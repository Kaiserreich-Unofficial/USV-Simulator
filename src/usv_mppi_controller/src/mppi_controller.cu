// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>

// Math-related
#include <math.h>
#include <stdio.h>
// Containers
#include <string>
#include <vector>
#include <deque>
#include <atomic>

// MPPI includes
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/Tube-MPPI/tube_mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

// USV model
#include <usv_dynamics.cuh>
#include <usv_mpc_plant.cuh>

// 检测到ctrl + c信号，退出程序
#include <signal.h>

using namespace std;

using DYN_T = wamv::USVDynamics;
using COST_T = QuadraticCost<DYN_T>;
using COST_PARAM_T = QuadraticCostTrajectoryParams<DYN_T>;
using FB_T = DDPFeedback<DYN_T, 100>;
using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, 100, 4096, SAMPLING_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;
using PLANT_T = wamv::USVMPCPlant<CONTROLLER_T>;
using state_array = DYN_T::state_array;
using control_array = DYN_T::control_array;

int DYN_BLOCK_X;
constexpr uint8_t DYN_BLOCK_Y = DYN_T::STATE_DIM;
state_array observed_state = state_array::Zero();
state_array target_state = state_array::Zero(); // 观测状态量和目标状态量

DYN_T dynamics;
COST_T cost;
COST_PARAM_T cost_params = cost.getParams();
SAMPLING_T sampler;
auto sampler_params = sampler.getParams();
CONTROLLER_PARAMS_T controller_params;
std::shared_ptr<CONTROLLER_T> controller; // mpc controller
std::shared_ptr<FB_T> fb_controller;      // feedback controller
std::shared_ptr<PLANT_T> plant;           // mpc plant

// 心跳保活机制
float heartbeat_duration;
double hbeat_target_time = 0.0;
bool hbeat_received_ = false;
bool continuous_hb_received_ = false; // 用于首次恢复/丢失时打印日志

// 推进消息发布
ros::Publisher pub_left;
ros::Publisher pub_right;

// Observer callback
void observer_cb(const nav_msgs::Odometry &state)
{
    observed_state[0] = state.pose.pose.position.x;
    observed_state[1] = state.pose.pose.position.y;
    observed_state[2] = tf::getYaw(state.pose.pose.orientation);
    observed_state[3] = state.twist.twist.linear.x;
    observed_state[4] = state.twist.twist.linear.y;
    observed_state[5] = state.twist.twist.angular.z;
}

// Target callback
void target_cb(const nav_msgs::Odometry &state)
{
    target_state[0] = state.pose.pose.position.x;
    target_state[1] = state.pose.pose.position.y;
    target_state[2] = tf::getYaw(state.pose.pose.orientation);
    target_state[3] = state.twist.twist.linear.x;
    target_state[4] = state.twist.twist.linear.y;
    target_state[5] = state.twist.twist.angular.z;
    if (!continuous_hb_received_)
    {
        ROS_INFO("心跳信号恢复，重新启动控制器...");
        continuous_hb_received_ = true;
    }
    hbeat_received_ = true;
}

void mySigintHandler(int sig)
{
    ROS_WARN("程序终止...");
    ros::shutdown(); // 通知 ROS 安全终止
}

// Timer callback: main MPC loop
void mpc_timer_cb(const ros::TimerEvent &event)
{
    // 心跳超时检测
    double now = ros::Time::now().toSec();
    if (now > hbeat_target_time)
    {
        if (hbeat_received_)
        {
            // 本周期收到心跳，允许输出推进命令
            hbeat_received_ = false;
            // （推进器正常由后面的 pub_left/pub_right 发布）
        }
        else
        {
            // 丢失心跳，停用推进器
            if (continuous_hb_received_)
            {
                ROS_WARN("心跳信号丢失 — 推进器已禁用！");
                continuous_hb_received_ = false;
            }
            std_msgs::Float32 zero;
            zero.data = 0.0f;
            pub_left.publish(zero);
            pub_right.publish(zero);
            return; // 跳过下发真实命令
        }
        // 设定下一个心跳检测点
        hbeat_target_time = now + heartbeat_duration;
    }

    static atomic<bool> alive(true); // 信号量，用于控制控制器的运行
    control_array cmd;
    memcpy(cost_params.s_goal, target_state.data(), DYN_T::STATE_DIM * sizeof(float));
    plant->setCostParams(cost_params);
    plant->updateState(observed_state, ros::Time::now().toSec());
    plant->runControlIteration(&alive);

    ROS_INFO_STREAM("平均优化时间: " << std::fixed << std::setprecision(1) << plant->getAvgOptimizationTime() << " ms, 上次优化时间: " << std::setprecision(1) << plant->getLastOptimizationTime() << " ms");
    // ROS_INFO("Avg Loop time: %f ms", plant->getAvgLoopTime());
    // ROS_INFO("Avg Optimization Hz: %f Hz", 1.0 / (plant->getAvgOptimizationTime() * 1e-3));

    cmd = controller->getControlSeq().col(0); // 从[-.5, .5] 转换到 [-1, 1]

    std_msgs::Float32 left_msg, right_msg;
    left_msg.data = static_cast<float>(cmd[0] * 2);
    right_msg.data = static_cast<float>(cmd[1] * 2);
    pub_left.publish(left_msg);
    pub_right.publish(right_msg);
}

void autoweight_mpc_timer_cb(const ros::TimerEvent &event_)
{
    // 心跳超时检测
    double now = ros::Time::now().toSec();
    if (now > hbeat_target_time)
    {
        if (hbeat_received_)
        {
            // 本周期收到心跳，允许输出推进命令
            hbeat_received_ = false;
            // （推进器正常由后面的 pub_left/pub_right 发布）
        }
        else
        {
            // 丢失心跳，停用推进器
            if (continuous_hb_received_)
            {
                ROS_WARN("心跳信号丢失 — 推进器已禁用！");
                continuous_hb_received_ = false;
            }
            std_msgs::Float32 zero;
            zero.data = 0.0f;
            pub_left.publish(zero);
            pub_right.publish(zero);
            return; // 跳过下发真实命令
        }
        // 设定下一个心跳检测点
        hbeat_target_time = now + heartbeat_duration;
    }

    CONTROLLER_T::state_trajectory traj = plant->getStateTraj();                       // 上一步计算的预测轨迹
    memcpy(cost_params.s_goal, target_state.data(), DYN_T::STATE_DIM * sizeof(float)); // 更新目标状态
    // 将目标状态扩展为与 traj 同样的列数
    CONTROLLER_T::state_trajectory goal_traj =
        target_state.replicate(1, traj.cols());

    // 计算代价的指数权重
    state_array cost_per_state = (traj - goal_traj).cwiseAbs().rowwise().mean(); // 计算每个状态变量的代价
    state_array cost_weight = cost_per_state.array().log() + 1;                  // 计算代价的指数权重
    Eigen::Map<state_array> weight_vec(cost_params.s_coeffs);                    // 每次映射一次，防止被优化器修改
    // weight的每个维度加上 0.1 * cost_weight
    weight_vec += 0.1 * cost_weight;
    // 限制权重大小
    weight_vec = weight_vec.cwiseMax(0.1f).cwiseMin(0.5f);
    weight_vec = weight_vec.array().log() + 1; // 归一化

    ROS_INFO_STREAM("新的状态权重: " << weight_vec.transpose().format(Eigen::IOFormat(1, 0, ", ", "\n", "[", "]")));
    // memcpy(cost_params.s_coeffs, weight_vec.data(), DYN_T::STATE_DIM * sizeof(float)); // 冗余的
    plant->setCostParams(cost_params);
    plant->updateState(observed_state, ros::Time::now().toSec());
    static std::atomic<bool> alive(true); // 优化线程的存活标志
    control_array cmd;
    plant->runControlIteration(&alive);

    ROS_INFO_STREAM("平均优化时间: " << std::fixed << std::setprecision(1) << plant->getAvgOptimizationTime() << " ms, 上次优化时间: " << std::setprecision(1) << plant->getLastOptimizationTime() << " ms");
    // ROS_INFO("Avg Loop time: %f ms", plant->getAvgLoopTime());
    // ROS_INFO("Avg Optimization Hz: %f Hz", 1.0 / (plant->getAvgOptimizationTime() * 1e-3));

    cmd = controller->getControlSeq().col(0); // 从[-.5, .5] 转换到 [-1, 1]

    std_msgs::Float32 left_msg, right_msg;
    left_msg.data = static_cast<float>(cmd[0] * 2);
    right_msg.data = static_cast<float>(cmd[1] * 2);
    pub_left.publish(left_msg);
    pub_right.publish(right_msg);
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "mppi_controller", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    // Load parameters
    static std::vector<float> x_weight = {10, 10, 10, 10, 10, 10};
    nh.param<int>("horizon", controller_params.num_timesteps_, 100); // 预测域
    nh.param<float>("dt", controller_params.dt_, 0.1);               // 步长
    nh.param<float>("lambda", controller_params.lambda_, 1.0);       // 温度参数
    nh.param<float>("alpha", controller_params.alpha_, 0.0);         // 探索参数
    nh.param<int>("max_iter", controller_params.num_iters_, 1);      // 最大迭代次数
    nh.param<int>("dyn_block_size", DYN_BLOCK_X, 32);
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);    // 动力学仿真块大小
    controller_params.cost_rollout_dim_ = dim3(controller_params.num_iters_, 1, 1); // 代价函数仿真块大小

    // 读取心跳超时参数
    nh.param<float>("heartbeat_duration", heartbeat_duration, 0.5);
    hbeat_target_time = ros::Time::now().toSec() + heartbeat_duration;

    // 读取并设置状态权重参数
    nh.getParam("x_weight", x_weight);
    memcpy(cost_params.s_coeffs, x_weight.data(), DYN_T::STATE_DIM * sizeof(float));
    cost.setParams(cost_params); // 设置状态权重

    // 读取并设置采样器参数
    static float stddev_;
    nh.param<float>("stddev", stddev_, .5); // 噪声标准差
    std::fill(sampler_params.std_dev, sampler_params.std_dev + DYN_T::CONTROL_DIM, stddev_);
    static float exponents_;
    nh.param<float>("exponents", exponents_, .5); // 有色噪声相关系数
    std::fill(sampler_params.exponents, sampler_params.exponents + DYN_T::CONTROL_DIM, exponents_);
    sampler.setParams(sampler_params); // 设置采样器参数

    fb_controller = std::make_shared<FB_T>(&dynamics, controller_params.dt_);                                        // 反馈控制器实例化
    controller = std::make_shared<CONTROLLER_T>(&dynamics, &cost, fb_controller.get(), &sampler, controller_params); // MPPI控制器实例化
    plant = std::make_shared<PLANT_T>(controller, 1 / controller_params.dt_, 1);                                     // PLANT实例化

    // 读取话题名称
    std::string obs_topic, tgt_topic, left_thrust_topic, right_thrust_topic;
    nh.param<std::string>("topics/observation", obs_topic, "/wamv/sensors/position/p3d_wamv");
    nh.param<std::string>("topics/target", tgt_topic, "/wamv/target_position");
    nh.param<std::string>("topics/left_thrust", left_thrust_topic, "/wamv/thrusters/left_thrust_cmd");
    nh.param<std::string>("topics/right_thrust", right_thrust_topic, "/wamv/thrusters/right_thrust_cmd");

    ROS_INFO_STREAM("MPPI控制器初始化完成!");
    ROS_INFO_STREAM("预测域: " << controller_params.num_timesteps_ << ", 步长: " << std::fixed << std::setprecision(1) << controller_params.dt_ << ", 控制标准差: " << stddev_);

    // 设置订阅和发布
    ros::Subscriber sub_obs = nh.subscribe(obs_topic, 10, observer_cb);
    ros::Subscriber sub_tgt = nh.subscribe(tgt_topic, 10, target_cb);
    pub_left = nh.advertise<std_msgs::Float32>(left_thrust_topic, 10);
    pub_right = nh.advertise<std_msgs::Float32>(right_thrust_topic, 10);

    // Timer for MPC at rate dt
    ros::Timer mpc_timer = nh.createTimer(ros::Duration(controller_params.dt_), &autoweight_mpc_timer_cb);
    signal(SIGINT, mySigintHandler); // 注册自定义SIGINT处理器

    // Spin to process callbacks
    ros::spin();
    return 0;
}
