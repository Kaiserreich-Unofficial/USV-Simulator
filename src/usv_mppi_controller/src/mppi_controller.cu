// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>

// Math相关
#include <math.h>
#include <stdio.h>
// 容器操作
#include <string>
#include <vector>
#include <deque>

// MPPI 相关
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/Tube-MPPI/tube_mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

// USV 模型
#include <usv_dynamics.cuh>
#include <usv_mpc_plant.cuh>

using namespace std;

int DYN_BLOCK_X;                                                                                // 动力学计算块大小
using DYN_T = heron::USVDynamics;                                                               // 动力学类型
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;                                                       // 状态维度
using COST_T = QuadraticCost<DYN_T>;                                                            // 成本函数类型
using COST_PARAM_T = QuadraticCostTrajectoryParams<DYN_T>;                                      // 成本函数参数类型
using FB_T = DDPFeedback<DYN_T, 100>;                                                           // 反馈控制器类型
using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>; // 采样分布类型
using CONTROLLER_T = TubeMPPIController<DYN_T, COST_T, FB_T, 100, 4096, SAMPLING_T>;            // 控制器类型
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;                                     // 控制器参数类型

using PLANT_T = heron::USVMPCPlant<CONTROLLER_T>;            // MPC Plnat类型
using state_array = DYN_T::state_array;                      // 状态数组类型
using control_array = DYN_T::control_array;                  // 控制数组类型
using control_trajectory = CONTROLLER_T::control_trajectory; // 控制轨迹类型

int num_rollouts;                                  // 前向采样次数
int horizon;                                       // 预测步长
float dt;                                          // 步长
vector<float> x_weight = {10, 10, 10, 10, 10, 10}; // 状态权重
float _lambda;                                     // 正则化/学习率参数
float _alpha;                                      // 探索参数
int max_iter;                                      // 最大迭代次数
bool sim_enable;                                   // 仿真开关
float sim_total_time;                              // 总仿真时间
int sim_times;                                     // 总仿真次数

state_array observed_state; // 观测状态
state_array target_state; // 目标状态
bool target_state_enable = false; // 目标状态开关

// 观测状态回调函数
void observer_cb(const nav_msgs::Odometry &state)
{
    observed_state[0] = state.pose.pose.position.x;
    observed_state[1] = state.pose.pose.position.y;
    observed_state[2] = tf::getYaw(state.pose.pose.orientation);
    observed_state[3] = state.twist.twist.linear.x;
    observed_state[4] = state.twist.twist.linear.y;
    observed_state[5] = state.twist.twist.angular.z;
    // ROS_INFO("新的观测状态: %f, %f, %f, %f, %f, %f", observed_state[0], observed_state[1], observed_state[2], observed_state[3], observed_state[4], observed_state[5]);
}

// 目标状态回调函数
void target_cb(const nav_msgs::Odometry &state)
{
    if (!target_state_enable) target_state_enable = true;
    target_state[0] = state.pose.pose.position.x;
    target_state[1] = state.pose.pose.position.y;
    target_state[2] = tf::getYaw(state.pose.pose.orientation);
    target_state[3] = state.twist.twist.linear.x;
    target_state[4] = state.twist.twist.linear.y;
    target_state[5] = state.twist.twist.angular.z;
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "mppi_controller", ros::init_options::AnonymousName);

    ros::NodeHandle nh;
    // 读取 yaml 配置参数
    nh.param<int>("num_rollouts", num_rollouts, 4096);
    nh.param<int>("dyn_block_size", DYN_BLOCK_X, 32);
    nh.param<int>("horizon", horizon, 100);
    nh.param<float>("dt", dt, 0.1);
    nh.getParam("x_weight", x_weight);
    nh.param<float>("lambda", _lambda, 1.0);
    nh.param<float>("alpha", _alpha, 0.0);
    nh.param<int>("max_iter", max_iter, 1);

    // 设置动力学和代价函数
    DYN_T dynamics;                      // 初始化动力学
    COST_T cost;                         // 初始化成本函数
    auto cost_params = cost.getParams(); // 获取并初始化代价参数
    for (uint8_t i = 0; i < DYN_T::STATE_DIM; i++)
    {
        cost_params.s_coeffs[i] = x_weight[i]; // 设置状态权重
    }
    cost.setParams(cost_params); // 设置代价参数

    // 定义 ILQR 反馈控制器
    FB_T fb_controller(&dynamics, dt);

    // 设置采样器
    SAMPLING_T sampler;
    auto sampler_params = sampler.getParams(); // 获取并初始化采样参数
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
        sampler_params.std_dev[i] = MAX_CTRL; // 设置控制标准差
        sampler_params.exponents[i] = 1.f;    // 设置 colored noise 指数
    }
    sampler.setParams(sampler_params); // 设置采样参数

    // 设置MPPI控制器参数
    CONTROLLER_PARAMS_T controller_params;
    controller_params.dt_ = dt;
    controller_params.lambda_ = _lambda;
    controller_params.alpha_ = _alpha;
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(horizon, 1, 1);
    controller_params.num_iters_ = max_iter;
    controller_params.num_timesteps_ = horizon;
    shared_ptr<CONTROLLER_T> controller =
        make_shared<CONTROLLER_T>(&dynamics, &cost, &fb_controller, &sampler, controller_params);

    // 创建MPC Plant
    PLANT_T plant(controller, (1.0 / dt), 1);
    ROS_INFO("MPPI控制器初始化完成!");
    ROS_INFO("预测域: %d, 步长: %.2f, 控制标准差: %.2f", horizon, dt, MAX_CTRL);

    // 订阅 USV 观测状态
    ros::Subscriber sub_observer = nh.subscribe("/wamv/p3d_position", 1, observer_cb); // 观测状态变量
    // 订阅 USV 目标轨迹
    ros::Subscriber sub_target = nh.subscribe("/wamv/target_position", 1, target_cb); // 观测状态变量
    // 发布消息到左螺旋桨转速控制话题
    ros::Publisher pub_left = nh.advertise<std_msgs::Float32>("/wamv/thrusters/left_thrust_cmd", 1);
    // 发布消息到右螺旋桨转速控制话题
    ros::Publisher pub_right = nh.advertise<std_msgs::Float32>("/wamv/thrusters/right_thrust_cmd", 1);

    ros::Rate rate(1.0 / dt);   // 控制频率
    atomic<bool> alive(true);         // MPPI求解器是否运行

    uint32_t step = 0;          // 当前步数
    // 主循环
    while (ros::ok())
    {
        control_array cmd;          // 控制指令消息
        if (target_state_enable)
        {
            memcpy(cost_params.s_goal, target_state.data(), DYN_T::STATE_DIM * sizeof(float));
            plant.setCostParams(cost_params);
            plant.updateState(observed_state, (step + 1) * dt);
            plant.runControlIteration(&alive);

            ROS_INFO("Avg Optimization time: %f ms", plant.getAvgOptimizationTime());
            ROS_INFO("Last Optimization time: %f ms", plant.getLastOptimizationTime());
            ROS_INFO("Avg Loop time: %f ms", plant.getAvgLoopTime());
            ROS_INFO("Avg Optimization Hz: %f Hz", 1.0 / (plant.getAvgOptimizationTime() * 1e-3));

            cmd = controller->getControlSeq().col(0); // 取出控制指令
        }
        else
        {
            cmd.setZero();
        }

        std_msgs::Float32 left_msg, right_msg;
        left_msg.data = cmd[0];
        right_msg.data = cmd[1];
        pub_left.publish(left_msg);
        pub_right.publish(right_msg);
        step++;
        rate.sleep();
        ros::spinOnce();
    }
    ROS_INFO("程序结束！");


    return 0;
}
