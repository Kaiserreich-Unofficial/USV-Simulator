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

using namespace std;

int DYN_BLOCK_X;
using DYN_T = heron::USVDynamics;
const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
using COST_T = QuadraticCost<DYN_T>;
using COST_PARAM_T = QuadraticCostTrajectoryParams<DYN_T>;
using FB_T = DDPFeedback<DYN_T, 100>;
using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, 100, 4096, SAMPLING_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;
using PLANT_T = heron::USVMPCPlant<CONTROLLER_T>;
using state_array = DYN_T::state_array;
using control_array = DYN_T::control_array;

// Globals
int num_rollouts;
int horizon;
float dt;
vector<float> x_weight = {10, 10, 10, 10, 10, 10};
float _lambda;
float _alpha;
int max_iter;
bool sim_enable;
float sim_total_time;
int sim_times;
state_array observed_state;
state_array target_state;
bool target_state_enable = false;
atomic<bool> alive(true);
uint32_t step = 0;

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
    if (!target_state_enable)
        target_state_enable = true;
    target_state[0] = state.pose.pose.position.x;
    target_state[1] = state.pose.pose.position.y;
    target_state[2] = tf::getYaw(state.pose.pose.orientation);
    target_state[3] = state.twist.twist.linear.x;
    target_state[4] = state.twist.twist.linear.y;
    target_state[5] = state.twist.twist.angular.z;
}

// Timer callback: main MPC loop
void mpc_timer_cb(const ros::TimerEvent &event,
                  shared_ptr<CONTROLLER_T> controller,
                  PLANT_T *plant,
                  COST_PARAM_T &cost_params)
{
    control_array cmd;
    if (target_state_enable)
    {
        memcpy(cost_params.s_goal, target_state.data(), DYN_T::STATE_DIM * sizeof(float));
        plant->setCostParams(cost_params);
        plant->updateState(observed_state, (step + 1) * dt);
        plant->runControlIteration(&alive);

        ROS_INFO("Avg Optimization time: %f ms", plant->getAvgOptimizationTime());
        ROS_INFO("Last Optimization time: %f ms", plant->getLastOptimizationTime());
        ROS_INFO("Avg Loop time: %f ms", plant->getAvgLoopTime());
        ROS_INFO("Avg Optimization Hz: %f Hz", 1.0 / (plant->getAvgOptimizationTime() * 1e-3));

        cmd = controller->getControlSeq().col(0);
        step++;
    }
    else
    {
        cmd.setZero();
    }
    std_msgs::Float32 left_msg, right_msg;
    left_msg.data = cmd[0] / MAX_CTRL;
    right_msg.data = cmd[1] / MAX_CTRL;
    pub_left.publish(left_msg);
    pub_right.publish(right_msg);
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "mppi_controller", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    // Load parameters
    nh.param<int>("num_rollouts", num_rollouts, 4096);
    nh.param<int>("dyn_block_size", DYN_BLOCK_X, 32);
    nh.param<int>("horizon", horizon, 100);
    nh.param<float>("dt", dt, 0.1);
    nh.getParam("x_weight", x_weight);
    nh.param<float>("lambda", _lambda, 1.0);
    nh.param<float>("alpha", _alpha, 0.0);
    nh.param<int>("max_iter", max_iter, 1);

    // Init dynamics, cost, controller
    DYN_T dynamics;
    COST_T cost;
    auto cost_params = cost.getParams();
    for (uint8_t i = 0; i < DYN_T::STATE_DIM; i++)
        cost_params.s_coeffs[i] = x_weight[i];
    cost.setParams(cost_params);
    FB_T fb_controller(&dynamics, dt);
    SAMPLING_T sampler;
    auto sampler_params = sampler.getParams();
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
    {
        sampler_params.std_dev[i] = MAX_CTRL;
        sampler_params.exponents[i] = 1.f;
    }
    sampler.setParams(sampler_params);
    CONTROLLER_PARAMS_T controller_params;
    controller_params.dt_ = dt;
    controller_params.lambda_ = _lambda;
    controller_params.alpha_ = _alpha;
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(horizon, 1, 1);
    controller_params.num_iters_ = max_iter;
    controller_params.num_timesteps_ = horizon;
    auto controller = make_shared<CONTROLLER_T>(&dynamics, &cost, &fb_controller, &sampler, controller_params);

    // Create MPC plant
    PLANT_T plant(controller, 1.0 / dt, 1);
    ROS_INFO("MPPI控制器初始化完成!");
    ROS_INFO("预测域: %d, 步长: %.2f, 控制标准差: %.2f", horizon, dt, MAX_CTRL);

    // Subscribers & Publishers
    ros::Subscriber sub_obs = nh.subscribe("/wamv/p3d_position", 1, observer_cb);
    ros::Subscriber sub_tgt = nh.subscribe("/wamv/target_position", 1, target_cb);
    pub_left = nh.advertise<std_msgs::Float32>("/wamv/thrusters/left_thrust_cmd", 100);
    pub_right = nh.advertise<std_msgs::Float32>("/wamv/thrusters/right_thrust_cmd", 100);

    // Timer for MPC at rate dt
    ros::Timer mpc_timer = nh.createTimer(ros::Duration(dt), boost::bind(&mpc_timer_cb, _1,
                                                                         controller, &plant, cost_params));

    // Spin to process callbacks
    ros::spin();

    ROS_INFO("Program ended");
    return 0;
}
