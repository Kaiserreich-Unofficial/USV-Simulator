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

// USV model
#include <usv_dynamics.cuh>

// 检测到ctrl + c信号，退出程序
#include <signal.h>

using namespace std;

using DYN_T = wamv::USVDynamics;

using state_array = DYN_T::state_array;
using control_array = DYN_T::control_array;
using output_array = DYN_T::output_array;

// Globals
float dt;
float u_left;
float u_right;

state_array observed_state;

ros::Publisher pub_left;
ros::Publisher pub_right;
ros::Publisher pub_target;

shared_ptr<DYN_T> dynamics;

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

// Timer callback: main MPC loop
void dyn_timer_cb(const ros::TimerEvent &event)
{
    static state_array tgt_state = observed_state; // Initial target state is the observed state
    state_array x_next, x_dot;
    output_array y;

    dynamics->step(tgt_state, x_next, x_dot, (control_array() << u_left, u_right).finished(), y, ros::Time::now().toSec(), dt);
    nav_msgs::Odometry tgt_msgs;

    tgt_msgs.header.stamp = ros::Time::now();
    tgt_msgs.header.frame_id = "map";      // parent frame
    tgt_msgs.child_frame_id = "base_link"; // child frame

    tgt_msgs.pose.pose.position.x = x_next[0];
    tgt_msgs.pose.pose.position.y = x_next[1];
    tgt_msgs.pose.pose.orientation = tf::createQuaternionMsgFromYaw(x_next[2]);
    tgt_msgs.twist.twist.linear.x = x_next[3];
    tgt_msgs.twist.twist.linear.y = x_next[4];
    tgt_msgs.twist.twist.angular.z = x_next[5];
    pub_target.publish(tgt_msgs);

    tgt_state = x_next;
    ROS_INFO_STREAM("Publish Thrust Command:" << fixed << setprecision(2) << u_left << " " << u_right);
    // ROS_INFO_STREAM("Target State: " << fixed << setprecision(2) << tgt_state.transpose());
    std_msgs::Float32 left_msg, right_msg;
    // 映射推力从[-100, 250]到[-1, 1]
    left_msg.data = u_left > 0 ? u_left / 250 : u_left / 100;
    right_msg.data = u_right > 0 ? u_right / 250 : u_right / 100;
    pub_left.publish(left_msg);
    pub_right.publish(right_msg);
}

void mySigintHandler(int sig)
{
    ROS_WARN("程序终止...");
    ros::shutdown(); // 通知 ROS 安全终止
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "dynamics_test", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    // Load parameters
    nh.param<float>("dt", dt, 0.1);
    nh.param<float>("dynamics_test/left_input", u_left, 50);
    nh.param<float>("dynamics_test/right_input", u_right, 50);

    // 读取话题名称
    std::string obs_topic, tgt_topic, left_thrust_topic, right_thrust_topic;
    nh.param<std::string>("topics/observation", obs_topic, "/wamv/p3d_position");
    nh.param<std::string>("topics/target", tgt_topic, "/wamv/target_position");
    nh.param<std::string>("topics/left_thrust", left_thrust_topic, "/wamv/thrusters/left_thrust_cmd");
    nh.param<std::string>("topics/right_thrust", right_thrust_topic, "/wamv/thrusters/right_thrust_cmd");

    dynamics = make_shared<DYN_T>();
    ROS_INFO_STREAM("动力学测试初始化完成!");
    ROS_INFO_STREAM("模型名称: " << dynamics->getDynamicsModelName().c_str() << ", 推力设置: " << fixed << setprecision(2) << u_left << " " << u_right);

    // 设置订阅和发布
    ros::Subscriber sub_obs = nh.subscribe(obs_topic, 1, observer_cb);
    // ros::Subscriber sub_tgt = nh.subscribe(tgt_topic, 1, target_cb);
    pub_target = nh.advertise<nav_msgs::Odometry>(tgt_topic, 100);
    pub_left = nh.advertise<std_msgs::Float32>(left_thrust_topic, 100);
    pub_right = nh.advertise<std_msgs::Float32>(right_thrust_topic, 100);

    // Timer for MPC at rate dt
    ros::Timer dyn_test_timer = nh.createTimer(ros::Duration(dt), &dyn_timer_cb);
    signal(SIGINT, mySigintHandler); // 注册自定义SIGINT处理器

    // Spin to process callbacks
    ros::spin();
    return 0;
}
