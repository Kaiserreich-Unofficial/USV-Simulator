#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <mavros_msgs/RCOut.h>
#include <mutex>

class RCOutPublisher
{
public:
    RCOutPublisher()
    {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        // 从参数服务器读取心跳间隔（秒），默认为 0.1 秒
        float hb_sec;
        std::string left_thrust_topic, right_thrust_topic; // 从参数服务器读取左右推力话题名称，默认为 "/wamv/thrusters/left_thrust_cmd" 和 "/wamv/thrusters/right_thrust_cmd"
        pnh.param<float>("heartbeat_interval", hb_sec, 0.1);
        pnh.param<std::string>("topics/left_thrust_topic", left_thrust_topic, "/wamv/thrusters/left_thrust_cmd");
        pnh.param<std::string>("topics/right_thrust_topic", right_thrust_topic, "/wamv/thrusters/right_thrust_cmd");
        heartbeat_interval_ = ros::Duration(hb_sec);

        left_sub_ = nh.subscribe(left_thrust_topic, 10, &RCOutPublisher::leftCallback, this);
        right_sub_ = nh.subscribe(right_thrust_topic, 10, &RCOutPublisher::rightCallback, this);

        rcout_pub_ = nh.advertise<mavros_msgs::RCOut>("/mavros/rc/out", 10);

        timer_ = nh.createTimer(ros::Duration(0.1), &RCOutPublisher::timerCallback, this);

        msg_.channels.resize(8);

        // 初始化：把最后命令时间设为现在，避免启动时误触发
        ros::Time now = ros::Time::now();
        last_left_cmd_time_  = now;
        last_right_cmd_time_ = now;
    }

private:
    void leftCallback(const std_msgs::Float32::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        left_thrust_ = msg->data;
        last_left_cmd_time_ = ros::Time::now();
    }

    void rightCallback(const std_msgs::Float32::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        right_thrust_ = msg->data;
        last_right_cmd_time_ = ros::Time::now();
    }

    void timerCallback(const ros::TimerEvent&)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ros::Time now = ros::Time::now();

        // 如果超过心跳间隔未收到新命令，则复位推力为 0（映射后即 1500）
        if ((now - last_left_cmd_time_) > heartbeat_interval_) {
            left_thrust_ = 0.0f;
        }
        if ((now - last_right_cmd_time_) > heartbeat_interval_) {
            right_thrust_ = 0.0f;
        }

        // 映射到 [1000,2000]
        uint16_t left_rc  = thrustToRC(left_thrust_);
        uint16_t right_rc = thrustToRC(right_thrust_);

        msg_.header.stamp = now;
        msg_.channels[0] = left_rc;
        msg_.channels[1] = 1500; // 中立位置
        msg_.channels[2] = right_rc;

        // 其他通道置 0（或中立）
        for (size_t i = 3; i < msg_.channels.size(); ++i) {
            msg_.channels[i] = 0;
        }

        rcout_pub_.publish(msg_);
    }

    uint16_t thrustToRC(float thrust)
    {
        // thrust 范围 [-1.0,1.0] 映射到 [1000,2000]
        float v = (thrust + 1.0f) * 500.0f + 1000.0f;
        v = std::min(std::max(v, 1000.0f), 2000.0f);
        return static_cast<uint16_t>(v);
    }

    ros::Subscriber left_sub_;
    ros::Subscriber right_sub_;
    ros::Publisher  rcout_pub_;
    ros::Timer      timer_;

    mavros_msgs::RCOut msg_;
    float left_thrust_{0.0f};
    float right_thrust_{0.0f};

    ros::Duration heartbeat_interval_;
    ros::Time     last_left_cmd_time_;
    ros::Time     last_right_cmd_time_;

    std::mutex mutex_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wamv_rcout_node");
    ROS_INFO("wamv_rcout_node started");
    RCOutPublisher node;
    ros::spin();
    return 0;
}
