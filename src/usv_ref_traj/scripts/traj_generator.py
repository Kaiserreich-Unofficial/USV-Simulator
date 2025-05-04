#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler

def generate_eight_traj(total_steps, dt):
    """
    生成八字轨迹：返回 shape=(N,6) 的数组，
    列分别为 [x, y, psi, u, v, r]
    """
    alpha1 = 0.05
    alpha2 = 0.025

    traj = np.zeros((total_steps, 6))
    for i in range(total_steps):
        t = i * dt
        # 位置
        x = 3.4 * np.sin(alpha1 * t)
        y = 4.8 * np.cos(alpha2 * t) - 6

        # 速度及加速度
        x_dot = 3.4 * alpha1 * np.cos(alpha1 * t)
        y_dot = 4.8 * (-alpha2) * np.sin(alpha2 * t)
        x_dd  = -3.4 * alpha1**2 * np.sin(alpha1 * t)
        y_dd  = 4.8 * (-alpha2**2) * np.cos(alpha2 * t)

        # 航向角及角速度
        psi     = np.arctan2(y_dot, x_dot)
        psi_dot = (x_dot * y_dd - y_dot * x_dd) / (x_dot**2 + y_dot**2)

        # 计算船体坐标系下的线速度 (u,v) 和角速度 r
        rot = np.array([
            [ np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [           0,           0, 1],
        ])
        uv_r = rot.dot(np.array([x_dot, y_dot, psi_dot]))

        traj[i, :] = [x, y, psi, uv_r[0], uv_r[1], uv_r[2]]
    return traj

def generate_sp_traj(total_steps, dt):
    traj = np.zeros((total_steps, 6))
    for i in range(total_steps):
        traj[i, :] = [5, 5, 1.57, 0, 0, 0]
    return traj

def generate_circle_traj(total_steps, dt):
    """
    生成八字轨迹：返回 shape=(N,6) 的数组，
    列分别为 [x, y, psi, u, v, r]
    """
    radius = 5.0
    omega = 0.2

    traj = np.zeros((total_steps, 6))
    for i in range(total_steps):
        t = i * dt
        # 位置
        x = radius * np.cos(omega * t - np.pi/2)
        y = radius * np.sin(omega * t - np.pi/2)

        # 航向角及角速度
        psi     = omega * t

        # 计算船体坐标系下的线速度 (u,v) 和角速度 r
        u = radius * omega
        v = 0
        r = omega

        traj[i, :] = [x, y, psi, u, v, r]
    return traj

def save_csv(filename, traj):
    """
    使用 numpy.savetxt 将轨迹保存为 CSV
    """
    header = 'x,y,psi,u,v,r'
    np.savetxt(filename, traj, delimiter=',', header=header, comments='')

def publish_trajectory():
    rospy.init_node('target_position_publisher')
    pub = rospy.Publisher('/wamv/target_position', Odometry, queue_size=10)


    total_steps = 50000
    dt          = 0.05
    rate = rospy.Rate(1/dt)  # 10Hz

    traj = generate_circle_traj(total_steps, dt)
    # save_csv('target_trajectory.csv', traj)

    for i in range(total_steps):
        if rospy.is_shutdown():
            break

        x, y, psi, u, v, r = traj[i]

        # 构造 Odometry 消息
        odom = Odometry()
        odom.header.stamp    = rospy.Time.now()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        # 位姿
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, psi)
        odom.pose.pose.orientation = Quaternion(*quat)

        # 速度
        odom.twist.twist.linear.x  = u
        odom.twist.twist.linear.y  = v
        odom.twist.twist.linear.z  = 0.0
        odom.twist.twist.angular.z = r
        rospy.loginfo("x: %.2f, y: %.2f, psi: %.2f, u: %.2f, v: %.2f, r: %.2f", x, y, psi, u, v, r)

        # 发布
        pub.publish(odom)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_trajectory()
    except rospy.ROSInterruptException:
        pass
