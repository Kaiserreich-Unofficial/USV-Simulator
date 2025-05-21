#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion

import time
import math


class ThrustControlNode:
    def __init__(self):
        rospy.init_node("thrust_control_node", anonymous=True)
        self.left_pub = rospy.Publisher(
            "/wamv/thrusters/left_thrust_cmd", Float32, queue_size=10)
        self.right_pub = rospy.Publisher(
            "/wamv/thrusters/right_thrust_cmd", Float32, queue_size=10)

        self.rate_hz = 10
        self.rate = rospy.Rate(self.rate_hz)

        # 推力映射范围
        self.pwm_mid = 1500
        self.pwm_min = 1000
        self.pwm_max = 2000

        # 控制参数
        self.duration = 2.0  # 控制一个动作的持续时间（秒）
        self.angle_threshold = 30  # Z字形换向角阈值（度）

        # 状态变量
        self.initial_yaw = None
        self.current_yaw = None
        self.last_yaw = 0
        self.accumulate_yaw = 0
        self.increasing_yaw = True
        self.z_count = 0
        self.spiral_change_flag = False

        # 推力控制初始值
        self.left_pwm = 2000
        self.right_pwm = 2000
        self.left_pwm_max = 2000
        self.right_pwm_max = 2000

        # 根据控制频率动态计算步长
        self.step_left = (self.left_pwm_max - self.pwm_mid) / (self.duration * self.rate_hz)
        self.step_right = (self.right_pwm_max - self.pwm_mid) / (self.duration * self.rate_hz)

        rospy.Subscriber("/wamv/sensors/position/p3d_wamv",
                         Odometry, self.odom_callback)

    def odom_callback(self, msg):
        # 四元数转 yaw
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y,
                            orientation_q.z, orientation_q.w]
        yaw = euler_from_quaternion(orientation_list)[2]

        if self.initial_yaw is None:
            self.initial_yaw = yaw
            self.last_yaw = yaw
        self.current_yaw = yaw

        # Z maneuver 判断换向
        dyaw = self.current_yaw - self.initial_yaw
        dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi  # Wrap to [-pi, pi]

        if self.increasing_yaw:
            if dyaw >= math.radians(self.angle_threshold - 1e-2):
                self.z_count += 1
                self.increasing_yaw = False
                rospy.loginfo(f"Z maneuver turn complete. Count: {self.z_count}")
        else:
            if dyaw <= -math.radians(self.angle_threshold - 1e-2):
                self.z_count += 1
                self.increasing_yaw = True
                rospy.loginfo(f"Z maneuver turn complete. Count: {self.z_count}")

        # Spiral 判断
        delta_yaw = self.current_yaw - self.last_yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
        self.accumulate_yaw += delta_yaw
        self.last_yaw = self.current_yaw

        if abs(self.accumulate_yaw) >= math.pi - 1e-2:
            self.accumulate_yaw = 0
            self.spiral_change_flag = True

    def pwm_to_thrust(self, pwm):
        return (pwm - self.pwm_mid) / 500.0

    def publish_thrust(self, left_pwm, right_pwm):
        self.left_pub.publish(Float32(self.pwm_to_thrust(left_pwm)))
        self.right_pub.publish(Float32(self.pwm_to_thrust(right_pwm)))

        rospy.loginfo_throttle(1.0, f"Left PWM: {left_pwm}, Right PWM: {right_pwm}, Yaw: {math.degrees(self.current_yaw):.1f}")

    def control_thrust(self):
        stage = 0
        start_time = time.time()

        while not rospy.is_shutdown():
            elapsed = time.time() - start_time

            if stage == 0:
                if elapsed < 10:
                    self.publish_thrust(2000, 2000)
                elif elapsed < 20:
                    self.publish_thrust(1100, 1100)
                elif elapsed < 30:
                    self.publish_thrust(2000, 2000)
                else:
                    stage = 1
                    start_time = time.time()
                    rospy.loginfo("Start Z maneuver")

            elif stage == 1:
                if self.z_count < 8: # Zigzag 次数
                    if self.increasing_yaw:
                        self.right_pwm = min(self.right_pwm + self.step_right, self.right_pwm_max)
                        self.left_pwm = max(self.left_pwm - self.step_left, self.pwm_mid)
                    else:
                        self.right_pwm = max(self.right_pwm - self.step_right, self.pwm_mid)
                        self.left_pwm = min(self.left_pwm + self.step_left, self.left_pwm_max)
                    self.publish_thrust(self.left_pwm, self.right_pwm)
                else:
                    stage = 2
                    start_time = time.time()
                    rospy.loginfo("Start left turn")

            elif stage == 2:
                if elapsed < 30:
                    self.publish_thrust(1800, 2000)
                else:
                    stage = 3
                    start_time = time.time()
                    rospy.loginfo("Start right turn")

            elif stage == 3:
                if elapsed < 30:
                    self.publish_thrust(2000, 1800)
                else:
                    stage = 4
                    start_time = time.time()
                    self.left_pwm = 1000
                    self.right_pwm = 2000
                    self.accumulate_yaw = 0
                    rospy.loginfo("Start left spiral")

            elif stage == 4:
                if self.spiral_change_flag:
                    self.spiral_change_flag = False
                    if self.left_pwm < 1900:
                        self.left_pwm += 100
                    else:
                        self.left_pwm = 2000
                        self.right_pwm = 1900
                        stage = 5
                        rospy.loginfo("Start right spiral")
                        continue
                self.publish_thrust(self.left_pwm, self.right_pwm)

            elif stage == 5:
                if self.spiral_change_flag:
                    self.spiral_change_flag = False
                    if self.right_pwm > 1000:
                        self.right_pwm -= 100
                    else:
                        stage = 6
                        rospy.loginfo("Finished!")
                        continue
                self.publish_thrust(self.left_pwm, self.right_pwm)

            self.rate.sleep()

    def run(self):
        rospy.sleep(2)
        self.control_thrust()
        rospy.spin()


if __name__ == "__main__":
    try:
        node = ThrustControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
