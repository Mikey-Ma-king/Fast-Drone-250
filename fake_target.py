#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3
import math
import tf
import time
import random

def publish_circle_motion():
    rospy.init_node('target_circle_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    radius = 4.0          # 圆的半径
    speed = 1.0           # 线速度 1 m/s
    omega = speed / radius  # 角速度 rad/s
    start_time = time.time()

    x = 1.0
    y = 0
    vx = 0.6
    vy = 0.1
    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        # print(t)

        # 圆周运动轨迹 (x = r*cos(θ), y = r*sin(θ)), 角度从 t * omega 增长
        theta = omega * t
        # x = radius * math.cos(theta) + 1.0  # x=1 为起始偏移
        # y = radius * math.sin(theta)
        # z = 1.0  # 保持恒定高度
        x += 1/30 * (vx) + random.gauss(0, 0.02)  # x=1 为起始偏移
        # print(x)
        y += 1/30 * (vy) + random.gauss(0, 0.02)
        z = 1.0  # 保持恒定高度
        vx += random.gauss(0, 0.01)
        vy += random.gauss(0, 0.01)
        vx = min(max(vx,-1),1)
        vy = min(max(vy,-1),1)

        # 速度方向
        # vx = -speed * math.sin(theta)
        # vy = speed * math.cos(theta)
        # vz = 0.0

        # 朝向的四元数（绕 z 轴）
        yaw = theta   # 方向与切线一致
        while abs(yaw) >math.pi:
            yaw -= 2*math.pi
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

        odom = Odometry()
        # odom.header.stamp = current_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "target_base"

        odom.pose.pose.position = Point(x, y, z)
        odom.pose.pose.orientation.w = math.atan2(vy, vx)
        odom.pose.pose.orientation.x  = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0

        odom.twist.twist.linear = Vector3(vx, vy, 0)
        odom.twist.twist.angular = Vector3(0, 0, omega)

        pub.publish(odom)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
