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

    # 圆周运动参数
    radius = rospy.get_param('~radius', 3.0)          # 圆的半径（m），可通过参数调整
    speed = rospy.get_param('~speed', 0.8)          # 线速度（m/s），可通过参数调整
    center_x = rospy.get_param('~center_x', 20.0)    # 圆心x坐标
    center_y = rospy.get_param('~center_y', 20.0)    # 圆心y坐标
    height = rospy.get_param('~height', 1.0)       # 高度（m）
    
    omega = speed / radius  # 角速度 rad/s
    start_time = time.time()
    
    rospy.loginfo("Starting circular motion: radius=%.2f m, speed=%.2f m/s, center=(%.2f, %.2f), height=%.2f m", 
                  radius, speed, center_x, center_y, height)

    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)

        # 圆周运动轨迹 (x = center_x + r*cos(θ), y = center_y + r*sin(θ))
        theta = omega * t
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        z = height  # 保持恒定高度

        # 速度方向（切线方向）
        vx = -speed * math.sin(theta)  # 速度在x方向的分量
        vy = speed * math.cos(theta)   # 速度在y方向的分量
        vz = 0.0

        # 朝向角度（与速度方向一致，即切线方向）
        yaw = theta + math.pi / 2.0  # 速度方向与半径垂直，所以角度+90度
        # 归一化到 [-pi, pi]
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "target_base"

        # 位置
        odom.pose.pose.position = Point(x, y, z)
        
        # 朝向（yaw存储在orientation.w中，与traj_server的格式一致）
        odom.pose.pose.orientation.w = yaw
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0

        # 速度
        odom.twist.twist.linear = Vector3(vx, vy, vz)
        odom.twist.twist.angular = Vector3(0, 0, omega)

        pub.publish(odom)
        rate.sleep()

def publish_oscillating_motion():
    """x轴速度跳变模式：10秒为1 m/s，10秒为-1 m/s"""
    rospy.init_node('target_oscillating_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    # 运动参数
    vx_positive = rospy.get_param('~vx_positive', 1.0)      # 正向速度（m/s）
    vx_negative = rospy.get_param('~vx_negative', -1.0)      # 负向速度（m/s）
    period = rospy.get_param('~period', 20.0)                # 每个方向的持续时间（秒）
    start_x = rospy.get_param('~start_x', 0.0)               # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)               # 起始y坐标
    height = rospy.get_param('~height', 1.0)                 # 高度（m）
    
    start_time = time.time()
    current_x = start_x
    current_y = start_y
    
    rospy.loginfo("Starting oscillating motion: vx_positive=%.2f m/s, vx_negative=%.2f m/s, period=%.2f s, start=(%.2f, %.2f), height=%.2f m", 
                  vx_positive, vx_negative, period, start_x, start_y, height)

    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        
        # 计算当前处于哪个周期（每20秒一个完整周期：10秒正向+10秒负向）
        cycle_time = t % (2 * period)  # 0 到 2*period
        
        # 根据时间判断当前速度方向
        if cycle_time < period:
            # 前10秒：正向速度
            vx = vx_positive
            yaw = 0.0  # 朝向x正方向
        else:
            # 后10秒：负向速度
            vx = vx_negative
            yaw = math.pi  # 朝向x负方向
        
        vy = 0.0  # y方向速度为0
        vz = 0.0  # z方向速度为0
        
        # 通过积分计算位置（使用速度和时间步长）
        dt = 1.0 / 30.0  # 时间步长（30 Hz）
        current_x += vx * dt
        current_y += vy * dt
        z = height  # 保持恒定高度

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "target_base"

        # 位置
        odom.pose.pose.position = Point(current_x, current_y, z)
        
        # 朝向（yaw存储在orientation.w中，与traj_server的格式一致）
        odom.pose.pose.orientation.w = yaw
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0

        # 速度
        odom.twist.twist.linear = Vector3(vx, vy, vz)
        odom.twist.twist.angular = Vector3(0, 0, 0)

        pub.publish(odom)
        rate.sleep()

if __name__ == '__main__':
    # 通过参数选择运动模式：'circle' 或 'oscillating'
    mode = 'oscillating'
    
    try:
        if mode == 'oscillating':
            publish_oscillating_motion()
        else:
            publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
