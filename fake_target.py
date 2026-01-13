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
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    target_rate = 15.0  # 15 Hz for target_ekf_odom
    last_target_time = time.time()

    # 圆周运动参数
    radius = rospy.get_param('~radius', 2.0)          # 圆的半径（m），可通过参数调整
    speed = rospy.get_param('~speed', 0.8)          # 线速度（m/s），可通过参数调整
    center_x = rospy.get_param('~center_x', 0.0)    # 圆心x坐标
    center_y = rospy.get_param('~center_y', 0.0)    # 圆心y坐标
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

        # 发布target_ekf_odom（15Hz）
        if current_time - last_target_time >= 1.0 / target_rate:
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

            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            odom.twist.twist.angular = Vector3(0, 0, omega)

            pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，速度需要转换到狗坐标系）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        
        # 位置（世界坐标系）
        dog_pos_msg.pose.pose.position = Point(x, y, z)
        
        # yaw（世界坐标系下的yaw，存储在orientation.w中）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度需要从世界坐标系转换到狗坐标系（狗头yaw=0的坐标系）
        # 从世界坐标系旋转到狗坐标系：旋转-yaw角度
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # vx_dog = vx_world * cos(yaw) + vy_world * sin(yaw)
        # vy_dog = -vx_world * sin(yaw) + vy_world * cos(yaw)
        vx_dog = vx * cos_yaw + vy * sin_yaw
        vy_dog = -vx * sin_yaw + vy * cos_yaw
        
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
        
        # 角速度
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        
        dog_pos_pub.publish(dog_pos_msg)
        dog_pos_rate.sleep()

def publish_oscillating_motion():
    """x轴速度跳变模式：10秒为1 m/s，10秒为-1 m/s"""
    rospy.init_node('target_oscillating_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    last_target_time = time.time()
    target_rate = 15.0  # 15 Hz for target_ekf_odom

    # 运动参数
    vx_positive = rospy.get_param('~vx_positive', 1.0)      # 正向速度（m/s）
    vx_negative = rospy.get_param('~vx_negative', -1.0)      # 负向速度（m/s）
    period = rospy.get_param('~period', 25.0)                # 每个方向的持续时间（秒）
    start_x = rospy.get_param('~start_x', -30.0)               # 起始x坐标
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
        
        # 通过积分计算位置（使用速度和时间步长，50Hz）
        dt = 1.0 / 50.0  # 时间步长（50 Hz）
        current_x += vx * dt
        current_y += vy * dt
        z = height  # 保持恒定高度

        # 发布target_ekf_odom（15Hz）
        if current_time - last_target_time >= 1.0 / target_rate:
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

            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            odom.twist.twist.angular = Vector3(0, 0, 0)

            pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，速度需要转换到狗坐标系）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        
        # 位置（世界坐标系）
        dog_pos_msg.pose.pose.position = Point(current_x, current_y, z)
        
        # yaw（世界坐标系下的yaw，存储在orientation.w中）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度需要从世界坐标系转换到狗坐标系（狗头yaw=0的坐标系）
        # 从世界坐标系旋转到狗坐标系：旋转-yaw角度
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # vx_dog = vx_world * cos(yaw) + vy_world * sin(yaw)
        # vy_dog = -vx_world * sin(yaw) + vy_world * cos(yaw)
        vx_dog = vx * cos_yaw + vy * sin_yaw
        vy_dog = -vx * sin_yaw + vy * cos_yaw
        
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
        
        # 角速度（振荡运动为0）
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
        
        dog_pos_pub.publish(dog_pos_msg)
        dog_pos_rate.sleep()

def publish_s_shape_motion():
    """S形运动：按照sin函数前进，x方向匀速，y方向按sin变化"""
    rospy.init_node('target_s_shape_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    target_rate = 15.0  # 15 Hz for target_ekf_odom
    last_target_time = time.time()

    # S形运动参数
    vx_forward = rospy.get_param('~vx_forward', 0.8)      # x方向前进速度（m/s）
    amplitude = rospy.get_param('~amplitude', 3.0)         # y方向振幅（m）
    period = rospy.get_param('~period', 20.0)              # 一个完整周期的长度（m，在x方向）
    start_x = rospy.get_param('~start_x', 0.0)             # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)             # 起始y坐标
    height = rospy.get_param('~height', 1.0)               # 高度（m）
    
    start_time = time.time()
    start_x_pos = start_x
    
    rospy.loginfo("Starting S-shape motion (sin wave): vx=%.2f m/s, amplitude=%.2f m, period=%.2f m, start=(%.2f, %.2f), height=%.2f m", 
                  vx_forward, amplitude, period, start_x, start_y, height)

    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        
        # x方向：匀速前进
        x = start_x_pos + vx_forward * t
        
        # y方向：按sin函数变化
        # 使用x位置作为sin的参数，使得轨迹是sin形状
        # y = amplitude * sin(2*pi * x / period)
        y = start_y + amplitude * math.sin(2 * math.pi * x / period)
        
        # 速度计算
        # vx = dx/dt = vx_forward（常数）
        vx = vx_forward
        # vy = dy/dt = amplitude * (2*pi/period) * cos(2*pi*x/period) * vx_forward
        vy = amplitude * (2 * math.pi / period) * math.cos(2 * math.pi * x / period) * vx_forward
        vz = 0.0
        
        # 朝向角度：速度方向
        yaw = math.atan2(vy, vx)
        
        # 归一化yaw到 [-pi, pi]
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi
        
        z = height  # 保持恒定高度

        # 发布target_ekf_odom（15Hz）
        if current_time - last_target_time >= 1.0 / target_rate:
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"

            # 位置
            odom.pose.pose.position = Point(x, y, z)
            
            # 朝向（yaw存储在orientation.w中）
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0

            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            # 角速度：yaw的变化率
            # omega = d(yaw)/dt = d(atan2(vy, vx))/dt
            # 简化计算：omega ≈ (d(vy)/dt * vx - d(vx)/dt * vy) / (vx^2 + vy^2)
            # 由于vx是常数，d(vx)/dt = 0
            # d(vy)/dt = -amplitude * (2*pi/period)^2 * sin(2*pi*x/period) * vx_forward^2
            dvy_dt = -amplitude * (2 * math.pi / period) ** 2 * math.sin(2 * math.pi * x / period) * vx_forward ** 2
            omega = (dvy_dt * vx) / (vx * vx + vy * vy) if (vx * vx + vy * vy) > 1e-6 else 0.0
            odom.twist.twist.angular = Vector3(0, 0, omega)

            pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，速度需要转换到狗坐标系）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        
        # 位置（世界坐标系）
        dog_pos_msg.pose.pose.position = Point(x, y, z)
        
        # yaw（世界坐标系下的yaw，存储在orientation.w中）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度需要从世界坐标系转换到狗坐标系（狗头yaw=0的坐标系）
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_dog = vx * cos_yaw + vy * sin_yaw
        vy_dog = -vx * sin_yaw + vy * cos_yaw
        
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
        
        # 角速度
        dvy_dt = -amplitude * (2 * math.pi / period) ** 2 * math.sin(2 * math.pi * x / period) * vx_forward ** 2
        omega = (dvy_dt * vx) / (vx * vx + vy * vy) if (vx * vx + vy * vy) > 1e-6 else 0.0
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        
        dog_pos_pub.publish(dog_pos_msg)
        dog_pos_rate.sleep()

if __name__ == '__main__':
    # 通过参数选择运动模式：'circle'、'oscillating' 或 's_shape'
    mode = 'circle'
    
    try:
        if mode == 'oscillating':
            publish_oscillating_motion()
        elif mode == 's_shape':
            publish_s_shape_motion()
        else:
            publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
