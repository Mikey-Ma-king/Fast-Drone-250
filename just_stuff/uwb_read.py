#!/usr/bin/env python

import serial
import time
import rospy
import numpy as np
from scipy.optimize import fsolve
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, PointStamped
from tf.transformations import quaternion_from_euler

# 打开串口
ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)

# 卡尔曼滤波参数
measurement_variance = 11722.54  # 设备标定出的测量噪声方差
process_variance = 500  # 适用于静止情况，移动可设 2000+

# 存储不同标签的卡尔曼滤波变量
kalman_state = {}  # 估计值 x_k
kalman_covariance = {}  # 估计协方差 P_k

# 存储两标签的测距值
distance_data = {}

# 已知的 A、B 之间的固定距离 (单位: mm)
d_AB = 300  # 例如，两标签固定相距 50cm

def hex2deci(hex_str):
    """ 将十六进制字符串转换为十进制整数 """
    try:
        return int(hex_str, 16)
    except ValueError:
        return 0

def apply_kalman_filter(tag_id, measurement):
    """ 对测距数据应用卡尔曼滤波 """
    global kalman_state, kalman_covariance

    if tag_id not in kalman_state:
        # 初始化卡尔曼滤波
        kalman_state[tag_id] = measurement
        kalman_covariance[tag_id] = 1
    else:
        # 预测步骤
        predicted_state = kalman_state[tag_id]
        predicted_covariance = kalman_covariance[tag_id] + process_variance

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance)
        new_state = predicted_state + kalman_gain * (measurement - predicted_state)
        new_covariance = (1 - kalman_gain) * predicted_covariance

        # 存储更新后的值
        kalman_state[tag_id] = new_state
        kalman_covariance[tag_id] = new_covariance

    return kalman_state[tag_id]

def parse_data(data):
    """ 解析 UWB 串口数据，区分标签 a0 和 a3 """
    parts = data.split()
    
    if len(parts) < 10:
        print("Invalid data format")
        return None
    
    tag_type = parts[-1]  # 例如 "a0:0" 或 "a3:0"
    tag_id = tag_type.split(":")[0]  # 取 "a0" 或 "a3"

    A0 = hex2deci(parts[2])  # 解析 A0 距离（毫米）
    
    # 进行卡尔曼滤波
    smoothed_A0 = apply_kalman_filter(tag_id, A0)

    # print(f"Tag {tag_id} - Raw A0: {A0}mm, Kalman Smoothed A0: {smoothed_A0:.1f}mm")

    return tag_id, smoothed_A0

def triangulate(d_A, d_B, d_AB):
    """
    计算 A, B 的相对基站的坐标
    :param d_A: 基站到 A 的距离
    :param d_B: 基站到 B 的距离
    :param d_AB: A 和 B 之间的固定距离
    :return: A 和 B 的坐标
    """
    def equations(vars):
        x_A, y_A, x_B, y_B = vars
        return [
            x_A**2 + y_A**2 - d_A**2,   # 基站到 A 的测距
            x_B**2 + y_B**2 - d_B**2,   # 基站到 B 的测距
            (x_B - x_A)**2 + (y_B - y_A)**2 - d_AB**2,  # A 到 B 的固定距离
            x_A  # 设定 A 在 x 轴上 (简化问题)
        ]

    # 设定初始猜测值
    initial_guess = [0, d_A, d_AB, d_B]  # 假设 A 在 x 轴上，B 在右侧

    # 求解方程组
    solution = fsolve(equations, initial_guess)

    x_A, y_A, x_B, y_B = solution
    return (x_A, y_A), (x_B, y_B)

def main():
    rospy.init_node('uwb_distance_publisher', anonymous=True)

    # 存储不同标签的 ROS 发布者
    odom_pubs = {}

    # 额外创建一个 ROS 发布者用于三角定位
    # triangulation_pub = rospy.Publisher("/uwb/triangulation", PointStamped, queue_size=10)

    rate = rospy.Rate(20)  # 20 Hz

    try:
        while not rospy.is_shutdown():
            if ser.in_waiting > 0:
                # 读取一行数据
                data = ser.readline().decode('utf-8', errors='replace').strip()
                # print(f"Received: {data}")

                # 解析数据
                result = parse_data(data)
                if result:
                    tag_id, smoothed_A0 = result
                    distance_data[tag_id] = smoothed_A0  # 记录当前标签的测距数据

                    # 检查是否已经为该标签创建了 ROS 话题
                    if tag_id not in odom_pubs:
                        topic_name = f"/uwb/odometry/{tag_id}"
                        odom_pubs[tag_id] = rospy.Publisher(topic_name, Odometry, queue_size=10)
                        rospy.loginfo(f"Created publisher for {topic_name}")

                    # 创建 Odometry 消息
                    odom_msg = Odometry()
                    odom_msg.header.stamp = rospy.Time.now()
                    odom_msg.header.frame_id = "odom"
                    odom_msg.child_frame_id = f"tag_{tag_id}"  # 记录标签 ID

                    # 设置位置
                    odom_msg.pose.pose.position.x = smoothed_A0 / 1000.0  # 转换为米
                    odom_msg.pose.pose.position.y = 0
                    odom_msg.pose.pose.position.z = 0

                    # 设置方向（假设方向未知，使用单位四元数）
                    odom_msg.pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, 0))

                    # 发布消息
                    odom_pubs[tag_id].publish(odom_msg)

                    # # **当两个标签数据都可用时，执行三角定位**
                    # if "a0" in distance_data and "a3" in distance_data:
                    #     d_A = distance_data["a0"]
                    #     d_B = distance_data["a3"]

                    #     # 计算 A, B 的相对位置
                    #     (x_A, y_A), (x_B, y_B) = triangulate(d_A, d_B, d_AB)

                    #     print(f"Triangulated Positions: A({x_A:.2f}, {y_A:.2f}), B({x_B:.2f}, {y_B:.2f})")

                    #     # 发布到 ROS 话题
                    #     point_msg = PointStamped()
                    #     point_msg.header.stamp = rospy.Time.now()
                    #     point_msg.header.frame_id = "map"
                    #     point_msg.point.x = x_B / 1000.0  # 转换为米
                    #     point_msg.point.y = y_B / 1000.0
                    #     point_msg.point.z = 0

                    #     triangulation_pub.publish(point_msg)

            rate.sleep()

    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        ser.close()  # 关闭串口

if __name__ == "__main__":
    main()
