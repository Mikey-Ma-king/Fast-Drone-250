#!/usr/bin/env python3
import struct
import serial
import time
import rospy
from nav_msgs.msg import Odometry
import math

# 串口配置
SERIAL_PORT = "/dev/USB_AOA"
BAUD_RATE = 921600
TIMEOUT = 0.1  # 超时时间，避免过度等待

measurement_variance = 0.03  # 设备标定出的测量噪声方差
process_variance = 0.001  # 适用于静止情况，移动可设 2000+

# 存储不同标签的卡尔曼滤波变量
kalman_state = {}  # 估计值 x_k
kalman_covariance = {}  # 估计协方差 P_k

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

# **解析 3 字节的 int24_t**
def parse_int24(data):
    """ 解析 3 字节的 24-bit 整数 (有符号数) """
    value = int.from_bytes(data, byteorder='little', signed=True)
    return value

# **解析 AOA NodeFrame0**
def parse_nlink_aoa_nodeframe0(data):
    FRAME_HEADER = b'\x55\x07'  # 帧头
    HEADER_SIZE = 21  # 固定头部大小
    ANCHOR_DATA_SIZE = 10  # 每个基站数据大小 (role + id + dis(3) + angle(2) + RSSI(2) + reserved(2))

    if len(data) < HEADER_SIZE:
        return None

    try:
        index = data.find(FRAME_HEADER)
        if index == -1:
            rospy.logwarn("❌ 未找到帧头")
            return None

        # **解析头部**
        frame_length, role, tag_id, local_time, system_time, _, voltage, valid_node_count = struct.unpack_from(
            '<H B B I I 4s H B', data, index + 2)

        parsed_data = {
            "frame_length": frame_length,
            "tag_role": role,
            "tag_id": tag_id,
            "local_time": local_time,
            "system_time": system_time,
            "voltage": voltage / 1000.0,  # 单位转换 (mV -> V)
            "valid_node_count": valid_node_count,
            "anchors": []
        }

        # **解析基站数据**
        offset = index + HEADER_SIZE
        for _ in range(valid_node_count):
            if offset + ANCHOR_DATA_SIZE > len(data):
                break  # 避免越界

            # **解析基站数据**
            node_role, anchor_id = struct.unpack_from('<B B', data, offset)
            distance_bytes = data[offset + 2: offset + 5]  # 3 字节距离
            angle, fp_rssi, rx_rssi = struct.unpack_from('<h B B', data, offset + 5)

            distance_mm = parse_int24(distance_bytes)  # 解析 int24_t 距离
            distance_m = distance_mm / 1000.0  # 转换为米
            angle_rad = math.radians(angle / 100.0)  # 角度转换为弧度制

            parsed_data["anchors"].append({
                "anchor_id": anchor_id,
                "distance_m": distance_m,
                "angle_rad": angle_rad
            })

            offset += ANCHOR_DATA_SIZE  # 移动到下一个基站

        return parsed_data

    except struct.error:
        rospy.logerr("❌ 解析失败")
        return None


# **ROS1 话题发布**
def main():
    rospy.init_node('aoa_tag_publisher', anonymous=True)
    pub = rospy.Publisher('AOA_Tag_data', Odometry, queue_size=10)
    start_time = time.time()
    pub_count = 0

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        rospy.loginfo(f"✅ 成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")

        while not rospy.is_shutdown():
            ser.reset_input_buffer()  # **清理缓冲区**
            time.sleep(0.005)

            available_bytes = ser.in_waiting
            if available_bytes == 0:
                continue  # **避免空读取**

            raw_data = ser.read(available_bytes or 32)
            # rospy.loginfo(f"🔍 原始数据: {raw_data.hex()}")  # **打印十六进制数据**

            parsed_result = parse_nlink_aoa_nodeframe0(raw_data)
            if parsed_result:
                for anchor in parsed_result["anchors"]:
                    distance = anchor["distance_m"]
                    distance1 = apply_kalman_filter("a0" , distance)
                    angle_rad = anchor["angle_rad"]

                    # **创建 ROS1 消息**
                    msg = Odometry()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "aoa_tag"

                    # **设置位置 (x = 距离)**
                    msg.pose.pose.position.x = distance

                    # **设置角度 (w = 角度转换为弧度制)**
                    msg.pose.pose.orientation.w = angle_rad

                    pub.publish(msg)
                    pub_count += 1
                    if (time.time() -start_time) > 4:
                        print(f"linktarck:Hz:{int(pub_count/4)},Updated data:{distance,int(angle_rad/3.14*180)}")
                        start_time = time.time()
                        pub_count = 0

                    # rospy.loginfo(f"📡 发布数据: 距离={distance:.3f} m, 角度 (弧度)={angle_rad:.4f} rad")

    except serial.SerialException:
        rospy.logerr(f"❌ 无法打开串口 {SERIAL_PORT}，请检查连接")
    finally:
        ser.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
