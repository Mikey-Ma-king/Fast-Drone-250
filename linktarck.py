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

# 卡尔曼滤波参数
measurement_variance_dis = 0.03
process_variance_dis = 0.001

measurement_variance_angle = 0.01
process_variance_angle = 0.001

# 存储不同标签的卡尔曼滤波变量
kalman_state = {}  # 估计值 x_k
kalman_covariance = {}  # 估计协方差 P_k

angle_pre = -1
angle_rad = 0
score = 0

timer = 1
def check_is_valid(delta_angle):
    """ 检查角速度是否有效 """
    global score, timer
    
    if delta_angle <= 15:
        timer += 0.2
        score += 0.005 * timer * timer
    else:
        timer -= 10
        score -= 5 * timer * timer
    timer = min(max(timer, 1), 30)
    score = min(max(score, 0), 10)

    # print(timer,score)

    return score >= 5


def apply_kalman_filter(tag_id, measurement, measurement_var, process_var, flag=True):
    """ 对测距数据应用卡尔曼滤波 """
    global kalman_state, kalman_covariance

    if not flag:
        kalman_state = {}
        kalman_covariance = {}

        
    if tag_id not in kalman_state:
        # 初始化卡尔曼滤波
        kalman_state[tag_id] = measurement
        kalman_covariance[tag_id] = 1
    else:
        # 预测步骤
        predicted_state = kalman_state[tag_id]
        predicted_covariance = kalman_covariance[tag_id] + process_var

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_var)
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
    global angle_pre, angle_rad

    FRAME_HEADER = b'\x55\x07'  # 帧头
    HEADER_SIZE = 21  # 固定头部大小
    ANCHOR_DATA_SIZE = 11  # 每个基站数据大小 (role + id + dis(3) + angle(2) + RSSI(2) + reserved(2))

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

            angle_pre = angle_rad
            angle_rad = math.radians(angle / 100.0)
            angle_delta = math.degrees(abs(angle_rad - angle_pre))
            # flag = check_is_valid(angle_delta)
            flag = True

            filtered_distance = apply_kalman_filter(f"{tag_id}_{anchor_id}_distance", distance_m, measurement_variance_dis, process_variance_dis)
            filtered_angle = apply_kalman_filter(f"{tag_id}_{anchor_id}_angle", angle_rad,measurement_variance_angle, process_variance_angle,flag)

            parsed_data["anchors"].append({
                "anchor_id": anchor_id,
                "distance_m": filtered_distance,
                "angle_rad": filtered_angle,
                "sending_flag": flag
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
    
    # 存储两个anchor的数据
    anchor1_data = None
    anchor2_data = None
    
    # 简单线性滤波：存储上一次的距离值
    anchor1_distance_filtered = None
    anchor2_distance_filtered = None
    filter_gain = 0.3  # 滤波增益，0.3表示新值权重30%，旧值权重70%

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
                # 收集所有anchor的数据
                for anchor in parsed_result["anchors"]:
                    anchor_id = anchor["anchor_id"]
                    distance = anchor["distance_m"]
                    angle_rad = anchor["angle_rad"]
                    
                    if anchor["sending_flag"]:
                        if anchor_id == 1:
                            # 简单线性滤波
                            if anchor1_distance_filtered is None:
                                anchor1_distance_filtered = distance
                            else:
                                anchor1_distance_filtered = (1.0 - filter_gain) * anchor1_distance_filtered + filter_gain * distance
                            anchor1_data = {"distance": anchor1_distance_filtered, "angle": angle_rad}
                        elif anchor_id == 2:
                            # 简单线性滤波
                            if anchor2_distance_filtered is None:
                                anchor2_distance_filtered = distance
                            else:
                                anchor2_distance_filtered = (1.0 - filter_gain) * anchor2_distance_filtered + filter_gain * distance
                            anchor2_data = {"distance": anchor2_distance_filtered, "angle": angle_rad}
                
                # 当两个anchor的数据都准备好时，发布一个消息
                if anchor1_data is not None and anchor2_data is not None:
                    msg = Odometry()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "aoa_tag"

                    # **position.x = anchor 1的距离, position.y = anchor 2的距离**
                    msg.pose.pose.position.x = anchor1_data["distance"]
                    msg.pose.pose.position.y = anchor2_data["distance"]
                    
                    # **orientation.w = anchor 1的角度, orientation.x = anchor 2的角度**
                    msg.pose.pose.orientation.x = anchor1_data["angle"]
                    msg.pose.pose.orientation.y = anchor2_data["angle"]

                    pub.publish(msg)
                    pub_count += 1
                    
                    if (time.time() - start_time) > 4:
                        print(f"linktrack:Hz:{int(pub_count/4)},Anchor1:dist={anchor1_data['distance']:.3f},angle={int(math.degrees(anchor1_data['angle']))},Anchor2:dist={anchor2_data['distance']:.3f},angle={int(math.degrees(anchor2_data['angle']))}")
                        start_time = time.time()
                        pub_count = 0
                    
                    # 清空数据，等待下一帧
                    anchor1_data = None
                    anchor2_data = None

    except serial.SerialException:
        rospy.logerr(f"❌ 无法打开串口 {SERIAL_PORT}，请检查连接")
    finally:
        ser.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
