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

# Manufacturer NodeFrame7 implementation (32-bit IDs, two angles):
# https://github.com/nooploop-dev/nlink_parser2/blob/main/src/nlink_utils/src/nlink_linktrack_nodeframe7/nlink_linktrack_nodeframe7.c
# Values: protocol name, header size, node size, ID size.
FRAME_LAYOUTS = {
    0x07: ('aoa_nodeframe0', 21, 11, 1),
    0x08: ('nodeframe5', 24, 10, 4),  # Tag output: range only.
    0x0B: ('nodeframe7', 24, 16, 4),  # Anchor output: range and angle0/angle1.
}

class NLinkStreamParser:
    """Accumulate serial chunks; emit complete, checksum-verified NLink frames.

    Serial reads may contain half a frame or several concatenated frames.
    """
    MAX_FRAME_SIZE = 24 + 255 * 16 + 1

    def __init__(self):
        self.buffer = bytearray()
        self.frames = 0
        self.bad_checksums = 0
        self.bad_lengths = 0
        self.discarded_bytes = 0

    def feed(self, data):
        self.buffer.extend(data)
        result = []
        while self.buffer:
            index = self.buffer.find(b'\x55')
            if index < 0:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if index:
                self.discarded_bytes += index
                del self.buffer[:index]
            if len(self.buffer) < 2:
                break  # Preserve a header split across two reads.
            mark = self.buffer[1]
            if mark not in FRAME_LAYOUTS:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < 4:
                break
            length = struct.unpack_from('<H', self.buffer, 2)[0]
            _, header_size, node_size, _ = FRAME_LAYOUTS[mark]
            if length < header_size + 1 or length > self.MAX_FRAME_SIZE:
                self.bad_lengths += 1
                del self.buffer[0]
                continue
            # All supported formats carry node count in the last header byte.
            if len(self.buffer) >= header_size:
                expected = header_size + node_size * self.buffer[header_size - 1] + 1
                if length != expected:
                    self.bad_lengths += 1
                    del self.buffer[0]
                    continue
            if len(self.buffer) < length:
                break  # Do not parse or discard partial measurements.
            frame = bytes(self.buffer[:length])
            if (sum(frame[:-1]) & 0xff) != frame[-1]:
                self.bad_checksums += 1
                del self.buffer[0]
                continue
            del self.buffer[:length]
            self.frames += 1
            result.append(frame)
        return result


def parse_nlink_aoa_nodeframe0(data):
    """Decode complete old/new frames; keep the historical function name for callers.

    NodeFrame7 angle0 feeds the existing bearing field. NodeFrame5 has no
    bearing and must never manufacture one for the combined AOA output.
    """
    global angle_pre, angle_rad
    if len(data) < 4 or data[0] != 0x55 or data[1] not in FRAME_LAYOUTS:
        return None
    mark = data[1]
    protocol, header_size, node_size, id_size = FRAME_LAYOUTS[mark]
    if len(data) < header_size + 1:
        return None
    length = struct.unpack_from('<H', data, 2)[0]
    count = data[header_size - 1]
    if (length != len(data) or length != header_size + node_size * count + 1 or
            (sum(data[:-1]) & 0xff) != data[-1]):
        return None
    fmt = '<H B B I I 4s H B' if id_size == 1 else '<H B I I I 4s H B'
    _, role, local_id, local_time, system_time, _, voltage, count = struct.unpack_from(fmt, data, 2)
    bearing_available = mark != 0x08
    parsed = dict(protocol=protocol, frame_length=length, tag_role=role, tag_id=local_id,
                  local_time=local_time, system_time=system_time, voltage=voltage / 1000.0,
                  valid_node_count=count, bearing_available=bearing_available, anchors=[])
    for index in range(count):
        offset = header_size + index * node_size
        node_role = data[offset]
        remote_id = int.from_bytes(data[offset + 1:offset + 1 + id_size], 'little')
        distance_offset = offset + 1 + id_size
        distance_m = parse_int24(data[distance_offset:distance_offset + 3]) / 1000.0
        if distance_m < 0:
            continue
        filtered_angle = None
        angle1_rad = None
        if bearing_available:
            angle = struct.unpack_from('<h', data, distance_offset + 3)[0]
            if abs(angle) > 18000:
                continue
            angle_pre = angle_rad
            angle_rad = math.radians(angle / 100.0)
            filtered_angle = apply_kalman_filter(
                f'{local_id}_{remote_id}_angle', angle_rad,
                measurement_variance_angle, process_variance_angle, True)
            if mark == 0x0B:
                angle1 = struct.unpack_from('<h', data, distance_offset + 5)[0]
                if abs(angle1) <= 18000:
                    angle1_rad = math.radians(angle1 / 100.0)
        filtered_distance = apply_kalman_filter(
            f'{local_id}_{remote_id}_distance', distance_m,
            measurement_variance_dis, process_variance_dis)
        parsed['anchors'].append(dict(anchor_id=remote_id, node_role=node_role,
                                      distance_m=filtered_distance, angle_rad=filtered_angle,
                                      angle1_rad=angle1_rad, sending_flag=bearing_available))
    return parsed


def main():
    rospy.init_node('aoa_tag_publisher', anonymous=True)
    pub = rospy.Publisher('AOA_Tag_data', Odometry, queue_size=10)
    port = rospy.get_param('~serial_port', SERIAL_PORT)
    baud_rate = int(rospy.get_param('~baud_rate', BAUD_RATE))
    parser = NLinkStreamParser()
    ser = None
    start_time = time.monotonic()
    pub_count = 0
    anchor1_distance_filtered = None
    filter_gain = 0.3
    last_publish_time = float('-inf')
    min_publish_interval = 0.2  # Preserve the original maximum 5 Hz output.
    last_bytes_time = time.monotonic()
    last_supported_time = time.monotonic()
    try:
        ser = serial.Serial(port, baud_rate, timeout=TIMEOUT, exclusive=True)
        rospy.loginfo('AOA 串口已打开: %s @ %d，使用持续缓存和完整帧校验', port, baud_rate)
        while not rospy.is_shutdown():
            raw_data = ser.read(min(ser.in_waiting or 1, 4096))
            if raw_data:
                last_bytes_time = time.monotonic()
            elif time.monotonic() - last_bytes_time > 3.0:
                rospy.logwarn_throttle(3.0, 'AOA 串口没有数据，请检查供电、端口和连接')
            for frame in parser.feed(raw_data):
                parsed = parse_nlink_aoa_nodeframe0(frame)
                if parsed is None:
                    continue
                rospy.loginfo_once('AOA 协议已识别: %s, local_id=0x%X', parsed['protocol'], parsed['tag_id'])
                if not parsed['bearing_available']:
                    rospy.logwarn_throttle(3.0,
                        '收到标签端 NodeFrame5：只有距离，没有角度。当前 /AOA_Tag_data '
                        '需要距离和角度，请接基站端 NodeFrame7；不会把缺失角度当作零度。')
                    continue
                last_supported_time = time.monotonic()
                anchor_data = None
                for anchor in parsed['anchors']:
                    if not anchor['sending_flag']:
                        continue
                    distance = anchor['distance_m']
                    if anchor1_distance_filtered is None:
                        anchor1_distance_filtered = distance
                    else:
                        anchor1_distance_filtered = ((1.0 - filter_gain) * anchor1_distance_filtered
                                                     + filter_gain * distance)
                    anchor_data = dict(distance=anchor1_distance_filtered, angle=anchor['angle_rad'])
                now = time.monotonic()
                if anchor_data is None or now - last_publish_time < min_publish_interval:
                    continue
                msg = Odometry()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = 'aoa_tag'
                # Keep the downstream legacy contract: range in x, bearing in orientation.x.
                msg.pose.pose.position.x = anchor_data['distance']
                msg.pose.pose.orientation.x = anchor_data['angle']
                pub.publish(msg)
                pub_count += 1
                last_publish_time = now
                if now - start_time > 4.0:
                    rospy.loginfo('linktrack: %.1f Hz, distance=%.3f m, angle=%.1f deg',
                                  pub_count / (now - start_time), anchor_data['distance'],
                                  math.degrees(anchor_data['angle']))
                    start_time, pub_count = now, 0
            if raw_data and time.monotonic() - last_supported_time > 3.0 and parser.frames == 0:
                rospy.logwarn_throttle(3.0,
                    '尚未收到校验通过的完整 NLink 帧：raw=%s；校验失败=%d，长度异常=%d。'
                    '请核对串口、波特率和设备输出协议。',
                    raw_data[:32].hex(' '), parser.bad_checksums, parser.bad_lengths)
    except serial.SerialException as exc:
        rospy.logerr('AOA 串口错误 %s: %s', port, exc)
    finally:
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
