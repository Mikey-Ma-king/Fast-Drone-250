import rospy
import serial
import struct
import math
from nav_msgs.msg import Odometry
import numpy as np

measurements = []
num_samples = 10000


SERIAL_PORT = '/dev/ttyUSB0'  # 请根据实际情况修改
BAUD_RATE = 115200
TIMEOUT = 0.1

def parse_int24(data):
    if len(data) != 3:
        raise ValueError("数据长度不足3字节")
    return int.from_bytes(data, byteorder='little', signed=True)

def parse_new_format(data):
    if len(data) < 37:
        return None

    try:
        # 根据新格式依次解析
        length = struct.unpack_from('>H', data, 4)[0]             # 长度 2字节（大端）
        reserved = struct.unpack_from('>H', data, 6)[0]           # 留号 2字节
        cmd = struct.unpack_from('>H', data, 8)[0]                # 命令 2字节
        version = struct.unpack_from('>H', data, 10)[0]           # 版本 2字节

        anchor_id = struct.unpack_from('<I', data, 12)[0]         # 基站ID 4字节（小端）
        tag_id = struct.unpack_from('<I', data, 16)[0]            # 信标ID 4字节（小端）

        distance_mm = struct.unpack_from('>I', data, 20)[0]       # 距离 4字节（小端）
        angle = struct.unpack_from('<h', data, 24)[0]             # 角度 2字节（小端）
        elevation = struct.unpack_from('<h', data, 26)[0]         # 仰角 2字节（小端）

        status = struct.unpack_from('<H', data, 28)[0]            # 状态 2字节（小端）
        sequence = struct.unpack_from('<H', data, 30)[0]          # 序号 2字节（小端）

        reserved2 = data[32:36]                                   # 预留 4字节（未使用）
        checksum = data[36]                                       # 校验 1字节

        return {
            'anchor_id': anchor_id,
            'tag_id': tag_id,
            'distance_m': distance_mm / 100.0,
            'angle_deg': angle / 100.0,
            'elevation_deg': elevation / 100.0,
            'status': status,
            'sequence': sequence,
            'checksum': checksum
        }

    except (struct.error, IndexError) as e:
        rospy.logerr(f"解析错误: {str(e)}")
        return None

def main():
    rospy.init_node('aoa_tag_publisher', anonymous=True)
    pub = rospy.Publisher('AOA_Tag_data', Odometry, queue_size=10)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        rospy.loginfo(f"✅ 成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")

        buffer = b''
        while not rospy.is_shutdown():
            buffer += ser.read(ser.in_waiting or 1)

            while True:
                # 查找帧头
                start_idx = buffer.find(b'\xFF\xFF\xFF\xFF')
                if start_idx == -1:
                    break

                # 检查长度字段是否完整
                if start_idx + 6 > len(buffer):
                    buffer = buffer[start_idx:]
                    break

                # 解析数据长度（大端）
                length = struct.unpack_from('>H', buffer, start_idx + 4)[0]

                # 检查完整帧是否到达
                frame_end = start_idx + length
                if frame_end > len(buffer):
                    buffer = buffer[start_idx:]
                    break

                # 提取完整帧数据
                frame = buffer[start_idx:frame_end]
                buffer = buffer[frame_end:]

                # 解析数据帧
                parsed_result = parse_new_format(frame)
                if parsed_result:
                    # 应用卡尔曼滤波
                    angle = parsed_result['elevation_deg']
                    if(len(measurements) < 10000):
                        measurements.append(angle)
                    else:
                        mean_distance = np.mean(measurements)
                        # 计算方差
                        variance = np.var(measurements)
                        # 计算标准差
                        std_dev = np.sqrt(variance)
                        print("\n===== UWB 设备噪声标定结果 =====")
                        print(f"测距均值: {mean_distance:.2f} mm")
                        print(f"测量方差 (σ²): {variance:.2f}")
                        print(f"测量标准差 (σ): {std_dev:.2f} mm")




    except serial.SerialException:
        rospy.logerr(f"❌ 无法打开串口 {SERIAL_PORT}，请检查连接")
    finally:
        if ser.is_open:
            ser.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
print("开始测量，请保持 UWB 标签静止...")
            # print(f"测量 {len(measurements)}/{num_samples}: {distance} mm")