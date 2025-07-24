import rospy
import serial
import struct
import math
from nav_msgs.msg import Odometry
import time
import numpy as np
from collections import deque

SERIAL_PORT = '/dev/USB_AOA'  # 请根据实际情况修改
BAUD_RATE = 115200
TIMEOUT = 0.1

slide_yaw_win = deque(maxlen=10)
slide_angle_win = deque(maxlen=10)

from scipy.spatial.transform import Rotation as R

q = np.array([1.0, 0.0, 0.0, 0.0])
def target_position(q, d, azimuth, elevation):
    # 四元数 -> scipy格式 [x, y, z, w]
    quat = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(quat)

    # 本地坐标（球坐标转直角坐标）
    x = d * np.cos(elevation) * np.cos(azimuth)
    y = d * np.cos(elevation) * np.sin(azimuth)
    z = d * np.sin(elevation)
    local_vec = np.array([x, y, z])

    # 应用旋转
    rotated_vec = r.apply(local_vec)

    return rotated_vec

class KalmanFilter1D:
    def __init__(self, measurement_variance=0.01, process_variance=1e-5):
        """
        一维卡尔曼滤波器类
        :param measurement_variance: 测量方差（传感器噪声）
        :param process_variance:     过程方差（系统模型噪声）
        """
        # 滤波器状态
        self.state = None           # 状态估计值
        self.covariance = None      # 状态协方差
        
        # 噪声参数
        self.measurement_var = measurement_variance
        self.process_var = process_variance

    def update(self, measurement):
        """
        输入新测量值并更新滤波器状态
        :param measurement: 新测量值
        :return: 滤波后的状态值
        """
        # 初始化检查
        if self.state is None:
            self.state = measurement
            self.covariance = 1.0  # 初始协方差
            return self.state

        # 预测步骤（假设状态不变）
        predicted_state = self.state
        predicted_cov = self.covariance + self.process_var

        # 更新步骤
        kalman_gain = predicted_cov / (predicted_cov + self.measurement_var)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.covariance = (1 - kalman_gain) * predicted_cov

        return self.state

class EDR_KalmanFilter:
    def __init__(self):
        self.A = np.matrix([[1, 0, 0.02, 0], [0, 1, 0, 0.02], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # 状态转移矩阵
        self.statePost = np.matrix(np.zeros((4,1),np.float32))  # 状态估计值
        self.statePre = np.matrix(np.zeros((4,1),np.float32))  # 状态预测值
        self.Q = np.matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0.1, 0],  # vx噪声从0.001减小到0.0001
            [0, 0, 0, 0.1]   # vy噪声同理
        ], dtype=np.float32)
        self.errorCovPost = np.matrix(np.zeros((4, 4),np.float32))  # 协方差矩阵的估计值
        self.errorCovPre = np.matrix(np.zeros((4, 4), np.float32))  # 协方差矩阵的预测值
        self.H = np.matrix(np.zeros((4, 2), np.float32))  # 卡尔曼滤波增益矩阵
        self.C = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # 状态观测矩阵
        self.R = np.matrix([
            [5.0, 0],     # x测量噪声
            [0, 5.0]      # y测量噪声
        ], dtype=np.float32)

    def predict(self):  # 按照卡尔曼滤波的公式1、2计算
        self.statePre = self.A * self.statePost  # 卡尔曼滤波公式1
        self.errorCovPre = self.A * self.errorCovPost * self.A.T + self.Q  # 卡尔曼滤波公式2
        return self.statePre  # 返回卡尔曼滤波的预测值

    def correct(self,measurement):  # 按照卡尔曼滤波的公式3、4、5计算
        self.H = self.errorCovPre * self.C.T * np.linalg.inv(self.C * self.errorCovPre * self.C.T + self.R)  # 卡尔曼滤波公式3
        self.statePost = self.statePre + self.H*(measurement - self.C * self.statePre)  # 卡尔曼滤波公式4
        self.errorCovPost = self.errorCovPre - self.H * self.C * self.errorCovPre  # 卡尔曼滤波公式5
        return self.statePost  # 返回卡尔曼滤波的估计值

    def getNext(self, x, y):  # 滤波器的输入输出在此
        measured = np.matrix([[np.float32(x)], [np.float32(y)]])
        self.predict()
        corrected = self.correct(measured)
        x, y = int(corrected[0]), int(corrected[1])
        return x, y

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

def odom_callback(msg):
    global q
    q[0] = msg.pose.pose.orientation.w
    q[1] = msg.pose.pose.orientation.x
    q[2] = msg.pose.pose.orientation.y
    q[3] = msg.pose.pose.orientation.z

def main():
    global q
    kf_yaw = KalmanFilter1D(
        measurement_variance=0.05,    # 角度测量噪声方差
        process_variance=0.001        # 系统过程噪声
)
    kf_angle = KalmanFilter1D(
        measurement_variance=0.05,    # 角度测量噪声方差
        process_variance=0.001        # 系统过程噪声
)
    kf_dis = KalmanFilter1D(
        measurement_variance=0.01,    # 角度测量噪声方差
        process_variance=1e-5        # 系统过程噪声
)
    ekf_yaw = EDR_KalmanFilter()
    rospy.init_node('aoa_tag_publisher', anonymous=True)
    pub = rospy.Publisher('AOA_Tag_data', Odometry, queue_size=10)
    sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, callback=odom_callback)

    pubcount = 0
    start_time = time.time()

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

                    slide_yaw_win.append(parsed_result['elevation_deg'])
                    slide_yaw = np.mean(slide_yaw_win)

                    slide_angle_win.append(parsed_result['angle_deg'])
                    slide_up_angle = np.mean(slide_angle_win)
                    
                    filtered_distance = kf_dis.update(parsed_result['distance_m'])

                    filtered_angle = kf_yaw.update(slide_yaw)

                    filtered_up_angle = kf_angle.update( slide_up_angle)

                    EDR_yaw , EDR_up_angle = ekf_yaw.getNext(slide_yaw, slide_up_angle)


                    # 准备ROS消息
                    msg = Odometry()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "aoa_tag"
                    
                    # 填充位置信息
                    # relate_p = target_position(q,filtered_distance,float(EDR_yaw)/180*math.pi,float(EDR_up_angle)/180*math.pi)
                    msg.pose.pose.position.x = filtered_distance
                    # msg.pose.pose.position.y = relate_p[1]
                    # msg.pose.pose.position.z = relate_p[2]
                    # print("relate_p:",relate_p)
                    # 转换角度为四元数
                    # yaw = math.radians(filtered_angle)
                    # msg.pose.pose.orientation.z = float(EDR_up_angle)/180*math.pi
                    msg.pose.pose.orientation.z = float(EDR_up_angle)/180*math.pi
                    # msg.pose.pose.orientation.w = float(filtered_angle)/180*math.pi
                    msg.pose.pose.orientation.w = float(EDR_yaw)/180*math.pi
                    

                    pub.publish(msg)
                    pubcount += 1
                    if (time.time() - start_time) > 4:
                        print(f"📡 ILANXIN: 距离={filtered_distance:.3f}m,角度={EDR_yaw:.1f}° , angle={EDR_up_angle:.1f}°")
                        start_time = time.time()
                        pubcount = 0
                    # rospy.loginfo(f"📡 已发布: 距离={filtered_distance:.3f}m, "
                    #              f"角度={filtered_angle:.1f}°")

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