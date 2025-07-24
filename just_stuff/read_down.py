import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from dt_apriltags import Detector
from geometry_msgs.msg import PoseStamped
import tf.transformations as transformations
import pyrealsense2 as rs
from nav_msgs.msg import Odometry
import time
from ctypes import cdll
import random
import subprocess
import os
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
import math
from collections import deque
import uuid
from datetime import datetime
from scipy.signal import butter, filtfilt
def low_pass_filter(data, cutoff=1.5, fs=14, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def least_squares_slope(data):
    """
    对传入的列表数据进行最小二乘拟合，并返回拟合后的斜率。

    参数:
    data (list or array-like): 一维数据列表

    返回:
    float: 拟合后的斜率
    """
    # 创建自变量 x，假设 x 为数据的索引
    x = np.arange(len(data))
    y = np.array(data)
    
    # 使用 numpy 的 polyfit 函数进行线性拟合（一次多项式拟合）
    # polyfit 返回的是多项式的系数，从高次项到低次项
    slope, intercept = np.polyfit(x, y, 1)
    
    return slope
class AccelerationEstimator:
    def __init__(self):
        self.velocity_samples = deque(maxlen=3)  # 用于存储最近的速度样本
        self.time_samples = deque(maxlen=3)  # 用于存储对应的时间样本
        self.last_update_time = None  # 记录上次更新的时间

    def update(self, velocity, timestamp):
        """
        更新加速度估计器的样本数据
        """
        # 检查是否超时（超过1秒没有新数据）
        if self.last_update_time and (timestamp - self.last_update_time > 1.5):
            self.reset_samples()  # 超时清空样本

        # 更新样本和时间
        self.velocity_samples.append(velocity)
        self.time_samples.append(timestamp)
        self.last_update_time = timestamp  # 更新最后接收时间

        # 检查是否有足够的样本进行计算
        if len(self.velocity_samples) == 3:
            return self.compute_acceleration()
        else:
            return [0.0, 0.0, 0.0]  # 样本不足时返回零加速度

    def compute_acceleration(self):
        """
        计算加速度
        """
        # 获取时间和速度样本
        v1, v2, v3 = self.velocity_samples
        t1, t2, t3 = self.time_samples

        # 计算两个时间段内的速度变化率
        dv1 = np.array(v2) - np.array(v1)
        dv2 = np.array(v3) - np.array(v2)
        dt1 = t2 - t1
        dt2 = t3 - t2

        # 计算加速度 (使用有限差分法)
        if dt1 > 0 and dt2 > 0:
            a1 = dv1 / dt1
            a2 = dv2 / dt2
            # 返回两个加速度值的平均值
            acceleration = (a1 + a2) / 2
            return acceleration.tolist()
        else:
            return [0.0, 0.0, 0.0]  # 时间间隔无效时返回零加速度

    def reset_samples(self):
        """
        清空加速度估计器的样本数据
        """
        self.velocity_samples.clear()
        self.time_samples.clear()
        self.last_update_time = None  # 重置最后更新时间


class MedianFilter:
    def __init__(self, size=5):
        self.size = size
        self.window = deque(maxlen=size)

    def update(self, new_value):
        self.window.append(new_value)
        return np.median(self.window)
glo_pos = [0,0,0]
glo_vel = [0,0,0]
averge_v = [0,0,0]
vel_time = time.time()
adx = 0
ady = 0
pos_copy = []
def rotate_around_x_90(R):
    # Define the rotation matrix for 90 degrees around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    
    # Compute the new rotation matrix
    R_new = np.dot(Rx, R)
    return R_new
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def radians_to_degrees(radians):
    return radians * 180 / math.pi
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return radians_to_degrees(z)
def rotationMatrixToEulerAngles1(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return radians_to_degrees(y)
class TagVelocityEstimator:
    def __init__(self):
        # self.pose_subscriber = rospy.Subscriber('/tag_pose', PoseStamped, self.pose_callback)
        # self.velocity_publisher = rospy.Publisher('/tag_velocity', TwistStamped, queue_size=10)
        
        self.positions = []
        self.times = []
        
        self.max_samples = 14
        self.initial_time = None  # 用于记录第一次订阅的时间

        # 定时器，用于检测消息接收是否超时
        self.timeout_duration = 0.8  # 超时时间，单位为秒
        self.last_msg_time = time.time()
        self.timer = self.create_timer(self.timeout_duration, self.check_timeout)
        self.timer1 = self.create_timer(1,self.set_triger)
    def pose_callback(self):
        # 记录当前时间
        global glo_pos
        global glo_vel
        
        current_time = time.time()
        
        # 如果这是第一次接收消息，则设置初始时间
        if self.initial_time is None:
            self.initial_time = current_time
        
        # 更新最后接收消息的时间
        self.last_msg_time = current_time
        
        # 计算相对于初始时间的时间差
        time_diff = current_time - self.initial_time
        
        # 记录位置
        position = glo_pos
        # position[0] = round(position[0],2)
        # position[1] = round(position[1],2)
        # position[2] = round(position[2],2)
        # 存储时间和位置
        self.positions.append(position)
        self.times.append(time_diff)
        
        # 保留最新的 `max_samples` 个样本
        if len(self.positions) > self.max_samples:
            self.positions.pop(0)
            self.times.pop(0)
        
        # 如果有足够的样本，则计算速度
        if len(self.positions) == self.max_samples:
            self.compute_and_publish_velocity()

    def compute_and_publish_velocity(self):
        global glo_vel
        times = np.array(self.times)
        positions = np.array(self.positions)
        
        velocities = []
        r_squared_values = []  # 用于存储每个轴的相关系数
        
        for i in range(3):  # 对于 x, y, z
            p = positions[:, i]
            # 执行线性拟合：p = a*t + b
            A = np.vstack([times, np.ones(len(times))]).T
            a, b = np.linalg.lstsq(A, p, rcond=None)[0]
            velocities.append(a)
            
            # 计算拟合值
            p_fit = a * times + b
            
            # 计算R² (相关系数)
            ss_total = np.sum((p - np.mean(p))**2)  # 总离差平方和
            ss_res = np.sum((p - p_fit)**2)         # 残差平方和
            r_squared = 1 - (ss_res / ss_total)
            r_squared_values.append(r_squared)
        velocity_threshold = 0.05  # 速度阈值，单位可以是m/s，根据实际情况调整
        # filtered_velocities = [v if abs(v) > velocity_threshold else 0.0 for v in velocities]
        if r_squared_values[0]>0.6:
            glo_vel[0] = velocities[0]
        if r_squared_values[1]>0.5:
            glo_vel[1] = velocities[1]
        
        glo_vel = [v if abs(v) > velocity_threshold else 0.0 for v in glo_vel]
        # glo_vel = velocities
        # print(r_squared_values)
        # glo_vel = [low_pass_filter(glo_vel[i]) for i in range(3)]
    def check_timeout(self, event):
        global pos_copy
        current_time = time.time()
        if current_time - self.last_msg_time > self.timeout_duration:
            # 超时，没有新的消息，清空列表
            self.positions.clear()
            self.times.clear()
            # rospy.set_param('/drone0/planning/triger_', 0.0)
            pos_copy = []
    def set_triger(self,event):
        current_time = time.time()
        if current_time - self.last_msg_time > 4:
            rospy.set_param('/drone0/planning/triger_', 0.0)
    def create_timer(self, duration, callback):
        import threading
        def timer_thread():
            while True:
                time.sleep(duration)
                callback(None)
        timer = threading.Thread(target=timer_thread)
        timer.daemon = True  #使线程在主程序退出时自动退出
        timer.start()
        return timer

def delete_file_in_directory(directory_path, file_name):
    file_path = os.path.join(directory_path, file_name)
    
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"文件 {file_path} 不存在")
        return
    
    try:
        # 删除文件
        os.remove(file_path)
        print(f"文件 {file_path} 已删除")
    except Exception as e:
        print(f"删除文件时出错: {e}")
params = {
    "perching_px": 3,
    "perching_py": 3,
    "perching_pz": 1
}


# 上一位置的存储
last_position1 = {
    "px": params["perching_px"],
    "py": params["perching_py"],
    "pz": params["perching_pz"]
}

last_position2 = {
    "px": params["perching_px"],
    "py": params["perching_py"],
    "pz": params["perching_pz"]
}

rough_velocity = {
    "vx": 0,
    "vy": 0,
    "vz": 0
}
t0 = 0
t1 = None
t2 = None
script_path = '/home/ros/github/Fast-Drone-250/land.sh'
def run_shell_script(script_path):
    subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def set_ros_param(param_key, param_value):
    """使用 rospy 设置参数值"""
    rospy.set_param(param_key, float(param_value))
class VelocitySubscriber:
    def __init__(self):
        # self.subscriber = rospy.Subscriber('/tag_velocity', TwistStamped, self.callback)
        # 设置初始值
        self.velocities = [0.0, 0.0, 0.0]
        self.velocity_samples = []
        self.sample_count = 0
        self.average_velocity = [0.0, 0.0, 0.0]
        self.cumulative_time = 0.0
        self.last_update_time = time.time()
        self.flag = 0
        self.last_vel = 0
        self.acceleration_estimator = AccelerationEstimator()  # 加速度估计器
    def callback(self):
        global flag3
        global glo_vel
        global averge_v
        global vel_time
        # 提取线速度数据
        current_velocities = [
            round(glo_vel[0], 2),
            round(glo_vel[1], 2),
            round(glo_vel[2], 2)
        ]

        if self.sample_count < 3:
            # 收集10次速度样本
            self.velocity_samples.append(current_velocities)
            self.sample_count += 1
            if self.sample_count == 3:
                # 计算平均速度
                self.average_velocity = [
                    sum(v[i] for v in self.velocity_samples) / 3 for i in range(3)
                ]
                # if self.flag == 1:
                #     if np.linalg.norm(np.array(self.average_velocity) - np.array(averge_v))/(time.time() - vel_time) > 6:
                #         self.average_velocity = self.average_velocity + (-self.average_velocity + averge_v)/np.linalg.norm(-self.average_velocity + averge_v)*6*(time.time() - vel_time)
                self.falg = 1
                vel_time = time.time()
                

                set_ros_param('/drone0/planning/perching_vx', round(self.average_velocity[0],2))
                set_ros_param('/drone0/planning/perching_vy', round(self.average_velocity[1],2))
                set_ros_param('/drone0/planning/perching_vz', 0.00)
                averge_v = self.average_velocity
                self.velocity_samples = []  # 清空样本列表
                self.sample_count = 0
        current_time = time.time()
        # acceleration = self.acceleration_estimator.update(current_velocities, current_time)
        # acceleration = [v if abs(v) > 0.1 else 0.0 for v in acceleration]
        # set_ros_param('/drone0/planning/perching_ax', round(acceleration[0], 2))
        # set_ros_param('/drone0/planning/perching_ay', round(acceleration[1], 2))
        # set_ros_param('/drone0/planning/perching_az', round(acceleration[2], 2))
        # print(acceleration)
        # 设置全局变量 flag3
    def change(self):
        self.cumulative_time = 0
        self.sample_count = 0 
last_call_time = None
def main(position):
    #rospy.init_node('param_updater', anonymous=True)
        # 5% 的概率改变速度
        global t0
        global t1
        global t2
        global last_call_time
        t0 =time.time()
        

    # 更新 last_call_time 为当前时间
        

        params["perching_px"] = position[0]
        params["perching_py"] = position[1]
        params["perching_pz"] = position[2]
        # params["perching_pz"] remains constant
        if (t1!= None and t2!= None) :
            rough_velocity["vx"] = 0*(params["perching_px"] - last_position1["px"])/(t0 - t1) + 0*(last_position1["px"] - last_position2["px"])/(t1 - t2)
            rough_velocity["vy"] = 0*(params["perching_py"] - last_position1["py"])/(t0-t1) + 0*(last_position1["py"] - last_position2["py"])/(t1-t2)
            rough_velocity["vz"] = 0*(params["perching_pz"] - last_position1["pz"])/(t0-t1) + 0*(last_position1["pz"] - last_position2["pz"])/(t1-t2)

        last_position2["px"] = last_position1["px"]
        last_position2["py"] = last_position1["py"]
        last_position2["pz"] = last_position1["pz"]
        
        t2 = t1
        
        last_position1["px"] = params["perching_px"]
        last_position1["py"] = params["perching_py"]
        last_position1["pz"] = params["perching_pz"]

        t1 = t0


        # 使用 rospy 设置参数值
        set_ros_param('/drone0/planning/perching_px', round(params["perching_px"],2))
        set_ros_param('/drone0/planning/perching_py', round(params["perching_py"],2))
        set_ros_param('/drone0/planning/perching_pz', round(params["perching_pz"],2))

        # set_ros_param('/drone0/planning/perching_vx', rough_velocity["vx"])
        # set_ros_param('/drone0/planning/perching_vy', rough_velocity["vy"])
        # set_ros_param('/drone0/planning/perching_vz', rough_velocity["vz"])
        # t4 = time.time()
        # if last_call_time is not None:
        #     time_interval = t4 - last_call_time
        #     frequency = 1.0 / time_interval
        #     print(f"当前函数调用频率: {frequency:.2f} 次/秒")
        # else:
        #     print("第一次调用，无法计算频率")
        # last_call_time = t4
        # 等待下一个周期

xlib = cdll.LoadLibrary('libX11.so')
xlib.XInitThreads()
global_image = None
class UAVStateListener:
    def __init__(self, c_flag):
        #rospy.init_node('uav_state_listener', anonymous=True)
        self.c_flag = c_flag
        self.trigger_condition_met = False
        self.T1 = None
        self.R1 = None
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.pose_cb)
        self.current_position = None
        self.current_orientation = None
        self.current_rotation_matrix = None
        self.list = []
        # 用于保存T1和R1
        # 触发条件的标志
        

    def pose_cb(self, msg):
                # 从 Odometry 消息中获取位置和姿态
                self.list.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                if len(self.list)==5:
                    self.T1 = np.array([np.mean(self.list,axis = 0)])
                    quaternion = (
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w
                    )
                    self.R1 = transformations.quaternion_matrix(quaternion)
                
                # 设置触发条件的标志为True
                    self.trigger_condition_met = True
                # 取消订阅
                    self.list = []


    def get_T1_R1(self):
        # 返回保存的T1和R1
        return self.T1, np.array(self.R1[:3,:3])
flag = True
def transform_to_world_coordinate(R_tag):
    # 将 AprilTag 旋转矩阵转换到世界坐标系
    R_world = np.dot(np.dot(M, R_tag), np.linalg.inv(M))
    return R_world
rospy.init_node('image_listener', anonymous=True)
uav_state_listener = UAVStateListener(flag)
pos = []
# v_sub = TagVelocityEstimator()
# fx = 640.68789851
# fy = 640.85089025
# cx = 618.95909174
# cy = 369.21605263
fx = 420.1543524
fy = 420.36231374
cx = 307.28393998
cy = 246.8883198
camera_params = (fx, fy, cx, cy)
dist_coeffs = np.zeros((4, 1))
tagsize = 0.07
detector = Detector(   families='tag25h9',   
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0) 
R1 = np.array([[1,0,0],
               [0,1,1],
               [0,0,1]])
T1 = np.array([[0,0,0]])
position = np.array([[0,0,0]])
pose = np.array([[1,0,0],
                 [0,1,1],
                 [0,0,1]])
# 检查摄像头是否成功打开
#相机内参
"""pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)

# 开始流
pipeline.start(config)"""

cap = cv2.VideoCapture(6,cv2.CAP_V4L2)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 禁用自动对焦
# cap.set(cv2.CAP_PROP_FOCUS, 3)      # 设置手动对焦距离（需要根据实际情况调整）
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 设置手动曝光，0.25表示手动模式
# cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 设置曝光值（需要根据实际情况调整）
flag1 = 0
M = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1]
])

last_deg = 0
median_filter = MedianFilter(size=5)
window = []
time_video = datetime.now().strftime("%H:%M:%S")
unique_filename = f"output_video_{time_video}.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用mp4v编码
out = cv2.VideoWriter(unique_filename, fourcc, 60.0, (640, 480)) 
averge_vel = [0,0,0]
v_sub = TagVelocityEstimator()
velocity_subscriber = VelocitySubscriber()
first_detection_time = None
last_detection_time = None
frame_count = 0
detection_rate = 0.0  # 初始化检测频率
while not rospy.is_shutdown():
    triger_ = rospy.get_param('/drone0/planning/triger_')
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（摄像头断开）")
        break
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    global_image = frame
    
    """frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()"""
        # 将图像转换为numpy数组以便显示
    #frame = np.asanyarray(global_image.get_data())

        # 将深度图像应用伪彩色以便观看
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    gray = cv2.cvtColor(global_image, cv2.COLOR_BGR2GRAY)
    # 使用CLAHE（自适应直方图均衡化）增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 使用高斯模糊降低噪声
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 使用锐化滤波器
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # gray = cv2.filter2D(blurred, -1, kernel)

    
    tags = detector.detect(gray,estimate_tag_pose = True , camera_params = [fx,fy,cx,cy],tag_size = tagsize)
    if len(tags)>0:
        
        # print(len(tags))
        flag = True
        flag1+=1
        #创建订阅实例
        timeout = time.time() + 0.005  # 等待5秒
        while not uav_state_listener.trigger_condition_met:
            rospy.sleep(0.1)  # 短暂睡眠以减少循环的CPU占用
            if time.time() > timeout:
                rospy.loginfo("Timeout waiting for the first message.")
                break
        T1, R1 = uav_state_listener.get_T1_R1()
        T1 = np.array([T1])
        if flag1 == 0:
            T8 = T1
        R1 = np.array(R1[:3, :3])
        # print("T1:",T1)
        # print("R1:",R1)
        for tag in tags:
            # print("tag.decision_margin:" , tag.decision_margin)
            if tag.decision_margin<10:
                continue
            center = tag.center
            corners = tag.corners
            corners = [(int(c[0]), int(c[1])) for c in corners]
            cv2.line(global_image, corners[0], corners[1], (0, 0, 255), 2)
            cv2.line(global_image, corners[1], corners[2], (0, 0, 255), 2)
            cv2.line(global_image, corners[2], corners[3], (0, 0, 255), 2)
            cv2.line(global_image, corners[3], corners[0], (0, 0, 255), 2)
            
            # 在图像上显示triger_值
            cv2.putText(global_image, f"position: {averge_vel}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # print("Tag ID:", tag.tag_id)
            # print("Pose:\n", tag.pose_R)
            # print("position:", tag.pose_t)
            T2 = np.array(tag.pose_t)
            T2 = T2.T
            #T3 = np.array([T2[0][2],-T2[0][0],-T2[0][1]])
            T3 = np.array([-T2[0][1],-T2[0][0],-T2[0][2]])
            R2 = np.array(tag.pose_R)
            R2_deg = rotationMatrixToEulerAngles(R2)
            # print("R2_deg:",R2_deg)
            R2 = rotate_around_x_90(R2)
            R2 = transform_to_world_coordinate(R2)
            # print("T2:",T2)
            # print("R2:",R2)
            #计算标签姿态和坐标
            position = np.dot(T3,R1.T)+T1
            position = np.array(position.flatten())
            position[0] = position[0]-0.10
            position[2] = position[2]-0.10
            averge_vel = position
            pose = np.dot(R2,R1)
            deg = rotationMatrixToEulerAngles1(pose)
            # deg = -deg
            R1_deg = rotationMatrixToEulerAngles(R1)
            fin_deg = 0.4*last_deg+0.6*(deg)
            # print("deg:",fin_deg)
            fin_deg = median_filter.update(fin_deg)
            fin_deg = float(fin_deg)
            if abs(fin_deg-last_deg)>20 and last_deg != 0:
                fin_deg = last_deg
            elif abs(fin_deg-last_deg)<4:
                fin_deg = last_deg
            else :
                last_deg = fin_deg
            # if flag1%3 == 0:
            #     rospy.set_param("/drone0/planning/target_yaw_deg", round(fin_deg,4))
            if len(window)<30:
                window.append(fin_deg)
            else:
                window.pop(0)
                window.append(fin_deg)
            # print("deg:",fin_deg)
            # print(R1_deg)
            # print("position:",position)
            if position[2]<=-0.4:
                continue
            init_position = "Initial Position: x={:.3f}, y={:.3f}, z={:.3f}\n".format(T1[0][0][0], T1[0][0][1], T1[0][0][2])
            position_info = "x={:.3f}\ny={:.3f}\nz={:.3f}\n".format(position[0], position[1], position[2])
            pos.append([position[0],position[1],position[2]])
            glo_pos = position
            v_sub.pose_callback()
            velocity_subscriber.callback()
            current_time = time.time()

            # 第一次检测标签，开始计时
            if first_detection_time is None:
                first_detection_time = current_time

            # 更新最后一次检测到标签的时间
            last_detection_time = current_time

            # 增加帧计数
            frame_count += 1

            # 计算持续识别时的频率
            elapsed_time = last_detection_time - first_detection_time
            if elapsed_time > 0:
                detection_rate = frame_count / elapsed_time

            # 显示检测频率
            # print(f"Detection rate: {detection_rate:.2f} detections per second")
            if flag1%3 == 0:
                pos = np.array(pos)
                pos = pos.T
                pos=[np.mean(sublist) for sublist in pos]
                # print(flag1)
                # print("pos:",pos)
                x_center = 320
                y_center = 240
                dx = center[0] - x_center
                dy = center[1] - y_center
                dead_zone = 80
                kx = 0.01
                # if abs(dx)>dead_zone:
                #     if dx>0:
                #         adx -= 0.0015
                #     else:
                #         adx += 0.0015
                # if abs(dy)>dead_zone:
                #     if dy>0:
                #         ady -= 0.0015
                #     else:
                #         ady += 0.0015

                # pos[1] = pos[1]+adx
                # pos[0] = pos[0]+ady
                if len(pos_copy)>0:
                    pos[0] = 0.4*pos_copy[0]+0.6*pos[0]
                    pos[1] = 0.4*pos_copy[1]+0.6*pos[1]
                    pos[2] = 0.4*pos_copy[2]+0.6*pos[2]
                main(pos)
                pos_copy = pos
                pos = []
                rospy.set_param('/drone0/planning/triger_', 3.0)
            if flag1 == 5:
                # run_shell_script(script_path)
                with open("average_position_info.txt", "w") as file:
                    file.write(init_position)
                # with open("/home/ros/github/Fast-Drone-250/coordinates.txt", "w") as file:
                #     file.write(position_info)
            if len(pos_copy)>0:
                pos_tar = pos_copy[2]
                T1_ =  T1[0][0][2]
                pos_tar1 = np.array([pos_copy[0],pos_copy[1]])
                T1_1 =  np.array([T1[0][0][0],T1[0][0][1]])
                distance = abs(pos_tar-T1_)
                # print(distance)
                distance1 = np.linalg.norm(pos_tar1-T1_1)
                
                # if flag1%20 == 0:
                #     distancex = pos_copy[0]-T1[0][0][0]
                #     distancey = pos_copy[1]-T1[0][0][1]
                #     vx = rospy.get_param('/drone0/planning/perching_vx')
                #     vy = rospy.get_param('/drone0/planning/perching_vy')
                #     delta_vx = 0.2*distancex
                #     vx = vx - delta_vx
                #     delta_vy = 0.2*distancey
                #     vy = vy - delta_vy
                #     rospy.set_param('/drone0/planning/perching_vx', float(vx))
                #     rospy.set_param('/drone0/planning/perching_vy', float(vy))
                # if T1_<1 and distance<=0.3:
                #     print("here open pub_triger2")
                #     run_shell_script(script_path)
                #     break
            # if len(pos_copy)>0:
            #     a= 2
                # pos_tar1 = np.array([pos_copy[0],pos_copy[1]])
                # T1_1 =  np.array([T1[0][0][0],T1[0][0][1]])
                # distance = np.linalg.norm(pos_tar1-T1_1)
                # if len(window)==30:
                #     a = least_squares_slope(window)
                # if abs(a)<0.8 or distance>1.2:
                #     rospy.set_param('/drone0/planning/triger_', 1.0)
                #     # print(a)
                #     # print("distance:",distance)
                #     main(pos_copy)
            # time.sleep(0.005)
        # break
    # else:
    #     if last_detection_time is not None:
    #         # last_detection_time = None
    #         print(f"No tags detected, waiting...")
    # cv2.imshow('img', gray)
    cv2.imshow('img', global_image)
    cv2.putText(global_image, f"triger_: {triger_}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(global_image, f"position: {averge_vel}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(global_image, f"vel: {averge_v}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(global_image, f"vins: {T1}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    out.write(global_image)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()

#rospy.init_node('tag_pose_publisher', anonymous=True)


# 创建一个发布器，发布到'/tag_pose'话题，使用PoseStamped消息类型
'''pose_publisher = rospy.Publisher('/tag_pose', PoseStamped, queue_size=10)
pose_msg = PoseStamped()

# 设置时间戳和参考框架
pose_msg.header.stamp = rospy.Time.now()
pose_msg.header.frame_id = "world"  # 或者其他参考框架

# 填充位置数据
pose_msg.pose.position.x = position[0]
pose_msg.pose.position.y = position[1]
pose_msg.pose.position.z = position[2]
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = pose  # 替换旋转部分
transformation_matrix[:3, 3] = position  # 替换位置部分
pose = transformations.quaternion_from_matrix(transformation_matrix)
# 填充姿态数据
pose_msg.pose.orientation.x = transformation_matrix[0]
pose_msg.pose.orientation.y = transformation_matrix[1]
pose_msg.pose.orientation.z = transformation_matrix[2]
pose_msg.pose.orientation.w = transformation_matrix[3]
pose_publisher.publish(pose_msg)'''
