import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from dt_apriltags import Detector
import tf.transformations as transformations
import pyrealsense2 as rs
from nav_msgs.msg import Odometry
import time
from ctypes import cdll
import random
import subprocess
import os
from geometry_msgs.msg import PoseStamped, TwistStamped
from rospy import Time
import threading
import math
from collections import deque
from std_msgs.msg import Float32
import uuid
from datetime import datetime
class MedianFilter:
    def __init__(self, size=5):
        self.size = size
        self.window = deque(maxlen=size)

    def update(self, new_value):
        self.window.append(new_value)
        return np.median(self.window)
class FinDegPublisher:
    def __init__(self, node_name='fin_deg_publisher', topic_name='/drone0/planning/target_y', filter_size=5):
        self.publisher = rospy.Publisher(topic_name, Float32, queue_size=10)
        # self.median_filter = MedianFilter(size=filter_size)

    def publish_fin_deg(self, fin_deg):
        self.publisher.publish(fin_deg)
glo_pos = [0,0,0]
glo_vel = [0,0,0]
averge_vel = [0,0,0]
def adjust_drone_position(x_tag, y_tag, width, height,tag):
    # 画面中心的坐标
    x_center = width / 2
    y_center = height / 2

    # 二维码中心的偏移量
    dx = x_tag - x_center
    dy = y_tag - y_center

    # 控制增益参数
    k_yaw = 0.1  # 旋转角度增益
    k_pitch = 0.001  # 垂直位移增益

    # 计算旋转角度和垂直移动距离
    yaw_adjust = k_yaw * abs(dx)
    pitch_adjust = k_pitch * dy

    # 死区阈值
    dead_zone = 80  # 可以根据需要调整

    # 判断水平偏移
    if abs(dx) > dead_zone:
        if dx > 0:
            rospy.set_param("/drone0/planning/target_yaw_deg", float(tag-yaw_adjust))
        elif dx < 0:
            rospy.set_param("/drone0/planning/target_yaw_deg", float(tag+yaw_adjust))

    # 判断垂直偏移

def run_at_100hz():
    # 目标频率为100Hz，周期为1/100秒
    interval = 1.0 / 100
    while True:
        start_time = time.time()
        
        # 在这里调用你希望以100Hz运行的函数
        velocity_subscriber.callback()
        
        # 计算函数运行结束后的时间
        end_time = time.time()
        
        # 计算下一个循环需要等待的时间
        elapsed_time = end_time - start_time
        sleep_time = max(0, interval - elapsed_time)
        
        # 如果函数执行时间较长，使得不可能达到100Hz，可能需要处理这种情况
        if sleep_time == 0:
            print("Warning: Function execution time is too long to maintain 100Hz.")
        
        # 等待下一个周期
        time.sleep(sleep_time)
class TagVelocityEstimator:
    def __init__(self):
        # self.pose_subscriber = rospy.Subscriber('/tag_pose', PoseStamped, self.pose_callback)
        # self.velocity_publisher = rospy.Publisher('/tag_velocity', TwistStamped, queue_size=10)
        
        self.positions = []
        self.times = []
        
        self.max_samples = 12
        self.initial_time = None  # 用于记录第一次订阅的时间

        # 定时器，用于检测消息接收是否超时
        self.timeout_duration = 1.0  # 超时时间，单位为秒
        self.last_msg_time = time.time()
        self.timer = self.create_timer(self.timeout_duration, self.check_timeout)

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
        # print("position_v", positions)
        # print("time_v", times)
        
        velocities = []
        for i in range(3):  # 对于 x, y, z
            p = positions[:, i]
            # 执行线性拟合：p = a*t + b
            A = np.vstack([times, np.ones(len(times))]).T
            a, b = np.linalg.lstsq(A, p, rcond=None)[0]
            velocities.append(a)
        
        glo_vel = velocities

    def check_timeout(self, event):
        current_time = time.time()
        if current_time - self.last_msg_time > self.timeout_duration:
            # 超时，没有新的消息，清空列表
            self.positions.clear()
            self.times.clear()

    def create_timer(self, duration, callback):
        import threading
        def timer_thread():
            while True:
                time.sleep(duration)
                callback(None)
        timer = threading.Thread(target=timer_thread)
        timer.daemon = True  # 使线程在主程序退出时自动退出
        timer.start()
        return timer


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
triger = 0
flag3 = 0
t0 = 0
t1 = None
t2 = None
script_path = '/home/ros/Fast-Perching/sh_utils/pub_triger.sh'
script_path2 = '/home/ros/github/Fast-Drone-250/takeoff.sh'
def run_shell_script(script_path):
    subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def cout_deg(pos):
    eigenvalues, eigenvectors = np.linalg.eig(pos)

# 提取旋转轴（特征值为1的特征向量）
    rotation_axis = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real

    # 计算旋转轴与x轴的夹角
    x_axis = np.array([1, 0, 0])

    # 计算内积
    dot_product = np.dot(rotation_axis, x_axis)

    # 计算向量的模长
    rotation_axis_magnitude = np.linalg.norm(rotation_axis)
    x_axis_magnitude = np.linalg.norm(x_axis)

    # 计算夹角（弧度）
    angle_rad = np.arccos(dot_product / (rotation_axis_magnitude * x_axis_magnitude))

    # 判断旋转方向
    cross_product = np.cross(x_axis, rotation_axis)
    if cross_product[2] < 0:
        angle_rad = -angle_rad

    # 转换为角度
    angle_deg = np.degrees(angle_rad)

    # 调整角度范围在 -180 度到 180 度之间
    # if angle_deg > 180:
    #     angle_deg -= 360
    # elif angle_deg < -180:
    #     angle_deg += 360
    return angle_deg
# Checks if a matrix is a valid rotation matrix.
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
    def callback(self):
        global flag3
        global glo_vel
        global triger
        global averge_vel
        if triger == 0:
            return
        triger = 0
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        self.last_update_time = current_time
        self.flag+=1
        # 提取线速度数据
        current_velocities = [
            round(glo_vel[0], 2),
            round(glo_vel[1], 2),
            round(glo_vel[2], 2)
        ]

        if self.sample_count < 10:
            # 收集10次速度样本
            self.velocity_samples.append(current_velocities)
            self.sample_count += 1
            if self.sample_count == 10:
                # 计算平均速度
                self.average_velocity = [
                    sum(v[i] for v in self.velocity_samples) / 10 for i in range(3)
                ]
                averge_vel = self.average_velocity
                self.velocity_samples = []  # 清空样本列表
        else:
            # 检查当前速度与平均速度的差值是否都小于0.3
            velocity_change = [abs(current_velocities[i] - self.average_velocity[i]) for i in range(3)]
            if all(change < 0.2 for change in velocity_change):
                self.cumulative_time += time_diff
            else:
                self.cumulative_time = 0.0
                self.sample_count = 0  # 重新开始采集样本

        # 设置全局变量 flag3
        if self.cumulative_time >= 3.0:
            flag3 = 1
            set_ros_param('/drone0/planning/perching_vx', self.average_velocity[0])
            set_ros_param('/drone0/planning/perching_vy', self.average_velocity[1])
            set_ros_param('/drone0/planning/perching_vz', 0.00)
            self.cumulative_time = 0
            self.sample_count = 0 
        elif self.flag%5 == 0 and flag3 == 0:
            # 修改参数服务器中的参数值
            set_ros_param('/drone0/planning/perching_vx', current_velocities[0])
            set_ros_param('/drone0/planning/perching_vy', current_velocities[1])
            set_ros_param('/drone0/planning/perching_vz', 0.00)
    def change(self):
        self.cumulative_time = 0
        self.sample_count = 0 
def main(position):
    #rospy.init_node('param_updater', anonymous=True)
        global t0
        global t1
        global t2
        t0 =time.time()
        params["perching_px"] = position[0]
        params["perching_py"] = position[1]
        params["perching_pz"] = position[2]
        # params["perching_pz"] remains constant
        if (t1!= None and t2!= None) :
            rough_velocity["vx"] = 0.8*(params["perching_px"] - last_position1["px"])/(t0 - t1) + 0.2*(last_position1["px"] - last_position2["px"])/(t1 - t2)
            rough_velocity["vy"] = 0.8*(params["perching_py"] - last_position1["py"])/(t0-t1) + 0.2*(last_position1["py"] - last_position2["py"])/(t1-t2)
            rough_velocity["vz"] = 0.8*(params["perching_pz"] - last_position1["pz"])/(t0-t1) + 0.2*(last_position1["pz"] - last_position2["pz"])/(t1-t2)

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



        # 等待下一个周期

xlib = cdll.LoadLibrary('libX11.so')
xlib.XInitThreads()
global_image = None
def callback(data):
    global global_image
    bridge = CvBridge()
    # 将ROS的图像消息转换为OpenCV的图像格式
    # 注意红外图像一般为单通道灰度图，这里使用"mono8"
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    # 显示图像
    global_image = cv_image

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

# fx = 1391.6163330078125
# fy = 1390.30322265625
# cx = 987.856201171875
# cy = 542.668701171875
fx = 601.7569580078125
fy = 601.7569580078125
cx = 314.7174987792969
cy = 241.04949951171875
camera_params = (fx, fy, cx, cy)
dist_coeffs = np.zeros((4, 1))
tagsize = 0.07
detector = Detector(   families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0,
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
M = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])
def transform_to_world_coordinate(R_tag):
    # 将 AprilTag 旋转矩阵转换到世界坐标系
    R_world = np.dot(np.dot(M, R_tag), np.linalg.inv(M))
    return R_world
# 检查摄像头是否成功打开
flag = True
#相机内参
"""pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)

# 开始流
pipeline.start(config)"""
rospy.init_node('image_list', anonymous=True)
uav_state_listener = UAVStateListener(flag)
rospy.set_param('/flag', 0)
rospy.Subscriber("/camera/color/image_raw", Image, callback)
# pose_publisher = rospy.Publisher('/tag_pose', PoseStamped, queue_size=10)
# pose_msg = PoseStamped()

# 设置时间戳和参考框架
# pose_msg.header.stamp = rospy.Time.now()
# pose_msg.header.frame_id = "world"  # 或者其他参考框架
flag1 = 0
pos = []
v_sub = TagVelocityEstimator()
velocity_subscriber = VelocitySubscriber()
# thread = threading.Thread(target=run_at_100hz)
# thread.daemon = True  # 设置为守护线程，使主程序退出时线程自动结束
# thread.start()
flag4 = 1#else
flag5 = 1#for auto takeoff
# while True:
#     takeoff_triger = rospy.get_param('/px4ctrl/takeoff_triger')
#     if takeoff_triger == 1 and flag5 == 1:
#         flag5 = 0
#         run_shell_script(script_path2)
#         break
# time.sleep(20)
last_deg = 0
median_filter = MedianFilter(size=5)
# fin_deg_publisher = FinDegPublisher()
pos_copy = []
time_video = datetime.now().strftime("%H:%M:%S")
unique_filename = f"output_video_{time_video}.mp4"
dabing = 0
else_enter_time = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
out = cv2.VideoWriter(unique_filename, fourcc, 20.0, (640, 480)) 
while not rospy.is_shutdown():
    triger_callback = rospy.get_param('/drone0/planning/triger_')
    #print(takeoff_triger)
    """frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()"""
    if global_image is None:
            # rospy.sleep(0.1)
            continue

        # 将图像转换为numpy数组以便显示
    #frame = np.asanyarray(global_image.get_data())

        # 将深度图像应用伪彩色以便观看
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    gray = cv2.cvtColor(global_image, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray,estimate_tag_pose = True , camera_params = [fx,fy,cx,cy],tag_size = tagsize)
    if len(tags)>0:
        if flag4 == 0:
            flag4 = 1
        else_enter_time = None
        flag1 +=1
        flag = True
        #创建订阅实例
        timeout = time.time() + 0.005  # 等待5秒
        # while not uav_state_listener.trigger_condition_met:
        #     rospy.sleep(0.1)  # 短暂睡眠以减少循环的CPU占用
        #     if time.time() > timeout:
        #         rospy.loginfo("Timeout waiting for the first message.")
        #         break
        
        T1, R1 = uav_state_listener.get_T1_R1()
        T1 = np.array([T1])
        if flag1 == 0:
            T8 = T1
        R1 = np.array(R1[:3, :3])
        # print("T1:",T1)
        # print("R1:",R1)
        for tag in tags:
            corners = tag.corners
            corners = [(int(c[0]), int(c[1])) for c in corners]
            cv2.line(global_image, corners[0], corners[1], (0, 0, 255), 2)
            cv2.line(global_image, corners[1], corners[2], (0, 0, 255), 2)
            cv2.line(global_image, corners[2], corners[3], (0, 0, 255), 2)
            cv2.line(global_image, corners[3], corners[0], (0, 0, 255), 2)
            center = tag.center
            # 在图像上显示triger_值
            
            
            # print(tag.tag_family)
            # print(tag.tag_id)
            # print("Pose:\n", tag.pose_R)
            # print("position:", tag.pose_t)
            T2 = np.array(tag.pose_t)
            T2 = T2.T
            T3 = np.array([T2[0][2],-T2[0][0],-T2[0][1]])
            R2 = np.array(tag.pose_R)
            R2_deg = rotationMatrixToEulerAngles(R2)
            # print("R2_deg:",R2_deg)
            R2 = transform_to_world_coordinate(R2)
            # print("T2:",T2)
            # print("R2:",R2)
            #计算标签姿态和坐标
            position = np.dot(T3,R1.T)+T1
            position = np.array(position.flatten())
            position[0] = position[0] - 0.07
            pose = np.dot(R1,R2)
            if  tag.tag_id == 0:
                deg = rotationMatrixToEulerAngles(pose)
            elif tag.tag_id == 4:
                deg = rotationMatrixToEulerAngles(pose)+90
                if deg>180:
                    deg = deg-360
            else:
                deg = rotationMatrixToEulerAngles(pose)-90
                if deg<-180:
                    deg = deg+360
            # print(deg)
            R1_deg = rotationMatrixToEulerAngles(R1)
            fin_deg = 0.35*last_deg+0.65*(deg)
            fin_deg = median_filter.update(fin_deg)
            fin_deg = float(fin_deg)
            if abs(fin_deg-last_deg)>20:
                fin_deg = last_deg
            elif abs(fin_deg-last_deg)<3:
                fin_deg = last_deg
            else :
                last_deg = fin_deg
            if flag1%3 == 0:
                rospy.set_param("/drone0/planning/target_yaw_deg", round(fin_deg,4))
            # fin_deg_publisher.publish_fin_deg(fin_deg)
            # print(round(deg,0)+round(R1_deg,0))
            # if deg>90:
            #     deg = deg-180q
            # print("deg:",deg+R1_deg)
            # print("R1_deg:",R1_deg)
            # print("position:",position)
            init_position = "Initial Position: x={:.3f}, y={:.3f}, z={:.3f}\n".format(T1[0][0][0], T1[0][0][1], T1[0][0][2])
            position_info = "x={:.3f}\ny={:.3f}\nz={:.3f}\n".format(position[0], position[1], position[2]+2)
            pos.append([position[0],position[1],position[2]])
            glo_pos = position
            v_sub.pose_callback()
            
            velocity_subscriber.callback()
            # print("glo_vel:",glo_vel)
            triger = 1
            # pose_msg.pose.position.x = position[0]
            # pose_msg.pose.position.y = position[1]
            # pose_msg.pose.position.z = position[2]
            # pose_publisher.publish(pose_msg)
            # rospy.set_param('/flag', 1)
            if flag1%5 == 0:
                pos = np.array(pos)
                pos = pos.T
                pos=[np.mean(sublist) for sublist in pos]
                # print(flag1)
                # print("pos:",pos)
                if triger_callback == 0:
                    main(pos)
                pos_copy = pos
                pos = []
            if flag1 >= 5 and triger_callback == 0:
                set_ros_param('/drone0/planning/triger_', 1.0)
            if triger_callback == 1:
                plane_tag = rospy.get_param("/drone0/planning/target_yaw_deg")
                adjust_drone_position(center[0], center[1], 649, 480 , plane_tag)
            if flag3 == 1 and triger_callback == 1:
                pos_tar = np.array([pos_copy[0],pos_copy[1]])
                T1_ =  np.array([T1[0][0][0],T1[0][0][1]])
                distance = np.linalg.norm(pos_tar-T1_)
                print("averge:",averge_vel)
                # print("distance",distance)
                if abs(distance-1.2)<=0.15:
                    dabing+=1
                    if dabing >= 10:
                        rospy.set_param('/drone0/planning/triger_', 2.0)
                        print("shit")
                    else :
                        print("牢晨的dabing: ",dabing)
                else :
                    dabing = 0
        time.sleep(0.01)
    else:
        if else_enter_time is None:
        # 第一次进入 else 分支，记录当前时间
            else_enter_time = time.time()
        elif time.time() - else_enter_time > 0.1:
            # 如果时间超过了 0.1 秒，则执行 else 下的代码
            velocity_subscriber.change()
            flag3 = 0
            # set_ros_param('/drone0/planning/perching_vx', 0)
            # set_ros_param('/drone0/planning/perching_vy', 0)
            # set_ros_param('/drone0/planning/perching_vz', 0)
             # 重置 else_enter_time 以防止重复执行
        # if time.time() - else_enter_time > 0.8 and flag4 == 1 :
        #     if len(pos_copy)>0:
        #         rospy.set_param('/drone0/planning/triger_', 3.0)
        #         flag4 = 0
    cv2.imshow('img', gray)
    cv2.putText(global_image, f"triger_: {triger_callback}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(global_image, f"vy: {rospy.get_param('/drone0/planning/perching_vy')}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(global_image, f"vx: {rospy.get_param('/drone0/planning/perching_vx')}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    out.write(global_image)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
