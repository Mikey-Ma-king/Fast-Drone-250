from quadrotor_msgs.msg import PositionCommand
import math
import time
import numpy as np
from nav_msgs.msg import Odometry
import subprocess
from geometry_msgs.msg import PoseStamped
from scipy.optimize import minimize
from scipy.optimize import fsolve,root_scalar
from geometry_msgs.msg import Quaternion, PointStamped
# from FlowDataMsg.msg import FlowDataMsg
from collections import deque
from MPC_python.MPC_python_complie import DroneMPC
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
from std_msgs.msg import Header

# 轨迹的起点
start_time = 0.0  # 初始时间

gos_V = []
gos_p = []
gos_yaw = 0
vins_p = np.array([0,0,0])
vins_v = np.array([0,0,0])
vins_yaw = 0
stop_triger = 0
read_triger = 0

dog_yaw0 = 0
dog_vins_p0 = []
dog_vins_yaw0 = 0
dog_p0 = []
target_p = np.array([0.0,0.0,0.3])
target_yaw = 0
data_list = [] #use for optimizing
optimize_count = 0
delta_r = np.array([0,0])

AOA_distance = 0.0
AOA_angle = 0.0

calibration_count = 50

flow_z = 1
target_ekf_odom =[0,0,0]

x_opt = None
u_opt = None
mpc = DroneMPC(N = 15)
mpc.N = 15
mpc.v_max = np.array([1.5, 1.5, 1.0])
mpc.a_max = np.array([1.2, 1.2, 1])
MPC_clcyle = time.time() - 10
shift = 0.3

# 定义目标函数
def objective(delta_r):
    """
    目标函数：最小化绝对误差和
    delta_r : 待优化的偏移量 [delta_x, delta_y]
    """
    delta_r = np.array(-delta_r)
    errors = []
    for i in range(len(data_list)):
        # 计算修正后的理论距离
        corrected_distance = np.linalg.norm(data_list[i][0] - data_list[i][1] - delta_r)
        # 计算与实测距离的绝对误差
        errors.append((corrected_distance - data_list[i][2])**2)
    return np.sum(errors)

import rospy
from quadrotor_msgs.msg import PositionCommand
from quadrotor_msgs.msg import TakeoffLand
from geometry_msgs.msg import Point

def publish_position(p, traj_id,v_x=0 , v_y=0 ,v_z = 0, yaw = 0):
    # 创建PositionCommand消息对象
    cmd = PositionCommand()

    # 设置时间戳和参考框架
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "world"  # 参考坐标系为world

    # 设置轨迹标识和状态
    cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
    cmd.trajectory_id = traj_id

    # 设置位置（从传入的p参数）
    cmd.position.x = p.x
    cmd.position.y = p.y
    cmd.position.z = p.z

    # 设置速度、加速度和yaw为零
    cmd.velocity.x = v_x
    cmd.velocity.y = v_y
    cmd.velocity.z = v_z
    cmd.acceleration.x = 0.0
    cmd.acceleration.y = 0.0
    cmd.acceleration.z = 0.0
    cmd.yaw = yaw

    cmd.yaw_dot = 0.0

    # 发布命令
    pos_cmd_pub.publish(cmd)

class VelocityEstimator:
    def __init__(self, tau=0.1, median_filter_window=3, 
                 stationary_threshold=0.001, stationary_window=5):
        """
        参数:
        - tau: float，EMA时间常数（秒），越小响应越快
        - median_filter_window: int，中值滤波窗口大小（0表示禁用）
        - stationary_threshold: float，静止判断的位置变化阈值（米）
        - stationary_window: int，连续检测到静止的次数才触发
        """
        self.tau = tau
        self.median_filter_window = median_filter_window
        self.stationary_threshold = stationary_threshold
        self.stationary_window = stationary_window
        
        self.prev_time = None
        self.prev_pos = None
        self.velocity = np.zeros(3)
        self.recent_v = []
        self.stationary_count = 0

    def reset(self):
        """重置估计器状态"""
        self.prev_time = None
        self.prev_pos = None
        self.velocity = np.zeros(3)
        self.recent_v = []
        self.stationary_count = 0

    def update(self, current_pos, current_time):
        """
        参数:
        - current_pos: ndarray，当前位置，形如 np.array([x, y, z])
        - current_time: float，当前时间戳（秒）

        返回:
        - 估计速度：np.array([vx, vy, vz])
        """
        # 输入校验
        current_pos = np.asarray(current_pos)
        if current_pos.shape != (3,):
            raise ValueError("current_pos must be a 3-element array")
        if not isinstance(current_time, (float, int)):
            raise ValueError("current_time must be numeric")

        # 处理时间戳异常
        if self.prev_time is not None and current_time < self.prev_time:
            print("Warning: 时间戳回退，重置估计器")
            self.reset()
            return np.zeros(3)

        # 首次更新不计算速度
        if self.prev_time is None or self.prev_pos is None:
            self.prev_time = current_time
            self.prev_pos = current_pos.copy()
            return np.zeros(3)

        dt = current_time - self.prev_time
        if dt <= 0.0:
            return self.velocity.copy()  # 忽略无效时间差

        delta_pos = current_pos - self.prev_pos
        pos_change = np.linalg.norm(delta_pos)
        v_now = delta_pos / dt

        # 零速检测
        if pos_change < self.stationary_threshold:
            self.stationary_count += 1
        else:
            self.stationary_count = 0

        if self.stationary_count >= self.stationary_window:
            self.velocity = np.zeros(3)
            self.recent_v = []
            return self.velocity.copy()

        # 中值滤波处理
        if self.median_filter_window > 1:
            self.recent_v.append(v_now)
            if len(self.recent_v) > self.median_filter_window:
                self.recent_v.pop(0)
            v_filtered = np.median(self.recent_v, axis=0)
        else:
            v_filtered = v_now

        # 动态计算EMA系数
        alpha = np.exp(-dt / self.tau)
        self.velocity = alpha * self.velocity + (1 - alpha) * v_filtered

        # 更新状态
        self.prev_time = current_time
        self.prev_pos = current_pos.copy()
        
        return self.velocity.copy()

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
        self.yaw = 0
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb(self, msg):
        global vins_p
        global vins_yaw
        global vins_v
        # 从 Odometry 消息中获取位置和姿态
        self.list.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        if len(self.list)==3:
            self.T1 = np.array([np.mean(self.list,axis = 0)])
            quaternion = (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            vins_q_w = msg.pose.pose.orientation.w
            vins_q_x = msg.pose.pose.orientation.x
            vins_q_y = msg.pose.pose.orientation.y
            vins_q_z = msg.pose.pose.orientation.z
            siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y)
            cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
        # 设置触发条件的标志为True
            self.trigger_condition_met = True
        # 取消订阅
            self.list = []
            vins_p = np.array(self.T1[0])
            vins_yaw = self.yaw
            vins_v[0] = msg.twist.twist.linear.x
            vins_v[1] = msg.twist.twist.linear.y

    def get_T1_R1(self):
        # 返回保存的T1和R1
        return self.T1[0],self.yaw
    
class dog_listerer:
    def __init__(self, c_flag):
        #rospy.init_node('uav_state_listener', anonymous=True)
        self.c_flag = c_flag
        self.trigger_condition_met = False
        self.T1 = None
        self.R1 = None
        self.V1 = None
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/dog_pos', Odometry, self.pose_cb)
        self.current_position = None
        self.current_orientation = None
        self.current_rotation_matrix = None
        self.list = []
        self.yaw = 0
        # 用于保存T1和R1
        # 触发条件的标志
        

    def pose_cb(self, msg):
        # 从 Odometry 消息中获取位置和姿态
        global gos_p
        global gos_V
        global gos_yaw
        self.list.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        if len(self.list)==1:
            self.T1 = np.array([np.mean(self.list,axis = 0)])
            quaternion = (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )
            vins_q_w = msg.pose.pose.orientation.w
            vins_q_x = msg.pose.pose.orientation.x
            vins_q_y = msg.pose.pose.orientation.y
            vins_q_z = msg.pose.pose.orientation.z
            siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y)
            cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z)
            self.yaw = msg.pose.pose.orientation.w
        # 设置触发条件的标志为True
            self.V1 = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            self.trigger_condition_met = True
        # 取消订阅
            self.list = []
            if self.T1[0][0] != 0 and self.T1[0][1] != 0:
                gos_p= np.array(self.T1[0])
                gos_V = self.V1
                gos_yaw = self.yaw


    def get_T1_R1(self):
        # 返回保存的T1和R1
        return self.T1[0],self.yaw

class take_listerer:
    def __init__(self):
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/px4ctrl/takeoff_land', TakeoffLand, self.pose_cb)
        # 用于保存T1和R1
        # 触发条件的标志

    def pose_cb(self, msg):
        global dog_yaw0
        global dog_vins_p0
        global dog_p0
        global dog_vins_yaw0
        global gos_p
        global gos_yaw 
        global vins_p 
        global vins_yaw
        dog_vins_p0 = vins_p
        dog_yaw0 = gos_yaw
        dog_p0 = gos_p
        dog_vins_yaw0 = 0
        
        print("坐标映射已完成")
        print("dog_yaw0:",dog_yaw0)
        
class flow_listerer:
    def __init__(self):
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/flow_data', Odometry, self.pose_cb)
        self.last_flow_z = 0

    def pose_cb(self, msg):
        global flow_z
        flow_z = msg.pose.pose.position.z*0.8 + self.last_flow_z*0.2
        self.last_flow_z = flow_z
             
class AOA_TAG:
    def __init__(self):
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('AOA_Tag_data', Odometry, self.pose_cb0)

        self.vel_estimator = VelocityEstimator()
        self.r_vel_estimator = VelocityEstimator()
        self.sub_count = 0
        self.est_v = np.array([0.0,0.0,0.0])
        self.est_r_v = np.array([0.0,0.0,0.0])
        self.est_pos = np.array([0.0,0.0,0.0])
        self.est_pos1 = np.array([0.0,0.0,0.0])
        self.est_pos2 = np.array([0.0,0.0,0.0])
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb0(self, msg):
        global AOA_distance
        global AOA_angle
        global vins_p
        global target_p
        global flow_z
        AOA_distance = msg.pose.pose.position.x
        AOA_angle = msg.pose.pose.orientation.w 
        self.sub_count += 1
        # print("[AOA]AOA_distance:")
        if self.sub_count > 10:
            self.sub_count = 0
            self.est_v = self.vel_estimator.update(target_p - vins_p,time.time())
            r = (max(AOA_distance**2 - (1)**2,0.5))**0.5
            self.est_r_v = self.r_vel_estimator.update(np.array([r,0,0]),time.time())
            v_norm = np.linalg.norm(self.est_v[:2])
            # print("v_norm",target_p - vins_p,v_norm)
            if v_norm > 0.1 and v_norm > abs(self.est_r_v[0]):
                theta1 = math.atan2(self.est_v[1],self.est_v[0])
                theta2 = math.acos(self.est_r_v[0]/v_norm)
                self.est_pos1[0] = r*math.cos(theta1-theta2) + vins_p[0]
                self.est_pos1[1] = r*math.sin(theta1-theta2) + vins_p[1]
                self.est_pos1[2] = 1
                self.est_pos2[0] = r*math.cos(theta1+theta2) + vins_p[0]
                self.est_pos2[1] = r*math.sin(theta1+theta2) + vins_p[1]
                self.est_pos2[2] = 1
                if (np.linalg.norm(self.est_pos1[:2] - (target_p[:2])) < np.linalg.norm(self.est_pos2[:2] - (target_p[:2]))):
                    self.est_pos = self.est_pos1
                else:
                    self.est_pos = self.est_pos2
                print("[est]est_pos:",self.est_pos)
                print("[est]error_norm:",np.linalg.norm(self.est_pos[:2] - (target_p[:2])))

class TrajectoryVisualizer:
    def __init__(self, frame_id="world"):
        """
        初始化轨迹可视化工具
        参数:
            frame_id: 坐标系名称 (默认 "world")
        """
        self.frame_id = frame_id
        self.publishers = {}  # 存储发布器 {topic_name: publisher}

    def visualize_traj(self, position_array, dt, topic="traj"):
        """
        发布轨迹到 RViz
        参数:
            position_array: 轨迹点数组，形状为 (N, 3) 的 numpy 数组，每行代表 [x, y, z]
            dt: 时间间隔 (秒)
            topic: 发布话题的基础名称 (路径发布到 topic，路点发布到 topic_wayPts)
        """
        # 1. 发布路径消息
        path_msg = self._build_path_msg(position_array)
        path_v_msg = self._build_path_v_msg(position_array)
        self._get_publisher(topic, Path).publish(path_msg)
        self._get_publisher("/traj_v", Path).publish(path_v_msg)
        
        # 2. 发布路点消息
        waypoints_msg = self._build_waypoints_msg(position_array)
        self._get_publisher(f"{topic}_wayPts", PointCloud2).publish(waypoints_msg)

    def _build_path_msg(self, positions):
        """构建 nav_msgs/Path 消息"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        # 直接遍历位置点
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            path.poses.append(pose)
        
        return path

    def _build_path_v_msg(self, positions):
        """构建 nav_msgs/Path 消息"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        # 直接遍历位置点
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[3]
            pose.pose.position.y = pt[4]
            pose.pose.position.z = pt[5]
            path.poses.append(pose)
        
        return path

    def _build_waypoints_msg(self, positions):
        """构建 sensor_msgs/PointCloud2 消息"""
        cloud = PointCloud2()
        cloud.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        # 设置点云字段
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # 直接使用所有位置点作为路点
        points = positions.flatten().astype(np.float32)
        cloud.data = points.tobytes()
        cloud.fields = fields
        cloud.height = 1
        cloud.width = positions.shape[0]
        cloud.point_step = 12  # 3*float32
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_bigendian = False
        cloud.is_dense = True
        
        return cloud

    def _get_publisher(self, topic, msg_type):
        """获取或创建发布器"""
        if topic not in self.publishers:
            self.publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=10)
        return self.publishers[topic]

def traj_get_state(x_opt, u_opt, future_time, dt):
    """
    输出未来某时刻的无人机状态（使用二次插值）
    参数:
        x_opt: MPC预测的状态轨迹 (N+1 x 6) [px,py,pz,vx,vy,vz]
        u_opt: MPC预测的控制输入 (N x 4) [ax,ay,az,yaw_rate]
        future_time: 未来时间偏移量（秒）
        dt: MPC时间步长
    """
    # 计算步数（可能为小数）
    steps = future_time / dt
    
    # 获取整数和小数部分
    idx_base = int(np.floor(steps))
    alpha = steps - idx_base  # 0 <= alpha < 1
    
    # 确保不越界
    idx_base = min(max(0, idx_base), x_opt.shape[0]-1)  # 需要至少3个点做二次插值
    
    # 取三个邻近点
    x0 = x_opt[idx_base]
    x1 = x_opt[idx_base + 1]
    
    # 二次插值公式（抛物线插值）
    state = (1 - alpha)*x0 + alpha*x1
    
    # 加速度提取（使用二次插值后的加速度分量）
    # 注意：这里假设控制输入的时间与状态轨迹对齐（控制输入作用于状态点之间）
    # 使用二次插值后的加速度需要更复杂的处理，这里简化处理为最近邻插值
    accel_idx = min(idx_base, u_opt.shape[0]-1)
    accel = u_opt[accel_idx, :3]
    
    return state, accel


def merge_trajectory(old_x_opt, new_x_opt, now,shift, mpc_N, dt):
    shift_idx = int((shift + now) / dt)
    now_idx = int(now / dt)

    x_head = old_x_opt[now_idx:shift_idx] if old_x_opt is not None and len(old_x_opt) > shift_idx else np.empty((0, new_x_opt.shape[1]))
    # u_head = old_u_opt[now_idx:shift_idx] if old_u_opt is not None and len(old_u_opt) > shift_idx else np.empty((0, new_u_opt.shape[1]))

    remain_x = 1 + mpc_N - x_head.shape[0]
    # remain_u = mpc_N - u_head.shape[0]

    x_tail = new_x_opt[:remain_x]
    # u_tail = new_u_opt[:remain_u]

    x_opt = np.concatenate((x_head, x_tail), axis=0)
    # u_opt = np.concatenate((u_head, u_tail), axis=0)

    return x_opt

def predict_target_trajectory(pos, vel, N, dt):
    """生成移动平台的参考轨迹"""
    return np.array([np.concatenate([pos + vel * t * dt, vel]) for t in range(N+1)])

class target:
    def __init__(self):
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('target_ekf_odom', Odometry, self.pose_cb0)
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb0(self, msg):
        global stop_triger
        global target_ekf_odom
        global read_triger
        global target_p
        # stop_triger = 1
        target_ekf_odom[0] = msg.pose.pose.position.x
        target_ekf_odom[1] = msg.pose.pose.position.y
        target_ekf_odom[2] = msg.pose.pose.position.z
        if read_triger == 1:
            if abs(target_p[0] - target_ekf_odom[0]) < 0.6 and abs(target_p[1] - target_ekf_odom[1]) < 0.6:
                stop_triger = 1
            else:
                print("识别不到目标")

def get_cmd_yaw(target_pxy, vins_pxy,vins_yaw):
    # 获取无人机和目标点的xy坐标
    x1, y1 = vins_pxy
    x2, y2 = target_pxy
    
    # 计算两点之间的差值
    dx = x2 - x1
    dy = y2 - y1
    
    # 使用atan2计算角度，返回值单位为弧度，转换为角度
    target_yaw = math.atan2(dy, dx)
    # angle_deg = math.degrees(angle_rad)

    yaw  = vins_yaw
    
    angle_diff = math.atan2(math.sin(target_yaw - vins_yaw), math.cos(target_yaw - vins_yaw))
    yaw = 0.2*angle_diff + vins_yaw
    # print("vins_yaw:",vins_yaw,"target_yaw:",target_yaw,"yaw:",yaw)
    
    return yaw

def triangulate(d_A, d_B, d_AB, dog_yaw):
    """
    计算 A, B 的相对基站的坐标
    :param d_A: 基站到 A 的距离
    :param d_B: 基站到 B 的距离
    :param d_AB: A 和 B 之间的固定距离
    :param theta: 向量 AB 相对于 x 轴的角度 (弧度制)
    :return: A 和 B 的坐标
    """
    def f(theta,d_A,d_B,d_AB):
        return (d_B**2 - (d_A * math.cos(theta))**2)**0.5 - d_A * math.sin(theta) - d_AB

    if (d_A < d_B):
        best_theta = 0
        if (abs(d_B - d_A) > d_AB):
            best_theta = 0
            print("warning: d_B - d_A too long.")
        elif (abs(d_B**2 - d_A**2) < d_AB):
            best_theta = 3.14159/2
            print("warning: d_AB too long.")
        else:
            # 定义搜索范围（0 到 90 度）
            bracket = (0, math.pi / 2)
            # 使用 root_scalar 求解
            result = root_scalar(f, args=(d_A, d_B, d_AB), bracket=bracket, method='brentq')
            best_theta = result.root
        x_A = -d_A*math.sin(best_theta) - 0.5*d_AB
        y_A = -d_A*math.cos(best_theta)
        # print(best_theta)
        # theta = dog_yaw + np.pi / 2  # 旋转角度
        # x_new = x_A * np.cos(theta) - y_A * np.sin(theta)
        # y_new = x_A * np.sin(theta) + y_A * np.cos(theta)
        return x_A, y_A

    else:
        d_C = d_A
        d_A = d_B
        d_B = d_C
        best_theta = 0
        if (abs(d_B - d_A) > d_AB):
            best_theta = 0
            print("warning: d_B - d_A too long.")
        # elif (abs(d_B**2 - d_A**2) < d_AB):
        #     best_theta = 3.14159/2
            # print("warning: d_AB too long.")
        else:
            # 定义搜索范围（0 到 90 度）
            bracket = (- math.pi / 2, math.pi / 2)
            # 使用 root_scalar 求解
            result = root_scalar(f, args=(d_A, d_B, d_AB), bracket=bracket, method='brentq')
            best_theta = result.root
        x_A = -d_A*math.sin(best_theta) - 0.5*d_AB
        y_A = -d_A*math.cos(best_theta)
        
        # print(best_theta)
        # theta = dog_yaw - np.pi / 2  # 旋转角度
        # x_new = x_A * np.cos(theta) - y_A * np.sin(theta)
        # y_new = x_A * np.sin(theta) + y_A * np.cos(theta)
        return x_A, y_A

def find_coordinates(OA, OB, L, theta):
    """
    求解点A和点B的坐标。

    参数:
    theta (float): 线段AB与x轴的夹角（弧度制）。
    L (float): 线段AB的长度。
    OA (float): OA的长度。
    OB (float): OB的长度。

    返回:
    (x1, y1), (x2, y2): 点A和点B的坐标。
    """

    if OA + OB < L or OA + L < OB or OB + L < OA:
        # raise ValueError("输入的参数不满足三角形不等式，无解。")
        print("OA + OB < L or OA + L < OB or OB + L < OA",OA,OB,L)
        return [((0,0),(0,0)),((0,0),(0,0))]
    # 计算斜率
    slope = math.tan(theta)

    # 计算 (x2 - x1) 的可能值
    delta_x = L / math.sqrt(1 + slope**2)

    # 考虑两种可能的情况（正负）
    solutions = []
    for sign in [1, -1]:
        dx = sign * delta_x
        dy = slope * dx

        # 根据 OA 和 OB 的长度求解坐标
        # 设 A 的坐标为 (x1, y1)，B 的坐标为 (x1 + dx, y1 + dy)
        # 根据 OA 的长度：x1^2 + y1^2 = OA^2
        # 根据 OB 的长度：(x1 + dx)^2 + (y1 + dy)^2 = OB^2

        # 解方程组
        # 设 A 的坐标为 (x1, y1)
        # 代入 B 的坐标：(x1 + dx)^2 + (y1 + dy)^2 = OB^2
        # 展开并利用 x1^2 + y1^2 = OA^2
        # 得到：2*x1*dx + 2*y1*dy + dx^2 + dy^2 = OB^2 - OA^2

        # 设常数项
        C = (OB**2 - OA**2 - L**2) / 2

        discriminant = OA**2 - (C / dy)**2
        # if discriminant < 0:
        #     print("discriminant",discriminant)
        #     return [((0,0),(0,0)),((0,0),(0,0))]

        # 代入 y1 = (C - x1*dx) / dy
        if dy != 0 and discriminant > 0:
            y1 = (C - dx * math.sqrt(OA**2 - (C / dy)**2)) / dy
            x1 = math.sqrt(OA**2 - y1**2)
        else:
            # 如果 dy = 0，AB 平行于 x 轴
            x1 = C / dx
            y1 = math.sqrt(OA**2 - x1**2)

        # 计算 B 的坐标
        x2 = x1 + dx
        y2 = y1 + dy

        # 检查 OB 的长度是否满足
        # if math.isclose(math.sqrt(x2**2 + y2**2), OB, rel_tol=1e-6):
        solutions.append([(x1, y1), (x2, y2)])

    # 返回所有可能的解
    return solutions

def eight_mission(t,p_z):
    # 定义8字形轨迹的参数
    radius = 1.5  # 圆的半径
    T = 10  # 总周期T
    angular_velocity = 2 * math.pi / T  # 角速度，单位弧度/秒
    total_time = 2 * T  # 完整运动的总时间，两圈
    total_time = 2*T
    finish = False
    p = [0,0,0]
    if t >= total_time:
        finish = True

    def get_position(t):
        """
        根据时间t，计算无人机的当前位置，t为时间（秒），返回(x, y)坐标
        """
        # 计算当前角度
        angle = angular_velocity * t
        angle -= math.pi
        # 确保angle在0到2pi范围内
        # angle = angle % (2 * math.pi)
        
        # 计算轨迹的x, y坐标
        if angle <= 0:
            # 第一部分轨迹，沿着第一个圆
            x = radius + radius * math.cos(angle)
            y = -radius * math.sin(angle)
        elif angle <= 2*math.pi:
            # 第二部分轨迹，沿着第二个圆
            # 在第二个圆中，角度应该调整为相对于第一个圆的偏移
            # angle -= math.pi
            x = radius * 3 - radius * math.cos(angle)
            y = -radius * math.sin(angle)
        else :
            angle -= 2*math.pi
            x = radius + radius * math.cos(angle)
            y = -radius * math.sin(angle)
        return x, y

    def get_yaw(t):
        yaw = 0
        if t < T/2:
            yaw = -math.pi/2 / (T/2) * t
        elif t <= 3*T/2:
            yaw = - math.pi/2 + (2*math.pi)/(T)*(t - T/2)
        else:
            yaw = 1.5*math.pi - (math.pi)/(T/2)*(t - 3*T/2)
        return yaw    
    
    p.x,p.y = get_position(t)
    p.z = p_z
    yaw = get_yaw(t)
    return finish,p,yaw

vis = TrajectoryVisualizer()

def MPC_mission():
    global x_opt,u_opt
    global MPC_clcyle
    global vins_p, vins_v
    global shift
    global vis
    target_p = np.array([20,20,3])
    target_v = np.array([0,0,0])
    if (x_opt is not None and u_opt is not None):
        drone_state,accel = traj_get_state(x_opt, u_opt, int((time.time() - MPC_clcyle + shift)/mpc.dt)*mpc.dt, mpc.dt)
        # drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
    else :
        drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
        # drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
    # print("time.time() - MPC_clcyle:",time.time() - MPC_clcyle)
    # print("drone_state:",drone_state)
    # print("x_opt:",x_opt)
    # print("drone_state:" , drone_state)
    target_traj = predict_target_trajectory(target_p, target_v, mpc.N, mpc.dt)
    start_time = time.time()
    new_x_opt, new_u_opt = mpc.solve(drone_state, target_traj)
    end_time = time.time()
    print("MPC solve time: " , end_time - start_time)

    x_opt = merge_trajectory(x_opt, new_x_opt, now=time.time() - MPC_clcyle ,shift=shift, mpc_N=mpc.N, dt=mpc.dt)
    # x_opt = new_x_opt
    u_opt = new_u_opt
    MPC_clcyle = time.time() 
    # Traj_splines.update(x_opt,u_opt)
    vis.visualize_traj(x_opt, mpc.dt, topic="/drone2/planning/traj")
    return 
    

if __name__ == '__main__':

    # 初始化ROS节点
    rospy.init_node('position_command_publisher', anonymous=True)
    flag = True
    uav_state_listener = UAVStateListener(flag)
    gos_listener = dog_listerer(flag)
    # takeoff_listerer = take_listerer()
    a_flow_listerer = flow_listerer()

    AOA_listener = AOA_TAG()
    target_listener = target()
    
    time.sleep(1)

    # number = input("是否开始执行任务：")
    # 读取文件内容
    with open("coordinate.txt", "r") as f:
        lines = f.readlines()
    # 去掉换行符和空格
    lines = [line.strip() for line in lines if line.strip()]

    # 提取并转换变量
    dog_vins_p0 = np.array([float(x) for x in lines[0].split(',')])
    dog_yaw0 = float(lines[1])
    dog_p0 = np.array([float(x) for x in lines[2].split(',')])
    dog_vins_yaw0 = float(lines[3])
    print("坐标映射已完成!")
    # print("dog_yaw0:",dog_yaw0)

    T1,_yaw  = uav_state_listener.get_T1_R1()
    # 创建发布器
    pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)
    triger_pub = rospy.Publisher('/triger', PoseStamped , queue_size=1)
    beacon_pub = rospy.Publisher('/beacon', Odometry , queue_size=1)
    # triangulation_pub = rospy.Publisher("/uwb/triangulation", PointStamped, queue_size=10)
    # 示例位置数据（geometry_msgs/Point类型）
    p = Point()
    p.x = T1[0]  # 设置X坐标
    p.y = T1[1]  # 设置Y坐标
    p.z = T1[2] # 设置Z坐标
    traj_id = 1
    rate = rospy.Rate(100)

      # 设置发布频率为100Hz
    T2,_yaw  = uav_state_listener.get_T1_R1()
    p.x = T2[0]  # 设置X坐标
    p.y = T2[1]  # 设置Y坐标
    p.z = T2[2] # 设置Z坐标

    init_p = T2

    start_time = rospy.get_time()
    print("开始任务")
    mission_finish = False
    while mission_finish is False:
        MPC_mission()
        if(x_opt is not None and abs(vins_p[0] - 20) > 1 and abs(vins_p[1] - 20) > 1 or np.linalg.norm(vins_v) > 0.3):
            p.x = x_opt[2][0]
            p.y = x_opt[2][1]
            p.z = x_opt[2][2] 
            publish_position(p, traj_id, v_x = x_opt[2][3], v_y = x_opt[2][4], v_z = x_opt[2][5], yaw = 0)
            print("当前位置：")
        else:
            p.x = 20
            p.y = 20
            p.z = 3
            publish_position(p, traj_id, v_x = 0, v_y = 0, v_z = 0, yaw = 0)
            mission_finish = True
            print("任务完成")
        time.sleep(0.10)
        if(x_opt is not None and abs(vins_p[0] - 20) > 1 and abs(vins_p[1] - 20) > 1 or np.linalg.norm(vins_v) > 0.3):
            p.x = x_opt[3][0]
            p.y = x_opt[3][1]
            p.z = x_opt[3][2] 
            publish_position(p, traj_id, v_x = x_opt[3][3], v_y = x_opt[3][4], v_z = x_opt[3][5], yaw = 0)
        else:
            p.x = 20
            p.y = 20
            p.z = 3
            publish_position(p, traj_id, v_x = 0, v_y = 0, v_z = 0, yaw = 0)
            mission_finish = True
            print("任务完成")
        time.sleep(0.10)
        
        #mission_finish,p,eight_yaw = eight_mission(rospy.get_time() - start_time,T2[2])
    #   eight_yaw = get_yaw(rospy.get_time() - start_time)
    #   rate.sleep()
    time.sleep(5)

    # rospy.loginfo("Finished publishing position commands for 2*T time period.")

    # for i in range(50):
    #     p.x = vins_p[0]
    #     p.y = vins_p[1]
    #     p.z = min(vins_p[2] + 0.5 , 1.0)
    #     publish_position(p, traj_id ,0,0,0, vins_yaw)
    #     rate.sleep()

    somethinggggg = "n"
    if somethinggggg == "y":
        look_point = vins_p
        for i in range(500):
            p.x = look_point[0]
            p.y = look_point[1]
            p.z = 1.5
            yaw = i * ((2 * math.pi) / 500)
            publish_position(p, traj_id ,v_x=0,v_y=0,v_z=0, yaw=yaw)
            rate.sleep()
            msg_beacon = Odometry()
            msg_beacon.header.stamp = rospy.Time.now()
            msg_beacon.header.frame_id = "world"
            AOA_distancexy = (AOA_distance**2 - (max(flow_z -0.47,0))**2)**0.5
            msg_beacon.pose.pose.position.x = float(AOA_distancexy*math.cos(target_yaw + AOA_angle) + vins_p[0])
            msg_beacon.pose.pose.position.y = float(AOA_distancexy*math.sin(target_yaw + AOA_angle) + vins_p[1])
            beacon_pub.publish(msg_beacon)
            # if i % 100 == 2:
            #     print("target_p:")

    # for i in range(100):
    #     x1, y1 = vins_pxy
    #     x2, y2 = target_pxy
    #     # 计算两点之间的差值
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     # 使用atan2计算角度，返回值单位为弧度，转换为角度
    #     yaw = math.atan2(dy, dx) 
    #     p.x = vins_p[0]
    #     p.y = vins_p[1]
    #     p.z = 0.5*vins_p[2] + 0.5*1.5
    #     publish_position(p, traj_id ,0,0,0, 0)
    #     rate.sleep()

    vins_p_now = np.array([0,0,0])
    vins_p_now[0] = vins_p[0]
    vins_p_now[1] = vins_p[1]
    vins_p_now[2] = vins_p[2]

    for i in range(100):
        yaw = get_cmd_yaw(np.array([target_p[0], target_p[1]]), np.array([vins_p[0], vins_p[1]]), vins_yaw)
        p.x = vins_p_now[0]
        p.y = vins_p_now[1]
        p.z = vins_p_now[2]
        publish_position(p, traj_id ,0,0,0, yaw=yaw)
        rate.sleep()
    
    r_p = gos_p - dog_p0
    r_yaw = dog_vins_yaw0 - dog_yaw0
    cos_theta = math.cos(r_yaw)
    sin_theta = math.sin(r_yaw)
    
    rotated_x = cos_theta * r_p[0] - sin_theta * r_p[1]
    rotated_y = sin_theta * r_p[0] + cos_theta * r_p[1]
    r_p[0] = rotated_x
    r_p[1] = rotated_y
    target_p = r_p + dog_vins_p0

    target_yaw = gos_yaw - dog_yaw0 + dog_vins_yaw0
    tracking_d = 1.5


    vins_pxy = np.array([vins_p[0],vins_p[1]])
    target_pxy = np.array([target_p[0] - tracking_d*math.cos(target_yaw),target_p[1] - tracking_d*math.sin(target_yaw)])
    stop_triger = 0
    # triger_listener = triger_listerer()
    
    last_count = 0

    print("开始返航")
    print("target_p:",target_p)

    unsure_target_pxy = np.array([target_p[0] - 4*math.cos(target_yaw),target_p[1] - 4*math.sin(target_yaw)])
    # tracking to a suitable distance for over-mapping
    if (np.linalg.norm(vins_p - target_p) >= 5):
        print("back from >5m")
        while np.linalg.norm(unsure_target_pxy - vins_pxy) > 0.4 and last_count < 80:
            if (np.linalg.norm(unsure_target_pxy - vins_pxy) < 0.4):
                last_count += 1
            # break
            if np.linalg.norm(unsure_target_pxy - vins_pxy) > 1:
                p.x = 1.4*(unsure_target_pxy[0] - vins_pxy[0])/np.linalg.norm(unsure_target_pxy - vins_pxy) + vins_pxy[0]
                p.y = 1.4*(unsure_target_pxy[1] - vins_pxy[1])/np.linalg.norm(unsure_target_pxy - vins_pxy) + vins_pxy[1]
                p.z = max(vins_p[2],1.5 + target_p[2])
                v_x = 1.3*(unsure_target_pxy[0] - vins_pxy[0])/np.linalg.norm(unsure_target_pxy - vins_pxy)
                v_y = 1.3*(unsure_target_pxy[1] - vins_pxy[1])/np.linalg.norm(unsure_target_pxy - vins_pxy)
                v_z = max(min(0.4 * (1.5 + target_p[2] - vins_p[2]),0.5),-0.5)
            else:
                p.x = unsure_target_pxy[0]
                p.y = unsure_target_pxy[1]
                p.z = max(vins_p[2],1.5 + target_p[2])
                v_x = 1.3 * (unsure_target_pxy[0] - vins_p[0])
                v_y = 1.3 * (unsure_target_pxy[1] - vins_p[1])
                v_z = max(min(0.4 * (1.5 + target_p[2] - vins_p[2]),0.5),-0.5)

            yaw = get_cmd_yaw(np.array([target_p[0],target_p[1]]),vins_pxy,vins_yaw)
            publish_position(p, traj_id , v_x ,v_y,v_z, yaw)
            vins_pxy = np.array([vins_p[0],vins_p[1]])
            gos_pxy = np.array([gos_p[0],gos_p[1]])

            r_p = gos_p - dog_p0
            r_yaw = dog_vins_yaw0 - dog_yaw0
            cos_theta = math.cos(r_yaw)
            sin_theta = math.sin(r_yaw)
            
            rotated_x = cos_theta * r_p[0] - sin_theta * r_p[1]
            rotated_y = sin_theta * r_p[0] + cos_theta * r_p[1]
            r_p[0] = rotated_x
            r_p[1] = rotated_y
            target_p = r_p + dog_vins_p0
            target_p[2] = vins_p[2] - (flow_z - 0.47)
            target_yaw = gos_yaw - dog_yaw0 + dog_vins_yaw0
            target_pxy = np.array([target_p[0] - tracking_d*math.cos(target_yaw),target_p[1] - tracking_d*math.sin(target_yaw)])
            unsure_target_pxy = np.array([target_p[0] - 4*math.cos(target_yaw),target_p[1] - 4*math.sin(target_yaw)])

            msg_beacon = Odometry()
            msg_beacon.header.stamp = rospy.Time.now()
            msg_beacon.header.frame_id = "world"
            msg_beacon.pose.pose.position.x = float(target_p[0])
            msg_beacon.pose.pose.position.y = float(target_p[1])
            msg_beacon.pose.pose.position.z = float(target_p[2])
            msg_beacon.pose.pose.orientation.w = float(target_yaw)
            msg_beacon.twist.twist.linear.x = 1
            beacon_pub.publish(msg_beacon)

            rate.sleep()
    else:
        print("back from <5m")
        while np.linalg.norm(target_pxy - vins_pxy) > 0.4  and last_count < 80:
            if (np.linalg.norm(target_pxy - vins_pxy) < 0.4):
                last_count += 1

            if np.linalg.norm(target_pxy - vins_pxy) > 1:
                p.x = 0.8*(target_pxy[0] - vins_pxy[0])/np.linalg.norm(target_pxy - vins_pxy) + vins_pxy[0]
                p.y = 0.8*(target_pxy[1] - vins_pxy[1])/np.linalg.norm(target_pxy - vins_pxy) + vins_pxy[1]
                p.z = max(vins_p[2] - 0.15,target_p[2] +0.05)
                v_x = 1*(target_pxy[0] - vins_pxy[0])/np.linalg.norm(target_pxy - vins_pxy)*0.4 + vins_v[0]*0.6
                v_y = 1*(target_pxy[1] - vins_pxy[1])/np.linalg.norm(target_pxy - vins_pxy)*0.4 + vins_v[1]*0.6
                v_z = max(min(0.4 * (target_p[2] +0.05 - vins_p[2]),0.3),-0.2)
            else:
                p.x = target_pxy[0]
                p.y = target_pxy[1]
                p.z = max(vins_p[2]-0.15,target_p[2] +0.05)
                v_x = 1.1 * (target_pxy[0] - vins_p[0])*0.4 + vins_v[0]*0.6
                v_y = 1.1 * (target_pxy[1] - vins_p[1])*0.4 + vins_v[1]*0.6
                v_z = max(min(0.4 * (target_p[2] +0.05 - vins_p[2]),0.3),-0.2)

                # if abs(target_pxy[0] - vins_p[0])<=0.4 and abs(target_pxy[1] - vins_p[1])<=0.4 and abs(target_p[2] + 0.6 - vins_p[2])<=0.5 and read_triger == 0:
                #     # run_shell_script(script_path)
                #     read_triger = 1
                #     # print("开始read")
            yaw = get_cmd_yaw(np.array([target_p[0],target_p[1]]),vins_pxy,vins_yaw)
            publish_position(p, traj_id , v_x ,v_y,v_z, yaw)
            vins_pxy = np.array([vins_p[0],vins_p[1]])
            gos_pxy = np.array([gos_p[0],gos_p[1]])

            r_p = gos_p - dog_p0
            r_yaw = dog_vins_yaw0 - dog_yaw0
            cos_theta = math.cos(r_yaw)
            sin_theta = math.sin(r_yaw)
            
            rotated_x = cos_theta * r_p[0] - sin_theta * r_p[1]
            rotated_y = sin_theta * r_p[0] + cos_theta * r_p[1]
            r_p[0] = rotated_x
            r_p[1] = rotated_y
            target_p = r_p + dog_vins_p0
            target_p[2] = vins_p[2] - (flow_z - 0.47)
            target_yaw = gos_yaw - dog_yaw0 + dog_vins_yaw0
            target_pxy = np.array([target_p[0] - tracking_d*math.cos(target_yaw),target_p[1] - tracking_d*math.sin(target_yaw)])
            unsure_target_pxy = np.array([target_p[0] - 4*math.cos(target_yaw),target_p[1] - 4*math.sin(target_yaw)])

            msg_beacon = Odometry()
            msg_beacon.header.stamp = rospy.Time.now()
            msg_beacon.header.frame_id = "world"
            msg_beacon.pose.pose.position.x = target_p[0]
            msg_beacon.pose.pose.position.y = target_p[1]
            msg_beacon.pose.pose.position.z = target_p[2]
            msg_beacon.pose.pose.orientation.w = float(target_yaw)
            msg_beacon.twist.twist.linear.x = -1
            beacon_pub.publish(msg_beacon)

    print("back by AOA")
    while np.linalg.norm(target_pxy - vins_pxy) > 1 or True:
        if np.linalg.norm(target_pxy - vins_pxy) > 2:
            p.x = 1.4*(target_pxy[0] - vins_pxy[0])/np.linalg.norm(target_pxy - vins_pxy) + vins_pxy[0]
            p.y = 1.4*(target_pxy[1] - vins_pxy[1])/np.linalg.norm(target_pxy - vins_pxy) + vins_pxy[1]
            p.z = max(vins_p[2] - 0.15,0.05 + target_p[2])
            v_x = 1.2*(target_pxy[0] - vins_pxy[0])/np.linalg.norm(target_pxy - vins_pxy)*0.5 + vins_v[0]*0.5
            v_y = 1.2*(target_pxy[1] - vins_pxy[1])/np.linalg.norm(target_pxy - vins_pxy)*0.5 + vins_v[1]*0.5
            v_z = max(min(0.4 * (0.05 + target_p[2] - vins_p[2]),0.3),-0.2)
        else:
            p.x = target_pxy[0]
            p.y = target_pxy[1]
            p.z = max(vins_p[2]-0.15,target_p[2] + 0.05)
            v_x = 1.2 * (target_pxy[0] - vins_p[0])*0.5 + vins_v[0]*0.5
            v_y = 1.2 * (target_pxy[1] - vins_p[1])*0.5 + vins_v[1]*0.5
            v_z = max(min(0.4 * (target_p[2] + 0.05 - vins_p[2]),0.3),-0.2)

            if abs(target_pxy[0] - vins_p[0])<=0.3 and abs(target_pxy[1] - vins_p[1])<=0.3 and abs(target_p[2] + 0.05 - vins_p[2])<=0.2 and read_triger == 0:
                # run_shell_script(script_path)
                read_triger = 1
                # print("开始read")
        yaw = get_cmd_yaw(np.array([target_p[0],target_p[1]]),vins_pxy,vins_yaw)
        publish_position(p, traj_id , v_x ,v_y,v_z, yaw)
        vins_pxy = np.array([vins_p[0],vins_p[1]])
        gos_pxy = np.array([gos_p[0],gos_p[1]])

        r_p = gos_p - dog_p0
        r_yaw = dog_vins_yaw0 - dog_yaw0
        cos_theta = math.cos(r_yaw)
        sin_theta = math.sin(r_yaw)
        
        rotated_x = cos_theta * r_p[0] - sin_theta * r_p[1]
        rotated_y = sin_theta * r_p[0] + cos_theta * r_p[1]
        r_p[0] = rotated_x
        r_p[1] = rotated_y
        target_p = r_p + dog_vins_p0
        target_p[2] = vins_p[2] - (flow_z - 0.47)
        target_yaw = gos_yaw - dog_yaw0 + dog_vins_yaw0
        calibration_count += 1
        if (calibration_count > 100 and AOA_distance**2 - (max(flow_z -0.47,0))**2 > 0):
            AOA_distancexy = (AOA_distance**2 - (max(flow_z -0.47,0))**2)**0.5
            target_p = np.array([AOA_distancexy*math.cos(target_yaw + AOA_angle) + vins_p[0],AOA_distancexy*math.sin(target_yaw + AOA_angle) + vins_p[1],target_p[2]])

            print("delta_r:",r_p + dog_vins_p0 - target_p,"error:",np.linalg.norm(r_p + dog_vins_p0 - target_p))
            dog_vins_p0 = target_p
            dog_p0 = gos_p
            dog_yaw0 = gos_yaw
            dog_vins_yaw0 = target_yaw
            calibration_count = 0

        target_pxy = np.array([target_p[0] - tracking_d*math.cos(target_yaw),target_p[1] - tracking_d*math.sin(target_yaw)])
        unsure_target_pxy = np.array([target_p[0] - 4*math.cos(target_yaw),target_p[1] - 4*math.sin(target_yaw)])



        msg_beacon = Odometry()
        msg_beacon.header.stamp = rospy.Time.now()
        msg_beacon.header.frame_id = "world"
        msg_beacon.pose.pose.position.x = float(target_p[0])
        msg_beacon.pose.pose.position.y = float(target_p[1])
        msg_beacon.pose.pose.position.z = float(target_p[2])
        msg_beacon.pose.pose.orientation.w = float(target_yaw)
        AOA_distancexy = (max(AOA_distance**2 - (max(flow_z -0.47,0)),0)**2)**0.5
        msg_beacon.pose.pose.orientation.x = float(AOA_distancexy*math.cos(target_yaw + AOA_angle) + vins_p[0])
        msg_beacon.pose.pose.orientation.y = float(AOA_distancexy*math.sin(target_yaw + AOA_angle) + vins_p[1])

        msg_beacon.twist.twist.linear.x = 2

        beacon_pub.publish(msg_beacon)

        
        if stop_triger == 1 and read_triger == 1:
            print("返航已完成")
            msg1 = PoseStamped()

            # 设置时间戳和参考框架
            msg1.header.stamp = rospy.Time.now()
            msg1.header.frame_id = "world"  # 参考坐标系为world
            # 发布命令
            triger_pub.publish(msg1)

            break

        rate.sleep()




