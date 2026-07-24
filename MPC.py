# -*- coding: utf-8 -*-
"""
============================================================================
MPC.py — 模型预测控制轨迹规划器
============================================================================

【在整个系统中的角色】
  本模块是规划管道,位于 dog_pos_processor (狗位姿对齐)和 traj_server (轨迹跟踪)
  之间。它接收处理后的目标位姿和无人机 VINS 状态,通过 MPC(Model Predictive Control)
  求解最优轨迹,发布给下游 traj_server 跟踪执行。

【核心管道】
  VINS 里程计 (/vins_fusion/imu_propagate) ─┐
  狗处理后位姿 (/dog_pos_processed)         ─┤
                                               ├→ [run_while_loop 80Hz]
                                               │    1. 目标轨迹预测 (匀速拟合)
                                               │    2. MPC 轨迹求解 (cvxpy + C++编译模型)
                                               │    3. 轨迹拼接 (新旧衔接)
                                               │    4. 轨迹可视化 (Rviz Path 发布)
                                               │
                                               └→ 发布 /drone2/planning/traj

【MPC 原理简述】
  模型预测控制是一种基于优化模型的控制方法,核心思想是:
    1. 在每个控制周期,求解一个有限时域(N 步)的优化问题
    2. 目标函数: 最小化无人机状态与目标轨迹之间的加权误差
    3. 约束条件: 速度/加速度/输入的上限
    4. 只执行优化结果的第一步,下一周期重新规划 (滚动时域)

  本项目使用 C++ 编译的 MPC 求解器 (DroneMPC),以保证实时性。

【依赖项】
  - numpy (数值计算)
  - cvxpy (优化问题建模,仅用于 MPC 内部)
  - scipy (CubicSpline 插值,备用)
  - rospy (ROS Python 接口)
  - DroneMPC (C++ 编译的 MPC 求解器,来自 MPC_python/MPC_python_complie)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Odometry
import rospy
import math
import time
from math import pi
import threading
from scipy.interpolate import CubicSpline
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point32
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# C++ 编译的 MPC 求解器 Python 绑定
# DroneMPC 内部使用 acados 或 cvxpy 求解最优控制问题
from MPC_python.MPC_python_complie import DroneMPC


# ============================================================================
# 一、全局状态变量 — 各传感器数据缓存
# ============================================================================

target_p = np.array([0.0, 0.0, 0.0])    # 目标在世界系下的位置(来自 /target_ekf_odom)
target_v = np.array([0.0, 0.0, 0.0])    # 目标在世界系下的速度(来自 /target_ekf_odom)
target_yaw = 0.0                          # 目标在世界系下的 yaw 角(来自 /target_ekf_odom)
target_received_ = 0                      # 目标数据是否已收到(>0 表示收到)
triger = 0                                # 控制触发标志(由 /mode_manager 话题设定)
vins_p = np.array([0.0, 0.0, 0.0])      # 无人机世界系位置(来自 VINS)
vins_v = np.array([0.0, 0.0, 0.0])      # 无人机世界系速度(来自 VINS)
vins_yaw = 0                              # 无人机 yaw 角(来自 VINS,弧度)
reset_mpc = 0                             # MPC 重置标志(模式切换时设为 1)

# ===== dog_pos_processor 相关变量 =====
# 这些变量接收经过坐标系对齐处理后的狗位姿(世界系)
dog_pos_p = np.array([0.0, 0.0, 0.0])   # 处理后狗在世界系下的位置
dog_pos_v = np.array([0.0, 0.0, 0.0])   # 处理后狗在世界系下的速度
dog_pos_yaw = 0.0                         # 处理后狗在世界系下的 yaw
dog_pos_acc = np.array([0.0, 0.0])       # 狗加速度(世界系 xy,来自 dog_pos_processor 的角速度估计)
dog_pos_received_ = 0                     # 狗数据是否已收到

# ===== command_pos 相关变量 (triger == 2 时使用) =====
# 来自外部指令(/command_pos 话题), 用于 agent 模式的直接位置控制
command_pos_p = np.array([0.0, 0.0, 0.0])
command_pos_v = np.array([0.0, 0.0, 0.0])
command_pos_yaw = 0.0
command_pos_received_ = 0

last_target_ekf_time = 0.0
hc14_offset_yaw_ready = False             # yaw offset 是否已收敛
hc14_offset_pos_ready = False             # pos offset 是否已收敛

# 目标速度历史记录 — 用于 predict_target_trajectory 中的速度拟合
target_vel_history = []                   # 最近 N 个速度点 [vel_x, vel_y, vel_z]
target_vel_history_times = []             # 对应的时间戳(秒)

# 狗的飞行高度限制区间 [下界, 上界] (m)
# 无人机跟踪时在狗上方 [1.5m, 1.8m] 的高度带内飞行
land_height_limit = [1.5, 1.8]


# ============================================================================
# 二、目标低通滤波器 (TargetFilter)
# ============================================================================

class TargetFilter:
    """
    一阶低通滤波器,用于平滑目标位置和 yaw 的跳变。

    原理:
      filtered = alpha * raw + (1-alpha) * filtered  (指数平滑)
      alpha=0.3 → 约 3 个周期收敛到新值的 70%

    防抖策略:
      当丢包时间超过 reset_interval (0.5s) 时,直接使用新测量值初始化,
      避免长时间丢包后滤波器需要多周期重新收敛。
    """

    def __init__(self, alpha=0.3, reset_interval=0.5):
        self.alpha = alpha                # 滤波系数(越小越平滑,越大响应越快)
        self.reset_interval = reset_interval  # 丢包重置超时(秒)

        self.filtered_pos = np.zeros(3)
        self.filtered_yaw = 0.0
        self.last_update_time = None
        self.initialized = False

    def _wrap_angle(self, angle):
        """角度归一化到 [-pi, pi],避免 yaw 的周期边界跳变"""
        return (angle + pi) % (2 * pi) - pi

    def _angle_lerp(self, a, b, alpha):
        """
        对角度的低通插值(含角度环绕处理)
        关键: 先取最短路径 delta = wrap(b-a),再线性插值
        如果直接做 a + alpha*(b-a), 在 ±π 边界会导致跳变
        """
        delta = self._wrap_angle(b - a)
        return a + alpha * delta

    def update(self, raw_pos, raw_yaw):
        """输入原始测量值,更新滤波状态"""
        now = time.time()
        # 首次初始化 或 超时重置 → 直接用新值初始化
        if not self.initialized or (self.last_update_time is not None
                                    and now - self.last_update_time > self.reset_interval):
            self.filtered_pos = np.array(raw_pos)
            self.filtered_yaw = raw_yaw
            self.initialized = True
        else:
            # 位置: 标准一阶低通
            self.filtered_pos = self.alpha * np.array(raw_pos) + (1 - self.alpha) * self.filtered_pos
            # Yaw: 含角度环绕的低通
            self.filtered_yaw = self._angle_lerp(self.filtered_yaw, raw_yaw, self.alpha)

        self.last_update_time = now

    def get_filtered(self):
        return self.filtered_pos, self.filtered_yaw


# ============================================================================
# 三、ROS 订阅器 (all_Subscriber) — 接收所有传感器和控制数据
# ============================================================================

class all_Subscriber:
    """
    集中管理所有 ROS 订阅,将消息解析为全局变量。
    订阅话题:
      /target_ekf_odom          → 目标视觉估计位姿
      /dog_pos_processed         → 坐标系对齐后的狗位姿
      /vins_fusion/imu_propagate → 无人机 VINS 里程计
      /mode_manager              → 模式切换控制信号
    """

    def __init__(self):
        self.T1 = None       # VINS 位置(世界系)
        self.R1 = None       # VINS 姿态(旋转矩阵)
        self.list = []       # VINS 位置滑动平均窗口
        self.yaw = 0         # VINS yaw 角

        # 订阅: 视觉目标估计 (/target_ekf_odom, 来自 read.cpp)
        self.pose_sub = rospy.Subscriber('/target_ekf_odom', Odometry, self.pose_cb0)
        # 订阅: 坐标系对齐后的狗位姿 (/dog_pos_processed, 来自 dog_pos_processor)
        self.dog_pos_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self.dog_pos_cb)
        # 订阅: 无人机 VINS 里程计
        self.command_pos_sub = rospy.Subscriber('/command_pos', Odometry, self.command_pos_cb)
        self.vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_cb)
        # 订阅: 模式管理器 (来自 traj_server 的模式切换指令)
        self.track_sub = rospy.Subscriber('/mode_manager', PoseStamped, self.mode_cb)

        # 目标低通滤波器实例
        self.target_filter = TargetFilter(alpha=0.3, reset_interval=0.5)

    def pose_cb0(self, msg):
        """
        接收视觉目标位姿 (/target_ekf_odom):
          - position: 目标世界系位置
          - orientation.w: 目标世界系 yaw
          - linear velocity: 目标世界系速度

        注意: 该话题来自 read.cpp 的 ArUco 检测结果,在视觉遮挡或标记丢失时
              可能不更新,此时使用 dog_pos 的数据作为备选。
        """
        global target_v, target_p, target_yaw, target_received_

        # 限制位标志: orientation.x == -1 表示此消息为限制位(不更新目标)
        if msg.pose.pose.orientation.x == -1:
            return

        target_received_ = 1

        # 从四元数的 w 分量直接读取 yaw (约定: orientation.w 存放 yaw 角)
        target_yaw = msg.pose.pose.orientation.w
        # 位置
        target_p[0] = msg.pose.pose.position.x
        target_p[1] = msg.pose.pose.position.y
        target_p[2] = msg.pose.pose.position.z
        # 速度
        target_v[0] = msg.twist.twist.linear.x
        target_v[1] = msg.twist.twist.linear.y
        target_v[2] = msg.twist.twist.linear.z

    def dog_pos_cb(self, msg):
        """
        接收坐标系对齐后的狗位姿 (/dog_pos_processed):
        这是本 MPC 节点实际使用的目标数据! 与 pose_cb0 不同的是,
        这里的数据经过了 dog_pos_processor 的坐标系变换和滤波,
        是狗在世界系下的最终估计位姿。

        消息字段复用约定 (Odom 格式的非常规使用):
          - position: 世界系位置 [x, y, z]
          - linear velocity: 世界系速度(狗坐标系旋转后) [vx, vy, vz]
          - angular.x: 世界系 yaw 角
          - orientation.w: precise_pos_offset_ready (1.0=ready)
          - orientation.x: precise_yaw_offset_ready (1.0=ready)
          - orientation.y: 加速度 x (世界系)
          - orientation.z: 加速度 y (世界系)
        """
        global dog_pos_p, dog_pos_v, dog_pos_yaw, dog_pos_acc
        global dog_pos_received_, hc14_offset_yaw_ready, hc14_offset_pos_ready

        dog_pos_p[0] = msg.pose.pose.position.x
        dog_pos_p[1] = msg.pose.pose.position.y
        dog_pos_p[2] = msg.pose.pose.position.z

        dog_pos_v[0] = msg.twist.twist.linear.x
        dog_pos_v[1] = msg.twist.twist.linear.y
        dog_pos_v[2] = msg.twist.twist.linear.z

        # angular.x 存储了世界系 yaw (dog_pos_processor 的约定)
        dog_pos_yaw = msg.twist.twist.angular.x

        # orientation.w/x 读出 offset ready 状态 (>0.5 表示 ready)
        hc14_offset_pos_ready = (msg.pose.pose.orientation.w > 0.5)
        hc14_offset_yaw_ready = (msg.pose.pose.orientation.x > 0.5)

        # orientation.y/z 读出狗加速度(世界系 xy)
        dog_pos_acc[0] = msg.pose.pose.orientation.y
        dog_pos_acc[1] = msg.pose.pose.orientation.z

        dog_pos_received_ = 1

    def command_pos_cb(self, msg):
        global command_pos_p
        global command_pos_v
        global command_pos_yaw
        global command_pos_received_

        command_pos_p[0] = msg.pose.pose.position.x
        command_pos_p[1] = msg.pose.pose.position.y
        command_pos_p[2] = msg.pose.pose.position.z

        command_pos_v[0] = msg.twist.twist.linear.x
        command_pos_v[1] = msg.twist.twist.linear.y
        command_pos_v[2] = msg.twist.twist.linear.z

        command_pos_yaw = msg.pose.pose.orientation.w
        command_pos_received_ = 1

    def vins_cb(self, msg):
        """
        接收 VINS 里程计 (/vins_fusion/imu_propagate):
          用于获取无人机当前的状态估计(位置+速度+yaw),作为 MPC 的初始状态。

        位置采用 3 帧滑动窗口均值滤波,抑制 VINS 的短期噪声。
        Yaw 从四元数解析 (atan2 公式)。
        """
        global vins_p, vins_v, vins_yaw
        # 3 帧滑动窗口取均值 — 减少 VINS 短期漂移噪声
        self.list.append([msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          msg.pose.pose.position.z])
        if len(self.list) == 3:
            self.T1 = np.array([np.mean(self.list, axis=0)])
            # 从四元数计算 yaw: yaw = atan2(2(qw*qz+qx*qy), 1-2(qy²+qz²))
            vins_q_w = msg.pose.pose.orientation.w
            vins_q_x = msg.pose.pose.orientation.x
            vins_q_y = msg.pose.pose.orientation.y
            vins_q_z = msg.pose.pose.orientation.z
            siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y)
            cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
            vins_yaw = self.yaw

            self.trigger_condition_met = True
            self.list = []
            vins_p = np.array(self.T1[0])       # 取 3 帧均值作为当前位置
            vins_v[0] = msg.twist.twist.linear.x
            vins_v[1] = msg.twist.twist.linear.y

    def mode_cb(self, msg):
        """
        使用 /mode_manager 话题控制:
        - orientation.w == 0:  triger=1，追踪 /dog_pos_processed
        - orientation.w == -2: triger=2，追踪 /command_pos
        - 其他: triger=0，关闭MPC
        """
        global triger, target_received_, dog_pos_received_, command_pos_received_, reset_mpc
        mode_w = msg.pose.orientation.w
        if mode_w == 0:
            triger = 1
        elif mode_w == -2:
            triger = 2
        else:
            triger = 0
        target_received_ = 0
        dog_pos_received_ = 0
        command_pos_received_ = 0
        reset_mpc = 1


# ============================================================================
# 四、轨迹可视化 (TrajectoryVisualizer)
# ============================================================================
# 将 MPC 输出的轨迹序列发布为 ROS Path/PointCloud2 消息,
# 可在 Rviz 中可视化轨迹形状和路点位置。

class TrajectoryVisualizer:
    def __init__(self, frame_id="world"):
        self.frame_id = frame_id
        self.publishers = {}  # topic → publisher 的缓存字典

    def visualize_traj(self, position_array, acceleration_array, dt, topic="traj"):
        """
        发布轨迹到 Rviz:
          - topic (Path):         位置轨迹线
          - /traj_v (Path):       速度轨迹
          - /traj_a (Path):       加速度轨迹
          - topic_wayPts (PointCloud2): 路点标记
        """
        path_msg = self._build_path_msg(position_array)
        path_v_msg = self._build_path_v_msg(position_array)
        path_a_msg = self._build_path_a_msg(acceleration_array)
        self._get_publisher(topic, Path).publish(path_msg)
        self._get_publisher("/traj_v", Path).publish(path_v_msg)
        self._get_publisher("/traj_a", Path).publish(path_a_msg)

        waypoints_msg = self._build_waypoints_msg(position_array)
        self._get_publisher(f"{topic}_wayPts", PointCloud2).publish(waypoints_msg)

    def _build_path_msg(self, positions):
        """构建 nav_msgs/Path — 位置轨迹"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            path.poses.append(pose)
        return path

    def _build_path_v_msg(self, positions):
        """构建速度轨迹消息 — 用 position 字段存储 vx/vy/vz"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[3]  # vx
            pose.pose.position.y = pt[4]  # vy
            pose.pose.position.z = pt[5]  # vz
            path.poses.append(pose)
        return path

    def _build_path_a_msg(self, positions):
        """构建加速度轨迹消息"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            path.poses.append(pose)
        return path

    def _build_waypoints_msg(self, positions):
        """构建 sensor_msgs/PointCloud2 — 路点散点"""
        cloud = PointCloud2()
        cloud.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        points = positions.flatten().astype(np.float32)
        cloud.data = points.tobytes()
        cloud.fields = fields
        cloud.height = 1
        cloud.width = positions.shape[0]
        cloud.point_step = 12   # 3 * float32 = 12 bytes
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_bigendian = False
        cloud.is_dense = True
        return cloud

    def _get_publisher(self, topic, msg_type):
        """延迟创建发布器 (按需缓存)"""
        if topic not in self.publishers:
            self.publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=10)
        return self.publishers[topic]


# ============================================================================
# 五、目标轨迹预测 (predict_target_trajectory)
# ============================================================================

def predict_target_trajectory(pos, vel, N, dt, target_yaw, direct_predict=False, ave_num=5, shift=0.0):
    """
    预测目标在未来 N 步的轨迹。

    目的: MPC 需要一条"参考轨迹"来追踪。由于目标(狗)的运动是未知的,
          需要根据历史速度预测其未来运动。

    三种预测模式:
      1. 历史不足 (<ave_num 个点): 匀速直线预测
      2. 直线预测 (direct_predict=True): 速度取历史均值 → 匀速外推
      3. 曲线预测 (direct_predict=False): 速度做一阶多项式拟合 → 积分得到位置

    数学原理 (模式 3):
      - 对每个轴的速度做线性拟合: v(t) = a*t + b (最小二乘)
        A = [t_i, 1] (N×2 设计矩阵), 求解 [a, b]ᵀ = (AᵀA)⁻¹Aᵀv
      - 位置积分: p(t) = ∫v(t)dt = (a/2)*t² + b*t + p₀

    @param pos            当前目标位置 [x, y, z]
    @param vel            当前目标速度 [vx, vy, vz]
    @param N              MPC 预测步数
    @param dt             MPC 时间步长
    @param target_yaw     目标 yaw (当前未使用)
    @param direct_predict 是否使用直线预测模式
    @param ave_num        用于拟合的历史速度点数
    @param shift          轨迹起始时间偏移(秒), >0 表示从未来某个时刻开始
    @return               (N+1)×6 矩阵,每行 [px,py,pz, vx,vy,vz]
    """
    global target_vel_history, target_vel_history_times

    vel_pro = vel.copy()

    # ----- 维护速度历史滑动窗口 -----
    current_time = time.time()
    target_vel_history.append(vel.copy())
    target_vel_history_times.append(current_time)

    if len(target_vel_history) > ave_num:
        target_vel_history.pop(0)
        target_vel_history_times.pop(0)

    # ----- 模式 1: 历史不足 → 匀速直线 -----
    if len(target_vel_history) < ave_num:
        return np.array([np.concatenate([pos + vel * (t * dt + shift), vel_pro])
                        for t in range(N+1)])

    # ----- 模式 2: 直线预测(速度平均) -----
    if direct_predict:
        vel_array = np.array(target_vel_history)         # shape: (ave_num, 3)
        avg_vel = np.mean(vel_array, axis=0)             # shape: (3,)
        vel_pro = avg_vel.copy()
        return np.array([np.concatenate([pos + avg_vel * (t * dt + shift), vel_pro])
                        for t in range(N+1)])

    # ----- 模式 3: 曲线预测(速度一阶多项式拟合→积分) -----
    vel_array = np.array(target_vel_history)              # shape: (ave_num, 3)
    time_array = np.array(target_vel_history_times)       # shape: (ave_num,)

    # 时间戳转为相对时间差(最新点=0, 历史点为负数)
    latest_time = time_array[-1]
    time_diffs = time_array - latest_time

    # 设计矩阵 A: [t, 1], v(t) = a*t + b
    A = np.vstack([time_diffs, np.ones(len(time_diffs))]).T

    # 逐轴最小二乘: coeffs[axis] = [a, b]
    coeffs = np.zeros((3, 2))
    for axis in range(3):
        coeffs[axis] = np.linalg.lstsq(A, vel_array[:, axis], rcond=None)[0]

    # 积分生成轨迹: p(t) = (a/2)*t² + b*t + p₀
    # 速度: v(t) = a*t + b
    return np.array([np.concatenate([
        # 位置 = 初始位置 + 速度的积分
        np.array([pos[axis] + (coeffs[axis][0]/2) * (t * dt + shift)**2
                  + coeffs[axis][1] * (t * dt + shift) for axis in range(3)]),
        # 速度
        np.array([coeffs[axis][0] * (t * dt + shift) + coeffs[axis][1] for axis in range(3)])
    ]) for t in range(N+1)])


# ============================================================================
# 六、轨迹状态插值 (traj_get_state)
# ============================================================================

def traj_get_state(x_opt, u_opt, future_time, dt):
    """
    从 MPC 轨迹中插值出未来某时刻的无人机状态。

    MPC 输出是离散时间点上的状态和控制量序列,
    但实际使用时可能需要任意时刻的状态(因为 MPC 求解和轨迹执行之间有延迟)。

    使用线性插值 (1-alpha)*x0 + alpha*x1,简单但有效(轨迹步长通常很小)。

    @param x_opt       MPC 状态轨迹 (N+1)×6 [px,py,pz,vx,vy,vz]
    @param u_opt       MPC 控制输入 (N)×4 [ax,ay,az,yaw_rate]
    @param future_time 需要查询的未来时间偏移(秒)
    @param dt          MPC 时间步长
    @return           (插值后的 6 维状态, 最近的控制输入)
    """
    steps = future_time / dt
    idx_base = int(np.floor(steps))
    alpha = steps - idx_base                 # 插值权重 (0 ≤ α < 1)

    idx_base = min(max(0, idx_base), x_opt.shape[0]-2)

    x0 = x_opt[idx_base]
    x1 = x_opt[idx_base + 1]

    # 线性插值: state = (1-α)*x0 + α*x1
    state = (1 - alpha)*x0 + alpha*x1

    # 加速度: 使用最近的控制输入(加速度不在状态向量中,在控制量中)
    accel_idx = min(idx_base, u_opt.shape[0]-1)
    accel = u_opt[accel_idx, :3]

    return state, accel


# ============================================================================
# 七、轨迹拼接 (merge_trajectory)
# ============================================================================

def merge_trajectory(old_x_opt, new_x_opt, old_u_opt, new_u_opt, now, shift, mpc_N, dt):
    """
    将上一轮 MPC 轨迹的"已执行部分"和新解"未来部分"无缝拼接。

    为什么需要拼接?
      MPC 每 0.2s 求解一次,但轨迹执行在 traj_server 中以 80Hz 运行。
      如果直接扔掉旧轨迹用新轨迹,在切换瞬间可能出现位置/速度跳变。
      拼接保证: 两个求解周期之间的过渡是连续的。

    拼接逻辑:
      1. 从旧轨迹取 [now_idx, shift_idx) 段(已经执行的区间)
      2. 从新轨迹取前 (N - 旧段长度) 个点(填充剩余)
      3. 无缝衔接

    示意:
      旧轨迹: |--已执行--|----未来----|
      新轨迹: |--------全部未来--------|
      拼接后: |--已执行--|--新未来----|

    @param old_x_opt  上一轮的状态轨迹
    @param new_x_opt  本轮新解的状态轨迹
    @param old_u_opt  上一轮的控制轨迹
    @param new_u_opt  本轮新解的控制轨迹
    @param now        从上次求解到现在的经过时间(秒)
    @param shift      MPC 求解和轨迹执行之间的时间偏移(秒)
    @param mpc_N      MPC 预测步数
    @param dt         MPC 时间步长
    @return           (拼接后的 x_opt, 拼接后的 u_opt)
    """
    # 旧轨迹中需要保留的步数区间 [now_idx, shift_idx)
    shift_idx = int((shift + now) / dt)
    now_idx = int(now / dt)

    # 从旧轨迹取头部(已执行段)
    x_head = old_x_opt[now_idx:shift_idx] if old_x_opt is not None and len(old_x_opt) > shift_idx \
             else np.empty((0, new_x_opt.shape[1]))
    u_head = old_u_opt[now_idx:shift_idx] if old_u_opt is not None and len(old_u_opt) > shift_idx \
             else np.empty((0, new_u_opt.shape[1]))

    # 从新轨迹取尾部(补足到 N 步)
    remain_x = 1 + mpc_N - x_head.shape[0]
    remain_u = mpc_N - u_head.shape[0]

    x_tail = new_x_opt[:remain_x]
    u_tail = new_u_opt[:remain_u]

    # 拼接
    x_opt = np.concatenate((x_head, x_tail), axis=0)
    u_opt = np.concatenate((u_head, u_tail), axis=0)

    return x_opt, u_opt


# ============================================================================
# 八、MPC 主循环 (run_while_loop)
# ============================================================================
# 这是本模块的核心,运行在 ROS 主线程中,以 80Hz 的额定频率(实际约 5Hz,
# 因为 MPC 求解器耗时 ~0.2s)定期调用 MPC 求解器,生成并发布轨迹。

# ----- ROS 初始化 -----
rospy.init_node('MPC_publisher', anonymous=True)
listener = all_Subscriber()

vis = TrajectoryVisualizer()

# ROS 发布器
pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)
stop_triger_pub = rospy.Publisher('/stop_triger', PoseStamped, queue_size=1)
# ===== MPC 求解器初始化 =====
# N=50: 预测步数, dt 在 DroneMPC 内部定义(通常为 0.1s)
mpc = DroneMPC(N=50)

# ===== MPC 约束参数 — 默认值(triger=1 dog 跟踪模式) =====
# 速度上限: [vx_max, vy_max, vz_max] (m/s)
#   水平 1.5m/s + 垂直 0.8m/s — 保守值,保证跟踪稳定性
MPC_V_MAX_DEFAULT = np.array([1.5, 1.5, 0.8])
# 加速度上限: [ax_max, ay_max, az_max] (m/s²)
MPC_A_MAX_DEFAULT = np.array([0.8, 0.8, 0.6])
# 状态权重矩阵: Q = diag([px_w, py_w, pz_w, vx_w, vy_w, vz_w])
#   z 权重 20 > xy 权重 10 → 高度跟踪优先级更高, 速度权重 5 → 适中
MPC_Q_DEFAULT = np.diag([10.0, 10.0, 20.0, 5.0, 5.0, 5.0])

# ===== MPC 约束参数 — agent 模式(triger=2 command_pos 跟踪) =====
# agent 模式下目标运动更慢更可控,使用更保守的速度/加速度限制
MPC_V_MAX_AGENT = np.array([0.3, 0.3, 0.3])
MPC_A_MAX_AGENT = np.array([0.3, 0.3, 0.3])
MPC_Q_AGENT = np.diag([5.0, 5.0, 5.0, 8.0, 8.0, 8.0])  # 位置略柔,速度权重更重

# 初始化为默认 dog 跟踪参数
mpc.v_max = MPC_V_MAX_DEFAULT.copy()
mpc.a_max = MPC_A_MAX_DEFAULT.copy()
mpc.Q = MPC_Q_DEFAULT.copy()

rate = rospy.Rate(80)                     # 额定循环频率 80Hz
MPC_clcyle = time.time() - 10             # MPC 求解周期计时起点

x_opt = None                               # 当前拼接后的最优状态轨迹 (初始为空)
u_opt = None                               # 当前拼接后的最优控制轨迹 (初始为空)

# 目标位置变量(用于跟踪逻辑)
target_p_dis = np.array([0.0, 0.0, 0.0])      # 实际发给 MPC 的目标位置
last_target_p_dis = None
last_last_target_p_dis = None
tracking_dist = 1.25                           # 跟踪距离(m)
shift = 0.2                                    # MPC 执行偏移(秒)
shift_dog = 0.23                               # 目标轨迹预测偏移(秒)


def run_while_loop():
    """MPC 主循环 — 周期性求解最优轨迹"""
    global x_opt, u_opt
    global MPC_clcyle
    global vins_p, vins_v
    global target_p, target_v
    global vis
    global target_p_dis, last_target_p_dis, last_last_target_p_dis
    global dog_pos_p, dog_pos_v, dog_pos_yaw, dog_pos_received_
    global hc14_offset_yaw_ready, hc14_offset_pos_ready
    global triger, reset_mpc, target_received_
    global command_pos_p, command_pos_v, command_pos_yaw, command_pos_received_

    while not rospy.is_shutdown():
        # ----- 等待条件: trigger 启动 + 对应数据 ready -----
        # triger=1: 等待狗数据 ready (dog_pos_received + yaw offset 收敛)
        # triger=2: 等待 command_pos 数据 ready
        # triger=0: 等待(不启动 MPC)
        while (triger == 1 and not (dog_pos_received_ and hc14_offset_yaw_ready)) \
            or (triger == 2 and not command_pos_received_) \
            or (triger == 0):
            time.sleep(0.1)

        # ----- 重置处理: 模式切换时清空轨迹和历史 -----
        if reset_mpc == 1:
            x_opt = None
            u_opt = None
            global target_vel_history, target_vel_history_times
            target_vel_history = []
            target_vel_history_times = []
            reset_mpc = 0
        # triger==1: dog_pos_processed; triger==2: command_pos (mode_manager.w == -2)
        if triger == 2:
            mpc.v_max = MPC_V_MAX_AGENT.copy()
            mpc.a_max = MPC_A_MAX_AGENT.copy()
            mpc.Q = MPC_Q_AGENT.copy()
            current_target_p = command_pos_p.copy()
            current_target_v = command_pos_v.copy()
            current_target_yaw = command_pos_yaw
        else:
            mpc.v_max = MPC_V_MAX_DEFAULT.copy()
            mpc.a_max = MPC_A_MAX_DEFAULT.copy()
            mpc.Q = MPC_Q_DEFAULT.copy()
            current_target_p = dog_pos_p.copy()
            current_target_v = dog_pos_v.copy()
            current_target_yaw = dog_pos_yaw

        # ----- 高度约束: 限幅到 [狗高度+1.5m, 狗高度+1.8m] -----
        # 无人机在狗上方 1.5-1.8m 的高度带内飞行
        # 同时不低于无人机当前高度(防止 MPC 规划向下穿过地面)
        target_p_dis[0] = current_target_p[0]
        target_p_dis[1] = current_target_p[1]
        if triger == 2:
            # agent / command_pos：不施加 land_height_limit
            target_p_dis[2] = current_target_p[2]
        else:
            target_p_dis[2] = min(
                current_target_p[2] + land_height_limit[1],
                max(current_target_p[2] + land_height_limit[0], vins_p[2]),
            )
        
        if x_opt is not None:
            drone_state, accel = traj_get_state(
                x_opt, u_opt,
                int((time.time() - MPC_clcyle + shift) / mpc.dt) * mpc.dt, mpc.dt)
        else:
            drone_state = np.array([vins_p[0], vins_p[1], vins_p[2],
                                    vins_v[0], vins_v[1], vins_v[2]])

        # ----- 目标轨迹预测 -----
        # 根据狗当前速度预测未来 N 步的轨迹
        target_traj = predict_target_trajectory(
            target_p_dis, current_target_v, mpc.N, mpc.dt,
            current_target_yaw, shift=shift_dog)

        # ----- MPC 求解 -----
        # 调用 C++ 编译的 MPC 求解器,输入无人机状态和参考轨迹,
        # 返回最优状态轨迹和控制序列
        target_position_for_mpc = None    # 额外的位置约束(预留接口)
        position_weight_for_mpc = 0.0     # 额外位置约束的权重(预留接口)

        start_time = time.time()
        new_x_opt, new_u_opt = mpc.solve(
            drone_state, target_traj,
            target_position=target_position_for_mpc,
            position_weight=position_weight_for_mpc)
        end_time = time.time()

        # 求解失败处理: 跳过本周期
        if new_x_opt is None or new_u_opt is None:
            print("MPC solver failed to find a solution.")
            continue

        # ----- 轨迹拼接: 将新旧轨迹无缝衔接 -----
        x_opt, u_opt = merge_trajectory(
            x_opt, new_x_opt, u_opt, new_u_opt,
            now=time.time() - MPC_clcyle, shift=shift,
            mpc_N=mpc.N, dt=mpc.dt)

        # 重置 MPC 周期计时器
        MPC_clcyle = time.time()

        # ----- 发布轨迹到 Rviz -----
        if not reset_mpc:
            vis.visualize_traj(x_opt, u_opt, mpc.dt, topic="/drone2/planning/traj")

        # 休眠: 约 5Hz MPC 更新(因为 solve 耗时约 0.2s + sleep 0.2s)
        time.sleep(0.2)


# ----- 变量初始化 -----
last_p_v = None     # 上一次发布的位置+速度命令
last_accel = None   # 上一次发布的加速度命令
land_sign = 0       # 降落标志(0=正常, 1=降落中)


# ============================================================================
# 九、入口 — 启动 MPC 主循环
# ============================================================================

run_while_loop()

# 下面的注释代码是旧版的位置命令直接发布逻辑,
# 当前版本改由 traj_server.cpp 消费 /drone2/planning/traj 并发布 /position_cmd
