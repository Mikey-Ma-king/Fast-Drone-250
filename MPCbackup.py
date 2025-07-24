import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Odometry
import rospy
import math,time
from math import pi
import threading
from scipy.interpolate import CubicSpline
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point32
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from MPC_python.MPC_python_complie import DroneMPC

class TargetFilter:
    def __init__(self, alpha=0.3, reset_interval=0.5):
        self.alpha = alpha  # 滤波系数，越小越平滑
        self.reset_interval = reset_interval  # 丢包时间阈值（秒）

        # 状态
        self.filtered_pos = np.zeros(3)
        self.filtered_yaw = 0.0
        self.last_update_time = None
        self.initialized = False

    def _wrap_angle(self, angle):
        """将角度限制到 [-pi, pi]"""
        return (angle + pi) % (2 * pi) - pi

    def _angle_lerp(self, a, b, alpha):
        """插值两个角度，避免跳变"""
        delta = self._wrap_angle(b - a)
        return a + alpha * delta

    def update(self, raw_pos, raw_yaw):
        now = time.time()
        if not self.initialized or (self.last_update_time is not None and now - self.last_update_time > self.reset_interval):
            # 丢包或首次初始化
            self.filtered_pos = np.array(raw_pos)
            self.filtered_yaw = raw_yaw
            self.initialized = True
        else:
            # 低通滤波位置
            self.filtered_pos = self.alpha * np.array(raw_pos) + (1 - self.alpha) * self.filtered_pos
            # 低通滤波偏航角（注意处理wrap）
            self.filtered_yaw = self._angle_lerp(self.filtered_yaw, raw_yaw, self.alpha)

        self.last_update_time = now

    def get_filtered(self):
        return self.filtered_pos, self.filtered_yaw

class all_Subscriber:
    def __init__(self):
        self.T1 = None
        self.R1 = None
        self.list = []
        self.yaw = 0
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/target_ekf_odom', Odometry, self.pose_cb0)
        self.triger_sub = rospy.Subscriber('/land_triger', PoseStamped, self.triger_cb)
        self.vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_cb)
        self.track_sub = rospy.Subscriber('/triger', PoseStamped, self.track_cb)
        # 用于保存T1和R1
        # 触发条件的标志
        self.target_filter = TargetFilter(alpha=0.3, reset_interval=0.5)
        
    def pose_cb0(self, msg):
        global target_v
        global target_p
        global target_yaw
        global target_received_
        target_received_ = 1
        # raw_target_p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z + 0.05])
        # raw_yaw = msg.pose.pose.orientation.w
        # self.target_filter.update(raw_pos= raw_target_p, raw_yaw=raw_yaw)

        # target_p[0] = self.target_filter.filtered_pos[0]
        # target_p[1] = self.target_filter.filtered_pos[1]
        # target_p[2] = self.target_filter.filtered_pos[2]
        # target_v[0] = msg.twist.twist.linear.x
        # target_v[1] = msg.twist.twist.linear.y
        # target_v[2] = msg.twist.twist.linear.z
        # target_yaw = self.target_filter.filtered_yaw
        target_p[0] = msg.pose.pose.position.x
        target_p[1] = msg.pose.pose.position.y
        target_p[2] = msg.pose.pose.position.z + 0.08
        target_v[0] = msg.twist.twist.linear.x
        target_v[1] = msg.twist.twist.linear.y
        target_v[2] = msg.twist.twist.linear.z
        target_yaw = msg.pose.pose.orientation.w

    def vins_cb(self, msg):
        global vins_p
        global vins_v
        global vins_yaw
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
            vins_yaw = self.yaw
        # 设置触发条件的标志为True
            self.trigger_condition_met = True
        # 取消订阅
            self.list = []
            vins_p = np.array(self.T1[0])
            vins_v[0] = msg.twist.twist.linear.x
            vins_v[1] = msg.twist.twist.linear.y
    def triger_cb(self, msg):
        global land_triger
        land_triger = 1

    def track_cb(self, msg):
        global triger
        triger = 1

class TrajectoryVisualizer:
    def __init__(self, frame_id="world"):
        """
        初始化轨迹可视化工具
        参数:
            frame_id: 坐标系名称 (默认 "world")
        """
        self.frame_id = frame_id
        self.publishers = {}  # 存储发布器 {topic_name: publisher}

    def visualize_traj(self, position_array, dt, topic="traj",frame_id=None):
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
        if frame_id is None:
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


class TrajectoryCache:
    """轨迹插值缓存类，预计算样条系数"""
    def __init__(self,dt):
        self.dt = dt
        self.initialized = False
        self.pos_splines = None
        self.accel_splines = None
        self.traj_end_time = 0

    def update(self, x_opt, u_opt):
        """
        更新轨迹数据（当MPC结果更新时调用）
        参数:
            x_opt: 状态轨迹 (N+1 x 6)
            u_opt: 控制输入 (N x 4)
        """
        if self.dt is None:
            raise ValueError("dt must be set before updating trajectory data")
        
        # 预计算位置样条（使用状态轨迹）
        t_state = np.arange(x_opt.shape[0]) * self.dt
        self.pos_splines = []
        for i in range(3):
            spl = CubicSpline(t_state, x_opt[:, i], bc_type='natural')
            self.pos_splines.append(spl)
        
        # 预计算加速度样条（使用控制输入）
        t_ctrl = np.arange(u_opt.shape[0]) * self.dt + self.dt/2
        self.accel_splines = []
        for i in range(3):
            spl = CubicSpline(t_ctrl, u_opt[:, i], bc_type='natural')
            self.accel_splines.append(spl)
        
        self.initialized = True
        self.traj_end_time = t_state[-1]  # 记录轨迹总时长

    def query(self, future_time):
        global x_opt
        """快速查询方法，时间复杂度O(1)"""
        if not self.initialized:
            raise RuntimeError("Trajectory data not initialized. Call update() first.")
        
        # 时间边界保护（关键修改点）
        clipped_time = np.clip(future_time, 0, self.traj_end_time)

        state = np.zeros(6)
        accel = np.zeros(3)
        
        print("clipped_time:",clipped_time)
        print("self.traj_end_time:",self.traj_end_time)
        print("len(self.pos_splines):",len(self.pos_splines))
        print(f"输入状态维度: {x_opt.shape[1]}")
        for i in range(3):
            # 位置和速度（来自状态样条）
            state[i] = self.pos_splines[i](clipped_time)
            state[i+3] = self.pos_splines[i](clipped_time, 1)
            
            # 加速度（来自控制输入样条，处理时间偏移）
            t_ctrl_adj = np.maximum(0, clipped_time - self.dt/2)
            accel[i] = self.accel_splines[i](t_ctrl_adj)
        
        return state, accel

def simulate_drone(x, u, dt):
    """简化的动力学仿真"""
    A = np.eye(6)
    A[0:3, 3:6] = np.eye(3) * dt
    B_acc = 0.5 * dt**2 * np.eye(3)  # 加速度对位置的影响 (3x3)
    B_vel = dt * np.eye(3)           # 加速度对速度的影响 (3x3)
    B_yaw = np.zeros((3, 1))         # 偏航角速度占位 (3x1)
    
    # 垂直拼接加速度和速度项 (6x3)
    B = np.vstack([B_acc, B_vel])

    B = np.hstack([B, np.vstack([B_yaw, B_yaw])])  # 确保维度对齐
    return A @ x + B @ u

def predict_target_trajectory(pos, vel, N, dt):
    """生成移动平台的参考轨迹"""
    return np.array([np.concatenate([pos + vel * t * dt, vel]) for t in range(N+1)])

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
