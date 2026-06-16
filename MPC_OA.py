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

try:
    from agent.config import (
        DEPTH_IMAGE_TOPIC,
        DEPTH_CX,
        DEPTH_CY,
        DEPTH_FX,
        DEPTH_FY,
        DEPTH_HEIGHT,
        DEPTH_WIDTH,
    )
except ImportError:
    DEPTH_IMAGE_TOPIC = "/camera/depth/image_rect_raw"
    DEPTH_WIDTH = 640
    DEPTH_HEIGHT = 480
    DEPTH_FX = 391.0926513671875
    DEPTH_FY = 391.0926513671875
    DEPTH_CX = 323.90478515625
    DEPTH_CY = 235.0640411376953

DEPTH_TOPIC = DEPTH_IMAGE_TOPIC
MPC_DEPTH_WIDTH = DEPTH_WIDTH
MPC_DEPTH_HEIGHT = DEPTH_HEIGHT
MPC_DEPTH_FX = DEPTH_FX
MPC_DEPTH_FY = DEPTH_FY
MPC_DEPTH_CX = DEPTH_CX
MPC_DEPTH_CY = DEPTH_CY

# 深度图避障总开关：True=订阅深度并传入 mpc.solve；False=一律不使用避障
MPC_ENABLE_OBSTACLE = False
# 格内深度分位数（5=近端 5% 分位，偏保守；可改为 95 等）
OBS_GRID_DEPTH_PERCENTILE = 5
OBS_GRID_MIN_VALID_DEPTH_M = 0.05
# 障碍 3D 点历史：最多保留最近 N 帧，每帧最多 grid_n^2 个点
OBS_POINT_HISTORY_FRAMES = 5

obs_dirs_body = None          # 机体系射线，reset_mpc 时预计算 (grid_n, grid_n, 3)
obstacle_point_history = []   # 最近若干帧的 (M, 3) 世界系障碍点
obstacle_points_lock = threading.Lock()
depth_received_ = False

target_p =np.array([0.0,0.0,0.0])
target_v =np.array([0.0,0.0,0.0])
target_yaw = 0.0
target_received_ = 0
triger = 0
vins_p =np.array([0.0,0.0,0.0])
vins_v =np.array([0.0,0.0,0.0])
vins_yaw = 0
vins_qx = 0.0
vins_qy = 0.0
vins_qz = 0.0
vins_qw = 1.0
reset_mpc = 0

# dog_pos_processor相关变量
dog_pos_p = np.array([0.0,0.0,0.0])
dog_pos_v = np.array([0.0,0.0,0.0])
dog_pos_yaw = 0.0
dog_pos_acc = np.array([0.0,0.0])  # 狗加速度（世界坐标系，x和y）
dog_pos_received_ = 0

# command_pos相关变量（mode_manager == -2 时使用）
command_pos_p = np.array([0.0,0.0,0.0])
command_pos_v = np.array([0.0,0.0,0.0])
command_pos_yaw = 0.0
command_pos_received_ = 0

last_target_ekf_time = 0.0
hc14_offset_yaw_ready = False
hc14_offset_pos_ready = False

# 目标速度历史记录（用于速度拟合）
target_vel_history = []  # 存储最近10个速度点
target_vel_history_times = []  # 存储对应的时间戳


land_height_limit = [1.5, 1.8]


def _depth_array_to_meters(depth_np, encoding):
    """深度图 → 米（真机 uint16 毫米；Gazebo 常为 float32 米）。"""
    if depth_np.dtype == np.uint16:
        return depth_np.astype(np.float64) * 0.001
    if depth_np.dtype == np.float32:
        return depth_np.astype(np.float64)
    enc = (encoding or "").lower()
    if "32f" in enc:
        return depth_np.astype(np.float64)
    return depth_np.astype(np.float64)


def depth_image_to_obstacle_grid(depth_m, grid_n=None, percentile=None):
    """分格取有效深度的分位数（默认 5% 分位，近障偏保守）。"""
    if grid_n is None:
        grid_n = mpc.obs_grid_n
    if percentile is None:
        percentile = OBS_GRID_DEPTH_PERCENTILE
    h, w = depth_m.shape[:2]
    out = np.zeros((grid_n, grid_n), dtype=np.float64)
    for gi in range(grid_n):
        r0 = int(gi * h / grid_n)
        r1 = int((gi + 1) * h / grid_n)
        if r1 <= r0:
            continue
        for gj in range(grid_n):
            c0 = int(gj * w / grid_n)
            c1 = int((gj + 1) * w / grid_n)
            if c1 <= c0:
                continue
            patch = depth_m[r0:r1, c0:c1].reshape(-1)
            valid = patch[(patch > OBS_GRID_MIN_VALID_DEPTH_M) & np.isfinite(patch)]
            if valid.size > 0:
                out[gi, gj] = float(np.percentile(valid, percentile))
    return out


def precompute_obs_dirs_body(grid_n=None):
    """机体系每格中心射线（单位向量），仅依赖内参，启动时算一次。"""
    if grid_n is None:
        grid_n = mpc.obs_grid_n if "mpc" in globals() else 8
    from agent.camera_geom import bearing_body_from_pixel

    intr = _mpc_depth_intrinsics()
    dirs = np.zeros((grid_n, grid_n, 3), dtype=np.float64)
    for gi in range(grid_n):
        for gj in range(grid_n):
            u, v = grid_cell_center_uv(gi, gj, grid_n)
            xb, yb, zb = bearing_body_from_pixel(u, v, intr)
            nlen = math.hypot(xb, math.hypot(yb, zb))
            if nlen > 1e-9:
                dirs[gi, gj, 0] = xb / nlen
                dirs[gi, gj, 1] = yb / nlen
                dirs[gi, gj, 2] = zb / nlen
    return dirs


def obstacle_points_world_from_grid(grid, body_dirs, origin, quat, depth_max_m=None):
    """每格：世界系方向 × 深度 → 障碍 3D 点 origin + r * n_world。"""
    from agent.camera_geom import bearing_world_from_body

    if depth_max_m is None:
        depth_max_m = mpc.obs_depth_max_m
    grid = np.asarray(grid, dtype=np.float64)
    body_dirs = np.asarray(body_dirs, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    q = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    gn = grid.shape[0]
    pts = []
    for gi in range(gn):
        for gj in range(gn):
            r_m = float(grid[gi, gj])
            if r_m <= OBS_GRID_MIN_VALID_DEPTH_M or r_m > depth_max_m:
                continue
            xb, yb, zb = body_dirs[gi, gj]
            xw, yw, zw = bearing_world_from_body(xb, yb, zb, quat=q)
            pts.append(origin + r_m * np.array([xw, yw, zw], dtype=np.float64))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pts)


def reset_obstacle_point_history():
    global obstacle_point_history
    with obstacle_points_lock:
        obstacle_point_history = []


def reset_obstacle_mpc_state(grid_n=None):
    """与 reset_mpc 同步：预计算机体系射线方向，并清空障碍点历史。"""
    global obs_dirs_body
    if grid_n is None:
        grid_n = mpc.obs_grid_n
    obs_dirs_body = precompute_obs_dirs_body(grid_n)
    reset_obstacle_point_history()


def push_obstacle_point_frame(points):
    """追加一帧障碍点，保留最近 OBS_POINT_HISTORY_FRAMES 帧。"""
    global obstacle_point_history
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=np.float64)
    with obstacle_points_lock:
        obstacle_point_history.append(pts)
        if len(obstacle_point_history) > OBS_POINT_HISTORY_FRAMES:
            obstacle_point_history.pop(0)


def flatten_obstacle_point_history():
    """合并历史帧为 (M, 3)，最多 grid_n^2 * OBS_POINT_HISTORY_FRAMES 个点。"""
    with obstacle_points_lock:
        if not obstacle_point_history:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(obstacle_point_history)


def _mpc_depth_intrinsics():
    from agent.camera_intrinsics import CameraIntrinsics
    return CameraIntrinsics(
        MPC_DEPTH_WIDTH,
        MPC_DEPTH_HEIGHT,
        MPC_DEPTH_FX,
        MPC_DEPTH_FY,
        MPC_DEPTH_CX,
        MPC_DEPTH_CY,
    )


def grid_cell_center_uv(gi, gj, grid_n=None):
    """格子 (gi, gj) 中心像素：gi 行向下，gj 列向右。"""
    if grid_n is None:
        grid_n = mpc.obs_grid_n
    u = (float(gj) + 0.5) * float(MPC_DEPTH_WIDTH) / float(grid_n)
    v = (float(gi) + 0.5) * float(MPC_DEPTH_HEIGHT) / float(grid_n)
    return u, v


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
        self.dog_pos_sub = rospy.Subscriber('/dog_pos_processed', Odometry, self.dog_pos_cb)
        self.command_pos_sub = rospy.Subscriber('/command_pos', Odometry, self.command_pos_cb)
        self.vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_cb)
        self.track_sub = rospy.Subscriber('/mode_manager', PoseStamped, self.mode_cb)
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        self.depth_bridge = None
        self.depth_sub = None
        if MPC_ENABLE_OBSTACLE:
            self.depth_bridge = CvBridge()
            self.depth_sub = rospy.Subscriber(
                DEPTH_TOPIC, Image, self.depth_cb, queue_size=1,
            )
        # 用于保存T1和R1
        # 触发条件的标志
        self.target_filter = TargetFilter(alpha=0.3, reset_interval=0.5)

    def depth_cb(self, msg):
        global obs_dirs_body, depth_received_
        global vins_p, vins_qx, vins_qy, vins_qz, vins_qw
        if self.depth_bridge is None or obs_dirs_body is None:
            return
        try:
            depth_np = self.depth_bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough",
            )
            depth_m = _depth_array_to_meters(depth_np, msg.encoding)
            grid = depth_image_to_obstacle_grid(depth_m)
            quat = (vins_qx, vins_qy, vins_qz, vins_qw)
            pts = obstacle_points_world_from_grid(
                grid, obs_dirs_body, vins_p, quat,
            )
            push_obstacle_point_frame(pts)
            depth_received_ = True
        except Exception as e:
            rospy.logwarn_throttle(2.0, "MPC depth_cb: %s", e)
        
    def pose_cb0(self, msg):
        global target_v
        global target_p
        global target_yaw
        global target_received_
        global land_target_z
        if msg.pose.pose.orientation.x == -1:
            return
        target_received_ = 1
        # raw_target_p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z + 0.12])
        # raw_yaw = msg.pose.pose.orientation.w
        # self.target_filter.update(raw_pos= raw_target_p, raw_yaw=raw_yaw)
        # target_p[0] = self.target_filter.filtered_pos[0]
        # target_p[1] = self.target_filter.filtered_pos[1]
        # target_p[2] = self.target_filter.filtered_pos[2]
        # target_v[0] = msg.twist.twist.linear.x
        # target_v[1] = msg.twist.twist.linear.y
        # target_v[2] = msg.twist.twist.linear.z
        # target_yaw = self.target_filter.filtered_yaw
        target_yaw = msg.pose.pose.orientation.w
        target_p[0] = msg.pose.pose.position.x
        target_p[1] = msg.pose.pose.position.y
        target_p[2] = msg.pose.pose.position.z
        
            # print("sign:",target_p[2])
        
        target_v[0] = msg.twist.twist.linear.x
        target_v[1] = msg.twist.twist.linear.y
        target_v[2] = msg.twist.twist.linear.z

    def dog_pos_cb(self, msg):
        global dog_pos_p
        global dog_pos_v
        global dog_pos_yaw
        global dog_pos_acc
        global dog_pos_received_
        global hc14_offset_yaw_ready
        global hc14_offset_pos_ready

        # 获取处理后的狗位置和速度
        dog_pos_p[0] = msg.pose.pose.position.x
        dog_pos_p[1] = msg.pose.pose.position.y
        dog_pos_p[2] = msg.pose.pose.position.z
        
        # 狗坐标系速度
        dog_pos_v[0] = msg.twist.twist.linear.x
        dog_pos_v[1] = msg.twist.twist.linear.y
        dog_pos_v[2] = msg.twist.twist.linear.z
        
        # 处理后的yaw
        dog_pos_yaw = msg.twist.twist.angular.x
        
        # 从orientation.w和x读取precise_pos_offset_ready和precise_yaw_offset_ready状态
        hc14_offset_pos_ready = (msg.pose.pose.orientation.w > 0.5)
        hc14_offset_yaw_ready = (msg.pose.pose.orientation.x > 0.5)
        
        # 从orientation.y和z读取加速度（世界坐标系）
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
        global vins_p
        global vins_v
        global vins_yaw
        global vins_qx, vins_qy, vins_qz, vins_qw
        vins_qx = msg.pose.pose.orientation.x
        vins_qy = msg.pose.pose.orientation.y
        vins_qz = msg.pose.pose.orientation.z
        vins_qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (vins_qw * vins_qz + vins_qx * vins_qy)
        cosy_cosp = 1.0 - 2.0 * (vins_qy * vins_qy + vins_qz * vins_qz)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        vins_yaw = self.yaw
        # 从 Odometry 消息中获取位置和姿态
        self.list.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        if len(self.list)==3:
            self.T1 = np.array([np.mean(self.list,axis = 0)])
        # 设置触发条件的标志为True
            self.trigger_condition_met = True
        # 取消订阅
            self.list = []
            vins_p = np.array(self.T1[0])
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

class TrajectoryVisualizer:
    def __init__(self, frame_id="world"):
        """
        初始化轨迹可视化工具
        参数:
            frame_id: 坐标系名称 (默认 "world")
        """
        self.frame_id = frame_id
        self.publishers = {}  # 存储发布器 {topic_name: publisher}

    def visualize_traj(self, position_array, acceleration_array, dt, topic="traj"):
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
        path_a_msg = self._build_path_a_msg(acceleration_array)
        self._get_publisher(topic, Path).publish(path_msg)
        self._get_publisher("/traj_v", Path).publish(path_v_msg)
        self._get_publisher("/traj_a", Path).publish(path_a_msg)
        
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

    def _build_path_a_msg(self, positions):
        """构建 nav_msgs/Path 消息（加速度）"""
        path = Path()
        path.header = Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        # 遍历加速度点（如果 position_array 包含加速度信息）
        # 假设 position_array 形状为 (N, 9)，包含 [x, y, z, vx, vy, vz, ax, ay, az]
        for pt in positions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
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

def predict_target_trajectory(pos, vel, N, dt, target_yaw, direct_predict=False, ave_num=5, shift=0.0):
    """
    生成移动平台的参考轨迹
    根据 direct_predict 参数选择：
    - True: 使用速度历史取平均，然后进行直线预测（如果历史点不足ave_num个，则使用传入的vel）
    - False: 使用速度历史进行一次函数（直线）拟合，然后通过积分得到位置（曲线预测），如果历史点不足ave_num个，则使用传入的vel进行直线预测
    返回的轨迹从shift时间之后开始
    """
    global target_vel_history, target_vel_history_times
    
    vel_pro = vel.copy()
    
    # 更新速度历史记录（无论哪种方法都要保存）
    current_time = time.time()
    target_vel_history.append(vel.copy())
    target_vel_history_times.append(current_time)
    
    # 只保留最近ave_num个速度点
    if len(target_vel_history) > ave_num:
        target_vel_history.pop(0)
        target_vel_history_times.pop(0)
    
    # 如果历史点不足ave_num个，使用传入的vel进行直线预测（从shift之后开始）
    if len(target_vel_history) < ave_num:
        return np.array([np.concatenate([pos + vel * (t * dt + shift), vel_pro]) for t in range(N+1)])
    
    # 如果使用直线预测（速度平均）
    if direct_predict:
        # 使用ave_num个历史速度点计算平均速度
        vel_array = np.array(target_vel_history)  # shape: (ave_num, 3)
        avg_vel = np.mean(vel_array, axis=0)  # shape: (3,)
        vel_pro = avg_vel.copy()
        
        # 使用平均速度进行直线预测（从shift之后开始）
        return np.array([np.concatenate([pos + avg_vel * (t * dt + shift), vel_pro]) for t in range(N+1)])
    
    # 以下为使用速度历史进行一次函数（直线）拟合的逻辑（曲线预测）
    vel_array = np.array(target_vel_history)  # shape: (ave_num, 3)
    time_array = np.array(target_vel_history_times)  # shape: (ave_num,)
    
    # 将时间戳转换为相对于最新点的时间差（秒）
    latest_time = time_array[-1]
    time_diffs = time_array - latest_time  # 负数，最新点为0
    
    # 对x, y, z三个轴分别进行一次函数拟合速度: v(t) = a*t + b
    # 构建系数矩阵 A: [t, 1] for each point
    A = np.vstack([time_diffs, np.ones(len(time_diffs))]).T
    
    # 对每个轴分别求解最小二乘，得到速度的系数矩阵 coeffs: shape (3, 2) [axis, coeff]
    coeffs = np.zeros((3, 2))  # [axis, (a, b)]
    for axis in range(3):
        coeffs[axis] = np.linalg.lstsq(A, vel_array[:, axis], rcond=None)[0]
    
    # 使用列表推导式生成轨迹（从shift之后开始）
    # 速度: v(t) = a*t + b
    # 位置: p(t) = ∫v(t)dt = (a/2)*t^2 + b*t + p0
    return np.array([np.concatenate([
        np.array([pos[axis] + (coeffs[axis][0]/2) * (t * dt + shift)**2 + coeffs[axis][1] * (t * dt + shift) for axis in range(3)]),  # 位置（速度的积分）
        np.array([coeffs[axis][0] * (t * dt + shift) + coeffs[axis][1] for axis in range(3)])  # 速度
    ]) for t in range(N+1)])

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
    idx_base = min(max(0, idx_base), x_opt.shape[0]-2)  # 需要至少3个点做二次插值
    
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

def merge_trajectory(old_x_opt, new_x_opt, old_u_opt, new_u_opt, now,shift, mpc_N, dt):
    shift_idx = int((shift + now) / dt)
    now_idx = int(now / dt)

    x_head = old_x_opt[now_idx:shift_idx] if old_x_opt is not None and len(old_x_opt) > shift_idx else np.empty((0, new_x_opt.shape[1]))
    u_head = old_u_opt[now_idx:shift_idx] if old_u_opt is not None and len(old_u_opt) > shift_idx else np.empty((0, new_u_opt.shape[1]))

    remain_x = 1 + mpc_N - x_head.shape[0]
    remain_u = mpc_N - u_head.shape[0]

    x_tail = new_x_opt[:remain_x]
    u_tail = new_u_opt[:remain_u]

    x_opt = np.concatenate((x_head, x_tail), axis=0)
    u_opt = np.concatenate((u_head, u_tail), axis=0)

    return x_opt, u_opt

# 初始化
rospy.init_node('MPC_publisher', anonymous=True)
listener = all_Subscriber()

vis = TrajectoryVisualizer()


pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)
stop_triger_pub = rospy.Publisher('/stop_triger', PoseStamped, queue_size=1)
mpc = DroneMPC(N = 50)
rospy.loginfo(
    "[MPC] depth obstacle avoidance (3D points, hinge loss): %s, history=%d frames, max %d points",
    "ON" if MPC_ENABLE_OBSTACLE else "OFF",
    OBS_POINT_HISTORY_FRAMES,
    mpc.obs_grid_n * mpc.obs_grid_n * OBS_POINT_HISTORY_FRAMES,
)
# mpc.v_max = np.array([2.0, 2.0, 0.8])  # 速度限制
# mpc.a_max = np.array([1.3, 1.3, 0.6])  # 加速度限制
# mpc.Q = np.diag([15, 15, 10, 3, 3, 0])  # 位置权重
MPC_V_MAX_DEFAULT = np.array([1.5, 1.5, 0.8])
MPC_V_MAX_AGENT = np.array([0.3, 0.3, 0.3])
MPC_A_MAX_DEFAULT = np.array([0.8, 0.8, 0.6])
MPC_A_MAX_AGENT = np.array([0.3, 0.3, 0.3])  # agent/command_pos：更柔的加速度
# Q 对角：[px, py, pz, vx, vy, vz]
MPC_Q_DEFAULT = np.diag([10.0, 10.0, 20.0, 5.0, 5.0, 5.0])
MPC_Q_AGENT = np.diag([5.0, 5.0, 5.0, 8.0, 8.0, 8.0])  # agent：位置略柔、速度略重
mpc.v_max = MPC_V_MAX_DEFAULT.copy()
mpc.a_max = MPC_A_MAX_DEFAULT.copy()
mpc.Q = MPC_Q_DEFAULT.copy()
# drone_state = np.array([0, 0, 0.4, 0, 0, 0])  # 初始状态 [px,py,pz,vx,vy,vz]
rate = rospy.Rate(80)
MPC_clcyle = time.time() - 10

x_opt = None
u_opt = None

target_p_dis = np.array([0.0,0.0,0.0])
last_target_p_dis = None
last_last_target_p_dis = None
tracking_dist = 1.25
shift = 0.2
shift_dog = 0.23

# 仿真模式参数：如果为True，直接使用target_p和target_v，否则使用dog_pos_p和dog_pos_v


def run_while_loop():
    global x_opt,u_opt
    global MPC_clcyle
    global vins_p, vins_v
    global target_p , target_v
    global vis
    global target_p_dist,last_target_p_dis,last_last_target_p_dis
    global target_p_dis
    global dog_pos_p, dog_pos_v, dog_pos_yaw, dog_pos_received_, hc14_offset_yaw_ready, hc14_offset_pos_ready
    global command_pos_p, command_pos_v, command_pos_yaw, command_pos_received_
    global aoa_fast_v_max, aoa_slow_v_max
    global triger
    global reset_mpc
    global target_received_

    while not rospy.is_shutdown():
        while (triger == 1 and not (dog_pos_received_ and hc14_offset_yaw_ready)) \
            or (triger == 2 and not command_pos_received_) \
            or (triger == 0):
            time.sleep(0.1)
        # 如果有起飞重置请求，则在主循环中清空轨迹和历史记录
        if reset_mpc == 1:
            x_opt = None
            u_opt = None
            # 清空速度历史记录
            global target_vel_history, target_vel_history_times
            target_vel_history = []
            target_vel_history_times = []
            reset_obstacle_mpc_state(mpc.obs_grid_n)
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

        # 在狗位置基础上添加提前量：沿着vins到狗的方向延伸0.5m
        # dog_to_vins = current_target_p[:2] - vins_p[:2]  # 狗位置到vins位置的向量（2D）
        # distance_to_vins = np.linalg.norm(dog_to_vins)
        # if distance_to_vins > 0.001:  # 避免除零
        #     # 单位方向向量：从狗指向vins
        #     direction_unit = dog_to_vins / distance_to_vins
        #     # 提前量：在狗到vins的方向上延伸0.5m
        #     lead_offset = 0.5 * direction_unit
        #     current_target_p[0] += lead_offset[0]
        #     current_target_p[1] += lead_offset[1]

        # mpc.v_max = np.array([0.6*1.2 + 0.4*abs(current_target_v[0]), 0.6*1.2 + 0.4*abs(current_target_v[1]), 0.8])
        # print("target_p:",current_target_p)
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
            drone_state,accel = traj_get_state(x_opt, u_opt, int((time.time() - MPC_clcyle + shift)/mpc.dt)*mpc.dt, mpc.dt)
        else :
            drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])

        # print("time.time() - MPC_clcyle:",time.time() - MPC_clcyle)
        # print("drone_state:",drone_state)
        # print("x_opt:",x_opt)
        # print("drone_state:" , drone_state)
        # if (last_target_p_dis is None):
        #     last_target_p_dis = target_p_dis
        #     last_last_target_p_dis = target_p_dis
        # else :
        #     target_p_dis = 0.3*last_last_target_p_dis + 0.4*last_target_p_dis + 0.3*target_p_dis
        #     last_last_target_p_dis = last_target_p_dis
        #     last_target_p_dis = target_p_dis

        # max_dis = 1.2
        # if (abs(target_p_dis[0] - vins_p[0]) > max_dis) : 
        #     target_p_dis[0] = max_dis*(target_p_dis[0] - vins_p[0])/abs(target_p_dis[0] - vins_p[0]) + vins_p[0]
        # if (abs(target_p_dis[1] - vins_p[1]) > max_dis) :
        #     target_p_dis[1] = max_dis*(target_p_dis[1] - vins_p[1])/abs(target_p_dis[1] - vins_p[1]) + vins_p[1]
        target_traj = predict_target_trajectory(target_p_dis, current_target_v, mpc.N, mpc.dt , current_target_yaw, shift=shift_dog)
        
        # 位置约束逻辑预留：可根据需要添加
        target_position_for_mpc = None
        position_weight_for_mpc = 0.0

        enable_obstacle = MPC_ENABLE_OBSTACLE
        obs_pts = None
        if enable_obstacle:
            obs_pts = flatten_obstacle_point_history()

        start_time = time.time()
        new_x_opt, new_u_opt = mpc.solve(
            drone_state, target_traj,
            target_position=target_position_for_mpc,
            position_weight=position_weight_for_mpc,
            obstacle_points=obs_pts,
            enable_obstacle=enable_obstacle,
        )
        end_time = time.time()
        # print("MPC solve time: " , end_time - start_time)

        if new_x_opt is None or new_u_opt is None:
            print("MPC solver failed to find a solution.")
            continue

        x_opt, u_opt = merge_trajectory(x_opt, new_x_opt, u_opt, new_u_opt, now=time.time() - MPC_clcyle ,shift=shift, mpc_N=mpc.N, dt=mpc.dt)

        MPC_clcyle = time.time() 
        if not reset_mpc:
            vis.visualize_traj(x_opt, u_opt, mpc.dt, topic="/drone2/planning/traj")
        time.sleep(0.2)
last_p_v = None
last_accel = None
land_sign = 0

# thread = threading.Thread(target=run_while_loop)
# thread.daemon = True  # 设置为守护线程，主程序退出时，子线程自动结束
# thread.start()

run_while_loop()
# while True:
#     time.sleep(0.1)
# while not rospy.is_shutdown():
#     while triger != 1:
#         time.sleep(0.1)

    

#     while triger == 1 and False:
#         if x_opt is None and u_opt is None:
#             continue
#         p_v, accel = traj_get_state(x_opt, u_opt, time.time() - MPC_clcyle, mpc.dt)
#         # p_v,accel = Traj_splines.query(time.time() - MPC_clcyle)

#         if last_accel is None:
#             last_accel = accel
#             last_p_v = p_v
#         cmd = PositionCommand()
#         cmd.position.x = p_v[0]
#         cmd.position.y = p_v[1]

#         cmd.position.z = 0.6*p_v[2] + 0.4*last_p_v[2]
#         cmd.velocity.x = p_v[3]
#         cmd.velocity.y = p_v[4]
#         cmd.velocity.z = 0.6*p_v[5] + 0.4*last_p_v[5]
#         cmd.acceleration.x = accel[0]
#         cmd.acceleration.y = accel[1]
#         cmd.acceleration.z = 0.6*accel[2] + 0.4*accel[2]
#         angle_diff = math.atan2(math.sin(target_yaw - vins_yaw), math.cos(target_yaw - vins_yaw))
#         cmd.yaw = 0.3*angle_diff + vins_yaw

#         last_p_v = p_v
#         last_accel = accel

#         if abs(vins_p[0] - target_p[0]) < 0.2 and abs(vins_p[1] - target_p[1]) < 0.15 or land_sign == 1:
#             cmd.position.z = vins_p[2] - 2
#             cmd.velocity.z = -2
#             if land_sign == 0:
#                 land_sign = 1
#         pos_cmd_pub.publish(cmd)

#         rate.sleep()