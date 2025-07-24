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

target_p =np.array([0.0,0.0,0.0])
target_v =np.array([0.0,0.0,0.0])
target_yaw = 0.0
target_received_ = 0
triger = 0
land_triger = 0
vins_p =np.array([0.0,0.0,0.0])
vins_v =np.array([0.0,0.0,0.0])
vins_yaw = 0

land_target_z = 0.0

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
        global land_target_z
        target_received_ = 1
        raw_target_p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z + 0.12])
        raw_yaw = msg.pose.pose.orientation.w
        self.target_filter.update(raw_pos= raw_target_p, raw_yaw=raw_yaw)
        # target_p[0] = self.target_filter.filtered_pos[0]
        # target_p[1] = self.target_filter.filtered_pos[1]
        # target_p[2] = self.target_filter.filtered_pos[2]
        # target_v[0] = msg.twist.twist.linear.x
        # target_v[1] = msg.twist.twist.linear.y
        # target_v[2] = msg.twist.twist.linear.z
        # target_yaw = self.target_filter.filtered_yaw
        target_yaw = msg.pose.pose.orientation.w
        target_p[0] = msg.pose.pose.position.x + math.sin(target_yaw) * 0.05
        target_p[1] = msg.pose.pose.position.y - math.cos(target_yaw) * 0.05
        if (land_triger == 1 and land_target_z == 0.0):
            land_target_z = self.target_filter.filtered_pos[2]
        if (land_target_z != 0.0):
            target_p[2] = land_target_z
            # print("sign0:",target_p[2])
        else:
            target_p[2] = msg.pose.pose.position.z + 0.09
            # print("sign:",target_p[2])
        
        if(land_triger == 1):
            target_v[0] = msg.twist.twist.linear.x * math.cos(target_yaw)* math.cos(target_yaw) + msg.twist.twist.linear.y * math.sin(target_yaw)* math.cos(target_yaw)
            target_v[1] = msg.twist.twist.linear.y * math.sin(target_yaw) * math.sin(target_yaw) + msg.twist.twist.linear.x * math.cos(target_yaw) * math.sin(target_yaw)
            target_v[2] = msg.twist.twist.linear.z
        else:
            target_v[0] = msg.twist.twist.linear.x
            target_v[1] = msg.twist.twist.linear.y
            target_v[2] = msg.twist.twist.linear.z
        # target_v[0] = msg.twist.twist.linear.x
        # target_v[1] = msg.twist.twist.linear.y
        # target_v[2] = msg.twist.twist.linear.z

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

def predict_target_trajectory(pos, vel, N, dt, target_yaw):
    """生成移动平台的参考轨迹"""
    global land_triger
    vel_pro = vel.copy()
    # if(land_triger == 1):
    #     vel_pro[2] = -0.08
    # if(land_triger == 1):
    #     vel_pro[0] = vel[0] * math.cos(target_yaw)
    #     vel_pro[1] = vel[1] * math.sin(target_yaw)
    #     vel_pro[2] = -0.08
    # else:
    #     vel_pro[0] = vel[0]
    #     vel_pro[1] = vel[1]
    #     vel_pro[2] = vel[2]
    return np.array([np.concatenate([pos + vel * t * dt, vel_pro]) for t in range(N+1)])

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

# 初始化
rospy.init_node('MPC_publisher', anonymous=True)
listener = all_Subscriber()

vis = TrajectoryVisualizer()


pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)
mpc = DroneMPC(N = 15)
mpc.N = 15
mpc.v_max = np.array([1.2, 1.2, 0.8])  # 速度限制
mpc.a_max = np.array([1.0, 1.0, 0.6])  # 加速度限制
mpc.Q = np.diag([10, 10, 20, 7, 7, 5])  # 位置权重
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


def run_while_loop():
    global x_opt,u_opt
    global MPC_clcyle
    global vins_p, vins_v
    global target_p , target_v
    global vis
    global target_p_dist,last_target_p_dis,last_last_target_p_dis
    global target_p_dis
    global target_received_

    land_init_ = 0
    land_time = time.time()
    while not rospy.is_shutdown():
        while triger != 1 or target_received_ != 1:
            time.sleep(0.1)
        target_p_dis[0] = target_p[0] - max(0.07,0 - 0.0*(time.time() - land_time))*math.cos(target_yaw)
        target_p_dis[1] = target_p[1] - max(0.07,0 - 0.0*(time.time() - land_time))*math.sin(target_yaw)
        target_p_dis[2] = target_p[2]
        # mpc.v_max = np.array([0.6*1.2 + 0.4*abs(target_v[0]), 0.6*1.2 + 0.4*abs(target_v[1]), 0.8])
        # print("target_p:",target_p)
        if (land_triger != 1):
            target_p_dis[0] = target_p[0] - tracking_dist*math.cos(target_yaw)
            target_p_dis[1] = target_p[1] - tracking_dist*math.sin(target_yaw)
            target_p_dis[2] = target_p[2] + 0.02
        if (land_triger == 0 and x_opt is not None or (land_triger == 1 and x_opt is not None and land_init_ == 1)):
            drone_state,accel = traj_get_state(x_opt, u_opt, int((time.time() - MPC_clcyle + shift)/mpc.dt)*mpc.dt, mpc.dt)
            # drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
        else :
            if (land_triger == 1 and land_init_ == 0):
                land_init_ = 1
                land_time = time.time()
                # mpc.v_max = np.array([1.2, 1.2, 0.8])
            drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
            # drone_state = np.array([vins_p[0], vins_p[1], vins_p[2], vins_v[0], vins_v[1], vins_v[2]])
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
        max_dis = 1.2 + land_triger*0.2
        if (abs(target_p_dis[0] - vins_p[0]) > max_dis) : 
            target_p_dis[0] = max_dis*(target_p_dis[0] - vins_p[0])/abs(target_p_dis[0] - vins_p[0]) + vins_p[0]
        if (abs(target_p_dis[1] - vins_p[1]) > max_dis) :
            target_p_dis[1] = max_dis*(target_p_dis[1] - vins_p[1])/abs(target_p_dis[1] - vins_p[1]) + vins_p[1]
        target_traj = predict_target_trajectory(target_p_dis, (1 + land_triger/10*0)*target_v, mpc.N, mpc.dt , target_yaw)
        start_time = time.time()
        new_x_opt, new_u_opt = mpc.solve(drone_state, target_traj)
        end_time = time.time()
        print("MPC solve time: " , end_time - start_time)

        if new_x_opt is None or new_u_opt is None:
            print("MPC solver failed to find a solution.")
            continue

        x_opt = merge_trajectory(x_opt, new_x_opt, now=time.time() - MPC_clcyle ,shift=shift, mpc_N=mpc.N, dt=mpc.dt)
        # x_opt = new_x_opt
        u_opt = new_u_opt
        MPC_clcyle = time.time() 
        # Traj_splines.update(x_opt,u_opt)
        # if (np.linalg.norm(x_opt[0,:3] - x_opt[-1,:3]) < 0.6 and land_triger == 1):
        #     print("short traj!")
        #     x_opt = None
        #     land_init_ = 0
        # else:
        #     vis.visualize_traj(x_opt, mpc.dt, topic="/drone2/planning/traj")
        time.sleep(0.2)
        vis.visualize_traj(x_opt, mpc.dt, topic="/drone2/planning/traj")

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