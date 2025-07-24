import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Odometry
import rospy
import math
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import PoseStamped

target_p =np.array([0.0,0.0,0.0])
target_v =np.array([0.0,0.0,0.0])
target_yaw = 0.0
triger = 0
vins_p =np.array([0.0,0.0,0.0])
vins_v =np.array([0.0,0.0,0.0])
vins_yaw = 0

class target:
    def __init__(self):
        self.T1 = None
        self.R1 = None
        self.list = []
        self.yaw = 0
        # 订阅 /vins_fusion/imu_propagate 主题
        self.pose_sub = rospy.Subscriber('/target_ekf_odom', Odometry, self.pose_cb0)
        self.triger_sub = rospy.Subscriber('/land_triger', PoseStamped, self.triger_cb)
        self.vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, self.vins_cb)
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb0(self, msg):
        global target_v
        global target_p
        global target_yaw
        # stop_triger = 1
        target_p[0] = msg.pose.pose.position.x
        target_p[1] = msg.pose.pose.position.y
        target_p[2] = msg.pose.pose.position.z
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
        global triger
        triger = 1


class DroneMPC:
    def __init__(self, N=10, dt=0.1):
        self.N = N
        self.dt = dt
        self.nx = 6  # [px,py,pz,vx,vy,vz]
        self.nu = 4  # [ax,ay,az,yaw_rate]

        # 权重矩阵
        self.Q = np.diag([10, 10, 20, 1, 1, 1])
        self.R = np.diag([0.1, 0.1, 0.1, 0.05])
        self.S = np.diag([0.5, 0.5, 0.5, 0.2])

        # 动力学模型
        self.A = np.eye(6)
        self.A[0:3, 3:6] = np.eye(3) * dt
        
        # B矩阵构建（修正后）
        B_acc = 0.5 * dt**2 * np.eye(3)
        B_vel = dt * np.eye(3)
        B_yaw = np.zeros((3, 1))
        self.B = np.hstack([np.vstack([B_acc, B_vel]), np.vstack([B_yaw, B_yaw])])
        
        # 约束
        self.v_max = np.array([1.5, 1.5, 1])
        self.a_max = np.array([2, 2, 1.5])
        self.yaw_rate_max = np.radians(30)
    
    def solve(self, x0, x_ref_traj):
        x = cp.Variable((self.N+1, self.nx))
        u = cp.Variable((self.N, self.nu))
        cost = 0
        constraints = [x[0] == x0]
        
        for k in range(self.N):
            cost += cp.quad_form(x[k] - x_ref_traj[k], self.Q)
            cost += cp.quad_form(u[k], self.R)
            if k > 0:
                cost += cp.quad_form(u[k] - u[k-1], self.S)
            constraints += [
                x[k+1] == self.A @ x[k] + self.B @ u[k],
                cp.abs(x[k, 3:6]) <= self.v_max,
                cp.abs(u[k, :3]) <= self.a_max,
                cp.abs(u[k, 3]) <= self.yaw_rate_max
            ]
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        return x.value, u.value if prob.status == cp.OPTIMAL else (None, None)

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

# 初始化
mpc = DroneMPC(N=10, dt=0.2)
drone_state = np.array([0, 0, 0.4, 0, 0, 0])  # 初始状态 [px,py,pz,vx,vy,vz]
target_initial_pos = np.array([5, 5, 2])
target_vel = np.array([0.5, 0.3, 0])     # 移动平台速度 [vx,vy,vz]

# 存储历史轨迹
drone_history = []
target_history = []

# 创建图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(0, 6)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('MPC-Based Drone Landing on Moving Platform')

# 绘制初始位置
drone_line, = ax.plot([], [], [], 'bo-', label='Drone')
target_line, = ax.plot([], [], [], 'r*-', label='Target')
pred_line, = ax.plot([], [], [], 'g--', lw=1, label='MPC Prediction')
ax.legend()

# 在全局区域初始化目标位置
target_pos = np.array([5.0, 5.0, 2.0])  # 初始目标位置
target_vel = np.array([0.8, 0.6, 0])  # 目标速度

def update(frame):
    global drone_state, target_p , target_v  # 声明为全局变量
    
    target_traj = predict_target_trajectory(target_p, target_v, mpc.N, mpc.dt)
    
    # noice = np.random.normal(0, 0.0, 3)
    target_p += target_v * mpc.dt

    # 求解MPC
    x_opt, u_opt = mpc.solve(drone_state, target_traj)
    
    # 存储历史数据
    drone_history.append(drone_state.copy())
    target_history.append(target_p.copy())
    
    # 更新无人机状态（仅执行第一步控制）
    if u_opt is not None:
        drone_state = simulate_drone(drone_state, u_opt[0], mpc.dt)
    
    # 可视化更新
    if len(drone_history) > 1:
        drone_data = np.array(drone_history)
        target_data = np.array(target_history)
        drone_line.set_data(drone_data[:, 0], drone_data[:, 1])
        drone_line.set_3d_properties(drone_data[:, 2])
        target_line.set_data(target_data[:, 0], target_data[:, 1])
        target_line.set_3d_properties(target_data[:, 2])
        
        if x_opt is not None:  # 显示MPC预测轨迹
            pred_line.set_data(x_opt[:, 0], x_opt[:, 1])
            pred_line.set_3d_properties(x_opt[:, 2])
    
    return drone_line, target_line, pred_line

# 运行动画
rospy.init_node('position_command_publisher', anonymous=True)
target_listener = target()
pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)
ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)
plt.tight_layout()
plt.show()
