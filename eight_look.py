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

# 定义8字形轨迹的参数
radius = 1.5  # 圆的半径
T = 10  # 总周期T
angular_velocity = 2 * math.pi / T  # 角速度，单位弧度/秒
total_time = 2 * T  # 完整运动的总时间，两圈

# 轨迹的起点
start_time = 0.0  # 初始时间
end_time = total_time  # 结束时间

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
delta_s = 0.1
# n 个可能的新 delta_r 值
directions = np.array([
        [0, 0],  # 保持不变
        [delta_s, 0], [-delta_s, 0], [0, delta_s], [0, -delta_s],  # 单步方向
        [delta_s, delta_s], [delta_s, -delta_s], [-delta_s, delta_s], [-delta_s, -delta_s]  # 单步对角线
        # [2*delta_s, 0], [-2*delta_s, 0], [0, 2*delta_s], [0, -2*delta_s],  # 双步方向
        # [2*delta_s, 2*delta_s], [2*delta_s, -2*delta_s], [-2*delta_s, 2*delta_s], [-2*delta_s, -2*delta_s]  # 双步对角线
    ])

AOA_distance = 0.0
AOA_angle = 0.0

calibration_count = 50

flow_z = 1
target_ekf_odom =[0,0,0]

def get_yaw(t):
    yaw = 0
    if t < T/2:
        yaw = -math.pi/2 / (T/2) * t
    elif t <= 3*T/2:
        yaw = - math.pi/2 + (2*math.pi)/(T)*(t - T/2)
    else:
        yaw = 1.5*math.pi - (math.pi)/(T/2)*(t - 3*T/2)
    return yaw       

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
        # 用于保存T1和R1
        # 触发条件的标志
        
    def pose_cb0(self, msg):
        global AOA_distance
        global AOA_angle
        AOA_distance = msg.pose.pose.position.x
        AOA_angle = msg.pose.pose.orientation.w 

        # vins_p2d = np.array([vins_p[0],vins_p[1]])
        # r_p1 = gos_p - dog_p0
        # r_yaw1= dog_vins_yaw0 - dog_yaw0
        # cos_theta1 = math.cos(r_yaw1)
        # sin_theta1 = math.sin(r_yaw1)
        
        # rotated_x1 = cos_theta1 * r_p1[0] - sin_theta1 * r_p1[1]
        # rotated_y1 = sin_theta1 * r_p1[0] + cos_theta1 * r_p1[1]
        # r_p1[0] = rotated_x1
        # r_p1[1] = rotated_y1
        # target_p1 = r_p1 + dog_vins_p0
        # target_p2d = np.array([target_p1[0],target_p1[1]])
        # distance  = (msg.pose.pose.position.x**2 - (vins_p[2] - target_p1[2])**2)**0.5
        # if distance > 2.5:
        #     data_list.append([vins_p2d, target_p2d, distance])
        # if len(data_list) > 20:
        #     data_list.pop(0)  # 移除最旧的元素
        #     optimize_count += 1
        #     if optimize_count > 1 and np.linalg.norm(data_list[0][0] - vins_p2d) + np.linalg.norm(data_list[0][1] - target_p2d) > 0.3:
        #         # 优化求解
        #         # result = minimize(objective, [0,0], method='Nelder-Mead')
        #         # delta_r = result.x
        #         candidates = [(delta_r + direction, objective(delta_r + direction)) for direction in directions]
        #         # 选择 objective 最小的 delta_r
        #         delta_r, best_value = min(candidates, key=lambda x: x[1])
        #         print("delta_r:",delta_r,f"       error:{best_value:.4f}")
        #         optimize_count = 0

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
    

    if (target_yaw - vins_yaw > 0.2 and target_yaw - vins_yaw<3.14) or target_yaw - vins_yaw <= -3.14:
        yaw = vins_yaw + 0.08
    if (target_yaw - vins_yaw < -0.2 and target_yaw - vins_yaw > -3.14) or target_yaw - vins_yaw >= 3.14:
        yaw = vins_yaw - 0.08
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

script_path = '/home/pc/Fast-Perching/read.sh'
def run_shell_script(script_path):
    subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == '__main__':

    # 初始化ROS节点
    rospy.init_node('position_command_publisher', anonymous=True)
    flag = True
    uav_state_listener = UAVStateListener(flag)
    gos_listener = dog_listerer(flag)
    takeoff_listerer = take_listerer()
    a_flow_listerer = flow_listerer()

    AOA_listener = AOA_TAG()
    target_listener = target()
    
    time.sleep(1)

    number = input("是否开始执行任务：")

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

    # print("开始飞高")
    # for i in range(500):
    #     p.z = T1[2]+0.5+0.5*i/500
    #     publish_position(p, traj_id)
    #     rate.sleep()
    # 假设轨迹ID为1

      # 设置发布频率为100Hz
    T2,_yaw  = uav_state_listener.get_T1_R1()
    p.x = T2[0]  # 设置X坐标
    p.y = T2[1]  # 设置Y坐标
    p.z = T2[2] # 设置Z坐标
    publish_position(p, traj_id)
    publish_position(p, traj_id)
    init_p = T2

    # start_time = rospy.get_time()
    # print("开始八字")
    # while rospy.get_time() - start_time < total_time:
    #     # 发布位置命令
    #     p.x,p.y = get_position(rospy.get_time() - start_time)
    #     # print(rospy.get_time() - start_time , (p.x,p.y))
    #     p.z = T2[2]
    #     eight_yaw = get_yaw(rospy.get_time() - start_time)
    #     publish_position(p, traj_id, yaw = eight_yaw)

    #     # 睡眠直到下一次发布
    #     rate.sleep()

    # rospy.loginfo("Finished publishing position commands for 2*T time period.")

    # target_p = np.array(init_p)
    # target_p[0] += 4
    # target_p[1] += 2
    # target_yaw = 0
    # vins_p0xy = np.array([vins_p0[0],vins_p0[1]])
    # dog_p0xy = np.array([dog_p0[0],dog_p0[1]])
    # gos_pxy = np.array([gos_p[0],gos_p[1]])
    # r_pxy = np.array([gos_pxy[0] - dog_p0xy[0] + vins_p0xy[0] , gos_pxy[1] - dog_p0xy[1] + vins_p0xy[1]])
    # target_p = np.array([np.linalg.norm(r_pxy)*math.cos(math.atan2(r_pxy[1],r_pxy[0]) - yaw0),np.linalg.norm(r_pxy)*math.sin(math.atan2(r_pxy[1],r_pxy[0]) - yaw0),vins_p0[2] + 0.0])
    # target_yaw = gos_yaw - yaw0

    for i in range(100):
        p.x = vins_p[0]
        p.y = vins_p[1]
        p.z = min(vins_p[2] + 0.5 , 1.0)
        publish_position(p, traj_id ,0,0,0, 0)
        rate.sleep()

    somethinggggg = "y"
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
    #     p.z = 1.5
    #     (p, traj_id ,0,0,0, 0)
    #     rate.sleep(publish_position)
    
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




