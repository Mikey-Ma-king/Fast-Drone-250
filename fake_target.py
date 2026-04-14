#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3
import math
import tf
import time
import random
import os

# 从环境变量读取随机种子，如果存在则设置
# 这样每次实验可以使用不同的随机种子，确保随机生成的颠簸区域不同
random_seed = os.environ.get('FAKE_TARGET_RANDOM_SEED')
if random_seed is not None:
    try:
        seed_value = int(random_seed)
        random.seed(seed_value)
        print(f"[fake_target] Set random seed to {seed_value} from environment variable")
    except ValueError:
        print(f"[fake_target] Warning: Invalid random seed value '{random_seed}', using default random seed")
else:
    # 如果没有设置环境变量，使用当前时间作为种子，确保每次运行都不同
    import time as time_module
    seed_value = int(time_module.time() * 1000000) % (2**32)
    random.seed(seed_value)
    print(f"[fake_target] Using time-based random seed: {seed_value}")


def transform_dog_pos(x, y, z, yaw, rotation_angle_deg=90.0, x_offset=5.0):
    """
    Transform dog position and yaw: rotate by specified angle (counterclockwise), then add offset to x-axis
    
    Args:
        x, y, z: Original position coordinates
        yaw: Original yaw angle (radians)
        rotation_angle_deg: Rotation angle in degrees (counterclockwise, default: 90.0)
        x_offset: Offset to add to x-axis after rotation (default: 5.0)
    
    Returns:
        Transformed (x, y, z, yaw) coordinates
    """
    # Convert rotation angle to radians
    rotation_angle = math.radians(rotation_angle_deg)
    cos_angle = math.cos(rotation_angle)
    sin_angle = math.sin(rotation_angle)
    
    # Rotate by specified angle counterclockwise: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
    x_rotated = x * cos_angle - y * sin_angle
    y_rotated = x * sin_angle + y * cos_angle
    
    # Add offset to x-axis
    x_transformed = x_rotated + x_offset
    
    # Rotate yaw by the same angle
    yaw_transformed = yaw + rotation_angle
    
    # Normalize yaw to [-pi, pi]
    while yaw_transformed > math.pi:
        yaw_transformed -= 2 * math.pi
    while yaw_transformed < -math.pi:
        yaw_transformed += 2 * math.pi
    
    return x_transformed, y_rotated, z, yaw_transformed

def publish_circle_motion():
    rospy.init_node('target_circle_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    # 圆周运动参数
    radius = rospy.get_param('~radius', 3.0)          # 圆的半径（m），可通过参数调整
    speed = rospy.get_param('~speed', 0.8)          # 线速度（m/s），可通过参数调整
    center_x = rospy.get_param('~center_x', 20.0)    # 圆心x坐标
    center_y = rospy.get_param('~center_y', 20.0)    # 圆心y坐标
    height = rospy.get_param('~height', 1.0)       # 高度（m）
    
    omega = speed / radius  # 角速度 rad/s
    start_time = time.time()
    
    rospy.loginfo("Starting circular motion: radius=%.2f m, speed=%.2f m/s, center=(%.2f, %.2f), height=%.2f m", 
                  radius, speed, center_x, center_y, height)
    rospy.loginfo("  target_ekf_odom_hz=%.1f, dog_pos_hz=%.1f, target_ekf_odom_noise_std=%.4f, dog_pos_noise_std=%.4f",
                  target_ekf_odom_hz, dog_pos_hz, target_ekf_odom_noise_std, dog_pos_noise_std)
    rospy.loginfo("  dog_pos_drift: offset_initial=(%.4f,%.4f,%.4f) m, offset_rate=(%.4f,%.4f,%.4f) m/s, yaw_initial=%.4f rad, yaw_rate=%.4f rad/s",
                  dog_pos_drift_offset_x, dog_pos_drift_offset_y, dog_pos_drift_offset_z,
                  dog_pos_drift_rate_x, dog_pos_drift_rate_y, dog_pos_drift_rate_z,
                  dog_pos_drift_yaw_initial, dog_pos_drift_yaw_rate)
    if vel_filter_alpha > 0:
        rospy.loginfo("  vel_filter_alpha=%.3f (velocity one-pole low-pass)", vel_filter_alpha)
    if publish_vins_zero:
        rospy.loginfo("  publish_vins_zero=true: /vins_fusion/imu_propagate -> (0,0,0) at 50 Hz")
    if target_ekf_odom_stop_after_sec >= 0:
        rospy.loginfo("  target_ekf_odom_stop_after_sec=%.1f s (will stop N s after param is set, e.g. after convergence)", target_ekf_odom_stop_after_sec)
    target_ekf_odom_gap_enabled = (target_ekf_odom_gap_window_sec > 0 and target_ekf_odom_gap_duration_sec > 0)
    if target_ekf_odom_gap_enabled:
        rospy.loginfo("  target_ekf_odom_gap: every window=%.1f s pick random start, then stop for duration=%.1f s (repeating); only target_ekf_odom lost, dog_pos unchanged",
                      target_ekf_odom_gap_window_sec, target_ekf_odom_gap_duration_sec)

    # target_ekf_odom：速度由位置有限差分
    last_odom_pos = None
    last_odom_time = None
    last_odom_vel = None
    # dog_pos：位置加噪声→用位置算速度→速度滤波→转 body→速度漂移；位置最后漂移
    last_pos_base = None   # 上一帧 (x+dx, y+dy, z+dz)，用于速度有限差分
    last_dog_vel_world = None  # 上一帧滤波后世界系速度，用于线性滤波
    stop_timer_start = None  # 从首次读到 target_ekf_odom_stop_after_sec>0 时开始计时（收敛后再停发）

    while not rospy.is_shutdown():
        # 每帧重新读取 freq、漂移参数与 target 丢失参数，便于实验脚本在收敛后再下发 stop 参数
        target_rate = max(0.0, float(rospy.get_param('~target_ekf_odom_hz', 15.0)))
        target_ekf_odom_stop_after_sec = rospy.get_param('~target_ekf_odom_stop_after_sec', -1.0)
        if target_ekf_odom_stop_after_sec >= 0 and stop_timer_start is None:
            stop_timer_start = time.time()
        dog_pos_drift_offset_x = rospy.get_param('~dog_pos_drift_offset_x', 0.0)
        dog_pos_drift_offset_y = rospy.get_param('~dog_pos_drift_offset_y', 0.0)
        dog_pos_drift_offset_z = rospy.get_param('~dog_pos_drift_offset_z', 0.0)
        dog_pos_drift_rate_x = rospy.get_param('~dog_pos_drift_rate_x', 0.0)
        dog_pos_drift_rate_y = rospy.get_param('~dog_pos_drift_rate_y', 0.0)
        dog_pos_drift_rate_z = rospy.get_param('~dog_pos_drift_rate_z', 0.0)
        dog_pos_drift_yaw_initial = rospy.get_param('~dog_pos_drift_yaw_initial', 0.0)
        dog_pos_drift_yaw_rate = rospy.get_param('~dog_pos_drift_yaw_rate', 0.0)

        current_time = time.time()
        t = (current_time - start_time)
        # 停止发布：从首次读到 param>0 起再过 target_ekf_odom_stop_after_sec 秒（实验在收敛后设置 param）
        target_ekf_lost = (stop_timer_start is not None and (current_time - stop_timer_start) >= target_ekf_odom_stop_after_sec)
        # 随机间歇（反复）：每 window 内随机选一起点，停发 duration 秒；用 segment_id 做种子保证同一段内起点固定
        if target_ekf_odom_gap_enabled:
            segment_id = int(t // target_ekf_odom_gap_window_sec)
            segment_begin = segment_id * target_ekf_odom_gap_window_sec
            max_offset = max(0.0, target_ekf_odom_gap_window_sec - target_ekf_odom_gap_duration_sec)
            rng = random.Random(segment_id + 12345)
            gap_start_t = segment_begin + rng.uniform(0.0, max_offset) if max_offset > 0 else segment_begin
            gap_end_t = gap_start_t + target_ekf_odom_gap_duration_sec
            if gap_start_t <= t < gap_end_t:
                target_ekf_lost = True

        # 圆周运动轨迹 (x = center_x + r*cos(θ), y = center_y + r*sin(θ))
        theta = omega * t
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        z = height  # 保持恒定高度

        # 速度方向（切线方向）
        vx = -speed * math.sin(theta)  # 速度在x方向的分量
        vy = speed * math.cos(theta)   # 速度在y方向的分量
        vz = 0.0

        # 朝向角度（与速度方向一致，即切线方向）
        yaw = theta + math.pi / 2.0  # 速度方向与半径垂直，所以角度+90度
        # 归一化到 [-pi, pi]
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "target_base"

        # 位置
        odom.pose.pose.position = Point(x, y, z)
        
        # 朝向（yaw存储在orientation.w中，与traj_server的格式一致）
        odom.pose.pose.orientation.w = yaw
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0

        # 速度
        odom.twist.twist.linear = Vector3(vx, vy, vz)
        odom.twist.twist.angular = Vector3(0, 0, omega)

        pub.publish(odom)
        rate.sleep()

def publish_oscillating_motion():
    """x轴速度跳变模式：10秒为1 m/s，10秒为-1 m/s"""
    rospy.init_node('target_oscillating_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    last_target_time = time.time()
    target_rate = 15.0  # 15 Hz for target_ekf_odom

    # 运动参数
    vx_positive = rospy.get_param('~vx_positive', 1.0)      # 正向速度（m/s）
    vx_negative = rospy.get_param('~vx_negative', -1.0)      # 负向速度（m/s）
    period = rospy.get_param('~period', 25.0)                # 每个方向的持续时间（秒）
    start_x = rospy.get_param('~start_x', 0.0)               # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)               # 起始y坐标
    height = rospy.get_param('~height', 0.4)                 # 高度（m）
    
    start_time = time.time()
    current_x = start_x
    current_y = start_y
    
    rospy.loginfo("Starting oscillating motion: vx_positive=%.2f m/s, vx_negative=%.2f m/s, period=%.2f s, start=(%.2f, %.2f), height=%.2f m", 
                  vx_positive, vx_negative, period, start_x, start_y, height)

    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        
        # 计算当前处于哪个周期（每20秒一个完整周期：10秒正向+10秒负向）
        cycle_time = t % (2 * period)  # 0 到 2*period
        
        # 根据时间判断当前速度方向
        if cycle_time < period:
            # 前10秒：正向速度
            vx = vx_positive
            yaw = 0.0  # 朝向x正方向
        else:
            # 后10秒：负向速度
            vx = vx_negative
            yaw = math.pi  # 朝向x负方向
        
        vy = 0.0  # y方向速度为0
        vz = 0.0  # z方向速度为0
        
        # 通过积分计算位置（使用速度和时间步长，50Hz）
        dt = 1.0 / 50.0  # 时间步长（50 Hz）
        current_x += vx * dt
        current_y += vy * dt
        z = height  # 保持恒定高度

        # 发布target_ekf_odom（15Hz）
        if current_time - last_target_time >= 1.0 / target_rate:
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"

            # 位置
            odom.pose.pose.position = Point(current_x, current_y, z)
            
            # 朝向（yaw存储在orientation.w中，与traj_server的格式一致）
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0

            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            odom.twist.twist.angular = Vector3(0, 0, 0)

        pub.publish(odom)
        rate.sleep()

if __name__ == '__main__':
    # 通过参数选择运动模式：'circle' 或 'oscillating'
    mode = 'oscillating'
    
    try:
        if mode == 'oscillating':
            publish_oscillating_motion()
        else:
            publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
