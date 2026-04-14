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
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    ground_truth_pub = rospy.Publisher('/ground_truth_traj', Odometry, queue_size=10)
    # 若打开则每帧向 /vins_fusion/imu_propagate 发送 (0,0,0)
    publish_vins_zero = rospy.get_param('~publish_vins_zero', False)
    vins_zero_pub = rospy.Publisher('/vins_fusion/imu_propagate', Odometry, queue_size=10) if publish_vins_zero else None
    # 发布频率（Hz），可调参数
    target_ekf_odom_hz = rospy.get_param('~target_ekf_odom_hz', 15.0)
    dog_pos_hz = rospy.get_param('~dog_pos_hz', 50.0)
    dog_pos_rate = rospy.Rate(dog_pos_hz)
    target_rate = target_ekf_odom_hz  # Hz for target_ekf_odom
    last_target_time = time.time()

    # 圆周运动参数
    radius = rospy.get_param('~radius', 3.5)          # 圆的半径（m），可通过参数调整
    speed = rospy.get_param('~speed', 0.8)          # 线速度（m/s），可通过参数调整
    center_x = rospy.get_param('~center_x', 10.0)    # 圆心x坐标
    center_y = rospy.get_param('~center_y', 5.0)    # 圆心y坐标
    height = rospy.get_param('~height', 0.4)       # 高度（m）

    # target_ekf_odom 噪声：位置各轴高斯噪声标准差（m），0 表示无噪声
    target_ekf_odom_noise_std = rospy.get_param('~target_ekf_odom_noise_std', 0.05)
    # dog_pos 噪声：位置各轴高斯噪声标准差（m）
    dog_pos_noise_std = rospy.get_param('~dog_pos_noise_std', 0.0)
    # dog_pos 漂移：坐标系变换 = 平移(初始+累计) + 绕z旋转(初始+累计)
    # 平移：初始 offset（m）+ 漂移速率（m/s）每帧累加
    dog_pos_drift_offset_x = rospy.get_param('~dog_pos_drift_offset_x', 0.0)
    dog_pos_drift_offset_y = rospy.get_param('~dog_pos_drift_offset_y', 0.0)
    dog_pos_drift_offset_z = rospy.get_param('~dog_pos_drift_offset_z', 0.0)
    dog_pos_drift_rate_x = rospy.get_param('~dog_pos_drift_rate_x', 0.01)
    dog_pos_drift_rate_y = rospy.get_param('~dog_pos_drift_rate_y', 0.01)
    dog_pos_drift_rate_z = rospy.get_param('~dog_pos_drift_rate_z', 0.0)
    # 旋转：初始 yaw（rad）+ 漂移角速度（rad/s）每帧累加，最后对位置做绕 z 的旋转
    dog_pos_drift_yaw_initial = rospy.get_param('~dog_pos_drift_yaw_initial', 0.0)
    dog_pos_drift_yaw_rate = rospy.get_param('~dog_pos_drift_yaw_rate', 0.0)

    # 平移漂移累计量（m）、旋转漂移累计量（rad）
    drift_accum_x, drift_accum_y, drift_accum_z = 0.0, 0.0, 0.0
    yaw_drift_accum = 0.0
    last_dog_pos_time = None  # 首帧不累加漂移

    # 速度线性滤波：一阶指数平滑 v_out = alpha * v_prev + (1-alpha) * v_raw，0 表示不滤波
    vel_filter_alpha = rospy.get_param('~vel_filter_alpha', 0.95)
    vel_filter_alpha = max(0.0, min(1.0, float(vel_filter_alpha)))

    # target_ekf_odom 周期性丢失（只停发 target_ekf_odom，不丢失 dog_pos）：每 window 秒内随机停发 duration 秒
    target_ekf_odom_gap_window_sec = rospy.get_param('~target_ekf_odom_gap_window_sec', 15.0)
    target_ekf_odom_gap_duration_sec = rospy.get_param('~target_ekf_odom_gap_duration_sec', 2.0)
    # target_ekf_odom 永久丢失：>=0 表示从首次读到该参数为正起计时，N 秒后停止发布（可与 launch 或实验脚本设置）
    target_ekf_odom_stop_after_sec = rospy.get_param('~target_ekf_odom_stop_after_sec', -1.0)
    
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

        # 发布target_ekf_odom（15Hz），可加位置噪声；若 target_ekf_odom_stop_after_sec 触发则不再发布（模拟丢失）；freq=0 时不发布
        if target_rate > 0 and not target_ekf_lost and ((current_time - last_target_time) >= (1.0 / target_rate)):
            # 位置（加噪声）
            nx = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            ny = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            nz = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            pos_final_x = x + nx
            pos_final_y = y + ny
            pos_final_z = z + nz

            # 速度：根据最终位置有限差分，与发布的位置一致（带噪声）
            if last_odom_pos is not None and last_odom_time is not None:
                dt_odom = current_time - last_odom_time
                if dt_odom > 1e-6:
                    vx_out = (pos_final_x - last_odom_pos[0]) / dt_odom
                    vy_out = (pos_final_y - last_odom_pos[1]) / dt_odom
                    vz_out = (pos_final_z - last_odom_pos[2]) / dt_odom
                else:
                    vx_out, vy_out, vz_out = vx, vy, vz
            else:
                vx_out, vy_out, vz_out = vx, vy, vz
            # 速度线性滤波：一阶指数平滑
            if vel_filter_alpha > 0 and last_odom_vel is not None:
                vx_out = vel_filter_alpha * last_odom_vel[0] + (1.0 - vel_filter_alpha) * vx_out
                vy_out = vel_filter_alpha * last_odom_vel[1] + (1.0 - vel_filter_alpha) * vy_out
                vz_out = vel_filter_alpha * last_odom_vel[2] + (1.0 - vel_filter_alpha) * vz_out
            last_odom_vel = (vx_out, vy_out, vz_out)
            last_odom_pos = (pos_final_x, pos_final_y, pos_final_z)
            last_odom_time = current_time

            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"
            odom.pose.pose.position = Point(pos_final_x, pos_final_y, pos_final_z)
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0
            odom.twist.twist.linear = Vector3(vx_out, vy_out, vz_out)
            odom.twist.twist.angular = Vector3(0, 0, omega)
            pub.publish(odom)
            last_target_time = current_time
        
        # 漂移：每帧把平移速率*dt、旋转角速度*dt 累加（首帧 dt=0）
        last_dog_pos_time_prev = last_dog_pos_time  # 保存上一帧时间，供速度有限差分用
        if last_dog_pos_time is not None:
            dt_dog = current_time - last_dog_pos_time
            drift_accum_x += dog_pos_drift_rate_x * dt_dog
            drift_accum_y += dog_pos_drift_rate_y * dt_dog
            drift_accum_z += dog_pos_drift_rate_z * dt_dog
            yaw_drift_accum += dog_pos_drift_yaw_rate * dt_dog
        last_dog_pos_time = current_time

        # ---------- dog_pos：先加噪声 → 用位置算速度 → 速度滤波 → 转 body → 最后漂移 ----------
        # 1. 噪声
        dx = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0
        dy = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0
        dz = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0

        # 2. 位置先算好（真值 + 噪声）
        pos_base_wx = x + dx
        pos_base_wy = y + dy
        pos_base_wz = z + dz

        # 3. 速度按位置有限差分得到（世界系）
        if last_pos_base is not None and last_dog_pos_time_prev is not None:
            dt_vel = current_time - last_dog_pos_time_prev
            if dt_vel > 1e-6:
                vw_x = (pos_base_wx - last_pos_base[0]) / dt_vel
                vw_y = (pos_base_wy - last_pos_base[1]) / dt_vel
                vw_z = (pos_base_wz - last_pos_base[2]) / dt_vel
            else:
                vw_x, vw_y, vw_z = vx, vy, vz
        else:
            vw_x, vw_y, vw_z = vx, vy, vz
        last_pos_base = (pos_base_wx, pos_base_wy, pos_base_wz)

        # 4. 速度滤波（世界系，一阶指数平滑）
        if vel_filter_alpha > 0 and last_dog_vel_world is not None:
            vw_x = vel_filter_alpha * last_dog_vel_world[0] + (1.0 - vel_filter_alpha) * vw_x
            vw_y = vel_filter_alpha * last_dog_vel_world[1] + (1.0 - vel_filter_alpha) * vw_y
            vw_z = vel_filter_alpha * last_dog_vel_world[2] + (1.0 - vel_filter_alpha) * vw_z
        last_dog_vel_world = (vw_x, vw_y, vw_z)

        # 5. 速度转到 body 系（用真实朝向 yaw）
        cos_yaw_true = math.cos(yaw)
        sin_yaw_true = math.sin(yaw)
        vx_body = vw_x * cos_yaw_true + vw_y * sin_yaw_true
        vy_body = -vw_x * sin_yaw_true + vw_y * cos_yaw_true
        vz_body = vw_z

        # 6. 漂移量（位置旋转+平移，速度 body 系加偏置，朝向）
        yaw_drift_total = dog_pos_drift_yaw_initial + yaw_drift_accum
        offset_x = dog_pos_drift_offset_x + drift_accum_x
        offset_y = dog_pos_drift_offset_y + drift_accum_y
        offset_z = dog_pos_drift_offset_z + drift_accum_z
        c, s = math.cos(yaw_drift_total), math.sin(yaw_drift_total)
        # 位置最后施加漂移
        pos_final_wx = pos_base_wx * c - pos_base_wy * s + offset_x
        pos_final_wy = pos_base_wx * s + pos_base_wy * c + offset_y
        pos_final_wz = pos_base_wz + offset_z
        # 速度最后施加漂移（body 系加偏置）
        vx_dog = vx_body + dog_pos_drift_rate_x
        vy_dog = vy_body + dog_pos_drift_rate_y
        vz_dog = vz_body + dog_pos_drift_rate_z

        yaw_out = yaw + yaw_drift_total
        while yaw_out > math.pi:
            yaw_out -= 2 * math.pi
        while yaw_out < -math.pi:
            yaw_out += 2 * math.pi

        # 发布dog_pos（50Hz），位置世界系，速度 body 系
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        dog_pos_msg.pose.pose.position = Point(pos_final_wx, pos_final_wy, pos_final_wz)
        dog_pos_msg.pose.pose.orientation.w = yaw_out
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz_dog)
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        dog_pos_pub.publish(dog_pos_msg)

        # 发布真实轨迹（不带噪声和漂移，50Hz）
        ground_truth_msg = Odometry()
        ground_truth_msg.header.stamp = rospy.Time.now()
        ground_truth_msg.header.frame_id = "world"
        ground_truth_msg.pose.pose.position = Point(x, y, z)
        ground_truth_msg.pose.pose.orientation.w = yaw
        ground_truth_msg.pose.pose.orientation.x = 0.0
        ground_truth_msg.pose.pose.orientation.y = 0.0
        ground_truth_msg.pose.pose.orientation.z = 0.0
        ground_truth_msg.twist.twist.linear = Vector3(vx, vy, vz)
        ground_truth_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        ground_truth_pub.publish(ground_truth_msg)

        # 若打开：每帧向 /vins_fusion/imu_propagate 发送 (0,0,0)
        if vins_zero_pub is not None:
            vins_msg = Odometry()
            vins_msg.header.stamp = rospy.Time.now()
            vins_msg.header.frame_id = "world"
            vins_msg.child_frame_id = "body"
            vins_msg.pose.pose.position = Point(0.0, 0.0, 0.0)
            vins_msg.pose.pose.orientation.w = 1.0
            vins_msg.pose.pose.orientation.x = 0.0
            vins_msg.pose.pose.orientation.y = 0.0
            vins_msg.pose.pose.orientation.z = 0.0
            vins_msg.twist.twist.linear = Vector3(0.0, 0.0, 0.0)
            vins_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
            vins_zero_pub.publish(vins_msg)

        dog_pos_rate.sleep()

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
    period = rospy.get_param('~period', 5.0)                # 每个方向的持续时间（秒）
    start_x = rospy.get_param('~start_x', -30.0)               # 起始x坐标
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
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，速度需要转换到狗坐标系）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        
        # 位置（世界坐标系）
        dog_pos_msg.pose.pose.position = Point(current_x, current_y, z)
        
        # yaw（世界坐标系下的yaw，存储在orientation.w中）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度需要从世界坐标系转换到狗坐标系（狗头yaw=0的坐标系）
        # 从世界坐标系旋转到狗坐标系：旋转-yaw角度
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # vx_dog = vx_world * cos(yaw) + vy_world * sin(yaw)
        # vy_dog = -vx_world * sin(yaw) + vy_world * cos(yaw)
        vx_dog = vx * cos_yaw + vy * sin_yaw
        vy_dog = -vx * sin_yaw + vy * cos_yaw
        
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
        
        # 角速度（振荡运动为0）
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
        
        dog_pos_pub.publish(dog_pos_msg)
        dog_pos_rate.sleep()

def publish_straight_line_motion():
    """直线行走：沿着指定方向匀速直线运动（功能与 publish_circle_motion 一致：噪声、漂移、滤波、ground_truth、vins_zero、target_ekf_odom_stop 等）"""
    rospy.init_node('target_straight_line_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    ground_truth_pub = rospy.Publisher('/ground_truth_traj', Odometry, queue_size=10)
    publish_vins_zero = rospy.get_param('~publish_vins_zero', True)
    vins_zero_pub = rospy.Publisher('/vins_fusion/imu_propagate', Odometry, queue_size=10) if publish_vins_zero else None
    target_ekf_odom_hz = rospy.get_param('~target_ekf_odom_hz', 15.0)
    dog_pos_hz = rospy.get_param('~dog_pos_hz', 50.0)
    dog_pos_rate = rospy.Rate(dog_pos_hz)
    target_rate = target_ekf_odom_hz
    last_target_time = time.time()

    # 直线运动参数
    speed = rospy.get_param('~speed', 1.0)
    direction = rospy.get_param('~direction', math.pi)
    start_x = rospy.get_param('~start_x', 10.0)
    start_y = rospy.get_param('~start_y', 10.0)
    height = rospy.get_param('~height', 0.4)

    target_ekf_odom_noise_std = rospy.get_param('~target_ekf_odom_noise_std', 0.1)
    dog_pos_noise_std = rospy.get_param('~dog_pos_noise_std', 0.05)
    dog_pos_drift_offset_x = rospy.get_param('~dog_pos_drift_offset_x', 0.0)
    dog_pos_drift_offset_y = rospy.get_param('~dog_pos_drift_offset_y', 0.0)
    dog_pos_drift_offset_z = rospy.get_param('~dog_pos_drift_offset_z', 0.0)
    dog_pos_drift_rate_x = rospy.get_param('~dog_pos_drift_rate_x', 0.01)
    dog_pos_drift_rate_y = rospy.get_param('~dog_pos_drift_rate_y', 0.01)
    dog_pos_drift_rate_z = rospy.get_param('~dog_pos_drift_rate_z', 0.0)
    dog_pos_drift_yaw_initial = rospy.get_param('~dog_pos_drift_yaw_initial', 0.0)
    dog_pos_drift_yaw_rate = rospy.get_param('~dog_pos_drift_yaw_rate', 0.01)

    drift_accum_x, drift_accum_y, drift_accum_z = 0.0, 0.0, 0.0
    yaw_drift_accum = 0.0
    last_dog_pos_time = None

    vel_filter_alpha = rospy.get_param('~vel_filter_alpha', 0.95)
    vel_filter_alpha = max(0.0, min(1.0, float(vel_filter_alpha)))

    try:
        _stop_sec = rospy.get_param('/target_straight_line_publisher/target_ekf_odom_stop_after_sec', -1.0)
    except Exception:
        _stop_sec = -1.0
    last_odom_vel = None
    last_pos_base = None
    last_dog_vel_world = None

    start_time = time.time()
    omega = 0.0  # 直线运动无角速度

    rospy.loginfo("Starting straight line motion: speed=%.2f m/s, direction=%.2f rad, start=(%.2f, %.2f), height=%.2f m",
                  speed, direction, start_x, start_y, height)
    rospy.loginfo("  target_ekf_odom_hz=%.1f, dog_pos_hz=%.1f, target_ekf_odom_noise_std=%.4f, dog_pos_noise_std=%.4f",
                  target_ekf_odom_hz, dog_pos_hz, target_ekf_odom_noise_std, dog_pos_noise_std)
    rospy.loginfo("  dog_pos_drift: offset_initial=(%.4f,%.4f,%.4f) m, offset_rate=(%.4f,%.4f,%.4f) m/s, yaw_initial=%.4f rad, yaw_rate=%.4f rad/s",
                  dog_pos_drift_offset_x, dog_pos_drift_offset_y, dog_pos_drift_offset_z,
                  dog_pos_drift_rate_x, dog_pos_drift_rate_y, dog_pos_drift_rate_z,
                  dog_pos_drift_yaw_initial, dog_pos_drift_yaw_rate)
    if vel_filter_alpha > 0:
        rospy.loginfo("  vel_filter_alpha=%.3f (velocity one-pole low-pass)", vel_filter_alpha)
    if publish_vins_zero:
        rospy.loginfo("  publish_vins_zero=true: /vins_fusion/imu_propagate -> (0,0,0) at %.1f Hz", dog_pos_hz)
    if _stop_sec >= 0:
        rospy.loginfo("  target_ekf_odom_stop_after_sec=%.1f s (will stop N s after param is set, e.g. after convergence)", _stop_sec)

    stop_timer_start = None
    while not rospy.is_shutdown():
        try:
            target_ekf_odom_stop_after_sec = rospy.get_param('/target_straight_line_publisher/target_ekf_odom_stop_after_sec', -1.0)
        except Exception:
            target_ekf_odom_stop_after_sec = -1.0
        if target_ekf_odom_stop_after_sec >= 0 and stop_timer_start is None:
            stop_timer_start = time.time()
        current_time = time.time()
        t = (current_time - start_time)
        target_ekf_lost = (stop_timer_start is not None and (current_time - stop_timer_start) >= target_ekf_odom_stop_after_sec)

        # 直线轨迹
        x = start_x + speed * math.cos(direction) * t
        y = start_y + speed * math.sin(direction) * t
        z = height

        vx = speed * math.cos(direction)
        vy = speed * math.sin(direction)
        vz = 0.0

        yaw = direction
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi

        # 发布 target_ekf_odom
        if not target_ekf_lost and ((current_time - last_target_time) >= (1.0 / target_rate)):
            nx = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            ny = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            nz = random.gauss(0, target_ekf_odom_noise_std) if target_ekf_odom_noise_std > 0 else 0.0
            pos_final_x = x + nx
            pos_final_y = y + ny
            pos_final_z = z + nz

            if last_odom_pos is not None and last_odom_time is not None:
                dt_odom = current_time - last_odom_time
                if dt_odom > 1e-6:
                    vx_out = (pos_final_x - last_odom_pos[0]) / dt_odom
                    vy_out = (pos_final_y - last_odom_pos[1]) / dt_odom
                    vz_out = (pos_final_z - last_odom_pos[2]) / dt_odom
                else:
                    vx_out, vy_out, vz_out = vx, vy, vz
            else:
                vx_out, vy_out, vz_out = vx, vy, vz
            if vel_filter_alpha > 0 and last_odom_vel is not None:
                vx_out = vel_filter_alpha * last_odom_vel[0] + (1.0 - vel_filter_alpha) * vx_out
                vy_out = vel_filter_alpha * last_odom_vel[1] + (1.0 - vel_filter_alpha) * vy_out
                vz_out = vel_filter_alpha * last_odom_vel[2] + (1.0 - vel_filter_alpha) * vz_out
            last_odom_vel = (vx_out, vy_out, vz_out)
            last_odom_pos = (pos_final_x, pos_final_y, pos_final_z)
            last_odom_time = current_time

            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"
            odom.pose.pose.position = Point(pos_final_x, pos_final_y, pos_final_z)
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0
            odom.twist.twist.linear = Vector3(vx_out, vy_out, vz_out)
            odom.twist.twist.angular = Vector3(0, 0, omega)
            pub.publish(odom)
            last_target_time = current_time

        last_dog_pos_time_prev = last_dog_pos_time
        if last_dog_pos_time is not None:
            dt_dog = current_time - last_dog_pos_time
            drift_accum_x += dog_pos_drift_rate_x * dt_dog
            drift_accum_y += dog_pos_drift_rate_y * dt_dog
            drift_accum_z += dog_pos_drift_rate_z * dt_dog
            yaw_drift_accum += dog_pos_drift_yaw_rate * dt_dog
        last_dog_pos_time = current_time

        dx = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0
        dy = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0
        dz = random.gauss(0, dog_pos_noise_std) if dog_pos_noise_std > 0 else 0.0

        pos_base_wx = x + dx
        pos_base_wy = y + dy
        pos_base_wz = z + dz

        if last_pos_base is not None and last_dog_pos_time_prev is not None:
            dt_vel = current_time - last_dog_pos_time_prev
            if dt_vel > 1e-6:
                vw_x = (pos_base_wx - last_pos_base[0]) / dt_vel
                vw_y = (pos_base_wy - last_pos_base[1]) / dt_vel
                vw_z = (pos_base_wz - last_pos_base[2]) / dt_vel
            else:
                vw_x, vw_y, vw_z = vx, vy, vz
        else:
            vw_x, vw_y, vw_z = vx, vy, vz
        last_pos_base = (pos_base_wx, pos_base_wy, pos_base_wz)

        if vel_filter_alpha > 0 and last_dog_vel_world is not None:
            vw_x = vel_filter_alpha * last_dog_vel_world[0] + (1.0 - vel_filter_alpha) * vw_x
            vw_y = vel_filter_alpha * last_dog_vel_world[1] + (1.0 - vel_filter_alpha) * vw_y
            vw_z = vel_filter_alpha * last_dog_vel_world[2] + (1.0 - vel_filter_alpha) * vw_z
        last_dog_vel_world = (vw_x, vw_y, vw_z)

        cos_yaw_true = math.cos(yaw)
        sin_yaw_true = math.sin(yaw)
        vx_body = vw_x * cos_yaw_true + vw_y * sin_yaw_true
        vy_body = -vw_x * sin_yaw_true + vw_y * cos_yaw_true
        vz_body = vw_z

        yaw_drift_total = dog_pos_drift_yaw_initial + yaw_drift_accum
        offset_x = dog_pos_drift_offset_x + drift_accum_x
        offset_y = dog_pos_drift_offset_y + drift_accum_y
        offset_z = dog_pos_drift_offset_z + drift_accum_z
        c, s = math.cos(yaw_drift_total), math.sin(yaw_drift_total)
        pos_final_wx = pos_base_wx * c - pos_base_wy * s + offset_x
        pos_final_wy = pos_base_wx * s + pos_base_wy * c + offset_y
        pos_final_wz = pos_base_wz + offset_z
        vx_dog = vx_body + dog_pos_drift_rate_x
        vy_dog = vy_body + dog_pos_drift_rate_y
        vz_dog = vz_body + dog_pos_drift_rate_z

        yaw_out = yaw + yaw_drift_total
        while yaw_out > math.pi:
            yaw_out -= 2 * math.pi
        while yaw_out < -math.pi:
            yaw_out += 2 * math.pi

        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        dog_pos_msg.pose.pose.position = Point(pos_final_wx, pos_final_wy, pos_final_wz)
        dog_pos_msg.pose.pose.orientation.w = yaw_out
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz_dog)
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        dog_pos_pub.publish(dog_pos_msg)

        ground_truth_msg = Odometry()
        ground_truth_msg.header.stamp = rospy.Time.now()
        ground_truth_msg.header.frame_id = "world"
        ground_truth_msg.pose.pose.position = Point(x, y, z)
        ground_truth_msg.pose.pose.orientation.w = yaw
        ground_truth_msg.pose.pose.orientation.x = 0.0
        ground_truth_msg.pose.pose.orientation.y = 0.0
        ground_truth_msg.pose.pose.orientation.z = 0.0
        ground_truth_msg.twist.twist.linear = Vector3(vx, vy, vz)
        ground_truth_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        ground_truth_pub.publish(ground_truth_msg)

        if vins_zero_pub is not None:
            vins_msg = Odometry()
            vins_msg.header.stamp = rospy.Time.now()
            vins_msg.header.frame_id = "world"
            vins_msg.child_frame_id = "body"
            vins_msg.pose.pose.position = Point(0.0, 0.0, 0.0)
            vins_msg.pose.pose.orientation.w = 1.0
            vins_msg.pose.pose.orientation.x = 0.0
            vins_msg.pose.pose.orientation.y = 0.0
            vins_msg.pose.pose.orientation.z = 0.0
            vins_msg.twist.twist.linear = Vector3(0.0, 0.0, 0.0)
            vins_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
            vins_zero_pub.publish(vins_msg)

        dog_pos_rate.sleep()

def publish_sin_curve_motion():
    """Sin曲线行走：沿着x方向前进，y方向按sin函数变化"""
    rospy.init_node('target_sin_curve_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    target_rate = 15.0  # 15 Hz for target_ekf_odom
    last_target_time = time.time()

    # Sin曲线运动参数
    vx_forward = rospy.get_param('~vx_forward', 1.0)    # x方向前进速度（m/s）
    amplitude = rospy.get_param('~amplitude', 3.0)      # y方向振幅（m）
    period = rospy.get_param('~period', 16.0)          # 一个完整周期的长度（m，在x方向）
    start_x = rospy.get_param('~start_x', -10.0)          # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)          # 起始y坐标
    height = rospy.get_param('~height', 0.4)            # 高度（m）
    
    start_time = time.time()
    start_x_pos = start_x
    current_x = start_x  # 当前x位置
    direction = 1.0  # 方向：1.0表示正向，-1.0表示反向
    last_loop_time = time.time()
    
    rospy.loginfo("Starting sin curve motion: vx=%.2f m/s, amplitude=%.2f m, period=%.2f m, start=(%.2f, %.2f), height=%.2f m", 
                  vx_forward, amplitude, period, start_x, start_y, height)

    while not rospy.is_shutdown():
        current_time = time.time()
        loop_dt = current_time - last_loop_time
        last_loop_time = current_time
        
        # 检查是否需要反向
        if current_x > 100.0 and direction > 0:
            direction = -1.0  # 反向
            rospy.loginfo("Reversing direction at x=%.2f (x > 50)", current_x)
        elif current_x < -50.0 and direction < 0:
            direction = 1.0  # 正向
            rospy.loginfo("Reversing direction at x=%.2f (x < -50)", current_x)
        
        # 计算实际速度（考虑方向）
        actual_vx = vx_forward * direction
        
        # 更新x位置（使用积分方式）
        current_x += actual_vx * loop_dt
        
        x = current_x
        
        # y方向：按sin函数变化
        # y = amplitude * sin(2*pi * x / period)
        y = start_y + amplitude * math.sin(2 * math.pi * x / period)
        
        # 速度计算
        # vx = dx/dt = vx_forward * direction（考虑方向）
        vx = actual_vx
        # vy = dy/dt = amplitude * (2*pi/period) * cos(2*pi*x/period) * vx_forward * direction
        vy = amplitude * (2 * math.pi / period) * math.cos(2 * math.pi * x / period) * actual_vx
        vz = 0.0
        
        # 朝向角度：速度方向
        yaw = math.atan2(vy, vx)
        
        # 归一化yaw到 [-pi, pi]
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi
        
        z = height  # 保持恒定高度

        # 发布target_ekf_odom（15Hz）
        if current_time - last_target_time >= 1.0 / target_rate:
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"

            # 位置
            odom.pose.pose.position = Point(x, y, z)
            
            # 朝向（yaw存储在orientation.w中）
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0

            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            # 角速度：yaw的变化率
            # omega = d(yaw)/dt = d(atan2(vy, vx))/dt
            # 简化计算：omega ≈ (d(vy)/dt * vx - d(vx)/dt * vy) / (vx^2 + vy^2)
            # 由于vx是常数，d(vx)/dt = 0
            # d(vy)/dt = -amplitude * (2*pi/period)^2 * sin(2*pi*x/period) * actual_vx^2
            dvy_dt = -amplitude * (2 * math.pi / period) ** 2 * math.sin(2 * math.pi * x / period) * actual_vx * actual_vx
            omega = (dvy_dt * vx) / (vx * vx + vy * vy) if (vx * vx + vy * vy) > 1e-6 else 0.0
            odom.twist.twist.angular = Vector3(0, 0, omega)

            pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，速度需要转换到狗坐标系）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        
        # 位置（世界坐标系）
        dog_pos_msg.pose.pose.position = Point(x, y, z)
        
        # yaw（世界坐标系下的yaw，存储在orientation.w中）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度需要从世界坐标系转换到狗坐标系（狗头yaw=0的坐标系）
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_dog = vx * cos_yaw + vy * sin_yaw
        vy_dog = -vx * sin_yaw + vy * cos_yaw
        
        dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
        
        # 角速度
        dvy_dt = -amplitude * (2 * math.pi / period) ** 2 * math.sin(2 * math.pi * x / period) * actual_vx * actual_vx
        omega = (dvy_dt * vx) / (vx * vx + vy * vy) if (vx * vx + vy * vy) > 1e-6 else 0.0
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, omega)
        
        dog_pos_pub.publish(dog_pos_msg)
        dog_pos_rate.sleep()

def publish_triangle_motion_with_sensors():
    """三角形轨迹运动，根据vins反馈和传感器类型添加噪声和误差"""
    rospy.init_node('target_triangle_sensor_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    ground_truth_pub = rospy.Publisher('/ground_truth_traj', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    target_rate = 15.0  # 15 Hz for target_ekf_odom
    last_target_time = time.time()
    
    # 三角形轨迹参数（按用户指定的方式）
    speed = rospy.get_param('~speed', 1.0)              # 速度（m/s）
    start_x = rospy.get_param('~start_x', 0.0)          # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)          # 起始y坐标
    height = rospy.get_param('~height', 0.4)            # 高度（m）
    triangle_side_length = rospy.get_param('~triangle_side_length', 7.0)  # 三角形边长（m）
    segment2_acceleration = rospy.get_param('~segment2_acceleration', 0.5)  # 第二段加速度（m/s²），默认0.5
    
    # 颠簸区域定义（矩形区域）
    # 如果参数为 None，则在 triangle_sensor 模式下会随机生成
    bumpy_area_x_min = rospy.get_param('~bumpy_area_x_min', None)  # 颠簸区域x最小值，None表示随机生成
    bumpy_area_x_max = rospy.get_param('~bumpy_area_x_max', None)  # 颠簸区域x最大值，None表示随机生成
    bumpy_area_y_min = rospy.get_param('~bumpy_area_y_min', None)  # 颠簸区域y最小值，None表示随机生成
    bumpy_area_y_max = rospy.get_param('~bumpy_area_y_max', None)  # 颠簸区域y最大值，None表示随机生成
    bumpy_area_size = rospy.get_param('~bumpy_area_size', 25.0)  # 随机生成颠簸区域的面积（m²），默认50
    bumpy_area_aspect_ratio_min = rospy.get_param('~bumpy_area_aspect_ratio_min', 0.5)  # 长宽比下限（宽/长或长/宽），默认0.5
    bumpy_area_aspect_ratio_max = rospy.get_param('~bumpy_area_aspect_ratio_max', 2.0)  # 长宽比上限（宽/长或长/宽），默认2.0
    bumpy_area_random_x_min = rospy.get_param('~bumpy_area_random_x_min', start_x - 3.0)  # 随机生成区域的x最小值
    bumpy_area_random_x_max = rospy.get_param('~bumpy_area_random_x_max', start_x + 3.0)  # 随机生成区域的x最大值
    bumpy_area_random_y_min = rospy.get_param('~bumpy_area_random_y_min', start_y - 3.0)  # 随机生成区域的y最小值
    bumpy_area_random_y_max = rospy.get_param('~bumpy_area_random_y_max', start_y + 12.0)  # 随机生成区域的y最大值

    # ARUCO传感器参数（用于/target_ekf_odom）
    aruco_max_horizontal_dist_coeff = rospy.get_param('~aruco_max_horizontal_dist_coeff', 1.0)  # 水平距离限制系数，实际限制 = 系数 × 高度差
    aruco_min_vertical_dist = rospy.get_param('~aruco_min_vertical_dist', 0.2)      # 最小竖直距离（m），小于此距离不可见
    aruco_noise_normal = rospy.get_param('~aruco_noise_normal', 0.10)               # 正常区域噪声幅值（m）
    aruco_noise_bumpy = rospy.get_param('~aruco_noise_bumpy', 0.30)                 # 颠簸区域噪声幅值（m）
    aruco_error_bumpy_x_min = rospy.get_param('~aruco_error_bumpy_x_min', 0.20)      # 颠簸区域x方向稳态误差最小值（m）
    aruco_error_bumpy_x_max = rospy.get_param('~aruco_error_bumpy_x_max', 0.30)      # 颠簸区域x方向稳态误差最大值（m）
    
    # HC14传感器参数（用于/dog_pos）
    hc14_noise = rospy.get_param('~hc14_noise', 0.05)                               # 噪声幅值（m）
    hc14_error_x_rate = rospy.get_param('~hc14_error_x_rate', 0.0)                # x方向稳态误差增加速率（m/s），0.5cm/s = 0.005m/s
    hc14_error_y_rate = rospy.get_param('~hc14_error_y_rate', 0.005)                  # y方向稳态误差增加速率（m/s）
    
    # 发布模式参数
    dog_pos_publish_mode = rospy.get_param('~dog_pos_publish_mode', 'always')  # 'always' 或 'aruco_visible'
    # 'always': 始终发布（50Hz）
    # 'aruco_visible': 仅在ARUCO可见时发布（与target_ekf_odom相同条件）
    
    # dog_pos变换参数
    dog_pos_rotation_angle_deg = rospy.get_param('~dog_pos_rotation_angle_deg', 90.0)  # 旋转角度（度，逆时针，默认90度）
    dog_pos_x_offset = rospy.get_param('~dog_pos_x_offset', 5.0)  # x轴偏移量（米，默认5.0米）
    
    # VINS位置（用于计算相对距离和噪声）
    vins_pos = [0.0, 0.0, 0.0]
    vins_pos_lock = threading.Lock()
    
    # HC14稳态误差累积
    hc14_x_error = 0.0
    hc14_x_error_start_time = None
    hc14_y_error = 0.0
    hc14_y_error_start_time = None
    
    # ARUCO颠簸区域稳态误差（随机生成一次）
    aruco_bumpy_x_error = random.uniform(aruco_error_bumpy_x_min, aruco_error_bumpy_x_max)
    
    # 如果颠簸区域参数为 None，则随机生成一个矩形区域
    if (bumpy_area_x_min is None or bumpy_area_x_max is None or 
        bumpy_area_y_min is None or bumpy_area_y_max is None):
        # 使用参数指定的随机生成区域范围
        random_area_x_min = bumpy_area_random_x_min
        random_area_x_max = bumpy_area_random_x_max
        random_area_y_min = bumpy_area_random_y_min
        random_area_y_max = bumpy_area_random_y_max
        
        # 计算可用区域大小
        available_width = random_area_x_max - random_area_x_min
        available_height = random_area_y_max - random_area_y_min
        available_area = available_width * available_height
        
        # 确保请求的面积不超过可用区域
        target_area = min(bumpy_area_size, available_area)
        
        # 随机生成矩形：随机长宽比 -> 计算边长 -> 随机中心点 -> clip到范围内
        # 1. 随机选择长宽比（在限制范围内）
        aspect_ratio = random.uniform(bumpy_area_aspect_ratio_min, bumpy_area_aspect_ratio_max)
        
        # 2. 根据面积和长宽比计算边长
        # aspect_ratio = width/height，area = width * height
        # area = aspect_ratio * height^2，所以 height = sqrt(area / aspect_ratio)
        # width = area / height = area / sqrt(area / aspect_ratio) = sqrt(area * aspect_ratio)
        rect_height = math.sqrt(target_area / aspect_ratio)
        rect_width = target_area / rect_height
        
        # 3. 随机生成中心点（在可用区域内，确保矩形能完全放入）
        center_x = random.uniform(random_area_x_min + rect_width / 2.0, 
                                 random_area_x_max - rect_width / 2.0)
        center_y = random.uniform(random_area_y_min + rect_height / 2.0, 
                                 random_area_y_max - rect_height / 2.0)
        
        # 4. 根据中心点和尺寸计算边界
        x_min = center_x - rect_width / 2.0
        x_max = center_x + rect_width / 2.0
        y_min = center_y - rect_height / 2.0
        y_max = center_y + rect_height / 2.0
        
        # 5. Clip到指定范围内
        x_min = max(x_min, random_area_x_min)
        x_max = min(x_max, random_area_x_max)
        y_min = max(y_min, random_area_y_min)
        y_max = min(y_max, random_area_y_max)
        
        # 更新颠簸区域参数
        bumpy_area_x_min = x_min
        bumpy_area_x_max = x_max
        bumpy_area_y_min = y_min
        bumpy_area_y_max = y_max
        rospy.loginfo("Generated bumpy area: center=(%.2f, %.2f), size=(%.2f x %.2f), aspect_ratio=%.2f, clipped to x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                     center_x, center_y, rect_width, rect_height, aspect_ratio,
                     bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max)
    
    # 将颠簸区域参数写回到参数服务器，以便其他节点可以读取
    node_name = rospy.get_name()  # 获取节点名称（包含命名空间）
    rospy.set_param(f'{node_name}/bumpy_area_x_min', bumpy_area_x_min)
    rospy.set_param(f'{node_name}/bumpy_area_x_max', bumpy_area_x_max)
    rospy.set_param(f'{node_name}/bumpy_area_y_min', bumpy_area_y_min)
    rospy.set_param(f'{node_name}/bumpy_area_y_max', bumpy_area_y_max)
    rospy.loginfo("颠簸区域参数已设置到参数服务器: x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                  bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max)
    
    def is_in_bumpy_area(x, y):
        """
        检查点(x, y)是否在颠簸区域内
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            bool: 如果点在颠簸区域内返回True，否则返回False
        """
        # 如果所有边界都是None，则没有定义颠簸区域，返回False
        if (bumpy_area_x_min is None and bumpy_area_x_max is None and 
            bumpy_area_y_min is None and bumpy_area_y_max is None):
            return False
        
        # 检查x坐标
        if bumpy_area_x_min is not None and x < bumpy_area_x_min:
            return False
        if bumpy_area_x_max is not None and x > bumpy_area_x_max:
            return False
        
        # 检查y坐标
        if bumpy_area_y_min is not None and y < bumpy_area_y_min:
            return False
        if bumpy_area_y_max is not None and y > bumpy_area_y_max:
            return False
        
        return True
    
    def vins_callback(msg):
        """VINS位置回调函数"""
        nonlocal vins_pos
        with vins_pos_lock:
            vins_pos[0] = msg.pose.pose.position.x
            vins_pos[1] = msg.pose.pose.position.y
            vins_pos[2] = msg.pose.pose.position.z
    
    # 订阅VINS位置
    vins_sub = rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, vins_callback, queue_size=10)
    
    # 三角形轨迹参数（按用户指定的方式）
    # 第一段：沿x正方向跑
    segment1_x_end = start_x + triangle_side_length / 2.0  # 第一段结束的x坐标
    
    # 计算三角形轨迹
    # 第一段：从(start_x, start_y)沿x正方向跑到(segment1_x_end, start_y)
    # 第二段：从(segment1_x_end, start_y)沿105度方向跑，直到x=start_x
    # 第三段：从(start_x, y_apex)沿-105度方向跑，直到y=start_y
    # 然后循环回到第一段
    
    angle_105 = math.radians(105.0)  # 105度
    angle_minus_105 = math.radians(-105.0)  # -105度
    
    # 第二段：从(segment1_x_end, start_y)沿105度方向
    # x = segment1_x_end + speed * cos(105°) * t
    # 当x=start_x时：segment1_x_end + speed * cos(105°) * t = start_x
    # t = (start_x - segment1_x_end) / (speed * cos(105°))
    # 由于cos(105°) < 0，且start_x < segment1_x_end，所以t > 0
    cos_105 = math.cos(angle_105)
    sin_105 = math.sin(angle_105)
    segment2_length = (segment1_x_end - start_x) / abs(cos_105)  # 第二段长度
    y_apex = start_y + segment2_length * sin_105  # 第二段结束时的y坐标（顶点y坐标）
    
    # 第三段：从(start_x, y_apex)沿-105度方向，直到y=start_y
    # y = y_apex + speed * sin(-105°) * t
    # 当y=start_y时：y_apex + speed * sin(-105°) * t = start_y
    # t = (start_y - y_apex) / (speed * sin(-105°))
    cos_minus_105 = math.cos(angle_minus_105)
    sin_minus_105 = math.sin(angle_minus_105)
    segment3_length = (y_apex - start_y) / abs(sin_minus_105)  # 第三段长度
    
    # 计算第三段结束位置
    x_segment3_end = start_x + segment3_length * cos_minus_105
    y_segment3_end = start_y  # 第三段结束时y=start_y
    
    # 第四段：绕起始点的半圆（下半部分，180~360度）
    # 半径
    circle_radius = triangle_side_length / 2.0
    # 半圆长度（π * radius）
    segment4_length = math.pi * circle_radius
    
    # 计算各段长度
    segment1_length = segment1_x_end - start_x
    # 总路径长度：第一段 + 第二段 + 第三段 + 第四段（半圆）
    total_path_length = segment1_length + segment2_length + segment3_length + segment4_length
    
    # 第0段：静止在起点5秒
    segment0_duration = rospy.get_param('~segment0_duration', 1.0)  # 第0段持续时间（秒）
    
    start_time = time.time()
    current_path_distance = 0.0  # 当前沿路径的距离
    
    rospy.loginfo("Starting triangle motion with sensors: speed=%.2f m/s", speed)
    rospy.loginfo("  Bumpy area: x=[%.2f, %.2f], y=[%.2f, %.2f] (area=%.2f m²)", 
                 bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max,
                 (bumpy_area_x_max - bumpy_area_x_min) * (bumpy_area_y_max - bumpy_area_y_min))
    rospy.loginfo("  Triangle segments: 0->静止%.1f秒, 1->x=%.2f (x+), 2->x=%.2f (105deg), 3->y=%.2f (-105deg), 4->semicircle (radius=%.2f, 180-360deg), then 2->3->4 loop", 
                  segment0_duration, segment1_x_end, start_x, start_y, circle_radius)
    rospy.loginfo("  /target_ekf_odom: ARUCO sensor characteristics")
    rospy.loginfo("    Max horizontal dist coeff: %.2f (limit = coeff × height_diff), Min vertical distance: %.2f m", 
                  aruco_max_horizontal_dist_coeff, aruco_min_vertical_dist)
    rospy.loginfo("    Normal noise: ±%.2f m, Bumpy noise: ±%.2f m", 
                  aruco_noise_normal, aruco_noise_bumpy)
    rospy.loginfo("    Bumpy x error: %.2f~%.2f m (current: %.2f m)", 
                  aruco_error_bumpy_x_min, aruco_error_bumpy_x_max, aruco_bumpy_x_error)
    rospy.loginfo("  /dog_pos: HC14 sensor characteristics")
    rospy.loginfo("    Noise: ±%.2f m, X error rate: %.4f m/s, Y error rate: %.4f m/s", 
                  hc14_noise, hc14_error_x_rate, hc14_error_y_rate)
    rospy.loginfo("    Publish mode: %s", dog_pos_publish_mode)
    rospy.loginfo("    Transform: rotation=%.1f deg (counterclockwise), x_offset=%.2f m", 
                  dog_pos_rotation_angle_deg, dog_pos_x_offset)
    
    while not rospy.is_shutdown():
        current_time = time.time()
        t = (current_time - start_time)
        
        # 第0段：静止在起点5秒
        if t < segment0_duration:
            x = 0
            y = 0
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw = 0.0  # 面向x正方向
            is_bumpy = is_in_bumpy_area(x, y)
        else:
            # 更新路径距离（循环），减去第0段的时间
            motion_time = t - segment0_duration
            
            # 计算一个完整周期（第一段+第二段+第三段+第四段）的长度
            full_cycle_length = segment1_length + segment2_length + segment3_length + segment4_length
            # 后续周期（第二段+第三段+第四段）的长度
            subsequent_cycle_length = segment2_length + segment3_length + segment4_length
            
            # 判断是否在第一个周期
            if motion_time * speed < full_cycle_length:
                # 第一个周期：包含第一段
                current_path_distance = speed * motion_time
            else:
                # 后续周期：跳过第一段，从第二段开始
                remaining_time = motion_time * speed - full_cycle_length
                current_path_distance = segment1_length + (remaining_time % subsequent_cycle_length)
            
            # 计算当前位置（沿三角形路径）
            if current_path_distance <= segment1_length:
                # 第一段：沿x正方向跑
                x = start_x + current_path_distance
                y = start_y
                vx = speed
                vy = 0.0
                is_bumpy = is_in_bumpy_area(x, y)
            elif current_path_distance <= segment1_length + segment2_length:
                # 第二段：沿105度方向跑，直到x=start_x
                # 速度控制：先减速后加速
                # 减速点：当x > start_x + triangle_side_length/4时开始减速
                # 前半段减速，后半段加速
                segment_dist = current_path_distance - segment1_length
                
                # 计算减速点的x坐标
                deceleration_x = start_x + triangle_side_length / 4.0
                # 从segment1_x_end到deceleration_x的沿轨迹距离
                # x方向距离 = segment1_x_end - deceleration_x
                # 沿轨迹距离 = (segment1_x_end - deceleration_x) / abs(cos_105)
                deceleration_dist = (segment1_x_end - deceleration_x) / abs(cos_105)
                
                # 计算中点（第二段的中点）
                segment2_midpoint_dist = segment2_length / 2.0
                
                # 计算当前速度
                if segment_dist <= deceleration_dist:
                    # 从起点到减速点：匀速（保持初始速度）
                    current_speed = speed
                elif segment_dist <= segment2_midpoint_dist:
                    # 减速段：从减速点到中点
                    decel_segment_dist = segment_dist - deceleration_dist
                    decel_length = segment2_midpoint_dist - deceleration_dist
                    
                    # 使用匀减速运动：v^2 = v0^2 - 2*a*s
                    # v0 = speed, a = segment2_acceleration, s = decel_segment_dist
                    v_squared = speed * speed - 2 * segment2_acceleration * decel_segment_dist
                    current_speed = math.sqrt(max(0.01, v_squared))  # 确保速度不为负
                else:
                    # 加速段：从中点到终点（后半段）
                    accel_segment_dist = segment_dist - segment2_midpoint_dist
                    accel_length = segment2_length - segment2_midpoint_dist
                    
                    # 计算中点速度（从减速段得到）
                    decel_total_dist = segment2_midpoint_dist - deceleration_dist
                    v_mid_squared = speed * speed - 2 * segment2_acceleration * decel_total_dist
                    v_mid = math.sqrt(max(0.01, v_mid_squared))
                    
                    # 使用匀加速运动：v^2 = v0^2 + 2*a*s
                    # v0 = v_mid, a = segment2_acceleration, s = accel_segment_dist
                    v_squared = v_mid * v_mid + 2 * segment2_acceleration * accel_segment_dist
                    current_speed = math.sqrt(v_squared)
                    # 限制最大速度不超过初始速度的2倍
                    current_speed = min(speed * 2.0, current_speed)
                
                # 计算位置（使用segment_dist，因为位置计算基于路径距离）
                x = segment1_x_end + segment_dist * cos_105
                y = start_y + segment_dist * sin_105
                
                # 速度方向沿105度
                vx = current_speed * cos_105
                vy = current_speed * sin_105
                is_bumpy = is_in_bumpy_area(x, y)
            elif current_path_distance <= segment1_length + segment2_length + segment3_length:
                # 第三段：沿-105度方向跑，直到y=start_y
                segment_dist = current_path_distance - segment1_length - segment2_length
                x = start_x + segment_dist * cos_minus_105
                y = y_apex + segment_dist * sin_minus_105
                vx = speed * cos_minus_105
                vy = speed * sin_minus_105
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                # 第四段：绕起始点的半圆（下半部分，180~360度）
                segment_dist = current_path_distance - segment1_length - segment2_length - segment3_length
                # 计算角度：从180度（π）开始，到360度（2π）结束
                # angle = π + (segment_dist / circle_radius)
                angle = math.pi + (segment_dist / circle_radius)
                # 限制角度范围在 [π, 2π]
                angle = min(angle, 2 * math.pi)
                
                # 计算半圆上的位置（相对于起始点）
                x_relative = circle_radius * math.cos(angle)
                y_relative = circle_radius * math.sin(angle)
                
                # 转换为世界坐标
                x = start_x + x_relative
                y = start_y + y_relative
                
                # 计算速度方向（切线方向）
                # 切线方向垂直于半径方向，角度为 angle + π/2
                vx = speed * math.cos(angle + math.pi / 2.0)
                vy = speed * math.sin(angle + math.pi / 2.0)
                is_bumpy = is_in_bumpy_area(x, y)
            
            vz = 0.0
            yaw = math.atan2(vy, vx)
            
            # 归一化yaw到 [-pi, pi]
            while yaw > math.pi:
                yaw -= 2 * math.pi
            while yaw < -math.pi:
                yaw += 2 * math.pi
        
        # 获取VINS位置
        with vins_pos_lock:
            vins_x, vins_y, vins_z = vins_pos
        
        # 计算相对距离
        horizontal_dist = math.sqrt((x - vins_x)**2 + (y - vins_y)**2)
        vertical_dist = abs(height - vins_z)
        
        # ARUCO传感器特性（用于target_ekf_odom）
        aruco_noise_x = 0.0
        aruco_noise_y = 0.0
        aruco_error_x = 0.0
        aruco_visible = True
        
        # 计算实际水平距离限制（系数 × 高度差）
        aruco_max_horizontal_dist = aruco_max_horizontal_dist_coeff * vertical_dist
        
        # 检查ARUCO是否可见
        if horizontal_dist > aruco_max_horizontal_dist or vertical_dist < aruco_min_vertical_dist:
            aruco_visible = False
            rospy.logwarn_throttle(1.0, "ARUCO: Target not visible (h_dist=%.2f m > %.2f m (coeff=%.2f × v_dist=%.2f m) or v_dist=%.2f m < %.2f m)", 
                                  horizontal_dist, aruco_max_horizontal_dist, aruco_max_horizontal_dist_coeff, vertical_dist, vertical_dist, aruco_min_vertical_dist)
        else:
            # ARUCO噪声（仅在xy平面）
            if is_bumpy:
                # 颠簸区域：噪声±aruco_noise_bumpy
                aruco_noise_x = random.uniform(-aruco_noise_bumpy, aruco_noise_bumpy)
                aruco_noise_y = random.uniform(-aruco_noise_bumpy, aruco_noise_bumpy)
                # x方向稳态误差
                aruco_error_x = aruco_bumpy_x_error
            else:
                # 正常区域：噪声±aruco_noise_normal
                aruco_noise_x = random.uniform(-aruco_noise_normal, aruco_noise_normal)
                aruco_noise_y = random.uniform(-aruco_noise_normal, aruco_noise_normal)
                aruco_error_x = 0.0
        
        # HC14传感器特性（用于dog_pos，噪音仅在xy平面）
        hc14_noise_x = random.uniform(-hc14_noise, hc14_noise)
        hc14_noise_y = random.uniform(-hc14_noise, hc14_noise)
        
        # x方向持续增加的稳态误差
        if hc14_x_error_start_time is None:
            hc14_x_error_start_time = current_time
        elapsed_time = current_time - hc14_x_error_start_time
        hc14_x_error = hc14_error_x_rate * elapsed_time
        hc14_error_x = hc14_x_error
        
        # y方向持续增加的稳态误差
        if hc14_y_error_start_time is None:
            hc14_y_error_start_time = current_time
        elapsed_time_y = current_time - hc14_y_error_start_time
        hc14_y_error = hc14_error_y_rate * elapsed_time_y
        hc14_error_y = hc14_y_error
        
        # 应用噪声和误差（ARUCO用于target_ekf_odom，z方向无噪音）
        x_aruco = x + aruco_noise_x + aruco_error_x
        y_aruco = y + aruco_noise_y
        z_aruco = height
        
        # 应用噪声和误差（HC14用于dog_pos，z方向无噪音）
        x_hc14 = x + hc14_noise_x + hc14_error_x
        y_hc14 = y + hc14_noise_y + hc14_error_y
        z_hc14 = height
        
        # 发布target_ekf_odom（15Hz，使用ARUCO特性）
        if current_time - last_target_time >= 1.0 / target_rate:
            # 只有在ARUCO可见时才发布
            if aruco_visible:
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = "world"
                odom.child_frame_id = "target_base"
                
                # 位置（带ARUCO噪声和误差）
                odom.pose.pose.position = Point(x_aruco, y_aruco, z_aruco)
                
                # 朝向
                odom.pose.pose.orientation.w = yaw
                odom.pose.pose.orientation.x = 0
                odom.pose.pose.orientation.y = 0
                odom.pose.pose.orientation.z = 0
                
                # 速度（世界坐标系）
                odom.twist.twist.linear = Vector3(vx, vy, vz)
                odom.twist.twist.angular = Vector3(0, 0, 0)
                
                pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，使用HC14特性，速度需要转换到狗坐标系）
        # 根据发布模式决定是否发布
        should_publish_dog_pos = True
        if dog_pos_publish_mode == 'aruco_visible':
            # 仅在ARUCO可见时发布（与target_ekf_odom相同条件）
            should_publish_dog_pos = aruco_visible
        
        if should_publish_dog_pos:
            dog_pos_msg = Odometry()
            dog_pos_msg.header.stamp = rospy.Time.now()
            dog_pos_msg.header.frame_id = "world"
            
            # 位置和yaw变换：先旋转指定角度，然后x轴加上指定偏移（在应用HC14噪声和误差之后）
            x_transformed, y_transformed, z_transformed, yaw_transformed = transform_dog_pos(
                x_hc14, y_hc14, z_hc14, yaw, 
                rotation_angle_deg=dog_pos_rotation_angle_deg, 
                x_offset=dog_pos_x_offset)
            dog_pos_msg.pose.pose.position = Point(x_transformed, y_transformed, z_transformed)
            
            # yaw（已旋转90度）
            dog_pos_msg.pose.pose.orientation.w = yaw_transformed
            dog_pos_msg.pose.pose.orientation.x = 0.0
            dog_pos_msg.pose.pose.orientation.y = 0.0
            dog_pos_msg.pose.pose.orientation.z = 0.0
            
            # 速度需要从世界坐标系转换到狗坐标系
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            vx_dog = vx * cos_yaw + vy * sin_yaw
            vy_dog = -vx * sin_yaw + vy * cos_yaw
            
            dog_pos_msg.twist.twist.linear = Vector3(vx_dog, vy_dog, vz)
            dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
            
            dog_pos_pub.publish(dog_pos_msg)
        
        # 发布真实轨迹（不带噪声和误差，50Hz）
        ground_truth_msg = Odometry()
        ground_truth_msg.header.stamp = rospy.Time.now()
        ground_truth_msg.header.frame_id = "world"
        
        # 位置（真实位置，无噪声和误差）
        ground_truth_msg.pose.pose.position = Point(x, y, height)
        
        # yaw
        ground_truth_msg.pose.pose.orientation.w = yaw
        ground_truth_msg.pose.pose.orientation.x = 0.0
        ground_truth_msg.pose.pose.orientation.y = 0.0
        ground_truth_msg.pose.pose.orientation.z = 0.0
        
        # 速度（世界坐标系）
        ground_truth_msg.twist.twist.linear = Vector3(vx, vy, vz)
        ground_truth_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
        
        ground_truth_pub.publish(ground_truth_msg)
        dog_pos_rate.sleep()

def publish_triangle_motion_with_aruco_trigger():
    """三角形轨迹运动，使用ARUCO触发逻辑：初始无条件发布，ARUCO可见后切换为仅在ARUCO可见时发布"""
    rospy.init_node('target_triangle_aruco_trigger_publisher')
    pub = rospy.Publisher('/target_ekf_odom', Odometry, queue_size=10)
    dog_pos_pub = rospy.Publisher('/dog_pos', Odometry, queue_size=10)
    ground_truth_pub = rospy.Publisher('/ground_truth_traj', Odometry, queue_size=10)
    dog_pos_rate = rospy.Rate(50)  # 50 Hz for dog_pos
    target_rate = 15.0  # 15 Hz for target_ekf_odom
    last_target_time = time.time()
    
    # 三角形轨迹参数（按用户指定的方式）
    speed = rospy.get_param('~speed', 1.0)              # 速度（m/s）
    start_x = rospy.get_param('~start_x', 0.0)          # 起始x坐标
    start_y = rospy.get_param('~start_y', 0.0)          # 起始y坐标
    height = rospy.get_param('~height', 0.4)            # 高度（m）
    triangle_side_length = rospy.get_param('~triangle_side_length', 8.0)  # 三角形边长（m）
    segment2_acceleration = rospy.get_param('~segment2_acceleration', 0.5)  # 第二段加速度（m/s²），默认0.5
    
    # 颠簸区域定义（矩形区域）
    bumpy_area_x_min = rospy.get_param('~bumpy_area_x_min', None)  # 颠簸区域x最小值，None表示随机生成
    bumpy_area_x_max = rospy.get_param('~bumpy_area_x_max', None)  # 颠簸区域x最大值，None表示随机生成
    bumpy_area_y_min = rospy.get_param('~bumpy_area_y_min', None)  # 颠簸区域y最小值，None表示随机生成
    bumpy_area_y_max = rospy.get_param('~bumpy_area_y_max', None)  # 颠簸区域y最大值，None表示随机生成
    bumpy_area_size = rospy.get_param('~bumpy_area_size', 30.0)  # 随机生成颠簸区域的面积（m²），默认30
    bumpy_area_aspect_ratio_min = rospy.get_param('~bumpy_area_aspect_ratio_min', 0.5)  # 长宽比下限（宽/长或长/宽），默认0.5
    bumpy_area_aspect_ratio_max = rospy.get_param('~bumpy_area_aspect_ratio_max', 2.0)  # 长宽比上限（宽/长或长/宽），默认2.0
    bumpy_area_random_x_min = rospy.get_param('~bumpy_area_random_x_min', start_x - 3.0)  # 随机生成区域的x最小值
    bumpy_area_random_x_max = rospy.get_param('~bumpy_area_random_x_max', start_x + 3.0)  # 随机生成区域的x最大值
    bumpy_area_random_y_min = rospy.get_param('~bumpy_area_random_y_min', start_y - 3.0)  # 随机生成区域的y最小值
    bumpy_area_random_y_max = rospy.get_param('~bumpy_area_random_y_max', start_y + 12.0)  # 随机生成区域的y最大值
    
    # ARUCO传感器参数（用于/target_ekf_odom和/dog_pos）
    aruco_max_horizontal_dist_coeff = rospy.get_param('~aruco_max_horizontal_dist_coeff', 1.0)  # 水平距离限制系数，实际限制 = 系数 × 高度差
    aruco_min_vertical_dist = rospy.get_param('~aruco_min_vertical_dist', 0.2)      # 最小竖直距离（m），小于此距离不可见
    aruco_noise_normal = rospy.get_param('~aruco_noise_normal', 0.10)               # 正常区域噪声幅值（m）
    aruco_noise_bumpy = rospy.get_param('~aruco_noise_bumpy', 0.30)                 # 颠簸区域噪声幅值（m）
    aruco_error_bumpy_x_amplitude = rospy.get_param('~aruco_error_bumpy_x_amplitude', 0.25)  # 颠簸区域x方向稳态误差幅值（m）
    aruco_error_bumpy_y_amplitude = rospy.get_param('~aruco_error_bumpy_y_amplitude', 0.25)  # 颠簸区域y方向稳态误差幅值（m）
    
    # VINS位置（用于计算相对距离和噪声）
    vins_pos = [0.0, 0.0, 0.0]
    vins_pos_lock = threading.Lock()
    
    # ARUCO颠簸区域稳态误差（随机生成一次）
    aruco_bumpy_x_error = random.uniform(-aruco_error_bumpy_x_amplitude, aruco_error_bumpy_x_amplitude)
    aruco_bumpy_y_error = random.uniform(-aruco_error_bumpy_y_amplitude, aruco_error_bumpy_y_amplitude)
    
    # 如果颠簸区域参数为 None，则随机生成一个矩形区域
    if (bumpy_area_x_min is None or bumpy_area_x_max is None or 
        bumpy_area_y_min is None or bumpy_area_y_max is None):
        # 使用参数指定的随机生成区域范围
        random_area_x_min = bumpy_area_random_x_min
        random_area_x_max = bumpy_area_random_x_max
        random_area_y_min = bumpy_area_random_y_min
        random_area_y_max = bumpy_area_random_y_max
        
        # 计算可用区域大小
        available_width = random_area_x_max - random_area_x_min
        available_height = random_area_y_max - random_area_y_min
        available_area = available_width * available_height
        
        # 确保请求的面积不超过可用区域
        target_area = min(bumpy_area_size, available_area)
        
        # 随机生成矩形：随机长宽比 -> 计算边长 -> 随机中心点 -> clip到范围内
        # 1. 随机选择长宽比（在限制范围内）
        aspect_ratio = random.uniform(bumpy_area_aspect_ratio_min, bumpy_area_aspect_ratio_max)
        
        # 2. 根据面积和长宽比计算边长
        # aspect_ratio = width/height，area = width * height
        # area = aspect_ratio * height^2，所以 height = sqrt(area / aspect_ratio)
        # width = area / height = area / sqrt(area / aspect_ratio) = sqrt(area * aspect_ratio)
        rect_height = math.sqrt(target_area / aspect_ratio)
        rect_width = target_area / rect_height
        
        # 3. 随机生成中心点（在可用区域内，确保矩形能完全放入）
        center_x = random.uniform(random_area_x_min + rect_width / 2.0, 
                                 random_area_x_max - rect_width / 2.0)
        center_y = random.uniform(random_area_y_min + rect_height / 2.0, 
                                 random_area_y_max - rect_height / 2.0)
        
        # 4. 根据中心点和尺寸计算边界
        x_min = center_x - rect_width / 2.0
        x_max = center_x + rect_width / 2.0
        y_min = center_y - rect_height / 2.0
        y_max = center_y + rect_height / 2.0
        
        # 5. Clip到指定范围内
        x_min = max(x_min, random_area_x_min)
        x_max = min(x_max, random_area_x_max)
        y_min = max(y_min, random_area_y_min)
        y_max = min(y_max, random_area_y_max)
        
        # 更新颠簸区域参数
        bumpy_area_x_min = x_min
        bumpy_area_x_max = x_max
        bumpy_area_y_min = y_min
        bumpy_area_y_max = y_max
        rospy.loginfo("Generated bumpy area: center=(%.2f, %.2f), size=(%.2f x %.2f), aspect_ratio=%.2f, clipped to x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                     center_x, center_y, rect_width, rect_height, aspect_ratio,
                     bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max)
    
    # 将颠簸区域参数写回到参数服务器，以便其他节点可以读取
    node_name = rospy.get_name()  # 获取节点名称（包含命名空间）
    rospy.set_param(f'{node_name}/bumpy_area_x_min', bumpy_area_x_min)
    rospy.set_param(f'{node_name}/bumpy_area_x_max', bumpy_area_x_max)
    rospy.set_param(f'{node_name}/bumpy_area_y_min', bumpy_area_y_min)
    rospy.set_param(f'{node_name}/bumpy_area_y_max', bumpy_area_y_max)
    rospy.loginfo("颠簸区域参数已设置到参数服务器: x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                  bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max)
    
    def is_in_bumpy_area(x, y):
        """
        检查点(x, y)是否在颠簸区域内
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            bool: 如果点在颠簸区域内返回True，否则返回False
        """
        return (bumpy_area_x_min <= x <= bumpy_area_x_max and 
                bumpy_area_y_min <= y <= bumpy_area_y_max)
    
    # VINS位置订阅回调
    def vins_pose_callback(msg):
        with vins_pos_lock:
            vins_pos[0] = msg.pose.pose.position.x
            vins_pos[1] = msg.pose.pose.position.y
            vins_pos[2] = msg.pose.pose.position.z
    
    rospy.Subscriber('/vins_fusion/imu_propagate', Odometry, vins_pose_callback)
    
    # 距离触发状态量（平台和无人机在xy平面的距离超过2m时变为true）
    distance_trigger = False
    distance_threshold = 2.0  # 距离阈值（m）
    
    # ARUCO触发状态量（只有当distance_trigger为true且aruco_visible为true时，才会变为true）
    aruco_triggered = False
    
    # 外推状态量（当aruco_triggered为true且aruco_visible为false时，外推1秒）
    extrapolating = False
    extrapolation_start_time = None
    extrapolation_start_pos = None  # [x, y, z]
    extrapolation_start_vel = None  # [vx, vy, vz]
    extrapolation_start_yaw = None
    
    # 轨迹生成参数
    start_time = time.time()
    segment = 0  # 当前段：0=静止, 1=第一段, 2=第二段, 3=第三段, 4=第四段
    segment_start_time = start_time
    first_cycle_complete = False  # 标记第一轮是否完成
    
    rospy.loginfo("Starting triangle motion with ARUCO trigger logic")
    rospy.loginfo("  Speed: %.2f m/s", speed)
    rospy.loginfo("  Start: (%.2f, %.2f)", start_x, start_y)
    rospy.loginfo("  Height: %.2f m", height)
    rospy.loginfo("  Triangle side length: %.2f m", triangle_side_length)
    rospy.loginfo("  Bumpy area: x=[%.2f, %.2f], y=[%.2f, %.2f]", 
                  bumpy_area_x_min, bumpy_area_x_max, bumpy_area_y_min, bumpy_area_y_max)
    
    while not rospy.is_shutdown():
        current_time = time.time()
        elapsed = current_time - segment_start_time
        
        # 轨迹生成逻辑（与publish_triangle_motion_with_sensors相同）
        x = start_x
        y = start_y
        vx = 0.0
        vy = 0.0
        is_bumpy = False
        
        if segment == 0:
            # 段0：静止5秒，面向x正方向
            if elapsed < 5.0:
                x = start_x
                y = start_y
                vx = 0.0
                vy = 0.0
                yaw = 0.0
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                segment = 1
                segment_start_time = current_time
                elapsed = 0.0
        
        if segment == 1:
            # 段1：沿x轴正方向移动
            segment1_x_end = start_x + triangle_side_length / 2.0
            segment_length = triangle_side_length / 2.0
            if elapsed * speed < segment_length:
                x = start_x + elapsed * speed
                y = start_y
                vx = speed
                vy = 0.0
                yaw = 0.0
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                segment = 2
                segment_start_time = current_time
                elapsed = 0.0
        
        if segment == 2:
            # 段2：沿y轴正方向移动
            # 速度控制：先减速后加速
            # 减速点：当y > start_y + triangle_side_length/4时开始减速
            # 前半段减速，后半段加速
            segment_length = triangle_side_length
            
            # 计算已移动的距离（基于匀速假设）
            distance_traveled = elapsed * speed
            
            if distance_traveled < segment_length:
                # 计算减速点的y坐标和对应的距离
                deceleration_y = start_y + triangle_side_length / 4.0
                deceleration_dist = deceleration_y - start_y
                
                # 计算中点（第二段的中点）
                segment2_midpoint_dist = segment_length / 2.0
                
                # 计算当前速度
                if distance_traveled <= deceleration_dist:
                    # 从起点到减速点：匀速（保持初始速度）
                    current_speed = speed
                elif distance_traveled <= segment2_midpoint_dist:
                    # 减速段：从减速点到中点
                    decel_segment_dist = distance_traveled - deceleration_dist
                    
                    # 使用匀减速运动：v^2 = v0^2 - 2*a*s
                    # v0 = speed, a = segment2_acceleration, s = decel_segment_dist
                    v_squared = speed * speed - 2 * segment2_acceleration * decel_segment_dist
                    current_speed = math.sqrt(max(0.01, v_squared))  # 确保速度不为负
                else:
                    # 加速段：从中点到终点（后半段）
                    accel_segment_dist = distance_traveled - segment2_midpoint_dist
                    
                    # 计算中点速度（从减速段得到）
                    decel_total_dist = segment2_midpoint_dist - deceleration_dist
                    v_mid_squared = speed * speed - 2 * segment2_acceleration * decel_total_dist
                    v_mid = math.sqrt(max(0.01, v_mid_squared))
                    
                    # 使用匀加速运动：v^2 = v0^2 + 2*a*s
                    # v0 = v_mid, a = segment2_acceleration, s = accel_segment_dist
                    v_squared = v_mid * v_mid + 2 * segment2_acceleration * accel_segment_dist
                    current_speed = math.sqrt(v_squared)
                    # 限制最大速度不超过初始速度的2倍
                    current_speed = min(speed * 2.0, current_speed)
                
                # 计算位置（使用distance_traveled，因为位置计算基于路径距离）
                x = start_x + triangle_side_length / 2.0
                y = start_y + distance_traveled
                
                # 速度方向沿y轴正方向
                vx = 0.0
                vy = current_speed
                yaw = math.pi / 2.0
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                segment = 3
                segment_start_time = current_time
                elapsed = 0.0
        
        if segment == 3:
            # 段3：沿斜边返回
            segment_length = triangle_side_length
            if elapsed * speed < segment_length:
                x = start_x + triangle_side_length / 2.0 - elapsed * speed * math.cos(math.pi / 3.0)
                y = start_y + triangle_side_length - elapsed * speed * math.sin(math.pi / 3.0)
                vx = -speed * math.cos(math.pi / 3.0)
                vy = -speed * math.sin(math.pi / 3.0)
                yaw = math.pi / 3.0 + math.pi
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                segment = 4
                segment_start_time = current_time
                elapsed = 0.0
        
        if segment == 4:
            # 段4：半圆（180-360度）
            radius = triangle_side_length / 2.0
            angle_start = math.pi  # 180度
            angle_end = 2 * math.pi  # 360度
            segment_length = radius * (angle_end - angle_start)
            
            if elapsed * speed < segment_length:
                angle = angle_start + (elapsed * speed) / radius
                x_relative = radius * math.cos(angle)
                y_relative = radius * math.sin(angle)
                x = start_x + x_relative
                y = start_y + y_relative
                vx = speed * math.cos(angle + math.pi / 2.0)
                vy = speed * math.sin(angle + math.pi / 2.0)
                yaw = angle + math.pi / 2.0
                is_bumpy = is_in_bumpy_area(x, y)
            else:
                if not first_cycle_complete:
                    first_cycle_complete = True
                segment = 2  # 直接跳到段2
                segment_start_time = current_time
                elapsed = 0.0
        
        vz = 0.0
        if segment != 0:
            yaw = math.atan2(vy, vx)
            while yaw > math.pi:
                yaw -= 2 * math.pi
            while yaw < -math.pi:
                yaw += 2 * math.pi
        
        # 获取VINS位置
        with vins_pos_lock:
            vins_x, vins_y, vins_z = vins_pos
        
        # 计算相对距离
        horizontal_dist = math.sqrt((x - vins_x)**2 + (y - vins_y)**2)
        vertical_dist = abs(height - vins_z)
        
        # 更新距离触发状态（平台和无人机在xy平面的距离超过2m时变为true）
        if not distance_trigger:
            if horizontal_dist > distance_threshold:
                distance_trigger = True
                rospy.loginfo("Distance trigger activated: horizontal_dist=%.2f m > %.2f m", 
                             horizontal_dist, distance_threshold)
        
        # ARUCO传感器特性
        aruco_noise_x = 0.0
        aruco_noise_y = 0.0
        aruco_error_x = 0.0
        aruco_error_y = 0.0
        aruco_visible = True
        
        # 计算实际水平距离限制（系数 × 高度差）
        aruco_max_horizontal_dist = aruco_max_horizontal_dist_coeff * vertical_dist
        
        # 检查ARUCO是否可见
        if horizontal_dist > aruco_max_horizontal_dist or vertical_dist < aruco_min_vertical_dist:
            aruco_visible = False
        else:
            # ARUCO噪声（仅在xy平面）
            if is_bumpy:
                # 颠簸区域：噪声±aruco_noise_bumpy
                aruco_noise_x = random.uniform(-aruco_noise_bumpy, aruco_noise_bumpy)
                aruco_noise_y = random.uniform(-aruco_noise_bumpy, aruco_noise_bumpy)
                # 稳态误差（sin函数）
                aruco_error_x = math.sin(x) * aruco_error_bumpy_x_amplitude
                aruco_error_y = math.sin(y) * aruco_error_bumpy_y_amplitude
            else:
                # 正常区域：噪声±aruco_noise_normal
                aruco_noise_x = random.uniform(-aruco_noise_normal, aruco_noise_normal)
                aruco_noise_y = random.uniform(-aruco_noise_normal, aruco_noise_normal)
                aruco_error_x = 0.0
                aruco_error_y = 0.0
        
        # 更新ARUCO触发状态
        # 只有当distance_trigger为true且aruco_visible为true时，aruco_triggered才会变为true
        if not aruco_triggered:
            # 状态为False时，无条件发布，直到distance_trigger为true且aruco_visible为true
            if distance_trigger and aruco_visible:
                aruco_triggered = True
                rospy.loginfo("ARUCO triggered: distance_trigger=True and aruco_visible=True, switching to conditional publish mode")
                # 重置外推状态
                extrapolating = False
                extrapolation_start_time = None
        else:
            # 状态为True时，只有ARUCO可见时才发布
            if not aruco_visible:
                # ARUCO再次不可见，开始或继续外推
                if not extrapolating:
                    # 第一次检测到不可见，记录当前位置和速度，开始外推
                    extrapolating = True
                    extrapolation_start_time = current_time
                    extrapolation_start_pos = [x, y, height]
                    extrapolation_start_vel = [vx, vy, vz]
                    extrapolation_start_yaw = yaw
                    rospy.loginfo("ARUCO lost: Starting 1s extrapolation from pos=(%.2f, %.2f, %.2f), vel=(%.2f, %.2f, %.2f)", 
                                 x, y, height, vx, vy, vz)
                else:
                    # 正在外推，检查是否超过1秒
                    extrapolation_elapsed = current_time - extrapolation_start_time
                    if extrapolation_elapsed >= 1.0:
                        # 外推时间超过1秒，停止发布
                        rospy.loginfo("Extrapolation timeout (1s): Stopping publish")
                        dog_pos_rate.sleep()
                        continue
                    # 否则继续外推（使用外推的位置和速度）
                # 如果正在外推且未超时，使用外推的位置和速度
                if extrapolating:
                    extrapolation_elapsed = current_time - extrapolation_start_time
                    if extrapolation_elapsed < 1.0:
                        # 使用外推的位置和速度
                        x = extrapolation_start_pos[0] + extrapolation_start_vel[0] * extrapolation_elapsed
                        y = extrapolation_start_pos[1] + extrapolation_start_vel[1] * extrapolation_elapsed
                        height = extrapolation_start_pos[2] + extrapolation_start_vel[2] * extrapolation_elapsed
                        vx = extrapolation_start_vel[0]
                        vy = extrapolation_start_vel[1]
                        vz = extrapolation_start_vel[2]
                        yaw = extrapolation_start_yaw
                        # 外推时不在颠簸区域
                        is_bumpy = False
                        # 外推时ARUCO噪声和误差为0（在外推时，使用外推的位置，不添加噪声和误差）
                        aruco_noise_x = 0.0
                        aruco_noise_y = 0.0
                        aruco_error_x = 0.0
                        aruco_error_y = 0.0
            else:
                # ARUCO重新可见，停止外推
                if extrapolating:
                    extrapolating = False
                    extrapolation_start_time = None
                    rospy.loginfo("ARUCO visible again: Stopping extrapolation")
        
        # 应用噪声和误差（ARUCO用于target_ekf_odom和dog_pos，z方向无噪音）
        # 如果正在外推，使用外推的位置（已经包含了外推的位置更新），不添加噪声和误差
        if extrapolating:
            # 外推时直接使用外推的位置，不添加噪声和误差
            x_aruco = x
            y_aruco = y
            z_aruco = height
        else:
            # 正常情况，添加噪声和误差
            x_aruco = x + aruco_noise_x + aruco_error_x
            y_aruco = y + aruco_noise_y + aruco_error_y
            z_aruco = height
        
        # 发布target_ekf_odom（15Hz，使用ARUCO特性）
        if current_time - last_target_time >= 1.0 / target_rate:
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "world"
            odom.child_frame_id = "target_base"
            
            # 位置（带ARUCO噪声和误差）
            odom.pose.pose.position = Point(x_aruco, y_aruco, z_aruco)
            
            # 朝向
            odom.pose.pose.orientation.w = yaw
            odom.pose.pose.orientation.x = 0
            odom.pose.pose.orientation.y = 0
            odom.pose.pose.orientation.z = 0
            
            # 速度（世界坐标系）
            odom.twist.twist.linear = Vector3(vx, vy, vz)
            odom.twist.twist.angular = Vector3(0, 0, 0)
            
            pub.publish(odom)
            last_target_time = current_time
        
        # 发布dog_pos（50Hz，复制pub的内容，即使用ARUCO特性）
        dog_pos_msg = Odometry()
        dog_pos_msg.header.stamp = rospy.Time.now()
        dog_pos_msg.header.frame_id = "world"
        dog_pos_msg.child_frame_id = "target_base"
        
        # 位置（与pub相同，使用ARUCO特性）
        dog_pos_msg.pose.pose.position = Point(x_aruco, y_aruco, z_aruco)
        
        # 朝向（与pub相同）
        dog_pos_msg.pose.pose.orientation.w = yaw
        dog_pos_msg.pose.pose.orientation.x = 0.0
        dog_pos_msg.pose.pose.orientation.y = 0.0
        dog_pos_msg.pose.pose.orientation.z = 0.0
        
        # 速度（世界坐标系，与pub相同）
        dog_pos_msg.twist.twist.linear = Vector3(vx, vy, vz)
        dog_pos_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
        
        dog_pos_pub.publish(dog_pos_msg)
        
        # 发布真实轨迹（不带噪声和误差，50Hz）
        ground_truth_msg = Odometry()
        ground_truth_msg.header.stamp = rospy.Time.now()
        ground_truth_msg.header.frame_id = "world"
        
        # 位置（真实位置，无噪声和误差）
        ground_truth_msg.pose.pose.position = Point(x, y, height)
        
        # yaw
        ground_truth_msg.pose.pose.orientation.w = yaw
        ground_truth_msg.pose.pose.orientation.x = 0.0
        ground_truth_msg.pose.pose.orientation.y = 0.0
        ground_truth_msg.pose.pose.orientation.z = 0.0
        
        # 速度（世界坐标系）
        ground_truth_msg.twist.twist.linear = Vector3(vx, vy, vz)
        ground_truth_msg.twist.twist.angular = Vector3(0.0, 0.0, 0.0)
        
        ground_truth_pub.publish(ground_truth_msg)
        dog_pos_rate.sleep()

if __name__ == '__main__':
    # 通过参数选择运动模式：'circle'、'oscillating'、'straight_line'、'sin_curve'、'triangle_sensor' 或 'triangle_aruco_trigger'
    mode = 'circle'

    try:
        if mode == 'oscillating':
            publish_oscillating_motion()
        elif mode == 'straight_line':
            publish_straight_line_motion()
        elif mode == 'sin_curve':
            publish_sin_curve_motion()
        elif mode == 'triangle_sensor':
            publish_triangle_motion_with_sensors()
        elif mode == 'triangle_aruco_trigger':
            publish_triangle_motion_with_aruco_trigger()
        else:
            publish_circle_motion()
    except rospy.ROSInterruptException:
        pass
