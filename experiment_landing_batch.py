#!/usr/bin/env python3
"""
降落批量实验：4 方案 × 2 场景 × N 重复

每次 trial 流程：
1. 切换 dog_pos 方案（remap 到 /dog_pos_processed）
2. 运行 fake_target → rosbag record
3. 等待 /dog_pos_processed 首帧（四方案 ready 恒 true）
4. 发布 /mode_manager w=0 → 等待 MPC 输出 /traj_v
5. 监控降落 → **立即** w=-1 + 沿直线 ~2 m/s 引导复位 (0,0,2) → 再停 rosbag / fake_target

无随机等待（已去掉 5–35 s）；轨迹与外参 Exp B 相同（wavy_circle / sin_accel_straight），
通过 ROS param 设漂移速率、无初始跳变。

前提（由用户手动启动，本脚本不启动）：
  sim_fly.sh、xtdrone.sh、MPC.py、tracker_sim.sh（traj_server_sim，勿用 tracker.sh）

输出：results.json、summary_table.md/csv、summary_bars.png
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
FAKE_TARGET_SCRIPT = os.path.join(WORKSPACE, 'fake_target.py')
SETUP_BASH = os.path.join(WORKSPACE, 'devel/setup.bash')

METHODS = (
    ('ours', 'Ours (cascade)', 'b'),
    ('kf', 'Whole KF', 'r'),
    ('lkf', 'LKF incr', 'm'),
    ('vis', 'Vis fixed R/t', 'c'),
)

SCENARIOS = {
    'wavy_circle': 'wavy_circle',
    'sin_accel_straight': 'sin_accel_straight',
}

# fake_target 节点私有命名空间（与外参实验一致）
TRAJ_MODES = {
    'wavy_circle': '/target_wavy_circle_publisher',
    'sin_accel_straight': '/target_sin_accel_straight_publisher',
}

RESET_HOME = (0.0, 0.0, 2.0)
RESET_HOME_SPEED = 2.0  # m/s，沿直线引导回 home

# 与 fake_target EXTRINSIC_DRIFT_DEFAULTS / Exp B drift 一致
DEFAULT_DRIFT = {
    'dog_pos_drift_rate_x': 0.008,
    'dog_pos_drift_rate_y': 0.008,
    'dog_pos_drift_rate_z': 0.0,
    'dog_pos_drift_yaw_rate': 0.003,
    'vins_drift_rate_x': -0.008,
    'vins_drift_rate_y': -0.008,
    'vins_drift_rate_z': 0.0,
    'vins_drift_yaw_rate': -0.003,
}

EXTRINSIC_PARAM_KEYS = (
    'dog_pos_drift_offset_x', 'dog_pos_drift_offset_y', 'dog_pos_drift_offset_z',
    'dog_pos_drift_rate_x', 'dog_pos_drift_rate_y', 'dog_pos_drift_rate_z',
    'dog_pos_drift_yaw_initial', 'dog_pos_drift_yaw_rate',
    'vins_drift_rate_x', 'vins_drift_rate_y', 'vins_drift_rate_z', 'vins_drift_yaw_rate',
)

active_processes = {'fake_target': None, 'rosbag': None, 'dog_pos_launch': None}
active_processes_lock = threading.Lock()
shutdown_flag = threading.Event()
_current_method = None


class UAVPositionListener:
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber(
            '/vins_fusion/imu_propagate', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]

    def get_position(self):
        with self.position_lock:
            return self.position.copy() if self.position is not None else None


class TargetPositionListener:
    def __init__(self):
        self.position = None
        self.position_lock = threading.Lock()
        self.pose_sub = rospy.Subscriber(
            '/ground_truth_traj', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        with self.position_lock:
            self.position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]

    def get_position(self):
        with self.position_lock:
            return self.position.copy() if self.position is not None else None


class TrajVListener:
    """监听 MPC 是否已发布速度轨迹（traj_server 靠 /traj_v 置 traj_initialized）。"""

    def __init__(self):
        self.received = False
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber('/traj_v', Path, self.callback, queue_size=1)

    def callback(self, msg):
        if msg.poses:
            with self.lock:
                self.received = True

    def reset(self):
        with self.lock:
            self.received = False

    def is_ready(self):
        with self.lock:
            return self.received


class DogPosReadyListener:
    """监听 /dog_pos_processed 是否可用于触发 MPC。"""

    def __init__(self, method_key):
        self.method_key = method_key
        self.ready = False
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber(
            '/dog_pos_processed', Odometry, self.callback, queue_size=10)

    def callback(self, msg):
        with self.lock:
            # 四方案 landing 版均恒置 ready；此处收到首帧即可
            self.ready = True

    def is_ready(self):
        with self.lock:
            return self.ready


def clear_fake_target_params(ns):
    for key in EXTRINSIC_PARAM_KEYS:
        full = f'{ns}/{key}'
        try:
            if rospy.has_param(full):
                rospy.delete_param(full)
        except Exception:
            pass


def set_drift_params(ns, drift_cfg):
    """Exp B 同款：无初始跳变，仅持续漂移。"""
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_initial', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_x', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_y', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_offset_z', 0.0)
    rospy.set_param(f'{ns}/dog_pos_drift_yaw_rate', float(drift_cfg['dog_pos_drift_yaw_rate']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_x', float(drift_cfg['dog_pos_drift_rate_x']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_y', float(drift_cfg['dog_pos_drift_rate_y']))
    rospy.set_param(f'{ns}/dog_pos_drift_rate_z', float(drift_cfg['dog_pos_drift_rate_z']))
    rospy.set_param(f'{ns}/vins_drift_rate_x', float(drift_cfg['vins_drift_rate_x']))
    rospy.set_param(f'{ns}/vins_drift_rate_y', float(drift_cfg['vins_drift_rate_y']))
    rospy.set_param(f'{ns}/vins_drift_rate_z', float(drift_cfg['vins_drift_rate_z']))
    rospy.set_param(f'{ns}/vins_drift_yaw_rate', float(drift_cfg['vins_drift_yaw_rate']))


def is_mpc_running():
    try:
        r = subprocess.run(
            ['pgrep', '-f', r'MPC\.py'],
            capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def count_mode_manager_subscribers():
    """返回 /mode_manager 订阅者数量（期望 MPC + traj_server_sim ≥ 2）。"""
    try:
        r = subprocess.run(
            ['bash', '-c',
             f'source "{SETUP_BASH}" && rostopic info /mode_manager'],
            capture_output=True, text=True, timeout=8, cwd=WORKSPACE)
        if r.returncode != 0:
            return 0
        count = 0
        in_sub = False
        for line in r.stdout.splitlines():
            if line.strip().startswith('Subscribers:'):
                in_sub = True
                continue
            if in_sub:
                if line.strip().startswith('Publishers:'):
                    break
                if line.strip().startswith('*'):
                    count += 1
        return count
    except Exception:
        return 0


def check_prerequisites():
    ok = True
    if not is_mpc_running():
        rospy.logerr('MPC.py 未运行！请先启动: python3 MPC.py')
        ok = False
    else:
        rospy.loginfo('MPC.py is running.')
    n_sub = count_mode_manager_subscribers()
    if n_sub < 2:
        rospy.logwarn(
            '/mode_manager 订阅者=%d（期望≥2: MPC + traj_server_sim）', n_sub)
        if n_sub < 1:
            ok = False
    else:
        rospy.loginfo('/mode_manager subscribers: %d', n_sub)
    return ok


def wait_for_dog_pos_ready(method_key, timeout=15.0):
    """触发前等待 /dog_pos_processed 首帧（非随机 R/t 等待）。"""
    listener = DogPosReadyListener(method_key)
    start = time.time()
    rate = rospy.Rate(20)
    while time.time() - start < timeout:
        if shutdown_flag.is_set():
            return False
        if listener.is_ready():
            rospy.loginfo('dog_pos_processed ready (method=%s)', method_key)
            return True
        rate.sleep()
    rospy.logwarn('dog_pos_processed not ready after %.1fs (method=%s)',
                  timeout, method_key)
    return False


def wait_for_traj_v(traj_listener, timeout=20.0):
    """发布 w=0 后等待 MPC 规划并发布 /traj_v。"""
    traj_listener.reset()
    start = time.time()
    rate = rospy.Rate(20)
    while time.time() - start < timeout:
        if shutdown_flag.is_set():
            return False
        if traj_listener.is_ready():
            rospy.loginfo('MPC trajectory ready (/traj_v received)')
            return True
        rate.sleep()
    rospy.logwarn('No /traj_v within %.1fs — MPC 可能未收到 w=0 或未规划', timeout)
    return False


def publish_trigger(pub, w_value):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'world'
    msg.pose.position.x = 0.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 0.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = w_value
    pub.publish(msg)
    rospy.loginfo('Published trigger w=%.1f', w_value)


def publish_position_cmd(pub, x, y, z, traj_id=1, quiet=False):
    cmd = PositionCommand()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = 'world'
    cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
    cmd.trajectory_id = traj_id
    cmd.position.x = x
    cmd.position.y = y
    cmd.position.z = z
    cmd.velocity.x = 0.0
    cmd.velocity.y = 0.0
    cmd.velocity.z = 0.0
    cmd.acceleration.x = 0.0
    cmd.acceleration.y = 0.0
    cmd.acceleration.z = 0.0
    cmd.yaw = 0.0
    cmd.yaw_dot = 0.0
    pub.publish(cmd)
    if not quiet:
        rospy.loginfo('Published position command (%.2f, %.2f, %.2f)', x, y, z)


def guide_position_cmd_linear(pub, start_pos, target_pos, speed=RESET_HOME_SPEED,
                              rate_hz=20.0, hold_sec=1.0):
    """沿直线匀速更新 position_cmd，从 start 引导到 target。"""
    sx, sy, sz = start_pos
    tx, ty, tz = target_pos
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-3:
        publish_position_cmd(pub, tx, ty, tz)
        return

    duration = dist / speed
    rospy.loginfo(
        'Guide home: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f), '
        'dist=%.2f m, speed=%.1f m/s, duration=%.2f s',
        sx, sy, sz, tx, ty, tz, dist, speed, duration)
    rate = rospy.Rate(rate_hz)
    t0 = time.time()
    while not shutdown_flag.is_set():
        elapsed = time.time() - t0
        if elapsed >= duration:
            break
        alpha = elapsed / duration
        publish_position_cmd(
            pub,
            sx + alpha * dx,
            sy + alpha * dy,
            sz + alpha * dz,
            quiet=True,
        )
        rate.sleep()

    hold_steps = max(1, int(hold_sec * rate_hz))
    for _ in range(hold_steps):
        if shutdown_flag.is_set():
            break
        publish_position_cmd(pub, tx, ty, tz, quiet=True)
        rate.sleep()
    rospy.loginfo('Guide home complete: (%.2f, %.2f, %.2f)', tx, ty, tz)


def wait_for_landing(uav_listener, target_listener, timeout=60.0):
    """等待降落（高度约 0.4 m）或停止移动超过 5 s。"""
    start_time = time.time()
    rate = rospy.Rate(50)
    last_position = None
    last_position_time = None
    position_threshold = 0.05
    stop_duration = 5.0

    while time.time() - start_time < timeout:
        if shutdown_flag.is_set():
            uav_pos = uav_listener.get_position()
            target_pos = target_listener.get_position()
            if uav_pos is not None and target_pos is not None:
                ex = uav_pos[0] - target_pos[0]
                ey = uav_pos[1] - target_pos[1]
                return (time.time() - start_time, uav_pos, target_pos,
                        ex, ey, math.sqrt(ex ** 2 + ey ** 2), False)
            return time.time() - start_time, None, None, None, None, None, False

        uav_pos = uav_listener.get_position()
        target_pos = target_listener.get_position()

        if uav_pos is not None:
            current_time = time.time()
            if last_position is not None:
                change = math.sqrt(
                    (uav_pos[0] - last_position[0]) ** 2
                    + (uav_pos[1] - last_position[1]) ** 2)
                if change < position_threshold:
                    if last_position_time is None:
                        last_position_time = current_time
                    elif (current_time - last_position_time) >= stop_duration:
                        rospy.logwarn('UAV stopped moving >%.1fs', stop_duration)
                        if target_pos is not None:
                            ex = uav_pos[0] - target_pos[0]
                            ey = uav_pos[1] - target_pos[1]
                            return (time.time() - start_time, uav_pos, target_pos,
                                    ex, ey, math.sqrt(ex ** 2 + ey ** 2), True)
                        return time.time() - start_time, uav_pos, None, None, None, None, True
                else:
                    last_position_time = None
            last_position = uav_pos.copy()

            if target_pos is not None and abs(uav_pos[2] - 0.4) < 0.05:
                landing_time = time.time() - start_time
                ex = uav_pos[0] - target_pos[0]
                ey = uav_pos[1] - target_pos[1]
                e_xy = math.sqrt(ex ** 2 + ey ** 2)
                rospy.loginfo(
                    'Landing: height=%.3f, error_xy=%.3f m, time=%.2f s',
                    uav_pos[2], e_xy, landing_time)
                return landing_time, uav_pos, target_pos, ex, ey, e_xy, False

        rate.sleep()

    rospy.logwarn('Landing timeout %.1f s', timeout)
    uav_pos = uav_listener.get_position()
    target_pos = target_listener.get_position()
    if uav_pos is not None and target_pos is not None:
        ex = uav_pos[0] - target_pos[0]
        ey = uav_pos[1] - target_pos[1]
        return timeout, uav_pos, target_pos, ex, ey, math.sqrt(ex ** 2 + ey ** 2), False
    return timeout, None, None, None, None, None, False


def _stop_dog_pos_launch():
    global _current_method
    with active_processes_lock:
        proc = active_processes.get('dog_pos_launch')
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception as e:
                rospy.logwarn('Error stopping dog_pos launch: %s', e)
            active_processes['dog_pos_launch'] = None
    _current_method = None


def ensure_dog_pos_method(method_key):
    """切换 dog_pos 方案（输出 remap 到 /dog_pos_processed）。"""
    global _current_method
    if _current_method == method_key:
        return
    _stop_dog_pos_launch()
    cmd = (
        f'source "{SETUP_BASH}" && '
        f'roslaunch planning dog_pos_processor_landing.launch method:={method_key}'
    )
    with active_processes_lock:
        active_processes['dog_pos_launch'] = subprocess.Popen(
            ['bash', '-c', cmd],
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    _current_method = method_key
    time.sleep(2.5)
    rospy.loginfo('dog_pos method switched to: %s', method_key)


def cleanup_processes():
    with active_processes_lock:
        if active_processes['rosbag'] is not None:
            try:
                active_processes['rosbag'].send_signal(signal.SIGINT)
                active_processes['rosbag'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['rosbag'].kill()
                active_processes['rosbag'].wait()
            except Exception as e:
                rospy.logwarn('Error stopping rosbag: %s', e)
            active_processes['rosbag'] = None
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['fake_target'].kill()
                active_processes['fake_target'].wait()
            except Exception as e:
                rospy.logwarn('Error stopping fake_target: %s', e)
            active_processes['fake_target'] = None
    _stop_dog_pos_launch()
    rospy.loginfo('Processes cleaned up.')


def signal_handler(signum, frame):
    rospy.logwarn('Ctrl+C received. Shutting down safely...')
    shutdown_flag.set()
    cleanup_processes()


def run_single_experiment(trial_num, total_num, method_key, scenario_key,
                          uav_listener, target_listener, trigger_pub, pos_cmd_pub,
                          traj_listener, base_dir, record_rosbag=True):
    """单次降落 trial：fake_target 每次重启；无随机等待，保留 dog_pos/MPC 就绪等待。"""
    exp_dir = os.path.join(
        base_dir, scenario_key, method_key, f'repeat_{trial_num:03d}')
    os.makedirs(exp_dir, exist_ok=True)

    rospy.loginfo('=' * 60)
    rospy.loginfo(
        'Trial %d/%d  scenario=%s  method=%s  repeat=%d',
        trial_num, total_num, scenario_key, method_key, trial_num)
    rospy.loginfo('=' * 60)

    if shutdown_flag.is_set():
        return None

    ensure_dog_pos_method(method_key)

    # 每次 trial 重启 fake_target
    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except Exception:
                pass
            active_processes['fake_target'] = None

    # 1. 外参漂移参数（Exp B：无跳变，仅 rate）+ fake_target
    mode = SCENARIOS[scenario_key]
    ns = TRAJ_MODES[scenario_key]
    clear_fake_target_params(ns)
    set_drift_params(ns, DEFAULT_DRIFT)
    rospy.set_param(f'{ns}/publish_vins_zero', False)  # 飞机位姿由 sim 栈提供，fake_target 不覆盖 VINS
    time.sleep(0.2)

    rospy.loginfo('Step 1: Starting fake_target (mode=%s, drift only)...', mode)
    seed_value = (trial_num * 1000000 + int(time.time() * 1000)) % (2 ** 32)
    env = os.environ.copy()
    env['FAKE_TARGET_RANDOM_SEED'] = str(seed_value)
    env['FAKE_TARGET_MODE'] = mode
    with active_processes_lock:
        active_processes['fake_target'] = subprocess.Popen(
            [sys.executable, FAKE_TARGET_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=WORKSPACE,
        )
    time.sleep(3)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 2. rosbag
    if record_rosbag:
        rospy.loginfo('Step 2: Starting rosbag record...')
        bag_path = os.path.join(exp_dir, f'trial_{trial_num:03d}.bag')
        topics = [
            '/dog_pos', '/mode_manager', '/vins_fusion/imu_propagate', '/position_cmd',
            '/target_ekf_odom', '/dog_pos_processed', '/ground_truth_traj',
        ]
        with active_processes_lock:
            active_processes['rosbag'] = subprocess.Popen(
                ['rosbag', 'record', '-O', bag_path] + topics,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=WORKSPACE,
            )
        time.sleep(1)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 3. 等待 dog_pos 首帧（非随机等待）
    rospy.loginfo('Step 3: Wait dog_pos_processed...')
    wait_for_dog_pos_ready(method_key, timeout=15.0)
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 4. 返航触发 + 等待 MPC 轨迹
    rospy.loginfo('Step 4: Publish return trigger w=0...')
    for _ in range(3):
        publish_trigger(trigger_pub, 0.0)
        time.sleep(0.2)
    if not wait_for_traj_v(traj_listener, timeout=20.0):
        rospy.logwarn('Proceeding without /traj_v (MPC may not have planned yet)')
    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 5. 监控降落
    rospy.loginfo('Step 5: Monitoring landing...')
    landing_time, uav_pos, target_pos, error_x, error_y, error_xy, stopped = (
        wait_for_landing(uav_listener, target_listener))
    success = not stopped and error_xy is not None and error_xy < 0.1

    if shutdown_flag.is_set():
        cleanup_processes()
        return None

    # 6. 降落结束沿直线引导复位到 (0,0,2)，不等待 rosbag/fake_target 清理
    rospy.loginfo('Step 6: Guide reset to (0, 0, 2) at %.1f m/s...', RESET_HOME_SPEED)
    publish_trigger(trigger_pub, -1.0)
    time.sleep(0.2)
    start_pos = uav_pos if uav_pos is not None else uav_listener.get_position()
    if start_pos is None:
        rospy.logwarn('No UAV position for guided reset; using home as start')
        start_pos = list(RESET_HOME)
    guide_position_cmd_linear(pos_cmd_pub, start_pos, RESET_HOME, speed=RESET_HOME_SPEED)

    # 7. 停止 rosbag
    if record_rosbag:
        rospy.loginfo('Step 7: Stopping rosbag...')
        with active_processes_lock:
            if active_processes['rosbag'] is not None:
                try:
                    active_processes['rosbag'].send_signal(signal.SIGINT)
                    active_processes['rosbag'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    active_processes['rosbag'].kill()
                    active_processes['rosbag'].wait()
                active_processes['rosbag'] = None
        time.sleep(1)

    # 8. 停止 fake_target
    rospy.loginfo('Step 8: Stopping fake_target...')
    with active_processes_lock:
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                active_processes['fake_target'].kill()
                active_processes['fake_target'].wait()
            active_processes['fake_target'] = None

    return {
        'trial_num': trial_num,
        'scenario': scenario_key,
        'method': method_key,
        'waiting_time': 0.0,
        'landing_time': landing_time,
        'success': success,
        'error_x': error_x,
        'error_y': error_y,
        'error_xy': error_xy,
        'uav_pos': uav_pos,
        'target_pos': target_pos,
        'stopped': stopped,
        'exp_dir': exp_dir,
    }


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stat_from_values(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'n': len(vals),
    }


def aggregate_results(results):
    agg = {}
    for scenario_key in SCENARIOS:
        agg[scenario_key] = {}
        for method_key, label, _ in METHODS:
            runs = [r for r in results
                    if r.get('scenario') == scenario_key and r.get('method') == method_key]
            errs = [r.get('error_xy') for r in runs]
            agg[scenario_key][method_key] = {
                'label': label,
                'n_total': len(runs),
                'success_count': sum(1 for r in runs if r.get('success')),
                'error_xy': _stat_from_values(errs),
            }
    return agg


def _fmt_cell(stat, fmt='.3f'):
    m, s = stat.get('mean'), stat.get('std')
    if m is None or not math.isfinite(m):
        return '—'
    if s is not None and math.isfinite(s) and s > 0:
        return f'{m:{fmt}}±{s:{fmt}}'
    return f'{m:{fmt}}'


def save_summary_table(all_agg, base_dir, repeats):
    scenario_keys = list(SCENARIOS.keys())
    method_labels = {k: l for k, l, _ in METHODS}
    lines = [f'# Landing experiment summary (n={repeats} repeats per cell)', '']

    for title, metric in (
        ('Landing error — mean (m)', 'mean'),
        ('Landing error — max (m)', 'max'),
        ('Landing error — min (m)', 'min'),
    ):
        lines.extend([
            '', f'## {title}',
            '| Method | ' + ' | '.join(scenario_keys) + ' |',
            '|--------|' + '|'.join(['-------------'] * len(scenario_keys)) + '|',
        ])
        for method_key, label, _ in METHODS:
            cells = []
            for scenario_key in scenario_keys:
                st = all_agg[scenario_key][method_key]['error_xy']
                v = st.get(metric)
                if v is None or not math.isfinite(v):
                    cells.append('—')
                else:
                    if metric == 'mean':
                        cells.append(_fmt_cell(st))
                    else:
                        cells.append(f'{v:.3f}')
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    md_path = os.path.join(base_dir, 'summary_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    csv_path = os.path.join(base_dir, 'summary_table.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('scenario,method,method_label,mean,std,min,max,n,success_count\n')
        for scenario_key in scenario_keys:
            for method_key, label, _ in METHODS:
                cell = all_agg[scenario_key][method_key]
                st = cell['error_xy']
                def _v(k):
                    v = st.get(k)
                    return '' if v is None else v
                f.write(
                    f'{scenario_key},{method_key},{label},'
                    f'{_v("mean")},{_v("std")},{_v("min")},{_v("max")},'
                    f'{st.get("n", 0)},{cell.get("success_count", 0)}\n')
    rospy.loginfo('Saved %s', md_path)
    rospy.loginfo('Saved %s', csv_path)


def plot_summary_bars(all_agg, base_dir, repeats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scenario_keys = list(SCENARIOS.keys())
    method_keys = [m[0] for m in METHODS]
    method_labels = [m[1] for m in METHODS]
    colors = [m[2] for m in METHODS]

    panels = [
        ('mean', 'Mean landing error (m)'),
        ('max', 'Max landing error (m)'),
        ('min', 'Min landing error (m)'),
    ]

    fig, axes = plt.subplots(
        len(panels), len(scenario_keys),
        figsize=(5.5 * len(scenario_keys), 3.5 * len(panels)))
    if len(panels) == 1:
        axes = np.array([axes])
    if len(scenario_keys) == 1:
        axes = axes.reshape(len(panels), 1)

    x = np.arange(len(method_keys))
    width = 0.62

    for row, (metric, ylabel) in enumerate(panels):
        for col, scenario_key in enumerate(scenario_keys):
            ax = axes[row, col]
            heights, yerr, valid = [], [], []
            for method_key in method_keys:
                st = all_agg[scenario_key][method_key]['error_xy']
                v = st.get(metric)
                ok = v is not None and math.isfinite(v)
                valid.append(ok)
                heights.append(v if ok else 0.0)
                if metric == 'mean':
                    s = st.get('std') or 0.0
                    yerr.append(s if ok else 0.0)
                else:
                    yerr.append(0.0)
            bar_colors = [colors[i] if valid[i] else '#cccccc' for i in range(len(method_keys))]
            ax.bar(x, heights, width,
                   yerr=yerr if metric == 'mean' else None,
                   capsize=3, color=bar_colors, alpha=0.9, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=22, ha='right', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(scenario_key, fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(bottom=0.0)

    fig.suptitle(f'Landing error (n={repeats} repeats)', fontsize=12)
    fig.tight_layout()
    path = os.path.join(base_dir, 'summary_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    rospy.loginfo('Saved %s', path)
    return path


def save_results(results, base_dir, repeats):
    experiments = []
    for r in results:
        uav = r.get('uav_pos')
        tgt = r.get('target_pos')
        experiments.append({
            'trial_num': r.get('trial_num'),
            'scenario': r.get('scenario'),
            'method': r.get('method'),
            'waiting_time': _to_float(r.get('waiting_time')),
            'landing_time': _to_float(r.get('landing_time')),
            'success': bool(r.get('success', False)),
            'error_x': _to_float(r.get('error_x')),
            'error_y': _to_float(r.get('error_y')),
            'error_xy': _to_float(r.get('error_xy')),
            'stopped': bool(r.get('stopped', False)),
            'uav_position': {
                'x': _to_float(uav[0]) if uav and len(uav) > 0 else None,
                'y': _to_float(uav[1]) if uav and len(uav) > 1 else None,
                'z': _to_float(uav[2]) if uav and len(uav) > 2 else None,
            } if uav else None,
            'target_position': {
                'x': _to_float(tgt[0]) if tgt and len(tgt) > 0 else None,
                'y': _to_float(tgt[1]) if tgt and len(tgt) > 1 else None,
                'z': _to_float(tgt[2]) if tgt and len(tgt) > 2 else None,
            } if tgt else None,
        })

    all_agg = aggregate_results(results)
    data = {
        'total_trials': len(results),
        'repeats': repeats,
        'timestamp': datetime.now().isoformat(),
        'methods': [{'key': k, 'label': l} for k, l, _ in METHODS],
        'scenarios': list(SCENARIOS.keys()),
        'experiments': experiments,
        'aggregated': all_agg,
    }
    path = os.path.join(base_dir, 'results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    rospy.loginfo('Saved %s', path)
    return all_agg


def main():
    parser = argparse.ArgumentParser(
        description='降落批量：4 方案 × 2 场景 × N 重复（前提：sim_fly + xtdrone + MPC.py + tracker_sim）')
    parser.add_argument('--repeats', type=int, default=5, help='每格重复次数')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--no-rosbag', action='store_true')
    parser.add_argument('--methods', nargs='+', default=[m[0] for m in METHODS])
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS.keys()))
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('experiment_landing_batch', anonymous=True)

    base_dir = args.output_dir
    if not base_dir:
        base_dir = os.path.join(
            WORKSPACE, f'experiment_landing_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(base_dir, exist_ok=True)
    rospy.loginfo('Output directory: %s', base_dir)
    rospy.loginfo(
        'Prerequisite: sim_fly.sh + xtdrone.sh + MPC.py + tracker_sim.sh')

    if not check_prerequisites():
        rospy.logerr('请先启动完整栈后再跑实验。')
        sys.exit(1)

    trigger_pub = rospy.Publisher('/mode_manager', PoseStamped, queue_size=10)
    pos_cmd_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=10)
    uav_listener = UAVPositionListener()
    target_listener = TargetPositionListener()
    traj_listener = TrajVListener()
    time.sleep(2)

    results = []
    total = len(args.scenarios) * len(args.methods) * args.repeats
    trial_counter = 0

    for scenario_key in args.scenarios:
        if scenario_key not in SCENARIOS:
            rospy.logwarn('Unknown scenario %s, skip', scenario_key)
            continue
        for method_key in args.methods:
            if method_key not in {m[0] for m in METHODS}:
                rospy.logwarn('Unknown method %s, skip', method_key)
                continue
            for repeat_idx in range(1, args.repeats + 1):
                if shutdown_flag.is_set():
                    break
                trial_counter += 1
                try:
                    r = run_single_experiment(
                        repeat_idx, args.repeats,
                        method_key, scenario_key,
                        uav_listener, target_listener,
                        trigger_pub, pos_cmd_pub,
                        traj_listener,
                        base_dir,
                        record_rosbag=not args.no_rosbag,
                    )
                    if r is None:
                        break
                    results.append(r)
                    save_results(results, base_dir, args.repeats)
                except KeyboardInterrupt:
                    rospy.logwarn('KeyboardInterrupt. Saving and exiting.')
                    shutdown_flag.set()
                    break
                except Exception as e:
                    rospy.logerr('Trial %d failed: %s', trial_counter, e)
                    results.append({
                        'trial_num': repeat_idx,
                        'scenario': scenario_key,
                        'method': method_key,
                        'waiting_time': None,
                        'landing_time': None,
                        'success': False,
                        'error_x': None,
                        'error_y': None,
                        'error_xy': None,
                        'uav_pos': None,
                        'target_pos': None,
                        'stopped': True,
                    })

    # 结束时只停 fake_target / rosbag，保留当前 dog_pos 节点
    with active_processes_lock:
        if active_processes['rosbag'] is not None:
            try:
                active_processes['rosbag'].send_signal(signal.SIGINT)
                active_processes['rosbag'].wait(timeout=5)
            except Exception:
                pass
            active_processes['rosbag'] = None
        if active_processes['fake_target'] is not None:
            try:
                active_processes['fake_target'].terminate()
                active_processes['fake_target'].wait(timeout=5)
            except Exception:
                pass
            active_processes['fake_target'] = None

    if results:
        all_agg = save_results(results, base_dir, args.repeats)
        save_summary_table(all_agg, base_dir, args.repeats)
        plot_summary_bars(all_agg, base_dir, args.repeats)
    rospy.loginfo('Done. Total trials: %d. Results in %s', len(results), base_dir)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
