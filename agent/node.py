#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双层 VLM agent 入口：Planner 1Hz + Executor ~5Hz → /command_pos。
运行: python3 -m agent.node  （仓库根目录）
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Optional, Union

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

from agent.command import quat_to_yaw
from agent.camera_intrinsics import CameraIntrinsics, get_intrinsics, init_intrinsics
from agent.config import (
    CAM_CX,
    CAM_CY,
    CAM_FX,
    CAM_FY,
    CAM_HEIGHT,
    CAM_WIDTH,
    COMMAND_POS_TOPIC,
    DEPTH_IMAGE_TOPIC,
    ENABLE_COMMAND_POS_OBSTACLE_AVOIDANCE,
    EXECUTOR_2D_BBOX,
    EXECUTOR_MODE,
    EXECUTOR_PERIOD_S,
    EXECUTOR_REASONING,
    GLOBAL_PROMPT_FILE,
    MODE_MANAGER_TOPIC,
    PLANNER_PERIOD_S,
    PLANNER_REASONING,
    RGB_IMAGE_TOPIC,
    SAVE_VIDEO,
    SHOW_RGB,
    SIMULATE,
    PANEL_RATE_WINDOW_S,
    STATUS_CHECK_PERIOD_S,
    SENSOR_LOSS_TICKS,
    STATS_PRINT_PERIOD_S,
    VIDEO_SAVE_DIR,
    VIDEO_SAVE_FPS,
    VINS_TOPIC,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_HTTP_TIMEOUT_S,
    VLLM_MODEL,
)
from agent.command import reset_command_target_z
from agent.direct_control import get_direct_control_state, publish_direct_subtask
from agent.executor import Executor
from agent.executor_2d import Executor2D
from agent.executor_discrete import ExecutorDiscrete
from agent.overlay import PanelVideoWriter, make_video_save_path, render_agent_panel_bgr
from agent.obstacle_avoidance import get_obstacle_avoidance, init_obstacle_avoidance
from agent.planner import Planner
from agent.task_state import TaskState
from agent.vlm_utils import load_prompt_from_txt, pick_model_id


bridge = CvBridge()
_cam_info_logged = False
asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
vlm_client: Optional[AsyncOpenAI] = None
vlm_model: str = ""

rgb_image: Optional[np.ndarray] = None
rgb_count = 0
last_rgb_count = 0
last_rgb_timer = 0
rgb_received = False

vins_count = 0
last_vins_count = 0
last_vins_timer = 0
vins_received = False
vins_pos = np.zeros(3, dtype=np.float64)
vins_vel = np.zeros(3, dtype=np.float64)
vins_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # qx, qy, qz, qw
vins_yaw = 0.0

task_state: Optional[TaskState] = None
planner: Optional[Planner] = None
executor: Optional[Union[Executor, Executor2D, ExecutorDiscrete]] = None
cmd_pub: Optional[rospy.Publisher] = None
mode_pub: Optional[rospy.Publisher] = None
panel_video_writer: Optional[PanelVideoWriter] = None
_video_last_write_t = 0.0

# 面板：累计平均 RTT + send Hz（滑动窗口）
_panel_plan_rtt_sum_s = 0.0
_panel_plan_rtt_n = 0
_panel_exec_rtt_sum_s = 0.0
_panel_exec_rtt_n = 0
_panel_rate_t0 = 0.0
_panel_rate_plan_base = 0
_panel_rate_exec_base = 0
_panel_plan_hz = 0.0
_panel_exec_hz = 0.0

# 终端统计：每 STATS_PRINT_PERIOD_S 打印后清零
_stats_plan_ok_n = 0
_stats_plan_rtt_sum_s = 0.0
_stats_plan_send_base = 0
_stats_exec_ok_n = 0
_stats_exec_rtt_sum_s = 0.0
_stats_exec_send_base = 0

# VLM 在途请求（每路最多 1 个，避免 Timer 堆叠导致突发）
_planner_in_flight = False
_executor_in_flight = False
_planner_resend_pending = False
_executor_resend_pending = False


def _fmt_avg_rtt_ms(sum_s: float, n: int) -> str:
    if n <= 0:
        return "n/a"
    return f"{sum_s / n * 1000.0:.0f}ms"


def _fmt_rps(rps: float) -> str:
    if rps <= 0.0:
        return "n/a"
    return f"{rps:.2f}"


def _record_vlm_ok(role: str, latency_s: float) -> None:
    global _panel_plan_rtt_sum_s, _panel_plan_rtt_n
    global _panel_exec_rtt_sum_s, _panel_exec_rtt_n
    global _stats_plan_ok_n, _stats_plan_rtt_sum_s
    global _stats_exec_ok_n, _stats_exec_rtt_sum_s
    if not (math.isfinite(latency_s) and latency_s >= 0.0):
        return
    if role == "planner":
        _stats_plan_ok_n += 1
        _stats_plan_rtt_sum_s += latency_s
        _panel_plan_rtt_sum_s += latency_s
        _panel_plan_rtt_n += 1
    else:
        _stats_exec_ok_n += 1
        _stats_exec_rtt_sum_s += latency_s
        _panel_exec_rtt_sum_s += latency_s
        _panel_exec_rtt_n += 1


def _executor_send_count() -> int:
    n = executor.send_count if executor is not None else 0
    direct = get_direct_control_state()
    return n + direct.send_count


def _update_panel_send_hz() -> None:
    global _panel_rate_t0, _panel_rate_plan_base, _panel_rate_exec_base
    global _panel_plan_hz, _panel_exec_hz
    if planner is None:
        return
    now = time.perf_counter()
    exec_count = _executor_send_count()
    if _panel_rate_t0 <= 0.0:
        _panel_rate_t0 = now
        _panel_rate_plan_base = planner.request_seq
        _panel_rate_exec_base = exec_count
        return
    dt = now - _panel_rate_t0
    if dt < PANEL_RATE_WINDOW_S:
        return
    _panel_plan_hz = (planner.request_seq - _panel_rate_plan_base) / dt
    _panel_exec_hz = (exec_count - _panel_rate_exec_base) / dt
    _panel_rate_t0 = now
    _panel_rate_plan_base = planner.request_seq
    _panel_rate_exec_base = exec_count


def vins_snapshot() -> dict[str, float]:
    return {
        "x": float(vins_pos[0]),
        "y": float(vins_pos[1]),
        "z": float(vins_pos[2]),
        "vx": float(vins_vel[0]),
        "vy": float(vins_vel[1]),
        "vz": float(vins_vel[2]),
        "qx": float(vins_quat[0]),
        "qy": float(vins_quat[1]),
        "qz": float(vins_quat[2]),
        "qw": float(vins_quat[3]),
        "yaw_rad": float(vins_yaw),
    }


def publish_mode_manager_w0() -> None:
    if mode_pub is None:
        return
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.orientation.w = 0.0
    mode_pub.publish(msg)
    rospy.loginfo("[agent] 已发布 mode_manager.w=0（finish）")


def _run_async_loop() -> None:
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()


def _on_future_done(role: str, future: ConcurrentFuture) -> None:
    global _planner_in_flight, _executor_in_flight
    global _planner_resend_pending, _executor_resend_pending
    try:
        result = future.result()
    except Exception as e:
        result = {"seq": -1, "ok": False, "error": repr(e), "role": role}
    try:
        if role == "planner":
            _planner_response_cb(result)
        else:
            _executor_response_cb(result)
    finally:
        if role == "planner":
            _planner_in_flight = False
            if _planner_resend_pending:
                _planner_resend_pending = False
                _dispatch_planner_request()
        else:
            _executor_in_flight = False
            if _executor_resend_pending:
                _executor_resend_pending = False
                _dispatch_executor_request()


def _render_current_panel_bgr() -> Optional[np.ndarray]:
    if rgb_image is None:
        return None
    bgr_panel = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _update_panel_send_hz()
    return render_agent_panel_bgr(
        bgr_panel,
        task_state=task_state,
        executor=executor,
    )


def _refresh_agent_panel(*, imshow: bool = True) -> None:
    if not (SHOW_RGB or SAVE_VIDEO):
        return
    panel = _render_current_panel_bgr()
    if panel is None:
        return
    if imshow and SHOW_RGB:
        cv2.imshow("agent_rgb", panel)
        cv2.waitKey(1)
    _maybe_write_panel_video(panel)


def _maybe_write_panel_video(panel: np.ndarray) -> None:
    """按 VIDEO_SAVE_FPS 节流写入，与实时相机流对齐。"""
    global _video_last_write_t
    if not SAVE_VIDEO or panel_video_writer is None:
        return
    now = time.perf_counter()
    period = 1.0 / max(float(VIDEO_SAVE_FPS), 0.1)
    if now - _video_last_write_t < period:
        return
    _video_last_write_t = now
    panel_video_writer.write(panel)


def rgb_cb(msg: Image) -> None:
    global rgb_image, rgb_count, rgb_received, last_rgb_count, last_rgb_timer
    try:
        bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_image = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
        rgb_count += 1
        rgb_received = True
        last_rgb_count = rgb_count
        last_rgb_timer = 0
        if SHOW_RGB or SAVE_VIDEO:
            _refresh_agent_panel(imshow=SHOW_RGB)
    except CvBridgeError as e:
        rospy.logwarn_throttle(5.0, "RGB 转换失败: %s", e)


def depth_cb(msg: Image) -> None:
    oa = get_obstacle_avoidance()
    if oa is None:
        return
    oa.on_depth_msg(msg, bridge)


def vins_cb(msg: Odometry) -> None:
    global vins_count, vins_pos, vins_vel, vins_quat, vins_yaw
    q = msg.pose.pose.orientation
    vins_count += 1
    vins_pos[0] = msg.pose.pose.position.x
    vins_pos[1] = msg.pose.pose.position.y
    vins_pos[2] = msg.pose.pose.position.z
    vins_vel[0] = msg.twist.twist.linear.x
    vins_vel[1] = msg.twist.twist.linear.y
    vins_vel[2] = msg.twist.twist.linear.z
    vins_quat[0] = q.x
    vins_quat[1] = q.y
    vins_quat[2] = q.z
    vins_quat[3] = q.w
    vins_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
    oa = get_obstacle_avoidance()
    if oa is not None:
        oa.update_vins(vins_snapshot())


def status_check_cb(_event) -> None:
    global vins_received, last_vins_count, last_vins_timer
    global rgb_received, last_rgb_count, last_rgb_timer

    if vins_count != last_vins_count:
        vins_received = True
        last_vins_count = vins_count
        last_vins_timer = 0
    else:
        last_vins_timer += 1
        if last_vins_timer >= SENSOR_LOSS_TICKS:
            vins_received = False

    if rgb_count != last_rgb_count:
        rgb_received = True
        last_rgb_count = rgb_count
        last_rgb_timer = 0
    else:
        last_rgb_timer += 1
        if last_rgb_timer >= SENSOR_LOSS_TICKS:
            rgb_received = False


def _clear_terminal() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)


def stats_print_cb(_event) -> None:
    """每 STATS_PRINT_PERIOD_S 打印一次统计与最近回复（不在 VLM 回调里 print）。"""
    global _stats_plan_ok_n, _stats_plan_rtt_sum_s, _stats_plan_send_base
    global _stats_exec_ok_n, _stats_exec_rtt_sum_s, _stats_exec_send_base

    dt = STATS_PRINT_PERIOD_S
    if planner is not None:
        plan_send_hz = (planner.request_seq - _stats_plan_send_base) / dt
        exec_send_hz = (_executor_send_count() - _stats_exec_send_base) / dt
    else:
        plan_send_hz = 0.0
        exec_send_hz = 0.0
    plan_rps = _stats_plan_ok_n / dt
    exec_rps = _stats_exec_ok_n / dt

    _clear_terminal()
    print(
        f"========== [{dt:.0f}s] "
        f"plan rps={_fmt_rps(plan_rps)} send={plan_send_hz:.2f}Hz "
        f"rtt={_fmt_avg_rtt_ms(_stats_plan_rtt_sum_s, _stats_plan_ok_n)} (n={_stats_plan_ok_n}) | "
        f"exec rps={_fmt_rps(exec_rps)} send={exec_send_hz:.2f}Hz "
        f"rtt={_fmt_avg_rtt_ms(_stats_exec_rtt_sum_s, _stats_exec_ok_n)} (n={_stats_exec_ok_n}) ==========",
        flush=True,
    )

    if task_state:
        kind = task_state.current_subtask_kind()
        subtask = task_state.current_subtask_text()
        tag = f"[{kind}] " if kind else ""
        print(f"[subtask] {tag}{subtask or '(none)'}", flush=True)
    else:
        print("[subtask] (none)", flush=True)
    cmd = executor.last_command_display if executor else ""
    if not cmd:
        cmd = get_direct_control_state().last_command_display
    print(f"[command] {cmd or '(none)'}", flush=True)
    print("=" * 60, flush=True)

    if planner is not None:
        _stats_plan_send_base = planner.request_seq
    if executor is not None or task_state is not None:
        _stats_exec_send_base = _executor_send_count()
    _stats_plan_ok_n = 0
    _stats_plan_rtt_sum_s = 0.0
    _stats_exec_ok_n = 0
    _stats_exec_rtt_sum_s = 0.0


def _dispatch_planner_request() -> None:
    global _planner_in_flight
    if _planner_in_flight:
        return
    if not (vins_received and rgb_received) or rgb_image is None:
        return
    if asyncio_loop is None or planner is None or task_state is None:
        return
    if task_state.is_finished():
        return

    seq, messages, user_prompt, force_switch = planner.build_request(rgb_image)
    _planner_in_flight = True
    future = asyncio.run_coroutine_threadsafe(
        planner.run_request(seq, messages, user_prompt, force_switch),
        asyncio_loop,
    )
    future.add_done_callback(lambda f: _on_future_done("planner", f))


def _dispatch_direct_executor() -> bool:
    """rotate_scan / stop：不经 VLM，直接发指令。"""
    if task_state is None or task_state.is_finished():
        return False
    ctx = task_state.get_executor_context()
    if ctx is None:
        return False
    _, kind, _, _ = ctx
    if kind not in ("rotate_scan", "stop"):
        return False
    if cmd_pub is None:
        return False
    return publish_direct_subtask(
        kind,
        cmd_pub=cmd_pub,
        vins_snapshot_fn=vins_snapshot,
    )


def _dispatch_executor_request() -> None:
    global _executor_in_flight
    if _executor_in_flight:
        return
    if not (vins_received and rgb_received) or rgb_image is None:
        return
    if asyncio_loop is None or executor is None:
        return

    if _dispatch_direct_executor():
        return

    built = executor.build_request(rgb_image)
    if built is None:
        return
    seq, messages, user_prompt, vins_at_request, list_version = built
    _executor_in_flight = True
    future = asyncio.run_coroutine_threadsafe(
        executor.run_request(seq, messages, user_prompt, vins_at_request, list_version),
        asyncio_loop,
    )
    future.add_done_callback(lambda f: _on_future_done("executor", f))


def planner_send_cb(_event) -> None:
    global _planner_resend_pending
    if _planner_in_flight:
        _planner_resend_pending = True
        return
    _dispatch_planner_request()


def executor_send_cb(_event) -> None:
    global _executor_resend_pending
    if _executor_in_flight:
        _executor_resend_pending = True
        return
    _dispatch_executor_request()


def _planner_response_cb(result: dict[str, Any]) -> None:
    if planner is None or rgb_image is None:
        return
    planner.handle_response(result, rgb_image)
    if not result.get("ok"):
        return
    _record_vlm_ok("planner", float(result.get("latency_s", float("nan"))))


def _executor_response_cb(result: dict[str, Any]) -> None:
    if executor is None:
        return
    executor.handle_response(result)
    if not result.get("ok"):
        return
    _record_vlm_ok("executor", float(result.get("latency_s", float("nan"))))


def _close_rgb_preview() -> None:
    global panel_video_writer
    if SHOW_RGB:
        cv2.destroyAllWindows()
    if panel_video_writer is not None:
        path = panel_video_writer.path
        panel_video_writer.release()
        panel_video_writer = None
        rospy.loginfo("[agent] 视频已保存: %s", path)


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """与 example_executor&planner.py 相同的 reasoning 开关（兼容 Python 3.8）。"""
    parser = argparse.ArgumentParser(description="Hierarchical VLM drone agent")
    g_plan = parser.add_mutually_exclusive_group()
    g_plan.add_argument(
        "--planner-reasoning",
        dest="planner_reasoning",
        action="store_true",
        help="planner JSON includes reasoning field",
    )
    g_plan.add_argument(
        "--no-planner-reasoning",
        dest="planner_reasoning",
        action="store_false",
        help="planner JSON without reasoning",
    )
    g_exec = parser.add_mutually_exclusive_group()
    g_exec.add_argument(
        "--executor-reasoning",
        dest="executor_reasoning",
        action="store_true",
        help="executor JSON includes reasoning field",
    )
    g_exec.add_argument(
        "--no-executor-reasoning",
        dest="executor_reasoning",
        action="store_false",
        help="executor JSON without reasoning",
    )
    g_bbox = parser.add_mutually_exclusive_group()
    g_bbox.add_argument(
        "--executor-2d-bbox",
        dest="executor_2d_bbox",
        action="store_true",
        help="2D executor JSON includes bbox (visualization only)",
    )
    g_bbox.add_argument(
        "--no-executor-2d-bbox",
        dest="executor_2d_bbox",
        action="store_false",
        help="2D executor JSON without bbox",
    )
    parser.set_defaults(
        planner_reasoning=PLANNER_REASONING,
        executor_reasoning=EXECUTOR_REASONING,
        executor_2d_bbox=EXECUTOR_2D_BBOX,
    )
    parser.add_argument(
        "--executor-mode",
        choices=("3d", "2d", "discrete"),
        default=EXECUTOR_MODE,
        help="executor output: 3d=compact body delta; 2d=pixel waypoint; discrete=FRONT/BACK/...",
    )
    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def main(
    *,
    planner_reasoning: bool = PLANNER_REASONING,
    executor_reasoning: bool = EXECUTOR_REASONING,
    executor_2d_bbox: bool = EXECUTOR_2D_BBOX,
    executor_mode: str = EXECUTOR_MODE,
) -> None:
    executor_mode = str(executor_mode).lower()
    if executor_mode not in ("3d", "2d", "discrete"):
        raise ValueError(f"invalid executor_mode: {executor_mode!r}")
    global vlm_client, vlm_model, asyncio_loop
    global task_state, planner, executor, cmd_pub, mode_pub, panel_video_writer
    global _stats_plan_send_base, _stats_exec_send_base
    reset_command_target_z(None)
    rospy.init_node("hierarchical_vlm_agent", anonymous=False)
    init_intrinsics(
        CameraIntrinsics(
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            fx=CAM_FX,
            fy=CAM_FY,
            cx=CAM_CX,
            cy=CAM_CY,
        )
    )
    intr = get_intrinsics()
    rospy.loginfo(
        "[agent] 2D intrinsics from config only: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f (SIMULATE=%s)",
        intr.width,
        intr.height,
        intr.fx,
        intr.fy,
        intr.cx,
        intr.cy,
        SIMULATE,
    )
    if SHOW_RGB or SAVE_VIDEO:
        rospy.on_shutdown(_close_rgb_preview)

    if SAVE_VIDEO:
        video_path = make_video_save_path(VIDEO_SAVE_DIR)
        panel_video_writer = PanelVideoWriter(video_path, VIDEO_SAVE_FPS)
        rospy.loginfo("[agent] 开始录制: %s (%.1f fps, 随 RGB 流写入)", video_path, VIDEO_SAVE_FPS)

    global_task = load_prompt_from_txt(GLOBAL_PROMPT_FILE)
    task_state = TaskState(global_task)

    vlm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL.rstrip("/"),
        api_key=VLLM_API_KEY,
        timeout=VLLM_HTTP_TIMEOUT_S,
    )

    threading.Thread(target=_run_async_loop, daemon=True).start()
    while asyncio_loop is None and not rospy.is_shutdown():
        time.sleep(0.01)
    if asyncio_loop is None:
        raise RuntimeError("asyncio 事件循环未启动")

    if VLLM_MODEL:
        vlm_model = VLLM_MODEL
    else:
        model_fut = asyncio.run_coroutine_threadsafe(pick_model_id(vlm_client), asyncio_loop)
        vlm_model = model_fut.result(timeout=60.0)

    init_obstacle_avoidance()

    cmd_pub = rospy.Publisher(COMMAND_POS_TOPIC, Odometry, queue_size=10)
    mode_pub = rospy.Publisher(MODE_MANAGER_TOPIC, PoseStamped, queue_size=1, latch=True)

    planner = Planner(
        client=vlm_client,
        model=vlm_model,
        task_state=task_state,
        on_finish=publish_mode_manager_w0,
        with_reasoning=planner_reasoning,
    )
    if executor_mode == "2d":
        executor = Executor2D(
            client=vlm_client,
            model=vlm_model,
            task_state=task_state,
            cmd_pub=cmd_pub,
            vins_snapshot_fn=vins_snapshot,
            on_command_published=None,
            with_bbox=executor_2d_bbox,
        )
    elif executor_mode == "discrete":
        executor = ExecutorDiscrete(
            client=vlm_client,
            model=vlm_model,
            task_state=task_state,
            cmd_pub=cmd_pub,
            vins_snapshot_fn=vins_snapshot,
            on_command_published=None,
            with_reasoning=executor_reasoning,
        )
    else:
        executor = Executor(
            client=vlm_client,
            model=vlm_model,
            task_state=task_state,
            cmd_pub=cmd_pub,
            vins_snapshot_fn=vins_snapshot,
            on_command_published=None,
            with_reasoning=executor_reasoning,
        )

    _stats_plan_send_base = planner.request_seq
    _stats_exec_send_base = _executor_send_count()

    rospy.Subscriber(RGB_IMAGE_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Subscriber(VINS_TOPIC, Odometry, vins_cb, queue_size=10)
    if ENABLE_COMMAND_POS_OBSTACLE_AVOIDANCE:
        rospy.Subscriber(DEPTH_IMAGE_TOPIC, Image, depth_cb, queue_size=1)

    rospy.Timer(rospy.Duration(STATUS_CHECK_PERIOD_S), status_check_cb)
    rospy.Timer(rospy.Duration(PLANNER_PERIOD_S), planner_send_cb)
    rospy.Timer(rospy.Duration(EXECUTOR_PERIOD_S), executor_send_cb)
    rospy.Timer(rospy.Duration(STATS_PRINT_PERIOD_S), stats_print_cb)

    print(
        f"hierarchical_vlm_agent: simulate={SIMULATE} show_rgb={SHOW_RGB} save_video={SAVE_VIDEO} "
        f"RGB={RGB_IMAGE_TOPIC} intrinsics=config model={vlm_model} | "
        f"planner={PLANNER_PERIOD_S}s executor={EXECUTOR_PERIOD_S}s | "
        f"stats_print={STATS_PRINT_PERIOD_S}s | "
        f"planner_reasoning={planner_reasoning} executor_reasoning={executor_reasoning} "
        f"executor_2d_bbox={executor_2d_bbox} executor_mode={executor_mode}",
        flush=True,
    )
    rospy.spin()


if __name__ == "__main__":
    _cli = parse_cli_args(rospy.myargv(argv=sys.argv)[1:])
    main(
        planner_reasoning=_cli.planner_reasoning,
        executor_reasoning=_cli.executor_reasoning,
        executor_2d_bbox=_cli.executor_2d_bbox,
        executor_mode=_cli.executor_mode,
    )
