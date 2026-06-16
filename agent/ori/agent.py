#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM agent：txt prompt + RGB -> AsyncOpenAI -> /command_pos。
vlm_send_cb：按 VLM_SEND_PERIOD_S 固定周期发 VLM 请求（人工设定）。
vlm_response_cb：已收到更大 seq 后，更小 seq 晚到则丢弃；未超时才发布。

使用前手动设 mode_manager=-2。
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future as ConcurrentFuture
import io
import json
import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
PROMPT_TXT_FILE = "/home/pc/Fast-Drone-250/agent/ori/agent_prompt.txt"
SIMULATE = False  # True: Gazebo iris_fpv_cam；False: RealSense
SHOW_RGB = True  # True: 实时弹窗显示 RGB 图像
RGB_IMAGE_TOPIC_REAL = "/camera/color/image_raw"
RGB_IMAGE_TOPIC_SIM = "/iris/usb_cam/image_raw"
RGB_IMAGE_TOPIC = RGB_IMAGE_TOPIC_SIM if SIMULATE else RGB_IMAGE_TOPIC_REAL
COMMAND_POS_TOPIC = "/command_pos"
VINS_TOPIC = "/vins_fusion/imu_propagate"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL: Optional[str] = "/home/ps/ltc/Qwen3-VL-8B-Instruct-FP8/"
VLLM_MAX_TOKENS = 96
VLLM_HTTP_TIMEOUT_S = 300.0
USE_JSON_RESPONSE_FORMAT = True

IMG_MAX_SIZE = 224
STATUS_CHECK_PERIOD_S = 0.2
# 人工设定：VLM 请求发送周期（秒）
VLM_SEND_PERIOD_S = 0.2
# 单条请求超时（秒），超时丢弃
VLM_RESPONSE_TIMEOUT_S = 8.0
SENSOR_LOSS_TICKS = 2
STATS_PRINT_PERIOD_S = 4.0

# 发布 /command_pos 前硬限幅（不传 VLM，仅本地裁剪）
BODY_DELTA_MAX_M = 0.3
CMD_Z_MIN_M = 0.5
CMD_Z_MAX_M = 2.5

USER_TEXT_SUFFIX_ONE_RGB = " 附图为前视RGB。"

DRONE_REPLY_JSON_SPEC = (
    "输出必须是单行紧凑JSON，"
    '且仅含下列4个键，键名字符完全一致：'
    '"x_m","y_m","z_m","yaw_deg"。'
    "x_m/y_m/z_m 为 JSON number，单位米，表示相对当前机体系的期望目标位置；"
    "yaw_deg 为 JSON number，单位度，表示相对当前航向的期望偏航角增量。"
)

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
bridge = CvBridge()
cmd_pub: Optional[rospy.Publisher] = None
vlm_client: Optional[AsyncOpenAI] = None
vlm_model: str = ""
task_text: str = ""

asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

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
vins_yaw = 0.0

vlm_request_seq = 0
latest_responded_seq = -1
_state_lock = threading.Lock()

_stats_vlm_send_base = 0
_stats_rtt_sum_s = 0.0
_stats_rtt_n = 0
_last_pub_prompt = ""
_last_pub_reply = ""


def load_prompt_from_txt(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"prompt 文件不存在: {p.resolve()}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt 文件为空: {p.resolve()}")
    return text


def build_user_prompt(task: str) -> str:
    return (
        f"{task.strip()}\n"
        f"{DRONE_REPLY_JSON_SPEC}"
        f"{USER_TEXT_SUFFIX_ONE_RGB}"
    )


def array_to_png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def resize_rgb_uint8(arr: np.ndarray, max_size: int = IMG_MAX_SIZE) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_size:
        return arr
    scale = max_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = PILImage.fromarray(arr).resize((new_w, new_h), PILImage.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def build_messages_one_image(prompt: str, image_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def parse_drone_reply(text: str) -> dict[str, float]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return {
        "x_m": float(data["x_m"]),
        "y_m": float(data["y_m"]),
        "z_m": float(data["z_m"]),
        "yaw_deg": float(data["yaw_deg"]),
    }


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def body_delta_to_world(
    x_m: float, y_m: float, z_m: float, yaw_deg: float,
    *, vins_x: float, vins_y: float, vins_z: float, vins_yaw_rad: float,
) -> tuple[float, float, float, float]:
    cy = math.cos(vins_yaw_rad)
    sy = math.sin(vins_yaw_rad)
    wx = vins_x + cy * x_m - sy * y_m
    wy = vins_y + sy * x_m + cy * y_m
    wz = vins_z + z_m
    target_yaw = vins_yaw_rad + math.radians(yaw_deg)
    target_yaw = (target_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return wx, wy, wz, target_yaw


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_body_delta(parsed: dict[str, float]) -> dict[str, float]:
    raw = parsed
    clamped = {
        "x_m": _clamp(raw["x_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "y_m": _clamp(raw["y_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "z_m": _clamp(raw["z_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "yaw_deg": raw["yaw_deg"],
    }
    if (
        clamped["x_m"] != raw["x_m"]
        or clamped["y_m"] != raw["y_m"]
        or clamped["z_m"] != raw["z_m"]
    ):
        rospy.logwarn_throttle(
            1.0,
            "body delta clamped: (%.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f)",
            raw["x_m"], raw["y_m"], raw["z_m"],
            clamped["x_m"], clamped["y_m"], clamped["z_m"],
        )
    return clamped


def clamp_world_z(wz: float) -> float:
    wz_clamped = _clamp(wz, CMD_Z_MIN_M, CMD_Z_MAX_M)
    if wz_clamped != wz:
        rospy.logwarn_throttle(1.0, "world z clamped: %.3f->%.3f", wz, wz_clamped)
    return wz_clamped


async def pick_model_id(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("/v1/models 未返回任何模型，请检查 vLLM 是否已启动")
    return models.data[0].id


async def async_vlm_request(
    seq: int,
    messages: list[dict[str, Any]],
    user_prompt: str,
    vins_at_request: dict[str, float],
    timeout_s: float,
) -> dict[str, Any]:
    """异步调用 VLM；返回 dict 带 seq，供 vlm_response_cb 识别是哪一次请求。"""
    t_send = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "model": vlm_model,
            "messages": messages,
            "max_tokens": VLLM_MAX_TOKENS,
        }
        if USE_JSON_RESPONSE_FORMAT:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await asyncio.wait_for(
            vlm_client.chat.completions.create(**kwargs),
            timeout=timeout_s,
        )
        text = resp.choices[0].message.content
        if not text:
            raise RuntimeError("VLM 返回空内容")
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": True,
            "text": text,
            "user_prompt": user_prompt,
            "vins_at_request": vins_at_request,
            "latency_s": t_recv - t_send,
        }
    except asyncio.TimeoutError:
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": False,
            "error": "timeout",
            "user_prompt": user_prompt,
            "vins_at_request": vins_at_request,
            "latency_s": t_recv - t_send,
        }
    except Exception as e:
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": False,
            "error": repr(e),
            "user_prompt": user_prompt,
            "vins_at_request": vins_at_request,
            "latency_s": t_recv - t_send,
        }


def _run_async_loop() -> None:
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()


def _vlm_future_done(future: ConcurrentFuture) -> None:
    """VLM 返回后立即处理（publish 线程安全，不用 Duration(0) 的 Timer）。"""
    try:
        result = future.result()
    except Exception as e:
        result = {
            "seq": -1,
            "ok": False,
            "error": repr(e),
            "vins_at_request": None,
        }
    vlm_response_cb(result)


def rgb_cb(msg: Image) -> None:
    global rgb_image, rgb_count
    try:
        bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_image = np.ascontiguousarray(bgr[:, :, ::-1])
        rgb_count += 1
        if SHOW_RGB:
            cv2.imshow("agent_rgb", bgr)
            cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logwarn_throttle(5.0, "RGB 转换失败: %s", e)


def vins_cb(msg: Odometry) -> None:
    global vins_count, vins_pos, vins_vel, vins_yaw
    q = msg.pose.pose.orientation
    vins_count += 1
    vins_pos[0] = msg.pose.pose.position.x
    vins_pos[1] = msg.pose.pose.position.y
    vins_pos[2] = msg.pose.pose.position.z
    vins_vel[0] = msg.twist.twist.linear.x
    vins_vel[1] = msg.twist.twist.linear.y
    vins_vel[2] = msg.twist.twist.linear.z
    vins_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)


def stats_print_cb(_event) -> None:
    """每 STATS_PRINT_PERIOD_S 打印一次：prompt/返回值、VLM 返回频率(avg_fps)、发送Hz、平均 RTT。"""
    global _stats_vlm_send_base, _stats_rtt_sum_s, _stats_rtt_n

    dt = STATS_PRINT_PERIOD_S
    avg_fps = _stats_rtt_n / dt  # VLM 返回频率（本周期收到多少条回复 / 秒）
    vlm_send_hz = (vlm_request_seq - _stats_vlm_send_base) / dt
    if _stats_rtt_n:
        rtt_str = f"{_stats_rtt_sum_s / _stats_rtt_n * 1000.0:.0f}ms"
    else:
        rtt_str = "n/a"

    print(
        f"\n========== [{dt:.0f}s] avg_fps={avg_fps:.2f} "
        f"vlm_send={vlm_send_hz:.2f}Hz avg_rtt={rtt_str} (n={_stats_rtt_n}) ==========",
        flush=True,
    )
    if _last_pub_prompt:
        print("[prompt]\n", _last_pub_prompt, sep="", flush=True)
        print("[reply]\n", _last_pub_reply, sep="", flush=True)
    else:
        print("(本周期内无成功发布 /command_pos)", flush=True)
    print("=" * 60, flush=True)

    _stats_vlm_send_base = vlm_request_seq
    _stats_rtt_sum_s = 0.0
    _stats_rtt_n = 0


def _record_vlm_rtt(latency_s: float) -> None:
    global _stats_rtt_sum_s, _stats_rtt_n
    if math.isfinite(latency_s) and latency_s >= 0.0:
        _stats_rtt_sum_s += latency_s
        _stats_rtt_n += 1


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


def vlm_send_cb(_event) -> None:
    """固定周期 Timer（VLM_SEND_PERIOD_S）：到点即发 VLM 请求（可多条在飞）。"""
    global vlm_request_seq

    if not (vins_received and rgb_received) or rgb_image is None:
        return
    if asyncio_loop is None or vlm_client is None:
        return

    vins_at_request = {
        "x": float(vins_pos[0]),
        "y": float(vins_pos[1]),
        "z": float(vins_pos[2]),
        "vx": float(vins_vel[0]),
        "vy": float(vins_vel[1]),
        "vz": float(vins_vel[2]),
        "yaw_rad": float(vins_yaw),
    }

    rgb = resize_rgb_uint8(rgb_image.copy(), IMG_MAX_SIZE)
    user_prompt = build_user_prompt(task_text)
    messages = build_messages_one_image(user_prompt, array_to_png_data_url(rgb))

    seq = vlm_request_seq
    vlm_request_seq += 1

    future = asyncio.run_coroutine_threadsafe(
        async_vlm_request(
            seq, messages, user_prompt, vins_at_request, VLM_RESPONSE_TIMEOUT_S,
        ),
        asyncio_loop,
    )
    future.add_done_callback(_vlm_future_done)


def vlm_response_cb(result: dict[str, Any]) -> None:
    """
    已收到更大 seq 的回复后，更小 seq 的晚到回复直接丢弃（与发送先后无关）。
    未超时且解析成功才发布 /command_pos。
    """
    global latest_responded_seq, _last_pub_prompt, _last_pub_reply

    seq = result.get("seq", -1)
    ok = result.get("ok", False)
    latency = result.get("latency_s", float("nan"))
    _record_vlm_rtt(latency)

    with _state_lock:
        if seq < latest_responded_seq:
            return
        latest_responded_seq = max(latest_responded_seq, seq)

    if not ok:
        err = result.get("error", "")
        if err == "timeout":
            rospy.logwarn("[vlm_response_cb] seq=%d 超时丢弃", seq)
        else:
            rospy.logerr("[vlm_response_cb] seq=%d 失败: %s", seq, err)
        return

    vins_snap = result.get("vins_at_request")
    if not vins_snap:
        rospy.logerr("[vlm_response_cb] seq=%d 缺少 vins_at_request", seq)
        return

    if cmd_pub is None:
        rospy.logerr("[vlm_response_cb] seq=%d cmd_pub 未初始化", seq)
        return

    try:
        parsed = clamp_body_delta(parse_drone_reply(result["text"]))
        wx, wy, wz, target_yaw = body_delta_to_world(
            parsed["x_m"], parsed["y_m"], parsed["z_m"], parsed["yaw_deg"],
            vins_x=vins_snap["x"],
            vins_y=vins_snap["y"],
            vins_z=vins_snap["z"],
            vins_yaw_rad=vins_snap["yaw_rad"],
        )
        wz = clamp_world_z(wz)
    except Exception as e:
        rospy.logerr("[vlm_response_cb] seq=%d 解析/坐标变换失败: %s", seq, e)
        return

    _last_pub_prompt = result.get("user_prompt", "")
    _last_pub_reply = result.get("text", "")

    cmd = Odometry()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "world"
    cmd.pose.pose.position.x = wx
    cmd.pose.pose.position.y = wy
    cmd.pose.pose.position.z = wz
    cmd.pose.pose.orientation.w = 1.0
    cmd.twist.twist.angular.x = target_yaw
    cmd_pub.publish(cmd)


def _close_rgb_preview() -> None:
    if SHOW_RGB:
        cv2.destroyAllWindows()


def main() -> None:
    global cmd_pub, vlm_client, vlm_model, task_text, asyncio_loop
    global _stats_vlm_send_base

    rospy.init_node("vlm_agent", anonymous=False)
    if SHOW_RGB:
        rospy.on_shutdown(_close_rgb_preview)

    task_text = load_prompt_from_txt(PROMPT_TXT_FILE)
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

    cmd_pub = rospy.Publisher(COMMAND_POS_TOPIC, Odometry, queue_size=10)
    rospy.Subscriber(RGB_IMAGE_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Subscriber(VINS_TOPIC, Odometry, vins_cb, queue_size=10)
    _stats_vlm_send_base = vlm_request_seq
    rospy.Timer(rospy.Duration(STATUS_CHECK_PERIOD_S), status_check_cb)
    rospy.Timer(rospy.Duration(VLM_SEND_PERIOD_S), vlm_send_cb)
    rospy.Timer(rospy.Duration(STATS_PRINT_PERIOD_S), stats_print_cb)

    print(
        f"vlm_agent: simulate={SIMULATE} show_rgb={SHOW_RGB} prompt={PROMPT_TXT_FILE} "
        f"RGB={RGB_IMAGE_TOPIC} model={vlm_model} | VLM周期={VLM_SEND_PERIOD_S}s 超时={VLM_RESPONSE_TIMEOUT_S}s | "
        f"统计打印每 {STATS_PRINT_PERIOD_S}s",
        flush=True,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
