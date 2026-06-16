#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 版 2D VLM 测试：解析与坐标映射对齐 ori/example_2d.py；
订阅 RGB（+ 可选 VINS），周期请求 VLM，OpenCV 显示附图上航点。

运行（仓库根目录）: python3 agent/ori/agent_test_2d.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Config（与 example_2d 对齐；ROS topic / 可视化参数单独列出）
# ---------------------------------------------------------------------------

PROMPT_TXT_FILE = "/home/pc/Fast-Drone-250/agent/ori/agent_prompt.txt"
RGB_IMAGE_TOPIC = "/camera/color/image_raw"
VINS_TOPIC = "/vins_fusion/imu_propagate"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL: Optional[str] = "/home/ps/ltc/Qwen3-VL-8B-Instruct-FP8/"
VLLM_MAX_TOKENS = 256
VLLM_HTTP_TIMEOUT_S = 300.0

RESIZE_WH = (448, 448)  # example_2d.RESIZE_WH
NORM_SCALE = 1000.0
HORIZON_S = 1.0
MAX_FORWARD_SPEED_MPS = 2.0
OBJECT_LABEL = "vlm_target"

VLM_SEND_PERIOD_S = 0.5
VLM_RESPONSE_TIMEOUT_S = 30.0
STATUS_CHECK_PERIOD_S = 0.1
DISPLAY_PERIOD_S = 0.033
SENSOR_LOSS_TICKS = 5
REQUIRE_VINS_FOR_VLM = False

IMSHOW_WINDOW = "vlm_2d"
IMSHOW_SCALE = 2

# ---------------------------------------------------------------------------
# example_2d：JSON / 航点解析 / 0~1000 → 像素 → 原图
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def parse_points_regex(text: str) -> list[list[float]]:
    points: list[list[float]] = []
    for x_s, y_s in re.findall(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)", text):
        points.append([float(x_s), float(y_s)])
    for x_s, y_s in re.findall(
        r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text,
    ):
        points.append([float(x_s), float(y_s)])
    return points


def normalize_waypoints(
    data: Any, raw_text: str, default_label: str,
) -> list[dict[str, Any]]:
    waypoints: list[dict[str, Any]] = []
    if data is not None:
        if isinstance(data, dict):
            if "waypoints" in data and isinstance(data["waypoints"], list):
                data = data["waypoints"]
            elif "points" in data and isinstance(data["points"], list):
                label = data.get("object", data.get("label", default_label))
                for pt in data["points"]:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        waypoints.append({"label": label, "center": [pt[0], pt[1]]})
                return waypoints[:1]
            elif any(
                k in data
                for k in ("center_2d", "center", "point", "waypoint_2d", "waypoint")
            ):
                data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                label = item.get("label", item.get("object", default_label))
                for key in ("waypoint_2d", "center_2d", "waypoint", "center", "point"):
                    if key in item:
                        c = item[key]
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            waypoints.append({"label": label, "center": [c[0], c[1]]})
                        break
    if not waypoints:
        for pt in parse_points_regex(raw_text):
            waypoints.append({"label": default_label, "center": pt})
    return waypoints[:1]


def _extract_speed(value: Any) -> Optional[float]:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    if speed < 0:
        speed = 0.0
    return min(speed, MAX_FORWARD_SPEED_MPS)


def parse_step_output(data: Any, raw_text: str, default_label: str) -> dict[str, Any]:
    waypoints = normalize_waypoints(data, raw_text, default_label)
    center = waypoints[0]["center"] if waypoints else None
    label = waypoints[0]["label"] if waypoints else default_label
    speed: Optional[float] = None
    speed_keys = ("forward_speed_mps", "forward_speed", "speed_mps", "speed")
    candidates: list[Any] = []
    if isinstance(data, dict):
        candidates.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                candidates.append(item)
    for item in candidates:
        for key in speed_keys:
            if key in item:
                speed = _extract_speed(item[key])
                if speed is not None:
                    break
        if speed is not None:
            break
    if speed is None:
        m = re.search(
            r'"(?:forward_speed_mps|forward_speed|speed_mps|speed)"\s*:\s*(\d+(?:\.\d+)?)',
            raw_text,
        )
        if m:
            speed = _extract_speed(m.group(1))
    if speed is None:
        speed = 0.5
    return {"label": label, "center": center, "forward_speed_mps": speed}


def is_normalized_coord(x: float, y: float, width: int, height: int) -> bool:
    return max(x, y) > max(width, height) + 10


def map_to_pixels(
    center: list[Any], width: int, height: int,
) -> Optional[tuple[int, int]]:
    try:
        x, y = float(center[0]), float(center[1])
    except Exception:
        return None
    if is_normalized_coord(x, y, width, height):
        x = x * width / NORM_SCALE
        y = y * height / NORM_SCALE
    x = int(round(max(0, min(width - 1, x))))
    y = int(round(max(0, min(height - 1, y))))
    return x, y


def map_resized_to_original(
    x: int, y: int, rw: int, rh: int, ow: int, oh: int,
) -> tuple[int, int]:
    ox = int(round(x / max(rw - 1, 1) * (ow - 1)))
    oy = int(round(y / max(rh - 1, 1) * (oh - 1)))
    return max(0, min(ox, ow - 1)), max(0, min(oy, oh - 1))


def enrich_waypoint(
    center: list[Any], rw: int, rh: int, ow: int, oh: int, label: str,
) -> Optional[dict[str, Any]]:
    px = map_to_pixels(center, rw, rh)
    if px is None:
        return None
    rx, ry = px
    ox, oy = map_resized_to_original(rx, ry, rw, rh, ow, oh)
    return {
        "label": label,
        "waypoint_raw": center,
        "waypoint_resized_px": [rx, ry],
        "waypoint_original_px": [ox, oy],
        "coord_was_normalized": is_normalized_coord(
            float(center[0]), float(center[1]), rw, rh,
        ),
    }


def parse_and_map_reply(
    raw_text: str, rw: int, rh: int, ow: int, oh: int,
) -> dict[str, Any]:
    data = extract_json(raw_text)
    step = parse_step_output(data, raw_text, OBJECT_LABEL)
    if step["center"] is None:
        raise ValueError("未解析到 waypoint_2d / [x,y]")
    wp = enrich_waypoint(step["center"], rw, rh, ow, oh, step["label"])
    if wp is None:
        raise ValueError(f"无效航点: {step['center']}")
    return {
        **wp,
        "forward_speed_mps": step["forward_speed_mps"],
        "raw_text": raw_text,
    }


# ---------------------------------------------------------------------------
# Prompt（对齐 example_2d build_step_prompt，不写附图分辨率）
# ---------------------------------------------------------------------------


def load_mission_from_txt(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"prompt 文件不存在: {p.resolve()}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt 文件为空: {p.resolve()}")
    return text


def build_executor_prompt(mission: str) -> str:
    return f"""
You are a drone flight executor.

Overall mission:
{mission.strip()}

Planning horizon:
- Only the next {HORIZON_S:.0f} second(s) of flight.
- Return ONE image point along a collision-free direction toward the mission.
- Collisions are strictly forbidden: keep safe clearance from people, walls, furniture, and obstacles.
- You MUST output exactly one point and one forward speed.
- forward_speed_mps: recommended forward speed for this ~{HORIZON_S:.0f}s segment, in m/s (0.0 to {MAX_FORWARD_SPEED_MPS:.1f}).

Coordinate rule:
- origin is the top-left corner
- x increases to the right
- y increases downward
- the point is [x, y]

Output only valid JSON. Do not output markdown. Do not explain.

Return format (exactly one item):
[
  {{"waypoint_2d": [x, y], "forward_speed_mps": 0.5}}
]
""".strip()


# ---------------------------------------------------------------------------
# 图像：ROS → 正方形 RGB → PNG data URL
# ---------------------------------------------------------------------------

bridge = CvBridge()
rw, rh = RESIZE_WH

_rgb_topic = RGB_IMAGE_TOPIC
_rgb_lock = threading.Lock()
_rgb_bgr: Optional[np.ndarray] = None
_orig_wh: tuple[int, int] = (640, 480)
_rgb_count = 0
_last_rgb_count = 0
_last_rgb_timer = 0
_rgb_received = False
_rgb_decode_logged = False

_vins_count = 0
_last_vins_count = 0
_last_vins_timer = 0
_vins_received = False
_vins_vel = np.zeros(3, dtype=np.float64)
_vins_yaw = 0.0

_overlay_lock = threading.Lock()
_sent_bgr_by_seq: dict[int, np.ndarray] = {}
_overlay_bgr: Optional[np.ndarray] = None
_latest_result: Optional[dict[str, Any]] = None
_latest_status = "等待 VLM…"
_latest_seq = -1
_latest_latency_s = float("nan")

vlm_client: Optional[AsyncOpenAI] = None
vlm_model: str = ""
mission_text: str = ""
asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
vlm_request_seq = 0


def resize_rgb_square(bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def bgr_to_png_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError("PNG 编码失败")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


def _decode_3ch(msg: Image, channels: int) -> np.ndarray:
    h, w, step = msg.height, msg.width, msg.step
    row_bytes = w * channels
    if step < row_bytes:
        raise CvBridgeError(f"step={step} < width*ch={row_bytes}")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    need = h * step
    if buf.size < need:
        raise CvBridgeError(f"data 长度 {buf.size} < {need}")
    planar = buf[:need].reshape((h, step))[:, :row_bytes].reshape((h, w, channels))
    return np.ascontiguousarray(planar)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    try:
        return np.ascontiguousarray(bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
    except CvBridgeError:
        pass
    enc = (msg.encoding or "").lower()
    if enc in ("bgr8", "rgb8", "8uc3"):
        img = _decode_3ch(msg, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("mono8", "8uc1"):
        gray = _decode_3ch(msg, 1)[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if enc == "bgra8":
        bgra = _decode_3ch(msg, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    raise CvBridgeError(f"不支持的编码: {msg.encoding}")


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ---------------------------------------------------------------------------
# VLM 异步
# ---------------------------------------------------------------------------


def build_messages(prompt: str, image_url: str, exact_pixels: int) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "extra_fields": {
                        "mm_processor_kwargs": {
                            "min_pixels": exact_pixels,
                            "max_pixels": exact_pixels,
                        },
                    },
                },
            },
            {"type": "text", "text": prompt},
        ],
    }]


async def pick_model_id(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("/v1/models 为空")
    return models.data[0].id


async def async_vlm_request(
    seq: int,
    messages: list[dict[str, Any]],
    prompt: str,
    orig_wh: tuple[int, int],
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    base = {"seq": seq, "prompt": prompt, "orig_wh": orig_wh}
    try:
        resp = await asyncio.wait_for(
            vlm_client.chat.completions.create(
                model=vlm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=VLLM_MAX_TOKENS,
            ),
            timeout=timeout_s,
        )
        text = resp.choices[0].message.content or ""
        return {**base, "ok": True, "text": text, "latency_s": time.perf_counter() - t0}
    except asyncio.TimeoutError:
        return {**base, "ok": False, "error": "timeout", "latency_s": time.perf_counter() - t0}
    except Exception as e:
        return {**base, "ok": False, "error": repr(e), "latency_s": time.perf_counter() - t0}


def _run_async_loop() -> None:
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()


# ---------------------------------------------------------------------------
# OpenCV 可视化
# ---------------------------------------------------------------------------


def draw_waypoint_overlay(
    bgr: np.ndarray,
    result: Optional[dict[str, Any]],
    *,
    seq: int,
    latency_s: float,
    status: str,
) -> np.ndarray:
    vis = bgr.copy()
    h, w = vis.shape[:2]
    scale = IMSHOW_SCALE
    r = max(6, min(w, h) // 40)

    if result is not None:
        rx, ry = result["waypoint_resized_px"]
        cv2.circle(vis, (rx, ry), r + 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (rx, ry), r, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(vis, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

    big = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    st = "ok" if status == "ok" else status[:40]
    cv2.putText(
        big, f"seq={seq} rtt={latency_s:.2f}s {st}",
        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )
    if result is not None:
        raw = result["waypoint_raw"]
        flag = "0~1000" if result.get("coord_was_normalized") else "px"
        spd = result.get("forward_speed_mps", 0.0)
        ox, oy = result["waypoint_original_px"]
        line1 = f"raw=[{raw[0]:.0f},{raw[1]:.0f}] ({flag}) vlm=({rx},{ry})"
        line2 = f"orig=({ox},{oy}) v={spd:.2f}m/s"
        cv2.putText(big, line1, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(big, line2, (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return big


def _display_once() -> bool:
    with _overlay_lock:
        sent = _overlay_bgr.copy() if _overlay_bgr is not None else None
        result = None if _latest_result is None else dict(_latest_result)
        status = _latest_status
        seq = _latest_seq
        latency = _latest_latency_s

    if sent is None:
        with _rgb_lock:
            if _rgb_bgr is not None:
                sent = resize_rgb_square(_rgb_bgr.copy(), rw)

    if sent is None or sent.size == 0:
        blank = np.zeros((rh * IMSHOW_SCALE, rw * IMSHOW_SCALE, 3), dtype=np.uint8)
        cv2.putText(
            blank, f"No RGB ({_rgb_topic})", (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(IMSHOW_WINDOW, blank)
    else:
        cv2.imshow(IMSHOW_WINDOW, draw_waypoint_overlay(
            sent, result, seq=seq, latency_s=latency, status=status,
        ))

    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def _apply_vlm_result(api_result: dict[str, Any]) -> None:
    global _overlay_bgr, _latest_result, _latest_status, _latest_seq, _latest_latency_s

    seq = int(api_result.get("seq", -1))
    latency = float(api_result.get("latency_s", float("nan")))
    raw_text = str(api_result.get("text", "") or "")
    ow, oh = api_result.get("orig_wh", _orig_wh)
    if not isinstance(ow, int) or not isinstance(oh, int):
        ow, oh = _orig_wh

    print(f"\n{'=' * 60}\n[VLM seq={seq}] ok={api_result.get('ok')} rtt={latency:.2f}s")
    print(raw_text or api_result.get("error", ""))
    print("=" * 60, flush=True)

    parsed: Optional[dict[str, Any]] = None
    status = "ok"
    if not api_result.get("ok"):
        status = str(api_result.get("error", "unknown"))
    else:
        try:
            parsed = parse_and_map_reply(raw_text, rw, rh, ow, oh)
        except Exception as e:
            status = f"parse: {e}"

    with _overlay_lock:
        frame = _sent_bgr_by_seq.pop(seq, None)
        if frame is not None:
            _overlay_bgr = frame
        _latest_result = parsed
        _latest_status = status
        _latest_seq = seq
        _latest_latency_s = latency
        if len(_sent_bgr_by_seq) > 8:
            for k in sorted(_sent_bgr_by_seq.keys())[:-4]:
                _sent_bgr_by_seq.pop(k, None)

    if parsed is not None:
        flag = "0~1000" if parsed.get("coord_was_normalized") else "px"
        print(
            f"[mapped] raw={parsed['waypoint_raw']} ({flag}) "
            f"vlm={parsed['waypoint_resized_px']} "
            f"orig={parsed['waypoint_original_px']} "
            f"v={parsed['forward_speed_mps']:.2f}m/s",
            flush=True,
        )


def _vlm_done(fut: ConcurrentFuture) -> None:
    try:
        _apply_vlm_result(fut.result())
    except Exception as e:
        _apply_vlm_result({
            "seq": -1, "ok": False, "error": repr(e),
            "latency_s": float("nan"), "orig_wh": _orig_wh,
        })


# ---------------------------------------------------------------------------
# ROS 回调
# ---------------------------------------------------------------------------


def rgb_cb(msg: Image) -> None:
    global _rgb_bgr, _orig_wh, _rgb_count, _rgb_decode_logged
    try:
        bgr = imgmsg_to_bgr(msg)
    except CvBridgeError as e:
        rospy.logwarn_throttle(5.0, "RGB 失败: %s", e)
        return
    with _rgb_lock:
        _rgb_bgr = bgr
        _orig_wh = (int(msg.width), int(msg.height))
        _rgb_count += 1
    if not _rgb_decode_logged:
        _rgb_decode_logged = True
        rospy.loginfo(
            "首帧 RGB %dx%d enc=%s topic=%s",
            msg.width, msg.height, msg.encoding, _rgb_topic,
        )


def vins_cb(msg: Odometry) -> None:
    global _vins_count, _vins_vel, _vins_yaw
    q = msg.pose.pose.orientation
    _vins_count += 1
    _vins_vel[0] = msg.twist.twist.linear.x
    _vins_vel[1] = msg.twist.twist.linear.y
    _vins_vel[2] = msg.twist.twist.linear.z
    _vins_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)


def status_check_cb(_evt) -> None:
    global _rgb_received, _last_rgb_count, _last_rgb_timer
    global _vins_received, _last_vins_count, _last_vins_timer
    if _rgb_count != _last_rgb_count:
        _rgb_received = True
        _last_rgb_count = _rgb_count
        _last_rgb_timer = 0
    else:
        _last_rgb_timer += 1
        if _last_rgb_timer >= SENSOR_LOSS_TICKS:
            _rgb_received = False
    if _vins_count != _last_vins_count:
        _vins_received = True
        _last_vins_count = _vins_count
        _last_vins_timer = 0
    else:
        _last_vins_timer += 1
        if _last_vins_timer >= SENSOR_LOSS_TICKS:
            _vins_received = False


def vlm_send_cb(_evt) -> None:
    global vlm_request_seq
    if asyncio_loop is None or vlm_client is None:
        return
    if not _rgb_received:
        return
    if REQUIRE_VINS_FOR_VLM and not _vins_received:
        return

    with _rgb_lock:
        if _rgb_bgr is None:
            return
        bgr_orig = _rgb_bgr.copy()
        ow, oh = _orig_wh

    bgr_send = resize_rgb_square(bgr_orig, rw)
    seq = vlm_request_seq
    vlm_request_seq += 1

    with _overlay_lock:
        _sent_bgr_by_seq[seq] = bgr_send.copy()

    prompt = build_executor_prompt(mission_text)
    data_url = bgr_to_png_data_url(bgr_send)
    exact_pixels = rw * rh
    messages = build_messages(prompt, data_url, exact_pixels)

    print(f"[VLM SEND seq={seq}] orig={ow}x{oh} send={rw}x{rh}", flush=True)
    fut = asyncio.run_coroutine_threadsafe(
        async_vlm_request(seq, messages, prompt, (ow, oh), VLM_RESPONSE_TIMEOUT_S),
        asyncio_loop,
    )
    fut.add_done_callback(_vlm_done)


def main() -> None:
    global vlm_client, vlm_model, mission_text, asyncio_loop, _rgb_topic
    global REQUIRE_VINS_FOR_VLM

    rospy.init_node("vlm_agent_test_2d", anonymous=False)
    _rgb_topic = rospy.get_param("~rgb_topic", RGB_IMAGE_TOPIC)
    REQUIRE_VINS_FOR_VLM = bool(rospy.get_param("~require_vins_for_vlm", False))
    mission_text = load_mission_from_txt(PROMPT_TXT_FILE)

    print("连接 vLLM…", flush=True)
    vlm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL.rstrip("/"),
        api_key=VLLM_API_KEY,
        timeout=VLLM_HTTP_TIMEOUT_S,
    )
    threading.Thread(target=_run_async_loop, daemon=True).start()
    while asyncio_loop is None and not rospy.is_shutdown():
        time.sleep(0.01)

    if VLLM_MODEL:
        vlm_model = VLLM_MODEL
    else:
        fut = asyncio.run_coroutine_threadsafe(pick_model_id(vlm_client), asyncio_loop)
        vlm_model = fut.result(timeout=15.0)

    cv2.namedWindow(IMSHOW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(IMSHOW_WINDOW, rw * IMSHOW_SCALE, rh * IMSHOW_SCALE)

    rospy.Subscriber(_rgb_topic, Image, rgb_cb, queue_size=1, buff_size=2**24)
    rospy.Subscriber(VINS_TOPIC, Odometry, vins_cb, queue_size=10)
    rospy.Timer(rospy.Duration(STATUS_CHECK_PERIOD_S), status_check_cb)
    rospy.Timer(rospy.Duration(VLM_SEND_PERIOD_S), vlm_send_cb)

    print(
        f"agent_test_2d 就绪\n"
        f"  mission={PROMPT_TXT_FILE}\n"
        f"  rgb={_rgb_topic} vins={VINS_TOPIC} require_vins={REQUIRE_VINS_FOR_VLM}\n"
        f"  send={rw}x{rh} (example_2d) model={vlm_model}\n"
        f"  按 q / Esc 退出",
        flush=True,
    )

    rate = rospy.Rate(1.0 / DISPLAY_PERIOD_S)
    while not rospy.is_shutdown():
        if not _display_once():
            rospy.signal_shutdown("用户退出")
            break
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
