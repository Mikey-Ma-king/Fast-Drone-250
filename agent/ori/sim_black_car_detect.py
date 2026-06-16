#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真相机 + VLM：按 DETECT_PROMPT 检测目标，返回 bbox 并在窗口画框（无 label 字段）。

前提：agent/config.py 中 SIMULATE=True（订阅 /iris/usb_cam/image_raw）。

运行（仓库根目录）:
  python3 agent/ori/sim_black_car_detect.py
  python3 agent/ori/sim_black_car_detect.py --period 1.0
"""

from __future__ import annotations

import argparse
import asyncio
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
from sensor_msgs.msg import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.config import (  # noqa: E402
    IMG_MAX_SIZE,
    NORM_SCALE_2D,
    RGB_IMAGE_TOPIC,
    SIMULATE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_HTTP_TIMEOUT_S,
    VLLM_MODEL,
)
from agent.motion_parse import extract_json  # noqa: E402
from agent.parse_2d import norm1000_bbox_to_original  # noqa: E402
from agent.vlm_utils import (  # noqa: E402
    array_to_png_data_url,
    async_vlm_request,
    build_messages_single_image,
    exact_pixels_from_shape,
    pick_model_id,
    resize_rgb_for_vlm,
)

DETECT_PROMPT = """You are a visual detector for a forward-facing drone camera.

Task: find every road lamp in the image.

Return JSON only:
{"detections":[[x1,y1,x2,y2]]}
If none: {"detections":[]}
"""

IMSHOW_WINDOW = "sim_black_car_detect"
BBOX_COLOR = (0, 165, 255)  # BGR 橙
BBOX_THICKNESS = 2
DETECT_MAX_TOKENS = 256
DETECT_TIMEOUT_S = 30.0
DISPLAY_PERIOD_S = 0.033
SENSOR_LOSS_TICKS = 5

bridge = CvBridge()
_rgb_bgr: Optional[np.ndarray] = None
_orig_wh: tuple[int, int] = (0, 0)
_rgb_lock = threading.Lock()
_rgb_count = 0
_last_rgb_count = 0
_last_rgb_timer = 0
_rgb_received = False

_latest_bboxes: list[tuple[int, int, int, int]] = []
_latest_reply = ""
_latest_latency_ms = -1.0
_latest_vlm_hw: tuple[int, int] = (0, 0)
_overlay_lock = threading.Lock()

vlm_client: Optional[AsyncOpenAI] = None
vlm_model: str = ""
asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
vlm_request_seq = 0
_vlm_in_flight = False
_vlm_resend_pending = False


def _parse_bbox_values(raw: Any) -> Optional[tuple[float, float, float, float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for k1, k2, k3, k4 in (
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("xmin", "ymin", "xmax", "ymax"),
        ):
            if all(k in raw for k in (k1, k2, k3, k4)):
                try:
                    return (
                        float(raw[k1]), float(raw[k2]),
                        float(raw[k3]), float(raw[k4]),
                    )
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        except (TypeError, ValueError):
            return None
    return None


def _extract_detections(
    data: Any, raw_text: str,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    root = data if isinstance(data, dict) else None
    if root is not None:
        dets = root.get("detections")
        if isinstance(dets, list):
            for item in dets:
                if isinstance(item, (list, tuple)):
                    box = _parse_bbox_values(item)
                elif isinstance(item, dict):
                    box = _parse_bbox_values(item.get("bbox", item.get("box")))
                else:
                    box = None
                if box is not None:
                    boxes.append(box)
        if not boxes:
            box = _parse_bbox_values(root.get("bbox", root.get("box")))
            if box is not None:
                boxes.append(box)
    if not boxes:
        m = re.search(
            r'"bbox"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*'
            r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
            raw_text,
        )
        if m:
            boxes.append(
                (
                    float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), float(m.group(4)),
                ),
            )
    return boxes


def parse_detection_bboxes(
    data: Any,
    raw_text: str,
    *,
    ow: int,
    oh: int,
) -> list[tuple[int, int, int, int]]:
    return [
        norm1000_bbox_to_original(*box, ow=ow, oh=oh)
        for box in _extract_detections(data, raw_text)
    ]


def _draw_bboxes(bgr: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    vis = bgr.copy()
    h, w = vis.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS, cv2.LINE_AA)
        label = f"#{i + 1}"
        cv2.putText(
            vis, label, (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1, cv2.LINE_AA,
        )
    return vis


def rgb_cb(msg: Image) -> None:
    global _rgb_bgr, _orig_wh, _rgb_count, _rgb_received, _last_rgb_count, _last_rgb_timer
    try:
        bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logwarn_throttle(5.0, "RGB 转换失败: %s", e)
        return
    h, w = bgr.shape[:2]
    with _rgb_lock:
        _rgb_bgr = np.ascontiguousarray(bgr)
        _orig_wh = (int(w), int(h))
        _rgb_count += 1
        _rgb_received = True
        _last_rgb_count = _rgb_count
        _last_rgb_timer = 0


def status_check_cb(_evt) -> None:
    global _rgb_received, _last_rgb_count, _last_rgb_timer
    if _rgb_count != _last_rgb_count:
        _rgb_received = True
        _last_rgb_count = _rgb_count
        _last_rgb_timer = 0
    else:
        _last_rgb_timer += 1
        if _last_rgb_timer >= SENSOR_LOSS_TICKS:
            _rgb_received = False


def _run_async_loop() -> None:
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()


def _vlm_done(fut: ConcurrentFuture) -> None:
    global _vlm_in_flight, _vlm_resend_pending
    global _latest_bboxes, _latest_reply, _latest_latency_ms, _latest_vlm_hw
    try:
        result = fut.result()
    except Exception as e:
        result = {"seq": -1, "ok": False, "error": repr(e)}
    try:
        if result.get("ok"):
            ow, oh = result.get("orig_wh", (0, 0))
            raw = result.get("text", "")
            data = extract_json(raw)
            bboxes = parse_detection_bboxes(
                data, raw, ow=int(ow), oh=int(oh),
            )
            vlm_hw = result.get("image_hw") or [0, 0]
            with _overlay_lock:
                _latest_bboxes = bboxes
                _latest_reply = raw
                _latest_latency_ms = float(result.get("latency_s", 0.0)) * 1000.0
                if len(vlm_hw) == 2:
                    _latest_vlm_hw = (int(vlm_hw[1]), int(vlm_hw[0]))
            rospy.loginfo(
                "[detect] seq=%s n=%d rtt=%.0fms",
                result.get("seq"), len(bboxes), _latest_latency_ms,
            )
        else:
            rospy.logwarn("[detect] seq=%s failed: %s", result.get("seq"), result.get("error"))
    finally:
        _vlm_in_flight = False
        if _vlm_resend_pending:
            _vlm_resend_pending = False
            _dispatch_vlm_request()


def _dispatch_vlm_request() -> None:
    global vlm_request_seq, _vlm_in_flight
    if _vlm_in_flight or asyncio_loop is None or vlm_client is None:
        return
    if not _rgb_received:
        return
    with _rgb_lock:
        if _rgb_bgr is None:
            return
        bgr_orig = _rgb_bgr.copy()
        ow, oh = _orig_wh

    rgb_send, img_meta = resize_rgb_for_vlm(
        cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB), IMG_MAX_SIZE,
    )
    rw = int(img_meta["vlm_rw"])
    rh = int(img_meta["vlm_rh"])
    seq = vlm_request_seq
    vlm_request_seq += 1
    exact_pixels = exact_pixels_from_shape(rh, rw)
    messages = build_messages_single_image(
        DETECT_PROMPT,
        array_to_png_data_url(rgb_send),
        exact_pixels=exact_pixels,
    )
    _vlm_in_flight = True
    fut = asyncio.run_coroutine_threadsafe(
        async_vlm_request(
            vlm_client,
            vlm_model,
            seq,
            messages,
            DETECT_PROMPT,
            max_tokens=DETECT_MAX_TOKENS,
            timeout_s=DETECT_TIMEOUT_S,
            extra={"orig_wh": [ow, oh], "vlm_size": IMG_MAX_SIZE, "role": "black_car_detect"},
        ),
        asyncio_loop,
    )
    fut.add_done_callback(_vlm_done)


def vlm_send_cb(_evt) -> None:
    global _vlm_resend_pending
    if _vlm_in_flight:
        _vlm_resend_pending = True
        return
    _dispatch_vlm_request()


def _display_once() -> bool:
    with _rgb_lock:
        if _rgb_bgr is None:
            return True
        frame = _rgb_bgr.copy()
        ow, oh = _orig_wh
    with _overlay_lock:
        bboxes = list(_latest_bboxes)
        reply = _latest_reply
        rtt_ms = _latest_latency_ms
        vlm_w, vlm_h = _latest_vlm_hw
    vis = _draw_bboxes(frame, bboxes)
    vlm_tag = f" vlm={vlm_w}x{vlm_h}" if vlm_w and vlm_h else ""
    status = (
        f"sim={SIMULATE} topic={RGB_IMAGE_TOPIC} src={ow}x{oh}{vlm_tag} "
        f"boxes={len(bboxes)}"
    )
    if rtt_ms >= 0:
        status += f" rtt={rtt_ms:.0f}ms"
    cv2.putText(
        vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    if reply:
        short = reply.replace("\n", " ")[:80]
        cv2.putText(
            vis, short, (8, oh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )
    cv2.imshow(IMSHOW_WINDOW, vis)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="仿真相机 VLM 目标检测 (bbox)")
    p.add_argument("--period", type=float, default=1.0, help="VLM 请求周期 (秒)")
    p.add_argument(
        "--allow-real",
        action="store_true",
        help="允许 SIMULATE=False 时仍运行（使用 config 当前 RGB topic）",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    global vlm_client, vlm_model, asyncio_loop
    args = parse_args(argv)

    if not SIMULATE and not args.allow_real:
        print(
            "需要仿真相机：请在 agent/config.py 设置 SIMULATE=True，"
            "或加 --allow-real",
            file=sys.stderr,
        )
        sys.exit(1)

    rospy.init_node("sim_black_car_detect", anonymous=True)
    print(
        f"sim_black_car_detect\n"
        f"  SIMULATE={SIMULATE} RGB={RGB_IMAGE_TOPIC}\n"
        f"  vlm_period={args.period}s 按 q/Esc 退出",
        flush=True,
    )

    vlm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL.rstrip("/"),
        api_key=VLLM_API_KEY,
        timeout=VLLM_HTTP_TIMEOUT_S,
    )
    threading.Thread(target=_run_async_loop, daemon=True).start()
    while asyncio_loop is None and not rospy.is_shutdown():
        time.sleep(0.01)

    global vlm_model
    if VLLM_MODEL:
        vlm_model = VLLM_MODEL
    else:
        fut = asyncio.run_coroutine_threadsafe(pick_model_id(vlm_client), asyncio_loop)
        vlm_model = fut.result(timeout=60.0)
    print(f"  model={vlm_model}", flush=True)

    cv2.namedWindow(IMSHOW_WINDOW, cv2.WINDOW_NORMAL)
    rospy.Subscriber(RGB_IMAGE_TOPIC, Image, rgb_cb, queue_size=1, buff_size=2**24)
    rospy.Timer(rospy.Duration(0.1), status_check_cb)
    rospy.Timer(rospy.Duration(max(args.period, 0.2)), vlm_send_cb)

    rate = rospy.Rate(1.0 / DISPLAY_PERIOD_S)
    while not rospy.is_shutdown():
        if not _display_once():
            break
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
