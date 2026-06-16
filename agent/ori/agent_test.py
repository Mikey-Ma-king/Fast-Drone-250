#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 测试 agent：1s 间隔发请求，prompt 允许「场景分析 + 坐标」，只 print 返回，不发布 /command_pos。

依赖 ROS RGB；不依赖 VINS，有图即发 VLM。vLLM 经本机 8000（SSH 隧道）。
运行: python3 agent_test.py
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future as ConcurrentFuture
import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

from PIL import Image as PILImage

PROMPT_TXT_FILE = "/home/pc/Fast-Drone-250/agent/ori/agent_prompt.txt"
RGB_IMAGE_TOPIC = "/camera/color/image_raw"

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL: Optional[str] = "/home/ps/ltc/Qwen3-VL-8B-Instruct-FP8/"
VLLM_MAX_TOKENS = 512
VLLM_HTTP_TIMEOUT_S = 300.0

IMG_MAX_SIZE = 224
VLM_SEND_PERIOD_S = 1.0
VLM_RESPONSE_TIMEOUT_S = 30.0
STATUS_CHECK_PERIOD_S = 0.1
SENSOR_LOSS_TICKS = 5
STATUS_CHECK_PERIOD_S = 0.1
SENSOR_LOSS_TICKS = 5
CLEAR_SCREEN_ON_PRINT = True

USER_TEXT_SUFFIX_ONE_RGB = " 附图为前视RGB。"

TEST_REPLY_SPEC = (
    "请先用 2～5 句中文简要分析前视场景、障碍与可行策略。"
    "最后单独一行给出机相机坐标系下1s后的目标相对坐标，格式严格为："
    "坐标: x_m=<米>, y_m=<米>, z_m=<米>, yaw_deg=<度>"
    "（yaw_deg 为相对当前航向的偏航增量）。"
    "不要只输出 JSON，分析文字与坐标行都要有。"
)

bridge = CvBridge()
vlm_client: Optional[AsyncOpenAI] = None
vlm_model: str = ""
task_text: str = ""
asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

rgb_image: Optional[np.ndarray] = None
rgb_count = 0
last_rgb_count = 0
last_rgb_timer = 0
last_rgb_count = 0
last_rgb_timer = 0
rgb_received = False

vlm_request_seq = 0


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
        f"{TEST_REPLY_SPEC}"
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


async def pick_model_id(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("/v1/models 未返回任何模型")
    return models.data[0].id


async def async_vlm_request(
    seq: int,
    messages: list[dict[str, Any]],
    user_prompt: str,
    rgb_wh: tuple[int, int],
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    base = {"seq": seq, "user_prompt": user_prompt, "rgb_wh": rgb_wh}
    try:
        resp = await asyncio.wait_for(
            vlm_client.chat.completions.create(
                model=vlm_model,
                messages=messages,
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


def _clear_terminal() -> None:
    """清屏并把光标移到左上角，便于每次在同一位置刷新输出。"""
    if CLEAR_SCREEN_ON_PRINT and sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)


def _run_async_loop() -> None:
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_forever()


def _print_vlm_result(result: dict[str, Any]) -> None:
    _clear_terminal()

    seq = result.get("seq", -1)
    latency = result.get("latency_s", float("nan"))
    rgb_wh = result.get("rgb_wh", (0, 0))
    w, h = rgb_wh if isinstance(rgb_wh, (list, tuple)) and len(rgb_wh) == 2 else (0, 0)

    print(f"{'=' * 60}\n[seq={seq}] rtt={latency:.2f}s RGB={w}x{h}", flush=True)
    print("[prompt]\n", result.get("user_prompt", ""), sep="", flush=True)
    print("[reply]", flush=True)
    if not result.get("ok"):
        print(result.get("error", "unknown"), flush=True)
    else:
        print(result.get("text", ""), flush=True)
    print("=" * 60, flush=True)


def _vlm_future_done(future: ConcurrentFuture) -> None:
    try:
        result = future.result()
    except Exception as e:
        result = {
            "seq": -1,
            "ok": False,
            "error": repr(e),
            "user_prompt": "",
            "rgb_wh": (0, 0),
            "latency_s": float("nan"),
        }
    _print_vlm_result(result)


def rgb_cb(msg: Image) -> None:
    global rgb_image, rgb_count, rgb_received, last_rgb_count, last_rgb_timer
    try:
        bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_image = np.ascontiguousarray(bgr[:, :, ::-1])
        rgb_count += 1
        rgb_received = True
        last_rgb_count = rgb_count
        last_rgb_timer = 0
    except CvBridgeError as e:
        rospy.logwarn_throttle(5.0, "RGB 转换失败: %s", e)


def status_check_cb(_event) -> None:
    global rgb_received, last_rgb_count, last_rgb_timer

    if rgb_count != last_rgb_count:
        rgb_received = True
        last_rgb_count = rgb_count
        last_rgb_timer = 0
    else:
        last_rgb_timer += 1
        if last_rgb_timer >= SENSOR_LOSS_TICKS:
            rgb_received = False


def vlm_send_cb(_event) -> None:
    global vlm_request_seq

    if not rgb_received or rgb_image is None:
        return
    if asyncio_loop is None or vlm_client is None:
        return

    user_prompt = build_user_prompt(task_text)
    rgb = resize_rgb_uint8(rgb_image.copy(), IMG_MAX_SIZE)
    messages = build_messages_one_image(user_prompt, array_to_png_data_url(rgb))

    seq = vlm_request_seq
    vlm_request_seq += 1
    rgb_wh = (int(rgb.shape[1]), int(rgb.shape[0]))

    future = asyncio.run_coroutine_threadsafe(
        async_vlm_request(seq, messages, user_prompt, rgb_wh, VLM_RESPONSE_TIMEOUT_S),
        asyncio_loop,
    )
    future.add_done_callback(_vlm_future_done)


def main() -> None:
    global vlm_client, vlm_model, task_text, asyncio_loop

    rospy.init_node("vlm_agent_test", anonymous=False)
    task_text = load_prompt_from_txt(PROMPT_TXT_FILE)
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
        vlm_model = fut.result(timeout=60.0)

    rospy.Subscriber(RGB_IMAGE_TOPIC, Image, rgb_cb, queue_size=1)
    rospy.Timer(rospy.Duration(STATUS_CHECK_PERIOD_S), status_check_cb)
    rospy.Timer(rospy.Duration(VLM_SEND_PERIOD_S), vlm_send_cb)

    print(
        f"agent_test: 每 {VLM_SEND_PERIOD_S}s 发 VLM，只 print 返回，不发布 command_pos\n"
        f"  prompt={PROMPT_TXT_FILE} model={vlm_model}",
        flush=True,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
