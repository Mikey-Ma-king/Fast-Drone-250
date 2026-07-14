#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent / VLM 连通性诊断。

用法（仓库根目录）:
  python3 -m agent.test_connectivity              # 只测 VLM（需 8000 端口可达）
  python3 -m agent.test_connectivity --ros      # 额外检查 RGB/VINS 话题
  python3 -m agent.test_connectivity --full     # Planner+Executor 各发一次带图请求

run_agent 收不到返回时常见原因:
  1. SSH 隧道未建立 → curl http://127.0.0.1:8000/v1/models 失败
  2. vLLM 未启动或模型路径错
  3. 请求超时（远程 GPU 忙 / 首 token 慢）
  4. RGB 或 VINS 无数据 → agent 根本不会发 VLM 请求（用 --ros 查）
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np

from agent.config import (
    EXECUTOR_MODE,
    EXECUTOR_RESPONSE_TIMEOUT_S,
    IMG_MAX_SIZE,
    PLANNER_RESPONSE_TIMEOUT_S,
    RGB_IMAGE_TOPIC,
    VINS_TOPIC,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_HTTP_TIMEOUT_S,
    VLLM_MODEL,
)
from agent.prompts import build_executor_discrete_prompt, build_planner_prompt
from agent.task_state import TaskState
from agent.vlm_utils import (
    array_to_png_data_url,
    async_vlm_request,
    build_messages_multi_image,
    build_messages_single_image,
    exact_pixels_from_shape,
    pick_model_id,
    resize_rgb_for_vlm,
)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _parse_base_host_port(base_url: str) -> tuple[str, int]:
    # http://127.0.0.1:8000/v1
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if "://" in url:
        url = url.split("://", 1)[1]
    if "/" in url:
        url = url.split("/", 1)[0]
    if ":" in url:
        host, port_s = url.rsplit(":", 1)
        return host, int(port_s)
    return url, 80


def test_tcp_port(base_url: str) -> bool:
    _section("1. TCP 端口")
    host, port = _parse_base_host_port(base_url)
    print(f"  目标: {host}:{port}  (来自 VLLM_BASE_URL={base_url})")
    try:
        with socket.create_connection((host, port), timeout=3.0):
            _ok(f"端口 {port} 可连接")
            if host in ("127.0.0.1", "localhost") and port == 8000:
                print("  提示: 若 run_agent 依赖 SSH 隧道，请先运行 run_agent.sh 或手动:")
                print("        ssh -N -L 8000:127.0.0.1:8000 -p 2145 ps@202.120.36.186")
            return True
    except OSError as e:
        _fail(f"无法连接 {host}:{port}: {e}")
        print("  → 先确认 SSH 隧道 / vLLM 服务已启动")
        return False


def test_http_models(base_url: str) -> tuple[bool, str]:
    _section("2. HTTP /v1/models")
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    url = root + "/models"
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
        _ok(f"HTTP {resp.status}, 响应片段: {body[:200]}...")
        return True, body
    except urllib.error.HTTPError as e:
        _fail(f"HTTP {e.code}: {e.read(512).decode('utf-8', errors='replace')[:300]}")
        return False, ""
    except Exception as e:
        _fail(repr(e))
        return False, ""


def _dummy_rgb(h: int = 480, w: int = 640) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    r = ((x / max(w - 1, 1)) * 200).astype(np.uint8)
    g = ((y / max(h - 1, 1)) * 200).astype(np.uint8)
    b = np.full((h, w), 120, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


async def _async_probe(
    *,
    model: str,
    full: bool,
) -> bool:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        _fail("未安装 openai: pip install openai")
        return False

    _section("3. OpenAI 客户端")
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL.rstrip("/"),
        api_key=VLLM_API_KEY,
        timeout=VLLM_HTTP_TIMEOUT_S,
    )
    print(f"  base_url={VLLM_BASE_URL}")
    print(f"  timeout={VLLM_HTTP_TIMEOUT_S}s")

    # 3a 文本探针（最快）
    print("\n  --- 3a 纯文本探针 ---")
    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": 'Reply JSON only: {"ping":1}'}],
                max_tokens=32,
                temperature=0.0,
            ),
            timeout=30.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        dt = time.perf_counter() - t0
        _ok(f"文本回复 ({dt:.1f}s): {text[:120]}")
    except asyncio.TimeoutError:
        _fail("纯文本请求 30s 超时 → vLLM 可能未就绪或 GPU 排队")
        return False
    except Exception as e:
        _fail(f"纯文本请求失败: {e!r}")
        return False

    if not full:
        _warn("未加 --full，跳过带图 Planner/Executor 测试")
        return True

    rgb = _dummy_rgb()
    rgb_vlm, meta = resize_rgb_for_vlm(rgb, IMG_MAX_SIZE)
    exact_px = exact_pixels_from_shape(int(meta["vlm_rh"]), int(meta["vlm_rw"]))
    img_url = array_to_png_data_url(rgb_vlm)

    # 3b Planner 风格（双图）
    print("\n  --- 3b Planner 探针（双图） ---")
    task_state = TaskState("test: hover and scan")
    task_state.append_subtask("move forward slowly", kind="move")
    prompt = build_planner_prompt(task_state, force_switch=False)
    messages = build_messages_multi_image(
        prompt,
        [img_url, img_url],
        exact_pixels=exact_px,
    )
    result = await async_vlm_request(
        client,
        model,
        seq=1,
        messages=messages,
        user_prompt=prompt,
        max_tokens=128,
        timeout_s=PLANNER_RESPONSE_TIMEOUT_S,
        extra={"role": "planner"},
    )
    if result.get("ok"):
        _ok(
            f"Planner ({result['latency_s']:.1f}s): "
            f"{str(result.get('text', ''))[:200]}"
        )
    else:
        _fail(f"Planner: {result.get('error')} ({result.get('latency_s', 0):.1f}s)")
        if result.get("error") == "timeout":
            _warn(f"当前 PLANNER_RESPONSE_TIMEOUT_S={PLANNER_RESPONSE_TIMEOUT_S}")
        return False

    # 3c Executor 探针（单图，离散模式 prompt）
    print(f"\n  --- 3c Executor 探针 (mode={EXECUTOR_MODE}) ---")
    exec_prompt = build_executor_discrete_prompt("move toward center of scene")
    exec_messages = build_messages_single_image(
        exec_prompt, img_url, exact_pixels=exact_px,
    )
    result = await async_vlm_request(
        client,
        model,
        seq=2,
        messages=exec_messages,
        user_prompt=exec_prompt,
        max_tokens=64,
        timeout_s=EXECUTOR_RESPONSE_TIMEOUT_S,
        extra={"role": "executor"},
    )
    if result.get("ok"):
        _ok(
            f"Executor ({result['latency_s']:.1f}s): "
            f"{str(result.get('text', ''))[:200]}"
        )
    else:
        _fail(f"Executor: {result.get('error')} ({result.get('latency_s', 0):.1f}s)")
        if result.get("error") == "timeout":
            _warn(f"当前 EXECUTOR_RESPONSE_TIMEOUT_S={EXECUTOR_RESPONSE_TIMEOUT_S}")
        return False

    return True


def test_ros_topics(timeout_s: float = 3.0) -> bool:
    _section("4. ROS 传感器（agent 发 VLM 的前置条件）")
    try:
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image
    except ImportError:
        _warn("未安装 rospy，跳过 ROS 检查")
        return True

    if not rospy.core.is_initialized():
        try:
            rospy.init_node("agent_connectivity_test", anonymous=True, disable_signals=True)
        except rospy.ROSException as e:
            _warn(f"roscore 未运行: {e}")
            return False

    rgb_ok = {"got": False, "n": 0}
    vins_ok = {"got": False, "n": 0}

    def _rgb_cb(_msg: Image) -> None:
        rgb_ok["got"] = True
        rgb_ok["n"] += 1

    def _vins_cb(_msg: Odometry) -> None:
        vins_ok["got"] = True
        vins_ok["n"] += 1

    print(f"  等待 RGB: {RGB_IMAGE_TOPIC}")
    print(f"  等待 VINS: {VINS_TOPIC}")
    rospy.Subscriber(RGB_IMAGE_TOPIC, Image, _rgb_cb, queue_size=1)
    rospy.Subscriber(VINS_TOPIC, Odometry, _vins_cb, queue_size=1)

    t_end = time.time() + timeout_s
    while time.time() < t_end and not rospy.is_shutdown():
        if rgb_ok["got"] and vins_ok["got"]:
            break
        time.sleep(0.05)

    all_ok = True
    if rgb_ok["got"]:
        _ok(f"RGB 有数据 (收到 {rgb_ok['n']} 帧)")
    else:
        _fail(f"RGB 无数据 → agent 不会发 VLM 请求")
        all_ok = False

    if vins_ok["got"]:
        _ok(f"VINS 有数据 (收到 {vins_ok['n']} 帧)")
    else:
        _fail(f"VINS 无数据 → agent 不会发 VLM 请求")
        all_ok = False

    if not all_ok:
        print("\n  仿真需先: sim_fly + xtdrone(read)，并确认:")
        print(f"    rostopic hz {RGB_IMAGE_TOPIC}")
        print(f"    rostopic hz {VINS_TOPIC}")
    return all_ok


async def _main_async(args: argparse.Namespace) -> int:
    print("Agent VLM 连通性诊断")
    print(f"  VLLM_BASE_URL = {VLLM_BASE_URL}")
    print(f"  VLLM_MODEL    = {VLLM_MODEL or '(auto)'}")

    if not test_tcp_port(VLLM_BASE_URL):
        return 1

    models_ok, _ = test_http_models(VLLM_BASE_URL)
    if not models_ok:
        return 1

    model = VLLM_MODEL
    if not model:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=VLLM_BASE_URL.rstrip("/"),
            api_key=VLLM_API_KEY,
            timeout=VLLM_HTTP_TIMEOUT_S,
        )
        try:
            model = await asyncio.wait_for(pick_model_id(client), timeout=15.0)
            _ok(f"自动选择模型: {model}")
        except Exception as e:
            _fail(f"无法列出模型: {e!r}")
            return 1

    vlm_ok = await _async_probe(model=str(model), full=args.full)
    ros_ok = test_ros_topics() if args.ros else True

    _section("总结")
    if vlm_ok and ros_ok:
        print("  全部通过。若 run_agent 仍无回复，看终端 stats 里 plan/exec rps 是否为 0。")
        return 0
    if vlm_ok and not ros_ok:
        print("  VLM 正常，但 ROS 传感器缺失 → agent 不会发请求。")
        return 2
    print("  VLM 不可用，先修 SSH 隧道 / vLLM 服务。")
    return 1


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Agent VLM / ROS 连通性诊断")
    parser.add_argument(
        "--ros",
        action="store_true",
        help="额外检查 RGB/VINS 话题（需 roscore + 仿真/真机已在发图）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="除文本探针外，再测 Planner+Executor 带图请求",
    )
    args = parser.parse_args(argv)
    code = asyncio.run(_main_async(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
