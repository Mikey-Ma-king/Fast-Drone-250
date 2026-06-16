#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用本机 vLLM OpenAI 兼容接口。

默认：压测「无人机找床悬停」prompt + **2 张 224×224 RGB 图**（内存生成）下的峰值吞吐。
运行时会先打印一对参考：见常量 REFERENCE_EXAMPLE_USER_TEXT / REFERENCE_EXAMPLE_ASSISTANT_REPLY。

依赖: pip install openai numpy pillow
服务需已启动，例如:
  CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model \
    --port 8000 --host 0.0.0.0 ...
"""

from __future__ import annotations

import asyncio
import base64
import io
import statistics
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装: pip install openai", file=sys.stderr)
    raise

# 若服务端不支持 OpenAI 兼容的 json_object，可改为 False，仅依赖 prompt 约束
USE_JSON_RESPONSE_FORMAT = True


# 模型回复：机体系相对目标的三维坐标 + 期望航向，单行紧凑 JSON（共 4 个键）
DRONE_REPLY_JSON_SPEC = (
    '输出必须是单行紧凑JSON（不要markdown围栏、不要换行、不要前后说明），'
    '且仅含下列4个键，键名字符完全一致：'
    '"x_m","y_m","z_m","yaw_deg"。'
    "x_m/y_m/z_m 为 JSON number，单位米，表示相对「床/悬停目标」的机体系位置；"
    "yaw_deg 为 JSON number，单位度，表示期望机头航向（与飞控约定一致即可）。"
    "不要输出速度或其它键。"
    "示例（仅示意结构，数值须按图与状态重算）："
    '{"x_m":0.5,"y_m":-0.2,"z_m":1.0,"yaw_deg":15.0}'
)


# user 里「文字 + 两图」时，接在任务 prompt 后的固定一句（与 build_messages_drone_two_images 内一致）
USER_TEXT_SUFFIX_TWO_RGB = " 附图2张为前视RGB。"


def build_drone_hover_prompt(
    *,
    vx: float,
    vy: float,
    vz: float,
    yaw_deg: float,
) -> str:
    """
    假设输入为前视 RGB 实时图；任务找床并悬停于床上方。
    告知机体系速度与当前航向；回复格式由 DRONE_REPLY_JSON_SPEC 固定。
    """
    return (
        f"体速m/s vx,vy,vz={vx:.2f},{vy:.2f},{vz:.2f}；当前yaw={yaw_deg:.1f}°。"
        f"任务：找床并悬停床正上方。"
        f"{DRONE_REPLY_JSON_SPEC}"
    )


def array_to_png_data_url(arr: np.ndarray) -> str:
    """uint8 (H,W,3) RGB -> data:image/png;base64,..."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def make_two_arrays(h: int = 224, w: int = 224) -> list[np.ndarray]:
    """两张互不相同的 RGB uint8 图，作前视帧占位（默认 224×224）。"""
    a0 = np.zeros((h, w, 3), dtype=np.uint8)
    a0[:, : w // 3] = [200, 80, 80]
    a0[:, w // 3 : 2 * w // 3] = [80, 180, 80]
    a0[:, 2 * w // 3 :] = [80, 80, 200]
    yy = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    a0[:, :, 0] = np.minimum(a0[:, :, 0].astype(np.int16) + yy // 4, 255).astype(np.uint8)

    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    a1 = np.stack([gx, gy, (gx.astype(np.int16) ^ gy.astype(np.int16)).astype(np.uint8)], axis=-1)

    return [a0, a1]


def build_messages_drone_two_images(
    prompt: str,
    *,
    image_urls: list[str],
) -> list[dict[str, Any]]:
    if len(image_urls) != 2:
        raise ValueError("需要恰好 2 张图")
    text = prompt + USER_TEXT_SUFFIX_TWO_RGB
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# 参考实例：与「发给模型的 user 首段 text」及「期望的 assistant 单行 JSON」对齐
# 真实请求在同一条 user 里另有 2 个 image_url（PNG data URL），此处不展开 base64。
# ---------------------------------------------------------------------------
REFERENCE_EXAMPLE_VX = 0.35
REFERENCE_EXAMPLE_VY = -0.12
REFERENCE_EXAMPLE_VZ = 0.08
REFERENCE_EXAMPLE_YAW_DEG = 42.0

REFERENCE_EXAMPLE_USER_TEXT = (
    build_drone_hover_prompt(
        vx=REFERENCE_EXAMPLE_VX,
        vy=REFERENCE_EXAMPLE_VY,
        vz=REFERENCE_EXAMPLE_VZ,
        yaw_deg=REFERENCE_EXAMPLE_YAW_DEG,
    )
    + USER_TEXT_SUFFIX_TWO_RGB
)

# 示意模型输出：单行 x_m,y_m,z_m,yaw_deg（数值勿照抄，仅作解析参考）
REFERENCE_EXAMPLE_ASSISTANT_REPLY = '{"x_m":0.4,"y_m":-0.15,"z_m":0.9,"yaw_deg":12.5}'


def print_reference_prompt_reply_pair() -> None:
    """打印与代码常量一致的「user 文本段 | assistant 回复」参考（两图略）。"""
    print("\n========== 参考实例：prompt 文本段 | assistant 单行 JSON ==========")
    print("[user → multimodal 中首条 type=text；真实请求在此后还有 2×image_url]\n")
    print(REFERENCE_EXAMPLE_USER_TEXT)
    print("\n[assistant → 期望形态示例]\n")
    print(REFERENCE_EXAMPLE_ASSISTANT_REPLY)
    print("====================================================================\n")


async def pick_model_id(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("/v1/models 未返回任何模型，请检查 vLLM 是否正常启动")
    return models.data[0].id


async def one_request(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    seq: int,
) -> dict[str, Any]:
    t_send = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if USE_JSON_RESPONSE_FORMAT:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": False,
            "error": repr(e),
            "latency_s": t_recv - t_send,
            "t_send": t_send,
            "t_recv": t_recv,
        }
    t_recv = time.perf_counter()
    text = resp.choices[0].message.content
    return {
        "seq": seq,
        "ok": True,
        "latency_s": t_recv - t_send,
        "t_send": t_send,
        "t_recv": t_recv,
        "text": text,
    }


async def bench_at_concurrency(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    concurrency: int,
    num_requests: int,
) -> tuple[list[dict[str, Any]], float]:
    """最多同时 concurrency 条在飞，共 num_requests 条。"""
    sem = asyncio.Semaphore(concurrency)

    async def run_one(seq: int) -> dict[str, Any]:
        async with sem:
            return await one_request(client, model, messages, max_tokens, seq)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(i) for i in range(num_requests)])
    wall = time.perf_counter() - t0
    return list(results), wall


async def async_main() -> None:
    BASE_URL = "http://127.0.0.1:8000/v1"
    API_KEY = "EMPTY"
    MODEL = None
    MAX_TOKENS = 96  # 单行 JSON：x_m,y_m,z_m,yaw_deg
    HTTP_TIMEOUT_S = 300.0

    # 与飞控状态同步替换即可（示例值）
    VX, VY, VZ = 0.35, -0.12, 0.08
    YAW_DEG = 42.0
    IMG_H, IMG_W = 224, 224
    arrays = make_two_arrays(h=IMG_H, w=IMG_W)
    image_urls = [array_to_png_data_url(a) for a in arrays]

    prompt = build_drone_hover_prompt(vx=VX, vy=VY, vz=VZ, yaw_deg=YAW_DEG)
    messages = build_messages_drone_two_images(prompt, image_urls=image_urls)

    print_reference_prompt_reply_pair()

    client = AsyncOpenAI(
        base_url=BASE_URL.rstrip("/"),
        api_key=API_KEY,
        timeout=HTTP_TIMEOUT_S,
    )
    model = MODEL or await pick_model_id(client)

    concurrencies = [2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    err_stop_ratio = 0.08

    print("=== 无人机 prompt + 2×224×224 RGB 图 峰值吞吐 ===")
    print(f"model: {model}")
    print(f"图: {len(image_urls)} 张 {IMG_H}×{IMG_W} PNG(data URL)；prompt 长度: {len(prompt)} 字；max_tokens={MAX_TOKENS}")
    print(f"并发档: {concurrencies}")
    print("--- 各档结果（成功 RPS = 成功数 / 墙钟；并发=同时最多在飞请求数）---")

    best_rps = -1.0
    best_c = 0
    best_row: dict[str, Any] = {}

    for C in concurrencies:
        num_requests = min(160, max(48, C * 8))
        results, wall = await bench_at_concurrency(
            client=client,
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            concurrency=C,
            num_requests=num_requests,
        )
        oks = [r for r in results if r.get("ok")]
        errs = [r for r in results if not r.get("ok")]
        n_ok, n_err = len(oks), len(errs)
        rps = n_ok / wall if wall > 0 else 0.0
        err_ratio = n_err / len(results) if results else 0.0
        mean_lat = statistics.mean(float(r["latency_s"]) for r in oks) if oks else float("nan")

        print(
            f"  C={C:2d}  n={num_requests:3d}  成功={n_ok:3d}  失败={n_err:3d}  "
            f"墙钟={wall:6.2f}s  成功RPS={rps:6.3f}  平均RTT={mean_lat * 1000:8.1f}ms  错误率={err_ratio * 100:5.2f}%"
        )

        if rps > best_rps and n_ok > 0:
            best_rps = rps
            best_c = C
            best_row = {
                "C": C,
                "n": num_requests,
                "wall": wall,
                "n_ok": n_ok,
                "rps": rps,
                "mean_lat": mean_lat,
                "err_ratio": err_ratio,
            }

        if err_ratio > err_stop_ratio:
            print(f"  （错误率 > {err_stop_ratio * 100:.0f}%，停止继续加压）")
            break

    print("--- 结论 ---")
    if best_rps < 0:
        print("无有效成功请求，请检查 vLLM 是否启动、显存/并发配置是否足够。")
    else:
        print(
            f"本次探测到的最高成功吞吐: {best_rps:.3f} req/s（并发档 C={best_c}，"
            f"该档 {best_row.get('n_ok', 0)} 条成功 / {best_row.get('wall', 0):.2f}s 墙钟）"
        )
        print(
            "说明：RPS 随 vLLM 参数（如 max_num_seqs）、GPU、解码长度与本机网络而变；"
            "更高并发若错误率上升，通常表示服务端排队/拒收/超时。"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
