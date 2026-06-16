#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 请求与图像工具。"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from openai import AsyncOpenAI
from PIL import Image as PILImage

from agent.config import USE_JSON_RESPONSE_FORMAT, VLM_IMAGE_UINT8, VLM_TEMPERATURE


def load_prompt_from_txt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"prompt 文件不存在: {path.resolve()}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt 文件为空: {path.resolve()}")
    return text


def _prepare_rgb_for_vlm(arr: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(arr)
    if VLM_IMAGE_UINT8 and rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def array_to_png_data_url(arr: np.ndarray) -> str:
    rgb = _prepare_rgb_for_vlm(arr)
    buf = io.BytesIO()
    if rgb.dtype == np.uint8:
        pil = PILImage.fromarray(rgb, mode="RGB")
    else:
        pil = PILImage.fromarray(rgb)
    pil.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def resize_rgb_for_vlm(arr: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, dict[str, int]]:
    """长边缩放到 max_long_edge，保持宽高比（不拉成正方形）。

    返回 (rgb, meta)：vlm_rw/vlm_rh 为送入 VLM 的图像尺寸；src_w/src_h 为原图。
    VLM 输出的 0~1000 坐标相对 vlm_rw×vlm_rh（与原图等比，可直接映射回 src）。
    """
    rgb = _prepare_rgb_for_vlm(arr)
    h, w = rgb.shape[:2]
    scale = float(max_long_edge) / max(w, h, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if rgb.dtype == np.uint8:
        pil = PILImage.fromarray(rgb, mode="RGB")
        out_dtype = np.uint8 if VLM_IMAGE_UINT8 else rgb.dtype
    else:
        pil = PILImage.fromarray(rgb)
        out_dtype = rgb.dtype
    resized = pil.resize((new_w, new_h), PILImage.BILINEAR)
    meta = {
        "vlm_rw": new_w,
        "vlm_rh": new_h,
        "src_w": w,
        "src_h": h,
    }
    return np.asarray(resized, dtype=out_dtype), meta


def resize_rgb_uint8(arr: np.ndarray, size: int) -> np.ndarray:
    """仅返回正方形 RGB（Planner/3D 不需 letterbox meta 时用）。"""
    rgb, _ = resize_rgb_for_vlm(arr, size)
    return rgb


def resize_rgb_square_stretch(
    arr: np.ndarray, size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """已废弃：请用 resize_rgb_for_vlm（长边缩放）。保留别名避免旧脚本 import 报错。"""
    return resize_rgb_for_vlm(arr, size)


def vlm_exact_pixels(size: int) -> int:
    """正方形附图时的像素数（仅当宽高相等时使用）。"""
    return int(size) * int(size)


def exact_pixels_from_shape(height: int, width: int) -> int:
    """Qwen-VL mm_processor_kwargs：min/max_pixels = 附图宽×高。"""
    return int(height) * int(width)


def _image_url_block(image_url: str, exact_pixels: int) -> dict[str, Any]:
    """Match example_executor&planner.py call_vlm image_url structure."""
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
            "extra_fields": {
                "mm_processor_kwargs": {
                    "min_pixels": exact_pixels,
                    "max_pixels": exact_pixels,
                }
            },
        },
    }


def build_messages_single_image(
    prompt: str,
    image_url: str,
    *,
    exact_pixels: int,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                _image_url_block(image_url, exact_pixels),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_messages_multi_image(
    prompt: str,
    image_urls: list[str],
    *,
    exact_pixels: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        _image_url_block(url, exact_pixels) for url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


async def pick_model_id(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("/v1/models 未返回任何模型，请检查 vLLM 是否已启动")
    return models.data[0].id


async def async_vlm_request(
    client: AsyncOpenAI,
    model: str,
    seq: int,
    messages: list[dict[str, Any]],
    user_prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    t_send = time.perf_counter()
    meta = dict(extra or {})
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": VLM_TEMPERATURE,
        }
        if USE_JSON_RESPONSE_FORMAT:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
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
            "latency_s": t_recv - t_send,
            **meta,
        }
    except asyncio.TimeoutError:
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": False,
            "error": "timeout",
            "user_prompt": user_prompt,
            "latency_s": t_recv - t_send,
            **meta,
        }
    except Exception as e:
        t_recv = time.perf_counter()
        return {
            "seq": seq,
            "ok": False,
            "error": repr(e),
            "user_prompt": user_prompt,
            "latency_s": t_recv - t_send,
            **meta,
        }
