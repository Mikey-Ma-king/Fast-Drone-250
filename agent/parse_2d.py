#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D：VLM 坐标一律为 0~1000 归一化（相对送入模型的图像宽高）。"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.config import NORM_SCALE_2D


def norm1000_to_vlm_px(
    x: float, y: float, rw: int, rh: int,
) -> tuple[float, float]:
    """0~1000 → VLM 附图上的像素坐标。"""
    return (
        max(0.0, min(float(rw - 1), float(x) * rw / NORM_SCALE_2D)),
        max(0.0, min(float(rh - 1), float(y) * rh / NORM_SCALE_2D)),
    )


def norm1000_to_original_px(
    x: float, y: float, ow: int, oh: int,
) -> tuple[float, float]:
    """0~1000 → 相机原图分辨率像素（长边等比缩放时与附图线性一致）。"""
    return (
        max(0.0, min(float(ow - 1), float(x) * ow / NORM_SCALE_2D)),
        max(0.0, min(float(oh - 1), float(y) * oh / NORM_SCALE_2D)),
    )


# 兼容旧调用名
def map_to_vlm_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return norm1000_to_vlm_px(x, y, width, height)


def map_vlm_px_to_original(
    u_vlm: float, v_vlm: float, rw: int, rh: int, ow: int, oh: int,
) -> tuple[float, float]:
    """VLM 图像素 → 原图（经 rw/rh 与 ow/oh 比例）。"""
    ox = u_vlm / max(rw, 1) * ow
    oy = v_vlm / max(rh, 1) * oh
    return (
        max(0.0, min(float(ow - 1), ox)),
        max(0.0, min(float(oh - 1), oy)),
    )


def _extract_xy(data: Any, raw_text: str) -> tuple[Optional[float], Optional[float]]:
    x = y = None
    root = data if isinstance(data, dict) else None
    if root is not None:
        if "x" in root:
            x = float(root["x"])
        if "y" in root:
            y = float(root["y"])
        if x is None and "waypoint_2d" in root:
            w = root["waypoint_2d"]
            if isinstance(w, (list, tuple)) and len(w) >= 2:
                x, y = float(w[0]), float(w[1])
    if x is None:
        m = re.search(r'"x"\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        if m:
            x = float(m.group(1))
    if y is None:
        m = re.search(r'"y"\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        if m:
            y = float(m.group(1))
    if x is None or y is None:
        for px, py in re.findall(
            r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", raw_text,
        ):
            x, y = float(px), float(py)
            break
    return x, y


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


def _extract_bbox(
    data: Any, raw_text: str,
) -> Optional[tuple[float, float, float, float]]:
    root = data if isinstance(data, dict) else None
    if root is not None:
        box = _parse_bbox_values(root.get("bbox", root.get("box")))
        if box is not None:
            return box
    m = re.search(
        r'"bbox"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*'
        r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
        raw_text,
    )
    if m:
        return (
            float(m.group(1)), float(m.group(2)),
            float(m.group(3)), float(m.group(4)),
        )
    return None


def parse_executor_2d_output(
    data: Any,
    raw_text: str,
    *,
    rw: int,
    rh: int,
    ow: int,
    oh: int,
    with_bbox: bool = False,
) -> dict[str, Any]:
    x, y = _extract_xy(data, raw_text)
    if x is None or y is None:
        raise ValueError('no x/y in reply (expected {"x":int,"y":int})')
    center_raw = [x, y]
    u_vlm, v_vlm = norm1000_to_vlm_px(x, y, rw, rh)
    u_orig, v_orig = norm1000_to_original_px(x, y, ow, oh)
    out: dict[str, Any] = {
        "center_raw": center_raw,
        "u_vlm": u_vlm,
        "v_vlm": v_vlm,
        "waypoint_resized_px": [u_vlm, v_vlm],
        "waypoint_original_px": [u_orig, v_orig],
        "coord_norm1000": True,
    }
    if with_bbox:
        box = _extract_bbox(data, raw_text)
        if box is not None:
            out["bbox_raw"] = list(box)
            out["bbox_orig"] = norm1000_bbox_to_original(*box, ow=ow, oh=oh)
    return out


def norm1000_bbox_to_original(
    x1: float, y1: float, x2: float, y2: float,
    *, ow: int, oh: int,
) -> tuple[int, int, int, int]:
    ox1, oy1 = norm1000_to_original_px(x1, y1, ow, oh)
    ox2, oy2 = norm1000_to_original_px(x2, y2, ow, oh)
    return (
        int(round(min(ox1, ox2))),
        int(round(min(oy1, oy2))),
        int(round(max(ox1, ox2))),
        int(round(max(oy1, oy2))),
    )
