#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预览面板：左 RGB（2D 可标 waypoint），右侧仅当前 subtask + 指令。"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Mapping, Optional, TYPE_CHECKING, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.config import (
    PANEL_FONT_SIZE,
    PANEL_LEFT_PAD,
    PANEL_LINE_H,
    PANEL_TEXT_MIN_WIDTH,
    PANEL_TEXT_WIDTH_RATIO,
)

if TYPE_CHECKING:
    from agent.executor import Executor
    from agent.executor_2d import Executor2D
    from agent.task_state import TaskState

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

_OVERLAY_FONT: Optional[ImageFont.FreeTypeFont | ImageFont.ImageFont] = None
_PAD = 12
_TEXT_BG = (24, 24, 24)
_LEFT_BG = (16, 16, 16)
_WAYPOINT_COLOR = (0, 255, 0)
_WAYPOINT_RADIUS = 8
_DETECT_COLOR = (0, 165, 255)  # BGR 橙色
_DETECT_THICKNESS = 2


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    global _OVERLAY_FONT
    if _OVERLAY_FONT is not None:
        return _OVERLAY_FONT
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            _OVERLAY_FONT = ImageFont.truetype(path, PANEL_FONT_SIZE)
            return _OVERLAY_FONT
    _OVERLAY_FONT = ImageFont.load_default()
    return _OVERLAY_FONT


def _wrap_line(text: str, font: ImageFont.ImageFont, max_px: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    buf = ""
    for ch in text:
        trial = buf + ch
        w = font.getbbox(trial)[2] - font.getbbox(trial)[0]
        if w <= max_px or not buf:
            buf = trial
        else:
            lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines or [""]


def _build_panel_lines(
    *,
    task_state: Optional["TaskState"],
    executor: Optional[Union["Executor", "Executor2D"]],
) -> list[str]:
    lines = ["--- subtask ---"]
    if task_state is None or not task_state.has_subtask():
        lines.append("(none)")
    else:
        kind = task_state.current_subtask_kind()
        text = task_state.current_subtask_text().strip() or "(empty)"
        if kind:
            lines.append(f"[{kind}] {text}")
        else:
            lines.append(text)

    lines.append("--- command ---")
    cmd = ""
    if executor and executor.last_command_display:
        cmd = executor.last_command_display.strip()
    if not cmd:
        try:
            from agent.direct_control import get_direct_control_state

            direct = get_direct_control_state()
            if direct.last_command_display:
                cmd = direct.last_command_display.strip()
        except ImportError:
            pass
    lines.append(cmd or "(none)")

    if task_state is not None and task_state.is_finished():
        lines.append("---")
        lines.append("FINISHED")
    return lines


def _wrap_all_lines(raw_lines: list[str], font: ImageFont.ImageFont, max_text_w: int) -> list[str]:
    wrapped: list[str] = []
    for line in raw_lines:
        wrapped.extend(_wrap_line(line, font, max_text_w))
    return wrapped


def _render_text_panel_bgr(width: int, height: int, wrapped_lines: list[str]) -> np.ndarray:
    font = _load_font()
    rgb = np.full((height, width, 3), _TEXT_BG, dtype=np.uint8)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    y = _PAD
    for line in wrapped_lines:
        if y + PANEL_LINE_H > height - _PAD:
            break
        draw.text((_PAD, y), line, font=font, fill=(240, 240, 240))
        y += PANEL_LINE_H

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _vlm_px_to_image_px(
    u_vlm: float,
    v_vlm: float,
    img_w: int,
    img_h: int,
    vlm_rw: int,
    vlm_rh: int,
    *,
    image_meta: Optional[Mapping[str, int]] = None,
) -> tuple[int, int]:
    """VLM 附图坐标 → 当前显示原图 (img_w×img_h) 像素。"""
    if image_meta and "lb_ox" in image_meta:
        ox = int(image_meta["lb_ox"])
        oy = int(image_meta["lb_oy"])
        cw = max(1, int(image_meta["lb_w"]))
        ch = max(1, int(image_meta["lb_h"]))
        u = (float(u_vlm) - ox) / max(cw - 1, 1) * max(img_w - 1, 1)
        v = (float(v_vlm) - oy) / max(ch - 1, 1) * max(img_h - 1, 1)
    else:
        u = float(u_vlm) / max(vlm_rw - 1, 1) * max(img_w - 1, 1)
        v = float(v_vlm) / max(vlm_rh - 1, 1) * max(img_h - 1, 1)
    ui = int(round(max(0.0, min(img_w - 1, u))))
    vi = int(round(max(0.0, min(img_h - 1, v))))
    return ui, vi


def _draw_waypoint_on_bgr(
    bgr: np.ndarray,
    u_vlm: float,
    v_vlm: float,
    vlm_rw: int,
    vlm_rh: int,
    *,
    image_meta: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    u, v = _vlm_px_to_image_px(
        u_vlm, v_vlm, w, h, vlm_rw, vlm_rh, image_meta=image_meta
    )
    cv2.circle(out, (u, v), _WAYPOINT_RADIUS, _WAYPOINT_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (u, v),
        _WAYPOINT_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    label = f"({u},{v})"
    cv2.putText(
        out,
        label,
        (min(u + 10, max(0, w - 80)), max(16, v - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _WAYPOINT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_detection_bbox_on_bgr(
    bgr: np.ndarray,
    bbox_orig: tuple[int, int, int, int],
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = bbox_orig
    x1i = int(round(max(0.0, min(w - 1, float(x1)))))
    y1i = int(round(max(0.0, min(h - 1, float(y1)))))
    x2i = int(round(max(0.0, min(w - 1, float(x2)))))
    y2i = int(round(max(0.0, min(h - 1, float(y2)))))
    if x1i > x2i:
        x1i, x2i = x2i, x1i
    if y1i > y2i:
        y1i, y2i = y2i, y1i
    cv2.rectangle(
        out, (x1i, y1i), (x2i, y2i),
        _DETECT_COLOR, _DETECT_THICKNESS, lineType=cv2.LINE_AA,
    )
    label = "object"
    cv2.putText(
        out,
        label,
        (x1i, max(14, y1i - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _DETECT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out


def _build_padded_image_column(bgr: np.ndarray, col_w: int, col_h: int) -> np.ndarray:
    """左栏：灰底 + 居中放置带 padding 的原图。"""
    pad = PANEL_LEFT_PAD
    h, w = bgr.shape[:2]
    canvas = np.full((col_h, col_w, 3), _LEFT_BG, dtype=np.uint8)

    inner_w = w + 2 * pad
    inner_h = h + 2 * pad
    x0 = max(0, (col_w - inner_w) // 2)
    y0 = max(0, (col_h - inner_h) // 2)

    y1 = min(col_h, y0 + inner_h)
    x1 = min(col_w, x0 + inner_w)
    region_h = y1 - y0
    region_w = x1 - x0

    if region_h <= 2 * pad or region_w <= 2 * pad:
        return canvas

    avail_h = region_h - 2 * pad
    avail_w = region_w - 2 * pad
    if h <= avail_h and w <= avail_w:
        img_y = y0 + pad + (avail_h - h) // 2
        img_x = x0 + pad + (avail_w - w) // 2
        canvas[img_y : img_y + h, img_x : img_x + w] = bgr
    else:
        scale = min(avail_w / float(w), avail_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_y = y0 + pad + (avail_h - new_h) // 2
        img_x = x0 + pad + (avail_w - new_w) // 2
        canvas[img_y : img_y + new_h, img_x : img_x + new_w] = resized

    return canvas


def render_agent_panel_bgr(
    bgr: np.ndarray,
    *,
    task_state: Optional["TaskState"],
    executor: Optional[Union["Executor", "Executor2D"]],
) -> np.ndarray:
    """左栏 RGB；右栏 subtask + 指令。"""
    h, w = bgr.shape[:2]
    font = _load_font()

    display_bgr = bgr
    if executor is not None:
        wp = executor.last_waypoint_vlm
        hw = executor.last_waypoint_hw
        vlm_meta = getattr(executor, "last_vlm_image_meta", None)
        if wp is not None and hw is not None:
            vlm_rh, vlm_rw = int(hw[0]), int(hw[1])
            display_bgr = _draw_waypoint_on_bgr(
                bgr,
                wp[0],
                wp[1],
                vlm_rw,
                vlm_rh,
                image_meta=vlm_meta,
            )
        bbox = getattr(executor, "last_detection_bbox_orig", None)
        if bbox is not None:
            display_bgr = _draw_detection_bbox_on_bgr(display_bgr, bbox)

    text_w = max(PANEL_TEXT_MIN_WIDTH, int(w * PANEL_TEXT_WIDTH_RATIO))
    max_text_w = max(80, text_w - 2 * _PAD)

    raw_lines = _build_panel_lines(task_state=task_state, executor=executor)
    wrapped = _wrap_all_lines(raw_lines, font, max_text_w)

    lh, lw = display_bgr.shape[:2]
    left_col_w = max(w + 2 * PANEL_LEFT_PAD + 2 * _PAD, lw + 2 * PANEL_LEFT_PAD)
    left_col_h = lh + 2 * PANEL_LEFT_PAD + 2 * _PAD
    canvas_h = left_col_h

    left_col = _build_padded_image_column(display_bgr, left_col_w, canvas_h)
    text_col = _render_text_panel_bgr(text_w, canvas_h, wrapped)
    return np.hstack([left_col, text_col])


def make_video_save_path(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return save_dir / f"agent_panel_{stamp}.mp4"


class PanelVideoWriter:
    """写入左图+右文合成帧。"""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: tuple[int, int] = (0, 0)

    def _fit_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            return frame_bgr
        tw, th = self._size
        if w == tw and h == th:
            return frame_bgr
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        copy_w = min(w, tw)
        copy_h = min(h, th)
        out[:copy_h, :copy_w] = frame_bgr[:copy_h, :copy_w]
        return out

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self._size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"无法创建视频文件: {self.path}")
        frame_bgr = self._fit_frame(frame_bgr)
        self._writer.write(frame_bgr)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
