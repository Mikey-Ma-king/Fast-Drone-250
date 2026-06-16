#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D Executor：move 子任务 VLM 仅输出像素点；distance/yaw/z 本地偏心计算。"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Optional

import numpy as np
import rospy
from openai import AsyncOpenAI

from agent.camera_geom import waypoint_to_body_delta
from agent.attitude import quat_from_vins_snap
from agent.command import (
    body_delta_to_world_from_snap,
    clamp_body_delta,
    clamp_world_z,
    publish_command_pos,
)
from agent.config import (
    EXECUTOR_2D_BBOX,
    EXECUTOR_2D_DEBUG_PRINT,
    EXECUTOR_MAX_TOKENS,
    EXECUTOR_RESPONSE_TIMEOUT_S,
    IMG_MAX_SIZE,
)
from agent.motion_parse import extract_json
from agent.parse_2d import parse_executor_2d_output
from agent.prompts import build_executor_2d_prompt
from agent.task_state import SubTaskKind, TaskState
from agent.vlm_utils import (
    array_to_png_data_url,
    async_vlm_request,
    build_messages_single_image,
    exact_pixels_from_shape,
    resize_rgb_for_vlm,
)


def _print_executor_2d_debug(
    *,
    seq: int,
    parsed_2d: dict[str, Any],
    body: dict[str, float],
    parsed: dict[str, float],
    vins_snap: dict[str, float],
    wx: float,
    wy: float,
    wz: float,
    target_yaw: float,
) -> None:
    """EXECUTOR_2D_DEBUG_PRINT=True 时输出 bearing / 几何 / 指令中间量。"""
    ref_pitch = body.get("ref_pitch_deg")
    ref_pitch_s = (
        f"{ref_pitch:+.1f}°" if ref_pitch is not None else "?"
    )
    print(
        f"[executor_2d] seq={seq} norm=({parsed_2d['center_raw'][0]:.1f},"
        f"{parsed_2d['center_raw'][1]:.1f}) cam=({body.get('u_camera', 0):.0f},"
        f"{body.get('v_camera', 0):.0f}) FLIP_LR={body.get('_flip_lr', '?')}\n"
        f"  bearing body wp : {body.get('b_wp_body_str', body.get('b_wp_body'))}\n"
        f"  bearing world wp: {body.get('b_wp_world_str', body.get('b_wp_world'))}\n"
        f"  bearing world ref: {body.get('b_ref_world_str', body.get('b_ref_world'))}\n"
        f"  ref_pitch={ref_pitch_s}\n"
        f"  geom: ang3d={math.degrees(body.get('angle_3d_rad', 0)):.1f}° "
        f"dyaw={math.degrees(body.get('yaw_diff_rad', 0)):.1f}° "
        f"dpitch={math.degrees(body.get('pitch_diff_rad', 0)):.1f}° "
        f"dist={body.get('distance_m', 0):.3f}m\n"
        f"  cmd body: x={parsed['x_m']:.3f} y={parsed['y_m']:.3f} "
        f"z_world={parsed.get('z_m_world', parsed['z_m']):.3f} "
        f"yaw_deg={parsed['yaw_deg']:+.2f}\n"
        f"  world horiz: dx={body.get('dx_world', 0):.3f} "
        f"dy={body.get('dy_world', 0):.3f}\n"
        f"  vins@req: x={vins_snap['x']:.3f} y={vins_snap['y']:.3f} "
        f"z={vins_snap['z']:.3f} yaw={math.degrees(vins_snap['yaw_rad']):.1f}°\n"
        f"  cmd world: x={wx:.3f} y={wy:.3f} z={wz:.3f} "
        f"target_yaw={math.degrees(target_yaw):.1f}°",
        flush=True,
    )


class Executor2D:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        task_state: TaskState,
        cmd_pub: rospy.Publisher,
        vins_snapshot_fn: Callable[[], dict[str, float]],
        on_command_published: Optional[Callable[[], None]] = None,
        with_bbox: bool = EXECUTOR_2D_BBOX,
    ) -> None:
        self.client = client
        self.with_bbox = with_bbox
        self.model = model
        self.task_state = task_state
        self.cmd_pub = cmd_pub
        self.vins_snapshot_fn = vins_snapshot_fn
        self.on_command_published = on_command_published
        self.request_seq = 0
        self.latest_responded_seq = -1
        self._lock = threading.Lock()
        self._last_pub_prompt = ""
        self._last_pub_reply = ""
        self._last_command_display = ""
        self._last_waypoint_vlm: Optional[tuple[float, float]] = None
        self._last_waypoint_hw: Optional[tuple[int, int]] = None
        self._last_vlm_image_meta: Optional[dict[str, int]] = None
        self._last_detection_bbox_orig: Optional[tuple[int, int, int, int]] = None

    def build_request(
        self,
        rgb_image: np.ndarray,
    ) -> Optional[tuple[int, list[dict[str, Any]], str, dict[str, float], int]]:
        if self.task_state.is_finished():
            return None
        ctx = self.task_state.get_executor_context()
        if ctx is None:
            return None

        subtask_text, kind, _, list_version = ctx
        if kind != "move":
            return None

        src_h, src_w = rgb_image.shape[:2]
        rgb_current, img_meta = resize_rgb_for_vlm(rgb_image.copy(), IMG_MAX_SIZE)
        rw = int(img_meta["vlm_rw"])
        rh = int(img_meta["vlm_rh"])

        vins_at_request = dict(self.vins_snapshot_fn())
        vins_at_request["_image_hw"] = [rh, rw]
        vins_at_request["_orig_wh"] = [int(src_w), int(src_h)]
        vins_at_request["_vlm_image_meta"] = img_meta

        user_prompt = build_executor_2d_prompt(subtask_text, with_bbox=self.with_bbox)
        exact_pixels = exact_pixels_from_shape(rh, rw)
        messages = build_messages_single_image(
            user_prompt,
            array_to_png_data_url(rgb_current),
            exact_pixels=exact_pixels,
        )
        seq = self.request_seq
        self.request_seq += 1
        return seq, messages, user_prompt, vins_at_request, list_version

    async def run_request(
        self,
        seq: int,
        messages: list[dict[str, Any]],
        user_prompt: str,
        vins_at_request: dict[str, float],
        list_version: int,
    ) -> dict[str, Any]:
        rh, rw = vins_at_request.get("_image_hw", [IMG_MAX_SIZE, IMG_MAX_SIZE])
        return await async_vlm_request(
            self.client,
            self.model,
            seq,
            messages,
            user_prompt,
            max_tokens=EXECUTOR_MAX_TOKENS,
            timeout_s=EXECUTOR_RESPONSE_TIMEOUT_S,
            extra={
                "vins_at_request": vins_at_request,
                "list_version": list_version,
                "role": "executor",
                "executor_mode": "2d",
                "image_hw": [int(rh), int(rw)],
                "orig_wh": vins_at_request.get("_orig_wh"),
                "vlm_image_meta": vins_at_request.get("_vlm_image_meta"),
            },
        )

    def handle_response(self, result: dict[str, Any]) -> None:
        seq = result.get("seq", -1)
        with self._lock:
            if seq < self.latest_responded_seq:
                return
            self.latest_responded_seq = max(self.latest_responded_seq, seq)

        if self.task_state.is_finished():
            return

        list_version_at_send = int(result.get("list_version", -1))
        if list_version_at_send != self.task_state.get_list_version():
            rospy.logdebug("[executor_2d] seq=%d list updated, drop stale reply", seq)
            return

        if not result.get("ok"):
            err = result.get("error", "")
            if err == "timeout":
                rospy.logwarn("[executor_2d] seq=%d timeout, drop", seq)
            else:
                rospy.logerr("[executor_2d] seq=%d failed: %s", seq, err)
            return

        vins_snap = result.get("vins_at_request")
        if not vins_snap:
            rospy.logerr("[executor_2d] seq=%d missing vins_at_request", seq)
            return

        image_hw = result.get("image_hw")
        if not image_hw or len(image_hw) != 2:
            rh = rw = IMG_MAX_SIZE
        else:
            rh, rw = int(image_hw[0]), int(image_hw[1])

        orig_wh = result.get("orig_wh") or vins_snap.get("_orig_wh")
        if orig_wh and len(orig_wh) == 2:
            ow, oh = int(orig_wh[0]), int(orig_wh[1])
        else:
            meta = result.get("vlm_image_meta") or vins_snap.get("_vlm_image_meta")
            ow = int(meta.get("src_w", rw)) if isinstance(meta, dict) else rw
            oh = int(meta.get("src_h", rh)) if isinstance(meta, dict) else rh

        image_meta = result.get("vlm_image_meta") or vins_snap.get("_vlm_image_meta")
        if not isinstance(image_meta, dict):
            image_meta = None

        raw_text = result.get("text", "")
        try:
            parsed_2d = parse_executor_2d_output(
                extract_json(raw_text),
                raw_text,
                rw=rw,
                rh=rh,
                ow=ow,
                oh=oh,
                with_bbox=self.with_bbox,
            )
            quat = quat_from_vins_snap(vins_snap)
            body = waypoint_to_body_delta(
                float(parsed_2d["center_raw"][0]),
                float(parsed_2d["center_raw"][1]),
                quat=quat,
                rw=rw,
                rh=rh,
                image_meta=image_meta,
            )
            parsed = clamp_body_delta(body)
            wx, wy, wz, target_yaw = body_delta_to_world_from_snap(
                parsed["x_m"],
                parsed["y_m"],
                parsed["z_m"],
                parsed["yaw_deg"],
                vins_snap,
                z_m_is_world_vertical=True,
            )
            wz = clamp_world_z(wz)
            if EXECUTOR_2D_DEBUG_PRINT:
                _print_executor_2d_debug(
                    seq=seq,
                    parsed_2d=parsed_2d,
                    body=body,
                    parsed=parsed,
                    vins_snap=vins_snap,
                    wx=wx,
                    wy=wy,
                    wz=wz,
                    target_yaw=target_yaw,
                )
        except Exception as e:
            rospy.logerr(
                "[executor_2d] seq=%d parse/transform failed: %s | %s",
                seq, e, raw_text,
            )
            return

        self._last_pub_prompt = result.get("user_prompt", "")
        self._last_pub_reply = raw_text
        self._last_waypoint_vlm = (float(parsed_2d["u_vlm"]), float(parsed_2d["v_vlm"]))
        self._last_waypoint_hw = (int(rh), int(rw))
        self._last_vlm_image_meta = image_meta
        self._last_detection_bbox_orig = (
            parsed_2d.get("bbox_orig") if self.with_bbox else None
        )
        self._last_command_display = (
            f"cam=({body.get('u_camera', 0):.0f},{body.get('v_camera', 0):.0f}) "
            f"ang3d={math.degrees(body.get('angle_3d_rad', 0)):.1f}° "
            f"dyaw={math.degrees(body.get('yaw_diff_rad', 0)):.1f}° "
            f"dpitch={math.degrees(body.get('pitch_diff_rad', 0)):.1f}° "
            f"dist={body.get('distance_m', 0):.2f}m → "
            f"x={parsed['x_m']:.2f} y={parsed['y_m']:.2f} "
            f"z={parsed['z_m']:.2f} yaw={parsed['yaw_deg']:.1f}"
        )
        publish_command_pos(
            self.cmd_pub,
            wx=wx,
            wy=wy,
            wz=wz,
            target_yaw=target_yaw,
            vins_snap=vins_snap,
        )
        if self.on_command_published is not None:
            self.on_command_published()

    @property
    def last_pub_prompt(self) -> str:
        return self._last_pub_prompt

    @property
    def last_pub_reply(self) -> str:
        return self._last_pub_reply

    @property
    def last_command_display(self) -> str:
        return self._last_command_display

    @property
    def last_waypoint_vlm(self) -> Optional[tuple[float, float]]:
        return self._last_waypoint_vlm

    @property
    def last_waypoint_hw(self) -> Optional[tuple[int, int]]:
        return self._last_waypoint_hw

    @property
    def last_vlm_image_meta(self) -> Optional[dict[str, int]]:
        return self._last_vlm_image_meta

    @property
    def last_detection_bbox_orig(self) -> Optional[tuple[int, int, int, int]]:
        return self._last_detection_bbox_orig

    @property
    def send_count(self) -> int:
        return self.request_seq
