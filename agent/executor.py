#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下层 VLM Executor：~5Hz，英文 compact JSON，仅当前 RGB。"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np
import rospy
from openai import AsyncOpenAI

from agent.command import (
    body_delta_to_world_from_snap,
    clamp_body_delta,
    clamp_world_z,
    publish_command_pos,
)
from agent.config import (
    EXECUTOR_MAX_TOKENS,
    EXECUTOR_REASONING,
    EXECUTOR_RESPONSE_TIMEOUT_S,
    IMG_MAX_SIZE,
)
from agent.motion_parse import (
    executor_parsed_to_body_delta,
    extract_json,
    parse_executor_output,
)
from agent.prompts import build_executor_prompt
from agent.task_state import TaskState
from agent.vlm_utils import (
    array_to_png_data_url,
    async_vlm_request,
    build_messages_single_image,
    resize_rgb_for_vlm,
    exact_pixels_from_shape,
)


class Executor:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        task_state: TaskState,
        cmd_pub: rospy.Publisher,
        vins_snapshot_fn: Callable[[], dict[str, float]],
        on_command_published: Optional[Callable[[], None]] = None,
        with_reasoning: bool = EXECUTOR_REASONING,
    ) -> None:
        self.client = client
        self.model = model
        self.task_state = task_state
        self.cmd_pub = cmd_pub
        self.vins_snapshot_fn = vins_snapshot_fn
        self.on_command_published = on_command_published
        self.with_reasoning = with_reasoning
        self.request_seq = 0
        self.latest_responded_seq = -1
        self._lock = threading.Lock()
        self._last_pub_prompt = ""
        self._last_pub_reply = ""
        self._last_command_display = ""

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
        rgb_current, img_meta = resize_rgb_for_vlm(rgb_image.copy(), IMG_MAX_SIZE)
        vins_at_request = self.vins_snapshot_fn()
        user_prompt = build_executor_prompt(
            subtask_text,
            with_reasoning=self.with_reasoning,
        )
        exact_pixels = exact_pixels_from_shape(
            int(img_meta["vlm_rh"]), int(img_meta["vlm_rw"]),
        )
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
        max_tokens = 256 if self.with_reasoning else EXECUTOR_MAX_TOKENS
        return await async_vlm_request(
            self.client,
            self.model,
            seq,
            messages,
            user_prompt,
            max_tokens=max_tokens,
            timeout_s=EXECUTOR_RESPONSE_TIMEOUT_S,
            extra={
                "vins_at_request": vins_at_request,
                "list_version": list_version,
                "role": "executor",
                "executor_reasoning": self.with_reasoning,
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
            rospy.logdebug("[executor] seq=%d list updated, drop stale reply", seq)
            return

        if not result.get("ok"):
            err = result.get("error", "")
            if err == "timeout":
                rospy.logwarn("[executor] seq=%d timeout, drop", seq)
            else:
                rospy.logerr("[executor] seq=%d failed: %s", seq, err)
            return

        vins_snap = result.get("vins_at_request")
        if not vins_snap:
            rospy.logerr("[executor] seq=%d missing vins_at_request", seq)
            return

        raw_text = result.get("text", "")
        try:
            motion = parse_executor_output(
                extract_json(raw_text),
                raw_text,
                with_reasoning=self.with_reasoning,
            )
            parsed = clamp_body_delta(executor_parsed_to_body_delta(motion))
            wx, wy, wz, target_yaw = body_delta_to_world_from_snap(
                parsed["x_m"],
                parsed["y_m"],
                parsed["z_m"],
                parsed["yaw_deg"],
                vins_snap,
            )
            wz = clamp_world_z(wz)
        except Exception as e:
            rospy.logerr("[executor] seq=%d parse/transform failed: %s | %s", seq, e, raw_text)
            return

        self._last_pub_prompt = result.get("user_prompt", "")
        self._last_pub_reply = raw_text
        compact = motion.get("compact") or {}
        parts = []
        for key in ("fwd", "lat", "vert", "yaw"):
            val = compact.get(key)
            if val:
                parts.append(f"{key}={val}")
        if parts:
            self._last_command_display = " ".join(parts)
        else:
            self._last_command_display = (
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
        return None

    @property
    def last_waypoint_hw(self) -> Optional[tuple[int, int]]:
        return None

    @property
    def send_count(self) -> int:
        return self.request_seq
