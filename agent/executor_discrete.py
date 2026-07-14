#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离散动作 Executor：VLM 返回 FRONT/BACK/UP/DOWN/TURN_LEFT/TURN_RIGHT。"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np
import rospy
from openai import AsyncOpenAI

from agent.command import (
    body_delta_to_world_from_snap,
    clamp_body_delta,
    get_command_target_z,
    nudge_command_target_z,
    publish_command_pos,
)
from agent.config import (
    DISCRETE_STEP_DOWN_M,
    DISCRETE_STEP_UP_M,
    EXECUTOR_MAX_TOKENS,
    EXECUTOR_REASONING,
    EXECUTOR_RESPONSE_TIMEOUT_S,
    IMG_MAX_SIZE,
)
from agent.motion_parse import extract_json
from agent.parse_discrete import (
    discrete_action_to_body_delta,
    parse_executor_discrete_output,
)
from agent.prompts import build_executor_discrete_prompt
from agent.task_state import SUBTASK_KINDS_EXECUTOR_DISCRETE, TaskState
from agent.vlm_utils import (
    array_to_png_data_url,
    async_vlm_request,
    build_messages_single_image,
    exact_pixels_from_shape,
    resize_rgb_for_vlm,
)


class ExecutorDiscrete:
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
        self._last_action: Optional[str] = None

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
        if kind not in SUBTASK_KINDS_EXECUTOR_DISCRETE:
            return None

        rgb_current, img_meta = resize_rgb_for_vlm(rgb_image.copy(), IMG_MAX_SIZE)
        vins_at_request = self.vins_snapshot_fn()
        user_prompt = build_executor_discrete_prompt(
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
        max_tokens = 128 if self.with_reasoning else EXECUTOR_MAX_TOKENS
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
                "executor_mode": "discrete",
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
            rospy.logdebug("[executor_discrete] seq=%d list updated, drop stale reply", seq)
            return

        if not result.get("ok"):
            err = result.get("error", "")
            if err == "timeout":
                rospy.logwarn("[executor_discrete] seq=%d timeout, drop", seq)
            else:
                rospy.logerr("[executor_discrete] seq=%d failed: %s", seq, err)
            return

        vins_snap = result.get("vins_at_request")
        if not vins_snap:
            rospy.logerr("[executor_discrete] seq=%d missing vins_at_request", seq)
            return

        raw_text = result.get("text", "")
        try:
            parsed_discrete = parse_executor_discrete_output(
                extract_json(raw_text),
                raw_text,
                with_reasoning=self.with_reasoning,
            )
            action = str(parsed_discrete["action"])
            body = discrete_action_to_body_delta(action)
            parsed = clamp_body_delta(body)
            vins_z = float(vins_snap["z"])
            wx, wy, _, target_yaw = body_delta_to_world_from_snap(
                parsed["x_m"],
                parsed["y_m"],
                0.0,
                parsed["yaw_deg"],
                vins_snap,
            )
            if action == "UP":
                wz = nudge_command_target_z(float(DISCRETE_STEP_UP_M), vins_z)
            elif action == "DOWN":
                wz = nudge_command_target_z(-float(DISCRETE_STEP_DOWN_M), vins_z)
            else:
                wz = get_command_target_z(vins_z)
        except Exception as e:
            rospy.logerr(
                "[executor_discrete] seq=%d parse/transform failed: %s | %s",
                seq, e, raw_text,
            )
            return

        self._last_pub_prompt = result.get("user_prompt", "")
        self._last_pub_reply = raw_text
        self._last_action = action
        self._last_command_display = (
            f"{action} → "
            f"x={parsed['x_m']:.2f} y={parsed['y_m']:.2f} "
            f"z_hold={wz:.2f} yaw={parsed['yaw_deg']:+.1f}°"
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
    def last_action(self) -> Optional[str]:
        return self._last_action

    @property
    def last_waypoint_vlm(self) -> Optional[tuple[float, float]]:
        return None

    @property
    def last_waypoint_hw(self) -> Optional[tuple[float, float]]:
        return None

    @property
    def send_count(self) -> int:
        return self.request_seq
