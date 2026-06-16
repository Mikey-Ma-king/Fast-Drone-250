#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上层 VLM Planner：维护子任务 list（英文 prompt + 子任务起点/当前双图）。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import rospy
from openai import AsyncOpenAI

from agent.config import (
    IMG_MAX_SIZE,
    PLANNER_MAX_TOKENS,
    PLANNER_MAX_TOKENS_REASONING,
    PLANNER_REASONING,
    PLANNER_RESPONSE_TIMEOUT_S,
    SUBTASK_TIMEOUT_FALLBACK_DESC,
)
from agent.motion_parse import (
    extract_json,
    map_planner_to_list_action,
    parse_planner_vlm_output,
)
from agent.prompts import build_planner_prompt
from agent.task_state import TaskState
from agent.vlm_utils import (
    array_to_png_data_url,
    async_vlm_request,
    build_messages_multi_image,
    resize_rgb_for_vlm,
    exact_pixels_from_shape,
)


class Planner:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        task_state: TaskState,
        on_finish: Callable[[], None],
        with_reasoning: bool = PLANNER_REASONING,
    ) -> None:
        self.client = client
        self.model = model
        self.task_state = task_state
        self.on_finish = on_finish
        self.with_reasoning = with_reasoning
        self.request_seq = 0
        self.latest_responded_seq = -1
        self._last_reply = ""
        self._force_switch_at_send = False
        self._planner_image_count_at_send = 2

    def build_request(
        self,
        rgb_image: np.ndarray,
    ) -> tuple[int, list[dict[str, Any]], str, bool]:
        rgb_current, meta_cur = resize_rgb_for_vlm(rgb_image.copy(), IMG_MAX_SIZE)
        force_switch = self.task_state.force_switch()
        snapshot = self.task_state.get_current_subtask_rgb_snapshot()
        if snapshot is not None:
            rgb_start, _ = resize_rgb_for_vlm(snapshot, IMG_MAX_SIZE)
        else:
            rgb_start = rgb_current
        user_prompt = build_planner_prompt(
            self.task_state,
            force_switch=force_switch,
        )
        exact_pixels = exact_pixels_from_shape(
            int(meta_cur["vlm_rh"]), int(meta_cur["vlm_rw"]),
        )
        image_urls = [
            array_to_png_data_url(rgb_start),
            array_to_png_data_url(rgb_current),
        ]
        messages = build_messages_multi_image(
            user_prompt, image_urls, exact_pixels=exact_pixels
        )
        seq = self.request_seq
        self.request_seq += 1
        self._force_switch_at_send = force_switch
        self._planner_image_count_at_send = 2
        return seq, messages, user_prompt, force_switch

    async def run_request(
        self,
        seq: int,
        messages: list[dict[str, Any]],
        user_prompt: str,
        force_switch: bool,
    ) -> dict[str, Any]:
        max_tokens = (
            PLANNER_MAX_TOKENS_REASONING if self.with_reasoning else PLANNER_MAX_TOKENS
        )
        return await async_vlm_request(
            self.client,
            self.model,
            seq,
            messages,
            user_prompt,
            max_tokens=max_tokens,
            timeout_s=PLANNER_RESPONSE_TIMEOUT_S,
            extra={
                "force_switch": force_switch,
                "role": "planner",
                "planner_reasoning": self.with_reasoning,
                "planner_image_count": self._planner_image_count_at_send,
            },
        )

    def handle_response(
        self,
        result: dict[str, Any],
        rgb_image: np.ndarray,
    ) -> None:
        seq = result.get("seq", -1)
        if seq < self.latest_responded_seq:
            return
        self.latest_responded_seq = max(self.latest_responded_seq, seq)

        if not result.get("ok"):
            err = result.get("error", "")
            if err == "timeout":
                rospy.logwarn("[planner] seq=%d timeout, drop", seq)
            else:
                rospy.logerr("[planner] seq=%d failed: %s", seq, err)
            return

        force_switch = bool(result.get("force_switch", self._force_switch_at_send))
        raw_text = result.get("text", "")
        try:
            vlm_parsed = parse_planner_vlm_output(
                extract_json(raw_text),
                raw_text,
                with_reasoning=self.with_reasoning,
            )
            parsed = map_planner_to_list_action(
                vlm_parsed,
                current_subtask=self.task_state.current_subtask_text(),
                has_subtask=self.task_state.has_subtask(),
                force_switch=force_switch,
            )
        except Exception as e:
            rospy.logerr("[planner] seq=%d parse failed: %s | %s", seq, e, raw_text)
            if force_switch:
                self._force_append(rgb_image, SUBTASK_TIMEOUT_FALLBACK_DESC)
            return

        self._last_reply = raw_text

        if parsed.get("f") == 1:
            rospy.loginfo("[planner] mission_complete -> finish")
            self.task_state.mark_finished()
            self.on_finish()
            return

        if parsed.get("u") == 1:
            desc = parsed["d"]
            kind = parsed.get("kind", "move")
            ver = self.task_state.append_subtask(desc, rgb_image, kind=kind)
            rospy.loginfo("[planner] append subtask v%d [%s]: %s", ver, kind, desc)
            if self.with_reasoning and vlm_parsed.get("reasoning"):
                rospy.loginfo("[planner] reasoning: %s", vlm_parsed["reasoning"])
            return

        if force_switch and parsed.get("f") != 1 and parsed.get("u") != 2:
            desc = (vlm_parsed.get("subtask") or "").strip() or SUBTASK_TIMEOUT_FALLBACK_DESC
            rospy.logwarn_throttle(
                1.0,
                "[planner] force_switch: planner returned u=0, fallback append: %s",
                desc,
            )
            self.task_state.append_subtask(desc, rgb_image, kind=parsed.get("kind", "move"))
            return

        rospy.logdebug_throttle(2.0, "[planner] u=0, subtask list unchanged")

    def _force_append(self, rgb_image: np.ndarray, desc: str) -> None:
        self.task_state.append_subtask(desc, rgb_image, kind="move")
        rospy.logwarn("[planner] parse failed + timeout, fallback append: %s", desc)

    @property
    def last_reply(self) -> str:
        return self._last_reply
