#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rotate_scan / stop：不经执行器 VLM，直接发布机体系指令。"""

from __future__ import annotations

from typing import Callable, Optional

import rospy

from agent.command import (
    body_delta_to_world_from_snap,
    clamp_body_delta,
    get_command_target_z,
    publish_command_pos,
)
from agent.config import SCAN_YAW_DELTA_DEG
from agent.task_state import SubTaskKind


class DirectControlState:
    """与 Executor2D 相同的展示字段，供面板/统计使用。"""

    def __init__(self) -> None:
        self.send_count = 0
        self._last_command_display = ""

    @property
    def last_pub_prompt(self) -> str:
        return "(direct, no VLM)"

    @property
    def last_pub_reply(self) -> str:
        return ""

    @property
    def last_command_display(self) -> str:
        return self._last_command_display

    @property
    def last_waypoint_vlm(self):
        return None

    @property
    def last_waypoint_hw(self):
        return None

    @property
    def last_vlm_image_meta(self):
        return None

    @property
    def last_detection_bbox_orig(self):
        return None


_direct_state = DirectControlState()


def get_direct_control_state() -> DirectControlState:
    return _direct_state


def publish_direct_subtask(
    kind: SubTaskKind,
    *,
    cmd_pub: rospy.Publisher,
    vins_snapshot_fn: Callable[[], dict[str, float]],
    on_command_published: Optional[Callable[[], None]] = None,
) -> bool:
    """原地扫描或悬停停止。返回是否已发布。"""
    if kind not in ("rotate_scan", "stop"):
        return False

    vins = vins_snapshot_fn()
    if kind == "stop":
        body = {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "yaw_deg": 0.0}
        label = "stop hold"
    else:
        body = {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": -float(SCAN_YAW_DELTA_DEG),
        }
        label = f"scan yaw={body['yaw_deg']:+.1f}°"

    parsed = clamp_body_delta(body)
    wx, wy, _, target_yaw = body_delta_to_world_from_snap(
        parsed["x_m"],
        parsed["y_m"],
        parsed["z_m"],
        parsed["yaw_deg"],
        vins,
    )
    wz = get_command_target_z(float(vins["z"]))
    publish_command_pos(
        cmd_pub,
        wx=wx,
        wy=wy,
        wz=wz,
        target_yaw=target_yaw,
        vins_snap=vins,
    )
    _direct_state.send_count += 1
    _direct_state._last_command_display = (
        f"[{kind}] {label} → "
        f"x={parsed['x_m']:.2f} y={parsed['y_m']:.2f} "
        f"z={parsed['z_m']:.2f} yaw={parsed['yaw_deg']:.1f}"
    )
    if on_command_published is not None:
        on_command_published()
    return True
