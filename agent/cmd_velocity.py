#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""维护 /command_pos 目标点自身的运动速度（对连续目标位姿求导 + 可选平滑）。"""

from __future__ import annotations

import math
import time
from typing import Optional

from agent.config import (
    CMD_V_EMA_ALPHA,
    CMD_V_MAX_DT_S,
    CMD_V_MAX_XY_MPS,
    CMD_V_MAX_Z_MPS,
    CMD_V_MIN_STEP_M,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_velocity(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    v_xy = math.hypot(vx, vy)
    if v_xy > CMD_V_MAX_XY_MPS > 0.0:
        s = CMD_V_MAX_XY_MPS / v_xy
        vx *= s
        vy *= s
    vz = _clamp(vz, -CMD_V_MAX_Z_MPS, CMD_V_MAX_Z_MPS)
    return vx, vy, vz


class TargetCommandVelocityTracker:
    """跟踪相邻两次发布的目标世界坐标，估计目标点线速度。"""

    __slots__ = ("_last_t", "_last_wx", "_last_wy", "_last_wz", "_vx", "_vy", "_vz")

    def __init__(self) -> None:
        self._last_wx: Optional[float] = None
        self._last_wy: Optional[float] = None
        self._last_wz: Optional[float] = None
        self._last_t: Optional[float] = None
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0

    def reset(self) -> None:
        self._last_wx = self._last_wy = self._last_wz = self._last_t = None
        self._vx = self._vy = self._vz = 0.0

    def observe(
        self,
        wx: float,
        wy: float,
        wz: float,
        *,
        t_s: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """新目标点 (wx,wy,wz)；返回该目标轨迹的估计线速度（世界系）。"""
        t = float(t_s if t_s is not None else time.perf_counter())
        wx, wy, wz = float(wx), float(wy), float(wz)

        if self._last_t is None:
            self._last_wx, self._last_wy, self._last_wz, self._last_t = wx, wy, wz, t
            self._vx = self._vy = self._vz = 0.0
            return 0.0, 0.0, 0.0

        dt = t - self._last_t
        if dt <= 1e-6:
            return self._vx, self._vy, self._vz

        if dt > CMD_V_MAX_DT_S:
            self._last_wx, self._last_wy, self._last_wz, self._last_t = wx, wy, wz, t
            self._vx = self._vy = self._vz = 0.0
            return 0.0, 0.0, 0.0

        dx = wx - self._last_wx
        dy = wy - self._last_wy
        dz = wz - self._last_wz
        step = math.sqrt(dx * dx + dy * dy + dz * dz)

        if step < CMD_V_MIN_STEP_M:
            rx = ry = rz = 0.0
        else:
            inv_dt = 1.0 / dt
            rx = dx * inv_dt
            ry = dy * inv_dt
            rz = dz * inv_dt
            rx, ry, rz = _clamp_velocity(rx, ry, rz)

        a = CMD_V_EMA_ALPHA
        if a >= 1.0:
            self._vx, self._vy, self._vz = rx, ry, rz
        elif a <= 0.0:
            pass
        else:
            self._vx = a * rx + (1.0 - a) * self._vx
            self._vy = a * ry + (1.0 - a) * self._vy
            self._vz = a * rz + (1.0 - a) * self._vz
            self._vx, self._vy, self._vz = _clamp_velocity(
                self._vx, self._vy, self._vz,
            )

        self._last_wx, self._last_wy, self._last_wz, self._last_t = wx, wy, wz, t
        return self._vx, self._vy, self._vz


# 全进程单例：与 publish_command_pos 发布节奏一致
_target_velocity_tracker = TargetCommandVelocityTracker()


def get_target_velocity_tracker() -> TargetCommandVelocityTracker:
    return _target_velocity_tracker


def observe_target_command_velocity(
    wx: float,
    wy: float,
    wz: float,
    *,
    t_s: Optional[float] = None,
) -> tuple[float, float, float]:
    return _target_velocity_tracker.observe(wx, wy, wz, t_s=t_s)


def reset_target_command_velocity() -> None:
    _target_velocity_tracker.reset()
