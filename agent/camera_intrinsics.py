#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D 相机内参：仅由 agent.config 在 node 启动时 init_intrinsics 注入。"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


_lock = threading.Lock()
_active: CameraIntrinsics = CameraIntrinsics(0, 0, 1.0, 1.0, 0.0, 0.0)


def set_intrinsics(intr: CameraIntrinsics) -> None:
    global _active
    with _lock:
        _active = intr


def get_intrinsics() -> CameraIntrinsics:
    with _lock:
        return _active


def init_intrinsics(intr: CameraIntrinsics) -> None:
    set_intrinsics(intr)
