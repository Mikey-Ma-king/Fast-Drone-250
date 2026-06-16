#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VINS 四元数：机体系 ↔ 世界系向量旋转。"""

from __future__ import annotations

import math
from typing import Mapping, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (qx, qy, qz, qw)


def quat_normalize(qx: float, qy: float, qz: float, qw: float) -> Quat:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def quat_from_vins_snap(vins_snap: Mapping[str, float]) -> Quat:
    return quat_normalize(
        float(vins_snap["qx"]),
        float(vins_snap["qy"]),
        float(vins_snap["qz"]),
        float(vins_snap["qw"]),
    )


def rotate_body_to_world(
    xb: float,
    yb: float,
    zb: float,
    quat: Quat,
) -> Vec3:
    """v_world = R_bw @ v_body，quat 为 VINS 机体相对世界的姿态 (x,y,z,w)。"""
    qx, qy, qz, qw = quat_normalize(*quat)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xw = (1.0 - 2.0 * (yy + zz)) * xb + 2.0 * (xy - wz) * yb + 2.0 * (xz + wy) * zb
    yw = 2.0 * (xy + wz) * xb + (1.0 - 2.0 * (xx + zz)) * yb + 2.0 * (yz - wx) * zb
    zw = 2.0 * (xz - wy) * xb + 2.0 * (yz + wx) * yb + (1.0 - 2.0 * (xx + yy)) * zb
    return xw, yw, zw


def rotate_world_to_body(
    xw: float,
    yw: float,
    zw: float,
    quat: Quat,
) -> Vec3:
    """v_body = R_wb @ v_world，R_wb = R_bw^T；与 rotate_body_to_world 互逆。"""
    qx, qy, qz, qw = quat_normalize(*quat)
    return rotate_body_to_world(xw, yw, zw, (-qx, -qy, -qz, qw))
