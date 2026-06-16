#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D 航点控制（六步管线，见 waypoint_to_body_delta）。

1. 归一化像素 (0~1000) → 相机实际像素
2. 内参 → 机体系 bearing 单位向量
3. 请求时刻 VINS 四元数 → 世界系 bearing
4. 世界系对比：标准=机头水平分量；俯仰基准=Z_2D_REF_PITCH_DEG（度）
5. 仅世界系 bearing：水平位移、yaw、高度；机体系 x_m,y_m 仅作 command 接口（R_wb 水平分量）
6. command.publish_command_pos + cmd_velocity 发布
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

from agent.attitude import Quat, rotate_body_to_world, rotate_world_to_body
from agent.camera_intrinsics import CameraIntrinsics, get_intrinsics
from agent import config as agent_config
from agent.config import (
    BEARING_DEV_MAX_RAD,
    EXECUTOR_2D_MAX_DISTANCE_M,
    NORM_SCALE_2D,
    YAW_2D_GAIN_DEG,
    Z_2D_GAIN_M,
    Z_2D_REF_PITCH_DEG,
)
# BEARING_BODY_FLIP_LR 在 bearing_body_from_pixel 内经 agent_config 读取（改 config 后需重启 node）

Vec3 = Tuple[float, float, float]


def format_bearing_vec(v: Vec3) -> str:
    """格式化 bearing 三元组便于打印。"""
    return f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})"


# ---------------------------------------------------------------------------
# 1. 归一化像素 → 相机实际像素
# ---------------------------------------------------------------------------

def norm1000_to_camera_px(
    x_norm: float,
    y_norm: float,
    intr: CameraIntrinsics,
) -> tuple[float, float]:
    """VLM 0~1000 → 相机原图 (u, v)，u 右 v 下。"""
    uw = max(int(intr.width) - 1, 1)
    uh = max(int(intr.height) - 1, 1)
    u = float(x_norm) * uw / float(NORM_SCALE_2D)
    v = float(y_norm) * uh / float(NORM_SCALE_2D)
    return (
        max(0.0, min(float(uw), u)),
        max(0.0, min(float(uh), v)),
    )


def reference_norm1000_center() -> tuple[float, float]:
    """标准方向水平中心：归一化图像 (500, 500)。"""
    half = float(NORM_SCALE_2D) * 0.5
    return half, half


def reference_camera_pixel(intr: CameraIntrinsics) -> tuple[float, float]:
    """画面中心像素（仅日志/可视化）。"""
    return norm1000_to_camera_px(*reference_norm1000_center(), intr)


def reference_pitch_rad(
    ref_pitch_deg: Optional[float] = None,
) -> float:
    """标准俯仰（弧度），来自 Z_2D_REF_PITCH_DEG：0=水平，负值=向下。"""
    deg = (
        float(ref_pitch_deg)
        if ref_pitch_deg is not None
        else float(agent_config.Z_2D_REF_PITCH_DEG)
    )
    deg = max(-90.0, min(0.0, deg))
    return math.radians(deg)


def reference_bearing_world(quat: Quat) -> Vec3:
    """世界系标准 bearing：请求时刻机头前向在水平面的分量（z=0，单位向量）。"""
    fx, fy, fz = rotate_body_to_world(1.0, 0.0, 0.0, quat)
    h = math.hypot(float(fx), float(fy))
    if h < 1e-9:
        return 1.0, 0.0, 0.0
    return fx / h, fy / h, 0.0


# ---------------------------------------------------------------------------
# 2. 相机像素 → 机体系 bearing
# ---------------------------------------------------------------------------

def bearing_camera_from_pixel(
    u: float,
    v: float,
    intr: CameraIntrinsics,
) -> Vec3:
    """针孔反投影，相机光学系单位向量 (x 右, y 下, z 前)。"""
    xc = (float(u) - intr.cx) / max(intr.fx, 1e-6)
    yc = (float(v) - intr.cy) / max(intr.fy, 1e-6)
    zc = 1.0
    n = math.hypot(xc, yc, zc)
    if n < 1e-9:
        return 0.0, 0.0, 1.0
    return xc / n, yc / n, zc / n


def bearing_body_from_pixel(
    u: float,
    v: float,
    intr: CameraIntrinsics,
) -> Vec3:
    """机体系 bearing：+X 前、+Y 左、+Z 上（相机 z 沿机体前方）。"""
    xc, yc, zc = bearing_camera_from_pixel(u, v, intr)
    lat_sign = 1.0 if agent_config.BEARING_BODY_FLIP_LR else -1.0
    xb, yb, zb = zc, lat_sign * xc, -yc
    n = math.hypot(xb, yb, zb)
    if n < 1e-9:
        return 1.0, 0.0, 0.0
    return xb / n, yb / n, zb / n


# ---------------------------------------------------------------------------
# 3. 机体系 bearing → 世界系（请求时刻 VINS 四元数）
# ---------------------------------------------------------------------------

def bearing_world_from_body(
    xb: float,
    yb: float,
    zb: float,
    *,
    quat: Quat,
) -> Vec3:
    xw, yw, zw = rotate_body_to_world(xb, yb, zb, quat)
    n = math.hypot(xw, yw, zw)
    if n < 1e-9:
        return xw, yw, zw
    return xw / n, yw / n, zw / n


# ---------------------------------------------------------------------------
# 4. 标准方向对比 → 位移 / yaw / z
# ---------------------------------------------------------------------------

def bearing_angle_rad(a: Vec3, b: Vec3) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.acos(max(-1.0, min(1.0, dot)))


def world_yaw_rad(xw: float, yw: float) -> float:
    """世界系水平航向（ENU：绕 +Z，从 +X 朝 +Y 为正，弧度）。"""
    return math.atan2(float(yw), float(xw))


def world_pitch_rad(xw: float, yw: float, zw: float) -> float:
    """世界系俯仰：相对水平面抬高为正（弧度）。"""
    return math.atan2(float(zw), math.hypot(float(xw), float(yw)))


def world_horizontal_unit(xw: float, yw: float) -> Tuple[float, float]:
    """bearing 在世界水平面 (X-Y) 的单位方向。"""
    h = math.hypot(float(xw), float(yw))
    if h < 1e-9:
        return 1.0, 0.0
    return xw / h, yw / h


def distance_m_from_angle(
    angle_rad: float,
    *,
    max_distance_m: float = EXECUTOR_2D_MAX_DISTANCE_M,
    dev_max_rad: float = BEARING_DEV_MAX_RAD,
) -> float:
    if dev_max_rad <= 1e-9:
        return 0.0
    t = min(1.0, max(0.0, float(angle_rad)) / float(dev_max_rad))
    return max(0.0, float(max_distance_m) * (1.0 - t))


def analyze_waypoint_bearings(
    x_norm: float,
    y_norm: float,
    *,
    quat: Quat,
    intr: Optional[CameraIntrinsics] = None,
) -> dict[str, float]:
    """完整 1~4 步；控制量仅来自世界系 bearing（机体系 bearing 仅用于旋到世界）。"""
    intr = intr or get_intrinsics()

    u_wp, v_wp = norm1000_to_camera_px(x_norm, y_norm, intr)
    u_ref, v_ref = reference_camera_pixel(intr)

    b_wp_b = bearing_body_from_pixel(u_wp, v_wp, intr)
    b_wp_w = bearing_world_from_body(*b_wp_b, quat=quat)
    b_ref_w = reference_bearing_world(quat)
    pitch_ref = reference_pitch_rad()

    angle_3d = bearing_angle_rad(b_wp_w, b_ref_w)
    distance_m = distance_m_from_angle(angle_3d)

    yaw_wp = world_yaw_rad(b_wp_w[0], b_wp_w[1])
    yaw_ref = world_yaw_rad(b_ref_w[0], b_ref_w[1])
    pitch_wp = world_pitch_rad(*b_wp_w)
    yaw_diff = yaw_wp - yaw_ref
    pitch_diff = pitch_wp - pitch_ref

    hx_w, hy_w = world_horizontal_unit(b_wp_w[0], b_wp_w[1])
    yaw_deg = float(YAW_2D_GAIN_DEG) * math.degrees(yaw_diff)
    z_m_world = float(Z_2D_GAIN_M) * pitch_diff

    return {
        "u_camera": u_wp,
        "v_camera": v_wp,
        "u_ref": u_ref,
        "v_ref": v_ref,
        "b_wp_body": b_wp_b,
        "b_wp_world": b_wp_w,
        "b_ref_world": b_ref_w,
        "ref_pitch_rad": pitch_ref,
        "ref_pitch_deg": math.degrees(pitch_ref),
        "angle_3d_rad": angle_3d,
        "yaw_diff_rad": yaw_diff,
        "pitch_diff_rad": pitch_diff,
        "distance_m": distance_m,
        "yaw_deg": yaw_deg,
        "z_m_world": z_m_world,
        "horiz_dir_x_w": hx_w,
        "horiz_dir_y_w": hy_w,
    }


def world_delta_to_body_delta(
    dx_w: float,
    dy_w: float,
    dz_w: float,
    *,
    quat: Quat,
) -> Tuple[float, float, float]:
    """世界系位移增量 → 机体系（供 command.body_delta_to_world 使用）。"""
    return rotate_world_to_body(dx_w, dy_w, dz_w, quat)


def waypoint_to_body_delta(
    x_norm: float,
    y_norm: float,
    *,
    quat: Quat,
    rw: int = 0,
    rh: int = 0,
    image_meta: Optional[Mapping[str, int]] = None,
) -> dict[str, float]:
    """由归一化航点 (0~1000) 得指令；几何仅用世界系 bearing + 发包时刻 quat。

    x_m,y_m：世界水平位移 (dx_w,dy_w) 映到机体系（供 command）；z_m 为世界竖直增量。
    """
    _ = rw, rh, image_meta
    a = analyze_waypoint_bearings(x_norm, y_norm, quat=quat)
    distance_m = float(a["distance_m"])
    z_m_world = float(a["z_m_world"])

    hx_w = float(a["horiz_dir_x_w"])
    hy_w = float(a["horiz_dir_y_w"])
    if distance_m > 0.0:
        dx_w = hx_w * distance_m
        dy_w = hy_w * distance_m
    else:
        dx_w = dy_w = 0.0
    x_m, y_m, _ = world_delta_to_body_delta(dx_w, dy_w, 0.0, quat=quat)
    z_m = z_m_world

    b_wp_b = a["b_wp_body"]
    b_wp_w = a["b_wp_world"]
    b_ref_w = a["b_ref_world"]
    return {
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "z_m_world": z_m_world,
        "dx_world": dx_w,
        "dy_world": dy_w,
        "yaw_deg": float(a["yaw_deg"]),
        "distance_m": distance_m,
        "angle_3d_rad": float(a["angle_3d_rad"]),
        "yaw_diff_rad": float(a["yaw_diff_rad"]),
        "pitch_diff_rad": float(a["pitch_diff_rad"]),
        "u_camera": float(a["u_camera"]),
        "v_camera": float(a["v_camera"]),
        "b_wp_body": b_wp_b,
        "b_wp_world": b_wp_w,
        "b_ref_world": b_ref_w,
        "b_wp_body_str": format_bearing_vec(b_wp_b),
        "b_wp_world_str": format_bearing_vec(b_wp_w),
        "b_ref_world_str": format_bearing_vec(b_ref_w),
        "ref_pitch_rad": float(a["ref_pitch_rad"]),
        "ref_pitch_deg": float(a["ref_pitch_deg"]),
        "_flip_lr": bool(agent_config.BEARING_BODY_FLIP_LR),
    }
