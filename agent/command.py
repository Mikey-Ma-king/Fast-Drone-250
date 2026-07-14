#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机体系指令解析、限幅与 /command_pos 发布。"""

from __future__ import annotations

import json
import math
import re
import threading
import rospy
from nav_msgs.msg import Odometry

from agent.attitude import rotate_body_to_world
from agent.cmd_velocity import observe_target_command_velocity
from agent.config import BODY_DELTA_MAX_M, CMD_Z_MAX_M, CMD_Z_MIN_M, ENABLE_COMMAND_POS_VELOCITY, YAW_DELTA_MAX_DEG
from agent.obstacle_avoidance import adjust_command_pos_for_obstacles

_altitude_lock = threading.Lock()
_command_target_wz: float | None = None


def parse_drone_reply(text: str) -> dict[str, float]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return {
        "x_m": float(data["x_m"]),
        "y_m": float(data["y_m"]),
        "z_m": float(data["z_m"]),
        "yaw_deg": float(data["yaw_deg"]),
    }


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_yaw_rad(yaw_rad: float) -> float:
    return (yaw_rad + math.pi) % (2.0 * math.pi) - math.pi


def body_position_delta_to_world(
    x_m: float,
    y_m: float,
    z_m: float,
    *,
    vins_x: float,
    vins_y: float,
    vins_z: float,
    vins_qx: float,
    vins_qy: float,
    vins_qz: float,
    vins_qw: float,
) -> tuple[float, float, float]:
    """机体系位置增量 → 世界系；用 VINS 完整四元数 R_bw。"""
    dx, dy, dz = rotate_body_to_world(
        x_m, y_m, z_m, (vins_qx, vins_qy, vins_qz, vins_qw),
    )
    return vins_x + dx, vins_y + dy, vins_z + dz


def body_delta_to_world(
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_deg: float,
    *,
    vins_x: float,
    vins_y: float,
    vins_z: float,
    vins_qx: float,
    vins_qy: float,
    vins_qz: float,
    vins_qw: float,
    vins_yaw_rad: float,
) -> tuple[float, float, float, float]:
    """机体系 → 世界系；位置用四元数，target_yaw = vins_yaw + yaw_deg 增量。"""
    wx, wy, wz = body_position_delta_to_world(
        x_m,
        y_m,
        z_m,
        vins_x=vins_x,
        vins_y=vins_y,
        vins_z=vins_z,
        vins_qx=vins_qx,
        vins_qy=vins_qy,
        vins_qz=vins_qz,
        vins_qw=vins_qw,
    )
    target_yaw = wrap_yaw_rad(vins_yaw_rad + math.radians(yaw_deg))
    return wx, wy, wz, target_yaw


def body_delta_to_world_from_snap(
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_deg: float,
    vins_snap: dict[str, float],
    *,
    z_m_is_world_vertical: bool = False,
) -> tuple[float, float, float, float]:
    """由 vins_snapshot 字典计算机体系 → 世界系。

    z_m_is_world_vertical=True（2D）：x_m,y_m 为机体系水平增量；z_m 为世界系竖直增量（米）。
    """
    if z_m_is_world_vertical:
        dx, dy, _ = rotate_body_to_world(
            x_m,
            y_m,
            0.0,
            (
                float(vins_snap["qx"]),
                float(vins_snap["qy"]),
                float(vins_snap["qz"]),
                float(vins_snap["qw"]),
            ),
        )
        wx = float(vins_snap["x"]) + dx
        wy = float(vins_snap["y"]) + dy
        wz = float(vins_snap["z"]) + z_m
        target_yaw = wrap_yaw_rad(
            float(vins_snap["yaw_rad"]) + math.radians(yaw_deg),
        )
        return wx, wy, wz, target_yaw

    return body_delta_to_world(
        x_m,
        y_m,
        z_m,
        yaw_deg,
        vins_x=float(vins_snap["x"]),
        vins_y=float(vins_snap["y"]),
        vins_z=float(vins_snap["z"]),
        vins_qx=float(vins_snap["qx"]),
        vins_qy=float(vins_snap["qy"]),
        vins_qz=float(vins_snap["qz"]),
        vins_qw=float(vins_snap["qw"]),
        vins_yaw_rad=float(vins_snap["yaw_rad"]),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_body_delta(parsed: dict[str, float]) -> dict[str, float]:
    raw = parsed
    clamped = {
        "x_m": _clamp(raw["x_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "y_m": _clamp(raw["y_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "z_m": _clamp(raw["z_m"], -BODY_DELTA_MAX_M, BODY_DELTA_MAX_M),
        "yaw_deg": _clamp(raw["yaw_deg"], -YAW_DELTA_MAX_DEG, YAW_DELTA_MAX_DEG),
    }
    if (
        clamped["x_m"] != raw["x_m"]
        or clamped["y_m"] != raw["y_m"]
        or clamped["z_m"] != raw["z_m"]
        or clamped["yaw_deg"] != raw["yaw_deg"]
    ):
        rospy.logwarn_throttle(
            1.0,
            "body delta clamped: (%.3f,%.3f,%.3f,%.1f)->(%.3f,%.3f,%.3f,%.1f)",
            raw["x_m"],
            raw["y_m"],
            raw["z_m"],
            raw["yaw_deg"],
            clamped["x_m"],
            clamped["y_m"],
            clamped["z_m"],
            clamped["yaw_deg"],
        )
    return clamped


def clamp_world_z(wz: float) -> float:
    wz_clamped = _clamp(wz, CMD_Z_MIN_M, CMD_Z_MAX_M)
    if wz_clamped != wz:
        rospy.logwarn_throttle(1.0, "world z clamped: %.3f->%.3f", wz, wz_clamped)
    return wz_clamped


def reset_command_target_z(vins_z: float | None = None) -> None:
    """重置指令高度；None 表示下次发布时再从 VINS 初始化。"""
    global _command_target_wz
    with _altitude_lock:
        _command_target_wz = float(vins_z) if vins_z is not None else None


def get_command_target_z(vins_z: float) -> float:
    """读取维护的世界系目标高度（首次从 VINS 初始化）。"""
    global _command_target_wz
    with _altitude_lock:
        if _command_target_wz is None:
            _command_target_wz = float(vins_z)
        return clamp_world_z(_command_target_wz)


def nudge_command_target_z(delta_m: float, vins_z: float) -> float:
    """UP/DOWN：在维护高度上增减，避免每帧相对 VINS 叠加导致晃动。"""
    global _command_target_wz
    with _altitude_lock:
        if _command_target_wz is None:
            _command_target_wz = float(vins_z)
        _command_target_wz = clamp_world_z(_command_target_wz + float(delta_m))
        return _command_target_wz


def publish_command_pos(
    pub: rospy.Publisher,
    *,
    wx: float,
    wy: float,
    wz: float,
    target_yaw: float,
    vins_snap: dict[str, float] | None = None,
) -> None:
    wx, wy, wz = adjust_command_pos_for_obstacles(wx, wy, wz, vins_snap)
    wz = clamp_world_z(wz)
    if ENABLE_COMMAND_POS_VELOCITY:
        vx, vy, vz = observe_target_command_velocity(wx, wy, wz)
    else:
        vx, vy, vz = 0.0, 0.0, 0.0

    cmd = Odometry()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "world"
    cmd.pose.pose.position.x = wx
    cmd.pose.pose.position.y = wy
    cmd.pose.pose.position.z = wz
    cmd.pose.pose.orientation.x = 0.0
    cmd.pose.pose.orientation.y = 0.0
    cmd.pose.pose.orientation.z = 0.0
    cmd.pose.pose.orientation.w = target_yaw
    cmd.twist.twist.linear.x = vx
    cmd.twist.twist.linear.y = vy
    cmd.twist.twist.linear.z = vz
    pub.publish(cmd)
