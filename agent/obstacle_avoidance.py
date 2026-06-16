#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度图 → 障碍 3D 点；发布 /command_pos 前保证「当前位置→目标」线段硬安全距离。"""

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import Image

from agent.camera_geom import bearing_world_from_body
from agent.camera_intrinsics import CameraIntrinsics
from scipy.optimize import minimize

from agent.config import (
    DEPTH_CX,
    DEPTH_CY,
    DEPTH_FX,
    DEPTH_FY,
    DEPTH_HEIGHT,
    DEPTH_IMAGE_TOPIC,
    DEPTH_WIDTH,
    ENABLE_COMMAND_POS_OBSTACLE_AVOIDANCE,
    OA_OBS_DEPTH_MAX_M,
    OA_OBS_MIN_CLEARANCE_M,
    OBS_GRID_DEPTH_PERCENTILE,
    OBS_GRID_MIN_VALID_DEPTH_M,
    OBS_GRID_N,
    OBS_POINT_HISTORY_FRAMES,
)

_avoidance: Optional["CommandPosObstacleAvoidance"] = None
def _depth_array_to_meters(depth_np, encoding: str) -> np.ndarray:
    if depth_np.dtype == np.uint16:
        return depth_np.astype(np.float64) * 0.001
    if depth_np.dtype == np.float32:
        return depth_np.astype(np.float64)
    enc = (encoding or "").lower()
    if "32f" in enc:
        return depth_np.astype(np.float64)
    return depth_np.astype(np.float64)


def _depth_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        DEPTH_FX,
        DEPTH_FY,
        DEPTH_CX,
        DEPTH_CY,
    )


def _grid_cell_center_uv(gi: int, gj: int, grid_n: int) -> tuple[float, float]:
    u = (float(gj) + 0.5) * float(DEPTH_WIDTH) / float(grid_n)
    v = (float(gi) + 0.5) * float(DEPTH_HEIGHT) / float(grid_n)
    return u, v


def precompute_obs_dirs_body(grid_n: int = OBS_GRID_N) -> np.ndarray:
    from agent.camera_geom import bearing_body_from_pixel

    intr = _depth_intrinsics()
    dirs = np.zeros((grid_n, grid_n, 3), dtype=np.float64)
    for gi in range(grid_n):
        for gj in range(grid_n):
            u, v = _grid_cell_center_uv(gi, gj, grid_n)
            xb, yb, zb = bearing_body_from_pixel(u, v, intr)
            nlen = math.hypot(xb, math.hypot(yb, zb))
            if nlen > 1e-9:
                dirs[gi, gj, 0] = xb / nlen
                dirs[gi, gj, 1] = yb / nlen
                dirs[gi, gj, 2] = zb / nlen
    return dirs


def depth_image_to_obstacle_grid(
    depth_m: np.ndarray,
    grid_n: int = OBS_GRID_N,
    percentile: int = OBS_GRID_DEPTH_PERCENTILE,
) -> np.ndarray:
    h, w = depth_m.shape[:2]
    out = np.zeros((grid_n, grid_n), dtype=np.float64)
    for gi in range(grid_n):
        r0 = int(gi * h / grid_n)
        r1 = int((gi + 1) * h / grid_n)
        if r1 <= r0:
            continue
        for gj in range(grid_n):
            c0 = int(gj * w / grid_n)
            c1 = int((gj + 1) * w / grid_n)
            if c1 <= c0:
                continue
            patch = depth_m[r0:r1, c0:c1].reshape(-1)
            valid = patch[(patch > OBS_GRID_MIN_VALID_DEPTH_M) & np.isfinite(patch)]
            if valid.size > 0:
                out[gi, gj] = float(np.percentile(valid, percentile))
    return out


def obstacle_points_world_from_grid(
    grid: np.ndarray,
    body_dirs: np.ndarray,
    origin: np.ndarray,
    quat: tuple[float, float, float, float],
    depth_max_m: float,
) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    body_dirs = np.asarray(body_dirs, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    q = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    gn = grid.shape[0]
    pts = []
    for gi in range(gn):
        for gj in range(gn):
            r_m = float(grid[gi, gj])
            if r_m <= OBS_GRID_MIN_VALID_DEPTH_M or r_m > depth_max_m:
                continue
            xb, yb, zb = body_dirs[gi, gj]
            xw, yw, zw = bearing_world_from_body(xb, yb, zb, quat=q)
            pts.append(origin + r_m * np.array([xw, yw, zw], dtype=np.float64))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pts)


def _segment_point_distance(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    len_sq = float(ab @ ab)
    if len_sq < 1e-12:
        return float(np.linalg.norm(o - a))
    t = float(np.clip((o - a) @ ab / len_sq, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(o - closest))


def _segment_clearance_feasible(
    p_start: np.ndarray,
    p_end: np.ndarray,
    obstacles: np.ndarray,
    min_clear_m: float,
) -> bool:
    if obstacles.shape[0] == 0:
        return True
    for o in obstacles:
        if _segment_point_distance(o, p_start, p_end) < min_clear_m - 1e-9:
            return False
    return True


def _active_obstacles_for_segment(
    p_start: np.ndarray,
    p_cmd: np.ndarray,
    obstacles: np.ndarray,
    min_clear_m: float,
    margin_m: float = 1.0,
) -> np.ndarray:
    if obstacles.shape[0] == 0:
        return obstacles
    dists = np.array(
        [_segment_point_distance(o, p_start, p_cmd) for o in obstacles],
        dtype=np.float64,
    )
    mask = dists < (min_clear_m + margin_m)
    active = obstacles[mask]
    if active.shape[0] == 0:
        return obstacles
    return active


def closest_target_with_segment_clearance(
    p_start: np.ndarray,
    p_cmd: np.ndarray,
    obstacles: np.ndarray,
    min_clear_m: float,
) -> np.ndarray:
    """SLSQP：距 p_cmd 最近，且线段 [p_start, p] 与所有障碍距离 >= min_clear_m。"""
    p_start = np.asarray(p_start, dtype=np.float64).reshape(3)
    p_cmd = np.asarray(p_cmd, dtype=np.float64).reshape(3)
    obstacles = np.asarray(obstacles, dtype=np.float64)
    if obstacles.ndim != 2 or obstacles.shape[0] == 0 or min_clear_m <= 0.0:
        return p_cmd.copy()
    if _segment_clearance_feasible(p_start, p_cmd, obstacles, min_clear_m):
        return p_cmd.copy()

    active = _active_obstacles_for_segment(p_start, p_cmd, obstacles, min_clear_m)

    def objective(p: np.ndarray) -> float:
        d = p - p_cmd
        return float(d @ d)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda p, o=o.copy(): _segment_point_distance(o, p_start, p) - min_clear_m,
        }
        for o in active
    ]
    res = minimize(
        objective,
        p_cmd,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 400},
    )
    if res.success and _segment_clearance_feasible(p_start, res.x, obstacles, min_clear_m):
        return res.x

    rospy.logwarn_throttle(
        1.0,
        "[agent OA] segment clearance SLSQP failed, hold at current position",
    )
    return p_start.copy()


class CommandPosObstacleAvoidance:
    """维护障碍点历史；保证当前位置→command_pos 线段硬安全距离。"""

    def __init__(self) -> None:
        self.obs_depth_max_m = OA_OBS_DEPTH_MAX_M
        self.min_clearance_m = OA_OBS_MIN_CLEARANCE_M
        self.obs_dirs_body: Optional[np.ndarray] = None
        self._history: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._vins_snap: dict[str, float] = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }

    def reset(self) -> None:
        with self._lock:
            self._history = []
        self.obs_dirs_body = precompute_obs_dirs_body(OBS_GRID_N)

    def update_vins(self, snap: dict[str, float]) -> None:
        self._vins_snap = dict(snap)

    def on_depth_msg(self, msg: Image, bridge) -> None:
        if self.obs_dirs_body is None:
            self.reset()
        snap = self._vins_snap
        try:
            depth_np = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_m = _depth_array_to_meters(depth_np, msg.encoding)
            grid = depth_image_to_obstacle_grid(depth_m)
            quat = (
                float(snap["qx"]),
                float(snap["qy"]),
                float(snap["qz"]),
                float(snap["qw"]),
            )
            origin = np.array([snap["x"], snap["y"], snap["z"]], dtype=np.float64)
            pts = obstacle_points_world_from_grid(
                grid,
                self.obs_dirs_body,
                origin,
                quat,
                self.obs_depth_max_m,
            )
            with self._lock:
                self._history.append(pts)
                if len(self._history) > OBS_POINT_HISTORY_FRAMES:
                    self._history.pop(0)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[agent OA] depth_cb: %s", e)

    def _flatten_points(self) -> np.ndarray:
        with self._lock:
            if not self._history:
                return np.zeros((0, 3), dtype=np.float64)
            return np.vstack(self._history)

    def adjust_command(
        self,
        wx: float,
        wy: float,
        wz: float,
        vins_snap: dict[str, float],
    ) -> tuple[float, float, float]:
        pts = self._flatten_points()
        p_start = np.array(
            [float(vins_snap["x"]), float(vins_snap["y"]), float(vins_snap["z"])],
            dtype=np.float64,
        )
        p_cmd = np.array([wx, wy, wz], dtype=np.float64)
        if pts.shape[0] == 0:
            return wx, wy, wz

        p_safe = closest_target_with_segment_clearance(
            p_start,
            p_cmd,
            pts,
            self.min_clearance_m,
        )
        return float(p_safe[0]), float(p_safe[1]), float(p_safe[2])


def init_obstacle_avoidance() -> Optional[CommandPosObstacleAvoidance]:
    global _avoidance
    if not ENABLE_COMMAND_POS_OBSTACLE_AVOIDANCE:
        _avoidance = None
        return None
    _avoidance = CommandPosObstacleAvoidance()
    _avoidance.reset()
    rospy.loginfo(
        "[agent OA] ON depth=%s grid=%dx%d history=%d clearance=%.2fm depth_max=%.1fm",
        DEPTH_IMAGE_TOPIC,
        OBS_GRID_N,
        OBS_GRID_N,
        OBS_POINT_HISTORY_FRAMES,
        _avoidance.min_clearance_m,
        _avoidance.obs_depth_max_m,
    )
    return _avoidance


def get_obstacle_avoidance() -> Optional[CommandPosObstacleAvoidance]:
    return _avoidance


def adjust_command_pos_for_obstacles(
    wx: float,
    wy: float,
    wz: float,
    vins_snap: Optional[dict[str, float]],
) -> tuple[float, float, float]:
    if vins_snap is None or _avoidance is None:
        return wx, wy, wz
    return _avoidance.adjust_command(wx, wy, wz, vins_snap)
