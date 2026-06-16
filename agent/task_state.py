#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""子任务 list 与 TaskState。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import rospy

from agent.config import MAX_LIST_LEN, SUBTASK_MAX_DURATION_S

SubTaskKind = Literal["move", "rotate_scan", "stop"]
VALID_SUBTASK_KINDS: tuple[SubTaskKind, ...] = ("move", "rotate_scan", "stop")


@dataclass
class SubTask:
    description: str
    kind: SubTaskKind
    rgb_snapshot: np.ndarray
    started_at: rospy.Time
    ended_at: Optional[rospy.Time] = None  # append 下一条或 finish 时写入，显示用时不再增长


class TaskState:
    def __init__(self, global_task: str) -> None:
        self.global_task = global_task
        self.subtasks: list[SubTask] = []
        self.max_list_len = MAX_LIST_LEN
        self.finished = False
        self.list_version = 0
        self._lock = threading.Lock()

    def current_subtask_text(self) -> str:
        with self._lock:
            return self.subtasks[-1].description if self.subtasks else ""

    def current_subtask_kind(self) -> Optional[SubTaskKind]:
        with self._lock:
            return self.subtasks[-1].kind if self.subtasks else None

    def has_subtask(self) -> bool:
        with self._lock:
            return bool(self.subtasks)

    def _elapsed_s_unlocked(self, st: SubTask, now: rospy.Time) -> float:
        end = st.ended_at if st.ended_at is not None else now
        return (end - st.started_at).to_sec()

    def _close_current_subtask_unlocked(self, now: rospy.Time) -> None:
        if self.subtasks and self.subtasks[-1].ended_at is None:
            self.subtasks[-1].ended_at = now

    def current_subtask_elapsed_s(self) -> float:
        with self._lock:
            if not self.subtasks:
                return 0.0
            return self._elapsed_s_unlocked(self.subtasks[-1], rospy.Time.now())

    def force_switch(self) -> bool:
        with self._lock:
            if not self.subtasks:
                return False
            return self._elapsed_s_unlocked(self.subtasks[-1], rospy.Time.now()) >= SUBTASK_MAX_DURATION_S

    def append_subtask(
        self,
        description: str,
        rgb: np.ndarray,
        *,
        kind: SubTaskKind = "move",
    ) -> int:
        if kind not in VALID_SUBTASK_KINDS:
            raise ValueError(f"invalid subtask kind: {kind!r}")
        with self._lock:
            now = rospy.Time.now()
            self._close_current_subtask_unlocked(now)
            self.subtasks.append(
                SubTask(
                    description=description.strip(),
                    kind=kind,
                    rgb_snapshot=np.ascontiguousarray(rgb.copy()),
                    started_at=now,
                )
            )
            while len(self.subtasks) > self.max_list_len:
                self.subtasks.pop(0)
            self.list_version += 1
            return self.list_version

    def get_list_version(self) -> int:
        with self._lock:
            return self.list_version

    def get_subtask_summary_lines(self, *, include_elapsed: bool = False) -> list[str]:
        now = rospy.Time.now()
        with self._lock:
            if not self.subtasks:
                return ["  (no subtasks yet — output first subtask in JSON)"]
            lines: list[str] = []
            for i, st in enumerate(self.subtasks):
                marker = " <- 当前" if i == len(self.subtasks) - 1 else ""
                tag = f"[{st.kind}] "
                if include_elapsed:
                    elapsed = self._elapsed_s_unlocked(st, now)
                    lines.append(
                        f"  {i + 1}. {tag}{st.description} ({elapsed:.1f}s){marker}",
                    )
                else:
                    lines.append(f"  {i + 1}. {tag}{st.description}{marker}")
            return lines

    def get_current_subtask_rgb_snapshot(self) -> Optional[np.ndarray]:
        """当前子任务开始时的 RGB 快照（无子任务时返回 None）。"""
        with self._lock:
            if not self.subtasks:
                return None
            return np.ascontiguousarray(self.subtasks[-1].rgb_snapshot.copy())

    def get_executor_context(
        self,
    ) -> Optional[tuple[str, SubTaskKind, np.ndarray, int]]:
        with self._lock:
            if not self.subtasks:
                return None
            cur = self.subtasks[-1]
            return (
                cur.description,
                cur.kind,
                np.ascontiguousarray(cur.rgb_snapshot.copy()),
                self.list_version,
            )

    def mark_finished(self) -> None:
        with self._lock:
            self._close_current_subtask_unlocked(rospy.Time.now())
            self.finished = True

    def is_finished(self) -> bool:
        with self._lock:
            return self.finished
