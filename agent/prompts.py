#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM prompts (planner / 3D executor / 2D executor)."""

from __future__ import annotations

from agent.config import EXECUTOR_2D_BBOX, EXECUTOR_REASONING
from agent.task_state import TaskState

# Planner 顶层字段：u + type + detail（move 时 detail 为自由文本，原样给执行器）
PLANNER_JSON_SCHEMA = (
    '{"u":0|1|2,"type":null|"move"|"rotate_scan"|"stop",'
    '"detail":null|string|object}'
)
PLANNER_JSON_U0 = '{"u":0,"type":null,"detail":null}'
PLANNER_JSON_U2 = '{"u":2,"type":null,"detail":null}'
PLANNER_JSON_U1_MOVE = (
    '{"u":1,"type":"move","detail":"free-text instruction for the executor"}'
)
PLANNER_JSON_U1_ROTATE_SCAN = '{"u":1,"type":"rotate_scan","detail":null}'
PLANNER_JSON_U1_STOP = '{"u":1,"type":"stop","detail":null}'


def build_planner_prompt(
    task_state: TaskState,
    *,
    force_switch: bool,
) -> str:
    current = task_state.current_subtask_text().strip() or "(none)"
    cur_kind = task_state.current_subtask_kind()
    kind_line = f" ({cur_kind})" if cur_kind else ""
    lines = [
        "You are a UAV task planner. Your job is to decompose the global mission into "
        "subtasks and keep updating that plan as the drone moves.",
        "",
        "Global task:",
        task_state.global_task.strip(),
        "",
        f"Current subtask{kind_line}:",
        current,
        "",
        "Images:",
        "A = image when the current subtask started.",
        "B = current live image.",
        "",
        "Each cycle you decide whether to keep, replace, or finish the active subtask. "
        "Compare A vs B to judge progress, then output one JSON object.",
        "",
        "Output fields (always present):",
        "- u: 0=keep executing the current subtask; 1=end it and append a new subtask; "
        "2=global mission fully complete.",
        "- type: subtask kind when u=1; otherwise null.",
        "- detail: when type=move, a short free-text instruction for the executor; otherwise null.",
        "",
        "Subtask types (type when u=1):",
        "- move: set detail to any concise instruction for the executor.",
        "- rotate_scan: in-place yaw sweep to search; detail=null.",
        "- stop: hold position; detail=null.",
        "",
        "Rules:",
        "- Decompose the global task into steps. When the current step is done, blocked, "
        "or no longer fits what you see in B, use u=1 to append the next subtask.",
        "- u=0: the current subtask is still in progress and should continue; type=null, detail=null. "
        "Do NOT use u=0 if the subtask is finished, achieved, blocked, or obsolete.",
        "- u=1: the current subtask ends; you append one new subtask for the executor. "
        "Set type to move|rotate_scan|stop; include detail only for move.",
        "- u=2: the entire global mission is achieved; type=null, detail=null.",
        "- Re-plan whenever B shows the situation changed (target found/lost, obstacle, wrong direction).",
        "- Use rotate_scan when the mission target is not in view.",
        "- Use stop when the drone should hover and wait.",
        "- Return JSON only.",
        "",
        "Examples:",
        PLANNER_JSON_U0,
        PLANNER_JSON_U2,
        PLANNER_JSON_U1_MOVE,
        PLANNER_JSON_U1_ROTATE_SCAN,
        PLANNER_JSON_U1_STOP,
        "",
        f"Schema: {PLANNER_JSON_SCHEMA}",
        "",
        "Subtask list (oldest → newest):",
        *task_state.get_subtask_summary_lines(),
    ]
    if force_switch:
        lines.extend([
            "",
            f'[FORCED SWITCH] Current subtask "{current}" timed out. '
            "u=0 forbidden. Use u=1 to append the next subtask, or u=2 if the global mission is done.",
        ])
    return "\n".join(lines)


def build_executor_prompt(
    subtask: str,
    *,
    with_reasoning: bool = EXECUTOR_REASONING,
) -> str:
    fmt = '{"fwd":"F<m>|B<m>","lat":"L<m>|R<m>","vert":"U<m>|D<m>","yaw":"L<deg>|R<deg>"'
    if with_reasoning:
        fmt += ',"reasoning":"brief"}'
    else:
        fmt += "}"
    return "\n".join([
        "Drone 3D motion executor. JSON only.",
        "One forward-facing RGB photo (before text): current scene.",
        "Body +X forward, +Y left, +Z up.",
        f"Task: {subtask.strip()}",
        "Output body-frame delta: fwd/lat/vert/yaw.",
        "fwd/lat/vert/yaw tokens: F/B/L/R/U/D + magnitude; yaw L/R degrees (L=left/+deg, R=right/-deg).",
        "Motion heuristics:",
        "- Approach: move toward where the target sits in the image; keep safe standoff, do not fly through the target.",
        "- Search: slow in-place yaw clockwise to scan.",
        "- Obstacle avoidance (priority): if people, walls, furniture, or obstacles block the path or are too close ahead, steer around them.",
        f"Format: {fmt}",
    ])


def build_executor_2d_prompt(
    subtask: str,
    *,
    with_bbox: bool = EXECUTOR_2D_BBOX,
) -> str:
    lines = [
        "You are a UAV visual waypoint executor (move subtask only).",
        "",
        "Current subtask:",
        subtask.strip(),
        "",
        "You receive one image: current RGB.",
        "",
        "Pick ONE waypoint around the subtask object (on it or nearby).",
        "It may deviate from the object surface for obstacle avoidance.",
        "Do NOT output distance or other fields.",
    ]
    if with_bbox:
        lines.extend([
            "Also output a tight axis-aligned bbox around that object (visualization only).",
            "",
            "Return JSON only:",
            '{"x":int,"y":int,"bbox":[x1,y1,x2,y2]}',
        ])
    else:
        lines.extend([
            "",
            "Return JSON only:",
            '{"x":int,"y":int}',
        ])
    return "\n".join(lines)
