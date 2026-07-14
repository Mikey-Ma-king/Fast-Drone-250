#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM prompts (planner / 3D executor / 2D executor)."""

from __future__ import annotations

from agent.config import EXECUTOR_2D_BBOX, EXECUTOR_REASONING
from agent.task_state import TaskState


def build_planner_prompt(
    task_state: TaskState,
    *,
    force_switch: bool,
) -> str:
    current = task_state.current_subtask_text().strip() or "(none)"
    cur_kind = task_state.current_subtask_kind()
    kind_line = f" ({cur_kind})" if cur_kind else ""

    force_line = (
        "Force switch: YES. Current subtask timed out. "
        "You MUST output u=1 with a new concrete subtask (type + detail), "
        "or u=2 if the global mission is complete. u=0 is forbidden."
        if force_switch else
        "Force switch: NO."
    )

    lines = [
        "You are a UAV event-driven planner.",
        "The executor can only execute discrete actions: FRONT, BACK, UP, DOWN, TURN_LEFT, TURN_RIGHT.",
        "Your job is to maintain ONE local visual subtask for the executor.",
        "Return JSON only.",
        "",
        "Global task:",
        task_state.global_task.strip(),
        "",
        f"Current subtask{kind_line}:",
        current,
        "",
        force_line,
        "",
        "Images:",
        "A = image when current subtask started.",
        "B = current live image.",
        "",
        "Output JSON:",
        '{"u":0|1|2,"type":null|"move"|"turn_left"|"turn_right"|"rotate_scan"|"stop","detail":null|string}',
        "",
        "Meanings:",
        "- u=0: keep current subtask; type=null; detail=null.",
        "- u=1: current subtask ends; append one new subtask.",
        "- u=2: global task complete; type=null; detail=null.",
        "",
        "Subtask types:",
        "- move: approach, enter, follow, or move until a visible cue.",
        "- turn_left / turn_right: rotate in place.",
        "- rotate_scan: yaw search when target/cue is not visible.",
        "- stop: hover.",
        "",
        "Hard switch rules:",
        "- If current subtask is rotate_scan/search AND B shows target or any clear navigation cue (doorway/corridor/intersection/room entrance), immediately switch to move.",
        "- If searching for a room (e.g., kitchen) AND no target is visible, use rotate_scan; do NOT output move toward abstract target.",
        "- If B shows a doorway/corridor but target room is not visible, output move THROUGH the opening to continue search.",
        "",
        "- For 'turn right at intersection':",
        "  when B shows intersection or right opening → output turn_right immediately.",
        "- For 'turn left at intersection':",
        "  when B shows intersection or left opening → output turn_left immediately.",
        "",
        "- MOVE rule (important):",
        "- Only use move when at least ONE is true:",
        "  (1) target is visible",
        "  (2) doorway/corridor/opening is visible and aligned",
        "  (3) you are already inside a room/corridor and continuing forward is safe",
        "- If front view is blocked (wall/close obstacle), NEVER continue move forward; switch to turn or rotate_scan.",
        "",
        "- After turn_left/turn_right, next step should usually be move along the new corridor.",
        "- If target is reached or fully inside target room, output u=2.",
        "- Do not output u=0 when a visual event changes (new cue/door/intersection appears).",
        "",
        "Good move details:",
        "- move toward the visible kitchen entrance",
        "- move through the visible doorway",
        "- move forward until the first intersection is visible",
        "- move forward along the current corridor",
        "- move toward the right-side opening",
        "",
        "Examples:",
        '{"u":0,"type":null,"detail":null}',
        '{"u":2,"type":null,"detail":null}',
        '{"u":1,"type":"rotate_scan","detail":"rotate in place to search for the target"}',
        '{"u":1,"type":"move","detail":"move toward the visible kitchen entrance"}',
        '{"u":1,"type":"move","detail":"move forward until the first intersection is visible"}',
        '{"u":1,"type":"turn_right","detail":"turn right toward the right opening"}',
        '{"u":1,"type":"turn_left","detail":"turn left toward the left opening"}',
        '{"u":1,"type":"stop","detail":"hover and hold position"}',
        "",
        "Subtask list:",
        *task_state.get_subtask_summary_lines(),
    ]

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


def build_executor_discrete_prompt(
    subtask: str,
    *,
    with_reasoning: bool = EXECUTOR_REASONING,
) -> str:
    from agent.parse_discrete import VALID_DISCRETE_ACTIONS

    actions = ", ".join(VALID_DISCRETE_ACTIONS)
    fmt = '{"action":"FRONT|BACK|UP|DOWN|TURN_LEFT|TURN_RIGHT"'
    if with_reasoning:
        fmt += ',"reasoning":"brief"}'
    else:
        fmt += "}"

    return "\n".join([
        "Drone discrete-action executor. JSON only.",
        "One forward-facing RGB photo is provided before this text: current scene.",
        f"Task: {subtask.strip()}",
        "",
        "Pick exactly ONE action from this set:",
        actions,
        "",
        "Action semantics:",
        "- FRONT: move forward a small step",
        "- BACK: move backward a small step",
        "- UP: ascend a small step",
        "- DOWN: descend a small step",
        "- TURN_LEFT: rotate left in place",
        "- TURN_RIGHT: rotate right in place",
        "",
        "Policy:",
        "- Choose the action that best approaches the task target/cue.",
        "- Target/cue means the visible object, doorway, opening, corridor center, or route direction mentioned by the task.",
        "- If the target/cue is roughly in front and the path is open, output FRONT.",
        "- If the target/cue is clearly left of center, output TURN_LEFT.",
        "- If the target/cue is clearly right of center, output TURN_RIGHT.",
        "- If the target/cue is only slightly off-center but still reachable forward, output FRONT.",
        "- Do not turn repeatedly for small alignment errors.",
        "",
        "Room/door/corridor behavior:",
        "- To enter a room or doorway: first face the doorway/opening, then move FRONT through it.",
        "- If the doorway/opening is off-center, turn toward it before moving.",
        "- To follow a corridor: approach the corridor center; use FRONT when the corridor direction is roughly centered.",
        "",
        "Safety:",
        "- Do not output FRONT if the image center is blocked by a close wall, closed door, person, furniture, or obstacle.",
        "- If too close to an obstacle, output BACK.",
        "- If blocked ahead but a safe opening exists left/right, turn toward that opening.",
        "",
        "Search:",
        "- If the target/cue is not visible, rotate to search.",
        "- Prefer the direction suggested by the task; otherwise use TURN_LEFT.",
        "",
        "Vertical:",
        "- Use UP/DOWN only if the target/cue is clearly above/below or vertical motion is required.",
        "",
        "Avoid:",
        "- Do not blindly move FRONT just because the task says move/go/enter.",
        "- Do not alternate TURN_LEFT and TURN_RIGHT for tiny visual offsets.",
        "- Do not move into walls, closed doors, people, or furniture.",
        "",
        "Return JSON only.",
        f"Format: {fmt}",
    ])
