#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离散动作 Executor：解析 VLM 返回的 FRONT/BACK/UP/DOWN/TURN_LEFT/TURN_RIGHT。"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.config import (
    DISCRETE_STEP_BACK_M,
    DISCRETE_STEP_DOWN_M,
    DISCRETE_STEP_FORWARD_M,
    DISCRETE_STEP_UP_M,
    DISCRETE_TURN_DEG,
)

DiscreteAction = str

VALID_DISCRETE_ACTIONS: tuple[DiscreteAction, ...] = (
    "FRONT",
    "BACK",
    "UP",
    "DOWN",
    "TURN_LEFT",
    "TURN_RIGHT",
)

_ACTION_ALIASES: dict[str, DiscreteAction] = {
    "FRONT": "FRONT",
    "FORWARD": "FRONT",
    "FWD": "FRONT",
    "F": "FRONT",
    "GO_FORWARD": "FRONT",
    "BACK": "BACK",
    "BACKWARD": "BACK",
    "BWD": "BACK",
    "B": "BACK",
    "GO_BACK": "BACK",
    "UP": "UP",
    "U": "UP",
    "ASCEND": "UP",
    "CLIMB": "UP",
    "DOWN": "DOWN",
    "D": "DOWN",
    "DESCEND": "DOWN",
    "TURN_LEFT": "TURN_LEFT",
    "TURNLEFT": "TURN_LEFT",
    "TURN LEFT": "TURN_LEFT",
    "LEFT": "TURN_LEFT",
    "YAW_LEFT": "TURN_LEFT",
    "ROTATE_LEFT": "TURN_LEFT",
    "L": "TURN_LEFT",
    "TURN_RIGHT": "TURN_RIGHT",
    "TURNRIGHT": "TURN_RIGHT",
    "TURN RIGHT": "TURN_RIGHT",
    "RIGHT": "TURN_RIGHT",
    "YAW_RIGHT": "TURN_RIGHT",
    "ROTATE_RIGHT": "TURN_RIGHT",
    "R": "TURN_RIGHT",
}


def _normalize_action_token(raw: str) -> str:
    s = raw.strip().upper().replace("-", "_")
    s = re.sub(r"\s+", " ", s)
    if " " in s and s not in _ACTION_ALIASES:
        s = s.replace(" ", "_")
    return s


def normalize_discrete_action(raw: Any) -> Optional[DiscreteAction]:
    if raw is None:
        return None
    token = _normalize_action_token(str(raw))
    if token in _ACTION_ALIASES:
        return _ACTION_ALIASES[token]
    compact = token.replace("_", "")
    for alias, canonical in _ACTION_ALIASES.items():
        if alias.replace("_", "") == compact:
            return canonical
    return None


def _extract_action(data: Any, raw_text: str) -> Optional[DiscreteAction]:
    root = data if isinstance(data, dict) else None
    if root is not None:
        for key in ("action", "cmd", "command", "move", "discrete_action"):
            if key in root and root[key] is not None:
                act = normalize_discrete_action(root[key])
                if act:
                    return act
    m = re.search(
        r'"(?:action|cmd|command|move|discrete_action)"\s*:\s*"([^"]+)"',
        raw_text,
        re.I,
    )
    if m:
        return normalize_discrete_action(m.group(1))
    for act in VALID_DISCRETE_ACTIONS:
        if re.search(rf"\b{act}\b", raw_text, re.I):
            return act
    return None


def parse_executor_discrete_output(
    data: Any,
    raw_text: str,
    *,
    with_reasoning: bool = False,
) -> dict[str, Any]:
    action = _extract_action(data, raw_text)
    if action is None:
        raise ValueError(
            "no discrete action in reply "
            f'(expected one of {", ".join(VALID_DISCRETE_ACTIONS)})'
        )
    reasoning = ""
    if with_reasoning and isinstance(data, dict):
        reasoning = str(data.get("reasoning", data.get("reason", ""))).strip()
    if with_reasoning and not reasoning:
        m = re.search(r'"(?:reasoning|reason)"\s*:\s*"([^"]+)"', raw_text)
        if m:
            reasoning = m.group(1).strip()
    out: dict[str, Any] = {"action": action}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def discrete_action_to_body_delta(action: DiscreteAction) -> dict[str, float]:
    """离散动作 → 机体系增量（米 / 度）。"""
    if action == "FRONT":
        return {
            "x_m": float(DISCRETE_STEP_FORWARD_M),
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": 0.0,
        }
    if action == "BACK":
        return {
            "x_m": -float(DISCRETE_STEP_BACK_M),
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": 0.0,
        }
    if action == "UP":
        return {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": float(DISCRETE_STEP_UP_M),
            "yaw_deg": 0.0,
        }
    if action == "DOWN":
        return {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": -float(DISCRETE_STEP_DOWN_M),
            "yaw_deg": 0.0,
        }
    if action == "TURN_LEFT":
        return {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": float(DISCRETE_TURN_DEG),
        }
    if action == "TURN_RIGHT":
        return {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": -float(DISCRETE_TURN_DEG),
        }
    raise ValueError(f"unknown discrete action: {action!r}")
