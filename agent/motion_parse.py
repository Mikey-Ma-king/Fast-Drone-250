#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact motion JSON parsing (fwd/lat/vert/yaw) from example_executor&planner."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from agent.config import BODY_DELTA_MAX_M, SUBTASK_TIMEOUT_FALLBACK_DESC, YAW_DELTA_MAX_DEG
from agent.task_state import SubTaskKind, VALID_SUBTASK_KINDS


def extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _json_root(data: Any) -> Optional[dict[str, Any]]:
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _parse_compact_axis(
    raw: Any,
    pos_letter: str,
    neg_letter: str,
    pos_name: str,
    neg_name: str,
) -> tuple[str, float]:
    if raw is None:
        return "none", 0.0
    text = str(raw).strip().upper()
    if text in ("0", "NONE", "HOLD", "-", ""):
        return "none", 0.0
    m = re.match(rf"^([{pos_letter}{neg_letter}])?(\d+(?:\.\d+)?)$", text)
    if not m:
        return "none", 0.0
    letter, mag = m.group(1), float(m.group(2))
    if letter == neg_letter:
        return neg_name, mag
    if letter == pos_letter or letter is None:
        return pos_name, mag
    return "none", 0.0


def _parse_compact_yaw(raw: Any) -> tuple[str, float]:
    if raw is None:
        return "none", 0.0
    text = str(raw).strip().upper()
    if text in ("0", "NONE", "HOLD", "-", ""):
        return "none", 0.0
    m = re.match(r"^(?:Y)?([LR])?(\d+(?:\.\d+)?)$", text)
    if not m:
        return "none", 0.0
    letter, mag = m.group(1), float(m.group(2))
    if letter == "L":
        return "left", mag
    if letter == "R":
        return "right", mag
    return "none", mag


def _axis_sign(name: str, positive: str, negative: str) -> float:
    if name == positive:
        return 1.0
    if name == negative:
        return -1.0
    return 0.0


def _assemble_relative_3d(
    forward: str,
    forward_m: float,
    lateral: str,
    lateral_m: float,
    vertical: str,
    vertical_m: float,
) -> list[float]:
    xyz = [
        _axis_sign(forward, "forward", "back") * forward_m,
        _axis_sign(lateral, "left", "right") * lateral_m,
        _axis_sign(vertical, "up", "down") * vertical_m,
    ]
    dist = (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
    if dist > BODY_DELTA_MAX_M and dist > 1e-6:
        scale = BODY_DELTA_MAX_M / dist
        xyz = [v * scale for v in xyz]
    return xyz


def _extract_reasoning(data: Any, raw_text: str) -> str:
    root = _json_root(data)
    if root:
        for key in ("reasoning", "reason"):
            if root.get(key):
                return str(root[key]).strip()
    m = re.search(r'"(?:reasoning|reason)"\s*:\s*"([^"]+)"', raw_text)
    return m.group(1).strip() if m else ""


def _extract_compact(data: Any, raw_text: str) -> dict[str, str]:
    root = _json_root(data)
    if root and any(k in root for k in ("fwd", "lat", "vert", "yaw")):
        return {
            "fwd": str(root.get("fwd", "0")),
            "lat": str(root.get("lat", "0")),
            "vert": str(root.get("vert", root.get("ver", "0"))),
            "yaw": str(root.get("yaw", "0")),
        }
    compact: dict[str, str] = {}
    for key, pat in (
        ("fwd", r'"fwd"\s*:\s*"([^"]+)"'),
        ("lat", r'"lat"\s*:\s*"([^"]+)"'),
        ("vert", r'"(?:vert|ver)"\s*:\s*"([^"]+)"'),
        ("yaw", r'"yaw"\s*:\s*"([^"]+)"'),
    ):
        m = re.search(pat, raw_text)
        if m:
            compact[key] = m.group(1)
    if compact:
        compact.setdefault("fwd", "0")
        compact.setdefault("lat", "0")
        compact.setdefault("vert", "0")
        compact.setdefault("yaw", "0")
    return compact


def parse_executor_output(
    data: Any,
    raw_text: str,
    *,
    with_reasoning: bool = False,
) -> dict[str, Any]:
    reasoning = _extract_reasoning(data, raw_text) if with_reasoning else None
    compact = _extract_compact(data, raw_text)
    if not compact:
        out = _apply_motion("none", 0.0, "none", 0.0, "none", 0.0, "none", 0.0)
        if reasoning:
            out["reasoning"] = reasoning
        return out
    forward, forward_m = _parse_compact_axis(compact["fwd"], "F", "B", "forward", "back")
    lateral, lateral_m = _parse_compact_axis(compact["lat"], "L", "R", "left", "right")
    vertical, vertical_m = _parse_compact_axis(compact["vert"], "U", "D", "up", "down")
    yaw_turn, yaw_magnitude_deg = _parse_compact_yaw(compact["yaw"])
    return _apply_motion(
        forward,
        forward_m,
        lateral,
        lateral_m,
        vertical,
        vertical_m,
        yaw_turn,
        yaw_magnitude_deg,
        compact,
        reasoning,
    )


def _apply_motion(
    forward: str,
    forward_m: float,
    lateral: str,
    lateral_m: float,
    vertical: str,
    vertical_m: float,
    yaw_turn: str,
    yaw_magnitude_deg: float,
    compact: Optional[dict[str, str]] = None,
    reasoning: Optional[str] = None,
) -> dict[str, Any]:
    mag = float(yaw_magnitude_deg or 0.0)
    yaw_delta_deg = _axis_sign(yaw_turn, "left", "right") * min(mag, YAW_DELTA_MAX_DEG)
    out: dict[str, Any] = {
        "forward": forward,
        "forward_m": forward_m,
        "lateral": lateral,
        "lateral_m": lateral_m,
        "vertical": vertical,
        "vertical_m": vertical_m,
        "yaw_turn": yaw_turn,
        "yaw_magnitude_deg": mag,
        "relative_3d_m": _assemble_relative_3d(
            forward, forward_m, lateral, lateral_m, vertical, vertical_m
        ),
        "yaw_delta_deg": yaw_delta_deg,
    }
    if compact:
        out["compact"] = compact
    if reasoning:
        out["reasoning"] = reasoning
    return out


def executor_parsed_to_body_delta(parsed: dict[str, Any]) -> dict[str, float]:
    xyz = parsed["relative_3d_m"]
    return {
        "x_m": float(xyz[0]),
        "y_m": float(xyz[1]),
        "z_m": float(xyz[2]),
        "yaw_deg": float(parsed["yaw_delta_deg"]),
    }


def _parse_planner_u(val: Any) -> Optional[int]:
    """0=keep; 1=append subtask; 2=mission complete."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        u = int(val)
        if u in (0, 1, 2):
            return u
        return None
    s = str(val).strip().lower()
    if s in ("0", "false", "no"):
        return 0
    if s in ("1", "true", "yes"):
        return 1
    if s == "2":
        return 2
    return None


def compose_subtask_description(action: str, obj: str) -> str:
    """move：action + object → executor 单行 subtask。"""
    a = action.strip().replace("_", " ")
    o = obj.strip()
    if not a and not o:
        return ""
    return " ".join(p for p in (a, o) if p)


def _normalize_subtask_kind(raw: str) -> Optional[SubTaskKind]:
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "scan": "rotate_scan",
        "search": "rotate_scan",
        "scan_search": "rotate_scan",
        "rotate_scan": "rotate_scan",
        "scan_and_search": "rotate_scan",
        "turn_left": "turn_left",
        "left": "turn_left",
        "turn_right": "turn_right",
        "right": "turn_right",
        "hold": "stop",
        "halt": "stop",
        "stop": "stop",
        "move": "move",
    }
    kind = aliases.get(s)
    if kind in VALID_SUBTASK_KINDS:
        return kind  # type: ignore[return-value]
    return None


def _is_json_null(val: Any) -> bool:
    return val is None or str(val).strip().lower() in ("null", "none", "")


def _extract_planner_type(
    root: Optional[dict[str, Any]], raw_text: str,
) -> Optional[SubTaskKind]:
    """顶层 type（u=1 时）。"""
    if root and "type" in root and not _is_json_null(root["type"]):
        k = _normalize_subtask_kind(str(root["type"]))
        if k:
            return k
    m = re.search(r'"type"\s*:\s*"([^"]+)"', raw_text, re.I)
    if m:
        k = _normalize_subtask_kind(m.group(1))
        if k:
            return k
    return None


_SUBTASK_DETAIL_DEFAULTS: dict[SubTaskKind, str] = {
    "move": "",
    "turn_left": "turn left in place",
    "turn_right": "turn right in place",
    "rotate_scan": "rotate in place to search for the target",
    "stop": "hover and hold position",
}


def _extract_planner_detail_text(
    root: Optional[dict[str, Any]], raw_text: str,
) -> str:
    """u=1 时 detail：字符串原样返回；dict 兼容 action/object 或 text 字段。"""
    if root:
        det = root.get("detail")
        if isinstance(det, str) and det.strip():
            return det.strip()
        if isinstance(det, dict):
            for key in ("text", "desc", "description", "instruction", "subtask"):
                val = det.get(key)
                if val is not None and not isinstance(val, dict) and str(val).strip():
                    return str(val).strip()
            action, obj = _extract_move_detail(root, raw_text)
            composed = compose_subtask_description(action, obj)
            if composed:
                return composed
        legacy = _extract_legacy_subtask_line(root, raw_text)
        if legacy:
            return legacy
    m = re.search(r'"detail"\s*:\s*"([^"]+)"', raw_text)
    return m.group(1).strip() if m else ""


def _extract_move_subtask_text(
    root: Optional[dict[str, Any]], raw_text: str,
) -> str:
    return _extract_planner_detail_text(root, raw_text)


def _subtask_text_for_kind(
    kind: SubTaskKind,
    root: Optional[dict[str, Any]],
    raw_text: str,
    *,
    legacy_sub: str = "",
) -> str:
    detail = _extract_planner_detail_text(root, raw_text)
    if detail:
        return detail
    if kind == "move" and legacy_sub:
        return legacy_sub
    return _SUBTASK_DETAIL_DEFAULTS.get(kind, "")


def _extract_move_detail(
    root: Optional[dict[str, Any]], raw_text: str,
) -> tuple[str, str]:
    """move 的 detail.action / detail.object；兼容旧嵌套 subtask。"""
    action = obj = ""
    if root:
        det = root.get("detail")
        if isinstance(det, dict):
            action = str(det.get("action", "")).strip()
            obj = str(det.get("object", det.get("target", ""))).strip()
        if not action:
            action = str(root.get("action", "")).strip()
        if not obj:
            obj = str(
                root.get("object", root.get("target", root.get("object_target", ""))),
            ).strip()
        st = root.get("subtask")
        if isinstance(st, dict):
            if not action:
                action = str(st.get("action", "")).strip()
            if not obj:
                obj = str(st.get("object", st.get("target", ""))).strip()
    if not action:
        m = re.search(r'"action"\s*:\s*"([^"]+)"', raw_text)
        if m:
            action = m.group(1).strip()
    if not obj:
        m = re.search(r'"object"\s*:\s*"([^"]+)"', raw_text)
        if m:
            obj = m.group(1).strip()
    return action, obj


def _extract_legacy_subtask_line(
    root: Optional[dict[str, Any]], raw_text: str,
) -> str:
    if root:
        st = root.get("subtask")
        if st is not None and not isinstance(st, dict) and not _is_json_null(st):
            return str(st).strip()
    m = re.search(r'"subtask"\s*:\s*"([^"]+)"', raw_text)
    return m.group(1).strip() if m else ""


def _extract_subtask_kind_legacy(
    root: Optional[dict[str, Any]], raw_text: str,
) -> SubTaskKind:
    """旧格式：type 在 subtask 对象内或仅有 action/object。"""
    if root:
        st = root.get("subtask")
        if isinstance(st, dict):
            for key in ("type", "kind", "subtask_type"):
                if key in st and not _is_json_null(st[key]):
                    k = _normalize_subtask_kind(str(st[key]))
                    if k:
                        return k
        for key in ("kind", "subtask_type"):
            if key in root and not _is_json_null(root[key]):
                k = _normalize_subtask_kind(str(root[key]))
                if k:
                    return k
    m = re.search(r'"(?:kind|subtask_type)"\s*:\s*"([^"]+)"', raw_text, re.I)
    if m:
        k = _normalize_subtask_kind(m.group(1))
        if k:
            return k
    return "move"


def _extract_planner_subtask(
    root: Optional[dict[str, Any]], raw_text: str,
) -> tuple[SubTaskKind, str, str, str]:
    """u=1 时解析 (kind, action, object, legacy_line)。"""
    kind = _extract_planner_type(root, raw_text)
    legacy = _extract_legacy_subtask_line(root, raw_text)

    if kind is not None:
        if kind == "move":
            action, obj = _extract_move_detail(root, raw_text)
            return kind, action, obj, legacy
        return kind, "", "", legacy

    kind = _extract_subtask_kind_legacy(root, raw_text)
    action, obj = _extract_move_detail(root, raw_text)
    return kind, action, obj, legacy


def _extract_planner_u(root: Optional[dict[str, Any]], raw_text: str) -> Optional[int]:
    if root:
        for key in ("u", "update_subtask", "update"):
            if key in root:
                u = _parse_planner_u(root[key])
                if u is not None:
                    return u
    m = re.search(r'"(?:u|update_subtask)"\s*:\s*([012])', raw_text)
    if m:
        return int(m.group(1))
    return None


def parse_planner_vlm_output(
    data: Any,
    raw_text: str,
    *,
    with_reasoning: bool = True,
) -> dict[str, Any]:
    root = _json_root(data)
    subtask = ""
    reasoning = ""
    scene_summary = ""
    mission_complete = False
    u = _extract_planner_u(root, raw_text)
    kind: Optional[SubTaskKind] = None
    action = obj = ""

    if u == 2:
        mission_complete = True
    elif u is None and root:
        fin = root.get("f", root.get("mission_complete"))
        if isinstance(fin, bool) and fin:
            mission_complete = True
            u = 2
        elif fin is not None and str(fin).lower() in ("true", "yes", "1") or int(fin) == 1:
            mission_complete = True
            u = 2

    kind = "move"
    action = obj = ""
    legacy_sub = ""
    if u == 1:
        kind, action, obj, legacy_sub = _extract_planner_subtask(root, raw_text)
        subtask = _subtask_text_for_kind(kind, root, raw_text, legacy_sub=legacy_sub)

    if root and with_reasoning:
        reasoning = str(root.get("reasoning", root.get("reason", ""))).strip()
        scene_summary = str(root.get("scene_summary", "")).strip()

    if u == 1 and not subtask:
        kind, action, obj, legacy_sub = _extract_planner_subtask(None, raw_text)
        subtask = _subtask_text_for_kind(kind, root, raw_text, legacy_sub=legacy_sub)

    if with_reasoning and not reasoning:
        m = re.search(r'"(?:reasoning|reason)"\s*:\s*"([^"]+)"', raw_text)
        if m:
            reasoning = m.group(1).strip()
    if not scene_summary:
        m = re.search(r'"scene_summary"\s*:\s*"([^"]+)"', raw_text)
        if m:
            scene_summary = m.group(1).strip()
    if not mission_complete:
        m = re.search(r'"(?:f|mission_complete)"\s*:\s*([012]|true|false)', raw_text, re.I)
        if m:
            mission_complete = m.group(1).lower() in ("1", "true", "2")
        if not mission_complete:
            m = re.search(r'"u"\s*:\s*2', raw_text)
            if m:
                mission_complete = True
                u = 2

    return {
        "u": u,
        "scene_summary": scene_summary or "(none)",
        "kind": kind,
        "action": action,
        "object": obj,
        "subtask": subtask,
        "reasoning": reasoning,
        "mission_complete": mission_complete,
    }


def map_planner_to_list_action(
    parsed: dict[str, Any],
    *,
    current_subtask: str,
    has_subtask: bool,
    force_switch: bool,
) -> dict[str, Any]:
    u = parsed.get("u")
    if parsed.get("mission_complete") or u == 2:
        return {"f": 1}

    subtask = str(parsed.get("subtask", "")).strip()
    kind: SubTaskKind = parsed.get("kind") or "move"  # u=1 时必有 kind

    def _append_action() -> dict[str, Any]:
        if kind == "move" and not subtask:
            raise ValueError("u=1 type=move requires non-empty detail for the executor")
        return {"u": 1, "d": subtask or SUBTASK_TIMEOUT_FALLBACK_DESC, "kind": kind}

    if force_switch:
        if u in (0, None):
            return _append_action()
        if u == 1:
            return _append_action()
        if u == 2:
            return {"f": 1}
        return _append_action()

    if not has_subtask:
        if u != 1:
            raise ValueError("no subtask in list yet: require u=1")
        return _append_action()

    if u is None:
        raise ValueError("missing u (0=keep, 1=new subtask, 2=mission done)")

    if u == 0:
        return {"u": 0}

    if u == 1:
        return _append_action()

    if u == 2:
        return {"f": 1}

    raise ValueError(f"invalid u={u!r}")
