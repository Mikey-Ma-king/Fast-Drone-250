import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


# ======================
# Config
# ======================

IMAGE_PATH = Path("/home/ps/ltc/agent/IMG_20260521_153519.jpg")
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"

MISSION_PROMPT = (
    "Stop beside the door. "
    "Collisions must be avoided; plan each move reasonably and safely."
)
HORIZON_S = 1.0
MAX_STEP_DIST_M = 2.0
MAX_YAW_DELTA_DEG = 90.0

RESIZE_WH = (448, 448)

OUT_DIR = Path("./vlm_drone_3d_outputs")
RESIZED_IMAGE_PATH = OUT_DIR / "input_resized.jpg"
VIS_RESIZED_PATH = OUT_DIR / "waypoint_3d_vis_resized.jpg"
VIS_ORIGINAL_PATH = OUT_DIR / "waypoint_3d_vis_original.jpg"
JSON_PATH = OUT_DIR / "waypoint_3d_result.json"

PLANNER_REASONING = False
EXECUTOR_REASONING = False


# ======================
# Utility
# ======================

def get_original_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def resize_image(image_path: Path, out_path: Path, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def _json_root(data: Any) -> Optional[Dict[str, Any]]:
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
) -> Tuple[str, float]:
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


def _parse_compact_yaw(raw: Any) -> Tuple[str, float]:
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


def format_short_cmd(
    forward: str, forward_m: float,
    lateral: str, lateral_m: float,
    vertical: str, vertical_m: float,
    yaw_turn: str, yaw_magnitude_deg: float,
) -> str:
    parts: List[str] = []
    if forward != "none" and forward_m > 0:
        parts.append(f"{'F' if forward == 'forward' else 'B'}{forward_m:.1f}")
    if lateral != "none" and lateral_m > 0:
        parts.append(f"L{lateral_m:.1f}" if lateral == "left" else f"R{lateral_m:.1f}")
    if vertical != "none" and vertical_m > 0:
        parts.append(f"U{vertical_m:.1f}" if vertical == "up" else f"D{vertical_m:.1f}")
    if yaw_turn != "none" and yaw_magnitude_deg > 0:
        parts.append(f"Y{'L' if yaw_turn == 'left' else 'R'}{yaw_magnitude_deg:.0f}")
    return " ".join(parts) if parts else "HOLD"


def _assemble_relative_3d(
    forward: str, forward_m: float,
    lateral: str, lateral_m: float,
    vertical: str, vertical_m: float,
) -> List[float]:
    xyz = [
        _axis_sign(forward, "forward", "back") * forward_m,
        _axis_sign(lateral, "left", "right") * lateral_m,
        _axis_sign(vertical, "up", "down") * vertical_m,
    ]
    dist = (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
    if dist > MAX_STEP_DIST_M and dist > 1e-6:
        scale = MAX_STEP_DIST_M / dist
        xyz = [v * scale for v in xyz]
    return xyz


def _extract_reasoning(data: Any, raw_text: str) -> str:
    root = _json_root(data)
    if root:
        for key in ("reasoning", "reason"):
            if root.get(key):
                return str(root[key]).strip()
    m = re.search(r'"(?:reasoning|reason)"\s*:\s*"([^"]+)"', raw_text)
    return m.group(1).strip() if m else "(no reasoning parsed)"


def _apply_motion(
    forward: str, forward_m: float,
    lateral: str, lateral_m: float,
    vertical: str, vertical_m: float,
    yaw_turn: str, yaw_magnitude_deg: float,
    compact: Optional[Dict[str, str]] = None,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    mag = float(yaw_magnitude_deg or 0.0)
    yaw_delta_deg = _axis_sign(yaw_turn, "left", "right") * min(mag, MAX_YAW_DELTA_DEG)
    out: Dict[str, Any] = {
        "forward": forward,
        "forward_m": forward_m,
        "lateral": lateral,
        "lateral_m": lateral_m,
        "vertical": vertical,
        "vertical_m": vertical_m,
        "yaw_turn": yaw_turn,
        "yaw_magnitude_deg": mag,
        "relative_3d_m": _assemble_relative_3d(forward, forward_m, lateral, lateral_m, vertical, vertical_m),
        "yaw_delta_deg": yaw_delta_deg,
        "short_cmd": format_short_cmd(
            forward, forward_m, lateral, lateral_m, vertical, vertical_m, yaw_turn, mag,
        ),
    }
    if compact:
        out["compact"] = compact
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _extract_compact(data: Any, raw_text: str) -> Dict[str, str]:
    root = _json_root(data)
    if root and any(k in root for k in ("fwd", "lat", "vert", "yaw")):
        return {
            "fwd": str(root.get("fwd", "0")),
            "lat": str(root.get("lat", "0")),
            "vert": str(root.get("vert", root.get("ver", "0"))),
            "yaw": str(root.get("yaw", "0")),
        }
    compact: Dict[str, str] = {}
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


def parse_executor_output(data: Any, raw_text: str, with_reasoning: bool = False) -> Dict[str, Any]:
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
        forward, forward_m, lateral, lateral_m, vertical, vertical_m,
        yaw_turn, yaw_magnitude_deg, compact, reasoning,
    )


def parse_planner_output(data: Any, raw_text: str, with_reasoning: bool = True) -> Dict[str, Any]:
    root = _json_root(data)
    subtask = ""
    reasoning = ""
    scene_summary = ""
    target_visible: Optional[bool] = None

    if root:
        subtask = str(root.get("subtask", "")).strip()
        if with_reasoning:
            reasoning = str(root.get("reasoning", root.get("reason", ""))).strip()
        scene_summary = str(root.get("scene_summary", "")).strip()
        if "target_visible" in root:
            val = root["target_visible"]
            target_visible = val if isinstance(val, bool) else str(val).lower() in ("true", "yes", "1")

    if not subtask:
        m = re.search(r'"subtask"\s*:\s*"([^"]+)"', raw_text)
        if m:
            subtask = m.group(1).strip()
    if with_reasoning and not reasoning:
        m = re.search(r'"(?:reasoning|reason)"\s*:\s*"([^"]+)"', raw_text)
        if m:
            reasoning = m.group(1).strip()
    if not scene_summary:
        m = re.search(r'"scene_summary"\s*:\s*"([^"]+)"', raw_text)
        if m:
            scene_summary = m.group(1).strip()
    if target_visible is None:
        m = re.search(r'"target_visible"\s*:\s*(true|false)', raw_text, re.I)
        if m:
            target_visible = m.group(1).lower() == "true"

    out: Dict[str, Any] = {
        "target_visible": target_visible,
        "scene_summary": scene_summary or "(none)",
        "subtask": subtask or "Hold position and scan the scene until the next target becomes visible.",
    }
    if with_reasoning:
        out["reasoning"] = reasoning or "(no reasoning parsed)"
    return out


def build_planner_prompt(rw: int, rh: int, with_reasoning: bool = True) -> str:
    json_fmt = (
        '{"target_visible":true|false,"scene_summary":"what you see",'
        '"subtask":"action + object + until/stop condition"'
    )
    if with_reasoning:
        json_fmt += ',"reasoning":"why from the image"}'
    else:
        json_fmt += "}"
    return (
        f"Drone mission planner. Fwd RGB {rw}x{rh}.\n"
        f"Overall mission: {MISSION_PROMPT}\n"
        f"Horizon: next {HORIZON_S:.0f}s only.\n"
        "Analyze the IMAGE honestly. Do NOT invent objects not visible.\n"
        "Output ONE subtask for the next ~1s. Write it as a complete instruction, not a label.\n"
        "Subtask MUST include: (1) action — what to do; (2) object/target — what to act on; "
        "(3) stop condition — until what state is reached or what is achieved in this step.\n"
        "Example pattern: \"Move forward toward the red plane until it fills more of the center view.\"\n"
        "If mission target is NOT in view, subtask = search/hold/scan with clear object and stop condition; "
        "do NOT pretend to approach an unseen target.\n"
        f"JSON only:\n{json_fmt}"
    )


def build_executor_prompt(rw: int, rh: int, planner: Dict[str, Any], with_reasoning: bool = False) -> str:
    fmt = '{"fwd":"F<m>","lat":"L<m>|R<m>|0","vert":"U<m>|D<m>|0","yaw":"L<deg>|R<deg>|0"'
    if with_reasoning:
        fmt += ',"reasoning":"1-2 sentences: how motion fulfills the subtask from the image"}'
    else:
        fmt += "}"
    return (
        f"Drone motion executor. Fwd RGB {rw}x{rh}.\n"
        f"Subtask (next {HORIZON_S:.0f}s): {planner['subtask']}\n"
        "Execute ONLY this subtask. Do NOT infer any other mission.\n"
        "Body frame: +X forward, +Y left, +Z up. Camera looks along +X.\n"
        "Motion heuristics (pick what matches the subtask):\n"
        "- Approach / go toward / beside a target: reduce distance to where the target sits in the image; "
        "combine fwd with lat/yaw on the side the target appears; if already close, use smaller magnitudes; "
        "keep a safe standoff — do not drive through the target.\n"
        "- Search / scan / locate: if the target is not in view, slow clockwise in-place yaw (yaw R) with "
        "fwd/lat/vert near zero to sweep the scene; if partially visible at an edge, yaw or lat toward that edge.\n"
        "- Avoid obstacle / keep safe distance: if something blocks the path or is too close ahead, "
        "prefer lat away from it or vert up to clear it; reduce fwd or use B if moving forward would collide; "
        "steer around rather than through.\n"
        "- Hold / wait: all axes zero unless subtask asks for small adjustment.\n"
        "\n"
        "JSON only. Motion keys: fwd, lat, vert, yaw. Token=letter+magnitude: F/B m, L/R m, U/D m, yaw L/R deg; 0=none.\n"
        "Estimate magnitudes from target size, distance, and clearance in the image; do NOT copy placeholder values.\n"
        f"Format: {fmt}"
    )


def call_vlm(
    client: OpenAI,
    model: str,
    image_url: str,
    prompt: str,
    exact_pixels: int,
    max_tokens: int = 128,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "extra_fields": {
                            "mm_processor_kwargs": {
                                "min_pixels": exact_pixels,
                                "max_pixels": exact_pixels,
                            }
                        },
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def load_font(size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [text] if text else []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if text_width(draw, test, font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_overlay(
    image_path: Path,
    planner: Dict[str, Any],
    result_3d: Dict[str, Any],
    out_path: Path,
    planner_reasoning: bool = True,
    executor_reasoning: bool = False,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    pad = max(8, min(w, h) // 120)
    title_font = load_font(max(16, min(w, h) // 50))
    body_font = load_font(max(13, min(w, h) // 65))
    small_font = load_font(max(11, min(w, h) // 80))
    panel_w = min(w - 2 * pad, max(280, w * 2 // 3))
    max_text_w = panel_w - 2 * pad

    xyz = result_3d["relative_3d_m"]
    accent = (100, 220, 255)
    vis = planner.get("target_visible")
    vis_s = "?" if vis is None else ("yes" if vis else "no")

    lines_meta: List[Tuple[str, Tuple[int, int, int], ImageFont.ImageFont]] = [
        (f"Mission: {MISSION_PROMPT}", (255, 220, 80), title_font),
        (f"Next {HORIZON_S:.0f}s | +X fwd +Y left +Z up", (180, 180, 180), small_font),
        ("", (255, 255, 255), body_font),
        (f"target_visible: {vis_s}", (180, 255, 180), small_font),
    ]
    for line in wrap_text(draw, planner.get("scene_summary", ""), small_font, max_text_w):
        lines_meta.append((f"see: {line}", (200, 200, 200), small_font))
    lines_meta.append(("", (255, 255, 255), body_font))
    lines_meta.append(("Subtask:", (255, 200, 100), body_font))
    for line in wrap_text(draw, planner["subtask"], small_font, max_text_w):
        lines_meta.append((line, (255, 220, 120), small_font))
    if planner_reasoning and planner.get("reasoning"):
        lines_meta.append(("Planner reasoning:", (255, 200, 100), body_font))
        for line in wrap_text(draw, planner["reasoning"], small_font, max_text_w):
            lines_meta.append((line, (220, 220, 220), small_font))
    lines_meta.append(("", (255, 255, 255), body_font))
    lines_meta.append(("Executor:", (255, 200, 100), body_font))
    if executor_reasoning and result_3d.get("reasoning"):
        lines_meta.append(("Exec reasoning:", (255, 200, 100), body_font))
        for line in wrap_text(draw, result_3d["reasoning"], small_font, max_text_w):
            lines_meta.append((line, (220, 220, 220), small_font))
    if result_3d.get("compact"):
        c = result_3d["compact"]
        lines_meta.append((
            f'{{"fwd":"{c["fwd"]}","lat":"{c["lat"]}","vert":"{c["vert"]}","yaw":"{c["yaw"]}"}}',
            accent, body_font,
        ))
    lines_meta.extend([
        (f"CMD: {result_3d['short_cmd']}", (255, 200, 100), body_font),
        (f"rel_3d = ({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f}) m", accent, body_font),
        (f"yaw_delta = {result_3d['yaw_delta_deg']:+.1f} deg", accent, body_font),
    ])

    line_heights = [
        (font.size if hasattr(font, "size") else 12) if text else (font.size // 2 if hasattr(font, "size") else 6)
        for text, _, font in lines_meta
    ]
    panel_h = min(pad + sum(line_heights) + pad + (len(lines_meta) - 1) * 2, h - 2 * pad)
    draw.rectangle([pad, pad, pad + panel_w, pad + panel_h], fill=(0, 0, 0))
    ty = pad + pad
    for (text, color, font), lh in zip(lines_meta, line_heights):
        if text:
            draw.text((pad + pad, ty), text, fill=color, font=font)
        ty += lh + 2
        if ty > pad + panel_h - pad:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def run(planner_reasoning: bool = PLANNER_REASONING, executor_reasoning: bool = EXECUTOR_REASONING) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ow, oh = get_original_size(IMAGE_PATH)
    resize_image(IMAGE_PATH, RESIZED_IMAGE_PATH, RESIZE_WH)
    rw, rh = RESIZE_WH

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    model = client.models.list().data[0].id
    image_url = image_to_data_url(RESIZED_IMAGE_PATH)
    exact_pixels = rw * rh
    t_total = time.perf_counter()

    print("=== Step 1: Planner ===")
    planner_prompt = build_planner_prompt(rw, rh, planner_reasoning)
    print(planner_prompt)

    t0 = time.perf_counter()
    planner_raw = call_vlm(client, model, image_url, planner_prompt, exact_pixels, max_tokens=512)
    t1 = time.perf_counter()
    print("\n--- planner output ---")
    print(planner_raw)

    planner = parse_planner_output(extract_json(planner_raw), planner_raw, planner_reasoning)
    print("\n--- planner parsed ---")
    print(f"  target_visible = {planner['target_visible']}")
    print(f"  scene_summary = {planner['scene_summary']}")
    print(f"  subtask = {planner['subtask']}")
    if planner_reasoning:
        print(f"  reasoning = {planner.get('reasoning', '')}")

    print("\n=== Step 2: Executor ===")
    exec_prompt = build_executor_prompt(rw, rh, planner, executor_reasoning)
    print(exec_prompt)

    t2 = time.perf_counter()
    exec_raw = call_vlm(
        client, model, image_url, exec_prompt, exact_pixels,
        max_tokens=256 if executor_reasoning else 128,
    )
    t3 = time.perf_counter()
    print("\n--- executor output ---")
    print(exec_raw)

    result_3d = parse_executor_output(extract_json(exec_raw), exec_raw, executor_reasoning)
    t4 = time.perf_counter()
    xyz = result_3d["relative_3d_m"]
    print("\n--- executor parsed ---")
    if executor_reasoning:
        print(f"  reasoning = {result_3d.get('reasoning', '')}")
    if result_3d.get("compact"):
        print(f"  model: {result_3d['compact']}")
    print(f"  cmd = {result_3d['short_cmd']}")
    print(f"  rel_3d = ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m")
    print(f"  yaw_delta = {result_3d['yaw_delta_deg']:+.1f} deg")

    timing = {
        "planner_ms": round((t1 - t0) * 1000, 1),
        "executor_ms": round((t3 - t2) * 1000, 1),
        "parse_ms": round((t4 - t3) * 1000, 1),
        "total_ms": round((t4 - t_total) * 1000, 1),
    }
    print(f"\n--- timing ---")
    print(f"  planner: {timing['planner_ms']:.1f} ms")
    print(f"  executor: {timing['executor_ms']:.1f} ms")
    print(f"  parse: {timing['parse_ms']:.1f} ms")
    print(f"  total: {timing['total_ms']:.1f} ms")

    result = {
        "mission": MISSION_PROMPT,
        "horizon_s": HORIZON_S,
        "planner_reasoning": planner_reasoning,
        "executor_reasoning": executor_reasoning,
        "timing_ms": timing,
        "planner_raw_output": planner_raw,
        "planner": planner,
        "executor_raw_output": exec_raw,
        "result_3d": result_3d,
        "image_path": str(IMAGE_PATH),
        "original_size": [ow, oh],
        "resized_size": [rw, rh],
    }
    draw_overlay(RESIZED_IMAGE_PATH, planner, result_3d, VIS_RESIZED_PATH, planner_reasoning, executor_reasoning)
    draw_overlay(IMAGE_PATH, planner, result_3d, VIS_ORIGINAL_PATH, planner_reasoning, executor_reasoning)
    timing["total_with_io_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    result["timing_ms"] = timing

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {JSON_PATH}")
    print(f"Saved visualization: {VIS_RESIZED_PATH}, {VIS_ORIGINAL_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM drone planner + executor")
    parser.add_argument(
        "--planner-reasoning",
        action=argparse.BooleanOptionalAction,
        default=PLANNER_REASONING,
        help="planner outputs reasoning field (default: on)",
    )
    parser.add_argument(
        "--executor-reasoning",
        action=argparse.BooleanOptionalAction,
        default=EXECUTOR_REASONING,
        help="executor outputs reasoning field (default: off)",
    )
    args = parser.parse_args()
    run(planner_reasoning=args.planner_reasoning, executor_reasoning=args.executor_reasoning)
