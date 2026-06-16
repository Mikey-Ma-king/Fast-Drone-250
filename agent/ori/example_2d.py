import base64
import json
import re
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

# 无人机任务（改这里即可换任务）
MISSION_PROMPT = "Avoid the crowd and move forward; collisions are not allowed"
HORIZON_S = 1.0  # 每步子任务对应的 ~1s 飞行方向
MAX_SUBTASKS = 5
MAX_FORWARD_SPEED_MPS = 2.0  # 室内前进速度上限 (m/s)
OBJECT_LABEL = "safe_1s_target"

RESIZE_WH = (448, 448)
NORM_SCALE = 1000.0

OUT_DIR = Path("./vlm_drone_waypoint_outputs")
RESIZED_IMAGE_PATH = OUT_DIR / "input_resized.jpg"
VIS_RESIZED_PATH = OUT_DIR / "waypoint_path_vis_resized.jpg"
VIS_ORIGINAL_PATH = OUT_DIR / "waypoint_path_vis_original.jpg"
JSON_PATH = OUT_DIR / "waypoint_result.json"

PATH_COLORS = [
    (255, 64, 64),
    (255, 140, 0),
    (255, 200, 0),
    (100, 220, 100),
    (64, 180, 255),
]


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
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def parse_points_regex(text: str) -> List[List[float]]:
    points: List[List[float]] = []
    for x_s, y_s in re.findall(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)", text):
        points.append([float(x_s), float(y_s)])
    for x_s, y_s in re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text):
        points.append([float(x_s), float(y_s)])
    return points


def normalize_subtasks(data: Any, raw_text: str) -> List[Dict[str, Any]]:
    """统一为 [{"step": 1, "label": ..., "description": ...}, ...]。"""
    subtasks: List[Dict[str, Any]] = []

    if data is not None:
        if isinstance(data, dict):
            if "subtasks" in data and isinstance(data["subtasks"], list):
                data = data["subtasks"]
            elif "steps" in data and isinstance(data["steps"], list):
                data = data["steps"]
            elif "plan" in data and isinstance(data["plan"], list):
                data = data["plan"]

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    subtasks.append({
                        "step": i + 1,
                        "label": f"step_{i + 1}",
                        "description": item,
                    })
                    continue
                if not isinstance(item, dict):
                    continue
                step = item.get("step", item.get("index", i + 1))
                label = item.get("label", item.get("name", f"step_{step}"))
                desc = item.get("description", item.get("goal", item.get("task", "")))
                if desc:
                    subtasks.append({
                        "step": int(step),
                        "label": str(label),
                        "description": str(desc),
                    })

    if not subtasks:
        for i, line in enumerate(re.findall(r'"description"\s*:\s*"([^"]+)"', raw_text)):
            subtasks.append({
                "step": i + 1,
                "label": f"step_{i + 1}",
                "description": line,
            })

    subtasks.sort(key=lambda s: s["step"])
    return subtasks[:MAX_SUBTASKS]


def normalize_waypoints(data: Any, raw_text: str, default_label: str) -> List[Dict[str, Any]]:
    """统一为 [{"label": ..., "center": [x, y]}, ...]（模型原始数值）。"""
    waypoints: List[Dict[str, Any]] = []

    if data is not None:
        if isinstance(data, dict):
            if "waypoints" in data and isinstance(data["waypoints"], list):
                data = data["waypoints"]
            elif "points" in data and isinstance(data["points"], list):
                label = data.get("object", data.get("label", default_label))
                for pt in data["points"]:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        waypoints.append({"label": label, "center": [pt[0], pt[1]]})
                return waypoints[:1]
            elif any(k in data for k in ("center_2d", "center", "point", "waypoint_2d", "waypoint")):
                data = [data]
            elif "detections" in data and isinstance(data["detections"], list):
                data = data["detections"]

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                label = item.get("label", item.get("object", default_label))
                for key in ("waypoint_2d", "center_2d", "waypoint", "center", "point"):
                    if key in item:
                        c = item[key]
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            waypoints.append({"label": label, "center": [c[0], c[1]]})
                        break

    if not waypoints:
        for pt in parse_points_regex(raw_text):
            waypoints.append({"label": default_label, "center": pt})

    return waypoints[:1]


def _extract_speed(value: Any) -> Optional[float]:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    if speed < 0:
        speed = 0.0
    return min(speed, MAX_FORWARD_SPEED_MPS)


def parse_step_output(data: Any, raw_text: str, default_label: str) -> Dict[str, Any]:
    """解析单步输出：航点 + 前进速度 (m/s)。"""
    waypoints = normalize_waypoints(data, raw_text, default_label)
    center = waypoints[0]["center"] if waypoints else None
    label = waypoints[0]["label"] if waypoints else default_label
    speed: Optional[float] = None

    speed_keys = ("forward_speed_mps", "forward_speed", "speed_mps", "speed")
    candidates: List[Any] = []
    if isinstance(data, dict):
        candidates.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                candidates.append(item)

    for item in candidates:
        for key in speed_keys:
            if key in item:
                speed = _extract_speed(item[key])
                if speed is not None:
                    break
        if speed is not None:
            break

    if speed is None:
        m = re.search(
            r'"(?:forward_speed_mps|forward_speed|speed_mps|speed)"\s*:\s*(\d+(?:\.\d+)?)',
            raw_text,
        )
        if m:
            speed = _extract_speed(m.group(1))

    if speed is None:
        speed = 0.5

    return {
        "label": label,
        "center": center,
        "forward_speed_mps": speed,
    }


def is_normalized_coord(x: float, y: float, width: int, height: int) -> bool:
    return max(x, y) > max(width, height) + 10


def map_to_pixels(center: List[Any], width: int, height: int) -> Optional[Tuple[int, int]]:
    try:
        x, y = float(center[0]), float(center[1])
    except Exception:
        return None

    if is_normalized_coord(x, y, width, height):
        x = x * width / NORM_SCALE
        y = y * height / NORM_SCALE

    x = int(round(max(0, min(width - 1, x))))
    y = int(round(max(0, min(height - 1, y))))
    return x, y


def map_resized_to_original(
    x: int, y: int, rw: int, rh: int, ow: int, oh: int,
) -> Tuple[int, int]:
    ox = int(round(x / max(rw - 1, 1) * (ow - 1)))
    oy = int(round(y / max(rh - 1, 1) * (oh - 1)))
    return max(0, min(ox, ow - 1)), max(0, min(oy, oh - 1))


def enrich_waypoints(
    waypoints: List[Dict[str, Any]], rw: int, rh: int, ow: int, oh: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for det in waypoints:
        raw = det["center"]
        px = map_to_pixels(raw, rw, rh)
        if px is None:
            continue
        rx, ry = px
        ox, oy = map_resized_to_original(rx, ry, rw, rh, ow, oh)
        out.append({
            **det,
            "waypoint_raw": raw,
            "waypoint_resized_px": [rx, ry],
            "waypoint_original_px": [ox, oy],
            "coord_was_normalized": is_normalized_coord(float(raw[0]), float(raw[1]), rw, rh),
        })
    return out


def drone_origin_px(width: int, height: int) -> Tuple[int, int]:
    """前视相机近似机位：画面底部中央。"""
    return width // 2, height - 1


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
        return [text]
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


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: List[str],
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    line_gap: int = 2,
) -> int:
    """绘制带深色底色的文字块，返回占用高度。"""
    if not lines:
        return 0
    pad = 4
    widths = [text_width(draw, line, font) for line in lines]
    block_w = max(widths) + pad * 2
    line_h = font.size if hasattr(font, "size") else 12
    block_h = len(lines) * line_h + (len(lines) - 1) * line_gap + pad * 2
    draw.rectangle([x, y, x + block_w, y + block_h], fill=(0, 0, 0))
    ty = y + pad
    for line in lines:
        draw.text((x + pad, ty), line, fill=fill, font=font)
        ty += line_h + line_gap
    return block_h


def draw_path_curve(
    image_path: Path,
    path_points: List[Tuple[int, int]],
    step_infos: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    origin = drone_origin_px(w, h)
    r = max(8, min(w, h) // 80)
    font = load_font(max(14, r))
    small_font = load_font(max(11, r - 2))
    origin_color = (64, 200, 64)
    max_text_w = max(120, w // 3)

    chain = [origin] + path_points
    for i in range(len(chain) - 1):
        color = PATH_COLORS[i % len(PATH_COLORS)]
        draw.line([chain[i], chain[i + 1]], fill=color, width=max(3, r // 3))

    ox, oy = origin
    draw.ellipse([ox - r // 2, oy - r // 2, ox + r // 2, oy + r // 2], outline=origin_color, width=2)
    draw.text((ox + 6, oy - r - 18), "drone", fill=origin_color, font=font)

    for i, (px, py) in enumerate(path_points):
        info = step_infos[i] if i < len(step_infos) else {}
        color = PATH_COLORS[i % len(PATH_COLORS)]
        label = info.get("label", f"step_{i + 1}")
        desc = info.get("description", "")
        speed = info.get("forward_speed_mps")

        draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=max(2, r // 4))
        draw.line([(px - r, py), (px + r, py)], fill=color, width=2)
        draw.line([(px, py - r), (px, py + r)], fill=color, width=2)

        desc_lines = wrap_text(draw, desc, small_font, max_text_w) if desc else []
        speed_line = f"v={speed:.2f} m/s" if speed is not None else "v=? m/s"
        block_lines = [f"Step {i + 1}: {label}"] + desc_lines + [speed_line]

        tx = px + r + 8 if px < w * 2 // 3 else max(8, px - max_text_w - r - 8)
        ty = max(8, py - r - len(block_lines) * (small_font.size + 2))
        draw_text_block(draw, tx, ty, block_lines, small_font, color)

    draw.text((8, 8), f"{HORIZON_S:.0f}s path ({len(path_points)} steps)", fill=(255, 200, 0), font=font)
    y_text = 8 + max(14, r) + 4
    for i, info in enumerate(step_infos[:len(path_points)]):
        color = PATH_COLORS[i % len(PATH_COLORS)]
        desc = info.get("description", "")
        speed = info.get("forward_speed_mps")
        short = desc if len(desc) <= 50 else desc[:47] + "..."
        speed_s = f"{speed:.2f} m/s" if speed is not None else "?"
        draw.text((8, y_text), f"{i + 1}. {short}  [{speed_s}]", fill=color, font=small_font)
        y_text += max(11, r - 2) + 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def build_plan_prompt(rw: int, rh: int) -> str:
    return f"""
You are a drone mission planner.

Sensor:
- Forward-facing RGB camera image (already resized to {rw}x{rh}).

Mission:
{MISSION_PROMPT}

Task:
Break this mission into an ordered sequence of subtasks — intermediate goals the drone should reach in order.
Each subtask is one short flight segment (~{HORIZON_S:.0f}s of motion at typical indoor speed).
Think: first detour here, then advance there — a coarse forward path that stays clear of people and obstacles.

Rules:
- Return 2 to {MAX_SUBTASKS} subtasks, ordered from nearest/safest first to final goal.
- Each subtask must be achievable from the previous one with zero collision (no contact with people, walls, furniture, or obstacles).
- Use spatial language tied to what you see (crowd, people, corridor, open lane, forward path, etc.).
- Do NOT output image coordinates yet — only textual subtask goals.

Output only valid JSON. Do not output markdown. Do not explain.

Return format:
{{
  "mission": "{MISSION_PROMPT}",
  "subtasks": [
    {{"step": 1, "label": "detour", "description": "Side-step around the crowd into a clear lane"}},
    {{"step": 2, "label": "advance", "description": "Move forward along the open path ahead"}},
    {{"step": 3, "label": "continue", "description": "Keep advancing forward while maintaining clearance from people"}}
  ]
}}
"""


def build_step_prompt(
    rw: int,
    rh: int,
    subtask: Dict[str, Any],
    step_index: int,
    total_steps: int,
    prior_descriptions: List[str],
) -> str:
    step = subtask["step"]
    label = subtask["label"]
    desc = subtask["description"]
    prior_block = ""
    if prior_descriptions:
        lines = "\n".join(f"  - Step {i + 1}: {d}" for i, d in enumerate(prior_descriptions))
        prior_block = f"""
Already planned earlier steps (for context only — you are planning step {step} now):
{lines}
"""

    return f"""
You are a drone flight executor.

Sensor:
- Forward-facing RGB camera image (already resized to {rw}x{rh}).

Overall mission:
{MISSION_PROMPT}

Current subtask (step {step} of {total_steps}):
- label: {label}
- goal: {desc}
{prior_block}
Planning horizon:
- Only the next {HORIZON_S:.0f} second(s) of flight for THIS subtask.
- Return ONE image point along a collision-free direction toward this subtask goal.
- Collisions are strictly forbidden: keep safe clearance from people, walls, furniture, and all obstacles.
- Prefer detouring around the crowd rather than flying over or through it.
- You MUST always output exactly one point and one forward speed; never return an empty list.
- forward_speed_mps: recommended forward speed for this ~{HORIZON_S:.0f}s segment, in m/s (0.0 to {MAX_FORWARD_SPEED_MPS:.1f}).
  Use lower speed near people/obstacles; higher speed only in clear open space.

Coordinate rule:
- origin is the top-left corner
- x increases to the right
- y increases downward
- the point is [x, y]

Output only valid JSON. Do not output markdown. Do not explain.

Return format (always return exactly one item):
[
  {{"step": {step}, "label": "{label}", "waypoint_2d": [x, y], "forward_speed_mps": 0.5}}
]
"""


def call_vlm(
    client: OpenAI,
    image_url: str,
    prompt: str,
    exact_pixels: int,
    max_tokens: int = 512,
) -> str:
    resp = client.chat.completions.create(
        model=client.models.list().data[0].id,
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


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ow, oh = get_original_size(IMAGE_PATH)
    resize_image(IMAGE_PATH, RESIZED_IMAGE_PATH, RESIZE_WH)
    rw, rh = RESIZE_WH

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_url = image_to_data_url(RESIZED_IMAGE_PATH)
    exact_pixels = rw * rh

    # ----- Step 1: plan subtasks -----
    plan_prompt = build_plan_prompt(rw, rh)
    print("=== Step 1: mission planning ===")
    print("--- prompt ---")
    print(plan_prompt)

    plan_raw = call_vlm(client, image_url, plan_prompt, exact_pixels, max_tokens=768)
    print("\n--- model output ---")
    print(plan_raw)

    plan_parsed = extract_json(plan_raw)
    subtasks = normalize_subtasks(plan_parsed, plan_raw)
    if not subtasks:
        subtasks = [{
            "step": 1,
            "label": "direct",
            "description": MISSION_PROMPT,
        }]
        print("\nWarning: no subtasks parsed; fallback to single-step mission.")

    print(f"\n--- parsed subtasks ({len(subtasks)}) ---")
    for st in subtasks:
        print(f"  #{st['step']} [{st['label']}] {st['description']}")

    # ----- Step 2: one waypoint per subtask -----
    path_steps: List[Dict[str, Any]] = []
    prior_descriptions: List[str] = []

    print("\n=== Step 2: waypoint per subtask ===")
    for i, subtask in enumerate(subtasks):
        step_prompt = build_step_prompt(
            rw, rh, subtask, i, len(subtasks), prior_descriptions,
        )
        print(f"\n--- step {subtask['step']} prompt ---")
        print(step_prompt)

        step_raw = call_vlm(client, image_url, step_prompt, exact_pixels)
        print(f"\n--- step {subtask['step']} model output ---")
        print(step_raw)

        step_parsed = extract_json(step_raw)
        label = f"step_{subtask['step']}_{subtask['label']}"
        step_out = parse_step_output(step_parsed, step_raw, label)
        waypoints: List[Dict[str, Any]] = []
        if step_out["center"] is not None:
            waypoints = enrich_waypoints(
                [{"label": step_out["label"], "center": step_out["center"]}],
                rw, rh, ow, oh,
            )
            if waypoints:
                waypoints[0]["forward_speed_mps"] = step_out["forward_speed_mps"]

        entry: Dict[str, Any] = {
            "subtask": subtask,
            "raw_output": step_raw,
            "forward_speed_mps": step_out["forward_speed_mps"],
            "waypoint": waypoints[0] if waypoints else None,
        }
        path_steps.append(entry)

        if waypoints:
            wp = waypoints[0]
            flag = "0~1000" if wp["coord_was_normalized"] else "pixel"
            print(
                f"  mapped raw={wp['waypoint_raw']} ({flag}) "
                f"-> resized={wp['waypoint_resized_px']} -> original={wp['waypoint_original_px']} "
                f"speed={wp.get('forward_speed_mps', step_out['forward_speed_mps']):.2f} m/s"
            )
        else:
            print("  Warning: no waypoint parsed for this step.")

        prior_descriptions.append(subtask["description"])

    valid_steps = [s for s in path_steps if s.get("waypoint")]
    result = {
        "mission": MISSION_PROMPT,
        "horizon_s": HORIZON_S,
        "plan_raw_output": plan_raw,
        "subtasks": subtasks,
        "path_steps": path_steps,
        "image_path": str(IMAGE_PATH),
        "original_size": [ow, oh],
        "resized_size": [rw, rh],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {JSON_PATH}")

    if not valid_steps:
        print("No waypoints parsed; skip visualization.")
        return

    resized_pts = [tuple(s["waypoint"]["waypoint_resized_px"]) for s in valid_steps]
    original_pts = [tuple(s["waypoint"]["waypoint_original_px"]) for s in valid_steps]
    step_infos = [{
        "label": s["subtask"]["label"],
        "description": s["subtask"]["description"],
        "forward_speed_mps": s.get("forward_speed_mps"),
    } for s in valid_steps]

    draw_path_curve(RESIZED_IMAGE_PATH, resized_pts, step_infos, VIS_RESIZED_PATH)
    draw_path_curve(IMAGE_PATH, original_pts, step_infos, VIS_ORIGINAL_PATH)
    print(f"Saved visualization (resized): {VIS_RESIZED_PATH}")
    print(f"Saved visualization (original): {VIS_ORIGINAL_PATH}")


if __name__ == "__main__":
    run()
