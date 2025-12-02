"""
Lightweight helpers for drawing affordance-related overlays on perception snapshots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.utils.coordinates import normalized_to_pixel  # noqa: E402

PALETTE = [
    "#f94144",
    "#f3722c",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#577590",
    "#277da1",
    "#9b5de5",
    "#f15bb5",
    "#e07a5f",
]


def visualize_primitives(plan, world_dir: Path, output_dir: Path, snapshot_id: Optional[str] = None) -> None:
    """
    Draw primitives that reference interaction points onto the plan's snapshot image.
    """
    snapshot_id = snapshot_id or getattr(plan, "source_snapshot_id", None) or None
    if not snapshot_id:
        state_path = world_dir / "state.json"
        if state_path.exists():
            try:
                state_payload = json.loads(state_path.read_text())
                snapshot_id = state_payload.get("last_snapshot_id")
            except Exception:
                snapshot_id = None
    if not snapshot_id:
        print("No snapshot_id available; skipping visualization.")
        return

    img_path = world_dir / "perception_pool" / "snapshots" / snapshot_id / "color.png"
    if not img_path.exists():
        print(f"No snapshot image found at {img_path}; skipping visualization.")
        return

    print(f"Visualizing affordance targets on snapshot {snapshot_id}")
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    drew_any = False
    for idx, primitive in enumerate(plan.primitives):
        if not primitive.references.get("interaction_point"):
            continue
        pixel = _primitive_target_pixel(primitive)
        if not pixel:
            continue
        xy = _normalized_to_pixel_xy(pixel, image.size)
        if xy is None:
            continue
        x, y = xy
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="black")
        label = f"{idx}:{primitive.name}"
        draw.text((x + 6, y - 6), label, fill="yellow", stroke_width=1, stroke_fill="black")
        drew_any = True

    if not drew_any:
        print("No affordance-referencing primitives with pixel targets found; skipping visualization.")
        return

    vis_path = output_dir / f"{plan.action_name}_{snapshot_id}_affordances.png"
    image.save(vis_path)
    print(f"Wrote affordance visualization to {vis_path}")


def visualize_snapshot_affordances(snapshot_dir: Path, output_path: Path) -> bool:
    """
    Draw bounding boxes and affordance interaction points for each object in a snapshot.
    Returns True if any were drawn.
    """
    det_path = snapshot_dir / "detections.json"
    img_path = snapshot_dir / "color.png"
    if not (det_path.exists() and img_path.exists()):
        print(f"Missing detections or image for {snapshot_dir.name}; skipping.")
        return False

    detections = json.loads(det_path.read_text())
    objects = detections.get("objects", [])
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    drew_any = False
    for idx, obj in enumerate(objects):
        obj_id = obj.get("object_id", f"obj_{idx}")
        color = PALETTE[hash(obj_id) % len(PALETTE)]

        # Draw bounding box if available
        bbox = _resolve_bbox(obj, image.size)
        if bbox:
            y1, x1, y2, x2 = bbox
            top_left = _normalized_to_pixel_xy([y1, x1], image.size)
            bottom_right = _normalized_to_pixel_xy([y2, x2], image.size)
            if top_left and bottom_right:
                draw.rectangle([top_left, bottom_right], outline=color, width=2)
                label = f"{idx}:{obj_id}"
                draw.text((top_left[0] + 4, top_left[1] + 4), label, fill=color, stroke_width=1, stroke_fill="black")
                drew_any = True

        # Draw affordance points in the same color, prefixed by object index
        ips = (obj.get("interaction_points") or {}).items()
        for affordance, ip in ips:
            pos2d = ip.get("position_2d")
            if not (isinstance(pos2d, (list, tuple)) and len(pos2d) == 2):
                continue
            xy = _normalized_to_pixel_xy(pos2d, image.size)
            if xy is None:
                continue
            x, y = xy
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="black")
            label = f"{idx}:{affordance}"
            draw.text((x + 6, y - 6), label, fill=color, stroke_width=1, stroke_fill="black")
            drew_any = True

    if not drew_any:
        return False

    image.save(output_path)
    return True


def _primitive_target_pixel(primitive) -> Optional[Any]:
    """
    Resolve a target pixel from primitive metadata/parameters.
    Prefers resolved interaction point, falls back to target_pixel_yx (row, col).
    """
    meta_ip = primitive.metadata.get("resolved_interaction_point") if primitive.metadata else None
    if isinstance(meta_ip, dict):
        pos2d = meta_ip.get("position_2d")
        if pos2d and len(pos2d) == 2:
            return pos2d

    pix = primitive.parameters.get("target_pixel_yx") if primitive.parameters else None
    if pix and len(pix) == 2:
        # target_pixel_yx is [row, col]; convert to [x, y]
        return [pix[1], pix[0]]

    return None


def _normalized_to_pixel_xy(pos2d: Any, image_size: Any) -> Optional[tuple[int, int]]:
    """
    Convert stored [y, x] normalized (0-1000) to (x, y) pixel tuple.
    """
    if not (isinstance(pos2d, (list, tuple)) and len(pos2d) >= 2):
        return None
    width, height = image_size  # PIL returns (width, height)
    y, x = float(pos2d[0]), float(pos2d[1])

    # Stored as normalized [y, x] on 0-1000 scale.
    pixel_y, pixel_x = normalized_to_pixel([y, x], (height, width))
    return int(pixel_x), int(pixel_y)


def _resolve_bbox(obj: dict, image_size: Any) -> Optional[list[float]]:
    """
    Resolve a normalized [y1, x1, y2, x2] bbox for the object.
    Falls back to interaction point extents or center with padding if bbox is missing.
    """
    bbox = obj.get("bounding_box_2d")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    coords = []
    pos2d = obj.get("position_2d")
    if isinstance(pos2d, (list, tuple)) and len(pos2d) >= 2:
        coords.append([float(pos2d[0]), float(pos2d[1])])
    ips = (obj.get("interaction_points") or {}).values()
    for ip in ips:
        p = ip.get("position_2d")
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            coords.append([float(p[0]), float(p[1])])
    if not coords:
        return None

    ys = [c[0] for c in coords]
    xs = [c[1] for c in coords]
    pad = 30.0  # normalized padding
    y1 = max(0.0, min(ys) - pad)
    x1 = max(0.0, min(xs) - pad)
    y2 = min(1000.0, max(ys) + pad)
    x2 = min(1000.0, max(xs) + pad)
    return [y1, x1, y2, x2]
