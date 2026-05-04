"""
LLM-based contact relationship classifier.

Sends a single annotated scene image to a vision-capable LLM and asks it to
classify the spatial relationship between every pair of detected objects.
The result is a dict mapping (obj_a, obj_b) → ContactType that can replace or
override Phase-3 geometric classifications in compute_contact_graph.

Relationship definitions sent to the model
------------------------------------------
  supporting  — obj_a physically holds up obj_b (A is below B)
  nested      — obj_b is enclosed inside obj_a (e.g. object inside a bowl)
  none        — the two objects are not in a meaningful support/containment
                relationship
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .object_registry import DetectedObject

logger = logging.getLogger(__name__)

ContactType = str  # "supporting" | "nested" | "none"
LLMLabel = Tuple[ContactType, str, str]  # (relationship, directed_a, directed_b)

_VALID_TYPES = {"supporting", "nested", "none"}


@dataclass
class LLMContactResult:
    """Return value from LLM contact classifiers."""
    labels: Dict[Tuple[str, str], LLMLabel] = field(default_factory=dict)
    room_available: Dict[str, bool] = field(default_factory=dict)

_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant for a robot manipulation system.
You will be shown an RGB scene image with coloured mask overlays. \
Each detected object is labelled with its exact ID string.

You will be given a list of object IDs and a list of surface IDs. \
Return ONLY the relationships that actually exist — omit pairs with no \
meaningful relationship.

Relationship types:
  supporting  — A is the immediate surface directly beneath B (B rests on A).
  nested      — B is enclosed inside A (e.g. object inside a bowl or box).

Also assess each surface in surface_ids: does it have visible free space \
where a small object (~15 cm diameter) could be placed? \
Set room_available to true if yes, false if the surface appears fully occupied.

CRITICAL RULES:
- Use object IDs EXACTLY as given — do not rephrase, rename, or invent IDs.
- For "supporting": use the CLOSEST object directly beneath B. If a cup sits \
on a box on a table, output table supports box AND box supports cup — not \
table supports cup.
- Only include pairs where a real supporting or nested relationship exists.
- Respond with ONLY valid JSON. No markdown, no prose.

{"relationships": [{"a": "<id>", "b": "<id>", "relationship": "<label>"}, ...], \
"room_available": {"<surface_id>": true/false, ...}}
"""


_MASK_COLORS_BGR = [
    (214, 39,  40),  # tab10-0  red
    (255, 127,  14), # tab10-1  orange
    (44,  160,  44), # tab10-2  green
    (31,  119, 180), # tab10-3  blue
    (148, 103, 189), # tab10-4  purple
    (140,  86,  75), # tab10-5  brown
    (227, 119, 194), # tab10-6  pink
    (127, 127, 127), # tab10-7  grey
    (188, 189,  34), # tab10-8  olive
    (23,  190, 207), # tab10-9  teal
]


def _render_annotated_image(
    color_img: np.ndarray,
    obj_masks: Dict[str, np.ndarray],
) -> bytes:
    """Render scene image with semi-transparent mask overlays and centroid labels.

    No bounding boxes — the LLM should reason from the raw visual scene.
    Uses cv2 to avoid matplotlib backend conflicts.
    """
    import cv2

    canvas = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR).copy()
    overlay = canvas.copy()

    for idx, (oid, mask) in enumerate(obj_masks.items()):
        if mask is None or not mask.any():
            continue
        bgr = _MASK_COLORS_BGR[idx % len(_MASK_COLORS_BGR)]
        overlay[mask] = (
            overlay[mask] * 0.55 + np.array(bgr, dtype=np.float32) * 0.45
        ).astype(np.uint8)

        # Label at mask centroid
        ys, xs = np.where(mask)
        cx_px, cy_px = int(xs.mean()), int(ys.mean())
        (tw, th), _ = cv2.getTextSize(oid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(0, cx_px - tw // 2)
        ty = max(th + 2, cy_px)
        cv2.rectangle(overlay, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(overlay, oid, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    ok, buf_arr = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf_arr.tobytes()


def classify_contacts_with_llm(
    objects: List[DetectedObject],
    object_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    color_img: np.ndarray,
    obj_masks: Dict[str, np.ndarray],
    llm_client,
    surface_ids: Optional[List[str]] = None,
    debug_image_out: Optional[List[bytes]] = None,
) -> LLMContactResult:
    """Classify all pairwise contact relationships using a vision LLM.

    Also asks the model whether each surface has room to receive a displaced
    object (room_available), replacing the geometric surface-map computation.

    Args:
        objects: Detected objects in the scene.
        object_aabbs: Per-object world-frame AABB (min, max) tuples.
        color_img: RGB uint8 (H, W, 3) scene image.
        obj_masks: {object_id: bool ndarray (H, W)} segmentation masks.
        llm_client: Any LLMClient instance (must support multimodal generate).
        surface_ids: IDs of surface objects to assess for room_available.
        debug_image_out: If provided, the rendered PNG bytes are appended.

    Returns:
        LLMContactResult with labels and room_available dicts.
        Missing pairs fall back to geometric classification downstream.
    """
    from src.llm_interface.base import ImagePart, GenerateConfig

    obj_ids = [o.object_id for o in objects if o.object_id in object_aabbs]
    if len(obj_ids) < 2:
        return LLMContactResult()

    png_bytes = _render_annotated_image(color_img, obj_masks)

    if debug_image_out is not None:
        debug_image_out.append(png_bytes)

    surf_ids = surface_ids or []
    user_text = (
        f"Object IDs (use exactly as written): {json.dumps(obj_ids)}\n"
        f"Surface IDs (assess room_available for each): {json.dumps(surf_ids)}\n\n"
        f"List every supporting or nested relationship you can see in the image, "
        f"and for each surface ID indicate whether it has free space for a displaced object."
    )

    # Token budget: ~25 tokens per relationship + ~15 per surface
    max_tokens = max(512, len(obj_ids) * 25 + len(surf_ids) * 15)

    config = GenerateConfig(
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        system_instruction=_SYSTEM_PROMPT,
    )

    try:
        response = llm_client.generate(
            [ImagePart(data=png_bytes, mime_type="image/png"), user_text],
            config=config,
        )
    except Exception as exc:
        logger.error("LLM contact classification failed: %s", exc)
        return LLMContactResult()

    logger.info("LLM raw response:\n%s", response.text)
    return _parse_response(response.text, obj_ids, surf_ids)


def _parse_response(
    text: str,
    valid_ids: List[str],
    surface_ids: Optional[List[str]] = None,
) -> LLMContactResult:
    """Parse the LLM JSON response into a LLMContactResult.

    Only retains relationship entries where both IDs appear in valid_ids.
    room_available defaults to True for surfaces not mentioned.
    """
    labels: Dict[Tuple[str, str], LLMLabel] = {}
    room_available: Dict[str, bool] = {}
    valid_set = set(valid_ids)
    surf_set = set(surface_ids or [])

    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        relationships = data.get("relationships", [])
        llm_room = data.get("room_available", {})
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("LLM response JSON parse error: %s\nRaw: %s", exc, text[:500])
        return LLMContactResult()

    for item in relationships:
        a = str(item.get("a", ""))
        b = str(item.get("b", ""))
        rel = str(item.get("relationship", "none")).lower()

        if a not in valid_set or b not in valid_set:
            logger.debug("Skipping LLM pair (%s, %s) — unknown ID(s)", a, b)
            continue

        if rel not in _VALID_TYPES or rel == "none":
            continue

        key = (min(a, b), max(a, b))
        labels[key] = (rel, a, b)

    # Parse room_available; default True for surfaces not mentioned by LLM.
    for sid in surf_set:
        if sid in llm_room:
            room_available[sid] = bool(llm_room[sid])
        else:
            room_available[sid] = True  # conservative default

    logger.info(
        "LLM contact classifier: %d relationships, %d surface room assessments",
        len(labels), len(room_available),
    )
    return LLMContactResult(labels=labels, room_available=room_available)


# ---------------------------------------------------------------------------
# Natural-language prompt variant
# ---------------------------------------------------------------------------

_NL_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant for a robot manipulation system.
You will be shown an RGB scene image with coloured mask overlays. \
Each detected object is labelled with its exact ID string.

You will be given a list of object IDs. Describe only the real physical \
relationships you can see — one per line — using these exact templates:

  <A> is supported by <B>    (B is the immediate surface directly beneath A)
  <A> is nested in <B>       (A is inside B, e.g. object inside a bowl)

Rules:
- Use object IDs EXACTLY as given. Do not rename or invent IDs.
- For "supported by": use the CLOSEST object directly beneath A, not a \
surface further down the stack. If a cup sits on a box which sits on a table, \
write "cup is supported by box" — not "cup is supported by table".
- Only output lines for relationships that clearly exist in the image.
- If there are no relationships, output: none
- No other text, no explanation.
"""


def classify_contacts_nl(
    objects: List[DetectedObject],
    object_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    color_img: np.ndarray,
    obj_masks: Dict[str, np.ndarray],
    llm_client,
    surface_ids: Optional[List[str]] = None,
    debug_image_out: Optional[List[bytes]] = None,
) -> LLMContactResult:
    """Natural-language variant of classify_contacts_with_llm.

    Asks the model to produce one plain-text line per relationship rather than
    JSON, then parses those lines.  Fewer tokens, less truncation risk.
    room_available defaults to True for all surfaces (NL prompt doesn't assess it).
    """
    from src.llm_interface.base import ImagePart, GenerateConfig

    obj_ids = [o.object_id for o in objects if o.object_id in object_aabbs]
    if len(obj_ids) < 2:
        return LLMContactResult()

    png_bytes = _render_annotated_image(color_img, obj_masks)
    if debug_image_out is not None:
        debug_image_out.append(png_bytes)

    user_text = (
        f"Object IDs (use exactly as written): {json.dumps(obj_ids)}\n\n"
        f"List every supporting or nested relationship visible in the image."
    )

    config = GenerateConfig(
        temperature=0.0,
        max_output_tokens=max(256, len(obj_ids) * 15),
        system_instruction=_NL_SYSTEM_PROMPT,
    )

    try:
        response = llm_client.generate(
            [ImagePart(data=png_bytes, mime_type="image/png"), user_text],
            config=config,
        )
    except Exception as exc:
        logger.error("LLM NL contact classification failed: %s", exc)
        return LLMContactResult()

    logger.info("LLM NL raw response:\n%s", response.text)
    labels = _parse_nl_response(response.text, obj_ids)
    # NL prompt doesn't assess room_available — default True for all surfaces.
    room_avail = {sid: True for sid in (surface_ids or [])}
    return LLMContactResult(labels=labels, room_available=room_avail)


def _parse_nl_response(
    text: str,
    valid_ids: List[str],
) -> Dict[Tuple[str, str], LLMLabel]:  # returns raw labels dict (room_available added by caller)
    """Parse natural-language relationship lines into a sorted-key → (rel, a, b) dict.

    Accepted line formats (case-insensitive):
      <A> is supported by <B>
      <A> is nested in <B>
    """
    import re
    result: Dict[Tuple[str, str], LLMLabel] = {}
    valid_set = set(valid_ids)

    # Build a regex that matches either template, capturing the two IDs.
    # IDs may contain underscores, digits, letters.
    id_pat = r"(.+?)"
    supported_re = re.compile(
        rf"^\s*{id_pat}\s+is\s+supported\s+by\s+{id_pat}\s*$", re.IGNORECASE
    )
    nested_re = re.compile(
        rf"^\s*{id_pat}\s+is\s+nested\s+in\s+{id_pat}\s*$", re.IGNORECASE
    )

    for line in text.splitlines():
        line = line.strip(" -•*")
        if not line or line.lower() == "none":
            continue

        m = supported_re.match(line)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            rel = "supporting"
            supporter, supported_obj = b, a  # "A is supported by B" → B supports A
        else:
            m = nested_re.match(line)
            if m:
                a, b = m.group(1).strip(), m.group(2).strip()
                rel = "nested"
                supporter, supported_obj = b, a  # "A is nested in B" → B contains A
            else:
                logger.debug("NL parser: unrecognised line: %r", line)
                continue

        if supporter not in valid_set or supported_obj not in valid_set:
            logger.debug("NL parser: unknown ID(s) in line %r — skipping", line)
            continue

        key = (min(supporter, supported_obj), max(supporter, supported_obj))
        result[key] = (rel, supporter, supported_obj)
        logger.debug("NL parser: %s → %s (%s)", supporter, supported_obj, rel)

    logger.info("LLM NL contact classifier: %d relationships found", len(result))
    return result
