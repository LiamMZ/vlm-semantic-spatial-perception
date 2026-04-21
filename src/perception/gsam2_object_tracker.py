"""
GSAM2-based object tracker that replaces LLM detection with RAM+ + GroundingDINO + SAM2.

Provides GSAM2ObjectTracker (async detect_objects compatible with ObjectTracker) and
GSAM2ContinuousObjectTracker (drop-in for ContinuousObjectTracker).
"""

import asyncio
import collections
import io
import json
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image

from pathlib import Path

from .clearance import ClearanceProfile, GripperGeometry, compute_clearance_profile
from .contact_graph import compute_contact_graph
from .occlusion import CameraPose, ObservationRecord, compute_occlusion_map
from .surface_map import compute_surface_maps
from .gsam2 import IncrementalObjectTracker, OpenAITagger
from .object_registry import DetectedObject, DetectedObjectRegistry, InteractionPoint
from .object_tracker import ContinuousObjectTracker, ObjectTracker, TrackingStats, save_debug_frame
from .utils.coordinates import compute_3d_position, pixel_to_normalized
from ..utils.logging_utils import get_structured_logger

import re
from typing import Optional as _Optional

# Labels that GroundingDINO sometimes produces that don't represent real objects.
_JUNK_LABELS = frozenset({
    "image", "photo", "picture", "frame", "background", "scene", "view",
    "object", "thing", "item", "area", "region", "part", "surface",
    "unknown", "none", "null", "",
})

# Simple stopwords to skip when extracting object hints from task text
_STOPWORDS = frozenset({
    "a", "an", "the", "put", "place", "move", "pick", "up", "on", "onto",
    "in", "into", "to", "from", "and", "or", "of", "with", "get", "take",
    "stack", "grasp", "bring", "is", "are", "it", "its", "that", "this",
})


def _transform_cam_to_world(
    cam_pos: np.ndarray,
    robot_state: _Optional[dict],
) -> _Optional[np.ndarray]:
    """
    Transform a camera-frame 3-D point to world/robot-base frame.

    Reads the camera transform from robot_state["camera"] which must contain:
      - "position":        [x, y, z]  camera origin in world frame
      - "quaternion_xyzw": [qx, qy, qz, qw]  camera orientation in world frame

    Returns the world-frame position, or None if the transform is unavailable.
    """
    if robot_state is None:
        return None
    cam_tf = robot_state.get("camera")
    if cam_tf is None:
        return None
    try:
        from scipy.spatial.transform import Rotation
        cam_origin = np.array(cam_tf["position"], dtype=float)
        cam_rot    = Rotation.from_quat(cam_tf["quaternion_xyzw"])
        return cam_rot.apply(cam_pos) + cam_origin
    except Exception:
        return None


def _extract_noun_phrases(text: str) -> List[str]:
    """
    Extract candidate object noun phrases from a task description.

    Uses a simple adjective+noun pattern (e.g. "red block", "blue block")
    without requiring NLP dependencies.  Returns lowercased multi-word phrases
    and single content words that are not stopwords.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    phrases: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _STOPWORDS:
            i += 1
            continue
        # If next word is also a content word, emit a two-word phrase
        if i + 1 < len(words) and words[i + 1] not in _STOPWORDS:
            phrases.append(f"{w} {words[i + 1]}")
            i += 2
        else:
            phrases.append(w)
            i += 1
    return phrases




class GSAM2ObjectTracker:
    """
    Object tracker using RAM+ → GroundingDINO → SAM2 instead of an LLM.

    Compatible drop-in for the detect_objects() interface of ObjectTracker.
    Maintains a DetectedObjectRegistry populated from SAM2 mask outputs.

    Args:
        sam2_model_cfg: SAM2 config file path
        sam2_ckpt_path: SAM2 checkpoint path
        grounding_model_id: GroundingDINO HuggingFace model ID
        openai_api_key: OpenAI API key for the vision tagger (if None, reads OPENAI_API_KEY env var)
        tagger_model: OpenAI model ID to use for tagging (default: "gpt-4o-mini")
        detection_interval: Frames between GroundingDINO re-detections
        device: torch device string
        tag_interval: Frames between OpenAI re-tagging (1 = every frame)
        logger: Optional logger
    """


    def __init__(
        self,
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path: str = "./checkpoints/sam2.1_hiera_large.pt",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        openai_api_key: Optional[str] = None,
        tagger_model: str = "gpt-4o-mini",
        detection_interval: int = 20,
        score_threshold: float = 0.5,
        overlap_iou_threshold: float = 0.5,
        device: str = "cuda",
        tag_interval: int = 1,
        llm_client: Optional[Any] = None,
        compute_clearances: bool = True,
        gripper: Optional[GripperGeometry] = None,
        compute_contacts: bool = True,
        contact_threshold_m: float = 0.005,
        compute_occlusion: bool = True,
        occlusion_history_len: int = 10,
        occlusion_update_interval: int = 1,
        compute_surface_maps: bool = True,
        surface_map_resolution_m: float = 0.01,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or get_structured_logger("GSAM2ObjectTracker")
        self.device = device
        self.tag_interval = tag_interval
        self.registry = DetectedObjectRegistry()
        self._llm_client = llm_client
        self._compute_clearances = compute_clearances
        self._gripper = gripper or GripperGeometry()
        self._compute_contacts = compute_contacts
        self._contact_threshold_m = contact_threshold_m
        self._compute_occlusion = compute_occlusion
        self._occlusion_update_interval = occlusion_update_interval
        self._compute_surface_maps = compute_surface_maps
        self._surface_map_resolution_m = surface_map_resolution_m
        # Rolling window of ObservationRecords for multi-viewpoint occlusion analysis
        self._obs_history: Deque[ObservationRecord] = collections.deque(maxlen=occlusion_history_len)
        # Cache affordances by object type (safe_name) to avoid redundant LLM calls
        self._affordance_cache: Dict[str, Set[str]] = {}
        # PDDL predicates to evaluate per object (set via set_pddl_predicates)
        self._pddl_predicates: List[str] = []
        self._last_masks: Dict[str, np.ndarray] = {}  # populated after each detect_objects call

        self.logger.info("Loading GroundingDINO (%s) + SAM2 (%s) on %s…",
                         grounding_model_id, sam2_ckpt_path, device)
        self._gsam2 = IncrementalObjectTracker(
            grounding_model_id=grounding_model_id,
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            prompt_text="object.",
            detection_interval=detection_interval,
            score_threshold=score_threshold,
            overlap_iou_threshold=overlap_iou_threshold,
        )
        self.logger.info("GroundingDINO + SAM2 loaded.")

        self.logger.info("Loading OpenAI tagger (%s)…", tagger_model)
        self._tagger = OpenAITagger(api_key=openai_api_key, model=tagger_model)
        self.logger.info("OpenAI tagger loaded.")

        self._current_prompt: str = "object."
        self._frame_count: int = 0
        self._extra_tags: List[str] = []

    def set_tagger(self, tagger_callable):
        """Override the default tagger with any BaseTagger or callable tag(rgb_np) -> (prompt, raw)."""
        self._tagger = tagger_callable

    def set_pddl_predicates(self, predicates: List[str]) -> None:
        """Store PDDL predicate signatures to evaluate per detected object."""
        self._pddl_predicates = list(predicates or [])
        self.logger.debug("GSAM2ObjectTracker: %d PDDL predicates configured", len(self._pddl_predicates))

    def set_extra_tags(self, tags: List[str]) -> None:
        """Inject additional search terms (e.g. from task description) into the GroundingDINO prompt."""
        self._extra_tags = [t.strip().lower() for t in tags if t.strip()]
        # Immediately merge into current prompt so the next frame uses them
        self._current_prompt = self._merge_prompt(self._current_prompt, self._extra_tags)
        self._gsam2.set_prompt(self._current_prompt)
        self.logger.debug("Extra tags set: %s → prompt: %s", self._extra_tags, self._current_prompt)

    @staticmethod
    def _merge_prompt(ram_prompt: str, extra_tags: List[str]) -> str:
        """Append extra tags to RAM prompt, deduplicating existing terms."""
        existing = {t.rstrip(".").strip() for t in ram_prompt.split() if t.strip()}
        additions = [t for t in extra_tags if t not in existing]
        if not additions:
            return ram_prompt
        extra_str = " ".join(t + "." for t in additions)
        return (ram_prompt.rstrip() + " " + extra_str).strip()

    async def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        temperature: Optional[float] = None,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> List[DetectedObject]:
        """
        Detect and segment objects using RAM+ → GroundingDINO → SAM2.

        Args:
            color_frame: RGB image (np.ndarray HxWx3 or PIL Image)
            depth_frame: Optional depth image in metres (same HxW as color_frame)
            camera_intrinsics: Camera intrinsics for 3D back-projection
            temperature: Unused (kept for interface compatibility)
            robot_state: Optional robot state dict containing camera transform for
                         converting camera-frame positions to world frame.

        Returns:
            List of DetectedObject instances added to self.registry
        """
        if isinstance(color_frame, Image.Image):
            rgb_np = np.array(color_frame)
        else:
            rgb_np = color_frame

        loop = asyncio.get_event_loop()

        # --- Tagging (CPU-bound, run in thread pool) ---
        if self._tagger is not None and self._frame_count % self.tag_interval == 0:
            new_prompt, raw = await loop.run_in_executor(
                None, self._tagger, rgb_np, self._extra_tags or None
            )
            if new_prompt:
                merged = self._merge_prompt(new_prompt, self._extra_tags)
                if merged != self._current_prompt:
                    self.logger.debug("RAM+ prompt update: %s (was: %s)", merged, self._current_prompt)
                    self._current_prompt = merged
                    self._gsam2.set_prompt(merged)

        # --- SAM2 tracking (GPU-bound, run in thread pool to avoid blocking event loop) ---
        def _run_gsam2():
            return self._gsam2.add_image(rgb_np)

        await loop.run_in_executor(None, _run_gsam2)

        self._frame_count += 1

        # --- Convert SAM2 results → DetectedObject ---
        detected: List[DetectedObject] = []
        _obj_masks: Dict[str, np.ndarray] = {}  # object_id -> bool mask, for clearance
        mask_dict = self._gsam2.last_mask_dict
        # Clear stale predicate facts from the previous frame before rebuilding
        if self._pddl_predicates:
            self.registry.clear_predicates()

        if mask_dict is None or not mask_dict.labels:
            return detected

        h, w = rgb_np.shape[:2]

        for obj_id, obj_info in mask_dict.labels.items():
            class_name = obj_info.class_name or "object"
            # Normalise to a valid PDDL identifier: lowercase, spaces → underscores.
            # GroundingDINO sometimes bleeds token spans across adjacent prompt labels,
            # producing labels like "blue_block red_block" or "cube cube".
            # Fix: keep only the first occurrence of each word.
            _words = class_name.strip().lower().split()
            _seen_words: set = set()
            _deduped: list = []
            for _word in _words:
                if _word not in _seen_words:
                    _seen_words.add(_word)
                    _deduped.append(_word)
            safe_name = "_".join(_deduped)

            # Skip detections where GroundingDINO returned a non-object label
            if safe_name in _JUNK_LABELS:
                self.logger.debug("Skipping junk label '%s' (obj_id=%s)", safe_name, obj_id)
                continue

            object_id = f"{safe_name}_{obj_id}"

            # Pixel bounding box from ObjectInfo
            x1 = int(obj_info.x1) if obj_info.x1 is not None else 0
            y1 = int(obj_info.y1) if obj_info.y1 is not None else 0
            x2 = int(obj_info.x2) if obj_info.x2 is not None else w
            y2 = int(obj_info.y2) if obj_info.y2 is not None else h

            # Mask centroid for 2D position
            mask = obj_info.mask
            if mask is not None:
                import torch
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
                centroid_y = int(ys.mean())
                centroid_x = int(xs.mean())
            else:
                centroid_y = (y1 + y2) // 2
                centroid_x = (x1 + x2) // 2

            position_2d = pixel_to_normalized((centroid_y, centroid_x), (h, w))
            # bounding_box_2d: [y1, x1, y2, x2] normalized to 0-1000 (same scale as position_2d)
            bbox_2d = [
                int(y1 / h * 1000),
                int(x1 / w * 1000),
                int(y2 / h * 1000),
                int(x2 / w * 1000),
            ]

            # 3D position: back-project to camera frame, then transform to world frame
            position_3d = None
            if depth_frame is not None and camera_intrinsics is not None:
                cam_pos = compute_3d_position(position_2d, depth_frame, camera_intrinsics)
                if cam_pos is not None:
                    position_3d = _transform_cam_to_world(cam_pos, robot_state)
                    if position_3d is None:
                        position_3d = cam_pos  # fallback: keep camera-frame if no transform

            interaction_point = InteractionPoint(
                position_2d=position_2d,
                position_3d=position_3d,
            )

            # Crop the object from the frame (used for both affordance and predicate inference)
            crop = rgb_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                crop = rgb_np

            # Infer affordances via LLM if available, else use cache or default
            if self._llm_client is not None:
                affordances = await self._infer_affordances(safe_name, crop)
            else:
                affordances = self._affordance_cache.get(safe_name, {"graspable"})

            obj = DetectedObject(
                object_type=safe_name,
                object_id=object_id,
                affordances=affordances,
                interaction_points={"grasp": interaction_point},
                position_2d=position_2d,
                position_3d=position_3d,
                bounding_box_2d=bbox_2d,
            )

            self.registry.add_object(obj)
            detected.append(obj)
            # Accumulate boolean masks for post-loop clearance computation.
            # mask may be a torch.Tensor or np.ndarray; convert to bool ndarray.
            if mask is not None:
                bool_mask = mask.astype(bool) if isinstance(mask, np.ndarray) else mask.cpu().numpy().astype(bool)
            else:
                bool_mask = np.zeros((h, w), dtype=bool)
                bool_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            _obj_masks[object_id] = bool_mask

        self._last_masks = _obj_masks

        # --- Clearance profiles (depth-based, O(n·k)) ---
        # Runs after all objects are built so every object can serve as an
        # obstacle for every other object's ray-casting pass.
        if self._compute_clearances and depth_frame is not None and camera_intrinsics is not None:
            import time as _time
            _t0 = _time.perf_counter()
            for obj in detected:
                target_mask_bool = _obj_masks.get(obj.object_id)
                if target_mask_bool is None:
                    continue
                other_masks = {oid: m for oid, m in _obj_masks.items() if oid != obj.object_id}
                try:
                    obj.clearance_profile = compute_clearance_profile(
                        target_mask=target_mask_bool,
                        depth_frame=depth_frame,
                        camera_intrinsics=camera_intrinsics,
                        all_masks=other_masks,
                        gripper=self._gripper,
                    )
                    # Keep registry in sync with the updated object
                    self.registry.update_object(obj.object_id, obj)
                except Exception as e:
                    self.logger.warning("Clearance computation failed for '%s': %s", obj.object_id, e)
            _elapsed_ms = (_time.perf_counter() - _t0) * 1000
            self.logger.debug("Clearance profiles computed for %d objects in %.1f ms", len(detected), _elapsed_ms)

        # --- Contact graph (pairwise O(n²) + stability analysis) ---
        if self._compute_contacts and depth_frame is not None and camera_intrinsics is not None and len(detected) >= 2:
            import time as _time
            _t0 = _time.perf_counter()
            try:
                self.registry.contact_graph = compute_contact_graph(
                    objects=detected,
                    obj_masks=_obj_masks,
                    depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_threshold_m=self._contact_threshold_m,
                )
            except Exception as e:
                self.logger.warning("Contact graph computation failed: %s", e)
            _elapsed_ms = (_time.perf_counter() - _t0) * 1000
            self.logger.debug("Contact graph computed in %.1f ms", _elapsed_ms)

        # --- Surface free-space maps (requires contact graph for surface ID resolution) ---
        if self._compute_surface_maps and depth_frame is not None and camera_intrinsics is not None:
            import time as _time
            _t0 = _time.perf_counter()
            try:
                surface_maps = compute_surface_maps(
                    objects=detected,
                    obj_masks=_obj_masks,
                    depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_graph=self.registry.contact_graph,
                    resolution_m=self._surface_map_resolution_m,
                )
                for surface_id, smap in surface_maps.items():
                    obj = self.registry.get_object(surface_id)
                    if obj is not None:
                        obj.surface_map = smap
                        self.registry.update_object(surface_id, obj)
            except Exception as e:
                self.logger.warning("Surface map computation failed: %s", e)
            _elapsed_ms = (_time.perf_counter() - _t0) * 1000
            self.logger.debug("Surface maps computed in %.1f ms", _elapsed_ms)

        # --- Occlusion map (rolling history, depth-image-based) ---
        if self._compute_occlusion and depth_frame is not None and camera_intrinsics is not None:
            # Always append the current observation so history stays current
            self._obs_history.append(ObservationRecord(
                depth_frame=depth_frame,
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose.from_robot_state(robot_state),
                obj_masks=dict(_obj_masks),
            ))
            # Recompute map at the configured interval
            if self._frame_count % self._occlusion_update_interval == 0:
                import time as _time
                _t0 = _time.perf_counter()
                try:
                    self.registry.occlusion_map = compute_occlusion_map(
                        observations=list(self._obs_history),
                        object_ids=[o.object_id for o in detected],
                    )
                except Exception as e:
                    self.logger.warning("Occlusion map computation failed: %s", e)
                _elapsed_ms = (_time.perf_counter() - _t0) * 1000
                self.logger.debug("Occlusion map computed in %.1f ms", _elapsed_ms)

        # Infer PDDL predicates for all detected objects now that the full
        # detected list is built (binary predicates like "on obj1 obj2" need
        # all object IDs to be known before querying).
        if self._llm_client is not None and self._pddl_predicates and detected:
            all_object_ids = [o.object_id for o in detected]
            for obj in detected:
                oy1, ox1, oy2, ox2 = (obj.bounding_box_2d or [0, 0, h, w])
                obj_crop = rgb_np[max(0, oy1 * h // 1000):min(h, oy2 * h // 1000),
                                  max(0, ox1 * w // 1000):min(w, ox2 * w // 1000)]
                if obj_crop.size == 0:
                    obj_crop = rgb_np
                pred_strings = await self._infer_predicates(
                    obj.object_id, obj.object_type, obj_crop, all_object_ids
                )
                if pred_strings:
                    self.registry.add_predicates(pred_strings)

        # Evict registry entries whose GSAM2 track ID is no longer active.
        # object_ids have the form "<name>_<track_id>"; split on the last "_"
        # to recover the numeric track_id and compare against the current frame.
        active_track_ids: Set[int] = set(mask_dict.labels.keys())
        stale_ids = []
        for reg_id in list(self.registry._objects.keys()):
            parts = reg_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if int(parts[1]) not in active_track_ids:
                    stale_ids.append(reg_id)
        for stale_id in stale_ids:
            self.registry.remove_object(stale_id)
            self.logger.debug("Evicted stale registry entry '%s'", stale_id)

        self.logger.debug(
            "GSAM2 detected %d objects: %s",
            len(detected),
            [o.object_id for o in detected],
        )
        return detected

    async def _infer_affordances(
        self,
        object_type: str,
        crop_rgb: np.ndarray,
    ) -> Set[str]:
        """
        Query the LLM to infer affordances for an object given its cropped image.

        Results are cached by object_type so the LLM is only queried once per
        novel object class per tracker lifetime.  Falls back to {"graspable"}
        on any failure.
        """
        if object_type in self._affordance_cache:
            return self._affordance_cache[object_type]

        try:
            from ..llm_interface.base import GenerateConfig, ImagePart

            buf = io.BytesIO()
            Image.fromarray(crop_rgb).save(buf, format="PNG")
            crop_png = buf.getvalue()

            prompt = (
                f"You are a robotics perception system. Given an image of a '{object_type}', "
                "list the affordances that apply to this object for a robot manipulation task. "
                "Common affordances include: graspable, placeable_on, openable, closeable, "
                "pushable, containable, movable, fixed. "
                "Return only affordances that clearly apply based on the object's appearance. "
                "Always include 'graspable' if the object can be picked up.\n\n"
                'Respond with a JSON object in this exact format: {"affordances": ["label1", "label2"]}'
            )

            config = GenerateConfig(
                temperature=0.2,
                max_output_tokens=256,
            )

            response = await self._llm_client.generate_async(
                [prompt, ImagePart(data=crop_png, mime_type="image/png")],
                config=config,
            )

            parsed = json.loads(response.text)
            affordances = set(parsed.get("affordances", ["graspable"]))
            if not affordances:
                affordances = {"graspable"}

            self._affordance_cache[object_type] = affordances
            self.logger.debug("Affordances for '%s': %s", object_type, affordances)
            return affordances

        except Exception as e:
            self.logger.warning(
                "Affordance inference failed for '%s': %s — falling back to {'graspable'}",
                object_type, e,
            )
            return {"graspable"}

    async def _infer_predicates(
        self,
        object_id: str,
        object_type: str,
        crop_rgb: np.ndarray,
        all_object_ids: List[str],
    ) -> List[str]:
        """
        Query the LLM to evaluate which PDDL predicates hold TRUE for this object.

        Uses the masked crop image and the full list of detected object IDs so
        the model can produce both unary predicates (e.g. "clear blue_block_2")
        and binary predicates (e.g. "on blue_block_2 table_5").

        Returns a list of grounded predicate strings ready for registry.add_predicates().
        Falls back to an empty list on any failure.
        """
        if not self._pddl_predicates:
            return []

        try:
            from ..llm_interface.base import GenerateConfig, ImagePart

            buf = io.BytesIO()
            Image.fromarray(crop_rgb).save(buf, format="PNG")
            crop_png = buf.getvalue()

            other_ids = [oid for oid in all_object_ids if oid != object_id]
            predicates_str = "\n".join(f"  - {p}" for p in self._pddl_predicates)
            others_str = ", ".join(other_ids) if other_ids else "(none)"

            prompt = (
                f"You are a robotics perception system evaluating PDDL state predicates.\n"
                f"The image shows object '{object_id}' (type: {object_type}).\n"
                f"Other objects in the scene: {others_str}\n\n"
                f"Available predicates:\n{predicates_str}\n\n"
                "For each predicate that holds TRUE for this object right now, "
                "output one grounded predicate string per entry:\n"
                "  - Unary:  'predicate_name object_id'\n"
                "  - Binary: 'predicate_name object_id other_object_id'\n\n"
                "Only assert predicates you can confidently determine from the image. "
                "Do not assert spatial relations you cannot see (e.g. do not guess 'on' "
                "unless the object is visibly resting on another object in this image).\n\n"
                'Respond with a JSON object in this exact format: {"predicates": ["pred1 obj1", "pred2 obj1 obj2"]}'
            )

            config = GenerateConfig(
                temperature=0.1,
                max_output_tokens=512,
            )

            response = await self._llm_client.generate_async(
                [prompt, ImagePart(data=crop_png, mime_type="image/png")],
                config=config,
            )

            parsed = json.loads(response.text)
            pred_strings: List[str] = []
            for entry in parsed.get("predicates", []):
                s = str(entry).strip()
                if s:
                    pred_strings.append(s)

            self.logger.debug(
                "Predicates for '%s': %s", object_id, pred_strings
            )
            return pred_strings

        except Exception as e:
            self.logger.warning(
                "Predicate inference failed for '%s': %s — skipping",
                object_id, e,
            )
            return []

    def get_all_objects(self) -> List[DetectedObject]:
        return self.registry.get_all_objects()

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        return self.registry.get_object(object_id)

    def recompute_geometry(
        self,
        obj_masks: Dict[str, np.ndarray],
        depth_frame: np.ndarray,
        camera_intrinsics,
        robot_state: Optional[Dict[str, Any]] = None,
        affected_ids: Optional[List[str]] = None,
        force_occlusion: bool = False,
    ) -> None:
        """Recompute all geometry representations for a given depth snapshot.

        Called by T3/T5 triggers to refresh clearance, contact graph, surface
        maps, and optionally occlusion after a manipulation or precondition failure.

        Args:
            obj_masks: {object_id: bool ndarray} SAM2 masks for this snapshot.
            depth_frame: (H, W) float32 depth in metres.
            camera_intrinsics: Camera intrinsics object.
            robot_state: Optional robot state for camera transform.
            affected_ids: If provided, only recompute clearance for these objects
                (contact graph and surface maps always use all objects).
            force_occlusion: If True, force-update the occlusion history and map
                regardless of update_interval.
        """
        objects = self.registry.get_all_objects()
        if not objects:
            return

        # Clearance — per-object, optionally restricted to affected_ids
        if self._compute_clearances:
            target_objs = [o for o in objects if affected_ids is None or o.object_id in affected_ids]
            for obj in target_objs:
                mask = obj_masks.get(obj.object_id)
                if mask is None:
                    continue
                other_masks = {oid: m for oid, m in obj_masks.items() if oid != obj.object_id}
                try:
                    obj.clearance_profile = compute_clearance_profile(
                        target_mask=mask,
                        depth_frame=depth_frame,
                        camera_intrinsics=camera_intrinsics,
                        all_masks=other_masks,
                        gripper=self._gripper,
                    )
                    self.registry.update_object(obj.object_id, obj)
                except Exception as e:
                    self.logger.warning("Clearance recompute failed for '%s': %s", obj.object_id, e)

        # Contact graph — always uses full object set
        if self._compute_contacts and len(objects) >= 2:
            try:
                self.registry.contact_graph = compute_contact_graph(
                    objects=objects,
                    obj_masks=obj_masks,
                    depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_threshold_m=self._contact_threshold_m,
                )
            except Exception as e:
                self.logger.warning("Contact graph recompute failed: %s", e)

        # Surface maps — after contact graph update
        if self._compute_surface_maps:
            try:
                surface_maps = compute_surface_maps(
                    objects=objects,
                    obj_masks=obj_masks,
                    depth_frame=depth_frame,
                    camera_intrinsics=camera_intrinsics,
                    contact_graph=self.registry.contact_graph,
                    resolution_m=self._surface_map_resolution_m,
                )
                for surface_id, smap in surface_maps.items():
                    obj = self.registry.get_object(surface_id)
                    if obj is not None:
                        obj.surface_map = smap
                        self.registry.update_object(surface_id, obj)
            except Exception as e:
                self.logger.warning("Surface map recompute failed: %s", e)

        # Occlusion — append new observation and force recompute
        if self._compute_occlusion and force_occlusion:
            self._obs_history.append(ObservationRecord(
                depth_frame=depth_frame,
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose.from_robot_state(robot_state),
                obj_masks=dict(obj_masks),
            ))
            try:
                self.registry.occlusion_map = compute_occlusion_map(
                    observations=list(self._obs_history),
                    object_ids=[o.object_id for o in objects],
                )
            except Exception as e:
                self.logger.warning("Occlusion map recompute failed: %s", e)


class GSAM2ContinuousObjectTracker:
    """
    Background continuous tracker using GSAM2ObjectTracker.

    Mirrors the ContinuousObjectTracker interface (start/stop/set_frame_provider/registry).
    """

    def __init__(
        self,
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path: str = "./checkpoints/sam2.1_hiera_large.pt",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        openai_api_key: Optional[str] = None,
        tagger_model: str = "gpt-4o-mini",
        detection_interval: int = 20,
        score_threshold: float = 0.5,
        overlap_iou_threshold: float = 0.5,
        device: str = "cuda",
        tag_interval: int = 1,
        update_interval: float = 0.0,
        on_detection_complete: Optional[Callable[[int], None]] = None,
        llm_client: Optional[Any] = None,
        compute_clearances: bool = True,
        gripper: Optional[GripperGeometry] = None,
        compute_contacts: bool = True,
        contact_threshold_m: float = 0.005,
        compute_occlusion: bool = True,
        occlusion_history_len: int = 10,
        occlusion_update_interval: int = 1,
        compute_surface_maps: bool = True,
        surface_map_resolution_m: float = 0.01,
        t1_budget_s: float = 2.0,
        logger: Optional[logging.Logger] = None,
        debug_save_dir: Optional[Union[str, Path]] = None,
    ):
        self._tracker = GSAM2ObjectTracker(
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            grounding_model_id=grounding_model_id,
            openai_api_key=openai_api_key,
            tagger_model=tagger_model,
            detection_interval=detection_interval,
            score_threshold=score_threshold,
            overlap_iou_threshold=overlap_iou_threshold,
            device=device,
            tag_interval=tag_interval,
            llm_client=llm_client,
            compute_clearances=compute_clearances,
            gripper=gripper,
            compute_contacts=compute_contacts,
            contact_threshold_m=contact_threshold_m,
            compute_occlusion=compute_occlusion,
            occlusion_history_len=occlusion_history_len,
            occlusion_update_interval=occlusion_update_interval,
            compute_surface_maps=compute_surface_maps,
            surface_map_resolution_m=surface_map_resolution_m,
            logger=logger,
        )
        self.registry = self._tracker.registry
        self.update_interval = update_interval
        self.on_detection_complete = on_detection_complete
        self.logger = logger or get_structured_logger("GSAM2ContinuousObjectTracker")
        self.stats = TrackingStats()
        self._t1_budget_s = t1_budget_s   # max acceptable loop duration before throttling
        self._t1_throttled = False

        self._frame_provider: Optional[Callable] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Last detection bundle (for save_snapshot compatibility)
        self._last_bundle_lock = threading.Lock()
        self._last_bundle: Optional[Dict[str, Any]] = None

        # Debug image streaming
        self._debug_save_dir: Optional[Path] = Path(debug_save_dir) if debug_save_dir else None
        self._debug_frame_index: int = 0
        self._debug_lock = threading.Lock()

    def set_tagger(self, tagger_callable):
        self._tracker.set_tagger(tagger_callable)

    # ------------------------------------------------------------------ #
    # Stubs for ObjectTracker interface compatibility                      #
    # ------------------------------------------------------------------ #

    def set_pddl_predicates(self, predicates) -> None:
        """Forward PDDL predicates to the inner tracker for per-object LLM evaluation."""
        self._tracker.set_pddl_predicates(predicates)

    def set_pddl_actions(self, _actions) -> None:
        """No-op: action schemas are not needed for visual predicate evaluation."""

    def set_task_context(self, task_description=None, available_actions=None, goal_objects=None) -> None:
        """Extract object noun phrases from task description and inject into GroundingDINO prompt."""
        hints: List[str] = []
        if goal_objects:
            hints.extend(goal_objects)
        if task_description:
            hints.extend(_extract_noun_phrases(task_description))
        if hints:
            self._tracker.set_extra_tags(hints)
            self.logger.info("GSAM2 task hints injected: %s", hints)

    def set_frame_provider(self, provider: Callable[[], tuple]):
        self._frame_provider = provider

    def start(self):
        if self._running:
            return
        if self._frame_provider is None:
            raise ValueError("Frame provider not set. Call set_frame_provider() first.")
        self._running = True
        self.stats.is_running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._tracking_loop())
        self.logger.info("GSAM2ContinuousObjectTracker started")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        self.stats.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("GSAM2ContinuousObjectTracker stopped")

    async def _tracking_loop(self):
        while self._running:
            loop_start = time.time()
            try:
                provided = self._frame_provider()
                if isinstance(provided, tuple) and len(provided) == 4:
                    color_frame, depth_frame, intrinsics, robot_state = provided
                else:
                    color_frame, depth_frame, intrinsics = provided
                    robot_state = None

                detection_start = time.time()
                detected = await self._tracker.detect_objects(
                    color_frame, depth_frame, intrinsics, robot_state=robot_state
                )
                detection_time = time.time() - detection_start

                self.stats.total_frames += 1
                self.stats.total_detections += len(detected)
                self.stats.last_detection_time = detection_time
                alpha = 0.1
                self.stats.avg_detection_time = (
                    alpha * detection_time + (1 - alpha) * self.stats.avg_detection_time
                )

                # Cache detection bundle for save_snapshot compatibility
                buf = io.BytesIO()
                frame_img = color_frame if isinstance(color_frame, Image.Image) else Image.fromarray(color_frame)
                frame_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                with self._last_bundle_lock:
                    self._last_bundle = {
                        "timestamp": time.time(),
                        "color_png": png_bytes,
                        "depth": np.array(depth_frame, copy=True) if depth_frame is not None else None,
                        "intrinsics": intrinsics,
                        "objects": list(detected),
                        "robot_state": robot_state,
                    }

                if self.on_detection_complete:
                    if asyncio.iscoroutinefunction(self.on_detection_complete):
                        await self.on_detection_complete(len(detected))
                    else:
                        self.on_detection_complete(len(detected))

                # Stream annotated debug frame in background if enabled
                if self._debug_save_dir is not None:
                    frame_idx = self._debug_frame_index
                    self._debug_frame_index += 1
                    objects_snapshot = list(detected)
                    t = threading.Thread(
                        target=save_debug_frame,
                        args=(png_bytes, objects_snapshot, frame_idx, self._debug_save_dir, self._debug_lock),
                        daemon=True,
                    )
                    t.start()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("GSAM2 tracking loop error: %s", e, exc_info=True)

            elapsed = time.time() - loop_start

            # T1 budget guard: if the full detection+geometry cycle is taking too
            # long, throttle to 0.5 Hz (2 s interval) so we don't starve the event
            # loop.  Recover back to the configured rate once computation catches up.
            if self._t1_budget_s > 0 and elapsed > self._t1_budget_s:
                if not self._t1_throttled:
                    self._t1_throttled = True
                    self.logger.warning(
                        "T1 detection cycle took %.1f ms (budget %.0f ms) — "
                        "throttling to 0.5 Hz",
                        elapsed * 1000, self._t1_budget_s * 1000,
                    )
                target_interval = max(self.update_interval, 2.0)  # 0.5 Hz floor
            else:
                if self._t1_throttled:
                    self._t1_throttled = False
                    self.logger.info("T1 detection cycle back within budget — restoring rate")
                target_interval = self.update_interval

            if target_interval > elapsed:
                await asyncio.sleep(target_interval - elapsed)

    async def get_stats(self) -> TrackingStats:
        """Return a snapshot of current tracking statistics."""
        return TrackingStats(
            total_frames=self.stats.total_frames,
            total_detections=self.stats.total_detections,
            skipped_frames=self.stats.skipped_frames,
            avg_detection_time=self.stats.avg_detection_time,
            last_detection_time=self.stats.last_detection_time,
            cache_hit_rate=self.stats.cache_hit_rate,
            is_running=self.stats.is_running,
        )

    def get_all_objects(self) -> List[DetectedObject]:
        return self.registry.get_all_objects()

    def get_last_detection_bundle(self) -> Optional[Dict[str, Any]]:
        """Return the most recent detection frame and objects for snapshotting."""
        with self._last_bundle_lock:
            return dict(self._last_bundle) if self._last_bundle is not None else None

    def trigger_geometry_recompute(
        self,
        affected_ids: Optional[List[str]] = None,
        force_occlusion: bool = False,
    ) -> bool:
        """Recompute geometry representations from the most recent observation.

        Safe to call from any thread at any time (T2/T3/T5 triggers).  Uses the
        last cached depth frame and object masks.  Returns False if no observation
        is available yet.

        Args:
            affected_ids: Restrict clearance recompute to these object IDs.
                Contact graph and surface maps always use all objects.
            force_occlusion: If True, also update the occlusion history and map.
        """
        with self._last_bundle_lock:
            bundle = self._last_bundle

        if bundle is None:
            self.logger.debug("trigger_geometry_recompute: no observation available yet")
            return False

        depth = bundle.get("depth")
        intr = bundle.get("intrinsics")
        robot_state = bundle.get("robot_state")

        if depth is None or intr is None:
            return False

        # Reconstruct obj_masks from the inner tracker's observation history
        # (the most recent entry matches this bundle)
        obj_masks: Dict[str, np.ndarray] = {}
        if self._tracker._obs_history:
            last_obs = self._tracker._obs_history[-1]
            obj_masks = last_obs.obj_masks

        self._tracker.recompute_geometry(
            obj_masks=obj_masks,
            depth_frame=depth,
            camera_intrinsics=intr,
            robot_state=robot_state,
            affected_ids=affected_ids,
            force_occlusion=force_occlusion,
        )
        return True
