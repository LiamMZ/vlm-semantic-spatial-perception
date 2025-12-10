"""
Primitive executor that translates Gemini-friendly parameters before calling CuRobo primitives.

LLM outputs reference image-grounded cues (pixel [y, x] pointers, normals, standoffs).
This executor back-projects those cues into metric coordinates using the latest snapshot depth and
camera intrinsics, validates the plan, and optionally drives the configured motion planner.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import time

from src.perception.utils.coordinates import compute_3d_position
from src.primitives.skill_plan_types import PRIMITIVE_LIBRARY, PrimitiveCall, SkillPlan
from src.planning.utils.snapshot_utils import SnapshotArtifacts, SnapshotCache, load_snapshot_artifacts
from src.utils.logging_utils import get_structured_logger


@dataclass
class SnapshotCameraPose:
    position: np.ndarray
    rotation: Rotation


@dataclass
class PrimitiveExecutionResult:
    """Return payload for executor runs."""

    executed: bool
    primitive_results: List[Any] = field(default_factory=list)


class PrimitiveExecutor:
    """Translate and execute primitive plans against the configured primitives interface."""

    def __init__(
        self,
        primitives: Optional[Any],
        perception_pool_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.primitives = primitives
        self.perception_pool_dir = Path(perception_pool_dir)
        self._snapshot_cache = SnapshotCache()
        self.logger = logger or get_structured_logger("PrimitiveExecutor")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
        dry_run: bool = False,
    ) -> PrimitiveExecutionResult:
        """
        Translate (and optionally execute) a primitive plan.
        """
        self.logger.info("Executing plan: %s", plan)
        translated_plan = self.prepare_plan(plan, world_state)
        if dry_run:
            self.logger.info("Dry run requested; execution skipped.")
            return PrimitiveExecutionResult(executed=False)
        if self.primitives is None:
            raise RuntimeError("Primitives interface is required for execution (dry_run=False).")

        primitive_results: List[Any] = []
        for idx, primitive in enumerate(translated_plan.primitives):
            self.logger.info(
                "Executing primitive [%d/%d]: %s with parameters %s",
                idx + 1,
                len(translated_plan.primitives),
                primitive.name,
                primitive.parameters,
            )
            schema = PRIMITIVE_LIBRARY.get(primitive.name)
            if (
                schema is not None
                and "execute" in schema.optional_params
            ):
                primitive.parameters.setdefault("execute", True)
            method = getattr(self.primitives, primitive.name, None)
            if not callable(method):
                raise AttributeError(f"Primitives interface missing primitive '{primitive.name}'")
            self.logger.debug("Calling primitive method '%s' with parameters: %s", primitive.name, primitive.parameters)
            raw_result = method(**primitive.parameters)
            result = self._json_safe(raw_result)
            primitive_results.append(result)
            time.sleep(0.5)  # Small delay to avoid overwhelming the primitives interface

        return PrimitiveExecutionResult(executed=True, primitive_results=primitive_results)

    def prepare_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
    ) -> SkillPlan:
        """
        Translate parameters and validate the plan without executing it.
        """
        self.logger.info("[prepare_plan] Translating %d primitives", len(plan.primitives))

        artifacts = load_snapshot_artifacts(
            world_state,
            self.perception_pool_dir,
            cache=self._snapshot_cache,
            snapshot_id=getattr(plan, "source_snapshot_id", None),
        )
        self.logger.debug("[prepare_plan] Snapshot artifacts loaded (snapshot_id=%s)", artifacts.snapshot_id)

        if getattr(plan, "source_snapshot_id", None) and plan.source_snapshot_id != artifacts.snapshot_id:
            self.logger.warning(
                "Plan snapshot %s missing; using %s",
                plan.source_snapshot_id,
                artifacts.snapshot_id or "latest",
            )

        cam_pose: Optional[SnapshotCameraPose] = None
        joints = (artifacts.robot_state or {}).get("joints")
        self.logger.debug("[prepare_plan] Robot joints from snapshot: %s", joints)

        if self.primitives:
            helper = getattr(self.primitives, "camera_pose_from_joints", None)
            if helper:
                self.logger.debug("[prepare_plan] Getting camera pose from joints...")
                pos, rot = helper(joints)
                cam_pose = SnapshotCameraPose(position=np.asarray(pos, dtype=float), rotation=rot)
                self.logger.debug("[prepare_plan] Camera pose: pos=%s, rot=%s", cam_pose.position, cam_pose.rotation)
            else:
                self.logger.info("[prepare_plan] Primitives interface missing camera_pose_from_joints; skipping base-frame transform")
        else:
            self.logger.info("[prepare_plan] No primitives interface; skipping base-frame transform")

        # Translate each primitive
        for idx, primitive in enumerate(plan.primitives):
            self.logger.debug(
                "[prepare_plan] [%d/%d] %s parameters: %s",
                idx + 1,
                len(plan.primitives),
                primitive.name,
                primitive.parameters,
            )

            pixel = primitive.parameters.pop("target_pixel_yx", None)
            if pixel is not None:
                if artifacts.depth is None or artifacts.intrinsics is None:
                    raise RuntimeError(
                        f"{primitive.name}: unable to back-project normalized {pixel} (missing depth/intrinsics)"
                    )

                coords = [float(pixel[0]), float(pixel[1])]
                point = compute_3d_position(coords, artifacts.depth, artifacts.intrinsics)
                if point is None:
                    raise RuntimeError(f"{primitive.name}: back-projection returned no point for {pixel}")

                depth_offset = float(primitive.parameters.get("depth_offset_m", 0.0) or 0.0)
                if depth_offset:
                    point = [point[0], point[1], point[2] + depth_offset]

                primitive.parameters["target_position"] = point
                primitive.parameters.pop("depth_offset_m", None)

            # Transform coordinates to base frame
            if cam_pose:
                for key in ("target_position", "pivot_point"):
                    if key not in primitive.parameters:
                        continue
                    pos = primitive.parameters[key]
                    base_pos = cam_pose.rotation.apply(pos) + cam_pose.position
                    primitive.parameters[key] = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                    self.logger.debug("Transformed %s: %s -> %s", key, pos, primitive.parameters[key])

        validation_errors = plan.validate(PRIMITIVE_LIBRARY)
        if validation_errors:
            raise ValueError(f"Plan validation failed: {validation_errors}")
        self.logger.info("[prepare_plan] Validation passed")

        return plan

    # ------------------------------------------------------------------ #
    # Result normalization
    # ------------------------------------------------------------------ #
    def _json_safe(self, value: Any) -> Any:
        """
        Best-effort conversion of planner return values into JSON-safe objects.
        Keeps executor behavior unchanged while avoiding serialization failures.
        """
        # Primitive scalars / passthrough
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Containers
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        # Torch tensors
        try:  # Lazy import to avoid hard dependency when unused
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
        except Exception:
            pass

        # NumPy arrays
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        # cuRobo JointState
        try:
            from curobo.types.state import JointState  # type: ignore

            if isinstance(value, JointState):
                return {
                    "position": self._json_safe(getattr(value, "position", None)),
                    "velocity": self._json_safe(getattr(value, "velocity", None)),
                    "acceleration": self._json_safe(getattr(value, "acceleration", None)),
                    "jerk": self._json_safe(getattr(value, "jerk", None)),
                    "joint_names": self._json_safe(getattr(value, "joint_names", None)),
                }
        except Exception:
            pass

        # Generic dataclasses
        if is_dataclass(value):
            return {k: self._json_safe(v) for k, v in asdict(value).items()}

        # Fallback to string representation
        return str(value)
