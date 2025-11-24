"""
Primitive executor that translates Gemini-friendly parameters before calling CuRobo primitives.

LLM outputs reference image-grounded cues (pixel [y, x] pointers, normals, standoffs).
This executor back-projects those cues into metric coordinates using the latest snapshot depth and
camera intrinsics, validates the plan, and optionally drives the configured motion planner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.perception.utils.coordinates import compute_3d_position, normalized_to_pixel, pixel_to_normalized
from src.primitives.skill_plan_types import PRIMITIVE_LIBRARY, PrimitiveCall, SkillPlan
from src.planning.utils.snapshot_utils import SnapshotArtifacts, SnapshotCache, load_snapshot_artifacts


@dataclass
class PrimitiveExecutionResult:
    """Return payload for executor runs."""

    executed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    primitive_results: List[Any] = field(default_factory=list)


class PrimitiveExecutor:
    """Translate and execute primitive plans against the configured primitives interface."""

    def __init__(self, primitives: Optional[Any], perception_pool_dir: Path):
        self.primitives = primitives
        self.perception_pool_dir = Path(perception_pool_dir)
        self._snapshot_cache = SnapshotCache()

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
        translated_plan, warnings, errors = self.prepare_plan(plan, world_state)
        if errors:
            return PrimitiveExecutionResult(executed=False, warnings=warnings, errors=errors)
        if dry_run or self.primitives is None:
            return PrimitiveExecutionResult(executed=False, warnings=warnings)

        primitive_results: List[Any] = []
        runtime_errors: List[str] = []
        for idx, primitive in enumerate(translated_plan.primitives):
            schema = PRIMITIVE_LIBRARY.get(primitive.name)
            if (
                self.primitives is not None
                and not dry_run
                and schema is not None
                and "execute" in schema.optional_params
            ):
                primitive.parameters.setdefault("execute", True)
            method = getattr(self.primitives, primitive.name, None)
            if not callable(method):
                runtime_errors.append(f"[{idx}] primitives missing primitive '{primitive.name}'")
                break
            try:
                raw_result = method(**primitive.parameters)
                result = self._json_safe(raw_result)
                primitive_results.append(result)
            except Exception as exc:  # pragma: no cover - hardware dependent
                runtime_errors.append(f"[{idx}] execution failed for '{primitive.name}': {exc}")
                break

        executed = not runtime_errors
        warnings.extend(runtime_errors)
        return PrimitiveExecutionResult(
            executed=executed,
            warnings=warnings,
            errors=[] if executed else runtime_errors,
            primitive_results=primitive_results,
        )

    def prepare_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
    ) -> Tuple[SkillPlan, List[str], List[str]]:
        """
        Translate parameters and validate the plan without executing it.
        """
        warnings: List[str] = []
        errors: List[str] = []

        artifacts = load_snapshot_artifacts(
            world_state,
            self.perception_pool_dir,
            cache=self._snapshot_cache,
            snapshot_id=getattr(plan, "source_snapshot_id", None),
        )
        warnings.extend(self._robot_pose_warnings(world_state, artifacts))
        if getattr(plan, "source_snapshot_id", None) and plan.source_snapshot_id != artifacts.snapshot_id:
            warnings.append(
                f"Plan snapshot {plan.source_snapshot_id} missing on disk; "
                f"using {artifacts.snapshot_id or 'latest available'} instead"
            )
        warnings.extend(self._translate_plan(plan, artifacts))

        validation_errors = plan.validate(PRIMITIVE_LIBRARY)
        if validation_errors:
            errors.extend(validation_errors)

        return plan, warnings, errors

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #
    def _translate_plan(self, plan: SkillPlan, artifacts: SnapshotArtifacts) -> List[str]:
        warnings: List[str] = []
        for primitive in plan.primitives:
            warnings.extend(self._translate_helper_parameters(primitive, artifacts))
        return warnings

    def _robot_pose_warnings(
        self,
        world_state: Dict[str, Any],
        artifacts: SnapshotArtifacts,
    ) -> List[str]:
        """
        Surface potential transform mismatch when replaying cached perception with a moved robot.
        """
        warnings: List[str] = []
        snapshot_state = artifacts.robot_state or {}
        live_state = world_state.get("robot_state") or {}
        snap_joints = snapshot_state.get("joints")
        live_joints = live_state.get("joints")
        if snap_joints and live_joints and len(snap_joints) == len(live_joints):
            try:
                max_delta = max(abs(a - b) for a, b in zip(snap_joints, live_joints))
                if max_delta > 0.1:
                    warnings.append(
                        f"Snapshot robot joints differ from current by up to {max_delta:.3f} rad; "
                        "camera transform may be stale for this cached perception."
                    )
            except Exception:
                pass
        return warnings

    def _translate_helper_parameters(
        self,
        primitive: PrimitiveCall,
        artifacts: SnapshotArtifacts,
    ) -> List[str]:
        warnings: List[str] = []
        seeded_target_position = False

        # Prefer resolved interaction point (snapshot-grounded) when available
        rip = (primitive.metadata or {}).get("resolved_interaction_point") or {}
        color_shape = artifacts.color_shape or (artifacts.depth.shape[:2] if artifacts.depth is not None else None)
        pos3d = rip.get("position_3d")
        if isinstance(pos3d, (list, tuple)) and len(pos3d) >= 3:
            primitive.parameters["target_position"] = [float(pos3d[0]), float(pos3d[1]), float(pos3d[2])]
            seeded_target_position = True
        pos2d = rip.get("position_2d")
        if (
            isinstance(pos2d, (list, tuple))
            and len(pos2d) >= 2
            and color_shape is not None
            and "target_pixel_yx" not in primitive.parameters
        ):
            try:
                # position_2d stored as [x, y] normalized (0-1000); convert to pixel [y, x]
                norm_yx = [float(pos2d[1]), float(pos2d[0])]
                pixel_yx = normalized_to_pixel(norm_yx, color_shape)
                primitive.parameters["target_pixel_yx"] = [int(pixel_yx[0]), int(pixel_yx[1])]
            except Exception:
                warnings.append(f"{primitive.name}: unable to convert resolved_interaction_point.position_2d -> pixels")

        helper_map: Tuple[Tuple[str, str, bool, bool], ...] = (
            ("target_pixel_yx", "target_position", True, True),
            ("pivot_pixel_yx", "pivot_point", False, True),
        )
        for helper_key, target_field, apply_offset, is_pixel in helper_map:
            pixel = primitive.parameters.pop(helper_key, None)
            if pixel is None:
                continue
            if target_field == "target_position" and seeded_target_position:
                continue
            point = self._back_project(pixel, artifacts, is_pixel=is_pixel)
            if point is None:
                warnings.append(
                    f"{primitive.name}: unable to back-project pixel {pixel} (missing depth/intrinsics)"
                )
                continue
            if apply_offset:
                depth_offset = float(primitive.parameters.get("depth_offset_m", 0.0) or 0.0)
                if depth_offset:
                    point = [point[0], point[1], point[2] + depth_offset]
            primitive.parameters[target_field] = point
        primitive.parameters.pop("depth_offset_m", None)
        return warnings

    def _back_project(
        self,
        pixel_yx: List[int],
        artifacts: SnapshotArtifacts,
        *,
        is_pixel: bool = True,
    ) -> Optional[List[float]]:
        if artifacts.depth is None or artifacts.intrinsics is None:
            return None
        coords = pixel_yx
        if is_pixel:
            image_shape = artifacts.color_shape
            if image_shape is None and artifacts.depth is not None:
                image_shape = artifacts.depth.shape[:2]
            if image_shape is None:
                return None
            coords = pixel_to_normalized((int(pixel_yx[0]), int(pixel_yx[1])), image_shape)
        point = compute_3d_position(coords, artifacts.depth, artifacts.intrinsics)
        if point is None:
            return None
        return [float(point[0]), float(point[1]), float(point[2])]

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
