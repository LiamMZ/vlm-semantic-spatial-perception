"""
Skill decomposer that turns symbolic actions into validated primitive calls.

The decomposer pulls a world slice from the TaskOrchestrator (registry + snapshots),
builds a prompt that includes primitive documentation, and asks an LLM to emit
JSON aligned to the PrimitiveCall schema. Responses are validated against the
primitive library before being returned to the caller.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore
    _genai_types = None  # type: ignore
    _GENAI_AVAILABLE = False

from src.llm_interface.base import GenerateConfig, ImagePart, LLMClient

from src.primitives.skill_plan_types import (
    PrimitiveCall,
    SkillPlan,
    SkillPlanDiagnostics,
    compute_registry_hash,
)
from src.planning.task_orchestrator import TaskOrchestrator
from src.planning.utils.snapshot_utils import (
    SnapshotArtifacts,
    SnapshotCache,
    latest_snapshot_for_object_ids,
    load_snapshot_artifacts,
)

# Default prompts config path (all prompt text lives in YAML)
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "skill_decomposer_prompts.yaml"
DEFAULT_PRIMITIVE_CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "primitive_descriptions.md"


class SkillDecomposer:
    """LLM-backed decomposer that maps symbolic actions to executable primitives."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        orchestrator: Optional[TaskOrchestrator] = None,
        primitive_catalog_path: Optional[Path] = None,
        prompts_config_path: Optional[Path] = None,
        client: Optional[Any] = None,
        llm_config_kwargs: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self._llm_client = llm_client
        self.orchestrator = orchestrator

        root = Path(__file__).resolve().parents[2]
        self.primitive_catalog_path = Path(primitive_catalog_path or DEFAULT_PRIMITIVE_CATALOG_PATH)
        self.prompts_config_path = Path(prompts_config_path or DEFAULT_PROMPTS_PATH)
        self._prompts_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self._snapshot_cache = SnapshotCache()

        # Use orchestrator config to locate registry on disk if available
        self._state_dir = (
            orchestrator.config.state_dir
            if orchestrator and getattr(orchestrator, "config", None)
            else root / "outputs" / "orchestrator_state"
        )
        if orchestrator and getattr(orchestrator.config, "perception_pool_dir", None):
            self._perception_pool_dir = Path(orchestrator.config.perception_pool_dir)
        else:
            self._perception_pool_dir = Path(self._state_dir) / "perception_pool"

        # Legacy Gemini client (used only when llm_client is not provided)
        if llm_client is None:
            if not _GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai is required when llm_client is not provided. "
                    "Install it or pass an llm_client."
                )
            self.model_name = model_name
            self.client = client or _genai.Client(api_key=api_key)
            _THINKING_MODELS = ("gemini-2.5",)
            _supports_thinking = any(t in model_name for t in _THINKING_MODELS)
            default_llm_config: Dict[str, Any] = {
                "top_p": 0.8,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            }
            if _supports_thinking:
                default_llm_config["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=-1)
            self.llm_config_kwargs = {**default_llm_config, **(llm_config_kwargs or {})}
        else:
            self.model_name = ""
            self.client = None
            self.llm_config_kwargs = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def plan(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        world_hint: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> SkillPlan:
        """
        Generate a primitive skill plan for a symbolic action.

        Args:
            action_name: Symbolic action name (e.g., "pick", "place").
            parameters: Action parameters (object ids, target poses, etc.).
            world_hint: Optional world slice override (registry, snapshots).
            temperature: LLM temperature for Gemini.

        Returns:
            Validated SkillPlan
        """
        world_state = self._prepare_world_state(world_hint, parameters)
        registry_hash = compute_registry_hash(world_state.get("registry", {}))

        catalog_text = self.primitive_catalog_path.read_text()
        target_object_ids = self._extract_object_ids(parameters)
        snapshot_id = latest_snapshot_for_object_ids(
            world_state, self._perception_pool_dir, target_object_ids, cache=self._snapshot_cache
        ) or world_state.get("last_snapshot_id")
        snapshot_artifacts = load_snapshot_artifacts(
            world_state, self._perception_pool_dir, cache=self._snapshot_cache, snapshot_id=snapshot_id
        )
        prompts = self._load_prompts_config()
        prompt = self._build_prompt(
            action_name,
            parameters,
            world_state,
            catalog_text,
            snapshot_artifacts,
            prompts["template"],
        )
        media_parts = self._build_media_parts(snapshot_artifacts)
        response_text = self._call_llm(
            prompt,
            temperature=temperature,
            media_parts=media_parts,
            response_schema=prompts["response_schema"],
        )

        plan = self._parse_plan(
            response_text, action_name=action_name, registry_hash=registry_hash
        )
        plan.source_snapshot_id = snapshot_artifacts.snapshot_id
        self._post_process_plan(plan, world_state, snapshot_artifacts)

        return plan

    # --------------------------------------------------------------------- #
    # World state helpers
    # --------------------------------------------------------------------- #
    def _prepare_world_state(
        self, world_hint: Optional[Dict[str, Any]], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gather registry, snapshots, and robot state for prompting.
        """
        if world_hint is None:
            world_hint = {}

        # Prefer live orchestrator state
        if self.orchestrator is not None:
            world_state = self.orchestrator.get_world_state_snapshot()
        else:
            world_state = {}

        # Disk fallback when orchestrator is not running
        if not world_state.get("registry"):
            registry_path = Path(self._state_dir) / "registry.json"
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    world_state["registry"] = json.load(f)
        if "last_snapshot_id" not in world_state:
            state_path = Path(self._state_dir) / "state.json"
            if state_path.exists():
                try:
                    with open(state_path, "r") as f:
                        state_payload = json.load(f)
                        world_state["last_snapshot_id"] = state_payload.get("last_snapshot_id")
                except Exception:
                    pass

        # Merge user-provided hints last
        world_state.update(world_hint)

        registry = world_state.get("registry", {})
        objects = registry.get("objects", [])
        now = time.time()
        for obj in objects:
            ts = obj.get("timestamp")
            if isinstance(ts, (int, float)):
                obj["staleness_seconds"] = max(0.0, now - ts)

        snapshot_id = world_state.get("last_snapshot_id")
        world_state["latest_detections"] = self._load_snapshot_detections(snapshot_id)
        detection_map = {d.get("object_id"): d for d in world_state["latest_detections"]}
        merged_objects: List[Dict[str, Any]] = []
        for obj in registry.get("objects", []):
            detection = detection_map.get(obj.get("object_id")) or {}
            merged = dict(obj)
            if detection:
                merged.setdefault("interaction_points", detection.get("interaction_points"))
                merged.setdefault("latest_observation", snapshot_id)
                if detection.get("bounding_box_2d") is not None:
                    merged.setdefault("latest_bounding_box_2d", detection.get("bounding_box_2d"))
                if detection.get("position_2d") is not None:
                    merged.setdefault("latest_position_2d", detection.get("position_2d"))
                if detection.get("position_3d") is not None:
                    merged.setdefault("latest_position_3d", detection.get("position_3d"))
            merged_objects.append(merged)
        registry["objects"] = merged_objects
        world_state["registry"] = registry
        world_state["relevant_objects"] = self._filter_relevant_objects(
            merged_objects, parameters=parameters
        )
        return world_state

    def _extract_object_ids(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Pull likely target object ids from parameters.
        """
        ids: List[str] = []
        oid = parameters.get("object_id")
        if isinstance(oid, str):
            ids.append(oid)
        elif isinstance(oid, (list, tuple)):
            ids.extend([str(v) for v in oid])

        oid_plural = parameters.get("object_ids")
        if isinstance(oid_plural, (list, tuple)):
            ids.extend([str(v) for v in oid_plural if v is not None])

        return [i for i in ids if i]

    def _filter_relevant_objects(
        self, objects: List[Dict[str, Any]], parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter objects based on symbolic parameters (id or type matches).
        """
        if not objects:
            return []

        target_tokens: List[str] = []
        for value in parameters.values():
            if isinstance(value, str):
                target_tokens.append(value.lower())
            elif isinstance(value, (list, tuple)):
                target_tokens.extend(
                    [str(v).lower() for v in value if isinstance(v, (str, int))]
                )

        if not target_tokens:
            return objects[:5]

        relevant: List[Dict[str, Any]] = []
        for obj in objects:
            oid = str(obj.get("object_id", "")).lower()
            otype = str(obj.get("object_type", "")).lower()
            if any(token in (oid, otype) for token in target_tokens):
                relevant.append(obj)

        return relevant or objects[:5]

    # --------------------------------------------------------------------- #
    # Prompt assembly and LLM call
    # --------------------------------------------------------------------- #
    def _build_prompt(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        world_state: Dict[str, Any],
        primitive_catalog: str,
        snapshot_artifacts: SnapshotArtifacts,
        template: str,
    ) -> str:
        """
        Build a structured prompt for Gemini.
        """
        registry = world_state.get("registry", {})
        relevant = world_state.get("relevant_objects") or registry.get("objects", [])
        detection_map = {det.get("object_id"): det for det in world_state.get("latest_detections") or []}
        perception_context = self._format_perception_context(world_state, snapshot_artifacts)

        def _format_obj(obj: Dict[str, Any]) -> str:
            detection = detection_map.get(obj.get("object_id")) or {}
            stale = obj.get("staleness_seconds")
            stale_note = f"{stale:.1f}s old" if stale is not None else "freshness: unknown"
            pos3d = obj.get("position_3d")
            pos_note = f" position_3d={[round(v, 3) for v in pos3d]}" if pos3d else ""
            return (
                f"- {obj.get('object_type')} ({obj.get('object_id')}):"
                f"{pos_note}"
                f" latest_snapshot={detection.get('snapshot_id') or obj.get('latest_observation')}"
                f" {stale_note}"
            )

        object_section = "\n".join(_format_obj(o) for o in relevant[:10]) or "none"

        # Use simple sequential replacement instead of str.format() to avoid
        # IndexError / KeyError when substituted values contain curly braces
        # (e.g. JSON dicts in object_section or perception_context).
        action_description = self._resolve_action_description(action_name)
        substitutions = {
            "{primitive_catalog}": primitive_catalog.strip(),
            "{action_name}": action_name,
            "{action_description}": action_description,
            "{parameters}": json.dumps(parameters, ensure_ascii=True),
            "{object_section}": object_section,
            "{last_snapshot_id}": str(snapshot_artifacts.snapshot_id),
            "{perception_context}": perception_context,
        }
        result = template
        for placeholder, value in substitutions.items():
            result = result.replace(placeholder, value)
        return result

    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        media_parts: Optional[List[Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send prompt (plus optional media) to the LLM and return raw text."""
        if self._llm_client is not None:
            config = GenerateConfig(
                temperature=temperature,
                top_p=0.8,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            )
            contents: List[Any] = []
            if media_parts:
                contents.extend(media_parts)
            contents.append(prompt)
            response = self._llm_client.generate(
                contents if len(contents) > 1 else prompt,
                config=config,
            )
            print(f"  [SkillDecomposer] raw LLM response ({len(response.text)} chars): {response.text!r}")
            return response.text

        # Legacy Gemini path
        config_kwargs: Dict[str, Any] = {**self.llm_config_kwargs, "temperature": temperature}
        if response_schema:
            config_kwargs["response_json_schema"] = response_schema

        config_gemini = _genai_types.GenerateContentConfig(**config_kwargs)

        gemini_contents: List[Any] = []
        if media_parts:
            gemini_contents.extend(media_parts)
        gemini_contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config_gemini,
        )
        text = getattr(response, "text", None)
        if text is None:
            raise ValueError("LLM response missing text payload")
        return text if isinstance(text, str) else str(text)

    @property
    def llm_config_kwargs(self) -> Dict[str, Any]:
        return self._llm_config_kwargs

    @llm_config_kwargs.setter
    def llm_config_kwargs(self, value: Dict[str, Any]) -> None:
        self._llm_config_kwargs = dict(value) if value is not None else {}

    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompt template and response schema from YAML."""
        path = self.prompts_config_path
        mtime = path.stat().st_mtime  # Will raise if missing to avoid hidden inline defaults
        cached_mtime, cached_prompts = self._prompts_cache
        if mtime == cached_mtime and cached_prompts:
            return cached_prompts

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        template = data.get("template")
        response_schema = data.get("response_schema")
        if not template or not isinstance(response_schema, dict):
            raise ValueError(f"Prompt config {path} must define 'template' and 'response_schema'")

        prompts = {
            "template": template,
            "response_schema": json.loads(json.dumps(response_schema)),
        }
        self._prompts_cache = (mtime, prompts)
        return prompts

    # --------------------------------------------------------------------- #
    # Parsing and validation
    # --------------------------------------------------------------------- #
    def _parse_plan(
        self, response_text: str, action_name: str, registry_hash: Optional[str]
    ) -> SkillPlan:
        """Parse LLM JSON into a SkillPlan."""
        data = json.loads(response_text)

        primitives = [
            PrimitiveCall.from_dict(item) for item in data.get("primitives", [])
        ]

        diagnostics_block = data.get("diagnostics") or {}
        diagnostics = SkillPlanDiagnostics(
            assumptions=data.get("assumptions") or [],
            warnings=diagnostics_block.get("warnings") or [],
            freshness_notes=diagnostics_block.get("freshness_notes") or [],
            freshness=diagnostics_block.get("freshness", {}),
            rationale=diagnostics_block.get("rationale", ""),
            interaction_points=data.get("interaction_points") or [],
        )

        plan = SkillPlan(
            action_name=action_name,
            primitives=primitives,
            diagnostics=diagnostics,
            registry_hash=registry_hash,
        )
        return plan

    def _post_process_plan(
        self,
        plan: SkillPlan,
        world_state: Dict[str, Any],
        snapshot_artifacts: SnapshotArtifacts,
    ) -> None:
        """
        Resolve references against world data and append warnings for gaps.
        """
        objects = world_state.get("registry", {}).get("objects", [])
        indexed = {obj.get("object_id"): obj for obj in objects}
        detection_map = {
            det.get("object_id"): det for det in world_state.get("latest_detections") or []
        }

        for idx, primitive in enumerate(plan.primitives):
            ref_id = primitive.references.get("object_id")
            if ref_id and ref_id not in indexed:
                plan.diagnostics.warnings.append(
                    f"[{idx}] reference object '{ref_id}' not found in registry"
                )
                continue
            ip_label = primitive.references.get("interaction_point")
            if ref_id and ip_label:
                obj = indexed.get(ref_id, {})
                detection = detection_map.get(ref_id, {})
                ip = (detection.get("interaction_points") or {}).get(ip_label)
                if not ip:
                    plan.diagnostics.warnings.append(
                        f"[{idx}] missing interaction point '{ip_label}' on {ref_id}"
                    )
                else:
                    primitive.metadata.setdefault("resolved_interaction_point", ip)
            if ref_id and ref_id in indexed:
                obj = indexed[ref_id]
                detection = detection_map.get(ref_id, {})
                if not (detection.get("interaction_points") or {}):
                    plan.diagnostics.warnings.append(
                        f"[{idx}] object '{ref_id}' has no interaction points in snapshot {detection.get('snapshot_id')}"
                    )

        # pointing_guidance grounding is deferred to PrimitiveExecutor.prepare_plan
        # so that Molmo runs at execution time against the freshest snapshot.
        # Set point_label fallback now so the plan is valid even without a detector.
        objects = world_state.get("registry", {}).get("objects", [])
        indexed = {obj.get("object_id"): obj for obj in objects}
        for idx, primitive in enumerate(plan.primitives):
            if primitive.name != "move_gripper_to_pose":
                continue
            guidance = primitive.metadata.get("pointing_guidance")
            if not guidance:
                continue
            ref_id = primitive.references.get("object_id")
            if ref_id:
                primitive.parameters.setdefault("point_label", ref_id)
                plan.diagnostics.warnings.append(
                    f"[{idx}] pointing_guidance deferred to execution time; point_label={ref_id!r} set as fallback"
                )

        # Bubble object staleness into diagnostics
        for obj in world_state.get("relevant_objects", []):
            staleness = obj.get("staleness_seconds")
            if staleness is not None and staleness > 90:
                note = f"Object {obj.get('object_id')} observation is {staleness:.1f}s old"
                plan.diagnostics.warnings.append(note)
                plan.diagnostics.freshness_notes.append(note)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def _resolve_action_description(self, action_name: str) -> str:
        """
        Look up the human-readable description for a PDDL action.

        Priority:
          1. Live domain from orchestrator.task_analysis.action_context()
             (covers both hardcoded clutter actions and LLM-generated ones)
          2. Hardcoded _CLUTTER_ACTIONS fallback (when orchestrator is absent)
          3. Empty string (graceful degradation)
        """
        # 1. Live domain
        if self.orchestrator is not None:
            task_analysis = getattr(self.orchestrator, "task_analysis", None)
            if task_analysis is not None:
                for action in task_analysis.action_context():
                    if action.get("name") == action_name:
                        desc = action.get("description", "")
                        if desc:
                            return desc

        # 2. Hardcoded clutter actions fallback
        from src.planning.clutter_module import _CLUTTER_ACTIONS
        for action in _CLUTTER_ACTIONS:
            if action.get("name") == action_name:
                desc = action.get("description", "")
                if desc:
                    return desc

        return ""

    def _format_perception_context(
        self,
        world_state: Dict[str, Any],
        snapshot_artifacts: SnapshotArtifacts,
    ) -> str:
        """
        Summarize available perception artifacts (snapshots, registry, robot state).
        """
        last_snap = snapshot_artifacts.snapshot_id or world_state.get("last_snapshot_id")
        snap_meta = snapshot_artifacts.meta
        if snap_meta is None and last_snap:
            snap_index = world_state.get("snapshot_index") or {}
            snap_meta = (snap_index.get("snapshots") or {}).get(last_snap)

        def _fmt_snapshot(meta: Optional[Dict[str, Any]]) -> str:
            if not meta:
                return "none"
            files = meta.get("files", {})
            return (
                f"id={last_snap}, captured_at={meta.get('captured_at')}, "
                f"color={files.get('color')}, depth={files.get('depth_npz')}, "
                f"intrinsics={files.get('intrinsics')}, detections={files.get('detections')}"
            )

        robot_state = world_state.get("robot_state") or {}
        robot_provider = robot_state.get("provider", "unknown")
        robot_stamp = robot_state.get("stamp")

        registry_meta = world_state.get("registry", {})
        num_objects = registry_meta.get("num_objects", "unknown")
        detection_ts = registry_meta.get("detection_timestamp")

        return (
            f"registry: objects={num_objects}, detection_ts={detection_ts}; "
            f"latest_snapshot: {_fmt_snapshot(snap_meta)}; "
            f"robot_state: provider={robot_provider}, stamp={robot_stamp}"
        )

    def _build_media_parts(self, snapshot_artifacts: SnapshotArtifacts) -> List[Any]:
        if not snapshot_artifacts.color_bytes:
            return []
        if self._llm_client is not None:
            return [ImagePart(data=snapshot_artifacts.color_bytes, mime_type="image/png")]
        return [
            _genai_types.Part.from_bytes(
                data=snapshot_artifacts.color_bytes,
                mime_type="image/png",
            )
        ]

    def _load_snapshot_detections(self, snapshot_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Load detections (including interaction points) for a snapshot to ground prompts.
        """
        if not snapshot_id:
            return []
        det_path = Path(self._perception_pool_dir) / "snapshots" / snapshot_id / "detections.json"
        if not det_path.exists():
            return []
        try:
            payload = json.loads(det_path.read_text())
        except Exception:
            return []
        detections = payload.get("objects") or []
        for det in detections:
            det.setdefault("snapshot_id", snapshot_id)
        return detections
