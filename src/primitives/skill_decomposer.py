"""
Skill decomposer that turns symbolic actions into validated primitive calls.

The decomposer pulls a world slice from the TaskOrchestrator (registry + snapshots),
builds a prompt that includes primitive documentation, and asks Gemini to emit
JSON aligned to the PrimitiveCall schema. Responses are validated against the
primitive library before being returned to the caller.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from google import genai
from google.genai import types

from src.primitives.skill_plan_types import (
    PrimitiveCall,
    SkillPlan,
    SkillPlanDiagnostics,
    compute_registry_hash,
)
from src.planning.task_orchestrator import TaskOrchestrator
from src.planning.utils.snapshot_utils import SnapshotArtifacts, SnapshotCache, load_snapshot_artifacts

# Default prompts config path (all prompt text lives in YAML)
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "skill_decomposer_prompts.yaml"

class SkillDecomposer:
    """LLM-backed decomposer that maps symbolic actions to executable primitives."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gemini-robotics-er-1.5-preview",
        orchestrator: Optional[TaskOrchestrator] = None,
        primitive_catalog_path: Optional[Path] = None,
        prompts_config_path: Optional[Path] = None,
        cache_enabled: bool = True,
        client: Optional[genai.Client] = None,
    ):
        self.model_name = model_name or "gemini-robotics-er-1.5-preview"
        self.client = client or genai.Client(api_key=api_key)
        self.orchestrator = orchestrator
        self.cache_enabled = cache_enabled
        self._plan_cache: Dict[str, SkillPlan] = {}

        root = Path(__file__).resolve().parents[2]
        default_catalog = root / "config" / "primitive_descriptions.md"
        self.primitive_catalog_path = Path(primitive_catalog_path or default_catalog)
        self._catalog_cache: Tuple[float, str] = (0.0, "")
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

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def plan(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        world_hint: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        temperature: float = 0.1,
    ) -> SkillPlan:
        """
        Generate a primitive skill plan for a symbolic action.

        Args:
            action_name: Symbolic action name (e.g., "pick", "place").
            parameters: Action parameters (object ids, target poses, etc.).
            world_hint: Optional world slice override (registry, snapshots).
            use_cache: Reuse cached plan when registry hash matches.
            temperature: LLM temperature for Gemini.

        Returns:
            Validated SkillPlan
        """
        world_state = self._prepare_world_state(world_hint, parameters)
        registry_hash = compute_registry_hash(world_state.get("registry", {}))

        cache_key = self._make_cache_key(action_name, parameters, registry_hash)
        if self.cache_enabled and use_cache and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        catalog_text = self._load_primitive_catalog()
        snapshot_artifacts = load_snapshot_artifacts(
            world_state, self._perception_pool_dir, cache=self._snapshot_cache
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

        if self.cache_enabled:
            self._plan_cache[cache_key] = plan

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

        world_state["registry"] = registry
        world_state["relevant_objects"] = self._filter_relevant_objects(
            objects, parameters=parameters
        )
        return world_state

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
    def _load_primitive_catalog(self) -> str:
        """Load primitive docs from disk with a simple mtime cache."""
        path = self.primitive_catalog_path
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return ""

        cached_mtime, cached_text = self._catalog_cache
        if mtime == cached_mtime and cached_text:
            return cached_text

        text = path.read_text()
        self._catalog_cache = (mtime, text)
        return text

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
        perception_context = self._format_perception_context(world_state, snapshot_artifacts)

        def _format_obj(obj: Dict[str, Any]) -> str:
            affordances = ", ".join(obj.get("affordances", []))
            i_points = obj.get("interaction_points", {}) or {}
            ip_short = ", ".join(
                f"{k}@{v.get('snapshot_id', obj.get('latest_observation'))}"
                for k, v in i_points.items()
            )
            stale = obj.get("staleness_seconds")
            stale_note = f"{stale:.1f}s old" if stale is not None else "freshness: unknown"
            return (
                f"- {obj.get('object_type')} ({obj.get('object_id')}): "
                f"affordances=[{affordances}] "
                f"interaction_points=[{ip_short}] "
                f"latest_snapshot={obj.get('latest_observation')} "
                f"{stale_note}"
            )

        object_section = "\n".join(_format_obj(o) for o in relevant[:10]) or "none"

        return template.format(
            primitive_catalog=primitive_catalog.strip(),
            action_name=action_name,
            parameters=json.dumps(parameters, ensure_ascii=True),
            object_section=object_section,
            last_snapshot_id=snapshot_artifacts.snapshot_id,
            perception_context=perception_context,
        )

    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        media_parts: Optional[List[types.Part]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send prompt (plus optional media) to Gemini and return raw text."""
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": 0.8,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
            "thinking_config": types.ThinkingConfig(thinking_budget=-1),
        }
        if response_schema:
            config_kwargs["response_json_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        contents: List[Any] = []
        if media_parts:
            contents.extend(media_parts)
        contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text

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
            new_interaction_points=data.get("new_interaction_points") or [],
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
        missing_requests: Dict[str, Dict[str, Any]] = {}

        def _track_request(object_id: str, obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not object_id or obj is None:
                return {}
            payload = missing_requests.setdefault(
                object_id,
                {
                    "object": obj,
                    "desired_affordances": set(),
                    "needs_general_scan": False,
                    "source_indices": set(),
                },
            )
            return payload

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
                ip = (obj.get("interaction_points") or {}).get(ip_label)
                if not ip:
                    plan.diagnostics.warnings.append(
                        f"[{idx}] missing interaction point '{ip_label}' on {ref_id}"
                    )
                    payload = _track_request(ref_id, obj)
                    if payload:
                        payload["desired_affordances"].add(ip_label)
                        payload["source_indices"].add(idx)
                else:
                    primitive.metadata.setdefault("resolved_interaction_point", ip)
            if ref_id and ref_id in indexed:
                obj = indexed[ref_id]
                if not (obj.get("interaction_points") or {}):
                    payload = _track_request(ref_id, obj)
                    if payload:
                        payload["needs_general_scan"] = True
                        payload["source_indices"].add(idx)

        # Bubble object staleness into diagnostics
        for obj in world_state.get("relevant_objects", []):
            staleness = obj.get("staleness_seconds")
            if staleness is not None and staleness > 90:
                note = f"Object {obj.get('object_id')} observation is {staleness:.1f}s old"
                plan.diagnostics.warnings.append(note)
                plan.diagnostics.freshness_notes.append(note)

        if missing_requests:
            new_points, ip_warnings = self._synthesize_interaction_points(
                world_state, snapshot_artifacts, missing_requests
            )
            if ip_warnings:
                plan.diagnostics.warnings.extend(ip_warnings)
            if new_points:
                plan.diagnostics.new_interaction_points.extend(new_points)
                self._attach_candidate_points(plan, new_points)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
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

    def _build_media_parts(self, snapshot_artifacts: SnapshotArtifacts) -> List[types.Part]:
        if not snapshot_artifacts.color_bytes:
            return []
        return [
            types.Part.from_bytes(
                data=snapshot_artifacts.color_bytes,
                mime_type="image/png",
            )
        ]

    def _make_cache_key(
        self, action_name: str, parameters: Dict[str, Any], registry_hash: str
    ) -> str:
        payload = json.dumps({"action": action_name, "parameters": parameters}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{action_name}:{registry_hash}:{digest}"


    def _build_interaction_point_prompt(
        self,
        world_state: Dict[str, Any],
        snapshot_artifacts: SnapshotArtifacts,
        requests: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Build a Gemini Robotics-ER prompt asking for new affordance interaction points.
        """
        perception_context = self._format_perception_context(world_state, snapshot_artifacts)
        snapshot_id = snapshot_artifacts.snapshot_id or world_state.get("last_snapshot_id")

        request_blocks: List[str] = []
        for object_id, payload in requests.items():
            obj = payload.get("object") or {}
            desired: List[str] = sorted(payload.get("desired_affordances") or [])
            if payload.get("needs_general_scan"):
                desired.append("any stable affordance supporting grasp/push")
            bbox = obj.get("latest_bounding_box_2d") or obj.get("latest_position_2d")
            existing = ", ".join((obj.get("interaction_points") or {}).keys()) or "none"
            staleness = obj.get("staleness_seconds")
            request_blocks.append(
                "\n".join(
                    [
                        f"object_id: {object_id}",
                        f"object_type: {obj.get('object_type')}",
                        f"latest_snapshot: {obj.get('latest_observation')}",
                        f"bounding_box_2d: {bbox}",
                        f"latest_position_3d: {obj.get('latest_position_3d')}",
                        f"existing_interaction_points: {existing}",
                        f"requested_affordances: {desired or ['grasp']}",
                        f"staleness_seconds: {staleness}",
                    ]
                )
            )

        if not request_blocks:
            return ""

        request_section = "\n".join(request_blocks)

        response_schema = (
            "{\n"
            '  "interaction_points": [\n'
            "    {\n"
            '      "object_id": "id",\n'
            '      "affordance": "grasp",\n'
            '      "position_2d": [y, x],  # normalized 0-1000 per Gemini Robotics-ER demo\n'
            '      "confidence": 0.0-1.0,\n'
            '      "rationale": "why this point works"\n'
            "    }\n"
            "  ],\n"
            '  "notes": ["optional warnings when object not visible"]\n'
            "}"
        )

        header = (
            "You are the Gemini Robotics-ER 1.5 perception assistant. "
            "Use the attached RGB snapshot to locate precise affordance-level interaction points. "
            "Follow the object-tracking demo format (normalized [y, x] coordinates in the 0-1000 range) "
            "and only emit JSON that adheres to the schema below."
        )

        return (
            f"{header}\n\n"
            f"Snapshot id: {snapshot_id}\n"
            f"Perception context: {perception_context}\n"
            "Objects needing interaction points:\n"
            f"{'-'*48}\n"
            f"{request_section}\n"
            f"{'-'*48}\n"
            "Response schema:\n"
            f"{response_schema}"
        )

    def _synthesize_interaction_points(
        self,
        world_state: Dict[str, Any],
        snapshot_artifacts: SnapshotArtifacts,
        requests: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Invoke Gemini Robotics-ER to propose new interaction points when the registry lacks them.
        """
        warnings: List[str] = []
        if not requests:
            return [], warnings

        media_parts = self._build_media_parts(snapshot_artifacts)
        if not media_parts:
            warnings.append(
                "No snapshot media available for affordance enrichment; "
                "capture a snapshot to enable Gemini Robotics-ER image grounding."
            )
            return [], warnings

        prompt = self._build_interaction_point_prompt(world_state, snapshot_artifacts, requests)
        if not prompt:
            return [], warnings

        try:
            response_text = self._call_llm(prompt, temperature=0.0, media_parts=media_parts)
        except Exception as exc:
            warnings.append(f"Interaction point enrichment failed: {exc}")
            return [], warnings

        try:
            points = self._parse_interaction_points_response(response_text)
        except ValueError as exc:
            warnings.append(f"Could not parse interaction point response: {exc}")
            return [], warnings

        return points, warnings

    def _parse_interaction_points_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse Gemini Robotics-ER interaction point JSON payloads.
        """
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

        raw_points = (
            data.get("interaction_points")
            or data.get("new_interaction_points")
            or data.get("points")
            or []
        )
        if not isinstance(raw_points, list):
            raise ValueError("interaction_points must be a list")

        cleaned: List[Dict[str, Any]] = []
        for entry in raw_points:
            if not isinstance(entry, dict):
                continue
            object_id = entry.get("object_id")
            affordance = entry.get("affordance") or entry.get("label")
            if not object_id or not affordance:
                continue
            cleaned.append(
                {
                    "object_id": object_id,
                    "affordance": affordance,
                    "position_2d": entry.get("position_2d") or entry.get("point"),
                    "position_3d": entry.get("position_3d"),
                    "confidence": entry.get("confidence"),
                    "rationale": entry.get("rationale") or entry.get("reason"),
                    "source": entry.get("source") or "gemini_robotics_er_1_5",
                }
            )
        return cleaned

    def _attach_candidate_points(self, plan: SkillPlan, candidates: List[Dict[str, Any]]) -> None:
        """
        Thread newly proposed interaction points back into primitive metadata for executors.
        """
        if not candidates:
            return

        by_object: Dict[str, List[Dict[str, Any]]] = {}
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for point in candidates:
            object_id = point.get("object_id")
            affordance = point.get("affordance")
            if not object_id or not affordance:
                continue
            by_object.setdefault(object_id, []).append(point)
            by_key[(object_id, affordance)] = point

        for primitive in plan.primitives:
            ref_id = primitive.references.get("object_id")
            if not ref_id:
                continue
            ip_label = primitive.references.get("interaction_point")
            candidate = None
            if ip_label:
                candidate = by_key.get((ref_id, ip_label))
            if candidate is None:
                obj_candidates = by_object.get(ref_id)
                if obj_candidates:
                    candidate = obj_candidates[0]
            if candidate:
                primitive.metadata.setdefault("proposed_interaction_point", candidate)
