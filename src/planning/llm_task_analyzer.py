"""
LLM-powered staged task analysis for dynamic PDDL generation.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from PIL import Image
from google import genai
from google.genai import types

from ..utils.prompt_utils import render_prompt_template
from .utils.task_types import (
    AbstractGoal,
    ActionSchemaLibrary,
    GroundingSummary,
    PredicateInventory,
    TaskAnalysis,
)


class LLMTaskAnalyzer:
    """Analyze tasks with a staged abstraction order.

    Args:
        api_key: Gemini API key. When omitted, the analyzer will look for
            `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment.
        model_name: Model used for all staged analysis calls.
        prompts_config_path: Optional override for the prompt YAML.

    Example:
        >>> analyzer = LLMTaskAnalyzer(api_key="test-key")
        >>> goal = analyzer.analyze_goal("put the cup on the tray")
    """

    DEFAULT_PROMPTS_CONFIG = (
        Path(__file__).resolve().parents[2] / "config" / "llm_task_analyzer_prompts.yaml"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-robotics-er-1.5-preview",
        prompts_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.prompts_config_path = Path(prompts_config_path)

        with self.prompts_config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.prompt_templates = {
            key: value
            for key, value in config_data.items()
            if key.endswith("_prompt") and isinstance(value, str)
        }
        self.robot_description = (config_data.get("robot_description") or "").strip()

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.call_count = 0
        self.total_elapsed_seconds = 0.0

    def analyze_goal(
        self,
        task_description: str,
        observed_objects: Optional[List[Dict[str, Any]]] = None,
        observed_relationships: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 15.0,
    ) -> AbstractGoal:
        """Analyze the task into an abstract goal layer."""

        payload = self._generate_stage_json(
            template_key="goal_prompt",
            replacements={
                "TASK": task_description,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
                "OBJECTS_JSON": self._format_objects_json(observed_objects or []),
                "RELATIONSHIPS_JSON": self._format_relationships_json(observed_relationships or []),
            },
            environment_image=environment_image,
            timeout=timeout,
        )
        return AbstractGoal(
            summary=payload.get("summary", "").strip(),
            goal_literals=self._as_string_list(payload.get("goal_literals")),
            goal_objects=self._as_string_list(payload.get("goal_objects")),
            success_checks=self._as_string_list(payload.get("success_checks")),
        )

    def analyze_predicates(
        self,
        task_description: str,
        abstract_goal: AbstractGoal,
        observed_objects: Optional[List[Dict[str, Any]]] = None,
        observed_relationships: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 15.0,
    ) -> PredicateInventory:
        """Select the minimal predicate inventory needed for the goal."""

        payload = self._generate_stage_json(
            template_key="predicates_prompt",
            replacements={
                "TASK": task_description,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
                "ABSTRACT_GOAL_JSON": json.dumps(self._goal_to_dict(abstract_goal), indent=2),
                "OBJECTS_JSON": self._format_objects_json(observed_objects or []),
                "RELATIONSHIPS_JSON": self._format_relationships_json(observed_relationships or []),
            },
            environment_image=environment_image,
            timeout=timeout,
        )
        return PredicateInventory(
            predicates=self._as_string_list(payload.get("predicates")),
            selection_rationale=self._as_string_list(payload.get("selection_rationale")),
            omitted_predicates=self._as_string_list(payload.get("omitted_predicates")),
        )

    def analyze_actions(
        self,
        task_description: str,
        abstract_goal: AbstractGoal,
        predicate_inventory: PredicateInventory,
        observed_objects: Optional[List[Dict[str, Any]]] = None,
        observed_relationships: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 15.0,
    ) -> ActionSchemaLibrary:
        """Derive action schemas from the goal and predicate inventory."""

        payload = self._generate_stage_json(
            template_key="actions_prompt",
            replacements={
                "TASK": task_description,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
                "ABSTRACT_GOAL_JSON": json.dumps(self._goal_to_dict(abstract_goal), indent=2),
                "PREDICATE_INVENTORY_JSON": json.dumps(
                    {
                        "predicates": predicate_inventory.predicates,
                        "selection_rationale": predicate_inventory.selection_rationale,
                        "omitted_predicates": predicate_inventory.omitted_predicates,
                    },
                    indent=2,
                ),
                "OBJECTS_JSON": self._format_objects_json(observed_objects or []),
                "RELATIONSHIPS_JSON": self._format_relationships_json(observed_relationships or []),
            },
            environment_image=environment_image,
            timeout=timeout,
        )
        actions = payload.get("actions") or []
        if not isinstance(actions, list):
            raise ValueError("Expected `actions` to be a list in actions-stage response")
        normalized_actions: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            normalized_actions.append(
                {
                    "name": str(action.get("name", "")).strip(),
                    "parameters": self._as_string_list(action.get("parameters")),
                    "precondition": str(action.get("precondition", "")).strip(),
                    "effect": str(action.get("effect", "")).strip(),
                    "description": str(action.get("description", "")).strip(),
                }
            )
        return ActionSchemaLibrary(
            actions=normalized_actions,
            planning_notes=self._as_string_list(payload.get("planning_notes")),
        )

    def analyze_grounding(
        self,
        task_description: str,
        abstract_goal: AbstractGoal,
        predicate_inventory: PredicateInventory,
        action_schemas: ActionSchemaLibrary,
        observed_objects: Optional[List[Dict[str, Any]]] = None,
        observed_relationships: Optional[List[str]] = None,
        predicates: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 15.0,
    ) -> GroundingSummary:
        """Ground symbolic references against the observed world."""

        payload = self._generate_stage_json(
            template_key="grounding_prompt",
            replacements={
                "TASK": task_description,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
                "ABSTRACT_GOAL_JSON": json.dumps(self._goal_to_dict(abstract_goal), indent=2),
                "PREDICATE_INVENTORY_JSON": json.dumps(
                    {
                        "predicates": predicate_inventory.predicates,
                        "selection_rationale": predicate_inventory.selection_rationale,
                        "omitted_predicates": predicate_inventory.omitted_predicates,
                    },
                    indent=2,
                ),
                "ACTION_SCHEMAS_JSON": json.dumps(action_schemas.actions, indent=2),
                "OBJECTS_JSON": self._format_objects_json(observed_objects or []),
                "RELATIONSHIPS_JSON": self._format_relationships_json(observed_relationships or []),
                "OBSERVED_PREDICATES_JSON": json.dumps(predicates or [], indent=2),
            },
            environment_image=environment_image,
            timeout=timeout,
        )

        raw_bindings = payload.get("object_bindings") or {}
        object_bindings: Dict[str, List[str]] = {}
        if isinstance(raw_bindings, dict):
            for key, value in raw_bindings.items():
                if isinstance(value, list):
                    object_bindings[str(key)] = [str(item) for item in value if item]
                elif value:
                    object_bindings[str(key)] = [str(value)]

        return GroundingSummary(
            object_bindings=object_bindings,
            grounded_goal_literals=self._as_string_list(payload.get("grounded_goal_literals")),
            grounded_predicates=self._as_string_list(payload.get("grounded_predicates")),
            available_skills=self._as_string_list(payload.get("available_skills")),
            missing_references=self._as_string_list(payload.get("missing_references")),
            observed_object_ids=self._extract_object_ids(observed_objects or []),
        )

    def analyze_task(
        self,
        task_description: str,
        observed_objects: Optional[List[Dict[str, Any]]] = None,
        observed_relationships: Optional[List[str]] = None,
        predicates: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 15.0,
    ) -> TaskAnalysis:
        """Run the full staged analysis pipeline."""

        observed_objects = observed_objects or []
        observed_relationships = observed_relationships or []
        predicates = predicates or []

        abstract_goal = self.analyze_goal(
            task_description=task_description,
            observed_objects=observed_objects,
            observed_relationships=observed_relationships,
            environment_image=environment_image,
            timeout=timeout,
        )
        predicate_inventory = self.analyze_predicates(
            task_description=task_description,
            abstract_goal=abstract_goal,
            observed_objects=observed_objects,
            observed_relationships=observed_relationships,
            environment_image=environment_image,
            timeout=timeout,
        )
        action_schemas = self.analyze_actions(
            task_description=task_description,
            abstract_goal=abstract_goal,
            predicate_inventory=predicate_inventory,
            observed_objects=observed_objects,
            observed_relationships=observed_relationships,
            environment_image=environment_image,
            timeout=timeout,
        )
        grounding_summary = self.analyze_grounding(
            task_description=task_description,
            abstract_goal=abstract_goal,
            predicate_inventory=predicate_inventory,
            action_schemas=action_schemas,
            observed_objects=observed_objects,
            observed_relationships=observed_relationships,
            predicates=predicates,
            environment_image=environment_image,
            timeout=timeout,
        )
        return TaskAnalysis(
            abstract_goal=abstract_goal,
            predicate_inventory=predicate_inventory,
            action_schemas=action_schemas,
            grounding_summary=grounding_summary,
            diagnostics={},
        )

    def clear_cache(self) -> None:
        """Compatibility no-op. The staged analyzer no longer caches responses."""

    def _generate_stage_json(
        self,
        template_key: str,
        replacements: Dict[str, str],
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]],
        timeout: float,
    ) -> Dict[str, Any]:
        prompt = render_prompt_template(self._get_prompt_template(template_key), replacements)
        content_parts: List[Any] = []
        pil_image = self._prepare_image(environment_image) if environment_image is not None else None
        if pil_image is not None:
            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format="PNG")
            content_parts.append(types.Part.from_bytes(data=image_bytes.getvalue(), mime_type="image/png"))
        content_parts.append(prompt)

        response_text = self._generate_content(content_parts=content_parts, timeout=timeout)
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON for {template_key}: {exc}\n{response_text}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object for {template_key}, got {type(payload).__name__}")
        return payload

    def _generate_content(self, content_parts: List[Any], timeout: float) -> str:
        if self.client is None:
            raise RuntimeError(
                "LLMTaskAnalyzer requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` for live analysis."
            )

        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=8192,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content_parts,
            config=config,
        )
        elapsed = time.time() - start
        self.call_count += 1
        self.total_elapsed_seconds += elapsed
        _ = timeout
        if response.text:
            return response.text
        raise RuntimeError("Model returned an empty response body.")

    def _get_prompt_template(self, template_key: str) -> str:
        template = self.prompt_templates.get(template_key)
        if template is None:
            raise KeyError(f"Missing prompt template: {template_key}")
        return template

    def _format_robot_description(self) -> str:
        return self.robot_description or "No robot description provided."

    def _format_objects_json(self, objects: List[Dict[str, Any]]) -> str:
        if not objects:
            return "[]"

        summary: List[Dict[str, Any]] = []
        for obj in objects[:30]:
            summary.append(
                {
                    "object_id": obj.get("object_id", "unknown"),
                    "object_type": obj.get("object_type", "unknown"),
                    "affordances": obj.get("affordances", []),
                    "attributes": obj.get("attributes", {}),
                }
            )
        return json.dumps(summary, indent=2)

    def _format_relationships_json(self, relationships: List[str]) -> str:
        if not relationships:
            return "[]"
        return json.dumps(relationships[:50], indent=2)

    def _prepare_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        if isinstance(image, Image.Image):
            return image
        raise ValueError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _goal_to_dict(abstract_goal: AbstractGoal) -> Dict[str, Any]:
        return {
            "summary": abstract_goal.summary,
            "goal_literals": abstract_goal.goal_literals,
            "goal_objects": abstract_goal.goal_objects,
            "success_checks": abstract_goal.success_checks,
        }

    @staticmethod
    def _extract_object_ids(objects: List[Dict[str, Any]]) -> List[str]:
        return [str(obj.get("object_id")) for obj in objects if obj.get("object_id")]

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        if not value:
            return []
        if not isinstance(value, list):
            return [str(value)]
        return [str(item).strip() for item in value if str(item).strip()]
