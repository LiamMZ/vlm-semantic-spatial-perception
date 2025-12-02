"""
LLM-powered task analysis for dynamic PDDL generation.

This module uses an LLM to:
1. Parse natural language tasks in context of observed environment
2. Extract relevant predicates from scene
3. Generate goal conditions
4. Infer required actions
"""

import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from PIL import Image
from google import genai
from google.genai import types

from ..utils.prompt_utils import render_prompt_template
from .utils.task_types import TaskAnalysis


class LLMTaskAnalyzer:
    """
    Uses LLM to analyze tasks in context of visual observations.

    This provides dynamic, scene-aware task understanding that adapts
    to the actual environment rather than relying on fixed patterns.
    """

    DEFAULT_PROMPTS_CONFIG = str(Path(__file__).parent.parent.parent / "config" / "llm_task_analyzer_prompts.yaml")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        prompts_config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize LLM task analyzer.

        Args:
            api_key: Gemini API key (None to use environment variable)
            model_name: Model to use (flash for speed, pro for quality)
            prompts_config_path: Override path to prompts config YAML (defaults to config/llm_task_analyzer_prompts.yaml)
        """
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG

        self.api_key = api_key
        self.model_name = model_name
        self.prompts_config_path = Path(prompts_config_path)
        self.client = genai.Client(api_key=api_key)

        with open(self.prompts_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.prompt_templates = {
            key: value
            for key, value in config_data.items()
            if key.endswith("_prompt")
        }
        self.robot_description = config_data.get("robot_description")

        print(f"ℹ LLMTaskAnalyzer using GenAI SDK with model: {model_name}")
        if self.robot_description:
            print(f"  • Robot description configured ({len(self.robot_description)} chars)")

        # Response cache for identical queries
        self._cache: Dict[str, TaskAnalysis] = {}
        self._cache_ttl = 300  # 5 minutes

    def analyze_task(
        self,
        task_description: str,
        observed_objects: Optional[List[Dict]] = None,
        observed_relationships: Optional[List[str]] = None,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None,
        timeout: float = 10.0
    ) -> Optional[TaskAnalysis]:
        """
        Analyze task in context of observed environment.

        Args:
            task_description: Natural language task
            observed_objects: List of detected objects with properties (optional)
            observed_relationships: Current spatial relationships (optional)
            environment_image: Optional image of environment for visual context
            timeout: Max time for LLM call (seconds)

        Returns:
            TaskAnalysis with scene-aware task understanding
        """
        # Handle None defaults
        if observed_objects is None:
            observed_objects = []
        if observed_relationships is None:
            observed_relationships = []

        # Check cache
        cache_key = self._make_cache_key(task_description, observed_objects)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Prepare image if provided
        pil_image = None
        if environment_image is not None:
            pil_image = self._prepare_image(environment_image)

        # Build prompt (different for initial vs observed analysis)
        has_observations = len(observed_objects) > 0 or pil_image is not None
        if has_observations:
            prompt = self._build_analysis_prompt(
                task_description, observed_objects, observed_relationships
            )
        else:
            # Initial analysis without observations - predict requirements
            prompt = self._build_initial_analysis_prompt(task_description)

        # Call LLM with timeout
        start_time = time.time()
        try:
            # Build content parts
            content_parts = []
            if pil_image:
                # Encode image
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                content_parts.append(types.Part.from_bytes(data=img_bytes, mime_type='image/png'))
            content_parts.append(prompt)

            # Generate content
            config = types.GenerateContentConfig(
                temperature=0.1,  # Low for consistency
                top_p=0.9,
                max_output_tokens=8192,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=config
            )
            response_text = response.text

            elapsed = time.time() - start_time
            print(f"   → LLM analysis completed in {elapsed:.2f}s")

            # Parse response
            analysis = self._parse_response(response_text)

            # Cache result
            self._cache[cache_key] = analysis

            return analysis

        except Exception as e:
            print(f"   ⚠ LLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_initial_analysis_prompt(self, task: str) -> str:
        """
        Build prompt for INITIAL task analysis (before any observations).

        This prompt asks the LLM to predict what predicates, actions, and objects
        will likely be needed for the task, even without seeing the environment yet.
        """
        template = self._get_prompt_template("initial_prompt")
        return render_prompt_template(
            template,
            {
                "TASK": task,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
            },
        )

    def _build_analysis_prompt(
        self,
        task: str,
        objects: List[Dict],
        relationships: List[str]
    ) -> str:
        """Build prompt for task analysis with observations."""

        objects_json = self._format_objects_json(objects)
        relationships_json = self._format_relationships_json(relationships)

        template = self._get_prompt_template("analysis_prompt")
        return render_prompt_template(
            template,
            {
                "TASK": task,
                "ROBOT_DESCRIPTION": self._format_robot_description(),
                "OBJECTS_JSON": objects_json,
                "RELATIONSHIPS_JSON": relationships_json,
            },
        )

    def _parse_response(self, response_text: str) -> TaskAnalysis:
        """Parse LLM JSON response into TaskAnalysis."""
        try:
            data = json.loads(response_text)

            return TaskAnalysis(
                action_sequence=data.get("action_sequence", []),
                goal_predicates=data.get("goal_predicates", []),
                preconditions=data.get("preconditions", []),
                goal_objects=data.get("goal_objects", []),
                tool_objects=data.get("tool_objects", []),
                obstacle_objects=data.get("obstacle_objects", []),
                initial_predicates=data.get("initial_predicates", []),
                relevant_predicates=data.get("relevant_predicates", []),
                relevant_types=data.get("relevant_types", []),
                required_actions=data.get("required_actions", []),
                complexity=data.get("complexity", "medium"),
                estimated_steps=data.get("estimated_steps", 1)
            )
        except Exception as e:
            print(f"   ⚠ Failed to parse LLM response: {e}")
            raise

    def _make_cache_key(self, task: str, objects: List[Dict]) -> str:
        """Create cache key from task and objects."""
        obj_ids = sorted([obj.get("object_id", "") for obj in objects])
        return f"{task}_{','.join(obj_ids)}"

    def _prepare_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Convert image to PIL Image format."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()

    def _get_prompt_template(self, template_key: str) -> str:
        """Return prompt template by key."""
        template = self.prompt_templates.get(template_key)
        if template is None:
            raise KeyError(f"Missing prompt template: {template_key}")
        return template

    def _format_robot_description(self) -> str:
        """Return robot description text or empty string if not configured."""
        return (self.robot_description or "").strip()

    def _format_objects_json(self, objects: List[Dict]) -> str:
        """Serialize observed objects as compact JSON summary."""
        summary: List[Dict[str, Union[str, List[str]]]] = []
        for obj in objects[:20]:
            summary.append(
                {
                    "object_id": obj.get("object_id", "unknown"),
                    "object_type": obj.get("object_type", "unknown"),
                    "affordances": obj.get("affordances", []),
                }
            )
        return json.dumps(summary, indent=2)

    def _format_relationships_json(self, relationships: List[str]) -> str:
        """Serialize observed relationships as JSON."""
        rels = relationships[:30] if relationships else []
        return json.dumps(rels, indent=2)
