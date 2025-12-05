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
        model_name: str = "gemini-robotics-er-1.5-preview",
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

        # Build prompt (analysis template handles empty observations)
        prompt = self._build_analysis_prompt(
            task_description, observed_objects, observed_relationships
        )

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
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=config
            )
            response_text = response.text
            print(f"************************************ {response_text}")
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
        robot_context = ""
        if self.robot_description:
            robot_context = f"""
ROBOT CAPABILITIES:
{self.robot_description}

Consider the robot's capabilities when determining feasible actions and predicates.
"""

        return f"""Analyze this robotic task and predict required PDDL components.

TASK: {task}
{robot_context}

Return JSON with:
{{
  "action_sequence": ["action1", "action2", ...],
  "goal_predicates": ["predicate1 obj1 obj2", ...],
  "preconditions": ["predicate(obj)", ...],
  "initial_predicates": ["expected_initial_states"],
  "relevant_predicates": ["predicate_names"],
  "goal_objects": ["object_id1", ...],
  "global_predicates": ["global_predicate1", ...],
  "tool_objects": ["tool_id", ...],
  "obstacle_objects": ["obstacle_id", ...],
  "initial_predicates": ["current_predicate(obj)", ...],
  "relevant_predicates": ["predicate_type1", "predicate_type2", ...],
  "relevant_types": ["type1", "type2", ...],
  "required_actions": [
    {{
      "name": "pick",
      "parameters": ["?obj - object"],
      "precondition": "(and (graspable ?obj) (clear ?obj))",
      "effect": "(and (holding ?obj) (not (empty-hand)))"
    }}
  ],
}}

IMPORTANT: Global Predicates
- "global_predicates" should list predicates that represent robot/environment state, NOT object-specific predicates
- These are predicates that should be TRUE INITIALLY before task execution
- Examples:
  - hand_is_empty (or empty-hand): Robot gripper has no object
  - arm_at_home: Robot arm is at home position
  - gripper_open: Gripper is in open state
  - robot_ready: Robot system is initialized
- Do NOT include object-related predicates here (those go in initial_predicates)
- Global predicates typically have NO parameters or take robot as parameter

IMPORTANT PDDL RULES:
1. Parameters MUST use variables starting with ? (e.g., ?obj, ?location, ?container)
2. Preconditions and effects use ONLY variables - NO quoted strings or constants
3. If you need to reference a specific part (like "water_reservoir"), create a separate predicate:
   - WRONG: (is-empty ?machine "water_reservoir")
   - RIGHT: (water-reservoir-empty ?machine)
4. All predicates should be predicates applied to variables, not string constants
5. GLOBAL Predicates should not be returned with parenthases

Include 8-12 relevant_predicates (clean, dirty, on, holding, empty-hand, graspable, reachable, etc.) and 3-5 required_actions."""

    def _build_analysis_prompt(
        self,
        task: str,
        objects: List[Dict],
        relationships: List[str]
    ) -> str:
        """Build prompt for task analysis with observations."""

        object_list = [obj["object_id"] for obj in objects]
        objects_json = self._format_objects_json(objects)
        relationships_json = self._format_relationships_json(relationships)

        relationship_list = "\n".join([f"- {rel}" for rel in relationships[:30]])

        robot_context = ""
        if self.robot_description:
            robot_context = f"""
ROBOT CAPABILITIES:
{self.robot_description}

Consider the robot's capabilities when determining feasible actions.
"""

        return f"""You are a robotic task planner. Analyze this task given the observed scene.

TASK: {task}
{robot_context}
OBSERVED OBJECTS:
{object_list if object_list else "- No objects detected yet"}

OBSERVED RELATIONSHIPS:
{relationship_list if relationship_list else "- No relationships observed yet"}

Provide a JSON response with:
{{
  "action_sequence": ["action1", "action2", ...],
  "goal_predicates": ["predicate1 obj1 obj2", ...],
  "preconditions": ["predicate(obj)", ...],
  "initial_predicates": ["expected_initial_states"],
  "relevant_predicates": ["predicate_names"],
  "goal_objects": ["object_id1", ...],
  "global_predicates": ["global_predicate1", ...],
  "tool_objects": ["tool_id", ...],
  "obstacle_objects": ["obstacle_id", ...],
  "initial_predicates": ["current_predicate(obj)", ...],
  "relevant_predicates": ["predicate_type1", "predicate_type2", ...],
  "relevant_types": ["type1", "type2", ...],
  "required_actions": [
    {{
      "name": "pick",
      "parameters": ["?obj - object"],
      "precondition": "(and (graspable ?obj) (clear ?obj))",
      "effect": "(and (holding ?obj) (not (empty-hand)))"
    }}
  ],
  "complexity": "simple|medium|complex",
  "estimated_steps": 3
}}

CRITICAL PDDL FORMATTING RULES:
1. Parameters MUST use variables with ? prefix (e.g., ?obj, ?location, ?machine)
2. Preconditions and effects can ONLY use:
   - Variables (e.g., ?obj, ?container)
   - Predicate names (e.g., graspable, empty-hand, has-water)
3. NEVER use quoted strings or constants in preconditions/effects
4. If referencing a component, make it a predicate:
   - WRONG: (is-empty ?machine "water_reservoir")
   - RIGHT: (water-reservoir-empty ?machine) or (reservoir-has-water ?machine)
5. Multi-word predicates use hyphens: has-water, is-empty, water-reservoir-empty
6. GLOBAL Predicates should not be returned with parenthases


IMPORTANT: Global Predicates
    - "global_predicates" should list predicates that represent robot/environment state, NOT object-specific predicates
    - These are predicates that should be TRUE INITIALLY before task execution
    - Examples:
    - hand_is_empty (or empty-hand): Robot gripper has no object
    - arm_at_home: Robot arm is at home position
    - gripper_open: Gripper is in open state
    - robot_ready: Robot system is initialized
    - Do NOT include object-related predicates here (those go in initial_predicates)
    - Global predicates typically have NO parameters or take robot as parameter

Focus on:
1. Use observed objects and their actual IDs
2. Generate predicates matching observed affordances
3. Create action sequence using observed objects
4. Include all task-relevant predicates
5. Ensure all PDDL actions follow proper syntax (no quoted strings!)"""

    def _parse_response(self, response_text: str) -> TaskAnalysis:
        """Parse LLM JSON response into TaskAnalysis."""
        try:
            data = json.loads(response_text)

            # Accept both legacy and current action keys to stay compatible with prompt schema
            actions = (
                data.get("required_actions")
                or data.get("relevant_actions")
                or data.get("actions")
                or []
            )

            return TaskAnalysis(
                action_sequence=data.get("action_sequence", []),
                goal_predicates=data.get("goal_predicates", []),
                preconditions=data.get("preconditions", []),
                goal_objects=data.get("goal_objects", []),
                tool_objects=data.get("tool_objects", []),
                obstacle_objects=data.get("obstacle_objects", []),
                initial_predicates=data.get("initial_predicates", []),
                global_predicates=data.get("global_predicates", []),
                relevant_predicates=data.get("relevant_predicates", []),
                relevant_types=data.get("relevant_types", []),
                required_actions=actions,
                complexity=data.get("complexity", "medium"),
                estimated_steps=data.get("estimated_steps", 1)
            )
        except Exception as e:
            print(f"   ⚠ Failed to parse LLM response: {e}")
            raise

    def _create_fallback_analysis(
        self, task: str, objects: List[Dict]
    ) -> TaskAnalysis:
        """Create basic fallback analysis if LLM fails."""
        return TaskAnalysis(
            action_sequence=["navigate", "manipulate"],
            goal_predicates=["completed(task)"],
            preconditions=["ready(robot)"],
            goal_objects=[obj.get("object_id", "") for obj in objects[:3]],
            tool_objects=[],
            obstacle_objects=[],
            initial_predicates=[],
            global_predicates=["hand_is_empty"],
            relevant_predicates=["at", "holding", "clear"],
            required_actions=[],
            complexity="medium",
            estimated_steps=2
        )

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
        if not objects:
            return (
                "Initial analysis run; perception has not produced any objects yet. "
            )

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
        if not relationships:
            return (
                "Initial analysis run; perception has not produced any relationships yet."
            )

        rels = relationships[:30]
        return json.dumps(rels, indent=2)
