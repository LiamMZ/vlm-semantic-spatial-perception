"""
LLM-powered task analysis for dynamic PDDL generation.

This module uses an LLM to:
1. Parse natural language tasks in context of observed environment
2. Extract relevant predicates from scene
3. Generate goal conditions
4. Infer required actions
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image
import google.generativeai as genai


@dataclass
class TaskAnalysis:
    """Result of LLM task analysis."""

    # Task understanding
    action_sequence: List[str]  # High-level action steps
    goal_predicates: List[str]  # Goal state predicates
    preconditions: List[str]  # Required preconditions

    # Relevant objects
    goal_objects: List[str]  # Primary objects for task
    tool_objects: List[str]  # Tools/intermediate objects
    obstacle_objects: List[str]  # Objects to avoid

    # Scene-specific predicates
    initial_predicates: List[str]  # Current state predicates
    relevant_predicates: List[str]  # Task-relevant predicate types

    # Action definitions
    required_actions: List[Dict]  # PDDL action schemas

    # Metadata
    complexity: str  # 'simple', 'medium', 'complex'
    estimated_steps: int


class LLMTaskAnalyzer:
    """
    Uses LLM to analyze tasks in context of visual observations.

    This provides dynamic, scene-aware task understanding that adapts
    to the actual environment rather than relying on fixed patterns.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize LLM task analyzer.

        Args:
            api_key: Gemini API key (None to use environment variable)
            model_name: Model to use (flash for speed, pro for quality)
        """
        if api_key:
            genai.configure(api_key=api_key)

        # Use flash model for speed
        self.model = genai.GenerativeModel(model_name)

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
            # Build content (text + optional image)
            if pil_image:
                content = [pil_image, prompt]
            else:
                content = prompt

            response = self.model.generate_content(
                content,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low for consistency
                    top_p=0.9,
                    max_output_tokens=8192,  # Increased to avoid truncation
                    response_mime_type="application/json"
                )
            )
            elapsed = time.time() - start_time
            print(f"   → LLM analysis completed in {elapsed:.2f}s")

            # Check if response was truncated
            if response.candidates[0].finish_reason != 1:  # 1 = STOP (normal completion)
                print(f"   ⚠ Warning: Response truncated (finish_reason: {response.candidates[0].finish_reason})")
                # Try to salvage what we can by completing the JSON
                text = response.text
                # If it's incomplete JSON, we can't use it
                if not text.strip().endswith('}'):
                    raise ValueError(f"Incomplete response from LLM (finish_reason: {response.candidates[0].finish_reason})")

            # Parse response
            analysis = self._parse_response(response.text)

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
        return f"""Analyze this robotic task and predict required PDDL components.

TASK: {task}

Return JSON with:
{{
  "action_sequence": ["step1", "step2"],
  "goal_predicates": ["on(obj, location)"],
  "preconditions": ["graspable(obj)"],
  "goal_objects": ["obj_types_from_task"],
  "tool_objects": ["tools_needed"],
  "obstacle_objects": [],
  "initial_predicates": ["expected_initial_states"],
  "relevant_predicates": ["predicate_names"],
  "required_actions": [
    {{
      "name": "pick",
      "parameters": ["?obj - object"],
      "precondition": "(and (graspable ?obj) (empty-hand))",
      "effect": "(and (holding ?obj) (not (empty-hand)))"
    }}
  ],
  "complexity": "simple",
  "estimated_steps": 3
}}

Include 8-12 relevant_predicates (clean, dirty, on, holding, empty-hand, graspable, reachable, etc.) and 3-5 required_actions."""

    def _build_analysis_prompt(
        self,
        task: str,
        objects: List[Dict],
        relationships: List[str]
    ) -> str:
        """Build prompt for task analysis with observations."""

        # Format observed scene
        object_list = "\n".join([
            f"- {obj.get('object_type', 'unknown')} "
            f"(id: {obj.get('object_id', 'unknown')}, "
            f"affordances: {', '.join(obj.get('affordances', []))})"
            for obj in objects[:20]  # Limit for speed
        ])

        relationship_list = "\n".join([f"- {rel}" for rel in relationships[:30]])

        return f"""You are a robotic task planner. Analyze this task given the observed scene.

TASK: {task}

OBSERVED OBJECTS:
{object_list if object_list else "- No objects detected yet"}

OBSERVED RELATIONSHIPS:
{relationship_list if relationship_list else "- No relationships observed yet"}

Provide a JSON response with:
{{
  "action_sequence": ["action1", "action2", ...],
  "goal_predicates": ["predicate1(obj1, obj2)", ...],
  "preconditions": ["predicate(obj)", ...],
  "goal_objects": ["object_id1", ...],
  "tool_objects": ["tool_id", ...],
  "obstacle_objects": ["obstacle_id", ...],
  "initial_predicates": ["current_predicate(obj)", ...],
  "relevant_predicates": ["predicate_type1", "predicate_type2", ...],
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

Focus on:
1. Use observed objects and their actual IDs
2. Generate predicates matching observed affordances
3. Create action sequence using observed objects
4. Include all task-relevant predicates"""

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
                required_actions=data.get("required_actions", []),
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
