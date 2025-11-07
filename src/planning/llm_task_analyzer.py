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
from typing import Dict, List, Optional

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

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
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
        observed_objects: List[Dict],
        observed_relationships: List[str],
        timeout: float = 5.0
    ) -> TaskAnalysis:
        """
        Analyze task in context of observed environment.

        Args:
            task_description: Natural language task
            observed_objects: List of detected objects with properties
            observed_relationships: Current spatial relationships
            timeout: Max time for LLM call (seconds)

        Returns:
            TaskAnalysis with scene-aware task understanding
        """
        # Check cache
        cache_key = self._make_cache_key(task_description, observed_objects)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build prompt
        prompt = self._build_analysis_prompt(
            task_description, observed_objects, observed_relationships
        )

        # Call LLM with timeout
        start_time = time.time()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low for consistency
                    top_p=0.9,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )

            elapsed = time.time() - start_time
            print(f"   → LLM analysis completed in {elapsed:.2f}s")

            # Parse response
            analysis = self._parse_response(response.text)

            # Cache result
            self._cache[cache_key] = analysis

            return analysis

        except Exception as e:
            print(f"   ⚠ LLM analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(task_description, observed_objects)

    def _build_analysis_prompt(
        self,
        task: str,
        objects: List[Dict],
        relationships: List[str]
    ) -> str:
        """Build optimized prompt for task analysis."""

        # Format observed scene
        object_list = "\n".join([
            f"- {obj.get('object_type', 'unknown')} "
            f"(id: {obj.get('object_id', 'unknown')}, "
            f"affordances: {', '.join(obj.get('affordances', []))})"
            for obj in objects[:20]  # Limit for speed
        ])

        relationship_list = "\n".join([f"- {rel}" for rel in relationships[:30]])

        return f"""You are a robotic task planner. Analyze this task in the context of the observed scene.

TASK: {task}

OBSERVED OBJECTS:
{object_list if object_list else "- No objects detected"}

OBSERVED RELATIONSHIPS:
{relationship_list if relationship_list else "- No relationships"}

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
      "precondition": "graspable(?obj) and clear(?obj)",
      "effect": "holding(?obj)"
    }}
  ],
  "complexity": "simple|medium|complex",
  "estimated_steps": 3
}}

Focus on:
1. Use ONLY objects that actually exist in the scene
2. Generate predicates that match observed affordances
3. Create minimal, efficient action sequence
4. Include only task-relevant predicates
5. Be concise for speed"""

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

    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
