"""
Dynamic PDDL generation from LLM task analysis and world state.

Generates PDDL domain and problem files on-the-fly based on:
1. LLM-analyzed task requirements
2. Observed scene state
3. Available object affordances
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from .llm_task_analyzer import LLMTaskAnalyzer, TaskAnalysis


class DynamicPDDLGenerator:
    """
    Generates PDDL files dynamically from task analysis and world state.

    Unlike traditional approaches with fixed domains, this creates
    task-specific PDDL that only includes relevant predicates and actions.
    """

    def __init__(self, llm_analyzer: Optional[LLMTaskAnalyzer] = None):
        """
        Initialize dynamic PDDL generator.

        Args:
            llm_analyzer: LLM task analyzer (creates one if not provided)
        """
        self.llm_analyzer = llm_analyzer or LLMTaskAnalyzer()

    def generate(
        self,
        task_description: str,
        world_state: Dict,
        output_dir: str = "outputs/pddl",
        domain_name: str = "manipulation"
    ) -> Dict[str, str]:
        """
        Generate PDDL domain and problem files.

        Args:
            task_description: Natural language task
            world_state: Current world state with objects and relationships
            output_dir: Directory for output files
            domain_name: Name for PDDL domain

        Returns:
            Dict with 'domain_path' and 'problem_path'
        """
        start_time = time.time()

        # Extract world state components
        objects = world_state.get("objects", [])
        relationships = world_state.get("relationships", [])

        print(f"   → Analyzing task with LLM...")

        # Analyze task with LLM
        analysis = self.llm_analyzer.analyze_task(
            task_description, objects, relationships
        )

        print(f"   → Generating PDDL domain...")

        # Generate domain
        domain_content = self._generate_domain(domain_name, analysis, objects)

        print(f"   → Generating PDDL problem...")

        # Generate problem
        problem_content = self._generate_problem(
            domain_name, task_description, analysis, world_state
        )

        # Write files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        domain_file = output_path / f"{domain_name}_domain.pddl"
        problem_file = output_path / f"{domain_name}_problem.pddl"

        domain_file.write_text(domain_content)
        problem_file.write_text(problem_content)

        elapsed = time.time() - start_time
        print(f"   ✓ PDDL generated in {elapsed:.2f}s")
        print(f"     Domain: {domain_file}")
        print(f"     Problem: {problem_file}")

        return {
            "domain_path": str(domain_file),
            "problem_path": str(problem_file),
            "analysis": analysis
        }

    def _generate_domain(
        self, domain_name: str, analysis: TaskAnalysis, objects: List[Dict]
    ) -> str:
        """Generate PDDL domain file."""

        # Extract object types from observed objects
        types = self._extract_types(objects)

        # Generate predicates from analysis and observed affordances
        predicates = self._generate_predicates(analysis, objects)

        # Generate actions from LLM analysis
        actions = self._generate_actions(analysis)

        domain = f""";; Auto-generated PDDL domain from task analysis
;; Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

(define (domain {domain_name})

  (:requirements :strips :typing)

  ;; Types extracted from observed objects
  (:types
{self._format_types(types)}
  )

  ;; Predicates relevant to task
  (:predicates
{self._format_predicates(predicates)}
  )

  ;; Actions inferred from task requirements
{self._format_actions(actions)}
)
"""
        return domain

    def _generate_problem(
        self,
        domain_name: str,
        task_desc: str,
        analysis: TaskAnalysis,
        world_state: Dict
    ) -> str:
        """Generate PDDL problem file."""

        problem_name = f"{domain_name}_task"

        # Format objects
        objects_str = self._format_problem_objects(world_state.get("objects", []))

        # Format initial state from current observations
        init_str = self._format_initial_state(
            analysis.initial_predicates,
            world_state.get("predicates", [])
        )

        # Format goal from analysis
        goal_str = self._format_goal(analysis.goal_predicates)

        problem = f""";; Auto-generated PDDL problem
;; Task: {task_desc}
;; Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

(define (problem {problem_name})
  (:domain {domain_name})

  ;; Objects in current scene
  (:objects
{objects_str}
  )

  ;; Current state
  (:init
{init_str}
  )

  ;; Goal state
  (:goal
{goal_str}
  )
)
"""
        return problem

    def _extract_types(self, objects: List[Dict]) -> Set[str]:
        """Extract unique object types."""
        types = {"object"}  # Base type

        for obj in objects:
            obj_type = obj.get("object_type", "object")
            if obj_type and obj_type != "unknown":
                types.add(obj_type)

        return types

    def _generate_predicates(
        self, analysis: TaskAnalysis, objects: List[Dict]
    ) -> List[str]:
        """Generate predicates from analysis and affordances."""
        predicates = set()

        # Add predicates from LLM analysis
        for pred_type in analysis.relevant_predicates:
            # Convert to PDDL format
            if pred_type in ["at", "on", "in", "near"]:
                predicates.add(f"({pred_type} ?obj1 - object ?obj2 - object)")
            elif pred_type in ["holding", "clear", "graspable"]:
                predicates.add(f"({pred_type} ?obj - object)")

        # Add affordance-based predicates from observed objects
        affordances = set()
        for obj in objects:
            affordances.update(obj.get("affordances", []))

        for aff in affordances:
            predicates.add(f"({aff} ?obj - object)")

        # Add basic robot predicates
        predicates.add("(robot-at ?loc - object)")
        predicates.add("(empty-hand)")

        return sorted(list(predicates))

    def _generate_actions(self, analysis: TaskAnalysis) -> List[Dict]:
        """Generate PDDL actions from LLM analysis."""
        actions = []

        # Use LLM-generated actions if available
        if analysis.required_actions:
            return analysis.required_actions

        # Otherwise, provide basic manipulation actions
        actions = [
            {
                "name": "pick",
                "parameters": "?obj - object",
                "precondition": "(and (graspable ?obj) (empty-hand))",
                "effect": "(and (holding ?obj) (not (empty-hand)))"
            },
            {
                "name": "place",
                "parameters": "?obj - object ?loc - object",
                "precondition": "(holding ?obj)",
                "effect": "(and (at ?obj ?loc) (empty-hand) (not (holding ?obj)))"
            },
            {
                "name": "move-to",
                "parameters": "?loc - object",
                "precondition": "(empty-hand)",
                "effect": "(robot-at ?loc)"
            }
        ]

        return actions

    def _format_types(self, types: Set[str]) -> str:
        """Format types for PDDL."""
        return "    " + " ".join([f"{t}" for t in sorted(types) if t != "object"])

    def _format_predicates(self, predicates: List[str]) -> str:
        """Format predicates for PDDL."""
        return "\n".join([f"    {pred}" for pred in predicates])

    def _format_actions(self, actions: List[Dict]) -> str:
        """Format actions for PDDL."""
        action_strs = []

        for action in actions:
            action_str = f"""  (:action {action['name']}
    :parameters ({action['parameters']})
    :precondition {action['precondition']}
    :effect {action['effect']}
  )
"""
            action_strs.append(action_str)

        return "\n".join(action_strs)

    def _format_problem_objects(self, objects: List[Dict]) -> str:
        """Format objects for problem file."""
        obj_strs = []

        for obj in objects:
            obj_id = obj.get("object_id", "unknown")
            obj_type = obj.get("object_type", "object")
            obj_strs.append(f"{obj_id} - {obj_type}")

        return "\n    ".join(obj_strs)

    def _format_initial_state(
        self, analysis_predicates: List[str], world_predicates: List[str]
    ) -> str:
        """Format initial state predicates."""
        all_predicates = set(analysis_predicates + world_predicates)

        # Filter and format
        formatted = []
        for pred in sorted(all_predicates):
            if pred and "(" in pred:
                formatted.append(f"    {pred}")

        # Add robot state
        formatted.append("    (empty-hand)")

        return "\n".join(formatted)

    def _format_goal(self, goal_predicates: List[str]) -> str:
        """Format goal condition."""
        if not goal_predicates:
            return "    (and)"

        if len(goal_predicates) == 1:
            return f"    {goal_predicates[0]}"

        formatted = ["    (and"]
        for pred in goal_predicates:
            formatted.append(f"      {pred}")
        formatted.append("    )")

        return "\n".join(formatted)
