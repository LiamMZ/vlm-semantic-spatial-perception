"""
PDDL Domain Maintainer

Manages PDDL domain construction and updates based on tasks and environmental observations.
Maintains a consistent domain representation that evolves as more information is gathered.
"""

import asyncio
import re
from typing import Dict, List, Optional, Set, Union
from pathlib import Path

import numpy as np
from PIL import Image

from .llm_task_analyzer import LLMTaskAnalyzer, TaskAnalysis
from .pddl_representation import PDDLRepresentation


def sanitize_pddl_name(name: str, max_length: int = 50) -> str:
    """
    Sanitize a name for use in PDDL.

    PDDL identifiers must:
    - Start with a letter
    - Contain only letters, digits, underscores, and hyphens
    - Not contain spaces, parentheses, quotes, or other special characters
    - Be reasonably short for readability

    Args:
        name: Raw name (e.g., "Black electrical plug (unplugged)")
        max_length: Maximum length for sanitized name

    Returns:
        Valid PDDL identifier (e.g., "black_electrical_plug")

    Examples:
        >>> sanitize_pddl_name("Black electrical plug (unplugged)")
        'black_electrical_plug'
        >>> sanitize_pddl_name("Blue plastic water bottle with 'CHILL' label")
        'blue_plastic_water_bottle'
        >>> sanitize_pddl_name("Right black stove knob")
        'right_black_stove_knob'
    """
    # Convert to lowercase
    name = name.lower()

    # Remove content in parentheses (usually state information)
    name = re.sub(r'\([^)]*\)', '', name)

    # Remove quotes and their content
    name = re.sub(r"['\"].*?['\"]", '', name)

    # Remove common filler words that don't add semantic meaning
    filler_words = ['with', 'the', 'a', 'an', 'of']
    for word in filler_words:
        name = re.sub(rf'\b{word}\b', '', name)

    # Replace spaces and special chars with underscores
    name = re.sub(r'[^\w\s-]', '', name)  # Remove special chars
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
    name = name.strip('_')  # Remove leading/trailing underscores

    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = 'obj_' + name

    # Truncate if too long, but try to keep meaningful parts
    if len(name) > max_length:
        # Try to truncate at underscore boundary
        parts = name[:max_length].split('_')
        if len(parts) > 1:
            name = '_'.join(parts[:-1])  # Drop last partial word
        else:
            name = name[:max_length]

    # Fallback for empty names
    if not name:
        name = 'object'

    return name


def sanitize_pddl_formula(formula: str) -> str:
    """
    Sanitize a PDDL formula by removing quoted strings.

    LLMs sometimes generate invalid PDDL with quoted strings like:
      (is-empty ?machine "water_reservoir")

    This should be:
      (water-reservoir-empty ?machine)

    This function removes quoted strings as a safety measure,
    though the prompt should prevent them.

    Args:
        formula: PDDL formula (precondition or effect)

    Returns:
        Sanitized formula without quoted strings

    Examples:
        >>> sanitize_pddl_formula('(is-empty ?machine "water_reservoir")')
        '(is-empty ?machine)'
        >>> sanitize_pddl_formula('(and (graspable ?obj) (empty-hand))')
        '(and (graspable ?obj) (empty-hand))'
    """
    if not formula:
        return formula

    # Remove double-quoted strings
    formula = re.sub(r'"[^"]*"', '', formula)

    # Remove single-quoted strings
    formula = re.sub(r"'[^']*'", '', formula)

    # Clean up extra whitespace that may result from removal
    formula = re.sub(r'\s+', ' ', formula)

    # Clean up spaces before closing parens
    formula = re.sub(r'\s+\)', ')', formula)

    # Clean up spaces after opening parens
    formula = re.sub(r'\(\s+', '(', formula)

    return formula.strip()


class PDDLDomainMaintainer:
    """
    Maintains PDDL domain representation conditioned on tasks and observations.

    Responsibilities:
    - Generate initial domain from task analysis
    - Update domain as new observations arrive
    - Maintain consistency between task requirements and observed environment
    - Notify when domain changes significantly

    Example:
        >>> maintainer = PDDLDomainMaintainer(pddl_repr, api_key="...")
        >>>
        >>> # Initialize domain from task
        >>> await maintainer.initialize_from_task(
        ...     "Clean the mug and place it on the shelf",
        ...     environment_image=camera_frame
        ... )
        >>>
        >>> # Update as observations arrive
        >>> await maintainer.update_from_observations(detected_objects)
        >>>
        >>> # Check if domain is complete
        >>> is_complete = await maintainer.is_domain_complete()
    """

    def __init__(
        self,
        pddl_representation: PDDLRepresentation,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize domain maintainer.

        Args:
            pddl_representation: PDDL representation to maintain
            api_key: Gemini API key for LLM analysis
            model_name: LLM model to use
        """
        self.pddl = pddl_representation
        self.llm_analyzer = LLMTaskAnalyzer(api_key=api_key, model_name=model_name)

        # Current task context
        self.current_task: Optional[str] = None
        self.task_analysis: Optional[TaskAnalysis] = None

        # Tracking what we've observed
        self.observed_object_types: Set[str] = set()
        self.observed_predicates: Set[str] = set()
        self.goal_object_types: Set[str] = set()  # Objects mentioned in task

        # Domain evolution tracking
        self.domain_version: int = 0
        self.last_update_observations: int = 0

        # Object type mapping: raw_name -> sanitized_name
        self._type_name_mapping: Dict[str, str] = {}

    async def initialize_from_task(
        self,
        task_description: str,
        environment_image: Optional[Union[np.ndarray, Image.Image, str, Path]] = None
    ) -> TaskAnalysis:
        """
        Initialize PDDL domain from task description.

        Performs LLM-based analysis to predict required predicates, actions,
        and object types, then initializes the domain accordingly.

        Args:
            task_description: Natural language task
            environment_image: Optional environment image for context

        Returns:
            TaskAnalysis with predicted requirements
        """
        self.current_task = task_description

        # Analyze task with LLM (no observations yet)
        self.task_analysis = self.llm_analyzer.analyze_task(
            task_description=task_description,
            observed_objects=None,
            observed_relationships=None,
            environment_image=environment_image,
            timeout=15.0
        )

        # Check if analysis succeeded
        if self.task_analysis is None:
            raise RuntimeError(f"LLM task analysis failed for task: {task_description}")

        # Extract goal object types from task
        self.goal_object_types = set(self.task_analysis.goal_objects)

        # Initialize domain with predicted components
        await self._initialize_domain_from_analysis()

        self.domain_version += 1

        return self.task_analysis

    async def _initialize_domain_from_analysis(self) -> None:
        """Initialize PDDL domain based on task analysis."""
        if not self.task_analysis:
            return

        # Add predicted predicates
        for predicate_name in self.task_analysis.relevant_predicates:
            if predicate_name not in self.pddl.predicates:
                # Most predicates are unary
                await self.pddl.add_predicate_async(
                    predicate_name,
                    [("obj", "object")]
                )

        # Add LLM-generated actions
        for action_def in self.task_analysis.required_actions:
            try:
                # Parse parameters
                params = []
                if "parameters" in action_def:
                    param_list = action_def["parameters"]
                    if isinstance(param_list, list):
                        for p in param_list:
                            if isinstance(p, str) and " - " in p:
                                name, type_ = p.split(" - ")
                                params.append((name.strip("?"), type_.strip()))

                # Sanitize preconditions and effects to remove quoted strings
                # LLMs sometimes add invalid quoted strings like: (is-empty ?obj "reservoir")
                precondition = sanitize_pddl_formula(action_def.get("precondition", ""))
                effect = sanitize_pddl_formula(action_def.get("effect", ""))

                # Add action to domain
                # Note: Using sync method here as PDDLRepresentation doesn't have async action methods yet
                self.pddl.add_llm_generated_action(
                    name=action_def.get("name", "unknown"),
                    parameters=params,
                    precondition=precondition,
                    effect=effect,
                    description=action_def.get("description", "")
                )
            except Exception as e:
                print(f"⚠ Failed to add action {action_def.get('name')}: {e}")

        # Validate that all predicates used in actions are defined
        # Auto-add any missing predicates to ensure domain consistency
        validation_result = await self.validate_and_fix_action_predicates()
        if validation_result["missing_predicates"]:
            print(f"✓ Auto-added {len(validation_result['missing_predicates'])} missing predicates")
        if validation_result["invalid_actions"]:
            print(f"⚠ Found {len(validation_result['invalid_actions'])} actions with parsing issues")

    async def update_from_observations(
        self,
        detected_objects: List[Dict],
        detected_relationships: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Update domain based on new observations.

        Adds newly observed object types and updates the problem state.
        May trigger domain refinement if significant gaps are detected.

        Args:
            detected_objects: List of detected objects with types and properties
            detected_relationships: Optional spatial relationships

        Returns:
            Dict with update statistics
        """
        if detected_relationships is None:
            detected_relationships = []

        new_object_types = set()
        new_predicates = set()
        objects_added = 0

        # Process each detected object
        for obj in detected_objects:
            raw_obj_type = obj.get("object_type", "unknown")
            obj_id = obj.get("object_id", "unknown")

            # Sanitize object type name for PDDL
            # Map raw descriptive name to valid PDDL identifier
            if raw_obj_type not in self._type_name_mapping:
                sanitized_type = sanitize_pddl_name(raw_obj_type)
                self._type_name_mapping[raw_obj_type] = sanitized_type

            obj_type = self._type_name_mapping[raw_obj_type]

            # Track new object types
            if obj_type not in self.observed_object_types:
                new_object_types.add(obj_type)
                self.observed_object_types.add(obj_type)

                # Add object type to domain if not present
                if obj_type not in self.pddl.object_types:
                    await self.pddl.add_object_type_async(obj_type, parent="object")

            # Add object instance to problem
            if obj_id not in self.pddl.object_instances:
                await self.pddl.add_object_instance_async(obj_id, obj_type)
                objects_added += 1

            # Add predicates from PDDL state if available
            if "pddl_state" in obj and obj["pddl_state"]:
                for predicate, value in obj["pddl_state"].items():
                    if value:  # Only add true predicates (closed world assumption)
                        # Track observed predicates
                        if predicate not in self.observed_predicates:
                            new_predicates.add(predicate)
                            self.observed_predicates.add(predicate)

                        # Add to initial state
                        try:
                            await self.pddl.add_initial_literal_async(
                                predicate,
                                [obj_id],
                                negated=False
                            )
                        except ValueError:
                            # Predicate not in domain - this indicates domain incompleteness
                            pass

        self.last_update_observations += len(detected_objects)

        # Check if we observed any goal-relevant objects (with fuzzy matching)
        goal_objects_found = self._match_goal_objects(self.observed_object_types)
        goal_objects_missing = self.goal_object_types - goal_objects_found
        return {
            "objects_added": objects_added,
            "new_object_types": list(new_object_types),
            "new_predicates": list(new_predicates),
            "total_object_types": len(self.observed_object_types),
            "total_observations": self.last_update_observations,
            "goal_objects_found": list(goal_objects_found),
            "goal_objects_missing": list(goal_objects_missing)
        }

    async def refine_domain_from_observations(
        self,
        detected_objects: List[Dict],
        detected_relationships: List[str]
    ) -> None:
        """
        Refine domain based on accumulated observations.

        Re-analyzes the task in context of what we've actually observed,
        potentially adding new predicates or actions.

        Args:
            detected_objects: All currently detected objects
            detected_relationships: Current spatial relationships
        """
        if not self.current_task:
            return

        # Re-analyze task with observations
        refined_analysis = self.llm_analyzer.analyze_task(
            task_description=self.current_task,
            observed_objects=detected_objects,
            observed_relationships=detected_relationships,
            timeout=15.0
        )

        # Add any new predicates discovered
        for predicate_name in refined_analysis.relevant_predicates:
            if predicate_name not in self.pddl.predicates:
                await self.pddl.add_predicate_async(
                    predicate_name,
                    [("obj", "object")]
                )

        # Update task analysis
        self.task_analysis = refined_analysis
        self.domain_version += 1

    async def is_domain_complete(self) -> bool:
        """
        Check if domain is complete for the current task.

        Returns:
            True if all required predicates and actions are defined
        """
        if not self.task_analysis:
            return False

        # Check if all predicted predicates are in domain
        for predicate in self.task_analysis.relevant_predicates:
            if predicate not in self.pddl.predicates:
                return False

        # Check if we have any actions
        all_actions = await self.pddl.get_all_actions()
        if not all_actions:
            return False

        return True

    async def validate_and_fix_action_predicates(self) -> Dict[str, List[str]]:
        """
        Validate that all predicates used in actions are defined in the domain.

        Automatically adds missing predicates to the domain.

        Returns:
            Dict with 'missing_predicates' (list of predicates that were added)
                  and 'invalid_actions' (list of actions that couldn't be parsed)

        Example:
            >>> stats = await maintainer.validate_and_fix_action_predicates()
            >>> print(f"Added {len(stats['missing_predicates'])} missing predicates")
            >>> print(f"Predicates added: {stats['missing_predicates']}")
        """
        missing_predicates = []
        invalid_actions = []

        # Get all actions from the domain
        all_actions = await self.pddl.get_all_actions()

        for action_name, action in all_actions.items():
            try:
                # Extract predicates from preconditions
                precond_predicates = self._extract_predicates_from_formula(action.precondition)

                # Extract predicates from effects
                effect_predicates = self._extract_predicates_from_formula(action.effect)

                # Combine all predicates used in this action
                all_action_predicates = precond_predicates | effect_predicates

                # Check if each predicate is defined in the domain
                for pred_name in all_action_predicates:
                    if pred_name not in self.pddl.predicates:
                        if pred_name not in missing_predicates:
                            missing_predicates.append(pred_name)

                            # Add the missing predicate to the domain
                            # Use a generic signature (single object parameter)
                            await self.pddl.add_predicate_async(
                                pred_name,
                                [("obj", "object")]
                            )
                            print(f"  ℹ Added missing predicate: {pred_name}")

            except Exception as e:
                print(f"  ⚠ Failed to parse action '{action_name}': {e}")
                invalid_actions.append(action_name)

        if missing_predicates:
            print(f"✓ Validated and added {len(missing_predicates)} missing predicates")
        else:
            print("✓ All action predicates are properly defined")

        return {
            "missing_predicates": missing_predicates,
            "invalid_actions": invalid_actions
        }

    def _extract_predicates_from_formula(self, formula: str) -> Set[str]:
        """
        Extract predicate names from a PDDL formula.

        Parses formulas like:
          "(and (graspable ?obj) (empty-hand))"
        And extracts: {"graspable", "empty-hand"}

        Args:
            formula: PDDL formula string (precondition or effect)

        Returns:
            Set of predicate names used in the formula
        """
        if not formula:
            return set()

        predicates = set()

        # Pattern to match predicate names in PDDL formulas
        # Matches: (predicate-name ...) but excludes logical operators
        import re

        # Remove 'not' wrapper to get the actual predicate
        # e.g., (not (holding ?obj)) -> (holding ?obj)
        formula_clean = re.sub(r'\(not\s+', '(', formula)

        # Find all predicate-like patterns: (word ...)
        # Exclude logical operators: and, or, not
        pattern = r'\(([a-zA-Z][a-zA-Z0-9_-]*)\s'
        matches = re.findall(pattern, formula_clean)

        logical_operators = {'and', 'or', 'not', 'forall', 'exists', 'when'}

        for match in matches:
            if match not in logical_operators:
                predicates.add(match)

        # Also handle predicates without parameters like (empty-hand)
        pattern_no_params = r'\(([a-zA-Z][a-zA-Z0-9_-]*)\)'
        matches_no_params = re.findall(pattern_no_params, formula_clean)

        for match in matches_no_params:
            if match not in logical_operators:
                predicates.add(match)

        return predicates

    def _match_goal_objects(self, observed_types: Set[str]) -> Set[str]:
        """
        Match goal object types against observed types with fuzzy matching.

        Handles cases like "red_mug" (goal) matching "mug" (observed).
        Returns the set of goal object types that have been matched.

        Args:
            observed_types: Set of observed object types

        Returns:
            Set of goal object types that match observed types
        """
        matched = set()

        for goal_type in self.goal_object_types:
            # Exact match
            if goal_type in observed_types:
                matched.add(goal_type)
                continue

            # Fuzzy match: check if observed type is a substring of goal type
            # e.g., "mug" matches "red_mug"
            for obs_type in observed_types:
                if obs_type in goal_type or goal_type in obs_type:
                    matched.add(goal_type)
                    break

        return matched

    async def are_goal_objects_observed(self) -> bool:
        """
        Check if all goal-relevant objects have been observed.

        Returns:
            True if all goal objects have been seen
        """
        matched = self._match_goal_objects(self.observed_object_types)
        return len(matched) == len(self.goal_object_types)

    async def get_missing_goal_objects(self) -> List[str]:
        """
        Get list of goal objects that haven't been observed yet.

        Returns:
            List of object types mentioned in goal but not yet observed
        """
        matched = self._match_goal_objects(self.observed_object_types)
        return list(self.goal_object_types - matched)

    async def get_domain_statistics(self) -> Dict:
        """
        Get statistics about current domain state.

        Returns:
            Dict with domain completeness metrics
        """
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        return {
            "domain_version": self.domain_version,
            "task": self.current_task,
            "predicates_defined": len(domain_snapshot["predicates"]),
            "actions_defined": len(domain_snapshot["predefined_actions"]) + len(domain_snapshot["llm_generated_actions"]),
            "object_types_observed": len(self.observed_object_types),
            "object_instances": len(problem_snapshot["object_instances"]),
            "initial_literals": len(problem_snapshot["initial_literals"]),
            "goal_literals": len(problem_snapshot["goal_literals"]),
            "goal_objects_total": len(self.goal_object_types),
            "goal_objects_observed": len(self.goal_object_types.intersection(self.observed_object_types)),
            "domain_complete": await self.is_domain_complete(),
            "goals_observable": await self.are_goal_objects_observed()
        }

    async def set_goal_from_task_analysis(self) -> None:
        """
        Set goal state based on task analysis.

        Uses LLM-predicted goal predicates to populate PDDL goal state.
        """
        if not self.task_analysis:
            return

        # Clear existing goals
        await self.pddl.clear_goal_state_async()

        # Add goal predicates from analysis
        for goal_pred in self.task_analysis.goal_predicates:
            try:
                # Parse goal predicate format: "predicate(arg1, arg2)" or "not(predicate(arg))"
                negated = False
                pred_str = goal_pred.strip()

                if pred_str.startswith("not(") or pred_str.startswith("not "):
                    negated = True
                    # Extract inner predicate
                    if "(" in pred_str:
                        pred_str = pred_str[pred_str.index("(") + 1:]
                        if pred_str.endswith(")"):
                            pred_str = pred_str[:-1]

                # Parse "predicate(arg1, arg2)"
                if "(" in pred_str and ")" in pred_str:
                    pred_name = pred_str[:pred_str.index("(")]
                    args_str = pred_str[pred_str.index("(") + 1:pred_str.index(")")]
                    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

                    # Map generic object types to actual object instances
                    # e.g., "red_mug" -> "red_mug_1" if red_mug_1 exists
                    mapped_args = []
                    for arg in args:
                        # Check if this is an exact object instance
                        if arg in self.pddl.object_instances:
                            mapped_args.append(arg)
                        else:
                            # Try to find an instance with this type
                            found = False
                            for obj_name, obj in self.pddl.object_instances.items():
                                # Match if object type contains arg or vice versa
                                if arg in obj.object_type or obj.object_type in arg:
                                    mapped_args.append(obj_name)
                                    found = True
                                    break
                            if not found:
                                # Use original arg (will fail validation later)
                                mapped_args.append(arg)

                    # Only add if predicate is defined
                    if pred_name in self.pddl.predicates:
                        await self.pddl.add_goal_literal_async(
                            pred_name,
                            mapped_args,
                            negated=negated
                        )
            except Exception as e:
                print(f"⚠ Failed to parse goal predicate '{goal_pred}': {e}")
