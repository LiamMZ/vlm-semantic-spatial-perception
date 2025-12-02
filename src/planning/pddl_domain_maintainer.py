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
import yaml

from ..utils.prompt_utils import render_prompt_template
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

    DEFAULT_PROMPTS_CONFIG = str(
        Path(__file__).parent.parent.parent / "config" / "pddl_domain_maintainer_prompts.yaml"
    )

    def __init__(
        self,
        pddl_representation: PDDLRepresentation,
        api_key: Optional[str] = None,
        model_name: str = "gemini-robotics-er-1.5-preview",
        prompts_config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize domain maintainer.

        Args:
            pddl_representation: PDDL representation to maintain
            api_key: Gemini API key for LLM analysis
            model_name: LLM model to use
            prompts_config_path: Override path to prompts config YAML (defaults to config/pddl_domain_maintainer_prompts.yaml)
        """
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG

        self.pddl = pddl_representation
        self.llm_analyzer = LLMTaskAnalyzer(
            api_key=api_key,
            model_name=model_name,
        )
        self.robot_description = self.llm_analyzer.robot_description
        self.prompts_config_path = Path(prompts_config_path)

        with open(self.prompts_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.prompt_templates = {
            key: value
            for key, value in config_data.items()
            if key.endswith("_prompt") and isinstance(value, str)
        }

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

            # Use base 'object' type for STRIPS (no typing)
            obj_type = "object"

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

        # Note: With STRIPS (no typing), we don't track object types
        return {
            "objects_added": objects_added,
            "new_predicates": list(new_predicates),
            "total_observations": self.last_update_observations
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
            print("⚠ No task analysis available - cannot set goals")
            return

        print(f"  • Setting goals from task analysis...")
        print(f"    Task: {self.current_task}")
        print(f"    Goal predicates: {self.task_analysis.goal_predicates}")

        # Clear existing goals
        await self.pddl.clear_goal_state_async()

        if not self.task_analysis.goal_predicates:
            print("⚠ No goal predicates in task analysis - problem will have empty goals!")
            print("   This usually means the LLM failed to extract goals from the task description")
            return

        # Add goal predicates from analysis
        goals_added = 0
        for goal_pred in self.task_analysis.goal_predicates:
            try:
                # Parse goal predicate format
                # Supports:
                #   - PDDL S-expression: "(is-open bottle_1)" or "(not (is-open bottle_1))"
                #   - Function call format: "is-open(bottle_1, arg2)"
                negated = False
                pred_str = goal_pred.strip()

                # Handle negation for both formats
                if pred_str.startswith("(not "):
                    negated = True
                    # Extract inner predicate: "(not (pred args))" -> "(pred args)"
                    pred_str = pred_str[5:].strip()  # Remove "(not "
                    if pred_str.endswith(")"):
                        pred_str = pred_str[:-1]  # Remove trailing )
                elif pred_str.startswith("not(") or pred_str.startswith("not "):
                    negated = True
                    if "(" in pred_str:
                        pred_str = pred_str[pred_str.index("(") + 1:]
                        if pred_str.endswith(")"):
                            pred_str = pred_str[:-1]

                # Detect format: PDDL S-expression starts with "(", function call doesn't
                if pred_str.startswith("("):
                    # PDDL S-expression format: "(predicate arg1 arg2)"
                    pred_str = pred_str[1:]  # Remove leading (
                    if pred_str.endswith(")"):
                        pred_str = pred_str[:-1]  # Remove trailing )

                    # Split by whitespace
                    parts = pred_str.split()
                    if not parts:
                        continue

                    pred_name = parts[0]
                    args = parts[1:] if len(parts) > 1 else []

                elif "(" in pred_str and ")" in pred_str:
                    # Function call format: "predicate(arg1, arg2)"
                    pred_name = pred_str[:pred_str.index("(")]
                    args_str = pred_str[pred_str.index("(") + 1:pred_str.index(")")]
                    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

                else:
                    # Simple predicate with no args: "predicate"
                    pred_name = pred_str
                    args = []

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
                    goals_added += 1
                    print(f"    ✓ Added goal: {pred_name}({', '.join(mapped_args)}) {'(negated)' if negated else ''}")
                else:
                    print(f"    ⚠ Skipped goal '{goal_pred}': predicate '{pred_name}' not defined")
            except Exception as e:
                print(f"    ⚠ Failed to parse goal predicate '{goal_pred}': {e}")

        print(f"  ✓ Added {goals_added} goal literals")

    async def refine_domain_from_error(
        self,
        error_message: str,
        current_domain_pddl: Optional[str] = None,
        current_problem_pddl: Optional[str] = None
    ) -> None:
        """
        Refine the PDDL domain based on a planning error.

        Uses the LLM to analyze the error and fix domain issues like:
        - Predicate arity mismatches
        - Missing predicates
        - Action definition errors
        - Type mismatches
        - Goal object name mismatches with detected objects

        Args:
            error_message: The error message from the planner
            current_domain_pddl: Optional current domain PDDL for context
            current_problem_pddl: Optional current problem PDDL for context (includes detected objects)
        """
        if not self.llm_analyzer:
            print("⚠ No LLM analyzer available for refinement")
            return

        print("  • Analyzing planning error...")

        # Create refinement prompt
        task_desc = self.current_task if self.current_task else 'Unknown'

        # Extract goal objects from task analysis if available
        goal_objects_str = ""
        if self.task_analysis and self.task_analysis.goal_objects:
            # Filter out None values that might have been incorrectly added
            valid_goal_objects = [obj for obj in self.task_analysis.goal_objects if obj and obj != "None"]
            if valid_goal_objects:
                goal_objects_str = f"\nExpected Goal Objects (from task): {', '.join(valid_goal_objects)}"

        # Add problem file context and check if objects section is empty
        problem_context = ""
        objects_section_empty = False
        detected_objects_list = []

        if current_problem_pddl:
            # Check if :objects section is empty or missing
            if ":objects" not in current_problem_pddl:
                objects_section_empty = True
                print("  ⚠ DEBUG: Problem file has NO :objects section")
            else:
                # Extract content between (:objects and next section or (:init
                import re
                objects_match = re.search(r'\(:objects\s*(.*?)\s*\)', current_problem_pddl, re.DOTALL)
                if objects_match:
                    objects_content = objects_match.group(1).strip()
                    objects_section_empty = len(objects_content) == 0 or objects_content == ""
                    if not objects_section_empty:
                        # Extract object names for debugging
                        detected_objects_list = re.findall(r'(\w+)\s*', objects_content)
                        print(f"  ℹ DEBUG: Detected objects in problem file: {detected_objects_list}")
                    else:
                        print("  ⚠ DEBUG: :objects section exists but is EMPTY")
                else:
                    print("  ⚠ DEBUG: Could not parse :objects section")

            problem_context = f"""

Current PDDL Problem File:
{current_problem_pddl}

Note: The problem file shows the detected objects (in :objects section) and the goal state (in :goal section).
If the goal references objects that don't exist in :objects, this is the root cause of the failure.
"""
            if objects_section_empty:
                problem_context += """
⚠ WARNING: The :objects section appears to be EMPTY. This means NO objects were detected by the perception system.
This is a PERCEPTION PROBLEM, not a domain problem. The object tracker needs to detect objects before planning can work.
"""
            elif detected_objects_list:
                problem_context += f"""
ℹ INFO: Detected {len(detected_objects_list)} object(s) in problem file: {', '.join(detected_objects_list)}
"""

        # Add robot capabilities context
        robot_context = ""
        if self.robot_description:
            robot_context = f"""

ROBOT CAPABILITIES:
{self.robot_description}

Note: Actions and predicates should be consistent with the robot's capabilities.
"""

        template = self._get_prompt_template("refinement_prompt")
        refinement_prompt = render_prompt_template(
            template,
            {
                "ERROR_MESSAGE": error_message,
                "TASK_DESCRIPTION": task_desc,
                "GOAL_OBJECTS_SECTION": goal_objects_str,
                "ROBOT_CONTEXT_SECTION": robot_context,
                "CURRENT_DOMAIN_PDDL": current_domain_pddl if current_domain_pddl else "Not available",
                "PROBLEM_CONTEXT_SECTION": problem_context,
            },
        )


        try:
            # Get LLM's analysis and fix
            response = await asyncio.to_thread(
                self.llm_analyzer.client.models.generate_content,
                model=self.llm_analyzer.model_name,
                contents=refinement_prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            )

            fix_json = response.text.strip()
            print(f"\n  LLM Response:\n{fix_json[:500]}...\n")

            # Parse the JSON response
            import json
            fix_data = json.loads(fix_json)

            analysis = fix_data.get("analysis", "")
            fixes = fix_data.get("fixes", [])
            explanation = fix_data.get("explanation", "")
            goal_object_recommendations = fix_data.get("goal_object_recommendations", [])

            print(f"  Analysis: {analysis}")
            print(f"  Explanation: {explanation}")

            # Display goal object recommendations and update task analysis if any
            if goal_object_recommendations:
                print(f"\n  ⚠ Goal Object Name Mismatches Detected:")
                updates_applied = 0
                for rec in goal_object_recommendations:
                    expected = rec.get("goal_object_expected", rec.get("current_name", "?"))
                    actual = rec.get("detected_object_actual", rec.get("detected_as", "?"))
                    action = rec.get("action", "UPDATE_GOAL")

                    # Skip if no actual object was detected (None, empty, or "None" string)
                    if not actual or actual == "?" or actual == "None" or actual.lower() == "none":
                        print(f"    • Goal expects: '{expected}' but NO matching object was detected")
                        print(f"      ⚠ Cannot update goal - no objects detected. Check if perception is running.")
                        continue

                    print(f"    • Goal expects: '{expected}' but detected object is: '{actual}'")

                    # Update goal_objects AND goal_predicates in task analysis
                    if self.task_analysis and expected in self.task_analysis.goal_objects:
                        # Update goal_objects list
                        idx = self.task_analysis.goal_objects.index(expected)
                        self.task_analysis.goal_objects[idx] = actual
                        print(f"      ✓ Updated task_analysis.goal_objects: '{expected}' → '{actual}'")

                        # Update goal_predicates to use new object name
                        updated_predicates = []
                        for pred in self.task_analysis.goal_predicates:
                            # Replace old object name with new one in predicate strings
                            # Handle various formats: "(is-open water_bottle_1)" or "is-open(water_bottle_1)"
                            updated_pred = pred.replace(expected, actual)
                            updated_predicates.append(updated_pred)
                            if updated_pred != pred:
                                print(f"      ✓ Updated goal predicate: '{pred}' → '{updated_pred}'")

                        self.task_analysis.goal_predicates = updated_predicates
                        updates_applied += 1
                    else:
                        print(f"      → Will update goal when domain is regenerated")

                if updates_applied > 0:
                    print(f"\n  ✓ Updated {updates_applied} goal object name(s) in task analysis")
                    print(f"  ℹ Goal predicates will use new names on next problem generation\n")
                else:
                    print(f"\n  ⚠ No objects detected in perception - cannot update goal object names")
                    print(f"     Please ensure the object tracker has detected objects before planning\n")

            print(f"  Found {len(fixes)} fix(es) to apply\n")

            if not fixes and not goal_object_recommendations:
                print("  ⚠ LLM did not provide any fixes or recommendations")
                return

            # Get current domain text for domain-level fixes
            current_domain_str = self.pddl.get_domain_text()

            # Apply all fixes sequentially (domain-level fixes only)
            fixes_applied = 0
            fixes_failed = 0

            for i, fix in enumerate(fixes, 1):
                old_text = fix.get("old_text", "")
                new_text = fix.get("new_text", "")
                location = fix.get("location", "unknown location")

                if not old_text or not new_text:
                    print(f"  ⚠ Fix {i}: Invalid fix (missing old_text or new_text)")
                    fixes_failed += 1
                    continue

                # Skip problem file fixes since those are regenerated automatically
                if "problem" in location.lower() or "goal" in location.lower() or ":objects" in location.lower() or ":init" in location.lower():
                    print(f"\n  • Skipping fix {i}/{len(fixes)}: {location}")
                    print(f"    (Problem file is regenerated automatically; use goal_object_recommendations instead)")
                    continue

                print(f"\n  • Applying fix {i}/{len(fixes)}: {location}")
                print(f"    Replacing:\n{old_text[:150]}...")
                print(f"    With:\n{new_text[:150]}...")

                if old_text in current_domain_str:
                    # Replace text (only first occurrence for safety)
                    current_domain_str = current_domain_str.replace(old_text, new_text, 1)
                    print(f"    ✓ Applied successfully")
                    fixes_applied += 1
                else:
                    print(f"    ⚠ Could not find exact match in domain")
                    fixes_failed += 1

            # After all fixes, update the domain if any were applied
            if fixes_applied > 0:
                # Update text cache only (don't parse back to structured)
                self.pddl.set_domain_text(current_domain_str, update_structured=False)
                print(f"\n  ✓ Applied {fixes_applied} domain fix(es) successfully")
                if fixes_failed > 0:
                    print(f"  ⚠ Failed to apply {fixes_failed} fix(es)")
            elif fixes_failed > 0:
                print(f"\n  ✗ All {fixes_failed} fix(es) failed to match domain text")
            elif len(fixes) == 0 and goal_object_recommendations:
                print(f"  ℹ No domain fixes needed (only goal object updates)")
            else:
                print(f"\n  ✗ No fixes could be applied")

        except json.JSONDecodeError as e:
            print(f"⚠ Error parsing LLM JSON response: {e}")
        except Exception as e:
            print(f"⚠ Error during refinement: {e}")
            import traceback
            traceback.print_exc()

    async def _fix_predicate_arity_from_error(
        self,
        error_message: str,
        llm_analysis: str
    ) -> None:
        """Fix predicate arity issues based on error message."""
        # Extract predicate name from error if possible
        # Example error: "wrong number of arguments for predicate empty-hand..."

        # Run action validation which will fix arity issues
        fixes = await self.validate_and_fix_action_predicates()

        if fixes:
            print(f"  ✓ Applied {sum(len(v) for v in fixes.values())} predicate fixes")

    async def _add_missing_predicate_from_error(
        self,
        error_message: str,
        llm_analysis: str
    ) -> None:
        """Add missing predicates based on error message."""
        # Extract predicate name from error message
        # This is a placeholder - in production you'd parse the error more carefully

        # For now, trigger a re-validation
        await self.validate_and_fix_action_predicates()

        print("  ✓ Re-validated predicates")

    def _get_prompt_template(self, template_key: str) -> str:
        """Fetch prompt text for the given key."""
        template = self.prompt_templates.get(template_key)
        if template is None:
            raise KeyError(f"Missing prompt template: {template_key}")
        return template
