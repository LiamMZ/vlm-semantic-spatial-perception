"""
PDDL Representation Manager

Maintains PDDL domain and problem components as modular sets/lists with methods
for incremental updates and PDDL file generation.

Architecture:
1. Domain Components (static/semi-static):
   - Object types hierarchy
   - Predicate definitions
   - Action schemas (predefined + LLM-generated)

2. Problem Components (dynamic):
   - Object instances
   - Initial state literals
   - Goal literals

3. Feedback & Refinement:
   - Methods for updating goals, predicates, actions
   - Validation and insufficiency detection
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import field
from contextlib import asynccontextmanager
import re

from .utils.pddl_types import (
    PDDLRequirements,
    ObjectType,
    Predicate,
    Action,
    ObjectInstance,
    Literal
)


class PDDLRepresentation:
    """
    Manages PDDL domain and problem representation with incremental updates.

    Example:
        >>> pddl = PDDLRepresentation(domain_name="kitchen_manipulation")
        >>>
        >>> # Add types
        >>> pddl.add_object_type("container", parent="object")
        >>> pddl.add_object_type("cup", parent="container")
        >>>
        >>> # Add predicates
        >>> pddl.add_predicate("on", [("obj", "object"), ("surface", "object")])
        >>> pddl.add_predicate("holding", [("obj", "object")])
        >>>
        >>> # Add actions
        >>> pddl.add_predefined_action("pick", ...)
        >>>
        >>> # Add problem instances
        >>> pddl.add_object_instance("red_cup", "cup")
        >>> pddl.add_initial_literal("on", ["red_cup", "table"])
        >>> pddl.add_goal_literal("holding", ["red_cup"])
        >>>
        >>> # Generate files
        >>> paths = pddl.generate_files("outputs/pddl")
    """

    def __init__(
        self,
        domain_name: str = "manipulation",
        problem_name: Optional[str] = None,
        requirements: Optional[List[PDDLRequirements]] = None
    ):
        """
        Initialize thread-safe PDDL representation with async support.

        Args:
            domain_name: Name of the PDDL domain
            problem_name: Name of the problem (defaults to domain_name + "_problem")
            requirements: PDDL requirements (defaults to :strips :typing)
        """
        self.domain_name = domain_name
        self.problem_name = problem_name or f"{domain_name}_problem"

        # Thread safety with asyncio Lock for true parallel access
        self._lock = asyncio.Lock()

        self.requirements: Set[PDDLRequirements] = set(requirements or [
            PDDLRequirements.STRIPS,
            PDDLRequirements.TYPING,
            PDDLRequirements.CONDITIONAL_EFFECTS,
        ])

        # Domain components
        self.object_types: Dict[str, ObjectType] = {}
        self.predicates: Dict[str, Predicate] = {}
        self.predefined_actions: Dict[str, Action] = {}
        self.llm_generated_actions: Dict[str, Action] = {}

        # Problem components
        self.object_instances: Dict[str, ObjectInstance] = {}
        self.initial_literals: Set[Literal] = set()
        self.goal_literals: Set[Literal] = set()
        self.goal_formulas: List[str] = []

        # Metadata
        self.task_description: Optional[str] = None
        self.last_modified: float = time.time()

        # Cached text representations (generated on demand)
        self._domain_text_cache: Optional[str] = None
        self._problem_text_cache: Optional[str] = None
        self._cache_dirty: bool = True

        # Initialize with base type
        self.add_object_type("object")

    @asynccontextmanager
    async def _acquire_lock(self):
        """Context manager for acquiring async lock."""
        async with self._lock:
            yield

    # =====================================================================
    # Thread-Safe Accessors (for external planners)
    # =====================================================================

    async def get_domain_snapshot(self) -> Dict:
        """
        Get a complete snapshot of domain components (thread-safe).

        Returns:
            Dict with object_types, predicates, predefined_actions, llm_generated_actions
        """
        async with self._acquire_lock():
            return {
                "object_types": dict(self.object_types),
                "predicates": dict(self.predicates),
                "predefined_actions": dict(self.predefined_actions),
                "llm_generated_actions": dict(self.llm_generated_actions),
                "requirements": set(self.requirements)
            }

    async def get_problem_snapshot(self) -> Dict:
        """
        Get a complete snapshot of problem components (thread-safe).

        Returns:
            Dict with object_instances, initial_literals, goal_literals
        """
        async with self._acquire_lock():
            return {
                "object_instances": dict(self.object_instances),
                "initial_literals": set(self.initial_literals),
                "goal_literals": set(self.goal_literals)
            }

    async def get_all_object_types(self) -> Dict[str, ObjectType]:
        """Get all object types (thread-safe copy)."""
        async with self._acquire_lock():
            return dict(self.object_types)

    async def get_all_predicates(self) -> Dict[str, Predicate]:
        """Get all predicates (thread-safe copy)."""
        async with self._acquire_lock():
            return dict(self.predicates)

    async def get_all_actions(self) -> Dict[str, Action]:
        """Get all actions (thread-safe copy)."""
        async with self._acquire_lock():
            actions = dict(self.predefined_actions)
            actions.update(self.llm_generated_actions)
            return actions

    async def get_all_object_instances(self) -> Dict[str, ObjectInstance]:
        """Get all object instances (thread-safe copy)."""
        async with self._acquire_lock():
            return dict(self.object_instances)

    async def get_initial_state(self) -> Set[Literal]:
        """Get initial state literals (thread-safe copy)."""
        async with self._acquire_lock():
            return set(self.initial_literals)

    async def get_goal_state(self) -> Set[Literal]:
        """Get goal state literals (thread-safe copy)."""
        async with self._acquire_lock():
            return set(self.goal_literals)

    # =====================================================================
    # Thread-Safe Modification Methods
    # =====================================================================

    async def add_object_type_async(self, name: str, parent: Optional[str] = None) -> None:
        """Add an object type (async, thread-safe)."""
        async with self._acquire_lock():
            if parent and parent not in self.object_types:
                raise ValueError(f"Parent type '{parent}' does not exist")
            self.object_types[name] = ObjectType(name=name, parent=parent)
            self.last_modified = time.time()

    async def add_predicate_async(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        description: Optional[str] = None
    ) -> None:
        """Add a predicate (async, thread-safe)."""
        async with self._acquire_lock():
            predicate = Predicate(name=name, parameters=parameters, description=description)
            self.predicates[name] = predicate
            self.last_modified = time.time()

    async def add_object_instance_async(self, name: str, object_type: str) -> None:
        """Add an object instance (async, thread-safe)."""
        async with self._acquire_lock():
            if object_type not in self.object_types:
                raise ValueError(f"Object type '{object_type}' does not exist in domain")
            self.object_instances[name] = ObjectInstance(name=name, object_type=object_type)
            self.last_modified = time.time()

    async def add_initial_literal_async(
        self,
        predicate: str,
        arguments: List[str],
        negated: bool = False
    ) -> None:
        """Add an initial state literal (async, thread-safe)."""
        async with self._acquire_lock():
            if predicate not in self.predicates:
                raise ValueError(f"Predicate '{predicate}' not defined in domain")
            literal = Literal(predicate=predicate, arguments=arguments, negated=negated)
            self.initial_literals.add(literal)
            self.last_modified = time.time()

    async def add_goal_literal_async(
        self,
        predicate: str,
        arguments: List[str],
        negated: bool = False
    ) -> None:
        """Add a goal state literal (async, thread-safe)."""
        async with self._acquire_lock():
            if predicate not in self.predicates:
                raise ValueError(f"Predicate '{predicate}' not defined in domain")
            literal = Literal(predicate=predicate, arguments=arguments, negated=negated)
            self.goal_literals.add(literal)
            self.last_modified = time.time()

    async def clear_initial_state_async(self) -> None:
        """Clear initial state (async, thread-safe)."""
        async with self._acquire_lock():
            self.initial_literals.clear()
            self.last_modified = time.time()

    async def clear_goal_state_async(self) -> None:
        """Clear goal state (async, thread-safe)."""
        async with self._acquire_lock():
            self.goal_literals.clear()
            self.goal_formulas.clear()
            self.last_modified = time.time()

    async def add_goal_formula_async(self, formula: str) -> None:
        """
        Add a raw goal formula (async, thread-safe).

        This preserves quantified/boolean goal expressions as authored by the LLM.
        """
        async with self._acquire_lock():
            if formula and formula not in self.goal_formulas:
                self.goal_formulas.append(formula)
                self.last_modified = time.time()

    async def generate_domain_pddl_async(self) -> str:
        """Generate PDDL domain file content (async, thread-safe)."""
        async with self._acquire_lock():
            return self.generate_domain_pddl()

    async def generate_problem_pddl_async(self) -> str:
        """Generate PDDL problem file content (async, thread-safe)."""
        async with self._acquire_lock():
            return self.generate_problem_pddl()

    async def generate_files_async(
        self,
        output_dir: str = "outputs/pddl",
        domain_filename: Optional[str] = None,
        problem_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate and write PDDL domain and problem files (async, thread-safe).

        Args:
            output_dir: Output directory path
            domain_filename: Custom domain filename (default: {domain_name}_domain.pddl)
            problem_filename: Custom problem filename (default: {domain_name}_problem.pddl)

        Returns:
            Dictionary with 'domain_path' and 'problem_path' keys
        """
        async with self._acquire_lock():
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filenames
            domain_file = output_path / (domain_filename or f"{self.domain_name}_domain.pddl")
            problem_file = output_path / (problem_filename or f"{self.domain_name}_problem.pddl")

            # Generate content (within lock)
            domain_content = self.generate_domain_pddl()
            problem_content = self.generate_problem_pddl()

            # Write files
            domain_file.write_text(domain_content)
            problem_file.write_text(problem_content)

            return {
                "domain_path": str(domain_file),
                "problem_path": str(problem_file)
            }

    # =====================================================================
    # Domain Component Management (Legacy sync methods)
    # =====================================================================

    def add_object_type(self, name: str, parent: Optional[str] = None) -> None:
        """
        Add an object type to the domain.

        Args:
            name: Type name
            parent: Parent type (None for root types)
        """
        if parent and parent not in self.object_types:
            raise ValueError(f"Parent type '{parent}' does not exist")

        self.object_types[name] = ObjectType(name=name, parent=parent)
        self.last_modified = time.time()

    def add_predicate(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        description: Optional[str] = None
    ) -> None:
        """
        Add a predicate to the domain.

        Args:
            name: Predicate name
            parameters: List of (parameter_name, type) tuples
            description: Optional description
        """
        predicate = Predicate(name=name, parameters=parameters, description=description)
        self.predicates[name] = predicate
        self.last_modified = time.time()

    def add_predefined_action(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        precondition: str,
        effect: str,
        description: Optional[str] = None
    ) -> None:
        """
        Add a predefined action to the domain.

        Args:
            name: Action name
            parameters: List of (parameter_name, type) tuples
            precondition: Precondition as PDDL string
            effect: Effect as PDDL string
            description: Optional description
        """
        action = Action(
            name=name,
            parameters=parameters,
            precondition=precondition,
            effect=effect,
            description=description,
            is_llm_generated=False
        )
        self.predefined_actions[name] = action
        self.last_modified = time.time()

    def add_llm_generated_action(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        precondition: str,
        effect: str,
        description: Optional[str] = None
    ) -> None:
        """
        Add an LLM-generated action to the domain.

        Args:
            name: Action name
            parameters: List of (parameter_name, type) tuples
            precondition: Precondition as PDDL string
            effect: Effect as PDDL string
            description: Optional description
        """
        action = Action(
            name=name,
            parameters=parameters,
            precondition=precondition,
            effect=effect,
            description=description,
            is_llm_generated=True
        )
        self.llm_generated_actions[name] = action
        self.last_modified = time.time()

    def remove_predicate(self, name: str) -> None:
        """Remove a predicate from the domain."""
        if name in self.predicates:
            del self.predicates[name]
            self.last_modified = time.time()

    def remove_action(self, name: str) -> None:
        """Remove an action from the domain (checks both predefined and LLM-generated)."""
        removed = False
        if name in self.predefined_actions:
            del self.predefined_actions[name]
            removed = True
        if name in self.llm_generated_actions:
            del self.llm_generated_actions[name]
            removed = True
        if removed:
            self.last_modified = time.time()

    # =====================================================================
    # Problem Component Management
    # =====================================================================

    def add_object_instance(self, name: str, object_type: str) -> None:
        """
        Add an object instance to the problem.

        Args:
            name: Object instance name
            object_type: Object type (must exist in domain)
        """
        if object_type not in self.object_types:
            raise ValueError(f"Object type '{object_type}' not defined in domain")

        self.object_instances[name] = ObjectInstance(name=name, object_type=object_type)
        self.last_modified = time.time()

    def add_initial_literal(
        self,
        predicate: str,
        arguments: List[str],
        negated: bool = False
    ) -> None:
        """
        Add a literal to the initial state.

        Args:
            predicate: Predicate name
            arguments: List of object instance names
            negated: Whether this is a negated literal
        """
        literal = Literal(predicate=predicate, arguments=arguments, negated=negated)
        self.initial_literals.add(literal)
        self.last_modified = time.time()

    def add_goal_literal(
        self,
        predicate: str,
        arguments: List[str],
        negated: bool = False
    ) -> None:
        """
        Add a literal to the goal state.

        Args:
            predicate: Predicate name
            arguments: List of object instance names
            negated: Whether this is a negated literal
        """
        literal = Literal(predicate=predicate, arguments=arguments, negated=negated)
        self.goal_literals.add(literal)
        self.last_modified = time.time()

    def remove_object_instance(self, name: str) -> None:
        """Remove an object instance from the problem."""
        if name in self.object_instances:
            del self.object_instances[name]
            # Remove literals referencing this object
            self.initial_literals = {
                lit for lit in self.initial_literals if name not in lit.arguments
            }
            self.goal_literals = {
                lit for lit in self.goal_literals if name not in lit.arguments
            }
            self.last_modified = time.time()

    def clear_initial_state(self) -> None:
        """Clear all initial state literals."""
        self.initial_literals.clear()
        self.last_modified = time.time()

    def clear_goal_state(self) -> None:
        """Clear all goal literals."""
        self.goal_literals.clear()
        self.last_modified = time.time()

    def update_initial_state(self, literals: List[Tuple[str, List[str], bool]]) -> None:
        """
        Replace initial state with new literals.

        Args:
            literals: List of (predicate, arguments, negated) tuples
        """
        self.clear_initial_state()
        for predicate, arguments, negated in literals:
            self.add_initial_literal(predicate, arguments, negated)

    def update_goal_state(self, literals: List[Tuple[str, List[str], bool]]) -> None:
        """
        Replace goal state with new literals.

        Args:
            literals: List of (predicate, arguments, negated) tuples
        """
        self.clear_goal_state()
        for predicate, arguments, negated in literals:
            self.add_goal_literal(predicate, arguments, negated)

    # =====================================================================
    # Bulk Updates from World State
    # =====================================================================

    def update_from_world_state(self, world_state: Dict) -> None:
        """
        Update problem components from world state.

        Args:
            world_state: Dictionary with 'objects' and 'predicates' keys
        """
        # Update object instances
        for obj in world_state.get("objects", []):
            obj_id = obj.get("object_id")
            obj_type = obj.get("object_type", "object")

            if obj_id:
                # Ensure type exists
                if obj_type not in self.object_types:
                    self.add_object_type(obj_type, parent="object")

                # Add instance
                self.add_object_instance(obj_id, obj_type)

        # Update initial literals from predicates
        for pred_str in world_state.get("predicates", []):
            # Parse predicate string: "predicate(arg1, arg2)"
            parsed = self._parse_predicate_string(pred_str)
            if parsed:
                predicate, arguments, negated = parsed
                self.add_initial_literal(predicate, arguments, negated)

    # =====================================================================
    # PDDL File Generation
    # =====================================================================

    @staticmethod
    def _strip_forall_when_from_domain_text(domain_text: str) -> str:
        """
        Strip (forall ...) and (when ...) from every :effect block in a raw PDDL
        domain string.  Used when the domain comes from the LLM text cache.
        """
        # Operate character by character to handle balanced parens correctly.
        result = []
        text = domain_text
        i = 0
        n = len(text)
        while i < n:
            # Look for ':effect'
            if text[i:i+7].lower() == ':effect':
                result.append(text[i:i+7])
                i += 7
                # Skip whitespace
                while i < n and text[i] in ' \t\n\r':
                    result.append(text[i])
                    i += 1
                # Now collect the effect value (may start with '(' or be a simple atom)
                if i < n and text[i] == '(':
                    depth = 0
                    start = i
                    while i < n:
                        if text[i] == '(':
                            depth += 1
                        elif text[i] == ')':
                            depth -= 1
                            if depth == 0:
                                i += 1
                                break
                        i += 1
                    effect_expr = text[start:i]
                    result.append(PDDLRepresentation._strip_forall_when(effect_expr))
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    @staticmethod
    def _strip_forall_when(effect: str) -> str:
        """
        Remove (forall ...) and (when ...) sub-expressions from an effect string.

        These require :conditional-effects which pyperplan (STRIPS) doesn't support.
        The PostDisplacementHook handles these semantics at runtime, so the planner
        only needs the simple non-conditional part of each effect.
        """
        result = []
        i = 0
        n = len(effect)
        while i < n:
            # Look for (forall or (when at the current position
            if effect[i] == '(':
                # Peek at the keyword after the open paren
                j = i + 1
                while j < n and effect[j] == ' ':
                    j += 1
                word_end = j
                while word_end < n and effect[word_end] not in (' ', ')'):
                    word_end += 1
                keyword = effect[j:word_end].lower()
                if keyword in ('forall', 'when'):
                    # Skip the entire balanced paren expression
                    depth = 1
                    k = i + 1
                    while k < n and depth > 0:
                        if effect[k] == '(':
                            depth += 1
                        elif effect[k] == ')':
                            depth -= 1
                        k += 1
                    i = k
                    continue
            result.append(effect[i])
            i += 1
        # Collapse extra whitespace left by removed expressions
        cleaned = re.sub(r'\s{2,}', ' ', ''.join(result)).strip()
        # Remove empty (and) that may remain after stripping all sub-expressions
        cleaned = re.sub(r'\(\s*and\s*\)', '(and)', cleaned)
        return cleaned

    def generate_domain_pddl(self) -> str:
        """
        Generate PDDL domain file content.

        If a text cache exists (e.g., from LLM-based refinement), use that.
        Otherwise, generate from structured representation.
        """
        # If we have a cached text version (e.g., from refinement), use it.
        # Still strip forall/when if requirements are STRIPS-only.
        if self._domain_text_cache is not None and not self._cache_dirty:
            cached = self._domain_text_cache
            req_values = {r.value for r in self.requirements}
            if ":conditional-effects" not in req_values and (
                'forall' in cached or 'when' in cached
            ):
                cached = self._strip_forall_when_from_domain_text(cached)
            return cached

        # Otherwise generate from structured representation
        lines = [
            f";; Auto-generated PDDL domain: {self.domain_name}",
            f";; Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"(define (domain {self.domain_name})",
            ""
        ]

        # Requirements
        req_str = " ".join([req.value for req in sorted(self.requirements, key=lambda x: x.value)])
        lines.append(f"  (:requirements {req_str})")
        lines.append("")

        # Types block — emit when typing is required and non-root types exist
        if PDDLRequirements.TYPING in self.requirements:
            non_root = {
                name: ot for name, ot in self.object_types.items()
                if name != "object"
            }
            if non_root:
                lines.append("  (:types")
                for type_name, ot in sorted(non_root.items()):
                    parent = ot.parent or "object"
                    lines.append(f"    {type_name} - {parent}")
                lines.append("  )")
                lines.append("")

        # Predicates
        if self.predicates:
            lines.append("  (:predicates")
            for pred_name in sorted(self.predicates.keys()):
                predicate = self.predicates[pred_name]
                comment = f"  ; {predicate.description}" if predicate.description else ""
                lines.append(f"    {predicate.to_pddl()}{comment}")
            lines.append("  )")
            lines.append("")

        # Actions (predefined first, then LLM-generated)
        all_actions = []
        for action_name in sorted(self.predefined_actions.keys()):
            action = self.predefined_actions[action_name]
            comment = f"  ; Predefined: {action.description}" if action.description else "  ; Predefined action"
            all_actions.append((action, comment))

        for action_name in sorted(self.llm_generated_actions.keys()):
            action = self.llm_generated_actions[action_name]
            comment = f"  ; LLM-generated: {action.description}" if action.description else "  ; LLM-generated action"
            all_actions.append((action, comment))

        # Strip forall/when conditional effects when requirements are STRIPS-only.
        # These are handled at runtime by PostDisplacementHook; the planner must
        # not see them or it will reject the domain.
        req_values = {r.value for r in self.requirements}
        needs_strip = ":conditional-effects" not in req_values

        for action, comment in all_actions:
            if needs_strip and ('forall' in action.effect or 'when' in action.effect):
                stripped_effect = self._strip_forall_when(action.effect)
                from dataclasses import replace as _dc_replace
                action = _dc_replace(action, effect=stripped_effect)
            lines.append(comment)
            lines.append(action.to_pddl())
            lines.append("")

        lines.append(")")

        return "\n".join(lines)

    def generate_problem_pddl(self) -> str:
        """Generate PDDL problem file content."""
        lines = [
            f";; Auto-generated PDDL problem: {self.problem_name}",
        ]

        if self.task_description:
            lines.append(f";; Task: {self.task_description}")

        lines.extend([
            f";; Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"(define (problem {self.problem_name})",
            f"  (:domain {self.domain_name})",
            ""
        ])

        # Objects — emit with type annotations when :typing is active
        if self.object_instances:
            lines.append("  (:objects")
            if PDDLRequirements.TYPING in self.requirements:
                # Group by type: emit "obj1 obj2 - type" lines
                from collections import defaultdict as _dd
                by_type: dict = _dd(list)
                for obj_name in sorted(self.object_instances.keys()):
                    obj = self.object_instances[obj_name]
                    by_type[obj.object_type].append(obj_name)
                for type_name in sorted(by_type.keys()):
                    names = " ".join(by_type[type_name])
                    lines.append(f"    {names} - {type_name}")
            else:
                for obj_name in sorted(self.object_instances.keys()):
                    obj = self.object_instances[obj_name]
                    lines.append(f"    {obj.to_pddl()}")
            lines.append("  )")
            lines.append("")

        # Initial state
        lines.append("  (:init")
        if self.initial_literals:
            for literal in sorted(self.initial_literals, key=lambda x: (x.predicate, tuple(x.arguments))):
                lines.append(f"    {literal.to_pddl()}")
        else:
            lines.append("    ; Empty initial state")
        lines.append("  )")
        lines.append("")

        # Goal
        lines.append("  (:goal")
        goal_entries: List[str] = []

        goal_entries.extend(self.goal_formulas)

        for literal in sorted(self.goal_literals, key=lambda x: (x.predicate, tuple(x.arguments))):
            goal_entries.append(literal.to_pddl())

        if not goal_entries:
            lines.append("    (and)")
        elif len(goal_entries) == 1:
            lines.append(f"    {goal_entries[0]}")
        else:
            lines.append("    (and")
            for entry in goal_entries:
                lines.append(f"      {entry}")
            lines.append("    )")
        lines.append("  )")

        lines.append(")")

        return "\n".join(lines)

    def generate_files(
        self,
        output_dir: str = "outputs/pddl",
        domain_filename: Optional[str] = None,
        problem_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate and write PDDL domain and problem files.

        Args:
            output_dir: Output directory path
            domain_filename: Custom domain filename (default: {domain_name}_domain.pddl)
            problem_filename: Custom problem filename (default: {domain_name}_problem.pddl)

        Returns:
            Dictionary with 'domain_path' and 'problem_path' keys
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filenames
        domain_file = output_path / (domain_filename or f"{self.domain_name}_domain.pddl")
        problem_file = output_path / (problem_filename or f"{self.domain_name}_problem.pddl")

        # Generate content
        domain_content = self.generate_domain_pddl()
        problem_content = self.generate_problem_pddl()

        # Write files
        domain_file.write_text(domain_content)
        problem_file.write_text(problem_content)

        return {
            "domain_path": str(domain_file),
            "problem_path": str(problem_file)
        }

    # =====================================================================
    # Validation & Feedback
    # =====================================================================

    def validate_goal_completeness(self) -> Tuple[bool, List[str]]:
        """
        Check if goal literals are complete and achievable.

        Returns:
            Tuple of (is_complete, issues)
        """
        issues = []

        if not self.goal_literals and not self.goal_formulas:
            issues.append("No goal literals defined")

        # Check if goal predicates exist in domain
        for literal in self.goal_literals:
            if literal.predicate not in self.predicates:
                issues.append(f"Goal predicate '{literal.predicate}' not defined in domain")

        # Check if goal objects exist
        for literal in self.goal_literals:
            for arg in literal.arguments:
                if arg not in self.object_instances:
                    issues.append(f"Goal references undefined object '{arg}'")

        return (len(issues) == 0, issues)

    def validate_action_completeness(self) -> Tuple[bool, List[str]]:
        """
        Check if sufficient actions exist to achieve goals.

        Returns:
            Tuple of (is_complete, issues)
        """
        issues = []

        total_actions = len(self.predefined_actions) + len(self.llm_generated_actions)
        if total_actions == 0:
            issues.append("No actions defined in domain")

        # Check if goal predicates can be produced by actions
        goal_predicates = {lit.predicate for lit in self.goal_literals}
        for formula in self.goal_formulas:
            for match in re.finditer(r"\(([^()]+)\)", formula):
                tokens = match.group(1).strip().split()
                if not tokens or tokens[0] in {"and", "or", "not", "forall", "exists", "when"}:
                    continue
                goal_predicates.add(tokens[0])
        action_effects = set()

        for action in list(self.predefined_actions.values()) + list(self.llm_generated_actions.values()):
            # Simple heuristic: check if predicate name appears in effect
            for pred in goal_predicates:
                if pred in action.effect:
                    action_effects.add(pred)

        missing_effects = goal_predicates - action_effects
        if missing_effects:
            issues.append(f"No actions produce goal predicates: {missing_effects}")

        return (len(issues) == 0, issues)

    def get_statistics(self) -> Dict:
        """Get statistics about the current representation."""
        return {
            "domain": {
                "name": self.domain_name,
                "types": len(self.object_types),
                "predicates": len(self.predicates),
                "predefined_actions": len(self.predefined_actions),
                "llm_generated_actions": len(self.llm_generated_actions),
                "total_actions": len(self.predefined_actions) + len(self.llm_generated_actions)
            },
            "problem": {
                "name": self.problem_name,
                "object_instances": len(self.object_instances),
                "initial_literals": len(self.initial_literals),
                "goal_literals": len(self.goal_literals)
            },
            "last_modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_modified))
        }

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _parse_predicate_string(self, pred_str: str) -> Optional[Tuple[str, List[str], bool]]:
        """
        Parse predicate string into components.

        Args:
            pred_str: String like "on(cup, table)" or "not holding(cup)"

        Returns:
            Tuple of (predicate_name, arguments, negated) or None if invalid
        """
        import re

        # Check for negation
        negated = False
        if pred_str.strip().startswith("not "):
            negated = True
            pred_str = pred_str.strip()[4:]

        # Match pattern: predicate(arg1, arg2, ...)
        match = re.match(r'(\w+)\((.*?)\)', pred_str.strip())
        if not match:
            return None

        predicate = match.group(1)
        args_str = match.group(2)
        arguments = [arg.strip() for arg in args_str.split(",")] if args_str else []

        return (predicate, arguments, negated)

    def _initialize_predefined_actions(self) -> None:
        """Initialize library of predefined manipulation actions."""

        # Basic manipulation actions
        predefined = [
            {
                "name": "pick",
                "parameters": [("obj", "object")],
                "precondition": "(and (graspable ?obj) (empty-hand))",
                "effect": "(and (holding ?obj) (not (empty-hand)))",
                "description": (
                    "Grasp ?obj and lift it into a held state. "
                    "Primitive sequence: move_gripper_to_pose to the grasp contact point on ?obj "
                    "→ close_gripper → retract_gripper to lift clear. "
                    "Use pointing_guidance to describe the exact contact region "
                    "(e.g. 'the body of the cup just below the rim where fingers can close around it'). "
                    "Choose approach_direction based on object geometry and overhead clearance."
                ),
            },
            {
                "name": "place",
                "parameters": [("obj", "object"), ("loc", "object")],
                "precondition": "(and (holding ?obj) (supportable ?loc))",
                "effect": "(and (on ?obj ?loc) (empty-hand) (not (holding ?obj)))",
                "description": (
                    "Set down the currently held ?obj onto surface ?loc. "
                    "Primitive sequence: move_gripper_to_pose with is_place=true to a stable "
                    "position above ?loc → open_gripper → retract_gripper. "
                    "Use pointing_guidance to describe the target landing spot on ?loc "
                    "(e.g. 'the flat open area on the table 10 cm to the left of the blue cup'). "
                    "approach_direction should typically be top_down for placing."
                ),
            },
            {
                "name": "open",
                "parameters": [("obj", "object")],
                "precondition": "(and (openable ?obj) (not (opened ?obj)))",
                "effect": "(opened ?obj)",
                "description": (
                    "Operate the opening mechanism of ?obj (door, drawer, lid, valve). "
                    "Primitive sequence: move_gripper_to_pose to the handle/mechanism → "
                    "close_gripper → push_pull with has_pivot=true or force_direction as appropriate. "
                    "Use pointing_guidance to describe the handle or grip point "
                    "(e.g. 'the drawer handle grip bar at its midpoint'). "
                    "approach_direction is usually side for handles."
                ),
            },
            {
                "name": "close",
                "parameters": [("obj", "object")],
                "precondition": "(and (openable ?obj) (opened ?obj))",
                "effect": "(not (opened ?obj))",
                "description": (
                    "Close an already-open ?obj (door, drawer, lid). "
                    "Primitive sequence: move_gripper_to_pose to the closing mechanism → "
                    "push_pull in the closing direction. "
                    "Use pointing_guidance to describe the push/pull contact point. "
                    "approach_direction is usually side for handles."
                ),
            },
            {
                "name": "pour",
                "parameters": [("from_obj", "object"), ("to_obj", "object")],
                "precondition": "(and (holding ?from_obj) (pourable ?from_obj) (containable ?to_obj))",
                "effect": "(and (empty ?from_obj) (filled ?to_obj))",
                "description": "Pour contents from one object to another"
            },
            {
                "name": "push",
                "parameters": [("obj", "object"), ("direction", "object")],
                "precondition": "(pushable ?obj)",
                "effect": "(at ?obj ?direction)",
                "description": "Push an object in a direction"
            },
            {
                "name": "pull",
                "parameters": [("obj", "object")],
                "precondition": "(pullable ?obj)",
                "effect": "(near ?obj robot)",
                "description": "Pull an object closer"
            },
            {
                "name": "turn-on",
                "parameters": [("obj", "object")],
                "precondition": "(and (switchable ?obj) (not (on-state ?obj)))",
                "effect": "(on-state ?obj)",
                "description": "Turn on a switchable object"
            },
            {
                "name": "turn-off",
                "parameters": [("obj", "object")],
                "precondition": "(and (switchable ?obj) (on-state ?obj))",
                "effect": "(not (on-state ?obj))",
                "description": "Turn off a switchable object"
            }
        ]

        for action_def in predefined:
            self.add_predefined_action(
                name=action_def["name"],
                parameters=action_def["parameters"],
                precondition=action_def["precondition"],
                effect=action_def["effect"],
                description=action_def.get("description")
            )

        # Add corresponding predicates
        common_predicates = [
            ("graspable", [("obj", "object")]),
            ("openable", [("obj", "object")]),
            ("pourable", [("obj", "object")]),
            ("containable", [("obj", "object")]),
            ("pushable", [("obj", "object")]),
            ("pullable", [("obj", "object")]),
            ("switchable", [("obj", "object")]),
            ("supportable", [("obj", "object")]),
            ("holding", [("obj", "object")]),
            ("empty-hand", []),
            ("on", [("obj1", "object"), ("obj2", "object")]),
            ("at", [("obj", "object"), ("loc", "object")]),
            ("near", [("obj1", "object"), ("obj2", "object")]),
            ("opened", [("obj", "object")]),
            ("filled", [("obj", "object")]),
            ("empty", [("obj", "object")]),
            ("on-state", [("obj", "object")])
        ]

        for pred_name, params in common_predicates:
            if pred_name not in self.predicates:
                self.add_predicate(pred_name, params)

    # ========================================================================
    # Domain Text Management (for LLM-based refinement)
    # ========================================================================

    def get_domain_text(self, force_regenerate: bool = False) -> str:
        """
        Get the domain as PDDL text.

        Args:
            force_regenerate: If True, regenerate from structured representation
                            If False, return cached version if available

        Returns:
            PDDL domain text
        """
        if force_regenerate or self._cache_dirty or self._domain_text_cache is None:
            self._domain_text_cache = self.generate_domain_pddl()
            self._cache_dirty = False

        return self._domain_text_cache

    def set_domain_text(self, pddl_text: str, update_structured: bool = True) -> None:
        """
        Set the domain from PDDL text.

        Args:
            pddl_text: PDDL domain text
            update_structured: If True, parse and update structured representation
        """
        req_values = {r.value for r in self.requirements}
        if ":conditional-effects" not in req_values and (
            'forall' in pddl_text or 'when' in pddl_text
        ):
            pddl_text = self._strip_forall_when_from_domain_text(pddl_text)
        self._domain_text_cache = pddl_text
        self._cache_dirty = False

        if update_structured:
            self._update_from_domain_text(pddl_text)

    def _mark_cache_dirty(self) -> None:
        """Mark cache as dirty (call this when structured representation changes)."""
        self._cache_dirty = True

    def _update_from_domain_text(self, pddl_text: str) -> None:
        """
        Update structured representation from PDDL text.

        This parses the PDDL text and updates the internal structured representation.
        Uses a simplified parser focused on predicates and basic structure.

        Args:
            pddl_text: PDDL domain text
        """
        import re

        # Extract domain name
        domain_match = re.search(r'\(define\s+\(domain\s+(\w+)\)', pddl_text)
        if domain_match:
            self.domain_name = domain_match.group(1)

        # Extract and parse predicates section.
        # Use a balanced-paren scan instead of a greedy/non-greedy regex because
        # the (:predicates ...) block contains nested parens (one per predicate).
        pred_start = pddl_text.find('(:predicates')
        if pred_start != -1:
            depth = 0
            pred_end = pred_start
            for i, ch in enumerate(pddl_text[pred_start:], pred_start):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        pred_end = i
                        break
            # pred_section is the content between (:predicates and the closing )
            pred_section = pddl_text[pred_start + len('(:predicates'):pred_end]
            self._parse_predicates_from_text(pred_section)

        # Extract and parse actions
        # Pattern to match entire actions with their content
        action_pattern = r'\(:action\s+(\w+)\s+(.*?)\n\s*\)'
        action_matches = re.finditer(action_pattern, pddl_text, re.DOTALL)

        for action_match in action_matches:
            action_name = action_match.group(1)
            action_body = action_match.group(2)
            self._parse_action_from_text(action_name, action_body)

        self.last_modified = time.time()

    def _parse_predicates_from_text(self, pred_section: str) -> None:
        """
        Parse predicates from PDDL text section.

        Args:
            pred_section: Content of (:predicates ...) section
        """
        # Find all top-level predicate definitions by scanning for balanced parens.
        pred_lines = []
        i = 0
        while i < len(pred_section):
            if pred_section[i] == '(':
                depth = 0
                start = i
                while i < len(pred_section):
                    if pred_section[i] == '(':
                        depth += 1
                    elif pred_section[i] == ')':
                        depth -= 1
                        if depth == 0:
                            pred_lines.append(pred_section[start + 1:i])  # content without outer parens
                            i += 1
                            break
                    i += 1
            else:
                i += 1

        new_predicates = {}
        for pred_line in pred_lines:
            parts = pred_line.strip().split()
            if not parts:
                continue

            pred_name = parts[0]

            # Parse parameters: ?name - type
            params = []
            i = 1
            while i < len(parts):
                if parts[i].startswith('?'):
                    param_name = parts[i][1:]  # Remove ?
                    # Look for - type pattern
                    if i + 2 < len(parts) and parts[i + 1] == '-':
                        param_type = parts[i + 2]
                        i += 3
                    else:
                        param_type = 'object'
                        i += 1
                    params.append((param_name, param_type))
                else:
                    i += 1

            # Create Predicate object
            from .utils.pddl_types import Predicate
            new_predicates[pred_name] = Predicate(
                name=pred_name,
                parameters=params,
                description=f"Parsed from PDDL text"
            )

        # Update predicates
        self.predicates = new_predicates

    def _parse_action_from_text(self, action_name: str, action_body: str) -> None:
        """
        Parse action from PDDL text.

        Args:
            action_name: Name of the action
            action_body: Body of the action (parameters, preconditions, effects)
        """
        import re
        from .utils.pddl_types import Action

        # Extract parameters
        param_match = re.search(r':parameters\s+\((.*?)\)', action_body, re.DOTALL)
        parameters = []
        if param_match:
            param_str = param_match.group(1)
            # Parse ?name - type pairs
            param_parts = param_str.split()
            i = 0
            while i < len(param_parts):
                if param_parts[i].startswith('?'):
                    param_name = param_parts[i][1:]
                    if i + 2 < len(param_parts) and param_parts[i + 1] == '-':
                        param_type = param_parts[i + 2]
                        i += 3
                    else:
                        param_type = 'object'
                        i += 1
                    parameters.append((param_name, param_type))
                else:
                    i += 1

        # Extract preconditions (simplified - just store as string)
        precond_match = re.search(r':precondition\s+(.*?)(?=:effect|\Z)', action_body, re.DOTALL)
        preconditions = precond_match.group(1).strip() if precond_match else "(and)"

        # Extract effects (simplified - just store as string)
        effect_match = re.search(r':effect\s+(.*)', action_body, re.DOTALL)
        effects = effect_match.group(1).strip() if effect_match else "(and)"

        # Create Action object
        action = Action(
            name=action_name,
            parameters=parameters,
            precondition=preconditions,
            effect=effects,
            description=f"Parsed from PDDL text"
        )

        # Store in LLM-generated actions (since we don't know origin)
        self.llm_generated_actions[action_name] = action
