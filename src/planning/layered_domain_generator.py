"""
Layered Domain Generator

Implements the 5-layer gated domain generation pipeline described in
gtamp-update-plan.md. Each layer is validated before the next layer runs;
failures trigger targeted regeneration of only that layer.

Layers:
  L1: Goal Specification       — extract grounded goal predicates from NL task
  L2: Predicate Vocabulary     — minimal predicate set covering L1 goals
  L3: Action Schemas           — pure PDDL actions using only L2 predicates
  L4: Grounding Pre-check      — algorithmic scene feasibility (no LLM)
  L5: Initial State            — algorithmic state construction from scene (no LLM)

Usage:
    generator = LayeredDomainGenerator(api_key=os.getenv("GEMINI_API_KEY"))
    artifact = await generator.generate_domain(
        "Put the red block on the blue block",
        observed_objects=[
            {"object_id": "red_block_1", "object_type": "block",
             "affordances": ["graspable", "stackable"], "position_3d": [0.3, 0.0, 0.1]},
            ...
        ]
    )
    task_analysis = artifact.to_task_analysis()
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml
from google import genai
from google.genai import types

from ..utils.prompt_utils import render_prompt_template
from .utils.task_types import (
    L1GoalArtifact,
    L2PredicateArtifact,
    L3ActionArtifact,
    L4GroundingArtifact,
    L5InitialStateArtifact,
    LayeredDomainArtifact,
)

# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, reusable in tests)
# ---------------------------------------------------------------------------

_LOGICAL_OPS = {"and", "or", "not", "forall", "exists", "when", "imply"}


def extract_predicate_name_from_literal(literal: str) -> Optional[str]:
    """Return the predicate name from a PDDL literal like '(on red_block_1 blue_block_1)'."""
    m = re.match(r"^\(\s*([a-zA-Z][a-zA-Z0-9_-]*)", literal.strip())
    return m.group(1).lower() if m else None


def extract_predicates_from_formula(formula: str) -> Set[str]:
    """Extract all predicate names used in a PDDL formula string."""
    names: Set[str] = set()
    for match in re.finditer(r"\(([a-zA-Z][a-zA-Z0-9_-]*)", formula):
        name = match.group(1).lower()
        if name not in _LOGICAL_OPS:
            names.add(name)
    return names


def extract_variables_from_formula(formula: str) -> Set[str]:
    """Extract all ?variable names used in a PDDL formula."""
    return set(re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", formula))


def bfs_reachable_predicates(
    actions: List[Dict],
    initial_predicates: Set[str],
) -> Set[str]:
    """
    Relaxed-plan-graph BFS: find all predicates reachable from initial_predicates
    by repeatedly applying action positive effects (delete-relaxed).

    Returns the set of all reachable predicate names.
    """
    reachable = set(initial_predicates)
    changed = True
    while changed:
        changed = False
        for action in actions:
            pre = action.get("precondition", "")
            eff = action.get("effect", "")
            # Relaxed: action fires if all positive literals in precondition are reachable
            pre_preds = extract_predicates_from_formula(pre)
            if pre_preds <= reachable:
                eff_preds = extract_predicates_from_formula(eff)
                new = eff_preds - reachable
                if new:
                    reachable |= new
                    changed = True
    return reachable


# ---------------------------------------------------------------------------
# LayeredDomainGenerator
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS_PATH = (
    Path(__file__).parent.parent.parent / "config" / "layered_domain_generator_prompts.yaml"
)

_ROBOT_DESCRIPTION = (
    "The robot is a 7-DOF robot arm, with a two-finger rigid gripper as end effector. "
    "The end effector cannot grasp objects <1cm or >15cm. "
    "The robot has a Realsense RGBD camera mounted at the end effector."
)


class LayerValidationError(Exception):
    """Raised when a layer exceeds max retries without passing validation."""
    def __init__(self, layer: str, errors: List[str]) -> None:
        self.layer = layer
        self.errors = errors
        super().__init__(f"Layer {layer} validation failed after max retries: {errors}")


class LayeredDomainGenerator:
    """
    Generates a PDDL domain by running five ordered, validated layers.

    Each layer that relies on LLM output is automatically retried (up to
    `max_layer_retries`) with the validation errors appended to the prompt.
    L4 and L5 are fully algorithmic (no LLM call).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        prompts_config_path: Optional[Path] = None,
        max_layer_retries: int = 2,
        dkb: Optional[Any] = None,
        robot_description: Optional[str] = None,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_layer_retries = max_layer_retries
        self.dkb = dkb
        self.robot_description = robot_description or _ROBOT_DESCRIPTION

        path = Path(prompts_config_path) if prompts_config_path else _DEFAULT_PROMPTS_PATH
        with open(path, "r", encoding="utf-8") as f:
            self._prompts: Dict[str, str] = yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate_domain(
        self,
        task: str,
        observed_objects: Optional[List[Dict]] = None,
        image: Optional[Any] = None,
    ) -> LayeredDomainArtifact:
        """
        Run the full L1→L5 pipeline and return a LayeredDomainArtifact.

        Args:
            task: Natural language task description.
            observed_objects: List of object dicts from perception, each with at minimum
                              {"object_id": str, "object_type": str, "affordances": list,
                               "position_3d": list[float]}.
            image: Optional image (currently unused, reserved for future L4 VLM calls).

        Returns:
            LayeredDomainArtifact — bridge to legacy TaskAnalysis via .to_task_analysis().
        """
        scene_objects: List[Dict] = observed_objects or []

        print(f"\n[LayeredDomainGenerator] Generating domain for: '{task}'")
        print(f"  Scene objects: {[o.get('object_id') for o in scene_objects]}")

        # L1 — Goal Specification
        l1 = await self._run_layer_with_retry(
            layer_name="L1",
            run_fn=lambda errs: self._run_l1(task, scene_objects, validation_errors=errs),
            validate_fn=lambda art: self._validate_l1(art, scene_objects),
            repair_fn=None,
        )
        print(f"  [L1] Goals: {l1.goal_predicates}")

        # L2 — Predicate Vocabulary
        l2 = await self._run_layer_with_retry(
            layer_name="L2",
            run_fn=lambda errs: self._run_l2(task, l1, scene_objects, validation_errors=errs),
            validate_fn=lambda art: self._validate_l2(art, l1),
            repair_fn=lambda art: self._repair_l2_auto(art),
        )
        print(f"  [L2] Predicates: {l2.predicate_signatures}")
        if l2.checked_variants:
            print(f"  [L2] Auto-generated checked variants: {l2.checked_variants}")

        # L3 — Action Schemas
        l3 = await self._run_layer_with_retry(
            layer_name="L3",
            run_fn=lambda errs: self._run_l3(task, l1, l2, validation_errors=errs),
            validate_fn=lambda art: self._validate_l3(art, l2, l1),
            repair_fn=lambda art: self._repair_l3_auto(art, l2),
        )
        print(f"  [L3] Actions: {[a.get('name') for a in l3.actions]}")

        # L4 — Grounding Pre-check (algorithmic)
        l4 = self._run_l4_precheck(l1, l3, scene_objects)
        if l4.warnings:
            for w in l4.warnings:
                print(f"  [L4] Warning: {w}")

        # L5 — Initial State Construction (algorithmic)
        l5 = self._run_l5(l2, l3, scene_objects, l1)
        print(f"  [L5] Initial true literals: {len(l5.true_literals)}, false: {len(l5.false_literals)}")

        artifact = LayeredDomainArtifact(
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            l5=l5,
            task_description=task,
            scene_objects=scene_objects,
        )

        # Record to DKB if available (non-blocking)
        if self.dkb is not None:
            try:
                self.dkb.record_execution(task, artifact, solver_result=None)
            except Exception:
                pass

        return artifact

    # ------------------------------------------------------------------
    # Generic retry loop
    # ------------------------------------------------------------------

    async def _run_layer_with_retry(
        self,
        layer_name: str,
        run_fn: Callable[[List[str]], Any],
        validate_fn: Callable[[Any], List[str]],
        repair_fn: Optional[Callable[[Any], Any]],
    ) -> Any:
        """
        Run a layer, validate, optionally auto-repair, and retry on validation failure.

        On retry, the validation error list is passed to run_fn so it can inject
        the errors into the <<VALIDATION_ERRORS>> placeholder.
        """
        errors: List[str] = []
        artifact = None

        for attempt in range(self.max_layer_retries + 1):
            artifact = await run_fn(errors)
            errors = validate_fn(artifact)

            if not errors:
                artifact.validation_errors = []
                return artifact

            # Try auto-repair before next LLM retry
            if repair_fn is not None:
                artifact = repair_fn(artifact)
                errors = validate_fn(artifact)
                if not errors:
                    artifact.validation_errors = []
                    return artifact

            print(f"  [{layer_name}] Attempt {attempt + 1}/{self.max_layer_retries + 1} — "
                  f"validation errors: {errors}")
            artifact.generation_attempts = attempt + 1

        # Record errors on the artifact and return it (don't raise — allow partial results)
        artifact.validation_errors = errors
        print(f"  [{layer_name}] Max retries reached with errors: {errors}")
        return artifact

    # ------------------------------------------------------------------
    # L1: Goal Specification
    # ------------------------------------------------------------------

    async def _run_l1(
        self,
        task: str,
        scene_objects: List[Dict],
        validation_errors: Optional[List[str]] = None,
    ) -> L1GoalArtifact:
        template = self._prompts.get("l1_goal_spec_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "ROBOT_DESCRIPTION": self.robot_description,
            "OBJECTS_JSON": json.dumps(scene_objects, indent=2),
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = await self._call_llm(prompt)
        data = json.loads(text)
        return L1GoalArtifact(
            goal_predicates=data.get("goal_predicates", []),
            goal_objects=data.get("goal_objects", []),
            global_predicates=data.get("global_predicates", []),
        )

    def _validate_l1(
        self, artifact: L1GoalArtifact, scene_objects: List[Dict]
    ) -> List[str]:
        errors: List[str] = []
        known_ids: Set[str] = {o.get("object_id", "") for o in scene_objects}

        if not artifact.goal_predicates:
            errors.append("goal_predicates is empty — must have at least one goal")
            return errors

        for gp in artifact.goal_predicates:
            gp = gp.strip()
            # Must be a grounded PDDL literal (no ?-variables)
            if "?" in gp:
                errors.append(
                    f"Goal predicate contains a variable (must be grounded): {gp}"
                )
            # Must start/end with parentheses
            if not (gp.startswith("(") and gp.endswith(")")):
                errors.append(f"Goal predicate not a valid PDDL literal (missing parens): {gp}")
                continue
            # Check arguments are known object_ids (only when scene is non-empty)
            if known_ids:
                tokens = re.findall(r"[^\s()]+", gp)
                for tok in tokens[1:]:  # skip predicate name
                    if tok not in known_ids:
                        errors.append(
                            f"Goal predicate argument '{tok}' not in observed objects. "
                            f"Known: {sorted(known_ids)}"
                        )

        return errors

    # ------------------------------------------------------------------
    # L2: Predicate Vocabulary
    # ------------------------------------------------------------------

    async def _run_l2(
        self,
        task: str,
        l1: L1GoalArtifact,
        scene_objects: List[Dict],
        validation_errors: Optional[List[str]] = None,
    ) -> L2PredicateArtifact:
        dkb_hints = self._get_dkb_predicate_hints(task)
        template = self._prompts.get("l2_predicate_vocab_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "GOAL_PREDICATES_JSON": json.dumps(l1.goal_predicates, indent=2),
            "OBJECTS_JSON": json.dumps(scene_objects, indent=2),
            "DKB_HINTS": dkb_hints,
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = await self._call_llm(prompt)
        data = json.loads(text)
        return L2PredicateArtifact(
            predicate_signatures=data.get("predicate_signatures", []),
            sensed_predicates=data.get("sensed_predicates", []),
            checked_variants=[],
        )

    def _validate_l2(
        self, artifact: L2PredicateArtifact, l1: L1GoalArtifact
    ) -> List[str]:
        errors: List[str] = []
        defined_names = {
            extract_predicate_name_from_literal(sig) or sig.split()[0].lstrip("(")
            for sig in artifact.predicate_signatures
        }

        # All goal predicate names must be defined
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in defined_names:
                errors.append(
                    f"Goal predicate '{name}' from L1 is not defined in predicate vocabulary"
                )

        # Naming compliance
        name_pattern = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
        for sig in artifact.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name and not name_pattern.match(name):
                errors.append(
                    f"Predicate name '{name}' does not follow lowercase-hyphen convention"
                )

        return errors

    def _repair_l2_auto(self, artifact: L2PredicateArtifact) -> L2PredicateArtifact:
        """Auto-generate checked-X variants for all sensed predicates."""
        existing_names = {
            extract_predicate_name_from_literal(sig) or ""
            for sig in artifact.predicate_signatures
        }
        new_checked: List[str] = list(artifact.checked_variants)

        for sensed_name in artifact.sensed_predicates:
            checked_name = f"checked-{sensed_name}"
            if checked_name not in existing_names:
                # Find the arity of the base sensed predicate
                base_sig = next(
                    (s for s in artifact.predicate_signatures
                     if (extract_predicate_name_from_literal(s) or "") == sensed_name),
                    None
                )
                if base_sig:
                    # Build checked variant with same parameters
                    params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", base_sig)
                    if params:
                        checked_sig = f"({checked_name} {' '.join(params)})"
                    else:
                        checked_sig = f"({checked_name})"
                else:
                    checked_sig = f"({checked_name} ?obj)"

                if checked_sig not in artifact.predicate_signatures:
                    artifact.predicate_signatures.append(checked_sig)
                new_checked.append(checked_sig)

        artifact.checked_variants = new_checked
        return artifact

    # ------------------------------------------------------------------
    # L3: Action Schemas
    # ------------------------------------------------------------------

    async def _run_l3(
        self,
        task: str,
        l1: L1GoalArtifact,
        l2: L2PredicateArtifact,
        validation_errors: Optional[List[str]] = None,
    ) -> L3ActionArtifact:
        dkb_hints = self._get_dkb_action_hints(task)
        template = self._prompts.get("l3_action_schema_prompt", "")
        prompt = render_prompt_template(template, {
            "TASK": task,
            "ROBOT_DESCRIPTION": self.robot_description,
            "GOAL_PREDICATES_JSON": json.dumps(l1.goal_predicates, indent=2),
            "PREDICATE_VOCAB_JSON": json.dumps(l2.predicate_signatures, indent=2),
            "DKB_HINTS": dkb_hints,
            "VALIDATION_ERRORS": self._format_errors(validation_errors),
        })
        text = await self._call_llm(prompt)
        data = json.loads(text)
        return L3ActionArtifact(
            actions=data.get("actions", []),
            sensing_actions=data.get("sensing_actions", []),
        )

    def _validate_l3(
        self,
        artifact: L3ActionArtifact,
        l2: L2PredicateArtifact,
        l1: L1GoalArtifact,
    ) -> List[str]:
        errors: List[str] = []
        all_actions = artifact.actions + artifact.sensing_actions

        # Build vocabulary of defined predicate names
        vocab: Set[str] = set()
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                vocab.add(name)

        for action in all_actions:
            name = action.get("name", "<unnamed>")
            params = action.get("parameters", [])
            pre = action.get("precondition", "")
            eff = action.get("effect", "")
            declared_vars: Set[str] = set(params)

            # Symbolic closure: all predicates in pre/eff must be in vocabulary
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used = extract_predicates_from_formula(formula)
                undefined = used - vocab
                if undefined:
                    errors.append(
                        f"Action '{name}' {formula_name} uses undefined predicates: {undefined}"
                    )

            # Parameter consistency: all ?vars must be declared
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used_vars = extract_variables_from_formula(formula)
                undeclared = used_vars - declared_vars
                if undeclared:
                    errors.append(
                        f"Action '{name}' {formula_name} uses undeclared variables: {undeclared}"
                    )

        # Goal achievability: BFS from global predicates
        global_preds: Set[str] = {
            p.strip("()").split()[0] for p in l1.global_predicates
        } | {
            extract_predicate_name_from_literal(sig) or ""
            for sig in l2.predicate_signatures
            if not re.findall(r"\?", sig)  # zero-param predicates can be in init
        }
        reachable = bfs_reachable_predicates(all_actions, global_preds | vocab)

        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in reachable:
                errors.append(
                    f"Goal predicate '{name}' is not reachable via any action's effects"
                )

        return errors

    def _repair_l3_auto(
        self, artifact: L3ActionArtifact, l2: L2PredicateArtifact
    ) -> L3ActionArtifact:
        """Auto-generate sensing actions for any checked-X predicates in preconditions."""
        vocab_names: Set[str] = {
            extract_predicate_name_from_literal(sig) or "" for sig in l2.predicate_signatures
        }
        existing_action_names: Set[str] = {
            a.get("name", "") for a in artifact.actions + artifact.sensing_actions
        }
        # Find checked-* predicates that appear in preconditions but no action produces them
        produced: Set[str] = set()
        all_actions = artifact.actions + artifact.sensing_actions
        for action in all_actions:
            produced |= extract_predicates_from_formula(action.get("effect", ""))

        for action in all_actions:
            for pred in extract_predicates_from_formula(action.get("precondition", "")):
                if pred.startswith("checked-") and pred not in produced:
                    base = pred[len("checked-"):]
                    action_name = f"check-{base}"
                    if action_name not in existing_action_names:
                        # Find arity from vocabulary
                        base_sig = next(
                            (s for s in l2.predicate_signatures
                             if (extract_predicate_name_from_literal(s) or "") == base),
                            None
                        )
                        params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", base_sig) if base_sig else ["?obj"]
                        param_str = " ".join(params)
                        sensing = {
                            "name": action_name,
                            "parameters": params,
                            "precondition": "(and)" if not params else f"(near {params[0]})" if "near" in vocab_names else "(and)",
                            "effect": f"({pred} {param_str})" if params else f"({pred})",
                        }
                        artifact.sensing_actions.append(sensing)
                        existing_action_names.add(action_name)
                        produced.add(pred)
        return artifact

    # ------------------------------------------------------------------
    # L4: Grounding Pre-check (algorithmic, no LLM)
    # ------------------------------------------------------------------

    def _run_l4_precheck(
        self,
        l1: L1GoalArtifact,
        l3: L3ActionArtifact,
        scene_objects: List[Dict],
    ) -> L4GroundingArtifact:
        """
        Check that actions have at least one candidate object in the scene.
        Warns (non-blocking) when parameterized actions have no matching objects.
        """
        warnings: List[str] = []
        bindings: Dict[str, str] = {}

        all_actions = l3.actions + l3.sensing_actions
        obj_ids = [o.get("object_id", "") for o in scene_objects]
        graspable_ids = [
            o.get("object_id", "")
            for o in scene_objects
            if "graspable" in o.get("affordances", [])
        ]

        for action in all_actions:
            name = action.get("name", "")
            params = action.get("parameters", [])
            if not params:
                continue
            if not obj_ids:
                warnings.append(
                    f"Action '{name}' has parameters but no objects in scene"
                )
            else:
                candidates = graspable_ids or obj_ids
                bindings[name] = candidates[0]

        # Check goal objects exist in scene
        for goal_obj in l1.goal_objects:
            if goal_obj not in obj_ids and obj_ids:
                warnings.append(
                    f"Goal object '{goal_obj}' not found in scene objects {obj_ids}"
                )

        return L4GroundingArtifact(object_bindings=bindings, warnings=warnings)

    # ------------------------------------------------------------------
    # L5: Initial State Construction (algorithmic, no LLM)
    # ------------------------------------------------------------------

    def _run_l5(
        self,
        l2: L2PredicateArtifact,
        l3: L3ActionArtifact,
        scene_objects: List[Dict],
        l1: L1GoalArtifact,
    ) -> L5InitialStateArtifact:
        """
        Construct initial PDDL state from scene observation.

        Rules:
        - checked-* predicates: always FALSE
        - graspable: TRUE if "graspable" in object affordances
        - on/above/atop: TRUE if z_distance between two objects < threshold
        - hand-empty, arm-at-home, etc. (global_predicates): TRUE
        - Zero-param state predicates in global_predicates: TRUE
        """
        true_lits: List[Tuple[str, List[str]]] = []
        false_lits: List[Tuple[str, List[str]]] = []

        # Determine which predicate names are checked-* (always FALSE)
        checked_names: Set[str] = set()
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name and name.startswith("checked-"):
                checked_names.add(name)
        for cv in l2.checked_variants:
            name = extract_predicate_name_from_literal(cv)
            if name:
                checked_names.add(name)

        # Get all defined predicate names with their arities from signatures
        pred_arities: Dict[str, int] = {}
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", sig)
                pred_arities[name] = len(params)

        obj_ids = [o.get("object_id", "") for o in scene_objects]

        # Add checked-* predicates as FALSE for each object
        for cname in checked_names:
            arity = pred_arities.get(cname, 1)
            if arity == 0:
                false_lits.append((cname, []))
            else:
                for obj in scene_objects:
                    oid = obj.get("object_id", "")
                    if arity == 1:
                        false_lits.append((cname, [oid]))
                    # arity >= 2 handled below

        # Add zero-parameter global predicates as TRUE
        for gp in l1.global_predicates:
            gp_clean = gp.strip("() ")
            true_lits.append((gp_clean, []))

        # Add graspable(obj) for graspable objects
        if "graspable" in pred_arities:
            for obj in scene_objects:
                oid = obj.get("object_id", "")
                if "graspable" in obj.get("affordances", []):
                    true_lits.append(("graspable", [oid]))

        # Add on(obj, surface) where obj is above a surface
        if "on" in pred_arities and pred_arities["on"] == 2:
            surfaces = [o for o in scene_objects if "support_surface" in o.get("affordances", [])]
            manipulables = [o for o in scene_objects if "support_surface" not in o.get("affordances", [])]
            for obj in manipulables:
                obj_pos = obj.get("position_3d", [0, 0, 0])
                for surf in surfaces:
                    surf_pos = surf.get("position_3d", [0, 0, 0])
                    # Consider object "on" surface if it's above and within 0.3m horizontally
                    dz = obj_pos[2] - surf_pos[2]
                    dx = abs(obj_pos[0] - surf_pos[0])
                    dy = abs(obj_pos[1] - surf_pos[1]) if len(obj_pos) > 1 else 0
                    if 0 <= dz <= 0.3 and dx < 0.5 and dy < 0.5:
                        true_lits.append(("on", [obj.get("object_id", ""), surf.get("object_id", "")]))

        # Add on(obj1, obj2) style — only if no surface relationship already covers it
        # (For blocksworld, objects stacked on other objects)
        if "on" in pred_arities and pred_arities["on"] == 2:
            non_surfaces = [o for o in scene_objects if "support_surface" not in o.get("affordances", [])]
            already_on = {(a, b) for (p, (a, *_)) in [(lit[0], lit[1]) for lit in true_lits if lit[0] == "on"] for b in lit[1][1:]}
            for i, obj1 in enumerate(non_surfaces):
                for obj2 in non_surfaces:
                    if obj1 is obj2:
                        continue
                    pos1 = obj1.get("position_3d", [0, 0, 0])
                    pos2 = obj2.get("position_3d", [0, 0, 0])
                    dz = pos1[2] - pos2[2]
                    dx = abs(pos1[0] - pos2[0])
                    dy = abs(pos1[1] - pos2[1]) if len(pos1) > 1 else 0
                    if 0 < dz <= 0.15 and dx < 0.08 and dy < 0.08:
                        true_lits.append(("on", [obj1.get("object_id", ""), obj2.get("object_id", "")]))

        return L5InitialStateArtifact(true_literals=true_lits, false_literals=false_lits)

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        response_mime_type: str = "application/json",
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=4096,
            response_mime_type=response_mime_type,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config,
        )
        text = getattr(response, "text", None)
        if text is None:
            raise ValueError("LLM response missing text payload")
        return text if isinstance(text, str) else str(text)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_errors(errors: Optional[List[str]]) -> str:
        if not errors:
            return ""
        lines = "\n".join(f"  - {e}" for e in errors)
        return f"\n**Validation errors from previous attempt (fix these):**\n{lines}\n"

    def _get_dkb_predicate_hints(self, task: str) -> str:
        if self.dkb is None:
            return "(none)"
        try:
            hints = self.dkb.get_predicate_hints(task)
            return json.dumps(hints, indent=2) if hints else "(none)"
        except Exception:
            return "(none)"

    def _get_dkb_action_hints(self, task: str) -> str:
        if self.dkb is None:
            return "(none)"
        try:
            hints = self.dkb.get_action_hints(task)
            return json.dumps(hints, indent=2) if hints else "(none)"
        except Exception:
            return "(none)"
