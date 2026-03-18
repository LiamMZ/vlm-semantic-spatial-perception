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
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """
    Parse JSON from raw LLM output, tolerating markdown code fences and
    prose before/after the JSON block.
    """
    # 1. Try bare parse first (clean output)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Extract from ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3. Find the first {...} or [...] block in the text
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise json.JSONDecodeError("No valid JSON found in model output", text, 0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised PDDL object/parameter types. Callers may extend this set.
VALID_PDDL_TYPES: Set[str] = {
    "object", "block", "surface", "robot", "location", "gripper",
    "item", "container", "tool", "agent",
}

# Valid type_classification values for L2 predicate entries
VALID_TYPE_CLASSIFICATIONS: Set[str] = {
    "sensed", "checked", "derived", "action", "robot_state", "object_state", "external",
}

# Pattern for lowercase-hyphenated PDDL names
_PDDL_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

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
        llm_client: Optional[Any] = None,  # src.llm_interface.LLMClient
    ) -> None:
        self._llm_client = llm_client
        if llm_client is None:
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

        # L2-V5 (deferred): remove predicates unused by any action or goal
        l2 = self._prune_unused_predicates(l2, l3, l1)

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
        data = _extract_json(text)
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

        # L1-V1: non-empty goal set
        if not artifact.goal_predicates:
            errors.append(
                "goal_predicates is empty. Re-examine the task description and "
                "extract at least one desired end-state condition."
            )
            return errors

        # Track (predicate_name, first_arg) pairs for contradiction detection (L1-V3)
        # Maps predicate_name -> set of (first_arg, full_literal) to catch duplicates
        # where the same predicate places the same object in two different locations.
        pred_first_arg_values: Dict[str, Dict[str, str]] = {}

        for i, gp in enumerate(artifact.goal_predicates):
            gp = gp.strip()

            # L1-V2a: must start/end with parentheses
            if not (gp.startswith("(") and gp.endswith(")")):
                errors.append(
                    f"Goal {i}: predicate '{gp}' is not a valid PDDL literal (missing parentheses)"
                )
                continue

            # L1-V2b: no ?-variables — goals must be grounded
            if "?" in gp:
                errors.append(
                    f"Goal {i}: predicate contains a variable — goals must be grounded: {gp}"
                )
                continue

            tokens = re.findall(r"[^\s()]+", gp)
            if not tokens:
                errors.append(f"Goal {i}: empty literal")
                continue

            pred_name = tokens[0]
            args = tokens[1:]

            # L1-V2c: predicate name must match PDDL naming pattern
            if not _PDDL_NAME_RE.match(pred_name):
                errors.append(
                    f"Goal {i}: predicate name '{pred_name}' does not match "
                    f"PDDL naming pattern (lowercase-hyphenated)"
                )

            # L1-V2d: must have at least one argument
            if not args:
                errors.append(
                    f"Goal {i}: predicate '{pred_name}' has no arguments — "
                    f"grounded goal predicates must reference at least one object"
                )
                continue

            # L1-V2e: no empty-string arguments
            for j, arg in enumerate(args):
                if not arg:
                    errors.append(f"Goal {i}, argument {j}: empty argument in '{gp}'")

            # L1-V3: contradiction detection
            # If the same predicate asserts the same subject (first arg) goes to
            # two different second args, that's a contradiction (e.g., on(A, B) + on(A, C))
            if len(args) >= 2:
                subject = args[0]
                location = args[1]
                key = f"{pred_name}:{subject}"
                if key in pred_first_arg_values:
                    prev = pred_first_arg_values[key]
                    if prev != location:
                        errors.append(
                            f"Contradiction: predicate '{pred_name}' asserts '{subject}' "
                            f"maps to both '{prev}' and '{location}'. "
                            f"An entity can only satisfy one value per uniqueness-constrained argument."
                        )
                else:
                    pred_first_arg_values[key] = location

            # L1-V4: all arguments reference known scene entities (error when scene non-empty)
            if known_ids:
                for arg in args:
                    if arg and arg not in known_ids:
                        errors.append(
                            f"Goal {i}: argument '{arg}' not found in observed scene. "
                            f"Known objects: {sorted(known_ids)}"
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
        data = _extract_json(text)
        return L2PredicateArtifact(
            predicate_signatures=data.get("predicate_signatures", []),
            sensed_predicates=data.get("sensed_predicates", []),
            checked_variants=[],
        )

    def _validate_l2(
        self, artifact: L2PredicateArtifact, l1: L1GoalArtifact
    ) -> List[str]:
        errors: List[str] = []

        # Build name → parameter list mapping from signatures
        sig_params: Dict[str, List[str]] = {}
        for sig in artifact.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name:
                params = re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+-\s+[a-zA-Z][a-zA-Z0-9_-]*)?", sig)
                sig_params[name] = params

        defined_names: Set[str] = set(sig_params.keys())

        # L2-V1: all goal predicate names must be defined in P
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in defined_names:
                errors.append(
                    f"Goal predicate '{name}' from L1 is not defined in predicate vocabulary. "
                    f"Add a definition for it."
                )

        # L2-V2: predicate names AND type names must match lowercase-hyphen pattern
        # Use the raw name from the signature (before lowercasing) to catch uppercase letters
        for sig in artifact.predicate_signatures:
            raw_name_match = re.match(r"^\(\s*([^\s()]+)", sig.strip())
            raw_name = raw_name_match.group(1) if raw_name_match else None
            name = extract_predicate_name_from_literal(sig)  # lowercased version
            if raw_name and not _PDDL_NAME_RE.match(raw_name):
                errors.append(
                    f"Predicate name '{raw_name}' does not match PDDL naming pattern (lowercase-hyphenated)"
                )
            # Extract inline type annotations (?var - type_name) and validate type names
            for type_name in re.findall(r"\?\S+\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", sig):
                if not _PDDL_NAME_RE.match(type_name):
                    errors.append(
                        f"Predicate '{name}': type name '{type_name}' does not match "
                        f"PDDL naming pattern (lowercase-hyphenated)"
                    )

        # L2-V3 (type mismatch part): if checked-X exists, its params must match X's params
        for sensed_name in artifact.sensed_predicates:
            checked_name = f"checked-{sensed_name}"
            if checked_name in defined_names and sensed_name in defined_names:
                sensed_params = sig_params.get(sensed_name, [])
                checked_params = sig_params.get(checked_name, [])
                # Compare arity (type structure comparison is approximate — we only have names)
                if len(sensed_params) != len(checked_params):
                    errors.append(
                        f"Checked variant '{checked_name}' has {len(checked_params)} parameters "
                        f"but '{sensed_name}' has {len(sensed_params)} — they must match."
                    )

        # L2-V4: no conflicting type assignments for the same argument variable name
        # Collect all (var_name, type_name) pairs across all predicates
        var_types: Dict[str, Set[str]] = {}
        for sig in artifact.predicate_signatures:
            for var, type_name in re.findall(r"(\?\S+)\s+-\s+([a-zA-Z][a-zA-Z0-9_-]*)", sig):
                var_types.setdefault(var, set()).add(type_name.lower())
        for var, types_seen in var_types.items():
            if len(types_seen) > 1:
                errors.append(
                    f"Argument variable '{var}' has conflicting type assignments across predicates: "
                    f"{sorted(types_seen)}. Each argument name should map to exactly one type."
                )

        # L2-V6: every predicate entry must have a valid type_classification (if provided)
        # type_classifications is an optional dict in the artifact; skip if absent
        classifications = getattr(artifact, "type_classifications", {}) or {}
        for pred_name, classification in classifications.items():
            if classification not in VALID_TYPE_CLASSIFICATIONS:
                errors.append(
                    f"Predicate '{pred_name}' has invalid type_classification '{classification}'. "
                    f"Valid options: {sorted(VALID_TYPE_CLASSIFICATIONS)}"
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
        data = _extract_json(text)
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
            aname = action.get("name", "<unnamed>")
            raw_params = action.get("parameters", [])
            pre = action.get("precondition", "")
            eff = action.get("effect", "")

            # Declared variable names — strip PDDL type annotations (?var - type → ?var)
            declared_vars: Set[str] = set()
            for p in raw_params:
                p = str(p)
                var = p.split("-")[0].strip() if "-" in p else p.strip()
                if var.startswith("?"):
                    declared_vars.add(var)

            # L3-V1: symbolic closure — all predicates in pre/eff must be in vocabulary
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used = extract_predicates_from_formula(formula)
                undefined = used - vocab
                if undefined:
                    errors.append(
                        f"Action '{aname}' {formula_name} uses undefined predicates: "
                        f"{sorted(undefined)}. Available predicates: {sorted(vocab)}"
                    )

            # L3-V2: parameter consistency — all ?vars in pre/eff must be declared
            for formula_name, formula in [("precondition", pre), ("effect", eff)]:
                used_vars = extract_variables_from_formula(formula)
                undeclared = used_vars - declared_vars
                if undeclared:
                    errors.append(
                        f"Action '{aname}' {formula_name} uses undeclared variables: "
                        f"{sorted(undeclared)}"
                    )

        # L3-V3: sensing coverage — every checked-* in any precondition must have a producer
        produced: Set[str] = set()
        for action in all_actions:
            produced |= extract_predicates_from_formula(action.get("effect", ""))

        uncovered_checked: Set[str] = set()
        for action in all_actions:
            for pred in extract_predicates_from_formula(action.get("precondition", "")):
                if pred.startswith("checked-") and pred not in produced:
                    uncovered_checked.add(pred)

        if uncovered_checked:
            errors.append(
                f"No action produces the following checked predicates that appear in "
                f"preconditions: {sorted(uncovered_checked)}. "
                f"Sensing actions must be added to produce these."
            )

        # L3-V4: goal achievability — BFS reachability from initial predicates
        global_preds: Set[str] = {
            p.strip("()").split()[0] for p in l1.global_predicates
        } | {
            extract_predicate_name_from_literal(sig) or ""
            for sig in l2.predicate_signatures
            if not re.findall(r"\?", sig)  # zero-param predicates can be in init
        }
        reachable = bfs_reachable_predicates(all_actions, global_preds | vocab)

        unreachable_goals: List[str] = []
        no_producer: List[str] = []
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name and name not in reachable:
                unreachable_goals.append(name)
                if not any(
                    name in extract_predicates_from_formula(a.get("effect", ""))
                    for a in all_actions
                ):
                    no_producer.append(name)

        if unreachable_goals:
            errors.append(
                f"Goal predicates unreachable: {sorted(unreachable_goals)}. "
                f"Predicates with no producing action at all: {sorted(no_producer)}. "
                f"Add actions that produce these predicates or revise the goal specification."
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

    def _prune_unused_predicates(
        self,
        l2: L2PredicateArtifact,
        l3: L3ActionArtifact,
        l1: L1GoalArtifact,
    ) -> L2PredicateArtifact:
        """
        L2-V5 (deferred): remove predicates from L2 that are not referenced by
        any action formula or goal predicate. Runs after L3 is valid.
        """
        used: Set[str] = set()

        # Predicates referenced in goals
        for gp in l1.goal_predicates:
            name = extract_predicate_name_from_literal(gp)
            if name:
                used.add(name)

        # Predicates referenced in action pre/effects
        for action in l3.actions + l3.sensing_actions:
            used |= extract_predicates_from_formula(action.get("precondition", ""))
            used |= extract_predicates_from_formula(action.get("effect", ""))

        unused = [
            sig for sig in l2.predicate_signatures
            if (extract_predicate_name_from_literal(sig) or "") not in used
        ]
        if unused:
            unused_names = [extract_predicate_name_from_literal(s) for s in unused]
            print(f"  [L2-V5] Removed {len(unused)} unused predicates: {sorted(unused_names)}")
            l2.predicate_signatures = [
                sig for sig in l2.predicate_signatures if sig not in unused
            ]

        return l2

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
        Algorithmic feasibility checks (non-blocking — produces warnings, not errors).

        L4-V1: every PDDL type referenced in action parameters has at least one matching
               object in the scene (by affordance mapping).
        L4-V2: the scene contains at least one element for each type required by the domain.
        L4-V3: every entity in goal_objects exists in the scene.
        """
        warnings: List[str] = []
        bindings: Dict[str, str] = {}

        all_actions = l3.actions + l3.sensing_actions
        obj_ids = [o.get("object_id", "") for o in scene_objects]
        affordance_index: Dict[str, List[str]] = {}
        for obj in scene_objects:
            for aff in obj.get("affordances", []):
                affordance_index.setdefault(aff, []).append(obj.get("object_id", ""))

        graspable_ids = affordance_index.get("graspable", [])

        # Collect all PDDL types used in action parameters (from ?var - type annotations)
        domain_types: Set[str] = set()
        for action in all_actions:
            for param in action.get("parameters", []):
                if isinstance(param, str) and "-" in param:
                    type_part = param.split("-", 1)[1].strip()
                    domain_types.add(type_part.lower())

        # L4-V1: every domain type must have at least one scene object
        # We use a broad match: object type, affordances, or any substring
        for dtype in domain_types:
            has_match = any(
                dtype in o.get("object_type", "").lower() or
                dtype in [a.lower() for a in o.get("affordances", [])] or
                dtype == "object"  # "object" is universal
                for o in scene_objects
            )
            if not has_match and obj_ids:
                warnings.append(
                    f"L4-V1: No grounding rule / scene element found for PDDL type '{dtype}'. "
                    f"Available object types: {sorted({o.get('object_type','') for o in scene_objects})}"
                )

        # L4-V2: parameterized actions need at least one candidate object
        for action in all_actions:
            aname = action.get("name", "")
            params = action.get("parameters", [])
            if not params:
                continue
            if not obj_ids:
                warnings.append(
                    f"L4-V2: Action '{aname}' has parameters but scene has no objects. "
                    f"Trigger perception update and retry."
                )
            else:
                candidates = graspable_ids or obj_ids
                bindings[aname] = candidates[0]

        # L4-V3: every goal entity must exist in scene (exploration flag, non-blocking)
        for goal_obj in l1.goal_objects:
            if goal_obj not in obj_ids and obj_ids:
                warnings.append(
                    f"L4-V3: Goal references entity '{goal_obj}' not in scene. "
                    f"Exploration may be required to locate this object."
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
            already_on = {(lit[1][0], lit[1][1]) for lit in true_lits if lit[0] == "on" and len(lit[1]) >= 2}
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

        # L5-V2 (AUTO-REPAIR): initialise all sensed/external predicates as FALSE
        # (checked-* already covered above; this handles sensed predicates not yet checked)
        sensed_names: Set[str] = set()
        for sig in l2.predicate_signatures:
            name = extract_predicate_name_from_literal(sig)
            if name and name in (getattr(l2, "sensed_predicates", None) or []):
                sensed_names.add(name)
        # Also treat any predicate whose name is in sensed_predicates list
        for sname in (l2.sensed_predicates or []):
            sensed_names.add(sname)

        already_in_false = {lit[0] for lit in false_lits}
        for sname in sensed_names:
            if sname in already_in_false or sname in checked_names:
                continue  # already handled
            arity = pred_arities.get(sname, 1)
            if arity == 0:
                false_lits.append((sname, []))
            else:
                for obj in scene_objects:
                    oid = obj.get("object_id", "")
                    if arity == 1:
                        false_lits.append((sname, [oid]))

        # L5-V3 (REJECT): entity args in facts must reference known scene entities
        all_lits = true_lits + false_lits
        for pred_name, args in all_lits:
            for arg in args:
                if arg and arg not in obj_ids:
                    print(
                        f"  [L5-V3] WARNING: fact ({pred_name} {args}) references unknown "
                        f"entity '{arg}' — not in scene objects {obj_ids}"
                    )
                    # Remove the offending literal rather than crash
                    if (pred_name, args) in true_lits:
                        true_lits.remove((pred_name, args))
                    elif (pred_name, args) in false_lits:
                        false_lits.remove((pred_name, args))
                    break

        # L5-V5 (WARN): check if initial state already satisfies all goals
        true_facts: Set[str] = {
            f"({name} {' '.join(args)})".strip() if args else f"({name})"
            for name, args in true_lits
        }
        goal_already_satisfied = all(
            gp.strip() in true_facts
            for gp in l1.goal_predicates
            if gp.strip()
        )
        if goal_already_satisfied and l1.goal_predicates:
            print(
                "  [L5-V5] WARNING: Initial state already satisfies all goals. "
                "The task may be trivially complete. "
                "Verify goal specification and perception accuracy."
            )

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
        if self._llm_client is not None:
            from src.llm_interface.base import GenerateConfig
            cfg = GenerateConfig(
                temperature=temperature,
                top_p=0.9,
                max_output_tokens=1024,
                response_mime_type=response_mime_type,
            )
            response = await self._llm_client.generate_async(prompt, config=cfg)
            return response.text

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
