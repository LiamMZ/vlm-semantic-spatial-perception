"""
Hybrid Planning Architecture (spec §5)

Selects between deterministic FF-Replan and probabilistic planning based on
a lightweight risk assessment of the current scene and DKB action history.

Components
----------
ActionOutcomeStore
    Per-action success-rate accumulator.  Wraps the DKB execution_history JSONL
    and adds a lightweight in-memory outcome table that is updated after each
    execution.  Provides the ``get_success_rate(action_type, context)`` interface
    used by PlanningModeSelector.

RiskAssessment
    Dataclass returned by PlanningModeSelector.assess_risk.

PlanningModeSelector
    Decides deterministic vs probabilistic mode (spec §5.1).

AllOutcomesDeterminizer
    Expands a single stochastic action into multiple deterministic outcome-variant
    actions (spec §5.2).  The three outcome variants for displacement actions are:
      full_success  — obstruction cleared, access-clear granted (p=0.75 default)
      partial_success — object moved, obstruction reduced to improved-clearance (p=0.20)
      no_change     — action had no effect (p=0.05)

HindsightOptimizer
    Implements §5.5: sample outcome sequences, plan each deterministically, vote
    on the first action.

HybridPlanner
    Façade that owns the above components and exposes a single
    ``async plan(domain_path, problem_path, goal, registry, dkb)`` coroutine.
    Called by TaskOrchestrator.solve_and_plan_with_refinement when the mode
    selector returns 'probabilistic' or when deterministic replanning is exhausted.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (spec §5.2 default outcome probabilities for displacement actions)
# ---------------------------------------------------------------------------

_DEFAULT_OUTCOME_PROBS: Dict[str, Dict[str, float]] = {
    "displace": {
        "full_success":    0.75,
        "partial_success": 0.20,
        "no_change":       0.05,
    },
    "push-aside": {
        "full_success":    0.60,
        "partial_success": 0.30,
        "no_change":       0.10,
    },
}

# Risk thresholds from spec §5.1
_DEAD_END_RISK_THRESHOLD: float = 0.10
_OUTCOME_CRITICALITY_THRESHOLD: float = 5.0
_STATE_SPACE_DETERMINISTIC_FLOOR: int = 10_000

# Hindsight optimization defaults (spec §5.5)
_HINDSIGHT_HORIZON: int = 10
_HINDSIGHT_SAMPLES: int = 20


# ---------------------------------------------------------------------------
# ActionOutcomeStore
# ---------------------------------------------------------------------------


@dataclass
class OutcomeRecord:
    """Running tally for one (action_type, outcome_type) pair."""
    action_type: str
    outcome_type: str
    success_count: int = 0
    failure_count: int = 0

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total if self.total > 0 else 0.0


class ActionOutcomeStore:
    """Per-action outcome accumulator backed by the DKB execution history.

    On construction, replays the JSONL history to seed in-memory counts.
    The ``record`` method updates counts after each live execution so that
    subsequent ``get_success_rate`` calls reflect the latest observations.

    The store is intentionally decoupled from DKB internals — it reads the
    JSONL file directly so it can be constructed without a live DKB instance.
    """

    def __init__(self, dkb_dir: Optional[Path] = None) -> None:
        self._dkb_dir = Path(dkb_dir) if dkb_dir else Path("outputs/dkb")
        # action_type → outcome_type → OutcomeRecord
        self._table: Dict[str, Dict[str, OutcomeRecord]] = defaultdict(dict)
        self._load_from_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_success_rate(self, action_type: str, outcome_type: str = "full_success") -> float:
        """Return observed success rate for (action_type, outcome_type).

        Falls back to the spec default probability if no history exists.
        """
        record = self._table.get(action_type, {}).get(outcome_type)
        if record is None or record.total == 0:
            return _DEFAULT_OUTCOME_PROBS.get(action_type, {}).get(outcome_type, 0.5)
        return record.success_rate

    def get_outcome_probs(self, action_type: str) -> Dict[str, float]:
        """Return {outcome_type: probability} for action_type.

        Uses observed rates when available; falls back to spec defaults.
        Raw observed rates are re-normalized so they sum to 1.0.
        """
        defaults = _DEFAULT_OUTCOME_PROBS.get(action_type, {})
        outcome_types = list(defaults.keys())

        raw: Dict[str, float] = {}
        has_any_history = False
        for ot in outcome_types:
            record = self._table.get(action_type, {}).get(ot)
            if record and record.total > 0:
                raw[ot] = record.success_rate
                has_any_history = True
            else:
                raw[ot] = defaults.get(ot, 0.0)

        if not has_any_history:
            return dict(defaults)

        total = sum(raw.values())
        if total <= 0:
            return dict(defaults)
        return {ot: v / total for ot, v in raw.items()}

    def record(
        self,
        action_type: str,
        outcome_type: str,
        succeeded: bool,
    ) -> None:
        """Update outcome counts after one execution."""
        if outcome_type not in self._table[action_type]:
            self._table[action_type][outcome_type] = OutcomeRecord(
                action_type=action_type, outcome_type=outcome_type
            )
        rec = self._table[action_type][outcome_type]
        if succeeded:
            rec.success_count += 1
        else:
            rec.failure_count += 1

    # ------------------------------------------------------------------
    # History replay
    # ------------------------------------------------------------------

    def _load_from_history(self) -> None:
        history_path = self._dkb_dir / "execution_history.jsonl"
        if not history_path.exists():
            return
        loaded = 0
        try:
            with open(history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Entries written by record_execution have action_outcomes list
                    for ao in entry.get("action_outcomes", []):
                        atype = ao.get("action_type", "")
                        otype = ao.get("outcome_type", "full_success")
                        ok = bool(ao.get("succeeded", True))
                        if atype:
                            self.record(atype, otype, ok)
                            loaded += 1
        except Exception as exc:
            _log.warning("ActionOutcomeStore: history replay failed — %s", exc)
        if loaded:
            _log.debug("ActionOutcomeStore: replayed %d outcome records", loaded)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessment:
    dead_end_risk: float = 0.0
    outcome_criticality: float = 0.0
    state_space_size: int = 0

    @property
    def requires_probabilistic(self) -> bool:
        return (
            self.dead_end_risk > _DEAD_END_RISK_THRESHOLD
            or self.outcome_criticality > _OUTCOME_CRITICALITY_THRESHOLD
        )


class PlanningModeSelector:
    """Decides deterministic vs probabilistic mode (spec §5.1).

    Mode selection logic:
      - dead_end_risk > 0.1  → probabilistic
      - outcome_criticality > 5  → probabilistic
      - state_space_size > 10,000  → deterministic (too large for prob. planner)
      - otherwise  → deterministic (default)
    """

    def __init__(self, outcome_store: Optional[ActionOutcomeStore] = None) -> None:
        self._store = outcome_store

    def assess_risk(
        self,
        registry: Any,
        action_types: Optional[List[str]] = None,
    ) -> RiskAssessment:
        """Analyse the current scene and DKB statistics to produce a RiskAssessment.

        Args:
            registry: DetectedObjectRegistry snapshot.
            action_types: Action types to check for outcome criticality.
                          Defaults to displacement actions.
        """
        risk = RiskAssessment()
        action_types = action_types or ["displace", "push-aside"]

        # Dead-end risk: each cascade object contributes +0.1
        graph = getattr(registry, "contact_graph", None)
        if graph is not None:
            for obj_id, consequence in graph.removal_consequences.items():
                if consequence == "cascade":
                    risk.dead_end_risk += 0.10

        # Outcome criticality: low historical success rate on displacement actions
        if self._store is not None:
            for atype in action_types:
                sr = self._store.get_success_rate(atype, "full_success")
                if sr < 0.70:
                    risk.outcome_criticality += 1.0

        # State space estimate: O(n²) where n = object count
        n = len(registry.get_all_objects())
        risk.state_space_size = n * n

        return risk

    def select_mode(
        self,
        registry: Any,
        action_types: Optional[List[str]] = None,
    ) -> str:
        """Return ``'deterministic'`` or ``'probabilistic'``."""
        risk = self.assess_risk(registry, action_types)

        if risk.state_space_size > _STATE_SPACE_DETERMINISTIC_FLOOR:
            _log.info(
                "PlanningModeSelector: state_space=%d > %d → deterministic (size floor)",
                risk.state_space_size, _STATE_SPACE_DETERMINISTIC_FLOOR,
            )
            return "deterministic"

        if risk.requires_probabilistic:
            _log.info(
                "PlanningModeSelector: dead_end=%.2f  criticality=%.2f → probabilistic",
                risk.dead_end_risk, risk.outcome_criticality,
            )
            return "probabilistic"

        _log.info(
            "PlanningModeSelector: dead_end=%.2f  criticality=%.2f → deterministic (default)",
            risk.dead_end_risk, risk.outcome_criticality,
        )
        return "deterministic"


# ---------------------------------------------------------------------------
# All-Outcomes Determinizer (spec §5.2)
# ---------------------------------------------------------------------------


@dataclass
class DeterministicOutcomeAction:
    """One deterministic outcome variant of a stochastic action."""
    name: str            # e.g. "displace-outcome-success"
    base_action: str     # e.g. "displace"
    parameters: str      # PDDL parameter string
    precondition: str
    effect: str
    probability: float
    outcome_type: str    # "full_success" | "partial_success" | "no_change"


# Per-outcome PDDL effect templates for displacement actions.
# Each template uses {obj}, {from_surf}, {to_surf} format keys.
_DISPLACE_OUTCOME_EFFECTS: Dict[str, str] = {
    "full_success": (
        "(and "
        "(not (object-at {obj} {from_surf})) "
        "(object-at {obj} {to_surf}) "
        "(stable-on {obj} {to_surf}) "
        "(not (stable-on {obj} {from_surf})) "
        "(spatial-change-occurred) "
        "(forall (?neighbor - object) "
        "(when (obstructs {obj} ?neighbor) "
        "(and "
        "(not (obstructs {obj} ?neighbor)) "
        "(access-clear ?neighbor)))))"
    ),
    "partial_success": (
        "(and "
        "(not (object-at {obj} {from_surf})) "
        "(object-at {obj} {to_surf}) "
        "(stable-on {obj} {to_surf}) "
        "(not (stable-on {obj} {from_surf})) "
        "(spatial-change-occurred) "
        "(forall (?neighbor - object) "
        "(when (obstructs {obj} ?neighbor) "
        "(improved-clearance ?neighbor))))"
    ),
    "no_change": "(and )",
}

_PUSH_ASIDE_OUTCOME_EFFECTS: Dict[str, str] = {
    "full_success": (
        "(and "
        "(spatial-change-occurred) "
        "(forall (?neighbor - object) "
        "(when (obstructs ?obj ?neighbor) "
        "(and "
        "(not (obstructs ?obj ?neighbor)) "
        "(access-clear ?neighbor)))))"
    ),
    "partial_success": (
        "(and "
        "(spatial-change-occurred) "
        "(forall (?neighbor - object) "
        "(when (obstructs ?obj ?neighbor) "
        "(improved-clearance ?neighbor))))"
    ),
    "no_change": "(and )",
}


class AllOutcomesDeterminizer:
    """Expands stochastic displacement actions into deterministic outcome variants.

    For each action type that has outcome probabilities in the outcome store,
    generates DeterministicOutcomeAction instances for each outcome tier.
    These are appended to the PDDL domain for deterministic planning (spec §5.2).
    """

    def __init__(self, outcome_store: ActionOutcomeStore) -> None:
        self._store = outcome_store

    def expand_action(
        self,
        action_name: str,
        parameters: str,
        precondition: str,
    ) -> List[DeterministicOutcomeAction]:
        """Return deterministic outcome variants for one action.

        Args:
            action_name: Base action name (e.g. "displace").
            parameters:  PDDL parameter string (e.g. "(?obj - object ?from - surface ?to - surface)").
            precondition: PDDL precondition string for the base action.

        Returns:
            List of DeterministicOutcomeAction, one per outcome tier.
            Empty list if the action is not a displacement-type action.
        """
        probs = self._store.get_outcome_probs(action_name)
        if not probs:
            return []

        if action_name == "displace":
            templates = _DISPLACE_OUTCOME_EFFECTS
            # Extract parameter names for template substitution
            params_parsed = _parse_pddl_params(parameters)
            obj_name    = params_parsed[0] if len(params_parsed) > 0 else "?obj"
            from_name   = params_parsed[1] if len(params_parsed) > 1 else "?from"
            to_name     = params_parsed[2] if len(params_parsed) > 2 else "?to"
        elif action_name == "push-aside":
            templates = _PUSH_ASIDE_OUTCOME_EFFECTS
            obj_name = from_name = to_name = "?obj"
        else:
            return []

        outcome_suffix = {
            "full_success":    "success",
            "partial_success": "partial",
            "no_change":       "fail",
        }
        results: List[DeterministicOutcomeAction] = []
        for outcome_type, prob in probs.items():
            if outcome_type not in templates:
                continue
            effect_raw = templates[outcome_type]
            effect = effect_raw.format(
                obj=obj_name, from_surf=from_name, to_surf=to_name
            )
            suffix = outcome_suffix.get(outcome_type, outcome_type)
            results.append(DeterministicOutcomeAction(
                name=f"{action_name}-outcome-{suffix}",
                base_action=action_name,
                parameters=parameters,
                precondition=precondition,
                effect=effect,
                probability=prob,
                outcome_type=outcome_type,
            ))
        return results

    def expand_domain(
        self,
        domain_pddl: str,
    ) -> Tuple[str, Dict[str, Dict]]:
        """Inject outcome-variant actions into a PDDL domain string.

        Finds each displacement action definition in the domain, generates
        outcome variants, appends them, and returns the modified domain plus
        the action_outcome_map dict (spec §5.2).

        Args:
            domain_pddl: Raw PDDL domain text.

        Returns:
            (modified_domain_pddl, action_outcome_map)
        """
        action_outcome_map: Dict[str, Dict] = {}
        injections: List[str] = []

        for action_name in ("displace", "push-aside"):
            action_block = _extract_action_block(domain_pddl, action_name)
            if action_block is None:
                continue
            params = _extract_pddl_field(action_block, ":parameters")
            precond = _extract_pddl_field(action_block, ":precondition")
            if params is None or precond is None:
                continue

            variants = self.expand_action(action_name, params, precond)
            for v in variants:
                injections.append(_render_pddl_action(v))
                action_outcome_map[v.name] = {
                    "base_action":   v.base_action,
                    "probability":   v.probability,
                    "outcome_type":  v.outcome_type,
                }
                _log.debug(
                    "AllOutcomesDeterminizer: injected %s (p=%.2f)", v.name, v.probability
                )

        if not injections:
            return domain_pddl, action_outcome_map

        # Insert outcome-variant actions before the closing paren of the domain
        insert_point = domain_pddl.rfind(")")
        if insert_point == -1:
            modified = domain_pddl + "\n" + "\n".join(injections)
        else:
            modified = (
                domain_pddl[:insert_point]
                + "\n\n;; --- All-outcomes determinization (auto-generated) ---\n"
                + "\n".join(injections)
                + "\n"
                + domain_pddl[insert_point:]
            )

        return modified, action_outcome_map


# ---------------------------------------------------------------------------
# Hindsight Optimizer (spec §5.5)
# ---------------------------------------------------------------------------


class HindsightOptimizer:
    """Sample-based hindsight optimization for moderate-risk scenes.

    For each sample: draw a random outcome sequence of length H, build a
    non-stationary deterministic domain, plan with the FF solver, record a
    vote for the first action.  Return the most-voted first action.
    """

    def __init__(
        self,
        outcome_store: ActionOutcomeStore,
        solver_fn: Any,  # Callable[[str, str], Optional[List[str]]]
        horizon: int = _HINDSIGHT_HORIZON,
        num_samples: int = _HINDSIGHT_SAMPLES,
    ) -> None:
        self._store = outcome_store
        self._solver_fn = solver_fn
        self.horizon = horizon
        self.num_samples = num_samples

    def select_first_action(
        self,
        domain_pddl: str,
        problem_pddl: str,
        action_types: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return the most-voted first action across hindsight samples.

        Args:
            domain_pddl:  Current PDDL domain text.
            problem_pddl: Current PDDL problem text.
            action_types: Which actions to sample outcomes for.

        Returns:
            First action string, or None if no plan found in any sample.
        """
        import random
        action_types = action_types or ["displace", "push-aside"]
        votes: Dict[str, int] = defaultdict(int)

        for _ in range(self.num_samples):
            # Draw a random outcome sequence for each timestep in the horizon
            outcome_seq: List[str] = []
            for _t in range(self.horizon):
                atype = random.choice(action_types)
                probs = self._store.get_outcome_probs(atype)
                outcome_seq.append(_sample_outcome(probs))

            # Build a non-stationary deterministic domain for this sample
            det_domain = _build_hindsight_domain(domain_pddl, outcome_seq, self._store)

            # Plan and record first action vote
            try:
                plan = self._solver_fn(det_domain, problem_pddl)
            except Exception as exc:
                _log.debug("HindsightOptimizer: solver failed for sample — %s", exc)
                plan = None

            if plan:
                votes[plan[0]] += 1

        if not votes:
            return None
        best = max(votes.items(), key=lambda x: x[1])
        _log.info(
            "HindsightOptimizer: selected '%s' with %d/%d votes",
            best[0], best[1], self.num_samples,
        )
        return best[0]


# ---------------------------------------------------------------------------
# HybridPlanner — public façade
# ---------------------------------------------------------------------------


class HybridPlanner:
    """Façade for the hybrid deterministic / probabilistic planning architecture.

    Typical call sequence:
        planner = HybridPlanner(dkb_dir=Path("outputs/dkb"), solver=pddl_solver)
        result = await planner.plan(
            domain_path=Path("outputs/pddl/domain.pddl"),
            problem_path=Path("outputs/pddl/problem.pddl"),
            registry=registry,
        )

    ``plan`` returns a HybridPlanResult containing the selected mode, the plan,
    the action_outcome_map (for execution-time outcome classification), and
    the risk assessment.
    """

    def __init__(
        self,
        dkb_dir: Optional[Path] = None,
        solver: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._dkb_dir = Path(dkb_dir) if dkb_dir else Path("outputs/dkb")
        self._solver = solver
        self._log = logger or _log

        self.outcome_store = ActionOutcomeStore(self._dkb_dir)
        self.mode_selector = PlanningModeSelector(self.outcome_store)
        self.determinizer = AllOutcomesDeterminizer(self.outcome_store)

    async def plan(
        self,
        domain_path: Path,
        problem_path: Path,
        registry: Any,
        force_mode: Optional[str] = None,
    ) -> "HybridPlanResult":
        """Run hybrid planning and return a HybridPlanResult.

        Args:
            domain_path:  Path to the PDDL domain file.
            problem_path: Path to the PDDL problem file.
            registry:     Live DetectedObjectRegistry.
            force_mode:   Override mode selection ('deterministic'|'probabilistic').

        Returns:
            HybridPlanResult with plan, mode, risk, and action_outcome_map.
        """
        domain_pddl  = domain_path.read_text(encoding="utf-8")
        problem_pddl = problem_path.read_text(encoding="utf-8")

        risk = self.mode_selector.assess_risk(registry)
        mode = force_mode or self.mode_selector.select_mode(registry)

        self._log.info("HybridPlanner: mode=%s  risk=%s", mode, risk)

        if mode == "probabilistic":
            return await self._plan_probabilistic(
                domain_pddl, problem_pddl, risk, domain_path, problem_path
            )
        else:
            return await self._plan_deterministic(
                domain_pddl, problem_pddl, risk, domain_path, problem_path
            )

    def record_execution_outcome(
        self,
        action_type: str,
        outcome_type: str,
        succeeded: bool,
    ) -> None:
        """Update the outcome store after an action executes.

        Call this from the execution loop after each displacement action so the
        mode selector's risk assessment and determinizer probabilities stay
        calibrated over time.
        """
        self.outcome_store.record(action_type, outcome_type, succeeded)

    # ------------------------------------------------------------------
    # Deterministic planning path
    # ------------------------------------------------------------------

    async def _plan_deterministic(
        self,
        domain_pddl: str,
        problem_pddl: str,
        risk: RiskAssessment,
        domain_path: Path,
        problem_path: Path,
    ) -> "HybridPlanResult":
        """Expand to all-outcomes deterministic domain and solve with FF."""
        det_domain, action_outcome_map = self.determinizer.expand_domain(domain_pddl)

        # Write the expanded domain to a temp file so the solver can read it
        det_domain_path = domain_path.parent / (domain_path.stem + "_det.pddl")
        det_domain_path.write_text(det_domain, encoding="utf-8")
        self._log.info(
            "HybridPlanner(det): expanded domain written to %s  (%d outcome variants)",
            det_domain_path, len(action_outcome_map),
        )

        plan = self._run_solver(str(det_domain_path), str(problem_path))

        return HybridPlanResult(
            mode="deterministic",
            plan=plan or [],
            action_outcome_map=action_outcome_map,
            risk=risk,
            success=bool(plan),
            error=None if plan else "deterministic solver returned no plan",
        )

    # ------------------------------------------------------------------
    # Probabilistic planning path (PPDDL / hindsight fallback)
    # ------------------------------------------------------------------

    async def _plan_probabilistic(
        self,
        domain_pddl: str,
        problem_pddl: str,
        risk: RiskAssessment,
        domain_path: Path,
        problem_path: Path,
    ) -> "HybridPlanResult":
        """Generate PPDDL domain and attempt probabilistic solver; fall back to hindsight."""
        ppddl_domain, action_outcome_map = self._build_ppddl_domain(domain_pddl)

        ppddl_path = domain_path.parent / (domain_path.stem + "_ppddl.pddl")
        ppddl_path.write_text(ppddl_domain, encoding="utf-8")
        self._log.info("HybridPlanner(prob): PPDDL domain written to %s", ppddl_path)

        # Try the configured solver (it may not support PPDDL — catch gracefully)
        plan = None
        try:
            plan = self._run_solver(str(ppddl_path), str(problem_path))
        except Exception as exc:
            self._log.warning(
                "HybridPlanner(prob): probabilistic solver failed (%s) — falling back to hindsight",
                exc,
            )

        if plan:
            return HybridPlanResult(
                mode="probabilistic",
                plan=plan,
                action_outcome_map=action_outcome_map,
                risk=risk,
                success=True,
            )

        # Hindsight optimization fallback
        self._log.info("HybridPlanner(prob): using hindsight optimization")
        optimizer = HindsightOptimizer(
            outcome_store=self.outcome_store,
            solver_fn=lambda d, p: self._run_solver_from_text(d, p, domain_path.parent),
        )
        first_action = optimizer.select_first_action(domain_pddl, problem_pddl)

        if first_action:
            return HybridPlanResult(
                mode="hindsight",
                plan=[first_action],
                action_outcome_map=action_outcome_map,
                risk=risk,
                success=True,
            )

        return HybridPlanResult(
            mode="probabilistic",
            plan=[],
            action_outcome_map=action_outcome_map,
            risk=risk,
            success=False,
            error="all probabilistic planning strategies exhausted",
        )

    def _build_ppddl_domain(
        self, domain_pddl: str
    ) -> Tuple[str, Dict[str, Dict]]:
        """Inject probabilistic effects into displacement actions (spec §5.4)."""
        action_outcome_map: Dict[str, Dict] = {}
        modified = domain_pddl

        for action_name in ("displace", "push-aside"):
            action_block = _extract_action_block(domain_pddl, action_name)
            if action_block is None:
                continue
            params = _extract_pddl_field(action_block, ":parameters") or ""
            precond = _extract_pddl_field(action_block, ":precondition") or "(and)"

            probs = self.outcome_store.get_outcome_probs(action_name)
            if action_name == "displace":
                templates = _DISPLACE_OUTCOME_EFFECTS
                params_parsed = _parse_pddl_params(params)
                obj_n    = params_parsed[0] if len(params_parsed) > 0 else "?obj"
                from_n   = params_parsed[1] if len(params_parsed) > 1 else "?from"
                to_n     = params_parsed[2] if len(params_parsed) > 2 else "?to"
            else:
                templates = _PUSH_ASIDE_OUTCOME_EFFECTS
                obj_n = from_n = to_n = "?obj"

            prob_clauses: List[str] = []
            for ot in ("full_success", "partial_success", "no_change"):
                p = probs.get(ot, 0.0)
                effect_raw = templates.get(ot, "(and )")
                effect = effect_raw.format(obj=obj_n, from_surf=from_n, to_surf=to_n)
                prob_clauses.append(f"  {p:.2f} {effect}")
                action_outcome_map[f"{action_name}-{ot}"] = {
                    "base_action": action_name,
                    "probability": p,
                    "outcome_type": ot,
                }

            prob_effect = "(probabilistic\n" + "\n".join(prob_clauses) + ")"
            ppddl_action = (
                f"(:action {action_name}\n"
                f"  :parameters {params}\n"
                f"  :precondition {precond}\n"
                f"  :effect {prob_effect})"
            )
            # Replace the original action block
            modified = modified.replace(action_block.strip(), ppddl_action)

        return modified, action_outcome_map

    # ------------------------------------------------------------------
    # Solver bridge
    # ------------------------------------------------------------------

    def _run_solver(self, domain_path: str, problem_path: str) -> Optional[List[str]]:
        """Call the PDDLSolver synchronously and return the plan list."""
        if self._solver is None:
            return None
        try:
            result = self._solver.solve(
                domain_file=domain_path,
                problem_file=problem_path,
            )
            if result and result.success:
                return result.plan or []
        except Exception as exc:
            self._log.debug("HybridPlanner._run_solver: %s", exc)
        return None

    def _run_solver_from_text(
        self,
        domain_text: str,
        problem_text: str,
        tmp_dir: Path,
    ) -> Optional[List[str]]:
        """Write domain/problem to temp files and call the solver."""
        import tempfile, os
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pddl", dir=tmp_dir, delete=False, encoding="utf-8"
            ) as df:
                df.write(domain_text)
                d_path = df.name
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pddl", dir=tmp_dir, delete=False, encoding="utf-8"
            ) as pf:
                pf.write(problem_text)
                p_path = pf.name
            return self._run_solver(d_path, p_path)
        except Exception as exc:
            self._log.debug("HybridPlanner._run_solver_from_text: %s", exc)
            return None
        finally:
            for p in (d_path, p_path):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class HybridPlanResult:
    mode: str                               # 'deterministic' | 'probabilistic' | 'hindsight'
    plan: List[str]
    action_outcome_map: Dict[str, Dict]
    risk: RiskAssessment
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# PDDL parsing helpers (module-private)
# ---------------------------------------------------------------------------


def _extract_action_block(domain_pddl: str, action_name: str) -> Optional[str]:
    """Extract the full (:action <name> ...) block from a PDDL domain string.

    Uses a simple parenthesis-depth scanner so it handles nested s-expressions
    correctly without requiring a full PDDL parser.
    """
    pattern = re.compile(
        r"\(:action\s+" + re.escape(action_name) + r"\b",
        re.IGNORECASE,
    )
    m = pattern.search(domain_pddl)
    if m is None:
        return None

    start = m.start()
    depth = 0
    i = start
    while i < len(domain_pddl):
        if domain_pddl[i] == "(":
            depth += 1
        elif domain_pddl[i] == ")":
            depth -= 1
            if depth == 0:
                return domain_pddl[start:i + 1]
        i += 1
    return None


def _extract_pddl_field(action_block: str, field_name: str) -> Optional[str]:
    """Extract the s-expression value of a PDDL field (e.g. :parameters, :effect).

    Returns the full matched sub-expression including its outermost parens,
    or a bare token if the value is not parenthesised.
    """
    pattern = re.compile(re.escape(field_name) + r"\s*", re.IGNORECASE)
    m = pattern.search(action_block)
    if m is None:
        return None

    i = m.end()
    # Skip whitespace
    while i < len(action_block) and action_block[i].isspace():
        i += 1

    if i >= len(action_block):
        return None

    if action_block[i] != "(":
        # Bare token (e.g. ":precondition (and)") — read until whitespace
        end = i
        while end < len(action_block) and not action_block[end].isspace():
            end += 1
        return action_block[i:end]

    # Parenthesised expression — scan to matching close paren
    depth = 0
    start = i
    while i < len(action_block):
        if action_block[i] == "(":
            depth += 1
        elif action_block[i] == ")":
            depth -= 1
            if depth == 0:
                return action_block[start:i + 1]
        i += 1
    return None


def _parse_pddl_params(params_str: str) -> List[str]:
    """Return variable names from a PDDL parameter string.

    E.g. "(?obj - object ?from - surface ?to - surface)" → ["?obj", "?from", "?to"]
    """
    return re.findall(r"\?[a-zA-Z][a-zA-Z0-9_-]*", params_str)


def _render_pddl_action(action: DeterministicOutcomeAction) -> str:
    """Render a DeterministicOutcomeAction as a PDDL (:action ...) block."""
    return (
        f"(:action {action.name}\n"
        f"  :parameters {action.parameters}\n"
        f"  :precondition {action.precondition}\n"
        f"  :effect {action.effect})"
    )


def _sample_outcome(probs: Dict[str, float]) -> str:
    """Sample an outcome type given a probability dict."""
    import random
    r = random.random()
    cumulative = 0.0
    for outcome, p in probs.items():
        cumulative += p
        if r < cumulative:
            return outcome
    return list(probs.keys())[-1]


def _build_hindsight_domain(
    domain_pddl: str,
    outcome_sequence: List[str],
    store: ActionOutcomeStore,
) -> str:
    """Build a non-stationary deterministic domain for hindsight planning.

    For each timestep t in the sequence, the outcome at that step is
    encoded as the sole available deterministic variant.  Since classical
    PDDL cannot express per-timestep branching directly, we use the
    all-outcomes determinization but keep only the sampled outcome variant
    for each action — the planner will use whichever outcome action is
    available at each step.

    In practice this produces a deterministic domain with multiple outcome
    variants available, allowing the planner to choose the expected path.
    This is a practical approximation of the non-stationary hindsight domain.
    """
    determinizer = AllOutcomesDeterminizer(store)
    det_domain, _ = determinizer.expand_domain(domain_pddl)
    return det_domain
