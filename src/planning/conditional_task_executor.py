"""
ConditionalTaskExecutor

Executes a symbolic PDDL plan with runtime contingency branching.

Classical PDDL produces a flat action sequence that assumes all sensing actions
succeed. This executor wraps that sequence and handles the case where a
``check-*`` sensing action reveals a predicate is FALSE:

  check-access-clear(?obj)
    → TRUE:  continue to next step
    → FALSE: run recovery branch (displace a blocker, push-aside, or replan)

  check-visible(?obj) / check-removal-safe(?obj) / check-displaceable(?obj)
    → TRUE:  continue
    → FALSE: replan from current world state

The executor calls ``ClutterGrounder`` at runtime using the live registry, so
its decisions are always based on the current perceived world state — not the
state assumed at planning time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    action: str
    success: bool
    predicate_value: Optional[bool] = None   # None for non-check actions
    recovery_triggered: bool = False
    recovery_action: Optional[str] = None
    notes: str = ""


@dataclass
class ConditionalExecutionResult:
    success: bool
    steps: List[StepResult] = field(default_factory=list)
    final_plan: List[str] = field(default_factory=list)
    replan_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Recovery policy
# ---------------------------------------------------------------------------

# Maps predicate name → recovery strategy
# "replan"   — call the replan_fn callback (orchestrator re-solves)
# "displace" — find a displaceable blocker and displace it, then retry
# "push"     — push the blocking object aside, then retry
_RECOVERY_POLICY: Dict[str, str] = {
    "access-clear":       "displace",
    "object-graspable":   "displace",
    "graspable":          "displace",
    "obstructs":          "displace",
    "visible":            "replan",
    "removal-safe":       "replan",
    "displaceable":       "replan",
    "space-available":    "replan",
    "supports":           "replan",
    "occluded-by":        "replan",
    "improved-clearance": "replan",
}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ConditionalTaskExecutor:
    """
    Execute a PDDL plan with runtime contingency branching on check-* steps.

    Args:
        registry:              DetectedObjectRegistry for live predicate evaluation.
        execute_action:        Callable(action_str) → bool that runs a non-check action.
        replan_fn:             Async callable() → List[str] that re-solves and returns a
                               new plan.  Called when a recovery strategy is "replan".
        max_replan:            Maximum number of replanning attempts before aborting.
        post_displacement_hook: Optional PostDisplacementHook.  When provided, it is
                               invoked after every displace/push-aside action to
                               re-ground clearance for affected neighbors and validate
                               the remaining plan.  If the hook signals the plan is no
                               longer valid, the executor triggers replanning.
        logger:                Optional logger.
    """

    def __init__(
        self,
        registry: Any,
        execute_action: Callable[[str], bool],
        replan_fn: Optional[Callable[[], Any]] = None,
        max_replan: int = 2,
        post_displacement_hook: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._registry = registry
        self._execute_action = execute_action
        self._replan_fn = replan_fn
        self._max_replan = max_replan
        self._post_displacement_hook = post_displacement_hook
        self.logger = logger or _log

    async def execute(self, plan: List[str]) -> ConditionalExecutionResult:
        """Execute *plan* with runtime branching. Returns a ConditionalExecutionResult."""
        from .clutter_module import ClutterGrounder, SymbolicState
        grounder = ClutterGrounder(self._registry)
        sym_state = SymbolicState()

        steps: List[StepResult] = []
        replan_count = 0
        current_plan = list(plan)
        step_idx = 0

        while step_idx < len(current_plan):
            action_str = current_plan[step_idx]
            action_name, args = _parse_action(action_str)

            # ── Check-* sensing step ──────────────────────────────────────
            if action_name.startswith("check-"):
                pred_name = action_name[len("check-"):]
                truth, details = self._evaluate_predicate(grounder, pred_name, args)

                self.logger.info(
                    "[ConditionalExecutor] %s → %s  (%s)",
                    action_str, "TRUE" if truth else "FALSE", details
                )

                if truth:
                    steps.append(StepResult(
                        action=action_str, success=True,
                        predicate_value=True, notes=details
                    ))
                    step_idx += 1
                    continue

                # Predicate is FALSE — execute recovery branch
                strategy = _RECOVERY_POLICY.get(pred_name, "replan")
                self.logger.warning(
                    "[ConditionalExecutor] %s is FALSE — recovery strategy: %s",
                    pred_name, strategy
                )

                recovery_action: Optional[str] = None
                recovery_ok = False

                if strategy == "displace" and len(args) >= 1:
                    recovery_action, recovery_ok = self._recovery_displace(
                        grounder, args[0] if len(args) == 1 else args[1]
                    )
                    if not recovery_ok:
                        # Displace failed — fall back to replan
                        strategy = "replan"

                if strategy == "replan":
                    if self._replan_fn is None or replan_count >= self._max_replan:
                        self.logger.error(
                            "[ConditionalExecutor] Replan unavailable or limit reached (%d/%d)",
                            replan_count, self._max_replan
                        )
                        steps.append(StepResult(
                            action=action_str, success=False,
                            predicate_value=False,
                            recovery_triggered=True,
                            notes=f"{pred_name} FALSE, replan limit reached"
                        ))
                        return ConditionalExecutionResult(
                            success=False, steps=steps,
                            final_plan=current_plan, replan_count=replan_count,
                            error=f"Predicate {pred_name} FALSE and cannot replan"
                        )

                    self.logger.info("[ConditionalExecutor] Replanning...")
                    new_plan = await self._replan_fn()
                    replan_count += 1
                    if not new_plan:
                        steps.append(StepResult(
                            action=action_str, success=False,
                            predicate_value=False,
                            recovery_triggered=True,
                            notes=f"{pred_name} FALSE, replan returned empty plan"
                        ))
                        return ConditionalExecutionResult(
                            success=False, steps=steps,
                            final_plan=current_plan, replan_count=replan_count,
                            error=f"Replan failed after {pred_name} FALSE"
                        )

                    self.logger.info(
                        "[ConditionalExecutor] New plan (%d steps): %s",
                        len(new_plan), new_plan
                    )
                    current_plan = new_plan
                    step_idx = 0
                    steps.append(StepResult(
                        action=action_str, success=True,
                        predicate_value=False,
                        recovery_triggered=True,
                        recovery_action="replan",
                        notes=f"{pred_name} FALSE → replanned ({len(new_plan)} steps)"
                    ))
                    continue

                steps.append(StepResult(
                    action=action_str, success=recovery_ok,
                    predicate_value=False,
                    recovery_triggered=True,
                    recovery_action=recovery_action,
                    notes=f"{pred_name} FALSE → {strategy}"
                ))
                step_idx += 1
                continue

            # ── Regular action step ───────────────────────────────────────
            self.logger.info("[ConditionalExecutor] Executing: %s", action_str)
            try:
                ok = self._execute_action(action_str)
            except Exception as exc:
                self.logger.error("[ConditionalExecutor] Action failed: %s — %s", action_str, exc)
                ok = False

            steps.append(StepResult(action=action_str, success=ok))
            if not ok:
                return ConditionalExecutionResult(
                    success=False, steps=steps,
                    final_plan=current_plan, replan_count=replan_count,
                    error=f"Action failed: {action_str}"
                )

            # ── Post-displacement hook ────────────────────────────────────
            if ok and action_name in ("displace", "push-aside") and self._post_displacement_hook is not None:
                displaced_obj = args[0] if args else ""
                remaining = current_plan[step_idx + 1:]
                plan_valid = self._post_displacement_hook.run(
                    displaced_obj_id=displaced_obj,
                    symbolic_state=sym_state,
                    remaining_plan=remaining,
                )
                if not plan_valid:
                    self.logger.info(
                        "[ConditionalExecutor] Post-displacement hook: plan invalidated after %s — replanning",
                        action_str,
                    )
                    if self._replan_fn is not None and replan_count < self._max_replan:
                        new_plan = await self._replan_fn()
                        replan_count += 1
                        if new_plan:
                            self.logger.info(
                                "[ConditionalExecutor] Replanned after displacement (%d steps)",
                                len(new_plan),
                            )
                            steps.append(StepResult(
                                action=action_str, success=True,
                                recovery_triggered=True,
                                recovery_action="replan",
                                notes="post-displacement replan",
                            ))
                            current_plan = new_plan
                            step_idx = 0
                            continue

            step_idx += 1

        return ConditionalExecutionResult(
            success=True, steps=steps,
            final_plan=current_plan, replan_count=replan_count
        )

    # ------------------------------------------------------------------
    # Predicate evaluation
    # ------------------------------------------------------------------

    def _evaluate_predicate(
        self,
        grounder: Any,
        pred_name: str,
        args: List[str],
    ) -> Tuple[bool, str]:
        """Evaluate pred_name(args) via ClutterGrounder. Returns (truth, detail_str)."""
        try:
            if pred_name in ("access-clear", "object-graspable", "graspable") and len(args) >= 1:
                v = grounder.evaluate_access_clear(args[0])
                return v, f"graspability + corridors for {args[0]}"
            if pred_name == "obstructs" and len(args) >= 2:
                v = grounder.evaluate_obstructs(args[0], args[1])
                return v, f"{args[0]} blocks {args[1]}"
            if pred_name == "displaceable" and len(args) >= 1:
                v = grounder.evaluate_displaceable(args[0])
                return v, f"stability + cascade check for {args[0]}"
            if pred_name == "removal-safe" and len(args) >= 1:
                v = grounder.evaluate_removal_safe(args[0])
                return v, f"removal consequence for {args[0]}"
            if pred_name == "visible" and len(args) >= 1:
                v = grounder.evaluate_visible(args[0])
                return v, f"visible fraction for {args[0]}"
            if pred_name == "space-available" and len(args) >= 2:
                v = grounder.evaluate_space_available(args[0], args[1])
                return v, f"free space on {args[0]} for {args[1]}"
            if pred_name == "supports" and len(args) >= 2:
                v = grounder.evaluate_supports(args[0], args[1])
                return v, f"{args[0]} supports {args[1]}"
            if pred_name == "occluded-by" and len(args) >= 2:
                v = grounder.evaluate_occluded_by(args[0], args[1])
                return v, f"{args[1]} occludes {args[0]}"
            if pred_name == "improved-clearance" and len(args) >= 1:
                v = grounder.evaluate_improved_clearance(args[0])
                return v, f"graspability in (0.3, 0.5) and improving for {args[0]}"
            # Unknown predicate — assume TRUE (don't block execution)
            return True, f"unknown predicate '{pred_name}' — assuming TRUE"
        except Exception as exc:
            self.logger.warning("Predicate eval error for %s%s: %s", pred_name, args, exc)
            return True, f"eval error — assuming TRUE"

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def _recovery_displace(
        self, grounder: Any, target_obj_id: str
    ) -> Tuple[Optional[str], bool]:
        """
        Find a displaceable, removal-safe blocker of *target_obj_id* and
        execute a symbolic displace action.  Returns (action_str, success).
        """
        all_objects = self._registry.get_all_objects()
        for blocker in all_objects:
            bid = blocker.object_id
            if bid == target_obj_id:
                continue
            if not grounder.evaluate_obstructs(bid, target_obj_id):
                continue
            if not grounder.evaluate_displaceable(bid):
                continue
            if not grounder.evaluate_removal_safe(bid):
                continue
            # Find a surface with space
            surfaces = [
                o for o in all_objects
                if o.object_type.lower() in (
                    "surface", "table", "shelf", "tray", "counter", "desk"
                )
            ]
            for surf in surfaces:
                if grounder.evaluate_space_available(surf.object_id, bid):
                    action_str = f"(displace {bid} ? {surf.object_id})"
                    self.logger.info(
                        "[ConditionalExecutor] Recovery: %s", action_str
                    )
                    try:
                        ok = self._execute_action(action_str)
                        return action_str, ok
                    except Exception as exc:
                        self.logger.warning("Recovery displace failed: %s", exc)
                        return action_str, False
        return None, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(action_str: str) -> Tuple[str, List[str]]:
    """Parse ``(action-name arg1 arg2 ...)`` → (name, [arg1, arg2, ...])."""
    s = action_str.strip().strip("()")
    tokens = s.split()
    if not tokens:
        return action_str, []
    return tokens[0].lower(), tokens[1:]
