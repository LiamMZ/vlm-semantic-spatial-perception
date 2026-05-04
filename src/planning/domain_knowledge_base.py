"""
Domain Knowledge Base (DKB)

Pre-loaded domain modules
-------------------------
The clutter manipulation module is loaded automatically on ``load()`` via
``load_clutter_module()``.  It installs the predicate library entries for:
    access-clear, obstructs, displaceable, space-available,
    supports, removal-safe, stable-on, visible, occluded-by
These are available to L2 via ``get_predicate_hints()`` without any LLM call
generating them from scratch.


Lightweight JSON-based persistence for predicates, action schemas, and execution
history that accumulates across tasks. Implements the DKB architecture described
in gtamp-update-plan.md §V.

Files written:
    <dkb_dir>/predicate_library.json     — known predicate signatures by name
    <dkb_dir>/action_schema_library.json — known action schemas by name
    <dkb_dir>/execution_history.jsonl    — append-only log of all pipeline runs
    <dkb_dir>/domain_errors.jsonl        — append-only log of domain generation failures

Usage:
    dkb = DomainKnowledgeBase(Path("outputs/dkb"))
    dkb.load()

    hints = dkb.get_predicate_hints("pick up the red block")
    dkb.record_execution("pick up the red block", artifact, solver_result)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class DomainKnowledgeBase:
    """
    Persistent store for domain knowledge accumulated across tasks.

    Predicate and action libraries grow monotonically — new entries are
    appended, existing entries are updated. Execution history is append-only.
    """

    def __init__(self, dkb_dir: Optional[Path] = None) -> None:
        self.dkb_dir = Path(dkb_dir) if dkb_dir else Path("outputs/dkb")
        self._predicates: Dict[str, Dict] = {}   # name → {signature, usage_count, ...}
        self._actions: Dict[str, Dict] = {}      # name → {parameters, precondition, effect, ...}

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing library files from disk, then pre-load domain modules."""
        self.dkb_dir.mkdir(parents=True, exist_ok=True)

        pred_path = self.dkb_dir / "predicate_library.json"
        if pred_path.exists():
            try:
                self._predicates = json.loads(pred_path.read_text(encoding="utf-8"))
            except Exception:
                self._predicates = {}

        action_path = self.dkb_dir / "action_schema_library.json"
        if action_path.exists():
            try:
                self._actions = json.loads(action_path.read_text(encoding="utf-8"))
            except Exception:
                self._actions = {}

        # Pre-load built-in domain modules (idempotent — preserves usage_count)
        from .clutter_module import load_clutter_module
        load_clutter_module(self)

    def save(self) -> None:
        """Persist predicate and action libraries to disk."""
        self.dkb_dir.mkdir(parents=True, exist_ok=True)
        (self.dkb_dir / "predicate_library.json").write_text(
            json.dumps(self._predicates, indent=2), encoding="utf-8"
        )
        (self.dkb_dir / "action_schema_library.json").write_text(
            json.dumps(self._actions, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_predicate_hints(self, task: str) -> List[Dict]:
        """
        Return predicate hint dicts from the library that may be relevant to `task`.

        Entries from a built-in domain module (e.g. clutter) are always included —
        they are infrastructure predicates, not task-specific. Task-word matches and
        usage_count boost ordering for remaining entries.
        Returns at most 20 entries.
        """
        task_lower = task.lower()
        scored: List[tuple] = []
        for name, entry in self._predicates.items():
            score = entry.get("usage_count", 0)
            # Always include domain-module entries (clutter infrastructure etc.)
            if entry.get("domain_module"):
                score += 100
            if any(word in task_lower for word in name.replace("-", " ").split()):
                score += 10
            scored.append((score, {
                "signature": entry.get("signature", f"({name})"),
                "type_classification": entry.get("type_classification", ""),
                "grounding_rule": entry.get("grounding_rule", ""),
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [hint for _, hint in scored[:20]]

    def get_action_hints(self, task: str) -> List[Dict]:
        """
        Return action schemas from the library relevant to `task`.
        Domain-module actions (e.g. clutter) are always included.
        Returns at most 10 entries.
        """
        import re as _re
        task_lower = task.lower()
        scored: List[tuple] = []
        for name, entry in self._actions.items():
            score = entry.get("usage_count", 0)
            if entry.get("domain_module"):
                score += 100
            if any(word in task_lower for word in name.replace("-", " ").split()):
                score += 10
            schema = {k: v for k, v in entry.items() if k != "usage_count"}
            schema["name"] = name
            # Strip stale precondition atoms that should never appear in primary actions:
            # (checked-improved-clearance ...) — contingency only, not a universal pick gate
            # (spatial-change-occurred) — runtime effect flag, never a precondition
            if "precondition" in schema:
                precond = schema["precondition"]
                precond = _re.sub(r"\(checked-improved-clearance\s+[^)]*\)\s*", "", precond)
                precond = _re.sub(r"\(spatial-change-occurred\)\s*", "", precond)
                schema["precondition"] = precond.strip()
            scored.append((score, schema))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [schema for _, schema in scored[:10]]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        task: str,
        artifact: Any,
        solver_result: Optional[Any] = None,
    ) -> None:
        """
        Record a pipeline execution to the execution history and update libraries.

        Args:
            task: Natural language task description.
            artifact: LayeredDomainArtifact from the generator.
            solver_result: Optional SolverResult (may be None if solver not yet run).
        """
        self.dkb_dir.mkdir(parents=True, exist_ok=True)

        # Update predicate library
        try:
            for sig in getattr(artifact.l2, "predicate_signatures", []):
                name = _name_from_sig(sig)
                if name:
                    entry = self._predicates.get(name, {"signature": sig, "usage_count": 0})
                    entry["usage_count"] = entry.get("usage_count", 0) + 1
                    entry["last_task"] = task
                    self._predicates[name] = entry
        except Exception:
            pass

        # Update action schema library
        try:
            import re as _re
            for action in getattr(artifact.l3, "actions", []) + getattr(artifact.l3, "sensing_actions", []):
                name = action.get("name", "")
                if name:
                    precond = action.get("precondition", "")
                    # Strip atoms that should never persist in primary action preconditions
                    precond = _re.sub(r"\(checked-improved-clearance\s+[^)]*\)\s*", "", precond)
                    precond = _re.sub(r"\(spatial-change-occurred\)\s*", "", precond)
                    entry = self._actions.get(name, {})
                    entry.update({
                        "name": name,
                        "parameters": action.get("parameters", []),
                        "precondition": precond.strip(),
                        "effect": action.get("effect", ""),
                        "usage_count": entry.get("usage_count", 0) + 1,
                        "last_task": task,
                    })
                    self._actions[name] = entry
        except Exception:
            pass

        self.save()

        # Append to execution history (JSONL)
        history_path = self.dkb_dir / "execution_history.jsonl"
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
        }
        try:
            entry["l1_goal_count"] = len(getattr(artifact.l1, "goal_predicates", []))
            entry["l2_predicate_count"] = len(getattr(artifact.l2, "predicate_signatures", []))
            entry["l3_action_count"] = len(getattr(artifact.l3, "actions", []))
            entry["l1_validation_errors"] = getattr(artifact.l1, "validation_errors", [])
            entry["l2_validation_errors"] = getattr(artifact.l2, "validation_errors", [])
            entry["l3_validation_errors"] = getattr(artifact.l3, "validation_errors", [])
        except Exception:
            pass
        if solver_result is not None:
            try:
                entry["solver_success"] = bool(getattr(solver_result, "success", False))
                entry["plan_length"] = getattr(solver_result, "plan_length", None)
                entry["solver_error"] = getattr(solver_result, "error_message", None)
            except Exception:
                pass

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Mirror validation errors and L4 warnings into domain_errors.jsonl
        try:
            for layer, attr in (("L1", "l1"), ("L2", "l2"), ("L3", "l3")):
                errs = entry.get(f"{attr}_validation_errors") or []
                for err in errs:
                    if err:
                        self.record_domain_error(
                            error_type=f"validation_{layer.lower()}",
                            message=str(err),
                            task=task,
                            layer=layer,
                        )
            l4_warnings = getattr(getattr(artifact, "l4", None), "warnings", None) or []
            for w in l4_warnings:
                if w:
                    self.record_domain_error(
                        error_type="l4_warning",
                        message=str(w),
                        task=task,
                        layer="L4",
                    )
        except Exception:
            pass

    def record_domain_error(
        self,
        error_type: str,
        message: str,
        task: Optional[str] = None,
        layer: Optional[str] = None,
        run_id: Optional[str] = None,
        refinement_attempt: Optional[int] = None,
        pddl_snippet: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a domain-generation failure to domain_errors.jsonl.

        Args:
            error_type:          Short classifier: "solver_no_plan", "solver_parse_error",
                                 "validation_l1" … "validation_l5", "refinement_limit",
                                 "l4_warning", "hybrid_planner_failed", etc.
            message:             Full error/warning message string.
            task:                Natural-language task description (optional).
            layer:               Pipeline layer where failure occurred, e.g. "L3", "solver".
            run_id:              Run identifier (timestamp string or UUID).
            refinement_attempt:  Which refinement iteration produced this error (0-based).
            pddl_snippet:        Relevant excerpt from domain/problem PDDL (optional, ≤2 KB).
            extra:               Any additional key-value context.
        """
        self.dkb_dir.mkdir(parents=True, exist_ok=True)
        error_path = self.dkb_dir / "domain_errors.jsonl"
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "message": message,
        }
        if run_id is not None:
            record["run_id"] = run_id
        if task is not None:
            record["task"] = task
        if layer is not None:
            record["layer"] = layer
        if refinement_attempt is not None:
            record["refinement_attempt"] = refinement_attempt
        if pddl_snippet is not None:
            # Truncate very long snippets to keep the log readable
            record["pddl_snippet"] = pddl_snippet[:2048]
        if extra:
            record.update({k: v for k, v in extra.items() if k not in record})
        with open(error_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


    def get_domain_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent *limit* records from domain_errors.jsonl."""
        error_path = self.dkb_dir / "domain_errors.jsonl"
        if not error_path.exists():
            return []
        records = []
        try:
            with open(error_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return records[-limit:]

    def error_summary(self) -> Dict[str, Any]:
        """
        Return a compact summary of all errors in domain_errors.jsonl:
        total count, counts by error_type, and the 5 most recent messages.
        """
        records = self.get_domain_errors(limit=1000)
        if not records:
            return {"total": 0, "by_type": {}, "recent": []}
        by_type: Dict[str, int] = {}
        for r in records:
            by_type[r.get("error_type", "unknown")] = by_type.get(r.get("error_type", "unknown"), 0) + 1
        recent = [
            {"timestamp": r.get("timestamp", ""), "type": r.get("error_type", ""), "message": r.get("message", "")[:120]}
            for r in records[-5:]
        ]
        return {"total": len(records), "by_type": by_type, "recent": recent}


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _name_from_sig(sig: str) -> Optional[str]:
    """Extract predicate name from a signature like '(on ?obj ?surface)'."""
    import re
    m = re.match(r"^\(?\s*([a-zA-Z][a-zA-Z0-9_-]*)", sig.strip())
    return m.group(1).lower() if m else None
