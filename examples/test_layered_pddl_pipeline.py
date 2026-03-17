"""
Layered PDDL Pipeline — Integration Tests

Tests each layer of the LayeredDomainGenerator individually and as a full
end-to-end pipeline. All tests make real Gemini API calls and use a real
PDDL solver (pyperplan). No mock data.

Scene fixture: two blocks + table (blocksworld)
Task: "Put the red block on the blue block"

Usage:
    export GEMINI_API_KEY="..."
    uv run pytest examples/test_layered_pddl_pipeline.py -v
    uv run pytest examples/test_layered_pddl_pipeline.py::test_l1_goal_extraction -v
"""

import json
import os
import re
import sys
import asyncio
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

config_path = PROJECT_ROOT / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.planning.layered_domain_generator import (
    LayeredDomainGenerator,
    bfs_reachable_predicates,
    extract_predicate_name_from_literal,
    extract_predicates_from_formula,
)
from src.planning.domain_knowledge_base import DomainKnowledgeBase
from src.planning.utils.task_types import (
    L1GoalArtifact,
    L2PredicateArtifact,
    L3ActionArtifact,
    LayeredDomainArtifact,
    TaskAnalysis,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TASK = "Put the red block on the blue block"

SCENE_OBJECTS = [
    {
        "object_id": "red_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.3, 0.0, 0.1],
    },
    {
        "object_id": "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.5, 0.0, 0.1],
    },
    {
        "object_id": "table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.4, 0.0, 0.0],
    },
]


def _make_generator() -> LayeredDomainGenerator:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assert api_key, "GEMINI_API_KEY environment variable not set"
    return LayeredDomainGenerator(api_key=api_key, max_layer_retries=2)


def _skip_if_no_api():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")


# ---------------------------------------------------------------------------
# L1 tests
# ---------------------------------------------------------------------------

async def test_l1_goal_extraction():
    """L1 extracts grounded goal predicates from task + scene."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)

    print(f"\n[L1] goal_predicates: {l1.goal_predicates}")
    print(f"[L1] goal_objects: {l1.goal_objects}")
    print(f"[L1] global_predicates: {l1.global_predicates}")

    assert len(l1.goal_predicates) > 0, "L1 must produce at least one goal predicate"

    # Each goal predicate must be a grounded PDDL literal
    for gp in l1.goal_predicates:
        assert gp.startswith("(") and gp.endswith(")"), f"Goal predicate not a PDDL literal: {gp}"
        assert "?" not in gp, f"Goal predicate must be grounded (no ?-variables): {gp}"

    # Goal must reference one or both blocks
    goal_text = " ".join(l1.goal_predicates)
    assert "red_block_1" in goal_text or "blue_block_1" in goal_text, (
        f"Goal predicates don't mention either block: {l1.goal_predicates}"
    )

    # Validation must pass
    errors = generator._validate_l1(l1, SCENE_OBJECTS)
    assert errors == [], f"L1 validation errors: {errors}"


async def test_l1_empty_scene_records_errors():
    """L1 validation flags when goal objects aren't in an empty scene."""
    _skip_if_no_api()
    generator = _make_generator()

    # Run L1 against empty scene — validation should warn
    l1 = await generator._run_l1(TASK, scene_objects=[])

    print(f"\n[L1 empty scene] goal_predicates: {l1.goal_predicates}")

    # Validation should produce errors (can't ground against empty scene)
    errors = generator._validate_l1(l1, scene_objects=[])
    # Errors may occur if the LLM fabricated object IDs, but may not if it returned empty
    # Either way the artifact must be returned (not raise)
    print(f"[L1 empty scene] validation errors: {errors}")
    # The important check: no unhandled exception was raised


# ---------------------------------------------------------------------------
# L2 tests
# ---------------------------------------------------------------------------

async def test_l2_predicate_vocab():
    """L2 produces a minimal predicate vocabulary covering L1 goals."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)

    print(f"\n[L2] predicate_signatures: {l2.predicate_signatures}")
    print(f"[L2] sensed_predicates: {l2.sensed_predicates}")

    assert len(l2.predicate_signatures) > 0, "L2 must produce at least one predicate"

    # All signatures should be PDDL-formatted
    for sig in l2.predicate_signatures:
        name = extract_predicate_name_from_literal(sig)
        assert name is not None, f"Cannot parse predicate name from: {sig}"

    # Must contain goal predicate names from L1
    defined_names = {extract_predicate_name_from_literal(s) for s in l2.predicate_signatures}
    for gp in l1.goal_predicates:
        goal_name = extract_predicate_name_from_literal(gp)
        assert goal_name in defined_names, (
            f"Goal predicate '{goal_name}' from L1 not defined in L2 vocabulary. "
            f"Defined: {defined_names}"
        )

    # Validation must pass
    errors = generator._validate_l2(l2, l1)
    assert errors == [], f"L2 validation errors: {errors}"


async def test_l2_checked_variant_autorepair():
    """Auto-repair generates checked-X variants for sensed predicates."""
    generator = LayeredDomainGenerator.__new__(LayeredDomainGenerator)
    generator.dkb = None

    # Craft an L2 artifact with a sensed predicate but no checked variant
    l2 = L2PredicateArtifact(
        predicate_signatures=["(on ?obj ?surface)", "(holding ?obj)", "(graspable ?obj)"],
        sensed_predicates=["graspable"],
        checked_variants=[],
    )
    repaired = generator._repair_l2_auto(l2)

    print(f"\n[L2 repair] predicate_signatures after repair: {repaired.predicate_signatures}")
    print(f"[L2 repair] checked_variants: {repaired.checked_variants}")

    checked_names = {
        extract_predicate_name_from_literal(s) for s in repaired.predicate_signatures
    }
    assert "checked-graspable" in checked_names, (
        f"checked-graspable not auto-generated. Signatures: {repaired.predicate_signatures}"
    )
    assert any("checked-graspable" in cv for cv in repaired.checked_variants)


# ---------------------------------------------------------------------------
# L3 tests
# ---------------------------------------------------------------------------

async def test_l3_action_schemas():
    """L3 produces action schemas with symbolic closure against L2 vocabulary."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = await generator._run_l3(TASK, l1, l2)

    print(f"\n[L3] actions: {[a.get('name') for a in l3.actions]}")
    print(f"[L3] sensing_actions: {[a.get('name') for a in l3.sensing_actions]}")

    assert len(l3.actions) > 0, "L3 must produce at least one action"

    # Check symbolic closure manually
    vocab = {extract_predicate_name_from_literal(s) for s in l2.predicate_signatures}
    for action in l3.actions:
        name = action.get("name", "<unnamed>")
        for formula in [action.get("precondition", ""), action.get("effect", "")]:
            used = extract_predicates_from_formula(formula)
            undefined = used - vocab
            assert not undefined, (
                f"Action '{name}' uses undefined predicates: {undefined}. Vocab: {vocab}"
            )

    # Validation should pass or errors should be only auto-repaired warnings
    errors = generator._validate_l3(l3, l2, l1)
    assert errors == [], f"L3 validation errors: {errors}"


async def test_l3_goal_achievability():
    """BFS reachability confirms all L1 goal predicates are achievable."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = await generator._run_l3(TASK, l1, l2)

    all_actions = l3.actions + l3.sensing_actions

    # Build initial set: global predicates + all L2 predicate names (as potential initial facts)
    global_preds = {p.strip("()").split()[0] for p in l1.global_predicates}
    vocab = {extract_predicate_name_from_literal(s) or "" for s in l2.predicate_signatures}
    reachable = bfs_reachable_predicates(all_actions, global_preds | vocab)

    print(f"\n[L3 reachability] reachable predicates: {sorted(reachable)}")

    for gp in l1.goal_predicates:
        name = extract_predicate_name_from_literal(gp)
        assert name in reachable, (
            f"Goal predicate '{name}' not reachable from actions. "
            f"Reachable: {sorted(reachable)}"
        )


# ---------------------------------------------------------------------------
# L4 tests
# ---------------------------------------------------------------------------

async def test_l4_grounding_precheck():
    """L4 algorithmic pre-check runs without blocking errors for valid scene."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = await generator._run_l3(TASK, l1, l2)
    l4 = generator._run_l4_precheck(l1, l3, SCENE_OBJECTS)

    print(f"\n[L4] object_bindings: {l4.object_bindings}")
    print(f"[L4] warnings: {l4.warnings}")

    # L4 is warn-only — no hard failures for a valid scene
    assert isinstance(l4.object_bindings, dict)
    assert isinstance(l4.warnings, list)


# ---------------------------------------------------------------------------
# L5 tests
# ---------------------------------------------------------------------------

async def test_l5_initial_state():
    """L5 algorithmic state construction produces correct initial facts."""
    _skip_if_no_api()
    generator = _make_generator()

    l1 = await generator._run_l1(TASK, SCENE_OBJECTS)
    l2 = await generator._run_l2(TASK, l1, SCENE_OBJECTS)
    l3 = await generator._run_l3(TASK, l1, l2)
    l5 = generator._run_l5(l2, l3, SCENE_OBJECTS, l1)

    print(f"\n[L5] true_literals ({len(l5.true_literals)}): {l5.true_literals[:5]}...")
    print(f"[L5] false_literals ({len(l5.false_literals)}): {l5.false_literals[:5]}...")

    # Must produce some true literals (global predicates + spatial)
    assert len(l5.true_literals) > 0, "L5 produced no true literals"

    # All checked-* predicates must appear only in false_literals
    checked_in_true = [
        (pred, args) for (pred, args) in l5.true_literals if pred.startswith("checked-")
    ]
    assert checked_in_true == [], (
        f"checked-* predicates initialized to TRUE (must be FALSE): {checked_in_true}"
    )

    # No literal should reference an object_id not in the scene
    known_ids = {o["object_id"] for o in SCENE_OBJECTS}
    for pred, args in l5.true_literals + l5.false_literals:
        for arg in args:
            assert arg in known_ids, (
                f"Literal ({pred} {args}) references unknown object '{arg}'. "
                f"Known: {known_ids}"
            )


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

async def test_full_pipeline_generates_layered_artifact():
    """Full generate_domain() call produces a complete LayeredDomainArtifact."""
    _skip_if_no_api()
    generator = _make_generator()

    artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    print(f"\n[Full] L1 goals: {artifact.l1.goal_predicates}")
    print(f"[Full] L2 predicates: {artifact.l2.predicate_signatures}")
    print(f"[Full] L3 actions: {[a.get('name') for a in artifact.l3.actions]}")

    assert isinstance(artifact, LayeredDomainArtifact)
    assert len(artifact.l1.goal_predicates) > 0
    assert len(artifact.l2.predicate_signatures) > 0
    assert len(artifact.l3.actions) > 0
    assert artifact.l4 is not None
    assert artifact.l5 is not None


async def test_backward_compat_to_task_analysis():
    """LayeredDomainArtifact.to_task_analysis() produces a valid TaskAnalysis."""
    _skip_if_no_api()
    generator = _make_generator()

    artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)
    ta = artifact.to_task_analysis()

    print(f"\n[Compat] TaskAnalysis.goal_predicates: {ta.goal_predicates}")
    print(f"[Compat] TaskAnalysis.required_actions: {[a.get('name') for a in ta.required_actions]}")

    assert isinstance(ta, TaskAnalysis)
    assert len(ta.goal_predicates) > 0
    assert len(ta.relevant_predicates) > 0
    assert len(ta.required_actions) > 0


async def test_full_pipeline_solves_blocksworld():
    """Full pipeline from NL task → PDDL domain → solver produces a valid plan."""
    _skip_if_no_api()
    from src.planning.pddl_representation import PDDLRepresentation
    from src.planning.pddl_domain_maintainer import PDDLDomainMaintainer
    from src.planning.pddl_solver import PDDLSolver, SolverBackend
    from src.perception.object_registry import DetectedObject
    import numpy as np

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    generator = LayeredDomainGenerator(api_key=api_key, max_layer_retries=2)

    # Generate layered domain
    artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)
    print(f"\n[E2E] L3 actions: {[a.get('name') for a in artifact.l3.actions]}")

    # Feed into PDDLDomainMaintainer via bridge
    pddl = PDDLRepresentation(domain_name="blocksworld", problem_name="test")
    maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)
    ta = await maintainer.initialize_from_layered_artifact(artifact)

    print(f"[E2E] TaskAnalysis.goal_predicates: {ta.goal_predicates}")

    # Add objects and goal to problem
    objects_dict = [
        {
            "object_id": o["object_id"],
            "object_type": o["object_type"],
            "affordances": o["affordances"],
            "position_3d": o["position_3d"],
        }
        for o in SCENE_OBJECTS
    ]
    await maintainer.update_from_observations(objects_dict)
    await maintainer.set_goal_from_task_analysis()

    # Validate and fix action predicates
    await maintainer.validate_and_fix_action_predicates()

    # Generate PDDL files
    output_dir = PROJECT_ROOT / "outputs" / "test_layered_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_text = pddl.generate_domain_text()
    problem_text = pddl.generate_problem_pddl()

    domain_path = output_dir / "domain.pddl"
    problem_path = output_dir / "problem.pddl"
    domain_path.write_text(domain_text)
    problem_path.write_text(problem_text)

    print(f"\n[E2E] Domain:\n{domain_text[:500]}...")
    print(f"\n[E2E] Problem:\n{problem_text[:500]}...")

    # Solve
    solver = PDDLSolver(backend=SolverBackend.AUTO, verbose=True)
    result = solver.solve(str(domain_path), str(problem_path), timeout=30.0)

    print(f"\n[E2E] Solver result: success={result.success}, plan={result.plan}")
    if not result.success:
        print(f"[E2E] Error: {result.error_message}")

    assert result.success, (
        f"Solver failed to find a plan. Error: {result.error_message}\n"
        f"Domain:\n{domain_text}\n\nProblem:\n{problem_text}"
    )
    assert result.plan_length >= 2, f"Plan too short: {result.plan}"


# ---------------------------------------------------------------------------
# Retry / error recording test
# ---------------------------------------------------------------------------

async def test_retry_on_bad_scene_does_not_crash():
    """Generator completes gracefully even with an empty scene (no objects)."""
    _skip_if_no_api()
    generator = _make_generator()

    # Empty scene — L1 may produce goals with fabricated IDs, which triggers validation errors
    artifact = await generator.generate_domain(TASK, observed_objects=[])

    print(f"\n[Bad scene] L1 goal_predicates: {artifact.l1.goal_predicates}")
    print(f"[Bad scene] L1 validation_errors: {artifact.l1.validation_errors}")

    # Must return an artifact (not raise)
    assert isinstance(artifact, LayeredDomainArtifact)
    # L1 errors may be populated — that's fine, just must not crash
    assert isinstance(artifact.l1.validation_errors, list)


# ---------------------------------------------------------------------------
# DKB test
# ---------------------------------------------------------------------------

async def test_dkb_records_execution(tmp_path):
    """DKB records an execution to execution_history.jsonl."""
    _skip_if_no_api()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    dkb = DomainKnowledgeBase(tmp_path / "dkb")
    dkb.load()

    generator = LayeredDomainGenerator(api_key=api_key, dkb=dkb, max_layer_retries=1)
    artifact = await generator.generate_domain(TASK, SCENE_OBJECTS)

    history_path = tmp_path / "dkb" / "execution_history.jsonl"
    assert history_path.exists(), "execution_history.jsonl was not created"

    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1, "execution_history.jsonl is empty"

    entry = json.loads(lines[-1])
    assert entry["task"] == TASK
    assert "l1_goal_count" in entry
    assert entry["l1_goal_count"] > 0

    print(f"\n[DKB] History entry: {entry}")


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def run_all():
        print("Running layered PDDL pipeline tests...\n")
        tests = [
            test_l1_goal_extraction,
            test_l1_empty_scene_records_errors,
            test_l2_predicate_vocab,
            test_l2_checked_variant_autorepair,
            test_l3_action_schemas,
            test_l3_goal_achievability,
            test_l4_grounding_precheck,
            test_l5_initial_state,
            test_full_pipeline_generates_layered_artifact,
            test_backward_compat_to_task_analysis,
            test_full_pipeline_solves_blocksworld,
            test_retry_on_bad_scene_does_not_crash,
        ]
        for test_fn in tests:
            name = test_fn.__name__
            print(f"\n{'='*60}")
            print(f"  {name}")
            print('='*60)
            try:
                if name == "test_l2_checked_variant_autorepair":
                    await test_fn()
                else:
                    await test_fn()
                print(f"  PASS")
            except Exception as e:
                print(f"  FAIL: {e}")

    asyncio.run(run_all())
