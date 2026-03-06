# Migration Journal: Staged Representation Builder

## Scope

This journal tracks concrete behavior changes made while implementing
`agents/MIGRATION_PLAN.md`.

## Pre-Migration Baseline Attempt

- Date: 2026-03-06
- Command:
  - `uv run scripts/benchmark_saved_world.py --captured-root outputs/captured_worlds --run-all --benchmark-output-root outputs/benchmarks/migration_baseline_pre`
- Result:
  - Blocked locally because `GEMINI_API_KEY` / `GOOGLE_API_KEY` was not set in the shell.
  - No live LLM baseline artifacts were produced in this environment.

## Behavioral Changes

### Planner Construction

- Replaced the one-shot task analysis contract with staged outputs:
  - `abstract_goal`
  - `predicate_inventory`
  - `action_schemas`
  - `grounding_summary`
  - `diagnostics`
- `LLMTaskAnalyzer` now exposes:
  - `analyze_goal(...)`
  - `analyze_predicates(...)`
  - `analyze_actions(...)`
  - `analyze_grounding(...)`

### Domain Maintenance

- Replaced domain-text refinement as the primary repair path.
- `PDDLDomainMaintainer` now owns:
  - `build_representation(...)`
  - `ground_representation(...)`
  - embedded validation
  - `repair_representation(...)`
- Repair behavior now follows reverse abstraction order:
  - actions
  - predicates
  - goal

### Orchestration

- `TaskOrchestrator.process_task_request()` now builds stages 1-3 explicitly.
- Detection callbacks now refresh only grounding.
- Planning retries now classify likely failure layer and request targeted repair.

### Diagnostics

- Representation validation now records:
  - per-layer validity
  - validation issues
  - repair history
  - LLM call counts / elapsed time
- `scripts/benchmark_saved_world.py` now emits replay summaries with:
  - staged validation status
  - grounding gaps
  - undefined-symbol error counts
  - LLM call metrics

## Test Evidence

### Deterministic Migration Tests

- Added `tests/test_representation_builder_migration.py`
- Coverage:
  - staged build + grounding on cached `vegetables` world
  - missing grounding on cached `markers` world
  - targeted action-layer repair without mutating goal/predicate layers

### Repository Test Suite

- Command:
  - `uv run -m pytest tests`
- Result:
  - `10 passed, 1 skipped`

## Incidental Fixes Discovered During Validation

- `PrimitiveExecutor.prepare_plan()` now normalizes translated `target_position`
  outputs to plain rounded Python float lists.
- This fixed an existing test failure unrelated to the migration but surfaced by
  the full-suite run.
