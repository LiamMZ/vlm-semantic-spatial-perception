# Planning + Primitives

The planning layer turns natural-language tasks into PDDL artifacts and, when requested, into executable primitive plans grounded in the orchestrator’s snapshots.

**Code**: `src/planning/`, `src/primitives/`  
**Config**: `config/orchestrator_config.py`, `config/primitive_descriptions.md`, `config/skill_decomposer_prompts.yaml`

## Components
- `pddl_representation.py` – thread-safe domain/problem model and PDDL writers.
- `pddl_domain_maintainer.py` – seeds/updates the domain from tasks and observations.
- `task_state_monitor.py` – gates exploration vs. plan readiness (`TaskState.PLAN_AND_EXECUTE`).
- `task_orchestrator.py` – orchestrates perception, PDDL maintenance, snapshots, and world-state export.
- `primitives/skill_decomposer.py` – LLM-backed decomposition of symbolic actions to primitives; attaches latest snapshot bytes and registry slices.
- `primitives/skill_plan_types.py` – `PrimitiveCall`, `SkillPlan`, validators, and registry hashing for plan freshness tracking.
- `primitives/primitive_executor.py` – translates helper fields (e.g., `target_pixel_yx`) into metric targets using depth/intrinsics from the perception pool and optionally drives `CuRoboMotionPlanner`.
- `planning/utils/snapshot_utils.py` – loads color/depth/intrinsics for a snapshot ID from `perception_pool/index.json`.

## Workflow (end-to-end)
1. **Task analysis**: `PDDLDomainMaintainer.initialize_from_task(...)` via `TaskOrchestrator.process_task_request(...)`.
2. **Continuous updates**: `ContinuousObjectTracker` feeds detections; `update_from_observations(...)` keeps the domain and registry fresh.
3. **Gating**: `TaskStateMonitor.determine_state()` sets `READY_FOR_PLANNING` once predicates/objects meet `min_observations`.
4. **PDDL outputs**: `TaskOrchestrator.generate_pddl_files()` writes domain/problem under `state_dir/pddl/`.
5. **Primitive planning** (optional): `SkillDecomposer.plan(action_name, parameters, orchestrator=...)` pulls the latest registry + snapshot, uses `config/primitive_descriptions.md` and `config/skill_decomposer_prompts.yaml`, and validates against `PRIMITIVE_LIBRARY`. The decomposer reads interaction points from the latest snapshot detections (`perception_pool/snapshots/<id>/detections.json`), merges them into the working registry view, and passes their pixel `[y, x]` coordinates into the prompt; the LLM is instructed to reuse those points (leaving `interaction_points` empty) and only emit new affordance+point entries when choosing a novel location.
6. **Execution/translation**: `PrimitiveExecutor.execute_plan(...)` back-projects helper pixels to 3D using the perception pool and calls the motion planner (or dry-runs).

## Skill Decomposition + Execution (code-backed)
```python
import os
from pathlib import Path
from src.primitives import SkillDecomposer, PrimitiveExecutor

# world_state may come from orchestrator.get_world_state_snapshot()
world_state = {
    "registry": registry_payload,
    "last_snapshot_id": "20251123_183123_042-abc123",
    "snapshot_index": snapshot_index_payload,
}

decomposer = SkillDecomposer(
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    orchestrator=None,  # pass orchestrator instance to auto-pull state
)
plan = decomposer.plan(
    action_name="pick",
    parameters={"object_id": "towel_patch"},
    world_hint=world_state,
)

# Translate (and optionally execute) against the perception pool on disk
executor = PrimitiveExecutor(
    primitives=None,  # inject CuRoboMotionPlanner(...) to execute on hardware
    perception_pool_dir=Path("outputs/orchestrator_state/perception_pool"),
)
result = executor.execute_plan(plan, world_state, dry_run=True)
print(result.warnings)
```

Helper fields expected from the LLM (`target_pixel_yx`, `pivot_pixel_yx`, `depth_offset_m`, `motion_normal`, `tcp_standoff_m`) are normalized `[y, x]` inputs (0–1000, Robotics-ER style) that the executor converts to metric parameters; when `metadata.resolved_interaction_point.position_3d` is present the executor prefers that 3D point over recomputing from depth. Leave metric-only fields unset unless already known in meters.

## Cached Plans and Replay
- Translation and LLM plans from the pick pipeline live under `tests/artifacts/translation_pick/` and `tests/artifacts/llm_pick/`.
- After running tests (LLM suites skip without an API key), replay a cached plan:
```
uv run python scripts/run_cached_plan.py \
  --plan tests/artifacts/llm_pick/pick_plan_llm_translated.json \
  --world tests/assets/continuous_pick_fixture \
  --robot-ip 192.168.1.224
```
Add `--dry-run` to skip robot execution; the script expects `registry.json`, `state.json`, and `perception_pool/index.json` in the `--world` directory.

## State Expectations for Planners
`SkillDecomposer` and `PrimitiveExecutor` expect:
- `registry`: enhanced registry (v2.0) with `observations`/`latest_observation` per object.
- `snapshot_index`: contents of `perception_pool/index.json` (paths are relative to the pool root).
- `last_snapshot_id`: snapshot to ground helper fields; falls back to the latest in the index if missing.
- Optional `robot_state`: captured by the orchestrator if a robot provider supplies `get_robot_state()`.

## Useful Commands
- Interactive orchestration (representation only): `uv run python examples/orchestrator_demo.py`
- PDDL-only demos: `uv run python examples/pddl_representation_demo.py`, `uv run python examples/task_monitoring_demo.py`
- Tests (LLM-gated suites skip without key): `uv run python -m pytest -q`
