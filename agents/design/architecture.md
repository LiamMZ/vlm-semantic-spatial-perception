# Agent Design & API Overview

Agents sit on top of the production `TaskOrchestrator` pipeline plus the primitives layer. They share one perception backbone, one PDDL maintainer, and one snapshot pool to keep world state reproducible across personas.

## Current Stack (code-backed)
1. **Task analysis + PDDL maintenance** – `PDDLDomainMaintainer` (`llm_task_analyzer.py`) seeds and updates `PDDLRepresentation`; `TaskStateMonitor` gates readiness.
2. **Continuous perception** – `ContinuousObjectTracker` (prompted `ObjectTracker`) streams detections into `DetectedObjectRegistry`, honoring `scene_change_threshold` and `fast_mode`.
3. **State persistence + snapshots** – `TaskOrchestrator` writes `state.json`, `registry.json` (v2.0 with `observations`), PDDL outputs, and a perception pool (`color.png`, optional `depth.npz`, `intrinsics.json`, `detections.json`, `manifest.json`, optional `robot_state.json`) indexed by `perception_pool/index.json`.
4. **World-state export** – `get_world_state_snapshot()` bundles registry + `snapshot_index` + `last_snapshot_id` + `robot_state` for downstream planners.
5. **Primitive planning/execution** – `SkillDecomposer` (Gemini ER) + `PrimitiveExecutor` live under `src/primitives/`; they use `config/primitive_descriptions.md`, `config/skill_decomposer_prompts.yaml`, and snapshot utils to turn symbolic steps into validated primitive calls and (optionally) execute via `CuRoboMotionPlanner`.

## Data Flow
```
Task text
  │
  ▼
PDDLDomainMaintainer.initialize_from_task(...)
  │                ▲
  ▼                │
ContinuousObjectTracker → registry (v2.0) ─┐
  │                                        ├─> TaskStateMonitor → READY_FOR_PLANNING
  └─> save_snapshot(...) → perception_pool ┘
  │
  ▼
TaskOrchestrator.get_world_state_snapshot()
  │
  ▼
SkillDecomposer.plan(...) → SkillPlan → PrimitiveExecutor.execute_plan(...)
```

## Interfaces & Config
- `TaskOrchestrator` / `OrchestratorConfig` (`config/orchestrator_config.py`): detection cadence, snapshot cadence/retention, state directory, optional robot provider (duck-typed `get_robot_state()`).
- Prompts: `config/prompts_config.yaml` (perception), `config/skill_decomposer_prompts.yaml` (primitive planning). Keep them aligned with `agents/design/prompts_configuration.md`.
- Primitive catalog: `config/primitive_descriptions.md` must mirror callable signatures in `src/kinematics/xarm_curobo_interface.py` and stay in sync with `PRIMITIVE_LIBRARY`.
- Snapshot helpers: `src/planning/utils/snapshot_utils.py` resolve color/depth/intrinsics for a snapshot ID referenced in the registry or plan.
- Logging: use `configure_logging` (`src/utils/logging_utils.py`) to route `TaskOrchestrator`/`RealSenseCamera` logs into UI callbacks (e.g., the Textual demo) instead of redirecting stdout.

## Example Orchestration + Primitive Plan
```python
import asyncio, os
from pathlib import Path
from orchestrator_config import OrchestratorConfig
from src.planning.task_orchestrator import TaskOrchestrator
from src.primitives import SkillDecomposer, PrimitiveExecutor

async def main():
    cfg = OrchestratorConfig(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        update_interval=2.0,
        enable_snapshots=True,
        snapshot_every_n_detections=1,
        state_dir=Path("outputs/orchestrator_state"),
    )
    orch = TaskOrchestrator(cfg)
    await orch.initialize()
    await orch.process_task_request("organize the black folded fabric")
    await orch.start_detection()
    await asyncio.sleep(2.0)  # allow at least one snapshot
    world = orch.get_world_state_snapshot()

    decomposer = SkillDecomposer(api_key=cfg.api_key, orchestrator=orch)
    plan = decomposer.plan("pick", {"object_id": "black_folded_fabric"})

    executor = PrimitiveExecutor(
        primitives=None,  # inject CuRoboMotionPlanner(...) to execute on hardware
        perception_pool_dir=cfg.state_dir / "perception_pool",
    )
    result = executor.execute_plan(plan, world, dry_run=True)
    await orch.shutdown()
    return result.warnings

asyncio.run(main())
```

## API Touchpoints & Cross-links
- `src/planning/task_orchestrator.py`: registry builder (v2.0), perception pool lifecycle, world-state export, snapshot retention.
- `src/planning/utils/snapshot_utils.py`: shared snapshot loader for decomposer/executor.
- `src/primitives/skill_decomposer.py`: registry hashing, snapshot-aware prompting, validation.
- `src/primitives/primitive_executor.py`: helper-parameter translation, JSON-safe execution outputs.
- `config/primitive_descriptions.md`, `config/skill_decomposer_prompts.yaml`: keep aligned with `PRIMITIVE_LIBRARY` and CuRobo helper signatures.
- `scripts/run_cached_plan.py`: replay cached plans against a perception pool on disk.

## Extension Points
- **Sensors**: add camera adapters under `src/camera/`; keep `_get_camera_frames()` returning `(color, depth, intrinsics)`.
- **Robot providers**: supply a planner with methods matching primitives plus `get_robot_state()` for snapshots.
- **Snapshots**: tune cadence/retention via config instead of manual deletion to keep `index.json` consistent.
- **Prompts/Primitive catalog**: update YAML/MD sources first; keep snippets runnable and add commit references in `AGENTS.md` when these contracts change.

## Documentation Sources
- Architecture and operational guardrails live here and in `agents/operations/playbook.md`.
- Perception prompts/configuration guidance: `agents/design/prompts_configuration.md`.
- Snapshot behavior: `agents/design/orchestrator_observation_snapshot_plan.md`.
- Journal evidence: `agents/journal/YYYY-MM-DD.md` (include commit hash + timestamp for every change).
