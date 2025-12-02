# Task Orchestrator

The `TaskOrchestrator` (see `src/planning/task_orchestrator.py`) manages task analysis, continuous perception, PDDL state, and snapshot persistence. It is the source of truth for world-state handoffs to the primitives layer.

## What It Does
- **Task intake + PDDL**: Uses `PDDLDomainMaintainer` and `TaskStateMonitor` to analyze a natural-language task, gate exploration, and write domain/problem files.
- **Continuous perception**: Wraps `ContinuousObjectTracker` with prompt-driven `ObjectTracker` to keep a live registry in sync with task predicates.
- **Snapshots + perception pool**: Every detection can emit `color.png`, optional `depth.npz`, `intrinsics.json`, `detections.json`, `manifest.json`, and optional `robot_state.json` under `state_dir/perception_pool/` with an `index.json` linking snapshots to object IDs.
- **Snapshot alignment**: Snapshots reuse the exact frame from the last detection pass (color/depth/intrinsics) and build `detections.json` from only that pass, so stale registry entries do not leak into saved snapshots even if the camera moves between detection and write-out.
- **Persistence**: `save_state()` writes `state.json`, `registry.json` (v2.0 with `observations` / `latest_observation`), PDDL artifacts, and the perception pool index pointer. `load_state()` reloads registry + snapshot index when present.
- **World-state export**: `get_world_state_snapshot()` returns `registry`, `last_snapshot_id`, `snapshot_index`, and `robot_state` for downstream planners such as `SkillDecomposer` / `PrimitiveExecutor`.

## Quick Start (code-backed)
```python
import asyncio, os
from pathlib import Path
from src.planning.task_orchestrator import TaskOrchestrator
from orchestrator_config import OrchestratorConfig

async def main():
    cfg = OrchestratorConfig(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        update_interval=2.0,
        min_observations=2,
        auto_save=True,
        auto_save_on_detection=True,
        enable_snapshots=True,
        snapshot_every_n_detections=1,
        state_dir=Path("outputs/orchestrator_state"),
    )
    orch = TaskOrchestrator(cfg)
    await orch.initialize()
    await orch.process_task_request("inventory towels")
    await orch.start_detection()
    await asyncio.sleep(2.0)  # allow at least one detection/snapshot
    world = orch.get_world_state_snapshot()
    await orch.generate_pddl_files()
    await orch.shutdown()
    return world

asyncio.run(main())
```

Or run the interactive TUI demo (representation only) via `uv run python examples/orchestrator_demo.py` to step through task analysis, continuous detection, snapshotting, and PDDL generation without issuing robot commands.

## Configuration Highlights (`config/orchestrator_config.py`)
- Detection cadence: `update_interval`, `min_observations`, `fast_mode`.
- Persistence knobs: `state_dir`, `auto_save`, `auto_save_on_detection`, `auto_save_on_state_change`.
- Snapshot controls: `enable_snapshots`, `snapshot_every_n_detections`, `perception_pool_dir`, `max_snapshot_count`, `depth_encoding`.
- Robot provider: attach a duck-typed planner with `get_robot_state()` for snapshot context; if none is provided, the orchestrator attempts to attach `CuRoboMotionPlanner` by default.

## Artifact Layout (default `state_dir=outputs/orchestrator_state`)
```
state.json                # orchestrator metadata + last_snapshot_id + perception pool pointer
registry.json             # v2.0 registry with snapshot references
pddl/
  task_execution.pddl
  current_task.pddl
perception_pool/
  index.json              # links object_ids <-> snapshot_ids
  snapshots/<SNAPSHOT_ID>/color.png
  snapshots/<SNAPSHOT_ID>/depth.npz        # optional
  snapshots/<SNAPSHOT_ID>/intrinsics.json
  snapshots/<SNAPSHOT_ID>/detections.json
  snapshots/<SNAPSHOT_ID>/manifest.json
  snapshots/<SNAPSHOT_ID>/robot_state.json # optional
```

## Operational Notes
- Always call `initialize()` before processing tasks; shut down to release camera resources.
- Let `TaskStateMonitor` decide readiness (`READY_FOR_PLANNING`) instead of bypassing `min_observations`.
- `get_world_state_snapshot()` is the handoff for `SkillDecomposer.plan(...)` and `PrimitiveExecutor.execute_plan(...)`; it embeds the latest perception pool index so snapshot utilities can load color/depth/intrinsics.
- Snapshot manifests stamp `captured_at` from the detection frame timestamp (fallback to `recorded_at`); bounding boxes and color/depth are taken from that same detection bundle to avoid drift if the camera moves post-detection.
- Snapshots are IDed as `YYYYMMDD_HHMMSS_mmm-<shortid>` for stable ordering; timestamp authority is host time, not camera hardware.
- When storage is tight, tune `max_snapshot_count` instead of deleting folders manually to keep `index.json` consistent.
- Logging: orchestrator and camera now emit via Python logging. The Textual demo wires `configure_logging(callback=_write_log, include_console=False, level=logging.DEBUG)` to stream RealSense/debug lines into the UI without buffering.
