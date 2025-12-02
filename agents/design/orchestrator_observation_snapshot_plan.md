## Observation Snapshot + Perception Pool (implemented)

Goal (delivered): `TaskOrchestrator` now captures lightweight “observation snapshots” (RGB, optional depth, intrinsics, detections, optional robot state) and maintains a file-backed perception pool that links objects to every snapshot they appear in. The feature is **enabled by default** and hooks into detection callbacks.

### Scope
- Persist aligned RGB(+depth), intrinsics, and detections for later querying/re-detection.
- Maintain `perception_pool/index.json` mapping `objects -> [snapshot_ids]` and `snapshots[id] -> files/objects/timestamps`.
- Keep state persistence non-breaking: existing PDDL outputs remain unchanged; `registry.json` now embeds snapshot references via the orchestrator’s enhanced registry builder.

### Current Behavior (as shipped in `src/planning/task_orchestrator.py`)
- Snapshots are taken every detection by default (`enable_snapshots=True`, `snapshot_every_n_detections=1`). Adjust cadence/retention via `OrchestratorConfig`. Each snapshot now reuses the exact color/depth/intrinsics frame from the most recent detection cycle to avoid drift when the camera moves between perception and write-out.
- Snapshot contents (under `perception_pool/snapshots/<SNAPSHOT_ID>/`):
  - `color.png` (lossless)
  - `depth.npz` (optional; float32 meters) when depth is enabled
  - `intrinsics.json` (from camera `to_dict()`)
  - `detections.json` (stamp + objects with `object_id`, `object_type`, `affordances`, `pddl_state`, `position_3d`, `bounding_box_2d`)
  - `manifest.json` (`snapshot_id`, `captured_at`, `recorded_at`, sources, file list, optional label/reason)
  - `robot_state.json` (optional, via duck-typed `get_robot_state()`)
- `detections.json` is built from only the most recent detection pass (not the long-lived registry), so objects drop out of snapshots when they leave the frame, even though the registry may still carry them for re-ID context.
- Manifest timestamps: `captured_at` now mirrors the detection frame timestamp (when available from the last detection bundle) and falls back to `recorded_at`; IDs are `YYYYMMDD_HHMMSS_mmm-<shortid>` for sortability, not time authority.
- Perception pool index (`perception_pool/index.json`, version `1.0`):
  - `objects[object_id] -> [snapshot_ids]`
  - `snapshots[id] -> {captured_at, recorded_at, objects, files, label, reason}`
  - `last_snapshot_id` stored for quick lookups
  - Paths are relative to `perception_pool/`
- `get_world_state_snapshot()` bundles the enhanced registry, `snapshot_index`, `last_snapshot_id`, and optional `robot_state` for consumers like `SkillDecomposer` / `PrimitiveExecutor`.
- Retention: `max_snapshot_count` (default 200) prunes oldest snapshot folders and cleans index entries.
- Robot context: only `robot_state.json` is written (best-effort serialization of `get_robot_state()`); extrinsics are not persisted yet. If no provider is given, the orchestrator attempts to attach `CuRoboMotionPlanner` by default.

### Configuration (`config/orchestrator_config.py`)
- `enable_snapshots: bool = True`
- `snapshot_every_n_detections: int = 1` (0 disables)
- `perception_pool_dir: Optional[Path] = None` (defaults to `state_dir / "perception_pool"`)
- `max_snapshot_count: Optional[int] = 200`
- `depth_encoding: Literal["npz"] = "npz"`
- `robot: Optional[Any] = None` (duck-typed provider; default xArm CuRobo attached when available)

### Orchestrator Hooks
- `_on_detection_callback(...)` triggers `save_snapshot(reason="periodic")` per cadence; failures are non-fatal.
- `save_snapshot(reason: str = "", label: Optional[str] = None)` serializes artifacts with a dedicated lock, updates index, and enforces retention.
- `save_state()` writes `registry.json` (v2.0 with `observations`/`latest_observation`) plus PDDL files; `state.json` records `last_snapshot_id` and `files["perception_pool_index"]` if present.
- `load_state()` reloads the perception pool index into memory when available.

### Deviations from the original concept (tracked for future extension)
- No `extrinsics.json` is emitted; any transform data should be added to `robot_state.json` once a stable schema is agreed upon.
- `captured_at` now reflects detection time; `recorded_at` remains the write time.
- Snapshot retention is index-driven; avoid manual folder deletion to keep references consistent.

### Operator Notes
- Always cite snapshot IDs and the cadence/retention settings used when reporting results or incidents.
- When customizing storage paths, prefer `perception_pool_dir` over manual symlinks.
- If robot context is expected but missing, capture that in the daily journal with the base commit hash (`git log -1 --format="%H"`).
- Route orchestrator/camera logs through `configure_logging(..., callback=_write_log, include_console=False)` (see `examples/orchestrator_demo.py`) so snapshot/camera errors surface in real time without stdout redirection.
