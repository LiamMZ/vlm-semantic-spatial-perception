# VLM Semantic Spatial Perception

Ad-hoc world model generation in PDDL, powered by LLM (Gemini Robotics-ER), which decomposes tasks into a set of primitives that are executed on the robot.

## Capabilities
- Prompt-driven perception (`ObjectTracker`/`ContinuousObjectTracker`) with affordances, interaction points, and optional PDDL predicates.
- Orchestration with auto-save + perception pool snapshots (`color.png`, optional `depth.npz`, `intrinsics.json`, `detections.json`, `manifest.json`, optional `robot_state.json`).
- Skill decomposition (`SkillDecomposer`) that uses the latest registry + snapshot to propose primitives, validated against `PRIMITIVE_LIBRARY`.
- Primitive execution/translation (`PrimitiveExecutor`) that back-projects helper pixels to metric coordinates and optionally drives `CuRoboMotionPlanner`.

## Setup (uv)
1. Install [uv](https://docs.astral.sh/uv/).
2. Sync dependencies: `uv sync`
3. Set API key (required for Gemini calls):
   ```bash
   export GEMINI_API_KEY=...
   # or: export GOOGLE_API_KEY=...
   ```

## Running
- **Orchestrator demo (interactive, representations only)**  
  `uv run python examples/orchestrator_demo.py`  
  Drives task analysis, continuous detection, snapshots, and PDDL generation via a Textual TUI; does not command the robot.

- **Cached plan replay (from test artifacts)**  
  After tests, a translated plan is available at `tests/artifacts/llm_pick/pick_plan_llm_translated.json` with its world assets in `tests/assets/continuous_pick_fixture`. Replay or execute:
  ```bash
  uv run python scripts/run_cached_plan.py \
    --plan tests/artifacts/llm_pick/pick_plan_llm_translated.json \
    --world tests/assets/continuous_pick_fixture \
    --robot-ip 192.168.1.224
  ```
  Add `--dry-run` to translate/validate without touching hardware.

- **Tests**  
  `uv run -m pytest -q` (LLM-gated suites skip without an API key). Artifacts land under `tests/artifacts/`.


## Notes
- World state is exported via `TaskOrchestrator.get_world_state_snapshot()` (registry v2.0 + perception pool index + `last_snapshot_id`). Downstream planners/executors rely on that structure—avoid manual edits.
- Keep prompts/catalog files in sync with `src/perception/object_tracker.py`, `src/primitives/skill_decomposer.py`, and `PRIMITIVE_LIBRARY`.
- Use `uv` for all maintenance (installs, tests, examples) to stay aligned with `uv.lock`.

## Directory Structure
```
├── config/                   # Orchestrator defaults, perception prompts, primitive catalog/prompt schema
├── docs/                     # Product-facing docs (perception, orchestrator, planning/primitives, camera)
├── agents/                   # Agent-facing design, ops, and journal instructions
├── src/
│   ├── perception/           # ObjectTracker, ContinuousObjectTracker, registry, coordinate utils
│   ├── planning/             # PDDL maintainer, task monitor, orchestrator, snapshot utils
│   ├── primitives/           # SkillDecomposer, PrimitiveExecutor, SkillPlan types
│   ├── camera/               # RealSense/Webcam wrappers and 2D↔3D helpers
│   ├── kinematics/           # CuRobo/XArm primitives
│   └── utils/                # Shared utilities
├── examples/                 # Demos (orchestrator TUI, PDDL samples)
├── scripts/                  # Utilities (e.g., run_cached_plan.py)
└── tests/                    # Regression suites + cached translation/LLM artifacts
    ├── assets/               # World fixtures used by cached plans
    ├── artifacts/            # Translation/LLM artifacts
    └── conftest.py           # Test configuration
```
