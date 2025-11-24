# Documentation Index

This folder tracks the current behavior of the perception/orchestration stack. Use `uv` for all commands (syncing dependencies, running examples, or running tests).

## What to Read
- [ORCHESTRATOR.md](ORCHESTRATOR.md) – TaskOrchestrator lifecycle, perception pool snapshots, and state layout produced under `state_dir`.
- [perception.md](perception.md) – `ObjectTracker`/`ContinuousObjectTracker`, prompt loading from `config/prompts_config.yaml`, registry v2.0, and coordinate conventions.
- [planning.md](planning.md) – PDDL maintainer/task monitor plus the primitives layer (`SkillDecomposer`, `PrimitiveExecutor`), snapshot utilities, and cached-plan replay guidance.
- [camera.md](camera.md) – RealSense/Webcam wrappers, intrinsics, and 2D↔3D utilities used by perception and the executor.

## Quick References
- Examples: `uv run python examples/orchestrator_demo.py` (interactive orchestrator), `uv run python examples/object_tracker_demo.py`.
- Cached plan replay: `uv run python scripts/run_cached_plan.py --plan <plan.json> --world <world_dir> --dry-run`.
- Tests: `uv run python -m pytest -q` (LLM-gated suites skip without `GEMINI_API_KEY`/`GOOGLE_API_KEY`).

## Key Config Files
- `config/orchestrator_config.py` – detection cadence, snapshot cadence/retention, state_dir.
- `config/prompts_config.yaml` – Gemini Robotics prompts for perception.
- `config/primitive_descriptions.md` + `config/skill_decomposer_prompts.yaml` – primitive catalog and Gemini prompt/schema for skill decomposition.

Keep these docs in sync with the corresponding modules under `src/`; update them in the same PR whenever APIs or behaviors change.
