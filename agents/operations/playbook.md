# Agent Implementation Instructions

## Guiding Principles
- Ground everything in the orchestrator: use `TaskOrchestrator`/`get_world_state_snapshot()` instead of bespoke loops.
- Keep prompts/config in YAML/MD (`config/object_tracker_prompts.yaml`, `config/llm_task_analyzer_prompts.yaml`, `config/pddl_domain_maintainer_prompts.yaml`, `config/skill_decomposer_prompts.yaml`, `config/primitive_descriptions.md`); never inline LLM prompts.
- Preserve reproducibility: snapshots + registry v2.0 are the canonical evidence. Do not hand-edit `registry.json` or perception pool contents.
- Use `uv` for all commands (`uv run python ...`, `uv run python -m pytest -q`) and log what you ran in `agents/journal/YYYY-MM-DD.md` with timestamps from `date '+%Y-%m-%d-%H%M'`.

## Best Practices by Stage
1. **Task Intake**
   - Use `TaskOrchestrator.process_task_request()` so `TaskAnalysis` and predicate seeding are recorded.
   - Capture/attach an environment frame when available; note whether a frame was used in the journal.
   - Reject ambiguous tasks early and log the exchange with timestamp + base commit hash.
2. **Perception**
   - `ObjectTracker` must load prompts from YAML; keep `fast_mode` and update cadence aligned with hardware constraints and document overrides.
   - Ensure frame providers return `(color, depth, intrinsics)`; depth may be `None` for webcam-only runs—note this when interpreting executor warnings.
3. **Registry & Snapshots**
   - Registry saves are versioned (v2.0) via `save_state()`; do not edit `registry.json`.
   - Snapshots default to every detection (`enable_snapshots=True`, `snapshot_every_n_detections=1`); adjust cadence/`max_snapshot_count` via config and record deltas.
   - Always cite snapshot IDs and `state_dir/perception_pool/` when sharing results.
4. **Planning & Primitives**
   - Let `TaskStateMonitor` gate `READY_FOR_PLANNING`; avoid bypassing `min_observations`.
   - Keep `config/primitive_descriptions.md` and `config/skill_decomposer_prompts.yaml` synchronized with `PRIMITIVE_LIBRARY`.
   - When decomposing, prefer `SkillDecomposer(orchestrator=...)` so registry/snapshot freshness checks are automatic.
   - Use `PrimitiveExecutor.prepare_plan(...)`/`execute_plan(...)` to translate helper pixels; keep perception pool intact for back-projection.
5. **Execution Feedback**
   - Capture success/failure reasons, snapshot IDs, registry hash (from `compute_registry_hash`), and any manual overrides. Append them to the journal entry immediately.

## Design Choices to Preserve
- Modular camera abstraction (`src/camera` factory) and prompt templating in YAML.
- Event-driven persistence: rely on `auto_save_on_detection`/`auto_save_on_state_change` instead of timers.
- Snapshot retention via `max_snapshot_count` rather than manual deletion to keep `index.json` consistent.
- Primitives stay image-grounded: helper pixels in prompts, metric translation in the executor (not the LLM).

## Agent Review Checklist
1. Docs updated (`agents/design/architecture.md`, `agents/design/primitive_task_breakdown_plan.md`, `docs/`) plus commit hash references in `AGENTS.md`.
2. Prompt/config changes reflected in YAML/MD (perception + primitives), not inline.
3. Tests/demos exercised with `uv run python -m pytest -q` (note if LLM suites skipped) and/or `uv run python examples/orchestrator_demo.py`; log commands + timestamps.
4. Cached-plan replay verified when touching primitives: `uv run python scripts/run_cached_plan.py --plan tests/artifacts/translation_pick/pick_plan_prepare_plan.json --world tests/assets/continuous_pick_fixture --dry-run`.
5. Journal entry added with task_id (if applicable), snapshot IDs, detector/executor versions, and base git commit hash.

## Operational Instructions
- Keep secrets in `.env` (API keys) and load via `dotenv`/env vars; never commit them.
- Long experiments: record environment info (`python --version`, `uv pip list | rg google-genai`) and snapshot cadence in the journal.
- Orchestrator demo (representation only): `uv run python examples/orchestrator_demo.py`.
- Plan replay: `uv run python scripts/run_cached_plan.py --plan tests/artifacts/llm_pick/pick_plan_llm_translated.json --world tests/assets/continuous_pick_fixture --robot-ip 192.168.1.224` (add `--dry-run` for translation-only).
- Tests: `uv run python -m pytest -q` (LLM-gated suites skip without `GEMINI_API_KEY`/`GOOGLE_API_KEY`); artifacts land in `tests/artifacts/`.

## Adding New Agents
1. Build on `TaskOrchestrator` + `SkillDecomposer`/`PrimitiveExecutor`; avoid bespoke perception/planning loops.
2. Store agent defaults under `config/agents/<agent>.yaml` if needed and document overrides here.
3. Extend `AGENTS.md` if new upkeep is required (e.g., new prompt packs, snapshot labels).
4. Submit code + doc changes together; reviewers should reject code-only updates.

## Incident Response
- On failures, freeze the agent version, capture logs/stack traces, include snapshot IDs + `perception_pool/index.json` location, and log the incident immediately with timestamp + base commit hash.
- If external dependencies (Gemini, camera drivers) regress, note the workaround, snapshot cadence, and expected resolution date in the journal and here if persistent.
- For recurring failures, add mitigation steps (e.g., replay commands, perception pool cleanup commands) so operators are not guessing.
