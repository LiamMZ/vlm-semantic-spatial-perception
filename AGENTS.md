# Agents Documentation Entry Point

Use this file as the index for all agent-related docs. Each component lives in its own subfolder under `agents/` to keep the content concise and easy to maintain.

## Directory Map (agent-facing docs only)
- `agents/design/architecture.md` – data flow, API expectations, and extension points tying perception, world modeling, planning, and primitives together.
- `agents/operations/playbook.md` – implementation guardrails, stage-by-stage best practices, review checklist, and incident procedures.
- `agents/journal/` – daily logs of experiments, bug hunts, outages, and scheduled maintenance notes. Each day gets its own `YYYY-MM-DD.md` file. See `agents/journal/README.md` for logging guidelines.

## Maintenance Instructions
1. **Find the paired doc for your change**:
   - Perception/prompts changes (`src/perception/*`, `config/prompts_config.yaml`): update `agents/design/prompts_configuration.md` and `docs/perception.md`.
   - Orchestrator/state/snapshots (`src/planning/task_orchestrator.py`, `config/orchestrator_config.py`): update `agents/design/architecture.md`, `agents/design/orchestrator_observation_snapshot_plan.md`, and `docs/ORCHESTRATOR.md`.
   - Primitives/skill decomposition (`src/primitives/*`, `config/primitive_descriptions.md`, `config/skill_decomposer_prompts.yaml`): update `agents/design/primitive_task_breakdown_plan.md` and `docs/planning.md`.
   - Operational guardrails or runbooks: update `agents/operations/playbook.md`.
   - Top-level behavior/entry points: update the root `README.md` whenever setup, demos, or replay instructions change.
2. **Update in the same PR**: when code or config moves, edit the mapped docs above. If multiple areas changed, touch each relevant doc. Missing doc updates should block merges.
3. **Log changes**: every experiment or decision goes to `agents/journal/YYYY-MM-DD.md` using the template in `agents/journal/README.md`. Use system timestamps (`date '+%Y-%m-%d-%H%M'`) and reference the base commit (`git log -1 --format="%H"`).
4. **Keep snippets runnable**: any snippet you edit in these docs must match current APIs; re-run the referenced sample or demo when you change it and note the command in the journal.

## Repo Structure (for agent work)
- `agents/` – agent-facing design, operations, and journal files (authoritative for agent workflows).
- `docs/` – product-facing docs for perception/orchestrator/planning/primitives; keep in sync with `agents/design/*` when behaviors change.
- `src/` – implementation (`perception`, `planning`, `primitives`, `camera`, `kinematics`, `utils`).
- `config/` – orchestrator defaults, perception prompts, primitive catalog/prompt schema.
- `examples/` – demos (orchestrator TUI, perception/PDDL samples).
- `scripts/run_cached_plan.py` – replay translated plans against a saved perception pool.
- `tests/` – regression suites and cached artifacts for translation/LLM plans.
