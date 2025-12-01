## Primitive Task Breakdown Plan (implemented)

Goal: convert symbolic actions (e.g., `pick`, `place`, `push`) into executable xArm primitives grounded in the orchestrator’s world state and snapshots.

### Current Implementation
- **Catalog-driven prompting**: `config/primitive_descriptions.md` documents primitives from `src/kinematics/xarm_curobo_interface.py` and seeds the Gemini prompt template in `config/skill_decomposer_prompts.yaml`.
- **World-state ingestion**: `SkillDecomposer` pulls registry v2.0, snapshot index (`perception_pool/index.json`), and `last_snapshot_id` from `TaskOrchestrator.get_world_state_snapshot()` (or disk fallback). Relevant objects are filtered from the registry and staleness is annotated.
- **LLM call + validation**: Prompts are sent to Gemini Robotics-ER 1.5 with structured output; responses are validated against `PRIMITIVE_LIBRARY` in `src/primitives/skill_plan_types.py`. Registry hashing (`compute_registry_hash`) is recorded on the plan for freshness checks instead of driving a cache.
- **Helper parameters**: LLMs emit image-grounded helper fields (`target_pixel_yx`, `pivot_pixel_yx`, `depth_offset_m`, `motion_normal`, `tcp_standoff_m`) in pixel `[y, x]` instead of raw metric poses. The executor back-projects them using depth/intrinsics from the selected snapshot.
- **Execution**: `PrimitiveExecutor` translates helper fields, validates, and optionally drives `CuRoboMotionPlanner`. Results are normalized for JSON safety.
- **Prompt sources**: Both the primary plan template/response schema and the interaction-point enrichment prompt/schema come from `config/skill_decomposer_prompts.yaml`; no inline prompt strings remain in `SkillDecomposer`.
- **Interaction handling**: Interaction points and positions live on snapshot detections (`perception_pool/snapshots/<id>/detections.json`). The decomposer pulls the latest snapshot’s detections, merges interaction points into the working registry view, and warns when points are absent (no enrichment call is made). The prompt lists each interaction point with normalized `[y, x]` coordinates; the LLM is told to reuse existing points (no new `interaction_points` payload) and only propose new affordance+point pairs when it cannot use an existing one.
- **LLM tuning**: Static Gemini config (top_p, output tokens, mime type, thinking_config) lives on the class via `llm_config_kwargs`; callers can override at construction. Temperature stays per-call.
- **Interaction outputs**: The main plan JSON surfaces proposed affordance points in `interaction_points`; no backward compatibility for `new_interaction_points`.

### Inputs
- Symbolic action + parameters (object IDs/types, goal hints).
- Orchestrator world state (registry, snapshot index, last snapshot id, optional robot state).
- Primitive catalog + prompt template (`config/primitive_descriptions.md`, `config/skill_decomposer_prompts.yaml`).

### Outputs
- `SkillPlan` (ordered `PrimitiveCall` list) with diagnostics:
  - `diagnostics.warnings`, `freshness_notes`, and `interaction_points` if data was missing or stale.
  - `source_snapshot_id` to anchor executor back-projection.

### Usage Pattern
```python
from pathlib import Path
from src.primitives import SkillDecomposer, PrimitiveExecutor

world = orchestrator.get_world_state_snapshot()  # preferred; includes snapshot index

decomposer = SkillDecomposer(
    api_key=cfg.api_key,
    orchestrator=orchestrator,  # optional world pull + staleness checks
)
plan = decomposer.plan("pick", {"object_id": "black_folded_fabric"})

executor = PrimitiveExecutor(
    primitives=None,  # swap in CuRoboMotionPlanner(...) to execute
    perception_pool_dir=Path(cfg.state_dir) / "perception_pool",
)
result = executor.execute_plan(plan, world, dry_run=True)
print(result.warnings)
```

### Testing & Artifacts
- Translation-only and LLM plans for the pick pipeline live under `tests/artifacts/translation_pick/` and `tests/artifacts/llm_pick/`.
- Replay a cached plan after tests:  
  `uv run python scripts/run_cached_plan.py --plan tests/artifacts/llm_pick/pick_plan_llm_translated.json --world tests/assets/continuous_pick_fixture --robot-ip 192.168.1.224` (add `--dry-run` to skip hardware).

### Open Items / Watchpoints
- Keep `config/primitive_descriptions.md` and `PRIMITIVE_LIBRARY` in lockstep with CuRobo helper signatures.
- If new helper parameters are introduced, update the prompt schema (`config/skill_decomposer_prompts.yaml`) and executor translation rules together.
- Snapshot availability drives executor success; log missing snapshot IDs and keep perception pool pruning (`max_snapshot_count`) conservative for experiments.
