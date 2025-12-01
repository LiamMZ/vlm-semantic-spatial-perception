# Prompts Configuration

`ObjectTracker` and `ContinuousObjectTracker` always load prompts from YAML; do not inline prompt strings. `ContinuousObjectTracker` now subclasses `ObjectTracker` in the same module and reuses the exact prompt config (no separate scene-change tuning). The default path is `config/prompts_config.yaml` (also referenced by `ObjectTracker.DEFAULT_PROMPTS_CONFIG`). Detection prompts render an `existing_objects_section` plus `prior_images_section` (attached recent frames with per-object positions) so Gemini reuses existing IDs when objects persist and only introduces new IDs when necessary.

## File Layout (`config/prompts_config.yaml`)
- `detection.streaming` – scene-level object discovery (normalized `[y, x, y2, x2]` in the 0–1000 range). Streaming is the only detection path (batch prompt exists in the file but is not used in code).
- `analysis.fast_mode` / `analysis.cached_mode` / `analysis.full` – affordances, interaction points, properties, optional PDDL predicates.
- `interaction.update` – single-affordance interaction point refinement.
- `pddl.section_template` / `pddl.example_template` – injected when predicates are provided.

## Usage (code-backed)
```python
from src.perception.object_tracker import ObjectTracker

tracker = ObjectTracker(
    api_key=api_key,
    fast_mode=False,
    pddl_predicates=["clean", "opened"],
    prompts_config_path="config/prompts_config.yaml",  # optional override
)
print(tracker.prompts["analysis"]["full"][:80])  # confirm load
```

Template variables supported:
- `{object_name}`, `{crop_note}`, `{pddl_section}`, `{pddl_example}`
- `{cached_affordances}`, `{object_type}` (cached mode)
- Interaction update: `{affordance}`, `{object_id}`, `{task_context_section}`

## Guardrails
- Keep the YAML prompt config under version control; document why changes were made in the journal with timestamp + base commit hash.
- Match prompt expectations with coordinate conventions (normalized 0–1000) and the current Gemini model (`gemini-robotics-er-1.5-preview`).
- When adding new prompt variables, update the tracker formatting code and this doc together.
- Test prompt edits with `uv run python examples/object_tracker_demo.py` or a targeted script before merging.

Keep this doc synchronized with `config/prompts_config.yaml` and `src/perception/object_tracker.py`.
