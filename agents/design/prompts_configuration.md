# Prompts Configuration

`ObjectTracker` and `ContinuousObjectTracker` always load prompts from YAML; do not inline prompt strings. `ContinuousObjectTracker` now subclasses `ObjectTracker` in the same module and reuses the exact prompt config (no separate scene-change tuning). The default path is `config/object_tracker_prompts.yaml` (also referenced by `ObjectTracker.DEFAULT_PROMPTS_CONFIG`). Detection prompts render an `existing_objects_section` plus `prior_images_section` (attached recent frames with per-object positions) so Gemini reuses existing IDs when objects persist and only introduces new IDs when necessary.

## File Layout (`config/object_tracker_prompts.yaml`)
- `detection.streaming` – scene-level object discovery (normalized `[ymin, xmin, ymax, xmax]` integers in the 0–1000 range). Streaming is the only detection path and is split into two YAML templates: `prior` (IDs + prior images, no detection) and `current` (current frame + detection instructions). The tracker now sends these as two explicit content turns to Gemini: prior images + prior prompt first (context only), then a second turn with ONLY the current frame appended last; detection must run solely on that final image. The current turn follows the Gemini cookbook style: return ONLY a JSON array of `{box_2d, label}` entries (no code fences), reuse existing IDs in the `label` field when visible, create descriptive labels for new instances, clamp to 25 objects, and keep boxes tight.
- `analysis.response_schema` – structured JSON contract enforced via `response_json_schema`; requires `object_type` and `position`, and allows `affordances` as a list of `{affordance, position, reasoning}` entries plus optional `predicates`.
- `analysis.fast_mode` / `analysis.cached_mode` / `analysis.full` – affordances with embedded interaction points, optional PDDL predicates. Each prompt now inlines the schema above and relies on structured generation instead of a single example blob.
- `interaction.update` – single-affordance interaction point refinement.
- `pddl.section_template` / `pddl.example_template` – injected when predicates are provided.

## Usage (code-backed)
```python
from src.perception.object_tracker import ObjectTracker

tracker = ObjectTracker(
    api_key=api_key,
    fast_mode=False,
    pddl_predicates=["clean", "opened"],
    prompts_config_path="config/object_tracker_prompts.yaml",  # optional override
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

Keep this doc synchronized with `config/object_tracker_prompts.yaml` and `src/perception/object_tracker.py`.
