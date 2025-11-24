# Perception System

Perception is built around Gemini Robotics-ER prompts loaded from `config/prompts_config.yaml`. The stack detects objects, affordances, interaction points, and optional PDDL predicates, then maintains a thread-safe registry consumed by the orchestrator, planners, and the primitives layer.

**Code**: `src/perception/`

## Components
- `object_tracker.py` – async detector using Gemini (`gemini-robotics-er-1.5-preview` by default). Loads prompts from YAML, supports `fast_mode`, PDDL predicate extraction, and reuses cached affordances.
- `continuous_tracker.py` – wraps `ObjectTracker` for background detection with `update_interval`, `scene_change_threshold`, and an `on_detection_complete` callback used by the orchestrator.
- `object_registry.py` – registry with snapshot-friendly fields (`observations`, `latest_observation`, `latest_position_*`); query by id/type/affordance and serialize to JSON.
- `utils/coordinates.py` – conversions between normalized `[y, x]` (0–1000), pixels, and 3D using depth + intrinsics.

## Usage
```python
import asyncio, os
from src.camera import RealSenseCamera
from src.perception import ObjectTracker, ContinuousObjectTracker

async def detect_once():
    camera = RealSenseCamera(enable_depth=True, auto_start=True)
    tracker = ObjectTracker(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        fast_mode=False,
        pddl_predicates=["clean", "opened"],
    )
    color, depth = camera.get_aligned_frames()
    intrinsics = camera.get_camera_intrinsics()
    objects = await tracker.detect_objects(color, depth, intrinsics)
    return [(o.object_id, list(o.affordances)) for o in objects]

async def continuous_loop():
    cam = RealSenseCamera(enable_depth=True, auto_start=True)
    co = ContinuousObjectTracker(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        update_interval=1.5,
        scene_change_threshold=0.15,
    )
    co.set_frame_provider(cam.get_aligned_frames)
    co.start()
    await asyncio.sleep(3.0)
    await co.stop()

asyncio.run(detect_once())
```

Run the demo: `uv run python examples/object_tracker_demo.py`.

## Coordinate Conventions
- VLM outputs normalized `[y, x]` in the `0–1000` range. Registry fields keep both the normalized points and 3D positions when depth + intrinsics are provided.
- `compute_3d_position(...)` back-projects normalized coordinates using intrinsics and a depth frame.
- Perception and primitives share this convention: the executor back-projects helper fields like `target_pixel_yx` from the same normalized scale.

## Integration Points
- The orchestrator uses `ContinuousObjectTracker` and emits registry v2.0 plus perception pool snapshots; `get_world_state_snapshot()` bundles `registry`, `snapshot_index`, and `last_snapshot_id` for downstream planners.
- `SkillDecomposer` filters relevant objects from the registry and attaches the latest snapshot bytes when prompting Gemini for primitive plans.
- `PrimitiveExecutor` reloads the same snapshots (color/depth/intrinsics) via `snapshot_index` to translate helper parameters into metric targets.

## Configuration + Environment
- API key: `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- Prompts: `config/prompts_config.yaml` (keep it in sync with `agents/design/prompts_configuration.md`).
- Tuning: `fast_mode`, `max_parallel_requests`, `crop_target_size`, `scene_change_threshold`, and `enable_affordance_caching` on the trackers.

Keep perception snippets in docs aligned with the concrete function signatures in `src/perception/object_tracker.py` and `continuous_tracker.py`.
