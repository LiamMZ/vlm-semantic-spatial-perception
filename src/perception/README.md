# Perception Module

Vision-Language Model (VLM) based perception for robotic scene understanding.

## Overview

This module provides interfaces to VLMs for:
- Object detection with dynamic affordance inference
- Spatial reasoning and relationship analysis
- Trajectory planning
- Task decomposition

**Key principle**: All semantic information is **dynamically generated** from visual observations, not hard-coded.

## Components

### [gemini_robotics.py](gemini_robotics.py)

Core client for Gemini Robotics-ER 1.5:

```python
from src.perception import GeminiRoboticsClient

client = GeminiRoboticsClient(api_key="your_key")

# Detect objects
result = client.detect_objects(image, "Find all graspable objects")

# Spatial reasoning
spatial = client.analyze_spatial_relationships(image, "What's closest to the robot?")

# Task decomposition
tasks = client.decompose_task("Pick up the cup", image=image)

# Trajectory planning
path = client.plan_trajectory(image, start=[100, 100], end=[800, 600])
```

### [vlm_detector.py](vlm_detector.py)

Integration layer between Gemini and the world model:

```python
from src.perception import VLMObjectDetector

detector = VLMObjectDetector(gemini_client)

# Returns DetectedObject instances compatible with WorldState
objects = detector.detect(
    color_image=frame,
    depth_image=depth,
    camera_intrinsics=intrinsics,
    task_context="Organize workspace"
)

# Update world model
world.update(objects)
```

## Dynamic Affordance Inference

**No hard-coded mappings.** Affordances are inferred from visual observation:

```python
# VLM analyzes the scene and infers:
# - Can it be grasped? (shape, size, handles)
# - Can it contain things? (openings, cavities)
# - Can it support objects? (flat surfaces)
# - Can it be opened? (doors, lids, drawers)
# - Is it movable? (size, attachment points)

objects = detector.detect(frame)

for obj in objects:
    print(f"{obj.object_type}: {obj.affordances}")
    # "cup": ["graspable", "containable", "pourable"]
    # "table": ["supportable", "static"]
    # "drawer": ["openable", "containable", "graspable"]
```

## Task-Aware Detection

Detection adapts to task context:

```python
# Task-specific detection
objects = detector.detect(
    frame,
    task_context="Find tools for assembly"
)
# VLM focuses on tools and infers tool-specific affordances
```

## Usage Examples

### Basic Detection

```python
from src.perception import GeminiRoboticsClient, VLMObjectDetector
from src.camera import RealSenseCamera

# Setup
client = GeminiRoboticsClient(api_key="your_key")
detector = VLMObjectDetector(client)
camera = RealSenseCamera(auto_start=True)

# Capture and detect
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

objects = detector.detect(color, depth, intrinsics)
```

### With World Model

```python
from src.perception import VLMObjectDetector
from src.world_model import WorldState

detector = VLMObjectDetector(client)
world = WorldState()

# Detect and update
objects = detector.detect(frame, depth, intrinsics, task_context="Clean the table")
world.update(objects)

# Query world state
graspable = world.get_objects_with_affordance("graspable")
on_table = [obj for obj in graspable if "on" in [rel.relation_type.value for rel in world.spatial_map.get_relationships(obj.object_id)]]
```

### Full Pipeline

```python
from src.perception import GeminiRoboticsClient, VLMObjectDetector
from src.world_model import WorldState
from src.planning import DynamicPDDLGenerator, LLMTaskAnalyzer

# Initialize components
gemini_client = GeminiRoboticsClient(api_key="your_key")
detector = VLMObjectDetector(gemini_client)
world = WorldState()
analyzer = LLMTaskAnalyzer(api_key="your_key")
generator = DynamicPDDLGenerator(analyzer)

# User task
task = "Put the red cup on the shelf"

# 1. Detect objects (task-aware)
objects = detector.detect(frame, depth, intrinsics, task_context=task)
world.update(objects)

# 2. Generate PDDL from observed scene
world_dict = {
    "objects": [obj.to_dict() for obj in world.get_all_objects()],
    "relationships": [rel.to_predicate() for rel in world.get_all_relationships()]
}

result = generator.generate(task, world_dict)
print(f"Generated: {result['domain_path']}, {result['problem_path']}")
```

## Configuration

See [config/gemini.yaml](../../config/gemini.yaml):

```yaml
model:
  name: "gemini-2.0-flash-exp"
  temperature: 0.1

detection:
  confidence_threshold: 0.5
  max_objects: 50
  include_affordances: true
```

## Performance

- **Detection**: 3-10s per frame
- **Spatial reasoning**: 2-5s
- **Task decomposition**: 2-5s

Tips for optimization:
- Use `gemini-2.0-flash-exp` for speed
- Enable caching for repeated queries
- Resize images to 640x480
- Limit max_objects to 20-30

## Documentation

See [docs/GEMINI_INTEGRATION.md](../../docs/GEMINI_INTEGRATION.md) for:
- Complete API reference
- Best practices
- Performance optimization
- Troubleshooting

## Examples

- [examples/gemini_robotics_example.py](../../examples/gemini_robotics_example.py) - All capabilities
- [examples/dynamic_pddl_demo.py](../../examples/dynamic_pddl_demo.py) - Full pipeline integration
