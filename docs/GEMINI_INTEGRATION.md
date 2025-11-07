# Gemini Robotics Integration Guide

This guide explains how to use the Gemini Robotics-ER integration in the VLM Semantic Spatial Perception system.

## Overview

The system integrates [Gemini Robotics-ER 1.5](https://ai.google.dev/gemini-api/docs/robotics-overview), Google's vision-language model designed for advanced robotic reasoning. This integration provides:

- **Object Detection**: Detect objects with 2D coordinates and dynamically inferred affordances
- **Spatial Reasoning**: Analyze spatial relationships and scene layout
- **Trajectory Planning**: Plan collision-free paths for robot movement
- **Task Decomposition**: Break down natural language tasks into executable subtasks
- **Video Tracking**: Track objects across video frames

## Key Features

### 1. Dynamic Affordance Inference

**No hard-coded mappings** - All affordances are inferred from visual observation:

```python
from src.perception import GeminiRoboticsClient, VLMObjectDetector

client = GeminiRoboticsClient(api_key="your_key")
detector = VLMObjectDetector(client)

objects = detector.detect(
    color_image=frame,
    depth_image=depth,
    camera_intrinsics=intrinsics,
    task_context="Pick up tools for assembly"
)

# Objects have VLM-inferred affordances
for obj in objects:
    print(f"{obj.object_id}: {obj.affordances}")
    # e.g., "wrench": ["graspable", "tool", "rotatable"]
```

The VLM analyzes:
- Physical shape and size → graspability
- Openings and cavities → containability
- Flat surfaces → supportability
- Handles, lids, doors → openability
- Container properties → pourability

### 2. Task-Aware Detection

Detection adapts to the task context:

```python
# General detection
objects = detector.detect(frame, task_context=None)

# Task-specific detection
objects = detector.detect(
    frame,
    task_context="Find all items that need to be sorted into bins"
)
# VLM focuses on relevant objects and infers task-specific properties
```

### 3. Spatial Reasoning

Analyze scene layout and relationships:

```python
spatial_result = client.analyze_spatial_relationships(
    image,
    query="Which objects can the robot reach without moving obstacles?",
    focus_objects=["cup", "bottle", "box"]
)

print(spatial_result.reasoning)
# "The cup is closest to the robot at the front-left.
#  The bottle is behind the cup and requires moving it first..."

for rel in spatial_result.relationships:
    print(rel)
    # {"object1": "cup", "relation": "in-front-of", "object2": "bottle"}
```

### 4. Task Decomposition

Break down complex tasks:

```python
result = client.decompose_task(
    task_description="Put the apple in the bowl",
    image=scene_image,
    available_actions=["navigate", "grasp", "place"]
)

for subtask in result.subtasks:
    print(f"{subtask['id']}: {subtask['action']} - {subtask['description']}")
    # 0: navigate - Move to apple location
    # 1: grasp - Pick up the apple
    # 2: navigate - Move to bowl location
    # 3: place - Place apple in bowl
```

## Architecture

### Module Structure

```
src/perception/
├── __init__.py
├── gemini_robotics.py     # Core Gemini client
└── vlm_detector.py         # Integration with world model
```

### Integration Flow

```
Camera → RGB-D Frame → VLMObjectDetector → DetectedObject[] → WorldState
                              ↓
                       GeminiRoboticsClient
                              ↓
                       [Object Detection]
                       [Spatial Reasoning]
                       [Task Decomposition]
```

### Data Flow

1. **Camera Capture**: RGB-D frames from RealSense or webcam
2. **VLM Detection**: Gemini analyzes image, returns objects with properties
3. **3D Projection**: 2D positions + depth → 3D coordinates
4. **World Model**: DetectedObject instances with VLM-inferred affordances
5. **Spatial Map**: Relationships computed from 3D positions
6. **Planning**: PDDL generation uses observed affordances and relationships

## API Reference

### GeminiRoboticsClient

Main interface to Gemini Robotics-ER:

```python
client = GeminiRoboticsClient(
    api_key="your_key",
    model_name="gemini-2.0-flash-exp",  # or "gemini-1.5-pro"
    default_temperature=0.1,
    enable_thinking=True,
    thinking_budget=1024
)
```

#### Methods

**detect_objects(image, query, temperature, return_json)**
- Detect objects with 2D positions and properties
- Returns: `ObjectDetectionResult`

**analyze_spatial_relationships(image, query, focus_objects, temperature)**
- Analyze spatial layout and relationships
- Returns: `SpatialReasoningResult`

**plan_trajectory(image, start_point, end_point, query, temperature)**
- Plan collision-free path
- Returns: `TrajectoryResult`

**decompose_task(task_description, image, available_actions, temperature)**
- Break task into subtasks
- Returns: `TaskDecompositionResult`

**track_objects_in_video(video_frames, query, temperature)**
- Track objects across frames
- Returns: Dictionary with tracking data

### VLMObjectDetector

Integrates Gemini with the world model:

```python
detector = VLMObjectDetector(
    gemini_client=client,
    confidence_threshold=0.5,
    use_thinking=True,
    max_objects=50
)
```

#### Methods

**detect(color_image, depth_image, camera_intrinsics, query, task_context)**
- Detect objects and convert to DetectedObject instances
- Returns: `List[DetectedObject]`

**detect_with_spatial_context(color_image, depth_image, camera_intrinsics, task)**
- Detect objects + spatial analysis in one call
- Returns: `(List[DetectedObject], Dict[str, Any])`

## Configuration

Edit [config/gemini.yaml](../config/gemini.yaml) to configure:

```yaml
model:
  name: "gemini-2.0-flash-exp"
  temperature: 0.1

detection:
  confidence_threshold: 0.5
  max_objects: 50
  include_affordances: true

cache:
  enabled: true
  ttl: 300
```

## Examples

### Example 1: Basic Object Detection

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

for obj in objects:
    print(f"{obj.object_id}: {obj.object_type} at {obj.position}")
    print(f"  Affordances: {obj.affordances}")
```

### Example 2: Task-Aware Pipeline

```python
from src.perception import GeminiRoboticsClient, VLMObjectDetector
from src.world_model import WorldState

# Initialize
client = GeminiRoboticsClient(api_key="your_key")
detector = VLMObjectDetector(client)
world = WorldState()

# User task
task = "Pick up the red cup and place it on the shelf"

# Detect with task context
objects = detector.detect(
    color_frame,
    depth_frame,
    camera_intrinsics,
    task_context=task
)

# Update world model
world.update(objects)

# Get task-relevant objects
cups = world.get_objects_by_type("cup")
red_cups = [obj for obj in cups if obj.color == "red"]

print(f"Found {len(red_cups)} red cup(s)")
```

### Example 3: Full Pipeline Integration

See [examples/dynamic_pddl_demo.py](../examples/dynamic_pddl_demo.py) for complete integration with:
- Camera capture
- VLM detection
- World model
- Task analysis
- PDDL generation

### Example 4: All Capabilities

See [examples/gemini_robotics_example.py](../examples/gemini_robotics_example.py) for demonstrations of:
- Object detection
- Spatial reasoning
- Trajectory planning
- Task decomposition
- Video tracking

## Best Practices

### 1. Prompt Design

**Good prompts are specific and observational:**

```python
# ✓ Good - specific, asks for observation
query = """Detect all objects on the table.
For each object, observe:
- Can it be grasped based on its size and shape?
- Does it have openings or cavities that could contain things?
- What is its current state (open/closed, empty/full)?
"""

# ✗ Bad - vague, assumes categories
query = "Find all objects"
```

### 2. Task Context

Always provide task context when available:

```python
# ✓ With context - VLM focuses on relevant objects
objects = detector.detect(
    frame,
    task_context="Sort tools into the red toolbox"
)

# ✗ Without context - may detect irrelevant objects
objects = detector.detect(frame)
```

### 3. Confidence Thresholding

Adjust confidence based on task criticality:

```python
# High-stakes tasks - high threshold
detector = VLMObjectDetector(client, confidence_threshold=0.8)

# Exploratory tasks - lower threshold
detector = VLMObjectDetector(client, confidence_threshold=0.3)
```

### 4. Caching

Enable caching for repeated queries:

```python
client = GeminiRoboticsClient(api_key="your_key")
client.set_cache_enabled(True)

# First call - hits API
result1 = client.detect_objects(image, "Find cups")

# Second call with same image - uses cache
result2 = client.detect_objects(image, "Find cups")
```

### 5. Error Handling

Always handle API failures gracefully:

```python
try:
    objects = detector.detect(frame, depth, intrinsics)
except Exception as e:
    print(f"Detection failed: {e}")
    # Fallback to mock data or previous state
    objects = world.get_all_objects()  # Use previous observations
```

## Performance

### Typical Latencies

- **Object Detection**: 3-10s (depends on scene complexity)
- **Spatial Reasoning**: 2-5s
- **Task Decomposition**: 2-5s
- **Trajectory Planning**: 3-8s

### Optimization Tips

1. **Use Flash Model**: `gemini-2.0-flash-exp` is 5x faster than Pro
2. **Enable Caching**: Reuse results for identical queries
3. **Reduce Image Size**: Resize images to 640x480 or 1280x720
4. **Limit Max Objects**: Set `max_objects=20` for faster processing
5. **Adjust Thinking Budget**: Lower budget for simple detection tasks

### Cost Management

- Flash model: ~$0.001 per detection
- Pro model: ~$0.01 per detection
- Cache hits: Free (no API call)
- Consensus runs: Multiply cost by number of runs

## Troubleshooting

### Issue: No objects detected

**Causes:**
- Image quality (too dark, blurry, low resolution)
- Confidence threshold too high
- Task context too restrictive

**Solutions:**
```python
# Lower confidence threshold
detector = VLMObjectDetector(client, confidence_threshold=0.3)

# Remove task context
objects = detector.detect(frame, task_context=None)

# Improve image quality
frame = cv2.resize(frame, (1280, 720))
```

### Issue: Incorrect affordances

**Causes:**
- Ambiguous visual appearance
- Occluded objects
- Poor lighting

**Solutions:**
```python
# Provide more specific detection query
query = """For each object, carefully observe:
- Its shape, size, and handles to determine graspability
- Any openings, lids, or cavities for containability
- Surface flatness for supportability
"""

objects = detector.detect(frame, query=query)
```

### Issue: Slow performance

**Solutions:**
```python
# Use Flash model instead of Pro
client = GeminiRoboticsClient(
    api_key="your_key",
    model_name="gemini-2.0-flash-exp"  # Fast model
)

# Reduce image resolution
frame_small = cv2.resize(frame, (640, 480))

# Enable caching
client.set_cache_enabled(True)
```

## Limitations

1. **Preview Status**: Gemini Robotics-ER is in preview, APIs may change
2. **Latency**: 3-10s per detection, not suitable for real-time control loops
3. **Cost**: API calls cost money, use caching to minimize
4. **Hallucinations**: VLM may hallucinate objects in ambiguous scenes
5. **2D Positions**: Native output is 2D, requires depth for 3D

## Future Enhancements

- [ ] Function calling for robot action execution
- [ ] Fine-tuning on specific object types
- [ ] Multi-view fusion for better 3D understanding
- [ ] Real-time streaming mode
- [ ] Integration with grasp planning
- [ ] Video understanding for dynamic scenes

## References

- [Gemini Robotics Overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [Agentic Capabilities](https://ai.google.dev/gemini-api/docs/robotics-overview#agentic-capabilities)
- [Google AI Python SDK](https://github.com/google/generative-ai-python)

## Support

For issues or questions:
1. Check this documentation
2. Review [examples/](../examples/)
3. Check Gemini API documentation
4. Open an issue on GitHub
