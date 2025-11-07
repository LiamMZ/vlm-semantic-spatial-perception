# VLM Semantic Spatial Perception System

A comprehensive Vision Language Model (VLM) powered spatial perception system for robotics with PDDL planning integration. This system provides task-conditioned scene understanding, spatial relationship extraction, and automatic PDDL generation for robotic manipulation tasks.

## Features

- **Multi-Camera Support**: Abstraction layer supporting webcams and Intel RealSense depth cameras
- **Object Tracking System**: Efficient parallel detection with affordances and interaction points
  - Detects object names, then analyzes each object in parallel
  - Per-affordance interaction points (grasp, pour, push, etc.)
  - Task-conditioned interaction point refinement
  - 3D position tracking with depth integration
  - Object registry for querying and tracking
- **Gemini Robotics Integration**: Full integration with Gemini Robotics-ER 1.5 for advanced robotic perception
  - Dynamic affordance inference from visual observation (no hard-coded mappings)
  - Spatial reasoning and relationship analysis
  - Trajectory planning with obstacle avoidance
  - Task decomposition into executable subtasks
  - Interaction point detection with visualization
- **VLM-Powered Perception**: Task-aware object detection with semantic scene understanding
- **Spatial Relationship Extraction**: Automatic detection of spatial relationships (on, near, above, etc.)
- **Task-Conditioned Processing**: Perception focused on task-relevant objects and affordances
- **World Model**: Unified representation combining object tracking and spatial relationships
- **Dynamic PDDL Generation**: LLM-powered generation of task-specific planning domain and problem files

## Installation

### Basic Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional Dependencies

```bash
# For Intel RealSense camera support
pip install pyrealsense2

# For 3D visualization
pip install open3d matplotlib plotly

# For development
pip install -e ".[dev]"
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
GEMINI_API_KEY=your_api_key_here
```

3. Configure camera and perception settings in `config/` directory

## Quick Start

### Basic Camera Usage

```python
from src.camera import create_camera_from_config, WebcamCamera

# Using webcam
camera = WebcamCamera(device_id=0, width=640, height=480)
camera.start()

frame = camera.capture_frame()  # RGB image
intrinsics = camera.get_camera_intrinsics()

camera.stop()
```

### Task Management

```python
from src.task import TaskManager, TaskParser

parser = TaskParser()
manager = TaskManager()

# Parse natural language task
task_desc = "Place the blue bottle on the shelf"
parsed = parser.parse(task_desc)

# Create and activate task
task = manager.create_task(task_desc, parsed)
manager.set_current_task(task.task_id)

# Get task context for perception
context = manager.get_task_context()
print(context["goal_objects"])      # ['bottle']
print(context["goal_predicates"])   # ['on(bottle, shelf)']
```

### World Model

```python
from src.world_model import WorldState, DetectedObject
import numpy as np

world = WorldState()

# Add detected objects
obj1 = DetectedObject(
    object_id="cup_0",
    object_type="cup",
    position=np.array([0.5, 0.2, 0.1]),
    confidence=0.9,
    timestamp=time.time(),
    affordances=["graspable", "containable"]
)

world.update([obj1])

# Query spatial relationships
relationships = world.get_all_relationships()

# Get PDDL representation
pddl_state = world.get_pddl_state()
```

### Gemini Robotics Perception

```python
from src.perception import GeminiRoboticsClient, VLMObjectDetector
from src.camera import RealSenseCamera

# Initialize Gemini client
client = GeminiRoboticsClient(
    api_key="your_gemini_api_key",
    model_name="gemini-2.0-flash-exp"
)

# Create detector
detector = VLMObjectDetector(
    gemini_client=client,
    confidence_threshold=0.5
)

# Capture scene
camera = RealSenseCamera(auto_start=True)
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

# Detect objects with dynamic affordances (no hard-coded mappings!)
objects = detector.detect(
    color_image=color,
    depth_image=depth,
    camera_intrinsics=intrinsics,
    task_context="Pick up tools and place in toolbox"
)

# Objects have VLM-inferred properties
for obj in objects:
    print(f"{obj.object_id}: {obj.object_type}")
    print(f"  Position: {obj.position}")
    print(f"  Affordances: {obj.affordances}")  # Inferred from visual observation!
    print(f"  Properties: {obj.properties}")
```

### Full Pipeline Example

See [examples/dynamic_pddl_demo.py](examples/dynamic_pddl_demo.py) for complete integration:

```bash
python examples/dynamic_pddl_demo.py
```

This demonstrates:
1. RealSense camera capture
2. Gemini VLM object detection with dynamic affordances
3. World model construction with spatial relationships
4. LLM-powered task analysis
5. Dynamic PDDL generation

## Project Structure

```
vlm-spatial-perception/
├── config/                 # Configuration files
│   ├── camera_config.yaml
│   ├── gemini.yaml        # Gemini Robotics configuration ✓
│   ├── perception_config.yaml
│   └── vlm_config.yaml
├── src/
│   ├── camera/            # Camera abstraction layer ✓
│   ├── perception/        # VLM perception pipeline ✓
│   │   ├── gemini_robotics.py    # Gemini Robotics-ER client ✓
│   │   └── vlm_detector.py       # World model integration ✓
│   ├── world_model/       # World state representation ✓
│   ├── planning/          # Dynamic PDDL generation ✓
│   ├── task/              # Task management ✓
│   └── utils/             # Utilities (TODO)
├── docs/
│   └── GEMINI_INTEGRATION.md    # Full Gemini integration guide ✓
├── outputs/               # Generated outputs (PDDL, logs)
├── tests/                 # Unit tests
└── examples/              # Example scripts
    ├── dynamic_pddl_demo.py          # Full pipeline demo ✓
    └── gemini_robotics_example.py    # All Gemini capabilities ✓
```

## Configuration

### Camera Configuration ([config/camera_config.yaml](config/camera_config.yaml))

```yaml
default_camera: 'webcam'  # or 'realsense'

webcam:
  device_id: 0
  resolution:
    width: 640
    height: 480
```

### Perception Configuration ([config/perception_config.yaml](config/perception_config.yaml))

```yaml
vlm:
  confidence_threshold: 0.6
  max_objects: 30

spatial:
  relationships:
    thresholds:
      near_distance: 0.3  # meters
```

## Development Status

### Completed Components

- ✓ Camera abstraction layer ([src/camera/](src/camera/))
  - Base camera interface
  - Webcam implementation
  - RealSense implementation
  - Camera utilities

- ✓ World model ([src/world_model/](src/world_model/))
  - Object registry with tracking
  - Spatial map and relationships
  - Unified world state

- ✓ Task management ([src/task/](src/task/))
  - Natural language task parser
  - Task manager with context

- ✓ Configuration files
  - Camera, VLM, Gemini, and perception configs

- ✓ Gemini Robotics integration ([src/perception/](src/perception/))
  - Gemini Robotics-ER 1.5 client
  - VLM object detector with dynamic affordances
  - Spatial reasoning and trajectory planning
  - Task decomposition

- ✓ Dynamic PDDL generation ([src/planning/](src/planning/))
  - LLM-powered task analyzer
  - Dynamic PDDL generator

### Remaining Components

- TODO: Main control loop (system orchestration)
- TODO: Utilities and visualization
- TODO: Comprehensive testing suite

## Documentation

- **[Gemini Integration Guide](docs/GEMINI_INTEGRATION.md)**: Complete guide to using Gemini Robotics-ER
  - API reference
  - Best practices
  - Performance optimization
  - Troubleshooting

- **[Architecture](ARCHITECTURE.md)**: System architecture and design decisions
- **[Implementation Status](IMPLEMENTATION_STATUS.md)**: Development progress tracking

## Examples

Run the examples to see the system in action:

```bash
# Full pipeline with VLM detection and PDDL generation
python examples/dynamic_pddl_demo.py

# Comprehensive Gemini capabilities demonstration
python examples/gemini_robotics_example.py
```

## Key Advantages

### Dynamic vs. Static Perception

Traditional systems use **hard-coded object mappings**:
```python
# ✗ Static approach
AFFORDANCES = {
    "cup": ["graspable", "containable"],
    "table": ["supportable"]
}
```

This system uses **VLM-inferred affordances**:
```python
# ✓ Dynamic approach - VLM observes and infers
objects = detector.detect(frame)
# VLM analyzes visual features:
# - Shape, size, handles → graspability
# - Openings, cavities → containability
# - Flat surfaces → supportability
# - Current state (open/closed, empty/full)
```

### Benefits

1. **No Pre-Programming**: Works with novel objects never seen before
2. **Context-Aware**: Detection adapts to task requirements
3. **Observation-Based**: Affordances from visual analysis, not categories
4. **Flexible**: Handles variations in object appearance and state
5. **Scalable**: No manual mapping maintenance required

## Performance

- **Camera Capture**: <0.5s
- **VLM Object Detection**: 3-10s (depends on scene complexity)
- **LLM Task Analysis**: 2-5s per task
- **PDDL Generation**: <0.1s
- **Spatial Relationships**: <0.1s

Optimization tips in [Gemini Integration Guide](docs/GEMINI_INTEGRATION.md).

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vlm_spatial_perception,
  title = {VLM Semantic Spatial Perception System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/vlm-semantic-spatial-perception}
}
```

## Contributing

Contributions welcome! Please see contribution guidelines.

## Support

For questions or issues:
1. Check [documentation](docs/)
2. Review [examples](examples/)
3. Open an issue on GitHub