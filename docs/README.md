# VLM Semantic Spatial Perception - Documentation

## Overview

This directory contains comprehensive documentation for the VLM Semantic Spatial Perception system, organized by subdirectory within the `src/` folder.

---

## Documentation by Module

### [Perception System](perception.md)

**Location**: [src/perception/](../src/perception/)

VLM-powered semantic scene understanding for robotic manipulation.

**Key Topics**:
- Object detection with Gemini Robotics-ER
- Dynamic affordance prediction
- Interaction point detection
- 3D spatial awareness with depth integration
- Coordinate systems (normalized, pixel, 3D world)
- PDDL state tracking
- Continuous tracking service
- Performance optimization

**Core Classes**: `ObjectTracker`, `DetectedObject`, `InteractionPoint`, `DetectedObjectRegistry`, `ContinuousObjectTracker`

---

### [Planning System](planning.md)

**Location**: [src/planning/](../src/planning/)

LLM-driven PDDL generation for task-aware robotic manipulation.

**Key Topics**:
- LLM-based task analysis
- Dynamic domain generation
- Incremental domain updates from observations
- Task state monitoring (explore/plan/refine)
- Fuzzy goal matching
- PDDL file generation
- Perception-planning integration

**Core Classes**: `PDDLRepresentation`, `LLMTaskAnalyzer`, `PDDLDomainMaintainer`, `TaskStateMonitor`

---

### [Camera System](camera.md)

**Location**: [src/camera/](../src/camera/)

Unified abstraction for RGB-D sensing with RealSense and webcam support.

**Key Topics**:
- Camera abstraction and interfaces
- RealSense depth camera integration
- Webcam RGB capture
- Camera intrinsics and calibration
- Coordinate transformations (2D ↔ 3D)
- Point cloud generation
- Depth visualization
- Factory functions and configuration

**Core Classes**: `BaseCamera`, `RealSenseCamera`, `WebcamCamera`, `CameraIntrinsics`, `CameraFrame`

---

## System Architecture

### High-Level Data Flow

```
┌────────────────────────────────────────────────┐
│              Camera System                     │
│  RealSense / Webcam → RGB + Depth + Intrinsics│
└─────────────────┬──────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────┐
│           Perception System                    │
│  VLM Detection → Objects + Affordances + PDDL  │
└─────────────────┬──────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────┐
│            Planning System                     │
│  PDDL Domain Maintenance → domain.pddl/problem │
└────────────────────────────────────────────────┘
```

### Integration Points

1. **Camera → Perception**:
   - Camera provides: RGB images, depth maps, intrinsics
   - Perception uses: Images for VLM detection, depth for 3D positions
   - See: [Camera Integration](camera.md#perception-integration), [Coordinate Systems](perception.md#coordinate-systems)

2. **Perception → Planning**:
   - Perception provides: Detected objects with PDDL states
   - Planning uses: Object instances for domain, states for initial conditions
   - See: [PDDL Integration](perception.md#pddl-integration), [Perception Integration](planning.md#perception-integration)

3. **Planning → Perception** (Feedback Loop):
   - Planning provides: Required predicates from task analysis
   - Perception uses: Seeds tracking with specific predicates
   - See: [Planning System Integration](perception.md#planning-system-integration)

---

## Quick Start

### Basic Detection Pipeline

```python
from src.camera import RealSenseCamera
from src.perception import ObjectTracker

# Initialize
camera = RealSenseCamera(enable_depth=True, auto_start=True)
tracker = ObjectTracker(api_key=api_key)

# Capture and detect
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()
objects = await tracker.detect_objects(color, depth, intrinsics)

# Query results
for obj in objects:
    print(f"{obj.object_id}: {obj.affordances}")
```

### Full Task Execution

```python
from src.camera import RealSenseCamera
from src.perception import ObjectTracker
from src.planning import (
    PDDLRepresentation,
    PDDLDomainMaintainer,
    TaskStateMonitor,
    TaskState
)

# Initialize
camera = RealSenseCamera(enable_depth=True, auto_start=True)
tracker = ObjectTracker(api_key=api_key)
pddl = PDDLRepresentation(domain_name="task")
maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)
monitor = TaskStateMonitor(maintainer, pddl)

# Analyze task
task = "Clean the dirty mug and place it on the shelf"
analysis = await maintainer.initialize_from_task(task)

# Seed perception
tracker.set_pddl_predicates(analysis.required_predicate_types)

# Exploration loop
while True:
    decision = await monitor.determine_state()

    if decision.state == TaskState.EXPLORE:
        color, depth = camera.get_aligned_frames()
        intrinsics = camera.get_camera_intrinsics()
        objects = await tracker.detect_objects(color, depth, intrinsics)
        await maintainer.update_from_observations(objects)

    elif decision.state == TaskState.PLAN_AND_EXECUTE:
        await maintainer.set_goal_from_task_analysis()
        domain_path, problem_path = await pddl.generate_files_async("outputs/pddl")
        break

    elif decision.state == TaskState.REFINE_DOMAIN:
        await maintainer.refine_domain_from_observations(tracker.get_all_objects())
```

See individual module documentation for detailed examples.

---

## Configuration Files

### Camera Configuration

**File**: [config/camera_config.yaml](../config/camera_config.yaml)

```yaml
default_camera: 'realsense'  # or 'webcam'

realsense:
  resolution:
    width: 640
    height: 480
  framerate: 30
  depth_enabled: true
```

### Perception Configuration

**File**: [config/perception_config.yaml](../config/perception_config.yaml)

```yaml
detection_rate: 5  # Hz
max_objects: 30
confidence_threshold: 0.6
async_processing: true
```

### Planning Configuration

**File**: [config/planning_config.yaml](../config/planning_config.yaml)

```yaml
planner:
  algorithm: "lama"
  timeout: 30

domain:
  min_observations: 3
  max_observations: 30
```

---

## Common Workflows

### 1. Object Detection and 3D Localization

**Goal**: Detect objects in scene with 3D positions

**Modules**: Camera + Perception

**Steps**:
1. Capture RGB + depth with RealSense
2. Run VLM detection with depth integration
3. Query objects by type or affordance
4. Use 3D positions for spatial reasoning

**Documentation**: [Perception - Detection Pipeline](perception.md#detection-pipeline), [Camera - Coordinate Systems](camera.md#coordinate-systems)

### 2. Task-Driven PDDL Generation

**Goal**: Generate PDDL from natural language task

**Modules**: All three (Camera + Perception + Planning)

**Steps**:
1. Analyze task with LLM to extract requirements
2. Seed perception with required predicates
3. Explore environment and detect objects
4. Incrementally update PDDL domain
5. Generate domain.pddl and problem.pddl files

**Documentation**: [Planning - Complete Planning Pipeline](planning.md#complete-planning-pipeline)

### 3. Continuous Object Tracking

**Goal**: Maintain up-to-date object registry in background

**Modules**: Camera + Perception

**Steps**:
1. Initialize ContinuousObjectTracker
2. Set frame provider from camera
3. Start background tracking loop
4. Query registry from other threads
5. Optional: Set callbacks for detection events

**Documentation**: [Perception - Continuous Tracking](perception.md#continuous-tracking)

### 4. Point Cloud Generation

**Goal**: Generate 3D point cloud from RGB-D

**Modules**: Camera

**Steps**:
1. Capture aligned RGB + depth frames
2. Get camera intrinsics
3. Use depth_image_to_point_cloud utility
4. Optionally add RGB color to points

**Documentation**: [Camera - 3D Conversion Utilities](camera.md#3d-conversion-utilities)

---

## Troubleshooting

### Common Issues

Each module documentation includes a dedicated troubleshooting section:

- **Perception Issues**: [Troubleshooting](perception.md#troubleshooting)
  - No objects detected
  - Incorrect 3D positions
  - Slow performance
  - API rate limiting

- **Planning Issues**: [Troubleshooting](planning.md#troubleshooting)
  - Domain not completing
  - Invalid PDDL generated
  - LLM analysis errors

- **Camera Issues**: [Troubleshooting](camera.md#troubleshooting)
  - RealSense not detected
  - Depth alignment issues
  - Invalid depth values
  - Webcam wrong device
  - Low frame rate

---

## Performance Considerations

### Perception System

- **Full detection (10 objects)**: ~10-20 seconds
- **Fast mode**: 2-3x speedup (skip interaction points)
- **Optimization**: Affordance caching, parallel processing, crop resizing

See: [Perception - Performance](perception.md#performance)

### Planning System

- **Task analysis**: ~2-3 seconds (LLM call)
- **Observation update**: <100ms
- **Domain refinement**: ~2-3 seconds (LLM call)
- **Optimization**: Response caching (5-min TTL), incremental updates

See: [Planning - Performance Characteristics](planning.md#performance-characteristics)

### Camera System

- **RealSense frame capture**: ~30-60 FPS (depends on resolution)
- **Hardware alignment**: GPU-accelerated
- **Point cloud generation**: Real-time for 640x480

---

## API Keys and Environment

### Required Environment Variables

```bash
# Google Gemini API key (required for perception and planning)
export GOOGLE_API_KEY="your_api_key_here"
```

### Optional Configuration

```bash
# Custom config paths
export CAMERA_CONFIG="path/to/camera_config.yaml"
export PERCEPTION_CONFIG="path/to/perception_config.yaml"
export PLANNING_CONFIG="path/to/planning_config.yaml"
```

---

## Examples

Example scripts demonstrating system usage:

- **Object Detection**: `examples/object_tracker_demo.py`
- **Task Planning**: `examples/planning_demo.py`
- **Camera Capture**: `examples/camera_demo.py`

---

## Future Enhancements

### Planned Features

**Perception**:
- Multi-frame tracking (identity tracking across frames)
- Semantic relationships (spatial scene graphs)
- Active perception (viewpoint suggestions)
- Hierarchical object models (part-based)

**Planning**:
- Replanning on failures
- Hierarchical task decomposition
- Plan execution monitoring
- Multi-agent coordination

**Camera**:
- Custom calibration loading
- Depth post-processing filters
- Multi-camera synchronization
- Recording and playback

---

## Additional Resources

### External Documentation

- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [Google Gemini API](https://ai.google.dev/docs)
- [PDDL Documentation](https://planning.wiki/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Module READMEs

- [Perception README](../src/perception/README.md)
- [Planning README](../src/planning/README.md)
- [Camera README](../src/camera/README.md)

### Main Project README

- [Project README](../README.md)

---

## Contact & Support

For questions or issues:
1. Check the module-specific documentation above
2. Review troubleshooting sections
3. Check example scripts
4. File an issue on the project repository

---

*Last Updated: November 10, 2025*
*Version: 1.0*
