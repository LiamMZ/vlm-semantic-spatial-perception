# Task Orchestrator

The **Task Orchestrator** is the main production-ready class for managing the complete perception-planning pipeline. It provides a unified interface for task execution with state persistence, continuous detection, and PDDL generation.

## Features

- **Task Request Processing**: Accepts natural language tasks and automatically sets up PDDL domain
- **Continuous Object Detection**: Background object tracking with automatic domain updates
- **State Persistence**: Save and load complete system state (domain, problem, objects)
- **Task State Monitoring**: Intelligent monitoring to determine when ready for planning
- **Lifecycle Management**: Initialize, run, pause, resume, and shutdown capabilities
- **Auto-save**: Automatic state persistence at configurable intervals
- **Callbacks**: Extensible with custom callbacks for state changes and updates

## Quick Start

```python
import asyncio
import os
from pathlib import Path
from src.planning import TaskOrchestrator, OrchestratorConfig

async def main():
    # Configure orchestrator
    config = OrchestratorConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        update_interval=2.0,  # Detection every 2 seconds
        min_observations=3,   # Require 3+ objects before planning
        auto_save=True,       # Enable auto-save
        state_dir=Path("outputs/orchestrator_state")
    )

    # Create and initialize
    orchestrator = TaskOrchestrator(config)
    await orchestrator.initialize()

    # Process task request
    await orchestrator.process_task_request("make a cup of coffee")

    # Start continuous detection
    await orchestrator.start_detection()

    # Wait for system to be ready
    while not orchestrator.is_ready_for_planning():
        await asyncio.sleep(1.0)
        status = await orchestrator.get_status()
        print(f"State: {status['task_state']['state']}")

    # Generate PDDL files
    paths = await orchestrator.generate_pddl_files()
    print(f"Domain: {paths['domain_path']}")
    print(f"Problem: {paths['problem_path']}")

    # Cleanup
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

### Components

The orchestrator integrates the following components:

1. **RealSense Camera**: Captures RGB-D frames
2. **ContinuousObjectTracker**: Background object detection
3. **DetectedObjectRegistry**: Thread-safe object storage
4. **PDDLRepresentation**: Domain and problem management
5. **PDDLDomainMaintainer**: Updates domain from observations
6. **TaskStateMonitor**: Determines readiness for planning

### State Machine

The orchestrator operates as a state machine with the following states:

- `UNINITIALIZED`: Not yet initialized
- `IDLE`: Ready for task requests
- `ANALYZING_TASK`: Processing task description
- `DETECTING`: Running continuous detection
- `PAUSED`: Detection paused, can resume
- `READY_FOR_PLANNING`: Sufficient observations, ready for planner
- `EXECUTING_PLAN`: Plan is being executed
- `TASK_COMPLETE`: Task successfully completed
- `ERROR`: Error state

## Configuration

### OrchestratorConfig

```python
config = OrchestratorConfig(
    # API Configuration
    api_key="your-api-key",
    model_name="gemini-2.5-flash",

    # Camera Configuration
    camera_width=640,
    camera_height=480,
    camera_fps=30,
    enable_depth=True,

    # Detection Configuration
    update_interval=2.0,              # Seconds between detections
    min_observations=3,               # Minimum objects before planning
    fast_mode=False,                  # Skip interaction points for speed
    scene_change_threshold=0.15,      # Scene change detection threshold
    enable_scene_change_detection=True,

    # Persistence Configuration
    state_dir=Path("outputs/orchestrator_state"),
    auto_save=True,                   # Auto-save state
    auto_save_interval=30.0,          # Auto-save every 30 seconds

    # Task Configuration
    exploration_timeout=60.0,         # Max exploration time

    # Callbacks
    on_state_change=my_state_callback,
    on_detection_update=my_detection_callback,
    on_task_state_change=my_task_callback
)
```

## API Reference

### Lifecycle Management

#### `initialize()`
Initialize the orchestrator and all subsystems. Must be called before use.

```python
await orchestrator.initialize()
```

#### `shutdown()`
Shutdown the orchestrator and cleanup resources.

```python
await orchestrator.shutdown()
```

### Task Management

#### `process_task_request(task_description, environment_image=None)`
Process a new task request. Analyzes the task and initializes PDDL domain.

```python
task_analysis = await orchestrator.process_task_request(
    "pick up the red mug and place it on the shelf"
)
print(f"Goal objects: {task_analysis.goal_objects}")
```

#### `update_task(new_task_description)`
Update the current task without resetting observations.

```python
await orchestrator.update_task("now place the mug in the dishwasher")
```

### Detection Management

#### `start_detection()`
Start continuous object detection in the background.

```python
await orchestrator.start_detection()
```

#### `stop_detection()`
Stop continuous object detection.

```python
await orchestrator.stop_detection()
```

#### `pause_detection()`
Pause detection (can be resumed later).

```python
await orchestrator.pause_detection()
```

#### `resume_detection()`
Resume paused detection.

```python
await orchestrator.resume_detection()
```

### Status and Monitoring

#### `is_ready_for_planning()`
Check if system is ready for PDDL planning.

```python
if orchestrator.is_ready_for_planning():
    await orchestrator.generate_pddl_files()
```

#### `get_status()`
Get comprehensive status of the orchestrator.

```python
status = await orchestrator.get_status()
print(f"State: {status['orchestrator_state']}")
print(f"Objects detected: {status['registry']['num_objects']}")
print(f"Task state: {status['task_state']['state']}")
```

#### `get_task_decision()`
Get current task state decision with reasoning.

```python
decision = await orchestrator.get_task_decision()
print(f"State: {decision.state.value}")
print(f"Confidence: {decision.confidence:.1%}")
print(f"Reasoning: {decision.reasoning}")
print(f"Blockers: {decision.blockers}")
```

#### `get_detected_objects()`
Get all detected objects from registry.

```python
objects = orchestrator.get_detected_objects()
for obj in objects:
    print(f"{obj.object_id}: {obj.affordances}")
```

#### `get_objects_by_type(object_type)`
Get objects of a specific type.

```python
cups = orchestrator.get_objects_by_type("cup")
```

#### `get_objects_with_affordance(affordance)`
Get objects with a specific affordance.

```python
graspable = orchestrator.get_objects_with_affordance("graspable")
```

### PDDL Generation

#### `generate_pddl_files(output_dir=None, set_goals=True)`
Generate PDDL domain and problem files.

```python
paths = await orchestrator.generate_pddl_files(
    output_dir=Path("outputs/pddl"),
    set_goals=True
)
print(f"Domain: {paths['domain_path']}")
print(f"Problem: {paths['problem_path']}")
```

### State Persistence

#### `save_state(path=None)`
Save orchestrator state to disk.

```python
state_path = await orchestrator.save_state()
print(f"State saved to: {state_path}")
```

#### `load_state(path=None)`
Load orchestrator state from disk.

```python
await orchestrator.load_state()
print(f"State loaded, {len(orchestrator.registry)} objects restored")
```

## Callbacks

### State Change Callback

Called when orchestrator state changes.

```python
def on_state_change(old_state, new_state):
    print(f"State changed: {old_state.value} → {new_state.value}")

config = OrchestratorConfig(
    on_state_change=on_state_change,
    ...
)
```

### Detection Update Callback

Called after each detection cycle.

```python
def on_detection_update(object_count):
    print(f"Detected {object_count} objects")

config = OrchestratorConfig(
    on_detection_update=on_detection_update,
    ...
)
```

### Task State Change Callback

Called when task state changes (explore, plan, etc.).

```python
def on_task_state_change(decision):
    print(f"Task state: {decision.state.value}")
    print(f"Confidence: {decision.confidence:.1%}")
    if decision.state == TaskState.PLAN_AND_EXECUTE:
        print("Ready for planning!")

config = OrchestratorConfig(
    on_task_state_change=on_task_state_change,
    ...
)
```

## State Persistence Format

The orchestrator saves state in the following structure:

```
outputs/orchestrator_state/
├── state.json              # Orchestrator metadata
├── registry.json           # Detected objects
└── pddl/
    ├── task_execution.pddl     # Domain file
    └── current_task.pddl       # Problem file
```

### state.json Format

```json
{
  "version": "1.0",
  "timestamp": 1234567890.0,
  "orchestrator_state": "idle",
  "current_task": "make a cup of coffee",
  "detection_count": 15,
  "task_analysis": {
    "goal_objects": ["coffee_maker", "cup", "coffee_grounds"],
    "relevant_predicates": ["graspable", "fillable", "on", "in"],
    "estimated_steps": 8,
    "complexity": "moderate"
  },
  "files": {
    "registry": "outputs/orchestrator_state/registry.json",
    "domain": "outputs/orchestrator_state/pddl/task_execution.pddl",
    "problem": "outputs/orchestrator_state/pddl/current_task.pddl"
  }
}
```

## Usage Patterns

### Simple Task Execution

```python
# Create, execute, and cleanup
orchestrator = TaskOrchestrator(config)
await orchestrator.initialize()
await orchestrator.process_task_request("pick up the red mug")
await orchestrator.start_detection()

# Wait for ready
while not orchestrator.is_ready_for_planning():
    await asyncio.sleep(1.0)

await orchestrator.generate_pddl_files()
await orchestrator.shutdown()
```

### Resume from Saved State

```python
# Load previous state and continue
orchestrator = TaskOrchestrator(config)
await orchestrator.initialize()
await orchestrator.load_state()  # Restores task and objects
await orchestrator.start_detection()  # Continue detection

# ... monitoring and PDDL generation ...

await orchestrator.shutdown()
```

### Task Update During Execution

```python
# Start with one task
await orchestrator.process_task_request("pick up the red mug")
await orchestrator.start_detection()

# ... wait for some observations ...

# Update task without losing observations
await orchestrator.update_task("pick up the red mug and place it on the shelf")

# Continue detection with updated task
# Existing objects are re-processed with new task context
```

### Manual Control Loop

```python
await orchestrator.initialize()
await orchestrator.process_task_request("make coffee")
await orchestrator.start_detection()

while True:
    # Check status
    status = await orchestrator.get_status()

    # Monitor task state
    decision = await orchestrator.get_task_decision()
    print(f"State: {decision.state.value}")

    # Act based on state
    if decision.state == TaskState.EXPLORE:
        print(f"Exploring... {decision.reasoning}")
    elif decision.state == TaskState.PLAN_AND_EXECUTE:
        print("Ready for planning!")
        await orchestrator.generate_pddl_files()
        break
    elif decision.state == TaskState.REFINE_DOMAIN:
        print(f"Domain needs refinement: {decision.blockers}")
        # Could trigger domain refinement here

    await asyncio.sleep(2.0)

await orchestrator.shutdown()
```

## Best Practices

1. **Always initialize before use**: Call `initialize()` before any other operations
2. **Always shutdown**: Call `shutdown()` to properly cleanup resources
3. **Enable auto-save**: Use `auto_save=True` for production to avoid losing state
4. **Monitor task state**: Check `get_task_decision()` to understand system status
5. **Use callbacks**: Set up callbacks to react to state changes in real-time
6. **Save state periodically**: Call `save_state()` at important checkpoints
7. **Handle errors gracefully**: Wrap operations in try-except and always shutdown

## Example: Production Server

```python
import asyncio
from src.planning import TaskOrchestrator, OrchestratorConfig

class RobotTaskServer:
    def __init__(self, api_key):
        self.config = OrchestratorConfig(
            api_key=api_key,
            auto_save=True,
            auto_save_interval=30.0,
            on_task_state_change=self.on_task_ready
        )
        self.orchestrator = None
        self.ready_for_plan = False

    async def start(self):
        self.orchestrator = TaskOrchestrator(self.config)
        await self.orchestrator.initialize()

        # Try to load previous state
        try:
            await self.orchestrator.load_state()
            print("Resumed from previous state")
        except FileNotFoundError:
            print("Starting fresh")

    async def process_task(self, task_description):
        await self.orchestrator.process_task_request(task_description)
        await self.orchestrator.start_detection()
        return {"status": "started", "task": task_description}

    def on_task_ready(self, decision):
        if decision.state == TaskState.PLAN_AND_EXECUTE:
            self.ready_for_plan = True

    async def get_plan(self):
        if not self.ready_for_plan:
            return {"status": "not_ready", "message": "Still exploring"}

        paths = await self.orchestrator.generate_pddl_files()
        return {
            "status": "ready",
            "domain": paths['domain_path'],
            "problem": paths['problem_path']
        }

    async def shutdown(self):
        if self.orchestrator:
            await self.orchestrator.shutdown()

# Usage
server = RobotTaskServer(api_key="...")
await server.start()
await server.process_task("make coffee")
# ... wait for ready ...
plan = await server.get_plan()
await server.shutdown()
```

## Troubleshooting

### "Orchestrator not initialized"
Call `await orchestrator.initialize()` before using any methods.

### "No task set"
Call `await orchestrator.process_task_request(task)` before starting detection.

### "Frame provider not set"
The orchestrator automatically sets up the camera. Ensure camera is connected.

### Detection not starting
- Check if task was set with `process_task_request()`
- Ensure camera is connected and accessible
- Check for errors in camera initialization

### Not reaching PLAN_AND_EXECUTE state
- Check `get_task_decision()` for blockers
- Ensure `min_observations` objects are detected
- Verify goal objects are present in the scene
- Check domain completeness with `get_status()`

### State load fails
- Ensure state file exists at expected path
- Check file format is valid JSON
- Verify camera is initialized before loading
