# Task and Motion Planning (TAMP) System

Complete integration of perception, symbolic planning, skill decomposition, and primitive execution for robot task execution.

## Overview

The TAMP system provides a unified pipeline that takes natural language task descriptions and executes them on a robot through multiple levels of abstraction:

```
Natural Language Task
       ↓
TaskOrchestrator (Perception + Symbolic Planning)
       ↓
PDDL Plan [action1, action2, ...]
       ↓
SkillDecomposer (Symbolic → Primitives)
       ↓
Skill Plans [primitives for each action]
       ↓
PrimitiveExecutor (Translate + Execute)
       ↓
Robot Motion
```

## Architecture

### Components

#### 1. TaskOrchestrator
**Location:** `src/planning/task_orchestrator.py`

Manages perception and symbolic planning:
- Continuous object detection and tracking
- PDDL domain and problem generation
- Task analysis with LLM
- PDDL solving with automatic domain refinement
- State persistence and lifecycle management

**Key Methods:**
- `initialize()` - Initialize camera and perception systems
- `process_task_request(task)` - Process natural language task
- `start_detection()` - Begin continuous perception
- `solve_and_plan()` - Generate PDDL plan
- `solve_and_plan_with_refinement()` - Plan with automatic domain fixes

#### 2. SkillDecomposer
**Location:** `src/primitives/skill_decomposer.py`

Decomposes symbolic PDDL actions into executable primitives:
- Pulls world state from orchestrator (objects, relationships, images)
- Uses LLM to generate primitive sequences
- Validates against primitive library schema
- Maintains cache for efficiency

**Key Methods:**
- `plan(action_name, parameters)` - Decompose action to primitives
- Returns `SkillPlan` with validated primitive calls

#### 3. PrimitiveExecutor
**Location:** `src/primitives/primitive_executor.py`

Executes primitive motion plans:
- Translates image-grounded parameters (pixels) to metric coordinates
- Back-projects using depth and camera intrinsics
- Validates primitive calls against library
- Executes on robot interface or dry-run validates

**Key Methods:**
- `execute_plan(plan, world_state, dry_run)` - Execute skill plan
- `prepare_plan(plan, world_state)` - Translate and validate
- Returns `PrimitiveExecutionResult` with success status

#### 4. TaskAndMotionPlanner (Integration)
**Location:** `src/task_motion_planner.py`

Integrates all components into a unified system:
- Orchestrates complete pipeline
- Manages component lifecycle
- Provides high-level API
- Tracks execution state and results
- Supports callbacks for monitoring

**Key Methods:**
- `initialize()` - Initialize all components
- `perceive_environment(duration)` - Run perception until ready
- `plan_task(task)` - Generate PDDL plan only
- `decompose_action(action, params)` - Decompose single action
- `execute_skill_plan(action, plan)` - Execute primitives
- `plan_and_execute_task(task)` - Complete pipeline

## Usage

### Basic Usage

```python
import asyncio
import os
from pathlib import Path
from src.task_motion_planner import TaskAndMotionPlanner, TAMPConfig

async def main():
    # Configure TAMP
    config = TAMPConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        state_dir=Path("./outputs/tamp"),
        orchestrator_model="gemini-2.5-pro",
        decomposer_model="gemini-robotics-er-1.5-preview",
        solver_backend="auto",
        dry_run_default=True,  # Validate without executing
    )

    tamp = TaskAndMotionPlanner(config)

    # Initialize
    await tamp.initialize()

    # Perceive environment
    await tamp.perceive_environment(duration=10.0)

    # Plan and execute task
    result = await tamp.plan_and_execute_task(
        task="pick up the red block and place it on the table",
        dry_run=True  # Set False for live execution
    )

    if result.success:
        print(f"✓ Task completed in {result.planning_time + result.decomposition_time + result.execution_time:.1f}s")
    else:
        print(f"✗ Task failed at {result.failed_at_stage}: {result.error_message}")

    # Cleanup
    await tamp.shutdown()

asyncio.run(main())
```

### Interactive Demo

```bash
# Dry run mode (validation only, no robot)
python examples/tamp_demo.py --dry-run --interactive

# Live execution mode (requires robot interface)
python examples/tamp_demo.py --live --interactive

# Single task execution
python examples/tamp_demo.py --task "pick up the red block" --dry-run
```

Interactive commands:
- `task <description>` - Plan and execute a task
- `perceive [duration]` - Run perception for N seconds
- `status` - Show system status
- `log` - Show execution log
- `help` - Show available commands
- `quit` - Exit

### Testing

```bash
# Test complete integration with mocks
python examples/test_tamp_integration.py

# Test individual components
python examples/orchestrator_demo.py
python examples/test_skill_decomposer.py
```

## Configuration

### TAMPConfig Parameters

```python
@dataclass
class TAMPConfig:
    # API and models
    api_key: Optional[str] = None
    orchestrator_model: str = "gemini-2.5-pro"
    decomposer_model: str = "gemini-robotics-er-1.5-preview"

    # Paths
    state_dir: Path = Path("./outputs/tamp_state")
    perception_pool_dir: Optional[Path] = None
    primitive_catalog_path: Optional[Path] = None

    # Perception settings
    update_interval: float = 2.0  # Seconds between perception updates
    min_observations: int = 3      # Min observations per object

    # Planning settings
    auto_refine_on_failure: bool = True  # Auto-fix PDDL domain errors
    max_refinement_attempts: int = 3      # Max domain refinement attempts
    solver_backend: str = "auto"          # "auto", "pyperplan", "fast-downward-docker"
    solver_algorithm: str = "lama-first"  # PDDL search algorithm
    solver_timeout: float = 30.0          # Planning timeout (seconds)

    # Execution settings
    dry_run_default: bool = False         # Default execution mode
    primitives_interface: Optional[Any] = None  # Robot interface

    # Callbacks
    on_state_change: Optional[Callable] = None
    on_plan_generated: Optional[Callable] = None
    on_action_decomposed: Optional[Callable] = None
    on_action_executed: Optional[Callable] = None
```

### Solver Backends

The system supports multiple PDDL solver backends:

1. **pyperplan** (default, pure Python)
   - No Docker required
   - Slower but reliable
   - Best for development

2. **fast-downward-docker**
   - Fast, production-quality
   - Requires Docker permissions
   - Best for performance

3. **fast-downward-apptainer**
   - Fast, no Docker needed
   - Requires Apptainer installed
   - Good compromise

Backend selection:
```python
config = TAMPConfig(
    solver_backend="auto",  # Auto-select best available
    # OR specify explicitly:
    # solver_backend="pyperplan"
    # solver_backend="fast-downward-docker"
    # solver_backend="fast-downward-apptainer"
)
```

## Pipeline Flow

### 1. Perception Phase

```python
# Start continuous object detection
await tamp.perceive_environment(
    duration=10.0,        # Max time to perceive
    min_observations=3    # Min observations per object
)
```

**What happens:**
- Camera captures RGB-D frames
- Object detection runs continuously
- Objects tracked across frames
- Registry updated with stable detections
- Spatial relationships computed
- Ready when min_observations met

### 2. Planning Phase

```python
# Generate symbolic PDDL plan
result = await tamp.plan_task(
    task="pick up the red block",
    use_refinement=True  # Auto-fix domain errors
)
```

**What happens:**
- Task analyzed by LLM
- PDDL domain constructed from registry
- PDDL problem generated with task goals
- Solver generates action sequence
- If planning fails and refinement enabled:
  - Error analyzed by LLM
  - Domain fixes suggested
  - Domain updated
  - Planning retried (up to max_refinement_attempts)

**Output:** List of symbolic actions
```
["pick robot red_block table", "place robot red_block blue_block"]
```

### 3. Decomposition Phase

```python
# Decompose each action to primitives
for action in plan:
    skill_plan = await tamp.decompose_action(
        action=action_name,
        parameters=action_params
    )
```

**What happens:**
- World state extracted (objects, images, relationships)
- LLM generates primitive sequence with parameters
- Parameters use image coordinates (pixels, normals)
- Validated against primitive library schema
- Cached for efficiency

**Output:** SkillPlan with primitives
```json
{
  "action_name": "pick",
  "primitives": [
    {"name": "move_to_pre_grasp", "parameters": {"target_pixel": [320, 240], ...}},
    {"name": "approach_grasp", "parameters": {"standoff": 0.05, ...}},
    {"name": "close_gripper", "parameters": {}},
    {"name": "lift", "parameters": {"height": 0.1}}
  ]
}
```

### 4. Execution Phase

```python
# Execute primitives (with metric translation)
result = await tamp.execute_skill_plan(
    action_name=action,
    skill_plan=skill_plan,
    dry_run=False  # True = validate only
)
```

**What happens:**
- Pixel coordinates back-projected to 3D using depth
- Image normals transformed to world frame
- Parameters validated against primitive library
- If dry_run=False: primitives executed on robot
- Results collected and validated

**Output:** PrimitiveExecutionResult
```python
{
  "executed": True,
  "warnings": [],
  "errors": [],
  "primitive_results": [...]
}
```

## State Machine

### TAMP States

```python
class TAMPState(Enum):
    UNINITIALIZED = "uninitialized"  # Before initialize()
    IDLE = "idle"                     # Ready for tasks
    PERCEIVING = "perceiving"         # Running perception
    PLANNING = "planning"             # Generating PDDL plan
    DECOMPOSING = "decomposing"       # Converting to primitives
    EXECUTING = "executing"           # Running primitives
    COMPLETED = "completed"           # Task successful
    FAILED = "failed"                 # Task failed
```

Monitor state changes:
```python
def on_state_change(state: TAMPState):
    print(f"TAMP state: {state.value}")

config = TAMPConfig(on_state_change=on_state_change)
```

## Result Tracking

### TAMPResult

Complete execution result with timing and diagnostics:

```python
@dataclass
class TAMPResult:
    success: bool
    task_description: str

    # Planning
    pddl_plan: Optional[List[str]]
    planning_time: float
    refinement_attempts: int

    # Decomposition
    skill_plans: Dict[str, SkillPlan]
    decomposition_time: float

    # Execution
    execution_results: List[PrimitiveExecutionResult]
    execution_time: float

    # Errors
    error_message: Optional[str]
    failed_at_stage: Optional[str]  # "planning", "decomposition", "execution"
```

## Error Handling

### Planning Failures

If PDDL planning fails with `auto_refine_on_failure=True`:

1. Error analyzed by LLM
2. Domain fixes suggested (predicate arities, missing actions, etc.)
3. Domain updated
4. Planning retried
5. Repeats up to `max_refinement_attempts`

Example error handling:
```python
result = await tamp.plan_and_execute_task(task, dry_run=True)

if not result.success:
    if result.failed_at_stage == "planning":
        print(f"Planning failed after {result.refinement_attempts} refinement attempts")
        print(f"Error: {result.error_message}")
    elif result.failed_at_stage == "decomposition":
        print(f"Failed to decompose action to primitives")
    elif result.failed_at_stage == "execution":
        print(f"Primitive execution failed")
        for exec_result in result.execution_results:
            if exec_result.errors:
                print(f"Errors: {exec_result.errors}")
```

### Decomposition Failures

If skill decomposition fails:
- LLM may generate invalid primitives
- Validation catches schema mismatches
- Returns None, pipeline stops

### Execution Failures

If primitive execution fails:
- Translation may fail (invalid pixels, depth missing)
- Primitive call may fail (hardware error)
- Execution stops, result contains error details

## Callbacks

Monitor pipeline progress with callbacks:

```python
def on_plan_generated(result: SolverResult):
    print(f"Plan: {result.plan}")

def on_action_decomposed(action: str, skill_plan: SkillPlan):
    print(f"Action {action} → {len(skill_plan.primitives)} primitives")

def on_action_executed(action: str, result: PrimitiveExecutionResult):
    print(f"Action {action}: {'✓' if result.executed else '✗'}")

config = TAMPConfig(
    on_plan_generated=on_plan_generated,
    on_action_decomposed=on_action_decomposed,
    on_action_executed=on_action_executed,
)
```

## Dry Run vs Live Execution

### Dry Run Mode (Default)
```python
result = await tamp.plan_and_execute_task(task, dry_run=True)
```

- Validates complete pipeline
- No robot motion
- Safe for development
- Catches errors early

### Live Execution
```python
result = await tamp.plan_and_execute_task(task, dry_run=False)
```

- Executes on robot
- Requires `primitives_interface` configured
- Production mode
- Use with caution

## Examples

### Example 1: Simple Pick and Place

```python
async def pick_and_place():
    config = TAMPConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        dry_run_default=True
    )
    tamp = TaskAndMotionPlanner(config)

    await tamp.initialize()
    await tamp.perceive_environment(duration=10.0)

    result = await tamp.plan_and_execute_task(
        "pick up the red block and place it on the blue block"
    )

    await tamp.shutdown()
    return result.success
```

### Example 2: Multi-Step Task

```python
async def complex_task():
    config = TAMPConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        auto_refine_on_failure=True,
        max_refinement_attempts=3,
        dry_run_default=False  # Live execution
    )
    tamp = TaskAndMotionPlanner(config)

    await tamp.initialize()

    # Perceive
    await tamp.perceive_environment(duration=15.0)

    # Execute task
    result = await tamp.plan_and_execute_task(
        "clear the table by moving all blocks to the bin"
    )

    await tamp.shutdown()

    if result.success:
        print(f"Completed {len(result.pddl_plan)} actions")
        print(f"Total time: {result.planning_time + result.decomposition_time + result.execution_time:.1f}s")
```

### Example 3: Step-by-Step Execution

```python
async def step_by_step():
    config = TAMPConfig(api_key=os.getenv("GEMINI_API_KEY"))
    tamp = TaskAndMotionPlanner(config)

    await tamp.initialize()

    # Step 1: Perceive
    await tamp.perceive_environment(duration=10.0)

    # Step 2: Plan
    solver_result = await tamp.plan_task("pick up the red block")
    if not solver_result:
        return False

    # Step 3: Decompose and execute each action
    for action_str in solver_result.plan:
        parts = action_str.strip("()").split()
        action_name = parts[0]
        params = {f"param{i}": p for i, p in enumerate(parts[1:], 1)}

        # Decompose
        skill_plan = await tamp.decompose_action(action_name, params)
        if not skill_plan:
            return False

        # Execute
        result = await tamp.execute_skill_plan(action_str, skill_plan, dry_run=True)
        if not result.executed:
            return False

    await tamp.shutdown()
    return True
```

## Performance

Typical timing (dry run, local models):
- **Perception:** 5-15 seconds (depends on scene complexity)
- **Planning:** 1-5 seconds (PDDL solving)
- **Decomposition:** 2-10 seconds per action (LLM inference)
- **Execution:** Variable (depends on primitives and robot speed)

Example breakdown for "pick up block":
```
Perception:      8.2s
Planning:        2.1s
  - PDDL solve:  1.8s
  - Refinement:  0s (no errors)
Decomposition:   3.5s (1 action)
Execution:       12.4s (4 primitives, dry run validation)
Total:           26.2s
```

## Troubleshooting

### Issue: Docker permission denied
**Solution:** Use pyperplan backend
```python
config = TAMPConfig(solver_backend="pyperplan")
```

### Issue: Planning fails with domain errors
**Solution:** Enable auto-refinement
```python
config = TAMPConfig(
    auto_refine_on_failure=True,
    max_refinement_attempts=3
)
```

### Issue: Decomposition generates invalid primitives
**Solution:** Check primitive catalog and LLM model
```python
config = TAMPConfig(
    decomposer_model="gemini-robotics-er-1.5-preview",  # Use robotics-tuned model
    primitive_catalog_path=Path("config/primitive_descriptions.md")
)
```

### Issue: Execution fails with metric translation errors
**Solution:** Ensure perception pool has valid depth data
- Check camera calibration
- Verify depth images saved correctly
- Check snapshot cache in perception_pool_dir

## Files

### Core Implementation
- `src/task_motion_planner.py` - Main TAMP integration class
- `src/planning/task_orchestrator.py` - Perception + symbolic planning
- `src/primitives/skill_decomposer.py` - Action → primitives
- `src/primitives/primitive_executor.py` - Primitive execution

### Demos and Tests
- `examples/tamp_demo.py` - Interactive TAMP demo
- `examples/test_tamp_integration.py` - Integration test with mocks

### Configuration
- `config/orchestrator_config.py` - Orchestrator settings
- `config/primitive_descriptions.md` - Primitive library catalog
- `config/skill_decomposer_prompts.yaml` - Decomposition prompts

### Documentation
- `docs/tamp_system.md` - This file
- `docs/pddl_solver_integration.md` - PDDL solver details
