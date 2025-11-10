# Planning System Documentation

## Overview

The planning system provides LLM-driven PDDL generation for task-aware robotic manipulation. It dynamically analyzes natural language tasks, maintains evolving domain models based on perception observations, and generates PDDL files for automated planning.

**Location**: [src/planning/](../src/planning/)

---

## Key Features

- **LLM-Driven Domain Generation**: Predicates and actions extracted from task semantics (no hard-coding)
- **Adaptive Exploration**: System knows when to explore vs. plan vs. refine
- **Perception-Planning Feedback Loop**: Task → predicates → perception → PDDL
- **Incremental Updates**: Domain evolves as observations arrive (not batch processing)
- **Fuzzy Goal Matching**: "mug" automatically matches "red_mug_1"
- **Thread-Safe Concurrent Access**: Multiple systems can query PDDL safely
- **Validation and Guidance**: Identifies what's missing and recommends actions

---

## Module Structure

### Core Files

- **[pddl_representation.py](../src/planning/pddl_representation.py)** (36,635 bytes)
  - Core PDDL data structures (types, predicates, actions)
  - Domain and problem management
  - PDDL file generation (domain.pddl, problem.pddl)
  - Thread-safe async operations with `asyncio.Lock`

- **[llm_task_analyzer.py](../src/planning/llm_task_analyzer.py)** (14,112 bytes)
  - LLM-powered task analysis using Gemini API
  - Extracts predicates, actions, goals from natural language
  - Supports optional environment images for context
  - Response caching (5-minute TTL)

- **[pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py)** (16,731 bytes)
  - Domain maintenance and evolution
  - Observation integration from perception
  - Goal object tracking with fuzzy matching
  - Domain completeness monitoring

- **[task_state_monitor.py](../src/planning/task_state_monitor.py)** (15,832 bytes)
  - Intelligent state determination (explore/plan/refine)
  - Blocker identification
  - Recommendation generation
  - Decision history tracking

---

## Architecture

### Component Hierarchy

```
TaskStateMonitor
    ├─> PDDLDomainMaintainer
    │       ├─> LLMTaskAnalyzer (Gemini API)
    │       └─> PDDLRepresentation
    └─> PDDLRepresentation
```

### Design Principles

- **Thread-safe**: Uses asyncio locks for concurrent access
- **Incremental updates**: Domain evolves as observations arrive
- **LLM-driven**: Predicates and actions generated dynamically
- **Modular**: Clear separation between domain (static) and problem (dynamic)

---

## Complete Planning Pipeline

```
1. TASK INPUT
   "Clean the dirty mug and place it on the shelf"

2. INITIAL ANALYSIS (LLM)
   ↓ analyze_task(task, observed_objects=None)
   Predicts: predicates, actions, goal objects

3. DOMAIN INITIALIZATION
   ↓ PDDLDomainMaintainer.initialize_from_task()
   Creates domain with predicted components

4. PERCEPTION SEEDING
   ↓ tracker.set_pddl_predicates(predicates)
   Tells perception what to track

5. EXPLORATION LOOP
   while TaskState.EXPLORE:
       ├─> Camera captures frame
       ├─> VLM detects objects with predicates
       ├─> PDDLDomainMaintainer.update_from_observations()
       ├─> TaskStateMonitor.determine_state()
       └─> Check if goal objects found

6. DOMAIN REFINEMENT (if needed)
   ↓ refine_domain_from_observations(all_objects)
   Re-analyze with actual observations

7. GOAL SETTING
   ↓ set_goal_from_task_analysis()
   Map task goals to detected objects

8. VALIDATION
   ↓ validate_goal_completeness()
   ↓ validate_action_completeness()

9. PLANNING READY
   TaskState.PLAN_AND_EXECUTE

10. FILE GENERATION
    ↓ generate_files_async()
    domain.pddl + problem.pddl

11. EXTERNAL PLANNER
    FastDownward/LAMA reads PDDL → generates plan
```

---

## Core Classes

### PDDLRepresentation

**Location**: [src/planning/pddl_representation.py](../src/planning/pddl_representation.py)

Core PDDL data structures and file generation.

#### Key Data Structures

```python
@dataclass
class ObjectType:
    name: str
    parent: Optional[str] = None

@dataclass
class Predicate:
    name: str
    parameters: List[Tuple[str, str]]  # [(param_name, type)]

@dataclass
class Action:
    name: str
    parameters: List[Tuple[str, str]]
    preconditions: List[Literal]
    effects: List[Literal]

@dataclass
class ObjectInstance:
    name: str
    type: str

@dataclass
class Literal:
    predicate: str
    arguments: List[str]
    negated: bool = False
```

#### Constructor

```python
PDDLRepresentation(
    domain_name: str = "robot_task",
    requirements: List[str] = [":strips", ":typing"]
)
```

#### Key Methods

```python
# Domain management (async, thread-safe)
async def add_object_type(name: str, parent: Optional[str] = None)
async def add_predicate(name: str, parameters: List[Tuple[str, str]])
async def add_action(action: Action)

# Problem management
async def add_object_instance(name: str, type: str)
async def add_initial_literal(predicate: str, arguments: List[str])
async def add_goal_literal(predicate: str, arguments: List[str])

# File generation
async def generate_files_async(output_dir: str) -> Tuple[str, str]
def generate_domain_pddl() -> str
def generate_problem_pddl() -> str

# Validation
async def validate_goal_completeness() -> Tuple[bool, List[str]]
async def validate_action_completeness() -> Tuple[bool, List[str]]
```

#### Usage Example

```python
from src.planning import PDDLRepresentation

pddl = PDDLRepresentation(domain_name="kitchen_task")

# Add types
await pddl.add_object_type("mug", parent="object")
await pddl.add_object_type("shelf", parent="object")

# Add predicates
await pddl.add_predicate("clean", [("obj", "object")])
await pddl.add_predicate("on", [("obj1", "object"), ("obj2", "object")])

# Add object instances
await pddl.add_object_instance("red_mug_1", "mug")
await pddl.add_object_instance("shelf_1", "shelf")

# Set initial state
await pddl.add_initial_literal("dirty", ["red_mug_1"])

# Set goal
await pddl.add_goal_literal("clean", ["red_mug_1"])
await pddl.add_goal_literal("on", ["red_mug_1", "shelf_1"])

# Generate files
domain_path, problem_path = await pddl.generate_files_async("outputs/pddl")
```

### LLMTaskAnalyzer

**Location**: [src/planning/llm_task_analyzer.py](../src/planning/llm_task_analyzer.py)

LLM-powered task analysis for dynamic domain generation.

#### TaskAnalysis Output

```python
@dataclass
class TaskAnalysis:
    action_sequence: List[str]           # High-level steps
    goal_predicates: List[str]           # Desired final state
    preconditions: List[str]             # Required starting conditions
    relevant_objects: Dict[str, List[str]]  # goal, tools, obstacles
    initial_predicates: List[str]        # Current state
    required_predicate_types: List[str]  # 8-15 predicates needed
    required_actions: List[str]          # PDDL action schemas
    complexity: str                      # "simple", "moderate", "complex"
    estimated_steps: int                 # Number of actions in plan
```

#### Constructor

```python
LLMTaskAnalyzer(
    api_key: str,
    model_name: str = "gemini-2.0-flash-exp",
    cache_ttl: int = 300  # 5 minutes
)
```

#### Key Methods

```python
# Task analysis
def analyze_task(
    task_description: str,
    observed_objects: Optional[List[Dict]] = None,
    environment_image: Optional[np.ndarray] = None
) -> TaskAnalysis

# Clear cache
def clear_cache()
```

#### Usage Example

```python
from src.planning import LLMTaskAnalyzer

analyzer = LLMTaskAnalyzer(api_key=api_key)

# Initial analysis (no observations)
analysis = analyzer.analyze_task(
    "Clean the dirty mug and place it on the shelf",
    environment_image=camera_frame
)

print(f"Goal predicates: {analysis.goal_predicates}")
print(f"Required predicates: {analysis.required_predicate_types}")
print(f"Goal objects: {analysis.relevant_objects['goal']}")

# Later, with observations
objects = [
    {"object_id": "red_mug_1", "object_type": "mug"},
    {"object_id": "shelf_1", "object_type": "shelf"}
]
refined_analysis = analyzer.analyze_task(
    "Clean the dirty mug and place it on the shelf",
    observed_objects=objects
)
```

### PDDLDomainMaintainer

**Location**: [src/planning/pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py)

Manages domain evolution based on observations.

#### Constructor

```python
PDDLDomainMaintainer(
    pddl: PDDLRepresentation,
    api_key: str,
    min_observations: int = 3
)
```

#### Key Methods

```python
# Initialization
async def initialize_from_task(
    task_description: str,
    environment_image: Optional[np.ndarray] = None
) -> TaskAnalysis

# Observation integration
async def update_from_observations(
    detected_objects: List[Dict]
) -> Dict[str, Any]  # Returns stats

# Domain refinement
async def refine_domain_from_observations(
    detected_objects: List[Dict]
) -> TaskAnalysis

# Goal management
async def set_goal_from_task_analysis()

# Status queries
async def is_domain_complete() -> bool
async def get_goal_object_coverage() -> Dict[str, List[str]]
def get_statistics() -> Dict[str, Any]
```

#### Update Statistics

```python
{
    "objects_added": int,           # New objects added this update
    "goal_objects_found": List[str],  # Goal objects detected
    "goal_objects_missing": List[str],  # Goal objects still needed
    "total_objects": int,           # Total objects in domain
    "domain_version": int           # Domain version number
}
```

#### Usage Example

```python
from src.planning import PDDLDomainMaintainer, PDDLRepresentation

pddl = PDDLRepresentation(domain_name="kitchen_task")
maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)

# Initialize from task
analysis = await maintainer.initialize_from_task(
    "Put the red mug on the shelf",
    environment_image=frame
)

# Update as objects detected
detected = [
    {
        "object_id": "red_mug_1",
        "object_type": "mug",
        "pddl_state": {"graspable": True, "dirty": False}
    },
    {
        "object_id": "shelf_1",
        "object_type": "shelf",
        "pddl_state": {"supportable": True}
    }
]

stats = await maintainer.update_from_observations(detected)
print(f"Added {stats['objects_added']} objects")
print(f"Found goal objects: {stats['goal_objects_found']}")
print(f"Still need: {stats['goal_objects_missing']}")

# Check readiness
if await maintainer.is_domain_complete():
    await maintainer.set_goal_from_task_analysis()
    await pddl.generate_files_async("outputs/pddl")
```

### TaskStateMonitor

**Location**: [src/planning/task_state_monitor.py](../src/planning/task_state_monitor.py)

Intelligent decision-making about exploration vs. planning.

#### Task States

```python
class TaskState(Enum):
    EXPLORE = "explore"                    # Need more observations
    PLAN_AND_EXECUTE = "plan_and_execute"  # Ready for planning
    REFINE_DOMAIN = "refine_domain"        # Domain needs refinement
    GOAL_UNREACHABLE = "goal_unreachable"  # Cannot achieve goal
    COMPLETE = "complete"                  # Task completed
```

#### TaskStateDecision

```python
@dataclass
class TaskStateDecision:
    state: TaskState                 # Current state
    confidence: float                # 0.0-1.0
    reasoning: str                   # Why this state?
    blockers: List[str]              # What's preventing progress
    recommendations: List[str]       # Suggested next actions
    metrics: Dict[str, Any]          # Supporting data
```

#### Constructor

```python
TaskStateMonitor(
    domain_maintainer: PDDLDomainMaintainer,
    pddl: PDDLRepresentation,
    min_observations: int = 3
)
```

#### Key Methods

```python
# State determination
async def determine_state() -> TaskStateDecision

# Decision history
def get_decision_history() -> List[TaskStateDecision]

# Reset state
def reset()
```

#### Decision Logic

Checks performed in order:

1. Are goal objects observed?
2. Is domain complete (all predicates/actions defined)?
3. Sufficient observations (min 3 objects)?
4. Initial state populated?
5. Goals defined?
6. Goal validation passes?
7. Action completeness validated?

#### Usage Example

```python
from src.planning import TaskStateMonitor

monitor = TaskStateMonitor(
    domain_maintainer=maintainer,
    pddl=pddl,
    min_observations=3
)

# Exploration loop
while True:
    decision = await monitor.determine_state()

    print(f"State: {decision.state}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")

    if decision.state == TaskState.EXPLORE:
        print(f"Need to find: {decision.blockers}")
        # Continue observation
        objects = await tracker.detect_objects(color, depth, intrinsics)
        await maintainer.update_from_observations(objects)

    elif decision.state == TaskState.REFINE_DOMAIN:
        print("Refining domain...")
        await maintainer.refine_domain_from_observations(all_objects)

    elif decision.state == TaskState.PLAN_AND_EXECUTE:
        print("Ready for planning!")
        await maintainer.set_goal_from_task_analysis()
        domain_path, problem_path = await pddl.generate_files_async("outputs/pddl")
        # Call external planner
        break

    elif decision.state == TaskState.GOAL_UNREACHABLE:
        print(f"Cannot complete task: {decision.blockers}")
        break
```

---

## Perception Integration

### Data Flow

```
Camera → ObjectTracker → DetectedObject[]
                              ↓
              PDDLDomainMaintainer.update_from_observations()
                              ↓
                      PDDLRepresentation
                  (objects + initial_state)
```

### Integration Pattern

```python
# 1. Analyze task → extract predicates
analysis = await maintainer.initialize_from_task("Clean the mug")

# 2. Seed perception with predicates
tracker.set_pddl_predicates(analysis.required_predicate_types)

# 3. Detect objects with those predicates
objects = await tracker.detect_objects(color, depth, intrinsics)
# Objects now have pddl_state: {"clean": False, "graspable": True, ...}

# 4. Update PDDL domain
await maintainer.update_from_observations(objects)

# Creates feedback loop: task → predicates → perception → PDDL
```

### Continuous Tracking Integration

```python
from src.perception import ContinuousObjectTracker

# Continuous tracker with callback
tracker = ContinuousObjectTracker(api_key=api_key)

# Callback triggers PDDL update
async def on_detection(objects):
    await maintainer.update_from_observations(objects)
    decision = await monitor.determine_state()
    if decision.state == TaskState.PLAN_AND_EXECUTE:
        # Ready to plan!
        pass

tracker.set_detection_callback(on_detection)
tracker.start()
```

---

## PDDL Generation

### Domain File Structure

```pddl
(define (domain kitchen_task)
  (:requirements :strips :typing)

  (:types
    mug - object
    shelf - object
  )

  (:predicates
    (clean ?obj - object)
    (dirty ?obj - object)
    (graspable ?obj - object)
    (on ?obj1 - object ?obj2 - object)
    (holding ?obj - object)
    (empty-hand)
  )

  (:action pick
    :parameters (?obj - object)
    :precondition (and (graspable ?obj) (empty-hand))
    :effect (and (holding ?obj) (not (empty-hand)))
  )

  (:action place
    :parameters (?obj - object ?surface - object)
    :precondition (and (holding ?obj) (supportable ?surface))
    :effect (and (on ?obj ?surface) (empty-hand) (not (holding ?obj)))
  )

  ; More actions...
)
```

### Problem File Structure

```pddl
(define (problem clean_mug_task)
  (:domain kitchen_task)

  (:objects
    red_mug_1 - mug
    shelf_1 - shelf
    table_1 - table
  )

  (:init
    (dirty red_mug_1)
    (graspable red_mug_1)
    (on red_mug_1 table_1)
    (supportable shelf_1)
    (empty-hand)
  )

  (:goal
    (and
      (clean red_mug_1)
      (on red_mug_1 shelf_1)
    )
  )
)
```

### Maintenance Cycle

**1. Initial Domain** (from LLM prediction):
- Predicates: 8-15 predicted from task
- Actions: 3-6 LLM-generated + predefined library
- Objects: None yet

**2. Observation Phase** (incremental updates):
- Add object instances as detected
- Populate initial state from `pddl_state`
- Track goal object coverage

**3. Refinement** (if needed):
- Re-analyze with observations
- Add missing predicates
- Update actions if insufficient

**4. Validation**:
- Check goal completeness
- Check action completeness
- Verify all references valid

**5. File Generation**:
- Write `domain.pddl`
- Write `problem.pddl`
- Ready for planner (FastDownward, LAMA, etc.)

---

## Fuzzy Goal Matching

The system uses intelligent matching to handle natural language ambiguity:

```python
# Task: "Clean the mug"
# Detected: "red_mug_1", "blue_mug_1"

# System automatically matches "mug" to detected mugs
goal_objects = maintainer.get_goal_object_coverage()
# Returns: {"mug": ["red_mug_1", "blue_mug_1"]}

# Can prioritize based on additional context
# (e.g., "red mug" prefers "red_mug_1")
```

**Matching Rules**:
- Type-based: "mug" matches any object of type "mug"
- Color-based: "red mug" prefers objects with "red" in properties
- Position-based: "nearby mug" considers proximity
- Context-based: Uses task context for disambiguation

---

## Performance Characteristics

### Typical Timings

- **Initial task analysis**: ~2-3 seconds (LLM call)
- **Observation update**: <100ms (adding objects to PDDL)
- **Domain refinement**: ~2-3 seconds (LLM call)
- **File generation**: <100ms (PDDL string generation)
- **Total exploration phase**: Depends on observation rate (typically 10-30 seconds)

### Optimization Strategies

1. **Response Caching**: 5-minute TTL for identical task analyses
2. **Incremental Updates**: Only updates changed components
3. **Async Operations**: Thread-safe concurrent access
4. **Lazy Validation**: Only validates when needed
5. **Predicate Seeding**: Tells perception what to look for (avoids redundant detection)

---

## Configuration

### Planning Config

**File**: [config/planning_config.yaml](../config/planning_config.yaml)

```yaml
planner:
  algorithm: "lama"  # or "ff", "fast-downward"
  timeout: 30  # seconds

domain:
  min_observations: 3
  max_observations: 30
  refinement_threshold: 5  # Re-analyze after N new objects

llm:
  model: "gemini-2.0-flash-exp"
  cache_ttl: 300  # seconds

validation:
  strict_mode: true
  check_action_completeness: true
  check_goal_reachability: true
```

---

## Complete Example

### Full Task Execution

```python
import asyncio
from src.planning import (
    PDDLRepresentation,
    LLMTaskAnalyzer,
    PDDLDomainMaintainer,
    TaskStateMonitor,
    TaskState
)
from src.perception import ObjectTracker
from src.camera import RealSenseCamera

async def execute_task(task_description: str):
    # Initialize components
    pddl = PDDLRepresentation(domain_name="kitchen_task")
    maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)
    monitor = TaskStateMonitor(maintainer, pddl, min_observations=3)
    tracker = ObjectTracker(api_key=api_key)
    camera = RealSenseCamera(enable_depth=True, auto_start=True)

    # Step 1: Initialize domain from task
    print(f"Task: {task_description}")
    analysis = await maintainer.initialize_from_task(task_description)
    print(f"Goal objects: {analysis.relevant_objects['goal']}")

    # Step 2: Seed perception
    tracker.set_pddl_predicates(analysis.required_predicate_types)

    # Step 3: Exploration loop
    while True:
        decision = await monitor.determine_state()
        print(f"\nState: {decision.state} (confidence: {decision.confidence:.2f})")
        print(f"Reasoning: {decision.reasoning}")

        if decision.state == TaskState.EXPLORE:
            print(f"Exploring... Need: {decision.blockers}")

            # Capture and detect
            color, depth = camera.get_aligned_frames()
            intrinsics = camera.get_camera_intrinsics()
            objects = await tracker.detect_objects(color, depth, intrinsics)

            # Update domain
            stats = await maintainer.update_from_observations(objects)
            print(f"Found {len(objects)} objects")
            print(f"Goal coverage: {stats['goal_objects_found']}")

        elif decision.state == TaskState.REFINE_DOMAIN:
            print("Refining domain...")
            all_objects = tracker.get_all_objects()
            await maintainer.refine_domain_from_observations(all_objects)

        elif decision.state == TaskState.PLAN_AND_EXECUTE:
            print("Ready for planning!")

            # Set goals and generate PDDL
            await maintainer.set_goal_from_task_analysis()
            domain_path, problem_path = await pddl.generate_files_async("outputs/pddl")

            print(f"Generated PDDL files:")
            print(f"  Domain: {domain_path}")
            print(f"  Problem: {problem_path}")

            # Call external planner here
            # plan = call_planner(domain_path, problem_path)
            break

        elif decision.state == TaskState.GOAL_UNREACHABLE:
            print(f"Cannot complete task: {decision.blockers}")
            break

        await asyncio.sleep(1.0)  # Wait before next iteration

    camera.stop()

# Run
asyncio.run(execute_task("Clean the dirty mug and place it on the shelf"))
```

---

## Troubleshooting

### Domain Not Completing

**Symptoms**: Stuck in EXPLORE or REFINE_DOMAIN state

**Causes**: Missing goal objects, insufficient observations

**Solutions**:
```python
# Check what's missing
decision = await monitor.determine_state()
print(decision.blockers)  # Lists what's needed

# Check coverage
coverage = await maintainer.get_goal_object_coverage()
print(f"Goal objects found: {coverage}")

# Lower observation threshold
maintainer = PDDLDomainMaintainer(pddl, api_key, min_observations=2)
```

### Invalid PDDL Generated

**Symptoms**: Planner rejects PDDL files

**Causes**: Missing types, undefined predicates, invalid references

**Solutions**:
```python
# Validate before generation
goal_valid, goal_issues = await pddl.validate_goal_completeness()
action_valid, action_issues = await pddl.validate_action_completeness()

if not goal_valid:
    print(f"Goal issues: {goal_issues}")
if not action_valid:
    print(f"Action issues: {action_issues}")

# Check domain statistics
stats = maintainer.get_statistics()
print(f"Domain has {stats['total_objects']} objects")
print(f"Domain has {len(stats['predicates'])} predicates")
```

### LLM Analysis Errors

**Symptoms**: Task analysis fails or returns incomplete data

**Causes**: API errors, invalid task description, quota exceeded

**Solutions**:
```python
# Try with environment image
analysis = analyzer.analyze_task(
    task_description,
    environment_image=camera.capture_frame()
)

# Clear cache if stale
analyzer.clear_cache()

# Check API key
import os
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set")
```

---

## Key Innovations

1. **LLM-Driven Domain Generation**: No hard-coded predicates
2. **Adaptive Exploration**: System knows when to explore/plan/refine
3. **Perception-Planning Feedback**: Task → predicates → perception → PDDL
4. **Incremental Updates**: Domain evolves as observations arrive
5. **Fuzzy Goal Matching**: Natural language flexibility
6. **Thread-Safe Concurrent Access**: Multi-system integration
7. **Validation and Guidance**: Identifies missing components

---

## Related Documentation

- [Perception System](perception.md)
- [Camera System](camera.md)
- [Planning Module README](../src/planning/README.md)

---

*Last Updated: November 10, 2025*
