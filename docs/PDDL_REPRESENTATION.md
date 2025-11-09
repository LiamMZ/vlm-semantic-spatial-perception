# PDDL Representation System

A modular, incremental PDDL representation manager for dynamic task planning with LLM integration.

## Overview

The `PDDLRepresentation` class provides a clean separation between domain components (types, predicates, actions) and problem components (object instances, initial state, goals). This enables:

1. **Incremental construction** - Add components one at a time as they're discovered
2. **State updates** - Update initial state from continuous perception
3. **Goal refinement** - Iteratively refine goals based on planning feedback
4. **Validation** - Check completeness and detect issues
5. **Modular generation** - Generate PDDL files from component sets

## Architecture

### Domain Components (Static/Semi-Static)

These define the **planning vocabulary** and remain relatively stable:

- **Object Types**: Type hierarchy (e.g., `cup - container - object`)
- **Predicates**: State properties and relationships (e.g., `(on ?obj1 - object ?obj2 - object)`)
- **Actions**: Two categories:
  - **Predefined actions**: Standard manipulation primitives (pick, place, open, close, etc.)
  - **LLM-generated actions**: Task-specific actions proposed by the LLM

### Problem Components (Dynamic)

These define the **current scenario** and change frequently:

- **Object Instances**: Specific objects in the scene (e.g., `red_cup - cup`)
- **Initial State Literals**: Current world state (e.g., `(on red_cup table)`)
- **Goal Literals**: Desired end state (e.g., `(holding red_cup)`)

## Key Features

### 1. Predefined Action Library

The system initializes with a comprehensive library of manipulation actions:

- `pick` - Pick up graspable objects
- `place` - Place objects on surfaces
- `open` / `close` - Open/close openable objects
- `pour` - Pour from one container to another
- `push` / `pull` - Push/pull movable objects
- `turn-on` / `turn-off` - Control switchable devices

Each action comes with:
- Typed parameters
- Preconditions (what must be true)
- Effects (what changes)
- Human-readable descriptions

### 2. LLM-Generated Actions

For tasks requiring specialized actions not in the predefined library:

```python
pddl.add_llm_generated_action(
    name="slide-open-drawer",
    parameters=[("d", "drawer")],
    precondition="(and (slideable ?d) (not (opened ?d)))",
    effect="(and (opened ?d) (accessible-inside ?d))",
    description="Slide open a drawer"
)
```

LLM-generated actions are clearly marked in the output PDDL for debugging.

### 3. World State Integration

Direct integration with perception systems:

```python
world_state = {
    "objects": [
        {
            "object_id": "red_cup_1",
            "object_type": "cup",
            "affordances": ["graspable", "containable"]
        }
    ],
    "predicates": [
        "on(red_cup_1, table_1)",
        "clean(red_cup_1)"
    ]
}

pddl.update_from_world_state(world_state)
```

The system:
- Automatically creates object types if needed
- Instantiates objects
- Parses predicate strings into literals

### 4. Incremental State Updates

For continuous perception scenarios:

```python
# Frame 1
pddl.update_initial_state([
    ("on", ["cup", "table"], False),
    ("empty-hand", [], False)
])

# Frame 2 (robot picked up cup)
pddl.update_initial_state([
    ("holding", ["cup"], False),
    ("empty-hand", [], True),  # Negated
    ("on", ["cup", "table"], True)  # Negated
])
```

### 5. Goal Refinement

Iteratively refine goals based on feedback:

```python
# Initial goal
pddl.add_goal_literal("holding", ["cup"])

# Planning fails, add constraints
pddl.add_goal_literal("on", ["cup", "table"], negated=True)
pddl.add_goal_literal("clean", ["cup"])

# Validate
valid, issues = pddl.validate_goal_completeness()
```

### 6. Validation & Feedback

Built-in validation methods:

```python
# Check if goals are achievable
goal_valid, goal_issues = pddl.validate_goal_completeness()
# Returns:
# - False, ["Goal references undefined object 'cup'"]
# - False, ["Goal predicate 'holding' not defined in domain"]

# Check if actions can achieve goals
action_valid, action_issues = pddl.validate_action_completeness()
# Returns:
# - False, ["No actions produce goal predicates: {'holding'}"]
```

### 7. Statistics & Monitoring

Get detailed statistics:

```python
stats = pddl.get_statistics()
# {
#   "domain": {
#     "name": "manipulation",
#     "types": 5,
#     "predicates": 12,
#     "predefined_actions": 9,
#     "llm_generated_actions": 2,
#     "total_actions": 11
#   },
#   "problem": {
#     "name": "manipulation_problem",
#     "object_instances": 4,
#     "initial_literals": 6,
#     "goal_literals": 2
#   }
# }
```

## Usage Patterns

### Pattern 1: Static Task Planning

For well-defined tasks with known environment:

```python
# Create representation
pddl = PDDLRepresentation(domain_name="kitchen_task")

# Build domain
pddl.add_object_type("cup", parent="object")
pddl.add_predicate("clean", [("obj", "object")])

# Define problem
pddl.add_object_instance("red_cup", "cup")
pddl.add_initial_literal("on", ["red_cup", "table"])
pddl.add_initial_literal("empty-hand", [])

# Set goal
pddl.add_goal_literal("holding", ["red_cup"])

# Generate PDDL files
paths = pddl.generate_files("outputs/pddl")
# Use with off-the-shelf planner (FastDownward, etc.)
```

### Pattern 2: Dynamic World State

For continuous perception and replanning:

```python
pddl = PDDLRepresentation(domain_name="dynamic_world")

while True:
    # Get current world state from perception
    world_state = get_world_state_from_perception()

    # Update representation
    pddl.clear_initial_state()
    pddl.update_from_world_state(world_state)

    # Check if goal still valid
    valid, issues = pddl.validate_goal_completeness()

    if not valid:
        # Refine goal
        refine_goal(pddl, issues)

    # Generate and plan
    paths = pddl.generate_files()
    plan = run_planner(paths)

    # Execute plan step
    execute_action(plan[0])
```

### Pattern 3: LLM-Guided Construction

For novel tasks requiring LLM assistance:

```python
pddl = PDDLRepresentation(domain_name="novel_task")

# Get world state
world_state = get_world_state()
pddl.update_from_world_state(world_state)

# Ask LLM to analyze task
task = "Organize the cluttered desk"
analysis = llm_analyze_task(task, world_state)

# Add LLM-generated components
for pred in analysis.required_predicates:
    pddl.add_predicate(pred.name, pred.parameters)

for action in analysis.required_actions:
    pddl.add_llm_generated_action(
        name=action.name,
        parameters=action.parameters,
        precondition=action.precondition,
        effect=action.effect
    )

for goal in analysis.goal_literals:
    pddl.add_goal_literal(goal.predicate, goal.arguments)

# Validate before planning
valid, issues = pddl.validate_action_completeness()
if not valid:
    # Ask LLM to fill gaps
    additional_actions = llm_fill_action_gaps(issues)
    for action in additional_actions:
        pddl.add_llm_generated_action(...)

# Generate and plan
paths = pddl.generate_files()
```

## Component API

### Domain Methods

```python
# Types
pddl.add_object_type(name: str, parent: Optional[str] = None)

# Predicates
pddl.add_predicate(
    name: str,
    parameters: List[Tuple[str, str]],
    description: Optional[str] = None
)

# Actions
pddl.add_predefined_action(
    name: str,
    parameters: List[Tuple[str, str]],
    precondition: str,
    effect: str,
    description: Optional[str] = None
)

pddl.add_llm_generated_action(...)  # Same signature

# Removal
pddl.remove_predicate(name: str)
pddl.remove_action(name: str)
```

### Problem Methods

```python
# Objects
pddl.add_object_instance(name: str, object_type: str)
pddl.remove_object_instance(name: str)

# State
pddl.add_initial_literal(
    predicate: str,
    arguments: List[str],
    negated: bool = False
)

pddl.add_goal_literal(
    predicate: str,
    arguments: List[str],
    negated: bool = False
)

# Bulk updates
pddl.update_initial_state(literals: List[Tuple[str, List[str], bool]])
pddl.update_goal_state(literals: List[Tuple[str, List[str], bool]])

pddl.clear_initial_state()
pddl.clear_goal_state()

# World state integration
pddl.update_from_world_state(world_state: Dict)
```

### Validation Methods

```python
# Validate goals
valid, issues = pddl.validate_goal_completeness()
# Returns: (bool, List[str])

# Validate actions
valid, issues = pddl.validate_action_completeness()
# Returns: (bool, List[str])

# Statistics
stats = pddl.get_statistics()
# Returns: Dict with domain and problem stats
```

### Generation Methods

```python
# Generate PDDL strings
domain_str = pddl.generate_domain_pddl()
problem_str = pddl.generate_problem_pddl()

# Generate and write files
paths = pddl.generate_files(
    output_dir: str = "outputs/pddl",
    domain_filename: Optional[str] = None,
    problem_filename: Optional[str] = None
)
# Returns: {"domain_path": str, "problem_path": str}
```

## Integration with Existing System

The PDDL representation integrates with:

1. **Object Tracker** (`src/perception/object_tracker.py`)
   - Provides object instances with types and affordances
   - Supplies spatial relationships for initial state

2. **LLM Task Analyzer** (`src/planning/llm_task_analyzer.py`)
   - Analyzes tasks and proposes predicates/actions
   - Generates goal conditions

3. **Dynamic PDDL Generator** (`src/planning/dynamic_pddl_generator.py`)
   - Can be refactored to use `PDDLRepresentation` internally
   - Provides higher-level task-to-PDDL workflow

## Examples

See `examples/pddl_representation_demo.py` for comprehensive demos:

1. **Demo 1**: Basic manual construction
2. **Demo 2**: World state integration
3. **Demo 3**: LLM-generated actions
4. **Demo 4**: Goal refinement workflow
5. **Demo 5**: Incremental state updates
6. **Demo 6**: **End-to-end task-to-PDDL generation** ⭐
7. **Demo 7**: Interactive task input

### Running the Demos

Run all demos:
```bash
python examples/pddl_representation_demo.py
```

Run end-to-end demo only:
```bash
python examples/pddl_representation_demo.py --end-to-end
```

Run interactive demo (input your own task):
```bash
python examples/pddl_representation_demo.py --interactive
```

### Demo 6: End-to-End Workflow

The end-to-end demo (`demo_end_to_end_task_generation`) shows the complete pipeline:

**Task:** "Clean the coffee mug and place it in the dishwasher"

**Steps:**
1. **Task Input** - User provides natural language task
2. **Environment Observation** - Perception system detects objects:
   - coffee_mug_1 (dirty, on countertop)
   - kitchen_sink_1 (has faucet)
   - dishwasher_1 (closed, empty)
   - dish_soap_1 (on countertop)
3. **LLM Task Analysis** - Analyzes task and proposes:
   - Object types (mug, sink, dishwasher, etc.)
   - Custom predicates (clean, dirty, wet, inside)
   - Task-specific actions (wash, dry, place-inside)
   - Goal conditions (clean mug, inside dishwasher)
4. **Build Representation** - Constructs PDDL components
5. **Validation** - Checks completeness and feasibility
6. **Generate PDDL** - Outputs domain and problem files
7. **Action Sequence** - Shows expected plan steps

**Output:**
```
Expected action sequence (from LLM analysis):
  1. pick(coffee_mug_1)
  2. wash(coffee_mug_1, kitchen_sink_1, dish_soap_1)
  3. dry(coffee_mug_1)
  4. open(dishwasher_1)
  5. place-inside(coffee_mug_1, dishwasher_1)
```

## Output Format

### Domain File

```pddl
;; Auto-generated PDDL domain: kitchen_manipulation
;; Generated at: 2025-11-08 10:30:00

(define (domain kitchen_manipulation)

  (:requirements :strips :typing)

  (:types
    container - object
    cup - container
    table - object
  )

  (:predicates
    (graspable ?obj - object)
    (holding ?obj - object)
    (on ?obj1 - object ?obj2 - object)
    (empty-hand)
  )

  ; Predefined action
  (:action pick
    :parameters (?obj - object)
    :precondition (and (graspable ?obj) (empty-hand))
    :effect (and (holding ?obj) (not (empty-hand)))
  )

  ; LLM-generated: Specialized action for task
  (:action custom-action
    :parameters (...)
    :precondition (...)
    :effect (...)
  )
)
```

### Problem File

```pddl
;; Auto-generated PDDL problem: kitchen_manipulation_problem
;; Task: Pick up the red cup from the table
;; Generated at: 2025-11-08 10:30:00

(define (problem kitchen_manipulation_problem)
  (:domain kitchen_manipulation)

  (:objects
    red_cup - cup
    kitchen_table - table
  )

  (:init
    (on red_cup kitchen_table)
    (empty-hand)
  )

  (:goal
    (holding red_cup)
  )
)
```

## Best Practices

1. **Start with predefined actions** - Use the built-in library when possible
2. **Validate frequently** - Check completeness after major updates
3. **Clear state between frames** - Use `clear_initial_state()` for dynamic scenarios
4. **Type hierarchy** - Organize types hierarchically for cleaner predicates
5. **Descriptive names** - Use clear, descriptive object instance names
6. **Incremental goals** - Add goal constraints incrementally based on feedback
7. **Monitor statistics** - Track component counts to detect issues early

## Future Enhancements

Potential extensions:

- [ ] Automatic predicate inference from action effects
- [ ] Action conflict detection
- [ ] Plan cost estimation
- [ ] Multi-agent support (multiple robots)
- [ ] Temporal PDDL support (durative actions)
- [ ] Probabilistic outcomes
- [ ] HTN (Hierarchical Task Network) integration
- [ ] Auto-generation of derived predicates
- [ ] PDDL 3.0+ features (preferences, constraints)
