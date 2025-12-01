# PDDL Solver Integration with Intelligent Refinement

## Overview

The task orchestrator now has full PDDL solver integration with intelligent failure handling. When planning fails, the orchestrator automatically analyzes the error and refines the PDDL domain using LLM-guided fixes.

## Features

### 1. **PDDL Solver Integration**
- Integrated `PDDLSolver` into `TaskOrchestrator`
- Multiple backend support: pyperplan (default), Fast Downward (Docker/Apptainer)
- No Docker permissions needed with pyperplan backend
- Automatic backend selection based on availability

### 2. **Intelligent Failure Handling**
- **Automatic error detection**: Identifies refinable vs non-refinable errors
- **LLM-guided refinement**: Uses LLM to analyze planning failures and suggest fixes
- **Iterative refinement**: Automatically retries planning after domain fixes
- **Configurable attempts**: Set max refinement attempts via config

### 3. **Supported Error Types**
The system can automatically fix:
- Predicate arity mismatches (wrong number of arguments)
- Missing predicate definitions
- Action precondition/effect errors
- Type mismatches in parameters
- Parse errors in PDDL syntax

## Architecture

### New Orchestrator States
```python
class OrchestratorState(Enum):
    READY_FOR_PLANNING = "ready_for_planning"
    REFINING_DOMAIN = "refining_domain"  # NEW
    EXECUTING_PLAN = "executing_plan"
```

### Key Methods

#### `solve_and_plan_with_refinement()`
Main method for planning with automatic refinement:
```python
result = await orchestrator.solve_and_plan_with_refinement(
    algorithm=SearchAlgorithm.LAMA_FIRST,
    timeout=30.0,
    output_dir=Path("outputs/pddl")
)
```

**Flow:**
1. Try to solve the planning problem
2. If it fails with refinable error → refine domain
3. Retry with refined domain
4. Repeat up to `max_refinement_attempts` times

#### `refine_domain_from_failure()`
Refines PDDL domain based on planning error:
```python
success = await orchestrator.refine_domain_from_failure(
    error_message="wrong number of arguments for predicate empty-hand",
    pddl_files={"domain_path": "...", "problem_path": "..."}
)
```

**Process:**
1. Analyze error with LLM
2. Identify issue type (arity, missing predicate, etc.)
3. Apply appropriate fix
4. Update PDDL domain

## Configuration

### Orchestrator Config Options
```python
config = OrchestratorConfig(
    api_key=api_key,

    # Solver settings
    solver_backend="auto",  # "auto", "pyperplan", "fast-downward-docker", "fast-downward-apptainer"
    solver_algorithm="lama-first",
    solver_timeout=30.0,
    solver_verbose=False,

    # Automatic solving
    auto_solve_when_ready=False,  # Auto-solve when detection complete

    # Refinement settings
    auto_refine_on_failure=True,  # Enable automatic refinement
    max_refinement_attempts=3,    # Max refinement iterations

    # Callbacks
    on_plan_generated=callback_fn
)
```

## Usage Examples

### Example 1: Basic Planning with Refinement
```python
# Configure orchestrator
config = OrchestratorConfig(
    api_key=api_key,
    auto_refine_on_failure=True,
    max_refinement_attempts=3
)

orchestrator = TaskOrchestrator(config)
await orchestrator.initialize()

# Process task and detect objects
await orchestrator.process_task_request("Put red block on blue block")
await orchestrator.start_detection()

# ... detection runs ...

# Solve with automatic refinement
result = await orchestrator.solve_and_plan_with_refinement()

if result.success:
    print(f"Plan found: {result.plan}")
else:
    print(f"Planning failed: {result.error_message}")
```

### Example 2: Fully Autonomous Mode
```python
config = OrchestratorConfig(
    api_key=api_key,
    auto_solve_when_ready=True,      # Auto-solve when ready
    auto_refine_on_failure=True,     # Auto-refine on failure
    max_refinement_attempts=3
)

orchestrator = TaskOrchestrator(config)
await orchestrator.initialize()
await orchestrator.process_task_request("Stack all blocks")
await orchestrator.start_detection()

# Orchestrator will:
# 1. Detect objects continuously
# 2. Decide when ready for planning (via TaskStateMonitor)
# 3. Automatically generate PDDL and solve
# 4. If planning fails, refine domain and retry
# 5. Continue until success or max attempts reached
```

### Example 3: Manual Refinement
```python
config = OrchestratorConfig(
    api_key=api_key,
    auto_refine_on_failure=False  # Manual control
)

orchestrator = TaskOrchestrator(config)

# Try to solve
result = await orchestrator.solve_and_plan()

if not result.success:
    print(f"Planning failed: {result.error_message}")

    # Manually trigger refinement
    refined = await orchestrator.refine_domain_from_failure(
        error_message=result.error_message
    )

    if refined:
        # Retry
        result = await orchestrator.solve_and_plan()
```

## Demo Usage

### Interactive Demo
```bash
python examples/orchestrator_demo.py
```

Commands:
- `solve` - Solve for plan (uses refinement if enabled)
- `solve force` - Force solve even if not ready
- `generate` - Generate PDDL files only

The demo automatically enables refinement by default.

### Test Script
```bash
python examples/test_orchestrator_solver.py
```

This test demonstrates:
- Mock object detection (no camera needed)
- PDDL domain generation
- Automatic refinement on planning failures
- Full integration workflow

## Error Handling Flow

```
┌─────────────────────┐
│   Solve Planning    │
│      Problem        │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │   Success?   │
    └──────┬───────┘
           │
      ┌────┴────┐
      │         │
     Yes       No
      │         │
      │         ▼
      │  ┌─────────────────┐
      │  │ Refinable Error? │
      │  └────────┬─────────┘
      │           │
      │      ┌────┴────┐
      │      │         │
      │     Yes       No
      │      │         │
      │      ▼         ▼
      │  ┌───────┐  Return
      │  │Refine │  Failure
      │  │Domain │
      │  └───┬───┘
      │      │
      │      ▼
      │  ┌────────────┐
      │  │Max Attempts?│
      │  └─────┬──────┘
      │        │
      │   ┌────┴────┐
      │   │         │
      │  Yes       No
      │   │         │
      │   ▼         │
      │ Return      │
      │ Failure     │
      │             │
      └─────────────┴─► Return Success
```

## Refinement Process

1. **Error Detection**
   - Parse planner error message
   - Classify error type (arity, missing pred, etc.)

2. **LLM Analysis**
   - Send error + current domain to LLM
   - Get specific fix recommendations

3. **Domain Modification**
   - Apply fixes to PDDL representation
   - Run validation (`validate_and_fix_action_predicates`)
   - Update action definitions

4. **Retry Planning**
   - Regenerate PDDL files
   - Attempt planning again
   - Repeat if needed

## Callbacks

### `on_plan_generated`
Called when a plan is generated (success or failure):
```python
def on_plan_generated(result: SolverResult):
    if result.success:
        print(f"Plan: {result.plan}")
    else:
        print(f"Failed: {result.error_message}")

config = OrchestratorConfig(
    on_plan_generated=on_plan_generated
)
```

## Files Modified

### Core Integration
- `src/planning/task_orchestrator.py` - Main integration
- `src/planning/pddl_domain_maintainer.py` - Domain refinement
- `config/orchestrator_config.py` - Configuration options

### Demos & Tests
- `examples/orchestrator_demo.py` - Interactive demo with refinement
- `examples/test_orchestrator_solver.py` - Test script

### Solver Changes
- `src/planning/pddl_solver.py` - Prefer pyperplan backend

## Performance Notes

### Backend Comparison
- **pyperplan**: Pure Python, no setup, slower for complex problems
- **Fast Downward (Docker)**: Fast, requires Docker permissions
- **Fast Downward (Apptainer)**: Fast, no root needed

### Refinement Overhead
- Each refinement attempt: ~5-10 seconds (LLM call + validation)
- Recommended max attempts: 3
- Most errors fixed in 1-2 attempts

## Future Enhancements

1. **Smarter Error Parsing**
   - Extract predicate names from errors
   - Identify specific action/precondition causing issue

2. **Refinement History**
   - Track what fixes were tried
   - Avoid repeating failed fixes

3. **Multi-Error Handling**
   - Fix multiple issues in one refinement pass
   - Prioritize errors by severity

4. **Learning from Success**
   - Cache successful domain patterns
   - Apply known fixes faster

## Troubleshooting

### Planning Always Fails
1. Check PDDL files manually: `outputs/*/pddl/`
2. Enable verbose mode: `solver_verbose=True`
3. Increase refinement attempts: `max_refinement_attempts=5`
4. Check LLM API key is valid

### Refinement Not Working
1. Verify `auto_refine_on_failure=True`
2. Check error is refinable: see `_is_refinable_error()`
3. Review LLM analysis output in logs

### Solver Not Found
1. Check available backends: `solver.get_available_backends()`
2. Install pyperplan: `pip install pyperplan`
3. Or install Apptainer for Fast Downward
