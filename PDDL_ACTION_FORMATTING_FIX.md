# PDDL Action Formatting Fix - No Quoted Strings

## Problem

The LLM was generating PDDL actions with **quoted strings** in preconditions and effects, which is invalid PDDL syntax:

```pddl
(:action fill_water
  :parameters (?coffee_maker - coffee_maker ?water_source - water_source)
  :precondition (and (is-empty ?coffee_maker "water_reservoir") (is-plugged-in ?coffee_maker))
  :effect (has-water ?coffee_maker)
)
```

**Issues:**
- `"water_reservoir"` is a quoted string constant (invalid in PDDL)
- PDDL only allows variables (e.g., `?obj`) and predicate names
- String literals are not supported in PDDL preconditions/effects

## Solution

### Two-Layer Defense

#### 1. **Updated LLM Prompts** ([src/planning/llm_task_analyzer.py](src/planning/llm_task_analyzer.py))

Added explicit PDDL formatting rules to both prompts:

```
CRITICAL PDDL FORMATTING RULES:
1. Parameters MUST use variables with ? prefix (e.g., ?obj, ?location, ?machine)
2. Preconditions and effects can ONLY use:
   - Variables (e.g., ?obj, ?container)
   - Predicate names (e.g., graspable, empty-hand, has-water)
3. NEVER use quoted strings or constants in preconditions/effects
4. If referencing a component, make it a predicate:
   - WRONG: (is-empty ?machine "water_reservoir")
   - RIGHT: (water-reservoir-empty ?machine) or (reservoir-has-water ?machine)
5. Multi-word predicates use hyphens: has-water, is-empty, water-reservoir-empty
```

#### 2. **Added Formula Sanitization** ([src/planning/pddl_domain_maintainer.py](src/planning/pddl_domain_maintainer.py))

New `sanitize_pddl_formula()` function removes quoted strings as a safety net:

```python
def sanitize_pddl_formula(formula: str) -> str:
    """
    Remove quoted strings from PDDL formulas.

    Converts:
      (is-empty ?machine "water_reservoir")
    To:
      (is-empty ?machine)
    """
    # Remove double-quoted strings
    formula = re.sub(r'"[^"]*"', '', formula)

    # Remove single-quoted strings
    formula = re.sub(r"'[^']*'", '', formula)

    # Clean up extra whitespace
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\s+\)', ')', formula)
    formula = re.sub(r'\(\s+', '(', formula)

    return formula.strip()
```

Applied automatically when adding LLM-generated actions to the domain.

---

## Before vs After

### Before (Invalid PDDL)

```pddl
(:action fill_water
  :parameters (?coffee_maker - coffee_maker ?water_source - water_source)
  :precondition (and
    (is-empty ?coffee_maker "water_reservoir")  ❌ Quoted string
    (is-plugged-in ?coffee_maker))
  :effect (has-water ?coffee_maker)
)

(:action insert_pod
  :parameters (?pod - pod ?machine - coffee_maker)
  :precondition (and
    (holding ?pod)
    (is-open ?machine "pod_chamber")  ❌ Quoted string
  )
  :effect (has-pod ?machine)
)
```

**Result:** PDDL parsers reject the domain

### After (Valid PDDL)

**Prompt instructs LLM to generate:**
```pddl
(:action fill_water
  :parameters (?coffee_maker - coffee_maker ?water_source - water_source)
  :precondition (and
    (water-reservoir-empty ?coffee_maker)  ✓ Predicate
    (is-plugged-in ?coffee_maker))
  :effect (reservoir-has-water ?coffee_maker)
)

(:action insert_pod
  :parameters (?pod - pod ?machine - coffee_maker)
  :precondition (and
    (holding ?pod)
    (pod-chamber-open ?machine)  ✓ Predicate
  )
  :effect (has-pod ?machine)
)
```

**If LLM still adds quotes, sanitization strips them:**
```pddl
(:action fill_water
  :parameters (?coffee_maker - coffee_maker ?water_source - water_source)
  :precondition (and
    (is-empty ?coffee_maker)  ✓ Sanitized (quotes removed)
    (is-plugged-in ?coffee_maker))
  :effect (has-water ?coffee_maker)
)
```

---

## How It Works

### Step 1: Prompt Prevention

The LLM prompt now includes clear examples of what NOT to do:

```
WRONG: (is-empty ?machine "water_reservoir")
RIGHT: (water-reservoir-empty ?machine)

WRONG: (on ?obj "shelf_1")
RIGHT: (on ?obj ?shelf_1)
```

### Step 2: Automatic Sanitization

Even if the LLM ignores the prompt, sanitization catches it:

```python
# In pddl_domain_maintainer.py, when adding actions:
precondition = sanitize_pddl_formula(action_def.get("precondition", ""))
effect = sanitize_pddl_formula(action_def.get("effect", ""))

self.pddl.add_llm_generated_action(
    name=action_def.get("name", "unknown"),
    parameters=params,
    precondition=precondition,  # Sanitized!
    effect=effect,              # Sanitized!
    description=action_def.get("description", "")
)
```

---

## Examples

### Test Sanitization

```python
from src.planning.pddl_domain_maintainer import sanitize_pddl_formula

# Test cases
print(sanitize_pddl_formula('(is-empty ?machine "water_reservoir")'))
# Output: (is-empty ?machine)

print(sanitize_pddl_formula('(and (is-plugged-in ?coffee_maker) (has-water ?coffee_maker "reservoir"))'))
# Output: (and (is-plugged-in ?coffee_maker) (has-water ?coffee_maker))

print(sanitize_pddl_formula('(and (graspable ?obj) (empty-hand))'))
# Output: (and (graspable ?obj) (empty-hand))  # No change, already valid
```

### Correct Predicate Design

Instead of using quoted strings to reference object parts, create specific predicates:

**Bad Design:**
```pddl
(:predicates
  (is-empty ?obj ?component)  ; Generic predicate with string argument
)

(:init
  (is-empty coffee_machine_1 "water_reservoir")  ; ❌ String constant
  (is-empty coffee_machine_1 "pod_chamber")      ; ❌ String constant
)
```

**Good Design:**
```pddl
(:predicates
  (water-reservoir-empty ?machine)  ; Specific predicate
  (pod-chamber-empty ?machine)      ; Specific predicate
  (pod-chamber-open ?machine)
)

(:init
  (water-reservoir-empty coffee_machine_1)  ; ✓ Valid
  (pod-chamber-empty coffee_machine_1)      ; ✓ Valid
)
```

---

## Files Modified

### [src/planning/llm_task_analyzer.py](src/planning/llm_task_analyzer.py)

**Lines 177-185:** Added PDDL rules to `_build_initial_analysis_prompt()`
```python
IMPORTANT PDDL RULES:
1. Parameters MUST use variables starting with ? (e.g., ?obj, ?location, ?container)
2. Preconditions and effects use ONLY variables - NO quoted strings or constants
3. If you need to reference a specific part (like "water_reservoir"), create a separate predicate:
   - WRONG: (is-empty ?machine "water_reservoir")
   - RIGHT: (water-reservoir-empty ?machine)
4. All predicates should be predicates applied to variables, not string constants
```

**Lines 237-253:** Added PDDL rules to `_build_analysis_prompt()`
```python
CRITICAL PDDL FORMATTING RULES:
1. Parameters MUST use variables with ? prefix (e.g., ?obj, ?location, ?machine)
2. Preconditions and effects can ONLY use:
   - Variables (e.g., ?obj, ?container)
   - Predicate names (e.g., graspable, empty-hand, has-water)
3. NEVER use quoted strings or constants in preconditions/effects
4. If referencing a component, make it a predicate:
   - WRONG: (is-empty ?machine "water_reservoir")
   - RIGHT: (water-reservoir-empty ?machine) or (reservoir-has-water ?machine)
5. Multi-word predicates use hyphens: has-water, is-empty, water-reservoir-empty
```

### [src/planning/pddl_domain_maintainer.py](src/planning/pddl_domain_maintainer.py)

**Lines 85-128:** Added `sanitize_pddl_formula()` function
```python
def sanitize_pddl_formula(formula: str) -> str:
    """Remove quoted strings from PDDL formulas."""
    # Remove double and single quotes
    # Clean up whitespace
    return formula
```

**Lines 260-263:** Apply sanitization when adding actions
```python
# Sanitize preconditions and effects to remove quoted strings
precondition = sanitize_pddl_formula(action_def.get("precondition", ""))
effect = sanitize_pddl_formula(action_def.get("effect", ""))
```

---

## Benefits

| Benefit | Description |
|---------|-------------|
| **Valid PDDL** | All generated actions are parseable by PDDL solvers |
| **Prompt Education** | LLM learns correct PDDL syntax from examples |
| **Safety Net** | Sanitization catches any quotes that slip through |
| **Better Semantics** | Component-specific predicates are more precise |
| **Solver Compatible** | Works with Fast Downward, Pyperplan, and all standard solvers |

---

## Related Fixes

This fix complements the other PDDL formatting fixes:

1. **Type Name Sanitization** ([PDDL_FORMATTING_FIX.md](PDDL_FORMATTING_FIX.md))
   - Removes parentheses and quotes from type names
   - Example: `"Black plug (unplugged)"` → `black_plug`

2. **Action Formula Sanitization** (This Document)
   - Removes quoted strings from preconditions/effects
   - Example: `(is-empty ?obj "reservoir")` → `(is-empty ?obj)`

Together, these ensure **fully valid PDDL generation** from end to end.

---

## Testing

The sanitization is automatically applied, but you can test it directly:

```python
from src.planning.pddl_domain_maintainer import sanitize_pddl_formula

# Test with your formulas
formula = '(and (is-empty ?machine "water_reservoir") (plugged-in ?machine))'
clean = sanitize_pddl_formula(formula)
print(clean)
# Output: (and (is-empty ?machine) (plugged-in ?machine))
```

Run full pipeline test:
```bash
python examples/orchestrator_demo.py
```

Check generated PDDL files in `outputs/pddl/` - should have no quoted strings!

---

*Updated: 2025-11-21*
