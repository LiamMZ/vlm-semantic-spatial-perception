# PDDL Predicate Validation - Automatic Domain Consistency

## Overview

The PDDL Domain Maintainer now includes **automatic predicate validation** that ensures all predicates used in actions are properly defined in the domain. This prevents invalid PDDL generation and automatically repairs incomplete domains.

## Problem

When LLMs generate PDDL actions, they may reference predicates that weren't explicitly added to the domain's predicate definitions. This creates invalid PDDL that fails to parse:

```pddl
(:predicates
  (graspable ?obj - object)
  (empty-hand)
)

(:action pick_up
  :parameters (?obj - cup)
  :precondition (and (graspable ?obj) (empty-hand))
  :effect (holding ?obj)  ; ❌ 'holding' predicate not defined!
)
```

**Result:** PDDL parsers reject the domain because `holding` is used but not declared.

## Solution

The `PDDLDomainMaintainer` now automatically:

1. **Extracts all predicates** from action preconditions and effects
2. **Validates** that each predicate is defined in the domain
3. **Auto-adds missing predicates** with generic signatures
4. **Reports** what was fixed

### Automatic Integration

Predicate validation runs **automatically** during domain initialization:

```python
# In _initialize_domain_from_analysis()
await self.validate_and_fix_action_predicates()
```

This ensures domains are always consistent after task analysis completes.

### Manual Validation

You can also run validation manually:

```python
maintainer = PDDLDomainMaintainer(pddl_representation=pddl)

# Validate and fix predicates
result = await maintainer.validate_and_fix_action_predicates()

print(f"Missing predicates: {result['missing_predicates']}")
print(f"Invalid actions: {result['invalid_actions']}")
```

## How It Works

### 1. Predicate Extraction

The `_extract_predicates_from_formula()` method parses PDDL formulas using regex:

```python
def _extract_predicates_from_formula(self, formula: str) -> Set[str]:
    """
    Extract predicate names from PDDL formula.

    Handles:
    - Simple predicates: (holding ?obj)
    - Negated predicates: (not (filled ?cup))
    - Conjunctions: (and (graspable ?obj) (empty-hand))
    - Disjunctions: (or (on ?obj ?table) (on ?obj ?shelf))
    """
    # Remove 'not' wrappers
    # Extract predicate names with regex
    # Filter out logical operators (and, or, not, forall, etc.)
    return predicates
```

**Example:**
```python
formula = "(and (holding ?cup) (is-empty ?cup))"
predicates = _extract_predicates_from_formula(formula)
# Result: {'holding', 'is-empty'}
```

### 2. Validation Loop

For each action in the domain:

```python
for action_name, action in all_actions.items():
    # Extract predicates from preconditions
    precond_predicates = _extract_predicates_from_formula(action.precondition)

    # Extract predicates from effects
    effect_predicates = _extract_predicates_from_formula(action.effect)

    # Check if all predicates are defined
    for pred_name in (precond_predicates | effect_predicates):
        if pred_name not in pddl.predicates:
            # Add missing predicate!
            await pddl.add_predicate_async(pred_name, [("obj", "object")])
```

### 3. Auto-Add Missing Predicates

Missing predicates are added with a **generic signature**:

```python
await pddl.add_predicate_async(
    "holding",
    [("obj", "object")]  # Generic: single object parameter
)
```

**Generated PDDL:**
```pddl
(:predicates
  (holding ?obj - object)
)
```

This ensures the domain is valid, even if the predicate signature isn't perfect. The LLM can refine signatures in subsequent iterations.

## Example Run

```bash
$ python examples/test_predicate_validation.py
```

**Output:**
```
📋 Explicitly defined predicates:
   - graspable
   - empty-hand

📋 Adding actions with additional predicates...
   ✓ Added action: pick_up
     Uses predicate: holding (NOT DEFINED YET)

   ✓ Added action: fill_cup
     Uses predicates: is-empty, filled (NOT DEFINED YET)

   ✓ Added action: pour_water
     Uses predicates: water-level-low (NOT DEFINED YET)

🔍 Running predicate validation...

  ℹ Added missing predicate: holding
  ℹ Added missing predicate: is-empty
  ℹ Added missing predicate: filled
  ℹ Added missing predicate: water-level-low
✓ Validated and added 4 missing predicates

📊 Validation Results:
   ✓ Found and auto-added 4 missing predicates:
      - holding
      - is-empty
      - filled
      - water-level-low
   ✓ All actions are valid

📊 Predicates AFTER validation:
   (empty-hand)
   (filled ?obj - object)
   (graspable ?obj - object)
   (holding ?obj - object)
   (is-empty ?obj - object)
   (water-level-low ?obj - object)
```

**Generated PDDL:**
```pddl
(define (domain predicate_test)
  (:requirements :strips :typing)

  (:types
    cup - object
    water_source - object
  )

  (:predicates
    (empty-hand)
    (filled ?obj - object)
    (graspable ?obj - object)
    (holding ?obj - object)
    (is-empty ?obj - object)
    (water-level-low ?obj - object)
  )

  (:action pick_up
    :parameters (?obj - cup)
    :precondition (and (graspable ?obj) (empty-hand))
    :effect (holding ?obj)
  )

  (:action fill_cup
    :parameters (?cup - cup ?source - water_source)
    :precondition (and (holding ?cup) (is-empty ?cup))
    :effect (filled ?cup)
  )

  (:action pour_water
    :parameters (?source_cup - cup ?target_cup - cup)
    :precondition (and (holding ?source_cup) (filled ?source_cup) (is-empty ?target_cup))
    :effect (and (not (filled ?source_cup)) (filled ?target_cup) (water-level-low ?source_cup))
  )
)
```

**✅ Valid PDDL - ready for planning!**

## API Reference

### `validate_and_fix_action_predicates()`

```python
async def validate_and_fix_action_predicates(self) -> Dict[str, List[str]]:
    """
    Validate that all predicates used in actions are defined in the domain.
    Automatically adds missing predicates to the domain.

    Returns:
        Dictionary with:
        - 'missing_predicates': List of predicate names that were added
        - 'invalid_actions': List of action names that couldn't be parsed

    Example:
        >>> stats = await maintainer.validate_and_fix_action_predicates()
        >>> print(f"Added {len(stats['missing_predicates'])} missing predicates")
        >>> print(f"Predicates added: {stats['missing_predicates']}")
    """
```

**Returns:**
```python
{
    "missing_predicates": ["holding", "is-empty", "filled"],
    "invalid_actions": []
}
```

### `_extract_predicates_from_formula()`

```python
def _extract_predicates_from_formula(self, formula: str) -> Set[str]:
    """
    Extract predicate names from a PDDL formula.

    Handles complex formulas with:
    - Conjunctions: (and ...)
    - Disjunctions: (or ...)
    - Negations: (not ...)
    - Nested expressions

    Filters out logical operators: and, or, not, forall, exists, when

    Args:
        formula: PDDL formula string

    Returns:
        Set of predicate names used in the formula

    Example:
        >>> predicates = _extract_predicates_from_formula(
        ...     "(and (holding ?obj) (not (is-empty ?container)))"
        ... )
        >>> predicates
        {'holding', 'is-empty'}
    """
```

## Integration Points

### 1. Domain Initialization

Validation runs automatically after task analysis:

```python
# In pddl_domain_maintainer.py:_initialize_domain_from_analysis()
async def _initialize_domain_from_analysis(self):
    # ... add predicates and actions ...

    # Validate that all predicates used in actions are defined
    validation_result = await self.validate_and_fix_action_predicates()
    if validation_result["missing_predicates"]:
        print(f"✓ Auto-added {len(validation_result['missing_predicates'])} missing predicates")
```

### 2. Manual Validation

Call validation explicitly when needed:

```python
from src.planning import PDDLDomainMaintainer, PDDLRepresentation

pddl = PDDLRepresentation(domain_name="my_domain")
maintainer = PDDLDomainMaintainer(pddl_representation=pddl)

# Add types, predicates, actions...

# Validate before generating PDDL
result = await maintainer.validate_and_fix_action_predicates()

if result["invalid_actions"]:
    print(f"Warning: {len(result['invalid_actions'])} actions had parsing errors")
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **Automatic Repair** | No manual tracking of predicate definitions |
| **Valid PDDL** | Ensures generated domains are always parseable |
| **LLM-Friendly** | Handles incomplete LLM outputs gracefully |
| **Debugging** | Reports exactly what predicates were missing |
| **Zero Overhead** | Runs automatically during initialization |

## Limitations

### Generic Signatures

Auto-added predicates use a generic signature:

```pddl
(predicate-name ?obj - object)
```

This may not match the **actual intended signature**. For example:

```pddl
; Auto-added (generic)
(on ?obj - object)

; Intended signature
(on ?obj - object ?surface - surface)
```

**Solution:** The LLM can refine predicate signatures in subsequent iterations, or you can manually define predicates with correct signatures before validation.

### No Semantic Understanding

Validation uses **regex pattern matching** to extract predicate names. It doesn't understand:
- Predicate semantics
- Correct arity (number of parameters)
- Type constraints

It simply ensures **syntactic validity** - that all predicates are defined.

## Files Modified

### [src/planning/pddl_domain_maintainer.py](src/planning/pddl_domain_maintainer.py)

**Lines 424-530:** Added validation methods

```python
async def validate_and_fix_action_predicates(self) -> Dict[str, List[str]]:
    """Validate and auto-add missing predicates."""
    # Extract predicates from all actions
    # Check if defined in domain
    # Auto-add if missing
    return {"missing_predicates": [...], "invalid_actions": [...]}

def _extract_predicates_from_formula(self, formula: str) -> Set[str]:
    """Extract predicate names from PDDL formula."""
    # Regex pattern matching
    # Filter logical operators
    return predicates_set
```

**Lines 277-283:** Integrated validation into domain initialization

```python
# Validate that all predicates used in actions are defined
validation_result = await self.validate_and_fix_action_predicates()
if validation_result["missing_predicates"]:
    print(f"✓ Auto-added {len(validation_result['missing_predicates'])} missing predicates")
```

## Testing

Run the test to see validation in action:

```bash
python examples/test_predicate_validation.py
```

The test:
1. Creates a domain with 2 explicit predicates
2. Adds 3 actions that reference 4 additional predicates
3. Runs validation (auto-detects 4 missing predicates)
4. Generates valid PDDL domain file

**Output:** `outputs/test_predicates/domain.pddl` with all predicates defined

## Related Features

This validation complements other PDDL formatting features:

1. **Type Name Sanitization** ([PDDL_FORMATTING_FIX.md](PDDL_FORMATTING_FIX.md))
   - Removes parentheses and quotes from type names
   - Example: `"Black plug (unplugged)"` → `black_plug`

2. **Formula Sanitization** ([PDDL_ACTION_FORMATTING_FIX.md](PDDL_ACTION_FORMATTING_FIX.md))
   - Removes quoted strings from formulas
   - Example: `(is-empty ?obj "reservoir")` → `(is-empty ?obj)`

3. **Predicate Validation** (This Document)
   - Ensures all predicates used in actions are defined
   - Auto-adds missing predicates

Together, these ensure **fully valid PDDL generation** from end to end.

---

*Updated: 2025-11-22*
