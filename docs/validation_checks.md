# Validation Checks Reference

This document catalogues every validation check in the planning and execution pipeline, the layer or stage it belongs to, the exact implementation location, and the failure behavior.

---

## Table of Contents

1. [Skill Plan & Primitive Validation](#1-skill-plan--primitive-validation)
2. [Layered PDDL Domain Generation (L1–L5)](#2-layered-pddl-domain-generation-l1l5)
3. [PDDL Domain Maintenance & Refinement](#3-pddl-domain-maintenance--refinement)
4. [Task Orchestrator](#4-task-orchestrator)
5. [Object Registry & Perception](#5-object-registry--perception)
6. [Config](#6-config)
7. [Summary Table](#7-summary-table)
8. [Key Design Patterns](#8-key-design-patterns)

---

## 1. Skill Plan & Primitive Validation

### `src/primitives/skill_plan_types.py`

#### `PrimitiveSchema.validate()` — [lines 75–109](../src/primitives/skill_plan_types.py#L75-L109)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Required parameters | [85–87](../src/primitives/skill_plan_types.py#L85-L87) | All `required_params` present in call | Appends error to list |
| Unknown parameters | [90–93](../src/primitives/skill_plan_types.py#L90-L93) | No params outside `required_params ∪ optional_params` | Appends error |
| Frame | [96–100](../src/primitives/skill_plan_types.py#L96-L100) | `call.frame` is in `allowed_frames` | Appends error |
| Per-param validators | [103–107](../src/primitives/skill_plan_types.py#L103-L107) | Custom validators in `param_validators` dict | Appends validator message |

#### `SkillPlan.validate()` — [lines 218–236](../src/primitives/skill_plan_types.py#L218-L236)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Unknown primitive name | [231–232](../src/primitives/skill_plan_types.py#L231-L232) | Primitive name exists in `schema_map` | Appends "unknown primitive" error |
| Per-primitive call | [234–235](../src/primitives/skill_plan_types.py#L234-L235) | Each `PrimitiveCall` passes its schema | Appends indexed error messages |

#### `PRIMITIVE_LIBRARY` parameter validators — [lines 239–321](../src/primitives/skill_plan_types.py#L239-L321)

`_positive_number_validator` is applied to the following params:

| Primitive | Param | Lines |
|-----------|-------|-------|
| `open_gripper` | `timeout` | [265–267](../src/primitives/skill_plan_types.py#L265-L267) |
| `close_gripper` | `timeout` | [274–276](../src/primitives/skill_plan_types.py#L274-L276) |
| `retract_gripper` | `distance`, `speed_factor` | [283–286](../src/primitives/skill_plan_types.py#L283-L286) |
| `twist` | `rotation_angle_deg`, `speed_factor`, `timeout` | [315–319](../src/primitives/skill_plan_types.py#L315-L319) |

Failure for all: returns `"must be > 0"` or `"must be numeric"` error string (collected, not raised).

---

### `src/primitives/primitive_executor.py`

#### `prepare_plan()` — [lines 100–202](../src/primitives/primitive_executor.py#L100-L202)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Depth/intrinsics for back-projection | [154–159](../src/primitives/primitive_executor.py#L154-L159) | Snapshot has depth and intrinsics when `target_pixel_yx` is set | Log WARNING, fall back to registry position |
| Back-projection result | [163–168](../src/primitives/primitive_executor.py#L163-L168) | `compute_3d_position` returns a valid point | Log WARNING, fall back to registry position |
| `point_label` injection | [178–185](../src/primitives/primitive_executor.py#L178-L185) | If no `target_position`, injects `point_label` from `references.object_id` | Silently injects fallback |
| Base-frame coordinate transform | [188–195](../src/primitives/primitive_executor.py#L188-L195) | Applies cam→base transform to `target_position` / `pivot_point` when `cam_pose` available | Skips transform if no cam_pose |
| **Plan validation (critical gate)** | [197–200](../src/primitives/primitive_executor.py#L197-L200) | All primitives pass `PRIMITIVE_LIBRARY` schema validation | **Raises `ValueError`: "Plan validation failed: {errors}"** |

---

## 2. Layered PDDL Domain Generation (L1–L5)

All layers live in `src/planning/layered_domain_generator.py`.

### Retry loop — `_run_layer_with_retry()` — [lines 303–342](../src/planning/layered_domain_generator.py#L303-L342)

Each LLM-driven layer (L1–L3) runs through this generic loop:
1. Call `run_fn()` → get artifact
2. Call `validate_fn(artifact)` → collect errors
3. If errors and `repair_fn` provided → call `repair_fn(artifact)` → re-validate
4. If still errors → append errors to prompt and retry up to `max_retries`
5. On exhaustion → return artifact with errors recorded (non-fatal)

---

### L1 — Goal Specification

#### `_validate_l1()` — [lines 369–460](../src/planning/layered_domain_generator.py#L369-L460)

| Check ID | Lines | What is validated | Failure |
|----------|-------|-------------------|---------|
| L1-V1 Non-empty goals | [376–380](../src/planning/layered_domain_generator.py#L376-L380) | `goal_predicates` is not empty | Appends error, returns early |
| L1-V2a Parentheses | [391–396](../src/planning/layered_domain_generator.py#L391-L396) | Each literal starts with `(` and ends with `)` | Appends error |
| L1-V2b Grounded (no variables) | [398–403](../src/planning/layered_domain_generator.py#L398-L403) | No `?` variable tokens in goal predicates | Appends error |
| L1-V2c Empty literal | [405–408](../src/planning/layered_domain_generator.py#L405-L408) | Literal has content after stripping parens | Appends error |
| L1-V2d PDDL name pattern | [413–418](../src/planning/layered_domain_generator.py#L413-L418) | Predicate name matches `^[a-z][a-z0-9]*(-[a-z0-9]+)*$` | Appends error |
| L1-V2e Argument present | [420–426](../src/planning/layered_domain_generator.py#L420-L426) | At least one argument after predicate name | Appends error |
| L1-V2f Empty argument | [429–431](../src/planning/layered_domain_generator.py#L429-L431) | No empty-string argument tokens | Appends error |
| L1-V3 Contradiction detection | [436–449](../src/planning/layered_domain_generator.py#L436-L449) | Same subject not mapped to two different values by same predicate | Appends "Contradiction" error |
| L1-V4 Scene entity references | [452–458](../src/planning/layered_domain_generator.py#L452-L458) | All argument tokens appear in observed scene object IDs (warn-only if scene empty) | Appends "not found in observed scene" error |

---

### L2 — Predicate Vocabulary

#### `_validate_l2()` — [lines 490–568](../src/planning/layered_domain_generator.py#L490-L568)

| Check ID | Lines | What is validated | Failure |
|----------|-------|-------------------|---------|
| L2-V1 Goal predicate coverage | [505–512](../src/planning/layered_domain_generator.py#L505-L512) | Every predicate name from L1 goals defined in `predicate_signatures` | Appends "not defined in predicate vocabulary" error |
| L2-V2a Predicate name pattern | [514–523](../src/planning/layered_domain_generator.py#L514-L523) | Each predicate name matches `_PDDL_NAME_RE` | Appends naming error |
| L2-V2b Type name pattern | [524–530](../src/planning/layered_domain_generator.py#L524-L530) | Inline type names match `_PDDL_NAME_RE` | Appends type naming error |
| L2-V3 Checked variant arity | [532–543](../src/planning/layered_domain_generator.py#L532-L543) | If `checked-X` exists, its arity matches base `X` | Appends parameter count mismatch error |
| L2-V4 Variable type consistency | [545–556](../src/planning/layered_domain_generator.py#L545-L556) | No variable has conflicting type assignments across predicates | Appends "conflicting type assignments" error |
| L2-V6 Type classification validity | [558–566](../src/planning/layered_domain_generator.py#L558-L566) | `type_classifications` values are in `VALID_TYPE_CLASSIFICATIONS` | Appends "invalid type_classification" error |

#### `_repair_l2_auto()` — [lines 570–602](../src/planning/layered_domain_generator.py#L570-L602)

Auto-generates missing `checked-X` variants for predicates tagged `sensed`. Appends new signatures if not already present.

---

### L3 — Action Schemas

#### `_validate_l3()` — [lines 632–729](../src/planning/layered_domain_generator.py#L632-L729)

| Check ID | Lines | What is validated | Failure |
|----------|-------|-------------------|---------|
| L3-V1 Symbolic closure | [662–670](../src/planning/layered_domain_generator.py#L662-L670) | All predicates in pre/effects exist in L2 vocabulary | Appends "uses undefined predicates" error |
| L3-V2 Parameter consistency | [672–680](../src/planning/layered_domain_generator.py#L672-L680) | All `?vars` in pre/effects are declared in `:parameters` | Appends "uses undeclared variables" error |
| L3-V3 Sensing coverage | [682–698](../src/planning/layered_domain_generator.py#L682-L698) | Every `checked-*` predicate in any precondition has at least one action that adds it | Appends "No action produces" error |
| L3-V4 Goal achievability (BFS) | [700–727](../src/planning/layered_domain_generator.py#L700-L727) | All L1 goal predicate names reachable from initial state via positive effects (`bfs_reachable_predicates` at [lines 128–153](../src/planning/layered_domain_generator.py#L128-L153)) | Appends "unreachable" / "no producer" errors |

#### `_repair_l3_auto()` — [lines 731–770](../src/planning/layered_domain_generator.py#L731-L770)

Auto-generates sensing actions (`check-X`) for any `checked-*` predicate that has no producer.

---

### L4 — Grounding Pre-check (algorithmic, no LLM)

#### `_run_l4_precheck()` — [lines 812–884](../src/planning/layered_domain_generator.py#L812-L884)

Warnings only — does not block L5.

| Check ID | Lines | What is validated | Failure |
|----------|-------|-------------------|---------|
| L4-V1 Domain type grounding | [846–859](../src/planning/layered_domain_generator.py#L846-L859) | Every PDDL type used in actions has ≥1 matching scene object | Appends warning |
| L4-V2 Parameterized action feasibility | [861–874](../src/planning/layered_domain_generator.py#L861-L874) | Parameterized actions have viable scene objects | Appends warning |
| L4-V3 Goal entity existence | [876–882](../src/planning/layered_domain_generator.py#L876-L882) | Every entity named in L1 goals exists in scene | Appends warning |

---

### L5 — Initial State Construction (algorithmic, no LLM)

#### `_run_l5()` — [lines 890–1044](../src/planning/layered_domain_generator.py#L890-L1044)

| Check ID | Lines | What is validated / computed | Failure |
|----------|-------|------------------------------|---------|
| `checked-*` always FALSE | [910–941](../src/planning/layered_domain_generator.py#L910-L941) | All sensed predicates initialised as FALSE | — |
| L5-V1 Graspable derivation | [948–953](../src/planning/layered_domain_generator.py#L948-L953) | Affordance `"graspable"` → positive literal | — |
| L5-V2 Spatial predicates (on/above/stacked) | [955–1009](../src/planning/layered_domain_generator.py#L955-L1009) | Position-based proximity: on (z<0.3m, xy<0.5m), stacked (dz 0–0.15m, dx/dy<0.08m) | — |
| L5-V3 Entity reference validation | [1011–1025](../src/planning/layered_domain_generator.py#L1011-L1025) | All literal args refer to known scene entities | Log WARNING, auto-remove offending literal |
| L5-V5 Trivial goal detection | [1027–1042](../src/planning/layered_domain_generator.py#L1027-L1042) | Initial state does not already satisfy all goals | Log WARNING only |

---

## 3. PDDL Domain Maintenance & Refinement

### `src/planning/pddl_domain_maintainer.py`

#### `_initialize_domain_from_analysis()` — [lines 291–394](../src/planning/pddl_domain_maintainer.py#L291-L394)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Null task_analysis | [293–294](../src/planning/pddl_domain_maintainer.py#L293-L294) | `task_analysis` not None | Returns early |
| Predicate name non-empty | [320–321](../src/planning/pddl_domain_maintainer.py#L320-L321) | Predicate name after normalization is non-empty | Skips predicate |
| Action type check | [348–351](../src/planning/pddl_domain_maintainer.py#L348-L351) | `required_actions[i]` is a dict | **Raises `ValueError`** |
| Formula sanitization | [370–373](../src/planning/pddl_domain_maintainer.py#L370-L373) | Removes quoted strings from PDDL formulas via `sanitize_pddl_formula()` ([lines 89–132](../src/planning/pddl_domain_maintainer.py#L89-L132)) | Non-fatal, returns sanitized formula |

#### `validate_and_fix_action_predicates()` — [lines 545–603](../src/planning/pddl_domain_maintainer.py#L545-L603)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Undefined predicates in actions | [568–589](../src/planning/pddl_domain_maintainer.py#L568-L589) | All predicates used in preconditions/effects exist in domain | **Auto-adds missing predicate** to domain |
| Action parse errors | [591–593](../src/planning/pddl_domain_maintainer.py#L591-L593) | Action formulas parseable | Log WARNING, track in `invalid_actions` |

#### `refine_domain_from_error()` — [lines 923–1200](../src/planning/pddl_domain_maintainer.py#L923-L1200)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Problem file `:objects` non-empty | [968–999](../src/planning/pddl_domain_maintainer.py#L968-L999) | Problem file has objects defined | Log WARNING (perception issue indicator) |
| LLM refinement response JSON | [1065](../src/planning/pddl_domain_maintainer.py#L1065) | Response is valid JSON | Catches `json.JSONDecodeError` |
| Detected object not null | [1084–1088](../src/planning/pddl_domain_maintainer.py#L1084-L1088) | Suggested object not `None`/`"None"`/`"?"` | Skips update |
| Fix has old/new text | [1151–1154](../src/planning/pddl_domain_maintainer.py#L1151-L1154) | Fix dict contains `old_text` and `new_text` | Skips fix |

#### `sanitize_pddl_name()` — [lines 24–86](../src/planning/pddl_domain_maintainer.py#L24-L86)

Normalises names: lowercase, strip parens/quotes, replace spaces with `-`, ensure starts with letter, truncate. Non-fatal — returns sanitized form.

#### `_normalize_predicate_signature()` — [lines 664–710](../src/planning/pddl_domain_maintainer.py#L664-L710)

| Check | Lines | What is validated | Failure |
|-------|-------|-------------------|---------|
| Non-empty signature string | [677–686](../src/planning/pddl_domain_maintainer.py#L677-L686) | Signature has tokens after splitting | Returns empty string |
| Parameter deduplication | [688–702](../src/planning/pddl_domain_maintainer.py#L688-L702) | No duplicate variable names | Deduplicates silently |

---

## 4. Task Orchestrator

### `src/planning/task_orchestrator.py`

| Check | Method | Lines | What is validated | Failure |
|-------|--------|-------|-------------------|---------|
| Orchestrator initialized | `process_task_request` | [343–344](../src/planning/task_orchestrator.py#L343-L344) | State is not UNINITIALIZED | **Raises `RuntimeError`** |
| Goal object null filter | `process_task_request` | [389–390](../src/planning/task_orchestrator.py#L389-L390) | Filters `None`/`"None"` from goal_objects | Silently drops |
| PDDL system initialized | `generate_pddl_files` | [1111–1112](../src/planning/task_orchestrator.py#L1111-L1112) | `pddl` and `maintainer` not None | **Raises `RuntimeError`** |
| Objects detected (registry) | `generate_pddl_files` | [1131–1139](../src/planning/task_orchestrator.py#L1131-L1139) | Registry has ≥1 detected object | Log WARNING (non-fatal) |
| Unknown object type auto-register | `generate_pddl_files` | [1142–1154](../src/planning/task_orchestrator.py#L1142-L1154) | Object type exists in PDDL domain | **Auto-adds** type as child of `object` |
| Domain/problem files exist | `solve_and_plan` | [1323–1326](../src/planning/task_orchestrator.py#L1323-L1326) | Generated PDDL files present on disk | **Raises `FileNotFoundError`** |
| Object detection timeout | `solve_and_plan` | [1268–1286](../src/planning/task_orchestrator.py#L1268-L1286) | Objects appear within `max_wait_seconds` | Log WARNING if timeout, proceeds |
| Max refinement attempts | `solve_and_plan_with_refinement` | [1380–1382](../src/planning/task_orchestrator.py#L1380-L1382) | Attempt count < `max_refinement_attempts` | Returns `False` |
| Refinable error pattern | `_is_refinable_error` | [1534–1572](../src/planning/task_orchestrator.py#L1534-L1572) | Error message matches known refinable keywords | Returns boolean (non-fatal guard) |

#### Affordance-based initial literal derivation — [lines 1185–1199](../src/planning/task_orchestrator.py#L1185-L1199)

For each detected object, maps affordance names (e.g. `"graspable"`) to predicate names and adds them as initial literals — only if the predicate is already defined in the domain. Acts as a fallback for L5 when scene was empty during generation.

---

## 5. Object Registry & Perception

### `src/perception/object_registry.py`

| Check | Method | Lines | What is validated | Failure |
|-------|--------|-------|-------------------|---------|
| File existence | `load_from_json` | [363–364](../src/perception/object_registry.py#L363-L364) | JSON file exists at path | **Raises `FileNotFoundError`** |
| Object existence before update | `update_object` | [141–142](../src/perception/object_registry.py#L141-L142) | Object ID exists in registry | Returns `False` |

### `src/perception/object_tracker.py`

| Check | Method | Lines | What is validated | Failure |
|-------|--------|-------|-------------------|---------|
| `<think>` block stripping | `_parse_json_response` | (see file) | Strips `<think>…</think>` before JSON parse | Non-fatal pre-processing |
| JSON extraction fallback | `_parse_json_response` | (see file) | Tries bare parse → fenced block → first `{}`/`[]` | Raises `json.JSONDecodeError` on all failures |

---

## 6. Config

### `config/orchestrator_config.py`

`OrchestratorConfig` is a Python dataclass with typed fields. Validation relies on Python's type system — no explicit runtime validators. Key fields:

| Field | Type | Notes |
|-------|------|-------|
| `api_key` | `str` | Required (no default) |
| `use_sim_camera` | `bool` | Skips RealSense init when True |
| `use_layered_generation` | `bool` | Enables L1–L5 pipeline |
| `dkb_dir` | `Optional[Path]` | Domain knowledge base dir |
| `solver_timeout` | `float` | Max solver wall time |
| `max_refinement_attempts` | `int` | Refinement retry limit |

---

## 7. Summary Table

| Stage | Check ID | File | Lines | Severity | Failure Behavior |
|-------|----------|------|-------|----------|-----------------|
| Primitive schema | Required params | [skill_plan_types.py](../src/primitives/skill_plan_types.py#L85-L87) | 85–87 | Error | Collected |
| Primitive schema | Unknown params | [skill_plan_types.py](../src/primitives/skill_plan_types.py#L90-L93) | 90–93 | Error | Collected |
| Primitive schema | Frame check | [skill_plan_types.py](../src/primitives/skill_plan_types.py#L96-L100) | 96–100 | Error | Collected |
| Primitive schema | Param validators | [skill_plan_types.py](../src/primitives/skill_plan_types.py#L103-L107) | 103–107 | Error | Collected |
| Primitive schema | Unknown primitive name | [skill_plan_types.py](../src/primitives/skill_plan_types.py#L231-L232) | 231–232 | Error | Collected |
| **Execution gate** | **Plan validation** | [primitive_executor.py](../src/primitives/primitive_executor.py#L197-L200) | **197–200** | **HARD** | **Raises ValueError** |
| Execution | Depth/intrinsics | [primitive_executor.py](../src/primitives/primitive_executor.py#L154-L159) | 154–159 | Warning | Falls back to registry |
| Execution | Back-projection result | [primitive_executor.py](../src/primitives/primitive_executor.py#L163-L168) | 163–168 | Warning | Falls back to registry |
| L1 | Non-empty goals | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L376-L380) | 376–380 | Error | Collected, early return |
| L1 | Parentheses | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L391-L396) | 391–396 | Error | Collected |
| L1 | Grounded literals | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L398-L403) | 398–403 | Error | Collected |
| L1 | PDDL name pattern | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L413-L418) | 413–418 | Error | Collected |
| L1 | Argument present | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L420-L426) | 420–426 | Error | Collected |
| L1 | Contradiction | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L436-L449) | 436–449 | Error | Collected |
| L1 | Scene entity refs | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L452-L458) | 452–458 | Error | Collected |
| L2 | Goal pred coverage | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L505-L512) | 505–512 | Error | Collected |
| L2 | Predicate name pattern | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L514-L523) | 514–523 | Error | Collected |
| L2 | Type name pattern | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L524-L530) | 524–530 | Error | Collected |
| L2 | checked-X arity | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L532-L543) | 532–543 | Error | Collected |
| L2 | Var type consistency | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L545-L556) | 545–556 | Error | Collected |
| L2 | Type classification | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L558-L566) | 558–566 | Error | Collected |
| L2 (repair) | checked-X auto-generate | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L570-L602) | 570–602 | — | Auto-repair |
| L3 | Symbolic closure | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L662-L670) | 662–670 | Error | Collected |
| L3 | Parameter consistency | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L672-L680) | 672–680 | Error | Collected |
| L3 | Sensing coverage | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L682-L698) | 682–698 | Error | Collected |
| L3 | Goal achievability (BFS) | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L700-L727) | 700–727 | Error | Collected |
| L3 (repair) | Sensing action auto-generate | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L731-L770) | 731–770 | — | Auto-repair |
| L4 | Domain type grounding | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L846-L859) | 846–859 | Warning | Non-blocking |
| L4 | Action feasibility | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L861-L874) | 861–874 | Warning | Non-blocking |
| L4 | Goal entity existence | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L876-L882) | 876–882 | Warning | Non-blocking |
| L5 | checked-* always FALSE | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L910-L941) | 910–941 | — | Enforced |
| L5 | Entity reference validation | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L1011-L1025) | 1011–1025 | Warning | Auto-remove literal |
| L5 | Trivial goal detection | [layered_domain_generator.py](../src/planning/layered_domain_generator.py#L1027-L1042) | 1027–1042 | Warning | Log only |
| Domain init | Action type check | [pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py#L348-L351) | 348–351 | HARD | Raises ValueError |
| Domain init | Undefined predicates | [pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py#L568-L589) | 568–589 | — | Auto-adds to domain |
| Refinement | Problem :objects empty | [pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py#L968-L999) | 968–999 | Warning | Log only |
| Refinement | LLM response JSON | [pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py#L1065) | 1065 | Error | Catches JSONDecodeError |
| Refinement | Null object suggestion | [pddl_domain_maintainer.py](../src/planning/pddl_domain_maintainer.py#L1084-L1088) | 1084–1088 | — | Silently skips |
| Orchestrator | State initialized | [task_orchestrator.py](../src/planning/task_orchestrator.py#L343-L344) | 343–344 | HARD | Raises RuntimeError |
| Orchestrator | PDDL initialized | [task_orchestrator.py](../src/planning/task_orchestrator.py#L1111-L1112) | 1111–1112 | HARD | Raises RuntimeError |
| Orchestrator | Unknown object type | [task_orchestrator.py](../src/planning/task_orchestrator.py#L1142-L1154) | 1142–1154 | — | Auto-registers type |
| Orchestrator | PDDL files on disk | [task_orchestrator.py](../src/planning/task_orchestrator.py#L1323-L1326) | 1323–1326 | HARD | Raises FileNotFoundError |
| Orchestrator | Max refinements | [task_orchestrator.py](../src/planning/task_orchestrator.py#L1380-L1382) | 1380–1382 | — | Returns False |
| Registry | File existence | [object_registry.py](../src/perception/object_registry.py#L363-L364) | 363–364 | HARD | Raises FileNotFoundError |
| Registry | Object ID existence | [object_registry.py](../src/perception/object_registry.py#L141-L142) | 141–142 | — | Returns False |

---

## 8. Key Design Patterns

### Hard stops vs. soft failures

- **Hard stops** (`raise`) occur at system boundaries: missing files, uninitialized state, invalid plan at execution time.
- **Soft failures** (append error / log warning) occur inside LLM-driven generation layers so the pipeline can retry or auto-repair without crashing.

### Auto-repair before retry

L2 and L3 have dedicated repair methods that run *before* the retry prompt is sent:
- [`_repair_l2_auto()`](../src/planning/layered_domain_generator.py#L570-L602) — generates missing `checked-X` variants
- [`_repair_l3_auto()`](../src/planning/layered_domain_generator.py#L731-L770) — generates missing sensing actions (`check-X`)
- [`validate_and_fix_action_predicates()`](../src/planning/pddl_domain_maintainer.py#L545-L603) — auto-adds any undefined predicates referenced in actions

### Graceful degradation in execution

`PrimitiveExecutor.prepare_plan()` never hard-fails on missing depth/intrinsics — it falls back to resolving the target from `references.object_id` via the registry. The only hard failure is the final `plan.validate()` call after all translations are complete.

### Layer retry budget

Each L1–L3 layer gets `max_retries=2` attempts by default. Validation errors from the previous attempt are appended to the next prompt via a `<<VALIDATION_ERRORS>>` placeholder, guiding the LLM toward a valid response.

### L4 is advisory, L5 is constructive

L4 produces only warnings — it never blocks L5 from running. L5 is fully algorithmic (no LLM) and constructs the initial state from scene geometry and affordances, with L5-V3 silently removing any literals that reference unknown objects.
