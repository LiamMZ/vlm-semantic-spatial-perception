# GTAMP Validation Checks — Implementation Status

Cross-reference of every check in `validation_checks.md` against the actual implementation.

**Key:**
- ✅ Fully implemented and matches spec
- ⚠️ Partially implemented (behaviour deviates from spec)
- ❌ Not implemented

---

## Layer L1 — Goal Specification

All checks implemented in [`_validate_l1`](../src/planning/layered_domain_generator.py#L369).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L1-V1 | Goal set is non-empty | ✅ | [layered_domain_generator.py:375-381](../src/planning/layered_domain_generator.py#L375) — REJECT with early return |
| L1-V2 | Goal predicates are syntactically well-formed (parentheses, grounded, naming pattern, ≥1 arg, no empty args) | ✅ | [layered_domain_generator.py:391-431](../src/planning/layered_domain_generator.py#L391) — REJECT per sub-check |
| L1-V3 | No contradictory goals under uniqueness constraints | ✅ | [layered_domain_generator.py:433-449](../src/planning/layered_domain_generator.py#L433) — REJECT on `pred:subject` key collision |
| L1-V4 | All goal arguments reference entities in the scene | ⚠️ | [layered_domain_generator.py:451-458](../src/planning/layered_domain_generator.py#L451) — REJECT implemented; entity set includes object IDs, semantic labels (`object_type`, `class_name`), and `"robot"`. **EXPLORATION flag not raised** (missing entity not recorded as exploration target). |

---

## Layer L2 — Predicate Vocabulary

All checks implemented in [`_validate_l2`](../src/planning/layered_domain_generator.py#L490) and [`_repair_l2_auto`](../src/planning/layered_domain_generator.py#L570).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L2-V1 | All goal predicate names are defined in vocabulary | ✅ | [layered_domain_generator.py:505-512](../src/planning/layered_domain_generator.py#L505) — REJECT |
| L2-V2 | All names follow PDDL naming convention | ✅ | [layered_domain_generator.py:514-530](../src/planning/layered_domain_generator.py#L514) — REJECT for predicate name and type names |
| L2-V3 | Every sensed/external predicate has a checked variant | ✅ | Type-mismatch REJECT at [layered_domain_generator.py:532-543](../src/planning/layered_domain_generator.py#L532); AUTO-REPAIR (generate missing variant) at [layered_domain_generator.py:570-602](../src/planning/layered_domain_generator.py#L570) |
| L2-V4 | All argument types are in the valid type set | ✅ | [layered_domain_generator.py:545-556](../src/planning/layered_domain_generator.py#L545) — REJECT per invalid type against `VALID_PDDL_TYPES`; also detects cross-predicate variable type conflicts |
| L2-V5 | No unused predicates (deferred to after L3) | ✅ | Deferred and runs in [`_prune_unused_predicates`](../src/planning/layered_domain_generator.py#L772) after L3 validation passes; AUTO-REPAIR (remove unused) |
| L2-V6 | All type classifications are valid | ✅ | [layered_domain_generator.py:558-566](../src/planning/layered_domain_generator.py#L558) — `type_classifications` now populated by `_run_l2` from LLM JSON; L2 prompt explicitly requires all five classification strings and includes them in the example output |

---

## Layer L3 — Action Schemas

All checks implemented in [`_validate_l3`](../src/planning/layered_domain_generator.py#L632) and [`_repair_l3_auto`](../src/planning/layered_domain_generator.py#L731).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L3-V1 | All predicates in action formulas exist in vocabulary | ✅ | [layered_domain_generator.py:662-670](../src/planning/layered_domain_generator.py#L662) — REJECT with list of undefined predicates and available vocab |
| L3-V2 | All variables declared in parameter list with valid types | ⚠️ | [layered_domain_generator.py:672-680](../src/planning/layered_domain_generator.py#L672) — undeclared variable detection ✅; type validity for declared parameters **not checked** (spec L3-V2 second half) |
| L3-V3 | All checked-* in preconditions have a producing action | ✅ | [layered_domain_generator.py:682-698](../src/planning/layered_domain_generator.py#L682) detection + [layered_domain_generator.py:731-770](../src/planning/layered_domain_generator.py#L731) AUTO-REPAIR (generate sensing action) |
| L3-V4 | All goal predicates reachable via relaxed planning graph | ✅ | [layered_domain_generator.py:700-727](../src/planning/layered_domain_generator.py#L700) — delete-relaxation BFS; REJECT with per-predicate diagnosis |

---

## Layer L4 — Grounding and Skill Specifications

Implemented in [`_run_l4_precheck`](../src/planning/layered_domain_generator.py#L812). Checks V1–V3 are **non-blocking warnings** (spec calls for REJECT/EXPLORATION). Checks V4–V9 are **not implemented**.

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L4-V1 | Grounding rules exist for all PDDL types | ⚠️ | [layered_domain_generator.py:846-859](../src/planning/layered_domain_generator.py#L846) — emits WARNING, does not REJECT |
| L4-V2 | Scene has elements for all required categories | ⚠️ | [layered_domain_generator.py:861-874](../src/planning/layered_domain_generator.py#L861) — emits WARNING, does not trigger perception update + REJECT |
| L4-V3 | All goal entities present in scene | ⚠️ | [layered_domain_generator.py:876-882](../src/planning/layered_domain_generator.py#L876) — emits WARNING; **EXPLORATION mechanism not triggered** |
| L4-V4 | All primitives in sequence exist in library | ✅ | [`SkillPlan.validate`](../src/primitives/skill_plan_types.py#L218) called from [`PrimitiveExecutor.prepare_plan`](../src/primitives/primitive_executor.py#L197) — REJECT (raises `ValueError`) |
| L4-V5 | All semantic parameters valid per schema | ✅ | [`PrimitiveSchema.validate`](../src/primitives/skill_plan_types.py#L75) checks required params, unknown params, and per-param validators — REJECT |
| L4-V6 | All symbolic labels resolve in scene | ⚠️ | [`PrimitiveExecutor.prepare_plan`](../src/primitives/primitive_executor.py#L152) falls back to `point_label` from registry but does **not REJECT if label is unresolvable** — silent fallback |
| L4-V7 | All required parameters present | ✅ | [`PrimitiveSchema.validate`](../src/primitives/skill_plan_types.py#L84-L87) checks `required_params` — REJECT |
| L4-V8 | All constraints instantiable after grounding | ❌ | Not implemented — no constraint template system exists yet |
| L4-V9 | All target poses pass feasibility pre-checks (workspace, reach, aperture) | ❌ | Not implemented — PyBullet IK failure is the only runtime guard |

---

## Layer L5 — Initial State Construction

All checks embedded in [`_run_l5`](../src/planning/layered_domain_generator.py#L890).

| ID | Condition | Status | Implementation |
|----|-----------|--------|----------------|
| L5-V1 | All predicate-entity combinations assigned | ⚠️ | State is constructed algorithmically; there is no explicit enumeration-and-diff check. Missing combinations are implicitly FALSE under closed-world assumption but are not logged. |
| L5-V2 | Checked and sensed predicates initialized to FALSE | ✅ | [layered_domain_generator.py:987-1009](../src/planning/layered_domain_generator.py#L987) — AUTO-REPAIR (set to FALSE); `checked-*` handled at lines 931–941 |
| L5-V3 | Entity types in facts match predicate argument types | ⚠️ | [layered_domain_generator.py:1011-1025](../src/planning/layered_domain_generator.py#L1011) — detects unknown entity IDs and removes offending facts (WARNING + AUTO-REPAIR); does **not** verify declared PDDL type compatibility |
| L5-V4 | Spatial facts consistent with scene representation | ⚠️ | Spatial facts are derived from position-based heuristics (dz, dx, dy thresholds at [layered_domain_generator.py:963-985](../src/planning/layered_domain_generator.py#L963)); no separate post-construction validation pass |
| L5-V5 | Initial state does not already satisfy all goals | ✅ | [layered_domain_generator.py:1027-1042](../src/planning/layered_domain_generator.py#L1027) — WARNING (non-blocking), matches spec |

---

## Summary

| Layer | Checks | Fully ✅ | Partial ⚠️ | Missing ❌ |
|-------|--------|----------|-----------|-----------|
| L1 | 4 | 3 | 1 (exploration) | 0 |
| L2 | 6 | 4 | 2 (V4 type set, V6 activation) | 0 |
| L3 | 4 | 3 | 1 (V2 type validity) | 0 |
| L4 | 9 | 3 (V4, V5, V7) | 3 (V1–V3 non-blocking) | 2 (V8, V9) |
| L5 | 5 | 2 (V2, V5) | 3 (V1, V3, V4) | 0 |
| **Total** | **28** | **15** | **10** | **2** |

### Remaining gaps

1. **L4-V1/V2/V3** — upgrade from WARNING to REJECT / perception-update / EXPLORATION as specified.
2. **L4-V6** — REJECT (or trigger perception update) when a symbolic label cannot be resolved, instead of silently falling back.
3. **L4-V8** — constraint instantiation check (requires a constraint template registry).
4. **L4-V9** — workspace/reachability/aperture feasibility pre-checks before motion planning.
5. **L1-V4** — record missing entities as exploration targets (EXPLORATION mechanism not implemented).
