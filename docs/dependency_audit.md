# Dependency Audit Report

**Entry points audited:**
- `scripts/test_clutter_planning_realsense_robot.py`
- `scripts/test_obstructed_fridge_sim.py`

**Date:** 2026-05-07

---

## Section A — Required only by `test_clutter_planning_realsense_robot.py`

| File | Role |
|---|---|
| `src/kinematics/xarm_pybullet_interface.py` | xArm7 PyBullet FK/IK |
| `src/kinematics/xarm_pybullet_planned_primitives.py` | Plans in sim, executes on real xArm |
| `src/kinematics/xarm_robot_adapter.py` | Real xArm SDK wrapper |
| `src/kinematics/base_pybullet_interface.py` | Base PyBullet sim interface |
| `src/kinematics/depth_environment_collider.py` | Depth-based collision mesh |
| `src/grasp_planner/grasp_planner.py` | Antipodal grasp sampler |
| `src/grasp_planner/grasp_candidate.py` | Grasp candidate representation |
| `src/grasp_planner/__init__.py` | — |
| `src/kinematics/sim/urdfs/xarm7_camera/xarm7.urdf` | Robot URDF model |
| `src/kinematics/sim/urdfs/xarm7_camera/xarm7.srdf` | Semantic robot description (mimic joints, gripper groups) |

---

## Section B — Required only by `test_obstructed_fridge_sim.py`

| File | Role |
|---|---|
| `src/kinematics/sim/robocasa_camera.py` | MuJoCo frame provider |
| `src/kinematics/sim/robocasa_robot_interface.py` | Robot state provider for orchestrator |
| `src/kinematics/sim/robocasa_primitives.py` | MuJoCo action-space primitives |

---

## Section C — Required by Both Scripts

| File | Role |
|---|---|
| `src/camera/base_camera.py` | Base camera interface |
| `src/perception/__init__.py` | Exports GSAM2ContinuousObjectTracker etc. |
| `src/perception/object_registry.py` | Thread-safe detected object registry |
| `src/perception/clearance.py` | Gripper geometry + clearance profiles |
| `src/perception/utils/coordinates.py` | Pixel-to-3D back-projection |
| `src/planning/task_orchestrator.py` | Main orchestration loop |
| `src/planning/clutter_module.py` | Clutter predicate vocabulary + grounding |
| `src/planning/conditional_task_executor.py` | Plan execution with recovery |
| `src/planning/utils/snapshot_utils.py` | Snapshot loading/caching |
| `src/primitives/primitive_executor.py` | Primitive execution + Molmo grounding |
| `src/primitives/skill_decomposer.py` | LLM skill decomposition |
| `src/primitives/skill_plan_types.py` | Primitive schema definitions |
| `src/llm_interface/base.py` | LLMClient base classes |
| `src/llm_interface/openai_client.py` | GPT-4V planning client |
| `src/llm_interface/google_genai.py` | Gemini client |
| `src/llm_interface/qwen3vl.py` | Qwen3-VL client |
| `src/llm_interface/__init__.py` | — |
| `src/utils/logging_utils.py` | Structured logging |

**Config YAMLs lazily loaded by the shared modules above:**

| File | Loaded By |
|---|---|
| `config/skill_decomposer_prompts.yaml` | `src/primitives/skill_decomposer.py` |
| `config/layered_domain_generator_prompts.yaml` | `src/planning/layered_domain_generator.py` |
| `config/pddl_domain_maintainer_prompts.yaml` | `src/planning/pddl_domain_maintainer.py` |
| `config/llm_task_analyzer_prompts.yaml` | `src/planning/llm_task_analyzer.py` |
| `config/llm_task_analyzer_prompts_predicates.yaml` | `src/planning/llm_task_analyzer.py` |

**Perception/planning files that appear unused but are lazy-loaded at runtime:**

| File | Loaded By |
|---|---|
| `src/perception/gsam2_object_tracker.py` | `src/perception/__init__.py` (lazy) |
| `src/perception/molmo_point_detector.py` | `src/primitives/primitive_executor.py` (lazy) |
| `src/perception/molmo_interaction_point_detector.py` | Alias wrapper for above |
| `src/perception/object_tracker.py` | `src/perception/gsam2_object_tracker.py` |
| `src/perception/gsam2/tracker.py` | `src/perception/gsam2_object_tracker.py` |
| `src/perception/gsam2/common_utils.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/mask_dictionary_model.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/track_utils.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/video_utils.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/taggers/base.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/taggers/openai.py` | `src/perception/gsam2/tracker.py` |
| `src/perception/gsam2/taggers/ram.py` | `src/perception/gsam2/tracker.py` |
| `src/planning/layered_domain_generator.py` | `src/planning/task_orchestrator.py` (lazy) |
| `src/planning/llm_task_analyzer.py` | `src/planning/task_orchestrator.py` (lazy) |
| `src/planning/pddl_domain_maintainer.py` | `src/planning/task_orchestrator.py` (lazy) |
| `src/planning/pddl_solver.py` | `src/planning/task_orchestrator.py` |
| `src/planning/pddl_representation.py` | `src/planning/pddl_solver.py` |
| `src/planning/task_state_monitor.py` | `src/planning/task_orchestrator.py` (lazy) |
| `src/planning/domain_knowledge_base.py` | `src/planning/layered_domain_generator.py` |
| `src/planning/utils/pddl_types.py` | `src/planning/pddl_representation.py` |
| `src/planning/utils/task_types.py` | `src/planning/task_orchestrator.py`, `src/primitives/primitive_executor.py` |
| `src/perception/contact_graph.py` | `src/planning/task_orchestrator.py` (optional branch) |
| `src/perception/occlusion.py` | `src/planning/task_orchestrator.py` (optional branch) |
| `src/perception/surface_map.py` | `src/planning/task_orchestrator.py` (optional branch) |
| `src/perception/llm_contact_classifier.py` | `src/planning/task_orchestrator.py` (optional branch) |
| `src/utils/prompt_utils.py` | `src/planning/layered_domain_generator.py`, `src/primitives/skill_decomposer.py` |
| `src/utils/genai_logging.py` | `src/llm_interface/google_genai.py` |

---

## Section D — Not Required by Either Script

Files not in either dependency graph. Each row notes where the file IS used, or marks it as unused.

### Kinematics

| File | Used By |
|---|---|
| `src/kinematics/b1_robot_interface.py` | `src/kinematics/b1_z1_system.py` |
| `src/kinematics/b1_z1_system.py` | Referenced by deleted scripts only |
| `src/kinematics/b1_z1_transform_calculator.py` | `src/kinematics/b1_z1_system.py` |
| `src/kinematics/z1_robot_interface.py` | `src/kinematics/b1_z1_system.py` |
| `src/kinematics/stretch_pybullet_interface.py` | `src/kinematics/sim/stretch_pybullet_primitives.py` |
| `src/kinematics/xarm_curobo_interface.py` | **Unused — no references found** |
| `src/kinematics/sim/pybullet_camera.py` | `src/kinematics/sim/pybullet_primitives.py` |
| `src/kinematics/sim/pybullet_primitives.py` | `src/kinematics/sim/xarm_pybullet_primitives.py`, `stretch_pybullet_primitives.py` |
| `src/kinematics/sim/xarm_pybullet_primitives.py` | Deleted scripts only (`test_xarm_real_pybullet_planned_primitives.py` removed from examples) |
| `src/kinematics/sim/stretch_pybullet_primitives.py` | Deleted scripts only (`test_motion_planning.py`) |
| `src/kinematics/sim/scene_environment.py` | `src/kinematics/sim/pybullet_primitives.py` |
| `src/kinematics/sim/transform_calculator.py` | `src/kinematics/sim/pybullet_primitives.py` |
| `src/kinematics/sim/obstructed_fridge_item.py` | **Unused — no references found** |

### Camera

| File | Used By |
|---|---|
| `src/camera/realsense_camera.py` | Deleted scripts only (`test_grasp_sampling.py`, `test_motion_planning.py`) |
| `src/camera/webcam_camera.py` | **Unused — no references found** |
| `src/camera/camera_utils.py` | **Unused — no references found** |

### Planning

| File | Used By |
|---|---|
| `src/planning/hybrid_planner.py` | **Unused — no references found** |

### Root / Utils

| File | Used By |
|---|---|
| `src/task_motion_planner.py` | **Unused — no references found** |
| `src/utils/logger.py` | Legacy — superseded by `logging_utils.py`; some older modules may reference it |

### URDFs for Other Platforms

| Path | Used By |
|---|---|
| `src/kinematics/sim/urdfs/stretch/` | `stretch_pybullet_interface.py` (only used by deleted scripts) |
| `src/kinematics/sim/urdfs/b1z1_description/` | `b1_z1_system.py` (only used by deleted scripts) |
| `src/kinematics/sim/urdfs/robotiq_2f_140_gripper_visualization/` | `depth_environment_collider.py` (visualization only) |

---

## Section E — Summary and Recommendations

### Counts

| Category | Count |
|---|---|
| Used by robot script only | 10 |
| Used by fridge sim script only | 3 |
| Shared by both | 18 (+ 26 lazy-loaded) |
| Unused anywhere | 6 |
| Dead weight (only referenced by deleted scripts) | ~12 |

### Truly Unused — Safe to Delete

These files have no references anywhere in the codebase:

- `src/kinematics/xarm_curobo_interface.py`
- `src/kinematics/sim/obstructed_fridge_item.py`
- `src/planning/hybrid_planner.py`
- `src/camera/webcam_camera.py`
- `src/camera/camera_utils.py`
- `src/task_motion_planner.py`

### Dead Weight (Only Used by Deleted Scripts)

These files are only referenced by scripts that have been removed from the repo. They can be deleted unless those platforms are planned to be revived:

- `src/kinematics/b1_robot_interface.py`
- `src/kinematics/b1_z1_system.py`
- `src/kinematics/b1_z1_transform_calculator.py`
- `src/kinematics/z1_robot_interface.py`
- `src/kinematics/stretch_pybullet_interface.py`
- `src/kinematics/sim/stretch_pybullet_primitives.py`
- `src/kinematics/sim/xarm_pybullet_primitives.py`
- `src/kinematics/sim/pybullet_primitives.py` (transitively dead)
- `src/kinematics/sim/pybullet_camera.py` (transitively dead)
- `src/kinematics/sim/scene_environment.py` (transitively dead)
- `src/kinematics/sim/transform_calculator.py` (transitively dead)
- `src/camera/realsense_camera.py`
- `src/kinematics/sim/urdfs/stretch/` (directory)
- `src/kinematics/sim/urdfs/b1z1_description/` (directory)
