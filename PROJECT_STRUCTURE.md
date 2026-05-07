# Project Structure — VLM Semantic Spatial Perception

End-to-end robot manipulation system: live RGB-D perception → PDDL task planning → motion-primitive execution on a real xArm7 (or in simulation). The reference integration is [scripts/test_clutter_planning_realsense_robot.py](scripts/test_clutter_planning_realsense_robot.py).

---

## Top-level layout

```
src/
  camera/          — RealSense + webcam drivers
  kinematics/      — FK/IK, motion planning, robot interfaces
    sim/           — PyBullet simulation primitives
  llm_interface/   — LLM client abstraction (OpenAI, Gemini, Qwen)
  perception/      — Object detection, tracking, contact/clearance analysis
    gsam2/         — GSAM2 tracker internals (RAM+, GroundingDINO, SAM2)
    utils/         — 3-D coordinate helpers
  planning/        — PDDL domain generation, solving, execution
    utils/         — PDDL types, snapshot helpers, task types
  primitives/      — Skill decomposition and primitive execution
  utils/           — Logging, prompt utilities
config/            — YAML prompts, OrchestratorConfig dataclass
scripts/           — Integration test / demo entry points
tests/             — Unit and integration tests with fixture snapshots
```

---

## Pipeline overview

```
RealSense D435
     │ RGB-D frames
     ▼
GSAM2ContinuousObjectTracker          [src/perception/gsam2_object_tracker.py]
  RAM+ → GroundingDINO → SAM2
  + ClearanceProfile, ContactGraph, OcclusionMap, MolmoInteractionPoints
     │ DetectedObjectRegistry (object_id → DetectedObject)
     ▼
TaskOrchestrator                       [src/planning/task_orchestrator.py]
  process_task_request()
     │ observed_objects list
     ▼
LayeredDomainGenerator (L1–L5)         [src/planning/layered_domain_generator.py]
  L1 Goal Specification  (LLM)
  L2 Predicate Vocabulary (LLM)
  L3 Action Schemas       (LLM)
  L4 Grounding Pre-check  (algorithmic)
  L5 Initial State        (algorithmic)
     │ PDDL domain + problem
     ▼
PDDLSolver (Fast Downward)             [src/planning/pddl_solver.py]
     │ plan: List[str] of PDDL action strings
     ▼
ConditionalTaskExecutor                [src/planning/conditional_task_executor.py]
  evaluates check-* sensing actions against ClutterGrounder at runtime
  triggers recovery / replanning on FALSE predicates
     │ per-action call
     ▼
SkillDecomposer (LLM)                  [src/primitives/skill_decomposer.py]
  action_str → SkillPlan of PrimitiveCalls
  (falls back to hardcoded pick/place/displace plans)
     │ SkillPlan
     ▼
PrimitiveExecutor                      [src/primitives/primitive_executor.py]
  back-projects pixel/depth cues → metric positions
  validates parameters against PRIMITIVE_LIBRARY
     │ method calls
     ▼
XArmPybulletPlannedPrimitives          [src/kinematics/xarm_pybullet_planned_primitives.py]
  plans trajectory in PyBullet (SRMP wAstar or linear fallback)
  streams waypoints to real xArm SDK
```

---

## Module reference

### `src/camera/`
| File | Purpose |
|---|---|
| [base_camera.py](src/camera/base_camera.py) | Abstract camera interface |
| [realsense_camera.py](src/camera/realsense_camera.py) | RealSense D435 driver (RGB-D, intrinsics) |
| [webcam_camera.py](src/camera/webcam_camera.py) | Webcam fallback |
| [camera_utils.py](src/camera/camera_utils.py) | Shared frame helpers |

### `src/kinematics/`
| File | Purpose |
|---|---|
| [base_pybullet_interface.py](src/kinematics/base_pybullet_interface.py) | Base FK/IK/planning class; SRMP wAstar motion planner |
| [xarm_pybullet_interface.py](src/kinematics/xarm_pybullet_interface.py) | xArm7 subclass with URDF defaults |
| [xarm_pybullet_planned_primitives.py](src/kinematics/xarm_pybullet_planned_primitives.py) | Plans in PyBullet, executes on real xArm |
| [xarm_curobo_interface.py](src/kinematics/xarm_curobo_interface.py) | Legacy CuRobo-backed interface |
| [stretch_pybullet_interface.py](src/kinematics/stretch_pybullet_interface.py) | Hello Robot Stretch subclass |
| [depth_environment_collider.py](src/kinematics/depth_environment_collider.py) | Depth → PyBullet mesh bodies for collision checking |
| [b1_robot_interface.py](src/kinematics/b1_robot_interface.py) | Unitree B1 quadruped interface |
| [b1_z1_system.py](src/kinematics/b1_z1_system.py) | B1 + Z1 arm combined system |
| [z1_robot_interface.py](src/kinematics/z1_robot_interface.py) | Unitree Z1 arm interface |
| [b1_z1_transform_calculator.py](src/kinematics/b1_z1_transform_calculator.py) | Coordinate transforms for B1+Z1 |
| **sim/** | PyBullet simulation primitives, camera, RoboCasa environment |

### `src/perception/`
| File | Purpose |
|---|---|
| [gsam2_object_tracker.py](src/perception/gsam2_object_tracker.py) | Main tracker: RAM+ → GroundingDINO → SAM2 + Molmo |
| [object_registry.py](src/perception/object_registry.py) | `DetectedObject`, `DetectedObjectRegistry`, `InteractionPoint` |
| [object_tracker.py](src/perception/object_tracker.py) | Base `ObjectTracker` / `ContinuousObjectTracker` ABC |
| [clearance.py](src/perception/clearance.py) | Gripper clearance profiles from depth masks |
| [contact_graph.py](src/perception/contact_graph.py) | Object contact/support graph from depth + masks |
| [llm_contact_classifier.py](src/perception/llm_contact_classifier.py) | LLM-based contact relation classifier |
| [occlusion.py](src/perception/occlusion.py) | Per-object occlusion map computation |
| [surface_map.py](src/perception/surface_map.py) | Planar surface detection and mapping |
| [molmo_point_detector.py](src/perception/molmo_point_detector.py) | Molmo VLM → pixel interaction point detection |
| [molmo_interaction_point_detector.py](src/perception/molmo_interaction_point_detector.py) | Wraps Molmo into the tracker's interaction-point pipeline |
| **gsam2/** | `IncrementalObjectTracker`, RAM/OpenAI taggers, mask utilities |
| **utils/coordinates.py** | Depth back-projection, pixel↔normalised conversions |

### `src/planning/`
| File | Purpose |
|---|---|
| [task_orchestrator.py](src/planning/task_orchestrator.py) | Top-level async orchestration: perception → domain gen → solving |
| [layered_domain_generator.py](src/planning/layered_domain_generator.py) | L1–L5 PDDL domain generation pipeline |
| [clutter_module.py](src/planning/clutter_module.py) | Clutter predicate library + `ClutterGrounder` |
| [conditional_task_executor.py](src/planning/conditional_task_executor.py) | Executes plan with runtime check-* branching and replanning |
| [hybrid_planner.py](src/planning/hybrid_planner.py) | Hybrid task-motion planner entry point |
| [pddl_solver.py](src/planning/pddl_solver.py) | Fast Downward wrapper (`SolverResult`, `SolverBackend`) |
| [pddl_representation.py](src/planning/pddl_representation.py) | PDDL domain/problem data model |
| [pddl_domain_maintainer.py](src/planning/pddl_domain_maintainer.py) | Incremental domain updates across task cycles |
| [domain_knowledge_base.py](src/planning/domain_knowledge_base.py) | Persistent predicate/action knowledge base (DKB) |
| [llm_task_analyzer.py](src/planning/llm_task_analyzer.py) | LLM-based NL task → structured `TaskAnalysis` |
| [task_state_monitor.py](src/planning/task_state_monitor.py) | FSM over `TaskState` enum |
| **utils/** | `pddl_types.py`, `snapshot_utils.py`, `task_types.py` |

### `src/primitives/`
| File | Purpose |
|---|---|
| [skill_plan_types.py](src/primitives/skill_plan_types.py) | `PrimitiveCall`, `SkillPlan`, `PRIMITIVE_LIBRARY` |
| [skill_decomposer.py](src/primitives/skill_decomposer.py) | LLM decomposes PDDL action → `SkillPlan` |
| [primitive_executor.py](src/primitives/primitive_executor.py) | Validates + executes `SkillPlan` against robot interface |

### `src/llm_interface/`
| File | Purpose |
|---|---|
| [base.py](src/llm_interface/base.py) | `LLMClient` ABC, `GenerateConfig`, `ImagePart` |
| [openai_client.py](src/llm_interface/openai_client.py) | OpenAI / GPT-4o client |
| [google_genai.py](src/llm_interface/google_genai.py) | Google Gemini client |
| [qwen3vl.py](src/llm_interface/qwen3vl.py) | Local Qwen3-VL client |

---

## Key data types

| Type | Defined in | Role |
|---|---|---|
| `DetectedObject` | [object_registry.py](src/perception/object_registry.py) | Per-object perception snapshot: 3-D position, mask, clearance, contact graph, interaction points |
| `InteractionPoint` | [object_registry.py](src/perception/object_registry.py) | Molmo-grounded pixel + 3-D position for pick/place/displace |
| `PrimitiveCall` | [skill_plan_types.py](src/primitives/skill_plan_types.py) | `(name, parameters, references)` — single robot primitive invocation |
| `SkillPlan` | [skill_plan_types.py](src/primitives/skill_plan_types.py) | Ordered list of `PrimitiveCall`s for one symbolic action |
| `SolverResult` | [pddl_solver.py](src/planning/pddl_solver.py) | `success`, `plan: List[str]`, `plan_length`, `error_message` |
| `ConditionalExecutionResult` | [conditional_task_executor.py](src/planning/conditional_task_executor.py) | Per-step trace with predicate values, recovery flags |

---

## Robot interfaces

Two parallel surfaces share the same duck-typed API (`get_robot_joint_state`, `move_to_pose`, etc.):

- **Simulation / planning**: `XArmPybulletInterface` (extends `BasePybulletInterface`) — FK/IK only, no hardware, uses SRMP for collision-free trajectory planning.
- **Real hardware**: `XArmPybulletPlannedPrimitives` — wraps a live xArm SDK adapter; trajectories are planned via `XArmPybulletInterface` then streamed waypoint-by-waypoint to the real arm.
- **B1+Z1 mobile system**: `B1Z1System` — Unitree quadruped + Z1 arm, separate transform calculator.
- **Stretch**: `StretchPybulletInterface` — Hello Robot Stretch subclass.

`DepthEnvironmentCollider` converts a live RealSense depth frame into per-object PyBullet mesh bodies so the SRMP planner receives scene geometry as a point cloud.

---

## Configuration

| File | Purpose |
|---|---|
| [config/orchestrator_config.py](config/orchestrator_config.py) | `OrchestratorConfig` dataclass — single source of truth for all pipeline knobs |
| [config/layered_domain_generator_prompts.yaml](config/layered_domain_generator_prompts.yaml) | L1–L5 LLM prompt templates |
| [config/skill_decomposer_prompts.yaml](config/skill_decomposer_prompts.yaml) | Skill decomposition prompt template |
| [config/primitive_descriptions.md](config/primitive_descriptions.md) | Human-readable primitive docs injected into decomposer prompts |
| [config/camera_config.yaml](config/camera_config.yaml) | RealSense resolution / FPS defaults |

---

## Environment variables

| Variable | Used by |
|---|---|
| `SAM2_CKPT` | GSAM2 tracker model checkpoint path |
| `OPENAI_API_KEY` | OpenAI LLM client |
| `OPENAI_MODEL` | LLM model name (default `gpt-4o-mini`) |
| `ROBOT_IP` | xArm IP address (default `192.168.1.224`) |
| `OUTPUT_DIR` | Run output root (default `outputs/clutter_robot`) |

---

## Scripts

| Script | What it tests |
|---|---|
| [test_clutter_planning_realsense_robot.py](scripts/test_clutter_planning_realsense_robot.py) | **Full pipeline** — RealSense + GSAM2 + PDDL + xArm execution |
| [test_clutter_planning_realsense.py](scripts/test_clutter_planning_realsense.py) | Same pipeline, perception + planning only (no robot) |
| [test_clutter_planning.py](scripts/test_clutter_planning.py) | Planning with cached perception data |
| [test_perception_realsense.py](scripts/test_perception_realsense.py) | RealSense + GSAM2 perception only |
| [test_xarm_real_pybullet_planned_primitives.py](scripts/test_xarm_real_pybullet_planned_primitives.py) | PyBullet plan → real xArm execution |
| [test_xarm_pybullet_primitives.py](scripts/test_xarm_pybullet_primitives.py) | PyBullet primitives in simulation |
| [test_obstructed_fridge_sim.py](scripts/test_obstructed_fridge_sim.py) | RoboCasa obstructed-fridge scenario |
| [test_b1_z1.py](scripts/test_b1_z1.py) | B1+Z1 mobile manipulation |
| [test_molmo_point.py](scripts/test_molmo_point.py) | Molmo interaction-point detection |
| [decompose_pddl_action.py](scripts/decompose_pddl_action.py) | Offline skill decomposition debug tool |
| [replay_cached_demo.py](scripts/replay_cached_demo.py) | Replay a saved perception + plan run |
| [benchmark_saved_world.py](scripts/benchmark_saved_world.py) | Benchmark planner on saved world state |
