"""
Clutter Planning Integration Test — Live RealSense + xArm Robot Execution

Full end-to-end pipeline on real hardware:
  1. RealSense D435 captures aligned RGB-D frames
  2. GSAM2 detects and segments all objects; computes clearance/contact/occlusion
  3. ClutterGrounder evaluates clutter predicates against the registry
  4. LayeredDomainGenerator (L1–L5) produces a PDDL domain + problem
  5. Fast Downward solves the plan
  6. ConditionalTaskExecutor runs the plan; each symbolic action is decomposed into
     a SkillPlan of typed PrimitiveCalls, grounded from Molmo interaction points stored
     in the object registry, then executed via PrimitiveExecutor →
     XArmPybulletPlannedPrimitives (PyBullet IK + smooth trajectory through RawXArmRobotAdapter)

Action → primitive decomposition (positions from registry interaction points):
  pick ?obj ?surface       → open_gripper → move(approach) → move(ip["pick"]) → close_gripper → retract
  place ?obj ?surface      → move(approach, surface ip["place-on-top"]) → move(descend, is_place) → open_gripper → retract
  displace/push-aside/clear-obstruction ?blocker ...
                           → open_gripper → move(approach) → move(ip["displace"|"pick"]) → close_gripper → retract → open_gripper

Interaction-point priority (Molmo-grounded, fallback to centroid):
  pick        : ip["pick"]       → ip["displace"] → centroid
  place       : ip["place-on-top"] → ip["pick"]   → centroid
  displace    : ip["displace"]   → ip["pick"]     → ip["push-aside"] → centroid
  push-aside  : ip["push-aside"] → ip["displace"] → ip["pick"]       → centroid

Requires:
    SAM2_CKPT env var
    OPENAI_API_KEY env var
    ROBOT_IP env var (or --robot-ip; default 192.168.1.224)
    RealSense D435 via USB3

Run:
    uv run scripts/test_clutter_planning_realsense_robot.py
    uv run scripts/test_clutter_planning_realsense_robot.py --no-execute
    uv run scripts/test_clutter_planning_realsense_robot.py --task "Pick up the red cup"
    uv run scripts/test_clutter_planning_realsense_robot.py --robot-ip 192.168.1.224
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_CONFIG_PATH = PROJECT_ROOT / "config"
if str(_CONFIG_PATH) not in sys.path:
    sys.path.insert(0, str(_CONFIG_PATH))

from src.perception.object_registry import DetectedObjectRegistry
from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
from src.primitives.primitive_executor import PrimitiveExecutor
from src.primitives.skill_plan_types import PrimitiveCall, SkillPlan, SkillPlanDiagnostics

# ---------------------------------------------------------------------------
# Colours / logging
# ---------------------------------------------------------------------------

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_CYAN    = "\033[36m"
_RED     = "\033[31m"
_BLUE    = "\033[34m"
_GREY    = "\033[90m"


def _setup_logging() -> logging.Logger:
    class _Fmt(logging.Formatter):
        _MAP = {
            "DEBUG":    _GREY,
            "INFO":     _CYAN,
            "WARNING":  _YELLOW,
            "ERROR":    _RED,
            "CRITICAL": _RED + _BOLD,
        }
        def format(self, r: logging.LogRecord) -> str:
            color = self._MAP.get(r.levelname, "")
            ts    = self.formatTime(r, "%H:%M:%S")
            name  = r.name.split(".")[-1][:18]
            return (f"{_GREY}{ts}{_RESET} "
                    f"{color}{r.levelname:<8}{_RESET} "
                    f"{_GREY}[{name}]{_RESET} {r.getMessage()}")

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "models", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("clutter_robot")


log = _setup_logging()


def _banner(msg: str) -> None:
    print()
    print(_BOLD + _BLUE + "=" * 72 + _RESET)
    print(_BOLD + _BLUE + f"  {msg}" + _RESET)
    print(_BOLD + _BLUE + "=" * 72 + _RESET)


def _section(msg: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {msg} " + "─" * max(0, 68 - len(msg)) + _RESET)
    log.info("BEGIN: %s", msg)


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _info(label: str, value: Any = "") -> None:
    if value == "":
        print(f"    {_YELLOW}•{_RESET} {label}")
    else:
        print(f"    {_YELLOW}{label}:{_RESET} {value}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")
    log.warning(msg)


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")
    log.error(msg)


# ---------------------------------------------------------------------------
# Run output directory / timing
# ---------------------------------------------------------------------------

_RUN_ID          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/clutter_robot"))
OUTPUT_DIR       = _BASE_OUTPUT_DIR / "runs" / _RUN_ID

DEFAULT_TASK     = "Pick up the rubber duck from the cluttered table"
DEFAULT_ROBOT_IP = "192.168.1.224"

_timings: Dict[str, float] = {}
_t_total: float = 0.0


def _tick() -> float:
    return time.monotonic()


def _tock(label: str, t0: float) -> float:
    elapsed = time.monotonic() - t0
    _timings[label] = elapsed
    return elapsed


def _init_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = _BASE_OUTPUT_DIR / "runs" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUTPUT_DIR.resolve())
    log.info("Output dir: %s", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# XArm connection via RawXArmRobotAdapter
# ---------------------------------------------------------------------------

def connect_xarm(robot_ip: str) -> Optional[Any]:
    """Connect to the xArm via RawXArmRobotAdapter. Returns adapter or None."""
    _section("Connecting to xArm")
    try:
        # Import inline to avoid hard dependency when xarm SDK is absent
        import sys as _sys
        import importlib.util as _util
        _spec = _util.spec_from_file_location(
            "test_xarm_real_pybullet_planned_primitives",
            Path(__file__).parent / "test_xarm_real_pybullet_planned_primitives.py",
        )
        _mod = _util.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        RawXArmRobotAdapter = _mod.RawXArmRobotAdapter
    except Exception as exc:
        _warn(f"RawXArmRobotAdapter not importable ({exc}) — robot execution disabled")
        return None

    try:
        adapter = RawXArmRobotAdapter(robot_ip)
        joints = adapter.get_robot_joint_state()
        if joints is not None:
            _ok(f"xArm connected at {robot_ip}  joints={np.round(joints, 3).tolist()}")
        else:
            _warn("Connected but could not read joint state")
        return adapter
    except Exception as exc:
        _warn(f"xArm connection failed ({exc}) — running without execution")
        log.exception("connect_xarm failed")
        return None


# ---------------------------------------------------------------------------
# Perception-grounded primitive plan builder
# ---------------------------------------------------------------------------

_APPROACH_HEIGHT = 0.12

# Interaction-point key priority for each PDDL action.  The first key found in
# obj.interaction_points wins; the last entry is always "centroid" (sentinel).
_ACTION_IP_PRIORITY: Dict[str, List[str]] = {
    "pick":              ["pick", "displace"],
    "place":             ["place-on-top", "pick"],
    "displace":          ["displace", "pick", "push-aside"],
    "push-aside":        ["push-aside", "displace", "pick"],
    "clear-obstruction": ["displace", "pick", "push-aside"],
}


def _get_action_pos(
    registry: DetectedObjectRegistry,
    object_id: str,
    action: str,
) -> Optional[np.ndarray]:
    """Return the 3-D interaction point for *action*, trying keys in priority order.

    Falls back to the object centroid when no interaction point is available.
    """
    obj = registry.get_object(object_id)
    if obj is None:
        return None
    ips = obj.interaction_points or {}
    for key in _ACTION_IP_PRIORITY.get(action, [action]):
        ip = ips.get(key)
        if ip is not None and getattr(ip, "position_3d", None) is not None:
            pos = np.asarray(ip.position_3d, dtype=float)
            _info(f"  {object_id}/{action}", f"ip[{key!r}]={pos.round(3).tolist()}")
            return pos
    if obj.position_3d is not None:
        pos = np.asarray(obj.position_3d, dtype=float)
        _info(f"  {object_id}/{action}", f"centroid fallback={pos.round(3).tolist()}")
        return pos
    return None


def _build_pick_plan(
    obj_id: str,
    pos: np.ndarray,
    speed_factor: float,
) -> SkillPlan:
    """Approach → descend → close → retract."""
    approach = (pos + np.array([0.0, 0.0, _APPROACH_HEIGHT])).tolist()
    return SkillPlan(
        action_name="pick",
        primitives=[
            PrimitiveCall("open_gripper", {}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }, references={"object_id": obj_id}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": pos.tolist(),
                "preset_orientation": "top_down",
                "speed_factor": speed_factor * 0.6,
                "execute": True,
            }, references={"object_id": obj_id}),
            PrimitiveCall("close_gripper", {}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }),
            PrimitiveCall("retract_gripper", {"execute": True}),
        ],
        diagnostics=SkillPlanDiagnostics(),
    )


def _build_place_plan(
    surface_id: str,
    pos: np.ndarray,
    speed_factor: float,
) -> SkillPlan:
    """Approach → descend (is_place) → open → retract."""
    approach = (pos + np.array([0.0, 0.0, _APPROACH_HEIGHT])).tolist()
    return SkillPlan(
        action_name="place",
        primitives=[
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }, references={"object_id": surface_id}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": pos.tolist(),
                "preset_orientation": "top_down",
                "is_place": True,
                "speed_factor": speed_factor * 0.6,
                "execute": True,
            }, references={"object_id": surface_id}),
            PrimitiveCall("open_gripper", {}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }),
            PrimitiveCall("retract_gripper", {"execute": True}),
        ],
        diagnostics=SkillPlanDiagnostics(),
    )


def _build_displace_plan(
    obj_id: str,
    pos: np.ndarray,
    speed_factor: float,
) -> SkillPlan:
    """Approach → descend → close → retract (arm home) → open (drop clear)."""
    approach = (pos + np.array([0.0, 0.0, _APPROACH_HEIGHT])).tolist()
    return SkillPlan(
        action_name="displace",
        primitives=[
            PrimitiveCall("open_gripper", {}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }, references={"object_id": obj_id}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": pos.tolist(),
                "preset_orientation": "top_down",
                "speed_factor": speed_factor * 0.6,
                "execute": True,
            }, references={"object_id": obj_id}),
            PrimitiveCall("close_gripper", {}),
            PrimitiveCall("move_gripper_to_pose", {
                "target_position": approach,
                "preset_orientation": "top_down",
                "speed_factor": speed_factor,
                "execute": True,
            }),
            PrimitiveCall("retract_gripper", {"execute": True}),
            PrimitiveCall("open_gripper", {}),
        ],
        diagnostics=SkillPlanDiagnostics(),
    )


# ---------------------------------------------------------------------------
# Action dispatcher
# ---------------------------------------------------------------------------

def build_action_executor(
    primitives: Optional[XArmPybulletPlannedPrimitives],
    registry: DetectedObjectRegistry,
    no_execute: bool,
    speed_factor: float,
    executor: Optional[PrimitiveExecutor] = None,
):
    """Return a synchronous callable that maps PDDL action strings to robot motion.

    Each PDDL action is decomposed into a SkillPlan of typed PrimitiveCalls.
    Positions are grounded from Molmo interaction points stored in the registry,
    with a priority-ordered fallback chain before resorting to the centroid.
    The SkillPlan is executed via PrimitiveExecutor which handles coordinate
    transforms and parameter validation.
    """

    def _execute_action(action_str: str) -> bool:
        tokens      = action_str.strip("() ").split()
        action_name = tokens[0].lower() if tokens else ""
        params      = tokens[1:]

        _info("Execute", action_str)

        if no_execute or primitives is None:
            _info("    → dry-run (no robot motion)")
            return True

        try:
            plan: Optional[SkillPlan] = None

            if action_name == "pick" and len(params) >= 1:
                obj_id = params[0]
                pos = _get_action_pos(registry, obj_id, "pick")
                if pos is None:
                    _warn(f"pick: no position for {obj_id}")
                    return False
                plan = _build_pick_plan(obj_id, pos, speed_factor)

            elif action_name == "place" and len(params) >= 2:
                obj_id, surface_id = params[0], params[1]
                pos = _get_action_pos(registry, surface_id, "place")
                if pos is None:
                    _warn(f"place: no position for surface {surface_id}")
                    return False
                plan = _build_place_plan(surface_id, pos, speed_factor)

            elif action_name in ("displace", "push-aside", "clear-obstruction") and len(params) >= 1:
                obj_id = params[0]
                pos = _get_action_pos(registry, obj_id, action_name)
                if pos is None:
                    _warn(f"{action_name}: no position for {obj_id}")
                    return False
                plan = _build_displace_plan(obj_id, pos, speed_factor)

            else:
                _warn(f"    No executor for '{action_name}' — skipping")
                return True

            if plan is None:
                return False

            if executor is not None:
                world_state: Dict[str, Any] = {}
                result = executor.execute_plan(plan, world_state)
                return result.executed
            else:
                # Direct execution without PrimitiveExecutor (no coordinate transforms).
                for prim in plan.primitives:
                    method = getattr(primitives, prim.name, None)
                    if not callable(method):
                        _warn(f"    Missing primitive '{prim.name}' on interface — skipping")
                        continue
                    r = method(**prim.parameters)
                    if isinstance(r, dict) and not r.get("success", True):
                        _warn(f"    {prim.name} failed: {r.get('reason')}")
                        return False
                return True

        except Exception as exc:
            _fail(f"    Action '{action_name}' raised: {exc}")
            log.exception("execute_action failed for %s", action_str)
            return False

    return _execute_action


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    global _t_total
    _t_total = time.monotonic()

    _init_output_dir()

    if args.no_gui:
        plt.ioff()

    _banner("Clutter Planning — Live RealSense + xArm Robot")
    _info("Task",    args.task)
    _info("Run ID",  _RUN_ID)
    _info("Output",  OUTPUT_DIR)
    _info("Execute", f"{'DRY-RUN' if args.no_execute else 'LIVE (' + args.robot_ip + ')'}")

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set")
        return 1

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info("Device", device)

    # --- LLM client ---
    _section("Setting up LLM client")
    t0 = _tick()
    llm_client   = None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from src.llm_interface import OpenAIClient
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            llm_client = OpenAIClient(model=openai_model, api_key=key)
            _ok(f"LLM client ready: {openai_model}")
        else:
            _warn("OPENAI_API_KEY not set — planning disabled")
    except Exception as exc:
        _warn(f"LLM client setup failed ({exc})")
    _tock("llm_client_init", t0)

    # --- xArm connection via RawXArmRobotAdapter ---
    t0        = _tick()
    adapter   = None
    primitives: Optional[XArmPybulletPlannedPrimitives] = None
    if not args.no_execute:
        adapter = connect_xarm(args.robot_ip)
    _tock("robot_connect", t0)

    # --- PyBullet FK/IK interface (planning + transforms) ---
    _section("Setting up PyBullet FK/IK interface")
    t0 = _tick()
    pybullet_iface = XArmPybulletInterface(use_gui=args.pybullet_gui)
    initial_joints = adapter.get_robot_joint_state() if adapter is not None else None
    if initial_joints is None:
        raise RuntimeError("Cannot initialize FK: failed to read joint state from robot")
    pybullet_iface.set_current_joint_state(initial_joints)
    cam_pos, cam_rot = pybullet_iface.get_camera_transform()
    if cam_pos is not None:
        _ok(f"PyBullet FK ready  camera_pos={cam_pos.round(3).tolist()}")
    else:
        _warn("Camera transform unavailable from PyBullet FK")
    _tock("pybullet_init", t0)

    if adapter is not None:
        primitives = XArmPybulletPlannedPrimitives(
            robot=adapter,
            planner=pybullet_iface,
            logger=log.getChild("Primitives"),
            use_gui=args.pybullet_gui,
        )
        # Registry injected after tracker is initialized below.

    # --- Orchestrator + GSAM2 setup ---
    _section("Configuring TaskOrchestrator + GSAM2 tracker")
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator
    from src.perception import GSAM2ContinuousObjectTracker
    from src.perception.clearance import GripperGeometry

    dkb_dir = OUTPUT_DIR / "dkb"

    config = OrchestratorConfig(
        api_key="unused",
        llm_client=llm_client,

        camera_width=640,
        camera_height=480,
        camera_fps=30,
        enable_depth=True,

        update_interval=4.0,
        min_observations=1,
        fast_mode=False,

        state_dir=OUTPUT_DIR / "orchestrator_state",
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,

        enable_snapshots=True,
        snapshot_every_n_detections=1,
        perception_pool_dir=OUTPUT_DIR / "perception_pool",
        debug_frames_dir=OUTPUT_DIR / "debug_frames",

        use_layered_generation=True,
        dkb_dir=dkb_dir,

        solver_backend="auto",
        solver_timeout=60.0,
        solver_verbose=False,
        auto_solve_when_ready=False,
        max_refinement_attempts=3,
        auto_refine_on_failure=True,

        robot=pybullet_iface,

        on_state_change=lambda old, new: log.info(
            "Orchestrator: %s → %s", old.value, new.value
        ),
    )

    # Orchestrator initialises the RealSense camera internally.
    t0 = _tick()
    orchestrator = TaskOrchestrator(config=config)
    await orchestrator.initialize()
    _tock("orchestrator_init", t0)
    _ok("Orchestrator + RealSense initialised")

    # Sync FK from live joints now that the camera is up.
    if adapter is not None:
        live_joints = adapter.get_robot_joint_state()
        if live_joints is not None:
            pybullet_iface.set_current_joint_state(live_joints)
            cam_pos, cam_rot = pybullet_iface.get_camera_transform()
            _ok(f"FK synced from live joints  camera_pos={cam_pos.round(3).tolist() if cam_pos is not None else 'N/A'}")

    # Swap the default ContinuousObjectTracker for GSAM2 before any detection starts.
    _section("Loading GSAM2 tracker")
    t0 = _tick()
    gsam2_tracker = GSAM2ContinuousObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        update_interval=config.update_interval,
        on_detection_complete=orchestrator._on_detection_callback,
        llm_client=llm_client if args.llm else None,
        llm_mode=args.llm_mode,
        robot=pybullet_iface,
        gripper=GripperGeometry(),
        logger=log.getChild("GSAM2Tracker"),
        debug_save_dir=OUTPUT_DIR / "debug_frames",
        use_molmo=True,
    )
    orchestrator.tracker = gsam2_tracker
    # Wire GSAM2 to pull frames from the orchestrator's RealSense camera.
    gsam2_tracker.set_frame_provider(orchestrator._get_camera_frames)
    _tock("model_load", t0)
    _ok(f"GSAM2 tracker active on {device}")

    # --- Perception pass ---
    # Run detection FIRST so the registry is populated with real object IDs.
    # process_task_request (L1-L5 domain generation) uses get_detected_objects()
    # internally — if called with an empty registry it generates abstract type
    # names instead of grounded IDs, causing the grounding validation failure.
    #
    # start_detection() requires current_task to be set, so we seed it directly
    # here without calling process_task_request yet.  The tracker task context
    # tells GSAM2 what to look for during this perception pass.
    _section("Perception pass (RealSense + GSAM2)")
    orchestrator.current_task = args.task
    orchestrator.tracker.set_task_context(task_description=args.task)
    t0 = _tick()
    await orchestrator.start_detection()

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        if (orchestrator.detection_count >= 1
                and orchestrator.tracker.registry.get_all_objects()):
            break
        _info("  waiting for first detection", orchestrator.detection_count)

    await orchestrator.stop_detection()

    # Build depth → PyBullet collision meshes (per-object + ground plane boundary)
    # while tracker state is fresh.  The PyBullet plane.urdf sits at Z=0 (robot base
    # origin) and is already loaded by XArmPybulletInterface, so floor_z=0.0 in the
    # antipodal grasp sampler will reject any candidate that would hit it.
    _masks = dict(getattr(orchestrator.tracker, "_last_masks", {}))
    if orchestrator._camera is not None:
        _section("Depth → PyBullet collision meshes")
        try:
            from src.kinematics.depth_environment_collider import (
                DepthEnvironmentCollider, BACKGROUND_ID,
            )
            depth_collider = DepthEnvironmentCollider(pybullet_iface)
            build_results = depth_collider.update(orchestrator._camera, _masks)
            for label, ok in build_results.items():
                tag = "(background)" if label == BACKGROUND_ID else "(object)"
                body_id = (depth_collider.background_body_id if label == BACKGROUND_ID
                           else depth_collider.body_id_for(label))
                if ok:
                    _ok(f"  [{label}] {tag}  body_id={body_id}")
                else:
                    _warn(f"  [{label}] {tag}  mesh build failed")
            if any(build_results.values()):
                pybullet_iface.attach_collider(depth_collider)
                _ok("Collider attached to planner — trajectories will be collision-checked")
        except Exception:
            log.exception("Depth collider injection failed")

    _tock("perception_pass", t0)

    registry     = orchestrator.tracker.registry
    if primitives is not None:
        primitives._registry = registry
    detected_ids = [o.object_id for o in registry.get_all_objects()]
    _info("Detected objects", detected_ids)

    if not detected_ids:
        _warn("No objects detected — aborting")
        await orchestrator.save_state()
        return 1

    try:
        last_color = orchestrator.tracker.last_color_frame
        if last_color is not None:
            from PIL import Image as _PIL
            _PIL.fromarray(last_color).save(OUTPUT_DIR / "color_frame.png")
            _ok(f"Frame saved → {OUTPUT_DIR / 'color_frame.png'}")
    except Exception:
        log.exception("Failed to save camera frame")

    _ok("Perception complete")

    # Refresh FK from live joints and update cam_pos/cam_rot for PostDisplacementHook.
    if adapter is not None:
        live_joints = adapter.get_robot_joint_state()
        if live_joints is not None:
            pybullet_iface.set_current_joint_state(live_joints)
            cam_pos, cam_rot = pybullet_iface.get_camera_transform()
            _ok(f"FK re-synced  camera_pos={cam_pos.round(3).tolist() if cam_pos is not None else 'N/A'}")

    # --- Task analysis + PDDL domain generation (L1–L5) ---
    # Now that the registry has real object IDs, process_task_request will pass
    # them to the layered generator so goals are grounded to detected objects.
    _section("Task Analysis + Domain Generation (L1–L5)")
    t0 = _tick()
    try:
        last_frame = getattr(orchestrator.tracker, "last_color_frame", None)
        await orchestrator.process_task_request(args.task, environment_image=last_frame)
        _ok("Task analysis complete")
    except Exception:
        log.exception("Task analysis failed")
    _tock("pddl_generation", t0)

    _section("PDDL Solving")
    t0 = _tick()
    solver_result = None
    try:
        solver_result = await orchestrator.solve_and_plan_with_refinement(
            output_dir=OUTPUT_DIR / "pddl",
            wait_for_objects=False,
        )
        _tock("solving", t0)

        pddl_dir      = OUTPUT_DIR / "pddl"
        domain_files  = sorted(pddl_dir.glob("*_domain.pddl"))  if pddl_dir.exists() else []
        problem_files = sorted(pddl_dir.glob("*_problem.pddl")) if pddl_dir.exists() else []
        if domain_files:
            _section("Generated PDDL Domain")
            print(domain_files[-1].read_text())
        if problem_files:
            _section("Generated PDDL Problem")
            print(problem_files[-1].read_text())

        if solver_result and solver_result.success:
            _ok(f"Plan found — {solver_result.plan_length} steps")
            for i, step in enumerate(solver_result.plan, 1):
                _info(f"  {i}", step)
        else:
            err = (getattr(solver_result, "error_message", "unknown")
                   if solver_result else "no result")
            _warn(f"Planning failed: {err}")
    except Exception:
        log.exception("Solving failed")
        _tock("solving", t0)

    # --- Execution ---
    if solver_result and solver_result.success and solver_result.plan:
        _section("Executing plan on xArm")

        prim_executor = PrimitiveExecutor(
            primitives=primitives,
            perception_pool_dir=OUTPUT_DIR / "perception_pool",
            logger=log.getChild("PrimitiveExecutor"),
            orchestrator=orchestrator,
        ) if primitives is not None else None

        t0 = _tick()
        try:
            from src.planning.conditional_task_executor import ConditionalTaskExecutor
            from src.planning.clutter_module import PostDisplacementHook
            from src.primitives.skill_decomposer import SkillDecomposer

            cam_intrinsics = None
            try:
                if orchestrator._camera is not None:
                    cam_intrinsics = orchestrator._camera.get_camera_intrinsics()
            except Exception:
                pass

            post_hook = PostDisplacementHook(
                registry=registry,
                camera_intrinsics=cam_intrinsics,
                cam_position=cam_pos,
                cam_rotation=cam_rot,
                logger=log.getChild("PostDisplacementHook"),
            )

            decomposer = None
            if llm_client is not None:
                decomposer = SkillDecomposer(llm_client=llm_client, orchestrator=orchestrator)
            else:
                _warn("LLM client not available — skill decomposition will use hardcoded plans")

            hardcoded_execute_fn = build_action_executor(
                primitives=primitives,
                registry=registry,
                no_execute=args.no_execute,
                speed_factor=args.speed_factor,
                executor=prim_executor,
            )

            def execute_fn(action_str: str) -> bool:
                tokens      = action_str.strip("() ").split()
                action_name = tokens[0] if tokens else action_str
                param_ids   = tokens[1:] if len(tokens) > 1 else []
                params      = {"objects": param_ids, "action_str": action_str}

                if decomposer is not None:
                    try:
                        skill_plan = decomposer.plan(action_name, params)
                        _ok(f"    Decomposed → {[p.name for p in skill_plan.primitives]}")
                        for w in (skill_plan.diagnostics.warnings or []):
                            _warn(f"    {w}")
                        if skill_plan.primitives and prim_executor is not None:
                            # Capture a fresh frame so Molmo sees the current scene,
                            # not the snapshot from before task execution began.
                            fresh_snapshot_id = orchestrator.capture_fresh_snapshot_sync(
                                reason=f"pre-execution:{action_name}"
                            )
                            world_state_for_exec = {"state_dir": str(orchestrator.config.state_dir)}
                            if fresh_snapshot_id:
                                world_state_for_exec["last_snapshot_id"] = fresh_snapshot_id
                            result = prim_executor.execute_plan(
                                skill_plan,
                                world_state=world_state_for_exec,
                                dry_run=args.no_execute,
                            )
                            return result.executed or args.no_execute
                        _warn(f"    Decomposer returned no primitives — falling back to hardcoded plan")
                    except Exception as exc:
                        _warn(f"    Decomposition failed for '{action_name}': {exc}")

                return hardcoded_execute_fn(action_str)

            async def _replan() -> List[str]:
                _info("Replanning: fresh perception pass...")
                # Sync FK before re-perceiving so transforms are current.
                if adapter is not None:
                    live_joints = adapter.get_robot_joint_state()
                    if live_joints is not None:
                        pybullet_iface.set_current_joint_state(live_joints)
                # Run a short detection pass to update the registry.
                await orchestrator.start_detection()
                replan_deadline = time.monotonic() + 30.0
                while time.monotonic() < replan_deadline:
                    await asyncio.sleep(1.0)
                    if orchestrator.tracker.registry.get_all_objects():
                        break
                await orchestrator.stop_detection()
                last_frame = getattr(orchestrator.tracker, "last_color_frame", None)
                await orchestrator.process_task_request(args.task, environment_image=last_frame)
                result = await orchestrator.solve_and_plan_with_refinement(
                    output_dir=OUTPUT_DIR / "pddl", wait_for_objects=False
                )
                return result.plan if result and result.success else []

            cond_executor = ConditionalTaskExecutor(
                registry=registry,
                execute_action=execute_fn,
                replan_fn=_replan,
                max_replan=getattr(orchestrator.config, "max_refinement_attempts", 2),
                post_displacement_hook=post_hook,
                logger=log.getChild("ConditionalExecutor"),
            )

            cond_result = await cond_executor.execute(solver_result.plan)

            if cond_result.success:
                _ok(f"Execution complete — {len(cond_result.steps)} steps, "
                    f"{cond_result.replan_count} replan(s)")
            else:
                _warn(f"Execution failed: {cond_result.error}")

            _section("Step-by-step trace")
            for step in cond_result.steps:
                pred_tag     = (f"  [{'TRUE' if step.predicate_value else 'FALSE'}]"
                                if step.predicate_value is not None else "")
                recovery_tag = f"  → {step.recovery_action}" if step.recovery_triggered else ""
                _info(f"  {'✓' if step.success else '✗'} {step.action}{pred_tag}{recovery_tag}",
                      step.notes or "")

        except Exception:
            log.exception("Conditional execution failed")
        _tock("execution", t0)

        if primitives is not None and not args.no_execute:
            primitives.retract_gripper(execute=True)
            _ok("Arm returned to home")

    # --- Save state ---
    _section("Saving run state")
    t0 = _tick()
    await orchestrator.save_state()
    _tock("save_state", t0)

    # Disconnect
    if adapter is not None:
        try:
            adapter.disconnect()
        except Exception:
            pass

    pybullet_iface.cleanup()

    # --- Timing report ---
    t_total_elapsed = time.monotonic() - _t_total
    _banner("Run Complete")
    _info("Run ID", _RUN_ID)
    _info("Output", OUTPUT_DIR)

    _TIMING_LABELS = [
        ("llm_client_init",   "LLM client init"),
        ("pybullet_init",     "PyBullet FK/IK init"),
        ("robot_connect",     "xArm SDK connect"),
        ("orchestrator_init", "Orchestrator init"),
        ("model_load",        "GSAM2 model load"),
        ("perception_pass",   "Perception pass"),
        ("pddl_generation",   "Task analysis + domain gen (L1–L5)"),
        ("solving",           "PDDL solving"),
        ("execution",         "Plan execution on xArm"),
        ("save_state",        "Save state"),
    ]

    print()
    print(_BOLD + _BLUE + "── Timing Report " + "─" * 55 + _RESET)
    col_w = 38
    for key, label in _TIMING_LABELS:
        if key in _timings:
            t   = _timings[key]
            pct = 100.0 * t / t_total_elapsed if t_total_elapsed > 0 else 0.0
            print(f"  {_YELLOW}{label:<{col_w}}{_RESET} {t:6.2f}s  ({pct:4.1f}%)")
    print(f"  {_BOLD}{'Total':<{col_w + 2}}{_RESET} {t_total_elapsed:6.2f}s")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clutter planning — live RealSense + xArm robot"
    )
    parser.add_argument("--no-gui",      action="store_true", help="Headless mode (no matplotlib plots)")
    parser.add_argument("--pybullet-gui", action="store_true",
                        help="Open PyBullet GUI to visualize planned trajectories")
    parser.add_argument("--no-execute", action="store_true",
                        help="Plan but do not send any motion commands")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Task description for the planner")
    parser.add_argument("--robot-ip",
                        default=os.getenv("ROBOT_IP", DEFAULT_ROBOT_IP),
                        help=f"xArm IP (default: {DEFAULT_ROBOT_IP} or $ROBOT_IP)")
    parser.add_argument(
        "--speed-factor", type=float, default=0.25,
        help="Trajectory speed multiplier 0–1 for all arm moves (default: 0.25)",
    )
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM for contact graph classification")
    parser.add_argument("--llm-mode", choices=["json", "nl"], default="nl")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
