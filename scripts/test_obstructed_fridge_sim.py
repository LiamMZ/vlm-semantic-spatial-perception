"""
Obstructed Fridge Sim — Full Perception + Planning Test
========================================================
Wires the ObstructedFridgeItem robocasa environment into TaskOrchestrator with:
  • RobocasaCamera   — frame provider from MuJoCo eye-in-hand camera
  • RobocasaRobotInterface — robot state / camera transform provider
  • RobocasaPrimitives     — action-space implementation of PRIMITIVE_LIBRARY
  • GSAM2ContinuousObjectTracker swapped into the orchestrator
  • LayeredDomainGenerator (L1–L5) for PDDL generation
  • PrimitiveExecutor + SkillDecomposer for plan execution

Pipeline
--------
1.  Create ObstructedFridgeItem env, reset, register sim adapters
2.  Initialise TaskOrchestrator with use_sim_camera=True
3.  Inject RobocasaCamera as the frame provider
4.  Swap in GSAM2ContinuousObjectTracker
5.  Collect N frames of random-action observations
6.  Process task request → L1-L5 PDDL generation
7.  Solve for plan, optionally execute via RobocasaPrimitives
8.  Save all state, print timing report

Requires:
    OPENAI_API_KEY (planning + optionally LLM contact classification)
    OPENAI_MODEL   (optional, default: gpt-4o-mini)
    SAM2_CKPT      (path to SAM2 checkpoint for GSAM2 tracker)

Run:
    uv run scripts/test_obstructed_fridge_sim.py
    uv run scripts/test_obstructed_fridge_sim.py --frames 5 --seed 1
    uv run scripts/test_obstructed_fridge_sim.py --no-plan
    uv run scripts/test_obstructed_fridge_sim.py --execute
    uv run scripts/test_obstructed_fridge_sim.py --no-gui
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CONFIG_PATH = _ROOT / "config"
if str(_CONFIG_PATH) not in sys.path:
    sys.path.insert(0, str(_CONFIG_PATH))

# ---------------------------------------------------------------------------
# ANSI colours / logging
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_YELLOW= "\033[33m"
_CYAN  = "\033[36m"
_RED   = "\033[31m"
_BLUE  = "\033[34m"
_GREY  = "\033[90m"


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
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib",
                  "robosuite", "robocasa"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("fridge_sim")


log = _setup_logging()


def _banner(msg: str) -> None:
    w = 72
    print()
    print(_BOLD + _BLUE + "=" * w + _RESET)
    print(_BOLD + _BLUE + f"  {msg}" + _RESET)
    print(_BOLD + _BLUE + "=" * w + _RESET)


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


# ---------------------------------------------------------------------------
# Run output directory
# ---------------------------------------------------------------------------

_RUN_ID          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/obstructed_fridge_sim"))
OUTPUT_DIR       = _BASE_OUTPUT_DIR / "runs" / _RUN_ID

DEFAULT_TASK   = None  # use robocasa's per-episode generated language
DEFAULT_FRAMES = 3
# agentview_left has a clear side-on view into the open fridge at reset pose.
# eye_in_hand points at the robot body until the arm is repositioned, so we
# use agentview_left as the primary perception camera.
CAM_NAME  = "robot0_agentview_left"
EIH_CAM   = "robot0_eye_in_hand"


def _init_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = _BASE_OUTPUT_DIR / "runs" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUTPUT_DIR.resolve())
    log.info("Output dir: %s", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_timings: Dict[str, float] = {}
_t_total: float = 0.0


def _tick() -> float:
    return time.monotonic()


def _tock(label: str, t0: float) -> float:
    elapsed = time.monotonic() - t0
    _timings[label] = elapsed
    return elapsed


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    global _t_total
    _t_total = time.monotonic()

    _init_output_dir()
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    debug_dir  = OUTPUT_DIR / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    _banner("Obstructed Fridge Sim — Full Perception + Planning Test")
    _info("Task",   args.task)
    _info("Run ID", _RUN_ID)
    _info("Output", OUTPUT_DIR)
    _info("Frames", args.frames)
    _info("Seed",   args.seed)
    _info("Robot",  args.robot)

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        return 1

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info("Device", device)

    # -----------------------------------------------------------------------
    _section("1 / LLM Client")
    # -----------------------------------------------------------------------
    t0 = _tick()
    llm_client = None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from src.llm_interface import OpenAIClient
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            _warn("OPENAI_API_KEY not set — planning disabled")
        else:
            llm_client = OpenAIClient(model=openai_model, api_key=openai_key)
            _ok(f"LLM client ready: {openai_model}")
    except Exception as exc:
        _warn(f"LLM client setup failed: {exc}")
    _tock("llm_client_init", t0)

    # -----------------------------------------------------------------------
    _section("2 / Robocasa Environment")
    # -----------------------------------------------------------------------
    t0 = _tick()
    from robocasa.utils.env_utils import create_env
    from robocasa.environments.kitchen.composite.loading_fridge.obstructed_fridge_item import (
        ObstructedFridgeItem,
    )

    cam_w, cam_h = args.cam_width, args.cam_height
    cameras = [CAM_NAME, EIH_CAM, "robot0_frontview"]

    env = create_env(
        env_name="ObstructedFridgeItem",
        robots=args.robot,
        camera_names=cameras,
        camera_widths=cam_w,
        camera_heights=cam_h,
        render_onscreen=False,
        seed=args.seed,
    )
    # create_env hardcodes camera_depths=False; RobocasaCamera renders depth
    # on demand via sim.render(depth=True) when get_depth() is called.
    obs = env.reset()

    # Tilt the wrist ~15° upward so the eye-in-hand camera faces more
    # toward the fridge interior rather than straight ahead.
    # Action [3:6] is EEF delta orientation (axis-angle); positive Y rotates
    # the wrist upward (pitch up from the robot's perspective).
    _WRIST_TILT_RAD = np.deg2rad(15.0)
    _wrist_action = np.zeros(getattr(env, "action_dim", 12))
    _wrist_action[4] = -_WRIST_TILT_RAD  # EEF delta pitch (Y-axis)
    for _ in range(5):  # apply over several steps so the controller tracks it
        obs, *_ = env.step(_wrist_action)

    _tock("env_create", t0)

    ep_meta  = env.get_ep_meta()
    task_str = (
        args.task
        or ep_meta.get("lang")
        or "Retrieve the target item from the fridge and place it on the counter"
    )
    _ok(f"Env ready — task: {task_str}")
    _info("Objects", list(env.objects.keys()))

    # -----------------------------------------------------------------------
    _section("3 / Sim Adapters (Camera, Robot, Primitives)")
    # -----------------------------------------------------------------------
    from src.kinematics.sim.robocasa_camera import RobocasaCamera
    from src.kinematics.sim.robocasa_robot_interface import RobocasaRobotInterface
    from src.kinematics.sim.robocasa_primitives import RobocasaPrimitives

    sim_camera = RobocasaCamera(env, camera_name=CAM_NAME, width=cam_w, height=cam_h)
    sim_camera.update(obs)

    robot_iface = RobocasaRobotInterface(env, camera_name=CAM_NAME)
    robot_iface.update(obs)

    # Matplotlib viewer — shows agentview_left + eye_in_hand side by side.
    # opencv-python on this system lacks GTK so we use matplotlib/TkAgg instead.
    _mpl_fig = _mpl_axes = _mpl_imgs = None
    if not args.no_render:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        _mpl_fig, _mpl_axes = plt.subplots(1, 2, figsize=(12, 5))
        _mpl_fig.suptitle("Obstructed Fridge Sim", fontsize=11)
        _mpl_axes[0].set_title(CAM_NAME.replace("robot0_", ""), fontsize=9)
        _mpl_axes[1].set_title(EIH_CAM.replace("robot0_", ""), fontsize=9)
        for ax in _mpl_axes:
            ax.axis("off")
        blank = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        _mpl_imgs = [ax.imshow(blank) for ax in _mpl_axes]
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)
        _ok(f"Matplotlib viewer: [{CAM_NAME}] + [{EIH_CAM}]")

    # Registry will be injected after orchestrator initialises its tracker
    sim_primitives: Optional[RobocasaPrimitives] = None  # wired up after tracker init

    _ok("Sim camera, robot interface, and primitives adapters created")

    # -----------------------------------------------------------------------
    _section("4 / TaskOrchestrator Init")
    # -----------------------------------------------------------------------
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator

    config = OrchestratorConfig(
        api_key="unused",
        llm_client=llm_client,

        use_sim_camera=True,     # skip RealSense init
        camera_width=cam_w,
        camera_height=cam_h,

        update_interval=2.0,
        min_observations=args.frames,
        fast_mode=False,

        state_dir=OUTPUT_DIR / "orchestrator_state",
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,

        enable_snapshots=True,
        snapshot_every_n_detections=1,
        perception_pool_dir=OUTPUT_DIR / "perception_pool",
        debug_frames_dir=debug_dir,

        use_layered_generation=(llm_client is not None),
        dkb_dir=OUTPUT_DIR / "dkb",

        solver_backend="auto",
        solver_timeout=60.0,
        solver_verbose=False,
        auto_solve_when_ready=False,
        max_refinement_attempts=3,
        auto_refine_on_failure=True,

        robot=robot_iface,

        on_state_change=lambda old, new: log.info(
            "Orchestrator state: %s → %s", old.value, new.value
        ),
    )

    t0          = _tick()
    orchestrator = TaskOrchestrator(config=config)
    await orchestrator.initialize()
    _tock("orchestrator_init", t0)
    _ok("Orchestrator initialised")

    # -----------------------------------------------------------------------
    _section("5 / GSAM2 Tracker Swap")
    # -----------------------------------------------------------------------
    from src.perception import GSAM2ContinuousObjectTracker
    from src.perception.clearance import GripperGeometry

    t0 = _tick()
    gsam2_tracker = GSAM2ContinuousObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        update_interval=config.update_interval,
        on_detection_complete=orchestrator._on_detection_callback,
        llm_client=llm_client if args.llm else None,
        llm_mode=args.llm_mode,
        robot=robot_iface,
        gripper=GripperGeometry(),
        logger=log.getChild("GSAM2Tracker"),
        debug_save_dir=debug_dir,
        use_molmo=True,
    )
    orchestrator.tracker = gsam2_tracker

    def _sim_frame_provider():
        color       = sim_camera.capture_frame()
        depth       = sim_camera.get_depth()
        intrinsics  = sim_camera.get_camera_intrinsics()
        robot_state = robot_iface.get_robot_state()
        return color, depth, intrinsics, robot_state

    gsam2_tracker.set_frame_provider(_sim_frame_provider)
    _tock("model_load", t0)
    _ok(f"GSAM2 tracker loaded on {device}")

    # Wire up primitives now that the registry exists, then point the
    # orchestrator config at sim_primitives to keep the robot interface
    # consistent with test_clutter_planning_realsense_robot.py.
    sim_primitives = RobocasaPrimitives(
        env=env,
        registry=orchestrator.tracker.registry,
        robot_iface=robot_iface,
        step_sleep=args.step_sleep,
        verbose=args.verbose,
    )
    orchestrator.config.robot = sim_primitives

    # -----------------------------------------------------------------------
    _section(f"6 / Perception Pass ({args.frames} sim frames)")
    # -----------------------------------------------------------------------
    orchestrator.current_task = task_str
    if hasattr(orchestrator.tracker, "set_task_context"):
        orchestrator.tracker.set_task_context(task_description=task_str)

    t0 = _tick()
    import cv2

    def _step_sim(action: np.ndarray) -> bool:
        """Take one sim step, update all adapters and viewer. Returns done flag."""
        nonlocal obs
        obs, _, done, _ = env.step(action)
        sim_camera.update(obs)
        robot_iface.update(obs)
        sim_primitives.update(obs)
        if not args.no_render and _mpl_imgs is not None:
            frame0 = sim_camera.capture_frame()
            eih_raw = obs.get(f"{EIH_CAM}_image", np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
            frame1 = (np.clip(eih_raw, 0, 1) * 255).astype(np.uint8) if eih_raw.dtype != np.uint8 else eih_raw
            _mpl_imgs[0].set_data(frame0)
            _mpl_imgs[1].set_data(frame1)
            _mpl_fig.canvas.draw_idle()
            plt.pause(0.001)
        return done

    async def _dwell_at_viewpoint(label: str, dwell_secs: float, move_action: Optional[np.ndarray] = None) -> bool:
        """
        Hold a viewpoint for dwell_secs, giving GSAM2 time to process frames.
        If move_action is set, step the sim with it first to reposition, then idle.
        Saves a keyframe at the start of the dwell.
        Returns True if target detection count reached.
        """
        _info(f"  viewpoint: {label}", f"dwelling {dwell_secs:.0f}s for perception")

        # Reposition — run move_action for 0.5s of sim time before dwelling
        if move_action is not None:
            for _ in range(25):
                if _step_sim(move_action):
                    return orchestrator.detection_count >= args.frames

        # Save keyframe at this viewpoint
        img = sim_camera.capture_frame()
        cv2.imwrite(
            str(frames_dir / f"viewpoint_{label.replace(' ', '_')}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

        # Dwell: yield to event loop every 0.1s so GSAM2 can run
        t_start = time.monotonic()
        while time.monotonic() - t_start < dwell_secs:
            if orchestrator.detection_count >= args.frames:
                return True
            # Keep sim alive with idle actions while waiting
            _step_sim(np.zeros(env.action_dim))
            await asyncio.sleep(0.1)   # yield to GSAM2 async task
            _info(f"    detections", orchestrator.detection_count)

        return orchestrator.detection_count >= args.frames

    await orchestrator.start_detection()

    # Phase 1: reset pose — agentview_left already faces the fridge interior
    reached = await _dwell_at_viewpoint("reset pose", dwell_secs=args.dwell)

    # Phase 2: exploratory viewpoints if we still need more detections.
    # Move the base to change the viewing angle, dwell at each stop.
    if not reached:
        _info("Perception phase 2", "exploratory base sweep")
        # [0:3] EEF delta, [3:6] ori delta, [6] gripper, [7:9] base xy, [10:12] base rot
        _base = lambda dy: np.array([0,0,0, 0,0,0, 0, 0,dy,0, 0,0], dtype=float)
        viewpoints = [
            ("sweep left",   _base( 0.3)),
            ("sweep right",  _base(-0.6)),
            ("return centre",_base( 0.3)),
        ]
        for label, move_action in viewpoints:
            if orchestrator.detection_count >= args.frames:
                break
            action = move_action[:env.action_dim]
            reached = await _dwell_at_viewpoint(label, dwell_secs=args.dwell, move_action=action)

    await orchestrator.stop_detection()

    _tock("perception_pass", t0)

    registry     = orchestrator.tracker.registry
    detected_ids = [o.object_id for o in registry.get_all_objects()]
    _info("Detected objects", detected_ids)

    if not detected_ids:
        _warn("No objects detected — aborting before planning")
        await orchestrator.save_state()
        env.close()
        return 1

    # Save reference frame
    try:
        ref_frame = sim_camera.capture_frame()
        cv2.imwrite(
            str(OUTPUT_DIR / "reference_frame.jpg"),
            cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR),
        )
        _ok(f"Reference frame saved → {OUTPUT_DIR / 'reference_frame.jpg'}")
    except Exception as exc:
        log.exception("Could not save reference frame: %s", exc)

    _ok(f"Perception pass complete — {len(detected_ids)} objects")

    # -----------------------------------------------------------------------
    _section("7 / Task Request + PDDL Generation (L1–L5)")
    # -----------------------------------------------------------------------
    solver_result = None
    if args.no_plan or llm_client is None:
        _info("Planning", "skipped (--no-plan or no LLM client)")
    else:
        t0 = _tick()
        try:
            ctx_frame = sim_camera.capture_frame()
            await orchestrator.process_task_request(task_str, environment_image=ctx_frame)
            _ok("Task analysis complete")
        except Exception as exc:
            log.exception("process_task_request failed: %s", exc)
            _tock("pddl_generation", t0)
            await orchestrator.save_state()
            env.close()
            return 1
        _tock("pddl_generation", t0)

        pddl_dir = OUTPUT_DIR / "pddl"
        for pattern, label in [("*_domain.pddl", "Domain"), ("*_problem.pddl", "Problem")]:
            matches = sorted(pddl_dir.glob(pattern)) if pddl_dir.exists() else []
            if matches:
                _section(f"Generated PDDL {label}")
                print(matches[-1].read_text())

        # -----------------------------------------------------------------------
        _section("8 / Planning")
        # -----------------------------------------------------------------------
        t0 = _tick()
        try:
            solver_result = await orchestrator.solve_and_plan_with_refinement(
                output_dir=pddl_dir,
                wait_for_objects=False,
            )
            _tock("planning", t0)

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
            _tock("planning", t0)

        # -----------------------------------------------------------------------
        _section("9 / Execution (via RobocasaPrimitives + ConditionalTaskExecutor)")
        # -----------------------------------------------------------------------
        if args.execute and solver_result and solver_result.success and solver_result.plan:
            from src.planning.conditional_task_executor import ConditionalTaskExecutor
            from src.planning.clutter_module import PostDisplacementHook
            from src.primitives.skill_decomposer import SkillDecomposer
            from src.primitives.primitive_executor import PrimitiveExecutor

            pool_dir = Path(config.perception_pool_dir or OUTPUT_DIR / "perception_pool")
            prim_executor = PrimitiveExecutor(
                primitives=sim_primitives,
                perception_pool_dir=pool_dir,
                logger=log.getChild("PrimitiveExecutor"),
                orchestrator=orchestrator,
            )

            cam_intrinsics = sim_camera.get_camera_intrinsics()
            cam_state      = robot_iface.get_robot_state()
            cam_pos = np.array(cam_state["camera"]["position"]) if cam_state and "camera" in cam_state else None
            cam_rot = None
            if cam_state and "camera" in cam_state:
                from scipy.spatial.transform import Rotation as _R
                cam_rot = _R.from_quat(cam_state["camera"]["quaternion_xyzw"])

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
                _warn("LLM client not available — skill decomposition disabled")

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
                        if skill_plan.primitives:
                            fresh_snapshot_id = orchestrator.capture_fresh_snapshot_sync(
                                reason=f"pre-execution:{action_name}"
                            )
                            world_state_for_exec = {"state_dir": str(orchestrator.config.state_dir)}
                            if fresh_snapshot_id:
                                world_state_for_exec["last_snapshot_id"] = fresh_snapshot_id
                            result = prim_executor.execute_plan(
                                skill_plan,
                                world_state=world_state_for_exec,
                            )
                            # Step sim after each primitive action to let physics settle
                            for _ in range(10):
                                obs, _, done, _ = env.step(np.zeros(env.action_dim))
                                sim_camera.update(obs)
                                robot_iface.update(obs)
                                sim_primitives.update(obs)
                            return result.executed
                        _warn("    Decomposer returned no primitives — skipping action")
                    except Exception as exc:
                        _warn(f"    Decomposition failed for '{action_name}': {exc}")
                        log.exception("Decomposition error")

                _warn(f"    No executor for '{action_name}' — skipping")
                return True

            async def _replan() -> List[str]:
                _info("Replanning: fresh perception pass...")
                await orchestrator.start_detection()
                replan_deadline = time.monotonic() + 20.0
                while time.monotonic() < replan_deadline:
                    await asyncio.sleep(0.5)
                    _step_sim(np.zeros(env.action_dim))
                    if orchestrator.tracker.registry.get_all_objects():
                        break
                await orchestrator.stop_detection()
                ctx = sim_camera.capture_frame()
                await orchestrator.process_task_request(task_str, environment_image=ctx)
                res = await orchestrator.solve_and_plan_with_refinement(
                    output_dir=pddl_dir, wait_for_objects=False
                )
                return res.plan if res and res.success else []

            t0 = _tick()
            cond_executor = ConditionalTaskExecutor(
                registry=registry,
                execute_action=execute_fn,
                replan_fn=_replan,
                max_replan=getattr(orchestrator.config, "max_refinement_attempts", 2),
                post_displacement_hook=post_hook,
                logger=log.getChild("ConditionalExecutor"),
            )

            cond_result = await cond_executor.execute(solver_result.plan)
            _tock("execution", t0)

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

            success = env._check_success()
            if success:
                _ok("Task SUCCESS — target item retrieved!")
            else:
                _warn("Task not yet complete after plan execution")

        elif args.execute:
            _warn("Execution requested but no plan available")

    # -----------------------------------------------------------------------
    _section("10 / Save State")
    # -----------------------------------------------------------------------
    t0 = _tick()
    try:
        saved_path = await orchestrator.save_state()
        _ok(f"State saved → {saved_path}")
    except Exception as exc:
        log.exception("Save state failed: %s", exc)
    _tock("save_state", t0)

    # Save a final annotated frame
    try:
        import imageio
        final_obs = obs
        front_key = f"robot0_frontview_image"
        if front_key in final_obs:
            img = final_obs[front_key]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(str(OUTPUT_DIR / "final_frame_frontview.png"), img)
            _ok(f"Final front view → {OUTPUT_DIR / 'final_frame_frontview.png'}")
    except Exception:
        pass

    if not args.no_render and _mpl_fig is not None:
        plt.close(_mpl_fig)
    env.close()

    # -----------------------------------------------------------------------
    _banner("Run Complete")
    # -----------------------------------------------------------------------
    t_total_elapsed = time.monotonic() - _t_total
    _info("Run ID", _RUN_ID)
    _info("Task",   task_str)
    _info("Output", OUTPUT_DIR)

    _TIMING_LABELS = [
        ("llm_client_init",   "LLM client init"),
        ("env_create",        "Robocasa env create"),
        ("orchestrator_init", "Orchestrator init"),
        ("model_load",        "GSAM2 model load"),
        ("perception_pass",   "Perception pass"),
        ("task_request",      "Task request + LLM analysis"),
        ("pddl_generation",   "PDDL generation (L1–L5)"),
        ("planning",          "PDDL solving"),
        ("execution",         "Primitive execution"),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Obstructed fridge sim — full perception + planning test"
    )
    p.add_argument("--task",    default="get the cream cheese from behind the other groceries in the fridge",
                   help="Override the task description (default: use robocasa ep_meta lang)")
    p.add_argument("--frames",  type=int, default=DEFAULT_FRAMES,
                   help=f"Detection passes before planning (default: {DEFAULT_FRAMES})")
    p.add_argument("--seed",    type=int, default=0,
                   help="RNG seed for the environment")
    p.add_argument("--robot",   default="PandaOmron",
                   help="Robot model name (default: PandaOmron)")
    p.add_argument("--cam-width",  type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--no-plan",    action="store_true",
                   help="Skip PDDL planning step")
    p.add_argument("--execute",    action="store_true",
                   help="Execute the plan via RobocasaPrimitives after solving")
    p.add_argument("--llm",        action="store_true",
                   help="Enable LLM contact graph classification in GSAM2")
    p.add_argument("--llm-mode",   choices=["json", "nl"], default="nl",
                   help="LLM contact prompt mode (default: nl)")
    p.add_argument("--dwell",       type=float, default=8.0,
                   help="Seconds to dwell at each viewpoint for GSAM2 to process (default: 8)")
    p.add_argument("--step-sleep", type=float, default=0.0,
                   help="Seconds to sleep between sim steps during execution")
    p.add_argument("--verbose",    action="store_true",
                   help="Enable verbose primitive logging")
    p.add_argument("--no-render",  action="store_true",
                   help="Disable the OpenCV sim viewer window")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
