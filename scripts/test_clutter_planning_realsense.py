"""
Clutter Planning Integration Test — Live RealSense RGB-D

Full end-to-end pipeline on real hardware via TaskOrchestrator:
  1. RealSense D435 captures aligned RGB-D frames
  2. GSAM2 (GroundingDINO + SAM2) detects and segments all objects
     — tracker internally computes clearance profiles, contact graph,
       surface free-space maps, and occlusion map
  3. ClutterGrounder evaluates all clutter predicates against the registry,
     including improved-clearance (§3.3) and L4-V4 feasibility gate
  4. LayeredDomainGenerator (L1–L5) produces a PDDL domain + problem
  5. HybridPlanner selects deterministic vs probabilistic mode (§5.1),
     expands to all-outcomes determinization (§5.2), and escalates to
     probabilistic/hindsight if deterministic replanning exhausted (§5.3–5.5)
  6. ConditionalTaskExecutor runs the plan with runtime predicate branching,
     PostDisplacementHook re-grounds clearance after every displacement,
     SkillDecomposer decomposes each action to primitives, and
     XArmPybulletPrimitives motion-plans + executes each primitive in the
     PyBullet sim (with GUI trajectory replay when --pybullet-gui is set)

Requires:
    SAM2_CKPT env var pointing to the SAM2 checkpoint
    OPENAI_API_KEY env var (for planning and optional LLM contact classification)
    OPENAI_MODEL env var (optional, default: gpt-4o-mini)
    RealSense D435 connected via USB3

Run:
    uv run scripts/test_clutter_planning_realsense.py
    uv run scripts/test_clutter_planning_realsense.py --no-plan
    uv run scripts/test_clutter_planning_realsense.py --no-gui
    uv run scripts/test_clutter_planning_realsense.py --task "Pick up the red cup"
    uv run scripts/test_clutter_planning_realsense.py --frames 5
    uv run scripts/test_clutter_planning_realsense.py --llm
    uv run scripts/test_clutter_planning_realsense.py --force-mode probabilistic
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
import matplotlib.patches as mpatches
plt.ion()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_CONFIG_PATH = PROJECT_ROOT / "config"
if str(_CONFIG_PATH) not in sys.path:
    sys.path.insert(0, str(_CONFIG_PATH))

from src.perception.object_registry import DetectedObjectRegistry
from src.planning.clutter_module import ClutterGrounder, _SURFACE_KEYWORDS
from src.planning.domain_knowledge_base import DomainKnowledgeBase

# ---------------------------------------------------------------------------
# Colours / logging
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_GREY   = "\033[90m"
_MAGENTA = "\033[35m"


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
    return logging.getLogger("clutter_realsense")


from contextlib import contextmanager

_PLANNING_LOGGERS = (
    "TaskOrchestrator",
    "LayeredDomainGenerator",
    "PDDLSolver",
    "PDDLRepresentation",
    "PDDLDomainMaintainer",
    "HybridPlanner",
    "ClutterGrounder",
    "DomainKnowledgeBase",
)

@contextmanager
def _quiet_planning():
    """Suppress verbose INFO logs from planning modules while planning runs."""
    saved = {name: logging.getLogger(name).level for name in _PLANNING_LOGGERS}
    for name in _PLANNING_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


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
    log.info("PASS: %s", msg)


def _info(label: str, value: Any = "") -> None:
    if value == "":
        print(f"    {_YELLOW}•{_RESET} {label}")
    else:
        print(f"    {_YELLOW}{label}:{_RESET} {value}")
    log.info("%s %s", label, value)


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")
    log.warning("WARN: %s", msg)


def _risk(msg: str) -> None:
    print(f"  {_MAGENTA}⚡{_RESET}  {msg}")
    log.info("RISK: %s", msg)


# ---------------------------------------------------------------------------
# Run output directory
# ---------------------------------------------------------------------------

_RUN_ID          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/clutter_realsense"))
OUTPUT_DIR       = _BASE_OUTPUT_DIR / "runs" / _RUN_ID

DEFAULT_TASK   = "Pick up the rubber duck from the cluttered table"


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
# GUI helpers
# ---------------------------------------------------------------------------

def _wait(gui: bool, prompt: str = "Press Enter to continue...") -> None:
    if gui:
        try:
            input(f"\n  {prompt}\n")
        except EOFError:
            pass


def _show_and_wait(fig: Any, gui: bool, prompt: str = "Press Enter to continue...") -> None:
    if not gui:
        plt.close(fig)
        return
    plt.show(block=False)
    plt.pause(0.1)
    _wait(gui, prompt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Clutter predicate grounding
# ---------------------------------------------------------------------------

def run_clutter_grounding(
    registry: DetectedObjectRegistry,
    color: np.ndarray,
    masks: Dict[str, np.ndarray],
    gui: bool,
) -> None:
    _section("Clutter Predicate Grounding")

    t0 = _tick()
    grounder  = ClutterGrounder(registry)
    all_objs  = registry.get_all_objects()
    obj_ids   = [o.object_id for o in all_objs]
    block_ids = [
        o.object_id for o in all_objs
        if not any(kw in o.object_type.lower() for kw in _SURFACE_KEYWORDS)
    ]

    stable_facts = grounder.ground_stable_on()
    _info(f"stable-on TRUE facts ({len(stable_facts)})")
    for pred_name, args in stable_facts:
        _info(f"  ({pred_name} {' '.join(args)})")

    if block_ids:
        _info("\nPer-object predicate evaluation")
        header = (
            f"  {'Object':<28} "
            f"{'access-clear':>12} "
            f"{'displaceable':>12} "
            f"{'removal-safe':>12} "
            f"{'visible':>8} "
            f"{'improved-clr':>13}"
        )
        print(f"    {_YELLOW}{header}{_RESET}")
        print(f"    {'  ' + '-' * 92}")
        for oid in block_ids:
            access   = grounder.evaluate_access_clear(oid)
            displ    = grounder.evaluate_displaceable(oid)
            safe     = grounder.evaluate_removal_safe(oid)
            vis      = grounder.evaluate_visible(oid)
            improved = grounder.evaluate_improved_clearance(oid)
            row = (
                f"  {oid:<28} "
                f"{str(access):>12} "
                f"{str(displ):>12} "
                f"{str(safe):>12} "
                f"{str(vis):>8} "
                f"{str(improved):>13}"
            )
            print(f"    {row}")
            log.info(row)

    _info("\nObstruction pairs (obstructs ?blocker ?target)")
    found = False
    for blocker in block_ids:
        for target in block_ids:
            if blocker != target and grounder.evaluate_obstructs(blocker, target):
                _info(f"  (obstructs {blocker} {target})")
                found = True
    if not found:
        _info("  (none from clearance profiles)")

    _info("\nSupport (supports ?lower ?upper)")
    found = False
    for lower in obj_ids:
        for upper in obj_ids:
            if lower != upper and grounder.evaluate_supports(lower, upper):
                _info(f"  (supports {lower} {upper})")
                found = True
    if not found:
        _info("  (none detected)")

    _tock("clutter_grounding", t0)
    _ok("Clutter grounding complete")

    if not gui or not block_ids:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Clutter Predicate Grounding", fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    axes[0].imshow(color)
    legend_patches = []
    for idx, (oid, mask) in enumerate(masks.items()):
        if not mask.any():
            continue
        c    = cmap(idx % 10)
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask, :3] = c[:3]
        rgba[mask, 3]  = 0.45
        axes[0].imshow(rgba)
        legend_patches.append(mpatches.Patch(color=c, label=oid))
    if legend_patches:
        axes[0].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[0].set_title("Detected scene", fontsize=9)
    axes[0].axis("off")

    unary_preds = ["access-clear", "displaceable", "removal-safe", "visible", "improved-clr"]
    eval_fns    = [
        grounder.evaluate_access_clear,
        grounder.evaluate_displaceable,
        grounder.evaluate_removal_safe,
        grounder.evaluate_visible,
        grounder.evaluate_improved_clearance,
    ]
    grid = np.array([[fn(oid) for fn in eval_fns] for oid in block_ids], dtype=float)
    ax   = axes[1]
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(unary_preds)))
    ax.set_xticklabels(unary_preds, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(block_ids)))
    ax.set_yticklabels(block_ids, fontsize=8)
    ax.set_title("Sensed predicates (T=green / F=red)", fontsize=9)
    for i in range(len(block_ids)):
        for j in range(len(unary_preds)):
            ax.text(j, i, "T" if grid[i, j] else "F",
                    ha="center", va="center", fontsize=9, color="black", fontweight="bold")

    plt.tight_layout()
    _show_and_wait(fig, gui, "Grounding shown — press Enter for planning.")


# ---------------------------------------------------------------------------
# Hybrid planner risk report
# ---------------------------------------------------------------------------

def show_risk_report(registry: DetectedObjectRegistry, dkb_dir: Path) -> Optional[Any]:
    """Run risk assessment and print mode selection rationale. Returns RiskAssessment."""
    _section("Hybrid Planner Risk Assessment (§5.1)")
    try:
        from src.planning.hybrid_planner import ActionOutcomeStore, PlanningModeSelector
        store    = ActionOutcomeStore(dkb_dir=dkb_dir)
        selector = PlanningModeSelector(outcome_store=store)
        risk     = selector.assess_risk(registry)
        mode     = selector.select_mode(registry)

        _info("Dead-end risk",      f"{risk.dead_end_risk:.2f}  (threshold 0.10)")
        _info("Outcome criticality",f"{risk.outcome_criticality:.2f}  (threshold 5.0)")
        _info("State-space estimate",f"{risk.state_space_size}  (det floor 10 000)")
        _info("Selected mode",       f"{_BOLD}{mode.upper()}{_RESET}")

        # Show per-action outcome probabilities from DKB history
        for atype in ("displace", "push-aside"):
            probs = store.get_outcome_probs(atype)
            prob_str = "  ".join(f"{k}={v:.2f}" for k, v in probs.items())
            _info(f"  {atype} outcomes", prob_str)

        return risk
    except Exception as exc:
        _warn(f"Risk assessment failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Visibility summary
# ---------------------------------------------------------------------------

def show_visibility_summary(registry: DetectedObjectRegistry, gui: bool) -> None:
    occ_map = registry.occlusion_map
    if occ_map is None or not occ_map.per_object_visibility or not gui:
        return

    obj_ids = list(occ_map.per_object_visibility.keys())
    fracs   = [occ_map.per_object_visibility[oid].visible_fraction for oid in obj_ids]

    fig, ax = plt.subplots(figsize=(max(6, len(obj_ids) * 1.4), 4))
    fig.suptitle("Occlusion Map — Visible Fractions", fontsize=11, fontweight="bold")
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(obj_ids))]
    ax.bar(range(len(obj_ids)), fracs, color=colors)
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2, label="visible threshold 0.6")
    ax.set_xticks(range(len(obj_ids)))
    ax.set_xticklabels(obj_ids, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Visible fraction")
    ax.legend(fontsize=8)
    for i, (val, oid) in enumerate(zip(fracs, obj_ids)):
        occluders = occ_map.per_object_visibility[oid].occluding_objects
        label     = f"{val:.0%}"
        if occluders:
            label += f"\n({', '.join(occluders[:2])})"
        ax.text(i, val + 0.02, label, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Visibility summary shown — press Enter for planning.")


# ---------------------------------------------------------------------------
# L5 initial-state heatmap
# ---------------------------------------------------------------------------

def show_l5_heatmap(artifact: Any, observed_objects: List[Dict], gui: bool) -> None:
    if not gui or artifact is None or artifact.l5 is None:
        return

    block_ids     = [o["object_id"] for o in observed_objects
                     if not any(kw in o["object_type"].lower() for kw in _SURFACE_KEYWORDS)]
    preds_to_show = [
        "stable-on", "access-clear", "displaceable",
        "removal-safe", "visible", "improved-clearance",
        "obstructs", "supports",
    ]
    if not block_ids:
        return

    true_set: set = set()
    for lit in artifact.l5.true_literals:
        if isinstance(lit, str):
            true_set.add(lit.strip("() ").lower())
        elif isinstance(lit, (list, tuple)) and len(lit) == 2:
            pred_name, args = lit
            true_set.add(f"{pred_name} {' '.join(args)}".lower())

    grid = np.zeros((len(block_ids), len(preds_to_show)), dtype=float)
    for i, oid in enumerate(block_ids):
        for j, pred in enumerate(preds_to_show):
            for entry in true_set:
                if pred.lower() in entry and oid.lower() in entry:
                    grid[i, j] = 1.0
                    break

    fig, ax = plt.subplots(figsize=(12, max(3, len(block_ids) * 0.6 + 2)))
    fig.suptitle("L5 Initial State — TRUE literal heatmap", fontsize=11, fontweight="bold")
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(preds_to_show)))
    ax.set_xticklabels(preds_to_show, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(block_ids)))
    ax.set_yticklabels(block_ids, fontsize=8)
    for i in range(len(block_ids)):
        for j in range(len(preds_to_show)):
            ax.text(j, i, "T" if grid[i, j] else "F",
                    ha="center", va="center", fontsize=8, color="black", fontweight="bold")
    plt.tight_layout()
    _show_and_wait(fig, gui, "L5 initial state shown — done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    global _t_total
    _t_total = time.monotonic()

    _init_output_dir()

    gui = not args.no_gui
    if not gui:
        plt.ioff()

    _banner("Clutter Planning Integration Test — Live RealSense")
    _info("Task",    args.task)
    _info("Run ID",  _RUN_ID)
    _info("Output",  OUTPUT_DIR)
    _info("LLM",     f"{'enabled (' + args.llm_mode + ')' if args.llm else 'disabled (geometric only)'}")
    if args.force_mode:
        _info("Planning mode", f"FORCED → {args.force_mode.upper()}")

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        return 1

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info("Device", device)

    # --- LLM client ---
    _section("Setting up LLM client (OpenAI)")
    t0 = _tick()
    llm_client = None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from src.llm_interface import OpenAIClient
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            _warn("OPENAI_API_KEY not set — planning and LLM classification disabled")
        else:
            llm_client = OpenAIClient(model=openai_model, api_key=openai_key)
            _ok(f"LLM client ready: {openai_model}")
    except Exception as exc:
        _warn(f"LLM client setup failed ({exc})")
    _tock("llm_client_init", t0)

    # --- Robot interface (xArm7 FK + optional PyBullet GUI) ---
    robot_interface = None
    depth_collider  = None
    _section("Setting up robot interface (xArm7 FK)")
    t0 = _tick()
    try:
        from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
        from src.kinematics.depth_environment_collider import DepthEnvironmentCollider
        robot_interface = XArmPybulletInterface(use_gui=args.pybullet_gui)
        robot_interface.set_current_joint_state(args.joint_state)
        _info("Joint state", [f"{j:.3f}" for j in args.joint_state])
        pos, rot = robot_interface.get_camera_transform()
        if pos is not None:
            _ok(f"Camera transform: pos={pos.round(3).tolist()}")
        else:
            _warn("FK returned no camera transform — falling back to camera frame")
            robot_interface = None
    except Exception as exc:
        _warn(f"Robot interface unavailable ({exc}) — using camera frame")
    _tock("robot_interface_init", t0)

    # --- Build orchestrator config ---
    _section("Configuring TaskOrchestrator")
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator
    from src.perception.clearance import GripperGeometry

    dkb_dir = OUTPUT_DIR / "dkb"
    contact_llm_client = llm_client if args.llm else None

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

        robot=robot_interface,

        on_state_change=lambda old, new: log.info(
            "Orchestrator state: %s → %s", old.value, new.value
        ),
    )
    _ok("Config built")

    # --- Initialise orchestrator ---
    _section("Initialising orchestrator")
    t0           = _tick()
    orchestrator = TaskOrchestrator(config=config)
    await orchestrator.initialize()
    _tock("orchestrator_init", t0)
    _ok("Orchestrator initialised")

    if orchestrator._hybrid_planner is not None:
        _ok("Hybrid planner active (det/prob mode selection)")
    else:
        _warn("Hybrid planner not available — will use direct solver only")

    # --- Swap tracker to GSAM2 ---
    _section("Loading GSAM2 tracker")
    t0 = _tick()
    from src.perception import GSAM2ContinuousObjectTracker

    gsam2_tracker = GSAM2ContinuousObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        update_interval=config.update_interval,
        on_detection_complete=orchestrator._on_detection_callback,
        llm_client=contact_llm_client,
        llm_mode=args.llm_mode,
        robot=robot_interface,
        gripper=GripperGeometry(),
        logger=log.getChild("GSAM2Tracker"),
        debug_save_dir=OUTPUT_DIR / "debug_frames",
    )
    _tock("model_load", t0)

    orchestrator.tracker = gsam2_tracker
    gsam2_tracker.on_detection_complete = orchestrator._on_detection_callback
    gsam2_tracker.set_frame_provider(orchestrator._get_camera_frames)
    _ok(f"GSAM2 tracker active on {device}")

    # --- Perception pass (manual) so we can inject depth collision meshes
    #     before handing off to execute_task() for planning + execution. ---
    _section("Perception pass (first result with objects)")
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

    # Build depth → PyBullet collision meshes while tracker state is fresh.
    _masks = dict(getattr(orchestrator.tracker, "_last_masks", {}))
    if robot_interface is not None and orchestrator._camera is not None:
        _section("Depth → PyBullet collision meshes (per-object + background)")
        t0_col = _tick()
        try:
            from src.kinematics.depth_environment_collider import BACKGROUND_ID
            depth_collider = DepthEnvironmentCollider(robot_interface)
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
                robot_interface.attach_collider(depth_collider)
                _ok("Collider attached to planner — trajectories will be collision-checked")
            if any(build_results.values()) and args.pybullet_gui:
                _info("PyBullet GUI open — inspect collision meshes, then press Enter")
                _wait(gui, "Press Enter to continue after inspecting the PyBullet GUI...")
        except Exception:
            log.exception("Depth collider injection failed")
        _tock("depth_collider", t0_col)

    # Detection loop is stopped but the camera pipeline stays alive so Molmo
    # can capture fresh frames during skill execution.
    _tock("perception_pass", t0)

    registry     = orchestrator.tracker.registry
    detected_ids = [o.object_id for o in registry.get_all_objects()]
    _info("Detected objects", detected_ids)

    if not detected_ids:
        _warn("No objects detected — aborting")
        await orchestrator.save_state()
        return 1

    # Save initial camera view.
    try:
        last_color = orchestrator.tracker.last_color_frame
        if last_color is not None:
            from PIL import Image as _PIL
            _PIL.fromarray(last_color).save(OUTPUT_DIR / "color_frame.png")
            _ok(f"Camera frame saved → {OUTPUT_DIR / 'color_frame.png'}")
    except Exception:
        log.exception("Failed to save camera frame")

    masks = dict(getattr(orchestrator.tracker, "_last_masks", {}))

    if registry.contact_graph:
        g = registry.contact_graph
        _info("Contact graph", f"{len(g.edges)} edges")
        for edge in g.edges:
            _info(f"  {edge.obj_a} → {edge.obj_b}",
                  f"{edge.contact_type}  consequence={edge.removal_consequence}")

    if registry.occlusion_map:
        for oid, vis in registry.occlusion_map.per_object_visibility.items():
            _info(f"  occlusion {oid}",
                  f"{vis.visible_fraction:.0%}  occluders={vis.occluding_objects}")

    _ok("Perception complete")

    # --- Pre-execution visualisations ---
    color_for_gui = getattr(orchestrator.tracker, "last_color_frame", None)
    if color_for_gui is None:
        color_for_gui = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        run_clutter_grounding(registry, color_for_gui, masks, gui)
    except Exception:
        log.exception("Clutter grounding failed")

    try:
        show_visibility_summary(registry, gui)
    except Exception:
        log.exception("Visibility summary failed")

    risk_assessment = None
    try:
        risk_assessment = show_risk_report(registry, dkb_dir)
    except Exception:
        log.exception("Risk assessment failed")

    # --- Full pipeline: planning + skill decomposition + sim execution ---
    exec_result = None
    if not args.no_plan:
        # Build sim primitives (requires robot_interface).
        sim_primitives = None
        if robot_interface is not None:
            from src.kinematics.sim.xarm_pybullet_primitives import XArmPybulletPrimitives
            sim_primitives = XArmPybulletPrimitives(
                robot=robot_interface,
                registry=registry,
                env=robot_interface,
                logger=log.getChild("XArmPybulletPrimitives"),
            )
            _ok("Sim primitives ready (XArmPybulletPrimitives)")
        else:
            _warn("No robot_interface — sim execution unavailable (dry-run mode)")

        # Run the full pipeline via TaskOrchestrator.execute_task().
        # detection_timeout=0 skips re-detection since we already have objects.
        _section("Task Analysis → PDDL Solving → Skill Decomp → Sim Execution")
        t0 = _tick()
        try:
            exec_result = await orchestrator.execute_task(
                task_description=args.task,
                output_dir=OUTPUT_DIR / "pddl",
                detection_timeout=0.0,
                primitives=sim_primitives,
            )
        except Exception:
            log.exception("execute_task raised an exception")
        _tock("execute_task", t0)

        if exec_result is not None:
            if exec_result.success:
                _ok(f"Task complete — {len(exec_result.steps)} step(s), "
                    f"{exec_result.replan_count} replan(s)")
            else:
                _warn(f"Task failed: {exec_result.error}")

            if exec_result.plan:
                _section("PDDL Plan")
                for i, step in enumerate(exec_result.plan, 1):
                    _info(f"  {i}", step)

            _section("Step-by-step execution trace")
            for step in exec_result.steps:
                pred_tag     = ""
                if getattr(step, "predicate_value", None) is not None:
                    pred_tag = f"  [{'TRUE' if step.predicate_value else 'FALSE'}]"
                recovery_tag = (f"  → {step.recovery_action}"
                                if getattr(step, "recovery_triggered", False) else "")
                status = "✓" if step.success else "✗"
                _info(f"  {status} {step.action}{pred_tag}{recovery_tag}",
                      step.notes if step.notes else "")

        # Show generated PDDL files.
        pddl_dir = OUTPUT_DIR / "pddl"
        det_domain_files = sorted(pddl_dir.glob("*_det.pddl"))  if pddl_dir.exists() else []
        domain_files     = sorted(pddl_dir.glob("*_domain.pddl")) if pddl_dir.exists() else []
        problem_files    = sorted(pddl_dir.glob("*_problem.pddl")) if pddl_dir.exists() else []
        if det_domain_files:
            _section("All-Outcomes Determinized Domain (§5.2)")
            print(det_domain_files[-1].read_text())
        elif domain_files:
            _section("Generated PDDL Domain")
            print(domain_files[-1].read_text())
        if problem_files:
            _section("Generated PDDL Problem")
            print(problem_files[-1].read_text())

        # L5 initial-state heatmap.
        artifact = getattr(orchestrator, "_last_layered_artifact", None)
        observed_objects = [
            {
                "object_id":   o.object_id,
                "object_type": o.object_type,
                "affordances": list(o.affordances),
                "position_3d": o.position_3d.tolist() if o.position_3d is not None else [0.0, 0.0, 0.0],
            }
            for o in registry.get_all_objects()
        ]
        try:
            show_l5_heatmap(artifact, observed_objects, gui)
        except Exception:
            log.exception("L5 heatmap failed")
    else:
        _info("Planning", "skipped (--no-plan)")

    # --- Save orchestrator state ---
    _section("Saving run state")
    t0 = _tick()
    saved_path = await orchestrator.save_state()
    _tock("save_state", t0)
    _ok(f"State saved → {saved_path}")

    # --- Timing report ---
    t_total_elapsed = time.monotonic() - _t_total

    _banner("Run Complete")
    _info("Run ID",  _RUN_ID)
    _info("Task",    args.task)
    _info("Output",  OUTPUT_DIR)

    # Merge timings from execute_task result into the global dict.
    if exec_result is not None:
        for k, v in exec_result.timings.items():
            _timings[f"execute_task.{k}"] = v

    _TIMING_LABELS = [
        ("llm_client_init",              "LLM client init"),
        ("robot_interface_init",         "Robot interface init"),
        ("orchestrator_init",            "Orchestrator init"),
        ("model_load",                   "GSAM2 model load"),
        ("perception_pass",              "Perception pass"),
        ("depth_collider",               "Depth → collision mesh"),
        ("clutter_grounding",            "Clutter predicate grounding"),
        ("execute_task",                 "execute_task (total)"),
        ("execute_task.domain_generation","  └ task analysis + domain gen (L1–L5)"),
        ("execute_task.solving",         "  └ PDDL solving (hybrid mode-aware)"),
        ("execute_task.execution",       "  └ skill decomp + sim execution"),
        ("save_state",                   "Save state"),
    ]

    print()
    print(_BOLD + _BLUE + "── Timing Report " + "─" * 55 + _RESET)
    col_w = 44
    for key, label in _TIMING_LABELS:
        if key in _timings:
            t   = _timings[key]
            pct = 100.0 * t / t_total_elapsed if t_total_elapsed > 0 else 0.0
            print(f"  {_YELLOW}{label:<{col_w}}{_RESET} {t:6.2f}s  ({pct:4.1f}%)")
    print(f"  {_BOLD}{'Total':<{col_w + 2}}{_RESET} {t_total_elapsed:6.2f}s")

    # Per-component perception breakdown.
    component_labels = [
        ("gsam2",         "GSAM2 inference (GroundingDINO + SAM2)"),
        ("clearance",     "Grasp clearance profiles"),
        ("contact_graph", "Contact graph"),
        ("surface_maps",  "Surface free-space maps"),
        ("occlusion",     "Occlusion map"),
    ]
    if hasattr(gsam2_tracker, "get_component_timings"):
        comp = gsam2_tracker.get_component_timings()
        if any(comp.get(k, {}).get("calls", 0) > 0 for k, _ in component_labels):
            print()
            print(_BOLD + _BLUE + "── Perception Component Breakdown " + "─" * 38 + _RESET)
            for key, label in component_labels:
                entry = comp.get(key, {})
                total = entry.get("total_s", 0.0)
                calls = entry.get("calls", 0)
                avg   = entry.get("avg_s", 0.0)
                pct   = 100.0 * total / t_total_elapsed if t_total_elapsed > 0 else 0.0
                if calls > 0:
                    print(
                        f"  {_YELLOW}{label:<{col_w}}{_RESET} "
                        f"{total:6.2f}s total  "
                        f"{avg * 1000:6.1f} ms/call  "
                        f"({calls} calls,  {pct:4.1f}%)"
                    )

    # Hybrid planner summary.
    if risk_assessment is not None:
        print()
        print(_BOLD + _BLUE + "── Hybrid Planner Summary " + "─" * 46 + _RESET)
        print(f"  {_YELLOW}Dead-end risk:{_RESET}       {risk_assessment.dead_end_risk:.2f}")
        print(f"  {_YELLOW}Outcome criticality:{_RESET} {risk_assessment.outcome_criticality:.2f}")
        print(f"  {_YELLOW}State-space estimate:{_RESET} {risk_assessment.state_space_size}")
        mode_str = "PROBABILISTIC" if risk_assessment.requires_probabilistic else "DETERMINISTIC"
        print(f"  {_YELLOW}Mode selected:{_RESET}       {_BOLD}{mode_str}{_RESET}")

    # Domain error log summary.
    try:
        from src.planning.domain_knowledge_base import DomainKnowledgeBase
        _dkb = DomainKnowledgeBase(dkb_dir)
        summary = _dkb.error_summary()
        if summary["total"] > 0:
            print()
            print(_BOLD + _BLUE + "── Domain Error Log " + "─" * 52 + _RESET)
            print(f"  Total errors logged: {_BOLD}{summary['total']}{_RESET}  "
                  f"(file: {dkb_dir / 'domain_errors.jsonl'})")
            for etype, count in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
                print(f"  {_YELLOW}{etype:<30}{_RESET} {count:>4}x")
            if summary["recent"]:
                print(f"  {_BOLD}Most recent:{_RESET}")
                for r in summary["recent"]:
                    ts = r["timestamp"][:19].replace("T", " ")
                    print(f"    [{ts}] {r['type']}: {r['message']}")
    except Exception:
        pass

    return 0 if (exec_result is None or exec_result.success) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clutter planning integration test — live RealSense"
    )
    parser.add_argument("--no-gui",      action="store_true", help="Headless mode (no plots)")
    parser.add_argument("--pybullet-gui", action="store_true",
                        help="Open PyBullet GUI to inspect the depth collision mesh")
    parser.add_argument("--no-plan", action="store_true", help="Skip PDDL planning step")
    parser.add_argument("--task",    default=DEFAULT_TASK,
                        help="Task description passed to LayeredDomainGenerator")
    parser.add_argument(
        "--joint-state", type=float, nargs=7, metavar="J",
        default=[-0.141372, -1.317724, -0.441568, 1.532399, -0.132645, 2.028073, -0.60912],
        help="xArm7 joint angles (rad) for FK-based camera-to-world transform.",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use LLM to classify contact relationships in the contact graph.",
    )
    parser.add_argument(
        "--llm-mode", choices=["json", "nl"], default="nl",
        help="LLM prompt mode for contact classification (default: nl).",
    )
    parser.add_argument(
        "--force-mode", choices=["deterministic", "probabilistic"], default=None,
        help="Override hybrid planner mode selection (default: auto).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
