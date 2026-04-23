"""
Clutter Planning Integration Test — Live RealSense RGB-D

Full end-to-end pipeline on real hardware:
  1. RealSense D435 captures aligned RGB-D frames (with spatial/temporal filtering)
  2. GSAM2 (GroundingDINO + SAM2) detects and segments all objects
     — tracker internally computes clearance profiles, contact graph,
       surface free-space maps, and occlusion map
  3. ClutterGrounder evaluates all 9 clutter predicates against the
     populated registry
  4. LayeredDomainGenerator.generate_domain() produces a PDDL plan with
     ground-truth stable-on facts injected from the contact graph

Unlike the sim version there is no canonical ID remapping — object IDs
come directly from GSAM2 (e.g. "green_block_3", "table_1").

Requires:
    SAM2_CKPT env var pointing to the SAM2 checkpoint
    GEMINI_API_KEY env var (only for planning step)
    RealSense D435 connected via USB3

Run:
    uv run scripts/test_clutter_planning_realsense.py
    uv run scripts/test_clutter_planning_realsense.py --no-plan
    uv run scripts/test_clutter_planning_realsense.py --no-gui
    uv run scripts/test_clutter_planning_realsense.py --task "Pick up the red cup"
    uv run scripts/test_clutter_planning_realsense.py --frames 5   # capture N frames before planning
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ion()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_camera import RealSenseCamera
from src.perception.object_registry import DetectedObjectRegistry
from src.planning.clutter_module import ClutterGrounder
from src.planning.domain_knowledge_base import DomainKnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clutter_realsense")

_DIVIDER = "=" * 70

DEFAULT_TASK    = "Pick up the rubber duck from the cluttered table"
DEFAULT_FRAMES  = 5   # number of frames to accumulate before planning

DETECTION_TAGS = [
    "rubber duck", "red cup", "dark blue mug", "box", "table", "surface",
]

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{_DIVIDER}\n  {title}\n{_DIVIDER}")
    log.info("BEGIN: %s", title)

def _ok(msg: str)   -> None: print(f"  ✓  {msg}"); log.info("PASS: %s", msg)
def _info(msg: str) -> None: print(f"  •  {msg}"); log.info("INFO: %s", msg)
def _warn(msg: str) -> None: print(f"  ⚠  {msg}"); log.warning("WARN: %s", msg)

def _wait(gui: bool, prompt: str = "Press Enter to continue...") -> None:
    if gui:
        try:
            input(f"\n  {prompt}\n")
        except EOFError:
            pass

def _show_and_wait(fig, gui: bool, prompt: str = "Press Enter to continue...") -> None:
    if not gui:
        plt.close(fig)
        return
    plt.show(block=False)
    plt.pause(0.1)
    _wait(gui, prompt)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Detection + full perception — N frames from RealSense
# ---------------------------------------------------------------------------

def run_detection_and_perception(
    camera: RealSenseCamera,
    tracker,
    intrinsics,
    n_frames: int,
    gui: bool,
):
    """Capture n_frames from the RealSense and run detect_objects() on each.

    The first n_frames-1 calls accumulate the occlusion history with clearances
    and contacts disabled. The final call enables all perception outputs so
    everything is computed exactly once on the freshest frame.

    Returns (registry, last_color, last_depth).
    """
    _section(f"Object Detection + Perception ({n_frames} frames)")

    color_last = depth_last = None

    # Sweep frames: occlusion history only, defer expensive perception to last
    tracker._compute_clearances   = False
    tracker._compute_contacts     = False
    tracker._compute_surface_maps = False

    for i in range(n_frames - 1):
        color, depth = camera.get_aligned_frames()
        _info(f"Frame {i+1}/{n_frames}: depth [{depth.min():.3f}, {depth.max():.3f}] m")

        asyncio.run(tracker.detect_objects(
            color_frame=color,
            depth_frame=depth,
            camera_intrinsics=intrinsics,
        ))
        _info(f"  GSAM2 detections: {list(tracker._last_masks.keys())}")
        color_last, depth_last = color, depth

    # Final frame: full perception pipeline
    tracker._compute_clearances   = True
    tracker._compute_contacts     = True
    tracker._compute_surface_maps = True

    _info(f"Frame {n_frames}/{n_frames}: full perception pass...")
    color_last, depth_last = camera.get_aligned_frames()
    asyncio.run(tracker.detect_objects(
        color_frame=color_last,
        depth_frame=depth_last,
        camera_intrinsics=intrinsics,
    ))

    registry = tracker.registry
    detected_ids = [o.object_id for o in registry.get_all_objects()]
    _info(f"Registry: {len(detected_ids)} objects — {detected_ids}")

    nonempty = [k for k, v in tracker._last_masks.items() if v.any()]
    _info(f"Non-empty masks: {nonempty}")

    if registry.contact_graph:
        g = registry.contact_graph
        _info(f"Contact graph: {len(g.edges)} edges")
        for edge in g.edges:
            _info(f"  {edge.obj_a} → {edge.obj_b}  {edge.contact_type}"
                  f"  consequence={edge.removal_consequence}")
        for lower, uppers in g.support_tree.items():
            if uppers:
                _info(f"  support_tree: {lower} → {uppers}")

    if registry.occlusion_map:
        _info("Occlusion map visibility:")
        for oid, vis in registry.occlusion_map.per_object_visibility.items():
            _info(f"  {oid}: {vis.visible_fraction:.0%}  occluders={vis.occluding_objects}")

    _ok("Detection and perception complete")
    return registry, color_last, depth_last

# ---------------------------------------------------------------------------
# Scene overview figure
# ---------------------------------------------------------------------------

def show_scene_overview(
    color: np.ndarray,
    masks: Dict[str, np.ndarray],
    task: str,
    gui: bool,
) -> None:
    if not gui:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Detected Scene", fontsize=11, fontweight="bold")
    ax.imshow(color)

    # Overlay each mask with a distinct colour
    cmap = plt.get_cmap("tab10")
    legend_patches = []
    for idx, (oid, mask) in enumerate(masks.items()):
        if not mask.any():
            continue
        c = cmap(idx % 10)
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask, :3] = c[:3]
        rgba[mask, 3]  = 0.45
        ax.imshow(rgba)
        legend_patches.append(mpatches.Patch(color=c, label=oid))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
    ax.set_title(f"Task: {task}", fontsize=9)
    ax.axis("off")
    _show_and_wait(fig, gui, "Scene overview shown — press Enter for grounding.")

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

    grounder  = ClutterGrounder(registry)
    all_objs  = registry.get_all_objects()
    obj_ids   = [o.object_id for o in all_objs]
    # Treat anything without "table"/"surface" in its type as a graspable object
    block_ids = [
        o.object_id for o in all_objs
        if o.object_type.lower() not in ("surface", "table", "shelf", "tray", "counter", "desk")
    ]

    # stable-on TRUE facts from contact graph
    stable_facts = grounder.ground_stable_on()
    _info(f"stable-on TRUE facts ({len(stable_facts)}):")
    for pred_name, args in stable_facts:
        _info(f"  ({pred_name} {' '.join(args)})")

    # Per-object unary predicate evaluation
    if block_ids:
        _info("\nPer-object predicate evaluation:")
        header = f"  {'Object':<28} {'access-clear':>12} {'displaceable':>12} {'removal-safe':>12} {'visible':>8}"
        _info(header)
        _info("  " + "-" * 76)
        for oid in block_ids:
            access = grounder.evaluate_access_clear(oid)
            displ  = grounder.evaluate_displaceable(oid)
            safe   = grounder.evaluate_removal_safe(oid)
            vis    = grounder.evaluate_visible(oid)
            _info(f"  {oid:<28} {str(access):>12} {str(displ):>12} {str(safe):>12} {str(vis):>8}")

    # Obstruction pairs
    _info("\nObstruction pairs (obstructs ?blocker ?target):")
    found = False
    for blocker in block_ids:
        for target in block_ids:
            if blocker != target and grounder.evaluate_obstructs(blocker, target):
                _info(f"  (obstructs {blocker} {target})")
                found = True
    if not found:
        _info("  (none from clearance profiles)")

    # Support relationships
    _info("\nSupport (supports ?lower ?upper):")
    found = False
    for lower in obj_ids:
        for upper in obj_ids:
            if lower != upper and grounder.evaluate_supports(lower, upper):
                _info(f"  (supports {lower} {upper})")
                found = True
    if not found:
        _info("  (none detected)")

    _ok("Clutter grounding complete")

    if not gui or not block_ids:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Clutter Predicate Grounding", fontsize=12, fontweight="bold")

    # Left: live scene with per-mask overlays
    cmap = plt.get_cmap("tab10")
    axes[0].imshow(color)
    legend_patches = []
    for idx, (oid, mask) in enumerate(masks.items()):
        if not mask.any():
            continue
        c = cmap(idx % 10)
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask, :3] = c[:3]
        rgba[mask, 3]  = 0.45
        axes[0].imshow(rgba)
        legend_patches.append(mpatches.Patch(color=c, label=oid))
    if legend_patches:
        axes[0].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[0].set_title("Detected scene", fontsize=9)
    axes[0].axis("off")

    # Right: boolean heatmap — objects × unary predicates
    unary_preds = ["access-clear", "displaceable", "removal-safe", "visible"]
    eval_fns = [
        grounder.evaluate_access_clear,
        grounder.evaluate_displaceable,
        grounder.evaluate_removal_safe,
        grounder.evaluate_visible,
    ]
    grid = np.array([[fn(oid) for fn in eval_fns] for oid in block_ids], dtype=float)
    ax = axes[1]
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
    cmap = plt.get_cmap("tab10")
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
        label = f"{val:.0%}"
        if occluders:
            label += f"\n({', '.join(occluders[:2])})"
        ax.text(i, val + 0.02, label, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Visibility summary shown — press Enter for planning.")

# ---------------------------------------------------------------------------
# PDDL planning
# ---------------------------------------------------------------------------

def run_planning(
    registry: DetectedObjectRegistry,
    task: str,
    gui: bool,
) -> None:
    _section("PDDL Planning — LayeredDomainGenerator")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        _warn("GEMINI_API_KEY not set — skipping planning step")
        return

    from src.planning.layered_domain_generator import LayeredDomainGenerator

    dkb = DomainKnowledgeBase(Path("outputs/dkb"))
    dkb.load()
    hints = dkb.get_predicate_hints(task)
    _info(f"DKB predicate hints ({len(hints)}): {hints[:6]}")

    observed_objects = []
    for obj in registry.get_all_objects():
        pos = obj.position_3d.tolist() if obj.position_3d is not None else [0.0, 0.0, 0.0]
        observed_objects.append({
            "object_id":   obj.object_id,
            "object_type": obj.object_type,
            "affordances": list(obj.affordances),
            "position_3d": pos,
        })

    _info(f"Task: '{task}'")
    _info(f"Objects: {[o['object_id'] for o in observed_objects]}")

    generator = LayeredDomainGenerator(api_key=api_key)
    artifact  = asyncio.run(
        generator.generate_domain(task, observed_objects=observed_objects, registry=registry)
    )

    _info(f"\nL1 Goals:       {artifact.l1.goal_predicates}")
    _info(f"L2 Predicates:  {len(artifact.l2.predicate_signatures)}")
    _info(f"L3 Actions:     {len(artifact.l3.actions)}")
    if artifact.l3.sensing_actions:
        _info(f"L3 Sensing:     {len(artifact.l3.sensing_actions)}")

    _info(f"\nL5 TRUE literals ({len(artifact.l5.true_literals)}):")
    for lit in artifact.l5.true_literals[:25]:
        _info(f"  {lit}")
    if len(artifact.l5.true_literals) > 25:
        _info(f"  ... ({len(artifact.l5.true_literals) - 25} more)")

    plan = getattr(artifact.l5, "plan", None) or getattr(artifact, "plan", None)
    if plan:
        _info(f"\nPlan ({len(plan)} steps):")
        for i, step in enumerate(plan):
            _info(f"  {i+1:2d}. {step}")
    else:
        _info("No plan attached (solver not connected — L5 initial state is the output)")

    dkb.record_execution(task, artifact)
    _ok("DKB execution recorded")
    _ok("Planning complete")

    if not gui:
        return

    # L5 heatmap
    block_ids     = [o["object_id"] for o in observed_objects
                     if o["object_type"] not in ("surface", "table")]
    preds_to_show = ["stable-on", "access-clear", "displaceable",
                     "removal-safe", "visible", "obstructs", "supports"]
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

    fig, ax = plt.subplots(figsize=(10, max(3, len(block_ids) * 0.6 + 2)))
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clutter planning integration test — live RealSense"
    )
    parser.add_argument("--no-gui",  action="store_true", help="Headless mode (no plots)")
    parser.add_argument("--no-plan", action="store_true", help="Skip PDDL planning step")
    parser.add_argument("--frames",  type=int, default=DEFAULT_FRAMES,
                        help=f"Number of frames to capture before planning (default: {DEFAULT_FRAMES})")
    parser.add_argument("--task",    default=DEFAULT_TASK,
                        help="Task description passed to LayeredDomainGenerator")
    parser.add_argument(
        "--joint-state", type=float, nargs=7, metavar="J",
        default=[0.100085, -1.407677, -0.098652, 1.314592, 0.0, 2.0, -0.112296],
        help="xArm7 joint angles (rad) for FK-based camera-to-world transform "
             "(default: tabletop viewing pose).",
    )
    args = parser.parse_args()
    gui = not args.no_gui

    if not gui:
        plt.ioff()

    print(_DIVIDER)
    print("  Clutter Planning Integration Test — Live RealSense")
    print(_DIVIDER)

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info(f"Device: {device}")

    # --- Start RealSense camera ---
    _section("Starting RealSense camera")
    camera = RealSenseCamera(width=640, height=480, fps=30, enable_depth=True)
    intrinsics = camera.get_camera_intrinsics()
    _ok(f"Camera ready  fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}"
        f"  cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}")

    # --- Robot interface for world-frame transforms ---
    robot_interface = None
    _section("Setting up robot interface (xArm7 FK)")
    try:
        from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
        robot_interface = XArmPybulletInterface()
        robot_interface.set_current_joint_state(args.joint_state)
        _info(f"Joint state: {[f'{j:.3f}' for j in args.joint_state]}")
        pos, rot = robot_interface.get_camera_transform()
        if pos is not None:
            _ok(f"Camera transform: pos={pos.round(3).tolist()}")
        else:
            _warn("FK returned no camera transform — falling back to camera frame")
            robot_interface = None
    except Exception as exc:
        _warn(f"Robot interface unavailable ({exc}) — using camera frame")

    # --- Load GSAM2 tracker ---
    _section("Loading GSAM2 tracker")
    from src.perception.gsam2_object_tracker import GSAM2ObjectTracker
    from src.perception.clearance import GripperGeometry

    gripper = GripperGeometry()

    tracker = GSAM2ObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        compute_clearances=True,
        gripper=gripper,
        compute_contacts=True,
        compute_occlusion=True,
        occlusion_history_len=max(args.frames, 5),
        occlusion_update_interval=1,
        compute_surface_maps=True,
        robot_interface=robot_interface,
    )
    tracker.set_extra_tags(DETECTION_TAGS)
    _ok("GSAM2 tracker loaded")

    # --- Capture and detect ---
    try:
        registry, color, depth = run_detection_and_perception(
            camera, tracker, intrinsics, args.frames, gui,
        )
    except Exception:
        log.exception("Detection/perception failed")
        camera.stop()
        sys.exit(1)

    masks = dict(tracker._last_masks)

    # --- Scene overview ---
    try:
        show_scene_overview(color, masks, args.task, gui)
    except Exception:
        log.exception("Scene overview failed")

    # --- Clutter predicate grounding ---
    try:
        run_clutter_grounding(registry, color, masks, gui)
    except Exception:
        log.exception("Clutter grounding failed")

    # --- Visibility summary ---
    try:
        show_visibility_summary(registry, gui)
    except Exception:
        log.exception("Visibility summary failed")

    # --- PDDL planning ---
    if not args.no_plan:
        try:
            run_planning(registry, args.task, gui)
        except Exception:
            log.exception("Planning failed")
    else:
        _info("--no-plan set: skipping PDDL planning")

    print(f"\n{_DIVIDER}")
    print("  Done.")
    print(_DIVIDER)
    camera.stop()


if __name__ == "__main__":
    main()
