"""
Clutter Planning Integration Test — xArm7 Simulation

Full end-to-end pipeline:
  1. PyBullet scene: table + 5 blocks in a cluttered arrangement
  2. GSAM2 detection (GroundingDINO + SAM2) across 3 viewpoints
     — tracker internally computes clearance profiles, contact graph,
       surface free-space maps, and multi-viewpoint occlusion map
  3. GSAM2 object IDs remapped to canonical scene IDs via projected
     centroid matching; all perception outputs copied across
  4. ClutterGrounder evaluates all 9 clutter predicates against the
     populated registry
  5. LayeredDomainGenerator.generate_domain() with registry= produces
     a PDDL plan with ground-truth stable-on in the L5 initial state

Requires:
    SAM2_CKPT env var pointing to the SAM2 checkpoint
    GEMINI_API_KEY env var (only needed for planning step)

Run:
    uv run scripts/test_clutter_planning.py
    uv run scripts/test_clutter_planning.py --no-gui
    uv run scripts/test_clutter_planning.py --no-plan
    uv run scripts/test_clutter_planning.py --task "Move the blue block off the green block"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ion()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.sim.scene_environment import SceneEnvironment, CAMERA_AIM_JOINTS
from src.perception.object_registry import DetectedObject, DetectedObjectRegistry
from src.planning.clutter_module import ClutterGrounder
from src.planning.domain_knowledge_base import DomainKnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clutter_test")

_DIVIDER = "=" * 70

# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------
# Cluttered pile: green_block_1 is the grasp target; blue_block_1 is stacked
# on top of it; red_block_1 and orange_block_1 are adjacent lateral blockers.
# yellow_block_1 is free on the table as a distractor.

TABLE_POS        = [0.40,  0.00, 0.20]
GREEN_TARGET_POS = [0.38,  0.00, 0.27]
BLUE_STACK_POS   = [0.38,  0.00, 0.33]   # directly on green target
RED_BLOCK_POS    = [0.44, -0.05, 0.27]
ORANGE_BLOCK_POS = [0.32,  0.05, 0.27]
YELLOW_BLOCK_POS = [0.44,  0.06, 0.27]

SCENE_OBJECTS = [
    {"object_id": "table_1",        "object_type": "surface", "position_3d": TABLE_POS},
    {"object_id": "green_block_1",  "object_type": "block",   "position_3d": GREEN_TARGET_POS},
    {"object_id": "blue_block_1",   "object_type": "block",   "position_3d": BLUE_STACK_POS},
    {"object_id": "red_block_1",    "object_type": "block",   "position_3d": RED_BLOCK_POS},
    {"object_id": "orange_block_1", "object_type": "block",   "position_3d": ORANGE_BLOCK_POS},
    {"object_id": "yellow_block_1", "object_type": "block",   "position_3d": YELLOW_BLOCK_POS},
]

# GroundingDINO tags to inject so all scene objects are detected
DETECTION_TAGS = [
    "green block", "blue block", "red block", "orange block",
    "yellow block", "table", "surface",
]

VIEWPOINT_MAIN = CAMERA_AIM_JOINTS
VIEWPOINT_SIDE = [0.4,  -1.2, -0.15, 1.5,  0.0, 1.8, -0.1]
VIEWPOINT_LEFT = [-0.3, -1.4, -0.10, 1.3,  0.0, 2.0, -0.1]

DEFAULT_TASK = "Pick up the green block from the cluttered table"

# Per-object display colours (RGB 0-1)
OBJ_COLORS = {
    "table_1":        (0.55, 0.45, 0.30),
    "green_block_1":  (0.15, 0.75, 0.15),
    "blue_block_1":   (0.15, 0.35, 0.85),
    "red_block_1":    (0.85, 0.15, 0.15),
    "orange_block_1": (0.95, 0.50, 0.05),
    "yellow_block_1": (0.90, 0.85, 0.05),
}

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

def _mask_overlay(ax, color_img: np.ndarray, masks: Dict[str, np.ndarray],
                  title: str = "") -> None:
    ax.imshow(color_img)
    for oid, mask in masks.items():
        if not mask.any():
            continue
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        c = OBJ_COLORS.get(oid, (0.5, 0.5, 0.5))
        rgba[mask, 0] = 1.0 - c[0]
        rgba[mask, 1] = 1.0 - c[1]
        rgba[mask, 2] = 1.0 - c[2]
        rgba[mask, 3] = 0.45
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

# ---------------------------------------------------------------------------
# Projection helpers (for GSAM2 ID → canonical ID remapping)
# ---------------------------------------------------------------------------

def _world_to_cam(points_world: np.ndarray, cam: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    origin = np.array(cam["position"], dtype=float)
    rot = Rotation.from_quat(cam["quaternion_xyzw"])
    return rot.inv().apply(points_world - origin)

def _project(pts_cam: np.ndarray, intr) -> np.ndarray:
    z = pts_cam[:, 2]
    u = intr.fx * pts_cam[:, 0] / z + intr.cx
    v = intr.fy * pts_cam[:, 1] / z + intr.cy
    return np.stack([u, v], axis=1)

# ---------------------------------------------------------------------------
# GSAM2 ID → canonical scene ID remapping
# ---------------------------------------------------------------------------

def _remap_gsam2_to_canonical(
    raw_masks: Dict[str, np.ndarray],
    scene_objects: list,
    robot_state: dict,
    intrinsics,
    depth_shape: Tuple[int, int],
) -> Dict[str, str]:
    """Return {canonical_id: gsam2_id} by nearest projected centroid.

    Projects each scene object's known 3D position to pixel space and assigns
    the nearest unmatched GSAM2 detection centroid within 120 px.
    """
    cam = robot_state.get("camera")
    H, W = depth_shape

    gsam2_centroids: Dict[str, np.ndarray] = {}
    for gid, mask in raw_masks.items():
        if mask.any():
            ys, xs = np.where(mask)
            gsam2_centroids[gid] = np.array([xs.mean(), ys.mean()])

    mapping: Dict[str, str] = {}
    used: set = set()

    for obj in scene_objects:
        oid = obj["object_id"]
        if cam is None or not gsam2_centroids:
            continue
        pos_cam = _world_to_cam(np.array(obj["position_3d"])[None], cam)[0]
        if pos_cam[2] <= 0:
            continue
        uv = _project(pos_cam[None], intrinsics)[0]

        best_gid, best_dist = None, float("inf")
        for gid, centroid in gsam2_centroids.items():
            if gid in used:
                continue
            dist = float(np.linalg.norm(uv - centroid))
            if dist < best_dist:
                best_dist = dist
                best_gid = gid

        if best_gid is not None and best_dist < 120:
            mapping[oid] = best_gid
            used.add(best_gid)
            _info(f"  {oid} ← {best_gid}  (dist={best_dist:.1f}px)")
        else:
            _warn(f"  {oid}: no GSAM2 match (nearest={best_dist:.1f}px)")

    return mapping


def _build_canonical_registry(
    tracker_registry: DetectedObjectRegistry,
    raw_masks: Dict[str, np.ndarray],
    id_map: Dict[str, str],
    scene_objects: list,
) -> Tuple[DetectedObjectRegistry, Dict[str, np.ndarray]]:
    """Build a canonical registry keyed by scene object IDs.

    Copies DetectedObject entries from the tracker registry (keyed by GSAM2 IDs)
    into a new registry keyed by canonical scene IDs.  Perception outputs
    (clearance_profile, surface_map, contact_graph, occlusion_map) are
    transferred across, with object ID references in contact edges rewritten.
    """
    canonical = DetectedObjectRegistry()
    canonical_masks: Dict[str, np.ndarray] = {}

    # Build reverse map: gsam2_id → canonical_id
    reverse: Dict[str, str] = {v: k for k, v in id_map.items()}

    for obj in scene_objects:
        canon_id = obj["object_id"]
        gsam2_id = id_map.get(canon_id)

        if gsam2_id is not None:
            gsam2_obj = tracker_registry.get_object(gsam2_id)
        else:
            gsam2_obj = None

        if gsam2_obj is not None:
            # Create a copy with the canonical ID
            canon_obj = DetectedObject(
                object_type=obj["object_type"],
                object_id=canon_id,
                affordances=gsam2_obj.affordances,
                interaction_points=gsam2_obj.interaction_points,
                position_2d=gsam2_obj.position_2d,
                position_3d=gsam2_obj.position_3d,
                bounding_box_2d=gsam2_obj.bounding_box_2d,
                timestamp=gsam2_obj.timestamp,
                clearance_profile=gsam2_obj.clearance_profile,
                surface_map=gsam2_obj.surface_map,
            )
            canonical_masks[canon_id] = raw_masks.get(gsam2_id, np.zeros((1, 1), dtype=bool))
        else:
            # Not detected — create a stub from scene description
            canon_obj = DetectedObject(
                object_type=obj["object_type"],
                object_id=canon_id,
                affordances={"graspable", "stackable"} if obj["object_type"] == "block"
                            else {"placeable_on"},
                position_3d=np.array(obj["position_3d"], dtype=float),
            )
            canonical_masks[canon_id] = np.zeros(
                next(iter(raw_masks.values())).shape if raw_masks else (480, 640),
                dtype=bool,
            )

        canonical.add_object(canon_obj)

    # Transfer contact graph — rewrite all object_id references
    graph = tracker_registry.contact_graph
    if graph is not None:
        import copy
        canon_graph = copy.deepcopy(graph)
        for edge in canon_graph.edges:
            edge.obj_a = reverse.get(edge.obj_a, edge.obj_a)
            edge.obj_b = reverse.get(edge.obj_b, edge.obj_b)
        # Rewrite dicts keyed by object_id
        canon_graph.stability_scores = {
            reverse.get(k, k): v for k, v in graph.stability_scores.items()
        }
        canon_graph.removal_consequences = {
            reverse.get(k, k): v for k, v in graph.removal_consequences.items()
        }
        canon_graph.support_tree = {
            reverse.get(k, k): [reverse.get(u, u) for u in v]
            for k, v in graph.support_tree.items()
        }
        canonical.contact_graph = canon_graph

    # Transfer occlusion map — rewrite object_id keys
    occ = tracker_registry.occlusion_map
    if occ is not None:
        import copy
        canon_occ = copy.deepcopy(occ)
        canon_occ.per_object_visibility = {
            reverse.get(k, k): v
            for k, v in occ.per_object_visibility.items()
        }
        # Rewrite occluding_objects lists inside each PerObjectVisibility entry
        for vis in canon_occ.per_object_visibility.values():
            vis.occluding_objects = [
                reverse.get(o, o) for o in vis.occluding_objects
            ]
        canonical.occlusion_map = canon_occ

    return canonical, canonical_masks

# ---------------------------------------------------------------------------
# Detection + full perception across multiple viewpoints
# ---------------------------------------------------------------------------

def run_detection_and_perception(
    env: SceneEnvironment,
    tracker,
    scene_objects: list,
    gui: bool,
) -> Tuple[DetectedObjectRegistry, Dict[str, np.ndarray], np.ndarray, object]:
    """Sweep 3 viewpoints, running detect_objects() at each.

    The tracker accumulates the occlusion history internally across calls.
    After the final viewpoint we return to the main viewpoint and remap
    GSAM2 object IDs to canonical scene IDs.

    Returns:
        (canonical_registry, canonical_masks, color_frame, intrinsics)
    """
    _section("Object Detection + Perception (3 viewpoints)")

    color_final = depth_final = intrinsics_final = robot_state_final = None

    # Sweep viewpoints to build the occlusion history.
    # Clearances, contact graph, and surface maps are expensive and only need
    # to run once on the final main-viewpoint frame — disable them during the sweep.
    tracker._compute_clearances   = False
    tracker._compute_contacts     = False
    tracker._compute_surface_maps = False

    for vp_name, vp_joints in [
        ("main",  VIEWPOINT_MAIN),
        ("side",  VIEWPOINT_SIDE),
        ("left",  VIEWPOINT_LEFT),
    ]:
        env.set_robot_joints(vp_joints)
        env.set_status(f"Detecting: {vp_name} viewpoint", color=[0.3, 0.7, 0.9])
        time.sleep(0.4)

        color, depth, intrinsics = env.capture_camera_frame()
        robot_state = env.get_robot_state()
        if depth is None:
            _warn(f"No depth frame at {vp_name} viewpoint — skipping")
            continue

        _info(f"[{vp_name}] detect_objects() — masks + occlusion history only")
        asyncio.run(tracker.detect_objects(
            color_frame=color,
            depth_frame=depth,
            camera_intrinsics=intrinsics,
            robot_state=robot_state,
        ))
        _info(f"[{vp_name}] GSAM2 raw detections: {list(tracker._last_masks.keys())}")

    # Re-enable full perception for the final main-viewpoint pass
    tracker._compute_clearances   = True
    tracker._compute_contacts     = True
    tracker._compute_surface_maps = True

    env.set_robot_joints(VIEWPOINT_MAIN)
    time.sleep(0.3)
    color_final, depth_final, intrinsics_final = env.capture_camera_frame()
    robot_state_final = env.get_robot_state()

    # Final pass: clearances + contact graph + surface maps computed once here
    _info("Final pass at main viewpoint: clearances, contact graph, surface maps...")
    asyncio.run(tracker.detect_objects(
        color_frame=color_final,
        depth_frame=depth_final,
        camera_intrinsics=intrinsics_final,
        robot_state=robot_state_final,
    ))

    raw_masks = dict(tracker._last_masks)
    _info(f"Final GSAM2 detections ({len(raw_masks)}): {list(raw_masks.keys())}")
    _info(f"Tracker registry: {tracker.registry.count()} objects, "
          f"contact_graph={'yes' if tracker.registry.contact_graph else 'no'}, "
          f"occlusion_map={'yes' if tracker.registry.occlusion_map else 'no'}")

    # --- Remap GSAM2 IDs → canonical scene IDs ---
    _info("Remapping GSAM2 IDs → canonical scene IDs:")
    id_map = _remap_gsam2_to_canonical(
        raw_masks, scene_objects, robot_state_final, intrinsics_final,
        depth_final.shape,
    )

    # --- Build canonical registry ---
    canonical_registry, canonical_masks = _build_canonical_registry(
        tracker.registry, raw_masks, id_map, scene_objects,
    )

    detected_canon = [o.object_id for o in canonical_registry.get_all_objects()]
    _info(f"Canonical registry: {detected_canon}")

    nonempty = [k for k, v in canonical_masks.items() if v.any()]
    _info(f"Non-empty canonical masks: {nonempty}")

    if canonical_registry.contact_graph:
        g = canonical_registry.contact_graph
        _info(f"Contact graph: {len(g.edges)} edges")
        for edge in g.edges:
            _info(f"  {edge.obj_a} → {edge.obj_b}  {edge.contact_type}  "
                  f"consequence={edge.removal_consequence}")
        for lower, uppers in g.support_tree.items():
            if uppers:
                _info(f"  support_tree: {lower} → {uppers}")

    if canonical_registry.occlusion_map:
        _info("Occlusion map visibility:")
        for oid, vis in canonical_registry.occlusion_map.per_object_visibility.items():
            _info(f"  {oid}: {vis.visible_fraction:.0%}  occluders={vis.occluding_objects}")

    _ok("Detection and perception complete")
    return canonical_registry, canonical_masks, color_final, intrinsics_final

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

    grounder = ClutterGrounder(registry)
    all_objects = registry.get_all_objects()
    block_ids = [o.object_id for o in all_objects if o.object_type == "block"]
    obj_ids   = [o.object_id for o in all_objects]

    # stable-on (TRUE, grounded from contact graph)
    stable_facts = grounder.ground_stable_on()
    _info(f"stable-on TRUE facts ({len(stable_facts)}):")
    for pred_name, args in stable_facts:
        _info(f"  ({pred_name} {' '.join(args)})")

    # Per-block unary sensed predicates
    _info("\nPer-block predicate evaluation:")
    header = f"  {'Object':<22} {'access-clear':>12} {'displaceable':>12} {'removal-safe':>12} {'visible':>8}"
    _info(header)
    _info("  " + "-" * 70)
    for oid in block_ids:
        access = grounder.evaluate_access_clear(oid)
        displ  = grounder.evaluate_displaceable(oid)
        safe   = grounder.evaluate_removal_safe(oid)
        vis    = grounder.evaluate_visible(oid)
        _info(f"  {oid:<22} {str(access):>12} {str(displ):>12} {str(safe):>12} {str(vis):>8}")

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

    if not gui:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Clutter Predicate Grounding", fontsize=12, fontweight="bold")

    # Left: scene overview + masks
    _mask_overlay(axes[0], color, masks, title="Detected scene")
    legend_patches = [
        mpatches.Patch(color=OBJ_COLORS.get(oid, (0.5, 0.5, 0.5)), label=oid)
        for oid in obj_ids
    ]
    axes[0].legend(handles=legend_patches, loc="lower right", fontsize=7)

    # Right: boolean heatmap — blocks × unary predicates
    if block_ids:
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
    else:
        axes[1].text(0.5, 0.5, "No blocks detected", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Grounding result shown — press Enter for planning.")

# ---------------------------------------------------------------------------
# Visibility summary figure
# ---------------------------------------------------------------------------

def show_visibility_summary(registry: DetectedObjectRegistry, gui: bool) -> None:
    occ_map = registry.occlusion_map
    if occ_map is None or not gui:
        return

    block_ids = [o.object_id for o in registry.get_all_objects() if o.object_type == "block"]
    if not block_ids:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Occlusion Map — Visible Fractions", fontsize=11, fontweight="bold")

    fracs = [
        occ_map.per_object_visibility.get(oid, None) for oid in block_ids
    ]
    vals = [v.visible_fraction if v is not None else 0.0 for v in fracs]
    colors = [(*OBJ_COLORS.get(oid, (0.5, 0.5, 0.5)), 0.85) for oid in block_ids]
    ax.bar(range(len(block_ids)), vals, color=colors)
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1.2, label="visible threshold 0.6")
    ax.set_xticks(range(len(block_ids)))
    ax.set_xticklabels(block_ids, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Visible fraction")
    ax.legend(fontsize=8)

    for i, (val, vis) in enumerate(zip(vals, fracs)):
        occluders = vis.occluding_objects if vis is not None else []
        label = f"{val:.0%}"
        if occluders:
            label += f"\n({', '.join(occluders[:2])})"
        ax.text(i, val + 0.02, label, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Visibility summary shown — press Enter for planning.")

# ---------------------------------------------------------------------------
# PDDL planning via LayeredDomainGenerator
# ---------------------------------------------------------------------------

def run_planning(
    registry: DetectedObjectRegistry,
    scene_objects: list,
    task: str,
    gui: bool,
) -> None:
    _section("PDDL Planning — LayeredDomainGenerator")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        _warn("GEMINI_API_KEY not set — skipping planning step")
        _warn("Export GEMINI_API_KEY and re-run without --no-plan")
        return

    from src.planning.layered_domain_generator import LayeredDomainGenerator

    # Pre-load DKB — installs clutter predicate hints so L2 sees them
    dkb = DomainKnowledgeBase(Path("outputs/dkb"))
    dkb.load()
    hints = dkb.get_predicate_hints(task)
    _info(f"DKB predicate hints ({len(hints)}): {hints[:6]}")

    # Build observed_objects from the canonical registry
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
    _info(f"Objects passed to generator: {[o['object_id'] for o in observed_objects]}")

    generator = LayeredDomainGenerator(api_key=api_key)
    artifact = asyncio.run(
        generator.generate_domain(task, observed_objects=observed_objects, registry=registry)
    )

    _info(f"\nL1 Goals:           {artifact.l1.goal_predicates}")
    _info(f"L2 Predicates:      {len(artifact.l2.predicate_signatures)} signatures")
    _info(f"L3 Actions:         {len(artifact.l3.actions)} action schemas")
    if artifact.l3.sensing_actions:
        _info(f"L3 Sensing actions: {len(artifact.l3.sensing_actions)}")

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

    # L5 initial state heatmap
    block_ids = [o["object_id"] for o in observed_objects if o["object_type"] == "block"]
    if not block_ids:
        return

    preds_to_show = [
        "stable-on", "access-clear", "displaceable",
        "removal-safe", "visible", "obstructs", "supports",
    ]
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

    fig, ax = plt.subplots(figsize=(10, 4))
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
    parser = argparse.ArgumentParser(description="Clutter planning integration test — xArm7")
    parser.add_argument("--no-gui",  action="store_true", help="Headless mode (no plots)")
    parser.add_argument("--no-plan", action="store_true", help="Skip PDDL planning step")
    parser.add_argument("--task",    default=DEFAULT_TASK,
                        help="Task description passed to LayeredDomainGenerator")
    args = parser.parse_args()
    gui = not args.no_gui

    if not gui:
        plt.ioff()

    print(_DIVIDER)
    print("  Clutter Planning Integration Test — xArm7 Simulation")
    print(_DIVIDER)

    # --- Check required env vars before spending time on setup ---
    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info(f"Device: {device}")

    # --- Setup PyBullet environment ---
    env = SceneEnvironment()

    if not gui:
        try:
            import pybullet as p
            import pybullet_data
            def _headless_start():
                env._client = p.connect(p.DIRECT)
                c = env._client
                p.setGravity(0, 0, -9.81, physicsClientId=c)
                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
                p.loadURDF("plane.urdf", physicsClientId=c)
                from src.kinematics.sim.scene_environment import _DEFAULT_URDF
                if _DEFAULT_URDF.exists():
                    env._robot_id = p.loadURDF(
                        str(_DEFAULT_URDF), basePosition=[0, 0, 0],
                        baseOrientation=[0, 0, 0, 1], useFixedBase=True, physicsClientId=c,
                    )
                    env._build_joint_map()
                    env.set_robot_joints(env.initial_joints)
                env._step_running = False
            env.start = _headless_start
        except ImportError:
            pass

    env.start()
    env.add_scene_objects(SCENE_OBJECTS)
    env.set_robot_joints(VIEWPOINT_MAIN)
    time.sleep(0.5)

    # --- Load GSAM2 tracker with all perception modules enabled ---
    _section("Loading GSAM2 tracker")
    from src.perception.gsam2_object_tracker import GSAM2ObjectTracker
    from src.perception.clearance import GripperGeometry

    gripper = GripperGeometry.from_urdf(
        PROJECT_ROOT / "src" / "kinematics" / "sim" / "urdfs"
        / "xarm7_camera" / "xarm7.urdf"
    )

    tracker = GSAM2ObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        compute_clearances=True,
        gripper=gripper,
        compute_contacts=True,
        compute_occlusion=True,
        occlusion_history_len=10,
        occlusion_update_interval=1,
        compute_surface_maps=True,
    )
    # Inject scene-specific detection tags so GroundingDINO finds all objects
    tracker.set_extra_tags(DETECTION_TAGS)
    _ok("GSAM2 tracker loaded")

    # --- Run multi-viewpoint detection + perception ---
    try:
        registry, masks, color, intrinsics = run_detection_and_perception(
            env, tracker, SCENE_OBJECTS, gui,
        )
    except Exception:
        log.exception("Detection/perception failed")
        env.stop()
        sys.exit(1)

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
            run_planning(registry, SCENE_OBJECTS, args.task, gui)
        except Exception:
            log.exception("Planning failed")
    else:
        _info("--no-plan set: skipping PDDL planning")

    print(f"\n{_DIVIDER}")
    print("  Done.")
    print(_DIVIDER)
    env.stop()


if __name__ == "__main__":
    main()
