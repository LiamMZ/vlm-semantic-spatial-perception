"""
Perception component integration test — xArm7 simulation.

Exercises each new perception extension (clearance, contact graph, occlusion
map, surface free-space map) against synthetic RGB-D frames captured from the
PyBullet xArm7 simulation.  No real robot or ML models required.

Masks are generated analytically from ground-truth object positions + intrinsics
(depth-threshold segmentation), so the test is fully self-contained.

After each test a matplotlib figure is shown with relevant visualizations.
Press Enter (or close the figure) to advance to the next test.

Run:
    uv run scripts/test_perception_components.py
    uv run scripts/test_perception_components.py --no-gui   # headless (no plots)
    uv run scripts/test_perception_components.py --tests clearance contact
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Set backend before any other matplotlib import so interactive show() works.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.sim.scene_environment import SceneEnvironment, CAMERA_AIM_JOINTS
from src.perception.clearance import GripperGeometry, compute_clearance_profile
from src.perception.contact_graph import compute_contact_graph
from src.perception.occlusion import CameraPose, ObservationRecord, compute_occlusion_map
from src.perception.surface_map import compute_surface_maps
from src.perception.object_registry import DetectedObject
from src.perception.gsam2_object_tracker import GSAM2ObjectTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("perception_test")

_DIVIDER = "=" * 70

# ---------------------------------------------------------------------------
# Scene layout — cluttered pile, green_block_1 is the grasp target
# ---------------------------------------------------------------------------

TABLE_POS        = [0.40,  0.00, 0.20]
GREEN_TARGET_POS = [0.38,  0.00, 0.27]   # target block, partially buried
BLUE_STACK_POS   = [0.38,  0.00, 0.33]   # stacked directly on green target
RED_BLOCK_POS    = [0.44, -0.05, 0.27]   # adjacent lateral blocker
ORANGE_BLOCK_POS = [0.32,  0.05, 0.27]   # lateral blocker / partial occluder
YELLOW_BLOCK_POS = [0.44,  0.06, 0.27]   # free block on table

SCENE_OBJECTS = [
    {"object_id": "table_1",        "object_type": "surface", "position_3d": TABLE_POS},
    {"object_id": "green_block_1",  "object_type": "block",   "position_3d": GREEN_TARGET_POS},
    {"object_id": "blue_block_1",   "object_type": "block",   "position_3d": BLUE_STACK_POS},
    {"object_id": "red_block_1",    "object_type": "block",   "position_3d": RED_BLOCK_POS},
    {"object_id": "orange_block_1", "object_type": "block",   "position_3d": ORANGE_BLOCK_POS},
    {"object_id": "yellow_block_1", "object_type": "block",   "position_3d": YELLOW_BLOCK_POS},
]

# Object to clear during TEST 5 post-manipulation update (stacked on the target)
POSTMANIP_OBJECT  = "blue_block_1"
POSTMANIP_SRC_POS = BLUE_STACK_POS
POSTMANIP_DST_POS = [0.55, -0.15, 0.27]

VIEWPOINT_1 = CAMERA_AIM_JOINTS
VIEWPOINT_2 = [0.4, -1.2, -0.15, 1.5, 0.0, 1.8, -0.1]
VIEWPOINT_3 = [-0.3,-1.4, -0.10, 1.3, 0.0, 2.0, -0.1]

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
    msg = f"\n{_DIVIDER}\n  {title}\n{_DIVIDER}"
    print(msg)
    log.info("BEGIN: %s", title)

def _ok(msg: str)   -> None: print(f"  ✓  {msg}"); log.info("PASS: %s", msg)
def _info(msg: str) -> None: print(f"  •  {msg}"); log.info("INFO: %s", msg)
def _warn(msg: str) -> None: print(f"  ⚠  {msg}"); log.warning("WARN: %s", msg)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _wait(gui: bool, prompt: str = "Press Enter to continue...") -> None:
    """In GUI mode: show prompt and block on Enter. Headless: no-op."""
    if gui:
        try:
            input(f"\n  {prompt}\n")
        except EOFError:
            pass


def _show_and_wait(fig, gui: bool, prompt: str = "Press Enter to continue...") -> None:
    """Display a matplotlib figure non-blocking, then wait for Enter."""
    if not gui:
        plt.close(fig)
        return
    plt.show(block=False)
    plt.pause(0.1)
    _wait(gui, prompt)
    plt.close(fig)


def _mask_overlay(ax, color_img: np.ndarray, masks: Dict[str, np.ndarray],
                  title: str = "") -> None:
    """Draw color_img with coloured mask overlays."""
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
# Mask / depth helpers
# ---------------------------------------------------------------------------

# Known half-extents per object type (must match SceneEnvironment)
_HALF_EXTENTS = {
    "block":     np.array([0.03, 0.03, 0.03]),
    "surface":   np.array([0.25, 0.25, 0.01]),
    "container": np.array([0.07, 0.07, 0.02]),
}


def _world_to_cam(points_world: np.ndarray, cam: dict) -> np.ndarray:
    """Transform Nx3 world points into the camera frame."""
    from scipy.spatial.transform import Rotation
    origin = np.array(cam["position"], dtype=float)
    rot = Rotation.from_quat(cam["quaternion_xyzw"])
    return rot.inv().apply(points_world - origin)


def _project(pts_cam: np.ndarray, intr) -> np.ndarray:
    """Project Nx3 camera-frame points to Nx2 pixel coordinates (u, v)."""
    z = pts_cam[:, 2]
    u = intr.fx * pts_cam[:, 0] / z + intr.cx
    v = intr.fy * pts_cam[:, 1] / z + intr.cy
    return np.stack([u, v], axis=1)


def _build_masks(depth, scene_objects, env, robot_state, intrinsics=None):
    """Generate per-object masks using ground-truth AABB back-projection.

    For each object:
      1. Project the 8 AABB corners to get a pixel bounding box (coarse ROI).
      2. Back-project every pixel in that ROI from depth → world frame.
      3. Accept the pixel if the world point falls inside the object's AABB
         (with a small padding).  This avoids any depth-tolerance guesswork and
         correctly handles perspective distortion.
    """
    from scipy.spatial.transform import Rotation as _Rotation

    cam = robot_state.get("camera")
    H, W = depth.shape
    masks: Dict[str, np.ndarray] = {}

    if cam is None or intrinsics is None:
        for obj in scene_objects:
            masks[obj["object_id"]] = np.zeros((H, W), dtype=bool)
        return masks

    cam_pos = np.array(cam["position"], dtype=float)
    cam_rot = _Rotation.from_quat(cam["quaternion_xyzw"])  # world←cam
    fx, fy = intrinsics.fx, intrinsics.fy
    cx = getattr(intrinsics, "cx", getattr(intrinsics, "ppx", W / 2.0))
    cy = getattr(intrinsics, "cy", getattr(intrinsics, "ppy", H / 2.0))

    for obj in scene_objects:
        oid = obj["object_id"]
        pos = np.array(obj["position_3d"], dtype=float)
        half = _HALF_EXTENTS.get(obj["object_type"], np.array([0.03, 0.03, 0.03]))
        pad_m = 0.01  # 1 cm world-space AABB padding

        # AABB corners → pixel bounding box for the coarse ROI
        offsets = np.array([
            [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
            [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        ], dtype=float) * half
        corners_world = pos + offsets
        corners_cam = _world_to_cam(corners_world, cam)
        centroid_cam = _world_to_cam(pos[None], cam)[0]

        front = corners_cam[:, 2] > 0.01
        if not front.any() or centroid_cam[2] <= 0:
            masks[oid] = np.zeros((H, W), dtype=bool)
            continue

        uvs = _project(corners_cam[front], intrinsics)
        px_pad = 12
        u_min = max(0,   int(uvs[:, 0].min()) - px_pad)
        u_max = min(W-1, int(uvs[:, 0].max()) + px_pad)
        v_min = max(0,   int(uvs[:, 1].min()) - px_pad)
        v_max = min(H-1, int(uvs[:, 1].max()) + px_pad)

        # Back-project ROI pixels into world frame and test against AABB
        roi_depth = depth[v_min:v_max+1, u_min:u_max+1]  # (rH, rW)
        rH, rW = roi_depth.shape

        vs_px = np.arange(v_min, v_min + rH, dtype=np.float32)
        us_px = np.arange(u_min, u_min + rW, dtype=np.float32)
        uu, vv = np.meshgrid(us_px, vs_px)  # (rH, rW)

        valid_d = (roi_depth > 0) & np.isfinite(roi_depth)
        z = roi_depth.copy()
        z[~valid_d] = 0.0

        # Camera-frame 3D points
        x_cam = (uu - cx) * z / fx
        y_cam = (vv - cy) * z / fy
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)  # (N,3)

        # World frame
        pts_world = cam_rot.apply(pts_cam) + cam_pos  # (N,3)

        # AABB test with padding
        lo = pos - half - pad_m
        hi = pos + half + pad_m
        inside = (
            (pts_world[:, 0] >= lo[0]) & (pts_world[:, 0] <= hi[0]) &
            (pts_world[:, 1] >= lo[1]) & (pts_world[:, 1] <= hi[1]) &
            (pts_world[:, 2] >= lo[2]) & (pts_world[:, 2] <= hi[2]) &
            valid_d.reshape(-1)
        )

        mask = np.zeros((H, W), dtype=bool)
        mask[v_min:v_max+1, u_min:u_max+1] = inside.reshape(rH, rW)
        masks[oid] = mask

    return masks


def _detect_masks(
    tracker: "GSAM2ObjectTracker",
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics,
    robot_state: dict,
    scene_objects: list = SCENE_OBJECTS,
) -> Dict[str, np.ndarray]:
    """Run GSAM2 detection and remap masks to known scene object IDs.

    GSAM2 generates its own object_id keys based on detected class names.
    We remap by projecting each known scene object's 3D position into the
    image and assigning the GSAM2 mask whose centroid is nearest.
    """
    import asyncio

    # Force GroundingDINO to look for the actual objects in the scene
    tracker.set_extra_tags([
        "green block", "blue block", "red block",
        "orange block", "yellow block", "table", "surface",
    ])

    asyncio.run(tracker.detect_objects(
        color_frame=color,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        robot_state=robot_state,
    ))
    raw_masks = tracker._last_masks  # {gsam2_id: bool mask}
    _info(f"GSAM2 raw detections: {list(raw_masks.keys())}")

    if not raw_masks:
        return {obj["object_id"]: np.zeros(depth.shape, dtype=bool) for obj in scene_objects}

    # Compute centroid (u, v) for each GSAM2 mask
    gsam2_centroids: Dict[str, np.ndarray] = {}
    for gid, mask in raw_masks.items():
        if mask.any():
            ys, xs = np.where(mask)
            gsam2_centroids[gid] = np.array([xs.mean(), ys.mean()])

    # Project each known scene object's 3D position to pixel space
    cam = robot_state.get("camera")
    remapped: Dict[str, np.ndarray] = {}
    used_gids: set = set()
    H, W = depth.shape

    for obj in scene_objects:
        oid = obj["object_id"]
        if cam is None or intrinsics is None or not gsam2_centroids:
            remapped[oid] = np.zeros((H, W), dtype=bool)
            continue

        pos_cam = _world_to_cam(np.array(obj["position_3d"])[None], cam)[0]
        if pos_cam[2] <= 0:
            remapped[oid] = np.zeros((H, W), dtype=bool)
            continue

        uv = _project(pos_cam[None], intrinsics)[0]  # (2,)

        # Find nearest unused GSAM2 mask centroid
        best_gid, best_dist = None, float("inf")
        for gid, centroid in gsam2_centroids.items():
            if gid in used_gids:
                continue
            dist = float(np.linalg.norm(uv - centroid))
            if dist < best_dist:
                best_dist = dist
                best_gid = gid

        if best_gid is not None and best_dist < 120:  # 120 px max match radius
            remapped[oid] = raw_masks[best_gid].astype(bool)
            used_gids.add(best_gid)
            _info(f"  {oid} ← {best_gid} (dist={best_dist:.1f}px)")
        else:
            remapped[oid] = np.zeros((H, W), dtype=bool)
            _warn(f"  {oid}: no GSAM2 match (nearest={best_dist:.1f}px)")

    return remapped


def _make_detected_objects(scene_objects):
    return [
        DetectedObject(
            object_type=obj["object_type"],
            object_id=obj["object_id"],
            affordances={"graspable"} if obj["object_type"] == "block" else {"placeable_on"},
            position_3d=np.array(obj["position_3d"], dtype=float),
        )
        for obj in scene_objects
    ]


# ---------------------------------------------------------------------------
# TEST 1 — Clearance
# ---------------------------------------------------------------------------

def test_clearance(env, color, depth, masks, intrinsics, robot_state, gui) -> bool:

    _section("TEST 1 — Clearance Profiles")
    env.set_status("TEST 1: Clearance Profiles", color=[0.3, 0.9, 0.3])

    cam = robot_state.get("camera", {})
    cam_quat = np.array(cam["quaternion_xyzw"]) if "quaternion_xyzw" in cam else None
    cam_pos = np.array(cam["position"]) if "position" in cam else None

    block_ids = [oid for oid, m in masks.items() if m.any() and "table" not in oid and "surface" not in oid]
    profiles = {}

    for oid in block_ids:
        target_mask = masks.get(oid)
        if target_mask is None or not target_mask.any():
            _warn(f"{oid}: mask empty — skipping")
            continue
        _info(f"{oid}: computing clearance profile ({target_mask.sum()} mask pixels, "
              f"{len([k for k,v in masks.items() if k != oid and v.any()])} neighbours)...")
        other_masks = {k: v for k, v in masks.items() if k != oid}
        profile = compute_clearance_profile(
            target_mask=target_mask,
            depth_frame=depth,
            camera_intrinsics=intrinsics,
            all_masks=other_masks,
            gripper=GripperGeometry.from_urdf(
                PROJECT_ROOT / "src" / "kinematics" / "sim" / "urdfs"
                / "xarm7_camera" / "xarm7.urdf"
            ),
            camera_quaternion_xyzw=cam_quat,
            cam_position=cam_pos,
        )
        profiles[oid] = profile
        n = len(profile.approach_corridors)
        grasp_ok = any(c.grasp_compatible for c in profile.approach_corridors)
        _info(f"{oid}: {n} corridors | top_clearance={profile.top_clearance*100:.1f} cm"
              f" | graspability={profile.graspability_score:.2f} | grasp_compatible={grasp_ok}")

    # --- Figure: 3 rows × N cols ---
    # Row 0: camera image with mask overlay
    # Row 1: 3D ray visualisation — object point cloud + rays + collision markers
    # Row 2: bar chart of scalar clearance values
    # GridSpec is required because row 1 uses 3D axes (projection='3d') which
    # cannot be mixed with 2D axes via plt.subplots().
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
    from matplotlib.gridspec import GridSpec

    n_cols = len(block_ids)
    fig = plt.figure(figsize=(5 * n_cols, 14))
    fig.suptitle("TEST 1 — Clearance Profiles", fontsize=12, fontweight="bold")
    gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1, 1.6, 1], hspace=0.4, wspace=0.35)

    for col, oid in enumerate(block_ids):
        profile = profiles.get(oid)
        mask = masks.get(oid, np.zeros(depth.shape, dtype=bool))
        ax_img  = fig.add_subplot(gs[0, col])
        ax_rays = fig.add_subplot(gs[1, col], projection="3d")
        ax_bar  = fig.add_subplot(gs[2, col])

        # ── Row 0: mask overlay ──────────────────────────────────────────────
        _mask_overlay(ax_img, color, {oid: mask}, title=oid)

        # ── Row 1: 3D corridor visualisation ────────────────────────────────
        # Each corridor is drawn as a full rectangular box (the actual OBB
        # swept volume) at low opacity, coloured by grasp compatibility:
        #   green (low opacity) = grasp_compatible
        #   red   (low opacity) = blocked / too narrow
        # Where an obstructor clips the corridor, its AABB face is drawn as
        # a filled quad at the intersection depth, showing exactly where the
        # block happens.
        if profile is not None and profile.centroid is not None:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from src.perception.clearance import CORRIDOR_LENGTH as _CL
            ocx, ocy, ocz = profile.centroid
            origin = np.array([ocx, ocy, ocz])

            # Downsampled obstacle cloud for context
            if profile.obstacle_pts is not None and len(profile.obstacle_pts) > 0:
                obs = profile.obstacle_pts
                step = max(1, len(obs) // 3000)
                obs_s = obs[::step]
                ax_rays.scatter(obs_s[:, 0], obs_s[:, 1], obs_s[:, 2],
                                s=1, c=[[0.75, 0.75, 0.75]], alpha=0.3,
                                depthshade=True)

            corridors_vis = profile.approach_corridors

            for corr in corridors_vis:
                color_c = "#2ca02c" if corr.grasp_compatible else "#d62728"
                alpha_box = 0.10 if corr.grasp_compatible else 0.18

                # Build the 8 corners of the corridor OBB.
                # The OBB center sits half-way along the corridor from start.
                d  = corr.direction
                ga = corr.grasp_axis
                ha = corr.height_axis
                hl = _CL / 2.0           # half-length along approach
                hw = corr.half_width     # along grasp_axis
                hh = corr.half_height    # along height_axis
                center = corr.corridor_start + d * hl

                corners = np.array([
                    center + s0*d*hl + s1*ga*hw + s2*ha*hh
                    for s0 in (+1, -1)
                    for s1 in (+1, -1)
                    for s2 in (+1, -1)
                ])  # (8, 3)

                # 6 faces of the OBB as quads (index into corners)
                face_indices = [
                    [0,1,3,2],  # +d face
                    [4,5,7,6],  # -d face
                    [0,1,5,4],  # +grasp face
                    [2,3,7,6],  # -grasp face
                    [0,2,6,4],  # +height face
                    [1,3,7,5],  # -height face
                ]
                faces = [[corners[i] for i in face] for face in face_indices]
                poly = Poly3DCollection(faces, alpha=alpha_box,
                                        facecolor=color_c, edgecolor=color_c,
                                        linewidth=0.3)
                ax_rays.add_collection3d(poly)

                # Centre-line shaft for readability
                tip = corr.corridor_start + d * _CL
                ax_rays.plot(
                    [corr.corridor_start[0], tip[0]],
                    [corr.corridor_start[1], tip[1]],
                    [corr.corridor_start[2], tip[2]],
                    color=color_c, linewidth=0.8,
                    alpha=0.7 if corr.grasp_compatible else 0.5,
                )

                # ── Collision markers ──────────────────────────────────────
                # For each obstructor AABB that clips this corridor, draw the
                # cross-section of the AABB face at the longitudinal entry
                # point — i.e. project the AABB onto the approach axis to find
                # the depth where it first enters the corridor, then draw a
                # filled quad at that depth clipped to corridor width/height.
                for nb_min, nb_max in corr.obstructor_aabbs:
                    nb_cw = (nb_min + nb_max) * 0.5
                    nb_hw = (nb_max - nb_min) * 0.5

                    # Entry depth: AABB near face along d, as offset from corridor_start
                    nb_proj_c = float((nb_cw - corr.corridor_start) @ d)
                    nb_proj_r = float(np.sum(nb_hw * np.abs(d)))
                    near_t = nb_proj_c - nb_proj_r
                    far_t  = nb_proj_c + nb_proj_r

                    if far_t <= 0.0 or near_t >= _CL:
                        continue

                    entry_t  = float(np.clip(near_t, 0.0, _CL))
                    entry_pt = corr.corridor_start + d * entry_t

                    # Cross-section quad: AABB extents along ga/ha as offsets
                    # from box_center, clipped to corridor half-extents.
                    bc_ga = float(center @ ga)
                    bc_ha = float(center @ ha)
                    nb_ga_c = float(nb_cw @ ga) - bc_ga
                    nb_ha_c = float(nb_cw @ ha) - bc_ha
                    nb_ga_r = float(np.sum(nb_hw * np.abs(ga)))
                    nb_ha_r = float(np.sum(nb_hw * np.abs(ha)))

                    q_min_ga = max(nb_ga_c - nb_ga_r, -hw)
                    q_max_ga = min(nb_ga_c + nb_ga_r,  hw)
                    q_min_ha = max(nb_ha_c - nb_ha_r, -hh)
                    q_max_ha = min(nb_ha_c + nb_ha_r,  hh)

                    if q_max_ga > q_min_ga and q_max_ha > q_min_ha:
                        # Build quad in world frame: entry_pt + offset*ga + offset*ha
                        ep_ga = float(entry_pt @ ga) - bc_ga  # entry_pt offset from bc
                        ep_ha = float(entry_pt @ ha) - bc_ha
                        def _qpt(og, oh, _ep=entry_pt, _ga=ga, _ha=ha,
                                 _bc_ga=bc_ga, _bc_ha=bc_ha):
                            cur_og = float(_ep @ _ga) - _bc_ga
                            cur_oh = float(_ep @ _ha) - _bc_ha
                            return _ep + (og - cur_og) * _ga + (oh - cur_oh) * _ha
                        quad = [
                            _qpt(q_min_ga, q_min_ha),
                            _qpt(q_max_ga, q_min_ha),
                            _qpt(q_max_ga, q_max_ha),
                            _qpt(q_min_ga, q_max_ha),
                        ]
                        col_face = Poly3DCollection(
                            [quad], alpha=0.65,
                            facecolor="#ff7f0e", edgecolor="#cc4400",
                            linewidth=1.2,
                        )
                        ax_rays.add_collection3d(col_face)

                        # X marker at the centre of the collision face
                        cx_col = np.mean(quad, axis=0)
                        cross_r = min(hw, hh) * 0.5
                        for ax_c in [ga, ha]:
                            ax_rays.plot(
                                [cx_col[0]-ax_c[0]*cross_r, cx_col[0]+ax_c[0]*cross_r],
                                [cx_col[1]-ax_c[1]*cross_r, cx_col[1]+ax_c[1]*cross_r],
                                [cx_col[2]-ax_c[2]*cross_r, cx_col[2]+ax_c[2]*cross_r],
                                color="#cc4400", linewidth=1.5,
                            )

            # Label the 5 named axis corridors
            named_dirs = {
                "+x": np.array([1.0,  0.0, 0.0]),
                "-x": np.array([-1.0, 0.0, 0.0]),
                "+y": np.array([0.0,  1.0, 0.0]),
                "-y": np.array([0.0, -1.0, 0.0]),
                "+z": np.array([0.0,  0.0, 1.0]),
            }
            corr_dirs = np.stack([c.direction for c in corridors_vis], axis=0)
            for axis, wd in named_dirs.items():
                best_i = int(np.argmax(corr_dirs @ wd))
                corr = corridors_vis[best_i]
                tip = corr.corridor_start + corr.direction * _CL
                ax_rays.text(
                    tip[0], tip[1], tip[2],
                    f"  {axis}\n  {corr.min_clearance*100:.0f}cm",
                    fontsize=5, color="navy",
                )

            # Centroid marker
            ax_rays.scatter(ocx, ocy, ocz, s=80, marker="*", color="black",
                            zorder=10, depthshade=False)

            from matplotlib.lines import Line2D as _L2D
            from matplotlib.patches import Patch as _Patch
            ax_rays.legend(
                handles=[
                    _Patch(facecolor="#2ca02c", alpha=0.5, label="grasp OK"),
                    _Patch(facecolor="#d62728", alpha=0.5, label="blocked"),
                    _Patch(facecolor="#ff7f0e", alpha=0.7, label="collision face"),
                ],
                fontsize=6, loc="upper left",
            )

            ax_rays.set_xlabel("X (m)", fontsize=7)
            ax_rays.set_ylabel("Y (m)", fontsize=7)
            ax_rays.set_zlabel("Z (m)", fontsize=7)  # type: ignore[attr-defined]
            n_ok = sum(1 for c in corridors_vis if c.grasp_compatible)
            ax_rays.set_title(f"{oid} — corridors ({n_ok}/{len(corridors_vis)} ok)", fontsize=8)
            ax_rays.tick_params(labelsize=6)
        else:
            ax_rays.text(0.5, 0.5, 0.5, "no corridor data", ha="center", va="center")
            ax_rays.set_title(oid)

        # ── Row 2: bar chart ─────────────────────────────────────────────────
        if profile is not None:
            labels = ["top\n(+z)", "fwd\n(+x)", "back\n(-x)", "left\n(+y)", "right\n(-y)"]
            values = [
                profile.top_clearance,
                profile.lateral_clearances.get("+x (fwd)", 0),
                profile.lateral_clearances.get("-x (back)", 0),
                profile.lateral_clearances.get("+y (left)", 0),
                profile.lateral_clearances.get("-y (right)", 0),
            ]
            bar_colors = ["steelblue" if v > 0.05 else "salmon" for v in values]
            ax_bar.bar(labels, [v * 100 for v in values], color=bar_colors)
            ax_bar.axhline(8 + 2 * 1.2, color="red", linestyle="--", linewidth=1,
                           label="min required (aperture + 2×finger)")
            ax_bar.set_ylabel("transverse clearance (cm)")
            ax_bar.set_title(
                f"score={profile.graspability_score:.2f}  "
                f"corridors={len(profile.approach_corridors)}\n"
                f"grasp_ok={any(c.grasp_compatible for c in profile.approach_corridors)}",
                fontsize=8,
            )
            ax_bar.legend(fontsize=7)
        else:
            ax_bar.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax_bar.transAxes)
            ax_bar.set_title(oid)

    # Show the matplotlib figure non-blocking so it stays visible alongside PyBullet
    if gui:
        plt.show(block=False)
        plt.pause(0.1)

    # --- Interactive corridor browser in PyBullet viewport ---
    # n = next corridor (single), a = show all, Enter = next object
    if gui:
        for oid, profile in profiles.items():
            if profile.centroid is None or not profile.approach_corridors:
                continue
            corridors = profile.approach_corridors
            idx = 0        # current single-corridor index
            all_mode = True  # True = show all corridors, False = show single

            def _redraw(current_idx: int) -> None:
                env.clear_debug_items()
                # All scene AABBs — cyan for current target, yellow for others
                target_aabb = (
                    [(profile.target_aabb_min, profile.target_aabb_max)]
                    if profile.target_aabb_min is not None else []
                )
                env.draw_aabbs(target_aabb, color=(0.0, 0.9, 0.9), line_width=3.0)
                env.draw_aabbs(profile.neighbor_aabbs, color=(0.9, 0.8, 0.1), line_width=2.0)
                if current_idx == -1:
                    env.draw_clearance_corridors(
                        centroid=profile.centroid,
                        corridors=corridors,
                    )
                    env.set_status(
                        f"{oid} — all {len(corridors)} corridors\n"
                        "n=next  a=all  Enter=next object"
                    )
                else:
                    corr = corridors[current_idx]
                    compat = "OK" if corr.grasp_compatible else "BLOCKED"
                    env.draw_clearance_corridors(
                        centroid=profile.centroid,
                        corridors=[corr],
                    )
                    env.set_status(
                        f"{oid} — corridor {current_idx + 1}/{len(corridors)}\n"
                        f"dir=({corr.direction[0]:.2f},{corr.direction[1]:.2f},{corr.direction[2]:.2f})\n"
                        f"clearance={corr.min_clearance*100:.1f}cm  {compat}\n"
                        "n=next  a=all  Enter=next object"
                    )

            _redraw(-1)  # start with all corridors shown

            n_ok  = sum(1 for c in corridors if c.grasp_compatible)
            n_blk = len(corridors) - n_ok
            _info(f"{oid}: PyBullet browser — {len(corridors)} corridors "
                  f"({n_ok} grasp_ok, {n_blk} blocked)")

            import pybullet as _pb
            _KEY_N     = ord('n')
            _KEY_A     = ord('a')
            _KEY_ENTER = 65309  # PyBullet special key for Return

            print(f"\n  {oid}: n=next corridor  a=all corridors  Enter=next object")
            while True:
                time.sleep(0.05)
                try:
                    keys = _pb.getKeyboardEvents(physicsClientId=env._client)
                except Exception:
                    break
                # KEY_WAS_TRIGGERED = 2
                triggered = {k for k, v in keys.items() if v & 2}
                if _KEY_ENTER in triggered or 13 in triggered:
                    break
                if _KEY_N in triggered:
                    idx = (idx + 1) % len(corridors)
                    all_mode = False
                    _redraw(idx)
                elif _KEY_A in triggered:
                    all_mode = True
                    _redraw(-1)

            env.clear_debug_items()

    _wait(gui, "Both views shown — press Enter for next test.")
    plt.close(fig)
    _ok("Clearance profile test complete")
    return True


# ---------------------------------------------------------------------------
# TEST 2 — Contact Graph
# ---------------------------------------------------------------------------

def test_contact_graph(env, color, depth, masks, detected, intrinsics, gui) -> bool:
    from matplotlib.lines import Line2D

    _section("TEST 2 — Contact Graph")
    env.set_status("TEST 2: Contact Graph", color=[0.3, 0.5, 0.9])

    graph = compute_contact_graph(
        objects=detected,
        obj_masks=masks,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        contact_threshold_m=0.05,
    )

    _info(f"Edges found: {len(graph.edges)}")
    for edge in graph.edges:
        _info(f"  {edge.obj_a} → {edge.obj_b}  type={edge.contact_type}"
              f"  consequence={edge.removal_consequence}"
              f"  area={edge.contact_region.area*1e4:.1f} cm²")

    _info("Support tree:")
    for oid, supported in graph.support_tree.items():
        if supported:
            _info(f"  {oid} supports {supported}")

    _info("Stability scores:")
    for oid, score in graph.stability_scores.items():
        _info(f"  {oid}: {score:.2f}")

    # --- Figure: two panels ---
    fig, (ax_scene, ax_graph) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TEST 2 — Contact Graph", fontsize=12, fontweight="bold")

    # Left: mask overlay highlighting all objects
    _mask_overlay(ax_scene, color, masks, title="Scene masks")

    # Right: schematic node-edge graph
    # Lay nodes out by approximate 3D position projected to (x, y)
    positions = {}
    for obj in detected:
        positions[obj.object_id] = (obj.position_3d[0], obj.position_3d[1])

    # Normalise to plot space
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_range = max(xs) - min(xs) or 1.0
        y_range = max(ys) - min(ys) or 1.0
        norm_pos = {
            oid: ((p[0] - min(xs)) / x_range, (p[1] - min(ys)) / y_range)
            for oid, p in positions.items()
        }
    else:
        norm_pos = {}

    edge_style = {
        "supporting": dict(color="green",  linewidth=2.5, linestyle="-"),
        "nested":     dict(color="purple", linewidth=1.5, linestyle="-."),
        "none":       dict(color="grey",   linewidth=0.8, linestyle=":"),
    }

    for edge in graph.edges:
        if edge.obj_a in norm_pos and edge.obj_b in norm_pos:
            xa, ya = norm_pos[edge.obj_a]
            xb, yb = norm_pos[edge.obj_b]
            style = edge_style.get(edge.contact_type, dict(color="black", linewidth=1))
            ax_graph.annotate(
                "", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="->", **style),
            )
            mid_x, mid_y = (xa + xb) / 2, (ya + yb) / 2
            ax_graph.text(mid_x, mid_y, edge.contact_type, fontsize=7,
                          ha="center", va="bottom", color=style["color"])

    for oid, (nx, ny) in norm_pos.items():
        c = OBJ_COLORS.get(oid, (0.5, 0.5, 0.5))
        score = graph.stability_scores.get(oid, 0.0)
        ax_graph.scatter(nx, ny, s=300, c=[c], zorder=5)
        ax_graph.text(nx, ny + 0.06, f"{oid}\n(stab={score:.2f})",
                      ha="center", fontsize=7)

    legend_elems = [
        Line2D([0], [0], color=v["color"], linewidth=v["linewidth"],
               linestyle=v["linestyle"], label=k)
        for k, v in edge_style.items()
    ]
    ax_graph.legend(handles=legend_elems, fontsize=7, loc="lower right")
    ax_graph.set_xlim(-0.2, 1.2)
    ax_graph.set_ylim(-0.2, 1.2)
    ax_graph.set_title(f"{len(graph.edges)} contact edges", fontsize=9)
    ax_graph.axis("off")

    plt.tight_layout()
    _show_and_wait(fig, gui, "Contact graph shown — press Enter for next test.")
    _ok("Contact graph test complete")
    return graph


# ---------------------------------------------------------------------------
# TEST 3 — Occlusion Map
# ---------------------------------------------------------------------------

def test_occlusion(env, scene_objects, intrinsics, tracker, gui) -> bool:

    _section("TEST 3 — Occlusion Map (multi-viewpoint)")
    env.set_status("TEST 3: Occlusion Map", color=[0.9, 0.5, 0.1])

    observations: List[ObservationRecord] = []
    captured_colors: List[np.ndarray] = []
    captured_labels: List[str] = []

    for joints, label in [
        (VIEWPOINT_1, "front"),
        (VIEWPOINT_2, "right"),
        (VIEWPOINT_3, "left"),
    ]:
        env.set_status(f"T3 Occlusion: capturing {label}", color=[0.9, 0.5, 0.1])
        env.set_robot_joints(joints)
        time.sleep(0.4)
        color_vp, depth_vp, _ = env.capture_camera_frame()
        if depth_vp is None:
            _warn(f"Viewpoint {label}: no depth — skipping")
            continue
        robot_state = env.get_robot_state()
        cam_pose = CameraPose.from_robot_state(robot_state)
        masks_vp = _detect_masks(tracker, color_vp, depth_vp, intrinsics, robot_state, scene_objects)
        _info(f"Viewpoint {label}: non-empty={[k for k,v in masks_vp.items() if v.any()]}")
        observations.append(ObservationRecord(
            depth_frame=depth_vp,
            camera_intrinsics=intrinsics,
            camera_pose=cam_pose,
            obj_masks=masks_vp,
        ))
        captured_colors.append(color_vp)
        captured_labels.append(label)

    env.set_robot_joints(CAMERA_AIM_JOINTS)
    time.sleep(0.3)

    if not observations:
        _warn("No observations captured — occlusion test skipped")
        return False

    occ_map = compute_occlusion_map(
        observations=observations,
        object_ids=[o["object_id"] for o in scene_objects],
    )

    for oid, vis in occ_map.per_object_visibility.items():
        _info(f"  {oid}: visible={vis.visible_fraction:.0%}"
              f"  occluded_by={vis.occluding_objects}")

    # --- Figure ---
    n_vp = len(captured_colors)
    all_ids = [o["object_id"] for o in scene_objects]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("TEST 3 — Occlusion Map", fontsize=12, fontweight="bold")

    # Top row: one panel per viewpoint with mask overlays
    for i, (col_img, label) in enumerate(zip(captured_colors, captured_labels)):
        ax = fig.add_subplot(2, max(n_vp, len(all_ids)), i + 1)
        masks_vp = observations[i].obj_masks
        _mask_overlay(ax, col_img, masks_vp, title=f"Viewpoint: {label}")

    # Bottom row: per-object visibility bar chart
    ax_vis = fig.add_subplot(2, 1, 2)
    oids = list(occ_map.per_object_visibility.keys())
    fracs = [occ_map.per_object_visibility[o].visible_fraction for o in oids]
    bar_colors = [OBJ_COLORS.get(o, (0.5, 0.5, 0.5)) for o in oids]
    bars = ax_vis.bar(oids, [f * 100 for f in fracs], color=bar_colors)
    ax_vis.set_ylabel("visible fraction (%)")
    ax_vis.set_ylim(0, 105)
    ax_vis.set_title("Per-object visibility across all viewpoints", fontsize=9)
    for bar, frac, oid in zip(bars, fracs, oids):
        occluders = occ_map.per_object_visibility[oid].occluding_objects
        label_txt = f"{frac:.0%}"
        if occluders:
            label_txt += f"\noccluded by:\n{', '.join(occluders)}"
        ax_vis.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, label_txt,
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Occlusion map shown — press Enter for next test.")
    _ok("Occlusion map test complete")
    return True


# ---------------------------------------------------------------------------
# TEST 4 — Surface Free-Space Map
# ---------------------------------------------------------------------------

def test_surface_map(env, color, depth, masks, detected, intrinsics, graph, gui) -> bool:
    from matplotlib.colors import ListedColormap

    _section("TEST 4 — Surface Free-Space Map")
    env.set_status("TEST 4: Surface Free-Space Map", color=[0.8, 0.3, 0.8])

    surface_maps = compute_surface_maps(
        objects=detected,
        obj_masks=masks,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        contact_graph=graph,
        resolution_m=0.01,
    )

    if not surface_maps:
        _warn("No surface maps computed")
        return False

    for surface_id, smap in surface_maps.items():
        _info(f"{surface_id}: congestion={smap.congestion_score:.2f}"
              f"  free_regions={len(smap.free_space_regions)}")
        for i, region in enumerate(smap.free_space_regions):
            _info(f"  region {i}: area={region.area*1e4:.1f} cm²"
                  f"  radius={region.max_inscribed_circle*100:.1f} cm"
                  f"  neighbors={region.neighboring_objects}")

    # --- Figure: one column per surface ---
    n_surfaces = len(surface_maps)
    fig, axes = plt.subplots(1, n_surfaces + 1,
                             figsize=(5 * (n_surfaces + 1), 5))
    if n_surfaces == 0:
        plt.close(fig)
        return False
    if not hasattr(axes, "__len__"):
        axes = [axes]

    fig.suptitle("TEST 4 — Surface Free-Space Map", fontsize=12, fontweight="bold")

    # First panel: scene overview with surface mask
    surface_mask_union = np.zeros(depth.shape, dtype=bool)
    for sid in surface_maps:
        m = masks.get(sid)
        if m is not None:
            surface_mask_union |= m
    _mask_overlay(axes[0], color,
                  {sid: masks.get(sid, np.zeros(depth.shape, dtype=bool))
                   for sid in surface_maps},
                  title="Surface masks in scene")

    for col, (surface_id, smap) in enumerate(surface_maps.items(), start=1):
        ax = axes[col] if col < len(axes) else axes[-1]

        if smap.occupancy_grid is not None:
            grid = smap.occupancy_grid.astype(float)
            # 0=free, 1=occupied; colour: free=lightgreen, occupied=salmon
            cmap = ListedColormap(["#b8f5b8", "#f5b8b8"])
            ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=1)

            # Overlay free-space region boundaries
            for i, region in enumerate(smap.free_space_regions):
                if hasattr(region, "pixel_mask") and region.pixel_mask is not None:
                    ys, xs = np.where(region.pixel_mask)
                    if len(xs):
                        ax.scatter(xs, ys, s=1, c="royalblue", alpha=0.4)
                # Mark inscribed circle centre if available
                if hasattr(region, "center_uv") and region.center_uv is not None:
                    cx, cy = region.center_uv
                    r_px = region.max_inscribed_circle / (smap.resolution_m or 0.01)
                    circle = plt.Circle((cx, cy), r_px, color="blue",
                                       fill=False, linewidth=1.5)
                    ax.add_patch(circle)
                    ax.text(cx, cy, f"r={region.max_inscribed_circle*100:.1f}cm",
                            ha="center", va="center", fontsize=6, color="navy")

            legend = [
                mpatches.Patch(color="#b8f5b8", label="free"),
                mpatches.Patch(color="#f5b8b8", label="occupied"),
            ]
            ax.legend(handles=legend, fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no grid\n(mask empty?)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        ax.set_title(
            f"{surface_id}\ncongestion={smap.congestion_score:.0%}"
            f"  regions={len(smap.free_space_regions)}",
            fontsize=9,
        )
        ax.set_xlabel("U (px)")
        ax.set_ylabel("V (px)")

    plt.tight_layout()
    _show_and_wait(fig, gui, "Surface map shown — press Enter for next test.")
    _ok("Surface map test complete")
    return True


# ---------------------------------------------------------------------------
# TEST 5 — Post-Manipulation Update
# ---------------------------------------------------------------------------

def test_post_manipulation_update(env, scene_objects, intrinsics, tracker, gui) -> bool:

    _section("TEST 5 — Post-Manipulation Geometry Update")
    env.set_status("TEST 5: Post-Manipulation Update", color=[0.9, 0.7, 0.1])

    # Baseline
    color_before, depth_before, _ = env.capture_camera_frame()
    robot_state = env.get_robot_state()
    masks_before = _detect_masks(tracker, color_before, depth_before, intrinsics, robot_state, scene_objects)
    detected = _make_detected_objects(scene_objects)

    graph_before = compute_contact_graph(
        objects=detected, obj_masks=masks_before,
        depth_frame=depth_before, camera_intrinsics=intrinsics,
        contact_threshold_m=0.05,
    )
    n_before = len(graph_before.edges)
    _info(f"Before: {n_before} contact edges")

    # Simulate pick — move the stacked block off the target to clear it
    _info(f"Moving {POSTMANIP_OBJECT} off the stack...")
    env.set_status(f"T5: Simulating pick of {POSTMANIP_OBJECT}", color=[0.9, 0.7, 0.1])
    env.move_object(POSTMANIP_OBJECT, POSTMANIP_DST_POS)
    time.sleep(0.4)

    # Post-manipulation snapshot
    color_after, depth_after, _ = env.capture_camera_frame()
    robot_state_after = env.get_robot_state()
    scene_after = [o.copy() for o in scene_objects]
    for o in scene_after:
        if o["object_id"] == POSTMANIP_OBJECT:
            o["position_3d"] = POSTMANIP_DST_POS
    masks_after = _detect_masks(tracker, color_after, depth_after, intrinsics, robot_state_after, scene_after)
    detected_after = _make_detected_objects(scene_after)

    graph_after = compute_contact_graph(
        objects=detected_after, obj_masks=masks_after,
        depth_frame=depth_after, camera_intrinsics=intrinsics,
        contact_threshold_m=0.05,
    )
    n_after = len(graph_after.edges)
    _info(f"After: {n_after} contact edges")

    # --- Figure: before/after side by side ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("TEST 5 — Post-Manipulation Update", fontsize=12, fontweight="bold")

    _mask_overlay(axes[0, 0], color_before, masks_before, title="Before: scene")
    _mask_overlay(axes[0, 1], color_after,  masks_after,  title="After: scene")

    # Depth diff
    depth_diff = depth_after - depth_before
    vmax = np.nanpercentile(np.abs(depth_diff[np.isfinite(depth_diff)]), 95) or 0.1
    im = axes[1, 0].imshow(depth_diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title("Depth diff (after − before)", fontsize=9)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], label="m", fraction=0.046, pad=0.04)

    # Edge count comparison
    axes[1, 1].bar(["before", "after"], [n_before, n_after],
                   color=["steelblue", "salmon"])
    axes[1, 1].set_ylabel("contact edges")
    axes[1, 1].set_title(
        f"Contact edges: {n_before} → {n_after}\n"
        f"{'edges removed as expected' if n_after < n_before else 'edge count unchanged'}",
        fontsize=9,
    )
    for i, val in enumerate([n_before, n_after]):
        axes[1, 1].text(i, val + 0.05, str(val), ha="center", fontsize=11)

    plt.tight_layout()
    _show_and_wait(fig, gui, "Post-manipulation result shown — press Enter to finish.")

    # Restore
    env.move_object(POSTMANIP_OBJECT, POSTMANIP_SRC_POS)
    _ok("Post-manipulation geometry update test complete")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Perception component integration test")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless: skip PyBullet GUI and matplotlib figures")
    parser.add_argument("--tests", nargs="*",
                        choices=["clearance","contact","occlusion","surface","postmanip","all"],
                        default=["all"])
    args = parser.parse_args()
    gui = not args.no_gui

    run_all = "all" in (args.tests or ["all"])
    run = {
        "clearance": run_all or "clearance"  in args.tests,
        "contact":   run_all or "contact"    in args.tests,
        "occlusion": run_all or "occlusion"  in args.tests,
        "surface":   run_all or "surface"    in args.tests,
        "postmanip": run_all or "postmanip"  in args.tests,
    }

    if not gui:
        plt.ioff()

    print(_DIVIDER)
    print("  Perception Component Integration Test — xArm7 Simulation")
    print(_DIVIDER)

    env = SceneEnvironment()

    if not gui:
        try:
            import pybullet as p  # type: ignore[import-untyped]
            import pybullet_data  # type: ignore[import-untyped]
            def headless_start():
                env._client = p.connect(p.DIRECT)
                c = env._client
                p.setGravity(0, 0, -9.81, physicsClientId=c)
                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
                p.loadURDF("plane.urdf", physicsClientId=c)
                from src.kinematics.sim.scene_environment import _DEFAULT_URDF
                if _DEFAULT_URDF.exists():
                    env._robot_id = p.loadURDF(
                        str(_DEFAULT_URDF), basePosition=[0,0,0],
                        baseOrientation=[0,0,0,1], useFixedBase=True, physicsClientId=c,
                    )
                    env._build_joint_map()
                    env.set_robot_joints(env.initial_joints)
                env._step_running = False
            env.start = headless_start
        except ImportError:
            pass

    env.start()
    env.add_scene_objects(SCENE_OBJECTS)
    env.set_robot_joints(CAMERA_AIM_JOINTS)
    time.sleep(0.5)

    color, depth, intrinsics = env.capture_camera_frame()
    if depth is None:
        log.error("No depth frame from simulation")
        env.stop()
        sys.exit(1)

    # --- Load GSAM2 tracker ---
    import os, torch
    _sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not _sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        env.stop()
        sys.exit(1)
    _section("Loading GSAM2 (GroundingDINO + SAM2 + OpenAI tagger)...")
    tracker = GSAM2ObjectTracker(
        sam2_ckpt_path=_sam2_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_clearances=False,
        compute_contacts=False,
        compute_occlusion=False,
        compute_surface_maps=False,
    )
    _ok("GSAM2 loaded")

    _info(f"Baseline frame: depth [{depth.min():.3f}, {depth.max():.3f}] m  shape={depth.shape}")
    robot_state = env.get_robot_state()
    masks = _detect_masks(tracker, color, depth, intrinsics, robot_state, SCENE_OBJECTS)
    detected = _make_detected_objects(SCENE_OBJECTS)
    nonempty = [k for k, v in masks.items() if v.any()]
    _info(f"Non-empty masks: {nonempty}")
    if not nonempty:
        _warn("GSAM2 found no objects in baseline frame.")

    results: Dict[str, bool] = {}
    graph = None  # shared between contact + surface tests

    if run["clearance"]:
        try:
            results["clearance"] = test_clearance(env, color, depth, masks, intrinsics, robot_state, gui)
        except Exception as e:
            log.exception("Clearance test failed")
            results["clearance"] = False
            _warn(f"Clearance FAILED: {e}")

    if run["contact"] or run["surface"]:
        try:
            result = test_contact_graph(env, color, depth, masks, detected, intrinsics, gui)
            if result is not False:
                graph = result
                results["contact"] = True
            else:
                results["contact"] = False
        except Exception as e:
            log.exception("Contact graph test failed")
            results["contact"] = False
            _warn(f"Contact FAILED: {e}")

    if run["occlusion"]:
        try:
            results["occlusion"] = test_occlusion(env, SCENE_OBJECTS, intrinsics, tracker, gui)
        except Exception as e:
            log.exception("Occlusion test failed")
            results["occlusion"] = False
            _warn(f"Occlusion FAILED: {e}")
        finally:
            env.set_robot_joints(CAMERA_AIM_JOINTS)
            time.sleep(0.3)
            color, depth, intrinsics = env.capture_camera_frame()
            robot_state = env.get_robot_state()
            masks = _detect_masks(tracker, color, depth, intrinsics, robot_state, SCENE_OBJECTS)

    if run["surface"] and graph is not None:
        try:
            results["surface"] = test_surface_map(
                env, color, depth, masks, detected, intrinsics, graph, gui
            )
        except Exception as e:
            log.exception("Surface map test failed")
            results["surface"] = False
            _warn(f"Surface FAILED: {e}")

    if run["postmanip"]:
        try:
            results["postmanip"] = test_post_manipulation_update(
                env, SCENE_OBJECTS, intrinsics, tracker, gui
            )
        except Exception as e:
            log.exception("Post-manipulation test failed")
            results["postmanip"] = False
            _warn(f"Post-manip FAILED: {e}")

    # Summary
    _section("SUMMARY")
    all_passed = True
    for name, passed in results.items():
        sym = "✓" if passed else "✗"
        print(f"  {sym}  {name:20s} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    env.set_status(
        "Tests complete\n" + "\n".join(f"{'✓' if p else '✗'} {n}" for n, p in results.items()),
        color=[0.3, 0.9, 0.3] if all_passed else [0.9, 0.3, 0.3],
    )

    print()
    if all_passed:
        print("  All tests passed.")
    else:
        print(f"  Failed: {[n for n, p in results.items() if not p]}")

    if gui:
        _wait(gui, "All done — press Enter to close simulation.")

    env.stop()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
