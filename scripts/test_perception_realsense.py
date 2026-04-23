"""
Perception component integration test — live RealSense RGB-D

Exercises all perception components against live RGB-D frames from a
RealSense D435:

  TEST 1 — Clearance Profiles
  TEST 2 — Contact Graph
  TEST 3 — Occlusion Map  (accumulates across multiple captured frames)
  TEST 4 — Surface Free-Space Map

GSAM2 (GroundingDINO + SAM2) handles detection and segmentation.
No ground-truth masks or scene description required.

Run:
    uv run scripts/test_perception_realsense.py
    uv run scripts/test_perception_realsense.py --no-gui
    uv run scripts/test_perception_realsense.py --tests clearance contact
    uv run scripts/test_perception_realsense.py --frames 8
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
from src.perception.clearance import GripperGeometry, compute_clearance_profile
from src.perception.contact_graph import compute_contact_graph
from src.perception.occlusion import CameraPose, ObservationRecord, compute_occlusion_map
from src.perception.surface_map import compute_surface_maps
from src.perception.object_registry import DetectedObject, DetectedObjectRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("perception_realsense")

_DIVIDER = "=" * 70

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
    cmap = plt.get_cmap("tab10")
    for idx, (oid, mask) in enumerate(masks.items()):
        if not mask.any():
            continue
        c = cmap(idx % 10)
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask, :3] = c[:3]
        rgba[mask, 3]  = 0.45
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _legend_patches(masks: Dict[str, np.ndarray]) -> List[mpatches.Patch]:
    cmap = plt.get_cmap("tab10")
    return [
        mpatches.Patch(color=cmap(i % 10), label=oid)
        for i, oid in enumerate(masks)
        if masks[oid].any()
    ]


def _draw_aabbs(ax, intrinsics, cam_position, cam_rotation,
               object_aabbs: dict, alpha: float = 0.7) -> None:
    """Project world-frame AABBs as wireframe boxes onto the image.

    Each AABB is drawn as the 12 edges of its bounding box, using the same
    tab10 colour scheme as the mask overlays so each object is identifiable.
    Edges that project behind the camera are skipped.
    """
    if cam_position is None or cam_rotation is None or not object_aabbs:
        return

    fx = float(getattr(intrinsics, 'fx', 0))
    fy = float(getattr(intrinsics, 'fy', 0))
    cx = float(getattr(intrinsics, 'ppx', getattr(intrinsics, 'cx', 0)))
    cy = float(getattr(intrinsics, 'ppy', getattr(intrinsics, 'cy', 0)))
    if fx == 0 or fy == 0:
        return

    R_cw = cam_rotation.as_matrix().T
    t_wc = np.asarray(cam_position, float)

    def w2p(p):
        p_cam = R_cw @ (np.asarray(p, float) - t_wc)
        if p_cam[2] <= 0:
            return None
        return fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy

    # The 12 edges of a box defined by (mn, mx): pairs of corner indices
    # Corners: 8 combinations of (x_min/max, y_min/max, z_min/max)
    _EDGES = [
        (0,1),(1,3),(3,2),(2,0),  # bottom face
        (4,5),(5,7),(7,6),(6,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # vertical edges
    ]

    cmap = plt.get_cmap("tab10")
    for idx, (oid, (mn, mx)) in enumerate(object_aabbs.items()):
        color = cmap(idx % 10)
        # Build 8 corners
        xs = [mn[0], mx[0]]
        ys = [mn[1], mx[1]]
        zs = [mn[2], mx[2]]
        corners = [np.array([xs[i&1], ys[(i>>1)&1], zs[(i>>2)&1]])
                   for i in range(8)]
        for i, j in _EDGES:
            p0 = w2p(corners[i])
            p1 = w2p(corners[j])
            if p0 is None or p1 is None:
                continue
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=color, linewidth=1.2, alpha=alpha)
        # Label at projected centroid
        centroid_px = w2p((mn + mx) / 2)
        if centroid_px is not None:
            ax.text(centroid_px[0], centroid_px[1], oid,
                    color=color, fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, ec="none"))


def _draw_world_axes(ax, intrinsics, cam_position, cam_rotation,
                     origin_world=None, axis_len: float = 0.10) -> None:
    """Project world-frame XYZ unit axes onto the image and draw them.

    Draws a small RGB triad at `origin_world` (defaults to the camera
    position projected forward by 0.5 m so it lands on the table surface).
    Provides immediate visual confirmation that the camera-to-world transform
    is pointing in the right direction.

    Args:
        ax: matplotlib Axes showing the colour image.
        intrinsics: CameraIntrinsics with fx, fy, cx/ppx, cy/ppy.
        cam_position: (3,) camera origin in world frame.
        cam_rotation: scipy Rotation, camera-to-world.
        origin_world: (3,) point in world frame to draw axes at.
            If None, a point 0.5 m in front of the camera is used.
        axis_len: length of each axis arrow in metres.
    """
    if cam_position is None or cam_rotation is None:
        return

    fx = float(getattr(intrinsics, 'fx', 0))
    fy = float(getattr(intrinsics, 'fy', 0))
    cx = float(getattr(intrinsics, 'ppx', getattr(intrinsics, 'cx', 0)))
    cy = float(getattr(intrinsics, 'ppy', getattr(intrinsics, 'cy', 0)))
    if fx == 0 or fy == 0:
        return

    # cam_rotation rotates camera→world; inverse rotates world→camera
    R_wc = cam_rotation.as_matrix()          # world ← camera
    R_cw = R_wc.T                            # camera ← world
    t_wc = np.asarray(cam_position, float)   # camera origin in world

    def world_to_pixel(p_world):
        """Project a world-frame point to pixel (u, v). Returns None if behind camera."""
        p_cam = R_cw @ (np.asarray(p_world, float) - t_wc)
        if p_cam[2] <= 0:
            return None
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return float(u), float(v)

    # Default origin: 0.5 m along camera +Z in world frame
    if origin_world is None:
        cam_z_world = R_wc[:, 2]   # third column = camera Z in world
        origin_world = t_wc + 0.5 * cam_z_world

    o_px = world_to_pixel(origin_world)
    if o_px is None:
        return

    # Project the robot base frame's world axes into the image.
    # These are fixed unit vectors in world/base frame — the FK transform
    # (cam_position, cam_rotation) maps them into pixel space so we can
    # verify the extrinsic is correct.
    axes_def = [
        (np.array([1, 0, 0]), "r", "X"),
        (np.array([0, 1, 0]), "g", "Y"),
        (np.array([0, 0, 1]), "b", "Z"),
    ]
    for world_dir, color, label in axes_def:
        tip_px = world_to_pixel(origin_world + axis_len * world_dir)
        if tip_px is None:
            continue
        ax.annotate(
            "", xy=tip_px, xytext=o_px,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0),
        )
        # Label at tip, offset slightly so it doesn't overlap the arrowhead
        dx = tip_px[0] - o_px[0]
        dy = tip_px[1] - o_px[1]
        norm = max((dx**2 + dy**2) ** 0.5, 1e-6)
        ax.text(tip_px[0] + 8 * dx / norm, tip_px[1] + 8 * dy / norm,
                label, color=color, fontsize=9, fontweight="bold",
                ha="center", va="center")

# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _capture_frame(camera: RealSenseCamera, tracker) -> tuple:
    """Capture one aligned frame and run detect_objects. Returns (color, depth, masks, detected)."""
    color, depth = camera.get_aligned_frames()
    asyncio.run(tracker.detect_objects(
        color_frame=color,
        depth_frame=depth,
        camera_intrinsics=camera.get_camera_intrinsics(),
    ))
    masks    = dict(tracker._last_masks)
    detected = list(tracker.registry.get_all_objects())
    return color, depth, masks, detected


# ---------------------------------------------------------------------------
# TEST 1 — Clearance Profiles
# ---------------------------------------------------------------------------

def test_clearance(camera: RealSenseCamera, tracker, gui: bool,
                   cam_position=None, cam_rotation=None) -> bool:
    from matplotlib.gridspec import GridSpec

    _section("TEST 1 — Clearance Profiles")

    intrinsics = camera.get_camera_intrinsics()
    color, depth, masks, detected = _capture_frame(camera, tracker)

    _info(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}] m")
    _info(f"Detected: {[o.object_id for o in detected]}")

    # All objects with a non-empty mask get a clearance profile
    object_ids = [oid for oid, m in masks.items() if m.any()]
    if not object_ids:
        _warn("No objects detected — skipping clearance test")
        return False

    gripper = GripperGeometry()
    profiles = {}

    for oid in object_ids:
        target_mask = masks[oid]
        other_masks = {k: v for k, v in masks.items() if k != oid}
        _info(f"{oid}: computing clearance profile "
              f"({target_mask.sum()} px, {len([k for k,v in other_masks.items() if v.any()])} neighbours)...")
        try:
            profile = compute_clearance_profile(
                target_mask=target_mask,
                depth_frame=depth,
                camera_intrinsics=intrinsics,
                all_masks=other_masks,
                gripper=gripper,
                cam_position=cam_position,
                camera_quaternion_xyzw=cam_rotation.as_quat() if cam_rotation is not None else None,
            )
            profiles[oid] = profile
            n_ok  = sum(1 for c in profile.approach_corridors if c.grasp_compatible)
            n_blk = len(profile.approach_corridors) - n_ok
            _info(f"  {oid}: {len(profile.approach_corridors)} corridors "
                  f"({n_ok} ok, {n_blk} blocked)  "
                  f"graspability={profile.graspability_score:.2f}  "
                  f"top_clearance={profile.top_clearance*100:.1f} cm")
        except Exception as exc:
            _warn(f"{oid}: clearance failed — {exc}")

    if not profiles:
        _warn("No clearance profiles computed")
        return False

    if not gui:
        _ok("Clearance test complete (headless)")
        return True

    # Two rows: mask image on top, polar clearance rose on bottom
    n_cols = len(profiles)
    fig = plt.figure(figsize=(4 * n_cols, 8))
    fig.suptitle("TEST 1 — Clearance Profiles", fontsize=12, fontweight="bold")
    gs = GridSpec(2, n_cols, figure=fig, height_ratios=[1, 1.2], hspace=0.45, wspace=0.35)

    for col, (oid, profile) in enumerate(profiles.items()):
        mask = masks.get(oid, np.zeros(depth.shape, dtype=bool))
        ax_img   = fig.add_subplot(gs[0, col])
        ax_polar = fig.add_subplot(gs[1, col], projection="polar")

        _mask_overlay(ax_img, color, {oid: mask}, title=oid)

        # ---- Polar clearance rose ----
        # Each corridor is a spoke at the angle of its XZ projection (camera
        # frame: X=right, Z=depth-into-scene, Y=down).  Spoke length = clearance
        # normalised to corridor full-width.  Green = compatible, red = blocked.
        if profile.approach_corridors:
            for corridor in profile.approach_corridors:
                d = corridor.direction
                # angle in XZ plane (lateral vs depth) — most informative for
                # a front-facing camera
                theta = float(np.arctan2(d[0], d[2]))
                # normalise clearance to [0, 1] relative to corridor full-width
                full_w = corridor.half_width * 2.0 if corridor.half_width > 0 else 0.125
                r = min(corridor.min_clearance / max(full_w, 1e-6), 1.0)
                color_c = "#2ca02c" if corridor.grasp_compatible else "#d62728"
                alpha   = 0.8 if corridor.grasp_compatible else 0.4
                ax_polar.bar(theta, r, width=0.30, color=color_c,
                             alpha=alpha, align="center", edgecolor="none")

            # Unit circle = full corridor width reference
            thetas = np.linspace(0, 2 * np.pi, 200)
            ax_polar.plot(thetas, np.ones_like(thetas), "k--", linewidth=0.6, alpha=0.4)

            n_ok = sum(1 for c in profile.approach_corridors if c.grasp_compatible)
            blocked_by = set()
            for c in profile.approach_corridors:
                if not c.grasp_compatible:
                    blocked_by.update(c.obstructing_objects)

            title_lines = [
                f"graspability={profile.graspability_score:.2f}  "
                f"{n_ok}/{len(profile.approach_corridors)} clear",
            ]
            if blocked_by:
                title_lines.append(f"blocked by: {', '.join(sorted(blocked_by)[:3])}")
            ax_polar.set_title("\n".join(title_lines), fontsize=7, pad=6)
        else:
            ax_polar.text(0, 0, "no corridors", ha="center", va="center", fontsize=9)

        ax_polar.set_yticklabels([])
        ax_polar.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax_polar.set_xticklabels(["+Z (depth)", "+X (right)", "-Z", "-X (left)"], fontsize=6)

    _show_and_wait(fig, gui, "Clearance profiles shown — press Enter for next test.")
    _ok("Clearance test complete")
    return True


# ---------------------------------------------------------------------------
# TEST 2 — Contact Graph
# ---------------------------------------------------------------------------

def test_contact_graph(camera: RealSenseCamera, tracker, gui: bool,
                       cam_position=None, cam_rotation=None, llm_client=None,
                       llm_mode: str = "nl"):
    from matplotlib.lines import Line2D

    _section("TEST 2 — Contact Graph")

    intrinsics = camera.get_camera_intrinsics()
    color, depth, masks, detected = _capture_frame(camera, tracker)

    if len(detected) < 2:
        _warn(f"Only {len(detected)} object(s) detected — need ≥2 for contact graph")
        return False

    llm_debug_images: List[bytes] = []
    graph = compute_contact_graph(
        objects=detected,
        obj_masks=masks,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        contact_threshold_m=0.05,
        camera_position=cam_position,
        camera_rotation=cam_rotation,
        color_image=color,
        llm_client=llm_client,
        llm_mode=llm_mode,
        llm_debug_image_out=llm_debug_images if llm_client is not None else None,
    )

    if gui and llm_debug_images:
        import io as _io
        from PIL import Image as _PILImage
        png_arr = np.array(_PILImage.open(_io.BytesIO(llm_debug_images[0])))
        fig_llm, ax_llm = plt.subplots(1, 1, figsize=(10, 7))
        fig_llm.suptitle("LLM input — annotated scene image", fontsize=11, fontweight="bold")
        ax_llm.imshow(png_arr)
        ax_llm.axis("off")
        plt.show(block=False)
        plt.pause(0.1)

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

    if not gui:
        _ok("Contact graph test complete (headless)")
        return graph

    fig, (ax_scene, ax_graph) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TEST 2 — Contact Graph", fontsize=12, fontweight="bold")

    _mask_overlay(ax_scene, color, masks, title="Scene masks")
    _draw_aabbs(ax_scene, intrinsics, cam_position, cam_rotation, graph.object_aabbs)
    _draw_world_axes(ax_scene, intrinsics, cam_position, cam_rotation)
    patches = _legend_patches(masks)
    if patches:
        ax_scene.legend(handles=patches, loc="lower right", fontsize=7)

    # Node layout: use world-frame AABB midpoints if available (X lateral, Z height)
    # giving an intuitive side-view where stacked objects appear above their supports.
    # Fall back to camera-frame position_3d XZ if no AABBs.
    positions = {}
    if graph.object_aabbs and cam_position is not None:
        for oid, (mn, mx) in graph.object_aabbs.items():
            mid = (mn + mx) / 2.0
            positions[oid] = (float(mid[0]), float(mid[2]))  # world X, world Z
    else:
        for obj in detected:
            if obj.position_3d is not None:
                positions[obj.object_id] = (float(obj.position_3d[0]), float(obj.position_3d[2]))

    if positions:
        xs = [p[0] for p in positions.values()]
        zs = [p[1] for p in positions.values()]
        x_range = max(xs) - min(xs) or 1.0
        z_range = max(zs) - min(zs) or 1.0
        norm_pos = {
            oid: ((p[0] - min(xs)) / x_range, (p[1] - min(zs)) / z_range)
            for oid, p in positions.items()
        }
    else:
        norm_pos = {}

    edge_style = {
        "supporting": dict(color="green",  linewidth=2.5, linestyle="-"),
        "nested":     dict(color="purple", linewidth=1.5, linestyle="-."),
        "none":       dict(color="grey",   linewidth=0.8, linestyle=":"),
    }
    cmap = plt.get_cmap("tab10")
    for edge in graph.edges:
        if edge.obj_a in norm_pos and edge.obj_b in norm_pos:
            xa, ya = norm_pos[edge.obj_a]
            xb, yb = norm_pos[edge.obj_b]
            style = edge_style.get(edge.contact_type, dict(color="black", linewidth=1))
            ax_graph.annotate("", xy=(xb, yb), xytext=(xa, ya),
                              arrowprops=dict(arrowstyle="->", **style))
            ax_graph.text((xa+xb)/2, (ya+yb)/2, edge.contact_type,
                          fontsize=7, ha="center", color=style["color"])

    for i, (oid, (nx, ny)) in enumerate(norm_pos.items()):
        score = graph.stability_scores.get(oid, 0.0)
        ax_graph.scatter([nx], [ny], s=300, c=[cmap(i % 10)], zorder=5)
        ax_graph.text(nx, ny + 0.05, f"{oid}\n{score:.2f}",
                      ha="center", fontsize=7)

    legend_lines = [Line2D([0], [0], label=k, **v) for k, v in edge_style.items()]
    ax_graph.legend(handles=legend_lines, fontsize=7, loc="lower right")
    ax_graph.set_title("Contact graph", fontsize=9)
    ax_graph.set_xlim(-0.1, 1.1); ax_graph.set_ylim(-0.1, 1.2)
    ax_graph.axis("off")

    _show_and_wait(fig, gui, "Contact graph shown — press Enter for next test.")
    _ok("Contact graph test complete")
    return graph


# ---------------------------------------------------------------------------
# TEST 3 — Occlusion Map
# ---------------------------------------------------------------------------

def test_occlusion(camera: RealSenseCamera, tracker, n_frames: int, gui: bool) -> bool:
    _section(f"TEST 3 — Occlusion Map ({n_frames} frames, static viewpoint)")

    intrinsics  = camera.get_camera_intrinsics()
    observations: List[ObservationRecord] = []
    colors: List[np.ndarray] = []
    all_object_ids: set = set()

    for i in range(n_frames):
        color, depth = camera.get_aligned_frames()
        asyncio.run(tracker.detect_objects(
            color_frame=color,
            depth_frame=depth,
            camera_intrinsics=intrinsics,
        ))
        masks = dict(tracker._last_masks)
        detected_ids = list(masks.keys())
        all_object_ids.update(detected_ids)
        _info(f"Frame {i+1}/{n_frames}: detected {detected_ids}")

        # No robot state on real hardware — camera pose is fixed/unknown;
        # pass None so the occlusion map uses mask-AABB visibility estimation.
        observations.append(ObservationRecord(
            depth_frame=depth,
            camera_intrinsics=intrinsics,
            camera_pose=None,
            obj_masks=masks,
        ))
        colors.append(color)
        time.sleep(0.1)

    if not observations:
        _warn("No observations — skipping occlusion test")
        return False

    occ_map = compute_occlusion_map(
        observations=observations,
        object_ids=list(all_object_ids),
    )

    _info("Visibility per object:")
    for oid, vis in occ_map.per_object_visibility.items():
        _info(f"  {oid}: {vis.visible_fraction:.0%}  occluders={vis.occluding_objects}")

    if not gui:
        _ok("Occlusion test complete (headless)")
        return True

    # Figure: last frame + visibility bar chart
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TEST 3 — Occlusion Map", fontsize=12, fontweight="bold")

    last_masks = dict(tracker._last_masks)
    _mask_overlay(ax_img, colors[-1], last_masks, title="Last captured frame")
    patches = _legend_patches(last_masks)
    if patches:
        ax_img.legend(handles=patches, loc="lower right", fontsize=7)

    vis_ids   = list(occ_map.per_object_visibility.keys())
    vis_fracs = [occ_map.per_object_visibility[oid].visible_fraction for oid in vis_ids]
    cmap      = plt.get_cmap("tab10")
    ax_bar.bar(range(len(vis_ids)), vis_fracs,
               color=[cmap(i % 10) for i in range(len(vis_ids))])
    ax_bar.axhline(0.6, color="red", linestyle="--", linewidth=1.2,
                   label="visible threshold 0.6")
    ax_bar.set_xticks(range(len(vis_ids)))
    ax_bar.set_xticklabels(vis_ids, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Visible fraction")
    ax_bar.set_title("Per-object visibility", fontsize=9)
    ax_bar.legend(fontsize=8)
    for i, (val, oid) in enumerate(zip(vis_fracs, vis_ids)):
        occluders = occ_map.per_object_visibility[oid].occluding_objects
        label = f"{val:.0%}"
        if occluders:
            label += f"\n({', '.join(occluders[:2])})"
        ax_bar.text(i, val + 0.02, label, ha="center", va="bottom", fontsize=7)

    _show_and_wait(fig, gui, "Occlusion map shown — press Enter for next test.")
    _ok("Occlusion test complete")
    return True


# ---------------------------------------------------------------------------
# TEST 4 — Surface Free-Space Map
# ---------------------------------------------------------------------------

def test_surface_map(camera: RealSenseCamera, tracker, gui: bool,
                     cam_position=None, cam_rotation=None) -> bool:
    from matplotlib.colors import ListedColormap

    _section("TEST 4 — Surface Free-Space Map")

    intrinsics = camera.get_camera_intrinsics()
    color, depth, masks, detected = _capture_frame(camera, tracker)

    if len(detected) < 2:
        _warn("Too few objects for surface map — skipping")
        return False

    # Need contact graph first so surface map can resolve support relationships
    graph = compute_contact_graph(
        objects=detected,
        obj_masks=masks,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        contact_threshold_m=0.05,
        camera_position=cam_position,
        camera_rotation=cam_rotation,
    )

    surface_maps = compute_surface_maps(
        objects=detected,
        obj_masks=masks,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        contact_graph=graph,
        resolution_m=0.01,
    )

    if not surface_maps:
        _warn("No surface maps computed — is there a table/surface in the scene?")
        return False

    for surface_id, smap in surface_maps.items():
        _info(f"{surface_id}: congestion={smap.congestion_score:.2f}"
              f"  free_regions={len(smap.free_space_regions)}")
        for i, region in enumerate(smap.free_space_regions):
            _info(f"  region {i}: area={region.area*1e4:.1f} cm²"
                  f"  radius={region.max_inscribed_circle*100:.1f} cm"
                  f"  neighbors={region.neighboring_objects}")

    if not gui:
        _ok("Surface map test complete (headless)")
        return True

    n_surfaces = len(surface_maps)
    fig, axes = plt.subplots(1, n_surfaces + 1,
                             figsize=(5 * (n_surfaces + 1), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    fig.suptitle("TEST 4 — Surface Free-Space Map", fontsize=12, fontweight="bold")

    _mask_overlay(axes[0], color,
                  {sid: masks.get(sid, np.zeros(depth.shape, dtype=bool))
                   for sid in surface_maps},
                  title="Surface masks")

    for col, (surface_id, smap) in enumerate(surface_maps.items(), start=1):
        ax = axes[col] if col < len(axes) else axes[-1]
        if smap.occupancy_grid is not None:
            cmap_grid = ListedColormap(["#b8f5b8", "#f5b8b8"])
            ax.imshow(smap.occupancy_grid.astype(float),
                      cmap=cmap_grid, origin="lower", vmin=0, vmax=1)
            for region in smap.free_space_regions:
                if hasattr(region, "pixel_mask") and region.pixel_mask is not None:
                    ys, xs = np.where(region.pixel_mask)
                    if len(xs):
                        ax.scatter(xs, ys, s=1, c="royalblue", alpha=0.4)
                if hasattr(region, "center_uv") and region.center_uv is not None:
                    cx_r, cy_r = region.center_uv
                    r_px = region.max_inscribed_circle / (smap.resolution_m or 0.01)
                    ax.add_patch(plt.Circle((cx_r, cy_r), r_px,
                                           color="blue", fill=False, linewidth=1.5))
                    ax.text(cx_r, cy_r,
                            f"r={region.max_inscribed_circle*100:.1f}cm",
                            ha="center", va="center", fontsize=6, color="navy")
            ax.legend(handles=[
                mpatches.Patch(color="#b8f5b8", label="free"),
                mpatches.Patch(color="#f5b8b8", label="occupied"),
            ], fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no grid\n(mask empty?)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_title(
            f"{surface_id}\ncongestion={smap.congestion_score:.0%}"
            f"  regions={len(smap.free_space_regions)}",
            fontsize=9,
        )
        ax.set_xlabel("U (px)"); ax.set_ylabel("V (px)")

    plt.tight_layout()
    _show_and_wait(fig, gui, "Surface map shown — done.")
    _ok("Surface map test complete")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perception component integration test — live RealSense"
    )
    parser.add_argument("--no-gui", action="store_true", help="Headless mode")
    parser.add_argument(
        "--tests", nargs="*",
        choices=["clearance", "contact", "occlusion", "surface", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Number of frames to accumulate for the occlusion test (default: 5)",
    )
    parser.add_argument(
        "--joint-state", type=float, nargs=7, metavar="J",
        default=[0.100085, -1.407677, -0.098652, 1.314592, 0.0, 2.0, -0.112296],
        help="xArm7 joint angles (rad) for FK-based camera transform "
             "(default: tabletop viewing pose).",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use LLM to classify contact relationships instead of geometric heuristics.",
    )
    parser.add_argument(
        "--llm-mode", choices=["json", "nl"], default="nl",
        help="LLM prompt mode: 'json' for structured JSON output, "
             "'nl' for natural-language lines (default: nl).",
    )
    args = parser.parse_args()
    gui = not args.no_gui

    run_all = "all" in (args.tests or ["all"])
    run = {
        "clearance": run_all or "clearance" in args.tests,
        "contact":   run_all or "contact"   in args.tests,
        "occlusion": run_all or "occlusion" in args.tests,
        "surface":   run_all or "surface"   in args.tests,
    }

    if not gui:
        plt.ioff()

    print(_DIVIDER)
    print("  Perception Component Integration Test — Live RealSense")
    print(_DIVIDER)

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _info(f"Device: {device}")

    # --- Camera ---
    _section("Starting RealSense camera")
    camera = RealSenseCamera(width=640, height=480, fps=30, enable_depth=True)
    intrinsics = camera.get_camera_intrinsics()
    _ok(f"Camera ready  fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}"
        f"  cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}")

    # --- Robot interface for world-frame transforms ---
    robot_interface = None
    cam_position = None
    cam_rotation = None
    _section("Setting up robot interface (xArm7 FK)")
    try:
        from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
        robot_interface = XArmPybulletInterface()
        robot_interface.set_current_joint_state(args.joint_state)
        _info(f"Joint state: {[f'{j:.3f}' for j in args.joint_state]}")
        from scipy.spatial.transform import Rotation
        pos, rot = robot_interface.get_camera_transform()
        if pos is not None and rot is not None:
            cam_position = pos
            cam_rotation = rot
            _ok(f"Camera transform: pos={pos.round(3).tolist()}  "
                f"quat={rot.as_quat().round(3).tolist()}")
        else:
            _warn("FK returned no camera transform — falling back to camera frame")
    except Exception as exc:
        _warn(f"Robot interface unavailable ({exc}) — using camera frame")

    # --- GSAM2 tracker ---
    _section("Loading GSAM2 tracker")
    from src.perception.gsam2_object_tracker import GSAM2ObjectTracker

    tracker = GSAM2ObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        compute_clearances=False,   # tests call perception directly
        compute_contacts=False,
        compute_occlusion=False,
        compute_surface_maps=False,
        robot_interface=robot_interface,
    )
    _ok("GSAM2 loaded")

    # --- Optional LLM client for contact classification ---
    llm_client = None
    if args.llm:
        _section("Setting up LLM contact classifier")
        try:
            from src.llm_interface import OpenAIClient
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                _warn("OPENAI_API_KEY not set — LLM classification disabled")
            else:
                llm_client = OpenAIClient(
                    model="gpt-4o-mini",
                    api_key=openai_key,
                )
                _ok(f"LLM client ready: {llm_client._model}")
        except Exception as exc:
            _warn(f"LLM client setup failed ({exc}) — falling back to geometric")

    results: Dict[str, bool] = {}

    if run["clearance"]:
        try:
            results["clearance"] = test_clearance(camera, tracker, gui,
                                                   cam_position=cam_position,
                                                   cam_rotation=cam_rotation)
        except Exception as exc:
            log.exception("Clearance test failed")
            results["clearance"] = False
            _warn(f"Clearance FAILED: {exc}")

    if run["contact"]:
        try:
            result = test_contact_graph(camera, tracker, gui,
                                        cam_position=cam_position,
                                        cam_rotation=cam_rotation,
                                        llm_client=llm_client,
                                        llm_mode=args.llm_mode)
            results["contact"] = result is not False
        except Exception as exc:
            log.exception("Contact graph test failed")
            results["contact"] = False
            _warn(f"Contact FAILED: {exc}")

    if run["occlusion"]:
        try:
            results["occlusion"] = test_occlusion(camera, tracker, args.frames, gui)
        except Exception as exc:
            log.exception("Occlusion test failed")
            results["occlusion"] = False
            _warn(f"Occlusion FAILED: {exc}")

    if run["surface"]:
        try:
            results["surface"] = test_surface_map(camera, tracker, gui,
                                                   cam_position=cam_position,
                                                   cam_rotation=cam_rotation)
        except Exception as exc:
            log.exception("Surface map test failed")
            results["surface"] = False
            _warn(f"Surface FAILED: {exc}")

    # --- Summary ---
    print(f"\n{_DIVIDER}")
    print("  Results")
    print(_DIVIDER)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
    print(_DIVIDER)

    camera.stop()


if __name__ == "__main__":
    main()
