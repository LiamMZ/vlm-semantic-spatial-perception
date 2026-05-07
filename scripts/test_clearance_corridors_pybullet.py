"""
Clearance corridor visualisation + PyBullet approach alignment test.

For each detected object:
  1. Runs the full clearance pipeline (RealSense → GSAM2 → compute_clearance_profile)
  2. Shows the matplotlib 3-D corridor viewer from test_perception_realsense.py
  3. For each grasp-compatible corridor (sorted best-first), solves IK in PyBullet,
     moves the sim arm to that pose, and draws the corridor OBB as debug lines.
  4. Press N / → for next corridor, P / ← for previous, Q / Esc to quit object,
     then advances to the next object automatically.

Run:
    uv run scripts/test_clearance_corridors_pybullet.py
    uv run scripts/test_clearance_corridors_pybullet.py --no-gui
    uv run scripts/test_clearance_corridors_pybullet.py --object cup_0
    uv run scripts/test_clearance_corridors_pybullet.py --joint-state 0.1 -1.4 -0.1 1.3 0.0 2.0 -0.1
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
plt.ion()

import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_camera import RealSenseCamera
from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.perception.clearance import GripperGeometry, compute_clearance_profile, ApproachCorridor
from src.perception.object_registry import DetectedObject

log = logging.getLogger("clearance_corridors_pybullet")

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_GREY   = "\033[90m"

# OBB edge index pairs — same as in test_perception_realsense.py
_OBB_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _setup_logging() -> None:
    class _Fmt(logging.Formatter):
        _MAP = {"DEBUG": _GREY, "INFO": _CYAN, "WARNING": _YELLOW,
                "ERROR": _RED, "CRITICAL": _RED + _BOLD}
        def format(self, r: logging.LogRecord) -> str:
            color = self._MAP.get(r.levelname, "")
            ts    = self.formatTime(r, "%H:%M:%S")
            name  = r.name.split(".")[-1][:20]
            return (f"{_GREY}{ts}{_RESET} {color}{r.levelname:<8}{_RESET} "
                    f"{_GREY}[{name}]{_RESET} {r.getMessage()}")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "PIL", "matplotlib", "models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# PyBullet corridor drawing
# ---------------------------------------------------------------------------

def _obb_corners(corridor: ApproachCorridor) -> np.ndarray:
    """Return the 8 world-frame corners of a corridor OBB."""
    s = np.asarray(corridor.corridor_start, float)
    d = np.asarray(corridor.direction, float) * corridor.corridor_length
    g = np.asarray(corridor.grasp_axis, float) * corridor.half_width
    h = np.asarray(corridor.height_axis, float) * corridor.half_height
    return np.array([
        s + sg * g + sh * h + ud
        for ud in (np.zeros(3), d)
        for sg in (-1.0, +1.0)
        for sh in (-1.0, +1.0)
    ])


def _draw_corridor_pybullet(
    client: int,
    corridor: ApproachCorridor,
    color: List[float],
    line_width: float = 2.0,
) -> List[int]:
    """Draw corridor OBB wireframe in PyBullet. Returns list of debug item IDs."""
    corners = _obb_corners(corridor)
    ids: List[int] = []
    for i, j in _OBB_EDGES:
        lid = p.addUserDebugLine(
            corners[i].tolist(), corners[j].tolist(),
            lineColorRGB=color,
            lineWidth=line_width,
            physicsClientId=client,
        )
        ids.append(lid)
    # Approach direction arrow from corridor_start to far face centre
    start = np.asarray(corridor.corridor_start, float)
    tip   = start + np.asarray(corridor.direction, float) * corridor.corridor_length
    aid = p.addUserDebugLine(
        start.tolist(), tip.tolist(),
        lineColorRGB=color,
        lineWidth=3.0,
        physicsClientId=client,
    )
    ids.append(aid)
    return ids


def _clear_debug_items(client: int, ids: List[int]) -> None:
    for i in ids:
        try:
            p.removeUserDebugItem(i, physicsClientId=client)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# IK + arm positioning
# ---------------------------------------------------------------------------

def _corridor_tcp_quat(corridor: ApproachCorridor) -> np.ndarray:
    """Convert a corridor into a gripper TCP quaternion (xyzw).

    TCP-Z → -corridor.direction (gripper enters from outside the corridor).
    TCP-Y → -corridor.height_axis (world-up → gripper-down convention).
    Matches the sign convention used in corridors_to_seeds() in grasp_planner.py.
    """
    tcp_z = -np.asarray(corridor.direction,  float)
    tcp_y =  np.asarray(corridor.grasp_axis, float)
    rot, _ = Rotation.align_vectors(
        [tcp_z, tcp_y],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        weights=[10.0, 1.0],
    )
    return rot.as_quat()


def _move_sim_to_corridor(
    robot: XArmPybulletInterface,
    corridor: ApproachCorridor,
) -> Optional[np.ndarray]:
    """Solve IK for the corridor approach pose and apply it to the sim.

    The TCP is placed at corridor_start (the surface of the object) with
    orientation aligned to enter along the corridor direction.

    Returns the joint solution (7,) or None if IK failed.
    """
    target_pos  = np.asarray(corridor.corridor_start, float)
    target_quat = _corridor_tcp_quat(corridor)

    joints = robot._ik_solve(
        target_pos=target_pos,
        target_quat=target_quat,
        position_tolerance=0.05,
        orientation_tolerance_rad=0.6,
    )
    if joints is None:
        return None

    robot.set_current_joint_state(joints)
    robot._apply_joints_to_sim()
    return joints


# ---------------------------------------------------------------------------
# Matplotlib 3-D corridor viewer (interactive, one corridor at a time)
# ---------------------------------------------------------------------------

def _show_corridors_3d(
    profiles: Dict[str, any],
    color_frame: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: any,
    cam_position: Optional[np.ndarray],
    cam_rotation: Optional[Rotation],
    robot: XArmPybulletInterface,
    gui: bool,
) -> None:
    """Interactive corridor viewer that also moves the PyBullet sim arm."""
    objects = [oid for oid, p_ in profiles.items()
               if p_.target_pts is not None and len(p_.target_pts) > 0]
    if not objects or not gui:
        return

    # Build RGBD scene point cloud for context
    scene_xyz: Optional[np.ndarray] = None
    scene_rgb: Optional[np.ndarray] = None
    if (color_frame is not None and depth_frame is not None
            and intrinsics is not None
            and cam_position is not None and cam_rotation is not None):
        fx = float(getattr(intrinsics, 'fx', 0))
        fy = float(getattr(intrinsics, 'fy', 0))
        cx = float(getattr(intrinsics, 'ppx', getattr(intrinsics, 'cx', 0)))
        cy = float(getattr(intrinsics, 'ppy', getattr(intrinsics, 'cy', 0)))
        if fx > 0 and fy > 0:
            h, w = depth_frame.shape
            step = 4
            rows = np.arange(0, h, step)
            cols = np.arange(0, w, step)
            cc, rr = np.meshgrid(cols, rows)
            z = depth_frame[rr, cc].ravel()
            valid = (z > 0.05) & (z < 4.0) & np.isfinite(z)
            z = z[valid]
            u = cc.ravel()[valid]
            v = rr.ravel()[valid]
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z], axis=1)
            R_wc = cam_rotation.as_matrix()
            t_wc = np.asarray(cam_position, float)
            scene_xyz = (R_wc @ pts_cam.T).T + t_wc
            rgb = color_frame[rr, cc].reshape(-1, 3)[valid]
            scene_rgb = rgb.astype(float) / 255.0

    cmap = plt.get_cmap("tab10")

    # Flat nav list: one entry per compatible corridor across all objects
    nav: List[tuple] = []
    for oid in objects:
        for ci, corridor in enumerate(profiles[oid].approach_corridors):
            if corridor.grasp_compatible:
                nav.append((oid, ci))

    if not nav:
        log.warning("No grasp-compatible corridors found across all objects")
        return

    state = {"idx": 0, "debug_ids": [], "ik_ok": False}
    client = robot._physics_client

    def _pts_in_obb(pts: np.ndarray, corridor: ApproachCorridor) -> np.ndarray:
        s = np.asarray(corridor.corridor_start, float)
        d = np.asarray(corridor.direction, float)
        g = np.asarray(corridor.grasp_axis, float)
        h = np.asarray(corridor.height_axis, float)
        center = s + d * corridor.corridor_length * 0.5
        rel = pts - center
        return (
            (np.abs(rel @ d) <= corridor.corridor_length * 0.5) &
            (np.abs(rel @ g) <= corridor.half_width) &
            (np.abs(rel @ h) <= corridor.half_height)
        )

    fig = plt.figure(figsize=(10, 7))

    def _draw(idx: int) -> None:
        fig.clf()
        oid, ci = nav[idx]
        profile  = profiles[oid]
        corridor = profile.approach_corridors[ci]
        obj_color = cmap(objects.index(oid) % 10)

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        n_ok = sum(1 for c in profile.approach_corridors if c.grasp_compatible)
        ik_tag = "  IK: OK" if state["ik_ok"] else "  IK: FAILED"
        fig.suptitle(
            f"[{idx+1}/{len(nav)}]  {oid}  corridor {ci+1}  —  "
            f"N/→=next  P/←=prev  Q=quit{ik_tag}",
            fontsize=9,
        )

        tgt = profile.target_pts

        # Scene points inside this corridor, excluding target object
        if scene_xyz is not None and len(scene_xyz) > 0:
            tgt_sample = tgt[::max(1, len(tgt) // 500)]
            dists = np.linalg.norm(
                scene_xyz[:, None, :] - tgt_sample[None, :, :], axis=2
            ).min(axis=1)
            not_obj = dists > 0.015
            non_obj_xyz = scene_xyz[not_obj]
            non_obj_rgb = scene_rgb[not_obj]
            if len(non_obj_xyz) > 0:
                in_corr = _pts_in_obb(non_obj_xyz, corridor)
                if in_corr.any():
                    step = max(1, in_corr.sum() // 4000)
                    fg = np.where(in_corr)[0][::step]
                    ax.scatter(
                        non_obj_xyz[fg, 0], non_obj_xyz[fg, 1], non_obj_xyz[fg, 2],
                        s=2.0, c=non_obj_rgb[fg], alpha=0.85, depthshade=False,
                    )

        # Target object points
        step = max(1, len(tgt) // 1000)
        ax.scatter(tgt[::step, 0], tgt[::step, 1], tgt[::step, 2],
                   s=10, color="cyan", alpha=0.95, depthshade=False, zorder=5)

        # Corridor OBB
        s  = np.asarray(corridor.corridor_start, float)
        d  = np.asarray(corridor.direction, float)      * corridor.corridor_length
        g  = np.asarray(corridor.grasp_axis, float)    * corridor.half_width
        h_ = np.asarray(corridor.height_axis, float)   * corridor.half_height
        corners = np.array([
            s + sg * g + sh * h_ + ud
            for ud in (np.zeros(3), d)
            for sg in (-1.0, +1.0)
            for sh in (-1.0, +1.0)
        ])
        ec = "#2ca02c"
        segs = [[corners[i], corners[j]] for i, j in _OBB_EDGES]
        ax.add_collection3d(Line3DCollection(segs, colors=ec, linewidths=2.0, alpha=0.9))
        ax.quiver(s[0], s[1], s[2], d[0], d[1], d[2],
                  color=ec, alpha=0.9, arrow_length_ratio=0.15, linewidth=1.5)

        # Axis limits centred on object + corridor
        center = tgt.mean(axis=0)
        far = s + np.asarray(corridor.direction, float) * corridor.corridor_length
        span = max(
            np.percentile(tgt, 95, axis=0).max() - np.percentile(tgt, 5, axis=0).min(),
            np.linalg.norm(far - center) * 2.2,
            0.3,
        )
        half = span / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_title(
            f"{oid}   CLEAR   graspability={profile.graspability_score:.2f}"
            f"   {n_ok}/{len(profile.approach_corridors)} compatible",
            fontsize=9,
        )
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-60)
        fig.canvas.draw_idle()

    def _update_pybullet(idx: int) -> None:
        _clear_debug_items(client, state["debug_ids"])
        state["debug_ids"] = []

        oid, ci = nav[idx]
        corridor = profiles[oid].approach_corridors[ci]

        joints = _move_sim_to_corridor(robot, corridor)
        state["ik_ok"] = joints is not None

        color = [0.1, 0.9, 0.1] if state["ik_ok"] else [0.9, 0.2, 0.1]
        state["debug_ids"] = _draw_corridor_pybullet(client, corridor, color)

        d  = np.round(corridor.direction,    4).tolist()
        g  = np.round(corridor.grasp_axis,   4).tolist()
        h  = np.round(corridor.height_axis,  4).tolist()
        print(f"\n  corridor {idx+1}/{len(nav)}  [{oid}]")
        print(f"    direction   (approach): {d}")
        print(f"    grasp_axis  (fingers):  {g}")
        print(f"    height_axis (up):       {h}")

        if state["ik_ok"]:
            log.info("corridor %d/%d  %s: IK OK  joints=%s",
                     idx + 1, len(nav), oid, np.round(joints, 3).tolist())
        else:
            log.warning("corridor %d/%d  %s: IK FAILED — no solution found", idx + 1, len(nav), oid)

    def _refresh(idx: int) -> None:
        _update_pybullet(idx)
        _draw(idx)

    def _on_key(event) -> None:
        if event.key in ("n", "right"):
            state["idx"] = (state["idx"] + 1) % len(nav)
            _refresh(state["idx"])
        elif event.key in ("p", "left"):
            state["idx"] = (state["idx"] - 1) % len(nav)
            _refresh(state["idx"])
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _refresh(0)
    plt.tight_layout()
    plt.show(block=True)

    # Clean up PyBullet debug items on close
    _clear_debug_items(client, state["debug_ids"])
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Clearance corridor visualisation + PyBullet approach alignment"
    )
    parser.add_argument("--no-gui", action="store_true", help="Headless mode")
    parser.add_argument(
        "--joint-state", type=float, nargs=7, metavar="J",
        default=[0.100085, -1.407677, -0.098652, 1.314592, 0.0, 2.0, -0.112296],
        help="xArm7 joint angles (rad) for FK-based camera transform",
    )
    args = parser.parse_args()
    gui = not args.no_gui

    if not gui:
        plt.ioff()

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # --- PyBullet sim (GUI so we can watch the arm move) ---
    log.info("Starting PyBullet sim (GUI=%s)", gui)
    robot = XArmPybulletInterface(use_gui=gui)
    robot.set_current_joint_state(args.joint_state)
    cam_pos, cam_rot = robot.get_camera_transform()
    if cam_pos is not None:
        log.info("Camera transform: pos=%s  quat=%s",
                 np.round(cam_pos, 3).tolist(),
                 np.round(cam_rot.as_quat(), 3).tolist())
    else:
        log.warning("FK returned no camera transform — clearance will use camera frame")

    # --- RealSense ---
    log.info("Starting RealSense camera")
    camera = RealSenseCamera(width=640, height=480, fps=30, enable_depth=True)
    intrinsics = camera.get_camera_intrinsics()
    log.info("Camera ready  fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
             intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

    # --- GSAM2 tracker ---
    log.info("Loading GSAM2 tracker")
    from src.perception.gsam2_object_tracker import GSAM2ObjectTracker
    tracker = GSAM2ObjectTracker(
        sam2_ckpt_path=sam2_ckpt,
        device=device,
        compute_clearances=False,
        compute_contacts=False,
        compute_occlusion=False,
        robot_interface=robot,
    )
    log.info("GSAM2 loaded — capturing frame")

    color, depth = camera.get_aligned_frames()
    asyncio.run(tracker.detect_objects(
        color_frame=color,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
    ))
    masks    = dict(tracker._last_masks)
    detected = list(tracker.registry.get_all_objects())

    if not detected:
        log.error("No objects detected — check camera and lighting")
        camera.stop()
        robot.cleanup()
        sys.exit(1)

    log.info("Depth range: [%.3f, %.3f] m", depth.min(), depth.max())

    # --- Object selection ---
    detected_ids = [o.object_id for o in detected]
    print()
    print("Detected objects:")
    for i, oid in enumerate(detected_ids):
        print(f"  [{i}] {oid}")
    print()
    while True:
        try:
            raw = input("Select object (number or name, Enter for all): ").strip()
        except EOFError:
            raw = ""
        if raw == "":
            break
        if raw.isdigit() and int(raw) < len(detected_ids):
            detected = [detected[int(raw)]]
            break
        if raw in detected_ids:
            detected = [o for o in detected if o.object_id == raw]
            break
        print(f"  '{raw}' not recognised — enter a number 0–{len(detected_ids)-1} or an object ID")

    log.info("Targeting: %s", [o.object_id for o in detected])

    # --- Compute clearance profiles ---
    gripper = GripperGeometry()
    profiles: Dict[str, any] = {}

    for obj in detected:
        oid = obj.object_id
        target_mask = masks.get(oid)
        if target_mask is None or not target_mask.any():
            log.warning("%s: empty mask — skipping", oid)
            continue

        other_masks = {k: v for k, v in masks.items() if k != oid}
        n_px = int(target_mask.sum())
        log.info("%s: computing clearance profile (%d px, %d neighbours)...",
                 oid, n_px, sum(1 for v in other_masks.values() if v.any()))

        try:
            profile = compute_clearance_profile(
                target_mask=target_mask,
                depth_frame=depth,
                camera_intrinsics=intrinsics,
                all_masks=other_masks,
                gripper=gripper,
                cam_position=cam_pos,
                camera_quaternion_xyzw=cam_rot.as_quat() if cam_rot is not None else None,
            )
            profiles[oid] = profile

            n_ok  = sum(1 for c in profile.approach_corridors if c.grasp_compatible)
            n_blk = len(profile.approach_corridors) - n_ok
            log.info("  %s: %d corridors (%d ok, %d blocked)  graspability=%.2f",
                     oid, len(profile.approach_corridors), n_ok, n_blk,
                     profile.graspability_score)
        except Exception as exc:
            log.exception("%s: clearance profile failed — %s", oid, exc)

    if not profiles:
        log.error("No clearance profiles computed")
        camera.stop()
        robot.cleanup()
        sys.exit(1)

    # --- Run interactive viewer + PyBullet alignment ---
    _show_corridors_3d(
        profiles=profiles,
        color_frame=color,
        depth_frame=depth,
        intrinsics=intrinsics,
        cam_position=cam_pos,
        cam_rotation=cam_rot,
        robot=robot,
        gui=gui,
    )

    camera.stop()
    if not gui:
        robot.cleanup()
    else:
        log.info("PyBullet window left open — close it manually to exit")
        try:
            while True:
                p.stepSimulation(physicsClientId=robot._physics_client)
                time.sleep(1.0 / 240.0)
        except Exception:
            pass
        robot.cleanup()


if __name__ == "__main__":
    main()
