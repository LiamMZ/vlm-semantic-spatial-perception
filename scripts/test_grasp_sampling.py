"""
Interactive grasp sampling — detection → corridor-seeded grasp candidates.

For a selected object this script:
  1. Runs RealSense + GSAM2 detection and builds depth-mesh collision bodies
  2. Computes a ClearanceProfile (approach corridors) for the target object
  3. Feeds the profile into GraspPlanner to sample collision-free grasps
  4. For each candidate:
     - Moves the PyBullet sim arm to the grasp pose
     - Draws the approach corridor OBB and a TCP-frame triad in the sim
     - Prints seed name, approach angle, manipulability, and approach direction
     - Runs a detailed collision check and prints what (if anything) was hit
     - Waits for Enter / N (next) / Q (quit)

Run:
    uv run scripts/test_grasp_sampling.py
    uv run scripts/test_grasp_sampling.py --object cup_0
    uv run scripts/test_grasp_sampling.py --joint-state 0.1 -1.4 -0.1 1.3 0.0 2.0 -0.1
    uv run scripts/test_grasp_sampling.py --no-gui
    uv run scripts/test_grasp_sampling.py --seed side --cone 30 --max-grasps 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_camera import RealSenseCamera
from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.kinematics.depth_environment_collider import DepthEnvironmentCollider, BACKGROUND_ID
from src.perception.clearance import GripperGeometry, compute_clearance_profile, ApproachCorridor
from src.grasp_planner.grasp_planner import GraspPlanner, corridors_to_seeds
from src.grasp_planner.grasp_candidate import GraspCandidate

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"
_GREY   = "\033[90m"
_MAGENTA = "\033[35m"


def _setup_logging() -> None:
    class _Fmt(logging.Formatter):
        _MAP = {"DEBUG": _GREY, "INFO": _CYAN, "WARNING": _YELLOW,
                "ERROR": _RED, "CRITICAL": _RED + _BOLD}
        def format(self, r: logging.LogRecord) -> str:
            color = self._MAP.get(r.levelname, "")
            ts    = self.formatTime(r, "%H:%M:%S")
            name  = r.name.split(".")[-1][:22]
            return (f"{_GREY}{ts}{_RESET} {color}{r.levelname:<8}{_RESET} "
                    f"{_GREY}[{name}]{_RESET} {r.getMessage()}")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "PIL", "matplotlib", "models", "trimesh"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = logging.getLogger("grasp_sampling")

# ---------------------------------------------------------------------------
# OBB helpers (shared edge pairs)
# ---------------------------------------------------------------------------

_OBB_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _obb_corners(corridor: ApproachCorridor) -> np.ndarray:
    s = np.asarray(corridor.corridor_start, float)
    d = np.asarray(corridor.direction,    float) * corridor.corridor_length
    g = np.asarray(corridor.grasp_axis,   float) * corridor.half_width
    h = np.asarray(corridor.height_axis,  float) * corridor.half_height
    return np.array([
        s + sg * g + sh * h + ud
        for ud in (np.zeros(3), d)
        for sg in (-1.0, +1.0)
        for sh in (-1.0, +1.0)
    ])


# ---------------------------------------------------------------------------
# PyBullet debug drawing
# ---------------------------------------------------------------------------

def _draw_corridor(client: int, corridor: ApproachCorridor, color: List[float]) -> List[int]:
    ids: List[int] = []
    corners = _obb_corners(corridor)
    for i, j in _OBB_EDGES:
        ids.append(p.addUserDebugLine(
            corners[i].tolist(), corners[j].tolist(),
            lineColorRGB=color, lineWidth=2.0, physicsClientId=client,
        ))
    # Approach arrow
    start = np.asarray(corridor.corridor_start, float)
    tip   = start + np.asarray(corridor.direction, float) * corridor.corridor_length
    ids.append(p.addUserDebugLine(
        start.tolist(), tip.tolist(),
        lineColorRGB=color, lineWidth=3.0, physicsClientId=client,
    ))
    return ids


def _draw_tcp_frame(
    client: int,
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    length: float = 0.08,
) -> List[int]:
    """Draw RGB XYZ axes at the TCP pose."""
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    pos = np.asarray(position, float)
    ids: List[int] = []
    for axis, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        tip = pos + rot[:, axis] * length
        ids.append(p.addUserDebugLine(
            pos.tolist(), tip.tolist(),
            lineColorRGB=color, lineWidth=3.0, physicsClientId=client,
        ))
    return ids


def _draw_text(client: int, position: np.ndarray, text: str, color: List[float]) -> int:
    return p.addUserDebugText(
        text, position.tolist(),
        textColorRGB=color, textSize=1.2, physicsClientId=client,
    )


def _clear(client: int, ids: List[int]) -> None:
    for i in ids:
        try:
            p.removeUserDebugItem(i, physicsClientId=client)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Detailed collision diagnostics
# ---------------------------------------------------------------------------

class CollisionHit:
    """One collision or near-miss contact, with 3-D positions for visualization."""
    def __init__(
        self,
        description: str,
        pt_on_robot: np.ndarray,   # world-frame point on the robot link surface
        pt_on_other: np.ndarray,   # world-frame point on the obstacle surface
        distance_m: float,
    ) -> None:
        self.description = description
        self.pt_on_robot = pt_on_robot
        self.pt_on_other = pt_on_other
        self.distance_m  = distance_m


def _collision_details(
    robot: XArmPybulletInterface,
    joints: np.ndarray,
    collider: Optional[DepthEnvironmentCollider],
    margin: float = 0.005,
) -> Tuple[bool, List[CollisionHit]]:
    """Re-apply joints and run a detailed collision check.

    Returns:
        (collision_free, list of CollisionHit with 3-D contact positions)
    """
    client  = robot._physics_client
    rid     = robot._robot_id
    n_links = p.getNumJoints(rid, physicsClientId=client)

    for j, joint_idx in enumerate(robot._movable_joints):
        if j < len(joints):
            p.resetJointState(rid, joint_idx, float(joints[j]), physicsClientId=client)

    body_labels: Dict[int, str] = {}
    if collider is not None:
        for label, bid in collider._bodies.items():
            body_labels[bid] = "background" if label == BACKGROUND_ID else label

    hits: List[CollisionHit] = []

    def _closest_pt(pt_list) -> CollisionHit:
        """Pick the point with the smallest (most negative / closest) distance."""
        pt = min(pt_list, key=lambda x: x[8])
        return np.array(pt[5], float), np.array(pt[6], float), float(pt[8])

    # Robot vs environment meshes.
    if collider is not None:
        for label, body_id in collider._bodies.items():
            display = body_labels.get(body_id, str(body_id))
            for link_idx in range(-1, n_links):
                pts = p.getClosestPoints(
                    bodyA=rid, bodyB=body_id,
                    distance=margin,
                    linkIndexA=link_idx,
                    physicsClientId=client,
                )
                if pts:
                    pr, po, d = _closest_pt(pts)
                    link_name = _link_name(robot, link_idx)
                    hits.append(CollisionHit(
                        description=(
                            f"env-mesh [{display}]  link {link_idx} ({link_name})  "
                            f"dist={d*1000:.1f}mm"
                        ),
                        pt_on_robot=pr,
                        pt_on_other=po,
                        distance_m=d,
                    ))
                    break  # one hit per body is enough

    # Robot vs floor.
    floor_body = robot._floor_body_id
    if floor_body is not None:
        for link_idx in range(robot._floor_check_start_link, robot._floor_check_end_link):
            pts = p.getClosestPoints(
                bodyA=rid, bodyB=floor_body,
                distance=margin,
                linkIndexA=link_idx,
                physicsClientId=client,
            )
            if pts:
                pr, po, d = _closest_pt(pts)
                link_name = _link_name(robot, link_idx)
                hits.append(CollisionHit(
                    description=(
                        f"floor-plane  link {link_idx} ({link_name})  "
                        f"dist={d*1000:.1f}mm"
                    ),
                    pt_on_robot=pr,
                    pt_on_other=po,
                    distance_m=d,
                ))

    # Self-collision.
    end = robot._arm_self_collision_end_link
    for link_a in range(-1, end):
        for link_b in range(link_a + 2, end + 1):
            pts = p.getClosestPoints(
                bodyA=rid, bodyB=rid,
                distance=0.0,
                linkIndexA=link_a, linkIndexB=link_b,
                physicsClientId=client,
            )
            if pts:
                pr, po, d = _closest_pt(pts)
                na = _link_name(robot, link_a)
                nb = _link_name(robot, link_b)
                hits.append(CollisionHit(
                    description=(
                        f"self-collision  link {link_a} ({na}) ↔ link {link_b} ({nb})  "
                        f"dist={d*1000:.1f}mm"
                    ),
                    pt_on_robot=pr,
                    pt_on_other=po,
                    distance_m=d,
                ))

    return len(hits) == 0, hits


def _link_name(robot: XArmPybulletInterface, link_idx: int) -> str:
    """Return the joint/link name for a given link index, or 'base' for -1."""
    if link_idx == -1:
        return "base"
    try:
        info = p.getJointInfo(robot._robot_id, link_idx, physicsClientId=robot._physics_client)
        return info[1].decode("utf-8") if info else str(link_idx)
    except Exception:
        return str(link_idx)


# ---------------------------------------------------------------------------
# Terminal grasp report
# ---------------------------------------------------------------------------

def _print_separator() -> None:
    print(_BOLD + _GREY + "─" * 72 + _RESET)


def _print_grasp_report(
    idx: int,
    total: int,
    candidate: GraspCandidate,
    corridor: Optional[ApproachCorridor],
    collision_free: bool,
    collision_hits: List["CollisionHit"],
) -> None:
    _print_separator()
    status_color = _GREEN if collision_free else _RED
    status_label = "COLLISION-FREE" if collision_free else "IN COLLISION"
    print(
        f"  {_BOLD}Grasp {idx}/{total}{_RESET}  "
        f"seed={_CYAN}{candidate.seed_orientation}{_RESET}  "
        f"{status_color}{_BOLD}{status_label}{_RESET}"
    )
    print(
        f"    approach angle : {_YELLOW}{np.degrees(candidate.approach_angle_rad):.1f}°{_RESET}"
        f"  from seed"
    )
    print(
        f"    manipulability : {_YELLOW}{candidate.manipulability:.4f}{_RESET}"
    )

    # TCP frame in world.
    rot = Rotation.from_quat(candidate.orientation)
    tcp_z = rot.apply([0, 0, 1])   # approach direction of gripper
    tcp_y = rot.apply([0, 1, 0])   # finger-spread axis
    tcp_x = rot.apply([1, 0, 0])   # palm-normal axis
    print(
        f"    TCP position   : {_CYAN}{np.round(candidate.position, 4).tolist()}{_RESET}"
    )
    print(
        f"    TCP Z (approach): {_CYAN}{np.round(tcp_z, 3).tolist()}{_RESET}  "
        f"→ gripper enters from {_direction_label(-tcp_z)}"
    )
    print(
        f"    TCP Y (fingers) : {_CYAN}{np.round(tcp_y, 3).tolist()}{_RESET}"
    )

    if corridor is not None:
        oc = corridor.obstructing_objects
        compat = corridor.grasp_compatible
        print(
            f"    corridor dir   : {np.round(corridor.direction, 3).tolist()}  "
            f"min_clearance={corridor.min_clearance*1000:.1f}mm  "
            f"compatible={compat}"
        )
        if oc:
            print(f"    obstructors    : {_YELLOW}{', '.join(oc)}{_RESET}")

    print(
        f"    joints (rad)   : {_GREY}{np.round(candidate.joints, 3).tolist()}{_RESET}"
    )

    if collision_free:
        print(f"  {_GREEN}✓  No collisions detected{_RESET}")
    else:
        print(f"  {_RED}✗  Collision(s) detected:{_RESET}")
        for hit in collision_hits:
            print(f"      {_RED}•{_RESET} {hit.description}")


def _direction_label(v: np.ndarray) -> str:
    """Return a cardinal-direction string for the dominant component of v."""
    labels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    axes   = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
    ], dtype=float)
    idx = int(np.argmax(np.abs(axes @ v)))
    dot = float(axes[idx] @ v)
    # pick sign
    label = labels[idx] if dot > 0 else labels[idx ^ 1]
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Interactive grasp sampling: detect → corridor seeds → "
                    "collision-free grasp candidates"
    )
    parser.add_argument("--object", default=None,
                        help="Object ID to target (skip interactive selection)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless PyBullet (no 3-D window)")
    parser.add_argument("--seed", default="top_down",
                        choices=["top_down", "side"],
                        help="Nominal seed orientation for corridor ranking")
    parser.add_argument("--cone", type=float, default=45.0,
                        help="Grasp cone half-angle in degrees (default 45)")
    parser.add_argument("--max-grasps", type=int, default=30,
                        help="Max candidates to collect before stopping (default 30)")
    parser.add_argument(
        "--joint-state", type=float, nargs=7, metavar="J",
        default=[-0.141372, -1.314233, -0.434587, 1.53589, -0.132645, 2.028073, -0.60912],
        help="xArm7 joint angles (rad) used to compute the camera transform",
    )
    args = parser.parse_args()

    sam2_ckpt = os.getenv("SAM2_CKPT", "")
    if not sam2_ckpt:
        log.error("SAM2_CKPT env var not set — source .env before running")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── PyBullet sim ──────────────────────────────────────────────────────────
    log.info("Starting PyBullet (gui=%s)", not args.no_gui)
    robot = XArmPybulletInterface(use_gui=not args.no_gui)
    robot.set_current_joint_state(args.joint_state)
    cam_pos, cam_rot = robot.get_camera_transform()
    if cam_pos is not None:
        log.info("Camera pos=%s  quat=%s",
                 np.round(cam_pos, 3).tolist(),
                 np.round(cam_rot.as_quat(), 3).tolist())
    else:
        log.warning("FK returned no camera transform — corridors in camera frame")

    # ── RealSense ─────────────────────────────────────────────────────────────
    log.info("Starting RealSense camera")
    camera = RealSenseCamera(width=640, height=480, fps=30, enable_depth=True)
    intrinsics = camera.get_camera_intrinsics()
    log.info("Camera ready  fx=%.1f  fy=%.1f", intrinsics.fx, intrinsics.fy)

    # ── GSAM2 detection ───────────────────────────────────────────────────────
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

    # Build robot_state so detect_objects can transform position_3d into base frame.
    robot_state: Optional[Dict[str, Any]] = None
    if cam_pos is not None and cam_rot is not None:
        robot_state = {
            "camera": {
                "position": cam_pos.tolist(),
                "quaternion_xyzw": cam_rot.as_quat().tolist(),
            }
        }

    asyncio.run(tracker.detect_objects(
        color_frame=color,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        robot_state=robot_state,
    ))

    masks    = dict(tracker._last_masks)
    detected = list(tracker.registry.get_all_objects())

    if not detected:
        log.error("No objects detected — check camera and lighting")
        camera.stop(); robot.cleanup(); sys.exit(1)

    log.info("Depth range [%.3f, %.3f] m", depth.min(), depth.max())

    # ── Object selection ──────────────────────────────────────────────────────
    detected_ids = [o.object_id for o in detected]

    if args.object is not None:
        if args.object in detected_ids:
            detected = [o for o in detected if o.object_id == args.object]
        else:
            log.error("--object '%s' not in detected list: %s", args.object, detected_ids)
            camera.stop(); robot.cleanup(); sys.exit(1)
    else:
        print()
        print("Detected objects:")
        for i, oid in enumerate(detected_ids):
            print(f"  [{i}] {oid}")
        print()
        while True:
            try:
                raw = input("Select object (number or name): ").strip()
            except EOFError:
                raw = "0"
            if raw.isdigit() and int(raw) < len(detected_ids):
                detected = [detected[int(raw)]]
                break
            if raw in detected_ids:
                detected = [o for o in detected if o.object_id == raw]
                break
            print(f"  Not recognised — enter 0–{len(detected_ids)-1} or an object ID")

    target_obj = detected[0]
    target_id  = target_obj.object_id
    log.info("Target: %s", target_id)

    # ── Depth → PyBullet collision meshes ─────────────────────────────────────
    log.info("Building depth-mesh collision bodies")
    depth_collider = DepthEnvironmentCollider(robot)
    build_results  = depth_collider.update(camera, masks)
    for label, ok in build_results.items():
        tag = "(background)" if label == BACKGROUND_ID else "(object)"
        if ok:
            log.info("  [%s] %s body_id=%s", label, tag,
                     depth_collider.background_body_id
                     if label == BACKGROUND_ID else depth_collider.body_id_for(label))
        else:
            log.warning("  [%s] %s mesh build FAILED", label, tag)
    robot.attach_collider(depth_collider)

    # ── Clearance profile ─────────────────────────────────────────────────────
    target_mask = masks.get(target_id)
    if target_mask is None or not target_mask.any():
        log.error("Empty mask for %s — cannot compute corridors", target_id)
        camera.stop(); robot.cleanup(); sys.exit(1)

    other_masks = {k: v for k, v in masks.items() if k != target_id}
    gripper     = GripperGeometry()
    log.info("Computing clearance profile (%d px, %d neighbours)...",
             int(target_mask.sum()), sum(1 for v in other_masks.values() if v.any()))

    profile = compute_clearance_profile(
        target_mask=target_mask,
        depth_frame=depth,
        camera_intrinsics=intrinsics,
        all_masks=other_masks,
        gripper=gripper,
        cam_position=cam_pos,
        camera_quaternion_xyzw=cam_rot.as_quat() if cam_rot is not None else None,
    )

    n_ok  = sum(1 for c in profile.approach_corridors if c.grasp_compatible)
    n_blk = len(profile.approach_corridors) - n_ok
    log.info("Corridors: %d total  %d compatible  %d blocked  graspability=%.2f",
             len(profile.approach_corridors), n_ok, n_blk, profile.graspability_score)

    if n_ok == 0:
        log.warning("No grasp-compatible corridors — GraspPlanner will use preset seed only")

    # ── Grasp planning ────────────────────────────────────────────────────────
    # profile.centroid is the target point cloud centroid already transformed to
    # robot base frame by compute_clearance_profile (when cam_position is given).
    # Prefer it over obj.position_3d, which may be in camera frame if the FK
    # transform was unavailable at detection time.
    if profile.centroid is not None:
        target_pos = np.asarray(profile.centroid, float)
        log.info("Target position (from profile centroid, base frame): %s",
                 np.round(target_pos, 4).tolist())
    elif target_obj.position_3d is not None:
        target_pos = np.asarray(target_obj.position_3d, float)
        log.info("Target position (from obj.position_3d): %s",
                 np.round(target_pos, 4).tolist())
    else:
        log.error("No 3D position available for %s", target_id)
        camera.stop(); robot.cleanup(); sys.exit(1)

    log.info("Sampling grasps (seed=%s  cone=%.0f°  max=%d)…",
             args.seed, args.cone, args.max_grasps)

    from src.grasp_planner.grasp_planner import _SEED_ORIENTATIONS, _sample_cone, _manipulability
    import pybullet as pb

    corridor_seeds = corridors_to_seeds(profile, args.seed)
    if not corridor_seeds:
        corridor_seeds = [_SEED_ORIENTATIONS.get(args.seed, _SEED_ORIENTATIONS["top_down"])]

    env_bodies, floor_body = robot._build_collision_bodies(ignore_labels={target_id})
    n_pybullet_links = pb.getNumJoints(robot._robot_id, physicsClientId=robot._physics_client)

    def _col_free(joints: np.ndarray) -> bool:
        return robot._is_state_valid(joints, env_bodies, floor_body, 0.005, n_pybullet_links)

    # Map each seed index to its ApproachCorridor (None for preset fallback seeds).
    free_corridors = [c for c in profile.approach_corridors if c.grasp_compatible]
    seed_to_corridor: Dict[int, Optional[ApproachCorridor]] = {
        i: (free_corridors[i] if i < len(free_corridors) else None)
        for i in range(len(corridor_seeds))
    }

    # Flatten all cone samples across all seeds into a list so we can solve one
    # at a time and pause between each displayed grasp.
    # Each entry: (seed_name, seed_quat, candidate_quat, corridor)
    SeedSample = Tuple[str, np.ndarray, np.ndarray, Optional[ApproachCorridor]]
    all_samples: List[SeedSample] = []
    for seed_idx, seed_quat in enumerate(corridor_seeds):
        seed_name = f"corridor_{seed_idx}" if seed_to_corridor.get(seed_idx) else args.seed
        for candidate_quat in _sample_cone(seed_quat, np.deg2rad(args.cone), n_tilt=4, n_roll=8):
            all_samples.append((seed_name, seed_quat, candidate_quat, seed_to_corridor.get(seed_idx)))

    # ── Interactive sampling loop ─────────────────────────────────────────────
    # Solve IK one candidate at a time, display each result, and wait for input
    # before advancing.  This way the sim pose and terminal output are always in
    # sync with the user's current view.

    client = robot._physics_client
    debug_ids: List[int] = []

    def _display(
        candidate: GraspCandidate,
        corridor: Optional[ApproachCorridor],
        seed_quat: np.ndarray,
        grasp_num: int,
        total_found: int,
        sample_num: int,
        total_samples: int,
    ) -> None:
        nonlocal debug_ids
        _clear(client, debug_ids)
        debug_ids = []

        robot.set_current_joint_state(candidate.joints)
        robot._apply_joints_to_sim()

        col_free, hits = _collision_details(robot, candidate.joints, depth_collider)

        # Collision markers — one sphere on the robot surface, one on the
        # obstacle surface, connected by a line.  Red sphere = robot contact
        # point, orange sphere = obstacle contact point.
        for hit in hits:
            # Robot-side contact: red sphere
            debug_ids.append(p.addUserDebugText(
                "X",
                hit.pt_on_robot.tolist(),
                textColorRGB=[1.0, 0.0, 0.0],
                textSize=1.8,
                physicsClientId=client,
            ))
            # Obstacle-side contact: orange sphere
            debug_ids.append(p.addUserDebugText(
                "X",
                hit.pt_on_other.tolist(),
                textColorRGB=[1.0, 0.5, 0.0],
                textSize=1.8,
                physicsClientId=client,
            ))
            # Line connecting the two contact points
            debug_ids.append(p.addUserDebugLine(
                hit.pt_on_robot.tolist(),
                hit.pt_on_other.tolist(),
                lineColorRGB=[1.0, 0.2, 0.0],
                lineWidth=3.0,
                physicsClientId=client,
            ))

        # Corridor OBB — the free approach volume this seed came from.
        if corridor is not None:
            obb_color = [0.1, 0.9, 0.1] if col_free else [0.9, 0.2, 0.1]
            debug_ids += _draw_corridor(client, corridor, obb_color)

        # Seed TCP frame — dim white axes showing the nominal seed orientation
        # before any cone tilt is applied.  Offset slightly along seed Z so it
        # doesn't overlap the candidate triad exactly when tilt is zero.
        seed_rot = Rotation.from_quat(seed_quat)
        seed_z   = seed_rot.apply(np.array([0.0, 0.0, 1.0]))
        seed_pos = target_pos + seed_z * 0.02   # 2 cm offset along approach axis
        seed_frame_ids = _draw_tcp_frame(
            client, seed_pos, seed_quat, length=0.06,
        )
        # Re-draw seed axes in dim grey to visually distinguish from candidate.
        _clear(client, seed_frame_ids)
        seed_frame_ids = []
        for axis_vec, color in zip(seed_rot.as_matrix().T, [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]):
            tip = seed_pos + axis_vec * 0.06
            seed_frame_ids.append(p.addUserDebugLine(
                seed_pos.tolist(), tip.tolist(),
                lineColorRGB=color, lineWidth=1.5,
                physicsClientId=client,
            ))
        debug_ids += seed_frame_ids

        # Draw a thin line from seed TCP origin to candidate TCP origin so the
        # tilt magnitude is visually obvious.
        debug_ids.append(p.addUserDebugLine(
            seed_pos.tolist(), candidate.position.tolist(),
            lineColorRGB=[0.8, 0.8, 0.2], lineWidth=1.0,
            physicsClientId=client,
        ))

        # Candidate TCP frame — full-brightness RGB axes.
        debug_ids += _draw_tcp_frame(client, candidate.position, candidate.orientation)

        status = "OK" if col_free else "COLLISION"
        label_color = [0.1, 0.9, 0.1] if col_free else [0.9, 0.2, 0.1]
        debug_ids.append(_draw_text(
            client,
            candidate.position + np.array([0.0, 0.0, 0.06]),
            f"[grasp {grasp_num}  sample {sample_num}/{total_samples}] "
            f"{status}  {candidate.seed_orientation}  "
            f"{np.degrees(candidate.approach_angle_rad):.1f}deg",
            label_color,
        ))

        _print_grasp_report(grasp_num, total_found, candidate, corridor, col_free, hits)

    print()
    print(_BOLD + _BLUE + "=" * 72 + _RESET)
    print(_BOLD + _BLUE + f"  Grasp Sampler — {target_id}  "
          f"({len(all_samples)} orientation samples across "
          f"{len(corridor_seeds)} seed(s))" + _RESET)
    print(_BOLD + _BLUE + "=" * 72 + _RESET)
    print(f"  {_CYAN}Enter/N{_RESET} = next sample   "
          f"{_CYAN}S{_RESET} = skip to next seed   "
          f"{_CYAN}Q{_RESET} = quit")

    found: List[Tuple[GraspCandidate, Optional[ApproachCorridor]]] = []
    quit_requested = False
    sample_idx = 0

    while sample_idx < len(all_samples) and len(found) < args.max_grasps:
        seed_name, seed_quat, candidate_quat, corridor = all_samples[sample_idx]

        joints = robot._ik_solve(
            target_pos=target_pos,
            target_quat=candidate_quat,
            position_tolerance=0.02,
            orientation_tolerance_rad=0.6,
            collision_check_fn=_col_free,
        )

        if joints is not None:
            score = _manipulability(robot, joints)
            tilt = float(
                (Rotation.from_quat(candidate_quat)
                 * Rotation.from_quat(seed_quat).inv()).magnitude()
            )
            candidate = GraspCandidate(
                position=target_pos.copy(),
                orientation=candidate_quat,
                joints=joints,
                approach_angle_rad=tilt,
                manipulability=score,
                seed_orientation=seed_name,
            )
            found.append((candidate, corridor))
            _display(candidate, corridor, seed_quat, len(found), len(found), sample_idx + 1, len(all_samples))

            try:
                raw = input(
                    f"\n  [Enter/N=next  S=skip seed  Q=quit]"
                    f"  sample {sample_idx+1}/{len(all_samples)}"
                    f"  found {len(found)} > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                quit_requested = True
                break

            if raw == "q":
                quit_requested = True
                break
            elif raw == "s":
                # Skip remaining samples from the current seed.
                current_seed = seed_name
                while (sample_idx + 1 < len(all_samples)
                       and all_samples[sample_idx + 1][0] == current_seed):
                    sample_idx += 1
        else:
            log.debug("sample %d/%d  IK failed — skipping", sample_idx + 1, len(all_samples))

        sample_idx += 1

    _clear(client, debug_ids)
    debug_ids = []

    if not found:
        log.error("No collision-free grasps found for %s", target_id)
        camera.stop()
        if args.no_gui:
            robot.cleanup()
        else:
            log.info("PyBullet window left open — close manually to exit")
            try:
                while True:
                    p.stepSimulation(physicsClientId=robot._physics_client); time.sleep(1/240)
            except Exception:
                pass
            robot.cleanup()
        sys.exit(1)

    _print_separator()
    print(f"  {_GREEN}Done.{_RESET}  Found {len(found)} grasp candidates for {target_id} "
          f"across {sample_idx}/{len(all_samples)} samples.")

    camera.stop()
    if args.no_gui:
        robot.cleanup()
    else:
        log.info("PyBullet window left open — close manually to exit")
        try:
            while True:
                p.stepSimulation(physicsClientId=robot._physics_client)
                time.sleep(1.0 / 240.0)
        except Exception:
            pass
        robot.cleanup()


if __name__ == "__main__":
    main()
