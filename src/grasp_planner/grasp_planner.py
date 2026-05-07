"""
Antipodal grasp planner.

Samples gripper orientations by rotating a seed orientation about the
approach axis (TCP z-axis), evaluates each candidate with TracIK +
collision checking, and returns the best collision-free grasp.

Usage:
    planner = GraspPlanner(robot_interface)
    candidate = planner.plan(target_pos, seed_orientation="top_down")
    if candidate:
        # candidate.orientation is the collision-free xyzw quaternion
        # candidate.joints is the IK solution

Clearance-seeded usage:
    profile = obj.clearance_profile   # from GSAM2 tracker
    candidate = planner.plan(target_pos, seed_orientation="top_down",
                             clearance_profile=profile)
    # Free corridors closest to top_down are prepended as extra_seeds so
    # the sampler tries physically-open directions first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from src.grasp_planner.grasp_candidate import GraspCandidate

if TYPE_CHECKING:
    from src.perception.clearance import ClearanceProfile

logger = logging.getLogger(__name__)

# Preset seed orientations (xyzw). These define the nominal approach direction;
# antipodal sampling rotates around the approach axis from each seed.
_SEED_ORIENTATIONS = {
    "top_down": np.array([-0.9983, 0.0314, 0.0438, 0.0223]),
    "side":     np.array([-0.6894, 0.0305, -0.7237, 0.0033]),
}


def corridors_to_seeds(
    profile: "ClearanceProfile",
    seed_orientation: str = "top_down",
    max_seeds: int = 6,
) -> List[np.ndarray]:
    """Convert free clearance corridors to gripper-orientation quaternions.

    Sign convention:
      - TCP Z aligns with  -direction   (gripper enters the corridor; corridor
        direction points *away* from the object toward the robot).
      - TCP Y aligns with  +grasp_axis  (the finger-spread axis, i.e. the wide
        face of the corridor OBB, so the gripper opens across the clear volume).

    Seeds are sorted by alignment of -direction with the nominal seed's TCP Z
    so the corridor closest to the requested approach style is tried first.

    Args:
        profile: ClearanceProfile from compute_clearance_profile / obj.clearance_profile.
        seed_orientation: Key into _SEED_ORIENTATIONS used to rank corridors.
        max_seeds: Maximum number of corridor seeds to return.

    Returns:
        List of xyzw quaternions, closest to seed_orientation first.
    """
    free = [c for c in profile.approach_corridors if c.grasp_compatible]
    if not free:
        return []

    nominal_quat = _SEED_ORIENTATIONS.get(seed_orientation,
                                          _SEED_ORIENTATIONS["top_down"])
    nominal_rot = Rotation.from_quat(nominal_quat)
    nominal_tcp_z = nominal_rot.apply(np.array([0.0, 0.0, 1.0]))

    seeds: List[np.ndarray] = []
    for corridor in free:
        direction  = np.asarray(corridor.direction,  dtype=float)
        grasp_axis = np.asarray(corridor.grasp_axis, dtype=float)

        # TCP Z points opposite to corridor direction (gripper enters from outside).
        # TCP Y aligns with grasp_axis — the finger-spread axis, i.e. the wide face
        # of the corridor OBB — so the gripper opens across the clear volume.
        tcp_z_target = -direction
        tcp_y_target =  grasp_axis

        try:
            rot, _ = Rotation.align_vectors(
                [tcp_z_target, tcp_y_target],
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                weights=[10.0, 1.0],
            )
            # Alignment score: dot of resulting TCP Z with nominal TCP Z.
            # Higher = closer to the requested approach style.
            alignment = float(np.dot(rot.apply([0.0, 0.0, 1.0]), nominal_tcp_z))
            seeds.append((rot.as_quat(), alignment))
        except Exception:
            continue

    if not seeds:
        return []

    seeds.sort(key=lambda t: -t[1])   # best alignment first
    seeds = seeds[:max_seeds]

    logger.debug(
        "corridors_to_seeds: %d free corridors → %d seeds (seed=%s, top alignment=%.3f)",
        len(free), len(seeds), seed_orientation, seeds[0][1],
    )
    return [quat for quat, _ in seeds]


def _sample_cone(
    base_quat: np.ndarray,
    cone_half_angle_rad: float,
    n_tilt: int,
    n_roll: int,
) -> list[np.ndarray]:
    """Sample orientations within a spherical cone around base_quat.

    Distributes n_tilt tilt rings from 0 → cone_half_angle_rad, sweeping
    n_roll evenly-spaced roll angles per ring around the approach axis.

    Returns a flat list of xyzw quaternions (seed always included first).
    """
    base_rot = Rotation.from_quat(base_quat)
    samples: list[np.ndarray] = [base_quat.copy()]

    approach = base_rot.apply(np.array([0.0, 0.0, 1.0]))
    perp = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(approach, perp)
    perp /= np.linalg.norm(perp)

    tilt_angles = np.linspace(0, cone_half_angle_rad, n_tilt + 1)[1:]
    roll_angles = np.linspace(0, 2 * np.pi, n_roll, endpoint=False)

    for tilt in tilt_angles:
        for roll in roll_angles:
            tilt_axis = Rotation.from_rotvec(approach * roll).apply(perp)
            tilt_rot = Rotation.from_rotvec(tilt_axis * tilt)
            samples.append((tilt_rot * base_rot).as_quat())

    return samples


def _manipulability(robot_interface: Any, joints: np.ndarray) -> float:
    """Yoshikawa manipulability score via PyBullet Jacobian.

    Returns 0.0 if the interface doesn't support Jacobian queries.
    """
    try:
        import pybullet as p
        rid = robot_interface._robot_id
        client = robot_interface._physics_client
        tcp_idx = robot_interface._get_link_index(robot_interface.tcp_link_name)
        if tcp_idx is None:
            return 0.0

        # Set arm joints; leave other movable joints at their current state.
        for i, idx in enumerate(robot_interface._movable_joints[:len(joints)]):
            p.resetJointState(rid, idx, float(joints[i]), physicsClientId=client)

        # calculateJacobian requires the full movable-joint state vector.
        n_movable = p.getNumJoints(rid, physicsClientId=client)
        full_state = [
            p.getJointState(rid, j, physicsClientId=client)[0]
            for j in range(n_movable)
        ]
        zero = [0.0] * n_movable
        jac_t, jac_r, _ = p.calculateJacobian(
            rid, tcp_idx, [0, 0, 0],
            full_state, zero, zero,
            physicsClientId=client,
        )
        # Slice to arm-only columns so J is (6 x dof).
        dof = len(joints)
        J = np.vstack([np.array(jac_t)[:, :dof], np.array(jac_r)[:, :dof]])
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
    except Exception:
        return 0.0


class GraspPlanner:
    """Spherical-cone grasp sampler backed by TracIK + PyBullet collision checking.

    Samples candidate gripper orientations within a cone around the seed
    orientation: n_tilt tilt rings × n_roll in-plane rotations per ring,
    plus the seed itself. Evaluates each with IK + collision and returns the
    highest-manipulability collision-free candidate.

    Args:
        robot_interface: A BasePybulletInterface subclass.
        cone_half_angle_deg: Half-angle of the sampling cone in degrees.
        n_tilt: Number of tilt rings between 0 and cone_half_angle_deg.
        n_roll: Number of in-plane (roll) samples per tilt ring.
        position_tolerance: IK position tolerance in metres.
        orientation_tolerance_rad: IK orientation tolerance in radians.
        collision_margin: Collision check clearance in metres.
    """

    def __init__(
        self,
        robot_interface: Any,
        cone_half_angle_deg: float = 45.0,
        n_tilt: int = 4,
        n_roll: int = 8,
        position_tolerance: float = 0.02,
        orientation_tolerance_rad: float = 0.6,
        collision_margin: float = 0.005,
    ) -> None:
        self._robot = robot_interface
        self._cone_rad = np.deg2rad(cone_half_angle_deg)
        self._n_tilt = n_tilt
        self._n_roll = n_roll
        self._pos_tol = position_tolerance
        self._ori_tol = orientation_tolerance_rad
        self._col_margin = collision_margin

    def plan(
        self,
        target_pos: np.ndarray,
        seed_orientation: str = "top_down",
        extra_seeds: Optional[list] = None,
        clearance_profile: Optional["ClearanceProfile"] = None,
        ignore_labels: Optional[set] = None,
    ) -> Optional[GraspCandidate]:
        """Find the best collision-free grasp within the orientation cone.

        When a clearance_profile is supplied, free corridors are converted to
        quaternion seeds via corridors_to_seeds() and prepended before any
        extra_seeds.  They are sorted by angular proximity to seed_orientation
        so the closest open approach is tried first, before the nominal preset.

        Args:
            target_pos: Desired TCP position in robot base frame (3,).
            seed_orientation: Key into preset orientations ("top_down"/"side").
            extra_seeds: Additional xyzw quaternions to sample cones around.
            clearance_profile: Optional ClearanceProfile from the perception
                system.  Compatible corridors are injected as leading seeds.

        Returns:
            Best GraspCandidate by manipulability, or None if none found.
        """
        target_pos = np.asarray(target_pos, dtype=float)

        seeds: list[tuple[str, np.ndarray]] = []
        if seed_orientation in _SEED_ORIENTATIONS:
            seeds.append((seed_orientation, _SEED_ORIENTATIONS[seed_orientation]))
        else:
            logger.warning("Unknown seed orientation '%s' — using top_down", seed_orientation)
            seeds.append(("top_down", _SEED_ORIENTATIONS["top_down"]))

        # When a clearance profile is available and has free corridors, replace
        # the preset seed entirely — the preset may point into a blocked direction
        # and wasting IK attempts there causes the collision spam seen in logs.
        # Only fall back to the preset when no free corridor exists at all.
        if clearance_profile is not None:
            corridor_seeds = corridors_to_seeds(clearance_profile, seed_orientation)
            if corridor_seeds:
                logger.info(
                    "GraspPlanner: using %d corridor seed(s) from clearance profile "
                    "(dropping preset '%s' — clearance corridors are authoritative)",
                    len(corridor_seeds), seed_orientation,
                )
                seeds = [(f"corridor_{i}", q) for i, q in enumerate(corridor_seeds)]
            else:
                logger.warning(
                    "GraspPlanner: clearance profile has no free corridors — "
                    "falling back to preset seed '%s'",
                    seed_orientation,
                )

        for i, quat in enumerate(extra_seeds or []):
            seeds.append((f"extra_{i}", np.asarray(quat, dtype=float)))

        import pybullet as p
        env_bodies, floor_body = self._robot._build_collision_bodies(ignore_labels=ignore_labels)
        n_links = p.getNumJoints(self._robot._robot_id, physicsClientId=self._robot._physics_client)

        def _collision_free(joints: np.ndarray) -> bool:
            return self._robot._is_state_valid(
                joints, env_bodies, floor_body, self._col_margin, n_links
            )

        best: Optional[GraspCandidate] = None
        best_score = -1.0
        n_tried = n_ik_ok = n_col_free = 0

        for seed_name, seed_quat in seeds:
            candidates = _sample_cone(seed_quat, self._cone_rad, self._n_tilt, self._n_roll)
            for candidate_quat in candidates:
                joints = self._robot._ik_solve(
                    target_pos=target_pos,
                    target_quat=candidate_quat,
                    position_tolerance=self._pos_tol,
                    orientation_tolerance_rad=self._ori_tol,
                    collision_check_fn=_collision_free,
                )
                n_tried += 1
                if joints is None:
                    continue
                n_ik_ok += 1
                if not _collision_free(joints):
                    continue
                n_col_free += 1

                score = _manipulability(self._robot, joints)
                if score > best_score:
                    best_score = score
                    tilt = float(
                        (Rotation.from_quat(candidate_quat)
                         * Rotation.from_quat(seed_quat).inv()).magnitude()
                    )
                    best = GraspCandidate(
                        position=target_pos.copy(),
                        orientation=candidate_quat,
                        joints=joints,
                        approach_angle_rad=tilt,
                        manipulability=score,
                        seed_orientation=seed_name,
                    )

        logger.info(
            "GraspPlanner: %d candidates tried, %d IK solved, %d collision-free",
            n_tried, n_ik_ok, n_col_free,
        )
        if best is None:
            logger.warning(
                "GraspPlanner: no collision-free grasp found at pos=%s seed=%s",
                np.round(target_pos, 3).tolist(), seed_orientation,
            )
        else:
            logger.info(
                "GraspPlanner: best grasp tilt=%.1f° seed=%s manipulability=%.4f",
                np.degrees(best.approach_angle_rad), best.seed_orientation, best.manipulability,
            )

        return best
