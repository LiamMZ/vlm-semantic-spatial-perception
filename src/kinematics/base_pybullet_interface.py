"""
Base PyBullet Simulation Interface

General-purpose base class for PyBullet FK/transform computation.
Robot-specific subclasses provide the URDF path and link names.

Duck-typed interface (same signatures as CuRoboMotionPlanner):
  get_robot_joint_state()      -> np.ndarray | None
  get_robot_tcp_pose()         -> (pos, quat_xyzw) | None
  get_camera_transform()       -> (pos, Rotation) | (None, None)
  get_robot_state()            -> dict
  get_forward_kinematics()     -> (pos, quat_wxyz) | None
  plan_joint_trajectory_to_pose() -> np.ndarray[N, dof] | None
  move_to_pose()               -> (success, trajectory, dt)
  convert_cam_pose_to_base()   -> (pos, quat_xyzw)
  set_current_joint_state()    -> None

No robot connection, no CUDA, no curobo required.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None

try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
except ImportError:
    OMPL_AVAILABLE = False
    ob = None
    og = None

logger = logging.getLogger(__name__)


class BasePybulletInterface:
    """
    General PyBullet kinematics interface.

    Maintains a simulated joint state (no physical robot). FK and camera
    transform queries run against the loaded URDF.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path],
        camera_link_name: str,
        tcp_link_name: str,
        base_link_name: str,
        n_arm_joints: int,
        static_camera_tf: Optional[Any] = None,
        use_gui: bool = False,
    ) -> None:
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is required. Install with: pip install pybullet"
            )

        self.urdf_path = str(urdf_path)
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.camera_link_name = camera_link_name
        self.tcp_link_name = tcp_link_name
        self.base_link_name = base_link_name

        self._physics_client: Optional[int] = None
        self._robot_id: Optional[int] = None
        self._floor_body_id: Optional[int] = None
        self._movable_joints: List[int] = []  # indices of non-fixed joints
        self._link_name_to_index: Dict[str, int] = {}
        self._joint_name_to_index: Dict[str, int] = {}
        self._status_text_ids: List[int] = []
        self._use_gui = bool(use_gui)
        # Optional depth-image collision checker; set via attach_collider().
        self._collider: Optional[Any] = None
        # First link index to include in floor-plane collision checks.
        # Set to 1 to skip link_base (index 0) whose geometry is embedded in the
        # floor plane because the robot is physically bolted to the ground.
        self._floor_check_start_link: int = 1
        # Last arm link index (inclusive) for self-collision checking.
        # Links beyond this index belong to the end-effector / gripper /
        # camera assembly and have constant geometry — no need to self-check them.
        self._arm_self_collision_end_link: int = n_arm_joints

        # Current simulated joint state
        self._joints = np.zeros(n_arm_joints, dtype=np.float64)

        # Static camera transform (optional, same semantics as CuRoboMotionPlanner)
        self.static_camera_tf: Optional[np.ndarray] = None
        self.static_camera_position: Optional[np.ndarray] = None
        self.static_camera_rotation: Optional[Rotation] = None
        if static_camera_tf is not None:
            self._parse_static_camera_tf(static_camera_tf)

        self._init_pybullet()
        logger.info("%s ready (URDF=%s)", type(self).__name__, self.urdf_path)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_pybullet(self) -> None:
        if self._physics_client is not None:
            return
        connection_mode = p.GUI if self._use_gui else p.DIRECT
        self._physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._floor_body_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client)
        if self._use_gui:
            p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI,
                0,
                physicsClientId=self._physics_client,
            )
        self._robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            physicsClientId=self._physics_client,
        )
        self._build_joint_map()
        logger.debug(
            "PyBullet %s loaded robot_id=%d",
            "GUI" if self._use_gui else "DIRECT",
            self._robot_id,
        )

    def _build_joint_map(self) -> None:
        """Populate movable joint list and link-name → index map."""
        num = p.getNumJoints(self._robot_id)
        self._movable_joints = []
        self._link_name_to_index = {}
        self._joint_name_to_index = {}
        for i in range(num):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._physics_client)
            joint_name: str = info[1].decode("utf-8")
            link_name: str = info[12].decode("utf-8")
            joint_type: int = info[2]
            self._joint_name_to_index[joint_name] = i
            self._link_name_to_index[link_name] = i
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self._movable_joints.append(i)

    def _get_link_index(self, link_name: str) -> Optional[int]:
        if link_name == self.base_link_name:
            return -1
        return self._link_name_to_index.get(link_name)

    def _get_joint_index(self, joint_name: str) -> Optional[int]:
        return self._joint_name_to_index.get(joint_name)

    def _apply_joints_to_sim(self) -> None:
        """Push self._joints into PyBullet for FK computation."""
        for i, joint_idx in enumerate(self._movable_joints):
            if i < len(self._joints):
                p.resetJointState(
                    self._robot_id,
                    joint_idx,
                    float(self._joints[i]),
                    physicsClientId=self._physics_client,
                )

    def set_sim_joint_state_by_name(self, joint_name: str, value: float) -> bool:
        """Set a specific PyBullet joint by URDF joint name."""
        joint_idx = self._get_joint_index(joint_name)
        if joint_idx is None:
            return False
        p.resetJointState(
            self._robot_id,
            joint_idx,
            float(value),
            physicsClientId=self._physics_client,
        )
        return True

    def _get_link_world_pose(
        self, link_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion_xyzw) of a link in world frame."""
        if link_index == -1:
            pos, orn = p.getBasePositionAndOrientation(
                self._robot_id,
                physicsClientId=self._physics_client,
            )
        else:
            state = p.getLinkState(
                self._robot_id,
                link_index,
                physicsClientId=self._physics_client,
            )
            pos = state[4]
            orn = state[5]
        return np.array(pos, dtype=float), np.array(orn, dtype=float)

    # ------------------------------------------------------------------
    # Static camera transform (same as CuRoboMotionPlanner)
    # ------------------------------------------------------------------

    def _parse_static_camera_tf(self, static_camera_tf: Any) -> None:
        if isinstance(static_camera_tf, np.ndarray) and static_camera_tf.shape == (4, 4):
            self.static_camera_tf = static_camera_tf
            self.static_camera_position = static_camera_tf[:3, 3]
            self.static_camera_rotation = Rotation.from_matrix(static_camera_tf[:3, :3])
        elif isinstance(static_camera_tf, (tuple, list)) and len(static_camera_tf) == 2:
            pos, rot = static_camera_tf
            self.static_camera_position = np.array(pos, dtype=float)
            if isinstance(rot, Rotation):
                self.static_camera_rotation = rot
            elif isinstance(rot, np.ndarray) and rot.shape == (3, 3):
                self.static_camera_rotation = Rotation.from_matrix(rot)
            elif isinstance(rot, (list, tuple, np.ndarray)) and len(rot) == 4:
                self.static_camera_rotation = Rotation.from_quat(np.array(rot))
            else:
                logger.warning("Unrecognised rotation format in static_camera_tf")
                return
            # Build 4x4 matrix for reference
            T = np.eye(4)
            T[:3, :3] = self.static_camera_rotation.as_matrix()
            T[:3, 3] = self.static_camera_position
            self.static_camera_tf = T
        else:
            logger.warning("Unrecognised static_camera_tf format, ignoring")

    # ------------------------------------------------------------------
    # Duck-typed robot interface
    # ------------------------------------------------------------------

    def set_current_joint_state(self, joint_positions: Any) -> None:
        """Set the simulated joint state."""
        arr = np.asarray(joint_positions, dtype=float).flatten()
        self._joints = arr[: len(self._movable_joints)]

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return current simulated joint positions (radians)."""
        return self._joints.copy()

    def get_robot_tcp_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (position, quaternion_xyzw) of the TCP in robot base frame.

        Uses PyBullet FK at the current joint state.
        """
        try:
            self._apply_joints_to_sim()
            tcp_idx = self._get_link_index(self.tcp_link_name)
            base_idx = self._get_link_index(self.base_link_name)

            if tcp_idx is None:
                logger.warning("TCP link '%s' not found in URDF", self.tcp_link_name)
                return None

            tcp_pos_w, tcp_orn_w = self._get_link_world_pose(tcp_idx)
            base_pos_w, base_orn_w = self._get_link_world_pose(base_idx)

            T_w_tcp = _build_T(tcp_pos_w, tcp_orn_w)
            T_w_base = _build_T(base_pos_w, base_orn_w)
            T_base_tcp = np.linalg.inv(T_w_base) @ T_w_tcp

            pos = T_base_tcp[:3, 3]
            rot = Rotation.from_matrix(T_base_tcp[:3, :3])
            quat_xyzw = rot.as_quat()  # [x, y, z, w]
            return pos, quat_xyzw
        except Exception:
            logger.exception("get_robot_tcp_pose failed")
            return None

    def get_camera_transform(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[Rotation]]:
        """
        Return (position, Rotation) of the camera link in robot base frame.

        Matches CuRoboMotionPlanner.get_camera_transform() signature.
        If a static transform was set, that is returned directly.
        Otherwise computes via FK.
        """
        if self.static_camera_tf is not None:
            return self.static_camera_position, self.static_camera_rotation

        try:
            self._apply_joints_to_sim()
            cam_idx = self._get_link_index(self.camera_link_name)
            base_idx = self._get_link_index(self.base_link_name)

            if cam_idx is None:
                logger.warning("Camera link '%s' not found in URDF", self.camera_link_name)
                return None, None

            cam_pos_w, cam_orn_w = self._get_link_world_pose(cam_idx)
            base_pos_w, base_orn_w = self._get_link_world_pose(base_idx)

            T_w_cam = _build_T(cam_pos_w, cam_orn_w)
            T_w_base = _build_T(base_pos_w, base_orn_w)
            T_base_cam = np.linalg.inv(T_w_base) @ T_w_cam

            pos = T_base_cam[:3, 3]
            rot = Rotation.from_matrix(T_base_cam[:3, :3])
            return pos, rot
        except Exception:
            logger.exception("get_camera_transform failed")
            return None, None

    def get_forward_kinematics(
        self, joint_positions: Optional[Any] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute FK for given joint positions (or current state if None).

        Returns:
            (position, quaternion_wxyz) matching CuRoboMotionPlanner convention,
            or None on failure.
        """
        try:
            if joint_positions is not None:
                saved = self._joints.copy()
                self.set_current_joint_state(joint_positions)
                result = self.get_robot_tcp_pose()
                self._joints = saved
            else:
                result = self.get_robot_tcp_pose()

            if result is None:
                return None
            pos, quat_xyzw = result
            # CuRoboMotionPlanner returns wxyz convention
            x, y, z, w = quat_xyzw
            quat_wxyz = np.array([w, x, y, z], dtype=float)
            return pos, quat_wxyz
        except Exception:
            logger.exception("get_forward_kinematics failed")
            return None

    def get_robot_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable state dict.

        Same structure as CuRoboMotionPlanner.get_robot_state() so
        ObjectTracker snapshots stay consistent.
        """
        state: Dict[str, Any] = {
            "stamp": time.time(),
            "provider": type(self).__name__,
        }

        joints = self.get_robot_joint_state()
        if joints is not None:
            state["joints"] = joints.tolist()

        tcp = self.get_robot_tcp_pose()
        if isinstance(tcp, tuple) and len(tcp) == 2:
            pos, quat = tcp
            state["tcp_pose"] = {
                "position": pos.tolist() if pos is not None else None,
                "quaternion_xyzw": quat.tolist() if quat is not None else None,
            }

        cam_pos, cam_rot = self.get_camera_transform()
        if cam_pos is not None:
            state["camera"] = {
                "position": cam_pos.tolist(),
                "quaternion_xyzw": cam_rot.as_quat().tolist() if cam_rot is not None else None,
            }

        if self.static_camera_position is not None:
            state["static_camera"] = {
                "position": self.static_camera_position.tolist(),
                "quaternion_xyzw": self.static_camera_rotation.as_quat().tolist()
                if self.static_camera_rotation is not None
                else None,
            }

        return state

    def attach_collider(self, collider: Any) -> None:
        """Register a DepthEnvironmentCollider for trajectory collision checking."""
        self._collider = collider

    def _ik_solve(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        position_tolerance: float,
        orientation_tolerance_rad: float,
    ) -> Optional[np.ndarray]:
        """Run PyBullet IK and validate the result against pose tolerances.

        Returns the goal joint configuration (dof,) or None on failure.
        """
        tcp_idx = self._get_link_index(self.tcp_link_name)
        if tcp_idx is None:
            logger.warning("TCP link '%s' not found in URDF", self.tcp_link_name)
            return None

        self._apply_joints_to_sim()
        if target_quat is None:
            ik_solution = p.calculateInverseKinematics(
                bodyUniqueId=self._robot_id,
                endEffectorLinkIndex=tcp_idx,
                targetPosition=target_pos.tolist(),
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=self._physics_client,
            )
        else:
            ik_solution = p.calculateInverseKinematics(
                bodyUniqueId=self._robot_id,
                endEffectorLinkIndex=tcp_idx,
                targetPosition=target_pos.tolist(),
                targetOrientation=target_quat.tolist(),
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=self._physics_client,
            )

        if not ik_solution:
            logger.warning("IK solver returned no solution")
            return None

        goal_joints = np.asarray(ik_solution[: len(self._joints)], dtype=float)

        saved = self._joints.copy()
        try:
            self._joints = goal_joints
            reached = self.get_robot_tcp_pose()
        finally:
            self._joints = saved

        if reached is None:
            return None

        reached_pos, reached_quat = reached
        pos_error = float(np.linalg.norm(reached_pos - target_pos))
        if pos_error > position_tolerance:
            logger.warning("IK rejected: pos error %.4f > %.4f", pos_error, position_tolerance)
            return None

        if target_quat is not None:
            ang_error = float(
                (Rotation.from_quat(target_quat).inv() * Rotation.from_quat(reached_quat)).magnitude()
            )
            if ang_error > orientation_tolerance_rad:
                logger.warning(
                    "IK rejected: orient error %.4f rad > %.4f rad",
                    ang_error, orientation_tolerance_rad,
                )
                return None

        return goal_joints

    def _is_state_valid(
        self,
        joints: np.ndarray,
        env_bodies: List[int],
        floor_body: Optional[int],
        collision_margin: float,
        n_links: int,
    ) -> bool:
        """PyBullet collision check for a single joint configuration.

        Checks the robot against depth-mesh environment bodies, against the
        floor plane (skipping base-mounted links), and against itself
        (non-adjacent link pairs only).

        Args:
            joints: (dof,) joint configuration in radians.
            env_bodies: PyBullet body IDs for depth-mesh obstacles.
            floor_body: PyBullet body ID of the floor plane, or None.
            collision_margin: Contact distance threshold in metres.
            n_links: Number of joints/links in the robot URDF.

        Returns:
            True if the configuration is collision-free.
        """
        client = self._physics_client
        rid = self._robot_id

        for j, joint_idx in enumerate(self._movable_joints):
            if j < len(joints):
                p.resetJointState(rid, joint_idx, float(joints[j]), physicsClientId=client)

        # Robot vs depth-mesh environment bodies.
        for body_id in env_bodies:
            for link_idx in range(-1, n_links):
                if p.getClosestPoints(
                    bodyA=rid, bodyB=body_id,
                    distance=collision_margin,
                    linkIndexA=link_idx,
                    physicsClientId=client,
                ):
                    return False

        # Robot vs floor plane — skip base-mounted links (their geometry
        # straddles the floor in the URDF since the robot is bolted down).
        if floor_body is not None:
            for link_idx in range(self._floor_check_start_link, n_links):
                if p.getClosestPoints(
                    bodyA=rid, bodyB=floor_body,
                    distance=collision_margin,
                    linkIndexA=link_idx,
                    physicsClientId=client,
                ):
                    return False

        # Self-collision — arm links only, skipping adjacent pairs.
        # Gripper/camera links (index > _arm_self_collision_end_link) have fixed
        # relative geometry and produce spurious hits from PyBullet mimic joints.
        end = self._arm_self_collision_end_link
        for link_a in range(-1, end):
            for link_b in range(link_a + 2, end + 1):
                if p.getClosestPoints(
                    bodyA=rid, bodyB=rid,
                    distance=0.0,
                    linkIndexA=link_a, linkIndexB=link_b,
                    physicsClientId=client,
                ):
                    return False

        return True

    def _build_collision_bodies(self) -> Tuple[List[int], Optional[int]]:
        """Collect PyBullet body IDs for planning collision checks.

        Returns:
            (env_bodies, floor_body) — env_bodies are depth-mesh obstacle bodies;
            floor_body is the plane body ID (or None).
        """
        env_bodies: List[int] = []
        if self._collider is not None:
            env_bodies.extend(self._collider._bodies.values())
        return env_bodies, self._floor_body_id

    def plan_joint_trajectory_to_pose(
        self,
        target_position: Sequence[float],
        target_orientation: Optional[Sequence[float]] = None,
        position_tolerance: float = 0.02,
        orientation_tolerance_rad: float = 0.2,
        planning_time: float = 5.0,
        collision_margin: float = 0.005,
        interpolate_n: int = 64,
    ) -> Optional[np.ndarray]:
        """Plan a collision-free joint-space trajectory to a target TCP pose.

        Uses PyBullet IK to find the goal joint configuration, then runs
        OMPL RRTConnect in joint space with a PyBullet collision validity
        checker.  Falls back to linear interpolation if OMPL is unavailable.

        Args:
            target_position: Desired TCP position in the robot base frame.
            target_orientation: Desired TCP orientation as xyzw quaternion.
                If ``None``, IK solves for position only.
            position_tolerance: Maximum allowed final TCP position error (m).
            orientation_tolerance_rad: Maximum allowed final TCP angular error (rad).
            planning_time: OMPL planning time budget in seconds.
            collision_margin: Contact distance threshold for the validity checker (m).
            interpolate_n: Number of waypoints to interpolate from the OMPL path.

        Returns:
            Array of shape ``(N, dof)`` waypoints in radians, or ``None`` on failure.
        """
        target_pos = np.asarray(target_position, dtype=float).reshape(-1)
        if target_pos.shape[0] != 3:
            raise ValueError(f"target_position must have shape (3,), got {target_pos.shape}")

        target_quat: Optional[np.ndarray] = None
        if target_orientation is not None:
            target_quat = np.asarray(target_orientation, dtype=float).reshape(-1)
            if target_quat.shape[0] != 4:
                raise ValueError(f"target_orientation must have shape (4,), got {target_quat.shape}")
            norm = float(np.linalg.norm(target_quat))
            if norm < 1e-9:
                raise ValueError("target_orientation has near-zero norm")
            target_quat = target_quat / norm

        # Sync PyBullet to the current joint state before anything else.
        # self._joints may lag PyBullet if external callers reset joints directly,
        # so push self._joints to sim first, then snapshot as start.
        self._apply_joints_to_sim()
        start_joints = self._joints.copy()
        dof = len(start_joints)

        goal_joints = self._ik_solve(target_pos, target_quat, position_tolerance, orientation_tolerance_rad)
        if goal_joints is None:
            return None

        # IK validation temporarily sets self._joints = goal_joints and calls
        # _apply_joints_to_sim for FK — restore PyBullet to start before OMPL.
        self._apply_joints_to_sim()

        if not OMPL_AVAILABLE:
            logger.warning("OMPL not available — using linear interpolation (no collision checking)")
            return np.linspace(start_joints, goal_joints, num=max(10, interpolate_n), dtype=float)

        # --- OMPL setup ---
        space = ob.RealVectorStateSpace(dof)
        bounds = ob.RealVectorBounds(dof)
        joint_limits = self._get_joint_limits()
        for i in range(dof):
            bounds.setLow(i, joint_limits[i][0])
            bounds.setHigh(i, joint_limits[i][1])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        n_links = p.getNumJoints(self._robot_id, physicsClientId=self._physics_client)
        env_bodies, floor_body = self._build_collision_bodies()

        # Capture mutable state needed inside the checker subclass.
        saved_joints = start_joints.copy()
        interface = self

        class _ValidityChecker(ob.StateValidityChecker):
            def isValid(self, state):  # noqa: N802
                joints = np.array([state[i] for i in range(dof)])
                valid = interface._is_state_valid(
                    joints, env_bodies, floor_body, collision_margin, n_links,
                )
                # Restore joint state so FK queries remain consistent between checks.
                for j, ji in enumerate(interface._movable_joints):
                    if j < len(saved_joints):
                        p.resetJointState(
                            interface._robot_id, ji, float(saved_joints[j]),
                            physicsClientId=interface._physics_client,
                        )
                return valid

        checker = _ValidityChecker(si)
        si.setStateValidityChecker(checker)
        si.setup()

        # Allocate and populate start/goal states.
        start_state = si.allocState()
        goal_state = si.allocState()
        for i in range(dof):
            start_state[i] = float(start_joints[i])
            goal_state[i] = float(goal_joints[i])

        if not si.isValid(start_state):
            logger.warning("Start state is in collision — cannot plan")
            return None
        if not si.isValid(goal_state):
            logger.warning("Goal state is in collision — cannot plan")
            return None

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)

        planner = og.RRTConnect(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(planning_time)
        if not solved:
            logger.warning("OMPL failed to find a path within %.1f s", planning_time)
            return None

        # Simplify and interpolate to a fixed number of waypoints.
        path = pdef.getSolutionPath()
        path.interpolate(interpolate_n)

        n_states = path.getStateCount()
        trajectory = np.zeros((n_states, dof), dtype=float)
        for s in range(n_states):
            state = path.getState(s)
            for i in range(dof):
                trajectory[s, i] = state[i]

        logger.info(
            "OMPL RRTConnect: path found (%d waypoints, %.1f s budget)",
            n_states, planning_time,
        )
        return trajectory

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Return (lower, upper) radian limits for each movable arm joint."""
        limits = []
        dof = len(self._joints)
        for j, joint_idx in enumerate(self._movable_joints):
            if j >= dof:
                break
            info = p.getJointInfo(self._robot_id, joint_idx, physicsClientId=self._physics_client)
            lo, hi = float(info[8]), float(info[9])
            if lo >= hi:
                lo, hi = -3.14159, 3.14159
            limits.append((lo, hi))
        return limits

    def move_to_pose(
        self,
        target_position: Sequence[float],
        target_orientation: Optional[Sequence[float]] = None,
        planning_dt: float = 0.02,
        execute: bool = False,
        position_tolerance: float = 0.02,
        orientation_tolerance_rad: float = 0.2,
        planning_time: float = 5.0,
        collision_margin: float = 0.005,
        interpolate_n: int = 64,
        **_ignored: Any,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Plan a trajectory to a target TCP pose and optionally apply final joint state.

        This mirrors the high-level return contract used by the CuRobo planner:
        ``(success, trajectory, dt)``.

        Unknown kwargs (e.g. ``max_joint_step`` from legacy callers) are silently
        ignored to avoid breaking existing call sites.
        """
        trajectory = self.plan_joint_trajectory_to_pose(
            target_position=target_position,
            target_orientation=target_orientation,
            position_tolerance=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance_rad,
            planning_time=planning_time,
            collision_margin=collision_margin,
            interpolate_n=interpolate_n,
        )
        if trajectory is None:
            return False, None, None

        if execute:
            self.set_current_joint_state(trajectory[-1])
        return True, trajectory, planning_dt

    def step(self, seconds: float = 0.02) -> None:
        """Advance simulation, useful for GUI visualization."""
        if self._physics_client is None:
            return
        steps = max(1, int(seconds * 240))
        sleep_dt = max(0.0, float(seconds) / float(steps))
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._physics_client)
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)

    def set_status(self, text: str, color: Optional[List[float]] = None) -> None:
        """Display status text in the PyBullet GUI (no-op in DIRECT mode)."""
        if not self._use_gui or self._physics_client is None:
            return
        rgb = color or [0.2, 0.9, 0.2]
        for tid in self._status_text_ids:
            p.removeUserDebugItem(tid, physicsClientId=self._physics_client)
        self._status_text_ids.clear()
        y0 = 0.7
        for i, line in enumerate(str(text).splitlines()[:3]):
            tid = p.addUserDebugText(
                line,
                [-0.1, y0, 0.6 - i * 0.06],
                textColorRGB=rgb,
                textSize=1.0,
                physicsClientId=self._physics_client,
            )
            self._status_text_ids.append(tid)

    def convert_cam_pose_to_base(
        self,
        position: Any,
        orientation: Any,
        do_translation: bool = True,
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a pose expressed in camera frame to robot base frame.

        Matches CuRoboMotionPlanner.convert_cam_pose_to_base().

        Args:
            position: (3,) position in camera frame
            orientation: quaternion [x,y,z,w] or (3,3) rotation matrix
            do_translation: apply translation component (default True)
            debug: log the result

        Returns:
            (position_base, quaternion_xyzw_base)
        """
        cam_pos, cam_rot = self.get_camera_transform()
        if cam_rot is None:
            raise RuntimeError("Camera transform not available")

        pos = np.asarray(position, dtype=float)

        # Rotate position into base frame
        transformed_position = cam_rot.apply(pos)
        if do_translation and cam_pos is not None:
            transformed_position += cam_pos

        if debug:
            logger.debug("convert_cam_pose_to_base result: %s", transformed_position)

        # Handle orientation
        if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            input_rotation = Rotation.from_matrix(orientation)
        elif isinstance(orientation, (np.ndarray, list, tuple)) and np.asarray(orientation).shape == (4,):
            input_rotation = Rotation.from_quat(np.asarray(orientation))
        else:
            raise ValueError(f"Unsupported orientation format: {type(orientation)}")

        combined = cam_rot * input_rotation
        return transformed_position, combined.as_quat()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_T(pos: np.ndarray, orn_xyzw: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from position and xyzw quaternion."""
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
    T[:3, 3] = pos
    return T
