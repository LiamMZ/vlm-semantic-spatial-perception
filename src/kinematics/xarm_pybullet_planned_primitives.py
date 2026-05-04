"""
Real xArm primitives planned by the PyBullet xArm model.

This module mirrors `XArmPybulletPrimitives`, but sends successful PyBullet
joint trajectories to a real xArm-compatible interface for execution.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.utils.logging_utils import get_structured_logger

_HOME_JOINTS_DEG = [-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9]
_HOME_JOINTS = np.deg2rad(_HOME_JOINTS_DEG).tolist()
_DEFAULT_POSITION_TOLERANCE_M = 0.07
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.6
_DEFAULT_SAFE_SPEED_FACTOR = 0.25
_DEFAULT_SAFE_JOINT_SPEED = 0.5
_DEFAULT_SAFE_JOINT_ACCEL = 0.25


class XArmPybulletPlannedPrimitives:
    """Primitive API that plans in PyBullet and executes on a real xArm.

    Args:
        robot: Real xArm robot interface. It should expose
            `get_robot_joint_state`, `set_robot_joint_angles`, `open_gripper`,
            and `close_gripper`. `CuRoboMotionPlanner` satisfies this surface.
        planner: PyBullet xArm interface used for IK, FK, and frame transforms.
        registry: Optional object registry for resolving point labels.
        logger: Optional structured logger.

    Example:
        >>> real = CuRoboMotionPlanner(robot_ip="192.168.1.XXX")
        >>> primitives = XArmPybulletPlannedPrimitives(robot=real)
        >>> primitives.move_gripper_to_pose(target_position=[0.35, 0.0, 0.30])
    """

    def __init__(
        self,
        robot: Any,
        planner: Optional[XArmPybulletInterface] = None,
        registry: Optional[Any] = None,
        logger: Optional[Any] = None,
        use_gui: bool = False,
    ) -> None:
        self._robot = robot
        self._planner = planner or XArmPybulletInterface(use_gui=use_gui)
        self._registry = registry
        self._logger = logger or get_structured_logger("XArmPybulletPlannedPrimitives")
        self._gripper_open = True
        self._last_execution_error: Optional[str] = None
        self._use_gui = use_gui

    def camera_pose_from_joints(self, joints: Optional[List[float]]):
        """Return `(position, Rotation)` camera pose from PyBullet FK.

        Args:
            joints: Optional joint angles in radians. If omitted, the current
                real robot joints are queried first.

        Example:
            >>> primitives.camera_pose_from_joints([0.0] * 7)
        """
        if joints is None:
            joints = self._get_real_joint_state()
        if joints is not None:
            self._planner.set_current_joint_state(joints)
        return self._planner.get_camera_transform()

    def get_robot_tcp_pose(self):
        """Return the PyBullet TCP pose after syncing from the real xArm."""
        if not self._sync_planner_to_real_robot():
            return None
        return self._planner.get_robot_tcp_pose()

    def get_camera_transform(self):
        """Return the PyBullet camera transform after syncing real joints."""
        if not self._sync_planner_to_real_robot():
            return None, None
        return self._planner.get_camera_transform()

    def convert_cam_pose_to_base(
        self,
        position: Any,
        orientation: Any,
        do_translation: bool = True,
        debug: bool = False,
    ):
        """Convert a camera-frame pose into the xArm base frame using PyBullet.

        Args:
            position: Camera-frame position.
            orientation: Camera-frame orientation as xyzw quaternion or matrix.
            do_translation: Whether to include camera translation.
            debug: Forwarded to the PyBullet transform helper.

        Example:
            >>> primitives.convert_cam_pose_to_base([0, 0, 1], [0, 0, 0, 1])
        """
        if not self._sync_planner_to_real_robot():
            raise RuntimeError("cannot read real robot joint state")
        return self._planner.convert_cam_pose_to_base(
            position=position,
            orientation=orientation,
            do_translation=do_translation,
            debug=debug,
        )

    def move_gripper_to_pose(
        self,
        target_position: Optional[List[float]] = None,
        target_orientation: Optional[List[float]] = None,
        preset_orientation: str = "top_down",
        is_place: bool = False,
        point_label: Optional[str] = None,
        is_top_down_grasp: bool = True,
        is_side_grasp: bool = False,
        planning_dt: float = 0.02,
        max_joint_step: float = 0.05,
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plan a PyBullet IK trajectory and optionally execute it on xArm.

        Args:
            target_position: Desired TCP position in the xArm base frame.
            target_orientation: Optional xyzw TCP orientation. If omitted,
                `preset_orientation` selects a top-down or side orientation.
            speed_factor: Execution speed multiplier for waypoint timing.
            execute: If false, only returns planning metadata.

        Example:
            >>> primitives.move_gripper_to_pose([0.35, 0.0, 0.30], execute=False)
        """
        del is_top_down_grasp
        if target_position is None and point_label is not None:
            target_position = self._resolve_point_label(point_label)
        if target_position is None:
            return {"success": False, "reason": "cannot determine target_position"}
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}

        pos = np.asarray(target_position, dtype=float).tolist()
        if is_place:
            pos[2] += 0.04

        if target_orientation is None:
            use_side = (preset_orientation == "side") or is_side_grasp
            target_orientation = [-0.6894, 0.0305, -0.7237, 0.0033] if use_side else [-0.9983, 0.0314, 0.0438, 0.0223]

        current_joints = self._planner.get_robot_joint_state()
        current_tcp = self._planner.get_robot_tcp_pose()
        success, trajectory, dt = self._planner.move_to_pose(
            target_position=pos,
            target_orientation=target_orientation,
            planning_dt=planning_dt,
            execute=False,
            max_joint_step=max_joint_step,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
        )
        if (not success or trajectory is None) and target_orientation is not None:
            self._logger.warning(
                "move_gripper_to_pose orientation-constrained plan failed; "
                "retrying position-only. target=%s orientation=%s current_tcp=%s",
                pos,
                target_orientation,
                None if current_tcp is None else current_tcp[0].tolist(),
            )
            success, trajectory, dt = self._planner.move_to_pose(
                target_position=pos,
                target_orientation=None,
                planning_dt=planning_dt,
                execute=False,
                max_joint_step=max_joint_step,
                position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            )
        if not success or trajectory is None:
            self._logger.warning(
                "move_gripper_to_pose failed. target=%s orientation=%s "
                "current_joints=%s current_tcp=%s",
                pos,
                target_orientation,
                None if current_joints is None else current_joints.tolist(),
                None if current_tcp is None else current_tcp[0].tolist(),
            )
            return {
                "success": False,
                "reason": "pose planning failed",
                "target_position": pos,
                "target_orientation": list(target_orientation),
                "current_joints": None if current_joints is None else current_joints.tolist(),
                "current_tcp": None if current_tcp is None else current_tcp[0].tolist(),
            }

        if not execute:
            return {
                "success": True,
                "executed": False,
                "reason": "planned only; execute=False",
                "target_position": pos,
                "target_orientation": list(target_orientation),
                "trajectory_len": int(trajectory.shape[0]),
                "dt": dt,
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }

        executed = self._execute_joint_trajectory(trajectory, dt, speed_factor=speed_factor)
        execution_error = self._last_execution_error
        if execute:
            if not executed:
                return {
                    "success": False,
                    "reason": execution_error or "real robot trajectory execution failed",
                    "target_position": pos,
                    "trajectory_len": int(trajectory.shape[0]),
                    "start_joints": trajectory[0].tolist(),
                    "goal_joints": trajectory[-1].tolist(),
                }

        return {
            "success": True,
            "executed": True,
            "target_position": pos,
            "target_orientation": list(target_orientation),
            "trajectory_len": int(trajectory.shape[0]),
            "dt": dt,
            "start_joints": trajectory[0].tolist(),
            "goal_joints": trajectory[-1].tolist(),
        }

    def move_gripper_To_pose(self, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible alias for legacy primitive calls."""
        return self.move_gripper_to_pose(**kwargs)

    def push(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Push along the requested base-frame axis from the current TCP pose."""
        return self._cartesian_delta_motion(
            direction=self._force_direction_to_vector(force_direction),
            distance=distance,
            label="push",
            speed_factor=speed_factor,
            execute=execute,
        )

    def pull(
        self,
        distance: float = 0.08,
        force_direction: str = "forward",
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Pull opposite the requested base-frame axis from the current TCP pose."""
        return self._cartesian_delta_motion(
            direction=-self._force_direction_to_vector(force_direction),
            distance=distance,
            label="pull",
            speed_factor=speed_factor,
            execute=execute,
        )

    def pivot_pull(
        self,
        pivot_point: Optional[List[float]] = None,
        pull_distance: float = 0.10,
        arc_angle_deg: float = 25.0,
        direction: str = "counterclockwise",
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Approximate pivot-pull by planning a tangential/radial TCP motion."""
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}
        tcp = self._planner.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        tcp_pos, tcp_quat = tcp

        if pivot_point is None:
            pivot_point = (tcp_pos + np.array([0.0, -0.10, 0.0])).tolist()
        pivot = np.asarray(pivot_point, dtype=float)
        radius_vec = tcp_pos - pivot
        radius_xy = radius_vec.copy()
        radius_xy[2] = 0.0
        radius_norm = float(np.linalg.norm(radius_xy))
        if radius_norm < 1e-6:
            return {"success": False, "reason": "pivot radius too small"}

        radial_hat = radius_xy / radius_norm
        tangent_hat = np.array([-radial_hat[1], radial_hat[0], 0.0], dtype=float)
        if direction == "clockwise":
            tangent_hat *= -1.0
        arc_len = math.radians(float(arc_angle_deg)) * radius_norm
        delta = tangent_hat * arc_len - radial_hat * float(pull_distance)
        return self._plan_and_execute_pose(
            target_position=(tcp_pos + delta).tolist(),
            target_orientation=tcp_quat.tolist(),
            label="pivot_pull",
            speed_factor=speed_factor,
            execute=execute,
        )

    def push_pull(
        self,
        surface_label: str,
        force_direction: str = "perpendicular",
        is_button: bool = False,
        has_pivot: bool = False,
        hinge_location: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility primitive that dispatches to push, pull, or pivot-pull."""
        del surface_label, hinge_location
        if has_pivot:
            result = self.pivot_pull(**kwargs)
        elif force_direction == "perpendicular":
            result = self.push(distance=0.05 if is_button else 0.08, force_direction="down", **kwargs)
        elif force_direction == "parallel":
            result = self.pull(distance=0.08, force_direction="forward", **kwargs)
        else:
            result = self.push(distance=0.08, force_direction=force_direction, **kwargs)

        if is_button and result.get("success") and kwargs.get("execute", True):
            self.pull(distance=0.03, force_direction="down", **kwargs)
        return result

    def twist(
        self,
        direction: str = "clockwise",
        rotation_angle_deg: float = 90.0,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Twist by commanding the final xArm wrist joint."""
        joints = self._get_real_joint_state()
        if joints is None or len(joints) == 0:
            return {"success": False, "reason": "cannot read real robot joints"}
        joints = np.asarray(joints, dtype=float).copy()
        delta = math.radians(float(rotation_angle_deg))
        if direction == "clockwise":
            delta *= -1.0
        joints[-1] += delta
        if execute and not self._set_real_joint_angles(joints.tolist(), wait=True, speed=speed):
            return {"success": False, "reason": "real robot twist command failed"}
        self._planner.set_current_joint_state(joints)
        return {"success": True, "executed": execute, "joint_index": int(len(joints) - 1), "delta_rad": float(delta)}

    def open_gripper(self, **kwargs: Any) -> Dict[str, Any]:
        """Open the real xArm gripper."""
        self._gripper_open = True
        ok = self._robot.open_gripper(**kwargs)
        return {"success": bool(ok)}

    def close_gripper(self, **kwargs: Any) -> Dict[str, Any]:
        """Close the real xArm gripper."""
        self._gripper_open = False
        ok = self._robot.close_gripper(**kwargs)
        return {"success": bool(ok)}

    def retract_gripper(
        self,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        execute: bool = True,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Move the real xArm back to the configured home joint state."""
        if execute and not self._set_real_joint_angles(_HOME_JOINTS, wait=True, speed=speed):
            return {"success": False, "reason": "real robot retract command failed"}
        self._planner.set_current_joint_state(_HOME_JOINTS)
        return {"success": True, "executed": execute}

    def _cartesian_delta_motion(
        self,
        direction: np.ndarray,
        distance: float,
        label: str,
        speed_factor: float,
        execute: bool,
    ) -> Dict[str, Any]:
        if not self._sync_planner_to_real_robot():
            return {"success": False, "reason": "cannot read real robot joint state"}
        tcp = self._planner.get_robot_tcp_pose()
        if tcp is None:
            return {"success": False, "reason": "cannot read tcp pose"}
        pos, quat = tcp
        unit = direction / max(float(np.linalg.norm(direction)), 1e-8)
        return self._plan_and_execute_pose(
            target_position=(pos + unit * float(distance)).tolist(),
            target_orientation=quat.tolist(),
            label=label,
            speed_factor=speed_factor,
            execute=execute,
        )

    def _plan_and_execute_pose(
        self,
        target_position: List[float],
        target_orientation: List[float],
        label: str,
        speed_factor: float,
        execute: bool,
    ) -> Dict[str, Any]:
        success, trajectory, dt = self._planner.move_to_pose(
            target_position=target_position,
            target_orientation=target_orientation,
            execute=False,
            position_tolerance=_DEFAULT_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=_DEFAULT_ORIENTATION_TOLERANCE_RAD,
        )
        if not success or trajectory is None:
            return {"success": False, "reason": f"{label} planning failed"}
        if not execute:
            return {
                "success": True,
                "executed": False,
                "reason": "planned only; execute=False",
                "target_position": target_position,
                "trajectory_len": int(trajectory.shape[0]),
                "dt": dt,
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }
        if not self._execute_joint_trajectory(trajectory, dt, speed_factor=speed_factor):
            return {
                "success": False,
                "reason": self._last_execution_error or f"{label} execution failed",
                "target_position": target_position,
                "trajectory_len": int(trajectory.shape[0]),
                "start_joints": trajectory[0].tolist(),
                "goal_joints": trajectory[-1].tolist(),
            }
        return {
            "success": True,
            "executed": True,
            "target_position": target_position,
            "trajectory_len": int(trajectory.shape[0]),
            "dt": dt,
            "start_joints": trajectory[0].tolist(),
            "goal_joints": trajectory[-1].tolist(),
        }

    def _execute_joint_trajectory(
        self,
        trajectory: np.ndarray,
        dt: Optional[float],
        speed_factor: float = _DEFAULT_SAFE_SPEED_FACTOR,
    ) -> bool:
        self._last_execution_error = None
        frame_dt = (float(dt) if dt is not None else 0.02) / max(float(speed_factor), 1e-6)
        waypoints = np.asarray(trajectory, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[0] == 0:
            self._last_execution_error = "empty or malformed trajectory"
            return False

        if hasattr(self._robot, "set_robot_joint_angles"):
            start_time = time.time()
            last_idx = len(waypoints) - 1
            for idx, waypoint in enumerate(waypoints):
                target_t = idx * frame_dt
                elapsed = time.time() - start_time
                if elapsed < target_t:
                    time.sleep(target_t - elapsed)
                is_last = idx == last_idx
                if not self._robot.set_robot_joint_angles(
                    waypoint.tolist(),
                    wait=is_last,
                    speed=_DEFAULT_SAFE_JOINT_SPEED,
                ):
                    self._last_execution_error = f"set_robot_joint_angles failed at waypoint {idx}"
                    self._logger.warning(self._last_execution_error)
                    return False
                if self._use_gui:
                    self._planner.set_current_joint_state(waypoint.tolist())
                if idx % max(1, len(waypoints) // 5) == 0 or is_last:
                    self._logger.info(
                        "Executed waypoint %d/%d",
                        idx + 1,
                        len(waypoints),
                    )
            self._planner.set_current_joint_state(waypoints[-1])
            try:
                self._robot.set_current_joint_state(waypoints[-1])
            except Exception:
                pass
            return True

        arm = getattr(self._robot, "arm", None)
        arm_lock = getattr(self._robot, "arm_lock", None)
        if arm is not None:
            try:
                if arm_lock is not None:
                    arm_lock.acquire()
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.1)
                start_time = time.time()
                last_idx = len(waypoints) - 1
                for idx, waypoint in enumerate(waypoints):
                    target_t = idx * frame_dt
                    elapsed = time.time() - start_time
                    if elapsed < target_t:
                        time.sleep(target_t - elapsed)
                    code = arm.set_servo_angle(
                        angle=waypoint.tolist(),
                        speed=_DEFAULT_SAFE_JOINT_SPEED,
                        mvacc=_DEFAULT_SAFE_JOINT_ACCEL,
                        is_radian=True,
                        wait=(idx == last_idx),
                    )
                    if code != 0:
                        self._last_execution_error = (
                            f"xArm set_servo_angle failed at waypoint {idx}; code={code}"
                        )
                        self._logger.warning(self._last_execution_error)
                        return False
                    if self._use_gui:
                        self._planner.set_current_joint_state(waypoint.tolist())
            finally:
                if arm_lock is not None:
                    arm_lock.release()
        else:
            self._last_execution_error = "robot has neither set_robot_joint_angles nor arm"
            return False

        if arm is None:
            for waypoint in waypoints:
                if not self._set_real_joint_angles(waypoint.tolist(), wait=True):
                    self._last_execution_error = "real robot joint command failed"
                    return False
                if frame_dt > 0.0:
                    time.sleep(frame_dt)

        self._planner.set_current_joint_state(waypoints[-1])
        try:
            self._robot.set_current_joint_state(waypoints[-1])
        except Exception:
            pass
        return True

    def _get_real_joint_state(self) -> Optional[np.ndarray]:
        joints = self._robot.get_robot_joint_state()
        if joints is None:
            return None
        joints = np.asarray(joints, dtype=float).reshape(-1)
        if joints.size == 0:
            return None
        return joints[:7]

    def _sync_planner_to_real_robot(self) -> bool:
        joints = self._get_real_joint_state()
        if joints is None:
            return False
        self._planner.set_current_joint_state(joints)
        self._planner.get_robot_tcp_pose()
        return True

    def _set_real_joint_angles(
        self,
        joints: List[float],
        wait: bool = True,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
    ) -> bool:
        if hasattr(self._robot, "set_robot_joint_angles"):
            return bool(self._robot.set_robot_joint_angles(joints, wait=wait, speed=speed))
        arm = getattr(self._robot, "arm", None)
        if arm is None:
            return False
        code = arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=_DEFAULT_SAFE_JOINT_ACCEL,
            is_radian=True,
            wait=wait,
        )
        return code == 0

    def _force_direction_to_vector(self, force_direction: str) -> np.ndarray:
        mapping = {
            "forward": np.array([1.0, 0.0, 0.0], dtype=float),
            "backward": np.array([-1.0, 0.0, 0.0], dtype=float),
            "left": np.array([0.0, 1.0, 0.0], dtype=float),
            "right": np.array([0.0, -1.0, 0.0], dtype=float),
            "up": np.array([0.0, 0.0, 1.0], dtype=float),
            "down": np.array([0.0, 0.0, -1.0], dtype=float),
            "perpendicular": np.array([0.0, 0.0, -1.0], dtype=float),
            "parallel": np.array([1.0, 0.0, 0.0], dtype=float),
        }
        return mapping.get(force_direction, np.array([1.0, 0.0, 0.0], dtype=float))

    def _resolve_point_label(self, label: str) -> Optional[List[float]]:
        if self._registry is None:
            return None
        obj_id, _, point_name = label.partition("/")
        obj = self._registry.get_object(obj_id)
        if obj is None:
            return None
        if point_name and getattr(obj, "interaction_points", None):
            ip = obj.interaction_points.get(point_name)
            if ip is not None and ip.position_3d is not None:
                return list(ip.position_3d)
        if getattr(obj, "position_3d", None) is not None:
            return list(obj.position_3d)
        return None
