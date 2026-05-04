"""
Hardware smoke test for PyBullet-planned xArm primitives.

Plans with `XArmPybulletInterface` and executes on a real xArm through the raw
xArm SDK. By default this script only plans; pass `--execute` to move hardware.

Run:
    uv run scripts/test_xarm_real_pybullet_planned_primitives.py --robot-ip 192.168.1.224
    uv run scripts/test_xarm_real_pybullet_planned_primitives.py --robot-ip 192.168.1.224 --execute
    uv run scripts/test_xarm_real_pybullet_planned_primitives.py --execute --steps move,retract
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.kinematics.xarm_pybullet_planned_primitives import (
    XArmPybulletPlannedPrimitives,
)

_HOME_JOINTS_DEG = [-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9]
_DEFAULT_ROBOT_IP = "192.168.1.224"
_DEFAULT_SAFE_SPEED_FACTOR = 0.25
_DEFAULT_SAFE_JOINT_SPEED = 0.12
_DEFAULT_SAFE_JOINT_ACCEL = 0.25
_DEFAULT_SAFE_MAX_JOINT_STEP = 0.03
_GRIPPER_OPEN = 850
_GRIPPER_CLOSED = 0
_STEP_NAMES = ("move", "push", "pull", "pivot_pull", "twist", "retract")


class RawXArmRobotAdapter:
    """Small real-xArm adapter for `XArmPybulletPlannedPrimitives`.

    Args:
        robot_ip: xArm controller IP address.

    Example:
        >>> robot = RawXArmRobotAdapter("192.168.1.224")
        >>> robot.get_robot_joint_state()
        >>> robot.disconnect()
    """

    def __init__(self, robot_ip: str) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError("xarm SDK is not available") from exc

        self.robot_ip = robot_ip
        self.arm = XArmAPI(robot_ip, is_radian=True)
        self.arm_lock = threading.Lock()
        self.current_joints: Optional[np.ndarray] = None

        self.arm.connect()
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return current robot joints in radians."""
        with self.arm_lock:
            code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0 or angles is None:
            print(f"Failed to read xArm joints; code={code}")
            return None
        self.set_current_joint_state(angles)
        return self.current_joints.copy()

    def set_current_joint_state(self, joint_positions: Any) -> None:
        """Cache the latest known real xArm joint state."""
        self.current_joints = np.asarray(joint_positions, dtype=float).reshape(-1)[:7]

    def set_robot_joint_angles(
        self,
        joint_angles: List[float],
        wait: bool = True,
        speed: float = _DEFAULT_SAFE_JOINT_SPEED,
        acc: float = _DEFAULT_SAFE_JOINT_ACCEL,
    ) -> bool:
        """Move the real xArm to a joint target in radians."""
        with self.arm_lock:
            # Only reset controller state on blocking (last) waypoints to avoid
            # interrupting the motion pipeline mid-trajectory.
            if wait:
                self.arm.set_mode(0)
                self.arm.set_state(0)
            code = self.arm.set_servo_angle(
                angle=list(joint_angles),
                speed=speed,
                mvacc=acc,
                wait=wait,
                is_radian=True,
            )
        if code == 0:
            self.set_current_joint_state(joint_angles)
            return True
        print(f"set_servo_angle failed; code={code}")
        return False

    def open_gripper(self, wait: bool = True, **_: Any) -> bool:
        """Open the xArm gripper."""
        return self._set_gripper(_GRIPPER_OPEN, wait=wait)

    def close_gripper(self, wait: bool = True, **_: Any) -> bool:
        """Close the xArm gripper."""
        return self._set_gripper(_GRIPPER_CLOSED, wait=wait)

    def disconnect(self) -> None:
        """Disconnect from the xArm controller."""
        try:
            self.arm.set_state(0)
        finally:
            self.arm.disconnect()

    def _set_gripper(self, position: int, wait: bool = True) -> bool:
        with self.arm_lock:
            if hasattr(self.arm, "set_gripper_mode"):
                self.arm.set_gripper_mode(0)
            if hasattr(self.arm, "set_gripper_enable"):
                self.arm.set_gripper_enable(True)
            if hasattr(self.arm, "clean_gripper_error"):
                self.arm.clean_gripper_error()
            if hasattr(self.arm, "set_gripper_speed"):
                self.arm.set_gripper_speed(2000)
            if hasattr(self.arm, "set_gripper_position"):
                code = self.arm.set_gripper_position(
                    position,
                    wait=wait,
                    auto_enable=True,
                )
                return code == 0
            if position == _GRIPPER_OPEN and hasattr(self.arm, "open_lite6_gripper"):
                return self.arm.open_lite6_gripper() == 0
            if position == _GRIPPER_CLOSED and hasattr(self.arm, "close_lite6_gripper"):
                return self.arm.close_lite6_gripper() == 0
        return False


def _run_step(
    name: str,
    fn: Callable[[], Dict[str, Any]],
    results: List[Tuple[str, bool, Dict[str, Any]]],
    wait_for_input: bool = True,
) -> None:
    if wait_for_input:
        try:
            input(f"\nPress Enter to run hardware test: {name}")
        except EOFError:
            pass

    try:
        payload = fn()
        ok = bool(payload.get("success"))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {payload}")
        results.append((name, ok, payload))
    except Exception as exc:
        print(f"[FAIL] {name}: exception={exc}")
        results.append((name, False, {"success": False, "reason": str(exc)}))


def _parse_steps(value: str) -> List[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(steps) - set(_STEP_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown steps {unknown}; choices are {list(_STEP_NAMES)}"
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test real xArm primitives planned through PyBullet"
    )
    parser.add_argument(
        "--robot-ip",
        default=os.getenv("ROBOT_IP", _DEFAULT_ROBOT_IP),
        help="xArm controller IP. Defaults to ROBOT_IP or 192.168.1.224.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.35, 0.00, 0.28],
        metavar=("X", "Y", "Z"),
        help="Primary target position in xArm base frame (m).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the real robot. Without this flag, only planning is tested.",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Run selected tests without waiting for Enter before each step.",
    )
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default=list(_STEP_NAMES),
        help=f"Comma-separated subset of steps to run. Choices: {','.join(_STEP_NAMES)}.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=_DEFAULT_SAFE_SPEED_FACTOR,
        help="Real robot trajectory speed multiplier for planned waypoint timing.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=_DEFAULT_SAFE_MAX_JOINT_STEP,
        help="Max radian delta between planned joint waypoints.",
    )
    parser.add_argument(
        "--open-gripper-first",
        action="store_true",
        help="Open the gripper before running selected steps.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Real xArm PyBullet-Planned Primitive Smoke Test")
    print(f"Robot IP:       {args.robot_ip}")
    print(f"Target pose:    {[round(v, 4) for v in args.target]}")
    print(f"Steps:          {args.steps}")
    print(f"Execution:      {'REAL ROBOT COMMANDS ENABLED' if args.execute else 'dry-run planning only'}")
    print("=" * 78)
    if args.execute and not args.auto_run:
        input("Hardware execution is enabled. Confirm workspace is clear, then press Enter.")

    robot: Optional[RawXArmRobotAdapter] = None
    planner: Optional[XArmPybulletInterface] = None
    try:
        robot = RawXArmRobotAdapter(args.robot_ip)
        planner = XArmPybulletInterface(use_gui=False)
        primitives = XArmPybulletPlannedPrimitives(robot=robot, planner=planner)

        joints = robot.get_robot_joint_state()
        print(f"Initial joints: {None if joints is None else [round(v, 4) for v in joints.tolist()]}")
        tcp = primitives.get_robot_tcp_pose()
        print(f"Initial TCP:    {None if tcp is None else [round(v, 4) for v in tcp[0].tolist()]}")

        if args.open_gripper_first:
            _run_step(
                "open_gripper",
                lambda: primitives.open_gripper() if args.execute else {"success": True, "executed": False},
                [],
                wait_for_input=not args.auto_run,
            )

        target = np.asarray(args.target, dtype=float)
        approach = (target + np.array([0.0, 0.0, 0.05], dtype=float)).tolist()
        execute = bool(args.execute)
        results: List[Tuple[str, bool, Dict[str, Any]]] = []
        step_fns: Dict[str, Callable[[], Dict[str, Any]]] = {
            "move": lambda: primitives.move_gripper_to_pose(
                target_position=approach,
                preset_orientation="top_down",
                is_place=False,
                max_joint_step=args.max_joint_step,
                speed_factor=args.speed_factor,
                execute=execute,
            ),
            "push": lambda: primitives.push(
                distance=0.04,
                force_direction="forward",
                speed_factor=args.speed_factor,
                execute=execute,
            ),
            "pull": lambda: primitives.pull(
                distance=0.04,
                force_direction="forward",
                speed_factor=args.speed_factor,
                execute=execute,
            ),
            "pivot_pull": lambda: primitives.pivot_pull(
                pull_distance=0.03,
                arc_angle_deg=15.0,
                speed_factor=args.speed_factor,
                execute=execute,
            ),
            "twist": lambda: primitives.twist(
                direction="clockwise",
                rotation_angle_deg=20.0,
                speed=_DEFAULT_SAFE_JOINT_SPEED,
                execute=execute,
            ),
            "retract": lambda: primitives.retract_gripper(
                speed=_DEFAULT_SAFE_JOINT_SPEED,
                execute=execute,
            ),
        }

        for step in args.steps:
            _run_step(
                step,
                step_fns[step],
                results,
                wait_for_input=not args.auto_run,
            )

        final_joints = robot.get_robot_joint_state()
        final_tcp = primitives.get_robot_tcp_pose()
        print("-" * 78)
        print(f"Final TCP:    {None if final_tcp is None else [round(v, 4) for v in final_tcp[0].tolist()]}")
        print(f"Final joints: {None if final_joints is None else [round(v, 4) for v in final_joints.tolist()]}")
        print("-" * 78)

        total = len(results)
        passed = sum(1 for _, ok, _ in results if ok)
        print(f"Summary: {passed}/{total} passed")
        if passed != total:
            raise SystemExit(1)
    finally:
        if planner is not None:
            planner.cleanup()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
