"""
Integration smoke test for XArmPybulletPrimitives.

Runs a short sequence of primitive calls against the PyBullet-backed xArm
interface and prints pass/fail for each call.

Run:
    uv run scripts/test_xarm_pybullet_primitives.py
    uv run scripts/test_xarm_pybullet_primitives.py --target 0.35 0.00 0.28
    uv run scripts/test_xarm_pybullet_primitives.py --no-gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.sim.xarm_pybullet_primitives import XArmPybulletPrimitives
from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface

_HOME_JOINTS_DEG = [-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9]
_DEFAULT_JOINTS = np.deg2rad(_HOME_JOINTS_DEG).tolist()


def _run_step(
    name: str,
    fn: Callable[[], Dict[str, Any]],
    results: List[Tuple[str, bool, Dict[str, Any]]],
    viewer: Optional[Any] = None,
    wait_for_input: bool = True,
) -> None:
    if viewer is not None:
        viewer.set_status(f"Ready: {name}\nPress Enter in terminal to execute")
        viewer.step(0.1)
    if wait_for_input:
        try:
            input(f"\nPress Enter to run test: {name}")
        except EOFError:
            pass

    try:
        if viewer is not None:
            viewer.set_status(f"Executing: {name}")
            viewer.step(0.1)
        payload = fn()
        ok = bool(payload.get("success"))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {payload}")
        results.append((name, ok, payload))
        if viewer is not None:
            viewer.set_status(f"{status}: {name}")
            viewer.step(0.4)
    except Exception as exc:
        print(f"[FAIL] {name}: exception={exc}")
        results.append((name, False, {"success": False, "reason": str(exc)}))
        if viewer is not None:
            viewer.set_status(f"FAIL: {name}\n{exc}")
            viewer.step(0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test XArmPybulletPrimitives")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.35, 0.00, 0.28],
        metavar=("X", "Y", "Z"),
        help="Primary target position in base frame (m).",
    )
    parser.add_argument(
        "--joint-state",
        type=float,
        nargs=7,
        default=_DEFAULT_JOINTS,
        metavar="J",
        help="Initial xArm7 joint state (rad).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI playback of primitive execution.",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Run all tests without waiting for Enter before each step.",
    )
    parser.add_argument(
        "--move-replay-dt",
        type=float,
        default=0.08,
        help="GUI seconds to hold each move_gripper_to_pose trajectory waypoint.",
    )
    args = parser.parse_args()

    robot = XArmPybulletInterface(use_gui=not args.no_gui)
    primitives = XArmPybulletPrimitives(robot=robot, registry=None, env=robot)
    robot.set_current_joint_state(args.joint_state)
    robot.get_robot_tcp_pose()
    if not args.no_gui:
        robot.step(0.2)
        robot.set_status("Ready: starting primitive smoke test")
        robot.step(0.5)

    target = np.asarray(args.target, dtype=float)
    approach = (target + np.array([0.0, 0.0, 0.05], dtype=float)).tolist()

    print("=" * 70)
    print("XArm PyBullet Primitive Smoke Test")
    print(f"Initial joints: {[round(v, 4) for v in args.joint_state]}")
    print(f"Target pose:    {[round(v, 4) for v in target.tolist()]}")
    print("=" * 70)

    results: List[Tuple[str, bool, Dict[str, Any]]] = []
    _run_step(
        "move_gripper_to_pose(top_down)",
        lambda: primitives.move_gripper_to_pose(
            target_position=approach,
            preset_orientation="top_down",
            is_place=False,
            visualization_dt=args.move_replay_dt,
        ),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )
    _run_step(
        "push",
        lambda: primitives.push(distance=0.04, force_direction="forward"),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )
    _run_step(
        "pull",
        lambda: primitives.pull(distance=0.04, force_direction="forward"),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )
    _run_step(
        "pivot_pull",
        lambda: primitives.pivot_pull(pull_distance=0.03, arc_angle_deg=15.0),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )
    _run_step(
        "twist",
        lambda: primitives.twist(direction="clockwise", rotation_angle_deg=20.0),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )
    _run_step(
        "retract_gripper",
        lambda: primitives.retract_gripper(),
        results,
        viewer=robot if not args.no_gui else None,
        wait_for_input=not args.auto_run,
    )

    tcp = robot.get_robot_tcp_pose()
    joints = robot.get_robot_joint_state()
    print("-" * 70)
    print(f"Final TCP:   {None if tcp is None else [round(v, 4) for v in tcp[0].tolist()]}")
    print(f"Final joints:{None if joints is None else [round(v, 4) for v in joints.tolist()]}")
    print("-" * 70)

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Summary: {passed}/{total} passed")

    if not args.no_gui:
        robot.set_status(f"Complete: {passed}/{total} primitives passed")
        robot.step(1.0)
    robot.cleanup()
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
