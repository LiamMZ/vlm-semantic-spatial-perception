"""
Isolated motion planning test — xArm IK + OMPL trajectory to a target point.

Tests the full IK → BITstar planning pipeline without perception, PDDL, or
task orchestration. Useful for debugging IK rejections and collision failures.

Run:
    uv run scripts/test_motion_planning.py
    uv run scripts/test_motion_planning.py --pos 0.4 -0.22 0.08
    uv run scripts/test_motion_planning.py --orientation side
    uv run scripts/test_motion_planning.py --pybullet-gui
    uv run scripts/test_motion_planning.py --robot-ip 192.168.1.224
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface

# Target from the failing log.
DEFAULT_POS = [0.4155050281195155, -0.2270009107215674, 0.03572339888118636]
# side orientation quaternion used by XArmPybulletPlannedPrimitives
SIDE_QUAT   = [-0.6894, 0.0305, -0.7237, 0.0033]
TOP_QUAT    = [-0.9983, 0.0314, 0.0438, 0.0223]

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_GREY   = "\033[90m"


def _setup_logging() -> logging.Logger:
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
    root.setLevel(logging.DEBUG)
    return logging.getLogger("motion_plan_test")


log = _setup_logging()


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")


def _info(label: str, value: object = "") -> None:
    if value == "":
        print(f"    {_YELLOW}•{_RESET} {label}")
    else:
        print(f"    {_YELLOW}{label}:{_RESET} {value}")


def _section(msg: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {msg} " + "─" * max(0, 68 - len(msg)) + _RESET)


def connect_robot(robot_ip: str) -> Optional[object]:
    """Return a RawXArmRobotAdapter or None if the SDK is unavailable."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_xarm_real_pybullet_planned_primitives",
            Path(__file__).parent / "test_xarm_real_pybullet_planned_primitives.py",
        )
        mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                  # type: ignore[union-attr]
        adapter = mod.RawXArmRobotAdapter(robot_ip)
        joints  = adapter.get_robot_joint_state()
        if joints is not None:
            _ok(f"xArm connected at {robot_ip}  joints={np.round(joints, 3).tolist()}")
            return adapter
        log.warning("Connected but no joint state — continuing sim-only")
        return None
    except Exception as exc:
        log.warning("Robot connection failed (%s) — using sim home pose", exc)
        return None


def run(args: argparse.Namespace) -> int:
    import pybullet as _p
    target_pos  = np.array(args.pos, dtype=float)
    target_quat = np.array(SIDE_QUAT if args.orientation == "side" else TOP_QUAT, dtype=float)
    if args.no_orientation:
        target_quat = None

    _section("Configuration")
    _info("Target position",   target_pos.round(4).tolist())
    _info("Target orientation", args.orientation if not args.no_orientation else "position-only")
    _info("Planning time",     f"{args.planning_time}s")
    _info("PyBullet GUI",      args.pybullet_gui)

    # --- Robot connection (optional) ---
    adapter = None
    if args.robot_ip:
        _section("Robot connection")
        adapter = connect_robot(args.robot_ip)

    # --- PyBullet interface ---
    _section("PyBullet IK/planning interface")
    iface = XArmPybulletInterface(use_gui=args.pybullet_gui)

    if adapter is not None:
        joints = adapter.get_robot_joint_state()
        if joints is not None:
            iface.set_current_joint_state(joints)
            _ok(f"FK seeded from live robot joints: {np.round(joints, 3).tolist()}")
        else:
            _fail("Connected to robot but could not read joint state")
            iface.cleanup()
            return 1
    else:
        _info("No robot — querying current joint state from interface")
        joints = iface.get_robot_joint_state()
        _info("Interface joint state", np.round(joints, 4).tolist() if joints is not None else "None")

    tcp_pose = iface.get_robot_tcp_pose()
    if tcp_pose is not None:
        _ok(f"Current TCP  pos={np.round(tcp_pose[0], 4).tolist()}  "
            f"quat={np.round(tcp_pose[1], 4).tolist()}")
    else:
        _fail("Could not read current TCP pose")

    # --- IK test ---
    _section("IK solve (8 random restarts)")
    t0 = time.monotonic()
    goal_joints = iface._ik_solve(
        target_pos=target_pos,
        target_quat=target_quat,
        position_tolerance=args.pos_tol,
        orientation_tolerance_rad=args.ori_tol,
    )
    ik_time = time.monotonic() - t0

    if goal_joints is None:
        _fail(f"IK failed after {ik_time:.2f}s — target may be out of workspace")
        _info("Try", "--no-orientation to relax the orientation constraint")
        _info("Try", "--pos with a reachable position (e.g. 0.4 0.0 0.2)")
        if not args.pybullet_gui:
            iface.cleanup()
        return 1

    _ok(f"IK solved in {ik_time:.2f}s")
    _info("Goal joints (rad)", np.round(goal_joints, 4).tolist())

    # Forward-kinematics check on the IK solution.
    saved = iface._joints.copy()
    iface._joints = goal_joints
    fk_pose = iface.get_robot_tcp_pose()
    iface._joints = saved
    if fk_pose is not None:
        pos_err = float(np.linalg.norm(fk_pose[0] - target_pos))
        _info("FK position error", f"{pos_err:.4f} m  (tol={args.pos_tol})")
        if pos_err <= args.pos_tol:
            _ok("Position within tolerance")
        else:
            _fail(f"Position error {pos_err:.4f} > tolerance {args.pos_tol}")

    # --- Collision diagnostics on start state ---
    _section("Start-state collision diagnostics")
    client  = iface._physics_client
    rid     = iface._robot_id
    n_links = _p.getNumJoints(rid, physicsClientId=client)
    floor_id = iface._floor_body_id
    floor_start = iface._floor_check_start_link
    self_end = iface._arm_self_collision_end_link
    iface._apply_joints_to_sim()

    floor_hits = []
    if floor_id is not None:
        for lx in range(floor_start, n_links):
            pts = _p.getClosestPoints(bodyA=rid, bodyB=floor_id,
                                      distance=0.005, linkIndexA=lx,
                                      physicsClientId=client)
            if pts:
                floor_hits.append((lx, min(c[8] for c in pts)))
    if floor_hits:
        _fail(f"Floor collision on links: {floor_hits}  (floor_check_start_link={floor_start})")
        _info("Fix", f"Increase _floor_check_start_link above {max(l for l,_ in floor_hits)}")
    else:
        _ok("No floor collision on start state")

    self_hits = []
    for la in range(-1, self_end):
        for lb in range(la + 2, self_end + 1):
            pts = _p.getClosestPoints(bodyA=rid, bodyB=rid, distance=0.0,
                                      linkIndexA=la, linkIndexB=lb,
                                      physicsClientId=client)
            if pts:
                self_hits.append((la, lb, min(c[8] for c in pts)))
    if self_hits:
        _fail(f"Self-collision on link pairs: {self_hits}")
    else:
        _ok("No self-collision on start state")

    # --- Goal-state collision diagnostics ---
    _section("Goal-state collision diagnostics")
    if goal_joints is not None:
        for idx, val in zip(iface._movable_joints[:len(goal_joints)], goal_joints):
            _p.resetJointState(rid, idx, float(val), physicsClientId=client)

        goal_floor_hits = []
        if floor_id is not None:
            for lx in range(floor_start, n_links):
                pts = _p.getClosestPoints(bodyA=rid, bodyB=floor_id,
                                          distance=0.005, linkIndexA=lx,
                                          physicsClientId=client)
                if pts:
                    goal_floor_hits.append((lx, min(c[8] for c in pts)))
        if goal_floor_hits:
            _fail(f"Floor collision at goal on links: {goal_floor_hits}")
        else:
            _ok("No floor collision at goal state")

        goal_self_hits = []
        for la in range(-1, self_end):
            for lb in range(la + 2, self_end + 1):
                pts = _p.getClosestPoints(bodyA=rid, bodyB=rid, distance=0.0,
                                          linkIndexA=la, linkIndexB=lb,
                                          physicsClientId=client)
                if pts:
                    goal_self_hits.append((la, lb, min(c[8] for c in pts)))
        if goal_self_hits:
            _fail(f"Self-collision at goal on link pairs: {goal_self_hits}")
        else:
            _ok("No self-collision at goal state")

        _info("Goal joint values (rad)", np.round(goal_joints, 4).tolist())
        joint_limits = iface._get_joint_limits()
        out_of_range = [
            (i, round(v, 4), round(joint_limits[i][0], 4), round(joint_limits[i][1], 4))
            for i, v in enumerate(goal_joints)
            if not (joint_limits[i][0] <= v <= joint_limits[i][1])
        ]
        if out_of_range:
            _fail(f"Goal joints out of URDF limits: {out_of_range}")
        else:
            _ok("All goal joints within URDF limits")

        # Restore sim to start state.
        iface._apply_joints_to_sim()

    # --- Grasp sampler ---
    _section(f"Antipodal grasp sampling (seed={args.orientation}, n=36)")
    from src.grasp_planner import GraspPlanner
    t0 = time.monotonic()
    grasp_planner = GraspPlanner(iface)
    candidate = grasp_planner.plan(target_pos, seed_orientation=args.orientation)
    sample_time = time.monotonic() - t0

    if candidate is None:
        _fail(f"Grasp sampler found no collision-free candidate ({sample_time:.2f}s)")
    else:
        _ok(f"Grasp sampler succeeded ({sample_time:.2f}s)")
        _info("Selected orientation (xyzw)", np.round(candidate.orientation, 4).tolist())
        _info("Approach angle",  f"{np.degrees(candidate.approach_angle_rad):.1f}°")
        _info("Seed",            candidate.seed_orientation)
        _info("Manipulability",  f"{candidate.manipulability:.4f}")
        _info("Goal joints",     np.round(candidate.joints, 4).tolist())
        # Use the sampler's orientation for subsequent planning.
        target_quat = candidate.orientation
        goal_joints = candidate.joints

    # --- Motion planning ---
    _section(f"Motion planning  (BITstar, budget={args.planning_time}s)")
    plan_quat = target_quat if candidate is not None else (
        target_quat if not args.no_orientation else None
    )
    t0 = time.monotonic()
    traj = iface.plan_joint_trajectory_to_pose(
        target_position=target_pos.tolist(),
        target_orientation=plan_quat.tolist() if plan_quat is not None else None,
        position_tolerance=args.pos_tol,
        orientation_tolerance_rad=args.ori_tol,
        planning_time=args.planning_time,
        interpolate_n=64,
        collision_margin=0.005,
    )
    plan_time = time.monotonic() - t0

    if traj is None:
        _fail(f"Planning failed after {plan_time:.2f}s")
        _info("Possible causes",
              "goal in collision / workspace boundary / BITstar timeout")
        if goal_joints is not None:
            for idx, val in zip(iface._movable_joints[:len(goal_joints)], goal_joints):
                _p.resetJointState(iface._robot_id, idx, float(val),
                                   physicsClientId=iface._physics_client)
            _info("Sim set to goal joints for inspection")
        _section("Paused at goal joints — inspect PyBullet GUI")
        input("  Press Enter to exit...")
        iface.cleanup()
        return 1

    _ok(f"Trajectory found in {plan_time:.2f}s — {traj.shape[0]} waypoints × {traj.shape[1]} DOF")
    _info("Start joints", np.round(traj[0],  4).tolist())
    _info("End joints",   np.round(traj[-1], 4).tolist())

    joint_ranges = np.max(traj, axis=0) - np.min(traj, axis=0)
    _info("Joint ranges (rad)", np.round(joint_ranges, 3).tolist())

    # --- Optional execution ---
    if adapter is not None and not args.no_execute:
        _section("Executing trajectory on xArm")
        try:
            from src.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives
            prims = XArmPybulletPlannedPrimitives(
                robot=adapter, planner=iface, logger=log.getChild("Primitives"),
            )
            result = prims.move_gripper_to_pose(
                target_position=target_pos.tolist(),
                preset_orientation=args.orientation,
                execute=True,
                planning_time=args.planning_time,
            )
            if isinstance(result, dict) and result.get("success"):
                _ok("Execution complete")
            else:
                reason = result.get("reason", "unknown") if isinstance(result, dict) else str(result)
                _fail(f"Execution failed: {reason}")
        except Exception as exc:
            _fail(f"Execution raised: {exc}")
            log.exception("Execution error")

    # Leave sim at trajectory end pose (or sampler goal joints if planning failed).
    display_joints = traj[-1] if traj is not None else goal_joints
    if display_joints is not None:
        for idx, val in zip(iface._movable_joints[:len(display_joints)], display_joints):
            _p.resetJointState(iface._robot_id, idx, float(val),
                               physicsClientId=iface._physics_client)
        _info("Sim set to goal joints for inspection")

    _section("Paused — inspect PyBullet GUI")
    input("  Press Enter to exit...")

    iface.cleanup()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated xArm motion planning test")
    parser.add_argument(
        "--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=DEFAULT_POS,
        help="Target TCP position in robot base frame (default: failing log point)",
    )
    parser.add_argument(
        "--orientation", choices=["side", "top_down"], default="side",
        help="Preset orientation (default: side)",
    )
    parser.add_argument(
        "--no-orientation", action="store_true",
        help="Solve IK and plan position-only (ignore orientation constraint)",
    )
    parser.add_argument(
        "--pos-tol", type=float, default=0.02,
        help="IK position tolerance in metres (default: 0.02)",
    )
    parser.add_argument(
        "--ori-tol", type=float, default=0.6,
        help="IK orientation tolerance in radians (default: 0.6)",
    )
    parser.add_argument(
        "--planning-time", type=float, default=5.0,
        help="OMPL planning time budget in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--pybullet-gui", action="store_true",
        help="Open PyBullet GUI to visualize the planned trajectory",
    )
    parser.add_argument(
        "--robot-ip", default=None,
        help="xArm IP to seed FK from live joints and optionally execute",
    )
    parser.add_argument(
        "--no-execute", action="store_true",
        help="Plan only — do not send motion commands even if robot is connected",
    )
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
