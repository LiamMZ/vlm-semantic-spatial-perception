"""
B1Z1 URDF load test — PyBullet.

Loads the combined B1 quadruped + Z1 arm URDF, prints joint info, and
optionally opens a GUI for visual inspection.

Run:
    uv run scripts/test_b1z1_urdf.py
    uv run scripts/test_b1z1_urdf.py --no-gui   # headless / CI
    uv run scripts/test_b1z1_urdf.py --pause 10  # keep GUI open longer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not installed. Run: pip install pybullet")
    sys.exit(1)

URDF_PATH = (
    PROJECT_ROOT
    / "src/kinematics/sim/urdfs/b1z1_description/b1z1.urdf"
)

# Standing height for B1: legs are ~0.7 m extended, spawn above ground
_SPAWN_HEIGHT = 0.55


def load_and_inspect(gui: bool, pause: float) -> None:
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.3],
            physicsClientId=client,
        )

    print(f"\nLoading URDF: {URDF_PATH}")
    robot = p.loadURDF(
        str(URDF_PATH),
        basePosition=[0, 0, _SPAWN_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        physicsClientId=client,
    )
    print(f"  robot id: {robot}")

    num_joints = p.getNumJoints(robot, physicsClientId=client)
    print(f"  joints:   {num_joints}\n")

    joint_type_names = {
        p.JOINT_REVOLUTE:   "revolute",
        p.JOINT_PRISMATIC:  "prismatic",
        p.JOINT_SPHERICAL:  "spherical",
        p.JOINT_PLANAR:     "planar",
        p.JOINT_FIXED:      "fixed",
    }

    revolute_indices = []
    print(f"  {'idx':>3}  {'type':<10}  {'name'}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*40}")
    for i in range(num_joints):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        idx       = info[0]
        name      = info[1].decode()
        jtype     = info[2]
        tname     = joint_type_names.get(jtype, str(jtype))
        print(f"  {idx:>3}  {tname:<10}  {name}")
        if jtype == p.JOINT_REVOLUTE:
            revolute_indices.append(idx)

    print(f"\n  Actuated (revolute) joints: {len(revolute_indices)}")
    print(f"  Indices: {revolute_indices}")

    # Set a neutral standing pose for the legs and a tucked arm
    # B1 leg joints: hip, thigh, calf per leg (12 joints)
    # Z1 arm joints: joint1-6 + gripperMover (7 joints)
    leg_joint_names = {
        "FR_hip_joint":   0.0,   "FR_thigh_joint":  0.7,  "FR_calf_joint": -1.4,
        "FL_hip_joint":   0.0,   "FL_thigh_joint":  0.7,  "FL_calf_joint": -1.4,
        "RR_hip_joint":   0.0,   "RR_thigh_joint":  0.7,  "RR_calf_joint": -1.4,
        "RL_hip_joint":   0.0,   "RL_thigh_joint":  0.7,  "RL_calf_joint": -1.4,
    }
    arm_joint_names = {
        "joint1": 0.0,
        "joint2": 0.5,
        "joint3": -0.5,
        "joint4": 0.5,
        "joint5": 0.0,
        "joint6": 0.0,
        "finger_joint": 0.0,  # Robotiq 2F-140: 0=open, 0.7=closed
    }
    pose = {**leg_joint_names, **arm_joint_names}

    name_to_idx: dict[str, int] = {}
    for i in range(num_joints):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        name_to_idx[info[1].decode()] = info[0]

    for jname, angle in pose.items():
        if jname in name_to_idx:
            p.resetJointState(
                robot, name_to_idx[jname], angle, physicsClientId=client
            )

    print("\nInitial pose set. Running simulation...")

    if gui:
        # Step for a few seconds so the robot settles visually
        steps = int(pause / (1.0 / 240.0))
        for _ in range(steps):
            p.stepSimulation(physicsClientId=client)
            time.sleep(1.0 / 240.0)
        print(f"GUI open for {pause:.0f}s — close the window or Ctrl-C to exit.")
        try:
            while p.isConnected(client):
                p.stepSimulation(physicsClientId=client)
                time.sleep(1.0 / 240.0)
        except (KeyboardInterrupt, p.error):
            pass
    else:
        # Headless: just step a few frames to confirm no crashes
        for _ in range(100):
            p.stepSimulation(physicsClientId=client)
        print("Headless simulation: 100 steps completed without error.")

    p.disconnect(client)
    print("\nPASS: b1z1.urdf loaded and simulated successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="B1Z1 URDF PyBullet load test")
    parser.add_argument("--no-gui", action="store_true", help="Run headless (DIRECT mode)")
    parser.add_argument("--pause", type=float, default=5.0,
                        help="Seconds to run GUI before handing control to user (default: 5)")
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    load_and_inspect(gui=not args.no_gui, pause=args.pause)


if __name__ == "__main__":
    main()
