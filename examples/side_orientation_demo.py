#!/usr/bin/env python3
"""
Simple demo to test `move_to_pose_with_preparation` using preset_orientation="side".

This runs planning only (execute=False) for safety, printing planning results.
Assumes the workspace config and robot IP are set in `CuRoboMotionPlanner.initialize_default_config()`.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kinematics.xarm_curobo_interface import CuRoboMotionPlanner


def main():
    # Instantiate planner; no static camera TF provided for simplicity.
    planner = CuRoboMotionPlanner(static_camera_tf=None, robot_ip='192.168.1.224')

    # Example target in camera frame: slightly forward and above the table.
    # Adjust as needed for your setup. Units are meters.

    # Plan with side orientation preset, keeping camera-frame input.
    # - execute=False for safety; set True only when connected and safe.
    # - is_place=False so we don't add the extra Z offset.
    print("\n--- Planning with preset_orientation='side' (move_to_pose) ---")
    # Convert camera-frame target to base frame first for move_to_pose
    pos_base, quat_base = planner.convert_cam_pose_to_base([0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0])
    success, trajectory, dt = planner.move_to_pose(
        target_position=[
            0.4257207583414897,
            -0.33239454733845436,
            0.19909752107068024
          ],
        target_orientation=None,
        force_top_down=False,
        preset_orientation="top_down",
        unconstrained_orientation=False,
        planning_timeout=5.0,
        execute=True,
        speed_factor=1.0,
    )

    print(f"\nResult: success={success}, dt={dt}")
    if success and trajectory is not None:
        try:
            # Attempt to introspect trajectory length for a quick summary
            num_points = len(trajectory.position) if hasattr(trajectory, "position") else None
            print(f"Trajectory points: {num_points}")
        except Exception:
            print("Trajectory present, but unable to introspect points.")

    print("Demo completed.")


if __name__ == "__main__":
    sys.exit(main())
