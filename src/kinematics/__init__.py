"""
Kinematics module.

XArmPybulletInterface is the default sim-safe robot provider.
CuRoboMotionPlanner (requires CUDA + curobo) is available when connected
to physical hardware.
"""

from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface, create_sim_interface

__all__ = ["XArmPybulletInterface", "create_sim_interface"]
