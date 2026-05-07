from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GraspCandidate:
    """A collision-free grasp configuration returned by GraspPlanner."""

    position: np.ndarray          # TCP target position in robot base frame
    orientation: np.ndarray       # TCP orientation as xyzw quaternion
    joints: np.ndarray            # IK solution (dof,)
    approach_angle_rad: float     # Angular distance from seed orientation (geodesic, radians)
    manipulability: float = 0.0   # Yoshikawa manipulability score (higher = better)
    seed_orientation: str = ""    # Name of the seed orientation this was sampled from
