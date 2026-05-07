"""
Perception module for VLM-based object detection and spatial understanding.

This module provides interfaces to Vision-Language Models (VLMs) like
Gemini Robotics-ER for:
- Object detection and localization
- Spatial reasoning
- Trajectory planning
- Task decomposition
"""


from .object_tracker import ObjectTracker, ContinuousObjectTracker, TrackingStats
from .object_registry import (
    DetectedObject,
    InteractionPoint,
    DetectedObjectRegistry
)
try:
    from .gsam2_object_tracker import GSAM2ObjectTracker, GSAM2ContinuousObjectTracker
    from .molmo_point_detector import MolmoPointDetector
    MolmoInteractionPointDetector = MolmoPointDetector  # backward-compat alias
except ImportError:
    pass  # optional heavy deps (supervision, torch, SAM2, transformers) not installed

try:
    from .clearance import GripperGeometry, ClearanceProfile
except ImportError:
    pass

__all__ = [
    # Object tracking system (focused on detection & affordances)
    "ObjectTracker",
    "DetectedObject",
    "InteractionPoint",
    "DetectedObjectRegistry",
    # Continuous background tracking
    "ContinuousObjectTracker",
    "TrackingStats",
    # GSAM2-based tracking (RAM+ + GroundingDINO + SAM2)
    "GSAM2ObjectTracker",
    "GSAM2ContinuousObjectTracker",
    # Molmo-based interaction point detection
    "MolmoPointDetector",
    "MolmoInteractionPointDetector",  # backward-compat alias
    # Clearance / gripper geometry
    "GripperGeometry",
    "ClearanceProfile",
]
