"""
Perception module for VLM-based object detection and spatial understanding.

This module provides interfaces to Vision-Language Models (VLMs) like
Gemini Robotics-ER for:
- Object detection and localization
- Spatial reasoning
- Trajectory planning
- Task decomposition
"""


from .object_tracker import ObjectTracker
from .object_registry import (
    DetectedObject,
    InteractionPoint,
    DetectedObjectRegistry
)
from .continuous_tracker import ContinuousObjectTracker, TrackingStats

__all__ = [
    # Object tracking system (focused on detection & affordances)
    "ObjectTracker",
    "DetectedObject",
    "InteractionPoint",
    "DetectedObjectRegistry",
    # Continuous background tracking
    "ContinuousObjectTracker",
    "TrackingStats",
]
