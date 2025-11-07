"""
Perception module for VLM-based object detection and spatial understanding.

This module provides interfaces to Vision-Language Models (VLMs) like
Gemini Robotics-ER for:
- Object detection and localization
- Spatial reasoning
- Trajectory planning
- Task decomposition
"""

from .gemini_robotics import (
    GeminiRoboticsClient,
    ObjectDetectionResult,
    SpatialReasoningResult,
    TrajectoryResult,
    TaskDecompositionResult,
    InteractionPointResult
)
from .object_tracker import ObjectTracker
from .object_registry import (
    DetectedObject,
    InteractionPoint,
    DetectedObjectRegistry
)
from .vlm_detector import VLMObjectDetector

__all__ = [
    # Legacy Gemini client (high-level capabilities)
    "GeminiRoboticsClient",
    "ObjectDetectionResult",
    "SpatialReasoningResult",
    "TrajectoryResult",
    "TaskDecompositionResult",
    "InteractionPointResult",
    # Object tracking system (focused on detection & affordances)
    "ObjectTracker",
    "DetectedObject",
    "InteractionPoint",
    "DetectedObjectRegistry",
    # World model integration
    "VLMObjectDetector",
]
