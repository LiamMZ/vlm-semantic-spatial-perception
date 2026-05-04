# Backward-compatibility shim — real implementation is in molmo_point_detector.py
from .molmo_point_detector import (  # noqa: F401
    MolmoPointDetector,
    MolmoPointDetector as MolmoInteractionPointDetector,
    DEFAULT_ACTIONS,
    _ACTION_PROMPTS,
    _extract_points,
    _transform_cam_to_world,
)
