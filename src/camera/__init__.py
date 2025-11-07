"""Camera abstraction layer for various camera types"""

from .base_camera import BaseCamera, CameraFrame, CameraIntrinsics
from .camera_utils import (
    create_camera,
    create_camera_from_config,
    depth_image_to_point_cloud,
    estimate_object_depth,
    pixel_to_3d,
    visualize_depth,
)
from .webcam_camera import WebcamCamera

# Conditional import for RealSense
try:
    from .realsense_camera import RealSenseCamera
    __all__ = [
        "BaseCamera",
        "CameraFrame",
        "CameraIntrinsics",
        "WebcamCamera",
        "RealSenseCamera",
        "create_camera",
        "create_camera_from_config",
        "pixel_to_3d",
        "depth_image_to_point_cloud",
        "estimate_object_depth",
        "visualize_depth",
    ]
except ImportError:
    __all__ = [
        "BaseCamera",
        "CameraFrame",
        "CameraIntrinsics",
        "WebcamCamera",
        "create_camera",
        "create_camera_from_config",
        "pixel_to_3d",
        "depth_image_to_point_cloud",
        "estimate_object_depth",
        "visualize_depth",
    ]