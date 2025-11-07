"""
Utility functions for camera operations.
"""

from typing import Dict, Type

import numpy as np
import yaml

from .base_camera import BaseCamera, CameraIntrinsics
from .webcam_camera import WebcamCamera

try:
    from .realsense_camera import RealSenseCamera
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


def create_camera_from_config(config_path: str) -> BaseCamera:
    """
    Create a camera instance from configuration file.

    Args:
        config_path: Path to camera configuration YAML file

    Returns:
        BaseCamera: Camera instance

    Raises:
        ValueError: If camera type is not supported
        FileNotFoundError: If config file not found
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    camera_type = config.get('default_camera', 'webcam').lower()

    return create_camera(camera_type, config)


def create_camera(camera_type: str, config: Dict = None) -> BaseCamera:
    """
    Create a camera instance by type.

    Args:
        camera_type: Type of camera ('webcam', 'realsense')
        config: Configuration dictionary

    Returns:
        BaseCamera: Camera instance

    Raises:
        ValueError: If camera type is not supported
    """
    config = config or {}

    if camera_type == 'webcam':
        webcam_config = config.get('webcam', {})
        resolution = webcam_config.get('resolution', {})

        return WebcamCamera(
            device_id=webcam_config.get('device_id', 0),
            width=resolution.get('width', 640),
            height=resolution.get('height', 480),
            fps=webcam_config.get('framerate', 30)
        )

    elif camera_type == 'realsense':
        if not REALSENSE_AVAILABLE:
            raise ValueError(
                "RealSense camera not available. Install with: pip install pyrealsense2"
            )

        realsense_config = config.get('realsense', {})
        resolution = realsense_config.get('resolution', {})

        return RealSenseCamera(
            width=resolution.get('width', 640),
            height=resolution.get('height', 480),
            fps=realsense_config.get('framerate', 30),
            enable_depth=realsense_config.get('depth_enabled', True)
        )

    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")


def pixel_to_3d(
    u: int,
    v: int,
    depth: float,
    intrinsics: CameraIntrinsics
) -> np.ndarray:
    """
    Convert pixel coordinates and depth to 3D point in camera frame.

    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate
        depth: Depth value in meters
        intrinsics: Camera intrinsic parameters

    Returns:
        np.ndarray: 3D point [x, y, z] in meters
    """
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    z = depth

    return np.array([x, y, z])


def depth_image_to_point_cloud(
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_image: np.ndarray = None,
    max_depth: float = 10.0
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.

    Args:
        depth_image: Depth image (H, W) in meters
        intrinsics: Camera intrinsic parameters
        color_image: Optional RGB image (H, W, 3)
        max_depth: Maximum valid depth in meters

    Returns:
        np.ndarray: Point cloud of shape (N, 3) or (N, 6) if color provided
    """
    h, w = depth_image.shape

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Get valid depth points
    valid_mask = (depth_image > 0) & (depth_image < max_depth)

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_image[valid_mask]

    # Convert to 3D
    x = (u_valid - intrinsics.cx) * depth_valid / intrinsics.fx
    y = (v_valid - intrinsics.cy) * depth_valid / intrinsics.fy
    z = depth_valid

    points = np.stack([x, y, z], axis=-1)

    # Add color if provided
    if color_image is not None:
        colors = color_image[valid_mask] / 255.0
        points = np.concatenate([points, colors], axis=-1)

    return points


def estimate_object_depth(
    bbox: tuple,
    depth_image: np.ndarray,
    method: str = 'median'
) -> float:
    """
    Estimate object depth from bounding box region in depth image.

    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        depth_image: Depth image in meters
        method: Estimation method ('median', 'mean', 'min')

    Returns:
        float: Estimated depth in meters
    """
    x_min, y_min, x_max, y_max = bbox

    # Extract region
    region = depth_image[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Filter out invalid depths
    valid_depths = region[(region > 0) & (region < 10.0)]

    if len(valid_depths) == 0:
        return 0.0

    if method == 'median':
        return float(np.median(valid_depths))
    elif method == 'mean':
        return float(np.mean(valid_depths))
    elif method == 'min':
        return float(np.min(valid_depths))
    else:
        raise ValueError(f"Unknown depth estimation method: {method}")


def visualize_depth(depth_image: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """
    Convert depth image to visualization (color-coded).

    Args:
        depth_image: Depth image in meters
        max_depth: Maximum depth for color mapping

    Returns:
        np.ndarray: RGB visualization image
    """
    import cv2

    # Normalize to 0-255
    depth_normalized = np.clip(depth_image / max_depth * 255, 0, 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return depth_colored
