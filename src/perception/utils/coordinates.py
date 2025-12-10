"""
Coordinate transformation utilities for 2D/3D conversions.

This module handles all coordinate conversions between:
- Normalized coordinates (0-1000 scale used by VLM)
- Pixel coordinates (image space)
- 3D world coordinates (camera frame, meters)

The VLM only works with normalized 2D coordinates. All 3D reconstruction
is performed by these utility functions using depth images and camera intrinsics.
"""

from typing import Tuple, List, Optional, Any
import numpy as np


def normalized_to_pixel(
    normalized_pos: List[int],
    image_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert normalized 2D position (0-1000 scale) to pixel coordinates.

    Args:
        normalized_pos: [y, x] in 0-1000 scale (from VLM)
        image_shape: (height, width) of the image

    Returns:
        (pixel_y, pixel_x) clamped to image bounds

    Example:
        >>> normalized_to_pixel([500, 750], (480, 640))
        (240, 480)
    """
    height, width = image_shape

    # Convert normalized (0-1000) to pixels
    pixel_y = int((normalized_pos[0] / 1000.0) * height)
    pixel_x = int((normalized_pos[1] / 1000.0) * width)

    # Clamp to image bounds
    pixel_y = max(0, min(height - 1, pixel_y))
    pixel_x = max(0, min(width - 1, pixel_x))

    return pixel_y, pixel_x


def pixel_to_normalized(
    pixel_pos: Tuple[int, int],
    image_shape: Tuple[int, int]
) -> List[int]:
    """
    Convert pixel coordinates to normalized 2D position (0-1000 scale).

    Args:
        pixel_pos: (pixel_y, pixel_x)
        image_shape: (height, width) of the image

    Returns:
        [y, x] in 0-1000 scale (for VLM)

    Example:
        >>> pixel_to_normalized((240, 480), (480, 640))
        [500, 750]
    """
    pixel_y, pixel_x = pixel_pos
    height, width = image_shape

    # Convert pixels to normalized (0-1000)
    norm_y = int((pixel_y / height) * 1000.0)
    norm_x = int((pixel_x / width) * 1000.0)

    # Clamp to valid range
    norm_y = max(0, min(1000, norm_y))
    norm_x = max(0, min(1000, norm_x))

    return [norm_y, norm_x]


def compute_3d_position(
    position_2d: List[int],
    depth_frame: np.ndarray,
    camera_intrinsics: Any
) -> Optional[np.ndarray]:
    """
    Convert 2D normalized position to 3D world coordinates using depth.

    This function performs back projection from 2D image coordinates to 3D camera space.
    The VLM provides 2D positions in normalized coordinates (0-1000), which we convert
    to pixels, look up depth, and back-project to 3D using camera intrinsics.

    Args:
        position_2d: [y, x] in 0-1000 normalized scale (from VLM)
        depth_frame: Depth image in meters
        camera_intrinsics: Camera intrinsics with fx, fy, cx, cy (or ppx, ppy)

    Returns:
        [x, y, z] in meters (camera frame) or None if invalid depth

    Note:
        - The VLM never sees depth information, only RGB images
        - All 3D reconstruction happens in this function
        - Uses pinhole camera model for back projection:
          * x = (u - cx) * z / fx
          * y = (v - cy) * z / fy
          * z = depth

    Example:
        >>> depth = np.random.rand(480, 640) * 3.0  # meters
        >>> intrinsics = ...
        >>> compute_3d_position([500, 500], depth, intrinsics)
        array([0.125, -0.050, 0.850], dtype=float32)
    """
    try:
        height, width = depth_frame.shape[:2]

        # Convert normalized coordinates (0-1000) to pixels
        pixel_y, pixel_x = normalized_to_pixel(position_2d, (height, width))

        # Look up depth at this pixel
        depth = depth_frame[pixel_y, pixel_x]

        if depth <= 0 or np.isnan(depth):
            print(f"      Invalid depth at pixel ({pixel_y}, {pixel_x}): {depth}")
            return None

        # Get camera intrinsics (support both naming conventions)
        # Use width/height from intrinsics or from depth_frame if not available
        width = getattr(camera_intrinsics, 'width', depth_frame.shape[1])
        height = getattr(camera_intrinsics, 'height', depth_frame.shape[0])

        fx = getattr(camera_intrinsics, 'fx', width / 2)
        fy = getattr(camera_intrinsics, 'fy', height / 2)
        cx = getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', width / 2))
        cy = getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', height / 2))

        # Back-project to 3D using pinhole camera model
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    except Exception as e:
        print(f"3D position computation failed: {e}")
        return None


def project_3d_to_2d(
    position_3d: np.ndarray,
    camera_intrinsics: Any,
    image_shape: Tuple[int, int]
) -> Optional[List[int]]:
    """
    Project 3D world coordinates to 2D normalized position.

    This is the inverse of compute_3d_position. Useful for projecting
    known 3D points back to image coordinates.

    Args:
        position_3d: [x, y, z] in meters (camera frame)
        camera_intrinsics: Camera intrinsics with fx, fy, cx, cy (or ppx, ppy)
        image_shape: (height, width) of the image

    Returns:
        [y, x] in 0-1000 normalized scale or None if behind camera

    Note:
        - Uses pinhole camera projection model:
          * u = fx * (x / z) + cx
          * v = fy * (y / z) + cy
        - Returns None if z <= 0 (point behind camera)

    Example:
        >>> position_3d = np.array([0.125, -0.050, 0.850])
        >>> intrinsics = ...
        >>> project_3d_to_2d(position_3d, intrinsics, (480, 640))
        [500, 500]
    """
    try:
        x, y, z = position_3d

        if z <= 0:
            return None  # Point behind camera

        # Get camera intrinsics (support both naming conventions)
        # Use width/height from intrinsics or from image_shape
        width = getattr(camera_intrinsics, 'width', image_shape[1])
        height = getattr(camera_intrinsics, 'height', image_shape[0])

        fx = getattr(camera_intrinsics, 'fx', width / 2)
        fy = getattr(camera_intrinsics, 'fy', height / 2)
        cx = getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', width / 2))
        cy = getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', height / 2))

        # Project to pixel coordinates using pinhole model
        pixel_x = int(fx * (x / z) + cx)
        pixel_y = int(fy * (y / z) + cy)

        # Convert pixels to normalized coordinates
        normalized_pos = pixel_to_normalized((pixel_y, pixel_x), image_shape)

        return normalized_pos

    except Exception as e:
        print(f"3D to 2D projection failed: {e}")
        return None


def batch_compute_3d_positions(
    positions_2d: List[List[int]],
    depth_frame: np.ndarray,
    camera_intrinsics: Any
) -> List[Optional[np.ndarray]]:
    """
    Convert multiple 2D positions to 3D in batch.

    Args:
        positions_2d: List of [y, x] in 0-1000 normalized scale
        depth_frame: Depth image in meters
        camera_intrinsics: Camera intrinsics

    Returns:
        List of [x, y, z] positions or None for invalid depths

    Example:
        >>> positions_2d = [[500, 500], [600, 700]]
        >>> batch_compute_3d_positions(positions_2d, depth, intrinsics)
        [array([0.1, 0.0, 0.8]), array([0.2, 0.1, 0.9])]
    """
    return [
        compute_3d_position(pos_2d, depth_frame, camera_intrinsics)
        for pos_2d in positions_2d
    ]


def get_depth_at_normalized_position(
    position_2d: List[int],
    depth_frame: np.ndarray
) -> Optional[float]:
    """
    Get depth value at a normalized 2D position.

    Args:
        position_2d: [y, x] in 0-1000 normalized scale
        depth_frame: Depth image in meters

    Returns:
        Depth in meters or None if invalid

    Example:
        >>> get_depth_at_normalized_position([500, 500], depth_frame)
        0.850
    """
    try:
        height, width = depth_frame.shape[:2]
        pixel_y, pixel_x = normalized_to_pixel(position_2d, (height, width))

        depth = depth_frame[pixel_y, pixel_x]

        if depth <= 0 or np.isnan(depth):
            return None

        return float(depth)

    except Exception:
        return None
