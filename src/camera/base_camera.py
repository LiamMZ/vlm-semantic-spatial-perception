"""
Abstract base class for camera interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height
    distortion: Optional[np.ndarray] = None  # Distortion coefficients

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion": self.distortion.tolist() if self.distortion is not None else None,
        }

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 camera matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class CameraFrame:
    """Container for camera frame data."""

    color: np.ndarray  # RGB color image (H, W, 3)
    depth: Optional[np.ndarray] = None  # Depth image (H, W) in meters
    timestamp: Optional[float] = None  # Frame timestamp
    frame_id: Optional[int] = None  # Frame sequence number


class BaseCamera(ABC):
    """
    Abstract base class for camera interfaces.
    
    All camera implementations should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """
        Capture a single RGB frame from the camera.
        
        Returns:
            np.ndarray: RGB image with shape (H, W, 3)
        """
        pass
    
    @abstractmethod
    def get_depth(self) -> np.ndarray:
        """
        Get depth frame from the camera.
        
        Returns:
            np.ndarray: Depth image with shape (H, W)
                       Values represent distance in millimeters
        
        Note:
            For cameras without depth sensors, this should raise NotImplementedError
        """
        pass
    
    @abstractmethod
    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters.

        Returns:
            CameraIntrinsics: Camera intrinsic parameters
        """
        pass

    @abstractmethod
    def start(self):
        """
        Start the camera stream.
        """
        pass
    
    @abstractmethod
    def stop(self):
        """
        Stop the camera stream and release resources.
        """
        pass
    
    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get aligned color and depth frames.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (color_image, depth_image)

        Note:
            Default implementation calls capture_frame() and get_depth() separately.
            Override this for hardware-accelerated alignment.
        """
        color = self.capture_frame()
        depth = self.get_depth()
        return color, depth

    def get_frame(self) -> CameraFrame:
        """
        Get a complete camera frame with all available data.

        Returns:
            CameraFrame: Frame containing color, depth, and metadata
        """
        color = self.capture_frame()
        try:
            depth = self.get_depth()
        except NotImplementedError:
            depth = None
        return CameraFrame(color=color, depth=depth)

    def is_depth_available(self) -> bool:
        """Check if camera supports depth sensing."""
        try:
            _ = self.get_depth()
            return True
        except NotImplementedError:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()