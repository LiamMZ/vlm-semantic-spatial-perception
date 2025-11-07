"""
Webcam camera implementation.
"""

import time
from typing import Optional

import cv2
import numpy as np

from .base_camera import BaseCamera, CameraIntrinsics


class WebcamCamera(BaseCamera):
    """
    Webcam camera implementation using OpenCV.

    Note: This implementation does not provide depth information unless
    depth estimation is explicitly configured.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_start: bool = True
    ):
        """
        Initialize webcam camera.

        Args:
            device_id: Camera device ID (default 0)
            width: Frame width
            height: Frame height
            fps: Target frame rate
            auto_start: Automatically start camera on initialization
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self._start_time = None

        if auto_start:
            self.start()

    def start(self):
        """Start the camera stream."""
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {self.device_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual properties (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._start_time = time.time()
        print(f"Webcam started: {self.width}x{self.height} @ {self.fps} FPS")

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single RGB frame from the webcam.

        Returns:
            np.ndarray: RGB image with shape (H, W, 3)

        Raises:
            RuntimeError: If camera is not started or frame capture fails
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera not started")

        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        return frame_rgb

    def get_depth(self) -> np.ndarray:
        """
        Get depth frame. Not supported for standard webcams.

        Raises:
            NotImplementedError: Webcams do not provide depth information
        """
        raise NotImplementedError(
            "Depth sensing not available on standard webcam. "
            "Use a depth camera (e.g., RealSense) or enable depth estimation."
        )

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters.

        Returns:
            CameraIntrinsics: Estimated camera intrinsics

        Note:
            For uncalibrated webcams, this returns estimated intrinsics
            based on typical webcam FOV (~60 degrees).
        """
        # Estimate focal length assuming ~60 degree horizontal FOV
        focal_length = self.width / (2.0 * np.tan(np.radians(60.0) / 2.0))

        return CameraIntrinsics(
            fx=focal_length,
            fy=focal_length,
            cx=self.width / 2.0,
            cy=self.height / 2.0,
            width=self.width,
            height=self.height
        )

    def stop(self):
        """Stop the camera stream and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Webcam stopped")

    def __del__(self):
        """Destructor to ensure camera is released."""
        self.stop()

    def get_fps(self) -> float:
        """Calculate actual FPS based on frame count and elapsed time."""
        if self._start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
