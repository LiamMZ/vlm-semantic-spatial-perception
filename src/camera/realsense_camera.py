"""
Intel RealSense camera implementation.
"""

import logging
import time
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from .base_camera import BaseCamera, CameraIntrinsics
from ..utils.logging_utils import get_structured_logger


class RealSenseCamera(BaseCamera):
    """
    Intel RealSense camera implementation with RGB-D support.

    Supports depth-aligned color frames and provides accurate camera intrinsics.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        auto_start: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RealSense camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frame rate
            enable_depth: Enable depth stream
            auto_start: Automatically start camera on initialization
            logger: Logger instance (if None, creates a structured logger)

        Raises:
            ImportError: If pyrealsense2 is not installed
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed. Install with: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.logger = logger if logger is not None else get_structured_logger("RealSenseCamera")

        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.frame_count = 0
        self._start_time = None

        if auto_start:
            self.start()

    def start(self):
        """Start the RealSense camera stream and wait for first valid frame."""
        if self.pipeline is not None:
            self.logger.debug("Camera already started, skipping start()")
            return

        self.logger.info(f"Starting RealSense camera: {self.width}x{self.height} @ {self.fps} FPS")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.logger.debug(f"Configuring color stream: RGB8 @ {self.width}x{self.height}")
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )

        if self.enable_depth:
            self.logger.debug(f"Configuring depth stream: Z16 @ {self.width}x{self.height}")
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            # Create alignment object to align depth to color
            self.align = rs.align(rs.stream.color)

        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
            self._start_time = time.time()
            self.logger.info("RealSense pipeline started successfully")
        except RuntimeError as e:
            self.logger.error(f"Failed to start RealSense camera: {e}")
            raise RuntimeError(f"Failed to start RealSense camera: {e}")
        
        # Wait for camera to stabilize and ensure we can capture frames
        max_retries = 30
        retry_delay = 0.1  # 100ms between retries
        
        self.logger.debug(f"Waiting for first valid frame (max {max_retries} attempts)...")
        
        for attempt in range(max_retries):
            try:
                # Try to capture a test frame
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                if self.align and self.enable_depth:
                    frames = self.align.process(frames)
                
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    # Successfully got a frame
                    if attempt == 0:
                        self.logger.info("Camera initialized successfully on first attempt")
                    else:
                        self.logger.info(f"Camera initialized successfully after {attempt + 1} attempts")
                    return
                else:
                    self.logger.debug(f"Attempt {attempt + 1}: No color frame available")
                    
            except RuntimeError as e:
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            time.sleep(retry_delay)
        
        # If we get here, we failed to capture a frame
        self.logger.error(f"Failed to capture initial frame after {max_retries} attempts")
        self.stop()
        raise RuntimeError(f"Failed to capture initial frame after {max_retries} attempts")

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single RGB frame from the camera.

        Returns:
            np.ndarray: RGB image with shape (H, W, 3)

        Raises:
            RuntimeError: If camera is not started or frame capture fails
        """
        if self.pipeline is None:
            self.logger.error("Attempted to capture frame but camera not started")
            raise RuntimeError("Camera not started")

        self.logger.debug("Capturing RGB frame...")
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            if self.align and self.enable_depth:
                # Align depth frame to color frame
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()

            if not color_frame:
                self.logger.error("Failed to get color frame from pipeline")
                raise RuntimeError("Failed to capture color frame")

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            self.frame_count += 1
            
            self.logger.debug(f"Successfully captured frame #{self.frame_count}")
            return color_image
            
        except RuntimeError as e:
            self.logger.error(f"Error capturing frame: {e}")
            raise

    def get_depth(self) -> np.ndarray:
        """
        Get depth frame from the camera.

        Returns:
            np.ndarray: Depth image with shape (H, W) in meters

        Raises:
            RuntimeError: If depth is not enabled or frame capture fails
        """
        if not self.enable_depth:
            self.logger.error("Attempted to get depth but depth stream not enabled")
            raise NotImplementedError("Depth stream not enabled")

        if self.pipeline is None:
            self.logger.error("Attempted to get depth but camera not started")
            raise RuntimeError("Camera not started")

        self.logger.debug("Capturing depth frame...")
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            if self.align:
                frames = self.align.process(frames)

            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                self.logger.error("Failed to get depth frame from pipeline")
                raise RuntimeError("Failed to capture depth frame")

            # Convert to numpy array and scale to meters
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_image.astype(np.float32) * depth_scale
            
            self.logger.debug(f"Successfully captured depth frame (scale: {depth_scale})")
            return depth_meters
            
        except RuntimeError as e:
            self.logger.error(f"Error capturing depth frame: {e}")
            raise

    def get_aligned_frames(self):
        """
        Get hardware-aligned color and depth frames.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (color_image, depth_image in meters)
        """
        if self.pipeline is None:
            self.logger.error("Attempted to get aligned frames but camera not started")
            raise RuntimeError("Camera not started")

        if not self.enable_depth:
            self.logger.error("Attempted to get aligned frames but depth stream not enabled")
            raise RuntimeError("Depth stream not enabled")

        self.logger.debug("Capturing aligned color and depth frames...")
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            if self.align:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.logger.error("Failed to get aligned frames from pipeline")
                raise RuntimeError("Failed to capture aligned frames")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Scale depth to meters
            depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_image.astype(np.float32) * depth_scale

            self.frame_count += 1
            
            self.logger.debug(f"Successfully captured aligned frames #{self.frame_count}")
            return color_image, depth_meters
            
        except RuntimeError as e:
            self.logger.error(f"Error capturing aligned frames: {e}")
            raise

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters from RealSense device.

        Returns:
            CameraIntrinsics: Camera intrinsic parameters
        """
        if self.profile is None:
            self.logger.error("Attempted to get camera intrinsics but camera not started")
            raise RuntimeError("Camera not started")

        self.logger.debug("Retrieving camera intrinsics...")
        
        # Get color stream intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Get distortion coefficients
        distortion = np.array(intrinsics.coeffs) if intrinsics.coeffs else None

        cam_intrinsics = CameraIntrinsics(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height,
            distortion=distortion
        )
        
        self.logger.debug(f"Camera intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
                         f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
        
        return cam_intrinsics

    def stop(self):
        """Stop the camera stream and release resources."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.profile = None
            self.logger.info("RealSense stopped")

    def __del__(self):
        """Destructor to ensure camera is released."""
        if hasattr(self, 'logger') and hasattr(self, 'pipeline') and self.pipeline is not None:
            self.logger.debug("RealSenseCamera destructor called, stopping camera")
        self.stop()

    def get_fps(self) -> float:
        """Calculate actual FPS based on frame count and elapsed time."""
        if self._start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        self.logger.debug(f"Current FPS: {fps:.2f} ({self.frame_count} frames in {elapsed:.2f}s)")
        return fps
