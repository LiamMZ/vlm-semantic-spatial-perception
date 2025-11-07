"""
RealSense camera interface for capturing aligned RGB-D frames.
"""

import numpy as np
import pyrealsense2 as rs
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from .base_camera import BaseCamera


@dataclass
class RealSenseConfig:
    """Configuration for RealSense camera."""
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_width: int = 640
    depth_height: int = 480
    enable_color: bool = True
    enable_depth: bool = True
    enable_infrared: bool = False
    align_to_color: bool = True  # Align depth to color frame
    
    # Advanced settings
    visual_preset: str = "Default"  # Default, High Accuracy, High Density, Medium Density
    enable_emitter: bool = True
    laser_power: int = 150  # 0-360 for D435
    
    # Post-processing filters
    enable_decimation: bool = False
    enable_spatial: bool = True
    enable_temporal: bool = True
    enable_hole_filling: bool = True


class RealSenseCamera(BaseCamera):
    """
    RealSense camera wrapper for aligned RGB-D capture.
    
    Provides aligned color and depth frames, camera intrinsics,
    and point cloud generation capabilities.
    """
    
    def __init__(self, config: Optional[RealSenseConfig] = None, device_serial: Optional[str] = None):
        """
        Initialize RealSense camera.
        
        Args:
            config: Camera configuration
            device_serial: Specific device serial number (None for first available)
        """
        self.config = config or RealSenseConfig()
        self.device_serial = device_serial
        self.logger = logging.getLogger(__name__)
        
        # RealSense objects
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.align = None
        self.profile = None
        
        # Camera parameters
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # Post-processing filters
        self.decimation_filter = None
        self.spatial_filter = None
        self.temporal_filter = None
        self.hole_filling_filter = None
        
        # State
        self.is_streaming = False
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera with configured settings."""
        try:
            # Configure specific device if serial provided
            if self.device_serial:
                self.rs_config.enable_device(self.device_serial)
            
            # Enable streams
            if self.config.enable_color:
                self.rs_config.enable_stream(
                    rs.stream.color,
                    self.config.width,
                    self.config.height,
                    rs.format.bgr8,
                    self.config.fps
                )
            
            if self.config.enable_depth:
                self.rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.depth_width,
                    self.config.depth_height,
                    rs.format.z16,
                    self.config.fps
                )
            
            if self.config.enable_infrared:
                self.rs_config.enable_stream(
                    rs.stream.infrared,
                    1,  # Left infrared
                    self.config.width,
                    self.config.height,
                    rs.format.y8,
                    self.config.fps
                )
            
            # Start streaming
            self.profile = self.pipeline.start(self.rs_config)
            self.is_streaming = True
            
            # Get device and sensors
            device = self.profile.get_device()
            self.logger.info(f"Connected to RealSense device: {device.get_info(rs.camera_info.name)}")
            self.logger.info(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
            
            # Configure depth sensor settings
            if self.config.enable_depth:
                depth_sensor = device.first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.logger.info(f"Depth scale: {self.depth_scale}")
                
                # Set visual preset
                if hasattr(rs.rs400_visual_preset, self.config.visual_preset.lower().replace(" ", "_")):
                    preset = getattr(rs.rs400_visual_preset, self.config.visual_preset.lower().replace(" ", "_"))
                    depth_sensor.set_option(rs.option.visual_preset, preset)
                
                # Set laser power
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, self.config.laser_power)
                
                # Enable/disable emitter
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1 if self.config.enable_emitter else 0)
            
            # Create alignment object (align depth to color)
            if self.config.align_to_color and self.config.enable_depth and self.config.enable_color:
                self.align = rs.align(rs.stream.color)
            
            # Get intrinsics
            if self.config.enable_color:
                color_stream = self.profile.get_stream(rs.stream.color)
                self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            if self.config.enable_depth:
                depth_stream = self.profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # Initialize post-processing filters
            self._initialize_filters()
            
            # Allow auto-exposure to stabilize
            self.logger.info("Warming up camera (waiting for auto-exposure)...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            self.logger.info("RealSense camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RealSense camera: {e}")
            raise
    
    def _initialize_filters(self):
        """Initialize post-processing filters for depth data."""
        if not self.config.enable_depth:
            return
        
        if self.config.enable_decimation:
            self.decimation_filter = rs.decimation_filter()
            self.decimation_filter.set_option(rs.option.filter_magnitude, 2)
        
        if self.config.enable_spatial:
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        if self.config.enable_temporal:
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        if self.config.enable_hole_filling:
            self.hole_filling_filter = rs.hole_filling_filter()
            self.hole_filling_filter.set_option(rs.option.holes_fill, 1)  # Fill from farthest
    
    def _apply_filters(self, depth_frame):
        """Apply post-processing filters to depth frame."""
        filtered_frame = depth_frame
        
        if self.config.enable_decimation and self.decimation_filter:
            filtered_frame = self.decimation_filter.process(filtered_frame)
        
        if self.config.enable_spatial and self.spatial_filter:
            filtered_frame = self.spatial_filter.process(filtered_frame)
        
        if self.config.enable_temporal and self.temporal_filter:
            filtered_frame = self.temporal_filter.process(filtered_frame)
        
        if self.config.enable_hole_filling and self.hole_filling_filter:
            filtered_frame = self.hole_filling_filter.process(filtered_frame)
        
        return filtered_frame
    
    def capture_frame(self) -> np.ndarray:
        """
        Capture RGB frame from color camera.
        
        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        frames = self.pipeline.wait_for_frames()
        
        if self.align:
            frames = self.align.process(frames)
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame")
        
        # Convert to numpy array (BGR format from RealSense)
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image
    
    def get_depth(self) -> np.ndarray:
        """
        Capture depth frame.
        
        Returns:
            np.ndarray: Depth image in millimeters (H, W)
        """
        frames = self.pipeline.wait_for_frames()
        
        if self.align:
            frames = self.align.process(frames)
        
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("Failed to capture depth frame")
        
        # Apply filters
        depth_frame = self._apply_filters(depth_frame)
        
        # Convert to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return depth_image
    
    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture aligned color and depth frames.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (color_image, depth_image)
                - color_image: RGB image (H, W, 3)
                - depth_image: Depth in millimeters (H, W)
        """
        if not self.is_streaming:
            raise RuntimeError("Camera is not streaming")
        
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to color
        if self.align:
            frames = self.align.process(frames)
        
        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")
        
        # Apply filters to depth
        depth_frame = self._apply_filters(depth_frame)
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def get_pointcloud(self, color_image: Optional[np.ndarray] = None, 
                       depth_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate point cloud from depth image.
        
        Args:
            color_image: Optional RGB image for coloring points
            depth_image: Optional depth image (if None, captures new frame)
        
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 
                - points: Point cloud (N, 3) in meters
                - colors: RGB colors (N, 3) if color_image provided
        """
        if depth_image is None:
            _, depth_image = self.get_aligned_frames()
        
        # Create point cloud
        pc = rs.pointcloud()
        
        # Convert numpy depth to rs.frame if needed
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        depth_frame = self._apply_filters(depth_frame)
        
        # Calculate point cloud
        points = pc.calculate(depth_frame)
        
        # Get vertices
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Get colors if provided
        colors = None
        if color_image is not None:
            colors = color_image.reshape(-1, 3)
            # Remove invalid points (where depth is 0)
            valid_mask = vertices[:, 2] != 0
            vertices = vertices[valid_mask]
            colors = colors[valid_mask]
        
        return vertices, colors
    
    def get_camera_intrinsics(self) -> Dict:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Dict: Camera intrinsics with keys:
                - fx, fy: Focal lengths
                - cx, cy: Principal point
                - width, height: Image dimensions
                - distortion: Distortion coefficients
        """
        if self.color_intrinsics is None:
            raise RuntimeError("Camera not initialized")
        
        intrinsics = {
            'fx': self.color_intrinsics.fx,
            'fy': self.color_intrinsics.fy,
            'cx': self.color_intrinsics.ppx,
            'cy': self.color_intrinsics.ppy,
            'width': self.color_intrinsics.width,
            'height': self.color_intrinsics.height,
            'distortion': np.array(self.color_intrinsics.coeffs),
            'model': self.color_intrinsics.model
        }
        
        return intrinsics
    
    def get_depth_intrinsics(self) -> Dict:
        """Get depth camera intrinsic parameters."""
        if self.depth_intrinsics is None:
            raise RuntimeError("Depth camera not initialized")
        
        intrinsics = {
            'fx': self.depth_intrinsics.fx,
            'fy': self.depth_intrinsics.fy,
            'cx': self.depth_intrinsics.ppx,
            'cy': self.depth_intrinsics.ppy,
            'width': self.depth_intrinsics.width,
            'height': self.depth_intrinsics.height,
            'distortion': np.array(self.depth_intrinsics.coeffs),
            'model': self.depth_intrinsics.model
        }
        
        return intrinsics
    
    def get_depth_scale(self) -> float:
        """
        Get depth scale factor (converts depth units to meters).
        
        Returns:
            float: Depth scale (typically 0.001 for mm to m conversion)
        """
        return self.depth_scale
    
    def deproject_pixel_to_point(self, pixel: Tuple[int, int], depth: float) -> np.ndarray:
        """
        Convert 2D pixel + depth to 3D point in camera coordinates.
        
        Args:
            pixel: (x, y) pixel coordinates
            depth: Depth value at pixel in millimeters
        
        Returns:
            np.ndarray: 3D point [x, y, z] in meters
        """
        if self.depth_intrinsics is None:
            raise RuntimeError("Camera not initialized")
        
        point = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics,
            [pixel[0], pixel[1]],
            depth * self.depth_scale * 1000  # Convert to depth units
        )
        
        return np.array(point)
    
    def project_point_to_pixel(self, point: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D pixel coordinates.
        
        Args:
            point: 3D point [x, y, z] in meters
        
        Returns:
            Tuple[int, int]: (x, y) pixel coordinates
        """
        if self.color_intrinsics is None:
            raise RuntimeError("Camera not initialized")
        
        pixel = rs.rs2_project_point_to_pixel(
            self.color_intrinsics,
            point.tolist()
        )
        
        return (int(pixel[0]), int(pixel[1]))
    
    def get_frame_metadata(self) -> Dict:
        """Get metadata from the latest frame."""
        frames = self.pipeline.wait_for_frames()
        
        if self.align:
            frames = self.align.process(frames)
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        metadata = {
            'timestamp': frames.get_timestamp(),
            'frame_number': frames.get_frame_number(),
        }
        
        if color_frame:
            metadata['color_timestamp'] = color_frame.get_timestamp()
            metadata['color_frame_number'] = color_frame.get_frame_number()
        
        if depth_frame:
            metadata['depth_timestamp'] = depth_frame.get_timestamp()
            metadata['depth_frame_number'] = depth_frame.get_frame_number()
        
        return metadata
    
    def stop(self):
        """Stop the camera stream and release resources."""
        if self.is_streaming:
            self.pipeline.stop()
            self.is_streaming = False
            self.logger.info("RealSense camera stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.stop()


def list_realsense_devices() -> list:
    """
    List all connected RealSense devices.
    
    Returns:
        list: List of dictionaries with device information
    """
    ctx = rs.context()
    devices = []
    
    for device in ctx.query_devices():
        device_info = {
            'name': device.get_info(rs.camera_info.name),
            'serial': device.get_info(rs.camera_info.serial_number),
            'firmware': device.get_info(rs.camera_info.firmware_version),
            'usb_type': device.get_info(rs.camera_info.usb_type_descriptor),
        }
        devices.append(device_info)
    
    return devices


if __name__ == "__main__":
    # Example usage
    import cv2
    
    logging.basicConfig(level=logging.INFO)
    
    print("Available RealSense devices:")
    for i, device in enumerate(list_realsense_devices()):
        print(f"{i}: {device['name']} (Serial: {device['serial']})")
    
    # Create camera with custom config
    config = RealSenseConfig(
        width=1280,
        height=720,
        fps=30,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=True
    )
    
    try:
        with RealSenseCamera(config=config) as camera:
            print("\nCamera Info:")
            print(f"Color intrinsics: {camera.get_camera_intrinsics()}")
            print(f"Depth scale: {camera.get_depth_scale()}")
            
            print("\nPress 'q' to quit, 's' to save frame, 'p' to save point cloud")
            
            while True:
                # Get aligned frames
                color, depth = camera.get_aligned_frames()
                
                # Visualize
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Stack images side by side
                images = np.hstack((color, depth_colormap))
                cv2.imshow('RealSense: Color | Depth', images)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('s'):
                    cv2.imwrite('color_frame.png', color)
                    cv2.imwrite('depth_frame.png', depth_colormap)
                    print("Frames saved!")
                elif key & 0xFF == ord('p'):
                    points, colors = camera.get_pointcloud(color, depth)
                    np.save('pointcloud.npy', points)
                    if colors is not None:
                        np.save('pointcloud_colors.npy', colors)
                    print(f"Point cloud saved! ({len(points)} points)")
            
            cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()