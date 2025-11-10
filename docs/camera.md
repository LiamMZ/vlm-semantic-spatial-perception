# Camera System Documentation

## Overview

The camera system provides a unified abstraction for RGB-D sensing with support for Intel RealSense depth cameras and standard webcams. It integrates seamlessly with the VLM perception pipeline, providing RGB images, depth data, and camera intrinsics for 3D spatial reasoning.

**Location**: [src/camera/](../src/camera/)

---

## Key Features

- **Clean Abstraction**: Unified interface (`BaseCamera`) across camera types
- **RGB-D Support**: Full depth sensing with RealSense cameras
- **Hardware Alignment**: Automatic depth-to-color frame alignment
- **Calibrated Intrinsics**: Factory-calibrated parameters from RealSense
- **Graceful Degradation**: Webcam works without depth capability
- **Context Manager Support**: RAII pattern for resource management
- **Coordinate Utilities**: Complete 2D↔3D transformation pipeline
- **Flexible Creation**: Factory functions support config files or programmatic setup

---

## Module Structure

### Core Files

- **[base_camera.py](../src/camera/base_camera.py)** (150 lines)
  - Abstract base class defining camera interface
  - `CameraIntrinsics` and `CameraFrame` data structures
  - Common functionality for all camera types

- **[realsense_camera.py](../src/camera/realsense_camera.py)** (238 lines)
  - Intel RealSense camera implementation
  - RGB-D support with hardware alignment
  - Direct calibrated intrinsics from device
  - Robust error handling with timeouts

- **[webcam_camera.py](../src/camera/webcam_camera.py)** (149 lines)
  - Standard webcam using OpenCV
  - Automatic BGR→RGB conversion
  - Estimated intrinsics (no depth support)

- **[camera_utils.py](../src/camera/camera_utils.py)** (219 lines)
  - Factory functions for camera creation
  - 3D conversion utilities (pixel-to-3D, point clouds)
  - Depth processing and visualization

### Configuration

- **[config/camera_config.yaml](../config/camera_config.yaml)** (49 lines)
  - Camera settings for RealSense and webcam
  - Resolution, framerate, depth parameters

---

## Architecture

### Base Abstraction

```python
class BaseCamera(ABC):
    # Required abstract methods
    @abstractmethod
    def capture_frame() -> np.ndarray  # RGB (H, W, 3)

    @abstractmethod
    def get_depth() -> np.ndarray      # Depth in meters (H, W)

    @abstractmethod
    def get_camera_intrinsics() -> CameraIntrinsics

    @abstractmethod
    def start()  # Initialize stream

    @abstractmethod
    def stop()   # Release resources

    # Optional methods with defaults
    def get_aligned_frames() -> Tuple[color, depth]
    def get_frame() -> CameraFrame
    def is_depth_available() -> bool

    # Context manager
    def __enter__() / __exit__()
```

### Data Structures

#### CameraIntrinsics

```python
@dataclass
class CameraIntrinsics:
    fx: float                         # Focal length X
    fy: float                         # Focal length Y
    cx: float                         # Principal point X
    cy: float                         # Principal point Y
    width: int                        # Image width
    height: int                       # Image height
    distortion: Optional[List[float]] # Distortion coefficients

    def to_dict() -> Dict
    def to_matrix() -> np.ndarray  # 3x3 camera matrix K
```

**Camera Matrix (K)**:
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

#### CameraFrame

```python
@dataclass
class CameraFrame:
    color: np.ndarray                 # RGB (H, W, 3)
    depth: Optional[np.ndarray]       # Meters (H, W)
    timestamp: Optional[float]        # Unix timestamp
    frame_id: Optional[int]           # Sequence number
```

---

## Camera Implementations

### RealSense Camera

**Location**: [src/camera/realsense_camera.py](../src/camera/realsense_camera.py)

Full RGB-D support with factory-calibrated intrinsics.

#### Constructor

```python
RealSenseCamera(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    enable_depth: bool = True,
    auto_start: bool = True
)
```

#### Features

- **Hardware Alignment**: Uses `rs.align()` for depth-to-color alignment
- **Depth Scaling**: Automatically converts raw depth to meters
- **Real Intrinsics**: Factory-calibrated parameters from device
- **Robust Capture**: 5-second timeout with error handling
- **FPS Tracking**: Actual frame rate calculation
- **Distortion Model**: Includes distortion coefficients

#### Key Methods

```python
# Most efficient: hardware-aligned frames
def get_aligned_frames() -> Tuple[np.ndarray, np.ndarray]
    # Returns: (color_rgb, depth_meters)

# Standard interface
def capture_frame() -> np.ndarray  # RGB (H, W, 3)
def get_depth() -> np.ndarray      # Meters (H, W)
def get_camera_intrinsics() -> CameraIntrinsics

# Frame rate
def get_fps() -> float
```

#### Usage Example

```python
from src.camera import RealSenseCamera

# Context manager (recommended)
with RealSenseCamera(width=640, height=480) as camera:
    color, depth = camera.get_aligned_frames()
    intrinsics = camera.get_camera_intrinsics()

    print(f"Color: {color.shape}")
    print(f"Depth: {depth.shape}, range: {depth.min():.2f}-{depth.max():.2f}m")
    print(f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

# Manual lifecycle
camera = RealSenseCamera(width=1280, height=720, auto_start=True)

# Wait for stable frames (recommended for first use)
for _ in range(30):
    camera.capture_frame()

# Capture
color_image = camera.capture_frame()
depth_image = camera.get_depth()

camera.stop()
```

#### Implementation Details

- **Pipeline API**: Uses `pyrealsense2` pipeline
- **Stream Configuration**: RGB8 and Z16 formats
- **Depth Scale**: `depth_meters = raw_depth * depth_scale`
- **Alignment**: Hardware-accelerated via `rs.align(rs.stream.color)`

### Webcam Camera

**Location**: [src/camera/webcam_camera.py](../src/camera/webcam_camera.py)

Standard RGB camera using OpenCV (no depth).

#### Constructor

```python
WebcamCamera(
    device_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    auto_start: bool = True
)
```

#### Features

- **OpenCV Backend**: Uses `cv2.VideoCapture`
- **BGR→RGB Conversion**: Automatic color space conversion
- **Estimated Intrinsics**: Based on 60° horizontal FOV assumption
- **No Depth**: `get_depth()` raises `NotImplementedError`

#### Limitations

- No depth sensing (standard webcams lack depth sensors)
- Intrinsics are estimated, not calibrated
- Actual resolution may differ from requested

#### Usage Example

```python
from src.camera import WebcamCamera

# Initialize
camera = WebcamCamera(device_id=0, width=640, height=480)
camera.start()

# Capture RGB only
color_image = camera.capture_frame()

# Depth not available
try:
    depth = camera.get_depth()
except NotImplementedError:
    print("Webcam does not support depth")

# Intrinsics are estimated
intrinsics = camera.get_camera_intrinsics()
print(f"Estimated fx={intrinsics.fx:.1f}")

camera.stop()
```

#### Intrinsics Estimation

Assumes 60° horizontal field of view:

```python
focal_length = width / (2.0 * np.tan(np.radians(60.0) / 2.0))
intrinsics = CameraIntrinsics(
    fx=focal_length,
    fy=focal_length,  # Square pixels
    cx=width / 2.0,   # Center
    cy=height / 2.0,
    width=width,
    height=height
)
```

---

## Coordinate Systems

### Pinhole Camera Model

The system uses the standard pinhole camera model for 3D back-projection:

```
x = (pixel_x - cx) * depth / fx
y = (pixel_y - cy) * depth / fy
z = depth
```

Where:
- `(cx, cy)` = principal point (optical center)
- `(fx, fy)` = focal lengths
- `depth` = depth value at pixel location (meters)

### Camera Frame Convention

**Coordinate System**:
- **X-axis**: Points to the right
- **Y-axis**: Points down
- **Z-axis**: Points forward (depth direction)
- **Origin**: Camera optical center

**Units**: Meters (not millimeters)

---

## Camera Utilities

**Location**: [src/camera/camera_utils.py](../src/camera/camera_utils.py)

### Factory Functions

#### Config-Based Creation

```python
from src.camera import create_camera_from_config

# Reads config/camera_config.yaml
camera = create_camera_from_config('config/camera_config.yaml')
# Automatically creates RealSense or Webcam based on config
```

#### Programmatic Creation

```python
from src.camera import create_camera

# Webcam
config = {
    'webcam': {
        'device_id': 0,
        'resolution': {'width': 640, 'height': 480},
        'framerate': 30
    }
}
camera = create_camera('webcam', config)

# RealSense
config = {
    'realsense': {
        'resolution': {'width': 640, 'height': 480},
        'framerate': 30,
        'depth_enabled': True
    }
}
camera = create_camera('realsense', config)
```

### 3D Conversion Utilities

#### Pixel to 3D

```python
from src.camera import pixel_to_3d

point_3d = pixel_to_3d(
    u=320,           # pixel x
    v=240,           # pixel y
    depth=0.5,       # meters
    intrinsics=intrinsics
)
# Returns: [x, y, z] in meters
```

#### Depth Image to Point Cloud

```python
from src.camera import depth_image_to_point_cloud

# Generate point cloud
points = depth_image_to_point_cloud(
    depth_image=depth,
    intrinsics=intrinsics,
    color_image=None,   # Optional: adds RGB
    max_depth=10.0      # Filter distant points
)
# Returns: (N, 3) array of [x, y, z] points
# or (N, 6) with color: [x, y, z, r, g, b]

# With color
colored_points = depth_image_to_point_cloud(
    depth_image=depth,
    intrinsics=intrinsics,
    color_image=color,
    max_depth=5.0
)
# Returns: (N, 6) array
```

#### Object Depth Estimation

```python
from src.camera import estimate_object_depth

# Estimate depth for bounding box region
depth = estimate_object_depth(
    bbox=(x_min, y_min, x_max, y_max),
    depth_image=depth_frame,
    method='median'  # 'median', 'mean', or 'min'
)
# Returns: estimated depth in meters
```

#### Depth Visualization

```python
from src.camera import visualize_depth

# Convert depth to colored image
colored_depth = visualize_depth(
    depth_image=depth,
    max_depth=5.0  # Colormap range
)
# Returns: RGB image with JET colormap (H, W, 3)
```

---

## Perception Integration

### Coordinate Transformation Pipeline

**VLM → Pixel → 3D Workflow**:

```python
from src.perception.utils.coordinates import (
    normalized_to_pixel,
    compute_3d_position
)
from src.camera import RealSenseCamera

# 1. VLM provides normalized coordinates (0-1000)
vlm_position = [500, 750]  # [y, x]

# 2. Get camera frames
camera = RealSenseCamera()
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

# 3. Convert to pixel coordinates
pixel_y, pixel_x = normalized_to_pixel(vlm_position, color.shape[:2])

# 4. Back-project to 3D
position_3d = compute_3d_position(
    position_2d=vlm_position,
    depth_frame=depth,
    camera_intrinsics=intrinsics
)
# Returns: [x, y, z] in meters (camera frame)
```

### Object Tracker Integration

```python
from src.perception import ObjectTracker
from src.camera import RealSenseCamera

# Initialize
tracker = ObjectTracker(api_key=api_key)
camera = RealSenseCamera(enable_depth=True, auto_start=True)

# Wait for stable frames
for _ in range(30):
    camera.capture_frame()

# Detect objects with full 3D information
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

objects = await tracker.detect_objects(
    color,
    depth,
    intrinsics
)

# Objects have both 2D and 3D positions
for obj in objects:
    print(f"{obj.object_id}:")
    print(f"  2D: {obj.position_2d}")  # [y, x] in 0-1000
    print(f"  3D: {obj.position_3d}")  # [x, y, z] in meters
```

---

## Configuration

### Camera Config File

**Location**: [config/camera_config.yaml](../config/camera_config.yaml)

#### Default Camera

```yaml
default_camera: 'webcam'  # or 'realsense'
```

#### RealSense Settings

```yaml
realsense:
  resolution:
    width: 640
    height: 480
  framerate: 30
  depth_enabled: true
  color_scheme: 'RGB8'

  depth:
    min_distance: 0.1   # meters
    max_distance: 10.0  # meters

  # Post-processing filters (future feature)
  filters:
    spatial_filter: true
    temporal_filter: true
    hole_filling: true
```

#### Webcam Settings

```yaml
webcam:
  device_id: 0
  resolution:
    width: 640
    height: 480
  framerate: 30

  depth_estimation:
    enabled: false
    method: 'monocular'  # or 'none'
```

#### Capture Settings

```yaml
capture:
  buffer_size: 5
  timeout_ms: 5000
  retry_attempts: 3
```

#### Calibration

```yaml
calibration:
  auto_calibrate: true
  calibration_file: null  # Path to custom calibration YAML
```

---

## Usage Examples

### Basic RealSense Capture

```python
from src.camera import RealSenseCamera

# Context manager (automatic cleanup)
with RealSenseCamera(width=640, height=480) as camera:
    # Get aligned frames
    color, depth = camera.get_aligned_frames()
    intrinsics = camera.get_camera_intrinsics()

    print(f"Color: {color.shape}")
    print(f"Depth: {depth.shape}, range: {depth.min():.2f}-{depth.max():.2f}m")
    print(f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
```

### Point Cloud Generation

```python
from src.camera import RealSenseCamera, depth_image_to_point_cloud

camera = RealSenseCamera(enable_depth=True, auto_start=True)

# Capture
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

# Generate colored point cloud
points = depth_image_to_point_cloud(
    depth_image=depth,
    intrinsics=intrinsics,
    color_image=color,
    max_depth=5.0  # 5 meters max
)

print(f"Point cloud shape: {points.shape}")  # (N, 6)
print(f"Number of points: {len(points)}")

# Save or visualize
# np.save('pointcloud.npy', points)

camera.stop()
```

### Webcam with Perception

```python
from src.camera import WebcamCamera
from src.perception import ObjectTracker

# Webcam (RGB only)
camera = WebcamCamera(device_id=0, auto_start=True)
tracker = ObjectTracker(api_key=api_key, fast_mode=True)

# Capture
color = camera.capture_frame()

# Detect without depth (2D only)
objects = await tracker.detect_objects(
    color_frame=color,
    depth_frame=None,        # No depth
    camera_intrinsics=None
)

# Objects have 2D positions only
for obj in objects:
    print(f"{obj.object_id}: position_2d={obj.position_2d}")
    print(f"  position_3d={obj.position_3d}")  # Will be None

camera.stop()
```

### Multiple Cameras

```python
from src.camera import RealSenseCamera, WebcamCamera

# Both cameras simultaneously
realsense = RealSenseCamera(width=640, height=480, auto_start=True)
webcam = WebcamCamera(device_id=0, auto_start=True)

# Capture from both
rs_color, rs_depth = realsense.get_aligned_frames()
webcam_color = webcam.capture_frame()

# Use different cameras for different purposes
# e.g., RealSense for manipulation, webcam for wide-angle monitoring

realsense.stop()
webcam.stop()
```

### Config-Based Setup

```python
from src.camera import create_camera_from_config

# Load from config file
camera = create_camera_from_config('config/camera_config.yaml')

# Use normally
camera.start()
color, depth = camera.get_aligned_frames()
intrinsics = camera.get_camera_intrinsics()

camera.stop()
```

---

## Troubleshooting

### RealSense Not Detected

**Symptoms**: `RuntimeError: No RealSense devices found`

**Solutions**:
```bash
# Check device connection
rs-enumerate-devices

# Check USB permissions (Linux)
sudo usermod -a -G plugdev $USER
# Logout and login

# Install RealSense SDK
pip install pyrealsense2

# Try different USB port (USB 3.0 recommended)
```

### Depth Alignment Issues

**Symptoms**: Color and depth frames misaligned

**Solutions**:
```python
# RealSense handles alignment automatically
camera = RealSenseCamera(enable_depth=True)
color, depth = camera.get_aligned_frames()  # Already aligned

# Verify alignment
print(f"Color shape: {color.shape[:2]}")
print(f"Depth shape: {depth.shape}")
# Should match: (H, W)
```

### Invalid Depth Values

**Symptoms**: `depth == 0` or `depth == nan` in many pixels

**Causes**: Reflective surfaces, transparent objects, depth range limits

**Solutions**:
```python
# Filter invalid depth
valid_mask = (depth > 0) & (depth < 10.0)  # 0-10m range
depth[~valid_mask] = np.nan

# Use median for robust estimation
from src.camera import estimate_object_depth
depth_value = estimate_object_depth(bbox, depth, method='median')
```

### Webcam Wrong Device

**Symptoms**: Opens wrong camera or fails

**Solutions**:
```python
# Try different device IDs
for device_id in range(5):
    try:
        camera = WebcamCamera(device_id=device_id)
        print(f"Found camera at device {device_id}")
        break
    except:
        continue

# List available devices (Linux)
# ls /dev/video*

# List available devices (macOS)
# system_profiler SPCameraDataType
```

### Low Frame Rate

**Symptoms**: FPS much lower than requested

**Solutions**:
```python
# Lower resolution
camera = RealSenseCamera(width=640, height=480)  # Instead of 1920x1080

# Disable depth if not needed
camera = RealSenseCamera(enable_depth=False)

# Check actual FPS
fps = camera.get_fps()
print(f"Actual FPS: {fps:.1f}")

# Reduce processing load
# - Use fast_mode in ObjectTracker
# - Reduce detection frequency
```

---

## Design Decisions

### Architecture Choices

1. **Depth in Meters**: Consistent with robotics conventions (not mm)
2. **RGB Format**: Matches PIL/numpy (not BGR like OpenCV)
3. **Optional Depth**: Supports RGB-only workflows gracefully
4. **Context Manager**: RAII pattern prevents resource leaks
5. **Factory Functions**: Flexible creation from config or code
6. **Type Safety**: Dataclasses for structured data

### Performance Considerations

- **Hardware Alignment**: RealSense uses efficient GPU-accelerated alignment
- **Depth Scaling**: Cached depth scale for efficiency
- **Frame Timing**: FPS calculation based on actual capture times
- **Lazy Evaluation**: Intrinsics computed once and cached

---

## Future Enhancements

### Planned Features

1. **Custom Calibration**: Load calibration from YAML files
2. **Depth Filters**: Spatial, temporal, hole-filling filters
3. **Multi-Camera Sync**: Synchronized capture from multiple cameras
4. **Camera Calibration Tool**: Interactive calibration utility
5. **Stereo Depth**: Stereo camera support for depth estimation
6. **Recording/Playback**: Save and replay camera streams

---

## Related Documentation

- [Perception System](perception.md)
- [Planning System](planning.md)
- [Coordinate Utilities](../src/perception/utils/coordinates.py)
- [Camera Module README](../src/camera/README.md)

---

## External Resources

- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [PyRealSense2 Documentation](https://intelrealsense.github.io/librealsense/python_docs/)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

---

*Last Updated: November 10, 2025*
