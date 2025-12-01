# Camera System

The camera layer provides RGB(-D) inputs, intrinsics, and 2D↔3D utilities for perception and execution. It supports Intel RealSense (RGB-D) and OpenCV webcams (RGB only) behind a shared interface.

**Code**: `src/camera/`

## Components
- `base_camera.py` – `BaseCamera`, `CameraIntrinsics`, and `CameraFrame` contracts.
- `realsense_camera.py` – hardware-aligned RGB-D capture with intrinsics + FPS reporting.
- `webcam_camera.py` – RGB capture with estimated intrinsics (no depth).
- `camera_utils.py` – factories (`create_camera`, `create_camera_from_config`) and helpers (`depth_image_to_point_cloud`, `estimate_object_depth`, `pixel_to_3d`, `visualize_depth`).

## Usage
```python
from pathlib import Path
from src.camera import RealSenseCamera, WebcamCamera

# RealSense (recommended for depth-aware perception)
with RealSenseCamera(width=640, height=480, enable_depth=True) as cam:
    color, depth = cam.get_aligned_frames()
    intrinsics = cam.get_camera_intrinsics()
    print(color.shape, depth.shape, intrinsics.to_dict())

# Webcam (RGB only)
cam = WebcamCamera(device_id=0, width=640, height=480, auto_start=True)
frame = cam.capture_frame()
intrinsics = cam.get_camera_intrinsics()  # estimated focal lengths
cam.stop()
```

### From Config
```python
from src.camera import create_camera_from_config
camera = create_camera_from_config("config/camera_config.yaml")
camera.start()
color, depth = camera.get_aligned_frames()
camera.stop()
```

## Notes for Perception/Execution
- `get_aligned_frames()` returns `(color, depth)` where `depth` is meters when available. Pass both plus `CameraIntrinsics` to `ObjectTracker.detect_objects(...)`.
- Normalized `[y, x]` → pixel → 3D conversions live in `src/perception/utils/coordinates.py`; the primitives executor reuses the same conventions when back-projecting helper pixels.
- If RealSense is unavailable, the orchestrator falls back to the provided camera instance (or raises); depth-dependent features will be skipped when `depth` is `None`.
