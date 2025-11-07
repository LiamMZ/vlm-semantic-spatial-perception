"""
VLM-based object detector that integrates with the world model.

Converts Gemini Robotics detection results into DetectedObject instances
compatible with the WorldState system. All semantic information including
affordances is dynamically generated from VLM observations.
"""

import time
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image

from .gemini_robotics import GeminiRoboticsClient, ObjectDetectionResult
from ..world_model import DetectedObject
from ..camera import CameraIntrinsics


class VLMObjectDetector:
    """
    Vision-Language Model based object detector.

    Bridges Gemini Robotics-ER detection with the world model by:
    1. Detecting objects with 2D positions from VLM
    2. Converting to 3D coordinates using depth data
    3. Creating DetectedObject instances with VLM-inferred affordances
    4. Enriching with semantic properties from visual observations

    All task-related data (affordances, properties, relationships) is
    dynamically generated based on environmental observations and user requests.
    """

    def __init__(
        self,
        gemini_client: GeminiRoboticsClient,
        confidence_threshold: float = 0.5,
        use_thinking: bool = True,
        max_objects: int = 50
    ):
        """
        Initialize VLM object detector.

        Args:
            gemini_client: Configured GeminiRoboticsClient
            confidence_threshold: Minimum confidence for detections
            use_thinking: Enable extended thinking mode
            max_objects: Maximum objects to detect per frame
        """
        self.client = gemini_client
        self.confidence_threshold = confidence_threshold
        self.use_thinking = use_thinking
        self.max_objects = max_objects

        # Detection statistics
        self.total_detections = 0
        self.total_frames = 0
        self.avg_processing_time = 0.0

    def detect(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        query: Optional[str] = None,
        task_context: Optional[str] = None
    ) -> List[DetectedObject]:
        """
        Detect objects in RGB-D frame and return DetectedObject instances.

        Args:
            color_image: RGB image (H, W, 3)
            depth_image: Depth map in meters (H, W) - optional
            camera_intrinsics: Camera parameters for 3D projection - optional
            query: Custom detection query - optional
            task_context: Task description to guide detection - optional

        Returns:
            List of DetectedObject instances ready for WorldState

        Example:
            >>> detector = VLMObjectDetector(client)
            >>> objects = detector.detect(
            ...     color_frame,
            ...     depth_frame,
            ...     camera_intrinsics,
            ...     task_context="Pick up the red cup"
            ... )
            >>> world.update(objects)
        """
        start_time = time.time()

        # Build detection query based on context
        detection_query = self._build_detection_query(query, task_context)

        # Run Gemini detection
        result = self.client.detect_objects(
            color_image,
            query=detection_query,
            return_json=True
        )

        # Convert to DetectedObject instances
        detected_objects = self._convert_to_detected_objects(
            result,
            color_image,
            depth_image,
            camera_intrinsics
        )

        # Filter by confidence
        detected_objects = [
            obj for obj in detected_objects
            if obj.confidence >= self.confidence_threshold
        ]

        # Limit max objects
        detected_objects = detected_objects[:self.max_objects]

        # Update statistics
        self.total_detections += len(detected_objects)
        self.total_frames += 1
        processing_time = time.time() - start_time
        self.avg_processing_time = (
            (self.avg_processing_time * (self.total_frames - 1) + processing_time)
            / self.total_frames
        )

        return detected_objects

    def detect_with_spatial_context(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        task: Optional[str] = None
    ) -> Tuple[List[DetectedObject], Dict[str, Any]]:
        """
        Detect objects and analyze spatial relationships in one call.

        Args:
            color_image: RGB image
            depth_image: Depth map - optional
            camera_intrinsics: Camera parameters - optional
            task: Task description for context

        Returns:
            Tuple of (detected_objects, spatial_context_dict)
        """
        # Detect objects
        objects = self.detect(
            color_image,
            depth_image,
            camera_intrinsics,
            task_context=task
        )

        # Analyze spatial relationships
        spatial_result = self.client.analyze_spatial_relationships(
            color_image,
            query=f"Analyze spatial layout for task: {task}" if task else "Analyze spatial layout",
            focus_objects=[obj.object_type for obj in objects]
        )

        spatial_context = {
            "reasoning": spatial_result.reasoning,
            "relationships": spatial_result.relationships,
            "recommendations": spatial_result.recommendations
        }

        return objects, spatial_context

    def _build_detection_query(
        self,
        custom_query: Optional[str],
        task_context: Optional[str]
    ) -> str:
        """
        Build optimized detection query based on context.

        Prompts the VLM to dynamically infer affordances and properties
        from visual observations rather than relying on hard-coded mappings.
        """
        if custom_query:
            return custom_query

        base_query = """Analyze the scene and detect all objects with their properties.

For each object, observe and infer:
1. Precise label (include visible attributes like color, material, size)
2. Position as [y, x] in 0-1000 normalized scale
3. Confidence score (0.0-1.0)
4. Physical properties from visual observation:
   - Color, material, size estimate
   - Current state (open/closed, empty/full, on/off, etc.)
5. Affordances based on visual analysis:
   - Can it be grasped? (size, shape, handles)
   - Can it contain things? (openings, cavities)
   - Can it support other objects? (flat surfaces)
   - Can it be opened? (doors, lids, drawers)
   - Can it be poured? (liquid containers)
   - Is it movable or static? (size, attachment)
   - Any other action possibilities you observe"""

        if task_context:
            base_query += f"""

TASK CONTEXT: {task_context}

Given this task, pay special attention to:
- Objects that are directly relevant to completing the task
- Affordances that would be useful for this specific task
- Spatial relationships that matter for task execution"""

        base_query += """

Return detailed observations as structured data. Infer affordances from what you see, not from pre-defined categories."""

        return base_query

    def _convert_to_detected_objects(
        self,
        detection_result: ObjectDetectionResult,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        camera_intrinsics: Optional[CameraIntrinsics]
    ) -> List[DetectedObject]:
        """Convert Gemini detection results to DetectedObject instances."""
        detected_objects = []
        timestamp = time.time()

        for i, obj_data in enumerate(detection_result.objects):
            try:
                # Skip if obj_data is not a dict
                if not isinstance(obj_data, dict):
                    print(f"   ⚠ Skipping invalid object data (expected dict, got {type(obj_data)}): {obj_data}")
                    continue

                # Extract basic info
                label = obj_data.get("label", f"unknown_{i}")
                position_2d = obj_data.get("position", [500, 500])  # [y, x] in 0-1000 scale
                confidence = obj_data.get("confidence", 0.5)
                properties = obj_data.get("properties", {})

                # Parse object type and attributes from VLM output
                object_type, color, material = self._parse_label(label)

                # Convert 2D position to 3D if depth available
                position_3d = self._compute_3d_position(
                    position_2d,
                    color_image.shape,
                    depth_image,
                    camera_intrinsics
                )

                # Extract affordances directly from VLM properties
                affordances = self._extract_affordances_from_properties(properties)

                # Extract size if provided by VLM
                size = self._extract_size(properties)

                # Create unique object ID
                object_id = self._generate_object_id(object_type, color, i)

                # Preserve all VLM-observed properties
                all_properties = {
                    "label": label,
                    "vlm_observed": True,
                    **properties
                }

                # Add parsed attributes if not already present
                if color and "color" not in all_properties:
                    all_properties["color"] = color
                if material and "material" not in all_properties:
                    all_properties["material"] = material

                # Create DetectedObject
                detected_obj = DetectedObject(
                    object_id=object_id,
                    object_type=object_type,
                    position=position_3d,
                    confidence=confidence,
                    timestamp=timestamp,
                    color=color,
                    affordances=affordances,
                    properties=all_properties,
                    size=size
                )

                detected_objects.append(detected_obj)

            except Exception as e:
                print(f"   ⚠ Failed to convert object {i}: {e}")
                continue

        return detected_objects

    def _parse_label(self, label: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Parse object label into (type, color, material).

        Example:
            "red plastic cup" -> ("cup", "red", "plastic")
        """
        label_lower = label.lower()
        words = label_lower.split()

        # Common colors (for parsing compound labels)
        colors = ["red", "blue", "green", "yellow", "black", "white", "orange",
                  "purple", "brown", "gray", "grey", "pink", "silver", "gold"]
        # Common materials
        materials = ["plastic", "metal", "wood", "glass", "ceramic", "paper",
                     "fabric", "leather", "rubber", "cardboard", "steel", "aluminum"]

        color = None
        material = None
        object_type_words = words.copy()

        # Extract color
        for c in colors:
            if c in words:
                color = c
                if c in object_type_words:
                    object_type_words.remove(c)

        # Extract material
        for m in materials:
            if m in words:
                material = m
                if m in object_type_words:
                    object_type_words.remove(m)

        # Remaining words form object type
        object_type = " ".join(object_type_words) if object_type_words else label_lower

        return object_type, color, material

    def _compute_3d_position(
        self,
        position_2d: List[int],
        image_shape: Tuple[int, int, int],
        depth_image: Optional[np.ndarray],
        camera_intrinsics: Optional[CameraIntrinsics]
    ) -> np.ndarray:
        """
        Convert 2D position (0-1000 scale) to 3D coordinates using depth.

        Args:
            position_2d: [y, x] in 0-1000 normalized scale
            image_shape: (height, width, channels)
            depth_image: Depth map in meters
            camera_intrinsics: Camera parameters

        Returns:
            3D position [x, y, z] in meters (camera frame)
        """
        # Convert normalized coords to pixel coords
        y_norm, x_norm = position_2d
        height, width = image_shape[:2]

        # Scale from 0-1000 to pixel coordinates
        pixel_y = int((y_norm / 1000.0) * height)
        pixel_x = int((x_norm / 1000.0) * width)

        # Clamp to image bounds
        pixel_y = np.clip(pixel_y, 0, height - 1)
        pixel_x = np.clip(pixel_x, 0, width - 1)

        # If no depth, return position in image plane at default depth
        if depth_image is None or camera_intrinsics is None:
            # Return position in normalized coordinates (no real 3D)
            return np.array([
                (pixel_x - width / 2) / width,
                (pixel_y - height / 2) / height,
                1.0  # Default depth
            ])

        # Get depth at pixel
        depth = depth_image[pixel_y, pixel_x]

        # If depth invalid, use median of surrounding region
        if depth <= 0 or depth > 10.0:  # Invalid depth
            region = depth_image[
                max(0, pixel_y - 5):min(height, pixel_y + 5),
                max(0, pixel_x - 5):min(width, pixel_x + 5)
            ]
            valid_depths = region[(region > 0) & (region < 10.0)]
            depth = np.median(valid_depths) if len(valid_depths) > 0 else 1.0

        # Convert to 3D using camera intrinsics
        # Support both ppx/ppy (RealSense) and cx/cy (standard) naming
        cx = getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', camera_intrinsics.width / 2))
        cy = getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', camera_intrinsics.height / 2))

        x_3d = (pixel_x - cx) * depth / camera_intrinsics.fx
        y_3d = (pixel_y - cy) * depth / camera_intrinsics.fy
        z_3d = depth

        return np.array([x_3d, y_3d, z_3d])

    def _extract_affordances_from_properties(
        self,
        properties: Dict[str, Any]
    ) -> List[str]:
        """
        Extract affordances directly from VLM-observed properties.

        The VLM should provide affordances based on visual analysis.
        This method extracts them from the properties dict without
        applying any hard-coded rules.
        """
        affordances = []

        # VLM may provide affordances directly
        if "affordances" in properties:
            vlm_affordances = properties["affordances"]
            if isinstance(vlm_affordances, list):
                affordances.extend(vlm_affordances)
            elif isinstance(vlm_affordances, str):
                affordances.append(vlm_affordances)

        # VLM may also embed affordances as boolean properties
        # e.g., {"graspable": true, "openable": false}
        affordance_keys = [
            "graspable", "containable", "supportable", "pourable",
            "openable", "closable", "movable", "stackable", "cuttable",
            "sittable", "edible", "interactive", "pushable", "pullable"
        ]

        for key in affordance_keys:
            if key in properties and properties[key] is True:
                if key not in affordances:
                    affordances.append(key)

        return affordances

    def _extract_size(self, properties: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract object size from VLM properties."""
        if "size" in properties:
            size_val = properties["size"]
            if isinstance(size_val, (list, tuple)) and len(size_val) == 3:
                return np.array(size_val)
            elif isinstance(size_val, dict):
                # Size as dict: {"width": 0.1, "height": 0.2, "depth": 0.1}
                if all(k in size_val for k in ["width", "height", "depth"]):
                    return np.array([
                        size_val["width"],
                        size_val["height"],
                        size_val["depth"]
                    ])

        return None

    def _generate_object_id(
        self,
        object_type: str,
        color: Optional[str],
        index: int
    ) -> str:
        """Generate unique object ID from observed attributes."""
        parts = []
        if color:
            parts.append(color)
        parts.append(object_type.replace(" ", "_"))
        parts.append(str(index + 1))
        return "_".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detections": self.total_detections,
            "total_frames": self.total_frames,
            "avg_objects_per_frame": (
                self.total_detections / self.total_frames if self.total_frames > 0 else 0
            ),
            "avg_processing_time": self.avg_processing_time
        }

    def reset_statistics(self):
        """Reset detection statistics."""
        self.total_detections = 0
        self.total_frames = 0
        self.avg_processing_time = 0.0
