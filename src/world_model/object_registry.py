"""
Object registry for tracking detected objects over time.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DetectedObject:
    """
    Represents a detected object in the scene.
    """

    object_id: str  # Unique identifier
    object_type: str  # Object category (e.g., 'cup', 'bottle')
    position: np.ndarray  # 3D position [x, y, z] in meters
    confidence: float  # Detection confidence (0-1)
    timestamp: float  # Time of detection

    # Optional attributes
    bbox_2d: Optional[tuple] = None  # 2D bounding box (x_min, y_min, x_max, y_max)
    bbox_3d: Optional[np.ndarray] = None  # 3D bounding box
    orientation: Optional[np.ndarray] = None  # Orientation (quaternion or euler)
    size: Optional[np.ndarray] = None  # Object dimensions [width, height, depth]
    color: Optional[str] = None  # Dominant color
    material: Optional[str] = None  # Material type
    affordances: List[str] = field(default_factory=list)  # List of affordances
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    last_seen: float = field(default_factory=time.time)  # Last observation time

    def update_position(self, new_position: np.ndarray, alpha: float = 0.3):
        """
        Update position with exponential smoothing.

        Args:
            new_position: New observed position
            alpha: Smoothing factor (0=no update, 1=full update)
        """
        self.position = alpha * new_position + (1 - alpha) * self.position
        self.last_seen = time.time()

    def update(self, **kwargs):
        """Update object attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_seen = time.time()

    def distance_to(self, other: "DetectedObject") -> float:
        """Calculate Euclidean distance to another object."""
        return float(np.linalg.norm(self.position - other.position))

    def is_near(self, other: "DetectedObject", threshold: float = 0.3) -> bool:
        """Check if object is near another object."""
        return self.distance_to(other) < threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "position": self.position.tolist(),
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "bbox_2d": self.bbox_2d,
            "bbox_3d": self.bbox_3d.tolist() if self.bbox_3d is not None else None,
            "orientation": self.orientation.tolist() if self.orientation is not None else None,
            "size": self.size.tolist() if self.size is not None else None,
            "color": self.color,
            "material": self.material,
            "affordances": self.affordances,
            "properties": self.properties,
            "last_seen": self.last_seen,
        }


class ObjectRegistry:
    """
    Registry for tracking detected objects over time with persistence.
    """

    def __init__(self, persistence_time: float = 5.0, position_tolerance: float = 0.1):
        """
        Initialize object registry.

        Args:
            persistence_time: Time (seconds) to keep unobserved objects
            position_tolerance: Distance threshold for matching objects (meters)
        """
        self.objects: Dict[str, DetectedObject] = {}
        self.persistence_time = persistence_time
        self.position_tolerance = position_tolerance
        self._next_id = 0

    def add_or_update(self, detection: DetectedObject) -> str:
        """
        Add a new object or update existing one.

        Args:
            detection: Detected object

        Returns:
            str: Object ID (assigned or matched)
        """
        # Try to match with existing objects
        matched_id = self._match_object(detection)

        if matched_id:
            # Update existing object
            existing = self.objects[matched_id]
            existing.update_position(detection.position)
            existing.update(
                confidence=detection.confidence,
                bbox_2d=detection.bbox_2d,
                affordances=detection.affordances,
                **detection.properties
            )
            return matched_id
        else:
            # Add new object
            if not detection.object_id:
                detection.object_id = self._generate_id(detection.object_type)
            self.objects[detection.object_id] = detection
            return detection.object_id

    def _match_object(self, detection: DetectedObject) -> Optional[str]:
        """
        Match detection with existing objects based on position and type.

        Args:
            detection: New detection to match

        Returns:
            Optional[str]: Matched object ID or None
        """
        candidates = []

        for obj_id, obj in self.objects.items():
            # Must be same type
            if obj.object_type != detection.object_type:
                continue

            # Check position distance
            distance = np.linalg.norm(obj.position - detection.position)
            if distance < self.position_tolerance:
                candidates.append((obj_id, distance))

        # Return closest match
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def _generate_id(self, object_type: str) -> str:
        """Generate unique object ID."""
        obj_id = f"{object_type}_{self._next_id}"
        self._next_id += 1
        return obj_id

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """Get object by ID."""
        return self.objects.get(object_id)

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def get_all_objects(self) -> List[DetectedObject]:
        """Get all tracked objects."""
        return list(self.objects.values())

    def get_objects_in_region(
        self, center: np.ndarray, radius: float
    ) -> List[DetectedObject]:
        """
        Get all objects within a spherical region.

        Args:
            center: Center position [x, y, z]
            radius: Search radius in meters

        Returns:
            List of objects within region
        """
        objects_in_region = []
        for obj in self.objects.values():
            distance = np.linalg.norm(obj.position - center)
            if distance <= radius:
                objects_in_region.append(obj)
        return objects_in_region

    def prune_old_objects(self):
        """Remove objects that haven't been seen recently."""
        current_time = time.time()
        to_remove = []

        for obj_id, obj in self.objects.items():
            if current_time - obj.last_seen > self.persistence_time:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.objects[obj_id]

        return len(to_remove)

    def clear(self):
        """Clear all objects from registry."""
        self.objects.clear()
        self._next_id = 0

    def get_object_count(self) -> int:
        """Get total number of tracked objects."""
        return len(self.objects)

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """Get all objects with a specific affordance."""
        return [
            obj for obj in self.objects.values()
            if affordance in obj.affordances
        ]

    def to_dict(self) -> Dict:
        """Convert registry to dictionary."""
        return {
            "objects": {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            "object_count": len(self.objects),
            "persistence_time": self.persistence_time,
            "position_tolerance": self.position_tolerance,
        }
