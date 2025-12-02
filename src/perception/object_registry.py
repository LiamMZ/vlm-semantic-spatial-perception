"""
Thread-safe registry for managing detected objects.

This module provides a registry for storing and querying detected objects
with thread safety for concurrent access.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np


@dataclass
class InteractionPoint:
    """Interaction point for a specific affordance."""

    position_2d: List[int]  # [y, x] in 0-1000 normalized scale
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in meters (if depth available)
    confidence: float = 0.0
    reasoning: str = ""
    alternative_points: List[Dict[str, any]] = field(default_factory=list)


@dataclass
class DetectedObject:
    """
    Represents a detected object with affordances and interaction points.

    Attributes:
        object_type: Category/class of object (e.g., "cup", "bottle")
        object_id: Unique identifier (e.g., "red_cup_1")
        affordances: Set of possible robot actions (e.g., {"graspable", "pourable"})
        interaction_points: Dict mapping affordances to interaction points
        position_2d: Center position [y, x] in 0-1000 scale
        position_3d: Center position [x, y, z] in meters
        bounding_box_2d: Optional 2D bounding box
        properties: Additional properties (color, size, etc.)
        pddl_state: Dict mapping PDDL predicate names to boolean values (e.g., {"clean": True, "dirty": False})
    """

    object_type: str
    object_id: str
    affordances: set[str] = field(default_factory=set)
    interaction_points: Dict[str, InteractionPoint] = field(default_factory=dict)
    position_2d: Optional[List[int]] = None  # [y, x] in 0-1000 scale
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in meters
    bounding_box_2d: Optional[List[int]] = None  # [y1, x1, y2, x2]
    properties: Dict[str, any] = field(default_factory=dict)
    pddl_state: Dict[str, bool] = field(default_factory=dict)  # PDDL predicate states
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class DetectedObjectRegistry:
    """
    Thread-safe registry for managing detected objects.

    This class provides a centralized storage and query interface for detected
    objects with proper thread synchronization for concurrent access.

    Example:
        >>> registry = DetectedObjectRegistry()
        >>> registry.add_object(detected_obj)
        >>> obj = registry.get_object("red_cup_1")
        >>> all_cups = registry.get_objects_by_type("cup")
    """

    def __init__(self):
        """Initialize empty registry with thread safety."""
        self._objects: Dict[str, DetectedObject] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested access

    def add_object(self, obj: DetectedObject) -> None:
        """
        Add object to registry (thread-safe).

        Args:
            obj: DetectedObject to add
        """
        with self._lock:
            self._objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """
        Get object by ID from registry (thread-safe).

        Args:
            object_id: ID of object to retrieve

        Returns:
            DetectedObject or None if not found
        """
        with self._lock:
            return self._objects.get(object_id)

    def get_all_objects(self) -> List[DetectedObject]:
        """
        Get all objects from registry (thread-safe).

        Returns:
            List of all detected objects (snapshot)
        """
        with self._lock:
            return list(self._objects.values())

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """
        Get all objects of a specific type (thread-safe).

        Args:
            object_type: Type to filter by (e.g., "cup", "bottle")

        Returns:
            List of objects matching the type
        """
        with self._lock:
            return [obj for obj in self._objects.values() if obj.object_type == object_type]

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """
        Get all objects that have a specific affordance (thread-safe).

        Args:
            affordance: Affordance to filter by (e.g., "graspable", "pourable")

        Returns:
            List of objects with the affordance
        """
        with self._lock:
            return [obj for obj in self._objects.values() if affordance in obj.affordances]

    def update_object(self, object_id: str, obj: DetectedObject) -> bool:
        """
        Update existing object in registry (thread-safe).

        Args:
            object_id: ID of object to update
            obj: New DetectedObject data

        Returns:
            True if updated, False if object not found
        """
        with self._lock:
            if object_id in self._objects:
                self._objects[object_id] = obj
                return True
            return False

    def remove_object(self, object_id: str) -> bool:
        """
        Remove object from registry (thread-safe).

        Args:
            object_id: ID of object to remove

        Returns:
            True if removed, False if object not found
        """
        with self._lock:
            if object_id in self._objects:
                del self._objects[object_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all objects from registry (thread-safe)."""
        with self._lock:
            self._objects.clear()

    def count(self) -> int:
        """
        Get number of objects in registry (thread-safe).

        Returns:
            Number of objects
        """
        with self._lock:
            return len(self._objects)

    def contains(self, object_id: str) -> bool:
        """
        Check if object exists in registry (thread-safe).

        Args:
            object_id: ID to check

        Returns:
            True if object exists
        """
        with self._lock:
            return object_id in self._objects

    def generate_unique_id(self, object_name: str, object_type: str) -> str:
        """
        Generate unique object ID (thread-safe).

        Args:
            object_name: Descriptive name (e.g., "red cup")
            object_type: Object type (e.g., "cup")

        Returns:
            Unique ID like "red_cup" or "red_cup_2"
        """
        with self._lock:
            # Create ID like "red_cup_1"
            base_id = object_name.replace(" ", "_").lower()

            # Check if already exists
            if base_id not in self._objects:
                return base_id

            # Count existing objects of this type
            existing = [obj for obj in self._objects.values() if obj.object_type == object_type]
            count = len(existing) + 1

            # Add number suffix
            return f"{base_id}_{count}"

    def to_dict(self) -> dict:
        """
        Convert registry to dictionary format (thread-safe).

        Returns:
            Dictionary with 'num_objects' and 'objects' keys
        """
        from datetime import datetime

        # Convert objects to serializable format (thread-safe snapshot)
        with self._lock:
            objects_data = []
            for obj in self._objects.values():
                obj_dict = {
                    "object_type": obj.object_type,
                    "object_id": obj.object_id,
                    "affordances": list(obj.affordances),
                    "properties": obj.properties,
                    "confidence": obj.confidence,
                    "timestamp": obj.timestamp,
                }

                # Add positional data if available
                if obj.position_2d:
                    obj_dict["position_2d"] = obj.position_2d
                if obj.position_3d is not None:
                    obj_dict["position_3d"] = obj.position_3d.tolist() if hasattr(obj.position_3d, 'tolist') else obj.position_3d
                if obj.bounding_box_2d:
                    obj_dict["bounding_box_2d"] = obj.bounding_box_2d

                # Add PDDL state if available
                if obj.pddl_state:
                    obj_dict["pddl_state"] = obj.pddl_state

                objects_data.append(obj_dict)

        # Return dictionary
        return {
            "num_objects": len(objects_data),
            "snapshot_timestamp": datetime.now().isoformat(),
            "objects": objects_data
        }

    def save_to_json(self, output_path: str, include_timestamp: bool = True) -> str:
        """
        Save registry to JSON file (thread-safe).

        Args:
            output_path: Path to save JSON file
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved file
        """
        import json
        from pathlib import Path
        from datetime import datetime

        # Prepare output path
        output_path = Path(output_path)
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = output_path.stem
            output_path = output_path.parent / f"{stem}_{timestamp}{output_path.suffix}"

        # Convert objects to serializable format (thread-safe snapshot)
        with self._lock:
            objects_data = []
            for obj in self._objects.values():
                obj_dict = {
                    "object_type": obj.object_type,
                    "object_id": obj.object_id,
                    "affordances": list(obj.affordances),
                    "properties": obj.properties,
                    "confidence": obj.confidence,
                    "timestamp": obj.timestamp,
                }

                objects_data.append(obj_dict)

        # Create output data
        output_data = {
            "num_objects": len(objects_data),
            "detection_timestamp": datetime.now().isoformat(),
            "objects": objects_data
        }

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved {len(objects_data)} objects to {output_path}")
        return str(output_path)

    def load_from_json(self, input_path: str) -> List[DetectedObject]:
        """
        Load registry from JSON file (thread-safe).

        Args:
            input_path: Path to JSON file

        Returns:
            List of loaded DetectedObject instances
        """
        import json
        from pathlib import Path

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        with open(input_path, 'r') as f:
            data = json.load(f)
        
        loaded_objects = []
        with self._lock:
            for obj_dict in data.get("objects", []):
                obj = DetectedObject(
                    object_type=obj_dict["object_type"],
                    object_id=obj_dict["object_id"],
                    affordances=set(obj_dict.get("affordances", [])),
                    interaction_points={},
                    position_2d=None,
                    position_3d=None,
                    bounding_box_2d=None,
                    properties=obj_dict.get("properties", {}),
                    confidence=obj_dict.get("confidence", 1.0),
                    timestamp=obj_dict.get("timestamp", time.time())
                )

                loaded_objects.append(obj)
                # Add to registry
                self._objects[obj.object_id] = obj

        print(f"✓ Loaded {len(loaded_objects)} objects from {input_path}")
        return loaded_objects

    def __len__(self) -> int:
        """Get number of objects in registry."""
        return self.count()

    def __contains__(self, object_id: str) -> bool:
        """Check if object exists in registry."""
        return self.contains(object_id)

    def __repr__(self) -> str:
        """String representation of registry."""
        with self._lock:
            return f"DetectedObjectRegistry(objects={len(self._objects)})"
