"""
Unified world state combining object registry and spatial relationships.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .object_registry import DetectedObject, ObjectRegistry
from .spatial_map import RelationType, SpatialMap, SpatialRelationship


class WorldState:
    """
    Unified world state representation combining objects and spatial relationships.

    This class serves as the central hub for all spatial and semantic information
    about the environment, updated by perception and queried by planning.
    """

    def __init__(
        self,
        persistence_time: float = 5.0,
        position_tolerance: float = 0.1,
        near_threshold: float = 0.3,
        contact_threshold: float = 0.05,
        enable_history: bool = True,
        max_history_length: int = 1000
    ):
        """
        Initialize world state.

        Args:
            persistence_time: Time to keep unobserved objects (seconds)
            position_tolerance: Distance threshold for object matching (meters)
            near_threshold: Distance for 'near' relationships (meters)
            contact_threshold: Distance for contact relationships (meters)
            enable_history: Whether to maintain state history
            max_history_length: Maximum history entries to keep
        """
        # Core components
        self.object_registry = ObjectRegistry(persistence_time, position_tolerance)
        self.spatial_map = SpatialMap(near_threshold, contact_threshold)

        # State management
        self.frame_count = 0
        self.last_update_time = time.time()
        self.enable_history = enable_history
        self.max_history_length = max_history_length
        self.history: List[Dict] = []

        # Metadata
        self.scene_type: Optional[str] = None
        self.workspace_bounds: Optional[np.ndarray] = None

    def update(self, detections: List[DetectedObject]):
        """
        Update world state with new detections.

        Args:
            detections: List of detected objects from current frame
        """
        # Update object registry
        for detection in detections:
            self.object_registry.add_or_update(detection)

        # Prune old objects
        self.object_registry.prune_old_objects()

        # Update spatial relationships
        all_objects = self.object_registry.get_all_objects()
        self.spatial_map.update_relationships(all_objects)

        # Update metadata
        self.frame_count += 1
        self.last_update_time = time.time()

        # Save to history
        if self.enable_history:
            self._save_to_history()

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """Get object by ID."""
        return self.object_registry.get_object(object_id)

    def get_all_objects(self) -> List[DetectedObject]:
        """Get all tracked objects."""
        return self.object_registry.get_all_objects()

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Get all objects of a specific type."""
        return self.object_registry.get_objects_by_type(object_type)

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """Get all objects with a specific affordance."""
        return self.object_registry.get_objects_with_affordance(affordance)

    def get_relationships(
        self, object_id: str, relation_type: Optional[RelationType] = None
    ) -> List[SpatialRelationship]:
        """Get spatial relationships for an object."""
        return self.spatial_map.get_relationships(object_id, relation_type)

    def get_all_relationships(self) -> List[SpatialRelationship]:
        """Get all spatial relationships."""
        return self.spatial_map.get_all_relationships()

    def query_objects_near(self, position: np.ndarray, radius: float) -> List[DetectedObject]:
        """
        Query objects near a position.

        Args:
            position: 3D position [x, y, z]
            radius: Search radius in meters

        Returns:
            List of objects within radius
        """
        return self.object_registry.get_objects_in_region(position, radius)

    def query_relationship(
        self, subject_id: str, relation_type: RelationType, object_id: str
    ) -> bool:
        """Check if a specific relationship exists."""
        return self.spatial_map.has_relationship(subject_id, relation_type, object_id)

    def get_scene_description(self) -> Dict:
        """
        Get a high-level description of the current scene.

        Returns:
            Dictionary containing scene information
        """
        objects = self.get_all_objects()
        relationships = self.get_all_relationships()

        # Count object types
        type_counts = {}
        for obj in objects:
            type_counts[obj.object_type] = type_counts.get(obj.object_type, 0) + 1

        # Count relationship types
        relation_counts = {}
        for rel in relationships:
            rel_type = rel.relation.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

        return {
            "scene_type": self.scene_type,
            "total_objects": len(objects),
            "object_types": type_counts,
            "total_relationships": len(relationships),
            "relationship_types": relation_counts,
            "frame_count": self.frame_count,
            "last_update": self.last_update_time,
        }

    def get_pddl_state(self) -> Dict[str, List[str]]:
        """
        Get current state in PDDL format.

        Returns:
            Dictionary with 'objects' and 'predicates' lists
        """
        objects = self.get_all_objects()
        relationships = self.get_all_relationships()

        # Object declarations
        pddl_objects = []
        for obj in objects:
            pddl_objects.append(f"{obj.object_id} - {obj.object_type}")

        # Predicates
        pddl_predicates = []

        # Spatial relationship predicates
        for rel in relationships:
            pddl_predicates.append(rel.to_predicate())

        # Affordance predicates
        for obj in objects:
            for affordance in obj.affordances:
                pddl_predicates.append(f"{affordance}({obj.object_id})")

        # Property predicates
        for obj in objects:
            if obj.color:
                pddl_predicates.append(f"has-color({obj.object_id}, {obj.color})")
            if obj.material:
                pddl_predicates.append(f"has-material({obj.object_id}, {obj.material})")

        return {
            "objects": pddl_objects,
            "predicates": pddl_predicates,
        }

    def _save_to_history(self):
        """Save current state snapshot to history."""
        if not self.enable_history:
            return

        snapshot = {
            "timestamp": time.time(),
            "frame_count": self.frame_count,
            "object_count": self.object_registry.get_object_count(),
            "relationship_count": len(self.get_all_relationships()),
        }

        self.history.append(snapshot)

        # Trim history if too long
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def save_to_file(self, filepath: str):
        """
        Save world state to JSON file.

        Args:
            filepath: Path to save file
        """
        state = {
            "metadata": {
                "timestamp": time.time(),
                "frame_count": self.frame_count,
                "scene_type": self.scene_type,
            },
            "objects": self.object_registry.to_dict(),
            "spatial_map": self.spatial_map.to_dict(),
            "scene_description": self.get_scene_description(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def clear(self):
        """Clear all objects and relationships."""
        self.object_registry.clear()
        self.spatial_map.clear()
        self.frame_count = 0
        self.history.clear()

    def get_object_count(self) -> int:
        """Get total number of tracked objects."""
        return self.object_registry.get_object_count()

    def to_dict(self) -> Dict:
        """Convert world state to dictionary."""
        return {
            "frame_count": self.frame_count,
            "last_update_time": self.last_update_time,
            "scene_type": self.scene_type,
            "objects": self.object_registry.to_dict(),
            "spatial_map": self.spatial_map.to_dict(),
            "scene_description": self.get_scene_description(),
        }

    def __str__(self) -> str:
        """String representation of world state."""
        desc = self.get_scene_description()
        return (
            f"WorldState(objects={desc['total_objects']}, "
            f"relationships={desc['total_relationships']}, "
            f"frame={self.frame_count})"
        )
