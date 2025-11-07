"""
Spatial map for tracking object locations and relationships.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .object_registry import DetectedObject


class RelationType(Enum):
    """Types of spatial relationships."""

    ON = "on"
    IN = "in"
    NEAR = "near"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    TOUCHING = "touching"
    SUPPORTING = "supporting"


@dataclass
class SpatialRelationship:
    """Represents a spatial relationship between two objects."""

    subject_id: str  # Subject object ID
    relation: RelationType  # Type of relationship
    object_id: str  # Object ID
    confidence: float = 1.0  # Confidence in relationship (0-1)
    distance: Optional[float] = None  # Distance between objects (meters)

    def to_predicate(self) -> str:
        """Convert to PDDL predicate format."""
        return f"{self.relation.value}({self.subject_id}, {self.object_id})"

    def __str__(self) -> str:
        return self.to_predicate()


class SpatialMap:
    """
    Manages spatial relationships between objects.
    """

    def __init__(
        self,
        near_threshold: float = 0.3,
        contact_threshold: float = 0.05,
        vertical_threshold: float = 0.1
    ):
        """
        Initialize spatial map.

        Args:
            near_threshold: Distance threshold for 'near' relationship (meters)
            contact_threshold: Distance threshold for contact relationships (meters)
            vertical_threshold: Threshold for vertical relationships (meters)
        """
        self.near_threshold = near_threshold
        self.contact_threshold = contact_threshold
        self.vertical_threshold = vertical_threshold

        # Store relationships as adjacency list
        self.relationships: Dict[str, List[SpatialRelationship]] = {}

    def update_relationships(self, objects: List[DetectedObject]):
        """
        Update spatial relationships based on current object positions.

        Args:
            objects: List of detected objects
        """
        # Clear existing relationships
        self.relationships.clear()

        # Compute pairwise relationships
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                relations = self._compute_relationships(obj1, obj2)
                for relation in relations:
                    self._add_relationship(relation)

    def _compute_relationships(
        self, obj1: DetectedObject, obj2: DetectedObject
    ) -> List[SpatialRelationship]:
        """
        Compute spatial relationships between two objects.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            List of detected relationships
        """
        relationships = []

        # Calculate distance and relative position
        diff = obj2.position - obj1.position
        distance = float(np.linalg.norm(diff))

        # Near relationship (symmetric)
        if distance < self.near_threshold:
            relationships.append(
                SpatialRelationship(obj1.object_id, RelationType.NEAR, obj2.object_id, distance=distance)
            )
            relationships.append(
                SpatialRelationship(obj2.object_id, RelationType.NEAR, obj1.object_id, distance=distance)
            )

        # Touching relationship (symmetric)
        if distance < self.contact_threshold:
            relationships.append(
                SpatialRelationship(obj1.object_id, RelationType.TOUCHING, obj2.object_id, distance=distance)
            )
            relationships.append(
                SpatialRelationship(obj2.object_id, RelationType.TOUCHING, obj1.object_id, distance=distance)
            )

        # Vertical relationships (above/below)
        vertical_dist = abs(diff[2])  # Assuming z is vertical
        if vertical_dist > self.vertical_threshold:
            if diff[2] > 0:
                # obj2 is above obj1
                relationships.append(
                    SpatialRelationship(obj2.object_id, RelationType.ABOVE, obj1.object_id, distance=distance)
                )
                relationships.append(
                    SpatialRelationship(obj1.object_id, RelationType.BELOW, obj2.object_id, distance=distance)
                )
            else:
                # obj1 is above obj2
                relationships.append(
                    SpatialRelationship(obj1.object_id, RelationType.ABOVE, obj2.object_id, distance=distance)
                )
                relationships.append(
                    SpatialRelationship(obj2.object_id, RelationType.BELOW, obj1.object_id, distance=distance)
                )

        # Horizontal relationships (left/right, front/back)
        # Assuming: x = left/right, y = front/back
        horizontal_dist = np.sqrt(diff[0]**2 + diff[1]**2)

        if horizontal_dist > self.vertical_threshold:
            # Left/Right (x-axis)
            if abs(diff[0]) > abs(diff[1]):
                if diff[0] > 0:
                    relationships.append(
                        SpatialRelationship(obj2.object_id, RelationType.RIGHT_OF, obj1.object_id, distance=distance)
                    )
                    relationships.append(
                        SpatialRelationship(obj1.object_id, RelationType.LEFT_OF, obj2.object_id, distance=distance)
                    )
                else:
                    relationships.append(
                        SpatialRelationship(obj1.object_id, RelationType.RIGHT_OF, obj2.object_id, distance=distance)
                    )
                    relationships.append(
                        SpatialRelationship(obj2.object_id, RelationType.LEFT_OF, obj1.object_id, distance=distance)
                    )

            # Front/Back (y-axis)
            else:
                if diff[1] > 0:
                    relationships.append(
                        SpatialRelationship(obj2.object_id, RelationType.IN_FRONT_OF, obj1.object_id, distance=distance)
                    )
                    relationships.append(
                        SpatialRelationship(obj1.object_id, RelationType.BEHIND, obj2.object_id, distance=distance)
                    )
                else:
                    relationships.append(
                        SpatialRelationship(obj1.object_id, RelationType.IN_FRONT_OF, obj2.object_id, distance=distance)
                    )
                    relationships.append(
                        SpatialRelationship(obj2.object_id, RelationType.BEHIND, obj1.object_id, distance=distance)
                    )

        # Support relationship (on top of)
        # Object is "on" another if it's close, above, and velocities suggest support
        if distance < self.contact_threshold * 2 and diff[2] > 0 and vertical_dist < 0.2:
            # Check if obj2 is on top of obj1
            relationships.append(
                SpatialRelationship(
                    obj2.object_id, RelationType.ON, obj1.object_id,
                    confidence=0.8, distance=distance
                )
            )
            relationships.append(
                SpatialRelationship(
                    obj1.object_id, RelationType.SUPPORTING, obj2.object_id,
                    confidence=0.8, distance=distance
                )
            )

        return relationships

    def _add_relationship(self, relationship: SpatialRelationship):
        """Add a relationship to the map."""
        if relationship.subject_id not in self.relationships:
            self.relationships[relationship.subject_id] = []
        self.relationships[relationship.subject_id].append(relationship)

    def get_relationships(
        self, object_id: str, relation_type: Optional[RelationType] = None
    ) -> List[SpatialRelationship]:
        """
        Get relationships for an object.

        Args:
            object_id: Object ID
            relation_type: Optional filter by relation type

        Returns:
            List of relationships
        """
        if object_id not in self.relationships:
            return []

        relationships = self.relationships[object_id]

        if relation_type:
            relationships = [r for r in relationships if r.relation == relation_type]

        return relationships

    def get_objects_with_relation(
        self, subject_id: str, relation_type: RelationType
    ) -> List[str]:
        """
        Get all objects that have a specific relationship with the subject.

        Args:
            subject_id: Subject object ID
            relation_type: Type of relationship

        Returns:
            List of object IDs
        """
        relationships = self.get_relationships(subject_id, relation_type)
        return [r.object_id for r in relationships]

    def has_relationship(
        self, subject_id: str, relation_type: RelationType, object_id: str
    ) -> bool:
        """Check if a specific relationship exists."""
        relationships = self.get_relationships(subject_id, relation_type)
        return any(r.object_id == object_id for r in relationships)

    def get_all_relationships(self) -> List[SpatialRelationship]:
        """Get all relationships in the map."""
        all_relations = []
        for relations in self.relationships.values():
            all_relations.extend(relations)
        return all_relations

    def get_predicates(self) -> List[str]:
        """
        Get all relationships as PDDL predicates.

        Returns:
            List of predicate strings
        """
        predicates = []
        for relations in self.relationships.values():
            for relation in relations:
                predicates.append(relation.to_predicate())
        return predicates

    def clear(self):
        """Clear all relationships."""
        self.relationships.clear()

    def to_dict(self) -> Dict:
        """Convert spatial map to dictionary."""
        return {
            "relationships": {
                obj_id: [
                    {
                        "relation": r.relation.value,
                        "object_id": r.object_id,
                        "confidence": r.confidence,
                        "distance": r.distance,
                    }
                    for r in relations
                ]
                for obj_id, relations in self.relationships.items()
            },
            "total_relationships": sum(len(r) for r in self.relationships.values()),
        }
