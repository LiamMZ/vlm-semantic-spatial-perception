"""
Basic integration tests for implemented components.

Run with: pytest tests/test_basic_integration.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCameraLayer:
    """Test camera abstraction layer."""

    def test_webcam_camera_import(self):
        """Test webcam camera can be imported."""
        from src.camera import WebcamCamera

        assert WebcamCamera is not None

    def test_camera_intrinsics_dataclass(self):
        """Test CameraIntrinsics dataclass."""
        from src.camera import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )

        assert intrinsics.fx == 500.0
        assert intrinsics.width == 640

        # Test to_matrix
        matrix = intrinsics.to_matrix()
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 500.0

    def test_camera_frame_dataclass(self):
        """Test CameraFrame dataclass."""
        from src.camera import CameraFrame

        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32)

        frame = CameraFrame(color=color, depth=depth, timestamp=time.time())

        assert frame.color.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert frame.timestamp is not None


class TestWorldModel:
    """Test world model components."""

    def test_detected_object_creation(self):
        """Test DetectedObject creation."""
        from src.world_model import DetectedObject

        obj = DetectedObject(
            object_id="cup_0",
            object_type="cup",
            position=np.array([0.5, 0.2, 0.1]),
            confidence=0.9,
            timestamp=time.time(),
            affordances=["graspable"],
        )

        assert obj.object_id == "cup_0"
        assert obj.object_type == "cup"
        assert len(obj.affordances) == 1
        assert obj.confidence == 0.9

    def test_detected_object_distance(self):
        """Test distance calculation between objects."""
        from src.world_model import DetectedObject

        obj1 = DetectedObject(
            object_id="obj1",
            object_type="cup",
            position=np.array([0.0, 0.0, 0.0]),
            confidence=1.0,
            timestamp=time.time(),
        )

        obj2 = DetectedObject(
            object_id="obj2",
            object_type="bottle",
            position=np.array([1.0, 0.0, 0.0]),
            confidence=1.0,
            timestamp=time.time(),
        )

        distance = obj1.distance_to(obj2)
        assert abs(distance - 1.0) < 0.01

    def test_object_registry(self):
        """Test ObjectRegistry functionality."""
        from src.world_model import DetectedObject, ObjectRegistry

        registry = ObjectRegistry(persistence_time=5.0, position_tolerance=0.1)

        obj = DetectedObject(
            object_id="cup_0",
            object_type="cup",
            position=np.array([0.5, 0.2, 0.1]),
            confidence=0.9,
            timestamp=time.time(),
        )

        # Add object
        obj_id = registry.add_or_update(obj)
        assert obj_id is not None
        assert registry.get_object_count() == 1

        # Get object
        retrieved = registry.get_object(obj_id)
        assert retrieved is not None
        assert retrieved.object_type == "cup"

    def test_spatial_relationships(self):
        """Test spatial relationship detection."""
        from src.world_model import DetectedObject, SpatialMap

        spatial_map = SpatialMap(near_threshold=0.5)

        obj1 = DetectedObject(
            object_id="cup",
            object_type="cup",
            position=np.array([0.0, 0.0, 0.1]),
            confidence=1.0,
            timestamp=time.time(),
        )

        obj2 = DetectedObject(
            object_id="table",
            object_type="table",
            position=np.array([0.0, 0.0, 0.0]),
            confidence=1.0,
            timestamp=time.time(),
        )

        # Update relationships
        spatial_map.update_relationships([obj1, obj2])

        # Check relationships
        relationships = spatial_map.get_all_relationships()
        assert len(relationships) > 0

    def test_world_state_integration(self):
        """Test WorldState integration."""
        from src.world_model import DetectedObject, WorldState

        world = WorldState()

        obj = DetectedObject(
            object_id="cup_0",
            object_type="cup",
            position=np.array([0.5, 0.2, 0.1]),
            confidence=0.9,
            timestamp=time.time(),
            affordances=["graspable", "containable"],
        )

        # Update world
        world.update([obj])

        # Check state
        assert world.get_object_count() == 1

        # Get all objects
        objects = world.get_all_objects()
        assert len(objects) == 1

        # Get PDDL state
        pddl_state = world.get_pddl_state()
        assert "objects" in pddl_state
        assert "predicates" in pddl_state


class TestTaskManagement:
    """Test task management components."""

    def test_task_parser_import(self):
        """Test TaskParser can be imported."""
        from src.task import TaskParser

        parser = TaskParser()
        assert parser is not None

    def test_task_parsing_pick(self):
        """Test parsing a pick task."""
        from src.task import TaskParser

        parser = TaskParser()
        result = parser.parse("pick up the red cup")

        assert result["action"] == "pick"
        assert "cup" in str(result["objects"]).lower()
        assert len(result["goal_predicates"]) > 0

    def test_task_parsing_place(self):
        """Test parsing a place task."""
        from src.task import TaskParser

        parser = TaskParser()
        result = parser.parse("place the bottle on the table")

        assert result["action"] == "place"
        assert len(result["spatial_constraints"]) > 0

    def test_task_manager(self):
        """Test TaskManager functionality."""
        from src.task import TaskManager, TaskParser

        parser = TaskParser()
        manager = TaskManager()

        # Parse task
        task_desc = "pick up the red cup"
        parsed = parser.parse(task_desc)

        # Create task
        task = manager.create_task(task_desc, parsed)
        assert task.task_id is not None
        assert task.description == task_desc

        # Set current task
        manager.set_current_task(task.task_id)
        current = manager.get_current_task()
        assert current is not None
        assert current.task_id == task.task_id

        # Get task context
        context = manager.get_task_context()
        assert context is not None
        assert "action" in context

    def test_task_status_lifecycle(self):
        """Test task status transitions."""
        from src.task import Task, TaskStatus

        task = Task(task_id="test_0", description="test task")

        # Initial status
        assert task.status == TaskStatus.PENDING

        # Start task
        task.start()
        assert task.status == TaskStatus.ACTIVE
        assert task.started_at is not None

        # Complete task
        task.complete()
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

        # Check duration
        duration = task.get_duration()
        assert duration is not None
        assert duration >= 0


class TestIntegration:
    """Test integration between components."""

    def test_full_pipeline_mock(self):
        """Test a complete pipeline with mock data."""
        from src.task import TaskManager, TaskParser
        from src.world_model import DetectedObject, WorldState

        # 1. Create task
        parser = TaskParser()
        manager = TaskManager()

        task_desc = "pick up the red cup on the table"
        parsed = parser.parse(task_desc)
        task = manager.create_task(task_desc, parsed)
        manager.set_current_task(task.task_id)

        # 2. Get task context
        context = manager.get_task_context()
        assert context is not None

        # 3. Create world state
        world = WorldState()

        # 4. Add mock detections
        cup = DetectedObject(
            object_id="cup_0",
            object_type="cup",
            position=np.array([0.5, 0.2, 0.15]),
            confidence=0.9,
            timestamp=time.time(),
            color="red",
            affordances=["graspable"],
        )

        table = DetectedObject(
            object_id="table_0",
            object_type="table",
            position=np.array([0.5, 0.2, 0.0]),
            confidence=0.95,
            timestamp=time.time(),
            affordances=["supportable"],
        )

        # 5. Update world model
        world.update([cup, table])

        # 6. Verify state
        assert world.get_object_count() == 2

        # 7. Get PDDL representation
        pddl_state = world.get_pddl_state()
        assert len(pddl_state["objects"]) == 2
        assert len(pddl_state["predicates"]) > 0

        # 8. Check spatial relationships
        relationships = world.get_all_relationships()
        assert len(relationships) > 0

        # Success!
        print("\n✓ Full pipeline integration test passed!")
        print(f"  Task: {task_desc}")
        print(f"  Objects detected: {world.get_object_count()}")
        print(f"  Relationships: {len(relationships)}")
        print(f"  PDDL predicates: {len(pddl_state['predicates'])}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
