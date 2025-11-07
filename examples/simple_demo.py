#!/usr/bin/env python3
"""
Simple demonstration of the VLM Spatial Perception System components.

This script demonstrates the basic functionality of the implemented components:
- Camera capture
- Task parsing
- Object tracking
- Spatial relationships
- PDDL state generation

Note: This is a mock demo using simulated detections since the perception
pipeline is not yet implemented.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("VLM Semantic Spatial Perception System - Simple Demo")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Camera Setup
    # ========================================================================
    print("1. Setting up RealSense camera...")

    try:
        from src.camera import RealSenseCamera
        import cv2

        camera = RealSenseCamera(width=640, height=480, enable_depth=True, auto_start=True)
        print("   ✓ RealSense camera initialized")

        # Get camera intrinsics
        intrinsics = camera.get_camera_intrinsics()
        print(f"   ✓ Camera intrinsics: {intrinsics.width}x{intrinsics.height}")
        print(f"   ✓ Focal length: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        print(f"   ✓ Principal point: cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")

        # Capture aligned RGB-D frames
        print("   → Capturing RGB-D frames...")
        color_frame, depth_frame = camera.get_aligned_frames()
        print(f"   ✓ Color frame: {color_frame.shape}")
        print(f"   ✓ Depth frame: {depth_frame.shape}")
        print(f"   ✓ Depth range: {depth_frame.min():.3f}m to {depth_frame.max():.3f}m")

        # Display the captured image
        print("   → Displaying captured image (press any key to continue)...")

        # Convert RGB to BGR for OpenCV display
        color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        # Create depth visualization
        from src.camera import visualize_depth
        depth_colored = visualize_depth(depth_frame, max_depth=3.0)
        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # Stack images side by side
        display = np.hstack([color_bgr, depth_bgr])

        # Add labels
        cv2.putText(display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(display, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("RealSense RGB-D Capture", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("   ✓ Image displayed successfully")

        camera.stop()

    except ImportError:
        print("   ⚠ pyrealsense2 not installed. Install with: pip install pyrealsense2")
        print("   → Continuing with mock data...")
    except Exception as e:
        print(f"   ⚠ RealSense camera not available: {e}")
        print("   → Continuing with mock data...")

    print()

    # ========================================================================
    # 2. Task Parsing
    # ========================================================================
    print("2. Parsing natural language task...")
    from src.task import TaskManager, TaskParser

    parser = TaskParser()
    manager = TaskManager()

    task_desc = "pick up the red cup on the table"
    print(f'   Task: "{task_desc}"')

    parsed = parser.parse(task_desc)
    print(f"   ✓ Action: {parsed['action']}")
    print(f"   ✓ Objects: {parsed['objects']}")
    print(f"   ✓ Goal predicates: {parsed['goal_predicates']}")

    # Create and activate task
    task = manager.create_task(task_desc, parsed, priority=1)
    manager.set_current_task(task.task_id)
    print(f"   ✓ Task created: {task.task_id}")

    # Get task context
    context = manager.get_task_context()
    print(f"   ✓ Task context generated for perception")
    print()

    # ========================================================================
    # 3. World Model - Object Tracking
    # ========================================================================
    print("3. Creating world model and tracking objects...")
    from src.world_model import DetectedObject, WorldState

    world = WorldState(persistence_time=5.0)

    # Simulate object detections (in real system, these come from VLM)
    print("   → Simulating object detections...")

    red_cup = DetectedObject(
        object_id="cup_0",
        object_type="cup",
        position=np.array([0.5, 0.2, 0.15]),  # x, y, z in meters
        confidence=0.92,
        timestamp=time.time(),
        bbox_2d=(200, 150, 280, 250),
        color="red",
        size=np.array([0.08, 0.08, 0.12]),  # width, height, depth
        affordances=["graspable", "containable"],
    )

    table = DetectedObject(
        object_id="table_0",
        object_type="table",
        position=np.array([0.5, 0.2, 0.0]),
        confidence=0.95,
        timestamp=time.time(),
        bbox_2d=(50, 200, 600, 450),
        color="brown",
        size=np.array([1.0, 0.6, 0.02]),
        affordances=["supportable"],
    )

    blue_bottle = DetectedObject(
        object_id="bottle_0",
        object_type="bottle",
        position=np.array([0.7, 0.3, 0.18]),
        confidence=0.88,
        timestamp=time.time(),
        color="blue",
        affordances=["graspable", "containable"],
    )

    # Update world model
    world.update([red_cup, table, blue_bottle])
    print(f"   ✓ Objects tracked: {world.get_object_count()}")

    # Query objects
    cups = world.get_objects_by_type("cup")
    print(f"   ✓ Cups detected: {len(cups)}")

    graspable = world.get_objects_with_affordance("graspable")
    print(f"   ✓ Graspable objects: {len(graspable)}")

    print()

    # ========================================================================
    # 4. Spatial Relationships
    # ========================================================================
    print("4. Analyzing spatial relationships...")

    relationships = world.get_all_relationships()
    print(f"   ✓ Total relationships detected: {len(relationships)}")

    # Show some relationships
    print("   → Key relationships:")
    for rel in relationships[:5]:  # Show first 5
        print(f"      - {rel.to_predicate()}")

    print()

    # ========================================================================
    # 5. PDDL State Generation
    # ========================================================================
    print("5. Generating PDDL representation...")

    pddl_state = world.get_pddl_state()

    print(f"   ✓ Objects in PDDL: {len(pddl_state['objects'])}")
    print("   → Objects:")
    for obj in pddl_state["objects"]:
        print(f"      - {obj}")

    print(f"\n   ✓ Predicates: {len(pddl_state['predicates'])}")
    print("   → Sample predicates:")
    for pred in pddl_state["predicates"][:8]:  # Show first 8
        print(f"      - {pred}")

    print()

    # ========================================================================
    # 6. Scene Description
    # ========================================================================
    print("6. Scene summary...")

    scene_desc = world.get_scene_description()
    print(f"   ✓ Total objects: {scene_desc['total_objects']}")
    print(f"   ✓ Object types: {scene_desc['object_types']}")
    print(f"   ✓ Total relationships: {scene_desc['total_relationships']}")
    print(f"   ✓ Relationship types: {scene_desc['relationship_types']}")

    print()

    # ========================================================================
    # 7. Task Status
    # ========================================================================
    print("7. Task status...")

    current_task = manager.get_current_task()
    print(f"   Current task: {current_task.description}")
    print(f"   Status: {current_task.status.value}")
    print(f"   Goal objects: {current_task.goal_objects}")
    print(f"   Required affordances: {current_task.required_affordances}")

    # Simulate task completion
    print("\n   → Simulating task completion...")
    manager.complete_current_task()
    print("   ✓ Task completed!")

    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nWhat was demonstrated:")
    print("  ✓ Camera initialization and frame capture")
    print("  ✓ Natural language task parsing")
    print("  ✓ Object detection and tracking")
    print("  ✓ Spatial relationship extraction")
    print("  ✓ PDDL state generation")
    print("  ✓ Task lifecycle management")
    print("\nNext steps:")
    print("  → Implement VLM perception pipeline")
    print("  → Add Gemini Robotics ER integration")
    print("  → Complete PDDL generation system")
    print("  → Build main control loop")
    print()


if __name__ == "__main__":
    main()
