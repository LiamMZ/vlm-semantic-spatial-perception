#!/usr/bin/env python3
"""
Interactive menu-driven demo of Gemini Robotics-ER capabilities.

This script provides a selection menu for testing different features:
1. Object detection with affordances
2. Spatial reasoning and relationships
3. Task decomposition
4. Interaction point detection (with visualization)
5. Trajectory planning
6. Full integration with world model

Based on: https://ai.google.dev/gemini-api/docs/robotics-overview
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_client():
    """Initialize Gemini client and capture scene."""
    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY not set. Please set it in .env file")
        return None, None, None, None

    from src.perception import GeminiRoboticsClient

    print("Initializing Gemini Robotics client...")
    client = GeminiRoboticsClient(
        api_key=api_key,
        model_name="auto",  # Auto-select robotics model if available
        default_temperature=0.5,
        thinking_budget=0  # Disable thinking for speed
    )
    print()

    # Try to get real camera feed
    color_frame = None
    depth_frame = None
    camera_intrinsics = None

    try:
        from src.camera import RealSenseCamera

        print("Attempting to capture scene with RealSense...")
        camera = RealSenseCamera(width=640, height=480, auto_start=True)
        camera_intrinsics = camera.get_camera_intrinsics()
        color_frame, depth_frame = camera.get_aligned_frames()
        camera.stop()
        print("✓ Scene captured from RealSense")
        print()
    except Exception as e:
        print(f"⚠ RealSense not available: {e}")
        print()

        # Fallback to test image
        test_image_path = input("Enter path to test image (or press Enter to skip): ").strip()
        if test_image_path and Path(test_image_path).exists():
            from PIL import Image

            img = Image.open(test_image_path)
            color_frame = np.array(img)
            print(f"✓ Loaded test image: {test_image_path}")
            print()

    if color_frame is None:
        print("⚠ No image available. Some examples will be skipped.")
        print()

    return client, color_frame, depth_frame, camera_intrinsics


def demo_object_detection(client, color_frame):
    """Example 1: Object Detection with Affordances."""
    print("=" * 80)
    print("Object Detection with Affordances")
    print("=" * 80)
    print()

    if color_frame is None:
        print("⚠ No image available. Skipping.")
        return None

    print("Running VLM object detection...")
    result = client.detect_objects(
        color_frame,
        query="""Detect all objects in the scene with detailed analysis.

For each object, infer from visual observation:
- Precise label (include color, material if visible)
- Position as [y, x] in 0-1000 scale
- Confidence score
- Physical properties (color, size, material, state)
- Affordances: Can it be grasped, opened, poured, used as support, etc.?

Analyze the visual appearance to determine affordances, not pre-defined categories.""",
        return_json=True
    )

    print(f"✓ Detection completed in {result.processing_time:.2f}s")
    print(f"✓ Detected {len(result.objects)} objects")
    print()

    if result.objects:
        print("Detected Objects:")
        for i, obj in enumerate(result.objects[:5], 1):
            print(f"\n{i}. {obj.get('label', 'Unknown')}")
            print(f"   Position: {obj.get('position', [])}")
            print(f"   Confidence: {obj.get('confidence', 0):.2f}")
            props = obj.get('properties', {})
            if props:
                print(f"   Properties: {props}")
        if len(result.objects) > 5:
            print(f"\n... and {len(result.objects) - 5} more objects")
    else:
        print("⚠ No objects detected")

    print()
    input("Press Enter to continue...")
    return result


def demo_spatial_reasoning(client, color_frame, result):
    """Example 2: Spatial Reasoning and Relationships."""
    print("\n" + "=" * 80)
    print("Spatial Reasoning and Relationships")
    print("=" * 80)
    print()

    if color_frame is None:
        print("⚠ No image available. Skipping.")
        return

    print("Analyzing spatial relationships...")
    spatial_result = client.analyze_spatial_relationships(
        color_frame,
        query="""Analyze the spatial layout of the scene.

Identify:
1. Spatial relationships between objects (left-of, right-of, on-top-of, near, etc.)
2. Which objects are closest to the camera/robot
3. Which objects are easy to reach vs. obstructed
4. Recommendations for manipulation planning

Be specific about distances and positions."""
    )

    print(f"✓ Analysis completed in {spatial_result.processing_time:.2f}s")
    print()
    print("Spatial Reasoning:")
    print(spatial_result.reasoning)
    print()

    if spatial_result.relationships:
        print("Relationships:")
        for rel in spatial_result.relationships[:5]:
            print(f"  • {rel}")
    print()

    if spatial_result.recommendations:
        print("Recommendations:")
        for rec in spatial_result.recommendations:
            print(f"  • {rec}")
    print()
    input("Press Enter to continue...")


def demo_task_decomposition(client, color_frame):
    """Example 3: Task Decomposition."""
    print("\n" + "=" * 80)
    print("Task Decomposition")
    print("=" * 80)
    print()

    # Get task from user
    print("Example tasks:")
    print("  - Pick up the object closest to the robot")
    print("  - Organize items by color")
    print("  - Clear the table")
    print()
    task = input("Enter a task: ").strip()
    if not task:
        task = "Pick up the object closest to the robot and place it in a safe location"

    print(f"\nTask: {task}")
    print()
    print("Decomposing task into subtasks...")

    decomp_result = client.decompose_task(
        task_description=task,
        image=color_frame,
        available_actions=["navigate", "grasp", "place", "open", "close", "push", "pull"]
    )

    print(f"✓ Decomposition completed in {decomp_result.processing_time:.2f}s")
    print(f"✓ Complexity: {decomp_result.estimated_complexity}")
    print()

    print("Reasoning:")
    print(decomp_result.reasoning)
    print()

    if decomp_result.subtasks:
        print("Subtasks:")
        for subtask in decomp_result.subtasks:
            print(f"  {subtask.get('id', '?')}. {subtask.get('action', '?')}: "
                  f"{subtask.get('description', '')}")
    print()

    if decomp_result.dependencies:
        print("Dependencies:")
        for dep in decomp_result.dependencies:
            print(f"  Task {dep[0]} → Task {dep[1]}")
    print()
    input("Press Enter to continue...")


def demo_interaction_point(client, color_frame, result):
    """Example 4: Interaction Point Detection with Visualization."""
    print("\n" + "=" * 80)
    print("Interaction Point Detection")
    print("=" * 80)
    print()

    if color_frame is None:
        print("⚠ No image available. Skipping.")
        return

    if not result or not result.objects:
        print("⚠ No objects detected. Running detection first...")
        result = demo_object_detection(client, color_frame)
        if not result or not result.objects:
            print("⚠ Still no objects detected. Skipping.")
            return

    # Get object from detected objects
    print("Detected objects:")
    for i, obj in enumerate(result.objects[:5], 1):
        print(f"  {i}. {obj.get('label', 'Unknown')}")
    print()

    # Get object selection
    selection = input(f"Select object (1-{min(5, len(result.objects))}): ").strip()
    if selection.isdigit() and 1 <= int(selection) <= min(5, len(result.objects)):
        selected_obj = result.objects[int(selection) - 1]
    else:
        selected_obj = result.objects[0]

    object_label = selected_obj.get('label', 'object')
    print(f"\nSelected: {object_label}")
    print()

    # Get action type
    print("Action type:")
    print("  1. grasp")
    print("  2. push")
    print("  3. pull")
    print("  4. open")
    print("  5. custom")
    action_choice = input("Select action (1-5): ").strip()

    action_map = {"1": "grasp", "2": "push", "3": "pull", "4": "open"}
    if action_choice in action_map:
        action = action_map[action_choice]
    elif action_choice == "5":
        action = input("Enter custom action: ").strip() or "grasp"
    else:
        action = "grasp"

    # Get optional task context
    task_context = input(f"Optional task context (or Enter to skip): ").strip() or None

    print()
    print(f"Detecting interaction point for '{action}' on '{object_label}'...")

    interaction_result = client.detect_interaction_point(
        color_frame,
        object_label=object_label,
        action=action,
        task_context=task_context
    )

    print(f"✓ Detection completed in {interaction_result.processing_time:.2f}s")
    print()

    print(f"Interaction Point:")
    print(f"  Position: {interaction_result.interaction_point} (y, x in 0-1000 scale)")
    print(f"  Type: {interaction_result.interaction_type}")
    print(f"  Confidence: {interaction_result.confidence:.2f}")
    print()

    print(f"Reasoning:")
    print(f"  {interaction_result.reasoning}")
    print()

    if interaction_result.alternative_points:
        print(f"Alternative points:")
        for alt in interaction_result.alternative_points[:3]:
            print(f"  • {alt.get('point', [])} - {alt.get('reason', '')}")
        print()

    # Visualize interaction point
    print("Creating visualization...")
    try:
        import cv2

        # Convert to BGR for OpenCV
        vis_image = cv2.cvtColor(color_frame.copy(), cv2.COLOR_RGB2BGR)
        height, width = vis_image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        y_norm, x_norm = interaction_result.interaction_point
        pixel_x = int((x_norm / 1000.0) * width)
        pixel_y = int((y_norm / 1000.0) * height)

        # Draw crosshair at interaction point
        color = (0, 255, 0)  # Green
        thickness = 2
        size = 20

        # Crosshair
        cv2.line(vis_image, (pixel_x - size, pixel_y), (pixel_x + size, pixel_y), color, thickness)
        cv2.line(vis_image, (pixel_x, pixel_y - size), (pixel_x, pixel_y + size), color, thickness)

        # Circles
        cv2.circle(vis_image, (pixel_x, pixel_y), 10, color, thickness)
        cv2.circle(vis_image, (pixel_x, pixel_y), 30, color, 1)

        # Label
        label_text = f"{action.upper()}: {object_label}"
        cv2.putText(vis_image, label_text, (pixel_x + 40, pixel_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Confidence
        conf_text = f"Confidence: {interaction_result.confidence:.2f}"
        cv2.putText(vis_image, conf_text, (pixel_x + 40, pixel_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw alternative points
        if interaction_result.alternative_points:
            for i, alt in enumerate(interaction_result.alternative_points[:3]):
                alt_point = alt.get('point', [500, 500])
                alt_y_norm, alt_x_norm = alt_point
                alt_pixel_x = int((alt_x_norm / 1000.0) * width)
                alt_pixel_y = int((alt_y_norm / 1000.0) * height)

                # Draw smaller circles for alternatives
                alt_color = (0, 165, 255)  # Orange
                cv2.circle(vis_image, (alt_pixel_x, alt_pixel_y), 8, alt_color, 2)
                cv2.circle(vis_image, (alt_pixel_x, alt_pixel_y), 15, alt_color, 1)
                cv2.putText(vis_image, f"Alt {i+1}", (alt_pixel_x + 20, alt_pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, alt_color, 1)

        # Display
        cv2.imshow("Interaction Point Detection", vis_image)
        print("✓ Visualization displayed (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save option
        save = input("\nSave visualization? (y/n): ").strip().lower()
        if save == 'y':
            output_path = f"interaction_point_{object_label.replace(' ', '_')}_{action}.png"
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Saved to {output_path}")

    except ImportError:
        print("⚠ OpenCV not available for visualization")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")

    print()
    input("Press Enter to continue...")


def demo_trajectory_planning(client, color_frame, result):
    """Example 5: Trajectory Planning."""
    print("\n" + "=" * 80)
    print("Trajectory Planning")
    print("=" * 80)
    print()

    if color_frame is None:
        print("⚠ No image available. Skipping.")
        return

    if not result or not result.objects:
        print("⚠ No objects detected. Skipping.")
        return

    # Get target from detected objects
    target_obj = result.objects[0]
    target_label = target_obj.get('label', 'object')
    target_pos = target_obj.get('position', [500, 500])

    print(f"Planning trajectory to: {target_label}")
    print(f"Target position: {target_pos}")
    print()

    traj_result = client.plan_trajectory(
        color_frame,
        start_point=[250, 250],  # Assume robot starts at bottom-left
        end_point=target_pos,
        query=f"Plan a safe path to reach {target_label}, avoiding obstacles"
    )

    print(f"✓ Planning completed in {traj_result.processing_time:.2f}s")
    print()
    print("Path Description:")
    print(traj_result.description)
    print()

    if traj_result.waypoints:
        print("Waypoints:")
        for wp in traj_result.waypoints:
            print(f"  {wp.get('id', '?')}. {wp.get('label', '?')}: "
                  f"{wp.get('position', [])}")
    print()
    input("Press Enter to continue...")


def demo_world_model_integration(client, color_frame, depth_frame, camera_intrinsics):
    """Example 6: Integration with World Model."""
    print("\n" + "=" * 80)
    print("Integration with World Model")
    print("=" * 80)
    print()

    if color_frame is None:
        print("⚠ No image available. Skipping.")
        return

    print("Creating VLM detector and world state...")

    from src.perception import VLMObjectDetector
    from src.world_model import WorldState

    detector = VLMObjectDetector(
        gemini_client=client,
        confidence_threshold=0.5
    )

    world = WorldState()

    print("✓ Detector and world state initialized")
    print()

    print("Detecting objects with full integration...")
    detected_objects = detector.detect(
        color_image=color_frame,
        depth_image=depth_frame,
        camera_intrinsics=camera_intrinsics,
        task_context="General scene understanding for manipulation"
    )

    print(f"✓ Detected {len(detected_objects)} objects")
    print()

    # Update world state
    world.update(detected_objects)

    print(f"✓ World state updated")
    print(f"  - Objects tracked: {world.get_object_count()}")
    print(f"  - Spatial relationships: {len(world.get_all_relationships())}")
    print()

    # Show detected objects with full details
    if detected_objects:
        print("Detected Objects (full integration):")
        for obj in detected_objects[:3]:
            print(f"\n  • {obj.object_id}")
            print(f"    Type: {obj.object_type}")
            print(f"    Position: {obj.position}")
            print(f"    Confidence: {obj.confidence:.2f}")
            print(f"    Affordances: {', '.join(obj.affordances)}")
            if obj.color:
                print(f"    Color: {obj.color}")
            if obj.properties.get('vlm_observed'):
                print(f"    Source: VLM-observed (dynamic)")
    print()

    # Show spatial relationships
    relationships = world.get_all_relationships()
    if relationships:
        print("Spatial Relationships:")
        for rel in relationships[:5]:
            print(f"  • {rel.to_predicate()}")
        if len(relationships) > 5:
            print(f"  ... and {len(relationships) - 5} more")
    print()

    # Show statistics
    stats = detector.get_statistics()
    print("📊 Detection Statistics:")
    print(f"  • Total detections: {stats['total_detections']}")
    print(f"  • Avg objects/frame: {stats['avg_objects_per_frame']:.1f}")
    print(f"  • Avg processing time: {stats['avg_processing_time']:.2f}s")
    print()
    input("Press Enter to continue...")


def show_menu():
    """Display main menu."""
    print("\n" + "=" * 80)
    print("GEMINI ROBOTICS CAPABILITIES MENU")
    print("=" * 80)
    print()
    print("Select a capability to test:")
    print()
    print("  1. Object Detection with Affordances")
    print("  2. Spatial Reasoning and Relationships")
    print("  3. Task Decomposition")
    print("  4. Interaction Point Detection (with visualization)")
    print("  5. Trajectory Planning")
    print("  6. Full Integration with World Model")
    print()
    print("  7. Run All Examples")
    print("  0. Exit")
    print()
    print("=" * 80)


def main():
    """Main interactive loop."""
    print("=" * 80)
    print("Gemini Robotics-ER Interactive Demo")
    print("=" * 80)
    print()

    # Setup
    client, color_frame, depth_frame, camera_intrinsics = setup_client()
    if client is None:
        return

    # Cache detection result for reuse
    detection_result = None

    while True:
        show_menu()
        choice = input("Enter choice (0-7): ").strip()

        if choice == "0":
            print("\nExiting demo. Goodbye!")
            break

        elif choice == "1":
            detection_result = demo_object_detection(client, color_frame)

        elif choice == "2":
            demo_spatial_reasoning(client, color_frame, detection_result)

        elif choice == "3":
            demo_task_decomposition(client, color_frame)

        elif choice == "4":
            demo_interaction_point(client, color_frame, detection_result)

        elif choice == "5":
            demo_trajectory_planning(client, color_frame, detection_result)

        elif choice == "6":
            demo_world_model_integration(client, color_frame, depth_frame, camera_intrinsics)

        elif choice == "7":
            print("\n" + "=" * 80)
            print("Running All Examples")
            print("=" * 80)
            print()
            detection_result = demo_object_detection(client, color_frame)
            demo_spatial_reasoning(client, color_frame, detection_result)
            demo_task_decomposition(client, color_frame)
            demo_interaction_point(client, color_frame, detection_result)
            demo_trajectory_planning(client, color_frame, detection_result)
            demo_world_model_integration(client, color_frame, depth_frame, camera_intrinsics)
            print("\n✓ All examples completed!")

        else:
            print("\n⚠ Invalid choice. Please select 0-7.")

    print()


if __name__ == "__main__":
    main()
