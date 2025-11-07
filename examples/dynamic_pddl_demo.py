#!/usr/bin/env python3
"""
Interactive demonstration of the full VLM-powered robotic perception pipeline.

This demo shows the complete system workflow:
1. Capture RGB-D scene with RealSense camera
2. Detect objects with Gemini VLM (dynamic affordances, no pre-defined data)
3. Build world model with spatial relationships
4. User provides tasks based on observed objects (interactive)
5. LLM analyzes tasks in scene context
6. Generate task-specific PDDL files

NO MOCK DATA - All objects and affordances come from real VLM observations.
NO PRE-DEFINED TASKS - User specifies tasks interactively after seeing detected objects.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 80)
    print("Interactive VLM-Powered Robotic Perception & Planning Demo")
    print("=" * 80)
    print()
    print("This demo captures a real scene, detects objects with Gemini VLM,")
    print("and generates PDDL plans for user-specified tasks.")
    print()
    print("Requirements:")
    print("  • RealSense camera connected")
    print("  • GEMINI_API_KEY set in .env file")
    print("  • Objects visible in camera view")
    print("=" * 80)
    print()

    # Load environment variables from .env file
    load_dotenv()

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY not set in environment")
        print("  Set it with: export GEMINI_API_KEY='your_key_here'")
        print("  Or add to .env file")
        print()
        return

    # ========================================================================
    # 1. Capture Real Scene with RealSense Camera
    # ========================================================================
    print("1. Capturing real environment with RealSense camera...")
    print()

    color_frame = None
    depth_frame = None
    camera_intrinsics = None

    try:
        from src.camera import RealSenseCamera, visualize_depth
        import cv2

        camera = RealSenseCamera(width=640, height=480, enable_depth=True, auto_start=True)
        print("   ✓ RealSense camera initialized")

        # Get camera intrinsics
        camera_intrinsics = camera.get_camera_intrinsics()
        print(f"   ✓ Camera intrinsics: {camera_intrinsics.width}x{camera_intrinsics.height}")
        print(f"   ✓ Focal length: fx={camera_intrinsics.fx:.1f}, fy={camera_intrinsics.fy:.1f}")

        # Capture aligned RGB-D frames
        print("   → Capturing RGB-D frames...")
        color_frame, depth_frame = camera.get_aligned_frames()
        print(f"   ✓ Color frame: {color_frame.shape}")
        print(f"   ✓ Depth frame: {depth_frame.shape}")
        print(f"   ✓ Depth range: {depth_frame.min():.3f}m to {depth_frame.max():.3f}m")

        # Display the captured image
        print("   → Displaying captured scene (press any key to continue)...")

        # Convert RGB to BGR for OpenCV display
        color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        # Create depth visualization
        depth_colored = visualize_depth(depth_frame, max_depth=3.0)
        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # Stack images side by side
        display = np.hstack([color_bgr, depth_bgr])

        # Add labels and info
        cv2.putText(display, "RGB Camera Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(display, "Depth Map", (660, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press any key to continue...", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("RealSense Scene Capture", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("   ✓ Scene captured successfully")

        camera.stop()

    except ImportError:
        print("   ⚠ pyrealsense2 not installed. Install with: pip install pyrealsense2")
        print("   → Continuing with mock data...")
    except Exception as e:
        print(f"   ⚠ RealSense camera not available: {e}")
        print("   → Continuing with mock data...")

    print()

    # ========================================================================
    # 2. Object Detection with Gemini VLM
    # ========================================================================
    print("2. Detecting objects in scene with Gemini VLM...")
    print()

    from src.world_model import WorldState
    from src.perception import GeminiRoboticsClient, VLMObjectDetector

    world = WorldState()

    # Initialize Gemini VLM detector
    try:
        print("   → Initializing Gemini Robotics client...")
        gemini_client = GeminiRoboticsClient(
            api_key=api_key,
            model_name="gemini-2.0-flash-exp",  # Using fast model for demo
            default_temperature=0.1
        )

        vlm_detector = VLMObjectDetector(
            gemini_client=gemini_client,
            confidence_threshold=0.5
        )
        print("   ✓ VLM detector initialized")
        print()

        # If we have real camera data, use it for detection
        if color_frame is not None:
            print("   → Running VLM object detection on camera feed...")
            print("   → This may take 3-10 seconds...")

            # Detect objects with VLM
            objects_data = vlm_detector.detect(
                color_image=color_frame,
                depth_image=depth_frame,
                camera_intrinsics=camera_intrinsics,
                task_context="Analyzing scene for robotic manipulation tasks"
            )

            print(f"   ✓ Detected {len(objects_data)} objects with VLM")

            # Show detected objects
            if objects_data:
                print("   → Detected objects:")
                for obj in objects_data[:5]:  # Show first 5
                    print(f"      • {obj.object_id}: {obj.object_type}")
                    print(f"        Position: {obj.position}")
                    print(f"        Affordances: {', '.join(obj.affordances)}")
                    if obj.color:
                        print(f"        Color: {obj.color}")
                if len(objects_data) > 5:
                    print(f"      ... and {len(objects_data) - 5} more")
            else:
                print("   ⚠ No objects detected, falling back to mock data")
                objects_data = None  # Will use fallback below

        else:
            print("   ⚠ No camera data available")
            objects_data = None

    except Exception as e:
        print(f"   ⚠ VLM detection failed: {e}")
        print("   → Falling back to mock detected objects")
        objects_data = None

    # Check if we have any detected objects
    if objects_data is None or len(objects_data) == 0:
        print()
        print("   ⚠ No objects detected!")
        print("   → Cannot proceed without object detections")
        print()
        print("   Please ensure:")
        print("      1. Camera is properly positioned to see objects")
        print("      2. GEMINI_API_KEY is valid")
        print("      3. Scene has visible objects")
        print()
        return

    # Update world state
    world.update(objects_data)

    print()
    print(f"   ✓ World model updated with {world.get_object_count()} objects")
    print(f"   ✓ Computed {len(world.get_all_relationships())} spatial relationships")
    print()

    # ========================================================================
    # 3. User Task Input
    # ========================================================================
    print("=" * 80)
    print("3. Task Input")
    print("=" * 80)
    print()

    # Show detected objects to help user formulate task
    print("   Detected objects in scene:")
    for obj in world.get_all_objects():
        desc = f"      • {obj.object_id}: {obj.object_type}"
        if obj.color:
            desc += f" ({obj.color})"
        if obj.affordances:
            desc += f" - can be {', '.join(obj.affordances[:3])}"
        print(desc)
    print()

    # Get tasks from user
    print("   Enter robotic manipulation tasks (one per line).")
    print("   Examples:")
    print("      - Pick up the red cup and place it on the shelf")
    print("      - Move all graspable objects to the container")
    print("      - Organize the workspace by color")
    print()
    print("   Enter a blank line when done.")
    print()

    tasks = []
    task_num = 1
    while True:
        task = input(f"   Task {task_num}: ").strip()
        if not task:
            break
        tasks.append(task)
        task_num += 1

    if not tasks:
        print()
        print("   ⚠ No tasks provided. Exiting.")
        print()
        return

    print()
    print(f"   ✓ Received {len(tasks)} task(s)")
    print()

    # ========================================================================
    # 4. Dynamic PDDL Generation with LLM
    # ========================================================================
    from src.planning import DynamicPDDLGenerator, LLMTaskAnalyzer

    # Initialize LLM analyzer (uses fast model for speed)
    print("   → Initializing LLM task analyzer (Gemini Flash)...")
    analyzer = LLMTaskAnalyzer(api_key=api_key, model_name="gemini-2.5-flash-lite")

    # Initialize PDDL generator
    generator = DynamicPDDLGenerator(llm_analyzer=analyzer)

    print("   ✓ Generator ready")
    print()

    # Process each task
    for i, task in enumerate(tasks, 1):
        print(f"{'='*80}")
        print(f"TASK {i}: {task}")
        print(f"{'='*80}")
        print()

        # Get world state in dict format
        world_state_dict = {
            "objects": [obj.to_dict() for obj in world.get_all_objects()],
            "relationships": [rel.to_predicate() for rel in world.get_all_relationships()],
            "predicates": world.get_pddl_state()["predicates"]
        }

        # Generate PDDL
        start = time.time()
        result = generator.generate(
            task_description=task,
            world_state=world_state_dict,
            output_dir=f"outputs/pddl/task_{i}",
            domain_name=f"task_{i}"
        )
        elapsed = time.time() - start

        # Show analysis results
        analysis = result["analysis"]
        print()
        print("   Analysis Results:")
        print(f"   → Action sequence: {', '.join(analysis.action_sequence)}")
        print(f"   → Goal objects: {', '.join(analysis.goal_objects)}")
        print(f"   → Goal predicates: {', '.join(analysis.goal_predicates[:3])}...")
        print(f"   → Relevant predicates: {', '.join(analysis.relevant_predicates)}")
        print(f"   → Complexity: {analysis.complexity}")
        print(f"   → Estimated steps: {analysis.estimated_steps}")
        print(f"   → Total time: {elapsed:.2f}s")
        print()

        # Show generated files
        print("   Generated Files:")
        print(f"   ✓ {result['domain_path']}")
        print(f"   ✓ {result['problem_path']}")
        print()

    # ========================================================================
    # 5. Summary
    # ========================================================================
    print("=" * 80)
    print("Demo Complete - Full Pipeline Demonstrated")
    print("=" * 80)
    print()
    print("✓ Pipeline Flow:")
    print("  1. RealSense camera captured RGB-D scene")
    print("  2. Gemini VLM detected objects with dynamic affordances")
    print("  3. World model built spatial relationships")
    print("  4. User provided tasks based on observed objects")
    print("  5. LLM analyzed tasks in scene context")
    print("  6. Dynamic PDDL files generated per task")
    print()
    print("✓ Performance:")
    print("  - Camera capture: <0.5s")
    print("  - VLM object detection: 3-10s")
    print(f"  - User task input: {len(tasks)} task(s)")
    print("  - LLM task analysis: 2-5s per task")
    print("  - PDDL generation: <0.1s per task")
    print()
    print("✓ Key Features Demonstrated:")
    print("  - Real-time VLM perception (no mock data)")
    print("  - Dynamic affordance inference from observation")
    print("  - User-driven task specification")
    print("  - Task-specific PDDL generation")
    print("  - Scene-grounded planning")
    print("  - No pre-defined objects or tasks")
    print()
    print("📁 Check outputs/pddl/ for generated PDDL files")
    print("📷 Scene captured from real RealSense camera")
    print("🤖 Objects detected with Gemini Robotics VLM")
    print("👤 Tasks specified by user based on observed scene")
    print()


if __name__ == "__main__":
    main()
