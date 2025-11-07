#!/usr/bin/env python3
"""
Simple test script for Gemini detection to debug API responses.

This script helps identify issues with:
- API key configuration
- Model availability
- Response format
- JSON parsing
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_detection():
    """Test basic Gemini detection with a simple synthetic image."""
    print("=" * 80)
    print("Gemini Detection Test")
    print("=" * 80)
    print()

    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        print("   Set it in .env file or environment")
        return False

    print(f"✓ API key found (length: {len(api_key)})")
    print()

    # Import modules
    try:
        from src.perception import GeminiRoboticsClient
        print("✓ Modules imported successfully")
        print()
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False

    # Create test image (simple colored rectangle)
    print("Creating test image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red rectangle (simulating an object)
    test_image[100:200, 200:400] = [255, 0, 0]  # Red
    # Add a blue circle area (simulating another object)
    test_image[250:350, 300:450] = [0, 0, 255]  # Blue
    print("✓ Test image created (640x480 with 2 colored regions)")
    print()

    # Initialize client
    try:
        print("Initializing Gemini client...")
        client = GeminiRoboticsClient(
            api_key=api_key,
            model_name="gemini-2.0-flash-exp",
            default_temperature=0.1
        )
        print(f"✓ Client initialized with model: {client.model_name}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run detection
    try:
        print("Running object detection...")
        print("(This may take 5-15 seconds...)")
        print()

        result = client.detect_objects(
            test_image,
            query="Detect colored regions in the image. For each region provide its approximate position in [y, x] format (0-1000 scale) and color.",
            return_json=True
        )

        print(f"✓ Detection completed in {result.processing_time:.2f}s")
        print()

        # Show results
        print("=" * 80)
        print("Detection Results")
        print("=" * 80)
        print()

        print(f"Model used: {result.model_used}")
        print(f"Objects detected: {len(result.objects)}")
        print()

        if result.objects:
            print("Detected objects:")
            for i, obj in enumerate(result.objects, 1):
                print(f"\n{i}. {obj}")
        else:
            print("⚠ No objects detected")
            print("\nRaw response:")
            print(result.raw_response[:500])

        print()
        return True

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_camera():
    """Test with real camera if available."""
    print("\n" + "=" * 80)
    print("Camera Test (Optional)")
    print("=" * 80)
    print()

    try:
        from src.camera import RealSenseCamera
        from src.perception import GeminiRoboticsClient, VLMObjectDetector
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠ Skipping camera test (no API key)")
            return

        print("Attempting to use RealSense camera...")
        camera = RealSenseCamera(width=640, height=480, auto_start=True)

        print("✓ Camera initialized")
        print("Capturing frame...")

        color, depth = camera.get_aligned_frames()
        intrinsics = camera.get_camera_intrinsics()
        camera.stop()

        print(f"✓ Frame captured: {color.shape}")
        print()

        # Run detection
        print("Running VLM detection on camera frame...")
        client = GeminiRoboticsClient(api_key=api_key)
        detector = VLMObjectDetector(client, confidence_threshold=0.3)

        objects = detector.detect(
            color,
            depth,
            intrinsics,
            task_context="Detect all objects in the scene"
        )

        print(f"✓ Detected {len(objects)} objects")

        if objects:
            print("\nObjects:")
            for obj in objects[:5]:
                print(f"  • {obj.object_id}: {obj.object_type}")
                print(f"    Position: {obj.position}")
                print(f"    Affordances: {obj.affordances}")

        print()

    except ImportError:
        print("⚠ RealSense not available, skipping camera test")
    except Exception as e:
        print(f"⚠ Camera test failed: {e}")


if __name__ == "__main__":
    print("\nThis script tests Gemini API detection with debug output.\n")

    success = test_basic_detection()

    if success:
        print("\n✅ Basic detection test PASSED")

        # Optionally test with camera
        response = input("\nTest with RealSense camera? (y/n): ").strip().lower()
        if response == 'y':
            test_with_camera()
    else:
        print("\n❌ Basic detection test FAILED")
        print("\nTroubleshooting:")
        print("1. Check GEMINI_API_KEY is set correctly in .env")
        print("2. Verify you have access to gemini-2.0-flash-exp model")
        print("3. Check your internet connection")
        print("4. Review any error messages above")

    print()
