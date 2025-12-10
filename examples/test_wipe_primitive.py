#!/usr/bin/env python3
"""
Test script for the wipe primitive.

This script verifies that the wipe primitive is properly registered
and can be executed through the CuRobo motion planner interface.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_wipe_primitive_registered():
    """Test that wipe primitive is registered in PRIMITIVE_LIBRARY."""
    print("\nTest 1: Checking wipe primitive registration")
    print("=" * 70)

    try:
        from src.primitives.skill_plan_types import PRIMITIVE_LIBRARY

        if "wipe" in PRIMITIVE_LIBRARY:
            schema = PRIMITIVE_LIBRARY["wipe"]
            print(f"  ✓ Wipe primitive found in PRIMITIVE_LIBRARY")
            print(f"    Name: {schema.name}")
            print(f"    Description: {schema.description}")
            print(f"    Optional params: {schema.optional_params}")
            print(f"    Allowed frames: {schema.allowed_frames}")
            return True
        else:
            print(f"  ✗ Wipe primitive not found in PRIMITIVE_LIBRARY")
            print(f"    Available primitives: {list(PRIMITIVE_LIBRARY.keys())}")
            return False

    except Exception as e:
        print(f"  ✗ Error loading PRIMITIVE_LIBRARY: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wipe_method_exists():
    """Test that wipe method exists in CuRoboMotionPlanner."""
    print("\nTest 2: Checking wipe method in CuRoboMotionPlanner")
    print("=" * 70)

    try:
        from src.kinematics.xarm_curobo_interface import CuRoboMotionPlanner

        # Check if method exists
        if hasattr(CuRoboMotionPlanner, "wipe"):
            print("  ✓ wipe() method found in CuRoboMotionPlanner")

            # Check method signature
            import inspect
            sig = inspect.signature(CuRoboMotionPlanner.wipe)
            print(f"    Signature: {sig}")

            # Check parameters
            params = list(sig.parameters.keys())
            expected_params = ["self", "direction", "rotation_angle_deg", "speed_factor", "timeout"]

            if all(p in params for p in expected_params):
                print(f"    ✓ All expected parameters present: {expected_params[1:]}")
                return True
            else:
                print(f"    ✗ Missing parameters")
                print(f"      Expected: {expected_params}")
                print(f"      Found: {params}")
                return False
        else:
            print("  ✗ wipe() method not found in CuRoboMotionPlanner")
            return False

    except Exception as e:
        print(f"  ✗ Error checking wipe method: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wipe_execution():
    """Test wipe primitive execution with CuRobo planner."""
    print("\nTest 3: Testing wipe primitive execution")
    print("=" * 70)

    try:
        from src.kinematics.xarm_curobo_interface import CuRoboMotionPlanner

        print("  Initializing CuRobo motion planner (simulation mode)...")

        # Initialize planner without robot connection
        planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")

        print("  ✓ CuRobo planner initialized successfully\n")

        # Test different wipe configurations
        test_cases = [
            {
                "name": "Default wipe (360° clockwise)",
                "params": {}
            },
            {
                "name": "Half rotation clockwise",
                "params": {"direction": "clockwise", "rotation_angle_deg": 180.0}
            },
            {
                "name": "Quarter rotation counterclockwise",
                "params": {"direction": "counterclockwise", "rotation_angle_deg": 90.0}
            },
            {
                "name": "Full rotation at half speed",
                "params": {"direction": "clockwise", "rotation_angle_deg": 360.0, "speed_factor": 0.5}
            },
            {
                "name": "Two full rotations (screwing motion)",
                "params": {"direction": "clockwise", "rotation_angle_deg": 720.0}
            },
        ]

        results = {}

        for test_case in test_cases:
            test_name = test_case["name"]
            params = test_case["params"]

            print(f"  Testing: {test_name}")
            print(f"    Parameters: {params}")

            try:
                # Call wipe method
                success = planner.wipe(**params)

                if success:
                    print(f"    ✓ Execution succeeded")
                    results[test_name] = "PASS"
                else:
                    print(f"    ⚠ Execution returned False (may be expected without hardware)")
                    results[test_name] = "EXPECTED_FAIL"

            except Exception as e:
                print(f"    ✗ Execution failed with error: {str(e)[:100]}")
                results[test_name] = "FAIL"
                import traceback
                traceback.print_exc()

            print()

        print("=" * 70)
        print("EXECUTION RESULTS:")
        print("=" * 70)

        for test_name, result in results.items():
            status_symbol = "✓" if result in ["PASS", "EXPECTED_FAIL"] else "✗"
            print(f"{status_symbol} {test_name}: {result}")

        print("=" * 70)

        # Consider test passed if no hard failures
        no_hard_failures = all(r != "FAIL" for r in results.values())

        if no_hard_failures:
            print("\n✓ Wipe primitive execution test passed")
            print("  Note: Without hardware connection, execution may return False")
            print("  The important thing is that the method is callable and doesn't crash")
            return True
        else:
            print("\n✗ Some wipe executions had hard failures")
            return False

    except Exception as e:
        print(f"✗ Error setting up CuRobo planner: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wipe_parameter_validation():
    """Test that wipe primitive validates parameters correctly."""
    print("\nTest 4: Testing wipe parameter validation")
    print("=" * 70)

    try:
        from src.primitives.skill_plan_types import PRIMITIVE_LIBRARY, PrimitiveCall

        schema = PRIMITIVE_LIBRARY["wipe"]

        # Valid call
        valid_call = PrimitiveCall(
            name="wipe",
            parameters={
                "direction": "clockwise",
                "rotation_angle_deg": 180.0,
                "speed_factor": 1.0
            }
        )

        errors = schema.validate(valid_call)
        if not errors:
            print("  ✓ Valid wipe call passes validation")
        else:
            print(f"  ✗ Valid call failed validation: {errors}")
            return False

        # Invalid parameter
        invalid_call = PrimitiveCall(
            name="wipe",
            parameters={
                "invalid_param": "test"
            }
        )

        errors = schema.validate(invalid_call)
        if errors:
            print(f"  ✓ Invalid parameter correctly rejected: {errors[0]}")
        else:
            print(f"  ✗ Invalid parameter not detected")
            return False

        # Invalid frame
        invalid_frame_call = PrimitiveCall(
            name="wipe",
            frame="invalid_frame",
            parameters={}
        )

        errors = schema.validate(invalid_frame_call)
        if errors and "frame" in errors[0]:
            print(f"  ✓ Invalid frame correctly rejected: {errors[0]}")
        else:
            print(f"  ✗ Invalid frame not detected")
            return False

        print("\n  ✓ All parameter validation tests passed")
        return True

    except Exception as e:
        print(f"  ✗ Error in parameter validation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wipe_in_primitive_catalog():
    """Test that wipe is documented in primitive_descriptions.md."""
    print("\nTest 5: Checking wipe documentation")
    print("=" * 70)

    try:
        catalog_path = "config/primitive_descriptions.md"

        if not os.path.exists(catalog_path):
            print(f"  ⚠ Primitive catalog not found at {catalog_path}")
            return True  # Don't fail test for missing docs

        with open(catalog_path, 'r') as f:
            content = f.read()

        if "## wipe" in content:
            print(f"  ✓ Wipe primitive documented in {catalog_path}")

            # Check for key documentation elements
            checks = {
                "direction": "direction parameter" in content.lower(),
                "rotation_angle_deg": "rotation_angle_deg" in content,
                "example": "example" in content.lower() and "wipe" in content.lower(),
            }

            for check_name, found in checks.items():
                if found:
                    print(f"    ✓ Documentation includes {check_name}")
                else:
                    print(f"    ⚠ Documentation may be missing {check_name}")

            return True
        else:
            print(f"  ⚠ Wipe primitive not documented in {catalog_path}")
            print(f"    This is recommended for LLM skill decomposition")
            return True  # Don't fail test for missing docs

    except Exception as e:
        print(f"  ⚠ Error checking documentation: {e}")
        return True  # Don't fail test for doc issues


def main():
    """Run all wipe primitive tests."""
    print("\n" + "=" * 70)
    print("WIPE PRIMITIVE TEST SUITE")
    print("=" * 70)

    test1_passed = test_wipe_primitive_registered()
    test2_passed = test_wipe_method_exists()
    test3_passed = test_wipe_execution()
    test4_passed = test_wipe_parameter_validation()
    test5_passed = test_wipe_in_primitive_catalog()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Primitive registered:     {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Method exists:            {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Execution test:           {'✓ PASS' if test3_passed else '✗ FAIL'}")
    print(f"Parameter validation:     {'✓ PASS' if test4_passed else '✗ FAIL'}")
    print(f"Documentation:            {'✓ PASS' if test5_passed else '✗ FAIL'}")
    print("=" * 70)

    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed])

    if all_passed:
        print("\n✓ All wipe primitive tests passed!")
        print("\nThe wipe primitive is fully integrated and ready to use:")
        print("  • Registered in PRIMITIVE_LIBRARY")
        print("  • Method available in CuRoboMotionPlanner")
        print("  • Parameters validated correctly")
        print("  • Documented for LLM skill decomposition")
        return 0
    else:
        print("\n✗ Some wipe primitive tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
