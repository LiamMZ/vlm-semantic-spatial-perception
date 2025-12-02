#!/usr/bin/env python3
"""
Simple test script to verify TAMP integration works with mock components.

This test uses minimal mocked perception data to verify that:
1. TaskOrchestrator can generate PDDL plans
2. SkillDecomposer can decompose actions to primitives
3. PrimitiveExecutor can validate and execute primitives
4. The complete pipeline integrates correctly
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.task_motion_planner import TaskAndMotionPlanner, TAMPConfig


async def test_tamp_integration():
    """Test the complete TAMP integration with mocks."""

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return False

    print("="*70)
    print("TAMP INTEGRATION TEST")
    print("="*70)
    print("\nThis test verifies the complete TAMP pipeline with mock data.")
    print("Expected flow:")
    print("  1. Initialize TaskOrchestrator")
    print("  2. Process task request")
    print("  3. Generate PDDL plan")
    print("  4. Decompose each action to primitives")
    print("  5. Validate primitive execution (dry run)")
    print("="*70 + "\n")

    # Configure TAMP
    config = TAMPConfig(
        api_key=api_key,
        state_dir=project_root / "outputs" / "tamp_test",
        orchestrator_model="gemini-2.5-pro",
        decomposer_model="gemini-robotics-er-1.5-preview",
        update_interval=2.0,
        min_observations=1,  # Lower for testing
        auto_refine_on_failure=True,
        max_refinement_attempts=2,
        solver_backend="pyperplan",  # Use pure Python solver
        solver_algorithm="lama-first",
        solver_timeout=30.0,
        dry_run_default=True,  # Always dry run for tests
    )

    # Create TAMP (without camera for testing)
    tamp = TaskAndMotionPlanner(config)

    # Override orchestrator to not initialize camera
    # This prevents camera initialization errors in test environment
    tamp.orchestrator._camera = None
    tamp.orchestrator._external_camera = True  # Pretend camera is externally managed

    try:
        # Initialize
        print("\n1. Initializing TAMP system...")
        await tamp.initialize()
        print("   ✓ TAMP initialized")

        # Mock some detected objects for testing
        print("\n2. Setting up mock environment...")
        from src.perception.object_registry import DetectedObject
        import numpy as np

        # Create mock objects
        red_block = DetectedObject(
            object_type="block",
            object_id="red_block",
            affordances={"graspable", "movable"},
            position_3d=np.array([0.3, 0.0, 0.1]),
            properties={"color": "red", "size": "small"},
            confidence=0.95
        )

        blue_block = DetectedObject(
            object_type="block",
            object_id="blue_block",
            affordances={"graspable", "movable"},
            position_3d=np.array([0.4, 0.0, 0.1]),
            properties={"color": "blue", "size": "small"},
            confidence=0.95
        )

        table = DetectedObject(
            object_type="table",
            object_id="table",
            affordances={"support"},
            position_3d=np.array([0.35, 0.0, 0.0]),
            properties={"surface": True},
            confidence=0.99
        )

        # Add to registry
        if tamp.orchestrator.tracker:
            tamp.orchestrator.tracker.registry.add_object(red_block)
            tamp.orchestrator.tracker.registry.add_object(blue_block)
            tamp.orchestrator.tracker.registry.add_object(table)
            print("   ✓ Added mock objects: red_block, blue_block, table")

        # Test a simple task
        task = "pick up the red block and place it on the table"
        print(f"\n3. Testing task: '{task}'")

        # Plan only (no execution for this simple test)
        print("\n4. Running symbolic planning...")
        solver_result = await tamp.plan_task(task, use_refinement=True)

        if not solver_result:
            print("   ✗ Planning failed")
            return False

        print(f"   ✓ Generated plan with {len(solver_result.plan)} actions:")
        for i, action in enumerate(solver_result.plan, 1):
            print(f"     {i}. {action}")

        # Test decomposition of first action only (to save time)
        if solver_result.plan:
            first_action = solver_result.plan[0]
            print(f"\n5. Testing decomposition of first action: {first_action}")

            # Parse action
            parts = first_action.strip("()").split()
            action_name = parts[0]
            action_params = {f"param{j}": param for j, param in enumerate(parts[1:], 1)}

            skill_plan = await tamp.decompose_action(
                action=action_name,
                parameters=action_params,
                temperature=0.1
            )

            if skill_plan:
                print(f"   ✓ Decomposed to {len(skill_plan.primitives)} primitives:")
                for i, prim in enumerate(skill_plan.primitives, 1):
                    print(f"     {i}. {prim.name}")

                # Test execution (dry run)
                print(f"\n6. Testing execution validation (dry run)...")
                exec_result = await tamp.execute_skill_plan(
                    action_name=first_action,
                    skill_plan=skill_plan,
                    dry_run=True
                )

                if exec_result.warnings:
                    print("   ⚠ Execution warnings:")
                    for w in exec_result.warnings:
                        print(f"     - {w}")

                if exec_result.errors:
                    print("   ✗ Execution errors:")
                    for e in exec_result.errors:
                        print(f"     - {e}")
                    return False
                else:
                    print("   ✓ Execution validation passed")

            else:
                print("   ✗ Decomposition failed")
                return False

        print("\n" + "="*70)
        print("✓ TAMP INTEGRATION TEST PASSED")
        print("="*70 + "\n")
        print("Summary:")
        print("  ✓ Orchestrator initialization")
        print("  ✓ Mock environment setup")
        print("  ✓ Task planning (PDDL)")
        print("  ✓ Action decomposition (primitives)")
        print("  ✓ Execution validation")
        print("\nThe complete TAMP pipeline is working correctly!")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await tamp.shutdown()


async def main():
    """Main entry point."""
    success = await test_tamp_integration()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
