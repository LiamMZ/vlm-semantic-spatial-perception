"""
Test Orchestrator + PDDL Solver Integration

Simple test script to verify that the orchestrator can solve PDDL problems.
This uses a mock detection scenario without requiring camera hardware.
"""

import os
import sys
import asyncio
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.planning import TaskOrchestrator
from src.planning.pddl_solver import SolverResult

pytestmark = pytest.mark.asyncio

# Import config from config directory
config_path = Path(__file__).parent.parent / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
from orchestrator_config import OrchestratorConfig

# Load environment
load_dotenv()


async def test_solver_integration():
    """Test the solver integration with simulated objects."""
    print("="*70)
    print("ORCHESTRATOR + PDDL SOLVER INTEGRATION TEST")
    print("="*70)
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return

    # Configure orchestrator (no camera for this test)
    config = OrchestratorConfig(
        api_key=api_key,
        update_interval=2.0,
        min_observations=1,
        state_dir=Path("outputs/test_solver"),
        auto_save=False,  # Disable for test
        solver_backend="auto",  # Will use pyperplan
        solver_algorithm="lama-first",
        solver_timeout=30.0,
        solver_verbose=True,
        auto_refine_on_failure=True,  # Enable automatic domain refinement
        max_refinement_attempts=3
    )

    # Create orchestrator with mock camera
    print("1. Initializing orchestrator (without camera)...")
    orchestrator = TaskOrchestrator(config, camera=None)

    # Initialize (will skip camera if None)
    try:
        await orchestrator.initialize()
        print("   ✓ Orchestrator initialized")
        print()
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return

    # Process a simple task
    task = "Put the red block on the blue block"
    print(f"2. Processing task: '{task}'")
    try:
        analysis = await orchestrator.process_task_request(task)
        print(f"   ✓ Task analyzed")
        print(f"   • Goal objects: {', '.join(analysis.goal_objects)}")
        print("   • Estimated steps: n/a")
        print()
    except Exception as e:
        print(f"   ❌ Task analysis failed: {e}")
        await orchestrator.shutdown()
        return

    # Simulate detected objects
    print("3. Simulating object detection...")
    from src.perception.object_registry import DetectedObject
    import numpy as np

    mock_objects = [
        DetectedObject(
            object_type="block",
            object_id="red_block_1",
            position_2d=(100, 100),
            position_3d=np.array([0.3, 0.0, 0.1]),
            bounding_box_2d=(80, 80, 120, 120),
            affordances={"graspable", "stackable"}
        ),
        DetectedObject(
            object_type="block",
            object_id="blue_block_1",
            position_2d=(200, 200),
            position_3d=np.array([0.5, 0.0, 0.1]),
            bounding_box_2d=(180, 180, 220, 220),
            affordances={"graspable", "stackable"}
        ),
    ]

    # Add objects to registry
    if orchestrator.tracker and orchestrator.tracker:
        for obj in mock_objects:
            orchestrator.tracker.registry.add_object(obj)

        # Add predicates to registry
        predicates = [
            "graspable red_block_1",
            "clear red_block_1",
            "graspable blue_block_1",
            "clear blue_block_1"
        ]
        orchestrator.tracker.registry.add_predicates(predicates)

        print(f"   ✓ Added {len(mock_objects)} mock objects")
        print()

    # Update PDDL domain with observations
    print("4. Updating PDDL domain from observations...")
    objects_dict = [
        {
            "object_id": obj.object_id,
            "object_type": obj.object_type,
            "affordances": list(obj.affordances),
            "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None
        }
        for obj in mock_objects
    ]

    predicates = orchestrator.tracker.registry.get_all_predicates()
    await orchestrator.maintainer.update_from_observations(objects_dict, predicates=predicates)
    print("   ✓ PDDL domain updated")
    print()

    # Set goal
    print("5. Setting goal state...")
    await orchestrator.maintainer.set_goal_from_task_analysis()
    print("   ✓ Goal set")
    print()

    # Generate PDDL files
    print("6. Generating PDDL files...")
    paths = await orchestrator.generate_pddl_files()
    print(f"   ✓ Files generated:")
    print(f"     • Domain: {paths['domain_path']}")
    print(f"     • Problem: {paths['problem_path']}")
    print()

    # Solve for plan
    print("7. Solving for plan...")
    print(f"   Backend: {orchestrator.solver.backend.value}")
    print()

    try:
        # Use the refinement-enabled solver
        pddl_dir = Path("outputs/test_solver/pddl")
        result = await orchestrator.solve_and_plan_with_refinement(output_dir=pddl_dir)

        print()
        if result.success:
            print(f"✅ SUCCESS!")
            print(f"   • Plan length: {result.plan_length} steps")
            if result.plan_cost:
                print(f"   • Plan cost: {result.plan_cost}")
            if result.search_time:
                print(f"   • Search time: {result.search_time:.2f}s")
            print()
            print("   Plan:")
            for i, action in enumerate(result.plan, 1):
                print(f"     {i}. {action}")
        else:
            print(f"❌ FAILED: {result.error_message}")

    except Exception as e:
        print(f"❌ Solver error: {e}")

    print()

    # Cleanup
    print("8. Shutting down...")
    await orchestrator.shutdown()
    print("   ✓ Cleanup complete")

    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_solver_integration())
