"""
PDDL Solver Demo

Demonstrates how to use the PDDL solver with different backends.
Shows both standalone usage and integration with the task orchestrator.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.planning import (
    PDDLSolver,
    SolverBackend,
    SearchAlgorithm,
    solve_pddl
)


async def demo_solver_backends():
    """Demo: Test all available solver backends."""
    print("="*70)
    print("PDDL SOLVER BACKEND DETECTION")
    print("="*70)

    solver = PDDLSolver()

    print("\nAvailable backends:")
    for backend in solver.get_available_backends():
        print(f"  ✓ {backend.value}")

    print(f"\nAuto-selected backend: {solver.backend.value}")
    print()


async def demo_simple_solve():
    """Demo: Solve a simple PDDL problem."""
    print("="*70)
    print("SIMPLE PDDL SOLVING")
    print("="*70)

    # Find example PDDL files
    project_root = Path(__file__).parent.parent
    pddl_dir = project_root / "outputs" / "orchestrator_state" / "pddl"

    domain_path = pddl_dir / "task_execution_domain.pddl"
    problem_path = pddl_dir / "task_execution_problem.pddl"

    # Fallback to test data if orchestrator files don't exist
    if not domain_path.exists() or not problem_path.exists():
        print(f"ℹ Using example blocks world domain (orchestrator files not found)")
        test_data_dir = Path(__file__).parent / "test_data"
        domain_path = test_data_dir / "blocksworld_domain.pddl"
        problem_path = test_data_dir / "blocksworld_problem.pddl"

        if not domain_path.exists() or not problem_path.exists():
            print(f"⚠ Example PDDL files not found")
            print("  Please ensure test_data directory contains blocksworld_*.pddl files")
            return

    print(f"Domain:  {domain_path.name}")
    print(f"Problem: {problem_path.name}")
    print()

    # Solve using convenience function
    print("Solving with LAMA-first (fast, ignores cost)...")
    result = await solve_pddl(
        domain_path=str(domain_path),
        problem_path=str(problem_path),
        algorithm=SearchAlgorithm.LAMA_FIRST,
        timeout=30.0,
        verbose=True
    )

    print(f"\n{result}")

    if result.success:
        print(f"\nPlan ({result.plan_length} steps):")
        for i, action in enumerate(result.plan, 1):
            print(f"  {i}. {action}")
    else:
        print(f"\n✗ Solving failed: {result.error_message}")

        if result.raw_output:
            print("\nRaw solver output:")
            print(result.raw_output[:500])  # Show first 500 chars

    print()


async def demo_different_algorithms():
    """Demo: Compare different search algorithms."""
    print("="*70)
    print("COMPARING SEARCH ALGORITHMS")
    print("="*70)

    # Find example PDDL files
    project_root = Path(__file__).parent.parent
    pddl_dir = project_root / "outputs" / "orchestrator_state" / "pddl"

    domain_path = pddl_dir / "task_execution_domain.pddl"
    problem_path = pddl_dir / "task_execution_problem.pddl"

    # Fallback to test data if orchestrator files don't exist
    if not domain_path.exists() or not problem_path.exists():
        test_data_dir = Path(__file__).parent / "test_data"
        domain_path = test_data_dir / "blocksworld_domain.pddl"
        problem_path = test_data_dir / "blocksworld_problem.pddl"

        if not domain_path.exists() or not problem_path.exists():
            print(f"⚠ Example PDDL files not found")
            return

    algorithms = [
        (SearchAlgorithm.LAMA_FIRST, "Fast, ignores cost"),
        (SearchAlgorithm.LAMA, "Optimizes cost"),
        (SearchAlgorithm.ASTAR_LMCUT, "A* with landmark cut"),
    ]

    solver = PDDLSolver(verbose=False)

    results = []
    for algorithm, description in algorithms:
        print(f"\n{algorithm.value:30s} - {description}")
        print(f"  Solving...")

        result = await solver.solve(
            domain_path=str(domain_path),
            problem_path=str(problem_path),
            algorithm=algorithm,
            timeout=30.0
        )

        if result.success:
            print(f"  ✓ Plan length: {result.plan_length}")
            if result.plan_cost:
                print(f"    Cost: {result.plan_cost}")
            if result.search_time:
                print(f"    Time: {result.search_time:.2f}s")
            if result.nodes_expanded:
                print(f"    Nodes: {result.nodes_expanded}")
        else:
            print(f"  ✗ Failed: {result.error_message}")

        results.append((algorithm.value, result))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [r for r in results if r[1].success]
    if successful:
        fastest = min(successful, key=lambda x: x[1].search_time or float('inf'))
        shortest = min(successful, key=lambda x: x[1].plan_length)

        fastest_time = fastest[1].search_time if fastest[1].search_time is not None else 0.0
        print(f"Fastest: {fastest[0]} ({fastest_time:.2f}s)")
        print(f"Shortest plan: {shortest[0]} ({shortest[1].plan_length} steps)")
    else:
        print("No algorithms succeeded")

    print()


async def demo_backend_comparison():
    """Demo: Compare different solver backends."""
    print("="*70)
    print("BACKEND COMPARISON")
    print("="*70)

    # Find example PDDL files
    project_root = Path(__file__).parent.parent
    pddl_dir = project_root / "outputs" / "orchestrator_state" / "pddl"

    domain_path = pddl_dir / "task_execution_domain.pddl"
    problem_path = pddl_dir / "task_execution_problem.pddl"

    # Fallback to test data if orchestrator files don't exist
    if not domain_path.exists() or not problem_path.exists():
        test_data_dir = Path(__file__).parent / "test_data"
        domain_path = test_data_dir / "blocksworld_domain.pddl"
        problem_path = test_data_dir / "blocksworld_problem.pddl"

        if not domain_path.exists() or not problem_path.exists():
            print(f"⚠ Example PDDL files not found")
            return

    backends_to_test = [
        SolverBackend.FAST_DOWNWARD_APPTAINER,
        SolverBackend.FAST_DOWNWARD_DOCKER,
        SolverBackend.PYPERPLAN,
    ]

    for backend in backends_to_test:
        solver = PDDLSolver(backend=SolverBackend.AUTO)

        if not solver.is_backend_available(backend):
            print(f"\n{backend.value:30s} - NOT AVAILABLE")
            continue

        print(f"\n{backend.value:30s}")
        solver.backend = backend

        result = await solver.solve(
            domain_path=str(domain_path),
            problem_path=str(problem_path),
            algorithm=SearchAlgorithm.LAMA_FIRST,
            timeout=30.0
        )

        if result.success:
            print(f"  ✓ Plan length: {result.plan_length}")
            if result.search_time:
                print(f"    Time: {result.search_time:.2f}s")
        else:
            print(f"  ✗ Failed: {result.error_message}")

    print()


async def demo_orchestrator_integration():
    """Demo: Using solver with TaskOrchestrator."""
    print("="*70)
    print("ORCHESTRATOR + SOLVER INTEGRATION")
    print("="*70)

    from src.planning import TaskOrchestrator, OrchestratorConfig

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY not set, skipping orchestrator demo")
        return

    # Create minimal config for demo (no camera)
    config = OrchestratorConfig(
        api_key=api_key,
        update_interval=2.0,
        min_observations=1  # Lower for demo
    )

    print("\nThis demo shows how to integrate the solver with the orchestrator:")
    print("1. Orchestrator generates PDDL files")
    print("2. Solver finds a plan")
    print("3. Plan can be executed by robot controller")
    print()

    # Example code (not executed without camera)
    print("Example code:")
    print("""
    # Initialize
    orchestrator = TaskOrchestrator(config)
    await orchestrator.initialize()

    # Process task and detect objects
    await orchestrator.process_task_request("make coffee")
    await orchestrator.start_detection()

    # Wait until ready for planning
    while not orchestrator.is_ready_for_planning():
        await asyncio.sleep(1.0)

    # Generate PDDL files
    paths = await orchestrator.generate_pddl_files()

    # Solve with PDDL solver
    result = await solve_pddl(
        domain_path=paths['domain_path'],
        problem_path=paths['problem_path'],
        algorithm=SearchAlgorithm.LAMA_FIRST,
        timeout=30.0
    )

    if result.success:
        print(f"Plan: {result.plan}")
        # Execute plan with robot controller
        # for action in result.plan:
        #     await robot.execute(action)
    """)

    print()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("PDDL SOLVER DEMONSTRATION")
    print("="*70)
    print()

    demos = [
        ("Backend Detection", demo_solver_backends),
        ("Simple Solve", demo_simple_solve),
        ("Algorithm Comparison", demo_different_algorithms),
        ("Backend Comparison", demo_backend_comparison),
        ("Orchestrator Integration", demo_orchestrator_integration),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
