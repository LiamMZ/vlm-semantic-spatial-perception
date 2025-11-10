"""
Task Monitoring Demo

Demonstrates the new PDDLDomainMaintainer and TaskStateMonitor classes working together
to intelligently manage task execution with adaptive exploration and planning.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.planning import (
    PDDLRepresentation,
    PDDLDomainMaintainer,
    TaskStateMonitor,
    TaskState
)

# Load environment
load_dotenv()


async def main():
    """Demonstrate task monitoring workflow."""
    print("\n" + "=" * 70)
    print("TASK MONITORING DEMO")
    print("=" * 70)
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ ERROR: GEMINI_API_KEY not set")
        return

    # =========================================================================
    # Step 1: Initialize Components
    # =========================================================================
    print("Step 1: Initializing components...")
    print()

    # Create PDDL representation (thread-safe)
    pddl = PDDLRepresentation(
        domain_name="adaptive_task",
        problem_name="monitored_task"
    )

    # Create domain maintainer
    maintainer = PDDLDomainMaintainer(pddl, api_key=api_key)

    # Create task monitor
    monitor = TaskStateMonitor(
        maintainer,
        pddl,
        min_observations_before_planning=2
    )

    print("  ✓ PDDL representation initialized (thread-safe)")
    print("  ✓ Domain maintainer created")
    print("  ✓ Task state monitor created")
    print()

    # =========================================================================
    # Step 2: Initialize Domain from Task
    # =========================================================================
    print("Step 2: Analyzing task and initializing domain...")
    print()

    task = "Put the red mug on the shelf"
    print(f"  Task: \"{task}\"")
    print()

    task_analysis = await maintainer.initialize_from_task(task)

    print(f"  → Predicted predicates ({len(task_analysis.relevant_predicates)}):")
    for pred in task_analysis.relevant_predicates[:10]:
        print(f"      • {pred}")
    if len(task_analysis.relevant_predicates) > 10:
        print(f"      ... and {len(task_analysis.relevant_predicates) - 10} more")

    print(f"\n  → Goal objects: {', '.join(task_analysis.goal_objects)}")
    print(f"  → Estimated steps: {task_analysis.estimated_steps}")
    print()

    # =========================================================================
    # Step 3: Check Initial State
    # =========================================================================
    print("Step 3: Checking task state...")
    print()

    decision = await monitor.determine_state()
    print(f"  State: {decision.state.value}")
    print(f"  Confidence: {decision.confidence:.1%}")
    print(f"  Reasoning: {decision.reasoning}")
    print()

    if decision.blockers:
        print("  Blockers:")
        for blocker in decision.blockers:
            print(f"    ✗ {blocker}")
        print()

    if decision.recommendations:
        print("  Recommendations:")
        for rec in decision.recommendations:
            print(f"    → {rec}")
        print()

    # =========================================================================
    # Step 4: Simulate Exploration - First Observation
    # =========================================================================
    print("Step 4: Simulating exploration (first observation)...")
    print()

    # Simulate detecting a table (not the goal object)
    observed_objects_1 = [
        {
            "object_id": "table_1",
            "object_type": "table",
            "affordances": ["supportable"],
            "pddl_state": {
                "graspable": False,
                "reachable": True,
                "on": False
            }
        }
    ]

    update_stats = await maintainer.update_from_observations(observed_objects_1)
    print(f"  → Added {update_stats['objects_added']} object(s)")
    print(f"  → New types: {update_stats['new_object_types']}")
    print(f"  → Goal objects found: {update_stats['goal_objects_found']}")
    print(f"  → Still missing: {update_stats['goal_objects_missing']}")
    print()

    # Check state again
    decision = await monitor.determine_state()
    print(f"  State: {decision.state.value}")
    print(f"  Reasoning: {decision.reasoning}")
    print()

    # =========================================================================
    # Step 5: Continue Exploration - Find Goal Objects
    # =========================================================================
    print("Step 5: Continuing exploration (finding goal objects)...")
    print()

    # Now find the red mug
    observed_objects_2 = [
        {
            "object_id": "red_mug_1",
            "object_type": "mug",
            "affordances": ["graspable", "containable"],
            "pddl_state": {
                "graspable": True,
                "reachable": True,
                "on": True,
                "holding": False
            }
        },
        {
            "object_id": "shelf_1",
            "object_type": "shelf",
            "affordances": ["supportable"],
            "pddl_state": {
                "graspable": False,
                "reachable": True
            }
        }
    ]

    update_stats = await maintainer.update_from_observations(
        observed_objects_1 + observed_objects_2
    )

    print(f"  → Total objects: {update_stats['objects_added']}")
    print(f"  → Goal objects found: {update_stats['goal_objects_found']}")
    print(f"  → Goal objects missing: {update_stats['goal_objects_missing']}")
    print()

    # =========================================================================
    # Step 6: Set Goals
    # =========================================================================
    print("Step 6: Setting goal state from task analysis...")
    print()

    await maintainer.set_goal_from_task_analysis()

    goal_state = await pddl.get_goal_state()
    print(f"  → Goal literals: {len(goal_state)}")
    for literal in list(goal_state)[:5]:
        print(f"      • {literal.to_pddl()}")
    print()

    # =========================================================================
    # Step 7: Check if Ready for Planning
    # =========================================================================
    print("Step 7: Checking if ready for planning...")
    print()

    decision = await monitor.determine_state()
    print(f"  State: {decision.state.value}")
    print(f"  Confidence: {decision.confidence:.1%}")
    print(f"  Reasoning: {decision.reasoning}")
    print()

    if decision.state == TaskState.PLAN_AND_EXECUTE:
        print("  ✓ Ready for planning!")
        print()
        print("  Recommendations:")
        for rec in decision.recommendations:
            print(f"    → {rec}")
        print()

        # Show what we'd pass to planner
        print("  Domain & Problem Summary:")
        domain_snapshot = await pddl.get_domain_snapshot()
        problem_snapshot = await pddl.get_problem_snapshot()

        print(f"    • Predicates: {len(domain_snapshot['predicates'])}")
        print(f"    • Actions: {len(domain_snapshot['predefined_actions']) + len(domain_snapshot['llm_generated_actions'])}")
        print(f"    • Objects: {len(problem_snapshot['object_instances'])}")
        print(f"    • Initial literals: {len(problem_snapshot['initial_literals'])}")
        print(f"    • Goal literals: {len(problem_snapshot['goal_literals'])}")
        print()

    elif decision.blockers:
        print("  ✗ Not ready yet. Blockers:")
        for blocker in decision.blockers:
            print(f"    • {blocker}")
        print()

    # =========================================================================
    # Step 8: Generate PDDL Files
    # =========================================================================
    if decision.state == TaskState.PLAN_AND_EXECUTE:
        print("Step 8: Generating PDDL files for planner...")
        print()

        paths = await pddl.generate_files_async("outputs/pddl/task_monitoring")
        print(f"  ✓ Domain: {paths['domain_path']}")
        print(f"  ✓ Problem: {paths['problem_path']}")
        print()

    # =========================================================================
    # Step 9: Decision History
    # =========================================================================
    print("Step 9: Decision history summary...")
    print()

    summary = await monitor.get_decision_summary()
    print(f"  Total decisions: {summary['total_decisions']}")
    print(f"  Current state: {summary['current_state']}")
    print(f"  State progression: {' → '.join(summary['recent_states'])}")
    print()

    # =========================================================================
    # Complete
    # =========================================================================
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. PDDLDomainMaintainer manages domain evolution from task + observations")
    print("  2. TaskStateMonitor intelligently decides between EXPLORE, PLAN, or REFINE")
    print("  3. System adapts to incomplete information and guides data collection")
    print("  4. Thread-safe PDDL representation enables concurrent access")
    print("  5. Ready for integration with PDDL planners (FastDownward, etc.)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
