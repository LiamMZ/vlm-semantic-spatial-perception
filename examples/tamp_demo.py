#!/usr/bin/env python3
"""
Complete Task and Motion Planning (TAMP) Demo

Demonstrates the full pipeline from natural language task to execution:
1. Environment perception
2. Symbolic task planning (PDDL)
3. Skill decomposition (action → primitives)
4. Primitive execution (with metric translation)

Usage:
    # Dry run (validation only, no robot execution)
    python examples/tamp_demo.py --dry-run

    # Live execution (requires robot primitives interface)
    python examples/tamp_demo.py --task "pick up the red block"

    # Interactive mode
    python examples/tamp_demo.py --interactive
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.task_motion_planner import TaskAndMotionPlanner, TAMPConfig, TAMPState, TAMPResult


class TAMPDemo:
    """Interactive demo for the complete TAMP system."""

    def __init__(self, api_key: str, dry_run: bool = True):
        """
        Initialize TAMP demo.

        Args:
            api_key: Gemini API key
            dry_run: If True, validate without executing on robot
        """
        self.dry_run = dry_run

        # Configure TAMP
        config = TAMPConfig(
            api_key=api_key,
            state_dir=project_root / "outputs" / "tamp_demo",
            orchestrator_model="gemini-2.5-pro",
            decomposer_model="gemini-2.5-pro",
            update_interval=2.0,
            min_observations=3,
            auto_refine_on_failure=True,
            max_refinement_attempts=3,
            solver_backend="pyperplan",  # Use pure Python solver (no Docker needed)
            solver_algorithm="lama-first",
            solver_timeout=30.0,
            dry_run_default=dry_run,
            primitives_interface=None,  # Would be robot interface in real deployment
            # Optional: Describe robot capabilities to guide task planning
            robot_description="""
                SINGLE 7-DOF robotic arm with parallel jaw gripper.
                Capabilities: pick/place objects (max 2kg), open/close containers,
                push/pull objects, pour from containers.
                Limitations: Cannot grasp objects <1cm or >15cm, no fine manipulation.
            """,
            on_state_change=self._on_state_change,
            on_plan_generated=self._on_plan_generated,
            on_action_decomposed=self._on_action_decomposed,
            on_action_executed=self._on_action_executed,
        )

        self.tamp = TaskAndMotionPlanner(config)
        self.execution_log = []

    def _on_state_change(self, state: TAMPState):
        """Callback for TAMP state changes."""
        print(f"\n{'='*70}")
        print(f"STATE TRANSITION: {state.value.upper()}")
        print(f"{'='*70}")

    def _on_plan_generated(self, result):
        """Callback when PDDL plan is generated."""
        if result.success:
            print(f"\n✓ Plan Generated ({result.plan_length} steps):")
            for i, action in enumerate(result.plan, 1):
                print(f"  {i}. {action}")
        else:
            print(f"\n✗ Planning Failed: {result.error_message}")

    def _on_action_decomposed(self, action: str, skill_plan):
        """Callback when action is decomposed to primitives."""
        print(f"  → Skill plan for '{action}':")
        for i, primitive in enumerate(skill_plan.primitives, 1):
            print(f"    {i}. {primitive.name}({', '.join(f'{k}={v}' for k, v in primitive.parameters.items())})")

    def _on_action_executed(self, action: str, result):
        """Callback when action is executed."""
        self.execution_log.append({
            "action": action,
            "success": result.executed,
            "warnings": result.warnings,
            "errors": result.errors
        })

    async def run_single_task(self, task: str):
        """
        Run TAMP for a single task.

        Args:
            task: Natural language task description
        """
        print(f"\n{'='*70}")
        print("TAMP SINGLE TASK EXECUTION")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print(f"{'='*70}\n")

        try:
            # Initialize
            await self.tamp.initialize()

            # Perceive environment
            print("\nStep 1: Environment Perception")
            print("-" * 70)
            perception_success = await self.tamp.perceive_environment(
                duration=10.0,  # Max 10 seconds
                min_observations=3
            )

            if not perception_success:
                print("⚠ Warning: Insufficient observations, proceeding anyway...")

            # Plan and execute
            print("\nStep 2: Task Planning, Decomposition, and Execution")
            print("-" * 70)
            result = await self.tamp.plan_and_execute_task(
                task=task,
                dry_run=self.dry_run,
                use_refinement=True,
                decompose_temperature=0.1
            )

            # Print summary
            self._print_result_summary(result)

            return result

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            return None
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            await self.tamp.shutdown()

    async def run_interactive(self):
        """Run interactive TAMP demo with command loop."""
        print(f"\n{'='*70}")
        print("TAMP INTERACTIVE DEMO")
        print(f"{'='*70}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print("\nCommands:")
        print("  task <description>  - Plan and execute a task")
        print("  perceive [duration] - Run perception for N seconds")
        print("  status              - Show system status")
        print("  log                 - Show execution log")
        print("  clear               - Clear execution log")
        print("  help                - Show this help")
        print("  quit                - Exit demo")
        print(f"{'='*70}\n")

        # Initialize
        print("\nInitializing TAMP system...")
        await self.tamp.initialize()
        print("✓ TAMP initialized")
        try:
            while True:
                # Get command
                try:
                    cmd = input("\ntamp> ").strip()
                except EOFError:
                    break

                if not cmd:
                    continue

                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()

                # Handle commands
                if action in ["quit", "exit", "q"]:
                    break

                elif action == "help":
                    print("\nCommands:")
                    print("  task <description>  - Plan and execute a task")
                    print("  perceive [duration] - Run perception for N seconds")
                    print("  status              - Show system status")
                    print("  log                 - Show execution log")
                    print("  clear               - Clear execution log")
                    print("  help                - Show this help")
                    print("  quit                - Exit demo")

                elif action == "task":
                    if len(parts) < 2:
                        print("Usage: task <description>")
                        continue

                    task = parts[1]
                    result = await self.tamp.plan_and_execute_task(
                        task=task,
                        dry_run=self.dry_run
                    )
                    self._print_result_summary(result)

                elif action == "perceive":
                    duration = float(parts[1]) if len(parts) > 1 else 10.0
                    print(f"\nPerceiving environment for {duration}s...")
                    success = await self.tamp.perceive_environment(duration=duration)
                    if success:
                        print("✓ Environment sufficiently observed")
                    else:
                        print("⚠ Perception incomplete")

                elif action == "status":
                    status = await self.tamp.get_status()
                    print("\nSystem Status:")
                    print(f"  TAMP State: {status['tamp_state']}")
                    print(f"  Orchestrator State: {status['orchestrator_state']}")
                    print(f"  Current Task: {status['current_task'] or 'None'}")
                    print(f"  Detected Objects: {status['num_objects']}")
                    if status['last_result']:
                        print(f"  Last Result: {status['last_result']}")

                elif action == "log":
                    if not self.execution_log:
                        print("\nNo executions logged yet")
                    else:
                        print(f"\nExecution Log ({len(self.execution_log)} entries):")
                        for i, entry in enumerate(self.execution_log, 1):
                            status = "✓" if entry["success"] else "✗"
                            print(f"  {i}. {status} {entry['action']}")
                            if entry["warnings"]:
                                for w in entry["warnings"]:
                                    print(f"     ⚠ {w}")
                            if entry["errors"]:
                                for e in entry["errors"]:
                                    print(f"     ✗ {e}")

                elif action == "clear":
                    self.execution_log.clear()
                    print("✓ Execution log cleared")

                else:
                    print(f"Unknown command: {action}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
        finally:
            await self.tamp.shutdown()

    def _print_result_summary(self, result: TAMPResult):
        """Print a summary of TAMP execution result."""
        print(f"\n{'='*70}")
        print("TAMP RESULT SUMMARY")
        print(f"{'='*70}")

        if result.success:
            print(f"✓ SUCCESS")
            print(f"\nTask: {result.task_description}")
            print(f"\nTiming:")
            print(f"  Planning: {result.planning_time:.2f}s")
            print(f"  Decomposition: {result.decomposition_time:.2f}s")
            print(f"  Execution: {result.execution_time:.2f}s")
            print(f"  Total: {result.planning_time + result.decomposition_time + result.execution_time:.2f}s")

            print(f"\nPlan: {len(result.pddl_plan or [])} actions")
            for i, action in enumerate(result.pddl_plan or [], 1):
                print(f"  {i}. {action}")

            print(f"\nDecomposition: {len(result.skill_plans)} skill plans")
            total_primitives = sum(len(sp.primitives) for sp in result.skill_plans.values())
            print(f"  Total primitives: {total_primitives}")

            print(f"\nExecution: {len(result.execution_results)} actions executed")
        else:
            print(f"✗ FAILED")
            print(f"\nTask: {result.task_description}")
            print(f"Failed at: {result.failed_at_stage}")
            print(f"Error: {result.error_message}")

            if result.refinement_attempts > 0:
                print(f"\nRefinement attempts: {result.refinement_attempts}")

        print(f"{'='*70}\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Task and Motion Planning Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task description to execute"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without executing on robot (default for safety)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute on robot (overrides --dry-run)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key (default: GEMINI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Check API key
    if not args.api_key:
        print("Error: GEMINI_API_KEY not set. Either:")
        print("  export GEMINI_API_KEY=your_key")
        print("  or use --api-key YOUR_KEY")
        return 1

    # Determine dry run mode (default to True for safety)
    dry_run = not args.live if args.live else True
    if args.dry_run:
        dry_run = True

    # Create demo
    demo = TAMPDemo(api_key=args.api_key, dry_run=dry_run)

    # Run mode
    if args.interactive:
        await demo.run_interactive()
    elif args.task:
        result = await demo.run_single_task(args.task)
        return 0 if result and result.success else 1
    else:
        # Default: interactive mode
        print("No task specified, entering interactive mode...")
        print("(Use --task 'description' for single task execution)")
        await demo.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
