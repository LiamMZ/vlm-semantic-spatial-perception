"""
Task Orchestrator Demo

Demonstrates the complete task orchestrator for production use.

This demo shows:
1. Initializing the orchestrator
2. Processing task requests
3. Running continuous detection
4. Monitoring task status
5. Saving/loading state
6. Generating PDDL files when ready
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.planning import (
    TaskOrchestrator,
    OrchestratorState,
    TaskState
)

# Import config from config directory
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from orchestrator_config import OrchestratorConfig

# Load environment
load_dotenv()


async def main():
    """Run orchestrator demo."""
    print("\n" + "="*70)
    print("TASK ORCHESTRATOR DEMO")
    print("="*70)
    print()
    print("This demo shows production-ready task orchestration with:")
    print("  • Task request processing")
    print("  • Continuous object detection")
    print("  • Automatic PDDL domain updates")
    print("  • Task state monitoring")
    print("  • State persistence (save/load)")
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("Please set it in .env file or environment")
        return

    # Ask user if they want to load existing state
    print("Do you want to:")
    print("  1. Start fresh with a new task")
    print("  2. Load previous state and continue")
    choice = input("\nEnter choice (1 or 2): ").strip()
    print()

    load_state = choice == "2"

    # Get task from user (if starting fresh)
    task = None
    if not load_state:
        print("Enter the task description (or press Enter for default):")
        task = input("> ").strip()
        if not task:
            task = "make a cup of coffee"
            print(f"Using default task: \"{task}\"")
        print()

    # Get update interval
    print("Enter detection update interval in seconds (default: 2.0):")
    interval_str = input("> ").strip()
    if interval_str:
        try:
            update_interval = float(interval_str)
        except ValueError:
            update_interval = 2.0
            print(f"Invalid input, using default: {update_interval}s")
    else:
        update_interval = 2.0
        print(f"Using default: {update_interval}s")
    print()

    # Configure orchestrator
    config = OrchestratorConfig(
        api_key=api_key,
        update_interval=update_interval,
        min_observations=2,
        fast_mode=False,  # Full detection with interaction points
        auto_save=True,
        auto_save_interval=30.0,
        state_dir=Path("outputs/orchestrator_state"),
        on_state_change=on_state_change,
        on_detection_update=on_detection_update,
        on_task_state_change=on_task_state_change
    )

    # Initialize orchestrator
    print("="*70)
    orchestrator = TaskOrchestrator(config)

    try:
        # Initialize
        await orchestrator.initialize()

        # Load state or process new task
        if load_state:
            state_path = config.state_dir / "state.json"
            if state_path.exists():
                print(f"\n{'='*70}")
                print("LOADING PREVIOUS STATE")
                print(f"{'='*70}\n")
                await orchestrator.load_state()
                print()
            else:
                print(f"⚠ No saved state found at {state_path}")
                print("Starting fresh instead...")
                load_state = False

        if not load_state:
            # Process task request
            await orchestrator.process_task_request(task)

        # Start continuous detection
        await orchestrator.start_detection()

        # Wait for camera to stabilize
        print("Waiting for camera to stabilize...")
        await asyncio.sleep(2.0)
        print()

        # Main control loop
        print("="*70)
        print("ORCHESTRATOR RUNNING")
        print("="*70)
        print("\nCommands:")
        print("  'status'  - Show current status")
        print("  'objects' - List detected objects")
        print("  'save'    - Save current state")
        print("  'load'    - Load saved state")
        print("  'pause'   - Pause detection")
        print("  'resume'  - Resume detection")
        print("  'task'    - Update task")
        print("  'pddl'    - Generate PDDL files")
        print("  'stop'    - Stop and generate PDDL if ready")
        print("  'quit'    - Quit without generating PDDL")
        print()

        running = True
        while running:
            # Wait a bit
            await asyncio.sleep(1.0)

            # Check for user input (with timeout)
            try:
                print("Enter command (or press Ctrl+C to stop): ", end="", flush=True)

                async def get_input():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, input)

                try:
                    command = await asyncio.wait_for(get_input(), timeout=5.0)
                    command = command.strip().lower()

                    if command == 'status':
                        await show_status(orchestrator)
                    elif command == 'objects':
                        await show_objects(orchestrator)
                    elif command == 'save':
                        await orchestrator.save_state()
                    elif command == 'load':
                        await orchestrator.load_state()
                    elif command == 'pause':
                        await orchestrator.pause_detection()
                    elif command == 'resume':
                        await orchestrator.resume_detection()
                    elif command == 'task':
                        new_task = input("Enter new task: ").strip()
                        if new_task:
                            await orchestrator.update_task(new_task)
                    elif command == 'pddl':
                        await orchestrator.generate_pddl_files()
                    elif command == 'stop':
                        print("\nStopping orchestrator...")
                        running = False
                    elif command == 'quit':
                        print("\nQuitting without generating PDDL...")
                        await orchestrator.shutdown()
                        return
                    elif command:
                        print(f"Unknown command: '{command}'")

                except asyncio.TimeoutError:
                    # No input, continue
                    print("\r" + " "*50 + "\r", end="", flush=True)

            except KeyboardInterrupt:
                print("\n\nCtrl+C detected, stopping...")
                running = False

        # Stop detection
        await orchestrator.stop_detection()

        # Show final status
        print("\n" + "="*70)
        print("FINAL STATUS")
        print("="*70)
        await show_status(orchestrator)

        # Generate PDDL files if ready
        if orchestrator.is_ready_for_planning():
            print("\n✓ System ready for planning!")
            print("\nGenerating PDDL files...")
            paths = await orchestrator.generate_pddl_files()

            print("\n✓ PDDL files ready!")
            print("\nNext steps:")
            print(f"  1. Review files: {paths['domain_path']}, {paths['problem_path']}")
            print("  2. Run PDDL planner (e.g., FastDownward, LAMA)")
            print("  3. Execute the generated plan")
        else:
            decision = await orchestrator.get_task_decision()
            print("\n⚠ Not ready for planning yet")
            print(f"Current state: {decision.state.value}")
            print(f"Reason: {decision.reasoning}")

            if decision.blockers:
                print("\nBlockers:")
                for blocker in decision.blockers:
                    print(f"  • {blocker}")

            # Ask if user wants to generate anyway
            print("\nGenerate PDDL files anyway? (y/n): ", end="", flush=True)
            response = input().strip().lower()
            if response == 'y':
                await orchestrator.generate_pddl_files()

        # Save final state
        print("\nSaving final state...")
        await orchestrator.save_state()

        # Cleanup
        await orchestrator.shutdown()

        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)

        # Show summary
        status = await orchestrator.get_status()
        print(f"\nSummary:")
        print(f"  • Task: {orchestrator.current_task}")
        print(f"  • Detection cycles: {orchestrator.detection_count}")
        print(f"  • Total objects detected: {len(orchestrator.registry)}")
        print(f"  • Ready for planning: {'YES' if orchestrator.is_ready_for_planning() else 'NO'}")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        await orchestrator.shutdown()
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        await orchestrator.shutdown()


async def show_status(orchestrator: TaskOrchestrator):
    """Display orchestrator status."""
    status = await orchestrator.get_status()

    print(f"\n{'='*70}")
    print("ORCHESTRATOR STATUS")
    print(f"{'='*70}")
    print(f"State: {status['orchestrator_state']}")
    print(f"Task: {status['current_task'] or 'None'}")
    print(f"Detection running: {status['detection_running']}")
    print(f"Detection cycles: {status['detection_count']}")
    print(f"Ready for planning: {'YES' if status['ready_for_planning'] else 'NO'}")

    # Registry stats
    print(f"\nObject Registry:")
    print(f"  • Total objects: {status['registry']['num_objects']}")
    print(f"  • Object types: {', '.join(status['registry']['object_types']) or 'none'}")

    # Tracker stats
    if 'tracker' in status:
        tracker = status['tracker']
        print(f"\nDetection Stats:")
        print(f"  • Total frames: {tracker['total_frames']}")
        print(f"  • Frames with detection: {tracker['total_detections']}")
        print(f"  • Frames skipped: {tracker['skipped_frames']}")
        print(f"  • Cache hit rate: {tracker['cache_hit_rate']:.1%}")
        print(f"  • Avg detection time: {tracker['avg_detection_time']:.2f}s")

    # Domain stats
    if 'domain' in status:
        domain = status['domain']
        print(f"\nPDDL Domain:")
        print(f"  • Object instances: {domain['object_instances']}")
        print(f"  • Object types: {domain['object_types_observed']}")
        print(f"  • Predicates: {domain['predicates_defined']}")
        print(f"  • Actions: {domain['actions_defined']}")
        print(f"  • Domain complete: {domain['domain_complete']}")

    # Task state
    if 'task_state' in status:
        task = status['task_state']
        print(f"\nTask State:")
        print(f"  • Current: {task['state']}")
        print(f"  • Confidence: {task['confidence']:.1%}")
        print(f"  • Reasoning: {task['reasoning']}")

        if task['blockers']:
            print(f"\n  Blockers:")
            for blocker in task['blockers']:
                print(f"    ✗ {blocker}")

        if task['recommendations']:
            print(f"\n  Recommendations:")
            for rec in task['recommendations']:
                print(f"    → {rec}")

    print(f"{'='*70}\n")


async def show_objects(orchestrator: TaskOrchestrator):
    """Display detected objects."""
    objects = orchestrator.get_detected_objects()

    print(f"\n{'='*70}")
    print(f"DETECTED OBJECTS ({len(objects)} total)")
    print(f"{'='*70}")

    if not objects:
        print("No objects detected yet")
        print(f"{'='*70}\n")
        return

    # Group by type
    by_type = {}
    for obj in objects:
        if obj.object_type not in by_type:
            by_type[obj.object_type] = []
        by_type[obj.object_type].append(obj)

    for obj_type, objs in sorted(by_type.items()):
        print(f"\n{obj_type} ({len(objs)}):")
        for obj in objs:
            print(f"  • {obj.object_id}")
            if obj.affordances:
                print(f"    Affordances: {', '.join(obj.affordances)}")
            if obj.pddl_state:
                true_predicates = [k for k, v in obj.pddl_state.items() if v]
                if true_predicates:
                    print(f"    Predicates: {', '.join(true_predicates)}")

    print(f"\n{'='*70}\n")


def on_state_change(old_state: OrchestratorState, new_state: OrchestratorState):
    """Callback for state changes."""
    print(f"\n⚡ STATE CHANGE: {old_state.value} → {new_state.value}\n")


def on_detection_update(object_count: int):
    """Callback for detection updates."""
    # Silent - print statement would clutter output
    pass


def on_task_state_change(decision):
    """Callback for task state changes."""
    print(f"\n⚡ TASK STATE: {decision.state.value} (confidence: {decision.confidence:.1%})")
    print(f"   {decision.reasoning}\n")

    if decision.state == TaskState.PLAN_AND_EXECUTE:
        print("🎯 " + "="*68)
        print("🎯 READY FOR PLANNING!")
        print("🎯 " + "="*68)
        print()


if __name__ == "__main__":
    asyncio.run(main())
