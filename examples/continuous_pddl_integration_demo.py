"""
Continuous PDDL Integration Demo

Demonstrates the complete integration of:
1. Task analysis and initial PDDL domain generation
2. Continuous object detection in the background
3. Real-time PDDL domain updates from observations
4. Task state monitoring with adaptive decision-making
5. User-controlled loop with live status updates

This demo runs until the user decides to stop, continuously updating
the PDDL domain as new objects are detected.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import numpy as np

from src.camera import RealSenseCamera
from src.perception import ContinuousObjectTracker
from src.planning import (
    PDDLRepresentation,
    PDDLDomainMaintainer,
    TaskStateMonitor,
    TaskState
)

# Load environment
load_dotenv()


class ContinuousPDDLIntegration:
    """
    Manages the continuous integration of perception and planning.

    Coordinates the continuous object tracker, PDDL domain maintainer,
    and task state monitor in a unified system.
    """

    def __init__(
        self,
        api_key: str,
        task_description: str,
        camera: Optional[RealSenseCamera] = None,
        update_interval: float = 2.0,
        min_observations: int = 3
    ):
        """
        Initialize the integration system.

        Args:
            api_key: Google AI API key
            task_description: Natural language task description
            camera: RealSense camera (if None, will try to create one)
            update_interval: Seconds between detection updates
            min_observations: Minimum observations before planning
        """
        self.api_key = api_key
        self.task_description = task_description
        self.update_interval = update_interval

        # Initialize camera
        if camera is None:
            print("Initializing RealSense camera...")
            self.camera = RealSenseCamera(
                width=640,
                height=480,
                fps=30,
                enable_depth=True,
                auto_start=True
            )
            print("✓ Camera initialized")
        else:
            self.camera = camera

        # Initialize PDDL components
        print("\nInitializing PDDL system...")
        self.pddl = PDDLRepresentation(
            domain_name="continuous_task",
            problem_name="continuous_problem"
        )
        self.maintainer = PDDLDomainMaintainer(
            self.pddl,
            api_key=api_key
        )
        self.monitor = TaskStateMonitor(
            self.maintainer,
            self.pddl,
            min_observations_before_planning=min_observations
        )
        print("✓ PDDL system initialized")

        # Initialize continuous tracker
        print("\nInitializing continuous object tracker...")
        self.tracker = ContinuousObjectTracker(
            api_key=api_key,
            fast_mode=False,  # Full detection with interaction points
            update_interval=update_interval,
            on_detection_complete=self._on_detection_callback
        )

        # Set frame provider
        self.tracker.set_frame_provider(self._get_camera_frames)
        print("✓ Continuous tracker initialized")

        # State tracking
        self.task_analysis = None
        self.detection_count = 0
        self.last_state = None
        self.ready_for_planning = False
        self._running = False

    def _get_camera_frames(self):
        """Frame provider for continuous tracker."""
        try:
            color, depth = self.camera.get_aligned_frames()
            intrinsics = self.camera.get_camera_intrinsics()
            return color, depth, intrinsics
        except Exception as e:
            print(f"⚠ Camera error: {e}")
            return None, None, None

    async def _on_detection_callback(self, object_count: int):
        """
        Called after each detection cycle.

        Updates the PDDL domain and checks task state.
        """
        self.detection_count += 1

        print(f"\n{'='*70}")
        print(f"DETECTION UPDATE #{self.detection_count}")
        print(f"{'='*70}")
        print(f"Detected {object_count} objects")

        # Get all detected objects
        all_objects = self.tracker.get_all_objects()

        if not all_objects:
            print("  No objects detected yet")
            return

        # Convert to dict format for PDDL maintainer
        objects_dict = [
            {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "affordances": list(obj.affordances),
                "pddl_state": obj.pddl_state,
                "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None
            }
            for obj in all_objects
        ]

        # Update PDDL domain
        print("\nUpdating PDDL domain...")
        update_stats = await self.maintainer.update_from_observations(objects_dict)

        print(f"  • Objects added this update: {update_stats['objects_added']}")
        print(f"  • Total observations: {update_stats['total_observations']}")
        print(f"  • Object types: {update_stats['total_object_types']}")
        print(f"  • Goal objects found: {', '.join(update_stats['goal_objects_found']) or 'none'}")
        print(f"  • Goal objects missing: {', '.join(update_stats['goal_objects_missing']) or 'none'}")

        # Check task state
        print("\nChecking task state...")
        decision = await self.monitor.determine_state()

        # Print state change if different
        if self.last_state != decision.state:
            print(f"\n  ⚡ STATE CHANGED: {self.last_state or 'INIT'} → {decision.state.value}")
            self.last_state = decision.state

        print(f"  • Current state: {decision.state.value}")
        print(f"  • Confidence: {decision.confidence:.1%}")
        print(f"  • Reasoning: {decision.reasoning}")

        if decision.blockers:
            print(f"\n  Blockers:")
            for blocker in decision.blockers:
                print(f"    ✗ {blocker}")

        if decision.recommendations:
            print(f"\n  Recommendations:")
            for rec in decision.recommendations:
                print(f"    → {rec}")

        # Check if ready for planning
        if decision.state == TaskState.PLAN_AND_EXECUTE:
            if not self.ready_for_planning:
                print("\n" + "="*70)
                print("🎯 READY FOR PLANNING!")
                print("="*70)
                self.ready_for_planning = True

        # Show detected objects summary
        print("\nDetected Objects Summary:")
        object_types = {}
        for obj in all_objects:
            object_types[obj.object_type] = object_types.get(obj.object_type, 0) + 1

        for obj_type, count in sorted(object_types.items()):
            print(f"  • {obj_type}: {count}")

        print(f"\n{'='*70}\n")

    async def initialize_from_task(self):
        """
        Analyze task and initialize PDDL domain.
        """
        print(f"\n{'='*70}")
        print("TASK ANALYSIS")
        print(f"{'='*70}")
        print(f"Task: \"{self.task_description}\"")
        print()

        # Capture initial frame for context
        print("Capturing environment frame for context...")
        color_frame, _ = self.camera.get_aligned_frames()

        # Analyze task
        print("Analyzing task with LLM...")
        self.task_analysis = await self.maintainer.initialize_from_task(
            self.task_description,
            environment_image=color_frame
        )

        print(f"\n✓ Task analyzed!")
        print(f"  • Goal objects: {', '.join(self.task_analysis.goal_objects)}")
        print(f"  • Estimated steps: {self.task_analysis.estimated_steps}")
        print(f"  • Complexity: {self.task_analysis.complexity}")
        print(f"  • Required predicates: {len(self.task_analysis.relevant_predicates)}")

        # Show some predicates
        print(f"\n  Key predicates:")
        for pred in self.task_analysis.relevant_predicates[:8]:
            print(f"    • {pred}")
        if len(self.task_analysis.relevant_predicates) > 8:
            print(f"    ... and {len(self.task_analysis.relevant_predicates) - 8} more")

        # Seed perception with predicates
        print(f"\nSeeding perception with {len(self.task_analysis.relevant_predicates)} predicates...")
        self.tracker.tracker.set_pddl_predicates(self.task_analysis.relevant_predicates)
        print("✓ Perception configured")

        print(f"\n{'='*70}\n")

    def start_tracking(self):
        """Start continuous tracking loop."""
        print(f"\n{'='*70}")
        print("STARTING CONTINUOUS TRACKING")
        print(f"{'='*70}")
        print(f"Update interval: {self.update_interval}s")
        print()

        self.tracker.start()
        self._running = True
        print("✓ Tracking started")
        print()
        print("The system will now continuously:")
        print("  1. Detect objects in the scene")
        print("  2. Update the PDDL domain")
        print("  3. Monitor task state")
        print("  4. Report progress")
        print()
        print("Press Ctrl+C or type 'stop' to end tracking")
        print(f"\n{'='*70}\n")

    async def stop_tracking(self):
        """Stop continuous tracking loop."""
        print("\n\nStopping continuous tracking...")
        self._running = False
        await self.tracker.stop()
        print("✓ Tracking stopped")

    async def generate_pddl_files(self, output_dir: str = "outputs/pddl/continuous"):
        """
        Generate final PDDL files.

        Args:
            output_dir: Directory to save PDDL files

        Returns:
            Dict with paths to generated files
        """
        print(f"\n{'='*70}")
        print("GENERATING PDDL FILES")
        print(f"{'='*70}")

        # Set goals if not already set
        print("Setting goal state...")
        await self.maintainer.set_goal_from_task_analysis()

        # Generate files
        print(f"Generating files to {output_dir}...")
        paths = await self.pddl.generate_files_async(output_dir)

        print(f"\n✓ PDDL files generated:")
        print(f"  • Domain: {paths['domain_path']}")
        print(f"  • Problem: {paths['problem_path']}")

        # Show summary
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        print(f"\nDomain Summary:")
        print(f"  • Types: {len(domain_snapshot['object_types'])}")
        print(f"  • Predicates: {len(domain_snapshot['predicates'])}")
        print(f"  • Predefined actions: {len(domain_snapshot['predefined_actions'])}")
        print(f"  • LLM-generated actions: {len(domain_snapshot['llm_generated_actions'])}")

        print(f"\nProblem Summary:")
        print(f"  • Object instances: {len(problem_snapshot['object_instances'])}")
        print(f"  • Initial literals: {len(problem_snapshot['initial_literals'])}")
        print(f"  • Goal literals: {len(problem_snapshot['goal_literals'])}")

        print(f"\n{'='*70}\n")

        return paths

    async def show_status(self):
        """Show current system status."""
        stats = await self.tracker.get_stats()

        print(f"\n{'='*70}")
        print("SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"Tracking: {'RUNNING' if stats.is_running else 'STOPPED'}")
        print(f"Detection cycles: {stats.total_frames}")
        print(f"Frames with detection: {stats.total_frames - stats.skipped_frames}")
        print(f"Frames skipped (no scene change): {stats.skipped_frames}")
        print(f"Cache hit rate: {stats.cache_hit_rate:.1%}")
        print(f"Total detections: {stats.total_detections}")
        print(f"Avg detection time: {stats.avg_detection_time:.2f}s")
        print(f"Last detection time: {stats.last_detection_time:.2f}s")

        # PDDL status
        domain_stats = await self.maintainer.get_domain_statistics()
        print(f"\nPDDL Domain:")
        print(f"  • Object instances: {domain_stats['object_instances']}")
        print(f"  • Object types observed: {domain_stats['object_types_observed']}")
        print(f"  • Domain version: {domain_stats['domain_version']}")

        # Task status
        decision = await self.monitor.determine_state()
        print(f"\nTask State:")
        print(f"  • Current: {decision.state.value}")
        print(f"  • Confidence: {decision.confidence:.1%}")
        print(f"  • Ready for planning: {'YES' if self.ready_for_planning else 'NO'}")

        print(f"{'='*70}\n")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.camera.stop()
        print("✓ Camera stopped")


async def main():
    """Run continuous PDDL integration demo."""
    print("\n" + "="*70)
    print("CONTINUOUS PDDL INTEGRATION DEMO")
    print("="*70)
    print()
    print("This demo continuously detects objects and updates the PDDL domain")
    print("until you decide to stop.")
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("Please set it in .env file or environment")
        return

    # Get task from user
    print("Enter the task description (or press Enter for default):")
    task = input("> ").strip()
    if not task:
        task = "Pick up the red mug and place it on the shelf"
        print(f"Using default task: \"{task}\"")

    print()

    # Get update interval
    print("Enter update interval in seconds (default: 2.0):")
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

    # Initialize system
    try:
        system = ContinuousPDDLIntegration(
            api_key=api_key,
            task_description=task,
            update_interval=update_interval,
            min_observations=2
        )
    except Exception as e:
        print(f"\n⚠ Failed to initialize system: {e}")
        print("\nTroubleshooting:")
        print("  • Is RealSense camera connected?")
        print("  • Is the camera in use by another application?")
        print("  • Try running: rs-enumerate-devices")
        return

    try:
        # Initialize from task
        await system.initialize_from_task()

        # Start continuous tracking
        system.start_tracking()

        # Wait for stable frames
        print("Waiting for camera to stabilize...")
        await asyncio.sleep(2.0)

        # Main control loop
        print("\nCommands:")
        print("  'status' - Show current status")
        print("  'stop' - Stop tracking and generate PDDL")
        print("  'quit' - Quit without generating PDDL")
        print()

        while system._running:
            # Wait a bit
            await asyncio.sleep(1.0)

            # Check for user input (non-blocking)
            # Note: This is a simple demo. In production, use proper async input handling
            try:
                # Simple prompt
                print("Enter command (or press Ctrl+C to stop): ", end="", flush=True)

                # Create a task that waits for input with timeout
                async def get_input():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, input)

                try:
                    command = await asyncio.wait_for(get_input(), timeout=5.0)
                    command = command.strip().lower()

                    if command == 'status':
                        await system.show_status()
                    elif command == 'stop':
                        print("\nStopping tracking...")
                        break
                    elif command == 'quit':
                        print("\nQuitting without generating PDDL...")
                        await system.stop_tracking()
                        system.cleanup()
                        return
                    elif command:
                        print(f"Unknown command: '{command}'")

                except asyncio.TimeoutError:
                    # No input, continue
                    print("\r" + " "*50 + "\r", end="", flush=True)
                    pass

            except KeyboardInterrupt:
                print("\n\nCtrl+C detected, stopping...")
                break

        # Stop tracking
        await system.stop_tracking()

        # Show final status
        await system.show_status()

        # Generate PDDL files if ready
        decision = await system.monitor.determine_state()

        if decision.state == TaskState.PLAN_AND_EXECUTE or system.ready_for_planning:
            print("\nGenerating final PDDL files...")
            await system.generate_pddl_files()

            print("\n✓ PDDL files ready for planner!")
            print("\nNext steps:")
            print("  1. Review the generated PDDL files")
            print("  2. Run a PDDL planner (e.g., FastDownward, LAMA)")
            print("  3. Execute the generated plan")
        else:
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
                await system.generate_pddl_files()

        # Cleanup
        system.cleanup()

        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print()
        print("Summary:")
        print(f"  • Detection cycles: {system.detection_count}")
        print(f"  • Total objects detected: {len(system.tracker.get_all_objects())}")
        print(f"  • Final state: {decision.state.value}")
        print(f"  • PDDL files generated: {'YES' if system.ready_for_planning else 'NO'}")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        await system.stop_tracking()
        system.cleanup()
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        await system.stop_tracking()
        system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
