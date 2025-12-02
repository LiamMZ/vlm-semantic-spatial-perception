"""
Simple Continuous PDDL Integration Demo

A simplified version that runs for a fixed duration or number of detection cycles.
Perfect for testing and demonstrations without complex input handling.

This demo:
1. Analyzes the task and generates initial PDDL domain
2. Starts continuous object detection
3. Updates PDDL domain after each detection
4. Monitors task state
5. Stops after N detection cycles or when ready for planning
6. Generates final PDDL files
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

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


class SimpleContinuousDemo:
    """Simple continuous integration demo with auto-stop."""

    def __init__(
        self,
        api_key: str,
        task_description: str,
        max_cycles: int = 10,
        update_interval: float = 3.0
    ):
        """
        Initialize demo.

        Args:
            api_key: Google AI API key
            task_description: Task to perform
            max_cycles: Maximum detection cycles before stopping
            update_interval: Seconds between detections
        """
        self.api_key = api_key
        self.task_description = task_description
        self.max_cycles = max_cycles
        self.update_interval = update_interval
        self.cycle_count = 0
        self.should_stop = False

        # Initialize camera
        print("🎥 Initializing camera...")
        self.camera = RealSenseCamera(
            width=640,
            height=480,
            fps=30,
            enable_depth=True,
            auto_start=True
        )
        print("✓ Camera ready\n")

        # Initialize PDDL
        print("📋 Initializing PDDL system...")
        self.pddl = PDDLRepresentation(domain_name="robot_task")
        self.maintainer = PDDLDomainMaintainer(self.pddl, api_key=api_key)
        self.monitor = TaskStateMonitor(self.maintainer, self.pddl)
        print("✓ PDDL system ready\n")

        # Initialize tracker
        print("👁️  Initializing object tracker...")
        self.tracker = ContinuousObjectTracker(
            api_key=api_key,
            fast_mode=False,
            update_interval=update_interval,
            on_detection_complete=self._on_detection
        )
        self.tracker.set_frame_provider(self._get_frames)
        print("✓ Tracker ready\n")

        self.ready_for_planning = False

    def _get_frames(self):
        """Get camera frames."""
        color, depth = self.camera.get_aligned_frames()
        intrinsics = self.camera.get_camera_intrinsics()
        return color, depth, intrinsics

    async def _on_detection(self, object_count: int):
        """Called after each detection cycle."""
        self.cycle_count += 1

        print(f"\n{'='*70}")
        print(f"CYCLE {self.cycle_count}/{self.max_cycles} - Detected {object_count} objects")
        print(f"{'='*70}")

        # Get objects
        all_objects = self.tracker.get_all_objects()

        if all_objects:
            # Convert for PDDL
            objects_dict = [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "affordances": list(obj.affordances)
                }
                for obj in all_objects
            ]

            # Update PDDL
            stats = await self.maintainer.update_from_observations(objects_dict)

            print(f"\n📦 PDDL Update:")
            print(f"  • New objects: {stats['objects_added']}")
            print(f"  • Total observations: {stats['total_observations']}")
            print(f"  • Object types: {stats['total_object_types']}")
            print(f"  • Goal objects found: {', '.join(stats['goal_objects_found']) or 'none'}")
            print(f"  • Still missing: {', '.join(stats['goal_objects_missing']) or 'none'}")

            # Check state
            decision = await self.monitor.determine_state()

            print(f"\n🎯 Task State: {decision.state.value} ({decision.confidence:.0%})")
            print(f"  Reasoning: {decision.reasoning}")

            if decision.blockers:
                print(f"\n  ⚠️  Blockers:")
                for blocker in decision.blockers:
                    print(f"    • {blocker}")

            # Check if ready
            if decision.state == TaskState.PLAN_AND_EXECUTE:
                print(f"\n  ✅ READY FOR PLANNING!")
                self.ready_for_planning = True
                self.should_stop = True  # Auto-stop when ready

            # Show objects
            print(f"\n🔍 Detected Objects:")
            by_type = {}
            for obj in all_objects:
                by_type[obj.object_type] = by_type.get(obj.object_type, 0) + 1
            for obj_type, count in sorted(by_type.items()):
                print(f"  • {obj_type}: {count}")

        print(f"{'='*70}\n")

        # Check if should stop
        if self.cycle_count >= self.max_cycles:
            print(f"⏱️  Reached maximum cycles ({self.max_cycles})")
            self.should_stop = True

    async def run(self):
        """Run the demo."""
        try:
            # Step 1: Analyze task
            print(f"\n{'#'*70}")
            print("STEP 1: TASK ANALYSIS")
            print(f"{'#'*70}")
            print(f"\nTask: \"{self.task_description}\"\n")

            color, _ = self.camera.get_aligned_frames()
            analysis = await self.maintainer.initialize_from_task(
                self.task_description,
                environment_image=color
            )

            print(f"✓ Task analyzed:")
            print(f"  • Goal objects: {', '.join(analysis.goal_objects)}")
            print(f"  • Estimated steps: {analysis.estimated_steps}")
            print(f"  • Predicates: {len(analysis.relevant_predicates)}")
            print(f"\n  Key predicates: {', '.join(analysis.relevant_predicates[:5])}")
            if len(analysis.relevant_predicates) > 5:
                print(f"    ... and {len(analysis.relevant_predicates) - 5} more")

            # Seed perception
            self.tracker.set_pddl_predicates(analysis.relevant_predicates)

            # Step 2: Start continuous tracking
            print(f"\n{'#'*70}")
            print("STEP 2: CONTINUOUS TRACKING")
            print(f"{'#'*70}")
            print(f"\nStarting continuous tracking...")
            print(f"  • Update interval: {self.update_interval}s")
            print(f"  • Max cycles: {self.max_cycles}")
            print(f"  • Auto-stop when ready for planning\n")

            self.tracker.start()

            # Wait for tracking to complete
            while not self.should_stop:
                await asyncio.sleep(1.0)

            # Stop tracking
            print(f"\n⏹️  Stopping tracking...")
            await self.tracker.stop()

            # Step 3: Generate PDDL
            print(f"\n{'#'*70}")
            print("STEP 3: GENERATE PDDL FILES")
            print(f"{'#'*70}\n")

            if self.ready_for_planning:
                print("✓ System is ready for planning!")
            else:
                decision = await self.monitor.determine_state()
                print(f"⚠️  Not fully ready (state: {decision.state.value})")
                print(f"   {decision.reasoning}\n")
                print("Generating PDDL anyway for inspection...")

            # Set goals
            await self.maintainer.set_goal_from_task_analysis()

            # Generate files
            output_dir = "outputs/pddl/continuous_demo"
            paths = await self.pddl.generate_files_async(output_dir)

            print(f"\n✓ PDDL files generated:")
            print(f"  • Domain: {paths['domain_path']}")
            print(f"  • Problem: {paths['problem_path']}")

            # Show summary
            domain_snap = await self.pddl.get_domain_snapshot()
            problem_snap = await self.pddl.get_problem_snapshot()

            print(f"\n📊 Domain Summary:")
            print(f"  • Object types: {len(domain_snap['object_types'])}")
            print(f"  • Predicates: {len(domain_snap['predicates'])}")
            print(f"  • Actions: {len(domain_snap['predefined_actions']) + len(domain_snap['llm_generated_actions'])}")

            print(f"\n📊 Problem Summary:")
            print(f"  • Object instances: {len(problem_snap['object_instances'])}")
            print(f"  • Initial state literals: {len(problem_snap['initial_literals'])}")
            print(f"  • Goal literals: {len(problem_snap['goal_literals'])}")

            # Final summary
            print(f"\n{'#'*70}")
            print("DEMO COMPLETE")
            print(f"{'#'*70}\n")

            # Get tracker stats
            tracker_stats = await self.tracker.get_stats()

            print(f"📈 Statistics:")
            print(f"  • Detection cycles: {self.cycle_count}")
            print(f"  • Total frames processed: {tracker_stats.total_frames}")
            print(f"  • Avg detection time: {tracker_stats.avg_detection_time:.2f}s")
            print(f"  • Total objects: {len(self.tracker.get_all_objects())}")
            print(f"  • Ready for planning: {'YES ✅' if self.ready_for_planning else 'NO ⚠️'}")

            if self.ready_for_planning:
                print(f"\n🚀 Next Steps:")
                print(f"  1. Review PDDL files in {output_dir}/")
                print(f"  2. Run PDDL planner (e.g., FastDownward)")
                print(f"  3. Execute the generated plan")
            else:
                print(f"\n💡 Tips:")
                print(f"  • Ensure goal objects are visible to camera")
                print(f"  • Increase max_cycles for more detection attempts")
                print(f"  • Check lighting and camera positioning")

            print()

        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            self.camera.stop()
            print("✓ Done\n")


async def main():
    """Main entry point."""
    print("\n" + "="*70)
    print(" "*15 + "CONTINUOUS PDDL DEMO")
    print("="*70)
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("   Set it in .env file or environment variable")
        return

    # Configuration
    task = "Pick up the red mug and place it on the shelf"
    max_cycles = 5  # Stop after 5 detection cycles
    update_interval = 3.0  # 3 seconds between detections

    print(f"Configuration:")
    print(f"  • Task: \"{task}\"")
    print(f"  • Max cycles: {max_cycles}")
    print(f"  • Update interval: {update_interval}s")
    print(f"  • Total runtime: ~{max_cycles * update_interval}s")
    print()

    input("Press Enter to start (make sure objects are visible to camera)...")

    try:
        demo = SimpleContinuousDemo(
            api_key=api_key,
            task_description=task,
            max_cycles=max_cycles,
            update_interval=update_interval
        )
        await demo.run()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run with proper async event loop
    asyncio.run(main())
