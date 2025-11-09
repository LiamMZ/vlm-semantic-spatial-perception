"""
PDDL Predicate Tracking Demo with Continuous Updates

Demonstrates real-time PDDL generation with continuous object tracking:
1. User provides a natural language task
2. LLM analyzes task to extract relevant PDDL predicates (no keyword matching)
3. PDDL representation is initialized with LLM-extracted predicates
4. Continuous tracker detects objects with those predicates
5. PDDL representation updates in real-time as new objects are detected
6. Final PDDL files are generated for planning

This bridges perception and planning with live state updates.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import numpy as np
import cv2
from dotenv import load_dotenv
from typing import List, Dict

from src.perception import ContinuousObjectTracker, DetectedObject
from src.planning import PDDLRepresentation, LLMTaskAnalyzer
from src.camera import RealSenseCamera

# Load environment
load_dotenv()


class PDDLPredicateTrackingDemo:
    """
    Interactive demo showing continuous PDDL updates from object tracking.
    Uses LLM to dynamically extract predicates from task description.
    """

    def __init__(self, api_key: str):
        """Initialize demo components."""
        self.api_key = api_key
        self.camera = None
        self.tracker = None
        self.pddl = None
        self.task_analyzer = None
        self.task_description = ""
        self.task_analysis = None
        self.predicates = []

        # Stats
        self.update_count = 0
        self.last_object_count = 0

    async def setup(self):
        """Setup camera and analysis components."""
        print("\n" + "=" * 70)
        print("PDDL PREDICATE TRACKING DEMO - SETUP")
        print("=" * 70)
        print()

        # Initialize camera
        print("Step 1: Initializing camera...")
        try:
            self.camera = RealSenseCamera()
            print("  ✓ RealSense camera initialized")
        except Exception as e:
            print(f"  ⚠ Camera initialization failed: {e}")
            print("  → Demo will use simulated data")
            self.camera = None

        # Initialize task analyzer
        print("\nStep 2: Initializing LLM task analyzer...")
        self.task_analyzer = LLMTaskAnalyzer(api_key=self.api_key)
        print("  ✓ Task analyzer ready")

        print()

    def get_task_input(self) -> str:
        """Get task description from user."""
        print("=" * 70)
        print("TASK INPUT")
        print("=" * 70)
        print()
        print("Enter a robotic manipulation task. The LLM will:")
        print("  1. Analyze the task semantically (no keyword matching)")
        print("  2. Extract task-relevant PDDL predicates")
        print("  3. Determine required actions and goals")
        print("  4. Generate complete PDDL domain")
        print()
        print("Example tasks:")
        print('  - "Clean the dirty mug and place it on the shelf"')
        print('  - "Open the container and fill it with water"')
        print('  - "Organize the workspace by stacking items"')
        print('  - "Move the fragile items to a safe location"')
        print()

        task = input("Enter task description: ").strip()

        if not task:
            task = "Clean the dirty mug and place it on the shelf"
            print(f"\n  → Using default: '{task}'")

        return task

    async def analyze_task_with_llm(self, task: str):
        """Use LLM to analyze task and extract predicates."""
        print("\n" + "=" * 70)
        print("LLM TASK ANALYSIS")
        print("=" * 70)
        print()
        print(f"Task: {task}")
        print()

        print("Analyzing task with LLM (predictive analysis before observation)...")
        print()

        # Capture environment image if camera available
        environment_image = None
        if self.camera:
            try:
                print("Capturing environment image for context...")
                color, _ = self.camera.get_aligned_frames()
                environment_image = color
                print(f"  ✓ Environment image captured: {color.shape}")
            except Exception as e:
                print(f"  ⚠ Failed to capture image: {e}")

        # Initial analysis - no objects observed yet
        # LLM will predict what predicates/actions will be needed
        try:
            self.task_analysis = self.task_analyzer.analyze_task(
                task_description=task,
                observed_objects=None,  # No observations yet
                observed_relationships=None,
                environment_image=environment_image,  # Optional visual context
                timeout=15.0
            )

            # Extract predicates from analysis
            self.predicates = self.task_analysis.relevant_predicates

            # If LLM returned empty predicates, use basic fallback
            if not self.predicates:
                print(f"  ⚠ LLM returned no predicates, using task-based inference")
                self.predicates = self._infer_predicates_from_task(task)

            print(f"  → LLM identified {len(self.predicates)} relevant predicates:")
            for pred in self.predicates:
                print(f"      • {pred}")
            print()

            print(f"  → Task complexity: {self.task_analysis.complexity}")
            print(f"  → Estimated steps: {self.task_analysis.estimated_steps}")
            print()

            if self.task_analysis.action_sequence:
                print(f"  → Predicted action sequence ({len(self.task_analysis.action_sequence)} steps):")
                for i, action in enumerate(self.task_analysis.action_sequence, 1):
                    print(f"      {i}. {action}")
                print()

            if self.task_analysis.goal_objects:
                print(f"  → Goal objects: {', '.join(self.task_analysis.goal_objects)}")
            if self.task_analysis.tool_objects:
                print(f"  → Tool objects: {', '.join(self.task_analysis.tool_objects)}")

            if self.task_analysis.required_actions:
                print(f"  → LLM generated {len(self.task_analysis.required_actions)} custom actions")
            print()

        except Exception as e:
            print(f"  ⚠ LLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            print("  → Using task-based inference")
            self.predicates = self._infer_predicates_from_task(task)
            print(f"  → Inferred predicates: {', '.join(self.predicates)}")
            print()

    def _infer_predicates_from_task(self, task: str) -> List[str]:
        """Infer basic predicates from task description."""
        predicates = set()
        task_lower = task.lower()

        # Always include basic manipulation predicates
        predicates.update(["graspable", "reachable", "holding", "empty-hand", "on"])

        # Add task-specific predicates
        if any(word in task_lower for word in ["put", "place", "move"]):
            predicates.update(["at", "in", "inside"])

        if any(word in task_lower for word in ["clean", "dirty", "wash"]):
            predicates.update(["clean", "dirty", "wet"])

        if any(word in task_lower for word in ["open", "close", "container", "box"]):
            predicates.update(["opened", "closed", "containable"])

        if any(word in task_lower for word in ["fill", "empty", "pour"]):
            predicates.update(["filled", "empty"])

        if any(word in task_lower for word in ["stack", "organize"]):
            predicates.update(["stacked", "clear", "above", "below"])

        return sorted(list(predicates))

    async def setup_pddl(self):
        """Initialize PDDL representation BEFORE tracker."""
        print("=" * 70)
        print("PDDL INITIALIZATION (BEFORE TRACKER)")
        print("=" * 70)
        print()

        print("Creating PDDL representation with LLM-extracted predicates...")
        self.pddl = PDDLRepresentation(
            domain_name="perception_driven_task",
            problem_name="continuous_tracking_task"
        )
        self.pddl.task_description = self.task_description

        # Add predicates to domain from LLM analysis
        print("  Adding predicates to PDDL domain:")
        for pred in self.predicates:
            # Most predicates are unary (apply to single object)
            self.pddl.add_predicate(pred, [("obj", "object")])
            print(f"    + {pred}(obj - object)")

        # Add common robot predicates
        self.pddl.add_predicate("empty-hand", [])
        self.pddl.add_predicate("holding", [("obj", "object")])

        # If LLM provided required actions, add them
        if self.task_analysis and self.task_analysis.required_actions:
            print()
            print("  Adding LLM-generated actions:")
            for action in self.task_analysis.required_actions:
                try:
                    # Parse parameters
                    params = []
                    if "parameters" in action:
                        param_str = action["parameters"]
                        if isinstance(param_str, list):
                            # Already parsed
                            for p in param_str:
                                if " - " in p:
                                    name, type_ = p.split(" - ")
                                    params.append((name.strip("?"), type_.strip()))
                        elif isinstance(param_str, str):
                            # Parse string format
                            for p in param_str.split(","):
                                p = p.strip()
                                if " - " in p:
                                    name, type_ = p.split(" - ")
                                    params.append((name.strip("?"), type_.strip()))

                    self.pddl.add_llm_generated_action(
                        name=action.get("name", "unknown"),
                        parameters=params,
                        precondition=action.get("precondition", ""),
                        effect=action.get("effect", ""),
                        description=f"LLM-generated action for task"
                    )
                    print(f"    + {action.get('name', 'unknown')}")
                except Exception as e:
                    print(f"    ⚠ Failed to add action {action.get('name', 'unknown')}: {e}")

        print()
        print("  ✓ PDDL domain initialized with LLM-extracted components")
        print()

    async def setup_tracker(self):
        """Setup continuous tracker with PDDL predicates."""
        print("=" * 70)
        print("TRACKER SETUP (SEEDED WITH PDDL PREDICATES)")
        print("=" * 70)
        print()

        print("Initializing continuous object tracker...")
        self.tracker = ContinuousObjectTracker(
            api_key=self.api_key,
            model_name="auto",
            fast_mode=True,  # Fast mode for demo
            update_interval=2.0,  # Update every 2 seconds
            on_detection_complete=self.on_objects_detected
        )

        # Seed tracker with predicates from PDDL (extracted by LLM)
        self.tracker.tracker.set_pddl_predicates(self.predicates)

        print(f"  ✓ Tracker seeded with {len(self.predicates)} PDDL predicates")
        print(f"    Predicates: {', '.join(self.predicates)}")
        print()

    async def on_objects_detected(self, count: int):
        """Callback when objects are detected - updates PDDL."""
        print(f"  [DEBUG] Callback triggered with count={count}")

        if count == self.last_object_count:
            print(f"  [DEBUG] Count unchanged ({count}), skipping update")
            return  # No change

        self.last_object_count = count
        self.update_count += 1

        print(f"\n{'─' * 70}")
        print(f"UPDATE #{self.update_count}: {count} objects detected")
        print(f"{'─' * 70}")

        # Get all detected objects
        objects = self.tracker.get_all_objects()
        print(f"  [DEBUG] Retrieved {len(objects)} objects from registry")

        # Update PDDL representation
        await self.update_pddl_from_objects(objects)

        print()

    async def update_pddl_from_objects(self, objects: List[DetectedObject]):
        """Update PDDL initial state from detected objects."""
        if not objects:
            return

        # Clear previous initial state
        self.pddl.clear_initial_state()

        print("Updating PDDL initial state:")
        print()

        # Add objects and their states
        for obj in objects:
            # Ensure object type exists
            if obj.object_type not in self.pddl.object_types:
                self.pddl.add_object_type(obj.object_type, parent="object")

            # Ensure object instance exists
            if obj.object_id not in self.pddl.object_instances:
                self.pddl.add_object_instance(obj.object_id, obj.object_type)
                print(f"  + Object: {obj.object_id} ({obj.object_type})")

            # Add PDDL state predicates (closed world assumption - only true predicates)
            if hasattr(obj, 'pddl_state') and obj.pddl_state:
                for predicate, value in obj.pddl_state.items():
                    if value:  # Only add true predicates
                        self.pddl.add_initial_literal(predicate, [obj.object_id])
                        print(f"      • ({predicate} {obj.object_id})")

        # Add robot state
        self.pddl.add_initial_literal("empty-hand", [])

        print()
        print(f"  ✓ Initial state updated: {len(self.pddl.initial_literals)} literals")

    async def run_continuous_tracking(self, duration: float = 10.0):
        """Run continuous tracking for specified duration."""
        print("=" * 70)
        print("CONTINUOUS TRACKING")
        print("=" * 70)
        print()
        print(f"Starting continuous object detection for {duration:.0f} seconds...")
        print("The PDDL state will update as new objects are detected.")
        print()

        if self.camera:
            # Real camera tracking
            def frame_provider():
                print(f"  [DEBUG] Frame provider called")
                try:
                    color, depth = self.camera.get_aligned_frames()
                    intrinsics = self.camera.get_camera_intrinsics()
                    print(f"  [DEBUG] Got frames: color={color.shape if hasattr(color, 'shape') else type(color)}, depth={depth.shape if hasattr(depth, 'shape') else type(depth)}")
                    return color, depth, intrinsics
                except Exception as e:
                    print(f"  [DEBUG] Frame provider error: {e}")
                    raise

            self.tracker.set_frame_provider(frame_provider)

            # Show camera view
            print("Camera preview (press any key to continue):")
            color, _ = self.camera.get_aligned_frames()
            cv2.imshow("Camera View", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.waitKey(2000)
            cv2.destroyWindow("Camera View")
        else:
            # Simulated tracking
            print("⚠ Using simulated object detection (no camera available)")
            print()

            # Create simulated objects that appear over time
            simulated_objects = self._create_simulated_objects()

            async def simulate_detection():
                """Simulate gradual object detection."""
                for i, obj in enumerate(simulated_objects):
                    # Add object to tracker's registry
                    self.tracker.tracker.registry.add_or_update(obj)

                    # Trigger callback
                    await self.on_objects_detected(i + 1)

                    # Wait before next detection
                    await asyncio.sleep(duration / len(simulated_objects))

            # Run simulation instead of real tracking
            await simulate_detection()
            return

        # Start continuous tracking
        self.tracker.start()

        # Wait for tracking duration with status updates
        print(f"Tracking for {duration:.0f} seconds (Ctrl+C to stop early)...")
        print()

        start_time = asyncio.get_event_loop().time()
        try:
            while (asyncio.get_event_loop().time() - start_time) < duration:
                await asyncio.sleep(1.0)
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = duration - elapsed

                # Get current stats
                stats = await self.tracker.get_stats()

                if remaining > 0:
                    print(f"  ⏱  {elapsed:.0f}s elapsed, {remaining:.0f}s remaining... ({self.update_count} updates, {stats.total_frames} frames, {stats.total_detections} objects)")
        except KeyboardInterrupt:
            print("\n  → Interrupted by user")

        # Stop tracking
        print()
        print("Stopping continuous tracker...")
        await self.tracker.stop()

        # Get final stats
        final_stats = await self.tracker.get_stats()
        print()
        print(f"  ✓ Tracking complete: {self.update_count} updates, {self.last_object_count} objects detected")
        print(f"  ✓ Statistics: {final_stats.total_frames} frames processed, {final_stats.total_detections} total detections")

    def _create_simulated_objects(self) -> List[DetectedObject]:
        """Create simulated objects for demo."""
        objects = [
            DetectedObject(
                object_type="mug",
                object_id="coffee_mug_1",
                affordances={"graspable", "containable"},
                position_2d=[450, 320],
                properties={"color": "white"},
                pddl_state={
                    pred: False for pred in self.predicates
                }
            ),
            DetectedObject(
                object_type="sink",
                object_id="kitchen_sink_1",
                affordances={"containable"},
                position_2d=[600, 400],
                properties={"has_faucet": True},
                pddl_state={
                    pred: False for pred in self.predicates
                }
            ),
            DetectedObject(
                object_type="shelf",
                object_id="shelf_1",
                affordances={"supportable"},
                position_2d=[300, 200],
                properties={"material": "wood"},
                pddl_state={
                    pred: False for pred in self.predicates
                }
            ),
        ]

        # Set predicates based on LLM analysis
        # For mug
        if "dirty" in self.predicates:
            objects[0].pddl_state["dirty"] = True
        if "graspable" in self.predicates:
            objects[0].pddl_state["graspable"] = True
        if "reachable" in self.predicates:
            objects[0].pddl_state["reachable"] = True

        # For sink
        if "clean" in self.predicates:
            objects[1].pddl_state["clean"] = True
        if "reachable" in self.predicates:
            objects[1].pddl_state["reachable"] = True

        # For shelf
        if "clean" in self.predicates:
            objects[2].pddl_state["clean"] = True
        if "reachable" in self.predicates:
            objects[2].pddl_state["reachable"] = True
        if "clear" in self.predicates:
            objects[2].pddl_state["clear"] = True

        return objects

    async def set_goal_state(self):
        """Set goal state from LLM analysis."""
        print("=" * 70)
        print("GOAL STATE (FROM LLM ANALYSIS)")
        print("=" * 70)
        print()

        print("Setting goal state from LLM task analysis...")
        print()

        if self.task_analysis and self.task_analysis.goal_predicates:
            # Use LLM-provided goals
            print("  LLM-provided goal predicates:")
            for goal_pred in self.task_analysis.goal_predicates:
                print(f"    • {goal_pred}")

                # Parse goal predicate (simple parsing)
                # Format: "predicate(arg1, arg2)" or "predicate(arg1)"
                try:
                    if "(" in goal_pred and ")" in goal_pred:
                        pred_name = goal_pred[:goal_pred.index("(")]
                        args_str = goal_pred[goal_pred.index("(")+1:goal_pred.index(")")]
                        args = [arg.strip() for arg in args_str.split(",")]

                        # Check if predicate should be negated
                        negated = False
                        if pred_name.startswith("not ") or pred_name.startswith("¬"):
                            negated = True
                            pred_name = pred_name.replace("not ", "").replace("¬", "").strip()

                        # Add to PDDL
                        self.pddl.add_goal_literal(pred_name, args, negated=negated)
                        print(f"      → Added: {'(not ' if negated else ''}({pred_name} {' '.join(args)}){')' if negated else ''}")

                except Exception as e:
                    print(f"      ⚠ Failed to parse: {e}")
        else:
            # Fallback: extract goals from objects and task
            print("  No LLM goals available, using heuristic extraction...")
            objects = self.pddl.object_instances

            # Look for objects mentioned in goal_objects from analysis
            if self.task_analysis and self.task_analysis.goal_objects:
                goal_obj_types = self.task_analysis.goal_objects
                print(f"    Target object types: {', '.join(goal_obj_types)}")

                for obj_id in objects:
                    for goal_type in goal_obj_types:
                        if goal_type.lower() in obj_id.lower():
                            # Add a generic goal
                            if "clean" in self.predicates:
                                self.pddl.add_goal_literal("clean", [obj_id])
                                print(f"    + Goal: (clean {obj_id})")
                            if "reachable" in self.predicates:
                                self.pddl.add_goal_literal("reachable", [obj_id])
                                print(f"    + Goal: (reachable {obj_id})")

        if not self.pddl.goal_literals:
            # Ultimate fallback
            print("  ℹ No specific goals extracted, adding default goal")
            if self.pddl.object_instances:
                first_obj = next(iter(self.pddl.object_instances.keys()))
                if "graspable" in self.predicates:
                    self.pddl.add_goal_literal("graspable", [first_obj])
                    print(f"    + Goal: (graspable {first_obj})")

        print()

    async def generate_final_pddl(self):
        """Generate final PDDL files."""
        print("=" * 70)
        print("FINAL PDDL GENERATION")
        print("=" * 70)
        print()

        # Set goal state
        await self.set_goal_state()

        # Validate
        print("Validating PDDL representation...")
        goal_valid, goal_issues = self.pddl.validate_goal_completeness()
        action_valid, action_issues = self.pddl.validate_action_completeness()

        print(f"  Goal completeness: {'✓' if goal_valid else '⚠'}")
        if goal_issues:
            for issue in goal_issues:
                print(f"    • {issue}")

        print(f"  Action completeness: {'✓' if action_valid else '⚠'}")
        if action_issues:
            for issue in action_issues:
                print(f"    • {issue}")
        print()

        # Statistics
        stats = self.pddl.get_statistics()
        print("PDDL Statistics:")
        print(f"  Domain:")
        print(f"    • Types: {stats['domain']['types']}")
        print(f"    • Predicates: {stats['domain']['predicates']}")
        print(f"    • Predefined actions: {stats['domain']['predefined_actions']}")
        print(f"    • LLM-generated actions: {stats['domain']['llm_generated_actions']}")
        print(f"    • Total actions: {stats['domain']['total_actions']}")
        print(f"  Problem:")
        print(f"    • Objects: {stats['problem']['object_instances']}")
        print(f"    • Initial literals: {stats['problem']['initial_literals']}")
        print(f"    • Goal literals: {stats['problem']['goal_literals']}")
        print()

        # Generate files
        print("Generating PDDL files...")
        output_dir = "outputs/pddl/continuous_tracking_demo"
        paths = self.pddl.generate_files(output_dir)

        print(f"  ✓ Domain: {paths['domain_path']}")
        print(f"  ✓ Problem: {paths['problem_path']}")
        print()

        # Show initial state
        print("Generated Initial State:")
        print("  (:init")
        for literal in sorted(self.pddl.initial_literals, key=lambda x: (x.predicate, tuple(x.arguments))):
            print(f"    {literal.to_pddl()}")
        print("  )")
        print()

        # Show goal state
        print("Generated Goal State:")
        print("  (:goal")
        print("    (and")
        for literal in sorted(self.pddl.goal_literals, key=lambda x: (x.predicate, tuple(x.arguments))):
            print(f"      {literal.to_pddl()}")
        print("    )")
        print("  )")
        print()

    async def run(self):
        """Run the complete demo."""
        try:
            # Setup
            await self.setup()

            # Get task from user
            self.task_description = self.get_task_input()

            # Analyze task with LLM (extracts predicates dynamically)
            await self.analyze_task_with_llm(self.task_description)

            # Initialize PDDL FIRST (before tracker)
            await self.setup_pddl()

            # Setup tracker and seed with PDDL predicates
            await self.setup_tracker()

            # Run continuous tracking
            await self.run_continuous_tracking(duration=10.0)

            # Generate final PDDL
            await self.generate_final_pddl()

            # Done
            print("=" * 70)
            print("DEMO COMPLETE")
            print("=" * 70)
            print()
            print("Key takeaways:")
            print("  1. LLM analyzes task to extract relevant predicates (no keywords)")
            print("  2. PDDL initialized BEFORE tracker with LLM predicates")
            print("  3. Tracker seeded with PDDL predicates for detection")
            print("  4. PDDL representation updates continuously as objects detected")
            print("  5. Final PDDL files ready for planning")
            print()
            print("Next steps:")
            print("  • Use generated PDDL with a planner (FastDownward, etc.)")
            print("  • Execute planned actions with robot")
            print("  • Continue tracking to monitor state changes")

        except KeyboardInterrupt:
            print("\n\n⚠ Demo interrupted by user")
        except Exception as e:
            print(f"\n\n⚠ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            print("\nCleaning up resources...")
            if self.tracker and self.tracker.is_running():
                await self.tracker.stop()
            # Close async client to prevent warnings
            if self.tracker and hasattr(self.tracker.tracker, 'aclose'):
                await self.tracker.tracker.aclose()
            if self.camera:
                self.camera.stop()
            cv2.destroyAllWindows()
            print("✓ Cleanup complete")


async def main():
    """Run demo."""
    print("\n")
    print("*" * 70)
    print("PDDL PREDICATE TRACKING WITH CONTINUOUS UPDATES")
    print("*" * 70)
    print()

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print()
        print("Please set your API key:")
        print("  export GEMINI_API_KEY='your_key_here'")
        print("or add it to .env file")
        return

    # Run demo
    demo = PDDLPredicateTrackingDemo(api_key)
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
