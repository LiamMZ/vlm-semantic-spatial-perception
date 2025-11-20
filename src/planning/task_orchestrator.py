"""
Task Orchestrator

Main orchestration class for managing the complete perception-planning pipeline.
Handles task requests, continuous detection, PDDL domain management, and state persistence.

This class is optimized for actual execution and provides:
- State persistence (load/save domain, problem, and object registry)
- Task request handling with domain updates
- Continuous detection management
- Task status monitoring
- Lifecycle management (init, run, pause, resume, shutdown)
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum
import json

import numpy as np
from PIL import Image

from .pddl_representation import PDDLRepresentation
from .pddl_domain_maintainer import PDDLDomainMaintainer
from .task_state_monitor import TaskStateMonitor, TaskState, TaskStateDecision
from .llm_task_analyzer import TaskAnalysis
from ..perception import ContinuousObjectTracker
from ..perception.object_registry import DetectedObjectRegistry, DetectedObject
from ..camera import RealSenseCamera

# Import config from config directory
import sys
config_path = Path(__file__).parent.parent.parent / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
from orchestrator_config import OrchestratorConfig


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""
    UNINITIALIZED = "uninitialized"
    IDLE = "idle"  # Ready for task requests
    ANALYZING_TASK = "analyzing_task"  # Processing task description
    DETECTING = "detecting"  # Running continuous detection
    PAUSED = "paused"  # Detection paused, can resume
    READY_FOR_PLANNING = "ready_for_planning"  # Sufficient observations, ready for plan
    EXECUTING_PLAN = "executing_plan"  # Plan is being executed
    TASK_COMPLETE = "task_complete"  # Task successfully completed
    ERROR = "error"  # Error state


class TaskOrchestrator:
    """
    Main orchestration class for the perception-planning pipeline.

    Manages the complete lifecycle from task request to execution, including:
    - Camera and perception system initialization
    - PDDL domain and problem management
    - Continuous object detection
    - Task state monitoring
    - State persistence and recovery

    Example:
        >>> # Initialize orchestrator
        >>> config = OrchestratorConfig(
        ...     api_key=os.getenv("GEMINI_API_KEY"),
        ...     update_interval=2.0,
        ...     min_observations=3
        ... )
        >>> orchestrator = TaskOrchestrator(config)
        >>>
        >>> # Start with a task
        >>> await orchestrator.initialize()
        >>> await orchestrator.process_task_request("make a cup of coffee")
        >>>
        >>> # Start continuous detection
        >>> await orchestrator.start_detection()
        >>>
        >>> # Monitor status
        >>> status = await orchestrator.get_status()
        >>> print(f"State: {status['state']}, Objects: {status['num_objects']}")
        >>>
        >>> # When ready, generate PDDL
        >>> if orchestrator.is_ready_for_planning():
        ...     paths = await orchestrator.generate_pddl_files()
        >>>
        >>> # Cleanup
        >>> await orchestrator.shutdown()
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        camera: Optional[RealSenseCamera] = None,
        pddl_representation: Optional[PDDLRepresentation] = None
    ):
        """
        Initialize the task orchestrator.

        Args:
            config: Orchestrator configuration
            camera: Optional RealSense camera (will create if None)
            pddl_representation: Optional PDDL representation (will create if None)
        """
        self.config = config
        self._state = OrchestratorState.UNINITIALIZED
        self._camera = camera
        self._external_camera = camera is not None  # Track if camera is externally managed

        # Core components (initialized in initialize())
        self.pddl: Optional[PDDLRepresentation] = pddl_representation
        self.maintainer: Optional[PDDLDomainMaintainer] = None
        self.monitor: Optional[TaskStateMonitor] = None
        self.tracker: Optional[ContinuousObjectTracker] = None
        self.registry: DetectedObjectRegistry = DetectedObjectRegistry()

        # Task state
        self.current_task: Optional[str] = None
        self.task_analysis: Optional[TaskAnalysis] = None
        self.last_task_decision: Optional[TaskStateDecision] = None

        # Detection tracking
        self.detection_count: int = 0
        self.last_detection_time: float = 0.0
        self._detection_running: bool = False

        # Auto-save management
        self._auto_save_task: Optional[asyncio.Task] = None
        self._last_save_time: float = 0.0

        # Ensure state directory exists
        self.config.state_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """
        Initialize the orchestrator and all subsystems.

        This must be called before using the orchestrator.
        """
        if self._state != OrchestratorState.UNINITIALIZED:
            print(f"⚠ Orchestrator already initialized (state: {self._state.value})")
            return

        print("Initializing Task Orchestrator...")

        # Initialize camera if not provided
        if self._camera is None:
            print(f"  • Initializing RealSense camera ({self.config.camera_width}x{self.config.camera_height} @ {self.config.camera_fps} FPS)...")
            self._camera = RealSenseCamera(
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                enable_depth=self.config.enable_depth,
                auto_start=True
            )
            print("    ✓ Camera initialized")
        else:
            print("  • Using provided camera")

        # Initialize PDDL representation if not provided
        if self.pddl is None:
            self.pddl = PDDLRepresentation(
                domain_name="task_execution",
                problem_name="current_task"
            )
            print("  • PDDL representation created")

        # Initialize PDDL components
        self.maintainer = PDDLDomainMaintainer(
            self.pddl,
            api_key=self.config.api_key,
            model_name=self.config.model_name
        )

        self.monitor = TaskStateMonitor(
            self.maintainer,
            self.pddl,
            min_observations_before_planning=self.config.min_observations,
            exploration_timeout_seconds=self.config.exploration_timeout
        )
        print("  • PDDL domain maintainer and monitor initialized")

        # Initialize continuous tracker
        self.tracker = ContinuousObjectTracker(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            fast_mode=self.config.fast_mode,
            update_interval=self.config.update_interval,
            on_detection_complete=self._on_detection_callback,
            scene_change_threshold=self.config.scene_change_threshold,
            enable_scene_change_detection=self.config.enable_scene_change_detection
        )

        # Set frame provider
        self.tracker.set_frame_provider(self._get_camera_frames)
        print("  • Continuous object tracker initialized")

        # Start auto-save if enabled
        if self.config.auto_save:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
            print(f"  • Auto-save enabled (interval: {self.config.auto_save_interval}s)")

        self._set_state(OrchestratorState.IDLE)
        print("✓ Task Orchestrator initialized and ready\n")

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        print("\nShutting down Task Orchestrator...")

        # Stop detection if running
        if self._detection_running:
            await self.stop_detection()

        # Stop auto-save
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass

        # Final save
        if self.config.auto_save:
            await self.save_state()

        # Stop camera if we created it
        if self._camera and not self._external_camera:
            self._camera.stop()
            print("  • Camera stopped")

        self._set_state(OrchestratorState.UNINITIALIZED)
        print("✓ Task Orchestrator shutdown complete")

    # ========================================================================
    # Task Management
    # ========================================================================

    async def process_task_request(
        self,
        task_description: str,
        environment_image: Optional[np.ndarray] = None
    ) -> TaskAnalysis:
        """
        Process a new task request.

        Analyzes the task, initializes the PDDL domain, and prepares for detection.

        Args:
            task_description: Natural language task description
            environment_image: Optional environment image for context (will capture if None)

        Returns:
            TaskAnalysis with predicted requirements
        """
        if self._state == OrchestratorState.UNINITIALIZED:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        self._set_state(OrchestratorState.ANALYZING_TASK)
        print(f"\n{'='*70}")
        print("PROCESSING TASK REQUEST")
        print(f"{'='*70}")
        print(f"Task: \"{task_description}\"")
        print()

        self.current_task = task_description

        # Capture environment image if not provided
        if environment_image is None and self._camera:
            print("  • Capturing environment frame for context...")
            environment_image, _ = self._camera.get_aligned_frames()

        # Analyze task and initialize domain
        print("  • Analyzing task with LLM...")
        self.task_analysis = await self.maintainer.initialize_from_task(
            task_description,
            environment_image=environment_image
        )

        print(f"\n✓ Task analyzed!")
        print(f"  • Goal objects: {', '.join(self.task_analysis.goal_objects)}")
        print(f"  • Estimated steps: {self.task_analysis.estimated_steps}")
        print(f"  • Complexity: {self.task_analysis.complexity}")
        print(f"  • Required predicates: {len(self.task_analysis.relevant_predicates)}")

        # Seed perception with predicates
        if self.tracker:
            print(f"\n  • Configuring perception with {len(self.task_analysis.relevant_predicates)} predicates...")
            self.tracker.tracker.set_pddl_predicates(self.task_analysis.relevant_predicates)

        print(f"\n{'='*70}\n")

        self._set_state(OrchestratorState.IDLE)
        return self.task_analysis

    async def update_task(self, new_task_description: str) -> TaskAnalysis:
        """
        Update the current task without resetting observations.

        Args:
            new_task_description: New task description

        Returns:
            Updated TaskAnalysis
        """
        # Stop detection during task update
        was_detecting = self._detection_running
        if was_detecting:
            await self.stop_detection()

        # Get current observations
        all_objects = self.tracker.get_all_objects() if self.tracker else []

        # Process new task
        result = await self.process_task_request(new_task_description)

        # Re-process existing observations with new task context
        if all_objects:
            print(f"\n  • Re-processing {len(all_objects)} existing objects with new task...")
            objects_dict = self._convert_objects_to_dict(all_objects)
            await self.maintainer.update_from_observations(objects_dict)

        # Resume detection if it was running
        if was_detecting:
            await self.start_detection()

        return result

    # ========================================================================
    # Detection Management
    # ========================================================================

    async def start_detection(self) -> None:
        """
        Start continuous object detection.

        Requires a task to be set via process_task_request() first.
        """
        if self._state == OrchestratorState.UNINITIALIZED:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        if self.current_task is None:
            raise RuntimeError("No task set. Call process_task_request() first.")

        if self._detection_running:
            print("⚠ Detection already running")
            return

        print(f"\n{'='*70}")
        print("STARTING CONTINUOUS DETECTION")
        print(f"{'='*70}")
        print(f"Update interval: {self.config.update_interval}s")
        print(f"Minimum observations: {self.config.min_observations}")
        print()

        self.tracker.start()
        self._detection_running = True
        self._set_state(OrchestratorState.DETECTING)

        print("✓ Continuous detection started")
        print(f"{'='*70}\n")

    async def stop_detection(self) -> None:
        """
        Stop continuous object detection.
        """
        if not self._detection_running:
            return

        print("\nStopping continuous detection...")
        await self.tracker.stop()
        self._detection_running = False
        self._set_state(OrchestratorState.IDLE)
        print("✓ Detection stopped")

    async def pause_detection(self) -> None:
        """
        Pause detection (can be resumed later).
        """
        if not self._detection_running:
            return

        await self.stop_detection()
        self._set_state(OrchestratorState.PAUSED)
        print("✓ Detection paused")

    async def resume_detection(self) -> None:
        """
        Resume paused detection.
        """
        if self._state != OrchestratorState.PAUSED:
            print(f"⚠ Cannot resume from state: {self._state.value}")
            return

        await self.start_detection()

    def _get_camera_frames(self):
        """Frame provider for continuous tracker."""
        try:
            color, depth = self._camera.get_aligned_frames()
            intrinsics = self._camera.get_camera_intrinsics()
            return color, depth, intrinsics
        except Exception as e:
            print(f"⚠ Camera error: {e}")
            return None, None, None

    async def _on_detection_callback(self, object_count: int):
        """
        Called after each detection cycle.

        Updates PDDL domain and checks task state.
        """
        self.detection_count += 1
        self.last_detection_time = time.time()

        # Get all detected objects
        all_objects = self.tracker.get_all_objects()

        # Update registry
        for obj in all_objects:
            self.registry.add_object(obj)

        # Convert to dict format for PDDL maintainer
        objects_dict = self._convert_objects_to_dict(all_objects)

        # Update PDDL domain
        update_stats = await self.maintainer.update_from_observations(objects_dict)

        # Check task state
        decision = await self.monitor.determine_state()

        # Update last decision
        prev_decision = self.last_task_decision
        self.last_task_decision = decision

        # Notify on state change
        if prev_decision is None or prev_decision.state != decision.state:
            if self.config.on_task_state_change:
                self.config.on_task_state_change(decision)

            # Update orchestrator state if ready for planning
            if decision.state == TaskState.PLAN_AND_EXECUTE:
                self._set_state(OrchestratorState.READY_FOR_PLANNING)

        # Notify detection update callback
        if self.config.on_detection_update:
            self.config.on_detection_update(object_count)

    def _convert_objects_to_dict(self, objects: List[DetectedObject]) -> List[Dict]:
        """Convert DetectedObject list to dict format for PDDL maintainer."""
        return [
            {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "affordances": list(obj.affordances),
                "pddl_state": obj.pddl_state,
                "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None
            }
            for obj in objects
        ]

    # ========================================================================
    # Status and Monitoring
    # ========================================================================

    def is_ready_for_planning(self) -> bool:
        """Check if system is ready for PDDL planning."""
        return (
            self._state == OrchestratorState.READY_FOR_PLANNING or
            (self.last_task_decision and
             self.last_task_decision.state == TaskState.PLAN_AND_EXECUTE)
        )

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the orchestrator.

        Returns:
            Dict with current state, statistics, and task status
        """
        status = {
            "orchestrator_state": self._state.value,
            "current_task": self.current_task,
            "detection_running": self._detection_running,
            "detection_count": self.detection_count,
            "last_detection_time": self.last_detection_time,
            "ready_for_planning": self.is_ready_for_planning(),
        }

        # Add tracker stats if available
        if self.tracker:
            tracker_stats = await self.tracker.get_stats()
            status["tracker"] = {
                "total_frames": tracker_stats.total_frames,
                "total_detections": tracker_stats.total_detections,
                "skipped_frames": tracker_stats.skipped_frames,
                "cache_hit_rate": tracker_stats.cache_hit_rate,
                "avg_detection_time": tracker_stats.avg_detection_time,
            }

        # Add registry stats
        status["registry"] = {
            "num_objects": len(self.registry),
            "object_types": list(set(obj.object_type for obj in self.registry.get_all_objects())),
        }

        # Add PDDL domain stats
        if self.maintainer:
            domain_stats = await self.maintainer.get_domain_statistics()
            status["domain"] = domain_stats

        # Add task state decision
        if self.last_task_decision:
            status["task_state"] = {
                "state": self.last_task_decision.state.value,
                "confidence": self.last_task_decision.confidence,
                "reasoning": self.last_task_decision.reasoning,
                "blockers": self.last_task_decision.blockers,
                "recommendations": self.last_task_decision.recommendations,
            }

        return status

    async def get_task_decision(self) -> Optional[TaskStateDecision]:
        """
        Get current task state decision.

        Returns:
            Current TaskStateDecision or None if not available
        """
        if self.monitor:
            return await self.monitor.determine_state()
        return None

    def get_detected_objects(self) -> List[DetectedObject]:
        """Get all detected objects from registry."""
        return self.registry.get_all_objects()

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Get objects of a specific type."""
        return self.registry.get_objects_by_type(object_type)

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """Get objects with a specific affordance."""
        return self.registry.get_objects_with_affordance(affordance)

    # ========================================================================
    # PDDL Generation
    # ========================================================================

    async def generate_pddl_files(
        self,
        output_dir: Optional[Path] = None,
        set_goals: bool = True
    ) -> Dict[str, str]:
        """
        Generate PDDL domain and problem files.

        Args:
            output_dir: Directory to save files (defaults to config.state_dir / "pddl")
            set_goals: Whether to set goals from task analysis

        Returns:
            Dict with paths to generated files
        """
        if self.pddl is None or self.maintainer is None:
            raise RuntimeError("PDDL system not initialized")

        if output_dir is None:
            output_dir = self.config.state_dir / "pddl"

        output_dir = Path(output_dir)

        print(f"\n{'='*70}")
        print("GENERATING PDDL FILES")
        print(f"{'='*70}")

        # Set goals if requested
        if set_goals:
            print("  • Setting goal state from task analysis...")
            await self.maintainer.set_goal_from_task_analysis()

        # Generate files
        print(f"  • Generating files to {output_dir}...")
        paths = await self.pddl.generate_files_async(str(output_dir))

        print(f"\n✓ PDDL files generated:")
        print(f"  • Domain: {paths['domain_path']}")
        print(f"  • Problem: {paths['problem_path']}")

        # Show summary
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        print(f"\nDomain Summary:")
        print(f"  • Types: {len(domain_snapshot['object_types'])}")
        print(f"  • Predicates: {len(domain_snapshot['predicates'])}")
        print(f"  • Actions: {len(domain_snapshot['predefined_actions']) + len(domain_snapshot['llm_generated_actions'])}")

        print(f"\nProblem Summary:")
        print(f"  • Objects: {len(problem_snapshot['object_instances'])}")
        print(f"  • Initial literals: {len(problem_snapshot['initial_literals'])}")
        print(f"  • Goal literals: {len(problem_snapshot['goal_literals'])}")

        print(f"\n{'='*70}\n")

        return paths

    # ========================================================================
    # State Persistence
    # ========================================================================

    async def save_state(self, path: Optional[Path] = None) -> Path:
        """
        Save orchestrator state to disk.

        Saves:
        - PDDL domain and problem
        - Object registry
        - Task information
        - Orchestrator metadata

        Args:
            path: Path to save state (defaults to config.state_dir / "state.json")

        Returns:
            Path to saved state file
        """
        if path is None:
            path = self.config.state_dir / "state.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save object registry
        registry_path = path.parent / "registry.json"
        self.registry.save_to_json(str(registry_path), include_timestamp=False)

        # Save PDDL files
        pddl_dir = path.parent / "pddl"
        if self.pddl:
            await self.pddl.generate_files_async(str(pddl_dir))

        # Save orchestrator state
        state_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "orchestrator_state": self._state.value,
            "current_task": self.current_task,
            "detection_count": self.detection_count,
            "task_analysis": {
                "goal_objects": self.task_analysis.goal_objects if self.task_analysis else [],
                "relevant_predicates": self.task_analysis.relevant_predicates if self.task_analysis else [],
                "estimated_steps": self.task_analysis.estimated_steps if self.task_analysis else 0,
                "complexity": self.task_analysis.complexity if self.task_analysis else "unknown",
            } if self.task_analysis else None,
            "files": {
                "registry": str(registry_path),
                "domain": str(pddl_dir / f"{self.pddl.domain_name}.pddl") if self.pddl else None,
                "problem": str(pddl_dir / f"{self.pddl.problem_name}.pddl") if self.pddl else None,
            }
        }

        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2)

        self._last_save_time = time.time()
        print(f"✓ State saved to {path}")

        return path

    async def load_state(self, path: Optional[Path] = None) -> None:
        """
        Load orchestrator state from disk.

        Args:
            path: Path to state file (defaults to config.state_dir / "state.json")
        """
        if path is None:
            path = self.config.state_dir / "state.json"

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        print(f"Loading state from {path}...")

        with open(path, 'r') as f:
            state_data = json.load(f)

        # Load object registry
        registry_path = state_data["files"]["registry"]
        if Path(registry_path).exists():
            self.registry.load_from_json(registry_path)
            print(f"  • Loaded {len(self.registry)} objects")

        # Load task information
        self.current_task = state_data.get("current_task")
        self.detection_count = state_data.get("detection_count", 0)

        # Restore task analysis if available
        task_analysis_data = state_data.get("task_analysis")
        if task_analysis_data and self.current_task:
            # Re-analyze task to get full TaskAnalysis object
            print(f"  • Re-analyzing task: '{self.current_task}'")
            await self.process_task_request(self.current_task)

            # Re-process existing observations
            all_objects = self.registry.get_all_objects()
            if all_objects:
                print(f"  • Re-processing {len(all_objects)} objects...")
                objects_dict = self._convert_objects_to_dict(all_objects)
                await self.maintainer.update_from_observations(objects_dict)

        print(f"✓ State loaded successfully")

    async def _auto_save_loop(self):
        """Background task for auto-saving state."""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval)

                # Only save if we have meaningful state
                if self.current_task and self.detection_count > 0:
                    await self.save_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠ Auto-save error: {e}")

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _set_state(self, new_state: OrchestratorState) -> None:
        """Update orchestrator state and notify callback."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state and self.config.on_state_change:
            self.config.on_state_change(old_state, new_state)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TaskOrchestrator("
            f"state={self._state.value}, "
            f"task='{self.current_task or 'None'}', "
            f"objects={len(self.registry)}, "
            f"detections={self.detection_count}"
            f")"
        )
