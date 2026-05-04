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
import io
import asyncio
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import json
from datetime import datetime, timezone
import uuid
import logging

import numpy as np
from PIL import Image

from .pddl_representation import PDDLRepresentation
from .pddl_domain_maintainer import PDDLDomainMaintainer
from .task_state_monitor import TaskStateMonitor, TaskState, TaskStateDecision
from .utils.task_types import TaskAnalysis
from .pddl_solver import PDDLSolver, SolverBackend, SearchAlgorithm, SolverResult
from ..perception import ContinuousObjectTracker
from ..perception.object_registry import DetectedObject
from ..camera import RealSenseCamera
from ..utils.logging_utils import get_structured_logger

# Import config from config directory
import sys
config_path = Path(__file__).parent.parent.parent / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
from orchestrator_config import OrchestratorConfig


@dataclass
class TaskExecutionResult:
    """Return value from TaskOrchestrator.execute_task()."""
    success: bool
    task: str
    plan: List[str] = field(default_factory=list)
    steps: List[Any] = field(default_factory=list)   # List[StepResult]
    replan_count: int = 0
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""
    UNINITIALIZED = "uninitialized"
    IDLE = "idle"  # Ready for task requests
    ANALYZING_TASK = "analyzing_task"  # Processing task description
    DETECTING = "detecting"  # Running continuous detection
    PAUSED = "paused"  # Detection paused, can resume
    READY_FOR_PLANNING = "ready_for_planning"  # Sufficient observations, ready for plan
    REFINING_DOMAIN = "refining_domain"  # Refining PDDL domain after planning failure
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
        self.logger = get_structured_logger("TaskOrchestrator")
        self._state = OrchestratorState.UNINITIALIZED
        self._camera = camera
        self._external_camera = camera is not None  # Track if camera is externally managed

        # Core components (initialized in initialize())
        self.pddl: Optional[PDDLRepresentation] = pddl_representation
        self.maintainer: Optional[PDDLDomainMaintainer] = None
        self.monitor: Optional[TaskStateMonitor] = None
        self.tracker: Optional[ContinuousObjectTracker] = None
        self.solver: Optional[PDDLSolver] = None

        # Layered domain generator (optional gated/guardrailed pipeline)
        self._layered_generator: Optional[Any] = None
        self._last_layered_artifact: Optional[Any] = None

        # Domain Knowledge Base (shared across planning sessions for error logging)
        self._dkb: Optional[Any] = None

        # Hybrid planner (deterministic / probabilistic mode selection)
        self._hybrid_planner: Optional[Any] = None

        # Task state
        self.current_task: Optional[str] = None
        self.task_analysis: Optional[TaskAnalysis] = None
        self.last_task_decision: Optional[TaskStateDecision] = None
        self.current_plan: Optional[SolverResult] = None
        self.refinement_attempts: int = 0
        self.last_planning_error: Optional[str] = None

        # Detection tracking
        self.detection_count: int = 0
        self.last_detection_time: float = 0.0
        self._detection_running: bool = False
        
        # Track known objects for detecting new ones
        self._known_object_ids: set = set()

        # Auto-save management (event-driven)
        self._last_save_time: float = 0.0
        self._save_lock: asyncio.Lock = asyncio.Lock()

        # Perception pool / snapshots
        self._perception_pool_index: Optional[Dict[str, Any]] = None
        self._perception_pool_lock: asyncio.Lock = asyncio.Lock()
        self.last_snapshot_id: Optional[str] = None

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
            self.logger.warning("Orchestrator already initialized (state: %s)", self._state.value)
            return

        self.logger.info("Initializing Task Orchestrator...")

        # Initialize camera if not provided
        if self._camera is None and not getattr(self.config, "use_sim_camera", False):
            self.logger.info(
                "  • Initializing RealSense camera (%sx%s @ %s FPS)...",
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
            )
            self._camera = RealSenseCamera(
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                enable_depth=self.config.enable_depth,
                auto_start=True,
                logger=self.logger.getChild("RealSenseCamera"),
            )
            self.logger.info("    ✓ Camera initialized")
        elif self._camera is not None:
            self.logger.info("  • Using provided camera")
        else:
            self.logger.info("  • Sim camera mode — frame provider will be injected")

        # Initialize PDDL representation if not provided
        if self.pddl is None:
            self.pddl = PDDLRepresentation(
                domain_name="task_execution",
                problem_name="current_task"
            )
            self.logger.info("  • PDDL representation created")

        # Initialize PDDL components
        _llm_client = getattr(self.config, "llm_client", None)
        self.maintainer = PDDLDomainMaintainer(
            self.pddl,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            task_analyzer_prompts_path=self.config.task_analyzer_prompts_path,
            llm_client=_llm_client,
        )

        self.monitor = TaskStateMonitor(
            self.maintainer,
            self.pddl,
            min_observations_before_planning=self.config.min_observations,
            exploration_timeout_seconds=self.config.exploration_timeout
        )
        self.logger.info("  • PDDL domain maintainer and monitor initialized")

        # Initialize layered domain generator when enabled
        if getattr(self.config, "use_layered_generation", False):
            from .layered_domain_generator import LayeredDomainGenerator
            dkb = None
            dkb_dir = getattr(self.config, "dkb_dir", None)
            if dkb_dir is not None:
                try:
                    from .domain_knowledge_base import DomainKnowledgeBase
                    dkb = DomainKnowledgeBase(dkb_dir)
                    dkb.load()
                except Exception as e:
                    self.logger.warning("  • DKB load failed (%s), proceeding without DKB", e)
            self._dkb = dkb
            self._layered_generator = LayeredDomainGenerator(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                dkb=dkb,
                llm_client=_llm_client,
            )
            self.logger.info("  • Layered domain generator initialized (use_layered_generation=True)")

        # Initialize hybrid planner (always available when a solver exists)
        try:
            from .hybrid_planner import HybridPlanner
            dkb_dir = getattr(self.config, "dkb_dir", None)
            self._hybrid_planner = HybridPlanner(
                dkb_dir=Path(dkb_dir) if dkb_dir else None,
                solver=self.solver,
                logger=self.logger,
            )
            self.logger.info("  • Hybrid planner initialized (det/prob mode selection)")
        except Exception as e:
            self.logger.warning("  • Hybrid planner init failed (%s), planning will use direct solver", e)

        # Attach default robot provider if none supplied — use PyBullet sim interface
        if getattr(self.config, "robot", None) is None:
            try:
                from ..kinematics.xarm_pybullet_interface import XArmPybulletInterface
                self.config.robot = XArmPybulletInterface()
                self.logger.info("  • Default robot provider attached: XArmPybulletInterface (sim)")
            except Exception as e:
                self.logger.warning("  • No robot provider attached (sim initialization failed: %s)", e)
                
        # Initialize continuous tracker
        self.tracker = ContinuousObjectTracker(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            fast_mode=self.config.fast_mode,
            update_interval=self.config.update_interval,
            on_detection_complete=self._on_detection_callback,
            logger=self.logger.getChild("ObjectTracker"),
            robot=self.config.robot,
            llm_client=_llm_client,
            debug_save_dir=getattr(self.config, "debug_frames_dir", None),
        )

        # Set frame provider
        self.tracker.set_frame_provider(self._get_camera_frames)
        self.logger.info("  • Continuous object tracker initialized")

        # Initialize PDDL solver
        backend_str = self.config.solver_backend or "auto"
        backend_map = {
            "auto": SolverBackend.AUTO,
            "pyperplan": SolverBackend.PYPERPLAN,
            "fast-downward-docker": SolverBackend.FAST_DOWNWARD_DOCKER,
            "fast-downward-apptainer": SolverBackend.FAST_DOWNWARD_APPTAINER,
        }
        backend = backend_map.get(backend_str.lower(), SolverBackend.AUTO)

        self.solver = PDDLSolver(
            backend=backend,
            verbose=self.config.solver_verbose
        )

        available = self.solver.get_available_backends()
        backend_names = [b.value for b in available]
        print(f"  • PDDL solver initialized (backend: {self.solver.backend.value}, available: {', '.join(backend_names)})")


        # Configure auto-save (event-driven)
        if self.config.auto_save:
            save_triggers = []
            if self.config.auto_save_on_detection:
                save_triggers.append("detection updates")
            if self.config.auto_save_on_state_change:
                save_triggers.append("state changes")
            self.logger.info("  • Auto-save enabled on: %s", ", ".join(save_triggers))

        self._set_state(OrchestratorState.IDLE)
        self.logger.info("✓ Task Orchestrator initialized and ready")

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        self.logger.info("Shutting down Task Orchestrator...")

        # Stop detection if running
        if self._detection_running:
            await self.stop_detection()

        # Final save
        if self.config.auto_save and self.current_task:
            await self.save_state()

        # Stop camera if we created it
        if self._camera and not self._external_camera:
            self._camera.stop()
            self.logger.info("  • Camera stopped")

        self._set_state(OrchestratorState.UNINITIALIZED)
        self.logger.info("✓ Task Orchestrator shutdown complete")

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
        self.logger.info("%s", "=" * 70)
        self.logger.info("PROCESSING TASK REQUEST")
        self.logger.info("%s", "=" * 70)
        self.logger.info('Task: "%s"', task_description)

        self.current_task = task_description

        # Capture environment image if not provided
        if environment_image is None and self._camera:
            self.logger.info("  • Capturing environment frame for context...")
            environment_image, _ = self._camera.get_aligned_frames()

        # Analyze task and initialize domain
        self.logger.info("  • Analyzing task with LLM...")
        if self._layered_generator is not None:
            # Gated/guardrailed pipeline: L1→L5 with validation gates
            self.logger.info("  • Using layered domain generator (L1–L5 pipeline)...")
            # Use any objects already detected by the tracker
            detected = self.get_detected_objects()
            observed_objects = [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "affordances": list(obj.affordances),
                    "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None,
                    "position_2d": obj.position_2d,
                    "bounding_box_2d": obj.bounding_box_2d,
                }
                for obj in detected
            ]
            registry = getattr(self.tracker, "registry", None) if self.tracker else None
            artifact = await self._layered_generator.generate_domain(
                task_description,
                observed_objects=observed_objects,
                image=environment_image,
                registry=registry,
            )
            self._last_layered_artifact = artifact
            self.task_analysis = await self.maintainer.initialize_from_layered_artifact(artifact)
        else:
            # Legacy monolithic path
            self.task_analysis = await self.maintainer.initialize_from_task(
                task_description,
                environment_image=environment_image
            )

        self.logger.info("✓ Task analyzed!")
        valid_goal_objects = [obj for obj in self.task_analysis.goal_object_references() if obj and obj != "None"]
        self.logger.info("  • Goal objects: %s", ", ".join(valid_goal_objects) if valid_goal_objects else "None")
        self.logger.info("  • Goal summary: %s", self.task_analysis.abstract_goal.summary or "n/a")
        self.logger.info("  • Required predicates: %s", len(self.task_analysis.predicate_signatures()))
        self.logger.info("  • Action schemas: %s", len(self.task_analysis.action_context()))

        # Seed perception with predicates and task context
        if self.tracker:
            predicate_signatures = self.task_analysis.predicate_signatures()
            self.logger.info("  • Configuring perception with %s predicates...", len(predicate_signatures))
            self.tracker.set_pddl_predicates(predicate_signatures)

            self.logger.info("%s", "=" * 70)
            # Pass task context and available actions to tracker
            print(f"  • Setting task context for grounded object detection...")
            available_actions = [
                {
                    "name": action.get("name", "unknown"),
                    "params": action.get("parameters", []),
                    "description": action.get("description", "")
                }
                for action in self.task_analysis.action_context()
            ]
            self.tracker.set_task_context(
                task_description=task_description,
                available_actions=available_actions,
                goal_objects=self.task_analysis.goal_object_references()
            )

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

        # Get current observations from tracker's registry
        all_objects = self.get_detected_objects()

        # Process new task
        result = await self.process_task_request(new_task_description)

        # Re-process existing observations with new task context
        if all_objects:
            self.logger.info("  • Re-processing %s existing objects with new task...", len(all_objects))
            objects_dict = self._convert_objects_to_dict(all_objects)
            predicates = self.tracker.registry.get_all_predicates()
            await self.maintainer.update_from_observations(objects_dict, predicates=predicates)

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
            self.logger.warning("Detection already running")
            return

        self.logger.info("%s", "=" * 70)
        self.logger.info("STARTING CONTINUOUS DETECTION")
        self.logger.info("%s", "=" * 70)
        self.logger.info("Update interval: %ss", self.config.update_interval)
        self.logger.info("Minimum observations: %s", self.config.min_observations)

        self.tracker.start()
        self._detection_running = True
        self._set_state(OrchestratorState.DETECTING)

        self.logger.info("✓ Continuous detection started")
        self.logger.info("%s", "=" * 70)

    async def stop_detection(self) -> None:
        """
        Stop continuous object detection.
        """
        if not self._detection_running:
            return

        self.logger.info("Stopping continuous detection...")
        await self.tracker.stop()
        self._detection_running = False
        self._set_state(OrchestratorState.IDLE)
        self.logger.info("✓ Detection stopped")

    async def pause_detection(self) -> None:
        """
        Pause detection (can be resumed later).
        """
        if not self._detection_running:
            return

        await self.stop_detection()
        self._set_state(OrchestratorState.PAUSED)
        self.logger.info("✓ Detection paused")

    async def resume_detection(self) -> None:
        """
        Resume paused detection.
        """
        if self._state != OrchestratorState.PAUSED:
            self.logger.warning("Cannot resume from state: %s", self._state.value)
            return

        await self.start_detection()

    def _get_camera_frames(self):
        """Frame provider for continuous tracker."""
        if self._camera is None:
            return None, None, None, None
        try:
            color, depth = self._camera.get_aligned_frames()
            intrinsics = self._camera.get_camera_intrinsics()
            robot_state = self._get_robot_state_struct()
            return color, depth, intrinsics, robot_state
        except Exception as e:
            self.logger.warning("Camera error: %s", e)
            return None, None, None, None
    # ========================================================================
    # Perception Pool / Snapshot Helpers
    # ========================================================================

    def _get_perception_pool_dir(self) -> Path:
        """Resolve perception pool directory from config, defaulting under state_dir."""
        if getattr(self.config, "perception_pool_dir", None) is None:
            return self.config.state_dir / "perception_pool"
        return Path(self.config.perception_pool_dir)

    def _get_pool_index_path(self) -> Path:
        """Return path to perception pool index.json."""
        return self._get_perception_pool_dir() / "index.json"

    def _ensure_pool_dirs(self) -> None:
        """Create perception pool directory structure if missing."""
        pool_dir = self._get_perception_pool_dir()
        (pool_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    def _init_empty_pool_index(self) -> Dict[str, Any]:
        """Return a new empty perception pool index structure."""
        return {
            "version": "1.0",
            "last_snapshot_id": None,
            "objects": {},
            "snapshots": {}
        }

    def _read_pool_index(self) -> Dict[str, Any]:
        """Load perception pool index into memory."""
        if self._perception_pool_index is not None:
            return self._perception_pool_index
        index_path = self._get_pool_index_path()
        if not index_path.exists():
            self._perception_pool_index = self._init_empty_pool_index()
            return self._perception_pool_index
        try:
            with open(index_path, "r") as f:
                self._perception_pool_index = json.load(f)
        except Exception:
            # On parse failure, re-init to a safe empty index
            self._perception_pool_index = self._init_empty_pool_index()
        return self._perception_pool_index

    def _write_pool_index(self, index: Dict[str, Any]) -> Path:
        """Persist perception pool index to disk."""
        self._ensure_pool_dirs()
        index_path = self._get_pool_index_path()
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        self._perception_pool_index = index
        self.last_snapshot_id = index.get("last_snapshot_id")
        return index_path

    def _generate_snapshot_id(self) -> str:
        """Generate human-sortable snapshot ID: YYYYMMDD_HHMMSS_mmm-<shortid> (UTC)."""
        now = datetime.utcnow()
        ts_prefix = now.strftime("%Y%m%d_%H%M%S") + f"_{int(now.microsecond / 1000):03d}"
        shortid = uuid.uuid4().hex[:6]
        return f"{ts_prefix}-{shortid}"

    def _serialize_detections_for_snapshot(
        self,
        objects: Optional[List[DetectedObject]] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Build detections.json payload from provided objects (or current registry) and return (payload, object_ids).
        """
        objects = objects if objects is not None else self.get_detected_objects()
        object_ids: List[str] = []
        det_objects: List[Dict[str, Any]] = []
        for obj in objects:
            object_ids.append(obj.object_id)
            det_objects.append({
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "affordances": list(obj.affordances),
                "interaction_points": {
                    affordance: {
                        "snapshot_id": None,  # filled by snapshot_id at write-time
                        "position_2d": point.position_2d,
                        "position_3d": point.position_3d.tolist() if point.position_3d is not None else None,
                        "alternative_points": point.alternative_points,
                    }
                    for affordance, point in obj.interaction_points.items()
                },
                "position_2d": obj.position_2d,
                "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None,
                "bounding_box_2d": obj.bounding_box_2d
            })
        payload = {
            "stamp": time.time(),
            "predicates": self.tracker.registry.get_all_predicates(),
            "objects": det_objects
        }
        return payload, object_ids

    def _robot_get_optional(self, method_name: str):
        """Call a robot provider method if available, else return None."""
        robot = getattr(self.config, "robot", None)
        if robot is None:
            return None
        method = getattr(robot, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
        # Allow attribute-based static transform access (e.g., static_camera_tf)
        if hasattr(robot, method_name):
            return getattr(robot, method_name)
        return None

    def _get_robot_state_struct(self) -> Optional[Dict[str, Any]]:
        """
        Obtain robot state via duck-typed get_robot_state() on the provider.

        The orchestrator does not assume any internal structure beyond it being JSON-serializable.
        """
        robot = getattr(self.config, "robot", None)
        if robot is None:
            return None
        getter = getattr(robot, "get_robot_state", None)
        if not callable(getter):
            return None
        try:
            raw_state = getter()
            # Best-effort: ensure it's serializable (convert numpy arrays)
            def to_serializable(x):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, (list, dict, str, int, float, type(None), bool)):
                    return x
                if hasattr(x, "__dict__"):
                    # dataclass / custom object; use dict
                    return {k: to_serializable(v) for k, v in vars(x).items()}
                return str(x)
            if isinstance(raw_state, dict):
                return {k: to_serializable(v) for k, v in raw_state.items()}
            # If provider returned a non-dict, wrap it for stability
            return {"data": to_serializable(raw_state)}
        except Exception:
            return None

    def _enforce_snapshot_retention(self, index: Dict[str, Any]) -> None:
        """Ensure number of snapshots does not exceed max_snapshot_count by pruning oldest."""
        max_count = getattr(self.config, "max_snapshot_count", None)
        if not max_count or max_count <= 0:
            return
        snapshots = index.get("snapshots", {})
        if len(snapshots) <= max_count:
            return
        # Sort by recorded_at fallback to captured_at, oldest first
        def parse_iso8601(s: Optional[str]) -> float:
            if not s:
                return 0.0
            try:
                # Handle Z suffix
                if s.endswith("Z"):
                    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0
        items = []
        for sid, meta in snapshots.items():
            t = parse_iso8601(meta.get("recorded_at")) or parse_iso8601(meta.get("captured_at"))
            items.append((t, sid))
        items.sort(key=lambda x: x[0])
        num_to_remove = len(snapshots) - max_count
        to_remove = [sid for _, sid in items[:num_to_remove]]

        pool_dir = self._get_perception_pool_dir()
        for sid in to_remove:
            # Remove folder
            snap_dir = pool_dir / "snapshots" / sid
            try:
                if snap_dir.exists():
                    # Remove files then directory
                    for p in snap_dir.glob("**/*"):
                        try:
                            if p.is_file():
                                p.unlink()
                        except Exception:
                            pass
                    # Remove empty subdirs first
                    for p in sorted(snap_dir.glob("**/*"), reverse=True):
                        try:
                            if p.is_dir():
                                p.rmdir()
                        except Exception:
                            pass
                    if snap_dir.exists():
                        snap_dir.rmdir()
            except Exception:
                pass

            # Prune from index
            snapshots.pop(sid, None)
            for obj_id, ids in list(index.get("objects", {}).items()):
                index["objects"][obj_id] = [x for x in ids if x != sid]
                if not index["objects"][obj_id]:
                    # Keep empty list? Prefer to drop empty to reduce noise
                    index["objects"].pop(obj_id, None)

        # Update last_snapshot_id if needed
        if index.get("last_snapshot_id") in to_remove:
            index["last_snapshot_id"] = None

    # ========================================================================
    # Snapshot API
    # ========================================================================

    async def save_snapshot(self, reason: str = "", label: Optional[str] = None) -> Path:
        """
        Capture and persist a snapshot of current observation.

        Returns:
            Path to the created snapshot directory.
        """
        # Serialize writes with a dedicated pool lock
        async with self._perception_pool_lock:
            detection_bundle = self.tracker.get_last_detection_bundle() if self.tracker else None
            bundle_color_bytes = bundle_depth = bundle_intrinsics = None
            bundle_objects = None
            bundle_timestamp = None
            if detection_bundle:
                bundle_color_bytes = detection_bundle.get("color_png")
                bundle_depth = detection_bundle.get("depth")
                bundle_intrinsics = detection_bundle.get("intrinsics")
                bundle_objects = detection_bundle.get("objects")
                bundle_timestamp = detection_bundle.get("timestamp")
                robot_state = detection_bundle.get("robot_state")

            color = depth = intrinsics = None
            color_img = None

            if bundle_color_bytes is not None:
                depth = bundle_depth
                intrinsics = bundle_intrinsics
            else:
                color, depth, intrinsics, robot_state = self._get_camera_frames()
                if color is None or intrinsics is None:
                    raise RuntimeError("Cannot capture snapshot: camera frames unavailable")
                color_img = Image.fromarray(color) if isinstance(color, np.ndarray) else color

            self._ensure_pool_dirs()
            pool_dir = self._get_perception_pool_dir()
            snapshot_id = self._generate_snapshot_id()
            snapshot_dir = pool_dir / "snapshots" / snapshot_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Save color
            color_path = snapshot_dir / "color.png"
            if bundle_color_bytes is not None:
                color_path.write_bytes(bundle_color_bytes)
                if color_img is None:
                    color_img = Image.open(io.BytesIO(bundle_color_bytes))
            else:
                color_img.save(str(color_path), format="PNG")

            # Save depth (optional)
            depth_path = None
            if depth is not None and getattr(self.config, "depth_encoding", "npz") == "npz":
                depth_path = snapshot_dir / "depth.npz"
                np.savez_compressed(depth_path, depth_m=np.asarray(depth, dtype=np.float32))

            # Save intrinsics
            intr_path = snapshot_dir / "intrinsics.json"
            try:
                intr_dict = intrinsics.to_dict()
            except Exception:
                # Best-effort mapping if it's plain dict-like
                intr_dict = dict(intrinsics) if isinstance(intrinsics, dict) else {}
            with open(intr_path, "w") as f:
                json.dump(intr_dict, f, indent=2)

            # Save detections
            det_payload, object_ids = self._serialize_detections_for_snapshot(objects=bundle_objects)
            if bundle_timestamp is not None:
                det_payload["stamp"] = bundle_timestamp
            det_path = snapshot_dir / "detections.json"
            # Stamp snapshot_id into interaction points for grounding
            for dobj in det_payload.get("objects", []):
                for point in (dobj.get("interaction_points") or {}).values():
                    point["snapshot_id"] = snapshot_id
            with open(det_path, "w") as f:
                json.dump(det_payload, f, indent=2)

            # Robot context (optional, via duck-typed get_robot_state)
            robot_state_path = None
            if robot_state is not None:
                robot_state_path = snapshot_dir / "robot_state.json"
                with open(robot_state_path, "w") as f:
                    json.dump(robot_state, f, indent=2)

            # Manifest
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            captured_at = recorded_at
            if bundle_timestamp is not None:
                captured_at = datetime.fromtimestamp(bundle_timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            manifest = {
                "snapshot_id": snapshot_id,
                "captured_at": captured_at,
                "recorded_at": recorded_at,
                "sources": {
                    "camera": type(self._camera).__name__ if self._camera is not None else "unknown",
                    "robot": type(self.config.robot).__name__ if getattr(self.config, "robot", None) is not None else None
                },
                "files": {
                    "color": "color.png",
                    "depth_npz": "depth.npz" if depth_path is not None else None,
                    "intrinsics": "intrinsics.json",
                    "detections": "detections.json",
                    "robot_state": "robot_state.json" if robot_state_path is not None else None
                },
                "notes": label or "",
                "hashes": None
            }
            manifest_path = snapshot_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Update perception pool index
            index = self._read_pool_index()
            rel_base = Path("snapshots") / snapshot_id
            index_entry_files = {
                "color": str(rel_base / "color.png"),
                "intrinsics": str(rel_base / "intrinsics.json"),
                "depth_npz": str(rel_base / "depth.npz") if depth_path is not None else None,
                "detections": str(rel_base / "detections.json"),
                "robot_state": str(rel_base / "robot_state.json") if robot_state_path is not None else None
            }
            index_snapshot_meta = {
                "captured_at": captured_at,
                "recorded_at": recorded_at,
                "objects": object_ids,
                "files": index_entry_files,
                "label": label or None,
                "reason": reason or None
            }

            index.setdefault("snapshots", {})[snapshot_id] = index_snapshot_meta
            for oid in object_ids:
                index.setdefault("objects", {}).setdefault(oid, [])
                if snapshot_id not in index["objects"][oid]:
                    index["objects"][oid].append(snapshot_id)
            index["last_snapshot_id"] = snapshot_id

            # Retention policy
            self._enforce_snapshot_retention(index)

            self._write_pool_index(index)
            self.last_snapshot_id = snapshot_id

            return snapshot_dir

    def capture_fresh_snapshot_sync(self, reason: str = "pre-execution") -> Optional[str]:
        """
        Synchronous snapshot capture for use inside synchronous execution callbacks.

        Captures a fresh RGB+depth frame from the camera, writes it to the perception
        pool, updates the pool index, and returns the new snapshot_id.  Returns None
        if the camera is unavailable or capture fails.

        This is intentionally sync (no asyncio lock) because it is only called from
        the synchronous execute_fn between primitive executions, where no other async
        snapshot writers are active.
        """
        color, depth, intrinsics, robot_state = self._get_camera_frames()
        if color is None or intrinsics is None:
            self.logger.warning("[capture_fresh_snapshot_sync] Camera unavailable — skipping fresh capture")
            return None

        try:
            from PIL import Image
            import numpy as np
            from datetime import datetime, timezone

            self._ensure_pool_dirs()
            pool_dir = self._get_perception_pool_dir()
            snapshot_id = self._generate_snapshot_id()
            snapshot_dir = pool_dir / "snapshots" / snapshot_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Color
            color_path = snapshot_dir / "color.png"
            color_img = Image.fromarray(color) if isinstance(color, np.ndarray) else color
            color_img.save(str(color_path), format="PNG")

            # Depth
            depth_path = None
            if depth is not None:
                depth_path = snapshot_dir / "depth.npz"
                np.savez_compressed(depth_path, depth_m=np.asarray(depth, dtype=np.float32))

            # Intrinsics
            intr_path = snapshot_dir / "intrinsics.json"
            try:
                intr_dict = intrinsics.to_dict()
            except Exception:
                intr_dict = dict(intrinsics) if isinstance(intrinsics, dict) else {}
            with open(intr_path, "w") as f:
                json.dump(intr_dict, f, indent=2)

            # Robot state
            robot_state_path = None
            if robot_state is not None:
                robot_state_path = snapshot_dir / "robot_state.json"
                with open(robot_state_path, "w") as f:
                    json.dump(robot_state, f, indent=2)

            # Empty detections placeholder (objects not re-detected at this point)
            det_path = snapshot_dir / "detections.json"
            with open(det_path, "w") as f:
                json.dump({"objects": [], "snapshot_id": snapshot_id}, f)

            # Manifest
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            manifest = {
                "snapshot_id": snapshot_id,
                "captured_at": now,
                "recorded_at": now,
                "reason": reason,
                "files": {
                    "color": "color.png",
                    "depth_npz": "depth.npz" if depth_path else None,
                    "intrinsics": "intrinsics.json",
                    "detections": "detections.json",
                    "robot_state": "robot_state.json" if robot_state_path else None,
                },
            }
            with open(snapshot_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Update index
            index = self._read_pool_index()
            rel_base = Path("snapshots") / snapshot_id
            index.setdefault("snapshots", {})[snapshot_id] = {
                "captured_at": now,
                "recorded_at": now,
                "objects": [],
                "files": {
                    "color": str(rel_base / "color.png"),
                    "depth_npz": str(rel_base / "depth.npz") if depth_path else None,
                    "intrinsics": str(rel_base / "intrinsics.json"),
                    "detections": str(rel_base / "detections.json"),
                    "robot_state": str(rel_base / "robot_state.json") if robot_state_path else None,
                },
                "reason": reason,
            }
            index["last_snapshot_id"] = snapshot_id
            self._enforce_snapshot_retention(index)
            self._write_pool_index(index)
            self.last_snapshot_id = snapshot_id

            self.logger.info("[capture_fresh_snapshot_sync] Captured snapshot %s", snapshot_id)
            return snapshot_id

        except Exception as exc:
            self.logger.warning("[capture_fresh_snapshot_sync] Failed: %s", exc)
            return None

    async def _on_detection_callback(self, object_count: int):
        """
        Called after each detection cycle.

        Updates PDDL domain and checks task state.
        """
        self.detection_count += 1
        self.last_detection_time = time.time()

        # Get all detected objects from tracker's registry
        all_objects = self.tracker.get_all_objects()

        # Convert to dict format for PDDL maintainer
        objects_dict = self._convert_objects_to_dict(all_objects)

        # Update PDDL domain
        predicates = self.tracker.registry.get_all_predicates()
        update_stats = await self.maintainer.ground_representation(objects_dict, predicates=predicates)

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

                # Auto-solve if enabled
                if self.config.auto_solve_when_ready:
                    asyncio.create_task(self._auto_solve())

        # Notify detection update callback
        if self.config.on_detection_update:
            self.config.on_detection_update(object_count)
        
        # Auto-save on detection update if enabled
        if self.config.auto_save and self.config.auto_save_on_detection:
            await self._try_auto_save()

        # Periodic snapshot trigger (optional)
        if getattr(self.config, "enable_snapshots", False):
            N = getattr(self.config, "snapshot_every_n_detections", 0)
            if N and N > 0 and (self.detection_count % N == 0):
                try:
                    # Save snapshot on schedule regardless of object count
                    await self.save_snapshot(reason="periodic")
                except Exception as e:
                    # Non-fatal: continue detection
                    self.logger.warning("Snapshot failed: %s", e)

    def _convert_objects_to_dict(self, objects: List[DetectedObject]) -> List[Dict]:
        """Convert DetectedObject list to dict format for PDDL maintainer."""
        return [
            {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "affordances": list(obj.affordances),
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

        # Add registry stats (from tracker's registry)
        if self.tracker:
            registry = self.tracker.registry
            status["registry"] = {
                "num_objects": len(registry),
                "object_types": list(set(obj.object_type for obj in registry.get_all_objects())),
            }
        else:
            status["registry"] = {
                "num_objects": 0,
                "object_types": [],
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
        """Get all detected objects from tracker's registry."""
        if self.tracker:
            return self.tracker.registry.get_all_objects()
        return []

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """Get objects of a specific type."""
        if self.tracker:
            return self.tracker.registry.get_objects_by_type(object_type)
        return []

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """Get objects with a specific affordance."""
        if self.tracker:
            return self.tracker.registry.get_objects_with_affordance(affordance)
        return []
    
    def get_new_objects(self) -> List[DetectedObject]:
        """
        Get newly detected objects since last check.
        
        Returns:
            List of objects that are new since the last call to this method
        """
        all_objects = self.get_detected_objects()
        current_ids = {obj.object_id for obj in all_objects}
        new_ids = current_ids - self._known_object_ids
        self._known_object_ids = current_ids
        
        return [obj for obj in all_objects if obj.object_id in new_ids]

    def get_world_state_snapshot(self) -> Dict[str, Any]:
        """
        Build a world-state payload for downstream planners.

        Returns:
            Dict containing registry, last snapshot id, snapshot index, and robot state.
        """
        snapshot_index: Optional[Dict[str, Any]] = None
        try:
            snapshot_index = self._read_pool_index()
        except Exception:
            snapshot_index = None

        return {
            "registry": self._build_enhanced_registry(),
            "last_snapshot_id": self.last_snapshot_id,
            "snapshot_index": snapshot_index,
            "robot_state": self._get_robot_state_struct(),
        }

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

        self.logger.info("%s", "=" * 70)
        self.logger.info("GENERATING PDDL FILES")
        self.logger.info("%s", "=" * 70)

        # Sync objects from tracker to PDDL representation
        # Ensure the tracker has a registry before trying to read objects
        if self.tracker and getattr(self.tracker, 'registry', None) is not None:
            self.logger.info("  • Syncing objects from tracker to PDDL...")
            registry = self.tracker.registry
            all_objects = registry.get_all_objects()
            self.logger.info(f"    Found {len(all_objects)} objects in registry")

            # Check if objects have been detected
            if len(all_objects) == 0:
                self.logger.warning("⚠ No objects detected in tracker registry!")
                self.logger.warning("  Planning requires at least one detected object.")
                self.logger.warning("  Make sure object detection has run before calling generate_pddl_files().")
                print("\n⚠ WARNING: Cannot generate PDDL files - no objects detected!")
                print("  • Ensure the object tracker has detected objects before planning")
                print("  • Check that detect_objects() has been called and completed")
                print("  • Verify the camera is providing valid frames")

            # Add objects to PDDL problem
            _SURFACE_KEYWORDS = frozenset({
                "table", "counter", "shelf", "desk", "tray", "bench", "surface", "floor"
            })
            for obj in all_objects:
                # Add object instance if not already present
                if obj.object_id not in self.pddl.object_instances:
                    obj_type = obj.object_type
                    # Normalise surface-like objects to the PDDL "surface" type so that
                    # actions with typed (?surface - surface) parameters can be grounded.
                    raw_type_lower = obj_type.lower()
                    if any(kw in raw_type_lower for kw in _SURFACE_KEYWORDS):
                        obj_type = "surface"
                    # Auto-register unknown types as children of 'object'
                    if obj_type not in self.pddl.object_types:
                        await self.pddl.add_object_type_async(obj_type, parent="object")
                        self.logger.debug(f"      Auto-registered type '{obj_type}' (parent: object)")
                    await self.pddl.add_object_instance_async(
                        obj.object_id,
                        obj_type
                    )
                    self.logger.info(f"      Added: {obj.object_id} ({obj_type})")

            # Add global predicates to initial state
            # Exclude runtime-effect flags that are set by actions at runtime and must start FALSE.
            _RUNTIME_FALSE_PREDICATES: frozenset = frozenset({"spatial-change-occurred"})
            if self.maintainer:
                global_predicates = self.maintainer.get_global_predicates()
                if global_predicates:
                    self.logger.info(f"  • Adding {len(global_predicates)} global predicates to initial state...")
                    for pred_name in global_predicates:
                        if pred_name in _RUNTIME_FALSE_PREDICATES:
                            self.logger.info(f"      Skipping runtime-effect flag '{pred_name}' (starts FALSE)")
                            continue
                        # Global predicates typically have no parameters
                        try:
                            await self.pddl.add_initial_literal_async(pred_name, [], negated=False)
                            self.logger.info(f"      Added global predicate: {pred_name}")
                        except ValueError as e:
                            self.logger.warning(f"      Failed to add global predicate '{pred_name}': {e}")

                # Apply L5 initial state literals first (handles on, above, etc.)
                # These use position_3d proximity from the scene at generation time.
                added_initial: set = set()
                l5 = getattr(self.maintainer, "_l5_artifact", None)
                if l5 and l5.true_literals:
                    for pred_name, args in l5.true_literals:
                        if args and not all(a in self.pddl.object_instances for a in args):
                            continue
                        key = (pred_name, tuple(args))
                        if key not in added_initial:
                            try:
                                await self.pddl.add_initial_literal_async(pred_name, args, negated=False)
                                added_initial.add(key)
                            except ValueError:
                                pass

                domain_predicates = set(self.pddl.predicates.keys())

                # Mirror `on` facts as `object-at` when that predicate exists in the domain.
                # LLM-generated actions often use `object-at` for placement tracking, but L5
                # only grounds `on`/`stable-on`.  Without this, displace/push-aside are unusable.
                if "object-at" in domain_predicates and l5 and l5.true_literals:
                    _ON_SYNONYMS = frozenset({"on", "stable-on", "resting-on", "placed-on"})
                    for pred_name, args in l5.true_literals:
                        if pred_name in _ON_SYNONYMS and len(args) == 2:
                            if all(a in self.pddl.object_instances for a in args):
                                key = ("object-at", tuple(args))
                                if key not in added_initial:
                                    try:
                                        await self.pddl.add_initial_literal_async("object-at", list(args), negated=False)
                                        added_initial.add(key)
                                    except ValueError:
                                        pass

                # Inject `space-available` for every surface × non-surface pair when
                # the predicate is in the domain.  space-available now defaults True
                # (LLM room_available flag); without this injection displace is unusable
                # whenever the LLM didn't include space-available in the vocabulary.
                if "space-available" in domain_predicates:
                    _SURFACE_KWS = frozenset({
                        "table", "counter", "shelf", "desk", "tray", "bench", "surface", "floor"
                    })
                    _surface_ids = {
                        obj.object_id for obj in all_objects
                        if any(kw in obj.object_type.lower() for kw in _SURFACE_KWS)
                    }
                    _non_surface_ids = [
                        obj.object_id for obj in all_objects
                        if obj.object_id not in _surface_ids
                    ]
                    for sid in _surface_ids:
                        for oid in _non_surface_ids:
                            key = ("space-available", (sid, oid))
                            if key not in added_initial:
                                try:
                                    await self.pddl.add_initial_literal_async(
                                        "space-available", [sid, oid], negated=False
                                    )
                                    added_initial.add(key)
                                except ValueError:
                                    pass

                # Derive `clear` facts: an object is clear if nothing is `on` it.
                # This handles blocksworld-style domains where `clear` is added by refinement.
                if "clear" in domain_predicates:
                    occupied: set = set()
                    for pred_name, args in l5.true_literals if (l5 and l5.true_literals) else []:
                        if pred_name == "on" and len(args) == 2:
                            occupied.add(args[1])  # surface has something on it
                    for obj in all_objects:
                        if obj.object_id not in occupied:
                            key = ("clear", (obj.object_id,))
                            if key not in added_initial:
                                try:
                                    await self.pddl.add_initial_literal_async("clear", [obj.object_id], negated=False)
                                    added_initial.add(key)
                                except ValueError:
                                    pass

                # Re-derive unary affordance predicates from live detected objects.
                # This catches cases where L5 was built before perception ran (empty scene)
                # or where domain refinement re-added predicates that L2-V5 had pruned.
                # Try both bare name (e.g. `graspable`) and object-prefixed name
                # (e.g. `object-graspable`) since the prompt encourages prefixed naming.
                for obj in all_objects:
                    for affordance in (obj.affordances or set()):
                        base = affordance.replace(" ", "-").replace("_", "-")
                        candidates = [base, f"object-{base}"]
                        for pred_name in candidates:
                            if pred_name in domain_predicates:
                                key = (pred_name, (obj.object_id,))
                                if key not in added_initial:
                                    try:
                                        await self.pddl.add_initial_literal_async(pred_name, [obj.object_id], negated=False)
                                        added_initial.add(key)
                                    except ValueError:
                                        pass
                if added_initial:
                    self.logger.info(f"  • Added {len(added_initial)} initial literals from L5 + affordances")

        # Set goals if requested
        if set_goals:
            self.logger.info("  • Setting goal state from task analysis...")
            await self.maintainer.set_goal_from_task_analysis()

        # Generate files
        self.logger.info("  • Generating files to %s...", output_dir)
        paths = await self.pddl.generate_files_async(str(output_dir))

        self.logger.info("✓ PDDL files generated:")
        self.logger.info("  • Domain: %s", paths["domain_path"])
        self.logger.info("  • Problem: %s", paths["problem_path"])

        # Show summary
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        self.logger.info("Domain Summary:")
        self.logger.info("  • Types: %s", len(domain_snapshot["object_types"]))
        self.logger.info("  • Predicates: %s", len(domain_snapshot["predicates"]))
        self.logger.info(
            "  • Actions: %s",
            len(domain_snapshot["predefined_actions"]) + len(domain_snapshot["llm_generated_actions"]),
        )

        self.logger.info("Problem Summary:")
        self.logger.info("  • Objects: %s", len(problem_snapshot["object_instances"]))
        self.logger.info("  • Initial literals: %s", len(problem_snapshot["initial_literals"]))
        # goal_literals tracks structured Literal objects; goal_formulas tracks raw PDDL strings.
        # Goals are added via add_goal_formula_async so count goal_formulas.
        goal_count = len(problem_snapshot.get("goal_literals", [])) + len(self.pddl.goal_formulas)
        self.logger.info("  • Goal literals: %s", goal_count)

        self.logger.info("%s", "=" * 70)

        return paths

    async def solve_and_plan(
        self,
        algorithm: Optional[SearchAlgorithm] = None,
        timeout: Optional[float] = None,
        generate_files: bool = True,
        output_dir: Optional[Path] = None,
        wait_for_objects: bool = True,
        max_wait_seconds: float = 120.0
    ) -> SolverResult:
        """
        Generate PDDL files and solve for a plan.

        Args:
            algorithm: Search algorithm to use (defaults to config)
            timeout: Solver timeout in seconds (defaults to config)
            generate_files: Whether to generate PDDL files first
            output_dir: Directory for PDDL files (defaults to config.state_dir / "pddl")
            wait_for_objects: Whether to wait for object detection before planning
            max_wait_seconds: Maximum time to wait for objects (default: 30s)

        Returns:
            SolverResult with plan and statistics
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized")

        # Wait for object detection to complete if requested
        if wait_for_objects and self.tracker and hasattr(self.tracker, 'registry'):
            registry = self.tracker.registry
            objects_count = len(registry.get_all_objects())

            if objects_count == 0:
                print(f"\n⏳ Waiting for object detection to complete (max {max_wait_seconds}s)...")
                import asyncio
                waited = 0.0
                poll_interval = 0.5

                while objects_count == 0 and waited < max_wait_seconds:
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    objects_count = len(registry.get_all_objects())

                    if waited % 5.0 < poll_interval:  # Print every 5 seconds
                        print(f"  Still waiting... ({waited:.1f}s elapsed, {objects_count} objects detected)")

                if objects_count == 0:
                    print(f"\n⚠ WARNING: No objects detected after {max_wait_seconds}s")
                    print("  Planning will likely fail without detected objects")
                else:
                    print(f"✓ Object detection complete: {objects_count} objects detected")

        # Use config defaults if not specified
        if algorithm is None:
            algo_map = {
                "lama-first": SearchAlgorithm.LAMA_FIRST,
                "lama": SearchAlgorithm.LAMA,
                "astar(lmcut())": SearchAlgorithm.ASTAR_LMCUT,
                "lazy_greedy([ff()], preferred=[ff()])": SearchAlgorithm.LAZY_GREEDY_FF,
            }
            algorithm = algo_map.get(self.config.solver_algorithm.lower(), SearchAlgorithm.LAMA_FIRST)

        if timeout is None:
            timeout = self.config.solver_timeout

        print(f"\n{'='*70}")
        print("SOLVING PDDL PROBLEM")
        print(f"{'='*70}")
        print(f"Algorithm: {algorithm.value}")
        print(f"Backend: {self.solver.backend.value}")
        print(f"Timeout: {timeout}s")
        print()

        # Generate PDDL files if requested
        if generate_files:
            paths = await self.generate_pddl_files(output_dir=output_dir, set_goals=True)
            domain_path = paths["domain_path"]
            problem_path = paths["problem_path"]
        else:
            # Use existing files
            if output_dir is None:
                output_dir = self.config.state_dir / "pddl"
            output_dir = Path(output_dir)  # Ensure it's a Path object
            # Both files use domain_name as the prefix
            domain_path = output_dir / f"{self.pddl.domain_name}_domain.pddl"
            problem_path = output_dir / f"{self.pddl.domain_name}_problem.pddl"

            if not domain_path.exists() or not problem_path.exists():
                raise FileNotFoundError(
                    f"PDDL files not found at {output_dir}. Set generate_files=True or call generate_pddl_files() first."
                )

        # Solve
        print("  • Running solver...")
        result = await self.solver.solve(
            domain_path=str(domain_path),
            problem_path=str(problem_path),
            algorithm=algorithm,
            timeout=timeout
        )
        print("result done.")
        # Store result
        self.current_plan = result

        # Print result
        print()
        if result.success:
            print(f"✓ {result}")
            print(f"\nPlan ({result.plan_length} steps):")
            for i, action in enumerate(result.plan, 1):
                print(f"  {i}. {action}")
        else:
            print(f"✗ {result}")

        print(f"\n{'='*70}\n")

        # Notify callback
        if self.config.on_plan_generated:
            self.config.on_plan_generated(result)

        return result

    async def refine_domain_from_failure(
        self,
        error_message: str,
        pddl_files: Optional[Dict[str, str]] = None,
        raw_solver_output: Optional[str] = None,
    ) -> bool:
        """
        Refine the PDDL domain based on a planning failure.

        This method analyzes the planning error and attempts to fix the domain
        by working with the LLM to correct issues like:
        - Incorrect predicate arities
        - Missing predicates or actions
        - Malformed action definitions
        - Type mismatches

        Args:
            error_message: The error message from the planner
            pddl_files: Optional dict with domain_path and problem_path for context

        Returns:
            True if refinement was attempted, False if max attempts reached
        """
        if self.refinement_attempts >= self.config.max_refinement_attempts:
            print(f"\n⚠ Max refinement attempts ({self.config.max_refinement_attempts}) reached")
            return False

        self.refinement_attempts += 1
        self.last_planning_error = error_message
        self._set_state(OrchestratorState.REFINING_DOMAIN)

        print(f"\n{'='*70}")
        print(f"REFINING PDDL DOMAIN (Attempt {self.refinement_attempts}/{self.config.max_refinement_attempts})")
        print(f"{'='*70}")
        print(f"Planning Error: {error_message}")

        # T3: fresh geometry recompute including full contact-graph stability analysis.
        # Precondition failures often mean the world state has drifted from the last
        # perception snapshot; a full recompute gives the planner accurate spatial data.
        if self.tracker and hasattr(self.tracker, "trigger_geometry_recompute"):
            recomputed = self.tracker.trigger_geometry_recompute(force_occlusion=False)
            if recomputed:
                print("  • T3: full geometry recomputed (contact graph + clearance + surfaces)")
        print()

        try:
            domain_content = None
            problem_content = None
            if pddl_files and "domain_path" in pddl_files:
                domain_path = Path(pddl_files["domain_path"])
                if domain_path.exists():
                    domain_content = domain_path.read_text()
            if pddl_files and "problem_path" in pddl_files:
                problem_path = Path(pddl_files["problem_path"])
                if problem_path.exists():
                    problem_content = problem_path.read_text()

            if not self.maintainer:
                print("  ⚠ No maintainer available for refinement")
                return False

            validation = await self.maintainer.get_domain_statistics()
            layer = self.maintainer.classify_failure_layer(
                error_message=error_message,
                validation=validation.get("validation"),
            )

            # Print validation issues and suggested repair layer so the cause is visible
            val_detail = validation.get("validation") or {}
            issues = val_detail.get("issues") or []
            suggested = val_detail.get("suggested_repair_layer")
            if issues:
                print(f"  • Validation issues ({len(issues)}):")
                for issue in issues:
                    lyr = issue.get("layer", "?")
                    msg = issue.get("message", "")
                    print(f"      [{lyr}] {msg}")
            if suggested and suggested != layer:
                print(f"  • Suggested repair layer (from validator): {suggested}")
            print(f"  • Targeted repair layer: {layer}")
            repair_record = await self.maintainer.repair_representation(
                failure_context={
                    "error_message": error_message,
                    "domain_path": pddl_files.get("domain_path") if pddl_files else None,
                    "problem_path": pddl_files.get("problem_path") if pddl_files else None,
                    "current_domain_pddl": domain_content,
                    "current_problem_pddl": problem_content,
                    "raw_solver_output": (raw_solver_output or "")[-3000:] or None,
                },
                layer=layer,
            )
            print(f"  ✓ Layer repair complete (valid={repair_record['validation']['valid']})")

            if self.tracker and self._detection_running:
                print("  • Updating ObjectTracker with repaired predicates/actions...")
                await self.maintainer.update_object_tracker_from_domain(self.tracker)
                self.tracker.set_task_context(
                    task_description=self.current_task,
                    available_actions=self.task_analysis.action_context() if self.task_analysis else [],
                    goal_objects=self.task_analysis.goal_object_references() if self.task_analysis else [],
                )

            self.task_analysis = self.maintainer.task_analysis
            self._set_state(OrchestratorState.READY_FOR_PLANNING)
            return True
        except Exception as e:
            print(f"  ⚠ Refinement failed: {e}")
            self._set_state(OrchestratorState.READY_FOR_PLANNING)
            return False

    async def solve_and_plan_with_refinement(
        self,
        algorithm: Optional[SearchAlgorithm] = None,
        timeout: Optional[float] = None,
        output_dir: Optional[Path] = None,
        wait_for_objects: bool = True,
        max_wait_seconds: float = 120.0
    ) -> SolverResult:
        """
        Solve for a plan with automatic domain refinement on failures.

        This method attempts to solve the planning problem, and if it fails due to
        domain issues, it will automatically refine the domain and retry up to
        max_refinement_attempts times.

        Args:
            algorithm: Search algorithm to use (defaults to config)
            timeout: Solver timeout in seconds (defaults to config)
            output_dir: Directory for PDDL files (defaults to config.state_dir / "pddl")
            wait_for_objects: Whether to wait for object detection before planning
            max_wait_seconds: Maximum time to wait for objects (default: 30s)

        Returns:
            SolverResult with plan and statistics
        """
        # Reset refinement attempts for new planning session
        self.refinement_attempts = 0
        self.last_planning_error = None

        while self.refinement_attempts <= self.config.max_refinement_attempts:
            should_prevalidate = True
            if (
                wait_for_objects
                and self.refinement_attempts == 0
                and self.tracker
                and hasattr(self.tracker, "registry")
                and len(self.tracker.registry.get_all_objects()) == 0
            ):
                should_prevalidate = False

            if self.maintainer and should_prevalidate:
                domain_stats = await self.maintainer.get_domain_statistics()
                validation = domain_stats.get("validation", {})
                if validation and not validation.get("valid", True):
                    issues = validation.get("issues") or []
                    suggested = validation.get("suggested_repair_layer", "")
                    print("\n🔧 Representation validation failed before solving; attempting targeted repair...")
                    if issues:
                        for issue in issues:
                            print(f"   [{issue.get('layer','?')}] {issue.get('message','')}")
                    if suggested:
                        print(f"   Suggested repair layer: {suggested}")
                    refined = await self.refine_domain_from_failure(
                        error_message="Representation validation failed before solving",
                        pddl_files={
                            "domain_path": str(output_dir or (self.config.state_dir / "pddl"))
                            + f"/{self.pddl.domain_name}_domain.pddl",
                            "problem_path": str(output_dir or (self.config.state_dir / "pddl"))
                            + f"/{self.pddl.domain_name}_problem.pddl",
                        },
                    )
                    if not refined:
                        self._log_domain_error(
                            error_type="validation_presolve",
                            message="Representation validation failed before solving and could not be repaired",
                            layer=suggested or "unknown",
                            extra={"issues": [i.get("message", "") for i in issues]},
                        )
                        result = SolverResult(
                            success=False,
                            plan=[],
                            plan_length=0,
                            plan_cost=None,
                            search_time=None,
                            nodes_expanded=None,
                            error_message="Representation validation failed before solving",
                        )
                        return result
                    continue

            # Try to solve (wait for objects only on first attempt)
            result = await self.solve_and_plan(
                algorithm=algorithm,
                timeout=timeout,
                generate_files=True,  # Always regenerate after refinement
                output_dir=output_dir,
                wait_for_objects=wait_for_objects and self.refinement_attempts == 0,  # Only wait on first attempt
                max_wait_seconds=max_wait_seconds
            )

            # Snapshot PDDL files for post-run review
            _pddl_base = Path(output_dir) if output_dir else self.config.state_dir / "pddl"
            _snap_dir = _pddl_base / f"attempt_{self.refinement_attempts}"
            _snap_dir.mkdir(parents=True, exist_ok=True)
            for _pddl_file in _pddl_base.glob("*.pddl"):
                shutil.copy2(_pddl_file, _snap_dir / _pddl_file.name)

            # Write failure log (or success note) for this attempt
            _failure_log = _snap_dir / "result.txt"
            with open(_failure_log, "w") as _f:
                _f.write(f"attempt: {self.refinement_attempts}\n")
                _f.write(f"timestamp: {datetime.now(timezone.utc).isoformat()}\n")
                if result.success:
                    _f.write("status: SUCCESS\n")
                    if result.plan:
                        _f.write(f"\nplan ({len(result.plan)} steps):\n")
                        for _step in result.plan:
                            _f.write(f"  {_step}\n")
                else:
                    _f.write("status: FAILURE\n")
                    _f.write(f"error: {result.error_message or 'unknown'}\n")
                if result.raw_output:
                    _f.write(f"\nsolver output:\n{result.raw_output}\n")

            # Success - return result
            if result.success:
                if self.refinement_attempts > 0:
                    print(f"\n✓ Planning succeeded after {self.refinement_attempts} refinement(s)")
                return result

            if not self.config.auto_refine_on_failure:
                self._log_domain_error(
                    error_type="solver_failure",
                    message=result.error_message or "No plan found",
                    layer="solver",
                    refinement_attempt=self.refinement_attempts,
                    extra={"raw_output": (result.raw_output or "")[-2000:]},
                )
                print(f"\n⚠ Auto-refinement disabled. Use refine_domain_from_failure() manually.")
                if result.raw_output:
                    print(f"\n--- Solver raw output ---\n{result.raw_output[-3000:]}\n---")
                return result

            # Also check raw_output for refinable signals when error_message is generic.
            _refinable = self._is_refinable_error(result.error_message) or (
                result.raw_output and self._is_refinable_error(result.raw_output)
            )
            if not _refinable:
                self._log_domain_error(
                    error_type="solver_non_refinable",
                    message=result.error_message or "No plan found",
                    layer="solver",
                    refinement_attempt=self.refinement_attempts,
                    extra={"raw_output": (result.raw_output or "")[-2000:]},
                )
                print(f"\n✗ Planning failed with non-refinable error")
                if result.raw_output:
                    print(f"\n--- Solver raw output ---\n{result.raw_output[-3000:]}\n---")
                return result

            self._log_domain_error(
                error_type="solver_failure",
                message=result.error_message or "No plan found",
                layer="solver",
                refinement_attempt=self.refinement_attempts,
            )
            print(f"\n🔧 Planning failed; attempting targeted layer repair: {result.error_message}...")

            # Get PDDL file paths
            pddl_files = {
                "domain_path": str(output_dir or (self.config.state_dir / "pddl")) + f"/{self.pddl.domain_name}_domain.pddl",
                "problem_path": str(output_dir or (self.config.state_dir / "pddl")) + f"/{self.pddl.domain_name}_problem.pddl"
            }

            refined = await self.refine_domain_from_failure(
                error_message=result.error_message,
                pddl_files=pddl_files,
                raw_solver_output=result.raw_output,
            )

            if not refined:
                # Append refinement failure note to the existing log
                with open(_failure_log, "a") as _f:
                    _f.write("\nrefinement: FAILED (no improvement possible)\n")
                self._log_domain_error(
                    error_type="refinement_failed",
                    message=f"Refinement attempt {self.refinement_attempts} produced no improvement",
                    layer="solver",
                    refinement_attempt=self.refinement_attempts,
                )
                print(f"\n✗ Could not refine domain further")
                return result

            # Loop will retry with repaired representation

        self._log_domain_error(
            error_type="refinement_limit",
            message=f"Max refinement attempts ({self.config.max_refinement_attempts}) reached without a valid plan",
            layer="solver",
            refinement_attempt=self.refinement_attempts,
            extra={"last_error": self.last_planning_error},
        )
        print(f"\n✗ Max refinement attempts reached")

        # Escalate to hybrid planner (probabilistic / hindsight) when all
        # deterministic refinement attempts are exhausted (spec §5.3).
        if self._hybrid_planner is not None:
            registry = getattr(getattr(self, "tracker", None), "registry", None)
            if registry is not None:
                pddl_dir = Path(output_dir or (self.config.state_dir / "pddl"))
                domain_path = pddl_dir / f"{self.pddl.domain_name}_domain.pddl"
                problem_path = pddl_dir / f"{self.pddl.domain_name}_problem.pddl"
                if domain_path.exists() and problem_path.exists():
                    print("\n🔀 Escalating to hybrid probabilistic planner...")
                    try:
                        hybrid_result = await self._hybrid_planner.plan(
                            domain_path=domain_path,
                            problem_path=problem_path,
                            registry=registry,
                        )
                        if hybrid_result.success:
                            print(
                                f"✓ Hybrid planner succeeded (mode={hybrid_result.mode}, "
                                f"plan_length={len(hybrid_result.plan)})"
                            )
                            return SolverResult(
                                success=True,
                                plan=hybrid_result.plan,
                                plan_length=len(hybrid_result.plan),
                                plan_cost=None,
                                search_time=None,
                                nodes_expanded=None,
                                error_message=None,
                            )
                        else:
                            self._log_domain_error(
                                error_type="hybrid_planner_failed",
                                message=hybrid_result.error or "Hybrid planner returned no plan",
                                layer="hybrid",
                                extra={"mode": hybrid_result.mode},
                            )
                            print(f"✗ Hybrid planner also failed: {hybrid_result.error}")
                    except Exception as exc:
                        self._log_domain_error(
                            error_type="hybrid_planner_exception",
                            message=str(exc),
                            layer="hybrid",
                        )
                        self.logger.warning("Hybrid planner escalation failed: %s", exc)

        return result

    def _log_domain_error(
        self,
        error_type: str,
        message: str,
        layer: Optional[str] = None,
        refinement_attempt: Optional[int] = None,
        pddl_snippet: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Write one record to domain_errors.jsonl via the DKB (no-op if DKB unavailable)."""
        if self._dkb is None:
            return
        import re as _re
        run_id = _re.search(r"(\d{8}_\d{6})", str(getattr(self.config, "state_dir", "")))
        self._dkb.record_domain_error(
            error_type=error_type,
            message=message,
            task=self.current_task,
            layer=layer,
            run_id=run_id.group(1) if run_id else None,
            refinement_attempt=refinement_attempt,
            pddl_snippet=pddl_snippet,
            extra=extra,
        )

    def _is_refinable_error(self, error_message: Optional[str]) -> bool:
        """
        Check if an error message indicates a refinable domain issue.

        Returns:
            True if the error appears to be fixable via domain refinement
        """
        if not error_message:
            return False

        # Common refinable error patterns
        refinable_patterns = [
            # Domain syntax/structure errors
            "wrong number of arguments",
            "predicate",
            "arity",
            "parse error",
            "undefined predicate",
            "type error",
            "action",
            "precondition",
            "effect",
            "parameter",
            # Object definition errors
            "object",
            "referenced",
            "not defined",
            "but not defined",
            "undefined object",
            # Planning failures that might indicate domain issues
            "no plan found",
            "no solution",
            "unsolvable",
            "empty plan",
            "goals",  # Goal-related issues
        ]

        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in refinable_patterns)

    # ========================================================================
    # State Persistence
    # ========================================================================

    def _build_enhanced_registry(self) -> Dict[str, Any]:
        """
        Build an enhanced registry with snapshot references.
        
        The enhanced format includes snapshot references to show observation history,
        while keeping full position history in each snapshot's detections.json file.
        
        Returns:
            Dict with registry data including snapshot references
        """
        if not self.tracker:
            return {"num_objects": 0, "detection_timestamp": datetime.now().isoformat(), "objects": []}
        
        # Load perception pool index to get snapshot references
        index_path = self._get_pool_index_path()
        snapshot_refs = {}
        if index_path.exists():
            with open(index_path, 'r') as f:
                pool_index = json.load(f)
                snapshot_refs = pool_index.get("objects", {})
        
        # Build enhanced object entries
        registry = self.tracker.registry
        objects_data = []

        with registry._lock:
            for obj in registry._objects.values():
                # Get snapshot references for this object
                obj_snapshots = snapshot_refs.get(obj.object_id, [])
                latest_snapshot = obj_snapshots[-1] if obj_snapshots else None
                
                obj_dict = {
                    "object_type": obj.object_type,
                    "object_id": obj.object_id,
                    "affordances": list(obj.affordances),
                    "timestamp": obj.timestamp,
                    # Snapshot references - this is the key addition
                    "observations": obj_snapshots,
                    "latest_observation": latest_snapshot,
                }

                objects_data.append(obj_dict)
        
        return {
            "version": "2.0",
            "num_objects": len(objects_data),
            "detection_timestamp": datetime.now().isoformat(),
            "objects": objects_data
        }

    # ========================================================================
    # Full-pipeline execution
    # ========================================================================

    async def execute_task(
        self,
        task_description: str,
        output_dir: Optional[Path] = None,
        detection_timeout: float = 120.0,
        primitives: Optional[Any] = None,
    ) -> "TaskExecutionResult":
        """Drive the complete perception → planning → execution pipeline.

        This is the top-level entry point for task execution.  It orchestrates:

        1. **Perception** — starts GSAM2 detection, waits for at least one
           detection cycle with objects, then stops the camera.
        2. **Task analysis + domain generation** — runs the L1→L5 layered
           domain generator against the detected scene.
        3. **PDDL solving** — calls ``solve_and_plan_with_refinement`` with
           automatic domain repair on failure and hybrid-planner escalation.
        4. **Skill decomposition + execution** — for each symbolic action,
           ``SkillDecomposer`` breaks it into primitives; ``PrimitiveExecutor``
           translates coordinates and runs them on *primitives*.  If no
           ``primitives`` interface is supplied the step runs as a dry-run.
        5. **Conditional branching** — ``ConditionalTaskExecutor`` evaluates
           check-* predicates at runtime against the live registry and
           triggers replanning when they fail.

        Args:
            task_description: Natural-language task (e.g. "pick up the red cup").
            output_dir: Directory for PDDL files and run artefacts.  Defaults
                to ``config.state_dir / "pddl"``.
            detection_timeout: Seconds to wait for at least one detection before
                giving up (default 120 s).
            primitives: Robot primitives interface (e.g. ``XArmPybulletPrimitives``
                or ``XArmPybulletPlannedPrimitives``).  When *None* each primitive
                step is executed as a dry-run and logged rather than sent to a robot.

        Returns:
            ``TaskExecutionResult`` with success flag, plan, execution trace, and
            timing breakdown.
        """
        from src.primitives.skill_decomposer import SkillDecomposer
        from src.primitives.primitive_executor import PrimitiveExecutor
        from src.planning.conditional_task_executor import (
            ConditionalTaskExecutor,
            ConditionalExecutionResult,
        )
        from src.planning.clutter_module import PostDisplacementHook

        if output_dir is None:
            output_dir = self.config.state_dir / "pddl"
        output_dir = Path(output_dir)

        timings: Dict[str, float] = {}
        _llm_client = getattr(self.config, "llm_client", None)

        def _tick() -> float:
            return time.monotonic()

        def _tock(label: str, t0: float) -> None:
            timings[label] = time.monotonic() - t0

        # ── 1. Perception ────────────────────────────────────────────────
        self.logger.info("[execute_task] Phase 1: perception")
        t0 = _tick()
        self.current_task = task_description
        if self.tracker:
            self.tracker.set_task_context(task_description=task_description)

        await self.start_detection()
        deadline = time.monotonic() + detection_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            if self.detection_count >= 1 and self.get_detected_objects():
                break
            self.logger.debug(
                "[execute_task] waiting for detection (%d cycles)", self.detection_count
            )
        await self.stop_detection()

        registry = self.tracker.registry if self.tracker else None
        detected = self.get_detected_objects()
        _tock("perception", t0)
        self.logger.info(
            "[execute_task] Perception complete: %d objects in %.1fs",
            len(detected), timings["perception"],
        )

        if not detected:
            self.logger.error("[execute_task] No objects detected — aborting")
            return TaskExecutionResult(
                success=False,
                task=task_description,
                error="No objects detected after perception pass",
                timings=timings,
            )

        # Capture masks for depth collider (if a depth collider is attached to
        # the robot interface via config.robot)
        _masks: Dict[str, Any] = dict(
            getattr(getattr(self.tracker, "_last_masks", None), "__self__", {})
            if False else
            getattr(self.tracker, "_last_masks", {})
        )

        # ── 2. Task analysis + domain generation ─────────────────────────
        self.logger.info("[execute_task] Phase 2: task analysis + domain generation")
        t0 = _tick()
        last_frame = getattr(self.tracker, "last_color_frame", None)
        await self.process_task_request(task_description, environment_image=last_frame)
        _tock("domain_generation", t0)
        self.logger.info(
            "[execute_task] Domain generation complete in %.1fs",
            timings["domain_generation"],
        )

        # ── 3. PDDL solving ──────────────────────────────────────────────
        self.logger.info("[execute_task] Phase 3: PDDL solving")
        t0 = _tick()
        solver_result = await self.solve_and_plan_with_refinement(
            output_dir=output_dir,
            wait_for_objects=False,
        )
        _tock("solving", t0)
        self.logger.info(
            "[execute_task] Solving complete in %.1fs — success=%s plan_length=%d",
            timings["solving"], solver_result.success,
            solver_result.plan_length if solver_result else 0,
        )

        if not solver_result or not solver_result.success or not solver_result.plan:
            err = (solver_result.error_message if solver_result else None) or "no plan found"
            self.logger.error("[execute_task] Planning failed: %s", err)
            return TaskExecutionResult(
                success=False,
                task=task_description,
                plan=solver_result.plan if solver_result else [],
                error=err,
                timings=timings,
            )

        # ── 4. Skill decomposition + execution ───────────────────────────
        self.logger.info("[execute_task] Phase 4: skill decomposition + execution")
        t0 = _tick()

        # Build skill decomposer (requires LLM client)
        decomposer: Optional[SkillDecomposer] = None
        prim_executor: Optional[PrimitiveExecutor] = None
        if _llm_client is not None:
            decomposer = SkillDecomposer(
                llm_client=_llm_client, orchestrator=self
            )
            prim_executor = PrimitiveExecutor(
                primitives=primitives,
                perception_pool_dir=self._get_perception_pool_dir(),
                logger=self.logger.getChild("PrimitiveExecutor"),
                orchestrator=self,
            )
        else:
            self.logger.warning(
                "[execute_task] No LLM client — skill decomposition will be skipped"
            )

        # Camera pose for PostDisplacementHook
        cam_pos = cam_rot = None
        robot = getattr(self.config, "robot", None)
        if robot is not None:
            try:
                cam_pos, cam_rot = robot.get_camera_transform()
            except Exception:
                pass

        post_hook = PostDisplacementHook(
            registry=registry,
            camera_intrinsics=None,
            cam_position=cam_pos,
            cam_rotation=cam_rot,
            logger=self.logger.getChild("PostDisplacementHook"),
        )

        async def _replan() -> List[str]:
            self.logger.info("[execute_task] Replanning after contingency...")
            lf = getattr(self.tracker, "last_color_frame", None)
            await self.process_task_request(task_description, environment_image=lf)
            r = await self.solve_and_plan_with_refinement(
                output_dir=output_dir, wait_for_objects=False
            )
            return r.plan if r and r.success else []

        def _execute_action(action_str: str) -> bool:
            tokens = action_str.strip("() ").split()
            action_name = tokens[0] if tokens else action_str
            param_ids = tokens[1:] if len(tokens) > 1 else []
            params = {"objects": param_ids, "action_str": action_str}

            self.logger.info("[execute_task] execute: %s", action_str)
            if decomposer is None or prim_executor is None:
                self.logger.info(
                    "[execute_task] skill decomposition skipped (no LLM client)"
                )
                return True

            try:
                skill_plan = decomposer.plan(action_name, params)
                self.logger.info(
                    "[execute_task] decomposed → %s",
                    [p.name for p in skill_plan.primitives],
                )
                for w in (skill_plan.diagnostics.warnings or []):
                    self.logger.warning("[execute_task] %s", w)

                snapshot_id = self.capture_fresh_snapshot_sync(
                    reason=f"pre-execution:{action_name}"
                )
                world_state = {"state_dir": str(self.config.state_dir)}
                if snapshot_id:
                    world_state["last_snapshot_id"] = snapshot_id

                dry_run = primitives is None
                exec_result = prim_executor.execute_plan(
                    skill_plan, world_state=world_state, dry_run=dry_run
                )

                if dry_run:
                    self.logger.info("[execute_task] dry-run OK for %s", action_name)
                    return True

                self.logger.info(
                    "[execute_task] executed %d primitive(s) for %s",
                    len(exec_result.primitive_results), action_name,
                )
                succeeded = all(
                    r.get("success", True) if isinstance(r, dict) else True
                    for r in exec_result.primitive_results
                )

                # Calibrate hybrid planner
                if self._hybrid_planner is not None and action_name in (
                    "displace", "push-aside"
                ):
                    outcome = "full_success" if succeeded else "no_change"
                    self._hybrid_planner.record_execution_outcome(
                        action_name, outcome, succeeded
                    )

                return succeeded

            except Exception as exc:
                self.logger.error(
                    "[execute_task] action '%s' failed: %s", action_name, exc
                )
                return False

        cond_executor = ConditionalTaskExecutor(
            registry=registry,
            execute_action=_execute_action,
            replan_fn=_replan,
            max_replan=getattr(self.config, "max_refinement_attempts", 2),
            post_displacement_hook=post_hook,
            logger=self.logger.getChild("ConditionalExecutor"),
        )

        self.logger.info(
            "[execute_task] Executing plan (%d steps): %s",
            solver_result.plan_length, solver_result.plan,
        )
        self._set_state(OrchestratorState.EXECUTING_PLAN)
        cond_result: ConditionalExecutionResult = await cond_executor.execute(
            solver_result.plan
        )
        _tock("execution", t0)

        if cond_result.success:
            self._set_state(OrchestratorState.TASK_COMPLETE)
            self.logger.info(
                "[execute_task] Task complete in %.1fs (%d steps, %d replan(s))",
                timings["execution"],
                len(cond_result.steps),
                cond_result.replan_count,
            )
        else:
            self._set_state(OrchestratorState.ERROR)
            self.logger.error(
                "[execute_task] Task failed: %s", cond_result.error
            )

        return TaskExecutionResult(
            success=cond_result.success,
            task=task_description,
            plan=solver_result.plan,
            steps=cond_result.steps,
            replan_count=cond_result.replan_count,
            error=cond_result.error,
            timings=timings,
        )

    async def save_state(self, path: Optional[Path] = None) -> Path:
        """
        Save orchestrator state to disk.

        Saves:
        - PDDL domain and problem
        - Object registry (with snapshot references)
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

        # Save enhanced registry with snapshot references
        registry_path = path.parent / "registry.json"
        if self.tracker:
            enhanced_registry = self._build_enhanced_registry()
            with open(registry_path, 'w') as f:
                json.dump(enhanced_registry, f, indent=2)
            self.logger.info("✓ Saved %s objects to %s", enhanced_registry["num_objects"], registry_path)

        # Save PDDL files
        pddl_dir = path.parent / "pddl"
        if self.pddl:
            # Ensure goals are populated before writing files
            if self.maintainer and self.task_analysis:
                await self.maintainer.set_goal_from_task_analysis()

            # Generate files and capture actual paths to avoid mismatches
            try:
                pddl_paths = await self.pddl.generate_files_async(str(pddl_dir))
            except Exception:
                # Fall back to attempting generation without capturing paths
                await self.pddl.generate_files_async(str(pddl_dir))
                pddl_paths = None

        # Save orchestrator state
        state_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "orchestrator_state": self._state.value,
            "current_task": self.current_task,
            "detection_count": self.detection_count,
            "last_snapshot_id": self.last_snapshot_id,
            "task_analysis": {
                "abstract_goal": {
                    "summary": self.task_analysis.abstract_goal.summary,
                    "goal_literals": self.task_analysis.abstract_goal.goal_literals,
                    "goal_objects": self.task_analysis.abstract_goal.goal_objects,
                } if self.task_analysis else {},
                "predicate_inventory": self.task_analysis.predicate_inventory.predicates if self.task_analysis else [],
                "grounding_summary": {
                    "object_bindings": self.task_analysis.grounding_summary.object_bindings,
                    "missing_references": self.task_analysis.grounding_summary.missing_references,
                } if self.task_analysis else {},
                "diagnostics": self.task_analysis.diagnostics if self.task_analysis else {},
            } if self.task_analysis else None,
            "files": {
                "registry": str(registry_path),
                "domain": (pddl_paths.get("domain_path") if pddl_paths else str(pddl_dir / f"{self.pddl.domain_name}_domain.pddl")) if self.pddl else None,
                "problem": (pddl_paths.get("problem_path") if pddl_paths else str(pddl_dir / f"{self.pddl.domain_name}_problem.pddl")) if self.pddl else None,
                "perception_pool_index": str(self._get_pool_index_path()) if (self._get_pool_index_path().exists()) else None
            }
        }

        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2)

        self._last_save_time = time.time()
        self.logger.info("✓ State saved to %s", path)

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

        self.logger.info("Loading state from %s...", path)

        with open(path, 'r') as f:
            state_data = json.load(f)

        # Load object registry into tracker's registry
        registry_path = state_data["files"]["registry"]
        if Path(registry_path).exists() and self.tracker:
            self.tracker.registry.load_from_json(registry_path)
            self.logger.info("  • Loaded %s objects", len(self.tracker.registry))

        # Load task information
        self.current_task = state_data.get("current_task")
        self.detection_count = state_data.get("detection_count", 0)
        self.last_snapshot_id = state_data.get("last_snapshot_id")

        # Restore task analysis if available
        task_analysis_data = state_data.get("task_analysis")
        if task_analysis_data and self.current_task:
            # Re-analyze task to get full TaskAnalysis object
            self.logger.info("  • Re-analyzing task: '%s'", self.current_task)
            await self.process_task_request(self.current_task)

            # Re-process existing observations from tracker's registry
            all_objects = self.get_detected_objects()
            if all_objects:
                self.logger.info("  • Re-processing %s objects...", len(all_objects))
                objects_dict = self._convert_objects_to_dict(all_objects)
                predicates = self.tracker.registry.get_all_predicates()
                await self.maintainer.update_from_observations(objects_dict, predicates=predicates)

        # Load perception pool index into memory if present
        files = state_data.get("files", {})
        pool_index_path = files.get("perception_pool_index")
        if pool_index_path:
            p = Path(pool_index_path)
            if p.exists():
                try:
                    with open(p, "r") as f:
                        self._perception_pool_index = json.load(f)
                        self.last_snapshot_id = self._perception_pool_index.get("last_snapshot_id", self.last_snapshot_id)
                        self.logger.info(
                            "  • Perception pool index loaded with %s snapshots",
                            len(self._perception_pool_index.get("snapshots", {})),
                        )
                except Exception:
                    self._perception_pool_index = self._init_empty_pool_index()
            else:
                # Initialize empty if directory configured but index missing
                self._perception_pool_index = self._init_empty_pool_index()

        self.logger.info("✓ State loaded successfully")

    async def _try_auto_save(self) -> None:
        """
        Attempt to auto-save state (non-blocking, with lock).
        
        Only saves if we have meaningful state (task and detections).
        Silently handles errors to avoid disrupting the main flow.
        """
        # Quick check without lock
        if not self.current_task or self.detection_count == 0:
            return
        
        # Try to acquire lock without blocking
        if self._save_lock.locked():
            return  # Save already in progress, skip
        
        async with self._save_lock:
            try:
                path = await self.save_state()
                if self.config.on_save_state:
                    self.config.on_save_state(path)
            except Exception as e:
                # Silent failure - don't disrupt the main flow
                pass

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    async def _auto_solve(self) -> None:
        """
        Automatically solve for a plan when ready.

        Called internally when auto_solve_when_ready is enabled and
        the orchestrator transitions to READY_FOR_PLANNING state.
        """
        try:
            print("\n🤖 Auto-solve triggered...")
            # Use refinement if enabled
            if self.config.auto_refine_on_failure:
                await self.solve_and_plan_with_refinement()
            else:
                await self.solve_and_plan()
        except Exception as e:
            print(f"⚠ Auto-solve failed: {e}")

    def _set_state(self, new_state: OrchestratorState) -> None:
        """Update orchestrator state and notify callback."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            if self.config.on_state_change:
                self.config.on_state_change(old_state, new_state)
            
            # Auto-save on state change if enabled
            if self.config.auto_save and self.config.auto_save_on_state_change:
                asyncio.create_task(self._try_auto_save())

    def __repr__(self) -> str:
        """String representation."""
        num_objects = len(self.tracker.registry) if self.tracker else 0
        return (
            f"TaskOrchestrator("
            f"state={self._state.value}, "
            f"task='{self.current_task or 'None'}', "
            f"objects={num_objects}, "
            f"detections={self.detection_count}"
            f")"
        )
