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
import time
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
from .llm_task_analyzer import TaskAnalysis
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
        self.logger = get_structured_logger("TaskOrchestrator")
        self._state = OrchestratorState.UNINITIALIZED
        self._camera = camera
        self._external_camera = camera is not None  # Track if camera is externally managed

        # Core components (initialized in initialize())
        self.pddl: Optional[PDDLRepresentation] = pddl_representation
        self.maintainer: Optional[PDDLDomainMaintainer] = None
        self.monitor: Optional[TaskStateMonitor] = None
        self.tracker: Optional[ContinuousObjectTracker] = None

        # Task state
        self.current_task: Optional[str] = None
        self.task_analysis: Optional[TaskAnalysis] = None
        self.last_task_decision: Optional[TaskStateDecision] = None

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
        if self._camera is None:
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
        else:
            self.logger.info("  • Using provided camera")

        # Initialize PDDL representation if not provided
        if self.pddl is None:
            self.pddl = PDDLRepresentation(
                domain_name="task_execution",
                problem_name="current_task"
            )
            self.logger.info("  • PDDL representation created")

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
        self.logger.info("  • PDDL domain maintainer and monitor initialized")

        # Initialize continuous tracker
        self.tracker = ContinuousObjectTracker(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            fast_mode=self.config.fast_mode,
            update_interval=self.config.update_interval,
            on_detection_complete=self._on_detection_callback,
            logger=self.logger.getChild("ObjectTracker"),
        )

        # Set frame provider
        self.tracker.set_frame_provider(self._get_camera_frames)
        self.logger.info("  • Continuous object tracker initialized")

        # Attach default robot provider if none supplied (xArm CuRobo interface)
        if getattr(self.config, "robot", None) is None:
            try:
                from ..kinematics.xarm_curobo_interface import CuRoboMotionPlanner
                self.config.robot = CuRoboMotionPlanner()
                self.logger.info("  • Default robot provider attached: CuRoboMotionPlanner")
            except Exception as e:
                self.logger.warning("  • No robot provider attached (default xArm initialization failed: %s)", e)

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
        self.task_analysis = await self.maintainer.initialize_from_task(
            task_description,
            environment_image=environment_image
        )

        self.logger.info("✓ Task analyzed!")
        self.logger.info("  • Goal objects: %s", ", ".join(self.task_analysis.goal_objects))
        self.logger.info("  • Estimated steps: %s", self.task_analysis.estimated_steps)
        self.logger.info("  • Complexity: %s", self.task_analysis.complexity)
        self.logger.info("  • Required predicates: %s", len(self.task_analysis.relevant_predicates))

        # Seed perception with predicates
        if self.tracker:
            self.logger.info("  • Configuring perception with %s predicates...", len(self.task_analysis.relevant_predicates))
            self.tracker.set_pddl_predicates(self.task_analysis.relevant_predicates)

        self.logger.info("%s", "=" * 70)

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
        try:
            color, depth = self._camera.get_aligned_frames()
            intrinsics = self._camera.get_camera_intrinsics()
            return color, depth, intrinsics
        except Exception as e:
            self.logger.warning("Camera error: %s", e)
            return None, None, None

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
                "pddl_state": obj.pddl_state,
                "interaction_points": {
                    affordance: {
                        "snapshot_id": None,  # filled by snapshot_id at write-time
                        "position_2d": point.position_2d,
                        "position_3d": point.position_3d.tolist() if point.position_3d is not None else None,
                        "confidence": point.confidence,
                        "reasoning": point.reasoning,
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

            color = depth = intrinsics = None
            color_img = None

            if bundle_color_bytes is not None:
                depth = bundle_depth
                intrinsics = bundle_intrinsics
            else:
                color, depth, intrinsics = self._get_camera_frames()
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
            robot_state = self._get_robot_state_struct()
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
        self.logger.info("  • Goal literals: %s", len(problem_snapshot["goal_literals"]))

        self.logger.info("%s", "=" * 70)

        return paths

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
                    "properties": obj.properties,
                    "confidence": obj.confidence,
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
            await self.pddl.generate_files_async(str(pddl_dir))

        # Save orchestrator state
        state_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "orchestrator_state": self._state.value,
            "current_task": self.current_task,
            "detection_count": self.detection_count,
            "last_snapshot_id": self.last_snapshot_id,
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
                await self.maintainer.update_from_observations(objects_dict)

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
