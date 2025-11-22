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
import json
import time
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

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

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Log, Input, Static

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
        min_observations: int = 3,
        output_base_dir: Optional[str] = None,
        on_event: Optional[Callable[[str, dict], None]] = None
    ):
        """
        Initialize the integration system.

        Args:
            api_key: Google AI API key
            task_description: Natural language task description
            camera: RealSense camera (if None, will try to create one)
            update_interval: Seconds between detection updates
            min_observations: Minimum observations before planning
            output_base_dir: Base directory for outputs (None = auto-generate with timestamp)
            on_event: Callback for events (event_type: str, data: dict)
        """
        self.api_key = api_key
        self.task_description = task_description
        self.update_interval = update_interval
        self._on_event = on_event or (lambda event_type, data: None)
        
        # Set up timestamped output directory
        if output_base_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"outputs/pddl/continuous_{timestamp}"
        else:
            self.output_dir = output_base_dir

        # Initialize camera
        if camera is None:
            self._emit_event("camera_init_start")
            self.camera = RealSenseCamera(
                width=640,
                height=480,
                fps=30,
                enable_depth=True,
                auto_start=True
            )
            self._emit_event("camera_init_complete")
        else:
            self.camera = camera

        # Initialize PDDL components
        self._emit_event("pddl_init_start")
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
        self._emit_event("pddl_init_complete")

        # Initialize continuous tracker
        self._emit_event("tracker_init_start")
        self.tracker = ContinuousObjectTracker(
            api_key=api_key,
            fast_mode=False,  # Full detection with interaction points
            update_interval=update_interval,
            on_detection_complete=self._on_detection_callback
        )

        # Set frame provider
        self.tracker.set_frame_provider(self._get_camera_frames)
        self._emit_event("tracker_init_complete")

        # State tracking
        self.task_analysis = None
        self.detection_count = 0
        self.last_state = None
        self.ready_for_planning = False
        self._running = False
        self._known_object_ids = set()  # Track known objects to identify new ones

    def _emit_event(self, event_type: str, data: Optional[dict] = None):
        """Emit an event to the event handler."""
        self._on_event(event_type, data or {})

    def _get_camera_frames(self):
        """Frame provider for continuous tracker."""
        try:
            color, depth = self.camera.get_aligned_frames()
            intrinsics = self.camera.get_camera_intrinsics()
            return color, depth, intrinsics
        except Exception as e:
            self._emit_event("camera_error", {"error": str(e)})
            return None, None, None

    async def _on_detection_callback(self, object_count: int):
        """
        Called after each detection cycle.

        Updates the PDDL domain and checks task state.
        """
        self.detection_count += 1

        # Get all detected objects from tracker's built-in registry
        all_objects = self.tracker.get_all_objects()

        if not all_objects:
            self._emit_event("detection_update", {
                "detection_count": self.detection_count,
                "object_count": 0,
                "has_objects": False
            })
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
        update_stats = await self.maintainer.update_from_observations(objects_dict)

        # Check task state
        decision = await self.monitor.determine_state()

        # Track state change
        state_changed = self.last_state != decision.state
        old_state = self.last_state
        if state_changed:
            self.last_state = decision.state

        # Check if ready for planning
        newly_ready = False
        if decision.state == TaskState.PLAN_AND_EXECUTE:
            if not self.ready_for_planning:
                self.ready_for_planning = True
                newly_ready = True

        # Identify new objects
        current_object_ids = {obj.object_id for obj in all_objects}
        new_object_ids = current_object_ids - self._known_object_ids
        self._known_object_ids = current_object_ids
        
        # Build detailed info for new objects only
        new_objects_detail = []
        for obj in all_objects:
            if obj.object_id in new_object_ids:
                # Extract predicates/relationships from pddl_state
                predicates = []
                if obj.pddl_state:
                    for key, value in obj.pddl_state.items():
                        if isinstance(value, bool) and value:
                            predicates.append(key)
                        elif value and key not in ['object_id', 'object_type']:
                            predicates.append(f"{key}={value}")
                
                new_objects_detail.append({
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "predicates": predicates,
                    "affordances": list(obj.affordances)
                })

        # Emit single event with all data
        self._emit_event("detection_update", {
            "detection_count": self.detection_count,
            "object_count": object_count,
            "has_objects": True,
            "update_stats": update_stats,
            "decision": {
                "state": decision.state.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "blockers": decision.blockers,
                "recommendations": decision.recommendations
            },
            "state_changed": state_changed,
            "old_state": old_state.value if old_state else "INIT",
            "newly_ready": newly_ready,
            "new_objects": new_objects_detail
        })
        
        # Auto-save both PDDL and state continuously after each update
        try:
            await self.generate_pddl_files()
            self._emit_event("pddl_auto_saved", {"detection_count": self.detection_count})
        except Exception as e:
            self._emit_event("pddl_save_error", {"error": str(e)})
        
        # Save state (registry + metadata) alongside PDDL
        try:
            await self.save_state()
            self._emit_event("state_auto_saved", {"detection_count": self.detection_count})
        except Exception as e:
            self._emit_event("state_save_error", {"error": str(e)})

    async def initialize_from_task(self):
        """
        Analyze task and initialize PDDL domain.
        
        Returns:
            Task analysis result
        """
        self._emit_event("task_analysis_start", {"task": self.task_description})

        # Capture initial frame for context
        color_frame, _ = self.camera.get_aligned_frames()

        # Analyze task
        self.task_analysis = await self.maintainer.initialize_from_task(
            self.task_description,
            environment_image=color_frame
        )

        # Seed perception with predicates
        self.tracker.tracker.set_pddl_predicates(self.task_analysis.relevant_predicates)
        
        self._emit_event("task_analysis_complete", {
            "goal_objects": self.task_analysis.goal_objects,
            "estimated_steps": self.task_analysis.estimated_steps,
            "complexity": self.task_analysis.complexity,
            "predicates": self.task_analysis.relevant_predicates
        })
        
        return self.task_analysis

    def start_tracking(self):
        """Start continuous tracking loop."""
        self._emit_event("tracking_start", {"update_interval": self.update_interval})
        self.tracker.start()
        self._running = True
        self._emit_event("tracking_started")

    async def stop_tracking(self):
        """Stop continuous tracking loop."""
        self._emit_event("tracking_stop")
        self._running = False
        await self.tracker.stop()
        self._emit_event("tracking_stopped")

    async def generate_pddl_files(self, output_dir: Optional[str] = None):
        """
        Generate final PDDL files.

        Args:
            output_dir: Directory to save PDDL files (None = use default timestamped dir)

        Returns:
            Dict with paths to generated files and summaries
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        self._emit_event("pddl_generate_start", {"output_dir": output_dir})

        # Set goals if not already set
        await self.maintainer.set_goal_from_task_analysis()

        # Generate files
        paths = await self.pddl.generate_files_async(output_dir)

        # Get summary
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        result = {
            "paths": paths,
            "domain_summary": {
                "types": len(domain_snapshot['object_types']),
                "predicates": len(domain_snapshot['predicates']),
                "predefined_actions": len(domain_snapshot['predefined_actions']),
                "llm_generated_actions": len(domain_snapshot['llm_generated_actions'])
            },
            "problem_summary": {
                "object_instances": len(problem_snapshot['object_instances']),
                "initial_literals": len(problem_snapshot['initial_literals']),
                "goal_literals": len(problem_snapshot['goal_literals'])
            }
        }
        
        self._emit_event("pddl_generate_complete", result)
        return result

    async def get_status(self):
        """Get current system status.
        
        Returns:
            Dict with status information
        """
        stats = await self.tracker.get_stats()
        domain_stats = await self.maintainer.get_domain_statistics()
        decision = await self.monitor.determine_state()
        
        # Get PDDL snapshots for actions and goals
        domain_snapshot = await self.pddl.get_domain_snapshot()
        problem_snapshot = await self.pddl.get_problem_snapshot()

        return {
            "task_description": self.task_description,
            "tracking": {
                "is_running": stats.is_running,
                "detection_cycles": stats.total_frames,
                "frames_with_detection": stats.total_frames - stats.skipped_frames,
                "frames_skipped": stats.skipped_frames,
                "cache_hit_rate": stats.cache_hit_rate,
                "total_detections": stats.total_detections,
                "avg_detection_time": stats.avg_detection_time,
                "last_detection_time": stats.last_detection_time
            },
            "pddl": domain_stats,
            "task_state": {
                "current": decision.state.value,
                "confidence": decision.confidence,
                "ready_for_planning": self.ready_for_planning
            },
            "actions": {
                "predefined": domain_snapshot.get('predefined_actions', []),
                "llm_generated": domain_snapshot.get('llm_generated_actions', [])
            },
            "goals": problem_snapshot.get('goal_literals', [])
        }

    async def save_state(self, path: Optional[Path] = None) -> Path:
        """
        Save system state to disk.

        Saves:
        - PDDL domain and problem
        - Object registry
        - Task information
        - System metadata

        Args:
            path: Path to save state (defaults to output_dir / "state.json")

        Returns:
            Path to saved state file
        """
        if path is None:
            path = Path(self.output_dir) / "state.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._emit_event("save_state_start", {"path": str(path)})

        # Save object registry from tracker
        registry_path = path.parent / "registry.json"
        self.tracker.tracker.registry.save_to_json(str(registry_path), include_timestamp=False)

        # Save PDDL files
        pddl_dir = path.parent / "pddl"
        if self.pddl:
            await self.pddl.generate_files_async(str(pddl_dir))

        # Save system state
        state_data = {
            "version": "1.0",
            "timestamp": time.time(),
            "task_description": self.task_description,
            "detection_count": self.detection_count,
            "ready_for_planning": self.ready_for_planning,
            "last_state": self.last_state.value if self.last_state else None,
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

        self._emit_event("save_state_complete", {"path": str(path)})
        return path

    async def load_state(self, path: Optional[Path] = None) -> None:
        """
        Load system state from disk.

        Args:
            path: Path to state file (defaults to output_dir / "state.json")
        """
        if path is None:
            path = Path(self.output_dir) / "state.json"

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        self._emit_event("load_state_start", {"path": str(path)})

        with open(path, 'r') as f:
            state_data = json.load(f)

        # Load components using their respective load methods
        await self._load_registry(state_data)
        await self._load_task_state(state_data)
        await self._restore_pddl_domain(state_data)

        self._emit_event("load_state_complete", {"path": str(path)})

    async def _load_registry(self, state_data: dict) -> None:
        """Load object registry from saved state."""
        registry_path = state_data["files"]["registry"]
        if Path(registry_path).exists():
            self.tracker.tracker.registry.load_from_json(registry_path)
            self._emit_event("load_state_registry", {"object_count": len(self.tracker.tracker.registry)})

    async def _load_task_state(self, state_data: dict) -> None:
        """Load task state metadata."""
        self.task_description = state_data.get("task_description", "")
        self.detection_count = state_data.get("detection_count", 0)
        self.ready_for_planning = state_data.get("ready_for_planning", False)
        
        last_state_value = state_data.get("last_state")
        if last_state_value:
            self.last_state = TaskState(last_state_value)

    async def _restore_pddl_domain(self, state_data: dict) -> None:
        """Restore PDDL domain from task analysis and observations."""
        task_analysis_data = state_data.get("task_analysis")
        if task_analysis_data and self.task_description:
            # Re-analyze task to restore domain
            self._emit_event("load_state_reanalyze", {"task": self.task_description})
            await self.initialize_from_task()

            # Re-process observations through maintainer
            all_objects = self.tracker.tracker.registry.get_all_objects()
            if all_objects:
                self._emit_event("load_state_reprocess", {"object_count": len(all_objects)})
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
                await self.maintainer.update_from_observations(objects_dict)

    def cleanup(self):
        """Clean up resources."""
        self._emit_event("cleanup_start")
        self.camera.stop()
        self._emit_event("cleanup_complete")


class ContinuousPDDLTextualApp(App):
    """Textual TUI for the continuous PDDL integration demo."""
    
    dark = False  # Use light mode by default

    CSS = """
    Screen {
        layout: vertical;
        overflow-x: hidden;
    }

    #status-panel {
        padding: 1 2;
        height: 3;
        content-align: left middle;
        border-bottom: heavy $accent;
        overflow-x: hidden;
    }

    #log-panel {
        height: 1fr;
        border: tall $accent;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    Log {
        overflow-x: hidden;
        width: 100%;
    }

    #commands-panel {
        padding: 0 2;
        height: 2;
        content-align: left middle;
        background: $panel;
        overflow-x: hidden;
    }

    #command-input {
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "shutdown", "Quit"),
        ("ctrl+q", "shutdown", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.task_description: str = ""
        self.update_interval: float = 2.0
        self.system: Optional[ContinuousPDDLIntegration] = None
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._log_widget: Optional[Log] = None
        self._status_widget: Optional[Static] = None
        self._commands_widget: Optional[Static] = None
        self._command_input: Optional[Input] = None
        self._initializing: bool = True
        self._awaiting_task_input: bool = False
        self._init_failed: bool = False
        
        # Log file will be set up after system initialization
        self.log_file: Optional[Path] = None
        self._log_file_handle = None
    
    def _get_separator(self, char: str = "=") -> str:
        """Get a separator line that spans the current console width."""
        try:
            # Get console width and subtract for borders + padding
            # Log panel has: tall border (2 chars per side) + padding (1 char per side)
            # Total: 6 characters to subtract (3 per side)
            width = self.console.width - 6
            return char * max(width, 40)  # Minimum width of 40
        except (AttributeError, Exception):
            # Fallback to reasonable default if console width unavailable
            return char * 80

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)
        with Vertical():
            yield Static(id="status-panel")
            yield Log(id="log-panel", highlight=False, auto_scroll=True)
            yield Static(id="commands-panel")
            yield Input(placeholder="Enter task description...", id="command-input")
        yield Footer()

    def on_mount(self):
        """Cache widget references and display initial UI."""
        self._log_widget = self.query_one("#log-panel", Log)
        self._status_widget = self.query_one("#status-panel", Static)
        self._commands_widget = self.query_one("#commands-panel", Static)
        self._command_input = self.query_one("#command-input", Input)
        if self._command_input:
            self._command_input.focus()
        
        self._update_status()
        self._update_commands_panel()
        
        # Delay initialization to allow UI to render first
        if not self.api_key:
            self._write_log("⚠ GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
            self._write_log("Please set it as an environment variable.")
            self._write_log("")
            self._write_log("After setting the key, type 'restart' to retry.")
            self._initializing = False
            self._awaiting_task_input = False
            self._init_failed = True
            if self._command_input:
                self._command_input.disabled = False
                self._command_input.placeholder = "Type 'restart' to retry..."
                self._command_input.focus()
            self._update_status()
            self._update_commands_panel()
        else:
            # Start initialization after UI is rendered
            asyncio.create_task(self._delayed_initialization())
    
    def _setup_log_file(self, output_dir: str):
        """Set up log file in the same directory as PDDL outputs."""
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "session.log"
        
        try:
            self._log_file_handle = open(self.log_file, 'w', encoding='utf-8')
            self._write_log(f"✓ Log file: {self.log_file}")
        except Exception as e:
            self._write_log(f"⚠ Could not create log file: {e}")

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user command submissions."""
        command = event.value.strip()
        # Clear input immediately for responsive feedback
        event.input.value = ""
        
        # Handle initialization failure state - only allow restart
        if self._init_failed:
            if command.lower() in ["restart", "quit", "exit"]:
                await self.handle_command(command)
            else:
                self._write_log(f"> {command}")
                self._write_log("⚠ System initialization failed. Use 'restart' to retry or 'quit' to exit.")
            return
        
        # Handle initial task input
        if self._awaiting_task_input:
            if command:
                # Immediate feedback
                self._write_log(f"> Task: {command}")
                self.task_description = command
                self._write_log("✓ Task accepted. Analyzing task...")
                self._awaiting_task_input = False
                if self._command_input:
                    self._command_input.placeholder = "Type command (help for list)..."
                self._update_status()
                self._update_commands_panel()
                # Analyze task and start tracking
                await self._start_tracking_with_task()
            else:
                self._write_log("⚠ Please enter a task description.")
        else:
            await self.handle_command(command)

    async def handle_command(self, command: str):
        """Process commands entered by the user."""
        if not command:
            return

        # Echo command immediately
        self._write_log(f"> {command}")
        parts = command.split(maxsplit=1)
        action = parts[0].strip().lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if action == "interval":
            if not argument:
                self._write_log(f"Current update interval: {self.update_interval:.2f}s")
            else:
                try:
                    self._write_log("⚙ Updating interval...")
                    self.update_interval = float(argument)
                    self._write_log(f"✓ Update interval set to {self.update_interval:.2f}s")
                    self._update_status()
                except ValueError:
                    self._write_log("⚠ Invalid interval. Provide a numeric value, e.g., 'interval 2.5'")
        elif action == "status":
            self._write_log("⚙ Fetching status...")
            if self.system:
                status = await self.system.get_status()
                self._display_status(status)
            else:
                self._write_log("⚠ System not initialized yet.")
        elif action == "save":
            self._write_log("⚙ Saving state...")
            if self.system:
                try:
                    path = await self.system.save_state()
                    self._write_log(f"✓ State saved to {path}")
                except Exception as exc:
                    self._write_log(f"⚠ Failed to save state: {exc}")
            else:
                self._write_log("⚠ System not initialized yet.")
        elif action == "load":
            if not self.system:
                self._write_log("⚠ System not initialized yet.")
            else:
                # Parse optional path argument
                load_path = None
                if argument:
                    load_path = Path(argument) / "state.json" if not argument.endswith('.json') else Path(argument)
                    self._write_log(f"⚙ Loading state from {load_path}...")
                else:
                    self._write_log("⚙ Loading state...")
                
                try:
                    await self.system.load_state(load_path)
                    self._update_status()
                    self._write_log("✓ State loaded successfully")
                except FileNotFoundError as exc:
                    self._write_log(f"⚠ {exc}")
                except Exception as exc:
                    self._write_log(f"⚠ Failed to load state: {exc}")
        elif action == "stop":
            self._write_log("⚙ Stopping tracking...")
            await self._stop_tracking()
        elif action == "continue":
            if not self.system:
                self._write_log("⚠ System not initialized.")
            elif self.system._running:
                self._write_log("⚠ Tracking is already running.")
            else:
                self._write_log("⚙ Resuming tracking...")
                self.system.start_tracking()
                self._update_status()
                self._update_commands_panel()
                self._write_log("✓ Tracking resumed")
        elif action == "restart":
            self._write_log("⚙ Restarting system...")
            
            # Stop tracking if running
            if self.system and self.system._running:
                await self.system.stop_tracking()
            
            # Clean up existing system
            if self.system:
                self.system.cleanup()
                self.system = None
            
            # Clear init failed flag and reload API key
            self._init_failed = False
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            
            # Check for API key before initializing
            if not self.api_key:
                self._write_log("⚠ GEMINI_API_KEY or GOOGLE_API_KEY is still not set.")
                self._write_log("Please set it as an environment variable and try again.")
                self._init_failed = True
                if self._command_input:
                    self._command_input.placeholder = "Type 'restart' to retry..."
            else:
                # Reinitialize
                self._write_log("⚙ Reinitializing components...")
                await self._initialize_system()
                
                # If we had a task and init succeeded, start tracking automatically
                if self.task_description and self.system and not self._awaiting_task_input and not self._init_failed:
                    self._write_log("⚙ Restarting tracking with previous task...")
                    await self._start_tracking_with_task()
            
            self._update_status()
            self._update_commands_panel()
        elif action == "generate":
            # Generate PDDL files (inline logic)
            if not self.system:
                self._write_log("⚠ System not initialized.")
            else:
                force = argument.lower() == "force"
                self._write_log("⚙ Checking readiness...")
                decision = await self.system.monitor.determine_state()
                ready = (
                    decision.state == TaskState.PLAN_AND_EXECUTE
                    or self.system.ready_for_planning
                    or force
                )
                
                if not ready:
                    self._write_log("⚠ Not ready for planning yet.")
                    self._write_log(f"  Current state: {decision.state.value}")
                    self._write_log("  Use 'generate force' to export anyway.")
                    if decision.blockers:
                        self._write_log("  Blockers:")
                        for blocker in decision.blockers:
                            self._write_log(f"    ✗ {blocker}")
                else:
                    await self.system.generate_pddl_files()
                    self._write_log(f"✓ Output: {self.system.output_dir}")
        elif action in {"quit", "exit"}:
            self._write_log("⚙ Shutting down...")
            await self.action_shutdown()
        elif action == "help":
            self._log_commands()
        else:
            self._write_log(f"⚠ Unknown command: {action}")
            self._write_log("Type 'help' for available commands.")

    async def _delayed_initialization(self):
        """Wait for UI to render, then start initialization."""
        # Wait for UI to fully render
        await asyncio.sleep(0.1)
        
        self._write_log("✓ API key detected.")
        self._write_log("")
        self._write_log("⚙ Initializing system components...")
        
        # Disable input during initialization
        if self._command_input:
            self._command_input.disabled = True
        
        # Start initialization
        await self._initialize_system()
    
    async def _initialize_system(self):
        """Initialize system components (camera, PDDL, tracker) before task is known."""
        if not self.api_key:
            self._write_log("⚠ Missing API key.")
            self._initializing = False
            return
            
        try:
            # Initialize with empty task for now
            self.system = ContinuousPDDLIntegration(
                api_key=self.api_key,
                task_description="",  # Will be set later
                update_interval=self.update_interval,
                min_observations=2,
                on_event=self._on_system_event
            )
            
            self._write_log(f"✓ Output directory: {self.system.output_dir}")
            
            # Set up log file in the same directory as PDDL outputs
            self._setup_log_file(self.system.output_dir)
            
            self._write_log("")
            self._write_log("✓ System initialized")
            
            # Now ready for task input
            self._initializing = False
            self._awaiting_task_input = True
            if self._command_input:
                self._command_input.disabled = False
                self._command_input.placeholder = "Enter task description..."
                self._command_input.focus()
            
        except Exception as exc:
            self._write_log(f"⚠ Initialization failed: {exc}")
            self._write_log("Check: RealSense connection, device availability (rs-enumerate-devices)")
            self._write_log("")
            self._write_log("Type 'restart' to retry initialization.")
            self.system = None
            self._initializing = False
            self._awaiting_task_input = False
            self._init_failed = True
            if self._command_input:
                self._command_input.disabled = False
                self._command_input.placeholder = "Type 'restart' to retry..."
                self._command_input.focus()
        
        self._update_status()
        self._update_commands_panel()
    
    async def _start_tracking_with_task(self):
        """Analyze task and start tracking (called after task is entered)."""
        if not self.system:
            self._write_log("⚠ System not initialized.")
            return
        
        if self.system._running:
            self._write_log("⚠ Tracking already running.")
            return

        try:
            # Update task description
            self.system.task_description = self.task_description
            
            self._write_log("⚙ Analyzing task and configuring PDDL domain...")
            await self.system.initialize_from_task()
            
            self._write_log("⚙ Starting continuous tracking...")
            self.system.start_tracking()
            
            self._update_status()
            self._update_commands_panel()
            
            self._write_log("✓ System active. See available commands below.")
        except Exception as exc:
            self._write_log(f"⚠ Error during startup: {exc}")
            await self._safe_shutdown()

    async def _stop_tracking(self):
        """Stop the continuous tracker if running."""
        if not self.system or not self.system._running:
            self._write_log("⚠ Tracking is not running.")
            return

        await self.system.stop_tracking()
        
        # Always generate PDDL files when stopped
        try:
            await self.system.generate_pddl_files()
        except Exception as exc:
            self._write_log(f"⚠ PDDL generation failed: {exc}")
        
        self._update_status()
        self._update_commands_panel()
        self._write_log("")
        self._write_log("✓ Stopped. Use 'continue' to resume or 'quit' to exit.")

    def _display_status(self, status: dict):
        """Display system status."""
        sep = self._get_separator()
        tracking = status['tracking']
        pddl = status['pddl']
        task = status['task_state']
        actions = status['actions']
        goals = status['goals']
        
        self._write_log("")
        self._write_log(sep)
        self._write_log("SYSTEM STATUS")
        self._write_log(sep)
        
        # Task description
        self._write_log("Task:")
        self._write_log(f"  \"{status['task_description']}\"")
        
        self._write_log("")
        self._write_log(f"Tracking: {'RUNNING' if tracking['is_running'] else 'STOPPED'}")
        self._write_log(f"Detection cycles: {tracking['detection_cycles']}")
        self._write_log(f"Frames with detection: {tracking['frames_with_detection']}")
        self._write_log(f"Frames skipped: {tracking['frames_skipped']}")
        self._write_log(f"Cache hit rate: {tracking['cache_hit_rate']:.1%}")
        self._write_log(f"Total detections: {tracking['total_detections']}")
        self._write_log(f"Avg detection time: {tracking['avg_detection_time']:.2f}s")
        self._write_log(f"Last detection time: {tracking['last_detection_time']:.2f}s")
        
        self._write_log("")
        self._write_log("PDDL Domain:")
        self._write_log(f"  • Object instances: {pddl['object_instances']}")
        self._write_log(f"  • Object types observed: {pddl['object_types_observed']}")
        self._write_log(f"  • Domain version: {pddl['domain_version']}")
        
        self._write_log("")
        self._write_log("Task State:")
        self._write_log(f"  • Current: {task['current']}")
        self._write_log(f"  • Confidence: {task['confidence']:.1%}")
        self._write_log(f"  • Ready for planning: {'YES' if task['ready_for_planning'] else 'NO'}")
        
        # Show generated actions
        self._write_log("")
        self._write_log("Actions:")
        predefined = actions['predefined']
        llm_generated = actions['llm_generated']
        
        self._write_log(f"  • Predefined actions: {len(predefined)}")
        if predefined:
            for action_name in predefined:
                self._write_log(f"    - {action_name}")
        
        self._write_log(f"  • LLM-generated actions: {len(llm_generated)}")
        if llm_generated:
            for action_name in llm_generated:
                self._write_log(f"    - {action_name}")
        
        # Show goals
        self._write_log("")
        self._write_log("Goals:")
        if goals:
            self._write_log(f"  • Total goal literals: {len(goals)}")
            for goal in goals:
                self._write_log(f"    - {goal}")
        else:
            self._write_log("  • No goals set yet")
        
        # Show all objects with predicates
        if self.system:
            all_objects = self.system.tracker.get_all_objects()
            if all_objects:
                self._write_log("")
                self._write_log(f"Objects ({len(all_objects)}):")
                for obj in all_objects:
                    self._write_log(f"  • {obj.object_id} ({obj.object_type})")
                    # Extract predicates from pddl_state
                    if obj.pddl_state:
                        predicates = []
                        for key, value in obj.pddl_state.items():
                            if isinstance(value, bool) and value:
                                predicates.append(key)
                            elif value and key not in ['object_id', 'object_type']:
                                predicates.append(f"{key}={value}")
                        if predicates:
                            self._write_log(f"    Predicates: {', '.join(predicates)}")
                    if obj.affordances:
                        self._write_log(f"    Affordances: {', '.join(obj.affordances)}")
        
        self._write_log("")
        self._write_log(sep)

    async def _safe_shutdown(self):
        """Attempt to stop tracking and cleanup safely."""
        try:
            await self._cleanup_system()
        except Exception as exc:
            self._write_log(f"⚠ Error during cleanup: {exc}")

    async def _cleanup_system(self):
        """Stop tracking and cleanup hardware resources."""
        if self.system:
            if self.system._running:
                await self.system.stop_tracking()
                
                # Always generate PDDL files before cleanup
                try:
                    await self.system.generate_pddl_files()
                except Exception as exc:
                    self._write_log(f"⚠ PDDL generation failed: {exc}")
            
            # Save final state if we have data
            if self.system.task_description and self.system.detection_count > 0:
                self._write_log("⚙ Saving state...")
                try:
                    await self.system.save_state()
                    self._write_log("✓ State saved")
                except Exception as exc:
                    self._write_log(f"⚠ State save failed: {exc}")

            try:
                self.system.cleanup()
            finally:
                self.system = None
        
        # Close log file
        if self._log_file_handle and not self._log_file_handle.closed:
            try:
                self._write_log("")
                if self.system:
                    self._write_log(f"✓ All outputs saved to: {self.system.output_dir}")
                    self._write_log("  • PDDL files (domain & problem)")
                    self._write_log(f"  • Session log: {self.log_file.name if self.log_file else 'session.log'}")
                else:
                    self._write_log(f"✓ Log saved to: {self.log_file}")
                self._log_file_handle.close()
            except Exception:
                pass
        
        self._update_status()

    async def action_shutdown(self):
        """Exit the application after cleanup."""
        await self._cleanup_system()
        self.exit()

    def _log_commands(self):
        """Print the available commands to the log."""
        self._write_log("\nAvailable commands:")
        self._write_log("  • help                 – Show this list")
        self._write_log("  • status               – Show current system status")
        self._write_log("  • stop                 – Stop tracking loop")
        self._write_log("  • continue             – Resume stopped tracking")
        self._write_log("  • restart              – Restart entire system")
        self._write_log("  • save                 – Save current state to disk")
        self._write_log("  • load [path]          – Load saved state (from path or current output_dir)")
        self._write_log("  • generate [force]     – Export PDDL files (force to override readiness)")
        self._write_log("  • interval <seconds>   – Update detection interval")
        self._write_log("  • quit                 – Cleanup and exit")

    def _write_log(self, message: str = ""):
        """Write text to the log widget and file."""
        text = message or ""
        
        # Write to file immediately
        if self._log_file_handle and not self._log_file_handle.closed:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self._log_file_handle.write(f"[{timestamp}] {text}\n")
                self._log_file_handle.flush()
            except Exception:
                pass  # Fail gracefully
        
        # Write to widget
        if self._log_widget is None:
            return
        
        # Write directly if in the app thread, otherwise use call_from_thread
        try:
            self._log_widget.write_line(text if text else "")
        except Exception:
            # If direct write fails, we're likely in a different thread
            def _writer():
                if self._log_widget:
                    self._log_widget.write_line(text if text else "")
            
            try:
                self.call_from_thread(_writer)
            except RuntimeError:
                # If call_from_thread fails, just skip this log
                pass

    def _update_status(self):
        """Refresh the status panel."""
        if self._status_widget is None:
            return

        if self._initializing:
            status_text = "Initializing system components..."
        elif self._init_failed:
            status_text = "Initialization failed - type 'restart' to retry"
        elif self._awaiting_task_input:
            status_text = "Waiting for task description..."
        else:
            tracking_state = "RUNNING" if self.system and self.system._running else "STOPPED"
            detections = self.system.detection_count if self.system else 0
            ready = "YES" if self.system and self.system.ready_for_planning else "NO"

            status_text = (
                f"Task: {self.task_description} | "
                f"Interval: {self.update_interval:.2f}s | "
                f"Tracking: {tracking_state} | "
                f"Detections: {detections} | "
                f"Ready: {ready}"
            )
        self._status_widget.update(status_text)

    def _update_commands_panel(self):
        """Update the commands panel with available commands."""
        if self._commands_widget is None:
            return

        if self._initializing:
            commands_text = "Initializing hardware and components..."
        elif self._init_failed:
            commands_text = "Commands: restart | quit"
        elif self._awaiting_task_input:
            commands_text = "Input: task description"
        else:
            tracking_state = "RUNNING" if self.system and self.system._running else "STOPPED"
            if tracking_state == "STOPPED" and self.system:
                commands_text = "Commands: help | status | continue | restart | save | load [path] | generate [force] | interval <sec> | quit"
            else:
                commands_text = "Commands: help | status | stop | restart | save | load [path] | generate [force] | interval <sec> | quit"
        
        self._commands_widget.update(commands_text)
    
    def _on_system_event(self, event_type: str, data: dict):
        """Handle events from ContinuousPDDLIntegration."""
        sep = self._get_separator()
        
        if event_type == "camera_init_start":
            self._write_log("⚙ Initializing RealSense camera...")
        elif event_type == "camera_init_complete":
            self._write_log("✓ Camera initialized")
        elif event_type == "camera_error":
            self._write_log(f"⚠ Camera error: {data.get('error', 'Unknown')}")
        elif event_type == "pddl_init_start":
            self._write_log("⚙ Initializing PDDL system...")
        elif event_type == "pddl_init_complete":
            self._write_log("✓ PDDL system initialized")
        elif event_type == "tracker_init_start":
            self._write_log("⚙ Initializing continuous object tracker...")
        elif event_type == "tracker_init_complete":
            self._write_log("✓ Continuous tracker initialized")
        elif event_type == "task_analysis_start":
            self._write_log("")
            self._write_log(sep)
            self._write_log("TASK ANALYSIS")
            self._write_log(sep)
            self._write_log(f"Task: \"{data['task']}\"")
            self._write_log("")
            self._write_log("⚙ Capturing environment and analyzing task...")
        elif event_type == "task_analysis_complete":
            self._write_log("✓ Task analyzed!")
            self._write_log(f"  • Goal objects: {', '.join(data['goal_objects'])}")
            self._write_log(f"  • Estimated steps: {data['estimated_steps']}")
            self._write_log(f"  • Complexity: {data['complexity']}")
            self._write_log(f"  • Predicates: {len(data['predicates'])}")
            if data['predicates']:
                self._write_log("  Key predicates:")
                for pred in data['predicates'][:8]:
                    self._write_log(f"    • {pred}")
                if len(data['predicates']) > 8:
                    self._write_log(f"    ... and {len(data['predicates']) - 8} more")
            self._write_log("")
            self._write_log(sep)
        elif event_type == "tracking_start":
            self._write_log("")
            self._write_log(sep)
            self._write_log("STARTING CONTINUOUS TRACKING")
            self._write_log(sep)
            self._write_log(f"Update interval: {data['update_interval']}s")
        elif event_type == "tracking_started":
            self._write_log("✓ Tracking loop active")
            self._write_log("")
            self._write_log(sep)
        elif event_type == "detection_update":
            self._handle_detection_update(data)
        elif event_type == "tracking_stop":
            self._write_log("⚙ Stopping tracking...")
        elif event_type == "tracking_stopped":
            self._write_log("✓ Tracking stopped")
        elif event_type == "pddl_generate_start":
            # Silent - will show result on completion
            pass
        elif event_type == "pddl_generate_complete":
            # One-line summary
            problem_sum = data['problem_summary']
            self._write_log(f"✓ PDDL saved: {problem_sum['object_instances']} objects, {problem_sum['initial_literals']} literals, {problem_sum['goal_literals']} goals")
        elif event_type == "cleanup_start":
            self._write_log("⚙ Cleaning up resources...")
        elif event_type == "cleanup_complete":
            self._write_log("✓ Cleanup complete")
        elif event_type == "pddl_auto_saved":
            # Silent auto-save - don't spam the log
            pass
        elif event_type == "pddl_save_error":
            self._write_log(f"⚠ PDDL auto-save error: {data.get('error', 'Unknown')}")
        elif event_type == "state_auto_saved":
            # Silent auto-save - don't spam the log
            pass
        elif event_type == "state_save_error":
            self._write_log(f"⚠ State auto-save error: {data.get('error', 'Unknown')}")
        elif event_type == "save_state_start":
            # Silent - handled by command echo
            pass
        elif event_type == "save_state_complete":
            # Silent - handled by command completion
            pass
        elif event_type == "load_state_start":
            # Silent - handled by command echo
            pass
        elif event_type == "load_state_registry":
            self._write_log(f"  • Loaded {data['object_count']} objects from registry")
        elif event_type == "load_state_reanalyze":
            self._write_log(f"  • Re-analyzing task: '{data['task']}'")
        elif event_type == "load_state_reprocess":
            self._write_log(f"  • Re-processing {data['object_count']} objects...")
        elif event_type == "load_state_complete":
            # Silent - handled by command completion
            pass
    
    def _handle_detection_update(self, data: dict):
        """Handle detection update event."""
        sep = self._get_separator()
        
        self._write_log("")
        self._write_log(sep)
        self._write_log(f"DETECTION UPDATE #{data['detection_count']}")
        self._write_log(sep)
        self._write_log(f"Objects detected: {data['object_count']}")
        
        if not data['has_objects']:
            self._write_log("  No objects detected yet")
            return
        
        # Update stats
        stats = data['update_stats']
        self._write_log("")
        self._write_log("PDDL Update:")
        self._write_log(f"  • Objects added: {stats['objects_added']}")
        self._write_log(f"  • Total observations: {stats['total_observations']}")
        self._write_log(f"  • Object types: {stats['total_object_types']}")
        self._write_log(f"  • Goal objects found: {', '.join(stats['goal_objects_found']) or 'none'}")
        self._write_log(f"  • Goal objects missing: {', '.join(stats['goal_objects_missing']) or 'none'}")
        
        # Task state
        decision = data['decision']
        self._write_log("")
        self._write_log("Task State:")
        if data['state_changed']:
            self._write_log(f"  ⚡ STATE CHANGED: {data['old_state']} → {decision['state']}")
        self._write_log(f"  • Current: {decision['state']}")
        self._write_log(f"  • Confidence: {decision['confidence']:.1%}")
        self._write_log(f"  • Reasoning: {decision['reasoning']}")
        
        if decision['blockers']:
            self._write_log("  Blockers:")
            for blocker in decision['blockers']:
                self._write_log(f"    ✗ {blocker}")
        
        if decision['recommendations']:
            self._write_log("  Recommendations:")
            for rec in decision['recommendations']:
                self._write_log(f"    → {rec}")
        
        # Ready for planning
        if data['newly_ready']:
            self._write_log("")
            self._write_log(sep)
            self._write_log("🎯 READY FOR PLANNING!")
            self._write_log(sep)
        
        # New objects detail (show predicates/relationships)
        new_objects = data.get('new_objects', [])
        if new_objects:
            self._write_log("")
            self._write_log(f"New Objects ({len(new_objects)}):")
            for obj in new_objects:
                self._write_log(f"  • {obj['object_id']} ({obj['object_type']})")
                if obj['predicates']:
                    self._write_log(f"    Predicates: {', '.join(obj['predicates'])}")
                if obj['affordances']:
                    self._write_log(f"    Affordances: {', '.join(obj['affordances'])}")
        
        self._write_log("")
        self._write_log(sep)


if __name__ == "__main__":
    ContinuousPDDLTextualApp().run()
