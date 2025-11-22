"""
Task Orchestrator Demo

Interactive TUI for the TaskOrchestrator system.

Demonstrates:
1. Task analysis and PDDL domain generation
2. Continuous object detection with real-time updates
3. Dynamic PDDL domain updates from observations
4. Task state monitoring with adaptive decision-making
5. State persistence (save/load)
6. User-controlled execution flow

Uses TaskOrchestrator as the production-focused backend with a Textual TUI
providing interactive control and event logging.
"""

import os
import sys
import asyncio
import textwrap
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.planning import TaskOrchestrator, OrchestratorState, TaskState
# Import config from config directory
config_path = Path(__file__).parent.parent / "config"
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
from orchestrator_config import OrchestratorConfig

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, RichLog, Input, Static

# Load environment
load_dotenv()


class OrchestratorDemoApp(App):
    """Interactive Textual TUI for the Task Orchestrator."""
    
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

    RichLog {
        width: 100%;
        overflow-x: hidden;
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
        self.orchestrator: Optional[TaskOrchestrator] = None
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._log_widget: Optional[RichLog] = None
        self._status_widget: Optional[Static] = None
        self._commands_widget: Optional[Static] = None
        self._command_input: Optional[Input] = None
        self._initializing: bool = True
        self._awaiting_task_input: bool = False
        self._init_failed: bool = False
        
        # Log file will be set up after system initialization
        self.log_file: Optional[Path] = None
        self._log_file_handle = None
        
        # Track new objects for demo reporting
        self._known_object_ids: set = set()
    
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
            yield RichLog(id="log-panel", markup=True, auto_scroll=True)
            yield Static(id="commands-panel")
            yield Input(placeholder="Enter task description...", id="command-input")
        yield Footer()

    def on_mount(self):
        """Cache widget references and display initial UI."""
        self._log_widget = self.query_one("#log-panel", RichLog)
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
            if self.orchestrator:
                status = await self.orchestrator.get_status()
                self._display_status(status)
            else:
                self._write_log("⚠ System not initialized yet.")
        elif action == "save":
            self._write_log("⚙ Saving state...")
            if self.orchestrator:
                try:
                    path = await self.orchestrator.save_state()
                    self._write_log(f"✓ State saved to {path}")
                except Exception as exc:
                    self._write_log(f"⚠ Failed to save state: {exc}")
            else:
                self._write_log("⚠ System not initialized yet.")
        elif action == "load":
            if not self.orchestrator:
                self._write_log("⚠ System not initialized yet.")
            else:
                # Parse optional path argument
                load_path = None
                if argument:
                    # Determine if argument is a directory or file
                    arg_path = Path(argument)
                    if arg_path.is_dir() or (not arg_path.exists() and not argument.endswith('.json')):
                        load_path = arg_path / "state.json"
                    else:
                        load_path = arg_path
                    self._write_log(f"⚙ Loading state from {load_path}...")
                    self._write_log(f"  • Outputs will continue to: {self.orchestrator.config.state_dir}")
                else:
                    self._write_log("⚙ Loading state from current output directory...")
                
                try:
                    await self.orchestrator.load_state(load_path)
                    # Update task description from loaded state
                    if self.orchestrator.current_task:
                        self.task_description = self.orchestrator.current_task
                    self._update_status()
                    self._write_log("✓ State loaded successfully")
                except FileNotFoundError as exc:
                    self._write_log(f"⚠ {exc}")
                except Exception as exc:
                    self._write_log(f"⚠ Failed to load state: {exc}")
        elif action == "stop":
            self._write_log("⚙ Stopping tracking...")
            await self._stop_tracking()
        elif action == "pause":
            if not self.orchestrator:
                self._write_log("⚠ System not initialized.")
            elif not self.orchestrator._detection_running:
                self._write_log("⚠ Tracking is not running.")
            else:
                self._write_log("⚙ Pausing tracking...")
                await self.orchestrator.pause_detection()
                self._update_status()
                self._update_commands_panel()
                self._write_log("✓ Tracking paused")
        elif action == "continue":
            if not self.orchestrator:
                self._write_log("⚠ System not initialized.")
            elif self.orchestrator._detection_running:
                self._write_log("⚠ Tracking is already running.")
            else:
                self._write_log("⚙ Resuming tracking...")
                await self.orchestrator.resume_detection()
                self._update_status()
                self._update_commands_panel()
                self._write_log("✓ Tracking resumed")
        elif action == "restart":
            self._write_log("⚙ Restarting system...")
            
            # Shutdown existing orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()
                self.orchestrator = None
            
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
                if self.task_description and self.orchestrator and not self._awaiting_task_input and not self._init_failed:
                    self._write_log("⚙ Restarting tracking with previous task...")
                    await self._start_tracking_with_task()
            
            self._update_status()
            self._update_commands_panel()
        elif action == "generate":
            # Generate PDDL files
            if not self.orchestrator:
                self._write_log("⚠ System not initialized.")
            else:
                force = argument.lower() == "force"
                self._write_log("⚙ Checking readiness...")
                decision = await self.orchestrator.get_task_decision()
                ready = (
                    self.orchestrator.is_ready_for_planning()
                    or force
                )
                
                if not ready and decision:
                    self._write_log("⚠ Not ready for planning yet.")
                    self._write_log(f"  Current state: {decision.state.value}")
                    self._write_log("  Use 'generate force' to export anyway.")
                    if decision.blockers:
                        self._write_log("  Blockers:")
                        for blocker in decision.blockers:
                            self._write_log(f"    ✗ {blocker}")
                else:
                    await self.orchestrator.generate_pddl_files()
                    self._write_log(f"✓ Output: {self.orchestrator.config.state_dir}")
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
            # Create timestamp-based output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"outputs/pddl/continuous_{timestamp}")
            
            # Configure orchestrator with callbacks for demo
            config = OrchestratorConfig(
                api_key=self.api_key,
                update_interval=self.update_interval,
                min_observations=2,
                state_dir=output_dir,
                auto_save=True,
                auto_save_on_detection=True,
                auto_save_on_state_change=True,
                on_state_change=self._on_state_change,
                on_detection_update=self._on_detection_update,
                on_task_state_change=self._on_task_state_change,
                on_save_state=self._on_save_state
            )
            
            # Initialize orchestrator
            self.orchestrator = TaskOrchestrator(config)
            await self.orchestrator.initialize()
            
            self._write_log(f"✓ Output directory: {output_dir}")
            
            # Set up log file in the same directory as PDDL outputs
            self._setup_log_file(str(output_dir))
            
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
            self.orchestrator = None
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
        if not self.orchestrator:
            self._write_log("⚠ System not initialized.")
            return
        
        if self.orchestrator._detection_running:
            self._write_log("⚠ Tracking already running.")
            return

        try:
            self._write_log("⚙ Analyzing task and configuring PDDL domain...")
            await self.orchestrator.process_task_request(self.task_description)
            
            self._write_log("⚙ Starting continuous tracking...")
            await self.orchestrator.start_detection()
            
            self._update_status()
            self._update_commands_panel()
            
            self._write_log("✓ System active. See available commands below.")
        except Exception as exc:
            self._write_log(f"⚠ Error during startup: {exc}")
            await self._safe_shutdown()

    async def _stop_tracking(self):
        """Stop the continuous tracker if running."""
        if not self.orchestrator or not self.orchestrator._detection_running:
            self._write_log("⚠ Tracking is not running.")
            return

        await self.orchestrator.stop_detection()
        
        # Always generate PDDL files when stopped
        try:
            await self.orchestrator.generate_pddl_files()
        except Exception as exc:
            self._write_log(f"⚠ PDDL generation failed: {exc}")
        
        self._update_status()
        self._update_commands_panel()
        self._write_log("")
        self._write_log("✓ Stopped. Use 'continue' to resume or 'quit' to exit.")

    def _display_status(self, status: dict):
        """Display system status."""
        sep = self._get_separator()
        
        self._write_log("")
        self._write_log(sep)
        self._write_log("SYSTEM STATUS")
        self._write_log(sep)
        
        # Orchestrator state
        self._write_log("Orchestrator:")
        self._write_log(f"  • State: {status['orchestrator_state']}")
        self._write_log(f"  • Task: \"{status['current_task']}\"")
        self._write_log(f"  • Detection running: {'YES' if status['detection_running'] else 'NO'}")
        self._write_log(f"  • Detection count: {status['detection_count']}")
        self._write_log(f"  • Ready for planning: {'YES' if status['ready_for_planning'] else 'NO'}")
        
        # Tracker stats
        if 'tracker' in status:
            tracker = status['tracker']
            self._write_log("")
            self._write_log("Tracker:")
            self._write_log(f"  • Total frames: {tracker['total_frames']}")
            self._write_log(f"  • Total detections: {tracker['total_detections']}")
            self._write_log(f"  • Frames skipped: {tracker['skipped_frames']}")
            self._write_log(f"  • Cache hit rate: {tracker['cache_hit_rate']:.1%}")
            self._write_log(f"  • Avg detection time: {tracker['avg_detection_time']:.2f}s")
        
        # Registry stats
        if 'registry' in status:
            registry = status['registry']
            self._write_log("")
            self._write_log("Registry:")
            self._write_log(f"  • Objects: {registry['num_objects']}")
            if registry['object_types']:
                self._write_log(f"  • Types: {', '.join(registry['object_types'])}")
        
        # Domain stats
        if 'domain' in status:
            domain = status['domain']
            self._write_log("")
            self._write_log("PDDL Domain:")
            self._write_log(f"  • Object instances: {domain['object_instances']}")
            self._write_log(f"  • Object types: {domain['object_types_observed']}")
            self._write_log(f"  • Domain version: {domain['domain_version']}")
        
        # Task state
        if 'task_state' in status:
            task = status['task_state']
            self._write_log("")
            self._write_log("Task State:")
            self._write_log(f"  • Current: {task['state']}")
            self._write_log(f"  • Confidence: {task['confidence']:.1%}")
            self._write_log(f"  • Reasoning: {task['reasoning']}")
            if task['blockers']:
                self._write_log("  Blockers:")
                for blocker in task['blockers']:
                    self._write_log(f"    ✗ {blocker}")
            if task['recommendations']:
                self._write_log("  Recommendations:")
                for rec in task['recommendations']:
                    self._write_log(f"    → {rec}")
        
        # Show all objects with predicates
        if self.orchestrator:
            all_objects = self.orchestrator.get_detected_objects()
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
        if self.orchestrator:
            # Orchestrator handles stopping detection and final save in shutdown()
            await self.orchestrator.shutdown()
            
            # Close log file
            if self._log_file_handle and not self._log_file_handle.closed:
                try:
                    self._write_log("")
                    self._write_log(f"✓ All outputs saved to: {self.orchestrator.config.state_dir}")
                    self._write_log("  • PDDL files (domain & problem)")
                    self._write_log("  • Object registry")
                    self._write_log(f"  • Session log: {self.log_file.name if self.log_file else 'session.log'}")
                    self._log_file_handle.close()
                except Exception:
                    pass
            
            self.orchestrator = None
        
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
        self._write_log("  • pause                – Pause tracking (can resume)")
        self._write_log("  • continue             – Resume paused tracking")
        self._write_log("  • restart              – Restart entire system")
        self._write_log("  • save                 – Save current state to disk")
        self._write_log("  • load [dir]           – Load state from dir (keeps logging to current dir)")
        self._write_log("  • generate [force]     – Export PDDL files (force to override readiness)")
        self._write_log("  • interval <seconds>   – Update detection interval")
        self._write_log("  • quit                 – Cleanup and exit")

    def _write_log(self, message: str = ""):
        """Write text to the log widget and file."""
        text = message or ""
        
        # Write to file immediately (unwrapped)
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
        
        # Wrap text to console width (accounting for log panel borders and padding)
        # Log panel has: tall border (2 chars per side) + padding (1 char per side) = 6 chars total
        try:
            console_width = self.console.width - 6
            if console_width < 40:
                console_width = 40  # Minimum width
            
            # Wrap long lines
            if text and len(text) > console_width:
                wrapped_lines = textwrap.wrap(text, width=console_width, 
                                             break_long_words=True, 
                                             break_on_hyphens=False)
                for line in wrapped_lines:
                    self._write_log_line(line)
                return
        except (AttributeError, Exception):
            # If console width unavailable, proceed without wrapping
            pass
        
        # Write normally if short or wrapping failed
        self._write_log_line(text)
    
    def _write_log_line(self, text: str):
        """Internal method to write a single line to the log widget."""
        # Write directly if in the app thread, otherwise use call_from_thread
        try:
            self._log_widget.write(text if text else "")
        except Exception:
            # If direct write fails, we're likely in a different thread
            def _writer():
                if self._log_widget:
                    self._log_widget.write(text if text else "")
            
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
            tracking_state = "RUNNING" if self.orchestrator and self.orchestrator._detection_running else "STOPPED"
            detections = self.orchestrator.detection_count if self.orchestrator else 0
            ready = "YES" if self.orchestrator and self.orchestrator.is_ready_for_planning() else "NO"

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
            tracking_state = "RUNNING" if self.orchestrator and self.orchestrator._detection_running else "STOPPED"
            if tracking_state == "STOPPED" and self.orchestrator:
                commands_text = "Commands: help | status | continue | restart | save | load [dir] | generate [force] | interval <sec> | quit"
            else:
                commands_text = "Commands: help | status | stop | pause | restart | save | load [dir] | generate [force] | interval <sec> | quit"
        
        self._commands_widget.update(commands_text)
    
    # ========================================================================
    # Orchestrator Callback Methods
    # ========================================================================
    
    def _on_state_change(self, old_state: OrchestratorState, new_state: OrchestratorState):
        """Called when orchestrator state changes."""
        self._write_log(f"⚡ State changed: {old_state.value} → {new_state.value}")
        self._update_status()
    
    def _on_detection_update(self, object_count: int):
        """Called after each detection cycle."""
        if not self.orchestrator:
            return
        
        # Get new objects for detailed reporting
        new_objects = self.orchestrator.get_new_objects()
        
        # Build compact one-line summary with bold object names
        parts = [f"Detection Update #{self.orchestrator.detection_count}: {object_count} objects"]
        if new_objects:
            new_obj_types = [f"[bold]{obj.object_id}[/bold] ({obj.object_type})" for obj in new_objects]
            parts.append(f"New: {', '.join(new_obj_types)}")
        
        self._write_log(" | ".join(parts))
        self._update_status()
    
    def _on_task_state_change(self, decision):
        """Called when task state decision changes."""
        sep = self._get_separator()
        
        self._write_log("")
        self._write_log("Task State Update:")
        self._write_log(f"  • Current: {decision.state.value}")
        self._write_log(f"  • Confidence: {decision.confidence:.1%}")
        self._write_log(f"  • Reasoning: {decision.reasoning}")
        
        if decision.blockers:
            self._write_log("  Blockers:")
            for blocker in decision.blockers:
                self._write_log(f"    ✗ {blocker}")
        
        if decision.recommendations:
            self._write_log("  Recommendations:")
            for rec in decision.recommendations:
                self._write_log(f"    → {rec}")
        
        # Check if ready for planning
        if decision.state == TaskState.PLAN_AND_EXECUTE:
            self._write_log("")
            self._write_log(sep)
            self._write_log("🎯 READY FOR PLANNING!")
            self._write_log(sep)
        
        self._update_status()
    
    def _on_save_state(self, path: Path):
        """Called after successful state save (silent - auto-save)."""
        # Silent for auto-saves to avoid log spam
        pass


if __name__ == "__main__":
    OrchestratorDemoApp().run()
