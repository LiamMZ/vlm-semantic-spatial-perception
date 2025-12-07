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

    # Reuse perception data from a previous run (skips perception step)
    python examples/tamp_demo.py --task "stack the blocks" --snapshot outputs/tamp_demo/run_20250103_143022/perception_pool/snapshots/snapshot_001

Output Directory Structure:
    Each script execution creates a unique run directory with timestamp:

    outputs/tamp_demo/
    └── run_YYYYMMDD_HHMMSS/              # Unique run directory
        ├── run_YYYYMMDD_HHMMSS.log        # Complete log file for this run
        ├── run_YYYYMMDD_HHMMSS_timing.log # CSV timing log for performance analysis
        ├── run_metadata.json              # Run-level metadata
        ├── pddl/                          # Shared PDDL domain/problem files
        │   ├── tabletop_domain.pddl
        │   └── tabletop_problem.pddl
        ├── task_001/                      # First task results
        │   ├── task_metadata.json         # Task description, success, timing
        │   ├── decompositions.json        # Skill plans for each action
        │   ├── execution_results.json     # Execution outcomes
        │   └── pddl -> ../pddl            # Symlink to shared PDDL files
        ├── task_002/                      # Second task results
        │   ├── task_metadata.json
        │   ├── decompositions.json
        │   ├── execution_results.json
        │   └── pddl -> ../pddl
        └── ...

    This structure allows:
    - Multiple runs without overwriting previous results
    - Multiple tasks per run, each with isolated results
    - Shared PDDL domain/problem files across tasks in a run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.task_motion_planner import TaskAndMotionPlanner, TAMPConfig, TAMPState, TAMPResult
from src.utils.genai_logging import configure_genai_logging
from src.kinematics.xarm_curobo_interface import CuRoboMotionPlanner

class TAMPDemo:
    """Interactive demo for the complete TAMP system."""

    def __init__(
        self,
        api_key: str,
        dry_run: bool = True,
        snapshot_dir: Optional[Path] = None,
        task_analyzer_prompts_path: Optional[Path] = None,
    ):
        """
        Initialize TAMP demo.

        Args:
            api_key: Gemini API key
            dry_run: If True, validate without executing on robot
            snapshot_dir: Optional path to snapshot directory to load perception data from
        """
        self.dry_run = dry_run
        self.snapshot_dir = snapshot_dir
        self.task_analyzer_prompts_path = task_analyzer_prompts_path

        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.run_dir = project_root / "outputs" / "tamp_demo" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Task counter for this run
        self.task_counter = 0

        # PDDL files are shared across tasks in this run
        self.pddl_dir = self.run_dir / "pddl"
        self.pddl_dir.mkdir(parents=True, exist_ok=True)

        # Route GenAI request/response logs alongside the run outputs (also passed into TAMPConfig)
        self.genai_log_path = self.run_dir / "genai_logs"

        # Configure TAMP
        config = TAMPConfig(
            api_key=api_key,
            state_dir=self.run_dir,
            orchestrator_model="gemini-robotics-er-1.5-preview",
            decomposer_model="gemini-robotics-er-1.5-preview",
            update_interval=2.0,
            min_observations=3,
            auto_refine_on_failure=True,
            max_refinement_attempts=3,
            solver_backend="pyperplan",  # Use pure Python solver (no Docker needed)
            solver_algorithm="lama-first",
            solver_timeout=60.0,
            dry_run_default=dry_run,
            primitives_interface=CuRoboMotionPlanner(robot_ip="192.168.1.224"),  # Would be robot interface in real deployment
            on_state_change=self._on_state_change,
            on_plan_generated=self._on_plan_generated,
            on_action_decomposed=self._on_action_decomposed,
            on_action_executed=self._on_action_executed,
            task_analyzer_prompts_path=task_analyzer_prompts_path,
            genai_log_dir=self.genai_log_path,
        )

        self.tamp = TaskAndMotionPlanner(config)
        self.execution_log = []

        # Log run metadata
        self._save_run_metadata()

        # Set up logging for this run
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to write to both console and run-specific log file."""
        # Create log file paths
        log_file = self.run_dir / f"{self.run_id}.log"
        timing_log_file = self.run_dir / f"{self.run_id}_timing.log"

        # Create a custom logger for this run
        self.logger = logging.getLogger(f"tamp_demo_{self.run_id}")
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if re-initializing
        if self.logger.handlers:
            self.logger.handlers.clear()

        # File handler - captures everything
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler - shows info and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Don't propagate to root logger
        self.logger.propagate = False

        # Create separate timing file for easy analysis
        self.timing_log = open(timing_log_file, 'w')
        self.timing_log.write(f"# Timing Log for {self.run_id}\n")
        self.timing_log.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        self.timing_log.write(f"# Format: timestamp,event,duration_ms,details\n")
        self.timing_log.write(f"timestamp,event,duration_ms,details\n")
        self.timing_log.flush()

        self.logger.info(f"Logging initialized for {self.run_id}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Timing log: {timing_log_file}")
        if self.genai_log_path:
            self.logger.info(f"GenAI logs: {self.genai_log_path}")

    def _log_timing(self, event: str, duration_s: float, details: str = ""):
        """
        Log timing information to both main log and timing CSV file.

        Args:
            event: Event name (e.g., "snapshot_load", "perception", "detection_cycle")
            duration_s: Duration in seconds
            details: Additional details about the event
        """
        timestamp = datetime.now().isoformat()
        duration_ms = duration_s * 1000

        # Log to main logger
        self.logger.info(f"[TIMING] {event}: {duration_s:.3f}s - {details}")

        # Log to timing CSV file
        details_clean = details.replace(',', ';').replace('\n', ' ')
        self.timing_log.write(f"{timestamp},{event},{duration_ms:.1f},{details_clean}\n")
        self.timing_log.flush()

    def log_and_print(self, message: str, level: str = "info"):
        """
        Log a message to file and print to console.

        Args:
            message: Message to log/print
            level: Log level (info, warning, error, debug)
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def _save_run_metadata(self):
        """Save metadata about this run."""
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "run_dir": str(self.run_dir),
            "tasks_completed": 0
        }
        metadata_path = self.run_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"\n✓ Created new run: {self.run_id}")
        print(f"  Directory: {self.run_dir}\n")

    def _save_task_result(self, task_id: int, task_description: str, result: TAMPResult):
        """Save results for a specific task within this run."""
        # Create task-specific directory
        task_dir = self.run_dir / f"task_{task_id:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save task metadata
        task_metadata = {
            "task_id": task_id,
            "task_description": task_description,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "failed_at_stage": result.failed_at_stage if not result.success else None,
            "error_message": result.error_message if not result.success else None,
            "planning_time": result.planning_time,
            "decomposition_time": result.decomposition_time,
            "execution_time": result.execution_time,
            "total_time": result.planning_time + result.decomposition_time + result.execution_time,
            "pddl_plan": result.pddl_plan,
            "plan_length": len(result.pddl_plan or []),
            "refinement_attempts": result.refinement_attempts
        }

        # Save task metadata
        (task_dir / "task_metadata.json").write_text(json.dumps(task_metadata, indent=2))

        # Save skill plans (decompositions)
        if result.skill_plans:
            decompositions = {}
            for action_name, skill_plan in result.skill_plans.items():
                decompositions[action_name] = {
                    "primitives": [
                        {
                            "name": prim.name,
                            "parameters": prim.parameters
                        }
                        for prim in skill_plan.primitives
                    ]
                }
            (task_dir / "decompositions.json").write_text(json.dumps(decompositions, indent=2))

        # Save execution results
        if result.execution_results:
            executions = []
            # Match execution results with their corresponding actions from the plan
            for i, exec_result in enumerate(result.execution_results):
                action_name = result.pddl_plan[i] if result.pddl_plan and i < len(result.pddl_plan) else f"action_{i}"
                executions.append({
                    "action": action_name,
                    "success": exec_result.executed,
                    "warnings": exec_result.warnings,
                    "errors": exec_result.errors,
                    "primitive_count": len(exec_result.primitive_results) if hasattr(exec_result, 'primitive_results') else 0
                })
            (task_dir / "execution_results.json").write_text(json.dumps(executions, indent=2))

        # Create symlink to shared PDDL files
        pddl_link = task_dir / "pddl"
        if not pddl_link.exists():
            pddl_link.symlink_to(self.pddl_dir, target_is_directory=True)

        print(f"\n✓ Saved task results to: {task_dir}")

        # Update run metadata
        metadata_path = self.run_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata["tasks_completed"] = task_id
            metadata["last_updated"] = datetime.now().isoformat()
            metadata_path.write_text(json.dumps(metadata, indent=2))

    def _on_state_change(self, state: TAMPState):
        """Callback for TAMP state changes."""
        msg = f"\n{'='*70}\nSTATE TRANSITION: {state.value.upper()}\n{'='*70}"
        print(msg)
        if hasattr(self, 'logger'):
            self.logger.info(f"State transition: {state.value}")

    def _on_plan_generated(self, result):
        """Callback when PDDL plan is generated."""
        if result.success:
            print(f"\n✓ Plan Generated ({result.plan_length} steps):")
            for i, action in enumerate(result.plan, 1):
                print(f"  {i}. {action}")
            if hasattr(self, 'logger'):
                self.logger.info(f"Plan generated with {result.plan_length} steps: {result.plan}")
        else:
            print(f"\n✗ Planning Failed: {result.error_message}")
            if hasattr(self, 'logger'):
                self.logger.error(f"Planning failed: {result.error_message}")

    def _on_action_decomposed(self, action: str, skill_plan):
        """Callback when action is decomposed to primitives."""
        print(f"  → Skill plan for '{action}':")
        for i, primitive in enumerate(skill_plan.primitives, 1):
            print(f"    {i}. {primitive.name}({', '.join(f'{k}={v}' for k, v in primitive.parameters.items())})")
        if hasattr(self, 'logger'):
            primitives_str = [f"{p.name}({p.parameters})" for p in skill_plan.primitives]
            self.logger.info(f"Action decomposed '{action}': {primitives_str}")

    def _on_action_executed(self, action: str, result):
        """Callback when action is executed."""
        self.execution_log.append({
            "action": action,
            "success": result.executed,
            "warnings": result.warnings,
            "errors": result.errors
        })
        if hasattr(self, 'logger'):
            status = "SUCCESS" if result.executed else "FAILED"
            self.logger.info(f"Action executed '{action}': {status}")
            if result.warnings:
                for w in result.warnings:
                    self.logger.warning(f"  {w}")
            if result.errors:
                for e in result.errors:
                    self.logger.error(f"  {e}")

    async def run_single_task(self, task: str):
        """
        Run TAMP for a single task.

        Args:
            task: Natural language task description
        """
        # Increment task counter
        self.task_counter += 1
        task_id = self.task_counter

        print(f"\n{'='*70}")
        print(f"TAMP TASK EXECUTION (Task {task_id})")
        print(f"{'='*70}")
        print(f"Run: {self.run_id}")
        print(f"Task: {task}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print(f"{'='*70}\n")

        self.logger.info(f"=" * 70)
        self.logger.info(f"Starting Task {task_id}: {task}")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        self.logger.info(f"=" * 70)

        try:
            # Initialize
            await self.tamp.initialize()
            # Set the task before perception so detection can start
            await self.tamp.orchestrator.process_task_request(task)
            self.tamp.current_task = task

            # Try to load snapshot from specified directory or previous run
            snapshot_loaded = False
            try:
                print("\nStep 1: Loading Perception Data")
                print("-" * 70)
                self.logger.info("Step 1: Loading Perception Data")
                snapshot_start = asyncio.get_event_loop().time()
                snapshot_loaded = await self._try_load_latest_snapshot(self.snapshot_dir)
                snapshot_time = asyncio.get_event_loop().time() - snapshot_start
                self._log_timing("snapshot_load", snapshot_time, f"loaded={snapshot_loaded}")
            except Exception as e:
                print(f"  ⚠ Could not load snapshot: {e}")
                self.logger.warning(f"Snapshot loading error: {e}")

            # Perceive environment only if snapshot wasn't loaded
            if not snapshot_loaded:
                print("\nStep 2: Environment Perception")
                print("-" * 70)
                self.logger.info("Step 2: Environment Perception - starting...")
                perception_start = asyncio.get_event_loop().time()

                perception_success = await self.tamp.perceive_environment(
                    duration=10.0,  # Max 10 seconds
                    min_observations=3
                )

                perception_time = asyncio.get_event_loop().time() - perception_start
                self.logger.info(f"Perception took {perception_time:.2f}s, success={perception_success}")
                self._log_timing("perception", perception_time, f"success={perception_success}")

                if not perception_success:
                    print("⚠ Warning: Insufficient observations, proceeding anyway...")
                    self.logger.warning("Insufficient observations during perception")
            else:
                print(f"\n  ✓ Skipping perception (using loaded snapshot data)")
                self.logger.info("Skipping perception - using loaded snapshot data")

            # Plan and execute
            step_num = 2 if snapshot_loaded else 3
            print(f"\nStep {step_num}: Task Planning, Decomposition, and Execution")
            print("-" * 70)

            task_start = asyncio.get_event_loop().time()
            result = await self.tamp.plan_and_execute_task(
                task=task,
                dry_run=self.dry_run,
                use_refinement=True,
                decompose_temperature=0.1
            )
            task_time = asyncio.get_event_loop().time() - task_start

            # Log detailed timing breakdown
            if result:
                self._log_timing("task_total", task_time, f"success={result.success}")
                if result.planning_time:
                    self._log_timing("planning", result.planning_time, f"plan_length={len(result.pddl_plan or [])}")
                if result.decomposition_time:
                    self._log_timing("decomposition", result.decomposition_time, f"actions={len(result.skill_plans or {})}")
                if result.execution_time:
                    self._log_timing("execution", result.execution_time, f"executed={len(result.execution_results or [])}")

            # Print summary
            self._print_result_summary(result)

            # Save task results
            if result:
                self._save_task_result(task_id, task, result)
                if result.success:
                    self.logger.info(f"Task {task_id} completed successfully")
                else:
                    self.logger.error(f"Task {task_id} failed: {result.error_message}")

            return result

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            self.logger.warning(f"Task {task_id} interrupted by user")
            return None
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Task {task_id} error: {e}", exc_info=True)
            return None
        finally:
            await self.tamp.shutdown()
            # Close timing log file
            if hasattr(self, 'timing_log') and self.timing_log:
                self.timing_log.close()

    async def _try_load_latest_snapshot(self, snapshot_dir: Optional[Path] = None) -> bool:
        """
        Load the most recent snapshot's `detections.json` into the tracker registry.

        This allows reusing perception data from previous runs without re-running perception.

        Args:
            snapshot_dir: Optional path to a specific snapshot directory. If None, looks for
                         the most recent snapshot in the perception pool of the current or
                         previous runs.

        Returns True if a snapshot was loaded, False otherwise.
        """
        # If specific snapshot directory provided, use it directly
        if snapshot_dir is not None:
            self.logger.info(f"Loading snapshot from specified directory: {snapshot_dir}")
            detections_path = snapshot_dir / "detections.json"
            if not detections_path.exists():
                print(f"  ⚠ Snapshot detections file not found: {detections_path}")
                self.logger.warning(f"Snapshot detections file not found: {detections_path}")
                return False

            # Ensure tracker and registry exist
            if not getattr(self.tamp.orchestrator, 'tracker', None) or not getattr(self.tamp.orchestrator.tracker, 'registry', None):
                print("  ⚠ Orchestrator tracker/registry not available to load snapshot")
                self.logger.warning("Orchestrator tracker/registry not available to load snapshot")
                return False

            try:
                # Clear existing objects first to avoid conflicts
                self.tamp.orchestrator.tracker.registry._objects.clear()
                self.tamp.orchestrator.tracker.registry.load_from_json(str(detections_path))
                num_objects = len(self.tamp.orchestrator.tracker.registry)
                print(f"  ✓ Loaded snapshot from {snapshot_dir.name} into registry")
                self.logger.info(f"Loaded snapshot from {snapshot_dir.name}: {num_objects} objects")
                return True
            except Exception as e:
                print(f"  ⚠ Failed to load registry from {detections_path}: {e}")
                self.logger.error(f"Failed to load registry from {detections_path}: {e}", exc_info=True)
                return False

        # Otherwise, search for most recent snapshot in current or previous runs
        # First try current run directory
        self.logger.info("Searching for snapshots in current/previous runs")
        percep_dir = Path(self.tamp.config.state_dir) / "perception_pool"
        index_path = percep_dir / "index.json"

        # If not found in current run, search previous runs
        if not index_path.exists():
            self.logger.debug(f"No snapshot index in current run: {percep_dir}")
            parent_dir = Path(self.tamp.config.state_dir).parent
            run_dirs = sorted([d for d in parent_dir.glob("run_*") if d.is_dir()], reverse=True)
            self.logger.debug(f"Found {len(run_dirs)} previous run directories")

            # Try each previous run (newest first)
            for run_dir in run_dirs:
                if run_dir == Path(self.tamp.config.state_dir):
                    continue  # Skip current run

                percep_dir = run_dir / "perception_pool"
                index_path = percep_dir / "index.json"

                if index_path.exists():
                    print(f"  ℹ Using snapshot from previous run: {run_dir.name}")
                    self.logger.info(f"Using snapshot from previous run: {run_dir.name}")
                    break
            else:
                # No snapshots found in any run
                self.logger.info("No snapshots found in any run")
                return False

        try:
            import json
            idx = json.loads(index_path.read_text())
        except Exception:
            return False

        # Prefer explicit last_snapshot_id if present, otherwise pick newest by captured_at
        snapshot_id = idx.get("last_snapshot_id")
        if not snapshot_id:
            # Find newest snapshot by captured_at/recorded_at metadata
            snaps = idx.get("snapshots", {})
            latest_ts = None
            latest_id = None
            from datetime import datetime
            for sid, meta in snaps.items():
                ts = meta.get("recorded_at") or meta.get("captured_at")
                if not ts:
                    continue
                try:
                    # Handle Z suffix
                    if ts.endswith("Z"):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(ts)
                    if latest_ts is None or dt > latest_ts:
                        latest_ts = dt
                        latest_id = sid
                except Exception:
                    continue
            snapshot_id = latest_id

        if not snapshot_id:
            return False

        snapshot_dir = percep_dir / "snapshots" / snapshot_id
        detections_path = snapshot_dir / "detections.json"

        if not detections_path.exists():
            print(f"  ⚠ Snapshot detections file not found: {detections_path}")
            return False

        # Ensure tracker and registry exist
        if not getattr(self.tamp.orchestrator, 'tracker', None) or not getattr(self.tamp.orchestrator.tracker, 'registry', None):
            print("  ⚠ Orchestrator tracker/registry not available to load snapshot")
            return False

        # Load detections.json into registry
        try:
            self.logger.info(f"Loading snapshot '{snapshot_id}' from {detections_path}")
            # Clear existing objects first to avoid conflicts
            self.tamp.orchestrator.tracker.registry._objects.clear()
            self.tamp.orchestrator.tracker.registry.load_from_json(str(detections_path))
            print(f"  ✓ Loaded snapshot '{snapshot_id}' into registry")

            # Verify objects were loaded
            num_objects = len(self.tamp.orchestrator.tracker.registry)
            print(f"  ℹ Registry now contains {num_objects} objects")
            self.logger.info(f"Snapshot '{snapshot_id}' loaded successfully: {num_objects} objects")

            if num_objects == 0:
                print(f"  ⚠ Warning: Snapshot loaded but contains no objects")
                self.logger.warning(f"Snapshot '{snapshot_id}' contains no objects")
                return False

            return True
        except Exception as e:
            print(f"  ⚠ Failed to load registry from {detections_path}: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Failed to load snapshot '{snapshot_id}': {e}", exc_info=True)
            return False

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

                    # Increment task counter
                    self.task_counter += 1
                    task_id = self.task_counter

                    task = parts[1]
                    print(f"\n--- Task {task_id}: {task} ---")

                    task_start = asyncio.get_event_loop().time()
                    result = await self.tamp.plan_and_execute_task(
                        task=task,
                        dry_run=self.dry_run
                    )
                    task_time = asyncio.get_event_loop().time() - task_start

                    self._print_result_summary(result)

                    # Log timing
                    if result:
                        self._log_timing("task_total", task_time, f"task_id={task_id};success={result.success}")
                        if result.planning_time:
                            self._log_timing("planning", result.planning_time, f"task_id={task_id};plan_length={len(result.pddl_plan or [])}")
                        if result.decomposition_time:
                            self._log_timing("decomposition", result.decomposition_time, f"task_id={task_id};actions={len(result.skill_plans or {})}")
                        if result.execution_time:
                            self._log_timing("execution", result.execution_time, f"task_id={task_id};executed={len(result.execution_results or [])}")

                        # Save task results
                        self._save_task_result(task_id, task, result)

                elif action == "perceive":
                    duration = float(parts[1]) if len(parts) > 1 else 10.0
                    print(f"\nPerceiving environment for {duration}s...")

                    perception_start = asyncio.get_event_loop().time()
                    success = await self.tamp.perceive_environment(duration=duration)
                    perception_time = asyncio.get_event_loop().time() - perception_start

                    self._log_timing("perception", perception_time, f"success={success}")

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
            # Close timing log file
            if hasattr(self, 'timing_log') and self.timing_log:
                self.timing_log.close()

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
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to snapshot directory to load perception data from (skips perception step)"
    )
    parser.add_argument(
        "--task-analyzer-prompts",
        type=str,
        default='config/llm_task_analyzer_prompts_lmh.yaml',
        help="Path to LLM task analyzer prompts YAML (e.g., config/llm_task_analyzer_prompts_lmh.yaml)"
    )

    args = parser.parse_args()

    # Check API key
    if not args.api_key:
        print("Error: GEMINI_API_KEY not set. Either:")
        print("  export GEMINI_API_KEY=your_key")
        print("  or use --api-key YOUR_KEY")
        return 1

    # Determine dry run mode (default to True for safety)
    dry_run = False #not args.live if args.live else True
    # if args.dry_run:
    #     dry_run = True

    # Parse snapshot directory if provided
    snapshot_dir = Path(args.snapshot) if args.snapshot else None
    if snapshot_dir and not snapshot_dir.exists():
        print(f"Error: Snapshot directory not found: {snapshot_dir}")
        return 1

    prompt_override = Path(args.task_analyzer_prompts) if args.task_analyzer_prompts else None
    if prompt_override and not prompt_override.exists():
        print(f"Error: Task analyzer prompts file not found: {prompt_override}")
        return 1

    # Create demo
    demo = TAMPDemo(
        api_key=args.api_key,
        dry_run=dry_run,
        snapshot_dir=snapshot_dir,
        task_analyzer_prompts_path=prompt_override,
    )

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
