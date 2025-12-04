"""
Visualization Script for TAMP Execution Analysis

Generates presentation-quality visualizations showing:
1. Object Detection & Tracking with RGB images
2. Tracked objects with bounding boxes and labels
3. Generated PDDL domain and problem files
4. Action decomposition with execution flow

Usage:
    python examples/visualize_tamp_execution.py outputs/tamp_demo/run_20250103_143022
    python examples/visualize_tamp_execution.py outputs/tamp_demo/run_20250103_143022 --task-id 1
    python examples/visualize_tamp_execution.py outputs/tamp_demo/run_20250103_143022 --output-dir ./presentation_visuals
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TAMPExecutionVisualizer:
    """Generate visualizations from TAMP execution logs with actual perception data."""

    def __init__(self, run_dir: Path, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            run_dir: Path to run directory (e.g., outputs/tamp_demo/run_20250103_143022)
            output_dir: Optional output directory for visualizations (default: run_dir/visualizations)
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        self.output_dir = Path(output_dir) if output_dir else self.run_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load run metadata
        self.run_metadata = self._load_json(self.run_dir / "run_metadata.json")
        self.run_id = self.run_metadata.get("run_id", self.run_dir.name)

        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'bbox': '#00ff00',
            'text': '#ffffff',
            'background': '#000000',
            'object_type': {
                'block': '#ff0000',
                'marker': '#00ff00',
                'cup': '#0000ff',
                'bowl': '#ffff00',
                'default': '#ffffff'
            }
        }

    def _load_json(self, path: Path) -> Dict:
        """Load JSON file, return empty dict if not found."""
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def visualize_all(self, task_id: Optional[int] = None):
        """
        Generate all visualizations.

        Args:
            task_id: Optional task ID to focus on (default: all tasks or latest)
        """
        print(f"\n{'='*70}")
        print(f"Generating TAMP Execution Visualizations")
        print(f"{'='*70}")
        print(f"Run: {self.run_id}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        # 1. Object Detection Visualization (with RGB images)
        print("1. Generating object detection visualization...")
        self.visualize_object_detections()

        # 2. Object Tracking Timeline
        print("2. Generating object tracking timeline...")
        self.visualize_tracking_timeline()

        # 3. Generated PDDL Domain
        print("3. Generating PDDL domain visualization...")
        self.visualize_pddl_domain()

        # 4. Generated PDDL Problem
        print("4. Generating PDDL problem visualization...")
        self.visualize_pddl_problem()

        # 5. Action Decomposition (if task specified)
        if task_id is not None:
            print(f"5. Generating action decomposition for task {task_id}...")
            self.visualize_action_decomposition(task_id)
        else:
            # Generate for first task
            task_dirs = sorted(self.run_dir.glob("task_*"))
            if task_dirs:
                first_task_id = int(task_dirs[0].name.split('_')[1])
                print(f"5. Generating action decomposition for task {first_task_id}...")
                self.visualize_action_decomposition(first_task_id)

        # 6. Execution Summary
        print("6. Generating execution summary...")
        self.visualize_execution_summary(task_id)

        print(f"\n{'='*70}")
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")

    def visualize_object_detections(self):
        """Visualize detected objects with RGB images and bounding boxes."""
        # Find perception pool snapshots
        perception_pool = self.run_dir / "perception_pool"
        if not perception_pool.exists():
            print("  ⚠ No perception pool found")
            return

        snapshots_dir = perception_pool / "snapshots"
        if not snapshots_dir.exists():
            print("  ⚠ No snapshots found")
            return

        # Look for snapshot directories (various naming formats)
        snapshot_dirs = sorted(snapshots_dir.glob("*"))
        # Filter to only directories
        snapshot_dirs = [d for d in snapshot_dirs if d.is_dir()]

        if not snapshot_dirs:
            print("  ⚠ No snapshot directories found")
            return

        # Use the latest snapshot (already sorted)
        latest_snapshot = snapshot_dirs[-1]
        print(f"  Using snapshot: {latest_snapshot.name}")

        # Load detection data
        detections_file = latest_snapshot / "detections.json"
        if not detections_file.exists():
            print("  ⚠ No detections.json found")
            return

        detections = self._load_json(detections_file)
        objects = detections.get("objects", [])

        if not objects:
            print("  ⚠ No objects in detections")
            return

        # Load RGB image if available (try multiple possible names)
        rgb_image = None
        for image_name in ["color.png", "rgb.png", "image.png"]:
            rgb_file = latest_snapshot / image_name
            if rgb_file.exists():
                rgb_image = Image.open(rgb_file)
                break

        # Create figure
        if rgb_image:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(16, 10))
            ax1 = None

        # Left panel: RGB image with bounding boxes
        if rgb_image and ax1:
            ax1.imshow(rgb_image)
            ax1.set_title('Detected Objects in Scene', fontsize=16, fontweight='bold')
            ax1.axis('off')

            # Draw bounding boxes
            for obj in objects:
                if 'bounding_box' in obj:
                    bbox = obj['bounding_box']
                    x_min = bbox.get('x_min', 0)
                    y_min = bbox.get('y_min', 0)
                    width = bbox.get('width', 0)
                    height = bbox.get('height', 0)

                    # Get color based on object type
                    obj_type = obj.get('object_type', 'default').split('_')[0]
                    color = self.colors['object_type'].get(obj_type, self.colors['object_type']['default'])

                    # Draw rectangle
                    rect = Rectangle((x_min, y_min), width, height,
                                   linewidth=3, edgecolor=color, facecolor='none')
                    ax1.add_patch(rect)

                    # Add label
                    label = obj.get('object_id', 'unknown')
                    ax1.text(x_min, y_min - 5, label,
                           color=color, fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Right panel: Object details
        ax2.axis('off')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, max(10, len(objects) * 1.2))

        ax2.text(5, ax2.get_ylim()[1] - 0.5, 'Tracked Objects',
                ha='center', va='top', fontsize=16, fontweight='bold')

        y_pos = ax2.get_ylim()[1] - 1.5

        for i, obj in enumerate(objects, 1):
            obj_id = obj.get('object_id', 'unknown')
            obj_type = obj.get('object_type', 'unknown')
            affordances = obj.get('affordances', [])
            position = obj.get('position', {})

            # Object header
            obj_type_base = obj_type.split('_')[0]
            color = self.colors['object_type'].get(obj_type_base, self.colors['object_type']['default'])

            box = FancyBboxPatch((0.5, y_pos - 0.5), 9, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor=color, facecolor=color,
                                linewidth=2, alpha=0.3)
            ax2.add_patch(box)

            # Object ID and type
            ax2.text(1, y_pos, f"{i}. {obj_id}",
                    ha='left', va='top', fontsize=12, fontweight='bold', color=color)
            ax2.text(1, y_pos - 0.3, f"Type: {obj_type}",
                    ha='left', va='top', fontsize=10)

            # Position
            if position:
                pos_str = f"Position: ({position.get('x', 0):.3f}, {position.get('y', 0):.3f}, {position.get('z', 0):.3f})"
                ax2.text(6, y_pos, pos_str,
                        ha='left', va='top', fontsize=9, family='monospace')

            # Affordances
            if affordances:
                affordances_str = f"Affordances: {', '.join(affordances)}"
                ax2.text(6, y_pos - 0.3, affordances_str,
                        ha='left', va='top', fontsize=9, style='italic')

            y_pos -= 1.2

        # Add summary at bottom
        summary_text = (
            f"Snapshot: {latest_snapshot.name}\n"
            f"Total Objects: {len(objects)}\n"
            f"Timestamp: {detections.get('detection_timestamp', 'unknown')}"
        )
        ax2.text(5, 0.5, summary_text,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / "1_object_detections.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_tracking_timeline(self):
        """Visualize object tracking over time across snapshots."""
        perception_pool = self.run_dir / "perception_pool"
        if not perception_pool.exists():
            print("  ⚠ No perception pool found")
            return

        snapshots_dir = perception_pool / "snapshots"
        if not snapshots_dir.exists():
            print("  ⚠ No snapshots found")
            return

        # Look for snapshot directories (various naming formats)
        snapshot_dirs = sorted(snapshots_dir.glob("*"))
        # Filter to only directories
        snapshot_dirs = [d for d in snapshot_dirs if d.is_dir()]

        if len(snapshot_dirs) < 2:
            print("  ℹ Not enough snapshots for timeline (need at least 2)")
            return

        # Collect object tracking data across snapshots
        timeline_data = {}  # object_id -> [(snapshot_idx, position), ...]

        for idx, snapshot_dir in enumerate(snapshot_dirs):
            detections_file = snapshot_dir / "detections.json"
            if not detections_file.exists():
                continue

            detections = self._load_json(detections_file)
            for obj in detections.get("objects", []):
                obj_id = obj.get('object_id', 'unknown')
                position = obj.get('position', {})

                if obj_id not in timeline_data:
                    timeline_data[obj_id] = []

                timeline_data[obj_id].append({
                    'snapshot': idx,
                    'x': position.get('x', 0),
                    'y': position.get('y', 0),
                    'z': position.get('z', 0),
                    'obj_type': obj.get('object_type', 'unknown')
                })

        if not timeline_data:
            print("  ⚠ No tracking data found")
            return

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Object Tracking Timeline', fontsize=16, fontweight='bold')

        # Left: Object presence across snapshots
        ax1.set_title('Object Persistence Across Snapshots')
        ax1.set_xlabel('Snapshot Index')
        ax1.set_ylabel('Object ID')

        obj_ids = sorted(timeline_data.keys())
        y_positions = {obj_id: i for i, obj_id in enumerate(obj_ids)}

        for obj_id, detections in timeline_data.items():
            y = y_positions[obj_id]
            snapshots = [d['snapshot'] for d in detections]

            # Get color based on object type
            obj_type = detections[0]['obj_type'].split('_')[0] if detections else 'default'
            color = self.colors['object_type'].get(obj_type, self.colors['object_type']['default'])

            ax1.scatter(snapshots, [y] * len(snapshots), s=200, color=color, alpha=0.7, marker='o')
            ax1.plot(snapshots, [y] * len(snapshots), color=color, alpha=0.3, linewidth=2)

        ax1.set_yticks(list(y_positions.values()))
        ax1.set_yticklabels(list(y_positions.keys()))
        ax1.grid(axis='x', alpha=0.3)

        # Right: 3D position tracking
        ax2.set_title('Object Position Tracking (Top View: X-Y)')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_aspect('equal')

        for obj_id, detections in timeline_data.items():
            xs = [d['x'] for d in detections]
            ys = [d['y'] for d in detections]

            obj_type = detections[0]['obj_type'].split('_')[0] if detections else 'default'
            color = self.colors['object_type'].get(obj_type, self.colors['object_type']['default'])

            # Plot trajectory
            ax2.plot(xs, ys, 'o-', color=color, alpha=0.5, linewidth=2, markersize=8)

            # Label first and last positions
            if len(xs) > 0:
                ax2.text(xs[0], ys[0], f"{obj_id}\n(start)", ha='center', va='bottom',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                if len(xs) > 1:
                    ax2.text(xs[-1], ys[-1], f"{obj_id}\n(end)", ha='center', va='top',
                            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "2_tracking_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_pddl_domain(self):
        """Visualize the generated PDDL domain file."""
        pddl_dir = self.run_dir / "pddl"
        if not pddl_dir.exists():
            print("  ⚠ No PDDL directory found")
            return

        domain_files = list(pddl_dir.glob("*_domain.pddl"))
        if not domain_files:
            print("  ⚠ No domain file found")
            return

        domain_file = domain_files[0]
        domain_text = domain_file.read_text()

        # Create figure with domain text
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.98, 'Generated PDDL Domain',
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        # Domain file content
        ax.text(0.05, 0.93, domain_text,
                ha='left', va='top', fontsize=9, family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8),
                wrap=True)

        # Parse and show statistics
        predicates = self._extract_pddl_section(domain_text, ":predicates")
        actions = self._extract_actions(domain_text)

        stats_text = (
            f"Domain Statistics:\n"
            f"  Predicates: {len(predicates)}\n"
            f"  Actions: {len(actions)}\n"
            f"  File: {domain_file.name}"
        )
        ax.text(0.95, 0.05, stats_text,
                ha='right', va='bottom', fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.output_dir / "3_pddl_domain.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_pddl_problem(self):
        """Visualize the generated PDDL problem file."""
        pddl_dir = self.run_dir / "pddl"
        if not pddl_dir.exists():
            print("  ⚠ No PDDL directory found")
            return

        problem_files = list(pddl_dir.glob("*_problem.pddl"))
        if not problem_files:
            print("  ⚠ No problem file found")
            return

        problem_file = problem_files[0]
        problem_text = problem_file.read_text()

        # Create figure with problem text
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.98, 'Generated PDDL Problem',
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        # Problem file content
        ax.text(0.05, 0.93, problem_text,
                ha='left', va='top', fontsize=9, family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8),
                wrap=True)

        # Parse and show statistics
        objects = self._extract_pddl_section(problem_text, ":objects")
        init = self._extract_pddl_section(problem_text, ":init")
        goal = self._extract_pddl_section(problem_text, ":goal")

        stats_text = (
            f"Problem Statistics:\n"
            f"  Objects: {len(objects)}\n"
            f"  Initial Facts: {len(init)}\n"
            f"  Goal Conditions: {len(goal)}\n"
            f"  File: {problem_file.name}"
        )
        ax.text(0.95, 0.05, stats_text,
                ha='right', va='bottom', fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.output_dir / "4_pddl_problem.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_action_decomposition(self, task_id: int):
        """Visualize action decomposition from PDDL plan to primitives."""
        task_dir = self.run_dir / f"task_{task_id:03d}"
        if not task_dir.exists():
            print(f"  ⚠ Task {task_id} not found")
            return

        # Load task data
        task_metadata = self._load_json(task_dir / "task_metadata.json")
        decompositions = self._load_json(task_dir / "decompositions.json")
        execution_results = self._load_json(task_dir / "execution_results.json")

        if not decompositions:
            print(f"  ⚠ No decomposition data for task {task_id}")
            return

        pddl_plan = task_metadata.get("pddl_plan", [])
        task_desc = task_metadata.get("task_description", "Unknown task")

        # Create figure
        fig, ax = plt.subplots(figsize=(18, max(10, len(pddl_plan) * 2.5)))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, max(12, len(pddl_plan) * 3))
        ax.axis('off')

        # Title
        ax.text(9, ax.get_ylim()[1] - 0.5, f'Task {task_id}: Action Decomposition',
                ha='center', va='top', fontsize=18, fontweight='bold')
        ax.text(9, ax.get_ylim()[1] - 1.2, f'"{task_desc}"',
                ha='center', va='top', fontsize=13, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        y_pos = ax.get_ylim()[1] - 2.5

        for i, action in enumerate(pddl_plan, 1):
            # PDDL Action box
            action_box = FancyBboxPatch((1, y_pos), 6, 1,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='#e74c3c', facecolor='#e74c3c',
                                       linewidth=3, alpha=0.7)
            ax.add_patch(action_box)

            ax.text(4, y_pos + 0.5, f"Step {i}: {action}",
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white')

            # Arrow
            arrow = FancyArrowPatch((7, y_pos + 0.5), (10, y_pos + 0.5),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='black')
            ax.add_patch(arrow)

            # Skill Primitives
            if action in decompositions:
                primitives = decompositions[action].get("primitives", [])

                box_height = max(1, len(primitives) * 0.3 + 0.5)
                prim_box = FancyBboxPatch((10, y_pos + 0.5 - box_height/2), 7, box_height,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='#f39c12', facecolor='#f39c12',
                                         linewidth=3, alpha=0.5)
                ax.add_patch(prim_box)

                ax.text(13.5, y_pos + 0.5 + box_height/2 - 0.2,
                       f"Skill Primitives ({len(primitives)})",
                       ha='center', va='top', fontsize=10, fontweight='bold')

                # List primitives
                prim_y = y_pos + 0.3 + box_height/2 - 0.5
                for prim in primitives[:15]:  # Limit for readability
                    prim_name = prim.get("name", "unknown")
                    params = prim.get("parameters", {})

                    # Format parameters
                    param_items = list(params.items())
                    if len(param_items) <= 3:
                        param_str = ", ".join([f"{k}={v}" for k, v in param_items])
                    else:
                        param_str = ", ".join([f"{k}={v}" for k, v in param_items[:3]]) + ", ..."

                    ax.text(10.5, prim_y, f"→ {prim_name}({param_str})",
                           ha='left', va='top', fontsize=8, family='monospace')
                    prim_y -= 0.25

                if len(primitives) > 15:
                    ax.text(10.5, prim_y, f"... and {len(primitives) - 15} more primitives",
                           ha='left', va='top', fontsize=8, style='italic')

                y_pos -= max(2, box_height + 0.8)
            else:
                ax.text(13.5, y_pos + 0.5, "No decomposition data",
                       ha='center', va='center', fontsize=10, style='italic', color='red')
                y_pos -= 2

        # Execution summary
        if execution_results:
            success_count = sum(1 for r in execution_results if r.get('success', False))
            total_count = len(execution_results)

            summary_text = (
                f"Execution Summary:\n"
                f"  Actions Executed: {total_count}\n"
                f"  Successful: {success_count}/{total_count}\n"
                f"  Success Rate: {100*success_count/total_count:.0f}%" if total_count > 0 else ""
            )
            ax.text(9, 1, summary_text,
                   ha='center', va='bottom', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if success_count == total_count else 'lightyellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.output_dir / f"5_action_decomposition_task{task_id}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_execution_summary(self, task_id: Optional[int] = None):
        """Create an execution summary visualization."""
        # Get task directories
        task_dirs = sorted(self.run_dir.glob("task_*"))
        if task_id is not None:
            task_dirs = [d for d in task_dirs if int(d.name.split('_')[1]) == task_id]

        if not task_dirs:
            print("  ⚠ No tasks found")
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')

        # Title
        title = f"Execution Summary - Task {task_id}" if task_id else f"Execution Summary - All Tasks ({len(task_dirs)})"
        ax.text(0.5, 0.95, title,
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        # Collect statistics
        total_tasks = len(task_dirs)
        successful_tasks = 0
        total_planning_time = 0
        total_decomposition_time = 0
        total_execution_time = 0
        total_actions = 0

        task_details = []

        for task_dir in task_dirs:
            task_metadata = self._load_json(task_dir / "task_metadata.json")

            task_num = int(task_dir.name.split('_')[1])
            task_desc = task_metadata.get("task_description", "Unknown")
            success = task_metadata.get("success", False)

            if success:
                successful_tasks += 1

            planning_time = task_metadata.get("planning_time", 0)
            decomposition_time = task_metadata.get("decomposition_time", 0)
            execution_time = task_metadata.get("execution_time", 0)
            plan_length = task_metadata.get("plan_length", 0)

            total_planning_time += planning_time
            total_decomposition_time += decomposition_time
            total_execution_time += execution_time
            total_actions += plan_length

            task_details.append({
                'num': task_num,
                'desc': task_desc,
                'success': success,
                'time': planning_time + decomposition_time + execution_time,
                'actions': plan_length
            })

        # Overall statistics box
        stats_text = (
            f"Overall Statistics:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Total Tasks: {total_tasks}\n"
            f"Successful: {successful_tasks}/{total_tasks} ({100*successful_tasks/total_tasks:.0f}%)\n"
            f"Total Actions: {total_actions}\n"
            f"\n"
            f"Timing Breakdown:\n"
            f"  Planning:      {total_planning_time:.2f}s\n"
            f"  Decomposition: {total_decomposition_time:.2f}s\n"
            f"  Execution:     {total_execution_time:.2f}s\n"
            f"  Total:         {total_planning_time + total_decomposition_time + total_execution_time:.2f}s\n"
            f"\n"
            f"Average per Task:\n"
            f"  Time:          {(total_planning_time + total_decomposition_time + total_execution_time)/total_tasks:.2f}s\n"
            f"  Actions:       {total_actions/total_tasks:.1f}"
        )

        ax.text(0.05, 0.85, stats_text,
                ha='left', va='top', fontsize=11, family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # Task details table
        y_offset = 0.55
        ax.text(0.5, y_offset, "Task Details:",
                ha='left', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)

        y_offset -= 0.05
        for task in task_details:
            status_symbol = "✓" if task['success'] else "✗"
            status_color = 'green' if task['success'] else 'red'

            task_line = f"{status_symbol} Task {task['num']}: {task['desc'][:60]}"
            time_line = f"    Time: {task['time']:.2f}s | Actions: {task['actions']}"

            ax.text(0.52, y_offset, task_line,
                   ha='left', va='top', fontsize=10, color=status_color,
                   transform=ax.transAxes)
            y_offset -= 0.04
            ax.text(0.52, y_offset, time_line,
                   ha='left', va='top', fontsize=9, style='italic',
                   transform=ax.transAxes)
            y_offset -= 0.05

        plt.tight_layout()
        suffix = f"_task{task_id}" if task_id else ""
        plt.savefig(self.output_dir / f"6_execution_summary{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Helper methods
    def _extract_pddl_section(self, text: str, section: str) -> List[str]:
        """Extract items from a PDDL section."""
        items = []
        in_section = False
        paren_count = 0
        current_item = ""

        for line in text.split('\n'):
            line = line.strip()
            if section in line:
                in_section = True
                continue

            if in_section:
                for char in line:
                    if char == '(':
                        paren_count += 1
                        if paren_count == 1:
                            current_item = ""
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0 and current_item:
                            items.append(current_item.strip())
                            current_item = ""
                        if paren_count < 0:
                            in_section = False
                            break
                    elif paren_count > 0:
                        current_item += char

                if not in_section:
                    break

        return items

    def _extract_actions(self, text: str) -> List[str]:
        """Extract action names from PDDL domain."""
        actions = []
        for line in text.split('\n'):
            line = line.strip()
            if ':action' in line:
                parts = line.split()
                if len(parts) >= 2:
                    actions.append(parts[1])
        return actions


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for TAMP execution analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('run_dir', type=str, help='Path to run directory')
    parser.add_argument('--task-id', type=int, help='Focus on specific task ID')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')

    args = parser.parse_args()

    try:
        visualizer = TAMPExecutionVisualizer(args.run_dir, args.output_dir)
        visualizer.visualize_all(args.task_id)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
