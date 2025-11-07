"""
Task manager for tracking and managing robotic tasks.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    Represents a robotic task with parsed components.
    """

    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING

    # Parsed task components
    action: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    goal_objects: List[str] = field(default_factory=list)
    tool_objects: List[str] = field(default_factory=list)
    spatial_constraints: List[Dict] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)
    goal_predicates: List[str] = field(default_factory=list)

    # Task metadata
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Context for perception
    focus_keywords: List[str] = field(default_factory=list)
    required_affordances: List[str] = field(default_factory=list)

    def start(self):
        """Mark task as started."""
        self.status = TaskStatus.ACTIVE
        self.started_at = time.time()

    def complete(self):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()

    def cancel(self):
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()

    def get_duration(self) -> Optional[float]:
        """Get task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "action": self.action,
            "objects": self.objects,
            "goal_objects": self.goal_objects,
            "tool_objects": self.tool_objects,
            "spatial_constraints": self.spatial_constraints,
            "attributes": self.attributes,
            "goal_predicates": self.goal_predicates,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.get_duration(),
        }


class TaskManager:
    """
    Manages robotic tasks and provides task context for perception.
    """

    def __init__(self):
        """Initialize task manager."""
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[Task] = None
        self._task_counter = 0

    def create_task(self, description: str, parsed_task: Dict, priority: int = 0) -> Task:
        """
        Create a new task from description and parsed components.

        Args:
            description: Natural language task description
            parsed_task: Parsed task dictionary from TaskParser
            priority: Task priority (higher = more important)

        Returns:
            Created Task object
        """
        task_id = f"task_{self._task_counter}"
        self._task_counter += 1

        task = Task(
            task_id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            action=parsed_task.get("action"),
            objects=parsed_task.get("objects", []),
            goal_objects=parsed_task.get("goal_objects", []),
            tool_objects=parsed_task.get("tool_objects", []),
            spatial_constraints=parsed_task.get("spatial_constraints", []),
            attributes=parsed_task.get("attributes", {}),
            goal_predicates=parsed_task.get("goal_predicates", []),
            priority=priority,
        )

        # Infer required affordances from action
        task.required_affordances = self._infer_affordances(task.action)

        self.tasks[task_id] = task
        return task

    def _infer_affordances(self, action: Optional[str]) -> List[str]:
        """Infer required affordances from action."""
        affordance_map = {
            "pick": ["graspable"],
            "grasp": ["graspable"],
            "grab": ["graspable"],
            "push": ["pushable"],
            "pull": ["pullable"],
            "open": ["openable"],
            "close": ["closeable"],
            "pour": ["containable", "pourable"],
            "place": ["supportable"],
        }

        return affordance_map.get(action, []) if action else []

    def set_current_task(self, task_id: str):
        """Set the current active task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if self.current_task and self.current_task.status == TaskStatus.ACTIVE:
                # Pause previous task
                self.current_task.status = TaskStatus.PENDING

            self.current_task = task
            task.start()
        else:
            raise ValueError(f"Task {task_id} not found")

    def get_current_task(self) -> Optional[Task]:
        """Get the currently active task."""
        return self.current_task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        return [task for task in self.tasks.values() if task.status == status]

    def complete_current_task(self):
        """Mark current task as completed."""
        if self.current_task:
            self.current_task.complete()
            self.current_task = None

    def fail_current_task(self):
        """Mark current task as failed."""
        if self.current_task:
            self.current_task.fail()
            self.current_task = None

    def cancel_task(self, task_id: str):
        """Cancel a task."""
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            if self.current_task and self.current_task.task_id == task_id:
                self.current_task = None

    def get_task_context(self) -> Optional[Dict]:
        """
        Get task context for perception conditioning.

        Returns:
            Dictionary with task context for perception, or None if no active task
        """
        if not self.current_task:
            return None

        return {
            "task_id": self.current_task.task_id,
            "description": self.current_task.description,
            "action": self.current_task.action,
            "goal_objects": self.current_task.goal_objects,
            "tool_objects": self.current_task.tool_objects,
            "required_affordances": self.current_task.required_affordances,
            "goal_predicates": self.current_task.goal_predicates,
            "spatial_constraints": self.current_task.spatial_constraints,
            "attributes": self.current_task.attributes,
        }

    def get_focus_objects(self) -> List[str]:
        """Get objects to focus on for current task."""
        if not self.current_task:
            return []

        focus_objects = []
        focus_objects.extend(self.current_task.goal_objects)
        focus_objects.extend(self.current_task.tool_objects)

        # Add reference objects from spatial constraints
        for constraint in self.current_task.spatial_constraints:
            if "reference" in constraint:
                focus_objects.append(constraint["reference"])

        return list(set(focus_objects))

    def get_required_affordances(self) -> List[str]:
        """Get required affordances for current task."""
        if not self.current_task:
            return []
        return self.current_task.required_affordances

    def clear_completed_tasks(self):
        """Remove all completed tasks."""
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        ]
        for task_id in to_remove:
            del self.tasks[task_id]

    def get_statistics(self) -> Dict:
        """Get task statistics."""
        total = len(self.tasks)
        by_status = {status: 0 for status in TaskStatus}

        for task in self.tasks.values():
            by_status[task.status] += 1

        return {
            "total_tasks": total,
            "by_status": {status.value: count for status, count in by_status.items()},
            "current_task": self.current_task.task_id if self.current_task else None,
        }

    def to_dict(self) -> Dict:
        """Convert task manager state to dictionary."""
        return {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "current_task_id": self.current_task.task_id if self.current_task else None,
            "statistics": self.get_statistics(),
        }
