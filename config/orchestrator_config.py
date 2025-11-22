"""
Orchestrator Configuration

Configuration dataclass for the Task Orchestrator.
"""

from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

# Import types needed for type hints
# These are imported at runtime only for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.planning.task_orchestrator import OrchestratorState
    from src.planning.task_state_monitor import TaskStateDecision


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # API Configuration
    api_key: str
    model_name: str = "gemini-2.5-flash"

    # Camera Configuration
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    enable_depth: bool = True

    # Detection Configuration
    update_interval: float = 2.0  # Seconds between detections
    min_observations: int = 3  # Minimum objects before planning
    fast_mode: bool = False  # Skip interaction points for speed
    scene_change_threshold: float = 0.15
    enable_scene_change_detection: bool = True

    # Persistence Configuration
    state_dir: Path = field(default_factory=lambda: Path("outputs/orchestrator_state"))
    auto_save: bool = True  # Auto-save state on updates
    auto_save_on_detection: bool = True  # Save after each detection update
    auto_save_on_state_change: bool = True  # Save on state changes

    # Task Configuration
    exploration_timeout: float = 60.0  # Max time for exploration before forcing decision

    # Callbacks
    on_state_change: Optional[Callable[["OrchestratorState", "OrchestratorState"], None]] = None
    on_detection_update: Optional[Callable[[int], None]] = None
    on_task_state_change: Optional[Callable[["TaskStateDecision"], None]] = None
    on_save_state: Optional[Callable[[Path], None]] = None  # Called after successful save
