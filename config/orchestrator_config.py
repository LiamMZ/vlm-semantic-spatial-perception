"""
Orchestrator Configuration

Configuration dataclass for the Task Orchestrator.
"""

from pathlib import Path
from typing import Optional, Callable, Any, Literal
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

    # Snapshot / Perception Pool (optional, non-breaking)
    enable_snapshots: bool = True  # Enabled by default
    snapshot_every_n_detections: int = 1  # Save every detection update by default
    # When None, will default to state_dir / "perception_pool" at runtime
    perception_pool_dir: Optional[Path] = None
    max_snapshot_count: Optional[int] = 200  # Rotate oldest if exceeded
    depth_encoding: Literal["npz"] = "npz"  # Keep simple for now
    # Optional robot provider (duck-typed):
    # - get_robot_joint_state() -> List[float] | np.ndarray
    # - get_robot_tcp_pose() -> Tuple[List[float], List[float]] (position, quaternion_xyzw)
    # - get_camera_transform() -> Tuple[List[float], AnyRotation] (position, rotation)
    robot: Optional[Any] = None

    # Task Configuration
    exploration_timeout: float = 60.0  # Max time for exploration before forcing decision

    # Callbacks
    on_state_change: Optional[Callable[["OrchestratorState", "OrchestratorState"], None]] = None
    on_detection_update: Optional[Callable[[int], None]] = None
    on_task_state_change: Optional[Callable[["TaskStateDecision"], None]] = None
    on_save_state: Optional[Callable[[Path], None]] = None  # Called after successful save
