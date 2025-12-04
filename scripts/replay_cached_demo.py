"""
Replay a past orchestrator demo using its cached snapshots (no camera required).

Example:
    uv run python scripts/replay_cached_demo.py \
        --source outputs/demos/20251202_173132
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.orchestrator_config import OrchestratorConfig  # noqa: E402
from src.camera.base_camera import BaseCamera, CameraIntrinsics  # noqa: E402
from src.planning.task_orchestrator import TaskOrchestrator  # noqa: E402
from src.utils.genai_logging import configure_genai_logging  # noqa: E402


@dataclass
class CachedFrame:
    color: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics


class SnapshotCamera(BaseCamera):
    """Camera stub that replays cached frames sequentially."""

    def __init__(self, frames: List[CachedFrame]):
        if not frames:
            raise ValueError("No cached frames supplied")
        self._frames = frames
        self._cursor = 0
        self._current_intrinsics = frames[0].intrinsics

    def start(self):
        """No-op for compatibility with RealSense interface."""

    def stop(self):
        """No-op for compatibility with RealSense interface."""

    def capture_frame(self) -> np.ndarray:
        return self._frames[self._cursor].color

    def get_depth(self) -> np.ndarray:
        return self._frames[self._cursor].depth

    def get_aligned_frames(self):
        frame = self._frames[self._cursor]
        self._current_intrinsics = frame.intrinsics
        if self._cursor < len(self._frames) - 1:
            self._cursor += 1
        return frame.color, frame.depth

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        return self._current_intrinsics


class NullRobot:
    """Stub robot provider so the orchestrator skips CuRobo initialization."""

    def get_robot_state(self):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Finished demo directory (e.g., outputs/demos/20251202_173132)",
    )
    parser.add_argument(
        "--dest-root",
        default="outputs/demos",
        help="Where to save the replay (default: outputs/demos)",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.5,
        help="Seconds between synthetic detections (default: 0.5s)",
    )
    return parser.parse_args()


def _require_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")
    return key


def _load_task_description(source_dir: Path) -> str:
    state_path = source_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state: {state_path}")
    state = json.loads(state_path.read_text())
    task = state.get("current_task")
    if not task:
        raise ValueError("state.json does not contain 'current_task'.")
    return task


def _load_cached_frames(source_dir: Path) -> List[CachedFrame]:
    index_path = source_dir / "perception_pool" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing perception pool index: {index_path}")

    index_payload = json.loads(index_path.read_text())
    snapshots = index_payload.get("snapshots") or {}
    if not snapshots:
        raise ValueError("No snapshots recorded in perception_pool/index.json")

    def _ts(entry: dict) -> float:
        ts = entry.get("recorded_at") or entry.get("captured_at")
        if not ts:
            return 0.0
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return 0.0

    ordered = sorted(snapshots.items(), key=lambda item: _ts(item[1]))
    pool_dir = source_dir / "perception_pool"
    frames: List[CachedFrame] = []
    for snapshot_id, meta in ordered:
        files = meta.get("files") or {}
        color_path = pool_dir / files.get("color", "")
        depth_path = pool_dir / files.get("depth_npz", "")
        intr_path = pool_dir / files.get("intrinsics", "")
        if not (color_path.exists() and depth_path.exists() and intr_path.exists()):
            raise FileNotFoundError(
                f"Snapshot {snapshot_id} is missing color/depth/intrinsics files."
            )

        color = np.array(Image.open(color_path).convert("RGB"))
        depth_np = np.load(depth_path)
        depth = depth_np[depth_np.files[0]]

        intr_data = json.loads(intr_path.read_text())
        distortion = intr_data.get("distortion")
        intrinsics = CameraIntrinsics(
            fx=intr_data["fx"],
            fy=intr_data["fy"],
            cx=intr_data["cx"],
            cy=intr_data["cy"],
            width=intr_data["width"],
            height=intr_data["height"],
            distortion=np.array(distortion) if distortion else None,
        )
        frames.append(CachedFrame(color=color, depth=depth, intrinsics=intrinsics))

    return frames


def _prepare_output_dir(dest_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_replay")
    target = dest_root / timestamp
    target.mkdir(parents=True, exist_ok=False)
    return target


async def _wait_for_detections(
    orchestrator: TaskOrchestrator,
    target: int,
    timeout: float
) -> bool:
    """Wait until detection_count reaches target or timeout expires."""
    loop = asyncio.get_event_loop()
    start = loop.time()
    while orchestrator.detection_count < target:
        if loop.time() - start > timeout:
            print(
                f"⚠ Timeout waiting for detections "
                f"(expected {target}, got {orchestrator.detection_count})."
            )
            return False
        await asyncio.sleep(0.25)
    return True


async def _replay_demo(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    task_description = _load_task_description(source_dir)
    frames = _load_cached_frames(source_dir)

    dest_root = Path(args.dest_root).resolve()
    dest_dir = _prepare_output_dir(dest_root)

    configure_genai_logging(dest_dir / "genai_logs")

    config = OrchestratorConfig(
        api_key=_require_api_key(),
        update_interval=args.update_interval,
        state_dir=dest_dir,
    )
    config.robot = NullRobot()
    camera = SnapshotCamera(frames)

    orchestrator = TaskOrchestrator(config, camera=camera)
    await orchestrator.initialize()
    await orchestrator.process_task_request(task_description)

    await orchestrator.start_detection()
    target_detections = max(1, len(frames))
    timeout = 90.0 + len(frames) * 45.0
    await _wait_for_detections(orchestrator, target_detections, timeout)
    await asyncio.sleep(1.0)  # allow final detection to finish streaming
    await orchestrator.stop_detection()

    await orchestrator.save_state()
    await orchestrator.shutdown()

    return dest_dir


def main() -> None:
    args = _parse_args()
    try:
        output_dir = asyncio.run(_replay_demo(args))
    except KeyboardInterrupt:
        print("\nReplay cancelled.")
        return
    print(f"Replay complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

