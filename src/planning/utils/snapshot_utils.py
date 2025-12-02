"""
Shared helpers for loading snapshot artifacts from the perception pool.
"""

from __future__ import annotations

import json
import io
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class SnapshotArtifacts:
    """Container for snapshot data consumed by planners/executors."""

    snapshot_id: Optional[str]
    meta: Optional[Dict[str, Any]] = None
    color_bytes: Optional[bytes] = None
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[Any] = None
    color_shape: Optional[Tuple[int, int]] = None
    robot_state: Optional[Dict[str, Any]] = None


@dataclass
class SnapshotCache:
    """Lightweight cache for snapshot indices and artifacts."""

    index: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, SnapshotArtifacts] = field(default_factory=dict)


def load_snapshot_artifacts(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    cache: Optional[SnapshotCache] = None,
    snapshot_id: Optional[str] = None,
) -> SnapshotArtifacts:
    """
    Load snapshot artifacts (color bytes, depth, intrinsics) for the latest snapshot.
    """
    snapshot_id = snapshot_id or world_state.get("last_snapshot_id")
    if not snapshot_id:
        return SnapshotArtifacts(snapshot_id=None)

    if cache and snapshot_id in cache.artifacts:
        return cache.artifacts[snapshot_id]

    index = _resolve_snapshot_index(world_state, perception_pool_dir, cache)
    meta = None
    if index:
        meta = (index.get("snapshots") or {}).get(snapshot_id)
    if meta is None:
        artifacts = SnapshotArtifacts(snapshot_id=None)
        if cache:
            cache.artifacts[snapshot_id] = artifacts
        return artifacts

    files = meta.get("files") or {}
    artifacts = SnapshotArtifacts(snapshot_id=snapshot_id, meta=meta)
    artifacts.color_bytes = _read_optional_bytes(perception_pool_dir, files.get("color"))
    artifacts.depth = _read_depth_array(perception_pool_dir, files.get("depth_npz"))
    artifacts.intrinsics = _read_intrinsics(perception_pool_dir, files.get("intrinsics"))
    artifacts.robot_state = _read_robot_state(perception_pool_dir, files.get("robot_state"))

    if artifacts.color_bytes is not None and files.get("color"):
        try:
            from PIL import Image

            with Image.open(io.BytesIO(artifacts.color_bytes)) as img:
                artifacts.color_shape = (img.height, img.width)
        except Exception:
            artifacts.color_shape = None

    if cache:
        cache.artifacts[snapshot_id] = artifacts
    return artifacts


def latest_snapshot_for_object_ids(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    object_ids: list[str],
    cache: Optional[SnapshotCache] = None,
) -> Optional[str]:
    """
    Pick the newest snapshot that contains any of the provided object ids.
    Falls back to None if index data is missing or no snapshots match.
    """
    if not object_ids:
        return None

    index = _resolve_snapshot_index(world_state, perception_pool_dir, cache)
    if not index:
        return None

    objects_map = index.get("objects") or {}
    snapshots_meta = index.get("snapshots") or {}

    def _snapshot_timestamp(sid: str) -> str:
        meta = snapshots_meta.get(sid) or {}
        return meta.get("recorded_at") or meta.get("captured_at") or ""

    candidates: list[str] = []
    for oid in object_ids:
        if not oid:
            continue
        snaps = objects_map.get(oid)
        if snaps:
            # Assume index lists snapshots chronologically; take the last as newest.
            candidates.append(snaps[-1])

    if not candidates:
        return None

    # Pick the candidate with the latest recorded/captured timestamp (lexicographic ISO sort).
    return max(candidates, key=_snapshot_timestamp)


def _resolve_snapshot_index(
    world_state: Dict[str, Any],
    perception_pool_dir: Path,
    cache: Optional[SnapshotCache],
) -> Optional[Dict[str, Any]]:
    if cache and cache.index:
        return cache.index
    index = world_state.get("snapshot_index")
    if index:
        if cache:
            cache.index = index
        return index
    index_path = Path(perception_pool_dir) / "index.json"
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        index = None
    if cache:
        cache.index = index
    return index


def _read_optional_bytes(perception_pool_dir: Path, relative_path: Optional[str]) -> Optional[bytes]:
    if not relative_path:
        return None
    path = Path(perception_pool_dir) / relative_path
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _read_depth_array(perception_pool_dir: Path, relative_path: Optional[str]) -> Optional[np.ndarray]:
    if not relative_path:
        return None
    path = Path(perception_pool_dir) / relative_path
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return data.get("depth_m")
    except Exception:
        return None


def _read_intrinsics(perception_pool_dir: Path, relative_path: Optional[str]) -> Optional[Any]:
    if not relative_path:
        return None
    path = Path(perception_pool_dir) / relative_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return SimpleNamespace(**data)
    except Exception:
        return None


def _read_robot_state(perception_pool_dir: Path, relative_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not relative_path:
        return None
    path = Path(perception_pool_dir) / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
