"""
Continuous background object tracking service.

This module provides an async background service that continuously detects objects
and maintains an up-to-date registry accessible for external planning.
"""

import time
import asyncio
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2

from .object_tracker import ObjectTracker
from .object_registry import DetectedObject


@dataclass
class TrackingStats:
    """Statistics for the continuous tracker."""
    total_frames: int = 0
    total_detections: int = 0
    skipped_frames: int = 0
    avg_detection_time: float = 0.0
    last_detection_time: float = 0.0
    cache_hit_rate: float = 0.0
    is_running: bool = False


class ContinuousObjectTracker:
    """
    Background service for continuous object tracking.

    This service runs in a separate thread, continuously processing frames
    and maintaining an up-to-date object registry that can be queried externally.

    Example:
        >>> # Create tracker with fast mode enabled
        >>> tracker = ContinuousObjectTracker(
        ...     api_key="your_key",
        ...     fast_mode=True,
        ...     update_interval=0.5  # Update twice per second
        ... )
        >>>
        >>> # Start background tracking
        >>> tracker.start()
        >>>
        >>> # Query objects from another thread/process
        >>> graspable = tracker.get_objects_with_affordance("graspable")
        >>>
        >>> # Stop when done
        >>> tracker.stop()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",
        max_parallel_requests: int = 5,
        crop_target_size: int = 512,
        enable_affordance_caching: bool = True,
        fast_mode: bool = False,
        update_interval: float = 1.0,
        on_detection_complete: Optional[Callable[[int], None]] = None,
        scene_change_threshold: float = 0.15,
        enable_scene_change_detection: bool = True
    ):
        """
        Initialize continuous tracker.

        Args:
            api_key: Google AI API key
            model_name: Model to use
            max_parallel_requests: Max parallel VLM requests
            crop_target_size: Resize crops to this size (0=disable)
            enable_affordance_caching: Cache affordance results
            fast_mode: Skip interaction points for speed
            update_interval: Minimum time between detections (seconds)
            on_detection_complete: Callback called with object count after each detection
            scene_change_threshold: Threshold for scene change detection (0-1, lower=more sensitive)
            enable_scene_change_detection: Skip detection if scene hasn't changed
        """
        self.tracker = ObjectTracker(
            api_key=api_key,
            model_name=model_name,
            max_parallel_requests=max_parallel_requests,
            crop_target_size=crop_target_size,
            enable_affordance_caching=enable_affordance_caching,
            fast_mode=fast_mode
        )

        self.update_interval = update_interval
        self.on_detection_complete = on_detection_complete
        self.scene_change_threshold = scene_change_threshold
        self.enable_scene_change_detection = enable_scene_change_detection

        # Async control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Frame source
        self._frame_provider: Optional[Callable[[], tuple]] = None

        # Scene change detection
        self._last_frame_embedding: Optional[np.ndarray] = None
        self._last_detection_objects: int = 0

        # Statistics
        self.stats = TrackingStats()

    def set_frame_provider(
        self,
        provider: Callable[[], tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]]
    ):
        """
        Set the frame provider function.

        The provider should return (color_frame, depth_frame, camera_intrinsics).
        This will be called continuously by the background thread.

        Args:
            provider: Function that returns (color, depth, intrinsics) tuple
        """
        self._frame_provider = provider

    def start(self):
        """Start the background tracking task."""
        if self._running:
            print("⚠ Tracker already running")
            return

        if self._frame_provider is None:
            raise ValueError("Frame provider not set. Call set_frame_provider() first.")

        self._running = True
        self.stats.is_running = True

        # Create and run task in background
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._tracking_loop())
        print("✓ Continuous tracker started")

    async def stop(self):
        """Stop the background tracking task."""
        if not self._running:
            return

        self._running = False
        self.stats.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("✓ Continuous tracker stopped")

    def _compute_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a compact embedding/hash of the frame for scene change detection.

        Uses a combination of:
        - Downsampled grayscale histogram
        - Edge detection histogram
        - Color histogram

        Args:
            frame: RGB frame (H, W, 3)

        Returns:
            Embedding vector (fixed size)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Resize to standard size for consistency
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        # 1. Intensity histogram (64 bins)
        hist_intensity = cv2.calcHist([small], [0], None, [64], [0, 256])
        hist_intensity = hist_intensity.flatten() / hist_intensity.sum()

        # 2. Edge histogram (Canny edges)
        edges = cv2.Canny(small, 50, 150)
        hist_edges = cv2.calcHist([edges], [0], None, [32], [0, 256])
        hist_edges = hist_edges.flatten() / (hist_edges.sum() + 1e-6)

        # 3. Color histogram if color image
        if len(frame.shape) == 3:
            small_color = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            hist_r = cv2.calcHist([small_color], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([small_color], [1], None, [16], [0, 256]).flatten()
            hist_b = cv2.calcHist([small_color], [2], None, [16], [0, 256]).flatten()
            hist_color = np.concatenate([hist_r, hist_g, hist_b])
            hist_color = hist_color / (hist_color.sum() + 1e-6)
        else:
            hist_color = np.zeros(48)

        # Combine all features
        embedding = np.concatenate([hist_intensity, hist_edges, hist_color])

        return embedding

    def _compute_scene_difference(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute difference between two frame embeddings.

        Uses cosine distance for robustness to lighting changes.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Difference score (0 = identical, 1 = completely different)
        """
        # Cosine distance (more robust to lighting)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        cosine_sim = dot_product / (norm1 * norm2)
        cosine_distance = 1.0 - cosine_sim

        return float(cosine_distance)

    async def _tracking_loop(self):
        """Main tracking loop (runs as async task)."""
        print("→ Tracking loop started")
        if self.enable_scene_change_detection:
            print(f"  Scene change detection enabled (threshold: {self.scene_change_threshold})")

        while self._running:
            loop_start = time.time()

            try:
                # Get current frame
                color_frame, depth_frame, intrinsics = self._frame_provider()

                # Check if scene has changed (if enabled)
                should_detect = True
                if self.enable_scene_change_detection and self._last_frame_embedding is not None:
                    current_embedding = self._compute_frame_embedding(color_frame)
                    scene_diff = self._compute_scene_difference(
                        current_embedding,
                        self._last_frame_embedding
                    )

                    if scene_diff < self.scene_change_threshold:
                        # Scene hasn't changed significantly, skip detection
                        should_detect = False
                        async with self._lock:
                            self.stats.total_frames += 1
                            self.stats.skipped_frames += 1
                            # Update cache hit rate
                            total_attempted = self.stats.total_frames
                            self.stats.cache_hit_rate = self.stats.skipped_frames / total_attempted if total_attempted > 0 else 0.0

                        # Still call callback with last known count
                        if self.on_detection_complete:
                            if asyncio.iscoroutinefunction(self.on_detection_complete):
                                await self.on_detection_complete(self._last_detection_objects)
                            else:
                                self.on_detection_complete(self._last_detection_objects)

                if should_detect:
                    # Run async detection
                    detection_start = time.time()
                    detected_objects = await self.tracker.detect_objects(
                        color_frame,
                        depth_frame,
                        intrinsics
                    )
                    detection_time = time.time() - detection_start

                    # Update embedding for next comparison
                    if self.enable_scene_change_detection:
                        self._last_frame_embedding = self._compute_frame_embedding(color_frame)
                        self._last_detection_objects = len(detected_objects)

                    # Update statistics
                    async with self._lock:
                        self.stats.total_frames += 1
                        self.stats.total_detections += len(detected_objects)
                        self.stats.last_detection_time = detection_time

                        # Running average
                        alpha = 0.1  # Smoothing factor
                        self.stats.avg_detection_time = (
                            alpha * detection_time +
                            (1 - alpha) * self.stats.avg_detection_time
                        )

                        # Update cache hit rate
                        total_attempted = self.stats.total_frames
                        self.stats.cache_hit_rate = self.stats.skipped_frames / total_attempted if total_attempted > 0 else 0.0

                    # Callback notification
                    if self.on_detection_complete:
                        if asyncio.iscoroutinefunction(self.on_detection_complete):
                            await self.on_detection_complete(len(detected_objects))
                        else:
                            self.on_detection_complete(len(detected_objects))

            except Exception as e:
                print(f"⚠ Tracking loop error: {e}")
                import traceback
                traceback.print_exc()

            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < self.update_interval:
                await asyncio.sleep(self.update_interval - elapsed)

        print("→ Tracking loop stopped")

    # Proxy methods to tracker's registry (thread-safe)

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """Get object by ID (thread-safe)."""
        return self.tracker.get_object(object_id)

    def get_all_objects(self):
        """Get all detected objects (thread-safe)."""
        return self.tracker.get_all_objects()

    def get_objects_by_type(self, object_type: str):
        """Get objects of specific type (thread-safe)."""
        return self.tracker.get_objects_by_type(object_type)

    def get_objects_with_affordance(self, affordance: str):
        """Get objects with specific affordance (thread-safe)."""
        return self.tracker.get_objects_with_affordance(affordance)

    def clear_registry(self):
        """Clear all detected objects (thread-safe)."""
        self.tracker.clear_registry()

    async def get_stats(self) -> TrackingStats:
        """Get current tracking statistics (async-safe)."""
        async with self._lock:
            return TrackingStats(
                total_frames=self.stats.total_frames,
                total_detections=self.stats.total_detections,
                skipped_frames=self.stats.skipped_frames,
                avg_detection_time=self.stats.avg_detection_time,
                last_detection_time=self.stats.last_detection_time,
                cache_hit_rate=self.stats.cache_hit_rate,
                is_running=self.stats.is_running
            )

    def is_running(self) -> bool:
        """Check if tracker is currently running."""
        return self._running
