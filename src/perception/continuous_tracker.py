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

from .object_tracker import ObjectTracker
from .object_registry import DetectedObject


@dataclass
class TrackingStats:
    """Statistics for the continuous tracker."""
    total_frames: int = 0
    total_detections: int = 0
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
        on_detection_complete: Optional[Callable[[int], None]] = None
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

        # Async control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Frame source
        self._frame_provider: Optional[Callable[[], tuple]] = None

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

    async def _tracking_loop(self):
        """Main tracking loop (runs as async task)."""
        print("→ Tracking loop started")

        while self._running:
            loop_start = time.time()

            try:
                # Get current frame
                color_frame, depth_frame, intrinsics = self._frame_provider()

                # Run async detection
                detection_start = time.time()
                detected_objects = await self.tracker.detect_objects(
                    color_frame,
                    depth_frame,
                    intrinsics
                )
                detection_time = time.time() - detection_start

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
                avg_detection_time=self.stats.avg_detection_time,
                last_detection_time=self.stats.last_detection_time,
                cache_hit_rate=self.stats.cache_hit_rate,
                is_running=self.stats.is_running
            )

    def is_running(self) -> bool:
        """Check if tracker is currently running."""
        return self._running
