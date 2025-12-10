#!/usr/bin/env python3
"""
Visualize object detections from a perception snapshot.

Shows:
- RGB image with bounding boxes
- Object labels and IDs
- Interaction points (if available)
- Object properties and affordances
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


class SnapshotVisualizer:
    """Visualize object detections from perception snapshots."""

    def __init__(self, snapshot_path: Path):
        """
        Initialize visualizer with snapshot path.

        Args:
            snapshot_path: Path to snapshot directory
        """
        self.snapshot_path = Path(snapshot_path)
        if not self.snapshot_path.exists():
            raise ValueError(f"Snapshot path does not exist: {snapshot_path}")

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _load_image(self) -> Optional[Image.Image]:
        """Load RGB image from snapshot."""
        # Try multiple possible image names
        for image_name in ["color.png", "rgb.png", "image.png"]:
            image_path = self.snapshot_path / image_name
            if image_path.exists():
                return Image.open(image_path)
        return None

    def _load_detections(self) -> Dict[str, Any]:
        """Load detection data from snapshot."""
        detections_path = self.snapshot_path / "detections.json"
        if not detections_path.exists():
            raise ValueError(f"No detections.json found in {self.snapshot_path}")
        return self._load_json(detections_path)

    def _denormalize_bbox(self, bbox: List[int], img_width: int, img_height: int) -> tuple:
        """
        Convert normalized bounding box to pixel coordinates.

        Args:
            bbox: [ymin, xmin, ymax, xmax] normalized 0-1000
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            (x, y, width, height) in pixels
        """
        ymin, xmin, ymax, xmax = bbox

        # Convert from 0-1000 to 0-1
        ymin_norm = ymin / 1000.0
        xmin_norm = xmin / 1000.0
        ymax_norm = ymax / 1000.0
        xmax_norm = xmax / 1000.0

        # Convert to pixels
        x1 = int(xmin_norm * img_width)
        y1 = int(ymin_norm * img_height)
        x2 = int(xmax_norm * img_width)
        y2 = int(ymax_norm * img_height)

        width = x2 - x1
        height = y2 - y1

        return (x1, y1, width, height)

    def _denormalize_point(self, point: List[float], img_width: int, img_height: int) -> tuple:
        """
        Convert normalized point to pixel coordinates.

        Args:
            point: [x, y] normalized 0-1
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            (x, y) in pixels
        """
        x_norm, y_norm = point
        x_px = int(x_norm * img_width)
        y_px = int(y_norm * img_height)
        return (x_px, y_px)

    def visualize(self, show_interaction_points: bool = True, save_path: Optional[Path] = None):
        """
        Create visualization of object detections.

        Args:
            show_interaction_points: Whether to show interaction points
            save_path: Optional path to save figure
        """
        # Load data
        rgb_image = self._load_image()
        if rgb_image is None:
            print("⚠ No RGB image found in snapshot")
            return

        detections = self._load_detections()
        objects = detections.get("objects", [])

        if not objects:
            print("⚠ No objects in detections")
            return

        # Get image dimensions
        img_width, img_height = rgb_image.size

        # Create figure that exactly matches image size
        dpi = 100
        fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure

        # Display image with exact dimensions
        ax.imshow(rgb_image)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Flip Y axis to match image coordinates
        ax.axis('off')

        # Color palette for bounding boxes
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(objects), 20)))

        for idx, obj in enumerate(objects):
            color = colors[idx % len(colors)]

            # Get object data
            object_id = obj.get("object_id", "unknown")
            object_type = obj.get("object_type", "unknown")
            bbox = obj.get("bounding_box_2d", obj.get("bounding_box", None))
            interaction_points = obj.get("interaction_points", [])
            affordances = obj.get("affordances", [])
            properties = obj.get("properties", {})
            position_3d = obj.get("position_3d", None)
            position_2d = obj.get("position_2d", None)

            # Draw bounding box
            # Expected format: [ymin, xmin, ymax, xmax] normalized 0-1000
            # But VLM may return pixel coords, so we handle both cases
            if bbox and len(bbox) == 4:
                ymin, xmin, ymax, xmax = bbox

                # Check if coordinates are normalized (0-1000) or pixel coords
                # If max value > 1000, assume pixel coords; otherwise assume normalized
                max_coord = max(ymin, xmin, ymax, xmax)

                if max_coord <= 1000:
                    # Normalized 0-1000 coords - convert to pixels
                    ymin = int(ymin * img_height / 1000)
                    xmin = int(xmin * img_width / 1000)
                    ymax = int(ymax * img_height / 1000)
                    xmax = int(xmax * img_width / 1000)
                # else: already in pixel coordinates

                x = xmin
                y = ymin
                width = xmax - xmin
                height = ymax - ymin

                # Draw rectangle
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=4,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='-'
                )
                ax.add_patch(rect)

                # Add label with background
                label = f"{object_id}"
                ax.text(
                    x, y - 10,
                    label,
                    fontsize=11,
                    fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9, edgecolor='white', linewidth=1.5)
                )

            # Draw center position
            if show_interaction_points and position_2d:
                if isinstance(position_2d, (list, tuple)) and len(position_2d) == 2:
                    px, py = position_2d
                    # Draw center point
                    ax.plot(px, py, 'X', color=color, markersize=12,
                           markeredgecolor='white', markeredgewidth=2)

            # Draw interaction points (if dict format)
            if show_interaction_points and isinstance(interaction_points, dict):
                for point_type, point_data in interaction_points.items():
                    # Handle different formats
                    if isinstance(point_data, dict):
                        point = point_data.get("position_2d", point_data.get("point", None))
                    elif isinstance(point_data, (list, tuple)) and len(point_data) == 2:
                        point = point_data
                    else:
                        continue

                    if point and len(point) == 2:
                        px, py = point
                        # Draw interaction point
                        ax.plot(px, py, 'o', color=color, markersize=10,
                               markeredgecolor='white', markeredgewidth=2.5)

                        # Add small label
                        ax.text(
                            px + 15, py,
                            point_type[:3],
                            fontsize=9,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
                        )

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            print(f"✓ Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize object detections from a perception snapshot"
    )
    parser.add_argument(
        "snapshot_path",
        type=str,
        help="Path to snapshot directory"
    )
    parser.add_argument(
        "--no-interaction-points",
        action="store_true",
        help="Hide interaction points"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save visualization to file instead of displaying"
    )

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_path)
    visualizer = SnapshotVisualizer(snapshot_path)

    save_path = Path(args.save) if args.save else None
    visualizer.visualize(
        show_interaction_points=not args.no_interaction_points,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
