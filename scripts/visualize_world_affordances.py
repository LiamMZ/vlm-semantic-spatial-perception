"""
Render affordance interaction points for every snapshot in a world directory.

Example:
  uv run python scripts/visualize_world_affordances.py --world tests/assets/20251201_161659
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.affordance_viz import visualize_snapshot_affordances  # noqa: E402

DEFAULT_WORLD = "tests/assets/20251201_161659"
DEFAULT_OUTPUT = None  # defaults to snapshot directory per image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        default=DEFAULT_WORLD,
        help=f"World directory containing perception_pool/snapshots (default: {DEFAULT_WORLD}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory for overlays (default: alongside each snapshot).",
    )
    args = parser.parse_args()

    world_dir = Path(args.world).expanduser()
    snapshots_dir = world_dir / "perception_pool" / "snapshots"
    if not snapshots_dir.exists():
        raise SystemExit(f"No snapshots directory found at {snapshots_dir}")

    rendered = 0
    output_dir = Path(args.output).expanduser() if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: list[Path] = []
    for snapshot_dir in sorted(p for p in snapshots_dir.iterdir() if p.is_dir()):
        if output_dir:
            output_path = output_dir / f"{snapshot_dir.name}_affordances.png"
        else:
            output_path = snapshot_dir / f"{snapshot_dir.name}_affordances.png"
        if visualize_snapshot_affordances(snapshot_dir, output_path):
            rendered += 1
            overlay_paths.append(output_path)
            print(f"Wrote {output_path}")

    if not overlay_paths:
        print("No affordances rendered; check snapshot detections.")
        return

    # Combine all overlay images into a single vertically stacked PNG.
    stacked_dir = world_dir / "perception_pool"
    stacked_dir.mkdir(parents=True, exist_ok=True)
    stacked_path = stacked_dir / "affordances_stacked.png"

    images = [Image.open(path).convert("RGB") for path in overlay_paths]
    stacked_width = max(img.width for img in images)
    stacked_height = sum(img.height for img in images)
    stacked_image = Image.new("RGB", (stacked_width, stacked_height))

    y_offset = 0
    for img in images:
        stacked_image.paste(img, (0, y_offset))
        y_offset += img.height

    stacked_image.save(stacked_path)
    print(f"Wrote stacked affordances to {stacked_path}")


if __name__ == "__main__":
    main()
