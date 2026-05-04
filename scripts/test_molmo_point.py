"""
Smoke test for MolmoPointDetector.

Loads Molmo2-4B, runs a pointing query on either a supplied image file or a
live RealSense frame, and prints the resulting interaction points together with
the raw model output — mirroring the feedback visible in the main pipeline logs.

Run:
    # live camera, default actions
    uv run scripts/test_molmo_point.py

    # static image (no camera required)
    uv run scripts/test_molmo_point.py --image path/to/image.jpg

    # custom prompt (matches skill-decomposer guidance strings)
    uv run scripts/test_molmo_point.py --image img.jpg \\
        --prompt "the widest stable top edge of the cardboard box on the side facing the robot"

    # specific action key + object type
    uv run scripts/test_molmo_point.py --image img.jpg --object-type mug --actions pick place

    # headless
    uv run scripts/test_molmo_point.py --no-gui --image img.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ion()

from src.perception.molmo_point_detector import MolmoPointDetector, DEFAULT_ACTIONS


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_GREY   = "\033[90m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET}  {msg}")
def _info(msg: str) -> None: print(f"  {_YELLOW}•{_RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET}  {msg}")
def _section(msg: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {msg} " + "─" * max(0, 68 - len(msg)) + _RESET)


def _setup_logging() -> None:
    """Route all WARNING+ logs (including Molmo's raw-output warnings) to stdout."""
    class _Fmt(logging.Formatter):
        _MAP = {"DEBUG": _GREY, "INFO": _CYAN, "WARNING": _YELLOW, "ERROR": _RED}
        def format(self, r: logging.LogRecord) -> str:
            color = self._MAP.get(r.levelname, "")
            ts = self.formatTime(r, "%H:%M:%S")
            name = r.name.split(".")[-1][:20]
            return f"{_GREY}{ts}{_RESET} {color}{r.levelname:<8}{_RESET} [{name}] {r.getMessage()}"

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Image / camera helpers
# ---------------------------------------------------------------------------

def _load_image_file(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def _capture_realsense():
    from src.camera.realsense_camera import RealSenseCamera
    _info("Opening RealSense camera…")
    cam = RealSenseCamera(width=640, height=480, fps=30, enable_depth=True)
    time.sleep(0.5)
    color, depth = cam.get_aligned_frames()
    intrinsics = cam.get_camera_intrinsics()
    cam.stop()
    _ok(f"Captured frame {color.shape}  depth [{depth.min():.3f}, {depth.max():.3f}] m")
    return color, depth, intrinsics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _show(color: np.ndarray, ips: dict, title: str, gui: bool) -> None:
    if not gui:
        return

    h, w = color.shape[:2]
    cmap = plt.get_cmap("Set2")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle(f"MolmoPointDetector — {title}", fontsize=13, fontweight="bold")
    ax.imshow(color)
    ax.axis("off")

    legend_handles = []
    for idx, (action, ip) in enumerate(ips.items()):
        c = cmap(idx % 8)
        norm = ip.position_2d
        if norm is not None and len(norm) >= 2:
            # position_2d is normalized [y, x] in 0-1000 scale
            px = float(norm[1]) * w / 1000.0
            py = float(norm[0]) * h / 1000.0
            ax.plot(px, py, marker="+", markersize=18, color=c, markeredgewidth=3, zorder=10)
            ax.plot(px, py, marker="o", markersize=8,  color=c, alpha=0.55, zorder=9)
            label = f"{action}"
            if ip.position_3d is not None:
                label += f"\n{np.round(ip.position_3d, 3).tolist()}"
            ax.text(px + 8, py - 8, label, color=c, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))
            # Alternative points
            for alt in (ip.alternative_points or []):
                ap2d = alt.get("position_2d")
                if ap2d and len(ap2d) >= 2:
                    apx = float(ap2d[1]) * w / 1000.0
                    apy = float(ap2d[0]) * h / 1000.0
                    ax.plot(apx, apy, marker="x", markersize=10, color=c,
                            markeredgewidth=2, alpha=0.5, zorder=8)
        legend_handles.append(mpatches.Patch(color=c, label=action))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
                  title="Actions", title_fontsize=9)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    try:
        input("\n  Press Enter to exit…\n")
    except EOFError:
        pass
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for MolmoPointDetector")
    parser.add_argument("--image", metavar="PATH",
                        help="Path to an RGB image file (skips RealSense capture)")
    parser.add_argument("--object-type", default="object",
                        help="Object type label sent to Molmo (default: 'object')")
    parser.add_argument("--object-id", default=None,
                        help="Object ID string (default: <object-type>_0)")
    parser.add_argument("--actions", nargs="+", default=None,
                        help="Action keys to query (default: all DEFAULT_ACTIONS)")
    parser.add_argument("--prompt", metavar="TEXT", default=None,
                        help="Custom pointing prompt — sent as a custom_prompts entry "
                             "under key '_guided', matching the skill-decomposer path")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("Y1", "X1", "Y2", "X2"),
                        help="Bounding box [y1 x1 y2 x2] in 0-1000 normalized scale")
    parser.add_argument("--checkpoint", default="allenai/Molmo2-4B",
                        help="HuggingFace checkpoint or local path")
    parser.add_argument("--no-gui", action="store_true", help="Skip visualisation")
    args = parser.parse_args()

    _setup_logging()

    gui = not args.no_gui
    object_id = args.object_id or f"{args.object_type}_0"

    # Build actions set and custom_prompts — mirror the skill-decomposer call
    if args.prompt:
        actions = {"_guided"}
        custom_prompts = {"_guided": args.prompt}
    else:
        actions = set(args.actions) if args.actions else DEFAULT_ACTIONS
        custom_prompts = {}

    print("=" * 68)
    print(f"  {_BOLD}Molmo2-4B smoke test{_RESET}")
    print("=" * 68)
    _info(f"Object type  : {args.object_type}")
    _info(f"Object ID    : {object_id}")
    _info(f"Actions      : {sorted(actions)}")
    if custom_prompts:
        for k, v in custom_prompts.items():
            _info(f"Custom prompt [{k}]: {v!r}")
    _info(f"Checkpoint   : {args.checkpoint}")
    _info(f"BBox (0-1000): {args.bbox}")

    # --- acquire image -------------------------------------------------------
    _section("Acquiring image")
    depth: Optional[np.ndarray] = None
    intrinsics = None

    if args.image:
        _info(f"Loading {args.image}")
        color = _load_image_file(args.image)
        _ok(f"Image shape: {color.shape}")
    else:
        color, depth, intrinsics = _capture_realsense()

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            free = _torch.cuda.mem_get_info()[0] / 1e9
            _info(f"CUDA free before model load: {free:.2f} GB")
    except Exception:
        pass

    # --- run detector --------------------------------------------------------
    _section("Loading MolmoPointDetector")
    _info("Instantiating…")
    t0 = time.time()
    detector = MolmoPointDetector(checkpoint=args.checkpoint)
    _ok(f"Model loaded in {time.time() - t0:.1f}s")

    _section("Running get_interaction_points")
    _info(f"Querying {len(actions)} action(s)…")
    t0 = time.time()
    try:
        ips = detector.get_interaction_points(
            rgb_image=color,
            depth_frame=depth,
            camera_intrinsics=intrinsics,
            object_id=object_id,
            object_type=args.object_type,
            bounding_box_2d=args.bbox,
            actions=actions,
            custom_prompts=custom_prompts,
        )
        elapsed = time.time() - t0
        _ok(f"Completed in {elapsed:.1f}s — {len(ips)} / {len(actions)} action(s) returned points")
    except Exception as exc:
        _fail(f"get_interaction_points raised: {exc}")
        raise

    # --- print results -------------------------------------------------------
    _section("Results")
    if not ips:
        _warn("No interaction points returned for any action")
        _warn("Check the WARNING logs above for the raw Molmo output and prompt")
    else:
        for action, ip in ips.items():
            p2d  = ip.position_2d
            p3d  = ip.position_3d
            ori  = ip.approach_orientation
            alts = len(ip.alternative_points) if ip.alternative_points else 0
            _ok(f"{action:14s}  2d={p2d}  3d={p3d}  orient={ori}  alt_pts={alts}")

    # --- visualise -----------------------------------------------------------
    title = args.object_type
    if args.prompt:
        title += f" | {args.prompt[:50]}…" if len(args.prompt) > 50 else f" | {args.prompt}"
    _show(color, ips, title, gui)

    print()
    print("  PASS" if ips else "  WARN — no points returned (see WARNING logs above)")


if __name__ == "__main__":
    main()
