"""
Orchestrator Simulation Demo
=============================

Full end-to-end demo of the TaskOrchestrator with:
  • PyBullet GUI — xArm7 robot + coloured scene objects
  • Simulated wrist camera feeding the perception loop
  • Layered PDDL domain generation (L1–L5 gated pipeline)
  • Continuous object detection updating the PDDL domain
  • PDDL solver producing an executable plan
  • Terminal logging of every pipeline stage with timing

Perception backends (set USE_GSAM2=1 to use RAM+ + GroundingDINO + SAM2):
  • LLM (default) — Qwen3-VL or Google Gemini via GEMINI_API_KEY
  • GSAM2          — RAM+ tagger → GroundingDINO → SAM2 segmentation (no LLM needed)

Usage:
    # LLM backend (default):
    export GEMINI_API_KEY="..."
    uv run python examples/run_orchestrator_sim_demo.py

    # GSAM2 backend:
    USE_GSAM2=1 SAM2_CKPT=/path/to/sam2.1_hiera_large.pt \\
        RAM_CKPT=/path/to/ram_plus_swin_large_14m.pth \\
        uv run python examples/run_orchestrator_sim_demo.py

Optional flags (set as env vars):
    DEMO_TASK          Task description (default: "Put the red block on the blue block")
    DEMO_DETECT_CYCLES Max detection cycles before forcing solve (default: 3)
    DEMO_STEP_PAUSE    Seconds to pause at each milestone (default: 2)
    DEMO_OUTPUT_DIR    Output directory for PDDL files and state (default: outputs/sim_demo)

GSAM2-specific env vars (only used when USE_GSAM2=1):
    SAM2_CFG           SAM2 config file (default: configs/sam2.1/sam2.1_hiera_l.yaml)
    SAM2_CKPT          SAM2 checkpoint path (required)
    RAM_CKPT           RAM+ checkpoint path (required)
    GROUNDING_MODEL    GroundingDINO model ID (default: IDEA-Research/grounding-dino-tiny)
    GSAM2_DET_INTERVAL GroundingDINO re-detection interval in frames (default: 20)
    GSAM2_TAG_INTERVAL RAM+ re-tagging interval in frames (default: 30)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
_config_path = PROJECT_ROOT / "config"
if str(_config_path) not in sys.path:
    sys.path.insert(0, str(_config_path))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_YELLOW= "\033[33m"
_CYAN  = "\033[36m"
_RED   = "\033[31m"
_BLUE  = "\033[34m"
_GREY  = "\033[90m"
_MAG   = "\033[35m"


def _setup_logging() -> logging.Logger:
    class _Fmt(logging.Formatter):
        _MAP = {"DEBUG": _GREY, "INFO": _CYAN, "WARNING": _YELLOW,
                "ERROR": _RED, "CRITICAL": _RED + _BOLD}
        def format(self, r):
            color = self._MAP.get(r.levelname, "")
            ts = self.formatTime(r, "%H:%M:%S")
            name = r.name.split(".")[-1][:16]
            return f"{_GREY}{ts}{_RESET} {color}{r.levelname:<8}{_RESET} {_GREY}[{name}]{_RESET} {r.getMessage()}"

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Fmt())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    # Silence noisy sub-loggers
    for noisy in ("urllib3", "httpx", "httpcore", "google", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("sim_demo")


log = _setup_logging()


def _banner(msg: str) -> None:
    w = 72
    print()
    print(_BOLD + _BLUE + "=" * w + _RESET)
    print(_BOLD + _BLUE + f"  {msg}" + _RESET)
    print(_BOLD + _BLUE + "=" * w + _RESET)


def _section(msg: str) -> None:
    print()
    print(_BOLD + _CYAN + f"── {msg} " + "─" * max(0, 68 - len(msg)) + _RESET)


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _info(label: str, value: Any = "") -> None:
    print(f"    {_YELLOW}{label}:{_RESET} {value}")


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

TASK = os.getenv("DEMO_TASK", "Put the red block on the blue block")

# HuggingFace model — used by default. Set to empty string to fall back to Google GenAI.
# Override via env var: HF_MODEL="microsoft/Phi-3.5-vision-instruct"
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-VL-4B")

# GSAM2 perception backend (set USE_GSAM2=1 to enable)
USE_GSAM2 = os.getenv("USE_GSAM2", "").strip() in ("1", "true", "yes")
GSAM2_SAM2_CFG        = os.getenv("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
GSAM2_SAM2_CKPT       = os.getenv("SAM2_CKPT", "")
GSAM2_RAM_CKPT        = os.getenv("RAM_CKPT", "")
GSAM2_GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
GSAM2_DET_INTERVAL    = int(os.getenv("GSAM2_DET_INTERVAL", "20"))
GSAM2_TAG_INTERVAL    = int(os.getenv("GSAM2_TAG_INTERVAL", "30"))

SCENE_OBJECTS = [
    {
        "object_id":   "red_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.3, 0.0, 0.12],
    },
    {
        "object_id":   "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.5, 0.0, 0.12],
    },
    {
        "object_id":   "table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.4, 0.0, 0.0],
    },
]

MAX_DETECT_CYCLES  = int(os.getenv("DEMO_DETECT_CYCLES", "3"))
STEP_PAUSE         = float(os.getenv("DEMO_STEP_PAUSE", "2"))
OUTPUT_DIR         = Path(os.getenv("DEMO_OUTPUT_DIR", "outputs/sim_demo"))


# ---------------------------------------------------------------------------
# Simulated frame provider
# ---------------------------------------------------------------------------

class SimFrameProvider:
    """
    Wraps a SceneEnvironment and provides (color, depth, intrinsics, robot_state)
    tuples for the ContinuousObjectTracker frame provider slot.

    Also exposes robot_state via the XArmPybulletInterface so the orchestrator
    can snapshot joint angles alongside each detection.
    """

    def __init__(self, env, robot) -> None:
        self._env   = env    # SceneEnvironment (GUI)
        self._robot = robot  # XArmPybulletInterface (DIRECT, for FK)

    def __call__(self) -> Tuple:
        color, depth, intrinsics = self._env.capture_camera_frame()
        robot_state = self._robot.get_robot_state() if self._robot is not None else None
        if color is None:
            return None, None, None, None
        return color, depth, intrinsics, robot_state


# ---------------------------------------------------------------------------
# Detection progress tracker (callback)
# ---------------------------------------------------------------------------

class DetectionTracker:
    """Receives detection callbacks from the orchestrator and tracks progress."""

    def __init__(self, env, min_cycles: int) -> None:
        self._env       = env
        self._min_cycles = min_cycles
        self.cycle      = 0
        self.object_ids: List[str] = []
        self._ready     = asyncio.Event()

    def on_detection(self, object_count: int) -> None:
        self.cycle += 1
        log.info("Detection cycle %d — %d objects observed", self.cycle, object_count)
        self._env.set_status(
            f"Detection {self.cycle}/{self._min_cycles}: {object_count} objects",
            color=[0.3, 0.8, 1.0],
        )
        if self.cycle >= self._min_cycles:
            self._ready.set()

    async def wait_until_ready(self, timeout: float = 120.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run_demo() -> int:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    use_hf = bool(HF_MODEL) and not USE_GSAM2  # HF is default unless GSAM2 is selected

    _banner("Orchestrator Simulation Demo")
    _info("Task", TASK)
    _info("Scene", [o["object_id"] for o in SCENE_OBJECTS])
    _info("Output", OUTPUT_DIR)
    if USE_GSAM2:
        _info("Perception", f"GSAM2 (RAM+ + GroundingDINO + SAM2)")
        _info("SAM2 ckpt", GSAM2_SAM2_CKPT or "(not set — will fail)")
        _info("RAM+ ckpt", GSAM2_RAM_CKPT or "(not set — will fail)")
    else:
        _info("Perception", f"HuggingFace ({HF_MODEL})" if use_hf else "Google GenAI (gemini-2.0-flash)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Build LLM client (skipped for GSAM2 backend)
    # ------------------------------------------------------------------
    llm_client = None

    if not USE_GSAM2:
        from src.llm_interface import GoogleGenAIClient, Qwen3VLClient

        if use_hf:
            _section("Loading Qwen3-VL model")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _info("Model", HF_MODEL)
            _info("Device", device)
            llm_client = Qwen3VLClient(
                model=HF_MODEL,
                device="auto" if device == "cuda" else device,
                load_in_4bit=device == "cuda",
                model_kwargs={"max_memory": {0: "6GiB", "cpu": "16GiB"}} if device == "cuda" else None,
            )
            _ok(f"Qwen3-VL model loaded on {device}")
        else:
            if not api_key:
                log.error("GEMINI_API_KEY is not set. Set HF_MODEL to use a local model or USE_GSAM2=1 for GSAM2.")
                return 1
            llm_client = GoogleGenAIClient(model="gemini-2.0-flash", api_key=api_key)
            _ok("Google GenAI client ready")

    # ------------------------------------------------------------------
    # 1. Start PyBullet GUI environment
    # ------------------------------------------------------------------
    _section("Starting PyBullet simulation environment")
    from src.kinematics.sim import SceneEnvironment, CAMERA_AIM_JOINTS

    env = SceneEnvironment()
    env.start()
    env.add_scene_objects(SCENE_OBJECTS)
    env.set_status("Initialising orchestrator…")
    env.step(0.5)
    _ok("PyBullet GUI started — xArm7 aimed at work surface")

    # SceneEnvironment acts as both the visualiser and the robot state provider.
    # No second PyBullet instance is created.
    frame_provider = SimFrameProvider(env, env)

    # Verify we can capture a frame before handing to orchestrator
    color, depth, intrinsics, _ = frame_provider()
    assert color is not None, "Sim camera returned no frame — check URDF camera link"
    _ok(f"Sim camera verified: {color.shape[1]}×{color.shape[0]} RGB, depth [{depth.min():.2f}, {depth.max():.2f}]m")

    # Save a preview of the initial camera view
    preview_path = OUTPUT_DIR / "initial_camera_view.png"
    try:
        from PIL import Image
        Image.fromarray(color).save(preview_path)
        _info("Camera preview saved", preview_path)
    except Exception:
        pass

    env.step(0.3)

    # ------------------------------------------------------------------
    # 2. Build OrchestratorConfig
    # ------------------------------------------------------------------
    _section("Configuring TaskOrchestrator")
    from orchestrator_config import OrchestratorConfig
    from src.planning.task_orchestrator import TaskOrchestrator

    detect_tracker = DetectionTracker(env, min_cycles=MAX_DETECT_CYCLES)

    config = OrchestratorConfig(
        api_key=api_key or "unused",   # required field; ignored when llm_client is set
        llm_client=llm_client,

        # Sim camera — skip RealSense init entirely
        use_sim_camera=True,
        enable_depth=True,

        # Detection
        update_interval=4.0,          # seconds between VLM calls (sim is slow to render)
        min_observations=MAX_DETECT_CYCLES,
        fast_mode=True,               # skip interaction-point refinement for speed

        # Persistence
        state_dir=OUTPUT_DIR / "orchestrator_state",
        auto_save=True,
        auto_save_on_detection=True,
        auto_save_on_state_change=True,

        # Snapshots
        enable_snapshots=True,
        snapshot_every_n_detections=1,
        perception_pool_dir=OUTPUT_DIR / "perception_pool",

        # PDDL solver
        solver_backend="auto",
        solver_timeout=60.0,
        solver_verbose=False,
        auto_solve_when_ready=False,   # we drive this manually
        max_refinement_attempts=3,
        auto_refine_on_failure=True,

        # Layered generation — the whole point of this demo
        use_layered_generation=True,
        dkb_dir=OUTPUT_DIR / "dkb",

        # SceneEnvironment implements the duck-typed robot provider interface
        robot=env,

        # Callbacks
        on_state_change=lambda old, new: (
            log.info("Orchestrator state: %s → %s", old.value, new.value),
            env.set_status(f"State: {new.value}", color=[0.9, 0.7, 0.2]),
        ),
        on_detection_update=detect_tracker.on_detection,
        on_plan_generated=lambda result: (
            _ok(f"Plan generated! {result.plan_length} steps: {result.plan}"),
            env.set_status(f"Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2]),
        ),
    )

    backend_label = "GSAM2" if USE_GSAM2 else ("HuggingFace" if use_hf else "GenAI")
    _ok(f"Config built — use_layered_generation=True, backend={backend_label}, solver=auto")
    _info("state_dir", config.state_dir)
    _info("dkb_dir",   config.dkb_dir)

    # ------------------------------------------------------------------
    # 3. Initialise orchestrator — no physical camera
    # ------------------------------------------------------------------
    _section("Initialising orchestrator subsystems")
    env.set_status("Initialising…")

    # Pass no camera — we'll inject frames via set_frame_provider after init
    orchestrator = TaskOrchestrator(config=config, camera=None)
    await orchestrator.initialize()
    _ok("Orchestrator initialised")

    # ------------------------------------------------------------------
    # 3a. Optionally swap to GSAM2 perception backend
    # ------------------------------------------------------------------
    if USE_GSAM2:
        _section("Swapping to GSAM2 perception backend")
        if not GSAM2_SAM2_CKPT:
            log.error("SAM2_CKPT env var is required when USE_GSAM2=1")
            return 1

        import torch
        from src.perception import GSAM2ContinuousObjectTracker

        gsam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        gsam2_tracker = GSAM2ContinuousObjectTracker(
            sam2_model_cfg=GSAM2_SAM2_CFG,
            sam2_ckpt_path=GSAM2_SAM2_CKPT,
            grounding_model_id=GSAM2_GROUNDING_MODEL,
            ram_ckpt_path=GSAM2_RAM_CKPT or None,
            detection_interval=GSAM2_DET_INTERVAL,
            device=gsam2_device,
            tag_interval=GSAM2_TAG_INTERVAL,
            update_interval=config.update_interval,
            on_detection_complete=detect_tracker.on_detection,
            logger=log.getChild("GSAM2Tracker"),
        )
        # Replace the LLM-based tracker with the GSAM2 tracker
        orchestrator.tracker = gsam2_tracker
        _ok(f"GSAM2 tracker active (SAM2 + {'RAM+' if GSAM2_RAM_CKPT else 'no tagger'}) on {gsam2_device}")

    # Override frame provider with our sim camera
    orchestrator.tracker.set_frame_provider(frame_provider)
    _ok("Sim frame provider injected into tracker")

    env.step(0.3)

    # ------------------------------------------------------------------
    # 4. Quick perception pass — identify scene objects before planning
    # ------------------------------------------------------------------
    _section("Initial perception pass")
    env.set_status("Detecting scene objects…", color=[0.8, 0.6, 0.2])
    orchestrator.current_task = TASK  # needed by tracker before domain generation
    await orchestrator.start_detection()
    ready = await detect_tracker.wait_until_ready(timeout=60.0)
    await orchestrator.stop_detection()
    pre_detected = orchestrator.get_detected_objects()
    if pre_detected:
        _ok(f"Pre-detected {len(pre_detected)} object(s): {[o.object_id for o in pre_detected]}")
    else:
        _warn("No objects detected in pre-pass — domain generation will use empty scene")

    # ------------------------------------------------------------------
    # 5. Process task request (L1→L5 layered domain generation)
    # ------------------------------------------------------------------
    _section("Processing task request — layered domain generation")
    env.set_status("L1–L5 domain generation…", color=[0.3, 0.8, 1.0])
    log.info("Task: %s", TASK)

    t0 = time.monotonic()
    # Pass the initial sim camera frame as environment context
    task_analysis = await orchestrator.process_task_request(
        TASK,
        environment_image=color,
    )
    dt = time.monotonic() - t0

    _ok(f"Domain generated in {dt:.1f}s")
    _info("goal_predicates",    task_analysis.goal_predicates)
    _info("goal_objects",       task_analysis.goal_objects)
    _info("required_actions",   [a.get("name") for a in task_analysis.required_actions])
    _info("relevant_predicates",task_analysis.relevant_predicates)

    env.highlight_objects([o["object_id"] for o in SCENE_OBJECTS
                           if any(g in " ".join(task_analysis.goal_predicates)
                                  for g in [o["object_id"]])])
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 6. Start continuous detection — sim camera feeds VLM
    # ------------------------------------------------------------------
    _section("Starting continuous VLM detection")
    env.set_status("Detecting objects…", color=[0.3, 0.8, 1.0])
    await orchestrator.start_detection()
    _ok(f"Detection started — waiting for {MAX_DETECT_CYCLES} cycles")

    ready = await detect_tracker.wait_until_ready(timeout=MAX_DETECT_CYCLES * 30.0)
    if not ready:
        _warn("Detection timeout — proceeding with whatever was observed")

    await orchestrator.stop_detection()

    detected = orchestrator.get_detected_objects()
    _ok(f"Detection complete — {len(detected)} objects in registry")
    for obj in detected:
        _info(f"  {obj.object_id}", f"type={obj.object_type} affordances={obj.affordances} pos3d={getattr(obj,'position_3d',None)}")

    # Highlight everything detected
    env.highlight_objects([o.object_id for o in detected if o.object_id in {s["object_id"] for s in SCENE_OBJECTS}])
    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 6. Check orchestrator status
    # ------------------------------------------------------------------
    _section("Orchestrator status")
    status = await orchestrator.get_status()
    _info("state",             status.get("orchestrator_state"))
    _info("detection_count",   status.get("detection_count"))
    _info("ready_for_planning",status.get("ready_for_planning"))
    monitor = status.get("monitor", {})
    if monitor:
        _info("task_state_decision", getattr(monitor, "state", monitor))

    # ------------------------------------------------------------------
    # 7. Generate PDDL files
    # ------------------------------------------------------------------
    _section("Generating PDDL domain & problem files")
    env.set_status("Generating PDDL…", color=[0.9, 0.7, 0.2])

    pddl_paths = await orchestrator.generate_pddl_files(
        output_dir=OUTPUT_DIR / "pddl",
        set_goals=True,
    )
    domain_path  = pddl_paths.get("domain_path")
    problem_path = pddl_paths.get("problem_path")
    _ok(f"Domain  → {domain_path}")
    _ok(f"Problem → {problem_path}")

    if domain_path and Path(domain_path).exists():
        _section("Generated PDDL Domain")
        print(Path(domain_path).read_text())
    if problem_path and Path(problem_path).exists():
        _section("Generated PDDL Problem")
        print(Path(problem_path).read_text())

    env.step(STEP_PAUSE)

    # ------------------------------------------------------------------
    # 8. Solve — with automatic refinement on failure
    # ------------------------------------------------------------------
    _section("Solving PDDL plan (with refinement)")
    env.set_status("Solving…", color=[0.9, 0.7, 0.2])

    t0 = time.monotonic()
    result = await orchestrator.solve_and_plan_with_refinement(
        output_dir=OUTPUT_DIR / "pddl",
        wait_for_objects=False,   # objects already detected above
    )
    dt = time.monotonic() - t0

    _section("Solver Result")
    _info("success",      result.success)
    _info("plan_length",  result.plan_length)
    _info("search_time",  f"{result.search_time:.2f}s" if result.search_time else "n/a")
    _info("solve_total",  f"{dt:.2f}s")

    if result.success:
        _ok("Plan found!")
        for i, step in enumerate(result.plan, 1):
            _info(f"  step {i}", step)
        env.set_status(f"✓ Plan: {result.plan_length} steps", color=[0.2, 0.9, 0.2])
    else:
        _warn(f"No plan found: {result.error_message}")
        env.set_status("✗ No plan found", color=[1.0, 0.3, 0.3])

    env.step(STEP_PAUSE * 1.5)

    # ------------------------------------------------------------------
    # 9. Save state snapshot
    # ------------------------------------------------------------------
    _section("Saving orchestrator state")
    saved_path = await orchestrator.save_state()
    _ok(f"State saved → {saved_path}")

    # ------------------------------------------------------------------
    # 10. Shutdown
    # ------------------------------------------------------------------
    _section("Shutting down")
    await orchestrator.shutdown()
    env.step(1.0)
    env.stop()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _banner("Demo Complete")
    _info("Task",         TASK)
    _info("Plan success", result.success)
    if result.success:
        _info("Plan length", result.plan_length)
        _info("Plan",        result.plan)
    _info("Output dir",  OUTPUT_DIR)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_demo()))
