"""
Test script: load saved sim_demo state, decompose and execute a PDDL plan
in PyBullet using the AMP skill library.

Usage:
    python examples/test_skill_execution.py

Environment variables:
    GOOGLE_API_KEY  — Gemini API key for SkillDecomposer
    STATE_DIR       — override state directory (default: outputs/sim_demo/orchestrator_state)
    STEP_PAUSE      — seconds to pause between steps (default: 1.5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
STATE_DIR  = Path(os.getenv("STATE_DIR", "outputs/sim_demo/runs/latest/orchestrator_state"))
API_KEY    = os.getenv("GOOGLE_API_KEY", "")
STEP_PAUSE = float(os.getenv("STEP_PAUSE", "1.5"))

PLAN: List[str] = [
    "(pick red_block_1 table_1)",
    "(place red_block_1 blue_block_1)",
]

SCENE_OBJECTS = [
    {
        "object_id":   "red_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.3, 0.0, 0.04],
    },
    {
        "object_id":   "blue_block_1",
        "object_type": "block",
        "affordances": ["graspable", "stackable"],
        "position_3d": [0.5, 0.0, 0.04],
    },
    {
        "object_id":   "table_1",
        "object_type": "surface",
        "affordances": ["support_surface"],
        "position_3d": [0.4, 0.0, 0.0],
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
log = logging.getLogger("skill_exec_test")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _section(title: str) -> None:
    bar = "─" * (60 - len(title) - 2)
    print(f"\n── {title} {bar}")

def _ok(msg: str)   -> None: print(f"  ✓ {msg}")
def _info(k, v="")  -> None: print(f"    {k}: {v}")
def _warn(msg: str) -> None: print(f"  ⚠ {msg}")


# ── Main ─────────────────────────────────────────────────────────────────────
async def run() -> None:
    # ------------------------------------------------------------------
    # 1. Start PyBullet scene
    # ------------------------------------------------------------------
    _section("Starting PyBullet scene")
    from src.kinematics.sim import SceneEnvironment, CAMERA_AIM_JOINTS
    from src.kinematics.sim.scene_environment import OBJECT_COLORS

    env = SceneEnvironment()
    env.start()
    env.add_scene_objects(SCENE_OBJECTS)
    env.set_status("Loading state…")
    env.step(0.5)

    # ------------------------------------------------------------------
    # 2. Load saved registry
    # ------------------------------------------------------------------
    _section("Loading saved registry")
    from src.perception.object_registry import DetectedObjectRegistry, DetectedObject

    registry = DetectedObjectRegistry()
    registry_path = STATE_DIR / "registry.json"

    if registry_path.exists():
        registry.load_from_json(str(registry_path))
        _ok(f"Loaded {len(registry.get_all_objects())} objects from {registry_path}")
    else:
        _warn(f"Registry not found at {registry_path} — seeding from SCENE_OBJECTS")

    # Always ensure the sim objects are in the registry with known 3-D positions
    for sim_obj in SCENE_OBJECTS:
        if registry.get_object(sim_obj["object_id"]) is None:
            registry.add_object(DetectedObject(
                object_id=sim_obj["object_id"],
                object_type=sim_obj["object_type"],
                affordances=set(sim_obj.get("affordances", [])),
                position_3d=np.array(sim_obj["position_3d"]),
            ))
        else:
            # Update position_3d from sim so PyBulletPrimitives resolves correctly
            obj = registry.get_object(sim_obj["object_id"])
            obj.position_3d = np.array(sim_obj["position_3d"])

    for obj in registry.get_all_objects():
        _info(f"  {obj.object_id}", f"type={obj.object_type} pos={getattr(obj,'position_3d',None)}")

    # ------------------------------------------------------------------
    # 3. Build primitives + executor + decomposer
    # ------------------------------------------------------------------
    _section("Building primitives stack")
    from src.kinematics.sim.pybullet_primitives import PyBulletPrimitives
    from src.primitives.primitive_executor import PrimitiveExecutor
    from src.primitives.skill_decomposer import SkillDecomposer

    primitives = PyBulletPrimitives(env=env, registry=registry)
    executor   = PrimitiveExecutor(
        primitives=primitives,
        perception_pool_dir=STATE_DIR / "perception_pool",
    )
    decomposer = SkillDecomposer(
        api_key=API_KEY,
        orchestrator=None,  # no live orchestrator needed
    )
    _ok("PyBulletPrimitives / PrimitiveExecutor / SkillDecomposer ready")

    # Build world-state in the format SkillDecomposer._prepare_world_state expects:
    # registry must have an "objects" list (same shape as registry.json on disk)
    world_state: Dict[str, Any] = {
        "registry": {
            "objects": [
                {
                    "object_id":   obj.object_id,
                    "object_type": obj.object_type,
                    "affordances": list(obj.affordances),
                    "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None,
                }
                for obj in registry.get_all_objects()
            ]
        },
        "last_snapshot_id": None,
        "snapshot_index":   None,
        "robot_state":      env.get_robot_state(),
    }

    # ------------------------------------------------------------------
    # 4. Execute plan
    # ------------------------------------------------------------------
    _section(f"Executing {len(PLAN)}-step plan")
    env.set_status(f"Plan: {len(PLAN)} steps", color=[0.3, 0.8, 1.0])

    for i, step in enumerate(PLAN, 1):
        parts = step.strip("()").split()
        action_name = parts[0]
        pddl_args   = parts[1:]

        env.set_status(f"Step {i}/{len(PLAN)}: {step}", color=[0.3, 0.8, 1.0])
        _section(f"Step {i}: {step}")

        # Build parameter dict for SkillDecomposer
        param_keys = [f"param{j+1}" for j in range(len(pddl_args))]
        parameters: Dict[str, Any] = dict(zip(param_keys, pddl_args))
        parameters["object_ids"] = pddl_args

        _info("action",     action_name)
        _info("parameters", parameters)

        # Decompose
        try:
            skill_plan = decomposer.plan(
                action_name,
                parameters,
                world_hint=world_state,
            )
        except Exception as exc:
            _warn(f"Decomposition failed: {exc}")
            env.step(STEP_PAUSE)
            continue

        _ok(f"Decomposed → {len(skill_plan.primitives)} primitive(s):")
        for pc in skill_plan.primitives:
            _info(f"  {pc.name}", pc.parameters)

        if skill_plan.diagnostics.rationale:
            _info("rationale", skill_plan.diagnostics.rationale)

        # Execute
        try:
            result = executor.execute_plan(skill_plan, world_state=world_state)
            _ok(f"Executed (executed={result.executed})")
            for j, r in enumerate(result.primitive_results):
                _info(f"  result[{j}]", r)
        except Exception as exc:
            _warn(f"Execution failed: {exc}")

        env.step(STEP_PAUSE)

    env.set_status("✓ Done", color=[0.2, 0.9, 0.2])
    _section("Complete")
    env.step(3.0)
    env.stop()


if __name__ == "__main__":
    asyncio.run(run())
