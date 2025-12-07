"""
API-gated coverage for the SkillDecomposer LLM path using real fixture data from
tests/assets/continuous_pick_fixture. Calls SkillDecomposer.plan (real _call_llm wrapped
for recording) and PrimitiveExecutor.prepare_plan to translate helper params.

Artifacts: tests/artifacts/llm_pick/pick_plan_llm_response.json (raw LLM) and
tests/artifacts/llm_pick/pick_plan_llm_translated.json (executor-ready plan).

Assertions: plan contains at least one primitive, references the cloth object, has a graspable-like
interaction point, and includes helper parameters; artifacts are written.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primitives import PrimitiveExecutor, SkillDecomposer  # noqa: E402


pytestmark = pytest.mark.needs_api

FIXTURE_DIR = Path("tests/assets/continuous_pick_fixture")
LLM_ARTIFACTS_DIR = Path("tests/artifacts/llm_pick")
LLM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RESPONSE_PATH = LLM_ARTIFACTS_DIR / "pick_plan_llm_response.json"
TRANSLATED_PLAN_PATH = LLM_ARTIFACTS_DIR / "pick_plan_llm_translated.json"


def _load_world_state() -> Dict[str, Dict]:
    """Load registry/state/index from the fixture."""
    registry = json.loads((FIXTURE_DIR / "registry.json").read_text())
    state = json.loads((FIXTURE_DIR / "state.json").read_text())
    index = json.loads((FIXTURE_DIR / "perception_pool" / "index.json").read_text())
    return {
        "registry": registry,
        "last_snapshot_id": state["last_snapshot_id"],
        "snapshot_index": index,
        "robot_state": state.get("robot_state"),
    }


def _cloth_interaction(world_state: dict) -> dict:
    """Extract the cloth graspable interaction point from the latest snapshot detections."""
    snapshot_id = world_state["last_snapshot_id"]
    det_path = (
        FIXTURE_DIR / "perception_pool" / "snapshots" / snapshot_id / "detections.json"
    )
    detections = json.loads(det_path.read_text())
    for obj in detections.get("objects", []):
        if obj.get("object_id") == "black_fabric_garment":
            grasp = (obj.get("interaction_points") or {}).get("graspable") or {}
            pos2d = grasp.get("position_2d") or obj.get("position_2d")
            pos3d = grasp.get("position_3d") or obj.get("position_3d")
            return {
                "object_id": obj["object_id"],
                "snapshot_id": grasp.get("snapshot_id") or snapshot_id,
                "position_2d": pos2d,
                "position_3d": pos3d,
                "confidence": grasp.get("confidence", obj.get("confidence")),
                "reasoning": grasp.get("reasoning", ""),
            }
    raise AssertionError("black_fabric_garment not found in detections")


def _recording_llm_call(original_call):
    """Wrap the real LLM call to capture the raw response text to artifacts."""

    def _wrapper(self, *args, **kwargs):
        response_text = original_call(self, *args, **kwargs)
        LLM_RESPONSE_PATH.write_text(response_text)
        return response_text

    return _wrapper


def test_pick_plan_decomposes_with_real_llm(genai_api_key, caplog):
    """
    Source data: fixture registry/snapshots in tests/assets/continuous_pick_fixture; real Gemini call.
    Methods covered: SkillDecomposer.plan (real _call_llm wrapped for recording) and PrimitiveExecutor.prepare_plan.
    Artifacts emitted: tests/artifacts/llm_pick/pick_plan_llm_response.json (plan) and
    pick_plan_llm_translated.json.
    Assertions: plan includes cloth references and a graspable-like interaction point, helper params present,
    artifacts exist; also prepares translated plan without errors assertion beyond existence.
    """
    world_state = _load_world_state()
    cloth = _cloth_interaction(world_state)

    decomposer = SkillDecomposer(
        api_key=genai_api_key,
    )
    decomposer._perception_pool_dir = FIXTURE_DIR / "perception_pool"

    original_call = SkillDecomposer._call_llm
    with patch.object(SkillDecomposer, "_call_llm", _recording_llm_call(original_call)):
        plan = decomposer.plan(
            action_name="pick",
            parameters={"object_id": cloth["object_id"], "interaction": "graspable"},
            world_hint=world_state,
        )

    assert LLM_RESPONSE_PATH.exists()
    assert plan.action_name == "pick"
    assert len(plan.primitives) >= 1

    move_primitives = [p for p in plan.primitives if p.name.startswith("move_to_pose")]
    assert move_primitives, "expected at least one move_to_pose* primitive"
    cloth_move = next(
        (p for p in move_primitives if p.references.get("object_id") == cloth["object_id"]),
        None,
    )
    assert cloth_move, "expected a move primitive referencing the target cloth"
    ip_label = cloth_move.references.get("interaction_point", "")
    assert "graspable" in ip_label
    assert cloth_move.parameters.get("target_pixel_yx") or cloth_move.parameters.get("target_position")

    # Translate helper parameters to executor/xArm-friendly payload and persist.
    executor = PrimitiveExecutor(
        primitives=None,
        perception_pool_dir=FIXTURE_DIR / "perception_pool",
    )
    with caplog.at_level(logging.WARNING):
        executor.prepare_plan(plan, world_state)
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    TRANSLATED_PLAN_PATH.write_text(
        json.dumps(
            {
                "plan": plan.to_dict(),
            },
            indent=2,
        )
    )
    assert TRANSLATED_PLAN_PATH.exists()
    assert warnings == []
