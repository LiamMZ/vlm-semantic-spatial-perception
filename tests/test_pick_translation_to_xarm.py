"""
Integration-ish regression that uses fixture data from tests/assets/continuous_pick_fixture,
stubs the LLM response, and exercises SkillDecomposer.plan + PrimitiveExecutor.prepare_plan.

Generates artifacts in tests/artifacts/translation_pick/ (prepare_plan + call list)

Asserts the translated primitives include target_position and expected method ordering.
"""

import json
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primitives import PrimitiveExecutor, SkillDecomposer  # noqa: E402
from src.perception.utils.coordinates import compute_3d_position, pixel_to_normalized  # noqa: E402


FIXTURE_DIR = Path("tests/assets/continuous_pick_fixture")
TRANSLATION_ARTIFACTS_DIR = Path("tests/artifacts/translation_pick")
TRANSLATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_world_state() -> Dict[str, Dict]:
    """Load registry/state/index from the fixture."""
    registry = json.loads((FIXTURE_DIR / "registry.json").read_text())
    state = json.loads((FIXTURE_DIR / "state.json").read_text())
    index = json.loads((FIXTURE_DIR / "perception_pool" / "index.json").read_text())
    world_state = {
        "registry": registry,
        "last_snapshot_id": state["last_snapshot_id"],
        "snapshot_index": index,
        "robot_state": state.get("robot_state"),
    }
    return world_state


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


def _mock_llm_response(world_state: dict) -> str:
    """LLM stub grounded in fixture data; returns cloth pick plan JSON."""
    cloth = _cloth_interaction(world_state)
    # position_2d is stored as normalized [y, x]; executor helper expects the same
    target_pixel_yx = cloth["position_2d"]
    return json.dumps(
        {
            "primitives": [
                {
                    "name": "move_to_pose_with_preparation",
                    "frame": "camera",
                    "parameters": {
                        "target_pixel_yx": target_pixel_yx,
                        "depth_offset_m": 0.0,
                        "tcp_standoff_m": 0.04,
                    },
                    "references": {
                        "object_id": cloth["object_id"],
                        "interaction_point": "graspable",
                    },
                    "metadata": {"speed_factor": 0.35, "retries": 0},
                },
                {
                    "name": "close_gripper",
                    "frame": "base",
                    "parameters": {},
                    "references": {"object_id": cloth["object_id"]},
                    "metadata": {"simple_close": True},
                },
                {
                    "name": "retract_gripper",
                    "frame": "base",
                    "parameters": {"distance": 0.08},
                    "references": {"object_id": cloth["object_id"]},
                    "metadata": {"speed_factor": 0.5},
                },
            ],
            "assumptions": ["Cloth remains reachable and not entangled."],
            "diagnostics": {
                "rationale": "Approach the folded black cloth, close gripper, back away.",
                "freshness_notes": [],
                "warnings": [],
            },
            "interaction_points": [],
            "resolved_interaction_point": cloth,
        }
    )


@pytest.mark.parametrize(
    "action_parameters",
    [
        ({"object_id": "black_fabric_garment", "interaction": "graspable"}),
    ],
)
def test_pick_plan_translates_to_xarm_calls(action_parameters):
    """
    Source data: fixture registry/snapshots in tests/assets/continuous_pick_fixture and a mocked LLM JSON.
    Methods covered: SkillDecomposer.plan (stubbed _call_llm) and PrimitiveExecutor.prepare_plan translation.
    Artifacts emitted: tests/artifacts/translation_pick/pick_plan_prepare_plan.json and pick_plan_calls.json.
    Assertions: translated move_to_pose_with_preparation has target_position; method order is move/close/retract;
    all references point to the black cloth; no prepare_plan errors and warnings empty.
    """
    world_state = _load_world_state()

    decomposer = SkillDecomposer(
        api_key="test",
    )
    decomposer._perception_pool_dir = FIXTURE_DIR / "perception_pool"

    with patch.object(SkillDecomposer, "_call_llm", return_value=_mock_llm_response(world_state)):
        plan = decomposer.plan(
            action_name="pick",
            parameters=action_parameters,
            world_hint=world_state,
        )

    executor = PrimitiveExecutor(
        primitives=None,
        perception_pool_dir=FIXTURE_DIR / "perception_pool",
    )

    _, warnings, errors = executor.prepare_plan(plan, world_state)
    prepare_output_path = TRANSLATION_ARTIFACTS_DIR / "pick_plan_prepare_plan.json"
    prepare_output_path.write_text(
        json.dumps(
            {
                "plan": plan.to_dict(),
                "warnings": warnings,
                "errors": errors,
            },
            indent=2,
        )
    )
    assert errors == []

    curobo_calls = [
        {
            "method": primitive.name,
            "frame": primitive.frame,
            "parameters": primitive.parameters,
        }
        for primitive in plan.primitives
    ]

    output_path = TRANSLATION_ARTIFACTS_DIR / "pick_plan_calls.json"
    output_path.write_text(json.dumps(curobo_calls, indent=2))

    assert output_path.exists()
    logged = json.loads(output_path.read_text())

    assert logged[0]["method"] == "move_to_pose_with_preparation"
    assert "target_position" in logged[0]["parameters"]
    assert len(logged[0]["parameters"]["target_position"]) == 3
    # Target position should align with the resolved interaction point 3D from detections
    cloth = _cloth_interaction(world_state)
    assert logged[0]["parameters"]["target_position"] == pytest.approx(
        cloth["position_3d"], rel=1e-5, abs=1e-5
    )
    assert all(
        primitive.references.get("object_id") == "black_fabric_garment"
        for primitive in plan.primitives
    )
    assert logged[1]["method"] == "close_gripper"
    assert logged[2]["method"] == "retract_gripper"
    assert warnings == []
