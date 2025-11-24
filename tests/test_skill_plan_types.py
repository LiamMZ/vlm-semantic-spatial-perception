from src.primitives.skill_plan_types import (
    PRIMITIVE_LIBRARY,
    PrimitiveCall,
    compute_registry_hash,
)


def test_move_to_pose_validation_passes_with_position():
    call = PrimitiveCall(
        name="move_to_pose",
        parameters={"target_position": [0.1, 0.0, 0.2]},
        frame="base",
    )
    errors = call.validate(PRIMITIVE_LIBRARY["move_to_pose"])
    assert errors == []


def test_move_to_pose_validation_detects_missing_param():
    call = PrimitiveCall(name="move_to_pose", parameters={})
    errors = call.validate(PRIMITIVE_LIBRARY["move_to_pose"])
    assert any("missing required parameter" in msg for msg in errors)


def test_frame_is_extracted_from_parameters_on_deserialize():
    call = PrimitiveCall.from_dict(
        {
            "name": "move_to_pose",
            "parameters": {"frame": "camera", "target_position": [0, 0, 0]},
        }
    )
    assert call.frame == "camera"
    assert "frame" not in call.parameters


def test_registry_hash_is_stable_with_ordering():
    reg_a = {"objects": [{"object_id": "cup_1", "timestamp": 1.0}]}
    reg_b = {"objects": [{"timestamp": 1.0, "object_id": "cup_1"}]}
    assert compute_registry_hash(reg_a) == compute_registry_hash(reg_b)
