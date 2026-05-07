import numpy as np

from src.kinematics.xarm_pybullet_interface import XArmPybulletInterface
from src.kinematics.xarm_pybullet_planned_primitives import XArmPybulletPlannedPrimitives


class FakeRealXArm:
    def __init__(self, joints):
        self.joints = np.asarray(joints, dtype=float)
        self.commands = []
        self.gripper_calls = []

    def get_robot_joint_state(self):
        return self.joints.copy()

    def set_robot_joint_angles(self, joints, wait=True, speed=20.0):
        del wait, speed
        self.joints = np.asarray(joints, dtype=float)
        self.commands.append(self.joints.copy())
        return True

    def set_current_joint_state(self, joints):
        self.joints = np.asarray(joints, dtype=float)

    def open_gripper(self, **kwargs):
        self.gripper_calls.append(("open", kwargs))
        return True

    def close_gripper(self, **kwargs):
        self.gripper_calls.append(("close", kwargs))
        return True


def test_real_primitives_plan_with_pybullet_and_execute_on_real_interface():
    start = np.deg2rad([-8.1, -75.3, -24.9, 88.0, -7.6, 116.2, -34.9])
    real = FakeRealXArm(start)
    planner = XArmPybulletInterface(use_gui=False)
    primitives = XArmPybulletPlannedPrimitives(robot=real, planner=planner)

    target = [0.35, 0.0, 0.33]
    result = primitives.move_gripper_to_pose(
        target_position=target,
        preset_orientation="top_down",
        speed_factor=10.0,
        execute=True,
    )

    assert result["success"] is True
    assert result["executed"] is True
    assert result["trajectory_len"] == len(real.commands)
    assert not np.allclose(real.commands[0], real.commands[-1])
    assert np.allclose(real.joints, planner.get_robot_joint_state())
    planner.cleanup()


def test_real_primitives_forward_gripper_calls():
    real = FakeRealXArm([0.0] * 7)
    planner = XArmPybulletInterface(use_gui=False)
    primitives = XArmPybulletPlannedPrimitives(robot=real, planner=planner)

    assert primitives.open_gripper()["success"] is True
    assert primitives.close_gripper(simple_close=True)["success"] is True
    assert [name for name, _ in real.gripper_calls] == ["open", "close"]
    planner.cleanup()


def test_real_primitives_forward_pybullet_frame_transforms():
    real = FakeRealXArm([0.0] * 7)
    planner = XArmPybulletInterface(use_gui=False)
    primitives = XArmPybulletPlannedPrimitives(robot=real, planner=planner)

    cam_pos, cam_rot = primitives.get_camera_transform()
    base_pos, base_quat = primitives.convert_cam_pose_to_base(
        position=[0.0, 0.0, 0.1],
        orientation=[0.0, 0.0, 0.0, 1.0],
    )

    assert cam_pos is not None
    assert cam_rot is not None
    assert np.asarray(base_pos).shape == (3,)
    assert np.asarray(base_quat).shape == (4,)
    planner.cleanup()
