# Kinematics Primitives Notes

## 2026-04-30

- Added `XArmPybulletPrimitives` in `src/kinematics/sim/xarm_pybullet_primitives.py`.
- This class mirrors CuRobo-style primitive entrypoints but plans through
  `XArmPybulletInterface` / `BasePybulletInterface.move_to_pose`.
- Implemented core methods used by higher-level planners:
  - `move_gripper_to_pose` (plus alias `move_gripper_To_pose`)
  - `push`, `pull`, `push_pull`
  - `pivot_pull`
  - `twist`
  - `open_gripper`, `close_gripper`, `retract_gripper`
- The primitive layer currently executes by writing the final planned joint state
  into the internal PyBullet model (no hardware command streaming).
- `XArmPybulletInterface` motor-holds the gripper mimic joints at an explicit
  open/closed target. PyBullet does not enforce mimic joints during GUI stepping,
  so one-time `resetJointState` calls allow the fingers to sag under gravity.
- `scripts/test_xarm_pybullet_primitives.py` passes `--move-replay-dt` into the
  first move-to-pose primitive so the GUI replays every waypoint from the current
  joint state at a visible speed. The replay forces an FK query after each
  `set_current_joint_state` because the interface applies joints lazily.
- Added `XArmPybulletPlannedPrimitives` for real xArm execution with PyBullet
  planning/transforms. It syncs the PyBullet planner from the real robot's
  current joints before every plan, uses PyBullet IK/FK and
  `convert_cam_pose_to_base`, then executes the planned joint waypoints through
  the real xArm interface.
- `scripts/test_xarm_real_pybullet_planned_primitives.py` mirrors the PyBullet
  smoke test for hardware. It wraps the raw xArm SDK in a small adapter, defaults
  to dry-run planning, and requires `--execute` before commanding the real robot.
- Real xArm PyBullet-planned primitives intentionally default to conservative
  initial hardware speeds: `speed_factor=0.25`, joint speed `0.12`, and joint
  acceleration `0.25` for direct SDK commands. Callers can override these once
  the workspace and trajectory behavior have been validated.
