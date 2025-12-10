# xArm CuRobo Primitive Catalog

This is the ground-truth reference for every primitive callable on `CuRoboMotionPlanner` (see `src/kinematics/xarm_curobo_interface.py`). Use these names and parameters verbatim in skill plans. Units and frames are explicit:
- Positions: meters
- Orientations: quaternions `[x, y, z, w]`

Read this top-to-bottom if you are new to the stack—each entry explains what the primitive does, its arguments, defaults, typical uses, and safety notes. Gemini only sees RGB; emit the helper parameters listed for each primitive (e.g., `target_pixel_yx`, `depth_offset_m`) and the `PrimitiveExecutor` will convert them into metric poses/points immediately before execution. 

## move_to_pose
- **What it does**: Converts a camera-frame pose into the base frame and plans a trajectory to move the gripper to the pose
- **Parameters**:
- `target_pixel_yx` `[y, x]` (normalized 0-1000) – pixel for the grasp/approach point.
- `preset_orientation` set to either "top_down" or "side", where a top_down grasp points the end-effector towards teh ground and a side grasp points the end-effector away from the base (i.e., parallel to the ground).
- `is_place` boolean set to true when performing a place action or a wipe action to provide a small offset in the z-direction.

## retract_gripper
- **What it does**: Backs the TCP away from the workspace and returns to a standby position.
- **Parameters**:
None

## open_gripper
- **What it does**: Opens the xArm gripper.
- **Parameters**:
None

## close_gripper
- **What it does**: Closes the xArm gripper.
- **Parameters**:
None

## wipe
- **What it does**: Performs a wiping motion by rotating the wrist joint in one direction, then returning to the original position. This creates a complete wipe cycle. Useful for wiping surfaces with a held cloth or sponge.
- **Parameters**:
  - `direction` (optional): Initial rotation direction - `"clockwise"` or `"counterclockwise"` (default: `"clockwise"`)
  - `rotation_angle_deg` (optional): Angle to rotate in degrees (default: 90)
  - `speed_factor` (optional): Speed multiplier for the motion, range 0.1 to 2.0 (default: 1.0)
  - `timeout` (optional): Maximum time to wait for completion in seconds (default: 15)
- **Example use cases**:
  - Wiping a surface: After grasping a cloth, move to surface, then call `wipe()` for a 90° wipe cycle
  - Larger wipe motion: `wipe(direction="clockwise", rotation_angle_deg=180)` for a 180° wipe cycle
  - Note: The wrist automatically returns to its original angle after the wipe motion

## Quick pick/place recipe (example)
- Pick: `move_to_pose` (camera frame grasp pose, `preset_orientation="side"` if doing a side grasp), then `close_gripper`, then `retract_gripper`.
- Place: `move_to_pose` to place pose (`is_place=True` to lift out), then `open_gripper`, then `retract_gripper`.
