# xArm CuRobo Primitive Catalog

This is the ground-truth reference for every primitive callable on `CuRoboMotionPlanner` (see `src/kinematics/xarm_curobo_interface.py`). Use these names and parameters verbatim in skill plans. Units and frames are explicit:
- Positions: meters
- Orientations: quaternions `[x, y, z, w]`
- Frames: `"base"` (robot base) or `"camera"` when explicitly stated

Read this top-to-bottom if you are new to the stack—each entry explains what the primitive does, its arguments, defaults, typical uses, and safety notes. Gemini only sees RGB; emit the helper parameters listed for each primitive (e.g., `target_pixel_yx`, `pivot_pixel_yx`, `depth_offset_m`) and the `PrimitiveExecutor` will convert them into metric poses/points immediately before execution.

## move_to_pose
- **What it does**: Plans a Cartesian move to a target pose in the base frame and can execute it.
- **Parameters**:
  - `target_position` `[x, y, z]` (required) – base-frame coordinates.
  - `target_orientation` `[x, y, z, w]` (optional) – defaults to a safe top-down orientation.
  - `force_top_down` (bool) – force vertical approach; overrides provided orientation.
  - `unconstrained_orientation` (bool) – let the planner search orientation freely.
  - `planning_timeout` (float, seconds) – defaults to 5.0.
  - `execute` (bool) – if True, send the planned trajectory to the robot.
  - `speed_factor` (float) – scales execution speed; start at 0.3–0.6 for contact work, 1.0 for nominal free-space moves, and avoid >1.5 unless you have headroom and clearance.
- **Notes**: Requires a valid joint seed; the planner fetches the latest joint state when the robot is connected. Use when you already have a base-frame goal pose.
- **LLM helper fields to output**:
  - `target_pixel_yx` `[y, x]` (normalized 0-1000) – pixel for the grasp/approach point.
  - `depth_offset_m` (float, optional) – additional distance to move along the camera ray (positive values push farther from the camera).
  - `motion_normal` `[nx, ny, nz]` (optional) – desired contact normal; defaults to TCP Z.
  - `tcp_standoff_m` (optional) – end-effector offset before final approach.

## move_to_pose_with_preparation
- **What it does**: Converts a camera-frame pose into the base frame, plans the move, and optionally adjusts approach based on depth.
- **Parameters**:
  - `target_position` `[x, y, z]` (required) – camera-frame coords when `is_camera_frame=True`.
  - `target_orientation` `[x, y, z, w]` (optional) – defaults to top-down when omitted.
  - `is_camera_frame` (bool) – convert from camera to base (default True).
  - `is_place` (bool) – lift 10 cm after placing.
  - `force_top_down`, `unconstrained_orientation`, `planning_timeout`, `execute`, `speed_factor` – same meaning as `move_to_pose`.
  - `adjust_tcp_for_surface` (bool) – use depth to nudge approach when available.
  - `tcp_standoff_m`, `search_radius_m` (float, meters) – approach tuning knobs.
- **Notes**: Use when your perception output is in the camera frame or you need slight TCP adjustments before contact.
- **LLM helper fields to output**:
  - Same as `move_to_pose` (`target_pixel_yx`, `depth_offset_m`, `motion_normal`, `tcp_standoff_m`).
  - Set `is_camera_frame`/`is_place` booleans directly when appropriate.

## retract_gripper
- **What it does**: Backs the TCP away from the workspace and returns to a neutral joint seed.
- **Parameters**:
  - `distance` (float, meters, optional, default 0.05) – Z lift before homing.
  - `speed_factor` (float, optional) – scales retraction speed; 0.5–1.0 is typical, avoid >1.5 near clutter.
- **Notes**: Helpful after a place/close to clear collisions.

## open_gripper
- **What it does**: Opens the xArm gripper through the SDK.
- **Parameters**:
  - `wait` (bool, optional) – wait for completion.
  - `timeout` (float, seconds, optional) – operation timeout.
- **Notes**: Lite6 uses `open_lite6_gripper`; full xArm uses gripper position.

## close_gripper
- **What it does**: Closes the gripper (simple close or torque-monitored path).
- **Parameters**:
  - `wait` (bool, optional) – wait for completion.
  - `timeout` (float, seconds, optional) – operation timeout.
  - `simple_close` (bool, optional, default True) – disable torque adjustments when True.
- **Notes**: Torque-monitored path reduces speed and adjusts position if imbalance is detected.

## plan_push_pull
- **What it does**: Plans a push or pull along a surface normal or hinge arc.
- **Parameters**:
  - `distance` (float, meters, required) – linear travel when not using a pivot/hinge.
  - `is_push` (bool, optional) – True for push, False for pull.
  - `custom_normal` `[x, y, z]` (optional) – explicit normal; defaults to TCP normal.
  - `move_parallel` (bool, optional) – slide parallel to surface.
  - `planning_timeout` (float, seconds, optional) – defaults to 5.0.
  - `current_position` `[x, y, z]`, `current_orientation` `[x, y, z, w]` (optional) – overrides FK seed.
  - `execute` (bool, optional) – run the trajectory when True.
  - `speed_factor` (float, optional) – execution scaling; stay ≤1.0 until the path is proven collision-free.
  - `pivot_point` `[x, y, z]` (optional) – legacy arc center; overrides `distance`.
  - `arc_segments` (int, optional) – discretization for arc motion.
  - `hinge_location` (`top|bottom|left|right`, optional) – hinge-aware arc direction.
- **Notes**: Returns `(success, trajectory, dt)` when executed; keep normals unit length.
- **LLM helper fields to output**:
  - `pivot_pixel_yx` `[y, x]` – normalized pixel for the hinge/pivot reference (executor converts to meters).
  - `motion_normal` `[nx, ny, nz]` – push/pull normal when not following a hinge.
  - `distance` (meters) remains in metric form; emit `custom_normal`, `hinge_location`, etc. as usual.

## execute_wrist_twist
- **What it does**: Rotates only the wrist joint using velocity control.
- **Parameters**:
  - `direction` (`clockwise|counterclockwise`, optional) – rotation direction.
  - `rotation_angle` (float, radians, optional, default 2π) – angle to rotate.
  - `speed_factor` (float, optional) – scales wrist joint velocity; start at 0.2–0.5, avoid >1.0 unless clearance is guaranteed.
  - `timeout` (float, seconds, optional) – safety timeout.
- **Notes**: Uses `vc_set_joint_velocity` then returns to position mode; ensure collision clearance.

## Quick pick/place recipe (example)
- Pick: `move_to_pose_with_preparation` (camera frame grasp pose, `force_top_down=False` if side grasps), then `close_gripper`, then `retract_gripper`.
- Place: `move_to_pose_with_preparation` to place pose (`is_place=True` to lift out), then `open_gripper`, then `retract_gripper`.

## Safety checklist
- Always specify the frame when positions are derived from perception.
- Keep `speed_factor` conservative for contact tasks; raise only after validation.
- Ensure surface normals (`custom_normal`) are unit length to avoid scaled motion.
- Use `adjust_tcp_for_surface` only when depth is available and clean.
