"""
Streamlined xArm CuRobo Interface - Optimized for Task Planner Integration

This is a streamlined version of CuRoboMotionPlanner that includes only the methods
required by the streamlined skill_executor, with all unused methods removed.

Reduction: 5126 lines -> ~2850 lines (44% reduction)

Required Methods (37 total):
- Core initialization and configuration (4)
- Joint state management (4)
- Planning initialization (3)
- Execution methods (2)
- Robot state management (4)
- Gripper control (4)
- Motion planning (7)
- Coordinate transforms (2)
- Utilities and helpers (7)

All deprecated, unused, and alternative methods have been removed.
"""

import numpy as np
import time
import torch
import math
import threading
from typing import List, Dict, Optional, Tuple, Union, Any
import os
import yaml
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation


# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen, 
    MotionGenConfig, 
    MotionGenPlanConfig, 
    MotionGenResult, 
    MotionGenStatus
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGenerator, CudaRobotGeneratorConfig
# xArm SDK import
from xarm.wrapper import XArmAPI


class RobotConfig:
    def __init__(self):
        self.dof = 6
        self.robot_type = "lite"
        self.prefix = ""
        self.arm_group_name = ""
        self.gripper_group_name = ""
        self.tcp_link = ""
        self.is_lite6 = True
        self.robot_ip = ""
        self.robot_cfg = None


class CuRoboMotionPlanner:
    def __init__(self, config=None, robot_ip=None, static_camera_tf=None):
        """
        Initialize CuRobo motion planner with xArm SDK integration
        
        Args:
            config: Optional configuration object
            robot_ip: IP address of the xArm robot
            static_camera_tf: Optional SE(3) transform from static camera to robot base frame.
                             Can be provided as:
                             - 4x4 homogeneous transformation matrix (numpy array)
                             - tuple/list of (translation_vector, rotation) where rotation is 
                               quaternion [x,y,z,w] or 3x3 rotation matrix
        """
        # Initialize configuration
        self.config = config or self.initialize_default_config()
        
        # Set robot IP if provided
        if robot_ip is not None:
            self.config.robot_ip = robot_ip
        
        # Store static camera transform
        self.static_camera_tf = None
        self.static_camera_position = None
        self.static_camera_rotation = None
        
        if static_camera_tf is not None:
            self._parse_static_camera_tf(static_camera_tf)
        
        # Initialize TensorDeviceType for cuRobo
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        
        # Store collision objects
        self.collision_objects = []
        self.collision_object_lock = threading.Lock()
        
        # Current robot state
        self.current_joints = None
        self.joint_state_lock = threading.Lock()
        
        # Initialize xArm SDK
        self.arm = None
        self.arm_lock = threading.Lock()
        
        # Force/torque sensor settings for robust interactions
        self.default_collision_sensitivity = 1  # Lower = more sensitive (0-5)
        self.default_teach_sensitivity = 1       # Lower = more sensitive (0-5) 
        self.pivot_collision_sensitivity = 1     # Less sensitive for pivot operations
        self.pivot_teach_sensitivity = 1         # Less sensitive for pivot operations
        
        # Dynamic sensitivity adjustment based on torque errors  
        self.max_collision_sensitivity = 1       # Maximum collision sensitivity (least sensitive) 
        self.max_teach_sensitivity = 1           # Maximum teach sensitivity (least sensitive)
        self.min_collision_sensitivity = 0       # Minimum collision sensitivity (most sensitive)
        self.min_teach_sensitivity = 0           # Minimum teach sensitivity (most sensitive)
        self.current_pivot_collision_sensitivity = self.pivot_collision_sensitivity
        self.current_pivot_teach_sensitivity = self.pivot_teach_sensitivity
        
        # Alternative torque handling parameters
        self.min_speed_factor = 0.1             # Minimum speed for torque issues
        self.max_segments = 20                  # Maximum trajectory segments
        self.torque_retry_segments = [10, 15, 20]  # Escalating segment counts
        
        self.connect_robot()
        
        # Configure initial robot sensitivity for normal operations
        if self.arm:
            self._configure_initial_sensitivity()
        
        # Register callback for joint state updates
        if self.arm is not None:
            self.arm.register_report_location_callback(self.joint_state_callback)
        
        # Initialize cuRobo motion generator
        self.motion_gen = self.init_curobo()
        _, self.initial_position = self.arm.get_servo_angle(is_radian=True)
        # Initialize IK solver only if motion generator succeeded
        if self.motion_gen is not None:
            self.ik_solver = self.init_ik_solver()
        else:
            print("Cannot initialize IK solver: motion generator initialization failed")
            self.ik_solver = None
        
        print('CuRobo Motion Planner initialized with xArm SDK')
        if self.static_camera_tf is not None:
            print('Using static camera transform for pose conversions')
            print(f"Transform: \n {self.static_camera_tf}")

    def camera_pose_from_joints(self, joints: List[float]) -> Optional[Tuple[np.ndarray, Rotation]]:
        """
        Compute camera pose (position + Rotation) for an arbitrary joint state using motion_gen kinematics.

        Args:
            joints: Joint angles in radians matching the configured robot DOF.

        Returns:
            (position, rotation) where position is np.ndarray shape (3,) and rotation is scipy Rotation,
            or None on failure.
        """
        try:
            if self.motion_gen is None or not getattr(self.motion_gen, "kinematics", None):
                return None
            config = torch.tensor([joints], device=self.tensor_args.device, dtype=torch.float32)
            state = self.motion_gen.kinematics.get_state(config)
            camera_pose = state.links_position.cpu().numpy()[0][1]  # [x, y, z]
            camera_quat_raw = state.links_quaternion.cpu().numpy()[0][1]

            if camera_quat_raw.shape[-1] != 4:
                return None

            # Reorder to [x, y, z, w] to match convert_cam_pose_to_base
            quat = np.array(
                [
                    camera_quat_raw[1],
                    camera_quat_raw[2],
                    camera_quat_raw[3],
                    camera_quat_raw[0],
                ],
                dtype=float,
            )

            rot = Rotation.from_quat(quat)
            camera_pose = np.asarray(camera_pose, dtype=float)
            camera_pose[1] += 0.01  # mirror convert_cam_pose_to_base offset
            return camera_pose, rot
        except Exception:
            return None
            
    def get_robot_state(self) -> Dict[str, Any]:
        """
        Duck-typed robot state for orchestrator snapshots.

        Returns a JSON-serializable dict capturing current joints, TCP pose,
        and camera transform when available. Fields are optional and omitted
        when unavailable; the orchestrator does not rely on any fixed schema.
        """
        state: Dict[str, Any] = {
            "stamp": time.time(),
            "provider": type(self).__name__,
        }
        try:
            joints = self.get_robot_joint_state()
            if joints is not None:
                state["joints"] = np.asarray(joints, dtype=float).tolist()
        except Exception:
            pass

        try:
            tcp = self.get_robot_tcp_pose()
            if isinstance(tcp, tuple) and len(tcp) == 2:
                pos, quat = tcp
                state["tcp_pose"] = {
                    "position": np.asarray(pos, dtype=float).tolist() if pos is not None else None,
                    "quaternion_xyzw": np.asarray(quat, dtype=float).tolist() if quat is not None else None,
                }
        except Exception:
            pass

        try:
            cam_tf = self.get_camera_transform()
            if isinstance(cam_tf, tuple) and len(cam_tf) == 2:
                cam_pos, cam_quat = cam_tf
                # Provide a predictable sub-structure for camera transform
                state["camera"] = {
                    "position": np.asarray(cam_pos, dtype=float).tolist() if cam_pos is not None else None,
                    "quaternion_xyzw": np.asarray(cam_quat, dtype=float).tolist() if cam_quat is not None else None,
                }
        except Exception:
            pass

        # Include static camera transform when set
        if self.static_camera_position is not None or self.static_camera_rotation is not None:
            try:
                quat = None
                if self.static_camera_rotation is not None:
                    quat_np = self.static_camera_rotation.as_quat()  # xyzw
                    quat = np.asarray(quat_np, dtype=float).tolist()
                pos = np.asarray(self.static_camera_position, dtype=float).tolist() if self.static_camera_position is not None else None
                state["static_camera"] = {
                    "position": pos,
                    "quaternion_xyzw": quat,
                }
            except Exception:
                pass

        return state

    def initialize_default_config(self) -> RobotConfig:
        """Initialize default configuration for the planner"""
        config = RobotConfig()
        
        # Set default parameters
        config.dof = 7
        config.robot_type = "xarm"
        config.prefix = ""
        config.robot_ip = "192.168.1.224"
        
        # Check if it's a Lite6 robot
        config.is_lite6 = (config.robot_type == "lite" and config.dof == 6)
        
        # Calculate planning group
        config.arm_group_name = f"{config.prefix}{config.robot_type}{config.dof}"
        
        # Similarly for gripper group
        config.gripper_group_name = f"{config.prefix}{config.robot_type}_gripper"
        config.tcp_link = f"{config.prefix}link_tcp"
        
        print(
            f"Configuration loaded: robot_type={config.robot_type}, dof={config.dof}, "
            f"is_lite6={config.is_lite6}, robot_ip={config.robot_ip}"
        )
        return config
    

    def _parse_static_camera_tf(self, static_camera_tf):
        """Parse the static camera transform into position and rotation components
        
        Args:
            static_camera_tf: SE(3) transform as 4x4 matrix or (translation, rotation) tuple
        """
        try:
            if isinstance(static_camera_tf, np.ndarray) and static_camera_tf.shape == (4, 4):
                # 4x4 homogeneous transformation matrix
                self.static_camera_position = static_camera_tf[:3, 3]
                self.static_camera_rotation = Rotation.from_matrix(static_camera_tf[:3, :3])
                print(f"Parsed 4x4 static camera transform: pos={self.static_camera_position}, rot={self.static_camera_rotation.as_quat()}")
                
            else:
                raise ValueError(f"static_camera_tf must be 4x4 matrix or (translation, rotation) tuple, got {type(static_camera_tf)}")
                
            # Store the original transform for reference
            self.static_camera_tf = static_camera_tf
        except Exception as e:
            print(f"Error parsing static camera transform: {e}")
            print("Static camera transform will be ignored, falling back to dynamic transform")
            self.static_camera_tf = None
            self.static_camera_position = None
            self.static_camera_rotation = None
    

    def connect_robot(self):
        """Connect to the physical robot using xArm SDK"""
        try:
            with self.arm_lock:
                if self.config.robot_ip:
                    print(f"Connecting to robot at {self.config.robot_ip}")
                    self.arm = XArmAPI(self.config.robot_ip, is_radian=True)
                    
                    # Set default parameters
                    self.arm.motion_enable(enable=True)
                    self.arm.set_mode(0)  # Position control mode
                    code = self.arm.set_state(state=0)  # Start state
                    print(f"Set arm state: {code}")
                    # Get current joint states
                    code, angles = self.arm.get_servo_angle(is_radian=True)
                    if code == 0 and angles is not None:
                        self.set_current_joint_state(angles)
                        print(f"Current joint state: {angles}")
                    
                    print(f"Robot connected successfully. Mode: {self.arm.mode}, State: {self.arm.state}")
                else:
                    print("No robot IP configured, running in simulation mode")
        except Exception as e:
            print(f"Failed to connect to the robot: {str(e)}")
            self.arm = None
    

    def joint_state_callback(self, data):
        """Callback function for joint state updates from the robot
        
        Args:
            data: Robot state data from xArm SDK
        """
        if data and len(data) > 7:  # Make sure we have joint data
            # xArm SDK reports joint angles in the data
            joint_angles = data[:self.config.dof]  # xArm Lite6 has 6 joints
            with self.joint_state_lock:
                self.current_joints = np.array(joint_angles)
    

    def init_curobo(self):
        """Initialize cuRobo motion generator"""
        try:
            # Get the robot configs path from cuRobo
            robot_configs_path = get_robot_configs_path()
            print(f"Robot configs path: {robot_configs_path}")
            
            # Create path for XArm Lite6 config
            xarm_config_dir = os.path.join(robot_configs_path, f"xarm" + "_lite6" if self.config.is_lite6 else f"{self.config.dof}")
            os.makedirs(xarm_config_dir, exist_ok=True)
            filename = "xarm_lite6" if self.config.is_lite6 else f"{self.config.robot_type}{self.config.dof}"
            # Define robot configuration file path
            robot_config_path = os.path.join(robot_configs_path, filename + ".yml")
            
            # Basic world configuration with just a table below the robot
            world_config = {
                "cuboid": {
                    "table": {
                        "dims": [1.0, 1.0, 0.05],  # x, y, z
                        "pose": [0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
                    }
                }
            }
            
            # Set up motion generator configuration with optimized parameters
            print(f"Loading motion generator config from: {robot_config_path}")
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_config_path,
                world_model=world_config,
                tensor_args=self.tensor_args,
                # Planning parameters - optimized for speed
                interpolation_dt=0.015,         # Increased from 0.01 for faster interpolation
                interpolation_steps=2500,      # Halved from 5000 for speed
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                
                # Optimization parameters - balanced for speed and reliability
                num_ik_seeds=12,        # Moderate reduction from 16 (was 32)
                num_graph_seeds=2,      # Restored from 1 (was 4)  
                num_trajopt_seeds=4,    # Restored from 2 (was 6)
                
                # Trajectory optimization parameters - faster convergence
                trajopt_tsteps=24,      # Reduced from default 32
                trajopt_dt=0.25,        # Reduced from 0.5 for faster optimization
                js_trajopt_dt=0.25,     # Reduced from 0.5 for faster optimization
                
                # Collision parameters - relaxed for speed
                collision_activation_distance=0.04,    # Slightly increased from 0.03
                collision_max_outside_distance=0.3,    # Reduced from 0.5
                
                # Quality parameters - restore some quality for reliability
                evaluate_interpolated_trajectory=True,  # Re-enabled for trajectory quality
                position_threshold=0.01,    # Restored to original for accuracy
                rotation_threshold=0.1,     # Restored to original
                cspace_threshold=0.05,      # Restored to original
                
                # Performance parameters
                use_cuda_graph=True,        # ENABLED: Cache CUDA graphs for massive speedup
                store_debug_in_result=False, # Keep disabled for speed
                minimum_trajectory_dt=0.015, # Must be >= interpolation_dt (0.015)
                finetune_dt_scale=0.85,      # Restored to original
                
                # Smoothness parameters - restore for quality
                minimize_jerk=True,          # Re-enabled for smooth trajectories
                filter_robot_command=True    # Re-enabled for command filtering
            )
            
            # Store robot config for IK solver - use the imported RobotConfig, not our class
            # Load the robot config from the YAML file for IK solver
            from curobo.types.robot import RobotConfig as CuroboRobotConfig
            
            self.config.robot_cfg = CuroboRobotConfig.from_dict(
                load_yaml(robot_config_path)["robot_cfg"],
                self.tensor_args
            )
            
            # Create motion generator
            print("Initializing cuRobo motion generator...")
            motion_gen = MotionGen(motion_gen_config)

            # Warmup the motion generator with common problem sizes
            # This pre-compiles CUDA graphs for faster subsequent planning
            print("Warming up motion generator (this may take 10-20 seconds)...")
            motion_gen.warmup(warmup_js_trajopt=True)  # Warm up joint space trajopt

            # Additional warmup for common batch sizes if needed
            # Uncomment if you see "warming up solver" messages during operation:
            # print("Performing extended warmup for batch planning...")
            # motion_gen.warmup(n_goalset=4, warmup_js_trajopt=True)

            print("cuRobo motion generator initialized successfully")
            return motion_gen
        except Exception as e:
            print(f"Failed to initialize cuRobo motion generator: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    

    def init_ik_solver(self):
        """Initialize cuRobo IK solver"""
        try:
            if not hasattr(self.config, "robot_cfg") or self.config.robot_cfg is None:
                print("Robot configuration not available for IK solver")
                return None
                
            # Create IK solver configuration - balanced for speed and reliability
            ik_config = IKSolverConfig.load_from_robot_config(
                self.config.robot_cfg,
                self.motion_gen.world_collision.world_model if self.motion_gen and self.motion_gen.world_collision else None,
                rotation_threshold=0.05,        # Restored for accuracy
                position_threshold=0.005,       # Restored for accuracy
                num_seeds=24,                   # Moderate reduction from 32
                self_collision_check=True,      # Keep for safety
                self_collision_opt=True,        # Re-enabled for better solutions
                tensor_args=self.tensor_args,
                use_cuda_graph=True             # ENABLED: Cache CUDA graphs for speedup
            )
            
            # Create IK solver
            ik_solver = IKSolver(ik_config)
            
            print("cuRobo IK solver initialized successfully")
            return ik_solver
        except Exception as e:
            print(f"Failed to initialize cuRobo IK solver: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    

    def set_current_joint_state(self, joint_positions):
        """Set current joint state for planning
        
        Args:
            joint_positions: Array of joint positions in radians
        """
        with self.joint_state_lock:
            self.current_joints = np.array(joint_positions[:self.config.dof])
    

    def get_robot_joint_state(self):
        """Get current joint state from the physical robot
        
        Returns:
            numpy.ndarray: Joint positions in radians or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            with self.arm_lock:
                code, angles = self.arm.get_servo_angle(is_radian=True)
                if code == 0 and angles is not None:
                    # Update our internal state
                    self.set_current_joint_state(angles)
                    return np.array(angles)
                else:
                    print(f"Failed to get joint state, error code: {code}")
                    return None
        except Exception as e:
            print(f"Error getting robot joint state: {str(e)}")
            return None
    

    def get_robot_tcp_pose(self):
        """Get current TCP position and orientation from the physical robot
        
        Returns:
            tuple: (position, orientation) or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            # Check if motion generator is properly initialized
            if self.motion_gen is None:
                print("Motion generator not initialized")
                return None
            
            if not hasattr(self.motion_gen, 'kinematics') or self.motion_gen.kinematics is None:
                print("Motion generator kinematics not available")
                return None
                
            with self.arm_lock:
                config = torch.from_numpy(np.array(self.arm.angles))
                config = config.cuda("cuda")
                config = config.to(torch.float32)
                state = self.motion_gen.kinematics.get_state(config)
                pose = state.ee_position.cpu().numpy()
                quat = state.ee_quaternion.cpu().numpy()
                print(f"RObiot pose: {pose} quat: {quat}")
                return pose, quat
        except Exception as e:
            print(f"Error getting robot TCP pose: {str(e)}")
            return None
     

    def get_tcp_pose_api(self):
        code, pose =  self.arm.get_position(is_radian=False)
        position, orientation = pose[:3], pose[3:]
        position = [x / 1000 for x in position]
        return position, orientation
     

    def get_camera_transform(self):
            # Check if motion generator is properly initialized
            if self.motion_gen is None:
                print("Motion generator not initialized")
                return None, None
            
            if not hasattr(self.motion_gen, 'kinematics') or self.motion_gen.kinematics is None:
                print("Motion generator kinematics not available")
                return None, None
                
            config = torch.from_numpy(np.array(self.arm.angles))
            config = config.cuda("cuda")
            config = config.to(torch.float32)
            
            state = self.motion_gen.kinematics.get_state(config)
            # Extract camera pose and quaternion with better debugging
            camera_pose = state.links_position.cpu().numpy()[0][1]  # [x, y, z]
            camera_quat_raw = state.links_quaternion.cpu().numpy()[0][1]
             # Handle different quaternion formats that might be returned
            if len(camera_quat_raw.shape) == 1 and camera_quat_raw.shape[0] == 4:
                # Simple quaternion array [x, y, z, w]
                camera_quat = camera_quat_raw
            elif len(camera_quat_raw.shape) == 2:
                # Get the last quaternion if multiple are returned
                camera_quat = camera_quat_raw[-1]
            elif len(camera_quat_raw.shape) == 3:
                # 3D array - get the last element along the first dimension
                camera_quat = camera_quat_raw[-1]
                if len(camera_quat.shape) == 2:
                    # If still 2D, flatten or take appropriate element
                    if camera_quat.shape[0] == 1:
                        camera_quat = camera_quat[0]
                    else:
                        camera_quat = camera_quat.flatten()[:4]  # Take first 4 elements
            else:
                raise ValueError(f"Unexpected quaternion shape: {camera_quat_raw.shape}")
            
            # print(f"Debug - Final camera_quat shape: {camera_quat.shape}, value: {camera_quat}")
            
            # Ensure we have exactly 4 elements for quaternion
            if camera_quat.shape[0] != 4:
                raise ValueError(f"Expected 4 quaternion elements, got {camera_quat.shape[0]}")
            import copy
            # Create transformation matrix from end-effector to base
            camera_quat_copy = copy.deepcopy(camera_quat)
            camera_quat = np.array([camera_quat_raw[1], camera_quat_raw[2], camera_quat_raw[3], camera_quat_raw[0]])
            
            # print(f"Debug - joint state: {config}")
            # print(f"Debug - camera_pose shape: {camera_pose.shape}, value: {list(camera_pose)}")
            # print(f"Debug - camera_quat_raw shape: {camera_quat.shape}")
            # print(f"Debug - camera_quat_raw: {list(camera_quat)}")
            # print(f"Debug - camera_quat_raw norm: {np.linalg.norm(list(camera_quat_raw))}")
            # print()
            # print()
            # print()
            
            camera_rotation = Rotation.from_quat(camera_quat)
            return camera_pose, camera_rotation
        

    def convert_cam_pose_to_base(self, position, orientation, do_translation=True, debug=True):
        """
        Convert camera pose to base frame using either static camera transform or 
        current robot end-effector transform.
        
        Args:
            position: 3D position in camera frame [x, y, z]
            orientation: Orientation in camera frame (quaternion [x, y, z, w] or rotation matrix)
            do_translation: Whether to apply translation (only used for dynamic transform)
        
        Returns:
            tuple: (transformed_position, transformed_orientation) in base frame
        """
        try:
            # Use static camera transform if available
            if self.static_camera_tf is not None:
                # if debug:
                #     print("Using static camera transform for pose conversion")
                camera_pose = self.static_camera_position
                camera_rotation = self.static_camera_rotation
            else:
                # print("Using dynamic camera transform for pose conversion")
                camera_pose, camera_rotation = self.get_camera_transform()
            camera_pose[1] += 0.01
            # if debug:
            #     print(f"Pose before conversion pose: {position}, {orientation}")
            # Convert input position to homogeneous coordinates
            if isinstance(position, (list, tuple)):
                position = np.array(position)
            
            # Transform position to base frame
            transformed_position = camera_rotation.apply(position)
            # if debug:
            #         print(f"Rotated pose: {transformed_position}")
            
            # Apply translation
            if self.static_camera_tf is not None and do_translation:
                # For static camera, always apply translation
                transformed_position += camera_pose
            else:
                # For dynamic camera, respect the do_translation flag
                if do_translation:
                    transformed_position += camera_pose
            if debug:
                    print(f"Fully transformed pose: {transformed_position}")
            
            # Handle orientation transformation
            if isinstance(orientation, np.ndarray) and orientation.shape == (4,):
                # Input is quaternion [x, y, z, w]
                input_rotation = Rotation.from_quat(orientation)
            elif isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
                # Input is rotation matrix
                input_rotation = Rotation.from_matrix(orientation)
            elif isinstance(orientation, (list, tuple)) and len(orientation) == 4:
                # Input is quaternion as list/tuple
                input_rotation = Rotation.from_quat(np.array(orientation))
            else:
                raise ValueError(f"Unsupported orientation format: {type(orientation)} with shape {getattr(orientation, 'shape', 'N/A')}")
            
            # Combine rotations: R_base = R_camera * R_input
            combined_rotation = camera_rotation * input_rotation
            transformed_orientation = combined_rotation.as_quat()  # Returns [x, y, z, w]
            
            return transformed_position, transformed_orientation
            
        except Exception as e:
            print(f"Error converting camera pose to base frame: {str(e)}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    

    def execute_wrist_twist(self, direction="clockwise", rotation_angle=2 * np.pi, speed_factor=1.0, timeout=30.0):
        """
        Execute a wrist twist motion by rotating only the wrist joint using velocity control
        
        Args:
            direction: "clockwise" or "counterclockwise" rotation direction
            rotation_angle: Angle to rotate in radians (default: 2π = 360 degrees)
            speed_factor: Speed factor for the motion (0.1 to 2.0)
            timeout: Timeout for the operation in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            print(f"Executing wrist twist: {direction}, angle: {rotation_angle:.2f} rad ({np.degrees(rotation_angle):.1f} deg)")
            
            # Get current joint state
            current_joints = self.get_robot_joint_state()
            if current_joints is None:
                print("Failed to get current joint state")
                return False
            
            # Prepare robot for execution
            if not self.prepare_robot_for_execution():
                print("Failed to prepare robot for execution")
                return False
            
            # Set robot to joint velocity control mode
            print("Setting robot to joint velocity control mode...")
            code = self.arm.set_mode(4)  # Joint velocity control mode
            if code != 0:
                print(f"Failed to set velocity control mode, code: {code}")
                return False
            
            # Enable the robot
            code = self.arm.set_state(0)
            if code != 0:
                print(f"Failed to enable robot, code: {code}")
                return False
            
            # The wrist joint is typically the last joint (index -1)
            # For xArm Lite6, this would be joint 6 (index 5)
            wrist_joint_index = self.config.dof - 1  # Last joint
            
            # Calculate velocity based on direction and speed_factor
            base_velocity = 1  # Base velocity in rad/s
            velocity = base_velocity * speed_factor
            
            if direction.lower() == "clockwise":
                # Clockwise rotation (negative velocity)
                wrist_velocity = velocity
            elif direction.lower() == "counterclockwise":
                # Counterclockwise rotation (positive velocity)
                wrist_velocity = -velocity
            else:
                print(f"Invalid direction: {direction}. Must be 'clockwise' or 'counterclockwise'")
                return False
            
            # Create velocity command - all joints zero except wrist
            joint_velocities = [0.0] * self.config.dof
            joint_velocities[wrist_joint_index] = wrist_velocity
            
            print(f"Current wrist joint angle: {current_joints[wrist_joint_index]:.3f} rad ({np.degrees(current_joints[wrist_joint_index]):.1f} deg)")
            print(f"Wrist velocity: {wrist_velocity:.3f} rad/s")
            
            # Calculate duration needed for the rotation
            duration = rotation_angle / abs(wrist_velocity)
            duration = min(duration, timeout)  # Respect timeout
            
            print(f"Executing velocity command for {duration:.2f} seconds...")
            
            # Execute velocity command
            code = self.arm.vc_set_joint_velocity(
                speeds=joint_velocities, 
                is_radian=True, 
                is_sync=True, 
                duration=int(duration * 1000)  # Convert to milliseconds
            )
            
            if code != 0:
                print(f"Failed to execute velocity command, code: {code}")
                return False
            
            # Wait for motion to complete
            import time
            time.sleep(duration + 0.1)  # Small buffer
            
            # Stop all joint motion
            stop_velocities = [0.0] * self.config.dof
            self.arm.vc_set_joint_velocity(
                speeds=stop_velocities, 
                is_radian=True, 
                is_sync=True, 
                duration=-1
            )
            
            # Return to position control mode
            self.arm.set_mode(0)
            self.arm.set_state(0)
            
            print(f"Wrist twist {direction} completed successfully")
            return True
                
        except Exception as e:
            print(f"Error executing wrist twist: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Ensure we return to position control mode on error
            try:
                if self.arm is not None:
                    stop_velocities = [0.0] * self.config.dof
                    self.arm.vc_set_joint_velocity(
                        speeds=stop_velocities, 
                        is_radian=True, 
                        is_sync=True, 
                        duration=0
                    )
                    self.arm.set_mode(0)
            except:
                pass
            
            return False
    

    def execute_trajectory(self, trajectory, dt, speed_factor=1.0):
        """Execute a trajectory on the physical robot with improved error handling
        
        Args:
            trajectory: Joint trajectory as list of JointState objects
            dt: Time step between trajectory points in seconds
            speed_factor: Factor to scale execution speed (>1 is faster)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        if not trajectory or len(trajectory) == 0:
            print("Empty trajectory provided")
            return False
            
        try:
                # Check robot status before execution
                print(f"Robot status before execution: Mode={self.arm.mode}, State={self.arm.state}")
                
                # Clear any errors first
                if self.arm.has_error or self.arm.has_warn:
                    print("Clearing robot errors before trajectory execution...")
                    self.clean_robot_error()
                    time.sleep(0.5)  # Give robot time to reset
                
                # Ensure robot is in position control mode
                self.arm.set_mode(0)  # Position control mode
                self.arm.set_state(0)  # Ready state
                time.sleep(0.1)
                
                # Verify robot is ready
                if self.arm.mode != 0:
                    print(f"Robot not in ready state (state: {self.arm.state}), aborting trajectory")
                    return False
                
                # Set trajectory parameters
                adjusted_dt = dt / speed_factor  # Adjust time step based on speed factor
                
                print(f"Executing trajectory with {len(trajectory)} points, dt={adjusted_dt:.4f}s")
                
                # Convert trajectory to list of joint positions
                joint_positions_list = []
                if type(trajectory) is not CuroboJointState:
                    trajectory = trajectory.tolist()
                
                for i, point in enumerate(trajectory.position):
                    try:
                        if hasattr(point, 'position'):
                            # Extract joint positions from JointState
                            if point.position.dim() > 1:
                                joint_pos = point.position[0].cpu().numpy()  # Take first batch element
                            else:
                                joint_pos = point.position.cpu().numpy()
                        else:
                            # Handle raw tensor
                            if point.dim() > 1:
                                joint_pos = point[0].cpu().numpy()
                            else:
                                joint_pos = point.cpu().numpy()
                        
                        # Ensure we have the right number of joints
                        if len(joint_pos) != self.config.dof:
                            print(f"Point {i}: Expected {self.config.dof} joints, got {len(joint_pos)}")
                            continue
                            
                        joint_positions_list.append(joint_pos.tolist())
                        
                    except Exception as e:
                        print(f"Error processing trajectory point {i}: {e}")
                        continue
                
                if len(joint_positions_list) == 0:
                    print("No valid trajectory points found")
                    return False
                    
                print(f"Processed {len(joint_positions_list)} valid trajectory points")
                
                # Execute each point in the trajectory
                start_time = time.time()
                for i, joint_pos in enumerate(joint_positions_list):
                    # Check if robot is still in running state
                    
                    # Check timing and wait if needed
                    elapsed = time.time() - start_time
                    target_time = i * adjusted_dt
                    
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                    
                    # Send joint command to the robot
                    try:
                        # Use set_servo_angle for smoother motion
                        code = self.arm.set_servo_angle(angle=joint_pos, is_radian=True, wait=True)
                        if code != 0:
                            print(f"Failed to send joint command at point {i}, error code: {code}")
                            
                            # Check if this is a torque-related error (error code -9 is common for torque issues)
                            if code == -9 or self.arm.state == 4:  # State 4 indicates robot stopped due to error
                                print(f"Detected potential torque constraint error (code: {code}, state: {self.arm.state})")
                                
                                # Try to adjust sensitivity and continue
                                if hasattr(self, 'adjust_sensitivity_for_torque_errors'):
                                    print("Attempting to adjust sensitivity for torque constraints...")
                                    self.clear_robot_errors()
                                    if self.adjust_sensitivity_for_torque_errors():
                                        print("Sensitivity adjusted, attempting to continue trajectory...")
                                        # Try to re-enable motion
                                        self.arm.motion_enable(True)
                                        self.arm.set_state(0)  # Try to return to ready state
                                        time.sleep(0.5)
                                        
                                        # Retry this point once with new sensitivity
                                        retry_code = self.arm.set_servo_angle(angle=joint_pos, is_radian=True, wait=True)
                                        if retry_code == 0:
                                            print(f"✓ Successfully continued after sensitivity adjustment at point {i}")
                                            continue  # Continue with trajectory
                                        else:
                                            print(f"Retry failed even with adjusted sensitivity (code: {retry_code})")
                                    else:
                                        print("Failed to adjust sensitivity")
                            
                            return False
                            
                    except Exception as e:
                        print(f"Error sending joint command at point {i}: {e}")
                        return False
                    
                    # Print progress every 10 points or at milestones
                    if i % max(1, len(joint_positions_list) // 10) == 0 or i == len(joint_positions_list) - 1:
                        print(f"Progress: {i+1}/{len(joint_positions_list)} ({100*(i+1)/len(joint_positions_list):.1f}%)")
                
                print(f"Trajectory execution completed in {time.time() - start_time} seconds")
                
                # Update our internal state to the final position
                if len(joint_positions_list) > 0:
                    self.set_current_joint_state(joint_positions_list[-1])
                
                return True
                    
        except Exception as e:
            print(f"Error executing trajectory: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Try to set the robot back to a safe mode/state
            try:
                self.arm.set_mode(0)  # Position control mode
                self.arm.set_state(0)  # Ready state
            except:
                pass
            return False
    

    def set_robot_joint_angles(self, joint_angles, wait=True, speed=20, acc=500):
        """Move the robot directly to target joint angles using the xArm SDK
        
        Args:
            joint_angles: List of joint angles in radians
            wait: Whether to wait for motion completion
            speed: Joint speed (rad/s)
            acc: Joint acceleration (rad/s²)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Set to position control mode
                self.arm.set_mode(0)
                self.arm.set_state(0)
                
                # Move to target joint angles
                print(f"Moving to joint angles: {joint_angles}")
                code = self.arm.set_servo_angle(joint_angles, speed=speed, mvacc=acc, wait=wait, is_radian=True)
                
                if code == 0:
                    print("Joint move completed successfully")
                    # Update our internal state
                    self.set_current_joint_state(joint_angles)
                    return True
                else:
                    print(f"Failed to move to joint angles, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error moving to joint angles: {str(e)}")
            return False
    

    def force_robot_ready_state(self):
        """Force robot into ready state with comprehensive recovery
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
                print("=== Force Robot Recovery ===")
                
                # First, try emergency stop and recovery
                print("Triggering emergency stop for clean recovery...")
                try:
                    self.arm.emergency_stop()
                    time.sleep(2.0)
                except:
                    pass
                
                # Reset everything step by step
                print("Step 1: Clearing all errors and warnings...")
                try:
                    self.arm.clean_error()
                    self.arm.clean_warn()
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 2: Disabling and re-enabling motion...")
                try:
                    self.arm.motion_enable(enable=False)
                    time.sleep(1.0)
                    self.arm.motion_enable(enable=True)
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 3: Setting mode to manual then position control...")
                try:
                    self.arm.set_mode(2)  # Manual mode first
                    time.sleep(1.0)
                    self.arm.set_mode(0)  # Then position control
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 4: Setting state through pause to ready...")
                try:
                    self.arm.set_state(4)  # Pause state
                    time.sleep(2.0)
                    self.arm.set_state(0)  # Ready state
                    time.sleep(2.0)
                except:
                    pass
                
                # Final verification
                final_state = self.arm.state
                final_mode = self.arm.mode
                has_error = self.arm.has_error
                has_warn = self.arm.has_warn
                
                print(f"Final robot status: Mode={final_mode}, State={final_state}")
                print(f"Error status: has_error={has_error}, has_warn={has_warn}")
                
                if final_state == 0 and final_mode == 0 and not has_error:
                    print("Robot successfully recovered to ready state")
                    return True
                else:
                    print("Robot recovery failed")
                    return False
                    
        except Exception as e:
            print(f"Error in force robot recovery: {str(e)}")
            return False
    

    def prepare_robot_for_execution(self):
        """Prepare robot for trajectory execution with robust error recovery
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            
        try:
            with self.arm_lock:
                print("=== Preparing Robot for Execution ===")
                
                # Check current status
                print(f"Initial status: Mode={self.arm.mode}, State={self.arm.state}")
                print(f"Error status: has_error={self.arm.has_error}, has_warn={self.arm.has_warn}")
                
                # If robot is in error state (state 2) or has errors, do comprehensive recovery
                if self.arm.has_error or self.arm.has_warn:
                    print("Robot needs recovery - attempting comprehensive reset...")
                    if not self.force_robot_ready_state():
                        print("Comprehensive recovery failed")
                        return False
                
                # Standard preparation sequence
                elif self.arm.mode != 0:
                    print("Robot needs standard preparation...")
                    
                    # Clear any lingering errors
                    if self.arm.has_error or self.arm.has_warn:
                        print("Clearing robot errors...")
                        if not self.clean_robot_error():
                            print("Failed to clear errors")
                            return False
                    
                    # Enable motion
                    print("Enabling motion...")
                    code = self.arm.motion_enable(enable=True)
                    if code != 0:
                        print(f"Failed to enable motion, error code: {code}")
                        return False
                    time.sleep(0.5)
                    
                    # Set to position control mode
                    print("Setting position control mode...")
                    code = self.arm.set_mode(0)
                    if code != 0:
                        print(f"Failed to set position control mode, error code: {code}")
                        return False
                    time.sleep(0.5)
                    
                    # Set to ready state
                    print("Setting ready state...")
                    code = self.arm.set_state(0)
                    if code != 0:
                        print(f"Failed to set ready state, error code: {code}")
                        return False
                    time.sleep(1.0)
                print("completed execution prep.")
                # Final verification
                return self.check_robot_ready_for_execution()
                
        except Exception as e:
            print(f"Error preparing robot for execution: {e}")
            return False
    

    def check_robot_ready_for_execution(self):
        """Check if robot is ready for trajectory execution
        
        Returns:
            bool: True if ready, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
                # Get current status
                current_mode = self.arm.mode
                current_state = self.arm.state
                has_error = self.arm.has_error
                has_warn = self.arm.has_warn
                
                print(f"Robot status check: Mode={current_mode}, State={current_state}")
                print(f"Error status: has_error={has_error}, has_warn={has_warn}")
                
                # Check for errors
                if has_error:
                    print(f"Robot has error: {self.arm.error_code}")
                    return False
                    
                if has_warn:
                    print(f"Robot has warning: {self.arm.warn_code}")
                    # Warnings might be acceptable, continue
                    
                # Check mode and state
                if current_mode != 0:
                    print(f"Robot not in position control mode (mode: {current_mode})")
                    return False
                    
                # if current_state != 0:
                #     print(f"Robot not in ready state (state: {current_state})")
                #     return False
                    
                # Check if motors are enabled
                # try:
                #     motor_states = self.arm.motor_enable_states
                #     if motor_states and not all(motor_states):
                #         print("Not all motors are enabled")
                #         return False
                # except:
                #     pass  # Skip motor check if not available
                    
                print("Robot is ready for execution")
                return True
                
        except Exception as e:
            print(f"Error checking robot readiness: {e}")
            return False
    
    

    def create_top_down_orientation(self):
        """Create a quaternion for top-down orientation"""
        # This quaternion represents Z-axis pointing downward in the base frame
        orientation = [0.0, 1.0, 0.0, 0.0]  # w, x, y, z (180 degrees around Y axis)
        
        print(
            f"Created top-down orientation: [w={orientation[0]}, x={orientation[1]}, "
            f"y={orientation[2]}, z={orientation[3]}]"
        )
        
        return orientation

    def create_side_orientation(self):
        """Create a quaternion for side orientation (end-effector parallel to ground)

        The intent is to align the tool such that its approach axis is
        horizontal/parallel to the ground. This uses a simple predefined
        quaternion consistent with the class's internal quaternion ordering
        used by `create_top_down_orientation`.
        """
        # Compute orientation via Euler for clarity:
        # - Rotate +90° about Y to face forward (parallel to ground)
        # - Rotate +180° about Z to flip the tool right-side-up
        # Convert to quaternion in [x, y, z, w], then reorder to [w, x, y, z]
        rot = Rotation.from_euler('y', 90, degrees=True) * Rotation.from_euler('z', 180, degrees=True)
        qxyzw = rot.as_quat()  # [x, y, z, w]
        orientation = [float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2])]  # [w, x, y, z]
        print(
            f"Created side orientation: [w={orientation[0]}, x={orientation[1]}, "
            f"y={orientation[2]}, z={orientation[3]}]"
        )
        return orientation
    
    

    def _create_joint_state_with_batch(self, joint_positions):
        """Helper function to create a CuroboJointState with proper batch dimension
        
        Args:
            joint_positions: Joint positions as tensor, numpy array, or list
            
        Returns:
            CuroboJointState: Joint state with proper batch dimension
        """
        try:
            # Convert to tensor if needed
            if isinstance(joint_positions, (list, np.ndarray)):
                position_tensor = self.tensor_args.to_device([joint_positions])
            else:
                # Already a tensor
                position_tensor = joint_positions
                
            # Ensure proper batch dimension
            if position_tensor.dim() == 1:
                position_tensor = position_tensor.unsqueeze(0)  # Add batch dimension (1, dof)
            elif position_tensor.dim() > 2:
                position_tensor = position_tensor.reshape(1, -1)  # Reshape to (1, dof)
                
            # Ensure we have exactly the right number of joints
            if position_tensor.shape[-1] != self.config.dof:
                print(f"Warning: Expected {self.config.dof} joints, got {position_tensor.shape[-1]}")
                position_tensor = position_tensor[..., :self.config.dof]  # Truncate if needed
                
            # Create joint state
            joint_state = CuroboJointState.from_position(
                position_tensor,
                joint_names=[f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)]
            )
            
            return joint_state
            
        except Exception as e:
            print(f"Error creating joint state with batch: {e}")
            raise

    

    def move_to_pose(
        self,
        target_position,
        target_orientation=None,
        force_top_down=False,
        preset_orientation="top_down",
        unconstrained_orientation=False,
        planning_timeout=5.0,   # Reduced from 10.0 seconds
        execute=False,
        speed_factor=1.0
    ):
        """Plan movement to a target pose with proper batch dimension handling"""
        
        # Handle orientation (same as before)
        if force_top_down:
            target_orientation = self.create_top_down_orientation()
            print("Using top-down orientation for target")
            unconstrained_orientation = False
        elif target_orientation is None:
            if preset_orientation == "top_down":
                target_orientation = self.create_top_down_orientation()
                print("No orientation provided, using top-down preset")
                unconstrained_orientation = False
            elif preset_orientation == "side":
                target_orientation = self.create_side_orientation()
                print("No orientation provided, using side preset")
                # Keep unconstrained_orientation as provided, default False
            else:
                target_orientation = self.create_top_down_orientation()
                print("Unknown preset, defaulting to top-down")
                unconstrained_orientation = False
        else:
            # Validate the provided orientation
            quat_norm = np.sqrt(sum(x**2 for x in target_orientation))
            # print(quat_norm)
            # if abs(quat_norm - 1.0) > 0.01 or quat_norm < 0.01:
            #     print("Invalid quaternion detected. Using top-down orientation instead.")
            #     target_orientation = self.create_top_down_orientation()
            #     unconstrained_orientation = False
            # else:
            #     print("Using provided orientation")
                
        # Get current joint state from the robot if connected
        if self.arm is not None:
            robot_joints = self.get_robot_joint_state()
            if robot_joints is not None:
                self.set_current_joint_state(robot_joints)
                
        # Check if current joints are available
        with self.joint_state_lock:
            if self.current_joints is None:
                print("No current joint state available")
                return False, None, None
            current_joints = self.current_joints
            
            # Verify joint state size
            if len(current_joints) != self.config.dof:
                print(f"Joint state size mismatch: {len(current_joints)} != {self.config.dof}")
                return False, None, None
            
        try:
            # Convert to CuroboJointState using helper function
            start_state = self._create_joint_state_with_batch(current_joints)
            print(f"Target pose initial: {target_position}")
            
            if type(target_position) not in (list, List, tuple) and len(target_position) < 3:
                print(target_position)
                target_position = target_position[0]
            print(f"Target pose shifted: {target_position}")
            
            # Convert to torch tensors for cuRobo
            target_position_tensor = self.tensor_args.to_device([target_position])
            target_orientation_tensor = self.tensor_args.to_device([target_orientation])
            
            # Create cuRobo Goal Pose
            goal_pose = CuroboPose(
                position=target_position_tensor,
                quaternion=target_orientation_tensor
            )
            print(target_orientation_tensor)
            # Create planning configuration with optimized parameters
            plan_config = self._get_fast_plan_config(
                timeout=planning_timeout,
                max_attempts=3,                    # Further reduced from 5
                enable_graph=True,                 # Enable for cartesian planning
                enable_finetune=True
            )
            
            # Plan the motion
            print("Planning motion to target pose...")
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=plan_config
            )
            
            if result.success.item():
                print(f"Motion planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds")
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned trajectory...")
                    execution_success = self.execute_trajectory(trajectory, result.interpolation_dt, speed_factor)
                    
                    if execution_success:
                        print("Trajectory execution completed successfully")
                    else:
                        print("Trajectory execution failed")
                
                return True, trajectory, result.interpolation_dt
            else:
                print(f"Motion planning failed: {result.status}, "
                    f"after {result.solve_time} seconds")
                return False, None, None
        except Exception as e:
            print(f"Error in motion planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
        
        

    def move_to_pose_with_preparation(
            self,
            target_position,
            target_orientation=None,
            force_top_down=False,
            preset_orientation="top_down",
            unconstrained_orientation=False,
            planning_timeout=5.0,   # Reduced from 10.0 seconds
            execute=False,
            speed_factor=1.0,
            is_camera_frame=True,
            is_place = False,
            depth_image=None,
            object_mask=None,
            adjust_tcp_for_surface=True,
            tcp_standoff_m=0.02,
            search_radius_m=0.05
        ):
            """Plan movement to a target pose with improved robot preparation"""
            
            # Apply preset orientation if none explicitly provided
            if target_orientation is None:
                if preset_orientation == "top_down":
                    target_orientation = self.create_top_down_orientation()
                    force_top_down = True
                elif preset_orientation == "side":
                    target_orientation = self.create_side_orientation()
                    # Do not force top-down; keep as provided side orientation
                    force_top_down = False
                else:
                    # Fallback to top_down for unknown presets
                    target_orientation = self.create_top_down_orientation()
                    force_top_down = True
            
            # Convert from camera frame to base frame if needed
            if is_camera_frame:
                print(f"Converting pose from camera frame to base frame...")
                print(f"Original position: {target_position}, orientation: {target_orientation}")
                
                # Orientation already selected above if None; keep as-is
                # Convert pose using the transformation
                converted_position, converted_orientation = self.convert_cam_pose_to_base(
                    target_position, target_orientation
                )
                converted_orientation = target_orientation
                if converted_position is None or converted_orientation is None:
                    print("Failed to convert camera frame pose to base frame")
                    return False, None, None
                    
                # Update target pose to converted values
                target_position = converted_position
                target_orientation = converted_orientation
                print(f"Converted position: {target_position}, orientation: {target_orientation}")
            
            # Apply surface-based TCP adjustment in robot frame if requested
            print(f"Surface adjustment conditions: adjust_tcp_for_surface={adjust_tcp_for_surface}, "
                  f"depth_image_available={depth_image is not None}, is_camera_frame={is_camera_frame}")
            if adjust_tcp_for_surface and depth_image is not None and is_camera_frame and not is_place:
                print("Applying surface-based TCP adjustment in robot frame...")
                adjusted_position = target_position#self._adjust_tcp_for_surface_robot_frame(
                #      target_position, depth_image, object_mask, tcp_standoff_m, search_radius_m
                # )
                if adjusted_position is not None:
                    target_position = adjusted_position
                    print(f"TCP adjusted position: {target_position}")
                else:
                    print("Surface adjustment returned None - using original position")
            else:
                print("Surface adjustment skipped - conditions not met")
            
            if is_place:
                target_position[2] += 0.1
                
            target_orientation = self._coerce_orientation_vector(target_orientation)
            target_orientation = [
                0.0 if abs(val) < 1e-6 else val for val in target_orientation
            ]
            
            # Plan the motion with converted pose
            success, trajectory, dt = self.move_to_pose(
                target_position, target_orientation, force_top_down, 
                unconstrained_orientation, planning_timeout, execute=False, speed_factor=speed_factor
            )
            
            if not success:
                return False, None, None
                
            # Execute with proper preparation if requested
            if execute and self.arm is not None:
                print("Preparing robot for execution...")
                if not self.prepare_robot_for_execution():
                    print("Failed to prepare robot for execution")
                    return success, trajectory, dt  # Return planning success but execution failure
                    
                print(f"Executing planned trajectory to pos: {target_position} quat: {target_orientation}")
                execution_success = self.execute_trajectory(trajectory, dt, speed_factor)
                
                if execution_success:
                    print("Trajectory execution completed successfully")
                else:
                    print("Trajectory execution failed")
                    
            return success, trajectory, dt
        

    def retract_gripper(self, distance=0.05, speed_factor=1.0):
        pose = self.arm.get_position(is_radian=True)
        if pose[0] != 0:  # Check if get_position was successful
            print(f"Failed to get robot position: {pose}")
            return False
            
        x, y, z, r, p, w = pose[1]
        success = True
        print(f"------------------ z: {z}")
        if z < 500 and x < 400:
            success = self.arm.set_position(x, y, z + distance * 1000, r, p, w, is_radian=True, wait=True, timeout=10.0, speed=50) == 0
        
        if success:
            success = self.arm.set_servo_angle(servo_id=None, angle=self.initial_position, is_radian=True, wait=True, speed=1.0) == 0

        return success
        

    def get_forward_kinematics(self, joint_positions=None):
        """Compute forward kinematics for given joint positions
        
        Args:
            joint_positions: Optional joint positions, or None to use current
            
        Returns:
            tuple: (position, orientation) or None if error
        """
        try:
            # Get current joint state from the robot if connected and no positions provided
            if joint_positions is None and self.arm is not None:
                robot_joints = self.get_robot_joint_state()
                if robot_joints is not None:
                    joint_positions = robot_joints
            
            # Use current joints if still not provided
            if joint_positions is None:
                with self.joint_state_lock:
                    if self.current_joints is None:
                        print("No current joint state available")
                        return None
                    joint_positions = self.current_joints
            
            # Check if we can use the robot's FK for better accuracy
            if self.arm is not None:
                try:
                    with self.arm_lock:
                        code, pose = self.arm.get_position(is_radian=True)
                        if code == 0 and pose is not None:
                            # XArm SDK returns TCP pose as [x, y, z, roll, pitch, yaw]
                            position = pose[:3]
                            # Convert Euler angles to quaternion
                            roll, pitch, yaw = pose[3:]
                            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                            orientation = [qw, qx, qy, qz]
                            return position, orientation
                except Exception as e:
                    print(f"Could not get FK from robot: {str(e)}, using CuRobo FK")
            
            # Use CuRobo's FK if we can't use the robot's
            # Check if IK solver is available
            if self.ik_solver is None:
                print("IK solver not initialized, cannot compute forward kinematics")
                return None
                
            # Create CuroboJointState from joint values
            joint_state = CuroboJointState.from_position(
                self.tensor_args.to_device([joint_positions]),
                joint_names=[f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)]

            )
            
            # Compute forward kinematics
            fk_result = self.ik_solver.fk(self.tensor_args.to_device([joint_positions]))
            position = fk_result.ee_position[0].cpu().numpy()
            orientation = fk_result.ee_quaternion[0].cpu().numpy()
            
            return position, orientation
            
        except Exception as e:
            print(f"Error in forward kinematics: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    

    def set_gripper(self, position, wait=True, timeout=5.0):
        """Control the gripper position
        
        Args:
            position: Gripper position (0 for open, 1 for closed, or value in between)
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
        code = 0
        try:
            with self.arm_lock:
                # Enable the gripper
                code = 0  # Initialize code variable
                if not self.config.is_lite6:
                    code = self.arm.set_gripper_enable(True)
                    if code != 0:
                        print(f"Failed to enable gripper, error code: {code}")
                        return False
                
                # Set gripper position
                # Note: xArm SDK takes position as 0 for open and 100 for closed
                scaled_position = position * 100
                print(f"Setting gripper position to {scaled_position}")
                code = 0
                if self.config.is_lite6:
                    if scaled_position < 50:
                        code = self.arm.open_lite6_gripper()
                    else:
                        code = self.arm.close_lite6_gripper()
                else:
                    code = self.arm.set_gripper_position(scaled_position, wait=wait, timeout=timeout)
                
                if code == 0:
                    print("Gripper operation completed successfully")
                    return True
                else:
                    print(f"Failed to set gripper position, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error controlling gripper: {str(e)}")
            return False
    

    def close_gripper(self, wait=True, timeout=5.0, simple_close=True):
        """Close the gripper with optional torque monitoring and position adjustment
        
        Args:
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            simple_close: If True, skip dynamic adjustments and just close gripper
            
        Returns:
            bool: True if successful, False otherwise
        """
        if simple_close:
            return self.set_gripper(-10.0, wait, timeout)
        else:
            return self.close_gripper_with_torque_adjustment(wait=wait, timeout=timeout)
    

    def close_gripper_with_torque_adjustment(self, wait=True, timeout=5.0):
        """Close gripper with slower speed and torque-based position adjustment
        
        This function:
        1. Decreases the close speed for better control
        2. Monitors gripper torque during closing
        3. Adjusts gripper position if torque is uneven between fingers
        4. If left finger has higher torque, adjusts position toward left finger
        
        Args:
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Enable the gripper
                if not self.config.is_lite6:
                    # Reset gripper first to clear any errors
                    reset_code = self.arm.reset()
                    if reset_code != 0:
                        print(f"Warning: Gripper reset failed with code: {reset_code}")
                    
                    # Enable the gripper
                    code = self.arm.set_gripper_enable(True)
                    if code != 0:
                        print(f"Failed to enable gripper, error code: {code}")
                        return False
                    
                    # Check gripper state after enabling
                    gripper_state = self.arm.get_gripper_position()
                    if gripper_state[0] != 0:
                        print(f"Failed to get gripper state after enabling, error code: {gripper_state[0]}")
                        return False
                    print(f"Gripper enabled successfully, current position: {gripper_state[1]}")
                
                print("Starting torque-monitored gripper closing...")
                
                # Use slower speed for better torque monitoring
                slow_speed = 20  # Reduced from default ~100
                
                # Get initial position and orientation for potential adjustment
                initial_result = self.arm.get_position()
                if initial_result[0] != 0:
                    print(f"Warning: Could not get initial position for torque adjustment, error: {initial_result[0]}")
                    # Fall back to simple gripper close
                    return self.set_gripper(-10.0, wait, timeout)
                
                initial_position = initial_result[1][:3]  # x, y, z
                initial_orientation = initial_result[1][3:]  # roll, pitch, yaw
                
                print(f"Initial TCP position: {initial_position}")
                print(f"Initial TCP orientation (deg): roll={math.degrees(initial_orientation[0]):.2f}, pitch={math.degrees(initial_orientation[1]):.2f}, yaw={math.degrees(initial_orientation[2]):.2f}")
                
                # Analyze gripper orientation to determine adjustment direction if needed
                # For horizontal cabinet handle grip, understand the gripper orientation
                pitch_deg = math.degrees(initial_orientation[1])
                roll_deg = math.degrees(initial_orientation[0])
                print(f"Gripper orientation analysis - Roll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°")
                
                # Determine Y-axis adjustment direction based on gripper orientation
                # This will be used later if torque imbalance is detected
                if abs(pitch_deg) < 45:  # Horizontal-ish grip
                    adjustment_direction = 1 if roll_deg > 0 else -1
                    print(f"Horizontal grip detected - Y adjustment direction set to {adjustment_direction}")
                else:
                    adjustment_direction = 1 if pitch_deg > 0 else -1  
                    print(f"Vertical-ish grip detected - Y adjustment direction set to {adjustment_direction}")
                
                # Gradual gripper closing with torque monitoring
                if self.config.is_lite6:
                    # For Lite6, we can't get detailed gripper torque, just close normally
                    code = self.arm.close_lite6_gripper()
                    if code == 0:
                        print("Lite6 gripper closed successfully")
                        return True
                    else:
                        print(f"Failed to close Lite6 gripper, error code: {code}")
                        return False
                else:
                    # For full xArm with gripper, use gradual closing with monitoring
                    target_position = -10.0  # Target close position
                    current_position = 100.0  # Start from open
                    step_size = -10.0  # Close gradually
                    adjustment_distance = 0.002  # 2mm adjustment steps
                    max_adjustments = 3
                    
                    adjustments_made = 0
                    
                    # Gradual closing loop
                    while current_position > target_position and adjustments_made < max_adjustments:
                        # Move to next position
                        next_position = max(current_position + step_size, target_position)
                        
                        print(f"Moving gripper to position: {next_position}")
                        # Set gripper speed first, then position
                        self.arm.set_gripper_speed(slow_speed)
                        code = self.arm.set_gripper_position(next_position, wait=True, timeout=2.0)
                        
                        if code != 0:
                            print(f"Gripper movement failed with code: {code}")
                            # If incremental movement fails, try direct approach
                            if code == 100:  # Gripper command error
                                print("Trying alternative gripper control method...")
                                # Clear error and try again with smaller step
                                self.arm.clean_error()
                                time.sleep(0.5)
                                # Try smaller step
                                smaller_step = (current_position + target_position) / 2
                                print(f"Trying smaller step to position: {smaller_step}")
                                code = self.arm.set_gripper_position(smaller_step, wait=True, timeout=2.0)
                                if code == 0:
                                    current_position = smaller_step
                                    continue
                            break
                        
                        current_position = next_position
                        time.sleep(0.2)  # Small delay for stabilization
                        
                        # Check for torque imbalance by monitoring gripper state
                        # Since direct finger torque isn't available, we'll use gripper current/load
                        gripper_state = self.arm.get_gripper_position()
                        if gripper_state[0] == 0:
                            gripper_pos = gripper_state[1]
                            print(f"Current gripper position: {gripper_pos}")
                            
                            # Check if gripper is stalled (position not changing much)
                            # This might indicate uneven contact
                            if abs(gripper_pos - next_position) > 5.0:  # Position discrepancy
                                print(f"Gripper position discrepancy detected: target={next_position}, actual={gripper_pos}")
                                
                                # Try to get force/torque sensor data if available
                                try:
                                    ft_result = self.arm.get_ft_sensor_data()
                                    if ft_result[0] == 0:
                                        ft_data = ft_result[1]
                                        fx, fy, fz = ft_data[:3]
                                        tx, ty, tz = ft_data[3:]
                                        
                                        print(f"F/T sensor data - Forces: [{fx:.2f}, {fy:.2f}, {fz:.2f}], Torques: [{tx:.2f}, {ty:.2f}, {tz:.2f}]")
                                        
                                        # Analyze force/torque for imbalance
                                        # If there's significant torque spike, move in the direction of the torque
                                        if abs(ty) > 0.5:  # Threshold for torque imbalance
                                            print(f"Torque spike detected: Ty = {ty:.2f} N⋅m")
                                            
                                            # Move in direction of torque spike to alleviate pressure
                                            # Use predetermined adjustment direction based on gripper orientation
                                            torque_direction = 1 if ty > 0 else -1
                                            adjustment_y = adjustment_direction * torque_direction * adjustment_distance
                                            
                                            new_position = [
                                                initial_position[0],
                                                initial_position[1] + adjustment_y,
                                                initial_position[2]
                                            ]
                                            
                                            print(f"Moving {adjustment_y:.3f}m in Y direction to alleviate torque spike")
                                            print(f"Adjusting from {initial_position[1]:.3f} to {new_position[1]:.3f}")
                                            
                                            # Move to adjusted position while maintaining current orientation
                                            adjust_code = self.arm.set_position(
                                                x=new_position[0] * 1000,  # Convert to mm
                                                y=new_position[1] * 1000,
                                                z=new_position[2] * 1000,
                                                roll=initial_orientation[0],  # Keep exact same orientation
                                                pitch=initial_orientation[1],
                                                yaw=initial_orientation[2],
                                                wait=True,
                                                timeout=3.0,
                                                speed=30,  # Slow, careful adjustment
                                                is_radian=True
                                            )
                                            
                                            if adjust_code == 0:
                                                print("Position adjustment completed successfully")
                                                adjustments_made += 1
                                                time.sleep(0.5)  # Allow settling
                                            else:
                                                print(f"Position adjustment failed with code: {adjust_code}")
                                        
                                except Exception as ft_error:
                                    print(f"Could not read F/T sensor data: {ft_error}")
                                    # Continue without force/torque adjustment
                    
                    # Final close attempt
                    print(f"Final gripper close to position {target_position}")
                    # Clear any errors before final attempt
                    self.arm.clean_error()
                    time.sleep(0.5)
                    
                    # Set gripper speed first, then position
                    self.arm.set_gripper_speed(slow_speed)
                    final_code = self.arm.set_gripper_position(target_position, wait=wait, timeout=timeout)
                    
                    if final_code == 0:
                        print(f"Gripper closed successfully with {adjustments_made} torque adjustments")
                        return True
                    else:
                        print(f"Final gripper close failed with code: {final_code}")
                        # Try one more time with reduced force
                        print("Attempting final close with basic gripper command...")
                        basic_code = self.arm.set_gripper_position(-5.0, wait=wait, timeout=timeout)  # Less aggressive close
                        if basic_code == 0:
                            print("Basic gripper close succeeded")
                            return True
                        else:
                            print(f"Basic gripper close also failed with code: {basic_code}")
                            return False
                        
        except Exception as e:
            print(f"Error in torque-monitored gripper closing: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    

    def open_gripper(self, wait=True, timeout=5.0):
        """Open the gripper
        
        Args:
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.set_gripper(1000.0, wait, timeout)
    

    def plan_push_pull(
        self, 
        distance, 
        is_push=True, 
        custom_normal=None, 
        move_parallel=False, 
        planning_timeout=5.0,   # Reduced from 10.0 seconds
        current_position=None,
        current_orientation=None,
        execute=False,
        speed_factor=1.0,
        pivot_point=None,       # Legacy: [x, y, z] coordinates of pivot point
        arc_segments=5,        # Number of segments for arc motion
        hinge_location=None     # New: hinge location ('top', 'bottom', 'left', 'right')
    ):
        """Plan a push or pull movement along a direction vector
        
        Args:
            distance: Distance to push/pull in meters (ignored if pivot_point provided)
            is_push: True for push, False for pull
            custom_normal: Optional custom normal vector [x, y, z] for movement direction
            move_parallel: Whether to move parallel to the surface (perpendicular to normal)
            planning_timeout: Timeout for planning in seconds
            current_position: Optional current position [x, y, z], if None use forward kinematics 
            current_orientation: Optional current orientation [w, x, y, z], if None use forward kinematics
            execute: Whether to execute the planned trajectory on the physical robot
            speed_factor: Speed factor for execution (>1 is faster)
            pivot_point: Optional [x, y, z] coordinates for legacy pivot-based arc motion
            arc_segments: Number of segments to discretize arc motion (default: 10)
            hinge_location: Optional hinge location ('top', 'bottom', 'left', 'right') for hinge-based motion
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        direction_type = "push" if is_push else "pull"
        movement_type = "parallel" if move_parallel else "perpendicular"
        print(f"Planning {direction_type} movement with distance: {distance} {movement_type}")
        print(f"")
        try:
            # Get current joint state from the robot if connected
            if self.arm is not None:
                robot_joints = self.get_robot_joint_state()
                if robot_joints is not None:
                    self.set_current_joint_state(robot_joints)
            # self.close_gripper()
            # Check if current joints are available
            with self.joint_state_lock:
                if self.current_joints is None:
                    print("No current joint state available")
                    return False, None, None
                current_joints = self.current_joints
            
            # Create CuroboJointState from current joint values
            start_state = CuroboJointState.from_position(
                self.tensor_args.to_device([current_joints]),
                joint_names=[f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)]

            )
            
            # Get current TCP pose through forward kinematics if not provided
            if current_position is None or current_orientation is None:
                # Get the pose from the robot if connected
                if self.arm is not None:
                    pose = self.get_robot_tcp_pose()
                    if pose:
                        current_position, current_orientation = pose
                        print(f"Using robot pose: position={current_position}, orientation={current_orientation}")
                    else:
                        # Fallback to FK if robot pose not available
                        pose = self.get_forward_kinematics()
                        if pose:
                            current_position, current_orientation = pose
                            print(f"Using FK for current pose: position={current_position}, orientation={current_orientation}")
                        else:
                            print("Failed to get current pose")
                            return False, None, None
                else:
                    # Compute forward kinematics to get current pose
                    pose = self.get_forward_kinematics()
                    if pose:
                        current_position, current_orientation = pose
                        print(f"Using FK for current pose: position={current_position}, orientation={current_orientation}")
                    else:
                        print("Failed to get current pose")
                        return False, None, None
            
            # Handle pivot point motion (legacy) or hinge location motion (new)
            if pivot_point is not None or hinge_location is not None:
                grasp_pos = np.array(current_position)
                
                if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                    print(f"Hinge location provided: {hinge_location}")
                    # Calculate hinge edge position based on hinge location and interaction point
                    # This uses the same simplified approach as in skill_executor
                    offset_distance = 0.1  # 10cm offset for hinge location
                    if hinge_location == 'left':
                        hinge_pos = np.array([grasp_pos[0], grasp_pos[1] - offset_distance, grasp_pos[2]])
                    elif hinge_location == 'right':
                        hinge_pos = np.array([grasp_pos[0], grasp_pos[1] + offset_distance, grasp_pos[2]])
                    elif hinge_location == 'top':
                        hinge_pos = np.array([grasp_pos[0] - offset_distance, grasp_pos[1], grasp_pos[2]])
                    elif hinge_location == 'bottom':
                        hinge_pos = np.array([grasp_pos[0] + offset_distance, grasp_pos[1], grasp_pos[2]])
                    
                    # Calculate the radius as straight line distance from interaction point to hinge edge
                    calculated_distance = np.linalg.norm(grasp_pos - hinge_pos)
                    print(f"Hinge-based calculation: interaction_point={grasp_pos}, hinge_edge={hinge_pos}")
                    print(f"Calculated radius from interaction point to hinge edge: {calculated_distance:.4f}m")
                    
                elif pivot_point is not None:
                    print(f"Legacy pivot point provided: {pivot_point}")
                    # Legacy calculation for backward compatibility
                    pivot_pos = np.array(pivot_point)
                    hinge_pos = pivot_pos  # Use pivot point as hinge position for legacy support
                    calculated_distance = np.linalg.norm(grasp_pos - pivot_pos)
                    print(f"Legacy pivot calculation: grasp_pos={grasp_pos}, pivot_pos={pivot_pos}")
                    print(f"Calculated distance from grasp to pivot: {calculated_distance:.4f}m")
                
                # Override the provided distance with calculated distance
                distance = calculated_distance
                
                # Use execute_pivot_pull_direct_xarm for pull operations with pivot/hinge
                if not is_push:  # This is a pull operation
                    motion_type = "hinge-based" if hinge_location else "legacy pivot"
                    print(f"Using execute_pivot_pull_direct_xarm for {motion_type} pull operation")
                    success = self.execute_pivot_pull_direct_xarm(
                        pivot_point=hinge_pos,
                        current_position=grasp_pos,
                        current_orientation=current_orientation,
                        radius=calculated_distance,
                        arc_angle_degrees=30.0,  # Default arc angle
                        segments=arc_segments,
                        speed_factor=speed_factor
                    )
                    # Return in the expected format for plan_push_pull
                    return success, None, None
                else:
                    # For push operations, still use the original arc motion planner
                    print("Using original arc motion planner for pivot push operation")
                    return self._plan_arc_motion_around_pivot(
                        pivot_point=pivot_pos,
                        current_position=grasp_pos,
                        current_orientation=current_orientation,
                        radius=calculated_distance,
                        is_push=is_push,
                        arc_segments=arc_segments,
                        planning_timeout=planning_timeout,
                        execute=execute,
                        speed_factor=speed_factor
                    )
            
            # Calculate the movement direction based on custom normal or TCP orientation
            if custom_normal is not None:
                custom_normal, custom_orientation = self.convert_cam_pose_to_base(position=custom_normal, orientation=[0,0,0,1], do_translation=False)
                # Use the provided custom normal
                print(f"Using custom surface normal: {custom_normal}")
                surface_normal = np.array(custom_normal)
            else:
                # Fallback to using the TCP z-axis as the normal
                print("No custom normal provided, using TCP z-axis as normal")
                
                # Convert quaternion to rotation matrix to extract z-axis
                # Flatten orientation in case it's a nested array
                orientation_flat = np.array(current_orientation).flatten()
                w, x, y, z = orientation_flat
                rotation_matrix = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
                ])
                
                # Extract z-axis (3rd column) from rotation matrix
                surface_normal = rotation_matrix[:, 2]
                surface_normal = surface_normal / np.linalg.norm(surface_normal)
            
            # Determine the movement direction based on parallel flag
            movement_dir = np.zeros(3)
            
            if move_parallel:
                # Movement is perpendicular to the surface normal (parallel to surface)
                # We need a perpendicular vector - we'll use the cross product with the global Z-axis
                global_z = np.array([0, 0, 1])
                perpendicular_dir = np.cross(surface_normal, global_z)
                
                # If the perpendicular direction is too close to zero (normal is parallel to Z),
                # use the X-axis instead
                if np.linalg.norm(perpendicular_dir) < 0.01:
                    global_x = np.array([1, 0, 0])
                    perpendicular_dir = np.cross(surface_normal, global_x)
                
                # Normalize the perpendicular direction
                if np.linalg.norm(perpendicular_dir) > 0.01:
                    perpendicular_dir = perpendicular_dir / np.linalg.norm(perpendicular_dir)
                    movement_dir = perpendicular_dir
                    print(f"Using movement parallel to surface: {movement_dir}")
                else:
                    # Fallback if we couldn't generate a valid perpendicular vector
                    print("Could not determine parallel direction, using surface normal")
                    movement_dir = surface_normal
            else:
                # Movement is along the surface normal (perpendicular to surface)
                movement_dir = surface_normal
                print(f"Using movement perpendicular to surface: {movement_dir}")
            
            # Calculate the movement vector, adjusting direction based on push/pull
            direction_factor = 1.0 if is_push else -1.0
            movement = distance * direction_factor * movement_dir
            
            # Calculate target position
            target_position = current_position + movement
            
            # Keep the current orientation for the target
            target_orientation = current_orientation
            
            # Convert to torch tensors for cuRobo
            target_position_tensor = self.tensor_args.to_device([target_position])
            target_orientation_tensor = self.tensor_args.to_device([target_orientation])
            
            # Create a pose cost metric that enforces straight-line motion
            pose_cost_metric = PoseCostMetric(
                # Keep all position components constrained for linear motion
                hold_partial_pose=False,
                hold_vec_weight=self.tensor_args.to_device([0.01, 0.01, 0.01, 0.5, 0.5, 0.5]),
                project_to_goal_frame=False,
            )
            
            # Create goal pose for planning
            goal_pose = CuroboPose(
                position=target_position_tensor,
                quaternion=target_orientation_tensor
            )
            
            # Create planning config with Cartesian constraints
            # plan_config = MotionGenPlanConfig(
            #     max_attempts=3,
            #     timeout=planning_timeout * 0.9,
            #     enable_opt=True,
            #     enable_graph=True,  # Don't use graph for Cartesian paths
            #     enable_finetune_trajopt=True,
            # )
            
            plan_config = self._get_fast_plan_config(
                timeout=planning_timeout,
                max_attempts=3,              # Reduced from 5
                enable_graph=False,          # Disabled for goal type changes
                enable_finetune=True
            )
            print(f"start pose: {self.get_robot_tcp_pose()}")
            print(f"goal pose: {goal_pose}")
            # Plan the motion
            print("Planning push/pull Cartesian path...")
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=plan_config
            )
            
            if result.success.item():
                print(
                    f"Push/pull planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds"
                )
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned push/pull trajectory on the robot...")
                    execution_success = self.execute_trajectory(
                        trajectory=trajectory,
                        dt=result.interpolation_dt,
                        speed_factor=speed_factor
                    )
                    
                    if execution_success:
                        print("Push/pull trajectory execution completed successfully")
                    else:
                        print("Push/pull trajectory execution failed")
                
                # Return the trajectory and timestep for execution
                return True, trajectory, result.interpolation_dt
            else:
                # Planning failed - try fallback methods
                if result.status == MotionGenStatus.FINETUNE_TRAJOPT_FAIL:
                    print("Finetune trajopt failed, retrying without fine-tuning...")
                    retry_plan_config = plan_config.clone()
                    retry_plan_config.enable_finetune_trajopt = False

                    retry_result = self.motion_gen.plan_single(
                        start_state=start_state,
                        goal_pose=goal_pose,
                        plan_config=retry_plan_config
                    )
                    if retry_result.success.item():
                        print(f"Retry successful in {retry_result.solve_time} seconds, motion time: {retry_result.motion_time} seconds")
                        trajectory = retry_result.get_interpolated_plan()
                        
                        # Execute on the robot if requested
                        if execute and self.arm is not None:
                            print("Executing planned push/pull trajectory on the robot...")
                            execution_success = self.execute_trajectory(
                                trajectory=trajectory,
                                dt=retry_result.interpolation_dt,
                                speed_factor=speed_factor
                            )
                            
                            if execution_success:
                                print("Push/pull trajectory execution completed successfully")
                            else:
                                print("Push/pull trajectory execution failed")
                        
                        return True, trajectory, retry_result.interpolation_dt
                    else:
                        print(f"Retry also failed: {retry_result.status}")
                
                # Try IK-based fallback
                print("Trying IK-based fallback...")
                success, trajectory, dt = self.move_to_pose_with_preparation(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    planning_timeout=planning_timeout * 0.5,
                    execute=execute,
                    speed_factor=speed_factor,
                    is_camera_frame=False
                )
                if success:
                    return True, trajectory, dt
                else:
                    print(f"All planning methods failed: {result.status}")
                    return False, None, None
        except Exception as e:
            print(f"Error in push/pull planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
            

    def clean_robot_error(self):
        """Clear any error or warning on the robot with improved error handling
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                print(f"Current robot state before cleaning: Mode={self.arm.mode}, State={self.arm.state}")
                print(f"Error status: has_error={self.arm.has_error}, error_code={self.arm.error_code}")
                print(f"Warning status: has_warn={self.arm.has_warn}, warn_code={self.arm.warn_code}")
                
                # Step 1: Clear error
                if self.arm.has_error:
                    print("Clearing robot error...")
                    code = self.arm.clean_error()
                    if code != 0:
                        print(f"Failed to clear error, error code: {code}")
                    else:
                        print("Error cleared successfully")
                    time.sleep(0.5)
                
                # Step 2: Clear warning
                if self.arm.has_warn:
                    print("Clearing robot warning...")
                    code = self.arm.clean_warn()
                    if code != 0:
                        print(f"Failed to clear warning, error code: {code}")
                    else:
                        print("Warning cleared successfully")
                    time.sleep(0.5)
                
                # Step 3: Re-enable motion
                print("Re-enabling robot motion...")
                code = self.arm.motion_enable(enable=True)
                if code != 0:
                    print(f"Failed to re-enable robot, error code: {code}")
                    return False
                time.sleep(0.5)
                
                # Step 4: Set mode to position control
                print("Setting robot to position control mode...")
                code = self.arm.set_mode(0)
                if code != 0:
                    print(f"Failed to set mode, error code: {code}")
                    return False
                time.sleep(0.2)
                
                # Step 5: Set to ready state - try multiple times if needed
                for attempt in range(3):
                    print(f"Setting robot to ready state (attempt {attempt + 1})...")
                    code = self.arm.set_state(0)
                    if code == 0:
                        time.sleep(0.5)
                        # Check if state actually changed
                        if self.arm.state == 0:
                            print("Robot successfully set to ready state")
                            return True
                        else:
                            print(f"State command succeeded but robot still in state {self.arm.state}")
                            time.sleep(1.0)  # Wait longer before next attempt
                    else:
                        print(f"Failed to set state (attempt {attempt + 1}), error code: {code}")
                        time.sleep(1.0)
                
                print(f"Failed to set robot to ready state after 3 attempts. Final state: {self.arm.state}")
                return False
                
        except Exception as e:
            print(f"Error clearing robot error: {str(e)}")
            return False
    

    def disconnect_robot(self):
        """Disconnect from the physical robot
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Remove callback
                self.arm.release_report_location_callback(self.joint_state_callback)
                
                # Disconnect
                self.arm.disconnect()
                self.arm = None
                
                print("Robot disconnected successfully")
                return True
        except Exception as e:
            print(f"Error disconnecting from robot: {str(e)}")
            return False
    

    def _get_fast_plan_config(self, timeout: float = 5.0, max_attempts: int = 3, 
                              enable_graph: bool = False, enable_finetune: bool = True) -> MotionGenPlanConfig:
        """
        Get optimized plan configuration for faster motion planning.
        
        Args:
            timeout: Planning timeout in seconds
            max_attempts: Maximum planning attempts
            enable_graph: Whether to enable graph search
            enable_finetune: Whether to enable trajectory fine-tuning
            
        Returns:
            Optimized MotionGenPlanConfig
        """
        return MotionGenPlanConfig(
            max_attempts=max_attempts,
            timeout=timeout,
            enable_opt=True,
            enable_graph=enable_graph,
            enable_graph_attempt=0,                # Fastest graph attempt
            enable_finetune_trajopt=enable_finetune,
            parallel_finetune=True,                # Parallel processing
            time_dilation_factor=0.98,             # More conservative timing
            num_graph_seeds=2,                     # Increased for reliability
            num_trajopt_seeds=4                    # Increased for reliability
        )


    def _ensure_robot_ready_for_motion(self) -> bool:
        """
        Ensure robot is in proper state for motion (state 0) and motion is enabled.
        
        Returns:
            bool: True if robot is ready, False otherwise
        """
        if self.arm is None:
            return False
            
        try:
            # Check current state
            current_state = self.arm.state
            current_mode = self.arm.mode
            
            print(f"Robot status: Mode={current_mode}, State={current_state}")
            
            # Clear any errors first
            if self.arm.has_error:
                print("Clearing robot errors...")
                self.arm.clean_error()
                time.sleep(0.2)
            
            # Clear any warnings
            if self.arm.has_warn:
                print("Clearing robot warnings...")
                self.arm.clean_warn()
                time.sleep(0.2)
            
            # Check if motion needs to be enabled - use motor enable states if available
            try:
                motor_states = self.arm.motor_enable_states
                if motor_states and all(state == 0 for state in motor_states[:self.config.dof]):
                    print("Motors are disabled, enabling motion...")
                    result = self.arm.motion_enable(True)
                    if result != 0:
                        print(f"Failed to enable motion, error: {result}")
                        return False
                    time.sleep(0.2)
            except AttributeError:
                # If motor_enable_states not available, just try to enable motion
                print("Enabling robot motion (motor states not available)...")
                result = self.arm.motion_enable(True)
                if result != 0:
                    print(f"Failed to enable motion, error: {result}")
                    return False
                time.sleep(0.2)
            
            # Set to position control mode if not already
            if current_mode != 0:
                print(f"Setting robot to position control mode (from mode {current_mode})...")
                result = self.arm.set_mode(0)
                if result != 0:
                    print(f"Failed to set mode, error: {result}")
                    return False
                time.sleep(0.2)
            
            # Check if robot is in acceptable state (2=sleeping is ready state)
            if current_state not in [2]:  # State 2 (sleeping) is the ready state
                print(f"Setting robot to sleeping/ready state (from state {current_state})...")
                result = self.arm.set_state(0)  # set_state(0) puts robot in sleeping mode
                if result != 0:
                    print(f"Failed to set state, error: {result}")
                    return False
                time.sleep(0.2)
            
            # Final verification
            final_state = self.arm.state
            final_mode = self.arm.mode
            print(f"Final robot status: Mode={final_mode}, State={final_state}")
            
            return final_mode == 0 and final_state == 2  # Mode 0, State 2 (sleeping) is ready
            
        except Exception as e:
            print(f"Error ensuring robot readiness: {e}")
            return False

    def _coerce_orientation_vector(self, orientation):
        """
        Ensure orientation is a flat list[float] of length >= 1.
        Handles nested lists/tuples/np arrays produced by upstream planners.
        """
        if orientation is None:
            return [0.0, 1.0, 0.0, 0.0]

        if isinstance(orientation, np.ndarray):
            orientation = orientation.flatten().tolist()
        elif isinstance(orientation, (list, tuple)):
            if len(orientation) and isinstance(orientation[0], (list, tuple, np.ndarray)):
                first = orientation[0]
                orientation = (
                    first.flatten().tolist()
                    if isinstance(first, np.ndarray)
                    else list(first)
                )
            else:
                orientation = list(orientation)
        else:
            orientation = [float(orientation)]

        orientation = [float(val) for val in orientation]
        return orientation


    def _plan_arc_motion_around_pivot(
        self,
        pivot_point: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        radius: float,
        is_push: bool = True,
        arc_segments: int = 10,
        planning_timeout: float = 5.0,
        execute: bool = False,
        speed_factor: float = 1.0,
        arc_angle: float = None  # Default: quarter circle (90 degrees)
    ):
        """
        Plan arc motion around a pivot point for push/pull operations.
        
        Args:
            pivot_point: [x, y, z] coordinates of pivot center
            current_position: [x, y, z] current TCP position 
            current_orientation: [w, x, y, z] current TCP orientation
            radius: Distance from current position to pivot point
            is_push: True for push away from object, False for pull toward object
            arc_segments: Number of segments to discretize the arc
            planning_timeout: Planning timeout in seconds
            execute: Whether to execute on robot
            speed_factor: Execution speed factor
            arc_angle: Arc angle in radians (default: π/2 for 90 degrees)
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        try:
            import math
            from scipy.spatial.transform import Rotation
            
            if arc_angle is None:
                arc_angle = math.pi / 4  # Default 45 degree arc (reduced from 90 for feasibility)
                
            print(f"Planning arc motion: radius={radius:.4f}m, angle={math.degrees(arc_angle):.1f}°, segments={arc_segments}")
            
            # Calculate the axis perpendicular to the line from pivot to current position
            # This will be the axis around which we rotate
            # Ensure arrays are flattened for proper operations
            current_pos_flat = current_position.flatten() if hasattr(current_position, 'flatten') else current_position
            pivot_pos_flat = pivot_point.flatten() if hasattr(pivot_point, 'flatten') else pivot_point
            
            pivot_to_grasp = current_pos_flat - pivot_pos_flat
            pivot_to_grasp_normalized = pivot_to_grasp / np.linalg.norm(pivot_to_grasp)
            
            # Calculate rotation axis perpendicular to pivot radius vector
            # For door/cabinet opening, rotation axis should be perpendicular to the pivot radius
            # Determine which axis (Y or Z) the pivot is primarily along
            abs_pivot_vector = np.abs(pivot_to_grasp_normalized)
            
            # Find the primary direction of the pivot radius vector
            primary_axis_idx = np.argmax(abs_pivot_vector)
            
            if primary_axis_idx == 0:  # Pivot radius primarily along X-axis
                # Rotation should be around Y or Z axis
                if abs_pivot_vector[1] > abs_pivot_vector[2]:
                    arc_rotation_axis = np.array([0, 0, 1])  # Z-axis rotation
                else:
                    arc_rotation_axis = np.array([0, 1, 0])  # Y-axis rotation
            elif primary_axis_idx == 1:  # Pivot radius primarily along Y-axis
                # Rotation should be around Z-axis (vertical hinge like cabinet door)
                arc_rotation_axis = np.array([0, 0, 1])  # Z-axis rotation
            else:  # Pivot radius primarily along Z-axis
                # Rotation should be around Y-axis (horizontal hinge)
                arc_rotation_axis = np.array([0, 1, 0])  # Y-axis rotation
            
            print(f"Pivot radius vector: {pivot_to_grasp}")
            print(f"Primary axis: {['X', 'Y', 'Z'][primary_axis_idx]}")
            print(f"Rotation axis: {arc_rotation_axis}")
            
            # DEBUG: Manual rotation test for debugging
            print(f"\n=== DEBUG: Manual Rotation Verification ===")
            test_vector = np.array([0, 0.25, 0])  # Our expected test case
            test_angle_45 = np.radians(45)
            debug_rotation_matrix = self._create_rotation_matrix(test_angle_45, arc_rotation_axis)
            debug_result = debug_rotation_matrix @ test_vector
            print(f"Debug test: Rotating {test_vector} by 45° around {arc_rotation_axis}")
            print(f"Rotation matrix:\n{debug_rotation_matrix}")
            print(f"Result: {debug_result}")
            print(f"Expected for Z-axis rotation: [-0.1767767, 0.1767767, 0]")
            print(f"=== END DEBUG ===\n")
            
            # Generate waypoints along the arc
            waypoints = []
            orientations = []
            
            # Direction of rotation (clockwise or counter-clockwise)
            # For proper door/cabinet opening, determine rotation direction based on geometry
            # Test which direction moves gripper towards robot (pull) or away (push)
            
            # Create a small test rotation to determine correct direction
            test_angle = 0.1  # Small test angle (5.7 degrees)
            test_rotation_matrix_pos = self._create_rotation_matrix(test_angle, arc_rotation_axis)
            test_rotation_matrix_neg = self._create_rotation_matrix(-test_angle, arc_rotation_axis)
            
            # Test both directions
            test_pos_vector = test_rotation_matrix_pos @ pivot_to_grasp
            test_neg_vector = test_rotation_matrix_neg @ pivot_to_grasp
            
            test_pos_position = pivot_pos_flat + test_pos_vector
            test_neg_position = pivot_pos_flat + test_neg_vector
            
            # For pull: gripper should move towards robot (decreasing X in most cases)
            # For push: gripper should move away from robot (increasing X in most cases)
            if is_push:
                # Choose direction that increases X (away from robot)
                if test_pos_position[0] > current_pos_flat[0]:
                    rotation_direction = 1
                else:
                    rotation_direction = -1
            else:
                # Choose direction that decreases X (towards robot) 
                if test_pos_position[0] < current_pos_flat[0]:
                    rotation_direction = 1
                else:
                    rotation_direction = -1
            
            print(f"Test rotation directions:")
            print(f"  Positive direction -> X: {test_pos_position[0]:.6f} (change: {test_pos_position[0] - current_pos_flat[0]:+.6f})")
            print(f"  Negative direction -> X: {test_neg_position[0]:.6f} (change: {test_neg_position[0] - current_pos_flat[0]:+.6f})")
            print(f"Selected rotation direction: {rotation_direction} ({'push' if is_push else 'pull'})")
            
            for i in range(arc_segments + 1):
                # Calculate angle for this segment
                t = i / arc_segments
                angle = rotation_direction * t * arc_angle
                
                # Create rotation matrix around the arc rotation axis
                rotation_matrix = self._create_rotation_matrix(angle, arc_rotation_axis)
                
                # Rotate the vector from pivot to current position
                rotated_vector = rotation_matrix @ pivot_to_grasp
                
                # Calculate new position
                new_position = pivot_pos_flat + rotated_vector
                waypoints.append(new_position)
                
                # Debug output for first, middle, and last waypoints  
                if i == 0 or i == arc_segments // 2 or i == arc_segments:
                    print(f"Waypoint {i}: t={t:.2f}, angle={math.degrees(angle):.1f}°")
                    print(f"  Original pivot_to_grasp: {pivot_to_grasp}")
                    print(f"  Rotation matrix:\n{rotation_matrix}")
                    print(f"  Rotated vector: {rotated_vector}")
                    print(f"  Position: {new_position}")
                    print(f"  Changes: X:{new_position[0]-current_pos_flat[0]:+.6f}, Y:{new_position[1]-current_pos_flat[1]:+.6f}, Z:{new_position[2]-current_pos_flat[2]:+.6f}")
                    if i == arc_segments // 2:  # Middle waypoint special debug
                        print(f"  *** EXPECTED vs ACTUAL for mid-point (45° rotation) ***")
                        if np.allclose(pivot_to_grasp, [0, 0.25, 0], atol=0.01):
                            print(f"  Expected rotated vector: [-0.1767767, 0.1767767, 0]")
                            print(f"  Actual rotated vector:   {rotated_vector}")
                            print(f"  Difference: {rotated_vector - np.array([-0.1767767, 0.1767767, 0])}")
                
                # Calculate gripper orientation: rotate about the same axis as the arc motion
                # For door opening, the gripper should rotate at the same rate as the arc motion
                # to maintain the same relative orientation to the door handle
                
                # Apply the same rotation to the initial gripper orientation
                # Create rotation matrix for the current arc angle about the arc rotation axis
                orientation_rotation_matrix = self._create_rotation_matrix(angle, arc_rotation_axis)
                
                # Convert current orientation to rotation object
                current_rot = Rotation.from_quat(current_orientation)
                
                # Create a rotation object from our rotation matrix
                arc_rotation = Rotation.from_matrix(orientation_rotation_matrix)
                
                # Apply the arc rotation to the initial orientation
                final_rotation = arc_rotation * current_rot
                final_orientation = final_rotation.as_quat()
                
                # Debug output for orientation
                if i == 0 or i == arc_segments // 2 or i == arc_segments:
                    print(f"  Orientation angle: {math.degrees(angle):.1f}°")
                    print(f"  Arc rotation matrix:\n{orientation_rotation_matrix}")
                    print(f"  Final orientation: {final_orientation}")
                
                orientations.append(final_orientation)
            
            print(f"Generated {len(waypoints)} waypoints for arc motion")
            
            # Plan trajectory through waypoints using existing motion planner
            success = True
            trajectory = None
            dt = None
            
            # For now, plan to each waypoint sequentially
            # TODO: Implement proper trajectory planning through multiple waypoints
            if len(waypoints) > 1:
                final_position = waypoints[-1]
                final_orientation = orientations[-1]
                
                # Plan motion to final waypoint
                goal_pose = CuroboPose(
                    position=self.tensor_args.to_device(torch.tensor([final_position], dtype=torch.float32)),
                    quaternion=self.tensor_args.to_device(torch.tensor([final_orientation], dtype=torch.float32)),
                    rotation=None
                )
                
                # Get current joint state
                with self.joint_state_lock:
                    if self.current_joints is None:
                        print("No current joint state available for arc planning")
                        return False, None, None
                    current_joints = self.current_joints
                
                start_state = CuroboJointState.from_position(
                    self.tensor_args.to_device([current_joints]),
                    joint_names=[f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)]
                )
                
                # Plan motion with optimized config - disable finetune for arc motion
                plan_config = self._get_fast_plan_config(
                    timeout=planning_timeout,
                    max_attempts=3,              # Increased attempts for arc motion
                    enable_graph=True,           # Enable graph for better success rate
                    enable_finetune=False        # Disable finetune to avoid failure
                )
                
                print(f"Planning arc motion to final position: {final_position}")
                result = self.motion_gen.plan_single(
                    start_state=start_state,
                    goal_pose=goal_pose,
                    plan_config=plan_config
                )
                
                if result.success.item():
                    print(f"Arc motion planning successful in {result.solve_time} seconds")
                    trajectory = result.get_interpolated_plan()
                    dt = trajectory.dt.cpu().numpy() if hasattr(trajectory, 'dt') else 0.01
                    
                    # Execute if requested
                    if execute and self.arm is not None:
                        print("Executing planned arc trajectory...")
                        execution_success = self.execute_trajectory(trajectory, dt, speed_factor=speed_factor)
                        if execution_success:
                            print("Arc trajectory execution completed successfully")
                        else:
                            print("Arc trajectory execution failed")
                            return False, trajectory, dt
                else:
                    print(f"Arc motion planning failed: {result.status}")
                    
                    # Fallback: try with smaller arc angle
                    if arc_angle > math.pi / 6:  # Only retry if current angle > 30 degrees
                        print("Retrying with smaller arc angle (30 degrees)...")
                        return self._plan_arc_motion_around_pivot(
                            pivot_point, current_position, current_orientation, radius,
                            is_push, arc_segments, planning_timeout, execute, speed_factor,
                            arc_angle=math.pi / 6  # 30 degree fallback
                        )
                    
                    success = False
            
            return success, trajectory, dt
            
        except Exception as e:
            print(f"Error in arc motion planning: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None, None


    def execute_pivot_pull_direct_xarm(
        self,
        pivot_point: np.ndarray,
        current_position: np.ndarray, 
        current_orientation: np.ndarray,
        radius: float,
        arc_angle_degrees: float = 30.0,
        segments: int = 10,
        speed_factor: float = 0.1,
        is_quat = False,
        hinge_location: str = None,
        is_push: bool = False
    ):
        """
        Execute pivot pull using direct xArm API commands, bypassing CuRobo planning.
        This is a fallback when CuRobo motion planning fails.
        
        Args:
            pivot_point: [x, y, z] coordinates of pivot center (hinge edge for hinge-based motion)
            current_position: [x, y, z] current TCP position (interaction point)
            current_orientation: [w, x, y, z] current TCP orientation
            radius: Distance from interaction point to hinge edge (for hinge-based motion)
            arc_angle_degrees: Arc angle in degrees (default 30)
            segments: Number of waypoints in the arc
            speed_factor: Speed factor for motion (default 0.1 for safety)
            hinge_location: Optional hinge location ('top', 'bottom', 'left', 'right') for logging
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            motion_type = f"hinge-based ({hinge_location})" if hinge_location else "legacy pivot"
            print(f"\n=== Direct xArm API Pivot Pull ({motion_type}) ===")
            print(f"Arc angle: {arc_angle_degrees}°, Segments: {segments}, Speed: {speed_factor}")
            if hinge_location:
                print(f"Hinge location: {hinge_location}, radius represents interaction point to hinge edge distance")
            
            if self.arm is None:
                print("Robot not connected")
                return False
            
            # Configure robot for pivot motion
            # self.configure_sensitivity_for_pivot()
            
            # Generate arc waypoints (reuse the working waypoint generation)
            arc_angle_rad = np.radians(arc_angle_degrees)
            
            # Ensure arrays are properly shaped and flattened
            current_pos_flat = np.array(current_position).flatten()
            pivot_point_flat = np.array(pivot_point).flatten()
            
            # Validate that we have 3D coordinates
            if len(current_pos_flat) != 3:
                print(f"Error: Current position must be 3D, got {len(current_pos_flat)} dimensions: {current_pos_flat}")
                return False
            if len(pivot_point_flat) != 3:
                print(f"Error: Pivot point must be 3D, got {len(pivot_point_flat)} dimensions: {pivot_point_flat}")
                return False
            
            pivot_to_grasp = current_pos_flat - pivot_point_flat
            # pivot_to_grasp *= 1.05
            actual_radius = np.linalg.norm(pivot_to_grasp)
            
            print(f"Pivot point: {pivot_point_flat}")
            print(f"Current position: {current_pos_flat}")
            print(f"Pivot vector: {pivot_to_grasp}")
            print(f"Actual radius: {actual_radius:.3f}m")
            print(f"Action type: {'push' if is_push else 'pull'}")
            print(f"Hinge location: {hinge_location}")
            
            # Determine rotation axis (Z for door opening)
            rotation_axis = np.array([0, 0, 1])
            
            # Test rotation direction (same logic as before)
            test_angle = np.radians(5)
            test_rotation = self._create_rotation_matrix(test_angle, rotation_axis)
            test_vector = test_rotation @ pivot_to_grasp
            test_position = pivot_point_flat + test_vector
            
            print(f"Test rotation: test_position={test_position}, current_pos={current_pos_flat}")
            print(f"X comparison: test_position[0]={test_position[0]:.3f}, current_pos[0]={current_pos_flat[0]:.3f}")
            
            # Base direction from test rotation
            base_direction = -1 if test_position[0] < current_pos_flat[0] else 1
            print(f"Base direction from test rotation: {base_direction}")
            
            # Override based on hinge location and action type
            if hinge_location == 'right':
                # For right hinge pull: need positive rotation to get negative X and negative Y movement
                pos_direction = 1 if not is_push else -1  # pull = 1, push = -1
            elif hinge_location == 'left':
                # For left hinge pull: negative X, positive Y movement (per user requirement)  
                pos_direction = -1 if not is_push else 1  # pull = -1, push = 1
            elif hinge_location == 'top':
                # For top hinge: pull = move down (negative Y), push = move up (positive Y)
                pos_direction = -1 if not is_push else 1  # pull = -1, push = 1
            elif hinge_location == 'bottom':
                # For bottom hinge: pull = move up (positive Y), push = move down (negative Y)
                pos_direction = 1 if not is_push else -1  # pull = 1, push = -1
            else:
                # Fallback to base direction
                pos_direction = base_direction
            # (Similar logic for top/bottom if needed)
            action_type = "push" if is_push else "pull"
            print(f"Using rotation direction: {pos_direction} ({action_type})")
            print(f"Direction logic: hinge={hinge_location}, is_push={is_push}, base_direction={base_direction}, final_direction={pos_direction}")
            print(f"Expected motion: {'backward' if pos_direction == -1 else 'forward'} for {action_type}")
            
            waypoints = []
            for i in range(segments + 1):
                t = i / segments
                angle = pos_direction * t * arc_angle_rad
                
                # Calculate position
                rotation_matrix = self._create_rotation_matrix(angle, rotation_axis)
                rotated_vector = rotation_matrix @ pivot_to_grasp
                position = pivot_point_flat + rotated_vector
                
                # DEBUG: For now, maintain fixed orientation to isolate position movement issue
                # The gripper orientation should remain constant while we debug the arc motion
                co = np.array(current_orientation).flatten()
                print(f"##############CO {co}")
                # Validate orientation array size and provide defaults if needed
                if len(co) == 4:  # Quaternion [w, x, y, z]
                    # Convert quaternion to Euler angles for xArm API
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_quat([co[1], co[2], co[3], co[0]])  # scipy uses [x,y,z,w] format
                    euler_angles = r.as_euler('xyz', degrees=True)
                    orientation = euler_angles.tolist()
                elif len(co) == 3:  # Already Euler angles
                    orientation = co.tolist()
                elif len(co) == 1:  # Single value - assume it's yaw only
                    orientation = [0.0, 0.0, float(co[0])]  # roll=0, pitch=0, yaw=co[0]
                else:
                    print(f"Warning: Unexpected orientation size {len(co)}, using default [0,0,0]")
                    orientation = [0.0, 0.0, 0.0]
                
                # TODO: Re-enable orientation tracking once position arc is working:
                orientation[2] += np.degrees(angle)  # Rotate around Z-axis (yaw) to maintain pointing toward pivot
                waypoints.append((position, orientation))
                print(f"Waypoint {i}: pos={position}, orientation= {orientation}, angle={np.degrees(angle):.1f}°")
            
            # Execute waypoints using direct xArm API (skip first waypoint since it's current position)
            print(f"Executing {len(waypoints)-1} waypoints with direct xArm API (skipping first waypoint)...")
            
            # Set pause time for smooth motion like the official example
            # self.arm.set_pause_time(0.5)  # 0.5 second pause between movements
            
            for i, (target_pos, target_rot) in enumerate(waypoints):
                # if i == 0:
                #     print(f"✓ Skipped waypoint {i} (current position)")
                #     continue
                try:
                    # Ensure robot is ready for motion  
                    if not self._ensure_robot_ready_for_motion():
                        print(f"Robot not ready for motion at waypoint {i}")
                        return False
                    
                    # Convert target quaternion to Euler angles for xArm API
                    from scipy.spatial.transform import Rotation as R
                    try:
                        if is_quat:
                            # Handle quaternion conversion properly
                            if hasattr(target_rot, 'cpu'):  # torch tensor
                                quat = target_rot.cpu().numpy().flatten()
                            else:
                                quat = np.array(target_rot).flatten()
                            
                            # Ensure we have 4 elements [x, y, z, w]
                            if len(quat) == 4:
                                r = R.from_quat(quat)
                                roll, pitch, yaw = r.as_euler('xyz', degrees=True)
                                print(f"Using calculated orientation: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
                            else:
                                raise ValueError(f"Expected 4-element quaternion, got {len(quat)} elements")
                        else:
                            roll, pitch, yaw = target_rot
                    except Exception as e:
                        print(f"Error converting quaternion to Euler: {e}")
                        # Fallback to current TCP pose orientation
                        tcp_result = self.arm.get_position()
                        if tcp_result[0] != 0:
                            print(f"Failed to get TCP pose at waypoint {i}, error: {tcp_result[0]}, using default orientation")
                            roll, pitch, yaw = 0, 0, 0
                        else:
                            current_tcp = tcp_result[1]
                            roll, pitch, yaw = current_tcp[3], current_tcp[4], current_tcp[5]
                            print(f"Fallback TCP pose: {current_tcp}, using roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
                    
                    # Move to waypoint using Cartesian motion
                    target_x = target_pos[0] * 1000  # Convert m to mm
                    target_y = target_pos[1] * 1000  
                    target_z = target_pos[2] * 1000
                    move_speed = max(10, int(50 * speed_factor))
                    move_acc = max(50, int(100 * speed_factor))
                    
                    print(f"Moving to: x={target_x:.1f}mm, y={target_y:.1f}mm, z={target_z:.1f}mm")
                    print(f"With orientation: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
                    print(f"Speed: {move_speed}mm/s, Acc: {move_acc}mm/s²")
                    # Wait until robot is in ready state
                    while self.arm.get_state()[1] not in [0, 2, 3]:
                        time.sleep(0.5)
                        self.arm.set_state(0)
                        print(f"Waiting for robot ready state: {self.arm.get_state()}")
                        
                    # Execute the movement without torque monitoring for better performance
                    code = self._execute_standard_waypoint(
                        target_x, target_y, target_z, roll, pitch, yaw,
                        move_speed, move_acc
                    )
                    
                    print(f"set_position returned code: {code}")
                    
                    if code != 0:
                        print(f"Failed at waypoint {i}, error code: {code}")
                        return False
                    
                    print(f"✓ Completed waypoint {i}/{len(waypoints)-1} ({100*i/(len(waypoints)-1):.1f}%)")
                    
                    # Small pause between waypoints to reduce accumulated error
                    if i < len(waypoints) - 1:
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error at waypoint {i}: {e}")
                    return False
            
            print("✓ Direct xArm pivot pull completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in direct xArm pivot pull: {e}")
            return False
            
        finally:
            # Always restore default sensitivity
            self.restore_default_sensitivity()



    def _configure_initial_sensitivity(self) -> bool:
        """
        Configure initial robot sensitivity settings for normal operations
        
        Returns:
            bool: True if configuration successful, False otherwise
        """
        try:
            if self.arm is None:
                print("Warning: Robot not connected, cannot configure initial sensitivity")
                return False
                
            with self.arm_lock:
                # Set default collision sensitivity
                # collision_result = self.arm.set_collision_sensitivity(self.default_collision_sensitivity, wait=True)
                # if collision_result != 0:
                #     print(f"Warning: Failed to set initial collision sensitivity (code: {collision_result})")
                    
                # Set default teach sensitivity
                teach_result = self.arm.set_teach_sensitivity(self.default_teach_sensitivity, wait=True)
                if teach_result != 0:
                    print(f"Warning: Failed to set initial teach sensitivity (code: {teach_result})")
                
                print(f"✓ Configured initial sensitivity: collision={self.default_collision_sensitivity}, teach={self.default_teach_sensitivity}")
                return True
                
        except Exception as e:
            print(f"Error configuring initial sensitivity: {e}")
            return False


