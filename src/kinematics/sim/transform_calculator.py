"""
Camera-to-robot transform calculator using PyBullet.

Uses the robot URDF to automatically compute the transformation
from camera frame to robot base frame.
"""

import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

# Optional PyBullet import
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None


class TransformCalculator:
    """
    Calculate camera-to-robot transform using PyBullet and URDF.

    This uses PyBullet's forward kinematics to get the exact transform
    from the camera optical frame to the robot base frame.
    """

    def __init__(self, urdf_path: str, camera_link_name: str = "camera_color_optical_frame",
                 use_gui: bool = False):
        """
        Initialize transform calculator.

        Args:
            urdf_path: Path to robot URDF file
            camera_link_name: Name of camera link in URDF (optical frame for RealSense)
            use_gui: If True, use PyBullet GUI for visualization (default: False, headless)
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is required for transform calculation. "
                "Install with: pip install pybullet"
            )

        self.urdf_path = urdf_path
        self.camera_link_name = camera_link_name
        self.physics_client = None
        self.robot_id = None
        self.base_link_name = "link_base"
        self.use_gui = use_gui

        # Verify URDF exists
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        logging.info(f"Transform calculator initialized with URDF: {urdf_path}")
        if use_gui:
            logging.info("GUI visualization enabled")

    def _init_pybullet(self):
        """Initialize PyBullet in DIRECT or GUI mode."""
        if self.physics_client is not None:
            return

        # Connect to PyBullet (GUI or headless)
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            # Set up camera view for GUI
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.3]
            )
            # Add ground plane
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
        else:
            self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True
        )

        logging.info(f"Loaded robot URDF in PyBullet (robot_id={self.robot_id}, GUI={self.use_gui})")

    def _get_link_index(self, link_name: str) -> Optional[int]:
        """Get link index by name."""
        num_joints = p.getNumJoints(self.robot_id)

        # Base link has index -1
        if link_name == self.base_link_name:
            return -1

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            # info[12] is the link name
            if info[12].decode('utf-8') == link_name:
                return i

        return None

    def _get_link_state(self, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get link position and orientation in world frame.

        Returns:
            position (3,), orientation quaternion (4,) [x, y, z, w]
        """
        if link_index == -1:
            # Base link - get base position/orientation
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        else:
            # Other links - get link state
            link_state = p.getLinkState(self.robot_id, link_index)
            pos = link_state[4]  # World link frame position
            orn = link_state[5]  # World link frame orientation

        return np.array(pos), np.array(orn)

    def get_camera_to_base_transform(self, joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get 4x4 transformation matrix from camera frame to robot base frame.

        Args:
            joint_positions: Optional joint angles (radians). If None, uses zero position.

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Initialize PyBullet if needed
        self._init_pybullet()

        # Set joint positions if provided (only for movable joints)
        if joint_positions is not None:
            # Get movable joint indices (skip fixed joints like camera mounts)
            movable_joints = []
            for i in range(p.getNumJoints(self.robot_id)):
                joint_info = p.getJointInfo(self.robot_id, i)
                joint_type = joint_info[2]
                joint_name = joint_info[1].decode('utf-8')
                # Joint types: JOINT_REVOLUTE=0, JOINT_PRISMATIC=1, JOINT_FIXED=4
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    movable_joints.append((i, joint_name))

            logging.debug(f"Found {len(movable_joints)} movable joints")

            # Set joint angles for movable joints only
            for i, (joint_idx, joint_name) in enumerate(movable_joints):
                if i < len(joint_positions):
                    p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
                    logging.debug(f"  Set joint {joint_idx} ({joint_name}): {joint_positions[i]:.4f} rad")

        # Get camera link index
        camera_idx = self._get_link_index(self.camera_link_name)
        if camera_idx is None:
            raise ValueError(f"Camera link '{self.camera_link_name}' not found in URDF")

        # Get base link index
        base_idx = self._get_link_index(self.base_link_name)

        # Get transforms in world frame
        camera_pos_world, camera_orn_world = self._get_link_state(camera_idx)
        base_pos_world, base_orn_world = self._get_link_state(base_idx)

        # Convert quaternions to rotation matrices
        camera_rot_world = np.array(p.getMatrixFromQuaternion(camera_orn_world)).reshape(3, 3)
        base_rot_world = np.array(p.getMatrixFromQuaternion(base_orn_world)).reshape(3, 3)

        # Build homogeneous transforms
        T_world_camera = np.eye(4)
        T_world_camera[:3, :3] = camera_rot_world
        T_world_camera[:3, 3] = camera_pos_world

        T_world_base = np.eye(4)
        T_world_base[:3, :3] = base_rot_world
        T_world_base[:3, 3] = base_pos_world

        # Compute T_base_camera = T_base_world @ T_world_camera
        T_base_world = np.linalg.inv(T_world_base)
        T_base_camera = T_base_world @ T_world_camera

        logging.debug(f"Camera to base transform computed at joint config: {joint_positions}")
        logging.debug(f"Transform:\n{T_base_camera}")

        return T_base_camera

    def get_transform_at_current_state(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Get camera-to-base transform for current robot configuration.

        Args:
            joint_positions: Current joint angles (radians)

        Returns:
            4x4 transformation matrix from camera to base
        """
        return self.get_camera_to_base_transform(joint_positions)

    def cleanup(self):
        """Cleanup PyBullet connection."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            logging.debug("PyBullet connection closed")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


def get_urdf_path() -> str:
    """Get path to xArm7 URDF file."""
    # Assuming this file is in xarm_manipulation/perception/
    current_dir = Path(__file__).parent
    urdf_path = current_dir.parent / "sim" / "xarm7.urdf"
    return str(urdf_path.resolve())


def create_transform_calculator(camera_link: str = "camera_color_optical_frame",
                               use_gui: bool = True) -> TransformCalculator:
    """
    Create a TransformCalculator with default xArm7 URDF.

    Args:
        camera_link: Name of camera link in URDF
        use_gui: If True, visualize in PyBullet GUI

    Returns:
        TransformCalculator instance
    """
    urdf_path = get_urdf_path()
    return TransformCalculator(urdf_path, camera_link, use_gui=use_gui)


if __name__ == "__main__":
    # Test the transform calculator
    logging.basicConfig(level=logging.INFO)

    print("Testing TransformCalculator...")

    # Create calculator
    calc = create_transform_calculator()

    # Get transform at zero position
    print("\n1. Transform at zero position:")
    T = calc.get_camera_to_base_transform()
    print(T)

    # Get transform at a different configuration
    print("\n2. Transform at different joint angles:")
    joint_angles = np.array([0, 0.5, 0, 0.5, 0, 0.5, 0])  # Example configuration
    T2 = calc.get_camera_to_base_transform(joint_angles)
    print(T2)

    # Extract position from transform
    camera_pos_in_base = T[:3, 3]
    print(f"\nCamera position in base frame: {camera_pos_in_base}")
    print(f"Distance from base: {np.linalg.norm(camera_pos_in_base):.3f} meters")

    calc.cleanup()
    print("\n✓ Test completed")
