"""
Depth-image-based environment collision for PyBullet motion planning.

Converts a RealSense depth frame into per-object triangle-mesh collision bodies
plus a background body for the unmasked environment, so the planner can
selectively enable or disable individual objects during collision checking
(e.g. ignore the target object while planning a grasp).

The camera extrinsic is read from the XArmPybulletInterface FK at update time
so it automatically tracks the wrist-mounted camera as the arm moves.

Typical usage::

    collider = DepthEnvironmentCollider(planner)

    # Build one body per detected object + a background body
    collider.update(camera, masks)          # masks: {object_id: bool H×W ndarray}

    # Check trajectory against everything except the grasp target
    hit = collider.check_trajectory(trajectory, ignore={"red_cup_1"})

    # Or check only background (ignore all objects)
    hit = collider.check_trajectory(trajectory, ignore=set(collider.object_ids))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

try:
    import pybullet as p
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

from src.camera.base_camera import BaseCamera, CameraIntrinsics

if TYPE_CHECKING:
    from src.kinematics.base_pybullet_interface import BasePybulletInterface

logger = logging.getLogger(__name__)

_DEPTH_MIN_M = 0.15
_DEPTH_MAX_M = 2.0
_DEPTH_STRIDE = 4
_BPA_RADIUS_FACTOR = 3.0
_COLLISION_MARGIN_M = 0.005

# Label used for the body covering all non-object depth pixels
BACKGROUND_ID = "__background__"


class DepthEnvironmentCollider:
    """Per-object + background mesh collision bodies built from a live depth image.

    Each detected object gets its own PyBullet body (masked depth pixels).
    All remaining depth pixels form a single background body.  This lets the
    planner ignore specific objects during collision checking.

    Args:
        planner: XArmPybulletInterface (or any BasePybulletInterface subclass).
        collision_margin_m: Distance (metres) below which a robot link is
            considered in collision with a mesh body.
    """

    def __init__(
        self,
        planner: BasePybulletInterface,
        collision_margin_m: float = _COLLISION_MARGIN_M,
    ) -> None:
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required: pip install open3d")
        if not _PYBULLET_AVAILABLE:
            raise ImportError("pybullet is required: pip install pybullet")

        self._planner = planner
        self._margin = float(collision_margin_m)
        # object_id (or BACKGROUND_ID) -> pybullet body ID
        self._bodies: Dict[str, int] = {}
        self._cam_pos_world: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def object_ids(self) -> list[str]:
        """Object IDs for which individual bodies exist (excludes background)."""
        return [k for k in self._bodies if k != BACKGROUND_ID]

    @property
    def background_body_id(self) -> Optional[int]:
        """PyBullet body ID of the background mesh, or None."""
        return self._bodies.get(BACKGROUND_ID)

    def body_id_for(self, object_id: str) -> Optional[int]:
        """Return the PyBullet body ID for a specific object, or None."""
        return self._bodies.get(object_id)

    def update(
        self,
        camera: BaseCamera,
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        """Capture a depth frame and rebuild all collision bodies.

        Args:
            camera: RealSenseCamera (or any BaseCamera with depth support).
            masks: {object_id: bool H×W mask} from GSAM2 / _last_masks.

        Returns:
            Dict mapping each body label (object_id + BACKGROUND_ID) to
            True if its mesh was successfully built.
        """
        T_base_cam = self._get_camera_extrinsic()
        if T_base_cam is None:
            logger.warning("DepthEnvironmentCollider: cannot read camera extrinsic")
            return {}
        intrinsics = camera.get_camera_intrinsics()
        _, depth = camera.get_aligned_frames()
        return self._rebuild(depth, intrinsics, T_base_cam, masks)

    def update_from_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        """Rebuild from an already-captured depth array.

        Args:
            depth_m: (H, W) float32 depth image in metres.
            intrinsics: Camera intrinsic parameters.
            masks: {object_id: bool H×W mask}.

        Returns:
            Dict mapping each label to True if its mesh was built successfully.
        """
        T_base_cam = self._get_camera_extrinsic()
        if T_base_cam is None:
            logger.warning("DepthEnvironmentCollider: cannot read camera extrinsic")
            return {}
        return self._rebuild(depth_m, intrinsics, T_base_cam, masks)

    def check_trajectory(
        self,
        trajectory: np.ndarray,
        ignore: Optional[Set[str]] = None,
    ) -> Optional[int]:
        """Check each waypoint of a planned trajectory against the depth-mesh bodies.

        Floor and self-collision are handled by the OMPL planner's validity
        checker during path generation.  This method is used post-planning
        to verify the trajectory against depth-mesh obstacles (which may not
        have been present when the trajectory was planned, e.g. in the grasp
        sampler when ignoring the target object).

        Args:
            trajectory: (N, dof) array of joint configurations in radians.
            ignore: Set of object_id labels (or BACKGROUND_ID) to skip.

        Returns:
            Index of the first colliding waypoint, or None if collision-free.
        """
        active_bodies = {
            label: bid for label, bid in self._bodies.items()
            if (ignore is None or label not in ignore)
        }
        if not active_bodies:
            return None

        client = self._planner._physics_client
        rid = self._planner._robot_id
        movable = self._planner._movable_joints
        n_links = p.getNumJoints(rid, physicsClientId=client)
        saved_joints = self._planner._joints.copy()
        hit: Optional[int] = None

        try:
            for step_idx, joints in enumerate(trajectory):
                for j, joint_idx in enumerate(movable):
                    if j < len(joints):
                        p.resetJointState(rid, joint_idx, float(joints[j]),
                                          physicsClientId=client)

                for label, body_id in active_bodies.items():
                    for link_idx in range(-1, n_links):
                        contacts = p.getClosestPoints(
                            bodyA=rid,
                            bodyB=body_id,
                            distance=self._margin,
                            linkIndexA=link_idx,
                            physicsClientId=client,
                        )
                        if contacts:
                            logger.debug(
                                "Mesh collision at waypoint %d body=%s link=%d dist=%.4f",
                                step_idx, label, link_idx, contacts[0][8],
                            )
                            hit = step_idx
                            break
                    if hit is not None:
                        break

                if hit is not None:
                    break
        finally:
            for j, joint_idx in enumerate(movable):
                if j < len(saved_joints):
                    p.resetJointState(rid, joint_idx, float(saved_joints[j]),
                                      physicsClientId=client)

        return hit

    def remove(self, label: Optional[str] = None) -> None:
        """Remove one or all mesh bodies from PyBullet.

        Args:
            label: If given, remove only that body (object_id or BACKGROUND_ID).
                   If None, remove all bodies.
        """
        client = self._planner._physics_client
        to_remove = [label] if label is not None else list(self._bodies.keys())
        for lbl in to_remove:
            body_id = self._bodies.pop(lbl, None)
            if body_id is not None:
                try:
                    p.removeBody(body_id, physicsClientId=client)
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.remove()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_extrinsic(self) -> Optional[np.ndarray]:
        cam_pos, cam_rot = self._planner.get_camera_transform()
        if cam_pos is None or cam_rot is None:
            return None
        T = np.eye(4)
        T[:3, :3] = cam_rot.as_matrix()
        T[:3, 3] = cam_pos
        return T

    def _rebuild(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_base_cam: np.ndarray,
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        self.remove()
        self._cam_pos_world = T_base_cam[:3, 3].copy()

        h, w = depth_m.shape
        results: Dict[str, bool] = {}

        # Union of all object masks — pixels NOT in any mask become background
        union_mask = np.zeros((h, w), dtype=bool)
        for mask in masks.values():
            if mask.shape == (h, w):
                union_mask |= mask.astype(bool)

        # Build one body per detected object
        for obj_id, mask in masks.items():
            if mask.shape != (h, w):
                logger.warning("DepthEnvironmentCollider: mask shape mismatch for %s", obj_id)
                results[obj_id] = False
                continue
            pts = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                               pixel_mask=mask.astype(bool))
            results[obj_id] = self._build_body(pts, obj_id)

        # Build background body from all non-object pixels
        background_mask = ~union_mask
        pts_bg = self._depth_to_world_points(depth_m, intrinsics, T_base_cam,
                                              pixel_mask=background_mask)
        results[BACKGROUND_ID] = self._build_body(pts_bg, BACKGROUND_ID)

        built = [lbl for lbl, ok in results.items() if ok]
        logger.info(
            "DepthEnvironmentCollider: built %d bodies (%s)",
            len(built), ", ".join(built),
        )
        return results

    def _depth_to_world_points(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_base_cam: np.ndarray,
        pixel_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Backproject masked depth pixels to world-frame 3-D points."""
        h, w = depth_m.shape
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        rows = np.arange(0, h, _DEPTH_STRIDE)
        cols = np.arange(0, w, _DEPTH_STRIDE)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        # Subsample the pixel mask to match the stride grid
        sampled_mask = pixel_mask[rr, cc]
        d = depth_m[rr, cc]

        valid = sampled_mask & (d > _DEPTH_MIN_M) & (d < _DEPTH_MAX_M)
        if not np.any(valid):
            return None

        d_v = d[valid].astype(float)
        x = (cc[valid].astype(float) - cx) * d_v / fx
        y = (rr[valid].astype(float) - cy) * d_v / fy
        z = d_v
        pts_cam = np.stack([x, y, z], axis=1)

        R = T_base_cam[:3, :3]
        t = T_base_cam[:3, 3]
        return ((R @ pts_cam.T).T + t).astype(np.float32)

    def _build_body(self, pts: Optional[np.ndarray], label: str) -> bool:
        """Reconstruct mesh from points and add it to PyBullet. Returns success."""
        if pts is None or len(pts) < 10:
            logger.debug("DepthEnvironmentCollider: too few points for %s (%s)",
                         label, len(pts) if pts is not None else 0)
            return False

        mesh = self._points_to_mesh(pts)
        if mesh is None:
            logger.debug("DepthEnvironmentCollider: mesh failed for %s", label)
            return False

        body_id = self._mesh_to_pybullet(mesh)
        if body_id is None:
            return False

        self._bodies[label] = body_id
        logger.info(
            "DepthEnvironmentCollider: [%s] body=%d  triangles=%d  points=%d",
            label, body_id, len(np.asarray(mesh.triangles)), len(pts),
        )
        return True

    def _points_to_mesh(self, pts: np.ndarray) -> Optional[object]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        if self._cam_pos_world is not None:
            pcd.orient_normals_towards_camera_location(
                camera_location=self._cam_pos_world.tolist()
            )

        distances = pcd.compute_nearest_neighbor_distance()
        if len(distances) == 0:
            return None
        radius = _BPA_RADIUS_FACTOR * float(np.mean(distances))

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]),
        )
        if len(mesh.triangles) == 0:
            return None

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        return mesh

    def _mesh_to_pybullet(self, mesh) -> Optional[int]:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        if len(verts) == 0 or len(tris) == 0:
            return None
        try:
            col_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices=verts.tolist(),
                indices=tris.flatten().tolist(),
                physicsClientId=self._planner._physics_client,
            )
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self._planner._physics_client,
            )
        except Exception as exc:
            logger.warning("DepthEnvironmentCollider: pybullet mesh creation failed: %s", exc)
            return None
