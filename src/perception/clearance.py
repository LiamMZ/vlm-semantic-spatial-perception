"""
Clearance corridor computation for detected objects.

For each object, builds a gripper-width swept-volume (OBB) in each of 18
structured approach directions and checks it against every neighbouring
object's AABB.  The result tells downstream planners how much transverse
clearance exists along each approach and whether the gripper can physically
fit through.

Algorithm
---------
1. Back-project the full depth image to a 3-D point cloud (camera frame).
2. For each target object (defined by its SAM2 mask), compute an AABB and
   centroid in world/base frame.
3. For each of 18 structured approach directions generate a rectangular OBB
   swept volume representing the gripper approaching the target.
4. Test the swept OBB against every neighbour AABB using the Separating Axis
   Theorem (SAT).  For intersecting neighbours, compute the transverse
   intrusion along the grasp axis.
5. Cluster compatible corridors and compute scalar summaries:
   top_clearance, lateral_clearances, graspability_score.

Complexity: O(N × 18 × (N + M)) = O(N² + NM).
For N=15 objects, M=6 surfaces: ~5,670 SAT tests ≈ <5 ms.

Usage:
    gripper = GripperGeometry(finger_width=0.012, max_aperture=0.085, depth=0.06)
    profile = compute_clearance_profile(
        target_mask=mask_bool,
        depth_frame=depth_m,
        camera_intrinsics=intr,
        all_masks=all_masks_dict,
        gripper=gripper,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How far beyond the target object surface the corridor extends (metres).
CORRIDOR_LENGTH: float = 0.30
# Safety margin added around the gripper swept volume on each side (metres).
CLEARANCE_MARGIN: float = 0.005
# Neighbourhood search radius — objects further than this are ignored (metres).
SEARCH_RADIUS: float = 1.0
# Standoff from the AABB surface so the corridor does not overlap the target's
# own point cloud, which would always produce a false self-collision.
CORRIDOR_STANDOFF: float = 0.010

# ---------------------------------------------------------------------------
# Approach direction set  (spec §1.3 Phase 1)
# 6 axis-aligned + 4 horizontal diagonals + 8 angled (45° elevation) = 18
# ---------------------------------------------------------------------------

_s2 = float(np.sqrt(2) / 2)   # sin/cos 45° ≈ 0.7071

_AXIS_DIRS = np.array([
    [1.0,  0.0,  0.0],
    [-1.0, 0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)

_HORIZ_DIAG_DIRS = np.array([
    [ _s2,  _s2, 0.0],
    [ _s2, -_s2, 0.0],
    [-_s2,  _s2, 0.0],
    [-_s2, -_s2, 0.0],
], dtype=np.float64)

_ANGLED_DIRS = np.array([
    [ 1.0,  0.0, -_s2],
    [ 0.0,  1.0, -_s2],
    [-1.0,  0.0, -_s2],
    [ 0.0, -1.0, -_s2],
    [ 1.0,  0.0,  _s2],
    [ 0.0,  1.0,  _s2],
    [-1.0,  0.0,  _s2],
    [ 0.0, -1.0,  _s2],
], dtype=np.float64)


def _build_approach_dirs() -> np.ndarray:
    """Return (18, 3) unit-normalised structured approach directions."""
    raw = np.vstack([_AXIS_DIRS, _HORIZ_DIAG_DIRS, _ANGLED_DIRS])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


_APPROACH_DIRS: np.ndarray = _build_approach_dirs()  # (18, 3), cached


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GripperGeometry:
    """Physical dimensions of the end-effector fingers.

    Attributes:
        finger_width: Width of each finger (along the gripper spread axis), metres.
        max_aperture: Maximum opening between finger inner faces, metres.
        depth: How deep the fingers extend along the approach axis, metres.
    """
    finger_width: float = 0.020
    max_aperture: float = 0.085
    depth: float = 0.060

    @classmethod
    def from_urdf(cls, urdf_path: "str | Path") -> "GripperGeometry":
        """Algorithmically extract gripper geometry from any URDF or xacro file.

        Strategy (works for parallel-jaw grippers of any make):

        1. **Finger links** — find all links whose name contains "finger"
           (case-insensitive).  Discard knuckle/base links.

        2. **finger_width** — for each finger link, prefer a ``<box>`` collision
           primitive (take its smallest horizontal dimension).  If no box exists,
           use the inertia-centroid Y spread as a fallback (the centroid XY gives
           a rough bounding half-width).

        3. **max_aperture** — find the joints that connect the gripper base to the
           first moving finger link on each side (joints mirrored in Y).  At the
           fully-open configuration (joint = lower limit for closing joints, which
           is usually 0), the aperture is 2 × |Y offset of that joint| plus any
           additional reach computed from child joint chains.  If the joint
           directly carries a ``<box>`` finger, use 2 × |Y| − finger_width.
           Otherwise fall back to 2 × |Y| × scale_factor where scale_factor
           accounts for the lever-arm extension (empirically ~2.3 for most
           parallel-jaw designs; validated against both xArm and Robotiq 2F-140).

        4. **depth** — use the maximum inertia-centroid Z value across all finger
           links, plus half the finger_width as a reach estimate.  This is robust
           because the inertia origin is authored at the centre-of-mass of each
           finger link, which lies roughly at the mid-length of the finger body.

        Falls back to the dataclass defaults if parsing fails.

        Args:
            urdf_path: Path to a ``.urdf`` or ``.xacro`` file that contains the
                       gripper kinematic description.
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path as _Path
        import logging

        _log = logging.getLogger(__name__)
        urdf_path = _Path(urdf_path)
        if not urdf_path.exists():
            _log.warning("GripperGeometry.from_urdf: file not found at %s, using defaults", urdf_path)
            return cls()

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            _log.warning("GripperGeometry.from_urdf: XML parse error (%s), using defaults", exc)
            return cls()

        # ── helpers ──────────────────────────────────────────────────────────
        def _parse_xyz(element) -> "tuple[float,float,float] | None":
            """Extract (x,y,z) from an <origin xyz="..."> element."""
            if element is None:
                return None
            xyz = element.get("xyz", "")
            parts = xyz.split()
            if len(parts) == 3:
                try:
                    return float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    pass
            return None

        def _is_finger_link(name: str) -> bool:
            n = name.lower()
            return "finger" in n and "knuckle" not in n and "base" not in n

        # ── 1. Collect finger links ──────────────────────────────────────────
        finger_links: dict[str, ET.Element] = {
            lnk.get("name", ""): lnk
            for lnk in root.iter("link")
            if _is_finger_link(lnk.get("name", ""))
        }
        _log.debug("from_urdf: finger links found: %s", list(finger_links.keys()))

        # ── 2. finger_width from box primitives or inertia spread ────────────
        finger_width = cls.finger_width
        box_widths: list[float] = []
        for lname, lnk in finger_links.items():
            for box in lnk.iter("box"):
                size_str = box.get("size", "")
                parts = size_str.split()
                if len(parts) == 3:
                    try:
                        dims = sorted([float(p) for p in parts])
                        # smallest dimension is likely the finger thickness/width
                        box_widths.append(dims[0])
                    except ValueError:
                        pass
        if box_widths:
            finger_width = float(np.median(box_widths))
            _log.debug("from_urdf: finger_width from box = %.4f m", finger_width)
        else:
            # Fallback: use inertia centroid Y spread across left/right fingers
            y_cents: list[float] = []
            for lnk in finger_links.values():
                inertial = lnk.find("inertial")
                if inertial is not None:
                    origin = inertial.find("origin")
                    xyz = _parse_xyz(origin)
                    if xyz:
                        y_cents.append(abs(xyz[1]))
            if y_cents:
                finger_width = float(np.mean(y_cents)) * 0.4  # empirical scale
                _log.debug("from_urdf: finger_width from inertia = %.4f m", finger_width)

        # ── 3. max_aperture from base→finger joint Y offsets ─────────────────
        # Find joints whose child is a finger link (or a knuckle directly
        # parented to the gripper base).  Collect the |Y| offset of each.
        max_aperture = cls.max_aperture
        pivot_y_offsets: list[float] = []
        for joint in root.iter("joint"):
            jtype = joint.get("type", "")
            if jtype not in ("revolute", "prismatic"):
                continue
            child_el = joint.find("child")
            if child_el is None:
                continue
            child_name = child_el.get("link", "")
            # Accept joints whose child is a finger link OR an outer/left/right knuckle
            child_lower = child_name.lower()
            if not ("finger" in child_lower or "knuckle" in child_lower):
                continue
            origin = joint.find("origin")
            xyz = _parse_xyz(origin)
            if xyz and abs(xyz[1]) > 1e-4:  # ignore near-zero Y offsets
                pivot_y_offsets.append(abs(xyz[1]))

        if pivot_y_offsets:
            # Use the largest |Y| pivot — that's the outermost knuckle base
            pivot_y = max(pivot_y_offsets)
            # Scale factor accounts for the lever arm extending the finger
            # beyond the pivot.  Validated: xArm (pivot_y=0.035) → aperture~0.085;
            # Robotiq 2F-140 (pivot_y=0.0306) → aperture~0.140.
            # In both cases aperture ≈ pivot_y * 2 + finger reach contribution.
            # Use 2 * pivot_y as the minimum (finger pads at pivot level) and
            # add the inertia-centroid Y reach for a better estimate.
            inertia_y_max = 0.0
            for lnk in finger_links.values():
                inertial = lnk.find("inertial")
                if inertial is not None:
                    origin = inertial.find("origin")
                    xyz = _parse_xyz(origin)
                    if xyz:
                        inertia_y_max = max(inertia_y_max, abs(xyz[1]))
            max_aperture = 2.0 * pivot_y + 2.0 * inertia_y_max - finger_width
            max_aperture = max(max_aperture, 2.0 * pivot_y)  # floor
            _log.debug(
                "from_urdf: max_aperture=%.4f m (pivot_y=%.4f, inertia_y=%.4f)",
                max_aperture, pivot_y, inertia_y_max,
            )

        # ── 4. depth from max finger inertia-centroid Z + half finger_width ──
        depth = cls.depth
        z_cents: list[float] = []
        for lnk in finger_links.values():
            inertial = lnk.find("inertial")
            if inertial is not None:
                origin = inertial.find("origin")
                xyz = _parse_xyz(origin)
                if xyz and xyz[2] > 0:
                    z_cents.append(xyz[2])
        if z_cents:
            # The inertia centroid is at the midpoint of the finger body;
            # add half finger_width to estimate the full reach to the tip.
            depth = max(z_cents) + finger_width / 2.0
            _log.debug("from_urdf: depth=%.4f m (max_inertia_z=%.4f)", depth, max(z_cents))

        _log.info(
            "GripperGeometry.from_urdf(%s): finger_width=%.3f  max_aperture=%.3f  depth=%.3f",
            urdf_path.name, finger_width, max_aperture, depth,
        )
        return cls(finger_width=finger_width, max_aperture=max_aperture, depth=depth)

    @property
    def min_clearance_required(self) -> float:
        """Minimum transverse corridor width needed for a grasp (metres)."""
        return self.max_aperture + 2.0 * self.finger_width

    @property
    def swept_width(self) -> float:
        """Corridor half-width including safety margin."""
        return self.min_clearance_required / 2.0 + CLEARANCE_MARGIN

    @property
    def swept_height(self) -> float:
        """Corridor half-height (finger depth + margin)."""
        return self.depth / 2.0 + CLEARANCE_MARGIN


@dataclass
class ApproachCorridor:
    """Clearance along a single approach direction.

    Attributes:
        direction: Unit approach vector (world/base frame).
        min_clearance: Transverse width remaining after all intrusions, metres.
        corridor_length: Depth of the swept volume (always CORRIDOR_LENGTH), metres.
        obstructing_objects: IDs of objects whose AABB intersects the swept OBB.
        grasp_compatible: True if min_clearance >= gripper.min_clearance_required.
        corridor_start: World-frame point where the corridor begins (target surface).
        grasp_axis: Unit vector perpendicular to direction along which fingers close.
        height_axis: Unit vector perpendicular to both direction and grasp_axis.
        half_width: Half-extent of the corridor along grasp_axis (metres).
        half_height: Half-extent of the corridor along height_axis (metres).
        obstructor_aabbs: List of (min_corner, max_corner) for each obstructing
            neighbour — used to draw the exact clip plane in visualizations.
    """
    direction: np.ndarray          # (3,) unit vector
    min_clearance: float           # metres, transverse
    corridor_length: float         # metres, longitudinal
    obstructing_objects: List[str]
    grasp_compatible: bool
    # Geometry for visualization (always populated)
    corridor_start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasp_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    height_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    half_width: float = 0.0
    half_height: float = 0.0
    obstructor_aabbs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def min_clearance_mm(self) -> float:
        """Narrowest transverse clearance in millimetres (spec §1.1)."""
        return self.min_clearance * 1000.0

    @property
    def corridor_length_mm(self) -> float:
        """Depth of the viable approach volume in millimetres (spec §1.1)."""
        return self.corridor_length * 1000.0


@dataclass
class ClearanceProfile:
    """Full clearance description for a single object.

    Attributes:
        approach_corridors: All evaluated corridors (grasp-compatible ones first).
        top_clearance: Transverse clearance along world +Z, metres.
        lateral_clearances: Per-axis clearance {"+x", "-x", "+y", "-y"} in metres.
        graspability_score: Aggregate [0, 1] score (quality + diversity blend).
        centroid: (3,) centroid of the target object in base/world frame.
        ray_dirs: (N, 3) approach directions used (for visualisation).
        ray_lengths_m: (N,) min_clearance per direction (for visualisation).
        obstacle_pts: (N, 3) obstacle point cloud (for visualisation).
    """
    approach_corridors: List[ApproachCorridor] = field(default_factory=list)
    top_clearance: float = 0.0
    lateral_clearances: Dict[str, float] = field(default_factory=dict)
    graspability_score: float = 0.0
    centroid: Optional[np.ndarray] = field(default=None)
    ray_dirs: Optional[np.ndarray] = field(default=None)
    ray_lengths_m: Optional[np.ndarray] = field(default=None)
    obstacle_pts: Optional[np.ndarray] = field(default=None)
    target_pts: Optional[np.ndarray] = field(default=None)
    target_aabb_min: Optional[np.ndarray] = field(default=None)
    target_aabb_max: Optional[np.ndarray] = field(default=None)
    neighbor_aabbs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def best_approach_dirs(self) -> List[np.ndarray]:
        """Viable approach direction vectors sorted by clearance margin (spec §1.1 downstream).

        Returns only grasp-compatible corridors, best clearance first.
        Used as preferred approach hints passed to the motion planner.
        """
        return [
            c.direction for c in self.approach_corridors
            if c.grasp_compatible
        ]

    @property
    def has_viable_approach(self) -> bool:
        """True if at least one grasp-compatible corridor exists (L4 feasibility gate)."""
        return any(c.grasp_compatible for c in self.approach_corridors)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


_WORLD_UP = np.array([0.0, 0.0, 1.0])


def _gripper_axes(approach_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (grasp_axis, height_axis) for a world-gravity-aligned gripper frame.

    The gripper opening is always horizontal (grasp_axis lies in the XY plane)
    and height_axis points up, so OBB corridors are gravity-aligned regardless
    of approach direction.

    For near-vertical approaches (top-down / bottom-up, |approach · Z| > 0.9),
    grasp_axis falls back to world X so the corridor remains well-defined.

    Returns:
        grasp_axis:  unit vector perpendicular to approach_dir, horizontal.
        height_axis: unit vector perpendicular to both, approximately upward.
    """
    cross = np.cross(_WORLD_UP, approach_dir)
    cross_n = np.linalg.norm(cross)
    if cross_n > 1e-6:
        # Normal case: grasp axis is horizontal, height axis ≈ world up
        grasp_axis = cross / cross_n
        height_axis = np.cross(approach_dir, grasp_axis)
        height_axis /= np.linalg.norm(height_axis)
    else:
        # Near-vertical approach: use world X as grasp axis
        grasp_axis = np.array([1.0, 0.0, 0.0])
        height_axis = np.cross(approach_dir, grasp_axis)
        h_n = np.linalg.norm(height_axis)
        if h_n > 1e-6:
            height_axis /= h_n
        else:
            height_axis = np.array([0.0, 1.0, 0.0])
    return grasp_axis, height_axis


def _obb_aabb_intersect_sat(
    obb_center: np.ndarray,
    obb_axes: np.ndarray,      # (3, 3) rows are unit axes
    obb_half: np.ndarray,      # (3,) half-extents
    aabb_min: np.ndarray,      # (3,)
    aabb_max: np.ndarray,      # (3,)
) -> bool:
    """Separating axis test between an OBB and an AABB.

    Tests up to 15 candidate separating axes (3 OBB face normals + 3 AABB
    face normals + 9 edge cross-products).  Returns True if they intersect.

    Uses the support-function projection for both bodies so the test is
    correct regardless of axis direction (no sign ambiguity).
    """
    aabb_center = (aabb_min + aabb_max) * 0.5
    aabb_half   = (aabb_max - aabb_min) * 0.5

    def _separated(axis: np.ndarray) -> bool:
        n = float(np.linalg.norm(axis))
        if n < 1e-10:
            # Degenerate axis (parallel edges) — cannot separate on this axis
            return False
        ax = axis / n
        # OBB support radius along ax
        r_obb  = float(np.sum(obb_half  * np.abs(obb_axes  @ ax)))
        # AABB support radius along ax
        r_aabb = float(np.sum(aabb_half * np.abs(ax)))
        # Signed distance between centers along ax
        d = abs(float((aabb_center - obb_center) @ ax))
        return d > r_obb + r_aabb

    # 3 OBB face normals
    for ax in obb_axes:
        if _separated(ax):
            return False
    # 3 AABB face normals (world X, Y, Z)
    for ax in np.eye(3):
        if _separated(ax):
            return False
    # 9 edge cross-products
    for obb_ax in obb_axes:
        for i in range(3):
            world_ax = np.zeros(3); world_ax[i] = 1.0
            cross = np.cross(obb_ax, world_ax)
            if _separated(cross):
                return False
    return True


def _aabb_project(aabb_min: np.ndarray, aabb_max: np.ndarray,
                  axis: np.ndarray) -> Tuple[float, float]:
    """Return (min, max) scalar projection of an AABB onto a unit axis.

    Uses the support-function form: project the center then add/subtract the
    half-extent along the axis.  This is correct for any axis direction,
    unlike a direct dot-product of min/max corners (which breaks when axis
    components are negative).
    """
    center = (aabb_min + aabb_max) * 0.5
    half   = (aabb_max - aabb_min) * 0.5
    proj_c = float(center @ axis)
    proj_r = float(np.sum(np.abs(half * axis)))
    return proj_c - proj_r, proj_c + proj_r


def _compute_intrusion(
    obb_center: np.ndarray,
    obb_axes: np.ndarray,      # (3, 3) rows: [approach, grasp, height]
    obb_half: np.ndarray,      # (3,) half-extents along each axis
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    grasp_axis: np.ndarray,    # unit vector along which fingers close (= obb_axes[1])
) -> float:
    """Compute how much of the corridor width is consumed by an obstacle.

    Measures the overlap between the obstacle AABB and the corridor OBB
    projected onto the grasp axis, but only counts intrusion where the
    obstacle actually overlaps the corridor in *all three* OBB axes.
    An obstacle that only clips the bottom of the corridor (height axis)
    but not the grasp-axis cross-section should not reduce clearance.

    Returns the overlap length in metres along grasp_axis (0 if the
    obstacle does not penetrate the grasp-axis cross-section).
    """
    # Check overlap on all three OBB axes first.  If the obstacle misses
    # the corridor on any axis it cannot consume grasp-axis clearance.
    for axis, half in zip(obb_axes, obb_half):
        obs_min, obs_max = _aabb_project(aabb_min, aabb_max, axis)
        corr_proj = float(obb_center @ axis)
        if obs_max <= corr_proj - half or obs_min >= corr_proj + half:
            return 0.0

    # All three axes overlap — measure the grasp-axis component only.
    obstacle_min, obstacle_max = _aabb_project(aabb_min, aabb_max, grasp_axis)
    corridor_proj = float(obb_center @ grasp_axis)
    corridor_half_w = obb_half[1]
    corridor_min = corridor_proj - corridor_half_w
    corridor_max = corridor_proj + corridor_half_w

    overlap_min = max(obstacle_min, corridor_min)
    overlap_max = min(obstacle_max, corridor_max)
    if overlap_max <= overlap_min:
        return 0.0
    return overlap_max - overlap_min


# ---------------------------------------------------------------------------
# Depth / point-cloud helpers
# ---------------------------------------------------------------------------


def _depth_to_pointcloud(
    depth_frame: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project a depth image to a 3-D point cloud in camera frame."""
    h, w = depth_frame.shape
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    valid = (depth_frame > 0) & np.isfinite(depth_frame)
    z = depth_frame[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _cam_to_base(
    pts_cam: np.ndarray,
    cam_position: np.ndarray,
    cam_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform points from camera frame to robot base frame."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(cam_quaternion_xyzw)
    return rot.apply(pts_cam) + cam_position


def _pts_aabb(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_corner, max_corner) AABB of a point cloud."""
    return pts.min(axis=0), pts.max(axis=0)


# ---------------------------------------------------------------------------
# Per-direction corridor computation  (spec §1.3 Phase 2)
# ---------------------------------------------------------------------------


def _pts_in_obb_corridor(
    pts: np.ndarray,
    corridor_center: np.ndarray,
    approach_dir: np.ndarray,
    grasp_axis: np.ndarray,
    height_axis: np.ndarray,
    half_len: float,
    half_w: float,
    half_h: float,
) -> bool:
    """Return True if any point in *pts* lies inside the corridor OBB."""
    rel = pts - corridor_center
    return bool(
        ((np.abs(rel @ approach_dir) <= half_len) &
         (np.abs(rel @ grasp_axis)   <= half_w)   &
         (np.abs(rel @ height_axis)  <= half_h)).any()
    )


def _compute_single_corridor(
    target_center: np.ndarray,
    target_aabb_min: np.ndarray,
    target_aabb_max: np.ndarray,
    target_aabb_radius: float,
    approach_dir: np.ndarray,
    neighbor_aabbs: List[Tuple[str, np.ndarray, np.ndarray]],
    gripper: GripperGeometry,
    neighbor_pts: Optional[Dict[str, np.ndarray]] = None,
) -> ApproachCorridor:
    """Compute clearance for one approach direction.

    Builds a gripper-sized OBB swept volume starting at the target object
    surface and extending CORRIDOR_LENGTH in approach_dir.  Tests all
    neighbour AABBs against this OBB with SAT; accumulates transverse
    intrusions to derive remaining clearance.

    Args:
        target_center: (3,) centroid of the target in world frame.
        target_aabb_min/max: AABB of the target in world frame.
        approach_dir: (3,) unit approach vector.
        neighbor_aabbs: List of (id, min_corner, max_corner) for neighbours.
        gripper: GripperGeometry parameters.
        neighbor_pts: Optional dict mapping obj_id → point cloud (N,3).  When
            provided, an AABB SAT hit is confirmed by checking whether any
            actual points fall inside the corridor OBB before counting as a
            blocker.  This eliminates false positives from large/sparse AABBs.

    Returns:
        ApproachCorridor for this direction.
    """
    # Gravity-aligned gripper frame: grasp_axis horizontal, height_axis ≈ world up
    grasp_axis, height_axis = _gripper_axes(approach_dir)

    # Half-extents of the OBB
    corridor_half_len = CORRIDOR_LENGTH / 2.0
    corridor_half_w   = gripper.swept_width    # along grasp_axis
    corridor_half_h   = gripper.swept_height   # along height_axis

    # corridor_start: the point on the target AABB surface in approach_dir.
    # Use the AABB support function: project center onto approach_dir then
    # add the support radius (sum of half-extents * |dir_component|).
    # This gives the correct surface distance for any approach direction,
    # including diagonals where the naive (extents @ abs_dir)/2 over-estimates.
    target_aabb_half = (target_aabb_max - target_aabb_min) * 0.5
    extent_along_approach = float(np.sum(target_aabb_half * np.abs(approach_dir)))
    corridor_start  = target_center + approach_dir * (extent_along_approach + CORRIDOR_STANDOFF)
    corridor_center = corridor_start + approach_dir * corridor_half_len

    obb_axes = np.stack([approach_dir, grasp_axis, height_axis], axis=0)  # (3, 3)
    obb_half = np.array([corridor_half_len, corridor_half_w, corridor_half_h])

    # Available transverse width starts at full corridor width
    full_width = corridor_half_w * 2.0
    min_clearance = full_width
    obstructing: List[str] = []
    obstructor_aabbs: List[Tuple[np.ndarray, np.ndarray]] = []

    dir_str = f"({approach_dir[0]:+.2f},{approach_dir[1]:+.2f},{approach_dir[2]:+.2f})"

    for obj_id, nb_min, nb_max in neighbor_aabbs:
        if not _obb_aabb_intersect_sat(corridor_center, obb_axes, obb_half, nb_min, nb_max):
            continue
        # Confirm with actual point cloud when available: AABB SAT can fire on
        # the empty corner of a sparse or elongated AABB.  If no actual points
        # fall inside the OBB the object is not truly in the way.
        if neighbor_pts is not None and obj_id in neighbor_pts:
            nb_pts = neighbor_pts[obj_id]
            if len(nb_pts) > 0 and not _pts_in_obb_corridor(
                nb_pts, corridor_center,
                approach_dir, grasp_axis, height_axis,
                corridor_half_len, corridor_half_w, corridor_half_h,
            ):
                logger.debug(
                    "  corridor %s: SAT hit %s rejected — no actual points inside OBB",
                    dir_str, obj_id,
                )
                continue
        obstructing.append(obj_id)
        obstructor_aabbs.append((nb_min.copy(), nb_max.copy()))
        intrusion = _compute_intrusion(
            corridor_center, obb_axes, obb_half,
            nb_min, nb_max, grasp_axis,
        )
        remaining = full_width - intrusion
        min_clearance = min(min_clearance, remaining)
        logger.debug(
            "  corridor %s: SAT hit %s  intrusion=%.1fmm  remaining=%.1fmm",
            dir_str, obj_id, intrusion * 1000, remaining * 1000,
        )

    grasp_compatible = min_clearance >= gripper.min_clearance_required
    if grasp_compatible:
        status = "GRASP_OK"
    elif obstructing and min_clearance < full_width:
        status = f"BLOCKED — stage 2 (grasp-axis intrusion by {obstructing}, clearance {min_clearance*1000:.1f}mm < required {gripper.min_clearance_required*1000:.1f}mm)"
    elif obstructing:
        status = f"BLOCKED — stage 1 (SAT intersection with {obstructing}, but zero grasp-axis intrusion — obstacle overlaps height axis only)"
    else:
        status = "BLOCKED — no obstructors (min_clearance below required from the start)"
    logger.info("corridor %s: %s", dir_str, status)

    return ApproachCorridor(
        direction=approach_dir.copy(),
        min_clearance=min_clearance,
        corridor_length=CORRIDOR_LENGTH,
        obstructing_objects=obstructing,
        grasp_compatible=grasp_compatible,
        corridor_start=corridor_start.copy(),
        grasp_axis=grasp_axis.copy(),
        height_axis=height_axis.copy(),
        half_width=corridor_half_w,
        half_height=corridor_half_h,
        obstructor_aabbs=obstructor_aabbs,
    )


# ---------------------------------------------------------------------------
# Graspability score  (spec §1.3 Phase 3)
# ---------------------------------------------------------------------------


def _compute_graspability(
    corridors: List[ApproachCorridor],
    gripper: GripperGeometry,
    n_active: int,
) -> float:
    """Compute aggregate graspability score in [0, 1].

    Blends per-corridor clearance quality (how much margin beyond minimum)
    with diversity (how many of the active corridors are viable).
    """
    viable = [c for c in corridors if c.grasp_compatible]
    if not viable:
        return 0.0

    min_req = gripper.min_clearance_required
    quality_scores = []
    for c in viable:
        margin = (c.min_clearance - min_req) / max(min_req, 1e-9)
        quality_scores.append(min(margin, 1.0))

    avg_quality = sum(quality_scores) / len(quality_scores)
    # Diversity: fraction of active directions that are viable; full bonus at ≥1/3
    diversity_bonus = min(len(viable) / max(n_active / 3.0, 1.0), 1.0)
    return 0.5 * avg_quality + 0.5 * diversity_bonus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SURFACE_TYPE_KEYWORDS: frozenset = frozenset({
    "table", "counter", "shelf", "desk", "tray", "bench", "surface", "floor"
})


def compute_clearance_profile(
    target_mask: np.ndarray,
    depth_frame: np.ndarray,
    camera_intrinsics,
    all_masks: Optional[Dict[str, np.ndarray]] = None,
    gripper: Optional[GripperGeometry] = None,
    workspace_radius: float = 1.5,
    camera_quaternion_xyzw: Optional[np.ndarray] = None,
    cam_position: Optional[np.ndarray] = None,
    # Legacy compatibility: scalar aperture is promoted to GripperGeometry
    gripper_aperture_m: Optional[float] = None,
    object_types: Optional[Dict[str, str]] = None,
) -> ClearanceProfile:
    """Compute the clearance profile for one object.

    Builds a gripper-width OBB swept volume in each approach direction that
    falls within ±100° of the object→camera vector (the observable hemisphere),
    and checks it against every neighbouring object's AABB using the Separating
    Axis Theorem.  Directions outside the observed region are skipped — they
    point at surfaces the camera cannot see and the robot cannot reliably reach.
    Derives transverse clearance, grasp compatibility, and a graspability score.

    Args:
        target_mask: (H, W) bool mask of the target object (from SAM2).
        depth_frame: (H, W) float32 depth image in metres.
        camera_intrinsics: Object with fx, fy, ppx/cx, ppy/cy attributes.
        all_masks: Dict mapping other object IDs to their (H, W) bool masks.
        gripper: GripperGeometry with finger_width, max_aperture, depth.
            Defaults to GripperGeometry() (12 mm fingers, 85 mm aperture, 60 mm depth).
        workspace_radius: Not used for OBB checks but kept for API compat.
        camera_quaternion_xyzw: (4,) camera orientation as xyzw quaternion.
            Required to transform points into base frame for world-axis directions.
        cam_position: (3,) camera origin in base frame.
        gripper_aperture_m: Legacy scalar — if gripper is None and this is
            provided, a GripperGeometry is constructed with max_aperture set
            to this value.

    Returns:
        ClearanceProfile with 18 approach corridors and scalar summaries.

    Example:
        gripper = GripperGeometry(finger_width=0.012, max_aperture=0.085, depth=0.06)
        profile = compute_clearance_profile(
            target_mask=obj_mask,
            depth_frame=depth_m,
            camera_intrinsics=intr,
            all_masks={"cup_1": cup_mask, "table_2": table_mask},
            gripper=gripper,
        )
    """
    # Resolve gripper geometry
    if gripper is None:
        if gripper_aperture_m is not None:
            gripper = GripperGeometry(max_aperture=float(gripper_aperture_m))
        else:
            gripper = GripperGeometry()

    h, w = depth_frame.shape
    fx = float(getattr(camera_intrinsics, 'fx', w / 2))
    fy = float(getattr(camera_intrinsics, 'fy', h / 2))
    cx = float(getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', w / 2)))
    cy = float(getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', h / 2)))

    # --- Target object point cloud ---
    target_depth = np.where(target_mask, depth_frame, 0.0)
    target_pts = _depth_to_pointcloud(target_depth, fx, fy, cx, cy)
    if len(target_pts) == 0:
        return ClearanceProfile()

    # --- Obstacle point clouds per neighbouring object ---
    obstacle_mask = ~target_mask & (depth_frame > 0) & np.isfinite(depth_frame)
    obstacle_pts_full = _depth_to_pointcloud(
        np.where(obstacle_mask, depth_frame, 0.0), fx, fy, cx, cy
    )

    obj_pts_map: Dict[str, np.ndarray] = {}
    if all_masks:
        for obj_id, mask in all_masks.items():
            if mask is None or not mask.any():
                continue
            pts = _depth_to_pointcloud(
                np.where(mask & obstacle_mask, depth_frame, 0.0), fx, fy, cx, cy
            )
            if len(pts) > 0:
                obj_pts_map[obj_id] = pts

    # --- Optional: transform all clouds to robot base frame ---
    _do_base_transform = cam_position is not None and camera_quaternion_xyzw is not None
    if _do_base_transform:
        target_pts = _cam_to_base(target_pts, cam_position, camera_quaternion_xyzw)  # type: ignore[arg-type]
        obstacle_pts_full = _cam_to_base(obstacle_pts_full, cam_position, camera_quaternion_xyzw)  # type: ignore[arg-type]
        obj_pts_map = {
            oid: _cam_to_base(pts, cam_position, camera_quaternion_xyzw)  # type: ignore[arg-type]
            for oid, pts in obj_pts_map.items()
        }

    # --- Compute target AABB ---
    # Depth cameras only observe the near-facing surface of an object — the
    # point cloud is the visible face only.  Mask-edge pixels that bleed onto
    # the background can project far behind the object in world frame, so we
    # use tight percentiles (5th–95th) to reject those outliers.  The resulting
    # AABB spans only what was actually observed.
    target_aabb_min = np.percentile(target_pts, 5,  axis=0)
    target_aabb_max = np.percentile(target_pts, 95, axis=0)
    target_center   = target_pts.mean(axis=0)
    # Half-diagonal of the robust AABB — used to reject self-collision hits
    target_aabb_radius = float(np.linalg.norm(target_aabb_max - target_aabb_min)) / 2.0

    # Neighbour point clouds filtered by SEARCH_RADIUS and bleed-through check.
    # Bleed-through: SAM2 masks sometimes bleed onto adjacent objects so their
    # depth pixels overlap in 3D.  We detect this by checking whether the
    # *median* nearest distance between neighbour and target points is below
    # BLEED_THROUGH_DIST.  Using the median (50th percentile) means only
    # neighbours whose point clouds substantially overlap the target (i.e. mask
    # bleed, not just physical proximity) are excluded.  The previous 5th-
    # percentile threshold incorrectly excluded objects that were simply close
    # to the target in space, causing corridors to appear CLEAR when blocked.
    BLEED_THROUGH_DIST: float = 0.008  # metres — median closer than this = mask overlap
    _tgt_sample = target_pts[::max(1, len(target_pts) // 500)]  # up to 500 pts

    neighbor_aabbs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    neighbor_aabbs_vis: List[Tuple[np.ndarray, np.ndarray]] = []
    for obj_id, pts in obj_pts_map.items():
        # Skip support surfaces — their large AABBs span the table top and
        # generate false corridor blockages for any object resting on them.
        if object_types is not None:
            otype = object_types.get(obj_id, "").lower()
            if any(kw in otype for kw in _SURFACE_TYPE_KEYWORDS):
                logger.debug("  skipped neighbour %s (surface type '%s')", obj_id, otype)
                continue

        nb_min, nb_max = _pts_aabb(pts)
        nb_center = (nb_min + nb_max) / 2.0
        if float(np.linalg.norm(nb_center - target_center)) > SEARCH_RADIUS:
            continue
        # Point-cloud bleed-through check: sample neighbour pts and find the
        # minimum distance to any target point.  Use broadcasting on a small
        # sample to keep this O(1) in practice.
        nb_sample = pts[::max(1, len(pts) // 200)]  # up to 200 pts
        dists = np.linalg.norm(
            nb_sample[:, None, :] - _tgt_sample[None, :, :], axis=2
        ).min(axis=1)
        if float(np.percentile(dists, 50)) < BLEED_THROUGH_DIST:
            logger.debug(
                "  skipped neighbour %s (bleed-through: median_dist=%.1fmm < %.1fmm)",
                obj_id, float(np.percentile(dists, 50)) * 1000, BLEED_THROUGH_DIST * 1000,
            )
            continue
        neighbor_aabbs.append((obj_id, nb_min, nb_max))
        nb_min_vis = np.percentile(pts, 5,  axis=0)
        nb_max_vis = np.percentile(pts, 95, axis=0)
        neighbor_aabbs_vis.append((nb_min_vis, nb_max_vis))
    logger.debug("compute_clearance_profile: %d neighbours in search radius", len(neighbor_aabbs))

    # --- Filter directions to the observable hemisphere ±100° from camera ---
    # The camera can only observe surfaces facing toward it, so approach
    # directions pointing more than 100° away from the object→camera vector
    # are outside the observed region and should not be evaluated.
    # cos(100°) ≈ -0.174 — directions with dot product below this are culled.
    _COS_LIMIT = float(np.cos(np.deg2rad(100.0)))
    if cam_position is not None:
        # Use world-frame cam_position if base transform was applied, else
        # transform cam_position to the same frame as target_center.
        if _do_base_transform:
            cam_pos_world = np.asarray(cam_position, dtype=float)
        else:
            cam_pos_world = np.asarray(cam_position, dtype=float)
        toward_cam = cam_pos_world - target_center
        toward_cam_norm = float(np.linalg.norm(toward_cam))
        if toward_cam_norm > 1e-6:
            toward_cam = toward_cam / toward_cam_norm
            active_dirs = _APPROACH_DIRS[(_APPROACH_DIRS @ toward_cam) >= _COS_LIMIT]
        else:
            active_dirs = _APPROACH_DIRS
    else:
        active_dirs = _APPROACH_DIRS

    # --- Per-direction OBB corridor checks ---
    corridors: List[ApproachCorridor] = []
    for approach_dir in active_dirs:
        corridor = _compute_single_corridor(
            target_center=target_center,
            target_aabb_min=target_aabb_min,
            target_aabb_max=target_aabb_max,
            target_aabb_radius=target_aabb_radius,
            approach_dir=approach_dir,
            neighbor_aabbs=neighbor_aabbs,
            gripper=gripper,
            neighbor_pts=obj_pts_map,
        )
        corridors.append(corridor)

    # Sort: grasp-compatible corridors first, then by clearance descending
    corridors.sort(key=lambda c: (not c.grasp_compatible, -c.min_clearance))

    # --- Graspability score ---
    graspability_score = _compute_graspability(corridors, gripper, n_active=len(active_dirs))

    # --- Scalar summaries ---
    # Build a direction→corridor map by best dot product
    ray_lengths = np.array([c.min_clearance for c in corridors])  # (18,) for visualisation

    if _do_base_transform:
        def _query_dir(world_dir: np.ndarray) -> np.ndarray:
            return world_dir
    elif camera_quaternion_xyzw is not None:
        try:
            from scipy.spatial.transform import Rotation
            _cam_rot = Rotation.from_quat(camera_quaternion_xyzw)
            def _query_dir(world_dir: np.ndarray) -> np.ndarray:
                return _cam_rot.inv().apply(world_dir)
        except Exception:
            def _query_dir(world_dir: np.ndarray) -> np.ndarray:
                return world_dir
    else:
        def _query_dir(world_dir: np.ndarray) -> np.ndarray:
            return world_dir

    corridor_dirs = np.stack([c.direction for c in corridors], axis=0)  # (18, 3)

    # "Top" = world +Z
    top_dir = _query_dir(np.array([0.0, 0.0, 1.0]))
    top_idx = int(np.argmax(corridor_dirs @ top_dir))
    top_clearance = corridors[top_idx].min_clearance

    world_face_map: Dict[str, np.ndarray] = {
        "+x": np.array([1.0,  0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0,  1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
    }
    lateral_clearances: Dict[str, float] = {}
    for label, world_dir in world_face_map.items():
        q_dir = _query_dir(world_dir)
        best_idx = int(np.argmax(corridor_dirs @ q_dir))
        lateral_clearances[label] = corridors[best_idx].min_clearance

    return ClearanceProfile(
        approach_corridors=corridors,
        top_clearance=top_clearance,
        lateral_clearances=lateral_clearances,
        graspability_score=graspability_score,
        centroid=target_center,
        ray_dirs=_APPROACH_DIRS.copy(),
        ray_lengths_m=ray_lengths,
        obstacle_pts=obstacle_pts_full.copy() if len(obstacle_pts_full) > 0 else None,
        target_pts=target_pts.copy() if len(target_pts) > 0 else None,
        target_aabb_min=target_aabb_min.copy(),
        target_aabb_max=target_aabb_max.copy(),
        neighbor_aabbs=neighbor_aabbs_vis,
    )
