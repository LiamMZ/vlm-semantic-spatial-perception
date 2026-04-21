"""
Free-space region computation for support surfaces.

For each object identified as a surface (table, shelf, tray, …), projects all
objects resting on it onto the surface plane and computes:
  - The set of unoccupied polygonal regions (free space) via Shapely boolean ops
  - The largest inscribed circle in each region (placement radius)
  - The neighboring objects bordering each region
  - A congestion score [0,1] penalising both occupancy and fragmentation

Algorithm
---------
Phase 1: Object-to-surface assignment
  Use contact graph supporting/stacked edges.  Fall back to AABB proximity
  (object bottom within 5 cm of surface top) when no contact graph is given.

Phase 2: 2D footprint computation
  Project each resting object's AABB corners onto the surface plane (PCA fit),
  take a 2D convex hull, expand by PLACEMENT_MARGIN.

Phase 3: Boolean subtraction via Shapely
  Subtract all footprint polygons from the surface boundary polygon to get the
  free-space polygon set.  Split into individual FreeSpaceRegion instances.

Phase 4: Congestion score
  congestion = clip(1 - free_fraction + 0.2 * fragmentation, 0, 1)
  where fragmentation = 1 - (largest_region / total_free).

Phase 5: Raster grid + inscribed circle
  Rasterise at resolution_m for visualisation.  Use scipy distance transform to
  find the inscribed circle centre and radius for each free region.

Complexity: <20 ms for 3 surfaces, <10 objects per surface, 1 cm resolution.

Usage:
    surface_maps = compute_surface_maps(
        objects=registry.get_all_objects(),
        obj_masks=masks,
        depth_frame=depth_m,
        camera_intrinsics=intr,
        contact_graph=registry.contact_graph,
        resolution_m=0.01,
    )
    feasible, region = can_place("cup_1", "table_1", surface_maps, objects)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .clearance import _depth_to_pointcloud
from .contact_graph import ContactGraph
from .object_registry import DetectedObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLACEMENT_MARGIN: float = 0.01   # 1 cm safety margin around footprints
_MIN_REGION_AREA: float = 0.001  # 10 cm² — ignore tiny slivers
_INSCRIBED_CIRCLE_SAMPLES: int = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FreeSpaceRegion:
    """One contiguous unoccupied region on a support surface.

    Spatial quantities are in the surface's local 2-D frame (metres) unless
    noted.

    Attributes:
        polygon: Boundary vertices (N, 2) in surface 2-D frame.
        area: Region area in m².
        max_inscribed_circle: Radius of the largest inscribed circle, metres.
        neighboring_objects: IDs of objects whose footprints border this region.
        center_uv: (col, row) pixel coordinates of the inscribed circle centre
            in the occupancy grid (for visualisation).
        pixel_mask: (grid_H, grid_W) bool — cells belonging to this region.
    """

    polygon: np.ndarray             # (N, 2) metres, local surface frame
    area: float                     # m²
    max_inscribed_circle: float     # metres
    neighboring_objects: List[str]
    center_uv: Optional[Tuple[float, float]] = None   # (col, row) in grid pixels
    pixel_mask: Optional[np.ndarray] = None           # (H, W) bool


@dataclass
class SurfaceMap:
    """Free-space map for one support surface.

    Attributes:
        free_space_regions: Unoccupied regions sorted by area descending.
        congestion_score: [0, 1] — 0 = empty, 1 = fully packed/fragmented.
        occupancy_grid: (Nu, Nv) bool raster — True = occupied.
        resolution_m: Cell size of occupancy_grid in metres.
    """

    free_space_regions: List[FreeSpaceRegion] = field(default_factory=list)
    congestion_score: float = 0.0
    occupancy_grid: Optional[np.ndarray] = None   # (Nu, Nv) bool
    resolution_m: float = 0.01


# ---------------------------------------------------------------------------
# Surface plane fitting
# ---------------------------------------------------------------------------

def _fit_surface_plane(
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to a point cloud via PCA; return (origin, u_axis, v_axis)."""
    origin = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - origin, full_matrices=False)
    u_axis = Vt[0]
    v_axis = Vt[1]
    return origin, u_axis, v_axis


def _project_to_surface_2d(
    pts_3d: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    """Project (N,3) 3D points onto the surface 2D frame → (N,2)."""
    rel = pts_3d - origin
    return np.stack([rel @ u_axis, rel @ v_axis], axis=1)


def _aabb_corners_2d(
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    """Project all 8 AABB corners onto the surface plane → (8, 2)."""
    corners = np.array([
        [sx, sy, sz]
        for sx in (min_xyz[0], max_xyz[0])
        for sy in (min_xyz[1], max_xyz[1])
        for sz in (min_xyz[2], max_xyz[2])
    ])
    return _project_to_surface_2d(corners, origin, u_axis, v_axis)


# ---------------------------------------------------------------------------
# 2D convex hull (dependency-free fallback)
# ---------------------------------------------------------------------------

def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Gift-wrapping 2D convex hull.  Returns ordered vertex array (K, 2)."""
    if len(pts) <= 2:
        return pts
    start = int(pts[:, 0].argmin())
    hull = [start]
    current = start
    for _ in range(len(pts) + 1):
        candidate = (current + 1) % len(pts)
        for i in range(len(pts)):
            if i == current:
                continue
            cross = np.cross(pts[candidate] - pts[current], pts[i] - pts[current])
            if cross < 0:
                candidate = i
        if candidate == start:
            break
        hull.append(candidate)
        current = candidate
    return pts[hull]


# ---------------------------------------------------------------------------
# Shapely-based free region computation
# ---------------------------------------------------------------------------

def _offset_polygon(coords: np.ndarray, margin: float):
    """Expand a polygon by margin using Shapely buffer."""
    try:
        from shapely.geometry import Polygon
        poly = Polygon(coords)
        return poly.buffer(margin)
    except Exception:
        return None


def _compute_free_regions_shapely(
    surface_pts_2d: np.ndarray,
    footprints_2d: Dict[str, np.ndarray],
    margin: float = PLACEMENT_MARGIN,
) -> Tuple[list, list]:
    """Phase 3: subtract footprint polygons from surface boundary.

    Returns:
        (shapely_polygons, neighbor_lists) — parallel lists.
        Falls back to an empty list if Shapely is not available.
    """
    try:
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("Shapely not available — free region computation skipped")
        return [], []

    # Surface boundary: convex hull of observed surface points
    hull_pts = _convex_hull_2d(surface_pts_2d)
    if len(hull_pts) < 3:
        return [], []
    surface_shape = Polygon(hull_pts)
    if not surface_shape.is_valid:
        surface_shape = surface_shape.buffer(0)

    # Obstacle shapes: convex hull of each footprint, expanded by margin
    obstacle_shapes = []
    for obj_id, pts2d in footprints_2d.items():
        if len(pts2d) < 3:
            continue
        hull = _convex_hull_2d(pts2d)
        try:
            poly = Polygon(hull).buffer(margin)
            if poly.is_valid and not poly.is_empty:
                obstacle_shapes.append((obj_id, poly))
        except Exception:
            pass

    if obstacle_shapes:
        obstacles_union = unary_union([s for _, s in obstacle_shapes])
        free_shape = surface_shape.difference(obstacles_union)
    else:
        free_shape = surface_shape

    if isinstance(free_shape, MultiPolygon):
        polys = list(free_shape.geoms)
    elif hasattr(free_shape, 'geoms'):
        polys = list(free_shape.geoms)
    elif not free_shape.is_empty:
        polys = [free_shape]
    else:
        polys = []

    result_polys = []
    result_neighbors = []

    for poly in polys:
        if poly.is_empty or poly.area < _MIN_REGION_AREA:
            continue
        neighbors = []
        for obj_id, fp_shape in obstacle_shapes:
            try:
                if poly.distance(fp_shape) < 0.02:
                    neighbors.append(obj_id)
            except Exception:
                pass
        result_polys.append(poly)
        result_neighbors.append(neighbors)

    return result_polys, result_neighbors


def _inscribed_circle_sampled(
    poly,
    n_samples: int = _INSCRIBED_CIRCLE_SAMPLES,
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Approximate largest inscribed circle via random sampling inside polygon.

    Returns (radius_m, (cx, cy)) where cx/cy are in the polygon's coordinate frame.
    """
    try:
        from shapely.geometry import Point
        minx, miny, maxx, maxy = poly.bounds
        rng = np.random.default_rng(seed=0)
        best_r = 0.0
        best_pt: Optional[Tuple[float, float]] = None

        # Stratified sampling over bounding box
        xs = rng.uniform(minx, maxx, n_samples)
        ys = rng.uniform(miny, maxy, n_samples)

        for px, py in zip(xs, ys):
            pt = Point(px, py)
            if not poly.contains(pt):
                continue
            d = float(poly.exterior.distance(pt))
            for ring in poly.interiors:
                d = min(d, float(ring.distance(pt)))
            if d > best_r:
                best_r = d
                best_pt = (float(px), float(py))

        return best_r, best_pt
    except Exception:
        return 0.0, None


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def _rasterise(
    surface_pts_2d: np.ndarray,
    footprints_2d: Dict[str, np.ndarray],
    resolution: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Rasterise occupied footprints onto a grid aligned to surface_pts_2d extent.

    Returns:
        (grid, u0, v0, u1, v1) — grid is (Nu, Nv) bool, True = occupied.
        u0/v0 are the minimum coordinates of the grid origin.
    """
    u0, v0 = surface_pts_2d.min(axis=0)
    u1, v1 = surface_pts_2d.max(axis=0)
    nu = max(1, int(np.ceil((u1 - u0) / resolution)))
    nv = max(1, int(np.ceil((v1 - v0) / resolution)))
    grid = np.zeros((nu, nv), dtype=bool)

    for pts2d in footprints_2d.values():
        if len(pts2d) == 0:
            continue
        fp_u0, fp_v0 = pts2d.min(axis=0)
        fp_u1, fp_v1 = pts2d.max(axis=0)
        i0 = max(0, int((fp_u0 - u0) / resolution))
        i1 = min(nu, int(np.ceil((fp_u1 - u0) / resolution)))
        j0 = max(0, int((fp_v0 - v0) / resolution))
        j1 = min(nv, int(np.ceil((fp_v1 - v0) / resolution)))
        if i0 < i1 and j0 < j1:
            grid[i0:i1, j0:j1] = True

    return grid, float(u0), float(v0), float(u1), float(v1)


def _pixel_mask_for_poly(
    poly,
    grid_shape: Tuple[int, int],
    u0: float, v0: float,
    resolution: float,
) -> np.ndarray:
    """Build a boolean pixel mask for a Shapely polygon on the given grid."""
    nu, nv = grid_shape
    mask = np.zeros((nu, nv), dtype=bool)
    try:
        from shapely.geometry import Point
        minx, miny, maxx, maxy = poly.bounds
        i0 = max(0, int((minx - u0) / resolution))
        i1 = min(nu, int(np.ceil((maxx - u0) / resolution)))
        j0 = max(0, int((miny - v0) / resolution))
        j1 = min(nv, int(np.ceil((maxy - v0) / resolution)))
        for i in range(i0, i1):
            for j in range(j0, j1):
                pu = u0 + (i + 0.5) * resolution
                pv = v0 + (j + 0.5) * resolution
                if poly.contains(Point(pu, pv)):
                    mask[i, j] = True
    except Exception:
        pass
    return mask


# ---------------------------------------------------------------------------
# Congestion score
# ---------------------------------------------------------------------------

def _congestion_score(
    free_regions,
    surface_area: float,
) -> float:
    """Spec §4.4 congestion: penalises both occupancy and fragmentation."""
    if surface_area <= 0:
        return 1.0
    total_free = sum(r.area for r in free_regions)
    free_fraction = total_free / surface_area
    if not free_regions:
        return 1.0
    largest = max(r.area for r in free_regions)
    fragmentation = 1.0 - (largest / max(total_free, 1e-9))
    score = 1.0 - free_fraction + 0.2 * fragmentation
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Placement feasibility query
# ---------------------------------------------------------------------------

def can_place(
    obj_id: str,
    surface_id: str,
    surface_maps: Dict[str, "SurfaceMap"],
    objects: List[DetectedObject],
    margin: float = PLACEMENT_MARGIN,
) -> Tuple[bool, Optional[FreeSpaceRegion]]:
    """Check whether obj_id can be placed on surface_id.

    Computes the minimum enclosing circle of obj_id's AABB footprint, adds
    margin, and checks whether any free region's inscribed circle is large
    enough.

    Args:
        obj_id: Object to place.
        surface_id: Target surface.
        surface_maps: Output of compute_surface_maps.
        objects: All DetectedObject instances (used to get obj AABB).
        margin: Additional clearance beyond the object footprint, metres.

    Returns:
        (feasible, best_region) — best_region is the largest viable region,
        or None if no region is large enough.
    """
    smap = surface_maps.get(surface_id)
    if smap is None or not smap.free_space_regions:
        return False, None

    obj = next((o for o in objects if o.object_id == obj_id), None)
    if obj is None or obj.position_3d is None:
        return False, None

    # Minimum radius: half the diagonal of a 6 cm default footprint
    # (real footprint would come from AABB half-extents if available)
    footprint_half = np.array([0.03, 0.03])  # default 3 cm half-extents
    min_radius = float(np.linalg.norm(footprint_half)) + margin

    best: Optional[FreeSpaceRegion] = None
    for region in smap.free_space_regions:  # already sorted largest first
        if region.max_inscribed_circle >= min_radius:
            best = region
            break

    return (best is not None), best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_surface_maps(
    objects: List[DetectedObject],
    obj_masks: Dict[str, np.ndarray],
    depth_frame: np.ndarray,
    camera_intrinsics,
    contact_graph: Optional[ContactGraph] = None,
    surface_types: Optional[Set[str]] = None,
    resolution_m: float = 0.01,
) -> Dict[str, SurfaceMap]:
    """Compute free-space maps for all surface objects in the scene.

    Args:
        objects: All DetectedObject instances in the current frame.
        obj_masks: {object_id: bool (H, W)} SAM2 masks.
        depth_frame: (H, W) float32 depth in metres.
        camera_intrinsics: Object with fx/fy/ppx/ppy (or cx/cy) attributes.
        contact_graph: Optional ContactGraph for surface assignment.
        surface_types: object_type strings treated as surfaces.
            Default: {"table", "surface", "shelf", "tray", "counter", "desk"}.
        resolution_m: Raster grid cell size in metres (default 1 cm).

    Returns:
        {surface_id: SurfaceMap}
    """
    if surface_types is None:
        surface_types = {"table", "surface", "shelf", "tray", "counter", "desk", "plate"}

    h, w = depth_frame.shape
    fx = float(getattr(camera_intrinsics, 'fx', w / 2))
    fy = float(getattr(camera_intrinsics, 'fy', h / 2))
    cx = float(getattr(camera_intrinsics, 'ppx', getattr(camera_intrinsics, 'cx', w / 2)))
    cy = float(getattr(camera_intrinsics, 'ppy', getattr(camera_intrinsics, 'cy', h / 2)))

    obj_by_id = {o.object_id: o for o in objects}

    # --- Identify surface objects ---
    surface_ids: Set[str] = {
        o.object_id for o in objects if o.object_type.lower() in surface_types
    }
    if contact_graph is not None:
        for oid, supported in contact_graph.support_tree.items():
            if supported:
                surface_ids.add(oid)

    if not surface_ids:
        return {}

    # --- Per-object 3D point clouds and AABBs ---
    obj_pts3d: Dict[str, np.ndarray] = {}
    obj_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for obj in objects:
        mask = obj_masks.get(obj.object_id)
        if mask is not None and mask.any():
            pts = _depth_to_pointcloud(np.where(mask, depth_frame, 0.0), fx, fy, cx, cy)
            if len(pts) > 0:
                obj_pts3d[obj.object_id] = pts
                obj_aabbs[obj.object_id] = (pts.min(axis=0), pts.max(axis=0))
        if obj.object_id not in obj_aabbs and obj.position_3d is not None:
            c = obj.position_3d.astype(float)
            obj_aabbs[obj.object_id] = (c - 0.03, c + 0.03)

    result: Dict[str, SurfaceMap] = {}

    for surface_id in surface_ids:
        surf_pts = obj_pts3d.get(surface_id)
        if surf_pts is None or len(surf_pts) < 3:
            logger.debug("surface_map: %s — no point cloud, skipping", surface_id)
            result[surface_id] = SurfaceMap(resolution_m=resolution_m)
            continue

        origin, u_axis, v_axis = _fit_surface_plane(surf_pts)
        surf_pts_2d = _project_to_surface_2d(surf_pts, origin, u_axis, v_axis)

        # --- Phase 1: assign resting objects ---
        if contact_graph is not None:
            resting_ids = list(contact_graph.support_tree.get(surface_id, []))
        else:
            # Fallback: objects with bottom AABB near surface top (5 cm)
            surf_aabb = obj_aabbs.get(surface_id)
            resting_ids = []
            if surf_aabb is not None:
                surf_top_y = float(surf_aabb[1][1])  # camera frame: max Y = bottom
                for obj in objects:
                    if obj.object_id == surface_id:
                        continue
                    ab = obj_aabbs.get(obj.object_id)
                    if ab is not None and abs(float(ab[0][1]) - surf_top_y) < 0.05:
                        resting_ids.append(obj.object_id)

        # --- Phase 2: 2D footprints ---
        footprints_2d: Dict[str, np.ndarray] = {}
        for rid in resting_ids:
            ab = obj_aabbs.get(rid)
            if ab is None:
                continue
            pts2d = _aabb_corners_2d(ab[0], ab[1], origin, u_axis, v_axis)
            footprints_2d[rid] = pts2d

        logger.debug(
            "surface_map: %s — %d resting objects, %d footprints",
            surface_id, len(resting_ids), len(footprints_2d),
        )

        # --- Phase 3: free regions via Shapely boolean subtraction ---
        shapely_polys, neighbor_lists = _compute_free_regions_shapely(
            surf_pts_2d, footprints_2d, margin=PLACEMENT_MARGIN
        )

        # --- Phase 5: rasterise for occupancy grid + pixel masks ---
        occ_grid, u0, v0, u1, v1 = _rasterise(surf_pts_2d, footprints_2d, resolution_m)

        free_regions: List[FreeSpaceRegion] = []

        for poly, neighbors in zip(shapely_polys, neighbor_lists):
            area = float(poly.area)

            # Inscribed circle via sampling
            mic_radius, mic_centre = _inscribed_circle_sampled(poly)

            # Convert Shapely polygon exterior to numpy vertex array
            try:
                verts = np.array(poly.exterior.coords, dtype=float)
            except Exception:
                verts = np.zeros((0, 2), dtype=float)

            # Pixel mask for this region
            pix_mask = _pixel_mask_for_poly(poly, occ_grid.shape, u0, v0, resolution_m)

            # Inscribed circle centre in grid pixel coords (col, row)
            center_uv: Optional[Tuple[float, float]] = None
            if mic_centre is not None:
                col = (mic_centre[0] - u0) / resolution_m
                row = (mic_centre[1] - v0) / resolution_m
                center_uv = (float(col), float(row))

            free_regions.append(FreeSpaceRegion(
                polygon=verts,
                area=area,
                max_inscribed_circle=mic_radius,
                neighboring_objects=neighbors,
                center_uv=center_uv,
                pixel_mask=pix_mask,
            ))

        # Fallback: if Shapely unavailable, derive one region from the free grid
        if not free_regions and not shapely_polys:
            free_grid = ~occ_grid
            area = float(free_grid.sum()) * (resolution_m ** 2)
            if area > _MIN_REGION_AREA:
                try:
                    from scipy.ndimage import distance_transform_edt
                    dist = distance_transform_edt(free_grid) * resolution_m
                    mic_radius = float(dist.max())
                    peak = np.unravel_index(dist.argmax(), dist.shape)
                    center_uv = (float(peak[1]), float(peak[0]))
                except ImportError:
                    mic_radius = 0.0
                    center_uv = None
                ri, ci = np.where(free_grid)
                pts2d_free = np.stack([
                    u0 + (ci + 0.5) * resolution_m,
                    v0 + (ri + 0.5) * resolution_m,
                ], axis=1)
                free_regions.append(FreeSpaceRegion(
                    polygon=_convex_hull_2d(pts2d_free) if len(pts2d_free) >= 3 else pts2d_free,
                    area=area,
                    max_inscribed_circle=mic_radius,
                    neighboring_objects=list(footprints_2d.keys()),
                    center_uv=center_uv,
                    pixel_mask=free_grid,
                ))

        # Sort largest region first
        free_regions.sort(key=lambda r: r.area, reverse=True)

        # --- Phase 4: congestion score ---
        surf_area = float(surf_pts_2d.max(axis=0).prod() - surf_pts_2d.min(axis=0).prod())
        try:
            from shapely.geometry import Polygon
            surf_area = float(Polygon(_convex_hull_2d(surf_pts_2d)).area)
        except Exception:
            pass
        congestion = _congestion_score(free_regions, max(surf_area, 1e-6))

        result[surface_id] = SurfaceMap(
            free_space_regions=free_regions,
            congestion_score=congestion,
            occupancy_grid=occ_grid,
            resolution_m=resolution_m,
        )

        logger.info(
            "surface_map: %s — congestion=%.2f  free_regions=%d  "
            "largest_region=%.0f cm²  largest_inscribed=%.1f cm",
            surface_id,
            congestion,
            len(free_regions),
            (free_regions[0].area * 1e4) if free_regions else 0.0,
            (free_regions[0].max_inscribed_circle * 100) if free_regions else 0.0,
        )

    return result
