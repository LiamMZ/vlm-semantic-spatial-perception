# Perception Service Extensions (Ψ)

Documents the two geometric perception extensions added to the Ψ layer:
clearance volumes (§1) and contact graphs (§2).  Both run automatically each
frame inside `GSAM2ObjectTracker.detect_objects()` when depth is available.

---

## §1 Clearance Volumes

**Files:** `src/perception/clearance.py`, `DetectedObject.clearance_profile`

### Data model

```
DetectedObject.clearance_profile: Optional[ClearanceProfile]
ClearanceProfile
├── approach_corridors: List[ApproachCorridor]   # sorted by corridor_length desc
│   ├── direction: (3,) unit vector  # mean approach axis, camera frame
│   ├── min_clearance: float          # narrowest point, metres
│   ├── corridor_length: float        # median ray depth, metres
│   ├── obstructing_objects: [id]     # neighbours that bound the corridor
│   └── grasp_compatible: bool        # min_clearance ≥ gripper_aperture/2
├── top_clearance: float              # clearance along camera −Y (upward), metres
├── lateral_clearances: {+x,-x,+y,-y: float}
└── graspability_score: float         # [0,1], corridor-length-weighted compat fraction
```

### Algorithm

1. Back-project the full depth frame → 3-D point cloud (camera frame).
2. For the target object, use its SAM2 boolean mask to isolate its points → centroid.
3. Non-target depth pixels form the obstacle cloud.
4. Cast **42 rays** (26-connected cube + 8 diagonal refinements) from the centroid.
   For each ray, project obstacle points onto the ray axis and find the nearest hit
   within a 10 mm lateral tube.
5. Viable rays (>2 cm clearance) are greedily clustered by angular similarity
   (cosine threshold 0.6, ~53°) into `ApproachCorridor`s.
6. Scalar summaries derived from fixed-axis rays.

**Complexity:** O(n·k) — n total obstacle points, k=42 directions.
**Typical latency:** <50 ms for ≤30 objects at 640×480.

### Key parameters

| Parameter | Default | Where set |
|---|---|---|
| `compute_clearances` | `True` | `GSAM2ObjectTracker.__init__` |
| `gripper_aperture_m` | `0.08` m | `GSAM2ObjectTracker.__init__` |
| Ray directions | 42 | `clearance._RAY_DIRS` (module constant) |
| Tube radius | 10 mm | `clearance._cast_ray` |
| Cluster cos threshold | 0.6 | `clearance._cluster_corridors` |

### Usage

```python
profile = obj.clearance_profile          # None until first frame with depth
if profile and profile.graspability_score > 0.5:
    best = profile.approach_corridors[0] # deepest viable corridor
    if best.grasp_compatible:
        approach_dir = best.direction    # (3,) camera-frame unit vector
```

---

## §2 Contact Graph

**Files:** `src/perception/contact_graph.py`, `DetectedObjectRegistry.contact_graph`

The contact graph is scene-level (not per-object) and lives on the registry.

### Data model

```
DetectedObjectRegistry.contact_graph: Optional[ContactGraph]
ContactGraph
├── edges: List[ContactEdge]
│   ├── obj_a: id                   # lower / supporting object
│   ├── obj_b: id                   # upper / supported object
│   ├── contact_type: supporting | stacked | leaning | adjacent | nested
│   ├── contact_region
│   │   ├── point: (3,) camera frame
│   │   ├── normal: (3,) unit vector obj_a → obj_b
│   │   └── area: float  m²
│   └── removal_consequence: stable | unstable | cascade
├── support_tree: {id → [supported_ids]}
└── stability_scores: {id → float [0,1]}
```

### Contact classification (from relative AABB geometry)

| Type | Condition |
|---|---|
| `nested` | One AABB fully contained inside the other (all 3 axes) |
| `stacked` | obj_a below obj_b + obj_b COM projects onto obj_a + obj_b footprint ⊆ obj_a footprint |
| `supporting` | obj_a below obj_b + obj_b COM projects onto obj_a surface |
| `leaning` | Lateral contact dominant (XZ separation > Y separation) |
| `adjacent` | Within threshold, no force dependency |

Camera-frame convention: **Y increases downward** → lower object has larger Y centroid.

### Stability analysis

For each object `a`, simulate its removal:
1. Find all objects `b` directly supported by `a`.
2. For each `b`, check if its COM projects inside the convex hull of remaining support
   contact points (XZ plane).
3. If `b` becomes unsupported, check whether `b`'s instability cascades to objects
   `b` was supporting → mark `removal_consequence = "cascade"`.

`stability_scores[id]` = fraction of directly-supported objects that remain stable
after `id` is removed.

**Complexity:** O(n²) pairwise + O(n²) stability.
**Typical latency:** <20 ms pairwise, <100 ms total for ≤30 objects.

### Key parameters

| Parameter | Default | Where set |
|---|---|---|
| `compute_contacts` | `True` | `GSAM2ObjectTracker.__init__` |
| `contact_threshold_m` | `0.005` m (5 mm) | `GSAM2ObjectTracker.__init__` |

### Usage

```python
graph = registry.contact_graph          # None until ≥2 objects detected with depth
if graph:
    for edge in graph.edges:
        if edge.contact_type == "supporting":
            print(f"{edge.obj_a} supports {edge.obj_b}, "
                  f"consequence={edge.removal_consequence}")
    risky = [oid for oid, s in graph.stability_scores.items() if s < 0.5]
```

---

## §3 Occlusion Map

**Files:** `src/perception/occlusion.py`, `DetectedObjectRegistry.occlusion_map`

The occlusion map is scene-level (lives on the registry).  It accumulates
evidence across a rolling window of observations so visibility estimates
improve as the camera moves.

### Data model

```
DetectedObjectRegistry.occlusion_map: Optional[OcclusionMap]
OcclusionMap
├── camera_poses: List[CameraPose]          # one per observation in history
│   ├── position: (3,) metres
│   └── quaternion_xyzw: (4,)
├── per_object_visibility: {id → ObjectVisibility}
│   ├── visible_fraction: float             # [0,1] best across history
│   ├── occluding_objects: [id]             # union of occluders across history
│   ├── best_viewpoint: CameraPose          # pose with max visible fraction
│   └── hidden_regions: List[ndarray]       # (H,W) bool, one per observation
└── unobserved_volumes: List[UnobservedVolume]
    ├── bounds: AABB  (min_xyz, max_xyz)    # camera frame of last observation
    ├── blocking_objects: [id]              # object(s) casting this shadow
    └── estimated_capacity: int             # small objects (5 cm³) that could fit
```

### Algorithm

Per observation frame:

1. **Pixel-level occlusion** — for each pixel (u,v) in object A's SAM2 mask,
   check if any other object's mask has a smaller depth at the same pixel.
   Visible pixels: `obj_depth[u,v] ≤ min_other_depth[u,v] + 5 mm`.
2. **Visible fraction** — `#visible_pixels / #mask_pixels` for this observation.
3. **Occluding objects** — IDs of objects nearest at each occluded pixel.
4. **Shadow volume** — for each column (u) touched by the object mask, all
   pixels with depth beyond the object's surface form the shadow cone.
5. **Accumulation** — best visible fraction and corresponding pose are
   tracked per object.  Shadow pixels are *intersected* across observations:
   only pixels that are *always* in shadow become `UnobservedVolume` candidates.
6. **Unobserved volumes** — persistent shadow pixels back-projected to 3-D AABB.

**Complexity:** O(H·W·n) per frame — ~9 M ops for 30 objects at 640×480.
**Update frequency:** configurable; default every frame.

### Key parameters

| Parameter | Default | Where set |
|---|---|---|
| `compute_occlusion` | `True` | `GSAM2ObjectTracker.__init__` |
| `occlusion_history_len` | `10` frames | `GSAM2ObjectTracker.__init__` |
| `occlusion_update_interval` | `1` (every frame) | `GSAM2ObjectTracker.__init__` |
| `depth_tolerance` | `0.005` m | `compute_occlusion_map()` |

Set `occlusion_update_interval > 1` to reduce compute at the cost of staleness
(e.g. `5` for ~5× speedup with single-viewpoint scenes).

### Usage

```python
occ = registry.occlusion_map
if occ:
    vis = occ.per_object_visibility.get("cup_1")
    if vis and vis.visible_fraction < 0.5:
        print("cup_1 is mostly occluded by", vis.occluding_objects)
        print("best viewpoint:", vis.best_viewpoint.position)
    for vol in occ.unobserved_volumes:
        if vol.estimated_capacity > 0:
            print(f"hidden volume behind {vol.blocking_objects}, "
                  f"fits ~{vol.estimated_capacity} small objects")
```

---

## §4 Surface Free-Space Map

**Files:** `src/perception/surface_map.py`, `DetectedObject.surface_map`

Populated for surface objects (tables, shelves, etc.) and any object that the
contact graph identifies as supporting ≥1 other object.

### Data model

```
DetectedObject.surface_map: Optional[SurfaceMap]  # None for non-surface objects
SurfaceMap
├── free_space_regions: List[FreeSpaceRegion]   # sorted by area desc
│   ├── polygon: (N, 2) metres    # convex hull in surface 2-D frame (U,V)
│   ├── area: float               # m²
│   ├── max_inscribed_circle: float  # placement radius, metres
│   └── neighboring_objects: [id]   # objects bordering this region
└── congestion_score: float         # [0,1] fraction of surface occupied
```

### Algorithm

1. Back-project the surface mask to 3-D; fit a plane via PCA → local (U,V) frame.
2. Project each resting object's AABB footprint onto (U,V).
3. Rasterise to a binary grid at `surface_map_resolution_m` (default 1 cm).
4. Distance transform of free cells → largest inscribed circle radius per region.
5. Connected-component labelling of free cells → one `FreeSpaceRegion` per
   component; neighbors identified by 1-cell dilation intersection.
6. `congestion_score = occupied_cells / total_cells`.

**Complexity:** O(A/r²) — A = surface area, r = resolution.
**Typical latency:** <5 ms per surface at 1 cm resolution for a 1 m² table.

### Key parameters

| Parameter | Default | Where set |
|---|---|---|
| `compute_surface_maps` | `True` | `GSAM2ObjectTracker.__init__` |
| `surface_map_resolution_m` | `0.01` m (1 cm) | `GSAM2ObjectTracker.__init__` |
| `surface_types` | `{"table","surface","shelf","tray","counter","desk"}` | `compute_surface_maps()` |

### Usage

```python
table = registry.get_object("table_1")
if table and table.surface_map:
    smap = table.surface_map
    print(f"congestion: {smap.congestion_score:.2f}")
    if smap.congestion_score > 0.8:
        print("surface is too cluttered — need multi-step rearrangement")
    if smap.free_space_regions:
        best = smap.free_space_regions[0]  # largest free region
        print(f"best placement area: {best.area:.3f} m², "
              f"radius: {best.max_inscribed_circle:.3f} m")
```

---

## §5 Update Frequency Triggers

The geometry representations are kept fresh via event-driven triggers in
addition to the background T1 loop.

| Trigger | Where | What happens |
|---|---|---|
| **T1** (background, 1 Hz) | `GSAM2ContinuousObjectTracker._tracking_loop` | Normal per-frame clearance + contact + surface + occlusion recompute.  If loop takes >2 s, throttle to 0.5 Hz (`t1_budget_s` param). |
| **T2** (sensing action) | `TaskAndMotionPlanner.plan_and_execute_task` — observe segment | Force occlusion history update for newly observed regions (`force_occlusion=True`). |
| **T3** (precondition failure) | `TaskOrchestrator.refine_domain_from_failure` | Full geometry recompute (contact + clearance + surfaces) from latest depth snapshot to correct state drift before replanning. |
| **T5** (post-manipulation) | `TaskAndMotionPlanner.execute_skill_plan` | Recompute contact graph + clearance from latest snapshot after each executed primitive, catching displaced/fallen objects. |

All triggers call `tracker.trigger_geometry_recompute()` which is a no-op on
the VLM `ContinuousObjectTracker` and active on `GSAM2ContinuousObjectTracker`.
`recompute_geometry()` on `GSAM2ObjectTracker` is the shared implementation used
by both background loops and trigger calls.

`t1_budget_s=2.0` means "warn and throttle to 0.5 Hz if a full detection cycle
takes more than 2 seconds".  Set to 0 to disable budget enforcement.
