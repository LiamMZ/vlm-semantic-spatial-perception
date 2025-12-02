# Object Re-Identification Plan

## Problem Statement

Currently, the object detection system treats each detection cycle independently, leading to:
1. **Duplicate objects**: Same physical object gets multiple IDs (e.g., `red_cup`, `red_cup_2`)
2. **State fragmentation**: Object state updates are not merged across detections
3. **No temporal consistency**: System doesn't leverage previous observations
4. **Registry bloat**: Registry grows unbounded with duplicate entries

## Goals

1. Enable the system to recognize when an object has been seen before
2. Merge/update existing object entries instead of creating duplicates
3. Track object state changes over time
4. Maintain a stable object ID throughout multiple observations
5. Leverage Gemini Robotics 1.5's vision capabilities for visual re-identification

## Current Detection Flow

```
detect_objects()
  ├─> _detect_object_names_streaming()  [Step 1: Find all objects]
  │     └─> Returns: [(object_name, bbox), ...]
  │
  ├─> _analyze_single_object()  [Step 2: Analyze each object]
  │     ├─> Detects affordances
  │     ├─> Finds interaction points
  │     ├─> Evaluates PDDL predicates
  │     └─> Creates DetectedObject
  │
  └─> registry.add_object()  [Step 3: Add to registry]
        └─> Always creates NEW entry (no checking)
```

**Issue**: Line 394 in object_tracker.py always calls `registry.add_object()` which creates a new entry.

## Proposed Solution: Multi-Stage Re-Identification

### Architecture Overview

```
detect_objects()
  ├─> Step 1: Initial Detection (Gemini Robotics 1.5)
  │     ├─> Detect objects with bounding boxes
  │     └─> Generate visual descriptors for each object
  │
  ├─> Step 2: Re-Identification Matching
  │     ├─> For each detected object:
  │     │     ├─> Extract visual features (position, color, shape)
  │     │     ├─> Check registry for similar objects
  │     │     ├─> Use VLM to confirm identity if match candidates found
  │     │     └─> Decide: NEW object or UPDATE existing
  │     │
  │     └─> Return: [(object_name, bbox, matched_id), ...]
  │
  ├─> Step 3: Detailed Analysis (parallel)
  │     └─> Only for NEW objects or changed objects
  │
  └─> Step 4: Registry Update/Merge
        ├─> NEW: Add to registry
        └─> EXISTING: Merge/update state
```

## Implementation Plan

### Phase 1: Registry Enhancement (Foundation)

**File**: `src/perception/object_registry.py`

Add methods to `DetectedObjectRegistry`:

```python
def find_similar_objects(
    self,
    position_3d: Optional[np.ndarray],
    bbox_2d: Optional[List[int]],
    object_type: str,
    similarity_threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """
    Find objects in registry similar to the query.

    Returns:
        List of (object_id, similarity_score) tuples, sorted by score
    """
    # Spatial similarity (3D position)
    # Visual similarity (bbox overlap, size)
    # Type similarity (object_type matching)
    pass

def update_object_state(
    self,
    object_id: str,
    new_observation: DetectedObject,
    merge_strategy: str = "latest"
) -> bool:
    """
    Update existing object with new observation.

    Strategies:
    - "latest": Replace all fields with new values
    - "merge": Combine observations (average positions, merge predicates)
    - "confidence": Keep higher confidence values

    Returns:
        True if updated successfully
    """
    pass

def merge_objects(
    self,
    primary_id: str,
    duplicate_id: str
) -> bool:
    """
    Merge duplicate object into primary, remove duplicate.

    Useful when re-identification discovers duplicates.
    """
    pass
```

### Phase 2: Visual Re-Identification Prompt

**File**: `config/prompts_config.yaml`

Add new prompt section for re-identification:

```yaml
re_identification:
  detection_with_context: |
    You are analyzing a scene to detect objects.

    Previously detected objects (may or may not still be visible):
    {previous_objects}

    For each object in the current image:
    1. Detect the object with bounding box [y1, x1, y2, x2] (0-1000 scale)
    2. Determine if this is the SAME object as any previously detected, or a NEW object
    3. If SAME: provide the matching ID and confidence
    4. If NEW: provide a descriptive name

    Output format (streaming):
    OBJECT: <name or matched_id> | <bbox> | <match_confidence 0-1> | <is_new true/false>
    ...
    END

  visual_confirmation: |
    Compare these two object observations and determine if they are the SAME physical object.

    Object A (from registry):
    - Name: {obj_a_name}
    - Type: {obj_a_type}
    - Properties: {obj_a_properties}
    - Location: {obj_a_position}
    - Visual crop: <image_a>

    Object B (current detection):
    - Name: {obj_b_name}
    - Type: {obj_b_type}
    - Location: {obj_b_position}
    - Visual crop: <image_b>

    Consider:
    - Visual appearance (color, shape, texture, markings)
    - Spatial proximity (could object have moved this far?)
    - Object type consistency
    - Unique identifying features

    Return JSON:
    {{
      "is_same_object": <true/false>,
      "confidence": <0.0-1.0>,
      "reasoning": "<brief explanation>",
      "identified_features": ["<feature1>", "<feature2>", ...]
    }}
```

### Phase 3: Object Descriptor System

**File**: `src/perception/object_descriptor.py` (NEW)

```python
@dataclass
class ObjectDescriptor:
    """Visual and spatial descriptor for object re-identification."""

    # Spatial features
    position_3d: Optional[np.ndarray]  # [x, y, z] in meters
    bbox_2d: Optional[List[int]]  # [y1, x1, y2, x2] normalized
    bbox_center_2d: Tuple[int, int]  # Center point
    bbox_area: float  # Normalized area

    # Visual features (extracted from VLM)
    dominant_colors: List[str]  # ["red", "metallic", ...]
    shape_category: str  # "cylindrical", "rectangular", "irregular"
    distinctive_features: List[str]  # ["has_handle", "striped_pattern", ...]
    material: Optional[str]  # "plastic", "metal", "ceramic"

    # Temporal
    last_seen: float  # timestamp
    observation_count: int  # how many times seen

    # Confidence
    descriptor_confidence: float  # 0-1

def compute_similarity(desc_a: ObjectDescriptor, desc_b: ObjectDescriptor) -> float:
    """
    Compute similarity score between two descriptors.

    Weighted combination of:
    - Spatial distance (3D position)
    - Visual overlap (bbox IoU)
    - Feature matching (colors, shape, distinctive features)
    - Size consistency

    Returns: similarity score 0.0-1.0
    """
    pass

def extract_descriptor_from_detection(
    obj: DetectedObject,
    vlm_analysis: Dict
) -> ObjectDescriptor:
    """Extract descriptor from detection and VLM response."""
    pass
```

### Phase 4: Modified Detection Flow

**File**: `src/perception/object_tracker.py`

Update `detect_objects()` method:

```python
async def detect_objects(
    self,
    color_frame: Union[np.ndarray, Image.Image],
    depth_frame: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[Any] = None,
    temperature: Optional[float] = None,
    enable_reidentification: bool = True  # NEW parameter
) -> List[DetectedObject]:
    """Detect objects with re-identification support."""

    # Step 1: Get context from registry (previously seen objects)
    previous_objects = []
    if enable_reidentification and self.registry.count() > 0:
        previous_objects = self._format_previous_objects_for_context()

    # Step 2: Detection with context (streaming)
    detection_candidates = []
    await self._detect_with_reidentification(
        image=pil_image,
        previous_objects=previous_objects,
        callback=lambda obj_data: detection_candidates.append(obj_data)
    )

    # Step 3: Process each candidate
    final_objects = []
    for candidate in detection_candidates:
        if candidate['is_new']:
            # New object - full analysis needed
            obj = await self._analyze_single_object(...)
            self.registry.add_object(obj)
            final_objects.append(obj)
        else:
            # Matched existing object - verify and update
            matched_id = candidate['matched_id']
            existing_obj = self.registry.get_object(matched_id)

            if candidate['match_confidence'] < 0.8:
                # Low confidence - do visual confirmation
                confirmed = await self._confirm_object_identity(
                    existing_obj, candidate, pil_image
                )
                if not confirmed:
                    # False match - treat as new
                    obj = await self._analyze_single_object(...)
                    self.registry.add_object(obj)
                    final_objects.append(obj)
                    continue

            # High confidence match - update existing
            updated_obj = await self._update_existing_object(
                existing_obj, candidate, pil_image, depth_frame, camera_intrinsics
            )
            self.registry.update_object(matched_id, updated_obj)
            final_objects.append(updated_obj)

    return final_objects
```

New helper methods:

```python
def _format_previous_objects_for_context(self) -> List[Dict]:
    """Format registry objects for VLM context."""
    objects = self.registry.get_all_objects()
    return [
        {
            "id": obj.object_id,
            "type": obj.object_type,
            "position_2d": obj.position_2d,
            "properties": obj.properties,
            "last_seen": obj.timestamp
        }
        for obj in objects[-20:]  # Only recent 20 objects to avoid token overflow
    ]

async def _detect_with_reidentification(
    self,
    image: Image.Image,
    previous_objects: List[Dict],
    callback: Callable
):
    """Detection prompt that includes previous object context."""
    # Uses re_identification.detection_with_context prompt
    pass

async def _confirm_object_identity(
    self,
    existing_obj: DetectedObject,
    candidate: Dict,
    current_image: Image.Image
) -> bool:
    """Use VLM to visually confirm two detections are same object."""
    # Crop both images to object regions
    # Use re_identification.visual_confirmation prompt
    # Return True if confirmed same
    pass

async def _update_existing_object(
    self,
    existing_obj: DetectedObject,
    new_observation: Dict,
    image: Image.Image,
    depth_frame: Optional[np.ndarray],
    camera_intrinsics: Optional[Any]
) -> DetectedObject:
    """Update existing object with new observation."""
    # Update position (moving average or latest)
    # Update PDDL state (re-evaluate predicates)
    # Update confidence (boost if seen multiple times)
    # Increment observation_count
    # Update timestamp
    pass
```

### Phase 5: Configuration Options

**File**: `src/perception/object_tracker.py` `__init__`

```python
def __init__(
    self,
    # ... existing params ...
    enable_reidentification: bool = True,
    reidentification_confidence_threshold: float = 0.8,
    reidentification_spatial_threshold: float = 0.5,  # meters
    max_context_objects: int = 20,  # Max previous objects to include in prompt
    reidentification_model: str = "gemini-robotics-er-1.5-preview",  # Specific model
):
```

## Benefits

### 1. **Consistent Object Identity**
   - Same physical object maintains same ID across observations
   - Enables tracking objects as they move
   - Simplifies reasoning about object states

### 2. **Accurate State Tracking**
   - PDDL predicate states updated over time
   - Can detect state changes (e.g., `opened` → `closed`)
   - Historical tracking of object properties

### 3. **Reduced Computational Cost**
   - Skip full analysis for already-known objects
   - Only re-evaluate changed predicates
   - Faster detection cycles

### 4. **Better Planning**
   - Stable object IDs improve plan validity
   - Can track goal achievement over time
   - More reliable initial state for PDDL

### 5. **Spatial Reasoning**
   - Track object movements
   - Detect when objects appear/disappear
   - Maintain scene consistency

## Implementation Phases & Testing

### Phase 1: Foundation (Week 1)
- [ ] Enhance registry with similarity search
- [ ] Add object update/merge methods
- [ ] Unit tests for registry operations

### Phase 2: Descriptors (Week 1-2)
- [ ] Implement ObjectDescriptor dataclass
- [ ] Add descriptor extraction from VLM responses
- [ ] Similarity computation with weights
- [ ] Test descriptor matching accuracy

### Phase 3: Re-ID Prompts (Week 2)
- [ ] Design and test re-identification prompts
- [ ] Add visual confirmation prompts
- [ ] Validate with Gemini Robotics 1.5 model
- [ ] Tune confidence thresholds

### Phase 4: Integration (Week 3)
- [ ] Modify detect_objects() flow
- [ ] Implement detection with context
- [ ] Add visual confirmation logic
- [ ] Add object update logic
- [ ] Integration tests

### Phase 5: Optimization (Week 4)
- [ ] Tune similarity thresholds
- [ ] Optimize context window size
- [ ] Add caching for descriptors
- [ ] Performance benchmarking
- [ ] End-to-end testing with TAMP

## Testing Strategy

### Unit Tests
```python
def test_object_similarity_matching():
    """Test that similar objects get matched correctly."""
    # Create two observations of same object
    # Verify similarity score > threshold
    # Verify different objects score < threshold

def test_registry_update():
    """Test that updates merge correctly."""
    # Add object to registry
    # Detect same object with new state
    # Verify state was updated, not duplicated

def test_false_positive_rejection():
    """Test that similar but different objects aren't merged."""
    # Detect "red_cup" and "red_bowl"
    # Verify both get unique IDs
```

### Integration Tests
```python
async def test_multi_cycle_reidentification():
    """Test re-identification over multiple detection cycles."""
    # Cycle 1: Detect red_cup, blue_cup
    # Cycle 2: Detect same cups (verify IDs preserved)
    # Cycle 3: Move red_cup, detect again (verify ID preserved, position updated)
    # Cycle 4: Add green_cup (verify new ID created)

async def test_state_change_tracking():
    """Test that state changes are detected."""
    # Cycle 1: Detect box with opened=False
    # Cycle 2: Open box manually
    # Cycle 3: Detect box (verify opened=True, same ID)
```

## Gemini Robotics 1.5 Capabilities to Leverage

Based on Google's documentation, Gemini Robotics 1.5 has enhanced capabilities for:

1. **Visual Grounding**: Precise object localization with bounding boxes
2. **Object Re-Identification**: Can recognize objects across frames
3. **Spatial Reasoning**: Understanding object relationships and positions
4. **Temporal Consistency**: Maintaining context across multiple observations
5. **Fine-Grained Visual Details**: Detecting distinctive features

We should specifically use this model for:
- Initial detection with context (leverage temporal consistency)
- Visual confirmation (leverage re-identification capabilities)
- Descriptor extraction (leverage fine-grained details)

## Fallback Strategy

If re-identification confidence is low or unclear:
1. Create new object with temporary ID
2. Flag for manual verification
3. Provide both match candidates to higher-level system
4. Let task orchestrator decide based on planning context

## Configuration Knobs

Users should be able to tune:
- `enable_reidentification`: Toggle feature on/off
- `confidence_threshold`: How certain to be before matching (0.0-1.0)
- `spatial_threshold`: Max distance for same object (meters)
- `visual_confirmation_enabled`: Use second VLM call for low-confidence matches
- `max_context_objects`: Limit previous objects in prompt (token management)
- `descriptor_weights`: Tune importance of spatial vs visual features

## Metrics to Track

- **Re-identification accuracy**: % of correct matches
- **False positive rate**: % of incorrect merges
- **False negative rate**: % of missed matches (duplicates created)
- **Latency impact**: Additional time per detection cycle
- **Token usage**: Cost impact of context and confirmation
- **Registry size**: Growth rate over time

## Future Enhancements

1. **Visual embeddings**: Store compressed visual features for fast matching
2. **Persistent storage**: Save registry to disk, load on startup
3. **Scene graphs**: Track object relationships and spatial structure
4. **Action-based tracking**: Update object positions based on robot actions
5. **Multi-camera fusion**: Merge observations from multiple viewpoints
6. **Temporal filtering**: Smooth object positions over time
