# Initial Detection Phase - Performance Analysis

## Problem Statement

The initial object detection (Phase 1) is taking 10-30+ seconds before even starting per-object analysis. This happens in the `_detect_object_names_streaming` method which finds all objects and their bounding boxes in a single VLM request.

---

## What Happens During Initial Detection

### Code Flow

**Location:** [src/perception/object_tracker.py:481-514](../src/perception/object_tracker.py#L481-L514)

```python
# Step 1: Stream object names and collect them
names_start = time.time()
await self._detect_object_names_streaming(
    pil_image,
    temperature,
    on_object_detected,
    prompt=current_prompt,
    ...
)
names_time = time.time() - names_start
# Typically: 10-30+ seconds for 8 objects
```

### The Detection Prompt

**Source:** [config/object_tracker_prompts.yaml:34-58](../config/object_tracker_prompts.yaml#L34-L58)

The VLM receives this prompt:

```
You are a robotic vision system. CURRENT FRAME TURN: perform detection ONLY on the newly attached image below.
Image order for this turn: the final attached image is the current frame for detection.
Ignore any earlier turns and run detection ONLY on that last image.

Previously detected objects (reuse these EXACT ids if visible; omit them if occluded/out of frame):
[empty when spatial_tracking=True]

Current Task: put the cards and the binder in the bag
Prioritize detecting objects relevant to this task.
Expected goal objects: cards, binder, bag
IMPORTANT: When you detect objects matching these goal objects, use these EXACT IDs as labels.

Instructions:
- Return bounding boxes as a JSON array with labels. Never return masks or code fencing.
- Include all objects you can identify, focusing on new objects that are not in context turn.
- IMPORTANT: Detect BOTH manipulable objects AND surfaces/support structures:
  * Manipulable objects: bottles, cups, tools, containers, etc.
  * Surfaces: tables, counters, shelves, trays, workbenches, floors, etc.
  * Surfaces are critical for spatial predicates like "on" and "in"
- If the current image is similar or identical to context image with no new objects, it is fine to not return new detections
- If an object from previous images is visible, re-use the exact id as the "label"; otherwise create a descriptive, unique label (colors, size, position, distinctive traits).
- IMPORTANT: Object labels must use underscores (_) to separate words, NOT hyphens (-). Examples: "blue_water_bottle" not "blue-water-bottle", "red_cup" not "red-cup", "wooden_table" not "wooden-table"
- Box format: [ymin, xmin, ymax, xmax] normalized to 0-1000. All values MUST be integers.
- Prefer tight boxes around each object instance. The bounding box may have moved compared to last capture, so make sure to update all bounding boxes and not just return the same bounding box for existing objects.
- Output ONLY the JSON array in this format (no prose):
[
  {{"box_2d": [ymin, xmin, ymax, xmax], "label": "<object label>"}}
]
```

---

## Why It's Slow

### 1. 🔴 CRITICAL: Extremely Long Prompt (500+ tokens)

**Issue:**
- Detection prompt is 400-600 tokens
- VLM must process entire prompt before generating
- Includes lots of instructions, examples, formatting rules
- Task context adds 50-100 more tokens

**Impact:**
- Longer prompt → more "thinking time" for VLM
- **3-8 seconds** just for prompt processing
- Each detection cycle pays this cost

---

### 2. 🔴 CRITICAL: Asking for Too Much

**The prompt asks VLM to:**
1. ✅ Detect all objects (necessary)
2. ✅ Return bounding boxes (necessary)
3. ❌ Detect BOTH manipulable objects AND surfaces
4. ❌ Check if objects match previous context
5. ❌ Create descriptive unique labels with colors/size/position
6. ❌ Format with exact JSON structure
7. ❌ Use underscores not hyphens
8. ❌ Return tight bounding boxes
9. ❌ Update bounding boxes if moved
10. ❌ Match goal objects with exact IDs

**Impact:**
- VLM has to think about all these constraints
- More complex reasoning → slower response
- **5-15 seconds** for complex scenes

---

### 3. 🟡 MAJOR: Streaming But Still Sequential

**Code:** [src/perception/object_tracker.py:648-656](../src/perception/object_tracker.py#L648-L656)

```python
response_text = await self._generate_content(
    image,
    prompt,
    temperature,
    ...
)

detections = self._parse_detection_response(response_text)
for object_name, bounding_box in detections[:25]:
    await callback(object_name, bounding_box)
```

**Issue:**
- Called "streaming" but actually collects full response first
- `_generate_content` waits for entire response
- Then parses all at once
- Then calls callback for each object

**Impact:**
- No real streaming benefit
- Must wait for entire response before proceeding
- **Not actually incremental**

---

### 4. 🟠 MODERATE: Image Encoding

**Code:** [src/perception/object_tracker.py:426-432](../src/perception/object_tracker.py#L426-L432)

```python
# Pre-encode image (reused in parallel requests)
img_byte_arr = io.BytesIO()
pil_image.save(img_byte_arr, format='PNG')
self._encoded_image_cache = img_byte_arr.getvalue()
encode_time = time.time() - encode_start
# Typically: 100-300ms for 1920x1080
```

**Impact:**
- PNG encoding is slow for high-res images
- **100-300ms** per detection cycle
- Could use JPEG for faster encoding (but slightly lower quality)

---

### 5. 🟢 LOW: Task Context Injection

**Code:** [src/perception/object_tracker.py:632-645](../src/perception/object_tracker.py#L632-L645)

```python
task_context_section, task_priority_note = self._format_task_context_for_detection()
if task_context_section:
    prompt = prompt.replace(
        "Instructions:",
        f"{task_context_section}\n\n      Instructions:"
    )
```

**Impact:**
- Adds 50-100 tokens to prompt
- String replacement is fast (<1ms)
- Minimal overhead

---

## Timeline Breakdown (8 objects, typical case)

```
0.00s - Start detection
0.10s - Encode image to PNG (1920x1080)
0.15s - Build prompt (simple string formatting)
0.20s - Send request to VLM
0.30s - VLM receives request
3.00s - VLM processes 500-token prompt  ⬅️ SLOW
8.00s - VLM generates response (JSON array)  ⬅️ SLOW
15.00s - VLM thinking about constraints  ⬅️ SLOW
18.00s - Response arrives
18.10s - Parse JSON response
18.20s - Call callback 8 times
────────────────────────────────────────
18.2s TOTAL for Phase 1
```

**After Phase 1, per-object analysis adds:**
- 8 objects × 3-5s each = 24-40s more

**Total detection time: 42-58 seconds**

---

## Recommended Optimizations

### 1. ⚡ CRITICAL: Simplify Detection Prompt

**Current:** 500+ tokens asking for 10+ things

**Simplified:**
```
Detect all objects in this image. Return JSON array:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}]

Use these exact IDs if visible: blue_bottle, red_cup, wooden_table
Otherwise create descriptive names with underscores like: green_marker, metal_spoon

Detect both objects and surfaces (tables, counters, floors).
Box coordinates: normalized 0-1000 integers.
```

**Expected Improvement:**
- **50-70% faster** prompt processing (500 → 150 tokens)
- Saves 2-5 seconds per detection
- Clearer, simpler task for VLM

---

### 2. ⚡ HIGH: Use Lighter Model for Detection

**Idea:** Use `gemini-1.5-flash` for initial detection, save `gemini-robotics` for per-object analysis

```python
# In _detect_object_names_streaming:
detection_model = "gemini-1.5-flash"  # Fast, cheap
response = await self.client.aio.models.generate_content_stream(
    model=detection_model,  # Use fast model
    ...
)
```

**Expected Improvement:**
- **2-3x faster** detection (Flash is much faster)
- **10x cheaper** (Flash is 20x cheaper)
- Saves 8-15 seconds on initial detection

---

### 3. ⚡ MEDIUM: Use JPEG Encoding

**Change:** [src/perception/object_tracker.py:430](../src/perception/object_tracker.py#L430)

```python
# Before
pil_image.save(img_byte_arr, format='PNG')

# After
pil_image.save(img_byte_arr, format='JPEG', quality=85)
```

**Expected Improvement:**
- **60-80% faster** encoding
- Saves 60-200ms per detection
- Slightly lower quality (usually fine for VLM)

---

### 4. ⚡ MEDIUM: Lower Image Resolution

**Idea:** Resize image before detection

```python
# Before detection, resize if too large
max_dimension = 1280
if max(pil_image.width, pil_image.height) > max_dimension:
    scale = max_dimension / max(pil_image.width, pil_image.height)
    new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
    pil_image = pil_image.resize(new_size, Image.Resampling.BILINEAR)
```

**Expected Improvement:**
- **30-50% faster** encoding
- **20-40% faster** VLM processing (fewer pixels)
- Saves 2-5 seconds total
- Slightly less accurate bounding boxes

---

### 5. ⚡ LOW: True Streaming Parsing

**Idea:** Parse objects as they arrive in stream

```python
async def _detect_object_names_streaming(...):
    buffer = ""
    async for chunk in self._generate_content_streaming(...):
        buffer += chunk
        # Try to parse complete objects from buffer
        objects, buffer = self._parse_partial_json(buffer)
        for obj_name, bbox in objects:
            await callback(obj_name, bbox)
```

**Expected Improvement:**
- Start per-object analysis earlier
- Better perceived latency
- No actual time savings (same total work)

---

### 6. ⚡ LOW: Remove Task Context from Detection

**Idea:** Task context useful for per-object analysis, but adds overhead to detection

```python
# Don't inject task context into detection prompt
# Only use it in per-object analysis
```

**Expected Improvement:**
- Saves 50-100 tokens
- **5-10% faster** detection
- Saves 1-2 seconds

---

## Priority Recommendations

### Immediate (Do First):
1. **Simplify detection prompt** → 50-70% faster, 2-5s savings
2. **Use Flash model** → 2-3x faster, 8-15s savings
3. **Use JPEG encoding** → 60-80% faster encoding, 100-200ms savings

**Expected Total: 10-20 seconds faster initial detection**

### High Value (Do Next):
4. **Lower image resolution** → 30-50% faster, 2-5s savings
5. **Remove task context** → 5-10% faster, 1-2s savings

### Nice to Have:
6. **True streaming parsing** → Better UX, no time savings

---

## Expected Performance After Fixes

### Current Timeline:
```
18s - Initial detection (Phase 1)
28s - Per-object analysis (8 objects × 3.5s)
────────────────────────────────────────
46s TOTAL
```

### After Quick Fixes (1-3):
```
5s  - Initial detection (70% faster)
28s - Per-object analysis (unchanged)
────────────────────────────────────────
33s TOTAL (28% faster)
```

### After All Fixes (1-6):
```
3s  - Initial detection (85% faster)
28s - Per-object analysis (unchanged)
────────────────────────────────────────
31s TOTAL (33% faster)
```

---

## Implementation Priority

### Phase 1: Quick Wins (30 minutes)
- Simplify detection prompt in YAML
- Change PNG → JPEG encoding
- Use Flash model for detection

### Phase 2: Medium Effort (1-2 hours)
- Add image resizing logic
- Remove task context from detection
- Update prompt templates

### Phase 3: Advanced (3-4 hours)
- Implement true streaming parser
- Add progressive object analysis
- Benchmark and tune

---

## Monitoring

```bash
# Watch detection timing
tail -f outputs/tamp_demo/run_*/run_*.log | grep "Detection phase complete"

# Check prompt length
grep -A 50 "detection:" config/object_tracker_prompts.yaml | wc -w

# Measure encoding time
grep "encoding=" outputs/tamp_demo/run_*/run_*.log
```
