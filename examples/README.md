# Examples

Demonstration scripts for the VLM Semantic Spatial Perception system.

## Main Demos

### [object_tracker_demo.py](object_tracker_demo.py) - Object Detection Demo

**Interactive demonstration of VLM-based object detection.**

```bash
python examples/object_tracker_demo.py
```

**What it does:**
1. Captures RGB-D scene from RealSense camera (or uses synthetic image)
2. Detects objects using Gemini VLM with affordances and interaction points
3. Visualizes detection results with bounding boxes and affordance markers
4. Provides interactive queries for detected objects

**Key features:**
- ✅ Real VLM detection with Gemini Robotics-ER
- ✅ Affordance detection (graspable, pourable, etc.)
- ✅ Interaction point detection for manipulation
- ✅ 3D position computation with depth
- ✅ Visualization with OpenCV
- ✅ Interactive query interface

**Requirements:**
- RealSense camera (or will use synthetic image)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` set in `.env` file

**Interactive features:**
- Query objects by affordance
- Update interaction points with task context
- View objects by type
- Save/load detections to JSON

---

### [pddl_predicate_tracking_demo.py](pddl_predicate_tracking_demo.py) - PDDL Predicate Tracking

**Real-time PDDL generation with continuous object tracking and state updates.**

```bash
python examples/pddl_predicate_tracking_demo.py
```

**Interactive workflow:**
1. **User provides task** - Natural language description (e.g., "Clean the mug and place it on the shelf")
2. **LLM task analysis** - Semantically analyzes task to extract PDDL predicates (no keyword matching)
3. **PDDL initialization** - Creates domain with LLM-extracted predicates BEFORE tracking
4. **Tracker seeding** - Seeds continuous tracker with PDDL predicates from domain
5. **Continuous tracking** - Detects objects and predicate states in real-time (10 seconds)
6. **Live PDDL updates** - Initial state updates as new objects are detected
7. **Goal generation** - Extracts goal state from LLM analysis
8. **Final PDDL files** - Generates domain.pddl and problem.pddl for planning

**Key features:**
- ✅ **Interactive task input** - User defines task after startup
- ✅ **LLM-based predicate extraction** - No keyword matching, semantic understanding
- ✅ **PDDL-first architecture** - PDDL initialized before tracker to seed predicates
- ✅ **Continuous updates** - PDDL state updates in real-time as objects appear
- ✅ **Camera or simulation** - Works with RealSense camera or simulated detection
- ✅ **Complete pipeline** - Task → LLM Analysis → PDDL Domain → Tracker → Planning

**LLM extracts:**
- Task-relevant predicates (semantic, not keyword-based)
- Required actions for the task
- Goal conditions and constraints
- Object types and relationships

**Use this for:**
- Real-time perception-to-planning integration
- Continuous world state monitoring
- Task-specific predicate tracking
- Dynamic PDDL generation from vision

---

### [task_monitoring_demo.py](task_monitoring_demo.py) - Task Monitoring Demo

**Demonstrates PDDL domain maintenance and task state monitoring.**

```bash
python examples/task_monitoring_demo.py
```

**What it does:**
1. Analyzes task and initializes PDDL domain
2. Simulates object observations
3. Updates PDDL domain incrementally
4. Monitors task state (EXPLORE, REFINE, PLAN_AND_EXECUTE)
5. Generates final PDDL files

**Key features:**
- ✅ LLM-driven task analysis
- ✅ Incremental domain updates
- ✅ Adaptive state monitoring
- ✅ Goal object tracking with fuzzy matching
- ✅ Complete PDDL generation workflow

**Use this for:**
- Understanding the planning system
- Testing task analysis
- Debugging PDDL generation
- Learning state monitoring logic

---

### [pddl_representation_demo.py](pddl_representation_demo.py) - PDDL Representation Demo

**Demonstrates PDDL domain and problem construction.**

```bash
python examples/pddl_representation_demo.py
```

**What it does:**
- Shows how to build PDDL domains programmatically
- Demonstrates predicates, actions, types
- Shows problem instance creation
- Generates domain.pddl and problem.pddl files

**Use this for:**
- Learning PDDL structure
- Testing PDDL generation
- Understanding domain/problem separation

---

### [continuous_pddl_simple_demo.py](continuous_pddl_simple_demo.py) - Continuous PDDL Integration (Simple)

**NEW! Complete continuous integration with auto-stop - perfect for testing.**

```bash
python examples/continuous_pddl_simple_demo.py
```

**What it does:**
1. **Task Analysis** - Analyzes task and generates initial PDDL domain
2. **Continuous Tracking** - Runs background object detection loop
3. **Live PDDL Updates** - Updates domain after each detection cycle
4. **State Monitoring** - Monitors task state and readiness
5. **Auto-Stop** - Stops when ready for planning or max cycles reached
6. **PDDL Generation** - Generates final domain.pddl and problem.pddl

**Key features:**
- ✅ **Fully automated** - No complex input handling
- ✅ **Continuous loop** - Background detection with callbacks
- ✅ **Live updates** - PDDL domain evolves as objects detected
- ✅ **Task-aware** - Monitors state and knows when ready
- ✅ **Clean output** - Progress updates after each cycle
- ✅ **Configurable** - Set max cycles and update interval

**Configuration:**
```python
max_cycles = 5          # Stop after 5 detection cycles
update_interval = 3.0   # 3 seconds between detections
```

**Example output:**
```
CYCLE 1/5 - Detected 3 objects
================================
�� PDDL Update:
  • New objects: 3
  • Total objects: 3
  • Goal objects found: mug
  • Still missing: shelf

🎯 Task State: EXPLORE (85%)
  Reasoning: Goal objects partially found...

🔍 Detected Objects:
  • mug: 1
  • table: 1
  • box: 1

CYCLE 2/5 - Detected 4 objects
================================
...

✅ READY FOR PLANNING!
```

**Use this for:**
- Testing continuous integration
- Quick demonstrations
- Automated workflows
- Performance benchmarking

---

### [continuous_pddl_integration_demo.py](continuous_pddl_integration_demo.py) - Continuous PDDL Integration (Advanced)

**Advanced continuous integration with interactive controls.**

```bash
python examples/continuous_pddl_integration_demo.py
```

**What it does:**
- Same as simple demo, but with interactive commands
- User can query status, stop manually, or let it run
- More detailed progress information
- Suitable for presentations and debugging

**Interactive commands:**
```
Commands:
  'status' - Show current status
  'stop'   - Stop tracking and generate PDDL
  'quit'   - Quit without generating PDDL
```

**Use this for:**
- Interactive demonstrations
- Debugging integration issues
- Manual control over tracking
- Real-time system monitoring

---

## Quick Start

### First Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Connect camera** (for full demo)
   - Plug in RealSense camera
   - Arrange objects in view

4. **Test setup:**
   ```bash
   python examples/object_tracker_demo.py
   ```

### Running the Continuous Demo

```bash
python examples/continuous_pddl_simple_demo.py
```

The demo will:
1. Analyze the task and initialize PDDL domain
2. Start continuous object detection in background
3. Update PDDL domain after each detection cycle
4. Monitor task state and stop when ready
5. Generate final PDDL files

### Without Camera

The demos will gracefully handle missing camera by using synthetic images for testing.

## Comparison

| Demo | Objects | Tasks | VLM | Continuous | Interactive | Best For |
|------|---------|-------|-----|------------|-------------|----------|
| **continuous_pddl_simple_demo.py** | Real VLM | Pre-set | ✅ | ✅ | ⚪ | Continuous integration, auto-run |
| **continuous_pddl_integration_demo.py** | Real VLM | Pre-set | ✅ | ✅ | ✅ | Continuous with manual control |
| **pddl_predicate_tracking_demo.py** | Real VLM | User input | ✅ | ✅ | ✅ | Real-time PDDL tracking |
| **object_tracker_demo.py** | Real VLM | N/A | ✅ | ❌ | ✅ | Object detection & visualization |
| **task_monitoring_demo.py** | Simulated | Pre-set | ✅ | ❌ | ❌ | Planning system demo |
| **pddl_representation_demo.py** | N/A | Pre-set | ❌ | ❌ | ❌ | PDDL structure demo |

**Legend:**
- ✅ = Full support
- ⚪ = Minimal (auto-stop on ready)
- ❌ = Not supported
- N/A = Not applicable

## Tips

### Getting Good Detection Results

1. **Lighting:** Ensure good, even lighting
2. **Object placement:** Arrange objects with clear separation
3. **Camera angle:** Position camera to see objects clearly
4. **Variety:** Mix different object types and colors
5. **Background:** Use contrasting background

### Formulating Tasks

**Good tasks reference detected objects:**
```
✓ "Pick up the red cup and place it on the shelf"
✓ "Move all graspable objects to the container"
✓ "Organize items by color"
```

**Avoid tasks with undetected objects:**
```
✗ "Pick up the wrench" (if no wrench detected)
✗ "Open the drawer" (if no drawer detected)
```

### Performance

- **First detection:** 10-15s (model initialization)
- **Subsequent detections:** 3-8s (cached model)
- **Task analysis:** 2-5s per task
- **PDDL generation:** <0.1s

### Troubleshooting

**No objects detected:**
- Check camera connection: `python -c "from src.camera import RealSenseCamera; cam = RealSenseCamera()"`
- Verify API key: `python examples/object_tracker_demo.py`
- Ensure objects are visible in camera view
- Check lighting conditions

**Detection errors:**
- Run diagnostic: `python examples/object_tracker_demo.py`
- Check API key validity (should be `GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- Verify model availability (uses `gemini-2.0-flash-exp` by default)
- Check internet connection

**PDDL generation fails:**
- Ensure objects were detected successfully
- Check task references detected objects
- Verify LLM API key is valid
- Try running `python examples/task_monitoring_demo.py` to test planning system

## Output Files

Generated files are saved to `outputs/pddl/task_N/`:

```
outputs/pddl/
├── task_1/
│   ├── domain.pddl    # Domain definition
│   └── problem.pddl   # Problem instance
├── task_2/
│   ├── domain.pddl
│   └── problem.pddl
└── ...
```

Use these PDDL files with planners like Fast Downward, ENHSP, or other PDDL-compatible planners.

**NEW**: See [pddl_solver_demo.py](pddl_solver_demo.py) for integrated solver usage with automatic backend detection.

---

### [test_predicate_validation.py](test_predicate_validation.py) - Predicate Validation Test

**Demonstrates automatic predicate validation and addition.**

```bash
python examples/test_predicate_validation.py
```

**What it does:**
1. Creates a PDDL domain with explicit predicates
2. Adds actions that reference undefined predicates
3. Runs automatic validation to detect missing predicates
4. Auto-adds missing predicates to ensure domain consistency
5. Generates valid PDDL domain file

**Key features:**
- ✅ **Automatic detection** - Finds predicates used in actions but not defined
- ✅ **Auto-fix** - Adds missing predicates with generic signatures
- ✅ **Validation** - Ensures all actions are parseable and valid
- ✅ **Zero manual work** - No need to manually track predicate definitions

**Example scenario:**
```python
# Define only 2 predicates explicitly
pddl.add_predicate("graspable", [("obj", "object")])
pddl.add_predicate("empty-hand", [])

# Add actions that use MORE predicates
pddl.add_action("pick_up",
    precondition="(and (graspable ?obj) (empty-hand))",
    effect="(holding ?obj)"  # 'holding' not defined!
)

# Validation automatically detects and adds 'holding'
await maintainer.validate_and_fix_action_predicates()
# Output: ✓ Auto-added 1 missing predicate: holding
```

**Use this for:**
- Testing predicate validation logic
- Understanding automatic domain repair
- Debugging PDDL generation issues
- Learning domain consistency checks

---

## Next Steps

After running the demos:

1. **Experiment with different scenes** - Try various object arrangements
2. **Test complex tasks** - Multi-step manipulation tasks
3. **Integrate with planner** - Use generated PDDL with a planner
4. **Add custom objects** - Test with domain-specific objects
5. **Extend capabilities** - Add new affordances or actions

## Support

For issues or questions:
- Check module documentation in [docs/](../docs/)
  - [Perception System](../docs/perception.md)
  - [Planning System](../docs/planning.md)
  - [Camera System](../docs/camera.md)
- Run example scripts to test specific components
