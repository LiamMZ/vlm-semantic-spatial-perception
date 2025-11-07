# Examples

Demonstration scripts for the VLM Semantic Spatial Perception system.

## Main Demos

### [dynamic_pddl_demo.py](dynamic_pddl_demo.py) - Interactive Full Pipeline Demo

**The complete, real-world demonstration of the system.**

```bash
python examples/dynamic_pddl_demo.py
```

**What it does:**
1. Captures RGB-D scene from RealSense camera
2. Detects objects using Gemini VLM (real-time, no mock data)
3. Builds world model with spatial relationships
4. **Shows you detected objects and prompts for tasks (interactive!)**
5. Analyzes tasks with LLM in scene context
6. Generates task-specific PDDL files

**Key features:**
- ✅ NO mock data - all objects from real VLM detection
- ✅ NO pre-defined tasks - you specify tasks after seeing objects
- ✅ Dynamic affordance inference from visual observation
- ✅ Scene-grounded planning

**Requirements:**
- RealSense camera connected and visible objects in view
- `GEMINI_API_KEY` set in `.env` file
- Objects arranged in camera's field of view

**Example interaction:**
```
Detected objects in scene:
  • red_cup_1: cup (red) - can be graspable, containable
  • blue_bottle_1: bottle (blue) - can be graspable, pourable
  • table_1: table - can be supportable
  • shelf_1: shelf - can be supportable, containable

Enter robotic manipulation tasks (one per line).
Examples:
  - Pick up the red cup and place it on the shelf
  - Move all graspable objects to the container
  - Organize the workspace by color

Task 1: Pick up the red cup and place it on the shelf
Task 2: Move the blue bottle next to the red cup
Task 3: [press enter to finish]
```

**Output:**
- Generated PDDL domain and problem files in `outputs/pddl/task_N/`
- Task analysis results
- Performance metrics

---

### [gemini_robotics_example.py](gemini_robotics_example.py) - Interactive Gemini Capabilities Demo

**Menu-driven demonstration** of all Gemini Robotics-ER capabilities.

```bash
python examples/gemini_robotics_example.py
```

**Interactive Menu:**
```
GEMINI ROBOTICS CAPABILITIES MENU
==================================================
Select a capability to test:

  1. Object Detection with Affordances
  2. Spatial Reasoning and Relationships
  3. Task Decomposition
  4. Interaction Point Detection (with visualization)
  5. Trajectory Planning
  6. Full Integration with World Model

  7. Run All Examples
  0. Exit
```

**Features:**
- **Menu-driven interface** - Select which capability to test
- **Reusable detection** - Run detection once, use for multiple demos
- **Interactive prompts** - Input tasks, select objects, choose actions
- **Visual feedback** - OpenCV visualization for interaction points
- **Save outputs** - Option to save annotated images
- Works with RealSense camera or custom test images

---

### [test_gemini_detection.py](test_gemini_detection.py) - Diagnostic Tool

Test and debug Gemini API integration.

```bash
python examples/test_gemini_detection.py
```

**What it does:**
- Tests API key configuration
- Tests detection with synthetic image
- Shows detailed error messages and debug info
- Optionally tests with real camera

**Use this when:**
- Setting up for the first time
- Troubleshooting API issues
- Debugging detection problems
- Verifying camera integration

---

### [simple_demo.py](simple_demo.py) - Basic Components Demo

Legacy demonstration of individual system components with mock data.

```bash
python examples/simple_demo.py
```

**What it demonstrates:**
- Camera capture
- Task parsing
- Object tracking (mock objects)
- Spatial relationships
- PDDL state generation

**Note:** Uses mock object data. For real VLM detection, use `dynamic_pddl_demo.py`.

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
   python examples/test_gemini_detection.py
   ```

### Running the Interactive Demo

```bash
python examples/dynamic_pddl_demo.py
```

Follow the prompts:
1. Camera captures scene (press any key after viewing)
2. VLM detects objects (takes 5-15 seconds)
3. System shows detected objects
4. You enter tasks based on what you see
5. System generates PDDL files for each task

### Without Camera

If you don't have a RealSense camera, you can still test with an image:

```bash
python examples/gemini_robotics_example.py
# When prompted, provide path to a test image
```

## Comparison

| Demo | Objects | Tasks | VLM | Interactive | Best For |
|------|---------|-------|-----|-------------|----------|
| **dynamic_pddl_demo.py** | Real VLM | User input | ✅ | ✅ | Full pipeline, real-world use |
| **gemini_robotics_example.py** | Real VLM | User input | ✅ | ✅ | Testing Gemini features |
| **test_gemini_detection.py** | Synthetic | None | ✅ | ✅ | Debugging, setup |
| **simple_demo.py** | Mock data | Pre-defined | ❌ | ❌ | Learning components |

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
- Verify API key: `python examples/test_gemini_detection.py`
- Ensure objects are visible in camera view
- Check lighting conditions

**Detection errors:**
- Run diagnostic: `python examples/test_gemini_detection.py`
- Check API key validity
- Verify model availability (`gemini-2.0-flash-exp`)
- Check internet connection

**PDDL generation fails:**
- Ensure objects were detected successfully
- Check task references detected objects
- Verify LLM API key is valid

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

## Next Steps

After running the demos:

1. **Experiment with different scenes** - Try various object arrangements
2. **Test complex tasks** - Multi-step manipulation tasks
3. **Integrate with planner** - Use generated PDDL with a planner
4. **Add custom objects** - Test with domain-specific objects
5. **Extend capabilities** - Add new affordances or actions

## Support

For issues or questions:
- Check [GEMINI_INTEGRATION.md](../docs/GEMINI_INTEGRATION.md)
- Review [BUGFIX_SUMMARY.md](../BUGFIX_SUMMARY.md)
- Run diagnostic script for specific errors
