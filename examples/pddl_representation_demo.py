"""
PDDL Representation Demo

Demonstrates the modular PDDL representation system with:
1. Incremental domain construction
2. Problem instance updates
3. Goal refinement
4. Validation and feedback
5. File generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.planning import PDDLRepresentation, PDDLRequirements


def demo_basic_usage():
    """Demo 1: Basic usage with manual construction."""
    print("=" * 70)
    print("DEMO 1: Basic Manual Construction")
    print("=" * 70)
    print()

    # Create representation
    pddl = PDDLRepresentation(domain_name="kitchen_manipulation")

    # Add custom types
    print("Adding object types...")
    pddl.add_object_type("container", parent="object")
    pddl.add_object_type("cup", parent="container")
    pddl.add_object_type("bottle", parent="container")
    pddl.add_object_type("surface", parent="object")
    pddl.add_object_type("table", parent="surface")

    # Add custom predicates
    print("Adding custom predicates...")
    pddl.add_predicate("clean", [("obj", "object")])
    pddl.add_predicate("dirty", [("obj", "object")])

    # Object instances
    print("Adding object instances...")
    pddl.add_object_instance("red_cup", "cup")
    pddl.add_object_instance("blue_bottle", "bottle")
    pddl.add_object_instance("kitchen_table", "table")
    pddl.add_object_instance("robot", "object")

    # Initial state
    print("Setting initial state...")
    pddl.add_initial_literal("on", ["red_cup", "kitchen_table"])
    pddl.add_initial_literal("on", ["blue_bottle", "kitchen_table"])
    pddl.add_initial_literal("empty-hand", [])
    pddl.add_initial_literal("clean", ["red_cup"])
    pddl.add_initial_literal("dirty", ["blue_bottle"])

    # Goal state
    print("Setting goal state...")
    pddl.add_goal_literal("holding", ["red_cup"])
    pddl.add_goal_literal("clean", ["blue_bottle"])

    # Validate
    print("\nValidating representation...")
    goal_valid, goal_issues = pddl.validate_goal_completeness()
    action_valid, action_issues = pddl.validate_action_completeness()

    print(f"  Goal completeness: {'✓' if goal_valid else '✗'}")
    if goal_issues:
        for issue in goal_issues:
            print(f"    - {issue}")

    print(f"  Action completeness: {'✓' if action_valid else '✗'}")
    if action_issues:
        for issue in action_issues:
            print(f"    - {issue}")

    # Statistics
    print("\nStatistics:")
    stats = pddl.get_statistics()
    print(f"  Domain: {stats['domain']['name']}")
    print(f"    Types: {stats['domain']['types']}")
    print(f"    Predicates: {stats['domain']['predicates']}")
    print(f"    Predefined actions: {stats['domain']['predefined_actions']}")
    print(f"    LLM-generated actions: {stats['domain']['llm_generated_actions']}")
    print(f"  Problem: {stats['problem']['name']}")
    print(f"    Objects: {stats['problem']['object_instances']}")
    print(f"    Initial literals: {stats['problem']['initial_literals']}")
    print(f"    Goal literals: {stats['problem']['goal_literals']}")

    # Generate files
    print("\nGenerating PDDL files...")
    paths = pddl.generate_files("outputs/pddl/demo1")
    print(f"  Domain: {paths['domain_path']}")
    print(f"  Problem: {paths['problem_path']}")

    return pddl


def demo_world_state_integration():
    """Demo 2: Integration with world state updates."""
    print("\n" + "=" * 70)
    print("DEMO 2: World State Integration")
    print("=" * 70)
    print()

    # Create representation
    pddl = PDDLRepresentation(domain_name="table_manipulation")

    # Simulate world state from perception system
    world_state = {
        "objects": [
            {
                "object_id": "red_cup_1",
                "object_type": "cup",
                "affordances": ["graspable", "containable", "pourable"]
            },
            {
                "object_id": "blue_mug_1",
                "object_type": "mug",
                "affordances": ["graspable", "containable"]
            },
            {
                "object_id": "wooden_table_1",
                "object_type": "table",
                "affordances": ["supportable"]
            }
        ],
        "predicates": [
            "on(red_cup_1, wooden_table_1)",
            "on(blue_mug_1, wooden_table_1)",
            "clean(red_cup_1)",
            "filled(blue_mug_1)"
        ]
    }

    print("Updating from world state...")
    pddl.update_from_world_state(world_state)

    # Add additional state
    pddl.add_initial_literal("empty-hand", [])

    # Set task-specific goal
    print("Setting task goal: Move red cup and pour into blue mug")
    pddl.task_description = "Move the red cup and pour its contents into the blue mug"
    pddl.add_goal_literal("holding", ["red_cup_1"])
    pddl.add_goal_literal("empty", ["red_cup_1"])
    pddl.add_goal_literal("filled", ["blue_mug_1"])

    # Validate
    print("\nValidating...")
    goal_valid, goal_issues = pddl.validate_goal_completeness()
    print(f"  Goal valid: {goal_valid}")
    if goal_issues:
        for issue in goal_issues:
            print(f"    - {issue}")

    # Generate files
    print("\nGenerating files...")
    paths = pddl.generate_files("outputs/pddl/demo2")
    print(f"  Generated: {paths['domain_path']}")
    print(f"  Generated: {paths['problem_path']}")

    return pddl


def demo_llm_generated_actions():
    """Demo 3: Adding LLM-generated actions."""
    print("\n" + "=" * 70)
    print("DEMO 3: LLM-Generated Actions")
    print("=" * 70)
    print()

    # Create representation
    pddl = PDDLRepresentation(domain_name="advanced_manipulation")

    # Add custom types
    pddl.add_object_type("drawer", parent="object")
    pddl.add_object_type("item", parent="object")

    # Add object instances
    pddl.add_object_instance("kitchen_drawer", "drawer")
    pddl.add_object_instance("spoon", "item")

    # Add LLM-generated custom action
    print("Adding LLM-generated action: slide-open-drawer")
    pddl.add_llm_generated_action(
        name="slide-open-drawer",
        parameters=[("d", "drawer")],
        precondition="(and (slideable ?d) (not (opened ?d)) (empty-hand))",
        effect="(and (opened ?d) (accessible-inside ?d))",
        description="Slide open a drawer with two hands"
    )

    # Add corresponding predicates
    pddl.add_predicate("slideable", [("obj", "object")])
    pddl.add_predicate("accessible-inside", [("obj", "object")])

    # Another LLM-generated action
    print("Adding LLM-generated action: retrieve-from-drawer")
    pddl.add_llm_generated_action(
        name="retrieve-from-drawer",
        parameters=[("item", "item"), ("drawer", "drawer")],
        precondition="(and (in ?item ?drawer) (opened ?drawer) (empty-hand))",
        effect="(and (holding ?item) (not (in ?item ?drawer)) (not (empty-hand)))",
        description="Retrieve an item from an open drawer"
    )

    pddl.add_predicate("in", [("obj1", "object"), ("obj2", "object")])

    # Set initial state
    pddl.add_initial_literal("slideable", ["kitchen_drawer"])
    pddl.add_initial_literal("in", ["spoon", "kitchen_drawer"])
    pddl.add_initial_literal("empty-hand", [])

    # Set goal
    pddl.add_goal_literal("holding", ["spoon"])

    # Statistics
    stats = pddl.get_statistics()
    print(f"\nActions breakdown:")
    print(f"  Predefined: {stats['domain']['predefined_actions']}")
    print(f"  LLM-generated: {stats['domain']['llm_generated_actions']}")
    print(f"  Total: {stats['domain']['total_actions']}")

    # Generate
    paths = pddl.generate_files("outputs/pddl/demo3")
    print(f"\nGenerated files:")
    print(f"  {paths['domain_path']}")
    print(f"  {paths['problem_path']}")

    return pddl


def demo_goal_refinement():
    """Demo 4: Iterative goal refinement based on feedback."""
    print("\n" + "=" * 70)
    print("DEMO 4: Goal Refinement")
    print("=" * 70)
    print()

    pddl = PDDLRepresentation(domain_name="goal_refinement_test")

    # Add objects
    pddl.add_object_instance("cup", "object")
    pddl.add_object_instance("table", "object")

    # Initial attempt: incomplete goal
    print("Attempt 1: Setting incomplete goal...")
    pddl.add_goal_literal("holding", ["cup"])

    valid, issues = pddl.validate_goal_completeness()
    print(f"  Valid: {valid}")
    for issue in issues:
        print(f"  Issue: {issue}")

    # Refinement: Add missing initial state
    print("\nRefinement 1: Adding initial state...")
    pddl.add_initial_literal("on", ["cup", "table"])
    pddl.add_initial_literal("empty-hand", [])

    valid, issues = pddl.validate_goal_completeness()
    print(f"  Valid: {valid}")
    for issue in issues:
        print(f"  Issue: {issue}")

    # Check action completeness
    print("\nChecking action completeness...")
    valid, issues = pddl.validate_action_completeness()
    print(f"  Valid: {valid}")
    for issue in issues:
        print(f"  Issue: {issue}")

    # Update goal with additional constraints
    print("\nRefinement 2: Adding additional goal constraints...")
    pddl.add_goal_literal("on", ["cup", "table"], negated=True)  # Cup should NOT be on table

    # Final validation
    print("\nFinal validation:")
    goal_valid, goal_issues = pddl.validate_goal_completeness()
    action_valid, action_issues = pddl.validate_action_completeness()

    print(f"  Goal: {'✓' if goal_valid else '✗'}")
    print(f"  Actions: {'✓' if action_valid else '✗'}")

    paths = pddl.generate_files("outputs/pddl/demo4")
    print(f"\nGenerated: {paths['domain_path']}")

    return pddl


def demo_incremental_updates():
    """Demo 5: Incremental state updates (simulating continuous perception)."""
    print("\n" + "=" * 70)
    print("DEMO 5: Incremental State Updates")
    print("=" * 70)
    print()

    pddl = PDDLRepresentation(domain_name="dynamic_world")

    # Frame 1: Initial observation
    print("Frame 1: Initial observation")
    world_state_1 = {
        "objects": [
            {"object_id": "cup1", "object_type": "cup"},
            {"object_id": "table1", "object_type": "table"}
        ],
        "predicates": ["on(cup1, table1)", "empty-hand()"]
    }
    pddl.update_from_world_state(world_state_1)
    print(f"  Objects: {len(pddl.object_instances)}")
    print(f"  Literals: {len(pddl.initial_literals)}")

    # Frame 2: New object detected
    print("\nFrame 2: New object detected")
    world_state_2 = {
        "objects": [
            {"object_id": "cup1", "object_type": "cup"},
            {"object_id": "cup2", "object_type": "cup"},  # New!
            {"object_id": "table1", "object_type": "table"}
        ],
        "predicates": [
            "on(cup1, table1)",
            "on(cup2, table1)",  # New!
            "empty-hand()"
        ]
    }
    pddl.clear_initial_state()  # Clear old state
    pddl.update_from_world_state(world_state_2)
    print(f"  Objects: {len(pddl.object_instances)}")
    print(f"  Literals: {len(pddl.initial_literals)}")

    # Frame 3: State change (robot picks up cup1)
    print("\nFrame 3: State change (robot action)")
    pddl.update_initial_state([
        ("on", ["cup2", "table1"], False),
        ("holding", ["cup1"], False),
        ("empty-hand", [], True)  # Negated
    ])
    print(f"  Updated initial state")
    print(f"  Literals: {len(pddl.initial_literals)}")

    # Set goal and generate
    pddl.add_goal_literal("holding", ["cup2"])
    paths = pddl.generate_files("outputs/pddl/demo5")
    print(f"\nGenerated: {paths['problem_path']}")

    return pddl


def demo_end_to_end_task_generation():
    """
    Demo 6: End-to-end task-to-PDDL generation.

    Simulates the complete workflow:
    1. User provides task description
    2. System observes environment (simulated)
    3. LLM analyzes task and proposes domain components
    4. System builds PDDL representation
    5. Validates and generates files
    """
    print("\n" + "=" * 70)
    print("DEMO 6: End-to-End Task-to-PDDL Generation")
    print("=" * 70)
    print()

    # =====================================================================
    # Step 1: Task Input
    # =====================================================================
    task_description = "Clean the coffee mug and place it in the dishwasher"
    print(f"Task: {task_description}")
    print()

    # =====================================================================
    # Step 2: Simulated World Observation
    # =====================================================================
    print("Step 1: Observing environment...")

    # Simulate what the perception system would provide
    observed_world_state = {
        "objects": [
            {
                "object_id": "coffee_mug_1",
                "object_type": "mug",
                "affordances": ["graspable", "containable", "washable"],
                "properties": {
                    "color": "white",
                    "state": "dirty",
                    "material": "ceramic"
                }
            },
            {
                "object_id": "kitchen_sink_1",
                "object_type": "sink",
                "affordances": ["containable", "water_source"],
                "properties": {
                    "state": "empty",
                    "has_faucet": True
                }
            },
            {
                "object_id": "dishwasher_1",
                "object_type": "dishwasher",
                "affordances": ["openable", "containable", "supportable"],
                "properties": {
                    "state": "closed",
                    "is_full": False
                }
            },
            {
                "object_id": "countertop_1",
                "object_type": "countertop",
                "affordances": ["supportable"],
                "properties": {
                    "material": "granite"
                }
            },
            {
                "object_id": "dish_soap_1",
                "object_type": "soap_dispenser",
                "affordances": ["graspable", "pourable"],
                "properties": {
                    "contents": "soap"
                }
            }
        ],
        "predicates": [
            "on(coffee_mug_1, countertop_1)",
            "on(dish_soap_1, countertop_1)",
            "near(kitchen_sink_1, countertop_1)",
            "near(dishwasher_1, countertop_1)"
        ]
    }

    print(f"  Detected {len(observed_world_state['objects'])} objects")
    for obj in observed_world_state['objects']:
        print(f"    - {obj['object_id']} ({obj['object_type']})")
    print()

    # =====================================================================
    # Step 3: LLM Task Analysis (Simulated)
    # =====================================================================
    print("Step 2: Analyzing task with LLM...")

    # Simulate what LLM would propose based on task and observations
    llm_analysis = {
        "required_object_types": [
            ("mug", "container"),
            ("sink", "object"),
            ("dishwasher", "appliance"),
            ("countertop", "surface"),
            ("soap_dispenser", "object"),
            ("container", "object"),
            ("appliance", "object"),
            ("surface", "object")
        ],
        "required_predicates": [
            ("clean", [("obj", "object")]),
            ("dirty", [("obj", "object")]),
            ("inside", [("obj1", "object"), ("obj2", "object")]),
            ("wet", [("obj", "object")]),
            ("has_soap", [("obj", "object")])
        ],
        "llm_generated_actions": [
            {
                "name": "wash",
                "parameters": [("obj", "object"), ("sink", "sink"), ("soap", "object")],
                "precondition": "(and (holding ?obj) (dirty ?obj) (near robot ?sink) (graspable ?soap))",
                "effect": "(and (clean ?obj) (wet ?obj) (not (dirty ?obj)))",
                "description": "Wash a dirty object in the sink with soap"
            },
            {
                "name": "dry",
                "parameters": [("obj", "object")],
                "precondition": "(and (holding ?obj) (wet ?obj))",
                "effect": "(not (wet ?obj))",
                "description": "Dry a wet object"
            },
            {
                "name": "place-inside",
                "parameters": [("obj", "object"), ("container", "object")],
                "precondition": "(and (holding ?obj) (opened ?container) (containable ?container))",
                "effect": "(and (inside ?obj ?container) (not (holding ?obj)) (empty-hand))",
                "description": "Place object inside an open container"
            }
        ],
        "goal_literals": [
            ("clean", ["coffee_mug_1"], False),
            ("inside", ["coffee_mug_1", "dishwasher_1"], False),
            ("dirty", ["coffee_mug_1"], True)  # Negated: mug should NOT be dirty
        ],
        "action_sequence": [
            "pick(coffee_mug_1)",
            "wash(coffee_mug_1, kitchen_sink_1, dish_soap_1)",
            "dry(coffee_mug_1)",
            "open(dishwasher_1)",
            "place-inside(coffee_mug_1, dishwasher_1)"
        ]
    }

    print(f"  Identified {len(llm_analysis['required_object_types'])} object types")
    print(f"  Proposed {len(llm_analysis['required_predicates'])} custom predicates")
    print(f"  Generated {len(llm_analysis['llm_generated_actions'])} task-specific actions")
    print(f"  Estimated {len(llm_analysis['action_sequence'])} steps")
    print()

    # =====================================================================
    # Step 4: Build PDDL Representation
    # =====================================================================
    print("Step 3: Building PDDL representation...")

    pddl = PDDLRepresentation(
        domain_name="kitchen_cleaning",
        problem_name="clean_mug_task"
    )
    pddl.task_description = task_description

    # Add object types from LLM analysis
    # Sort by hierarchy: add parents before children
    print("  Adding object types...")
    type_hierarchy = llm_analysis["required_object_types"]

    # Add in order: first add types with parent "object", then others
    added = set(["object"])  # Base type already exists
    max_iterations = len(type_hierarchy) + 1
    iteration = 0

    while len(added) < len(type_hierarchy) + 1 and iteration < max_iterations:
        iteration += 1
        for type_name, parent in type_hierarchy:
            if type_name not in added and parent in added:
                if type_name not in pddl.object_types:
                    pddl.add_object_type(type_name, parent=parent if parent != "object" else None)
                added.add(type_name)

    # Add custom predicates from LLM analysis
    print("  Adding custom predicates...")
    for pred_name, params in llm_analysis["required_predicates"]:
        if pred_name not in pddl.predicates:
            pddl.add_predicate(pred_name, params)

    # Add LLM-generated actions
    print("  Adding LLM-generated actions...")
    for action in llm_analysis["llm_generated_actions"]:
        pddl.add_llm_generated_action(
            name=action["name"],
            parameters=action["parameters"],
            precondition=action["precondition"],
            effect=action["effect"],
            description=action["description"]
        )

    # Update from observed world state
    print("  Integrating world state...")
    pddl.update_from_world_state(observed_world_state)

    # Add additional initial state from observations
    pddl.add_initial_literal("dirty", ["coffee_mug_1"])
    pddl.add_initial_literal("empty-hand", [])
    pddl.add_initial_literal("near", ["robot", "countertop_1"])

    # Add goal from LLM analysis
    print("  Setting goal state...")
    for pred_name, args, negated in llm_analysis["goal_literals"]:
        pddl.add_goal_literal(pred_name, args, negated=negated)

    print()

    # =====================================================================
    # Step 5: Validation
    # =====================================================================
    print("Step 4: Validating representation...")

    goal_valid, goal_issues = pddl.validate_goal_completeness()
    action_valid, action_issues = pddl.validate_action_completeness()

    print(f"  Goal completeness: {'✓ Valid' if goal_valid else '✗ Issues found'}")
    if goal_issues:
        for issue in goal_issues:
            print(f"    ⚠ {issue}")

    print(f"  Action completeness: {'✓ Valid' if action_valid else '✗ Issues found'}")
    if action_issues:
        for issue in action_issues:
            print(f"    ⚠ {issue}")

    print()

    # =====================================================================
    # Step 6: Statistics
    # =====================================================================
    print("Step 5: Representation statistics...")
    stats = pddl.get_statistics()

    print(f"  Domain components:")
    print(f"    Object types: {stats['domain']['types']}")
    print(f"    Predicates: {stats['domain']['predicates']}")
    print(f"    Predefined actions: {stats['domain']['predefined_actions']}")
    print(f"    LLM-generated actions: {stats['domain']['llm_generated_actions']}")
    print(f"    Total actions: {stats['domain']['total_actions']}")

    print(f"  Problem components:")
    print(f"    Object instances: {stats['problem']['object_instances']}")
    print(f"    Initial literals: {stats['problem']['initial_literals']}")
    print(f"    Goal literals: {stats['problem']['goal_literals']}")
    print()

    # =====================================================================
    # Step 7: Generate PDDL Files
    # =====================================================================
    print("Step 6: Generating PDDL files...")
    paths = pddl.generate_files("outputs/pddl/demo6_end_to_end")

    print(f"  ✓ Domain: {paths['domain_path']}")
    print(f"  ✓ Problem: {paths['problem_path']}")
    print()

    # =====================================================================
    # Step 8: Show Expected Action Sequence
    # =====================================================================
    print("Expected action sequence (from LLM analysis):")
    for i, action in enumerate(llm_analysis["action_sequence"], 1):
        print(f"  {i}. {action}")
    print()

    print("=" * 70)
    print("Task-to-PDDL generation complete!")
    print("=" * 70)
    print()
    print("The generated PDDL files can now be used with any PDDL planner")
    print("(e.g., FastDownward, LAMA, FF) to find an optimal action sequence.")
    print()

    return pddl


def demo_interactive_task_input():
    """
    Demo 7: Interactive task input from user.

    Allows user to input their own task and see the generation process.
    """
    print("\n" + "=" * 70)
    print("DEMO 7: Interactive Task Input")
    print("=" * 70)
    print()

    print("This demo allows you to input a custom task and see the PDDL generation.")
    print("(Using simplified world state for demonstration)")
    print()

    # Get user input
    task = input("Enter your task description (or press Enter for default): ").strip()

    if not task:
        task = "Pick up the red cup and place it on the shelf"
        print(f"Using default task: {task}")

    print()
    print(f"Generating PDDL for task: '{task}'")
    print()

    # Create representation
    pddl = PDDLRepresentation(
        domain_name="custom_task",
        problem_name="user_task"
    )
    pddl.task_description = task

    # Simplified world state
    print("Using simplified world state:")
    world_state = {
        "objects": [
            {"object_id": "red_cup", "object_type": "cup"},
            {"object_id": "shelf", "object_type": "shelf"},
            {"object_id": "table", "object_type": "table"}
        ],
        "predicates": [
            "on(red_cup, table)",
            "supportable(shelf)"
        ]
    }

    for obj in world_state["objects"]:
        print(f"  - {obj['object_id']} ({obj['object_type']})")
    print()

    # Build representation
    pddl.update_from_world_state(world_state)
    pddl.add_initial_literal("empty-hand", [])

    # Simple goal extraction (just for demo - real system would use LLM)
    print("Extracting goal from task...")
    if "pick" in task.lower() and "place" in task.lower():
        # Extract object names (simplified)
        words = task.lower().split()
        if "cup" in words:
            pddl.add_goal_literal("holding", ["red_cup"])
        if "shelf" in words:
            pddl.add_goal_literal("on", ["red_cup", "shelf"])

    print("  Added goals based on task keywords")
    print()

    # Validate
    goal_valid, goal_issues = pddl.validate_goal_completeness()
    action_valid, action_issues = pddl.validate_action_completeness()

    print(f"Validation: Goals={'✓' if goal_valid else '✗'}, Actions={'✓' if action_valid else '✗'}")
    print()

    # Generate
    paths = pddl.generate_files("outputs/pddl/demo7_interactive")
    print(f"Generated PDDL files:")
    print(f"  {paths['domain_path']}")
    print(f"  {paths['problem_path']}")
    print()

    return pddl


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("PDDL REPRESENTATION SYSTEM DEMO")
    print("*" * 70)
    print()

    # Check if user wants interactive demo
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        demo_interactive_task_input()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--end-to-end":
        demo_end_to_end_task_generation()
        return

    # Run all demos
    demo_basic_usage()
    demo_world_state_integration()
    demo_llm_generated_actions()
    demo_goal_refinement()
    demo_incremental_updates()

    # Run the end-to-end demo
    demo_end_to_end_task_generation()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nCheck outputs/pddl/demo* for generated PDDL files")
    print("\nTo run specific demos:")
    print("  python examples/pddl_representation_demo.py --end-to-end")
    print("  python examples/pddl_representation_demo.py --interactive")


if __name__ == "__main__":
    main()
