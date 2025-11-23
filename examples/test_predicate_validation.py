"""
Test predicate validation in PDDL domain maintainer.

This script demonstrates the automatic predicate validation and addition
when actions reference predicates that aren't explicitly defined.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src.planning.pddl_representation import PDDLRepresentation


async def test_predicate_validation():
    """Test that missing predicates are automatically detected and added."""

    print("=" * 80)
    print("Testing Predicate Validation in PDDL Domain")
    print("=" * 80)
    print()

    # Create PDDL representation
    pddl = PDDLRepresentation(domain_name="predicate_test")

    # Add some types
    print("📋 Setting up PDDL domain...")
    pddl.add_object_type("cup", parent="object")
    pddl.add_object_type("water_source", parent="object")
    print("   ✓ Added types: cup, water_source")
    print()

    # Add some predicates explicitly (but not all that we'll use in actions)
    await pddl.add_predicate_async("graspable", [("obj", "object")])
    await pddl.add_predicate_async("empty-hand", [])
    print("📋 Explicitly defined predicates:")
    print("   - graspable")
    print("   - empty-hand")
    print()

    # Add actions that reference predicates NOT in the explicit list
    print("📋 Adding actions with additional predicates...")
    print()

    # Action 1: Uses 'holding' predicate (not defined)
    pddl.add_llm_generated_action(
        name="pick_up",
        parameters=[("obj", "cup")],
        precondition="(and (graspable ?obj) (empty-hand))",
        effect="(holding ?obj)",  # 'holding' not defined!
        description="Pick up an object"
    )
    print("   ✓ Added action: pick_up")
    print("     Uses predicate: holding (NOT DEFINED YET)")
    print()

    # Action 2: Uses 'is-empty' and 'filled' predicates (not defined)
    pddl.add_llm_generated_action(
        name="fill_cup",
        parameters=[("cup", "cup"), ("source", "water_source")],
        precondition="(and (holding ?cup) (is-empty ?cup))",  # 'is-empty' not defined!
        effect="(filled ?cup)",  # 'filled' not defined!
        description="Fill cup with water"
    )
    print("   ✓ Added action: fill_cup")
    print("     Uses predicates: is-empty, filled (NOT DEFINED YET)")
    print()

    # Action 3: Uses complex formula with multiple undefined predicates
    pddl.add_llm_generated_action(
        name="pour_water",
        parameters=[("source_cup", "cup"), ("target_cup", "cup")],
        precondition="(and (holding ?source_cup) (filled ?source_cup) (is-empty ?target_cup))",
        effect="(and (not (filled ?source_cup)) (filled ?target_cup) (water-level-low ?source_cup))",
        description="Pour water from one cup to another"
    )
    print("   ✓ Added action: pour_water")
    print("     Uses predicates: water-level-low (NOT DEFINED YET)")
    print()

    # Show initial predicates
    print("📊 Predicates BEFORE validation:")
    for pred_name in sorted(pddl.predicates.keys()):
        predicate = pddl.predicates[pred_name]
        if predicate.parameters:
            param_str = " ".join([f"?{name} - {type_}" for name, type_ in predicate.parameters])
            print(f"   ({pred_name} {param_str})")
        else:
            print(f"   ({pred_name})")
    print()

    # Import the maintainer to use its validation method
    from src.planning.pddl_domain_maintainer import PDDLDomainMaintainer

    # Create maintainer with our PDDL representation
    maintainer = PDDLDomainMaintainer(pddl_representation=pddl)

    # Run validation
    print("🔍 Running predicate validation...")
    print()
    validation_result = await maintainer.validate_and_fix_action_predicates()

    # Show results
    print("📊 Validation Results:")
    print()
    if validation_result["missing_predicates"]:
        print(f"   ✓ Found and auto-added {len(validation_result['missing_predicates'])} missing predicates:")
        for pred in validation_result["missing_predicates"]:
            print(f"      - {pred}")
    else:
        print("   ℹ No missing predicates found")

    if validation_result["invalid_actions"]:
        print(f"   ⚠ Found {len(validation_result['invalid_actions'])} actions with parsing issues:")
        for action in validation_result["invalid_actions"]:
            print(f"      - {action}")
    else:
        print("   ✓ All actions are valid")
    print()

    # Show final predicates
    print("📊 Predicates AFTER validation:")
    for pred_name in sorted(pddl.predicates.keys()):
        predicate = pddl.predicates[pred_name]
        if predicate.parameters:
            param_str = " ".join([f"?{name} - {type_}" for name, type_ in predicate.parameters])
            print(f"   ({pred_name} {param_str})")
        else:
            print(f"   ({pred_name})")
    print()

    # Show final domain
    print("📄 Final PDDL Domain Preview:")
    print("-" * 80)
    domain_str = await pddl.generate_domain_pddl_async()
    lines = domain_str.split('\n')
    for i, line in enumerate(lines[:60], 1):
        print(f"{i:3d}: {line}")
    if len(lines) > 60:
        print(f"... ({len(lines) - 60} more lines)")
    print("-" * 80)
    print()

    # Save domain file
    output_dir = Path("outputs/test_predicates")
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_file = output_dir / "domain.pddl"

    with open(domain_file, 'w') as f:
        f.write(domain_str)

    print(f"💾 Domain saved to: {domain_file}")
    print()

    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Started with 2 explicit predicates")
    print(f"  - Added 3 actions referencing additional predicates")
    print(f"  - Auto-detected {len(validation_result['missing_predicates'])} missing predicates")
    print(f"  - Final domain has {len(pddl.predicates)} predicates")
    print()


if __name__ == "__main__":
    asyncio.run(test_predicate_validation())
