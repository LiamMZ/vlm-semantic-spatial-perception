"""
Utility script to replay a cached plan with PrimitiveExecutor.

Examples:
  # Dry run (no robot connection)
  uv run python scripts/run_cached_plan.py \
      --plan tests/artifacts/pick_plan_prepare_plan.json \
      --world tests/assets/continuous_pick_fixture \
      --dry-run

  # Execute on the robot (default IP constant below)
  uv run python scripts/run_cached_plan.py \
      --plan tests/artifacts/pick_plan_prepare_plan.json \
      --world tests/assets/continuous_pick_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primitives import PrimitiveExecutor, SkillPlan  # noqa: E402

DEFAULT_ROBOT_IP = "192.168.1.224"


def _load_world_state(world_dir: Path) -> Dict[str, Any]:
    registry_path = world_dir / "registry.json"
    state_path = world_dir / "state.json"
    index_path = world_dir / "perception_pool" / "index.json"

    if not (registry_path.exists() and state_path.exists() and index_path.exists()):
        raise FileNotFoundError(
            "World directory must contain registry.json, state.json, "
            "and perception_pool/index.json"
        )

    registry = json.loads(registry_path.read_text())
    state = json.loads(state_path.read_text())
    snapshot_index = json.loads(index_path.read_text())

    world_state = {
        "registry": registry,
        "last_snapshot_id": state.get("last_snapshot_id"),
        "snapshot_index": snapshot_index,
        "robot_state": state.get("robot_state"),
    }
    return world_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Path to cached plan JSON")
    parser.add_argument(
        "--world",
        required=True,
        help="Path to world-state directory (registry/state/perception_pool)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip robot connection and only translate/validate the plan.",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"Robot IP for CuRoboMotionPlanner (default: {DEFAULT_ROBOT_IP}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path (defaults to stdout).",
    )
    args = parser.parse_args()

    plan_payload = json.loads(Path(args.plan).read_text())
    plan_dict = plan_payload.get("plan") or plan_payload
    plan = SkillPlan.from_dict(plan_dict)

    world_dir = Path(args.world)
    world_state = _load_world_state(world_dir)

    execute = not args.dry_run
    executor = PrimitiveExecutor(
        primitives=_build_primitives_interface(execute, args.robot_ip),
        perception_pool_dir=world_dir / "perception_pool",
    )
    result_payload = executor.execute_plan(
        plan,
        world_state,
        dry_run=not execute,
    )

    result = {
        "plan": plan.to_dict(),
        "warnings": result_payload.warnings,
        "errors": result_payload.errors,
        "executed": result_payload.executed,
    }
    if result_payload.primitive_results:
        result["primitive_results"] = result_payload.primitive_results

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Saved executor output to {args.output}")
    else:
        print(output_json)


def _build_primitives_interface(execute: bool, robot_ip: Optional[str]):
    if not execute:
        return None
    from src.kinematics.xarm_curobo_interface import CuRoboMotionPlanner

    print(
        f"⚠️  EXECUTION ENABLED: connecting to {robot_ip or DEFAULT_ROBOT_IP} "
        "via CuRoboMotionPlanner."
    )
    return CuRoboMotionPlanner(robot_ip=robot_ip)


if __name__ == "__main__":
    main()
