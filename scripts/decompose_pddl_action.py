"""
Minimal helper to LLM-decompose a PDDL action for a specific object using SkillDecomposer.

Examples:
  uv run python scripts/decompose_pddl_action.py \\
      --world tests/assets/continuous_pick_fixture \\
      --action pick --object-id black_fabric_garment

  uv run python scripts/decompose_pddl_action.py \\
      --world tests/assets/20251201_161659 \\
      --action pick --object-id brown_paper_bag \\
      --output outputs/llm_decompositions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primitives import PrimitiveExecutor, SkillDecomposer  # noqa: E402
from scripts.affordance_viz import visualize_primitives  # noqa: E402

DEFAULT_WORLD = "outputs/demos/20251201_210900"
DEFAULT_ACTION = "pick"
DEFAULT_OBJECT_ID = "black_cloth"
DEFAULT_INTERACTION = "graspable"
DEFAULT_OUTPUT_DIR = "outputs/llm_decompositions"


def _load_world_state(world_dir: Path) -> Dict[str, Any]:
    """Load registry/state/index from a world directory."""
    registry_path = world_dir / "registry.json"
    state_path = world_dir / "state.json"
    index_path = world_dir / "perception_pool" / "index.json"
    missing = [p for p in (registry_path, state_path, index_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"World dir must contain registry.json, state.json, and perception_pool/index.json (missing: {missing})"
        )

    registry = json.loads(registry_path.read_text())
    state = json.loads(state_path.read_text())
    index = json.loads(index_path.read_text())
    return {
        "registry": registry,
        "last_snapshot_id": state.get("last_snapshot_id"),
        "snapshot_index": index,
        "robot_state": state.get("robot_state"),
    }


def _recording_llm_call(output_path: Path, original_call):
    """Wrap SkillDecomposer._call_llm to persist the raw response."""

    def _wrapper(self, *args, **kwargs):
        response_text = original_call(self, *args, **kwargs)
        output_path.write_text(response_text)
        return response_text

    return _wrapper


def _get_api_key(explicit: Optional[str]) -> str:
    key = (
        explicit
        or os.environ.get("GENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not key:
        raise SystemExit(
            "Set GENAI_API_KEY/GOOGLE_API_KEY/GEMINI_API_KEY or pass --api-key to call the LLM."
        )
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        default=DEFAULT_WORLD,
        help=f"World directory with registry/state/perception_pool (default: {DEFAULT_WORLD}).",
    )
    parser.add_argument(
        "--action",
        default=DEFAULT_ACTION,
        help=f"PDDL action name to decompose (default: {DEFAULT_ACTION}).",
    )
    parser.add_argument(
        "--object-id",
        default=DEFAULT_OBJECT_ID,
        help=f"Target object_id for the action (default: {DEFAULT_OBJECT_ID}).",
    )
    parser.add_argument(
        "--interaction",
        default=DEFAULT_INTERACTION,
        help=f"Interaction label passed to the decomposer (default: {DEFAULT_INTERACTION}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional GenAI API key (falls back to GENAI_API_KEY/GOOGLE_API_KEY/GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1).",
    )
    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)
    world_dir = Path(args.world).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    world_state = _load_world_state(world_dir)

    base_name = f"{args.action}_{args.object_id}".replace(" ", "_")
    llm_response_path = output_dir / f"{base_name}_llm_response.json"
    translated_path = output_dir / f"{base_name}_plan_translated.json"

    decomposer = SkillDecomposer(api_key=api_key)
    decomposer._perception_pool_dir = world_dir / "perception_pool"  # align with provided world

    params = {"object_id": args.object_id, "interaction": args.interaction}
    original_call = SkillDecomposer._call_llm
    with patch.object(SkillDecomposer, "_call_llm", _recording_llm_call(llm_response_path, original_call)):
        plan = decomposer.plan(
            action_name=args.action,
            parameters=params,
            world_hint=world_state,
            temperature=args.temperature,
        )

    executor = PrimitiveExecutor(
        primitives=None,
        perception_pool_dir=world_dir / "perception_pool",
    )
    executor.prepare_plan(plan, world_state)

    translated_path.write_text(
        json.dumps(
            {
                "plan": plan.to_dict(),
            },
            indent=2,
        )
    )

    print(f"Wrote LLM response to {llm_response_path}")
    print(f"Wrote translated plan to {translated_path}")

    visualize_primitives(
        plan=plan,
        world_dir=world_dir,
        output_dir=output_dir,
        snapshot_id=getattr(plan, "source_snapshot_id", None),
    )


if __name__ == "__main__":
    main()
