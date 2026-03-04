"""
Benchmark full PDDL generation + planning/refinement from saved world bundles.

Single-world mode:
  uv run scripts/benchmark_saved_world.py --world-dir outputs/captured_worlds/blocks

Batch mode over outputs/captured_worlds/*:
  uv run scripts/benchmark_saved_world.py --captured-root outputs/captured_worlds --run-all

Behavioral guarantees:
- Does NOT start live camera detection.
- Does NOT execute robot primitives.
- Runs with isolated benchmark output dirs, not inside source world dirs.
- Verifies source world dirs were not modified (optional, enabled by default).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.orchestrator_config import OrchestratorConfig
from src.planning.pddl_solver import PDDLSolver, SearchAlgorithm, SolverBackend
from src.planning.task_orchestrator import TaskOrchestrator


class NoopCamera:
    """Placeholder camera for orchestrator init when replaying saved world state."""


class NullRobot:
    """Stub robot provider so no robot connection is attempted."""

    def get_robot_state(self):
        return None


def _require_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return key


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", default=None, help="Single world bundle directory")
    parser.add_argument(
        "--captured-root",
        default="outputs/captured_worlds",
        help="Root containing named world bundles for --run-all",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run benchmark on each subdir under --captured-root that contains TASK.md",
    )
    parser.add_argument("--task", default=None, help="Override task text (single-world mode only)")
    parser.add_argument(
        "--solver-backend",
        default="pyperplan",
        help="Solver backend: auto|pyperplan|fast-downward-docker|fast-downward-apptainer",
    )
    parser.add_argument(
        "--solver-timeout",
        type=float,
        default=120.0,
        help="Planner timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--use-existing-pddl",
        action="store_true",
        help="Skip LLM/orchestrator replay and solve existing world_dir/pddl files directly.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default="outputs/benchmarks/saved_world_replays",
        help="Root for replay outputs and summary files",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit JSON summary path. Defaults under --benchmark-output-root.",
    )
    parser.add_argument(
        "--enforce-source-immutable",
        action="store_true",
        default=True,
        help="Fail run if source world directory changed during replay (default: enabled)",
    )
    parser.add_argument(
        "--no-enforce-source-immutable",
        action="store_false",
        dest="enforce_source_immutable",
        help="Disable source immutability verification.",
    )
    return parser.parse_args()


def _load_task_from_world(world_dir: Path, task_override: str | None = None) -> str:
    if task_override:
        return task_override

    task_md = world_dir / "TASK.md"
    if task_md.exists():
        text = task_md.read_text(encoding="utf-8").strip()
        if text:
            return text

    state_path = world_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json in {world_dir}")
    state = json.loads(state_path.read_text())
    task = state.get("current_task")
    if not task:
        raise ValueError(f"No task found in {task_md} or state.json for {world_dir}")
    return task


def _validate_world_bundle(world_dir: Path) -> Dict[str, Any]:
    registry_path = world_dir / "registry.json"
    pool_index_path = world_dir / "perception_pool" / "index.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry.json in {world_dir}")
    if not pool_index_path.exists():
        raise FileNotFoundError(f"Missing perception_pool/index.json in {world_dir}")

    registry = json.loads(registry_path.read_text())
    num_objects = int(registry.get("num_objects", 0))
    if num_objects <= 0:
        raise RuntimeError(f"Registry empty in {world_dir} (num_objects={num_objects})")

    pool_index = json.loads(pool_index_path.read_text())
    num_snapshots = len(pool_index.get("snapshots", {}))
    if num_snapshots <= 0:
        raise RuntimeError(f"Perception pool empty in {world_dir}")

    return {
        "registry_path": registry_path,
        "pool_index_path": pool_index_path,
        "num_objects": num_objects,
        "num_snapshots": num_snapshots,
    }


def _fingerprint_dir(root: Path) -> Dict[str, tuple]:
    """Return path->(size, mtime_ns) fingerprint for files under root."""
    fp: Dict[str, tuple] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        fp[rel] = (st.st_size, st.st_mtime_ns)
    return fp


async def _run_single_world(
    world_dir: Path,
    task: str,
    solver_backend: str,
    solver_timeout: float,
    benchmark_output_root: Path,
    enforce_source_immutable: bool,
) -> Dict[str, Any]:
    bundle = _validate_world_bundle(world_dir)
    pre_fp = _fingerprint_dir(world_dir) if enforce_source_immutable else None

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_state_dir = benchmark_output_root / f"{world_dir.name}_{run_ts}"

    cfg = OrchestratorConfig(
        api_key=_require_api_key(),
        state_dir=run_state_dir,
        solver_backend=solver_backend,
        solver_timeout=solver_timeout,
        auto_refine_on_failure=True,
        max_refinement_attempts=5,
        auto_save=False,
    )
    cfg.robot = NullRobot()

    orchestrator = TaskOrchestrator(cfg, camera=NoopCamera())
    await orchestrator.initialize()

    start = time.time()
    try:
        environment_image = np.zeros((480, 640, 3), dtype=np.uint8)
        await orchestrator.process_task_request(task, environment_image=environment_image)

        orchestrator.tracker.registry.load_from_json(str(bundle["registry_path"]))
        objects = orchestrator.get_detected_objects()
        objects_dict = [
            {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "affordances": list(obj.affordances),
                "position_3d": obj.position_3d.tolist() if obj.position_3d is not None else None,
            }
            for obj in objects
        ]
        predicates = orchestrator.tracker.registry.get_all_predicates()
        await orchestrator.maintainer.update_from_observations(objects_dict, predicates=predicates)

        result = await orchestrator.solve_and_plan_with_refinement(wait_for_objects=False)

        elapsed = time.time() - start
        payload: Dict[str, Any] = {
            "world_name": world_dir.name,
            "world_dir": str(world_dir.resolve()),
            "task": task,
            "num_objects": bundle["num_objects"],
            "num_snapshots": bundle["num_snapshots"],
            "success": result.success,
            "plan": result.plan,
            "plan_length": result.plan_length,
            "error_message": result.error_message,
            "refinement_attempts": orchestrator.refinement_attempts,
            "solver_backend": orchestrator.solver.backend.value,
            "search_time": result.search_time,
            "elapsed_seconds": round(elapsed, 3),
            "run_state_dir": str(run_state_dir.resolve()),
        }
    finally:
        await orchestrator.shutdown()

    if enforce_source_immutable:
        post_fp = _fingerprint_dir(world_dir)
        payload["source_immutable"] = pre_fp == post_fp
        if pre_fp != post_fp:
            raise RuntimeError(f"Source world directory was modified during replay: {world_dir}")

    return payload


async def _run_batch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    benchmark_output_root = Path(args.benchmark_output_root).resolve()
    benchmark_output_root.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        captured_root = Path(args.captured_root).resolve()
        if not captured_root.exists():
            raise FileNotFoundError(f"Captured root not found: {captured_root}")
        world_dirs = [d for d in sorted(captured_root.iterdir()) if d.is_dir() and (d / "TASK.md").exists()]
        if not world_dirs:
            raise RuntimeError(f"No world dirs with TASK.md found under {captured_root}")
    else:
        if not args.world_dir:
            raise ValueError("Use --world-dir for single-world mode or --run-all for batch mode")
        world_dirs = [Path(args.world_dir).resolve()]

    results: List[Dict[str, Any]] = []
    for world_dir in world_dirs:
        task_override = args.task if not args.run_all else None
        task = _load_task_from_world(world_dir, task_override)
        print(f"[replay] world={world_dir.name} task={task}")
        try:
            if args.use_existing_pddl:
                result = await _run_single_world_existing_pddl(
                    world_dir=world_dir,
                    task=task,
                    solver_backend=args.solver_backend,
                    solver_timeout=args.solver_timeout,
                    benchmark_output_root=benchmark_output_root,
                    enforce_source_immutable=args.enforce_source_immutable,
                )
            else:
                result = await _run_single_world(
                    world_dir=world_dir,
                    task=task,
                    solver_backend=args.solver_backend,
                    solver_timeout=args.solver_timeout,
                    benchmark_output_root=benchmark_output_root,
                    enforce_source_immutable=args.enforce_source_immutable,
                )
        except Exception as exc:
            result = {
                "world_name": world_dir.name,
                "world_dir": str(world_dir.resolve()),
                "task": task,
                "success": False,
                "plan": [],
                "plan_length": 0,
                "error_message": str(exc),
                "refinement_attempts": 0,
                "solver_backend": args.solver_backend,
                "search_time": None,
                "elapsed_seconds": 0.0,
                "run_state_dir": None,
                "source_immutable": None,
            }
        results.append(result)
        print(
            f"[replay] done world={world_dir.name} success={result['success']} "
            f"plan_length={result['plan_length']} refinements={result['refinement_attempts']} "
            f"elapsed={result['elapsed_seconds']}s"
        )

    return results


async def _run_single_world_existing_pddl(
    world_dir: Path,
    task: str,
    solver_backend: str,
    solver_timeout: float,
    benchmark_output_root: Path,
    enforce_source_immutable: bool,
) -> Dict[str, Any]:
    _validate_world_bundle(world_dir)
    pre_fp = _fingerprint_dir(world_dir) if enforce_source_immutable else None

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_state_dir = benchmark_output_root / f"{world_dir.name}_{run_ts}"
    run_state_dir.mkdir(parents=True, exist_ok=True)

    domain_path = world_dir / "pddl" / "task_execution_domain.pddl"
    problem_path = world_dir / "pddl" / "task_execution_problem.pddl"
    if not domain_path.exists() or not problem_path.exists():
        raise FileNotFoundError(f"Missing existing PDDL files under {world_dir / 'pddl'}")

    backend_map = {
        "auto": SolverBackend.AUTO,
        "pyperplan": SolverBackend.PYPERPLAN,
        "fast-downward-docker": SolverBackend.FAST_DOWNWARD_DOCKER,
        "fast-downward-apptainer": SolverBackend.FAST_DOWNWARD_APPTAINER,
    }
    backend = backend_map.get(solver_backend.lower(), SolverBackend.PYPERPLAN)
    solver = PDDLSolver(backend=backend, verbose=False)

    start = time.time()
    result = await solver.solve(
        domain_path=str(domain_path),
        problem_path=str(problem_path),
        algorithm=SearchAlgorithm.LAMA_FIRST,
        timeout=solver_timeout,
        working_dir=str(run_state_dir / "solver_work"),
    )
    elapsed = time.time() - start

    payload: Dict[str, Any] = {
        "world_name": world_dir.name,
        "world_dir": str(world_dir.resolve()),
        "task": task,
        "num_objects": json.loads((world_dir / "registry.json").read_text()).get("num_objects", 0),
        "num_snapshots": len(json.loads((world_dir / "perception_pool" / "index.json").read_text()).get("snapshots", {})),
        "success": result.success,
        "plan": result.plan,
        "plan_length": result.plan_length,
        "error_message": result.error_message,
        "refinement_attempts": 0,
        "solver_backend": solver.backend.value,
        "search_time": result.search_time,
        "elapsed_seconds": round(elapsed, 3),
        "run_state_dir": str(run_state_dir.resolve()),
    }

    if enforce_source_immutable:
        post_fp = _fingerprint_dir(world_dir)
        payload["source_immutable"] = pre_fp == post_fp
        if pre_fp != post_fp:
            raise RuntimeError(f"Source world directory was modified during replay: {world_dir}")

    return payload


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    non_empty = sum(1 for r in results if (r.get("plan_length") or 0) > 0)
    failures = sum(1 for r in results if not r.get("success"))
    avg_refine = (sum(r.get("refinement_attempts", 0) for r in results) / total) if total else 0.0
    avg_elapsed = (sum(r.get("elapsed_seconds", 0.0) for r in results) / total) if total else 0.0

    return {
        "total_worlds": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": round(successes / total, 3) if total else 0.0,
        "non_empty_plan_count": non_empty,
        "non_empty_plan_rate": round(non_empty / total, 3) if total else 0.0,
        "avg_refinement_attempts": round(avg_refine, 3),
        "avg_elapsed_seconds": round(avg_elapsed, 3),
        "results": results,
    }


def main() -> None:
    args = _parse_args()
    try:
        results = asyncio.run(_run_batch(args))
        summary = _build_summary(results)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Benchmark failed: {exc}")
        return
    except KeyboardInterrupt:
        print("Benchmark cancelled.")
        return

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_path = Path(args.output_json).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.benchmark_output_root).resolve() / f"summary_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
