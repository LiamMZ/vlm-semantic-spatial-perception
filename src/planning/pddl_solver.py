"""
PDDL Solver Interface

Provides unified interface for solving PDDL problems using various backends:
- Fast Downward (Docker)
- Fast Downward (Apptainer/Singularity)
- Pyperplan (Python fallback)

Integrates seamlessly with the async task orchestrator.
"""

import os
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import tempfile


class SolverBackend(Enum):
    """Available PDDL solver backends."""
    FAST_DOWNWARD_DOCKER = "fast-downward-docker"
    FAST_DOWNWARD_APPTAINER = "fast-downward-apptainer"
    PYPERPLAN = "pyperplan"
    AUTO = "auto"  # Automatically detect best available


class SearchAlgorithm(Enum):
    """Common search algorithms."""
    LAMA_FIRST = "lama-first"  # Fast, ignores cost
    LAMA = "lama"  # Optimizes cost
    ASTAR_LMCUT = "astar(lmcut())"  # A* with landmark cut heuristic
    LAZY_GREEDY_FF = "lazy_greedy([ff()], preferred=[ff()])"  # Fast forward heuristic

    def __str__(self):
        return self.value


@dataclass
class SolverResult:
    """Result from PDDL solver."""
    success: bool
    plan: List[str]  # Sequence of action names
    plan_length: int
    plan_cost: Optional[float]
    search_time: Optional[float]
    nodes_expanded: Optional[int]
    error_message: Optional[str] = None
    raw_output: Optional[str] = None

    def __str__(self):
        if not self.success:
            return f"Solver failed: {self.error_message}"

        result = f"Plan found ({self.plan_length} steps"
        if self.plan_cost is not None:
            result += f", cost: {self.plan_cost}"
        if self.search_time is not None:
            result += f", time: {self.search_time:.2f}s"
        result += ")"
        return result


class PDDLSolver:
    """
    Unified interface for solving PDDL problems.

    Automatically detects available solver backends and provides
    a simple async interface for plan generation.

    Example:
        >>> solver = PDDLSolver(backend=SolverBackend.AUTO)
        >>> result = await solver.solve(
        ...     domain_path="domain.pddl",
        ...     problem_path="problem.pddl",
        ...     algorithm=SearchAlgorithm.LAMA_FIRST,
        ...     timeout=30.0
        ... )
        >>> if result.success:
        ...     print(f"Plan: {result.plan}")
    """

    def __init__(
        self,
        backend: SolverBackend = SolverBackend.AUTO,
        docker_image: str = "aibasel/downward:latest",
        apptainer_image: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize PDDL solver.

        Args:
            backend: Solver backend to use (AUTO detects best available)
            docker_image: Docker image for Fast Downward
            apptainer_image: Path to Apptainer .sif file (auto-downloads if None)
            verbose: Print solver output
        """
        self.backend = backend
        self.docker_image = docker_image
        self.apptainer_image = apptainer_image
        self.verbose = verbose

        # Detect available backends
        self._available_backends = self._detect_backends()

        # Auto-select backend if needed
        if self.backend == SolverBackend.AUTO:
            self.backend = self._select_best_backend()
            if self.verbose:
                print(f"Auto-selected backend: {self.backend.value}")

    def _detect_backends(self) -> Dict[SolverBackend, bool]:
        """Detect which solver backends are available."""
        available = {}

        # Check for Docker
        available[SolverBackend.FAST_DOWNWARD_DOCKER] = (
            shutil.which("docker") is not None
        )

        # Check for Apptainer/Singularity
        apptainer_cmd = shutil.which("apptainer") or shutil.which("singularity")
        available[SolverBackend.FAST_DOWNWARD_APPTAINER] = (
            apptainer_cmd is not None
        )

        # Check for Pyperplan
        try:
            import pyperplan
            available[SolverBackend.PYPERPLAN] = True
        except ImportError:
            available[SolverBackend.PYPERPLAN] = False

        return available

    def _select_best_backend(self) -> SolverBackend:
        """Select best available backend (preference order)."""
        preference_order = [
            SolverBackend.FAST_DOWNWARD_APPTAINER,  # Fastest, no overhead
            SolverBackend.FAST_DOWNWARD_DOCKER,     # Fast, some overhead
            SolverBackend.PYPERPLAN,                # Fallback
        ]

        for backend in preference_order:
            if self._available_backends.get(backend, False):
                return backend

        raise RuntimeError(
            "No PDDL solver backend available. Install one of:\n"
            "  - Apptainer: apt install apptainer\n"
            "  - Docker: apt install docker.io\n"
            "  - Pyperplan: pip install pyperplan"
        )

    async def solve(
        self,
        domain_path: str,
        problem_path: str,
        algorithm: SearchAlgorithm = SearchAlgorithm.LAMA_FIRST,
        timeout: float = 30.0,
        working_dir: Optional[str] = None
    ) -> SolverResult:
        """
        Solve a PDDL problem.

        Args:
            domain_path: Path to domain.pddl file
            problem_path: Path to problem.pddl file
            algorithm: Search algorithm to use
            timeout: Timeout in seconds
            working_dir: Working directory for solver output

        Returns:
            SolverResult with plan and statistics
        """
        # Validate inputs
        domain_path = Path(domain_path).resolve()
        problem_path = Path(problem_path).resolve()

        if not domain_path.exists():
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Domain file not found: {domain_path}"
            )

        if not problem_path.exists():
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Problem file not found: {problem_path}"
            )

        # Route to appropriate backend
        if self.backend == SolverBackend.FAST_DOWNWARD_DOCKER:
            return await self._solve_docker(
                domain_path, problem_path, algorithm, timeout, working_dir
            )
        elif self.backend == SolverBackend.FAST_DOWNWARD_APPTAINER:
            return await self._solve_apptainer(
                domain_path, problem_path, algorithm, timeout, working_dir
            )
        elif self.backend == SolverBackend.PYPERPLAN:
            return await self._solve_pyperplan(
                domain_path, problem_path, algorithm, timeout
            )
        else:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Unsupported backend: {self.backend}"
            )

    async def _solve_docker(
        self,
        domain_path: Path,
        problem_path: Path,
        algorithm: SearchAlgorithm,
        timeout: float,
        working_dir: Optional[str]
    ) -> SolverResult:
        """Solve using Fast Downward Docker image."""
        if working_dir is None:
            working_dir = tempfile.mkdtemp(prefix="pddl_solver_")
        working_dir = Path(working_dir).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)

        # Docker requires absolute paths
        domain_abs = domain_path.absolute()
        problem_abs = problem_path.absolute()
        mount_dir = domain_abs.parent.absolute()

        # Build Docker command
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{mount_dir}:/workspace",
            "-v", f"{working_dir}:/output",
            "-w", "/output",
            self.docker_image,
            "--alias", str(algorithm),
            f"/workspace/{domain_abs.name}",
            f"/workspace/{problem_abs.name}",
        ]

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        # Run solver
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')

            if self.verbose:
                print(stdout_text)
                if stderr_text:
                    print(f"STDERR: {stderr_text}")

            # Parse output
            return self._parse_fast_downward_output(
                stdout_text + stderr_text,
                working_dir
            )

        except asyncio.TimeoutError:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=timeout,
                nodes_expanded=None,
                error_message=f"Solver timeout after {timeout}s"
            )
        except Exception as e:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Docker error: {e}"
            )

    async def _solve_apptainer(
        self,
        domain_path: Path,
        problem_path: Path,
        algorithm: SearchAlgorithm,
        timeout: float,
        working_dir: Optional[str]
    ) -> SolverResult:
        """Solve using Fast Downward Apptainer image."""
        # Get or download Apptainer image
        if self.apptainer_image is None:
            self.apptainer_image = await self._ensure_apptainer_image()

        sif_path = Path(self.apptainer_image)
        if not sif_path.exists():
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Apptainer image not found: {sif_path}"
            )

        if working_dir is None:
            working_dir = tempfile.mkdtemp(prefix="pddl_solver_")
        working_dir = Path(working_dir).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)

        # Build Apptainer command
        cmd = [
            str(sif_path),
            "--alias", str(algorithm),
            str(domain_path),
            str(problem_path),
        ]

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        # Run solver
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')

            if self.verbose:
                print(stdout_text)
                if stderr_text:
                    print(f"STDERR: {stderr_text}")

            # Parse output
            return self._parse_fast_downward_output(
                stdout_text + stderr_text,
                working_dir
            )

        except asyncio.TimeoutError:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=timeout,
                nodes_expanded=None,
                error_message=f"Solver timeout after {timeout}s"
            )
        except Exception as e:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Apptainer error: {e}"
            )

    async def _solve_pyperplan(
        self,
        domain_path: Path,
        problem_path: Path,
        algorithm: SearchAlgorithm,
        timeout: float
    ) -> SolverResult:
        """Solve using Pyperplan (Python fallback)."""
        try:
            from pyperplan import planner
        except ImportError:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message="Pyperplan not installed: pip install pyperplan"
            )

        # Map algorithm to Pyperplan search and heuristic
        # Pyperplan uses string keys to look up search and heuristic
        search_name = "astar"
        heuristic_name = "hff"

        if algorithm == SearchAlgorithm.LAZY_GREEDY_FF:
            search_name = "gbf"  # Greedy best-first
            heuristic_name = "hff"

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def run_planner():
                # Get search and heuristic from planner dictionaries
                search_func = planner.SEARCHES[search_name]
                heuristic_class = planner.HEURISTICS[heuristic_name]

                return planner.search_plan(
                    str(domain_path),
                    str(problem_path),
                    search=search_func,
                    heuristic_class=heuristic_class
                )

            # Run with timeout
            plan = await asyncio.wait_for(
                loop.run_in_executor(None, run_planner),
                timeout=timeout
            )

            if plan is None:
                return SolverResult(
                    success=False,
                    plan=[],
                    plan_length=0,
                    plan_cost=None,
                    search_time=None,
                    nodes_expanded=None,
                    error_message="No plan found"
                )

            # Extract action names
            action_names = [action.name for action in plan]

            return SolverResult(
                success=True,
                plan=action_names,
                plan_length=len(action_names),
                plan_cost=float(len(action_names)),  # Pyperplan doesn't provide cost
                search_time=None,
                nodes_expanded=None
            )

        except asyncio.TimeoutError:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=timeout,
                nodes_expanded=None,
                error_message=f"Solver timeout after {timeout}s"
            )
        except Exception as e:
            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=f"Pyperplan error: {e}"
            )

    def _parse_fast_downward_output(
        self,
        output: str,
        working_dir: Path
    ) -> SolverResult:
        """Parse Fast Downward output and extract plan."""
        # Check for solution
        if "Solution found!" not in output:
            # Check for common failure messages
            if "unsolvable" in output.lower():
                error_msg = "Problem is unsolvable"
            elif "parse error" in output.lower():
                error_msg = "PDDL parse error"
            elif "no" in output.lower() and "found" in output.lower():
                error_msg = "No solution found"
            else:
                error_msg = "Solver failed (see raw output)"

            return SolverResult(
                success=False,
                plan=[],
                plan_length=0,
                plan_cost=None,
                search_time=None,
                nodes_expanded=None,
                error_message=error_msg,
                raw_output=output
            )

        # Extract statistics
        plan_cost = None
        search_time = None
        nodes_expanded = None

        for line in output.split('\n'):
            if "Plan cost:" in line:
                try:
                    plan_cost = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Search time:" in line:
                try:
                    search_time = float(line.split(":")[-1].strip().rstrip("s"))
                except:
                    pass
            elif "Expanded" in line and "state(s)" in line:
                try:
                    nodes_expanded = int(line.split()[1])
                except:
                    pass

        # Try to read plan from sas_plan file
        plan = []
        sas_plan_path = working_dir / "sas_plan"

        if sas_plan_path.exists():
            with open(sas_plan_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(';'):
                        # Extract action name (remove parameters)
                        action = line.strip('()')
                        plan.append(action)
        else:
            # Try to parse plan from output
            in_plan = False
            for line in output.split('\n'):
                if "Solution found!" in line or "Plan:" in line:
                    in_plan = True
                    continue

                if in_plan:
                    line = line.strip()
                    if line and not line.startswith(';') and '(' in line:
                        action = line.strip('()')
                        plan.append(action)
                    elif line == "":
                        break

        return SolverResult(
            success=True,
            plan=plan,
            plan_length=len(plan),
            plan_cost=plan_cost,
            search_time=search_time,
            nodes_expanded=nodes_expanded,
            raw_output=output
        )

    async def _ensure_apptainer_image(self) -> str:
        """Download Apptainer image if not present."""
        # Check common locations
        home_dir = Path.home()
        cache_dir = home_dir / ".cache" / "pddl_solver"
        cache_dir.mkdir(parents=True, exist_ok=True)

        sif_path = cache_dir / "fast-downward.sif"

        if sif_path.exists():
            return str(sif_path)

        print(f"Downloading Fast Downward Apptainer image to {sif_path}...")
        print("This is a one-time download (~200MB)...")

        cmd = [
            "apptainer", "pull",
            str(sif_path),
            f"docker://{self.docker_image}"
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and sif_path.exists():
                print(f"✓ Download complete: {sif_path}")
                return str(sif_path)
            else:
                raise RuntimeError(
                    f"Failed to download image: {stderr.decode('utf-8')}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to download Apptainer image: {e}")

    def get_available_backends(self) -> List[SolverBackend]:
        """Get list of available solver backends."""
        return [
            backend for backend, available in self._available_backends.items()
            if available
        ]

    def is_backend_available(self, backend: SolverBackend) -> bool:
        """Check if a specific backend is available."""
        return self._available_backends.get(backend, False)


# Convenience function for quick solving
async def solve_pddl(
    domain_path: str,
    problem_path: str,
    algorithm: SearchAlgorithm = SearchAlgorithm.LAMA_FIRST,
    timeout: float = 30.0,
    backend: SolverBackend = SolverBackend.AUTO,
    verbose: bool = False
) -> SolverResult:
    """
    Convenience function to solve a PDDL problem.

    Args:
        domain_path: Path to domain.pddl
        problem_path: Path to problem.pddl
        algorithm: Search algorithm to use
        timeout: Timeout in seconds
        backend: Solver backend (AUTO detects best)
        verbose: Print solver output

    Returns:
        SolverResult with plan and statistics

    Example:
        >>> result = await solve_pddl("domain.pddl", "problem.pddl")
        >>> if result.success:
        ...     print(f"Plan: {result.plan}")
    """
    solver = PDDLSolver(backend=backend, verbose=verbose)
    return await solver.solve(
        domain_path=domain_path,
        problem_path=problem_path,
        algorithm=algorithm,
        timeout=timeout
    )
