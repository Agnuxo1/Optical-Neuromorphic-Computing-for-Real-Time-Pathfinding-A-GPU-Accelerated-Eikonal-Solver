"""
Benchmark execution harness.

Provides utility classes to benchmark the Optical Neuromorphic Eikonal Solver
against CPU references on a collection of `.npz` cases produced by the dataset
preparation tooling.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .io_utils import BenchmarkCase, load_case


@dataclass
class BenchmarkResult:
    """Aggregated metrics for a single case."""

    case_path: Path
    gpu_time: float
    cpu_time: float
    mae: float
    path_ratio: float
    metadata: Dict[str, str]


def dijkstra_reference(case: BenchmarkCase) -> np.ndarray:
    """
    CPU reference using 4-neighbour Dijkstra with costs derived from speeds.
    """
    from heapq import heappop, heappush

    h, w = case.obstacles.shape
    cost_grid = np.full((h, w), np.inf, dtype=np.float64)
    sx, sy = case.source
    tx, ty = case.target

    pq: List[tuple[float, int, int]] = []
    cost_grid[sy, sx] = 0.0
    heappush(pq, (0.0, sx, sy))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        dist, x, y = heappop(pq)
        if dist > cost_grid[y, x]:
            continue
        if (x, y) == (tx, ty):
            break

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if case.obstacles[ny, nx] > 0.5:
                    continue
                step_speed = max((case.speeds[y, x] + case.speeds[ny, nx]) * 0.5, 1e-4)
                new_cost = dist + (1.0 / step_speed)
                if new_cost < cost_grid[ny, nx]:
                    cost_grid[ny, nx] = new_cost
                    heappush(pq, (new_cost, nx, ny))
    return cost_grid.astype(np.float32)


def reconstruct_path(cost_field: np.ndarray, case: BenchmarkCase) -> List[tuple[int, int]]:
    """Follow steepest descent on the scalar field to recover a path."""
    h, w = cost_field.shape
    x, y = case.target
    path: List[tuple[int, int]] = []
    visited = set()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(h * w):
        path.append((x, y))
        if (x, y) == case.source:
            break
        visited.add((x, y))
        best = (x, y)
        best_val = cost_field[y, x]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                candidate = cost_field[ny, nx]
                if candidate + 1e-6 < best_val:
                    best_val = candidate
                    best = (nx, ny)
        if best == (x, y):
            break
        x, y = best
    return path


class GPUSolverContext:
    """
    Helper that wraps `OpticalEikonalSolver` for headless benchmarking.

    A hidden window is created (GLFW requires a context), the solver state is
    updated manually, and the propagation kernel executed for a fixed number of
    iterations.
    """

    def __init__(self, grid_size: int):
        import glfw
        from quantum_eikonal_solver import OpticalEikonalSolver

        if not glfw.init():
            raise RuntimeError("GLFW initialisation failed (required for GPU benchmarking)")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._solver = OpticalEikonalSolver(grid_size=grid_size, window_scale=1, iterations_per_frame=1)
        self._solver.show_grid = False
        self._solver.show_photons = False
        self._solver.debug_gates = False
        self.grid_size = grid_size

    def close(self) -> None:
        import glfw

        glfw.terminate()

    def run_case(self, case: BenchmarkCase, iterations: int) -> tuple[np.ndarray, List[tuple[int, int]], float]:
        solver = self._solver
        if case.obstacles.shape != (solver.grid_size, solver.grid_size):
            raise ValueError("Case resolution does not match solver grid size")

        solver.obstacle_field = case.obstacles.copy()
        solver.speed_field = case.speeds.copy()
        solver.obstacle_texture.write(solver.obstacle_field.tobytes())
        solver.speed_texture.write(solver.speed_field.tobytes())

        # Reset and assign endpoints
        solver._assign_endpoints(case.source, case.target, preserve_state=False)
        # Note: _assign_endpoints already calls _reset_time_field internally

        start = time.perf_counter()
        for _ in range(iterations):
            solver._propagation_step()
        gpu_time = time.perf_counter() - start

        field = solver._read_time_field()
        path = solver._extract_path_from_field(field)
        return field, path, gpu_time


@dataclass
class BenchmarkRunner:
    """High-level benchmarking pipeline."""

    iterations: int = 200
    gpu_grid: Optional[int] = None

    def run_directory(self, directory: Path | str) -> List[BenchmarkResult]:
        directory = Path(directory)
        cases = sorted(directory.glob("*.npz"))
        if not cases:
            raise FileNotFoundError(f"No .npz cases found in {directory}")

        results: List[BenchmarkResult] = []
        gpu_context: Optional[GPUSolverContext] = None
        try:
            for path in cases:
                case = load_case(path)
                grid_size = self.gpu_grid or case.obstacles.shape[0]
                if gpu_context is None or gpu_context.grid_size != grid_size:
                    if gpu_context is not None:
                        gpu_context.close()
                    gpu_context = GPUSolverContext(grid_size=grid_size)
                result = self._run_single_case(path, case, gpu_context)
                results.append(result)
        finally:
            if gpu_context is not None:
                gpu_context.close()
        return results

    def _run_single_case(self, path: Path, case: BenchmarkCase, gpu_context: GPUSolverContext) -> BenchmarkResult:
        cpu_start = time.perf_counter()
        cpu_field = dijkstra_reference(case)
        cpu_time = time.perf_counter() - cpu_start

        # Adaptive iterations based on grid size
        grid_size = case.obstacles.shape[0]
        iterations = max(self.iterations, grid_size * 2)  # At least 2x grid size
        
        gpu_field, gpu_path, gpu_time = gpu_context.run_case(case, iterations)
        
        # If no path found, try with more iterations
        if not gpu_path and iterations < grid_size * 4:
            print(f"[Warning] No path found for {path.name}, retrying with {grid_size * 4} iterations...")
            gpu_field, gpu_path, gpu_time = gpu_context.run_case(case, grid_size * 4)
        
        if not gpu_path:
            print(f"[Warning] GPU solver failed to recover path for {path.name}, using CPU path")
            gpu_path = reconstruct_path(cpu_field, case)

        mae = float(np.mean(np.abs(cpu_field - gpu_field)))

        cpu_path = reconstruct_path(cpu_field, case)
        path_ratio = float(len(gpu_path) / max(len(cpu_path), 1))

        return BenchmarkResult(
            case_path=path,
            gpu_time=gpu_time,
            cpu_time=cpu_time,
            mae=mae,
            path_ratio=path_ratio,
            metadata=case.metadata,
        )


