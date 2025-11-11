"""
Synthetic benchmark generation utilities.

These helpers produce controlled obstacle patterns and speed fields to stress
different behaviours of the Optical Neuromorphic Eikonal Solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .io_utils import BenchmarkCase, save_case


@dataclass
class SyntheticConfig:
    """Specification for a synthetic benchmark case."""

    size: int
    obstacle_density: float
    speed_mode: str
    name: str
    seed: int = 0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_obstacles(size: int, density: float, seed: int) -> np.ndarray:
    gen = _rng(seed)
    grid = gen.random((size, size), dtype=np.float32)
    obstacles = (grid < density).astype(np.float32)
    return obstacles


def maze_obstacles(size: int, seed: int) -> np.ndarray:
    """Generate a perfect maze using depth-first search."""
    gen = _rng(seed)
    size = size - (size % 2 == 0)  # ensure odd
    grid = np.ones((size, size), dtype=np.float32)
    stack = [(1, 1)]
    grid[1, 1] = 0.0
    dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        x, y = stack[-1]
        gen.shuffle(dirs)
        carved = False
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and grid[ny, nx] > 0.5:
                grid[y + dy // 2, x + dx // 2] = 0.0
                grid[ny, nx] = 0.0
                stack.append((nx, ny))
                carved = True
                break
        if not carved:
            stack.pop()
    return grid


def speed_field(shape: Tuple[int, int], mode: str, seed: int) -> np.ndarray:
    h, w = shape
    gen = _rng(seed)
    if mode == "uniform":
        return np.ones(shape, dtype=np.float32)
    if mode == "gradient":
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        radial = np.sqrt(x * x + y * y)
        speeds = 1.5 - 0.8 * radial
        speeds = np.clip(speeds, 0.3, 2.0).astype(np.float32)
        return speeds
    if mode == "random":
        field = gen.uniform(0.4, 2.0, size=shape).astype(np.float32)
        return field
    if mode == "banded":
        bands = np.sin(np.linspace(0, np.pi * 4, h, dtype=np.float32))[:, None]
        speeds = 1.2 + 0.6 * bands
        return speeds.astype(np.float32)
    raise ValueError(f"Unknown speed mode '{mode}'")


def build_case(cfg: SyntheticConfig) -> BenchmarkCase:
    if cfg.speed_mode == "maze":
        obstacles = maze_obstacles(cfg.size, cfg.seed)
    else:
        obstacles = random_obstacles(cfg.size, cfg.obstacle_density, cfg.seed)
    speeds = speed_field(obstacles.shape, cfg.speed_mode if cfg.speed_mode != "maze" else "uniform", cfg.seed + 17)

    source = (1, 1)
    target = (cfg.size - 2, cfg.size - 2)
    if obstacles[source[1], source[0]] > 0.5:
        obstacles[source[1], source[0]] = 0.0
    if obstacles[target[1], target[0]] > 0.5:
        obstacles[target[1], target[0]] = 0.0

    metadata = {
        "dataset": "Synthetic",
        "name": cfg.name,
        "size": cfg.size,
        "obstacle_density": cfg.obstacle_density,
        "speed_mode": cfg.speed_mode,
        "seed": cfg.seed,
    }
    return BenchmarkCase(obstacles=obstacles, speeds=speeds, source=source, target=target, metadata=metadata)


def export_suite(configs: Iterable[SyntheticConfig], output_dir: Path | str) -> List[BenchmarkCase]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: List[BenchmarkCase] = []
    for cfg in configs:
        case = build_case(cfg)
        save_case(output_dir / f"{cfg.name}.npz", case)
        cases.append(case)
    return cases


