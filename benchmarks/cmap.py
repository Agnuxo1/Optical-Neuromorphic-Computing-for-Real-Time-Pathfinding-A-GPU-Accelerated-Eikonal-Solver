"""
Loader helpers for the CMAP Maze Benchmark (GECCO 2018).

Dataset reference: http://mazebenchmark.github.io
Original publication: \"Maze Benchmark for Testing Evolutionary Algorithms\"
by V. Koutník et al., GECCO 2018.

Expected raw format
-------------------
The public generator ships mazes as NumPy `.npy` arrays where cells contain
integer values in {0, 1}. We interpret:

- 0 → free cell
- 1 → wall / obstacle

This module converts those arrays into the internal `BenchmarkCase` structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .io_utils import BenchmarkCase


def _find_first_open_cell(grid: np.ndarray, start_corner: Tuple[int, int]) -> Tuple[int, int]:
    """Return the first free cell scanning row-major starting from a corner."""
    h, w = grid.shape
    x_range = range(w) if start_corner[0] == 0 else range(w - 1, -1, -1)
    y_range = range(h) if start_corner[1] == 0 else range(h - 1, -1, -1)

    for y in y_range:
        for x in x_range:
            if grid[y, x] < 0.5:
                return x, y
    raise ValueError("maze contains no free cells")


def load_cmap_maze(
    path: Path | str,
    *,
    source: Optional[Tuple[int, int]] = None,
    target: Optional[Tuple[int, int]] = None,
    connectivity: Optional[float] = None,
) -> BenchmarkCase:
    """
    Convert a CMAP `.npy` maze into a benchmark case.

    Parameters
    ----------
    path
        Path to the `.npy` file (binary array with values 0/1).
    source, target
        Optional explicit coordinates. If omitted, the first and last free cell
        in opposite corners are used respectively.
    connectivity
        Optional metadata copy of the generator's connectivity value (0-1).
    """
    array = np.load(Path(path))
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D maze array, got shape {array.shape}")
    maze = (array > 0).astype(np.float32)

    if source is None:
        source = _find_first_open_cell(maze, (0, 0))
    if target is None:
        target = _find_first_open_cell(maze, (maze.shape[1] - 1, maze.shape[0] - 1))

    speeds = np.ones_like(maze, dtype=np.float32)
    metadata = {
        "dataset": "CMAP-Maze",
        "connectivity": connectivity,
        "source_file": str(path),
    }
    return BenchmarkCase(
        obstacles=maze,
        speeds=speeds,
        source=source,
        target=target,
        metadata=metadata,
    )


def generate_connectivity_sweep(
    size: int,
    connectivities: Tuple[float, ...],
    output_dir: Path | str,
    generator_script: Optional[str] = None,
) -> None:
    """
    Convenience wrapper that invokes the official Python maze generator to
    create a sweep of mazes at different connectivity levels and stores them as
    `.npz` cases using :func:`load_cmap_maze`.

    Parameters
    ----------
    size
        Grid size (the CMAP generator supports 32-512).
    connectivities
        Tuple of connectivity ratios (0.0 – 1.0) to request.
    output_dir
        Destination directory for `.npz` benchmark cases.
    generator_script
        Path to `maze_generator.py`. If omitted, we assume it is available on
        the PYTHONPATH and importable.
    """
    import importlib.util
    import subprocess
    import sys

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if generator_script is None:
        generator_script = "maze_generator.py"

    script_path = Path(generator_script)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find CMAP maze generator at {script_path}. "
            "Download it from http://mazebenchmark.github.io"
        )

    for conn in connectivities:
        suffix = f"{size}_c{int(conn * 100):02d}"
        npy_path = output_dir / f"maze_{suffix}.npy"
        npz_path = output_dir / f"maze_{suffix}.npz"
        if not npy_path.exists():
            subprocess.run(
                [sys.executable, str(script_path), "--size", str(size), "--connectivity", str(conn), "--output", str(npy_path)],
                check=True,
            )
        case = load_cmap_maze(npy_path, connectivity=conn)
        from .io_utils import save_case

        save_case(npz_path, case)


