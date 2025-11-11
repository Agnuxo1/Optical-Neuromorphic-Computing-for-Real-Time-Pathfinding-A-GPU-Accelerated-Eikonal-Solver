"""
I/O helpers for benchmark datasets.

All benchmark cases are normalised to the following structure:

- `obstacles`: float32 array (1.0 = blocked, 0.0 = free)
- `speeds`: float32 array (propagation speed, default 1.0)
- `source`: tuple (sx, sy)
- `target`: tuple (tx, ty)
- `metadata`: dict with dataset-specific information

The canonical on-disk representation is an `.npz` archive so that cases can be
consumed easily by both the GPU solver and CPU references.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class BenchmarkCase:
    """Container for a single benchmark instance."""

    obstacles: np.ndarray
    speeds: np.ndarray
    source: Tuple[int, int]
    target: Tuple[int, int]
    metadata: Dict[str, Any]

    def validate(self) -> None:
        """Ensure internal consistency."""
        if self.obstacles.shape != self.speeds.shape:
            raise ValueError("obstacles and speeds must have matching shapes")
        if self.obstacles.dtype != np.float32:
            raise TypeError("obstacles must be float32")
        if self.speeds.dtype != np.float32:
            raise TypeError("speeds must be float32")
        sx, sy = self.source
        tx, ty = self.target
        h, w = self.obstacles.shape
        for name, (x, y) in {"source": (sx, sy), "target": (tx, ty)}.items():
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"{name} coordinate {(x, y)} out of bounds for grid {w}x{h}")
        for arr_name, arr in {"obstacles": self.obstacles, "speeds": self.speeds}.items():
            if np.any(np.isnan(arr)):
                raise ValueError(f"{arr_name} contains NaN values")


def _ensure_float_grid(grid: np.ndarray) -> np.ndarray:
    if grid.dtype != np.float32:
        grid = grid.astype(np.float32, copy=False)
    return grid


def save_case(path: Path | str, case: BenchmarkCase) -> None:
    """
    Persist a benchmark case as `.npz`.

    Parameters
    ----------
    path
        Output file path (created parent directories automatically).
    case
        Benchmark instance to serialise.
    """
    case.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obstacles=case.obstacles,
        speeds=case.speeds,
        source=np.array(case.source, dtype=np.int32),
        target=np.array(case.target, dtype=np.int32),
        metadata=json.dumps(case.metadata, ensure_ascii=True),
    )


def load_case(path: Path | str) -> BenchmarkCase:
    """
    Load a benchmark `.npz` archive.

    Returns
    -------
    BenchmarkCase
    """
    path = Path(path)
    with np.load(path) as data:
        obstacles = _ensure_float_grid(data["obstacles"])
        speeds = _ensure_float_grid(data["speeds"])
        source = tuple(int(v) for v in data["source"])  # type: ignore
        target = tuple(int(v) for v in data["target"])  # type: ignore
        metadata = json.loads(str(data["metadata"]))
    case = BenchmarkCase(
        obstacles=obstacles,
        speeds=speeds,
        source=source,  # type: ignore
        target=target,  # type: ignore
        metadata=metadata,
    )
    case.validate()
    return case


def case_to_dict(case: BenchmarkCase) -> Dict[str, Any]:
    """Return a serialisable dictionary (metadata already JSON serialisable)."""
    payload = asdict(case)
    payload["obstacles"] = case.obstacles.tolist()
    payload["speeds"] = case.speeds.tolist()
    return payload


