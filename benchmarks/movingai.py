"""
MovingAI benchmark loaders.

Reference: http://movingai.com/benchmarks/
Publication: Daniel Harabor & Alban Grastien (2011),
             \"Online Graph Pruning for Pathfinding on Grid Maps\"

The benchmark distributes grid maps (`.map`) and scenario files (`.scen`)
listing start/goal pairs. We expose utilities to read those and convert them
into :class:`BenchmarkCase` objects for the Eikonal solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .io_utils import BenchmarkCase

# Terrain legend based on MovingAI spec
CHAR_COST: Dict[str, float] = {
    ".": 1.0,  # walkable
    "G": 1.5,  # ground
    "S": 1.0,  # swamp
    "W": 2.0,  # water / slow terrain
    "w": 1.5,
    "T": np.inf,  # tree -> blocked
    "@": np.inf,  # stone wall -> blocked
    "O": 1.0,  # open door
    "X": np.inf,  # closed door -> blocked
}


@dataclass
class ScenarioEntry:
    """Single entry from a `.scen` file."""

    map_path: str
    bucket: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    optimal_length: float


def parse_map(path: Path | str) -> np.ndarray:
    """
    Parse a MovingAI `.map` file into boolean obstacles and float speeds.

    Returns
    -------
    np.ndarray
        Obstacles array (float32).
    np.ndarray
        Speeds array (float32).
    """
    path = Path(path)
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip().lower()
        if header != "type octile":
            raise ValueError(f"Unsupported map type '{header}' (expected 'type octile')")

        dimensions = {}
        for _ in range(2):
            key, value = handle.readline().split()
            dimensions[key.lower()] = int(value)
        width = dimensions.get("width")
        height = dimensions.get("height")
        if width is None or height is None:
            raise ValueError("Map header missing width/height")

        grid: List[List[str]] = []
        for _ in range(height):
            line = handle.readline().rstrip("\n")
            if len(line) != width:
                raise ValueError(f"Expected row of length {width}, got {len(line)} characters")
            grid.append(list(line))

    chars = np.array(grid)
    costs = np.zeros_like(chars, dtype=np.float32)
    for terrain, cost in CHAR_COST.items():
        mask = chars == terrain
        if cost == np.inf:
            costs[mask] = np.inf
        else:
            costs[mask] = cost

    if np.any(costs == 0):
        unknown = np.unique(chars[costs == 0])
        raise ValueError(f"Encountered unknown map symbols: {unknown}")

    obstacles = np.isinf(costs).astype(np.float32)
    costs[costs == np.inf] = 1.0  # placeholder to avoid divide by zero
    speeds = (1.0 / costs).astype(np.float32)
    speeds[obstacles > 0.5] = 0.0
    return obstacles, speeds


def parse_scen(path: Path | str) -> Iterator[ScenarioEntry]:
    """Yield entries from a `.scen` file."""
    path = Path(path)
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()
        if not header.startswith("version"):
            raise ValueError("Invalid .scen file (missing version header)")
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 9:
                continue
            _, map_name, bucket, sx, sy, gx, gy, _, optimal = fields[:9]
            yield ScenarioEntry(
                map_path=map_name,
                bucket=int(bucket),
                start=(int(sx), int(sy)),
                goal=(int(gx), int(gy)),
                optimal_length=float(optimal),
            )


def load_cases_from_map(
    map_path: Path | str,
    scen_entries: Iterable[ScenarioEntry],
    *,
    root: Optional[Path] = None,
) -> List[BenchmarkCase]:
    """
    Convert a map + scenario entries into benchmark cases.

    Parameters
    ----------
    map_path
        Path to the `.map` file (absolute or relative to `root`).
    scen_entries
        Iterable of :class:`ScenarioEntry` objects associated with `map_path`.
    root
        Optional base directory containing the MovingAI dataset.
    """
    if root is not None:
        map_path = root / map_path

    obstacles, speeds = parse_map(map_path)
    cases: List[BenchmarkCase] = []

    for entry in scen_entries:
        metadata = {
            "dataset": "MovingAI",
            "map_file": str(map_path),
            "bucket": entry.bucket,
            "optimal_length": entry.optimal_length,
            "scenario_source": entry.map_path,
        }
        case = BenchmarkCase(
            obstacles=obstacles.copy(),
            speeds=speeds.copy(),
            source=entry.start,
            target=entry.goal,
            metadata=metadata,
        )
        cases.append(case)
    return cases


def load_map_and_scen(
    map_path: Path | str,
    scen_path: Path | str,
    *,
    limit: Optional[int] = None,
    root: Optional[Path] = None,
) -> List[BenchmarkCase]:
    """
    Convenience wrapper for a single map file and its scenario pairs.

    Parameters
    ----------
    limit
        Optional maximum number of scenarios to convert (useful for quick tests).
    root
        Base directory to resolve relative paths stored in `.scen` files.
    """
    entries = list(parse_scen(scen_path))
    if root is None:
        root = Path(map_path).parent

    filtered = [e for e in entries if Path(e.map_path).name == Path(map_path).name]
    if not filtered:
        raise ValueError(f"No scenarios in {scen_path} reference map {map_path}")
    if limit is not None:
        filtered = filtered[:limit]

    return load_cases_from_map(map_path, filtered, root=root)


