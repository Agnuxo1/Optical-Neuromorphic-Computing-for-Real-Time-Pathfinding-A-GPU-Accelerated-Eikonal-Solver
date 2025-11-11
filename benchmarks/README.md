# Benchmark Preparation Guide

This directory centralises the tooling required to evaluate the **Optical Neuromorphic Eikonal Solver** on public benchmarks and synthetic suites.

## Available converters

| Command | Dataset | Source | Notes |
|---------|---------|--------|-------|
| `python -m benchmarks.prepare_datasets cmap` | CMAP Maze Benchmark | http://mazebenchmark.github.io | Accepts `.npy` mazes produced by the official generator. |
| `python -m benchmarks.prepare_datasets movingai` | MovingAI Pathfinding | http://movingai.com/benchmarks/ | Parses `.map` + `.scen` files and emits `.npz` cases. |
| `python -m benchmarks.prepare_datasets synthetic` | Synthetic suite | Local | Generates reproducible random/gradient/maze cases. |

Each command produces `.npz` archives containing:

```
obstacles -> float32 grid (1.0 = blocked)
speeds    -> float32 grid (propagation speed)
source    -> [sx, sy]
target    -> [tx, ty]
metadata  -> JSON string with provenance details
```

## Example usage

```bash
# 1) CMAP maze (connectivity sweep)
python -m benchmarks.prepare_datasets cmap \
    --input raw/cmap/maze_256_c03.npy \
    --output cases/cmap/maze_256_c03.npz \
    --connectivity 0.3

# 2) MovingAI map (limit first 50 scenarios)
python -m benchmarks.prepare_datasets movingai \
    --map raw/movingai/mazes/maze512-32-0.map \
    --scen raw/movingai/mazes/maze512-32-0.map.scen \
    --output cases/movingai/maze512-32-0 \
    --limit 50

# 3) Synthetic reference suite
python -m benchmarks.prepare_datasets synthetic \
    --output cases/synthetic
```

The generated `.npz` files can be consumed directly by the upcoming benchmarking harness (see plan step `benchmark-runner`), or inspected interactively via the existing solver.


