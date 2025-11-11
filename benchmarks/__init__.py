"""
Benchmark utilities for the Optical Neuromorphic Eikonal Solver.

This package centralises dataset loaders, converters and generators so that
public benchmarks (CMAP mazes, MovingAI maps, Grid Pathfinding competition data)
can be consumed in a reproducible way.
"""

from . import cmap, movingai, synthetic, io_utils  # noqa: F401


