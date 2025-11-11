#!/usr/bin/env python3
"""
Run GPU vs CPU benchmarks on prepared `.npz` cases.

Example:
    python -m benchmarks.run_suite --cases cases/cmap --output results/cmap.csv --iterations 300
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import List

from .runner import BenchmarkResult, BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="Directory with .npz benchmark cases")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV file for metrics")
    parser.add_argument("--iterations", type=int, default=200, help="Propagation iterations per case")
    parser.add_argument("--grid", type=int, default=None, help="Fixed GPU grid size (if cases already resampled)")
    return parser


def summarise(results: List[BenchmarkResult]) -> None:
    speedups = [r.cpu_time / max(r.gpu_time, 1e-9) for r in results]
    maes = [r.mae for r in results]
    path_ratios = [r.path_ratio for r in results]
    print(f"Cases evaluated : {len(results)}")
    print(f"Avg speedup     : {mean(speedups):.2f}x")
    print(f"Avg MAE         : {mean(maes):.4f}")
    print(f"Avg path ratio  : {mean(path_ratios):.4f}")


def write_csv(path: Path, results: List[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "gpu_time",
                "cpu_time",
                "speedup",
                "mae",
                "path_ratio",
                "metadata",
            ]
        )
        for r in results:
            speedup = r.cpu_time / max(r.gpu_time, 1e-9)
            writer.writerow(
                [
                    r.case_path.name,
                    f"{r.gpu_time:.6f}",
                    f"{r.cpu_time:.6f}",
                    f"{speedup:.2f}",
                    f"{r.mae:.6f}",
                    f"{r.path_ratio:.6f}",
                    r.metadata,
                ]
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runner = BenchmarkRunner(iterations=args.iterations, gpu_grid=args.grid)
    results = runner.run_directory(args.cases)
    write_csv(args.output, results)
    summarise(results)


if __name__ == "__main__":
    main()


