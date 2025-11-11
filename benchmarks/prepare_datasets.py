#!/usr/bin/env python3
"""
Dataset preparation CLI for the Optical Neuromorphic Eikonal Solver benchmarks.

This script exposes a uniform interface to convert public benchmarks into the
internal `.npz` format expected by the solver and reference harness.

Usage examples
--------------

Convert a CMAP maze:
    python -m benchmarks.prepare_datasets cmap --input maze_256.npy --output cases/cmap_256.npz --connectivity 0.3

Convert a MovingAI map + scenarios:
    python -m benchmarks.prepare_datasets movingai --map maps/maze512-32-0.map --scen maps/maze512-32-0.map.scen --output cases/movingai --limit 50

Generate the recommended synthetic suite:
    python -m benchmarks.prepare_datasets synthetic --output cases/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from . import cmap, movingai, synthetic
from .io_utils import BenchmarkCase, save_case


def parse_coord(value: str) -> Tuple[int, int]:
    try:
        x_str, y_str = value.split(",")
        return int(x_str), int(y_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coordinates must be 'x,y'") from exc


def command_cmap(args: argparse.Namespace) -> None:
    source = parse_coord(args.source) if args.source else None
    target = parse_coord(args.target) if args.target else None
    case = cmap.load_cmap_maze(
        args.input,
        source=source,
        target=target,
        connectivity=args.connectivity,
    )
    save_case(args.output, case)
    print(f"[cmap] Saved {args.output} ({case.obstacles.shape[1]}x{case.obstacles.shape[0]})")


def command_movingai(args: argparse.Namespace) -> None:
    map_path = Path(args.map)
    scen_path = Path(args.scen)
    cases = movingai.load_map_and_scen(
        map_path,
        scen_path,
        limit=args.limit,
        root=args.root and Path(args.root),
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(cases):
        save_case(output_dir / f"{map_path.stem}_{idx:04d}.npz", case)
    print(f"[movingai] Converted {len(cases)} scenarios → {output_dir}")


def command_synthetic(args: argparse.Namespace) -> None:
    configs = _synthetic_suite(args.scale)
    cases = synthetic.export_suite(configs, args.output)
    print(f"[synthetic] Generated {len(cases)} cases in {args.output}")


def _synthetic_suite(scale: float = 1.0) -> Iterable[synthetic.SyntheticConfig]:
    base = [
        synthetic.SyntheticConfig(128, 0.10, "uniform", "sparse_128", seed=1),
        synthetic.SyntheticConfig(256, 0.20, "uniform", "medium_256", seed=3),
        synthetic.SyntheticConfig(256, 0.20, "gradient", "gradient_256", seed=5),
        synthetic.SyntheticConfig(512, 0.30, "random", "complex_512", seed=7),
        synthetic.SyntheticConfig(511, 0.30, "maze", "maze_511", seed=11),
    ]
    if scale == 1.0:
        return base
    scaled: List[synthetic.SyntheticConfig] = []
    for cfg in base:
        new_size = max(64, int(cfg.size * scale))
        scaled.append(
            synthetic.SyntheticConfig(
                size=new_size if cfg.speed_mode != "maze" else new_size | 1,
                obstacle_density=cfg.obstacle_density,
                speed_mode=cfg.speed_mode,
                name=f"{cfg.name}_x{scale:.2f}",
                seed=cfg.seed,
            )
        )
    return scaled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmap = sub.add_parser("cmap", help="Convert a CMAP maze .npy into a benchmark case")
    p_cmap.add_argument("--input", required=True, type=Path, help="Input .npy maze")
    p_cmap.add_argument("--output", required=True, type=Path, help="Destination .npz path")
    p_cmap.add_argument("--connectivity", type=float, default=None, help="Connectivity metadata (0-1)")
    p_cmap.add_argument("--source", type=str, default=None, help="Source coordinate 'x,y'")
    p_cmap.add_argument("--target", type=str, default=None, help="Target coordinate 'x,y'")
    p_cmap.set_defaults(func=command_cmap)

    p_movingai = sub.add_parser("movingai", help="Convert MovingAI map + scenarios")
    p_movingai.add_argument("--map", required=True, type=Path, help="Path to .map file")
    p_movingai.add_argument("--scen", required=True, type=Path, help="Path to .scen file")
    p_movingai.add_argument("--output", required=True, type=Path, help="Output directory for .npz cases")
    p_movingai.add_argument("--limit", type=int, default=None, help="Optional limit of scenarios to convert")
    p_movingai.add_argument("--root", type=Path, default=None, help="Dataset root to resolve relative map paths")
    p_movingai.set_defaults(func=command_movingai)

    p_synth = sub.add_parser("synthetic", help="Generate the reference synthetic suite")
    p_synth.add_argument("--output", required=True, type=Path, help="Output directory for .npz cases")
    p_synth.add_argument("--scale", type=float, default=1.0, help="Uniform scaling factor for grid sizes")
    p_synth.set_defaults(func=command_synthetic)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


