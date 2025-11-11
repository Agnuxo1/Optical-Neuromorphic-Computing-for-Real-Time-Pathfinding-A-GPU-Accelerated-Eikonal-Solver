# Dataset Preparation Guide

This guide explains how to obtain and prepare benchmark datasets for the Optical Neuromorphic Eikonal Solver.

## Quick Start: Synthetic Datasets

The project includes a synthetic dataset generator that creates reproducible test cases:

```bash
python -m benchmarks.prepare_datasets synthetic --output cases/synthetic
```

This generates 5 reference cases covering multiple sizes and obstacle patterns:
- `sparse_128.npz` (128×128, 10% obstacles, uniform speed)
- `medium_256.npz` (256×256, 20% obstacles, uniform speed)
- `gradient_256.npz` (256×256, 20% obstacles, gradient speed field)
- `complex_512.npz` (512×512, 30% obstacles, random speeds)
- `maze_511.npz` (511×511, perfect maze, uniform speed)

## Official Benchmark Datasets

### MovingAI Pathfinding Benchmarks

**Source**: http://movingai.com/benchmarks/

The MovingAI benchmarks are the de-facto standard for pathfinding algorithm evaluation.

#### Download Steps

1. Visit http://movingai.com/benchmarks/grids.html
2. Download desired map packs (recommended: mazes, rooms, random)
3. Extract to `raw/movingai/` directory

#### Convert to Solver Format

```bash
# Example: Convert maze512-32-0 with first 50 scenarios
python -m benchmarks.prepare_datasets movingai \
    --map raw/movingai/mazes/maze512-32-0.map \
    --scen raw/movingai/mazes/maze512-32-0.map.scen \
    --output cases/movingai/maze512-32-0 \
    --limit 50
```

**Map format**: ASCII grid where:
- `.` = passable terrain
- `@` or `O` = obstacles
- `T` = trees (passable at reduced speed)

**Scenario format**: TSV file with columns:
```
bucket  map  width  height  startX  startY  goalX  goalY  optimal_length
```

#### Recommended Maps

| Category | Size | Difficulty | Map Name |
|----------|------|------------|----------|
| Mazes | 512×512 | Easy | `maze512-1-*.map` |
| Mazes | 512×512 | Hard | `maze512-32-*.map` |
| Rooms | 256×256 | Medium | `room-256-*.map` |
| Random | 512×512 | High | `random-512-*.map` |

### CMAP Maze Benchmark

**Source**: http://mazebenchmark.github.io

CMAP provides generated mazes with controlled connectivity parameters.

#### Download Steps

1. Clone the CMAP repository:
   ```bash
   git clone https://github.com/jvkersch/maze-benchmark.git
   cd maze-benchmark
   ```

2. Generate mazes with the official generator:
   ```bash
   python maze_generator.py --size 256 --connectivity 0.3 --output raw/cmap/maze_256_c03.npy
   python maze_generator.py --size 512 --connectivity 0.5 --output raw/cmap/maze_512_c05.npy
   ```

#### Connectivity Parameter

- `0.0` = minimum connectivity (sparse, tree-like maze)
- `0.5` = medium connectivity (multiple paths)
- `1.0` = maximum connectivity (many alternate routes)

#### Convert to Solver Format

```bash
python -m benchmarks.prepare_datasets cmap \
    --input raw/cmap/maze_256_c03.npy \
    --output cases/cmap/maze_256_c03.npz \
    --connectivity 0.3
```

Optionally specify source and target:
```bash
python -m benchmarks.prepare_datasets cmap \
    --input raw/cmap/maze_512_c05.npy \
    --output cases/cmap/maze_512_c05.npz \
    --connectivity 0.5 \
    --source "10,10" \
    --target "500,500"
```

## Dataset Format

All datasets are stored as `.npz` archives containing:

```python
{
    'obstacles': np.ndarray,  # float32 (H×W), 1.0 = blocked, 0.0 = free
    'speeds': np.ndarray,     # float32 (H×W), propagation speed per cell
    'source': np.ndarray,     # int32 (2,), [x, y] coordinates
    'target': np.ndarray,     # int32 (2,), [x, y] coordinates
    'metadata': str,          # JSON string with provenance information
}
```

### Loading a Dataset

```python
from benchmarks.io_utils import load_case

case = load_case('cases/synthetic/maze_511.npz')
print(f"Size: {case.obstacles.shape}")
print(f"Source: {case.source}, Target: {case.target}")
print(f"Metadata: {case.metadata}")
```

## Directory Structure

Recommended organization:

```
Quantum_Processor_Simulator/
├── raw/                      # Original downloaded datasets
│   ├── movingai/
│   │   ├── mazes/
│   │   ├── rooms/
│   │   └── random/
│   └── cmap/
│       └── *.npy
├── cases/                    # Converted benchmark cases
│   ├── synthetic/
│   │   └── *.npz
│   ├── movingai/
│   │   └── *.npz
│   └── cmap/
│       └── *.npz
└── results/                  # Benchmark results
    └── *.csv
```

## Creating Custom Datasets

### From NumPy Arrays

```python
import numpy as np
from benchmarks.io_utils import BenchmarkCase, save_case

# Create custom obstacle field
obstacles = np.random.random((128, 128)) < 0.2  # 20% obstacles
obstacles = obstacles.astype(np.float32)

# Create speed field (1.0 = normal speed)
speeds = np.ones((128, 128), dtype=np.float32)
speeds[obstacles > 0.5] = 0.0  # blocked cells have zero speed

# Define endpoints
source = (10, 10)
target = (110, 110)

# Package as case
case = BenchmarkCase(
    obstacles=obstacles,
    speeds=speeds,
    source=source,
    target=target,
    metadata={'dataset': 'Custom', 'description': 'My test case'}
)

# Save
save_case('cases/custom/my_case.npz', case)
```

### From Image

```python
from PIL import Image
import numpy as np
from benchmarks.io_utils import BenchmarkCase, save_case

# Load black and white image
img = Image.open('my_map.png').convert('L')
arr = np.array(img, dtype=np.float32) / 255.0

# Threshold to binary (dark = obstacle)
obstacles = (arr < 0.5).astype(np.float32)

# Uniform speeds
speeds = np.ones_like(obstacles)

case = BenchmarkCase(
    obstacles=obstacles,
    speeds=speeds,
    source=(5, 5),
    target=(arr.shape[1] - 5, arr.shape[0] - 5),
    metadata={'dataset': 'Image', 'source': 'my_map.png'}
)

save_case('cases/custom/from_image.npz', case)
```

## Validation

After preparing datasets, verify them with:

```python
python -c "from benchmarks.io_utils import load_case; c = load_case('cases/synthetic/maze_511.npz'); print(f'Valid: {c.obstacles.shape}, Source: {c.source}, Target: {c.target}')"
```

## Next Steps

Once datasets are prepared, proceed to benchmarking:

```bash
python -m benchmarks.run_suite --cases cases/synthetic --output results/synthetic.csv --iterations 300
```

See `BENCHMARK_GUIDE.md` for complete benchmarking instructions.

