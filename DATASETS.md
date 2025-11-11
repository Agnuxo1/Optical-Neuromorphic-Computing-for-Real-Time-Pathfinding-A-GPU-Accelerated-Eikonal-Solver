# Dataset Documentation

This document describes all benchmark datasets included with the Optical Neuromorphic Eikonal Solver.

## Quick Reference

| Dataset | Grid Size | Cells | Obstacles | Speed Field | File Size | Difficulty |
|---------|-----------|-------|-----------|-------------|-----------|------------|
| sparse_128 | 128×128 | 16,384 | 10% | Uniform | ~260 KB | Easy |
| medium_256 | 256×256 | 65,536 | 20% | Uniform | ~1 MB | Medium |
| gradient_256 | 256×256 | 65,536 | 20% | Gradient | ~1 MB | Medium |
| maze_511 | 511×511 | 261,121 | 30% (walls) | Uniform | ~4 MB | Hard |
| complex_512 | 512×512 | 262,144 | 30% | Random | ~4 MB | Hard |

**Total Size**: ~10 MB (uncompressed)

---

## Dataset Format

All datasets are stored as NumPy `.npz` archives containing:

```python
{
    'obstacles': np.ndarray,  # shape (H, W), dtype float32
                             # 1.0 = blocked cell
                             # 0.0 = free cell
    
    'speeds': np.ndarray,     # shape (H, W), dtype float32
                             # Propagation speed per cell
                             # 1.0 = normal speed
                             # >1.0 = faster (highways)
                             # <1.0 = slower (rough terrain)
    
    'source': np.ndarray,     # shape (2,), dtype int32
                             # [x, y] coordinates of start point
    
    'target': np.ndarray,     # shape (2,), dtype int32
                             # [x, y] coordinates of goal point
    
    'metadata': str,          # JSON string with provenance info
                             # {
                             #   "dataset": "Synthetic",
                             #   "name": "sparse_128",
                             #   "size": 128,
                             #   "obstacle_density": 0.1,
                             #   "speed_mode": "uniform",
                             #   "seed": 1
                             # }
}
```

---

## Synthetic Suite (Included)

The synthetic suite provides controlled test cases for benchmarking and validation.

### sparse_128.npz

**Purpose**: Fast baseline test case  
**Grid Size**: 128×128 (16,384 cells)  
**Obstacles**: 10% random placement  
**Speed Field**: Uniform (1.0 everywhere)  
**Source**: (1, 1)  
**Target**: (126, 126)  
**Seed**: 1

**Characteristics**:
- Low complexity
- Few obstacles
- Clear paths
- Fast to solve (~2ms GPU)
- Good for quick validation

**Use Cases**:
- Quick functionality testing
- Baseline performance measurement
- Algorithm validation

---

### medium_256.npz

**Purpose**: Standard benchmark case  
**Grid Size**: 256×256 (65,536 cells)  
**Obstacles**: 20% random placement  
**Speed Field**: Uniform (1.0 everywhere)  
**Source**: (1, 1)  
**Target**: (254, 254)  
**Seed**: 3

**Characteristics**:
- Medium complexity
- Moderate obstacle density
- Multiple possible paths
- Representative of typical scenarios
- ~3ms GPU solve time

**Use Cases**:
- Standard benchmark
- Algorithm comparison
- Performance profiling

---

### gradient_256.npz

**Purpose**: Variable cost testing  
**Grid Size**: 256×256 (65,536 cells)  
**Obstacles**: 20% random placement  
**Speed Field**: Radial gradient (0.7-1.5)  
**Source**: (1, 1)  
**Target**: (254, 254)  
**Seed**: 5

**Speed Distribution**:
- Center: 1.5 (faster)
- Edges: 0.7 (slower)
- Smooth gradient transition

**Characteristics**:
- Tests heterogeneous cost handling
- Non-uniform optimal paths
- Realistic terrain modeling
- ~3ms GPU solve time

**Use Cases**:
- Variable cost validation
- Realistic scenario testing
- Path quality evaluation

---

### maze_511.npz

**Purpose**: Perfect maze testing  
**Grid Size**: 511×511 (261,121 cells)  
**Obstacles**: ~30% (maze walls)  
**Speed Field**: Uniform (1.0 in corridors)  
**Source**: Near (1, 1)  
**Target**: Near (509, 509)  
**Seed**: 11

**Generation**: Depth-first search recursive backtracker

**Characteristics**:
- Single solution path (no shortcuts)
- Long winding corridors
- Tests convergence in constrained spaces
- ~4ms GPU solve time
- Worst-case scenario for some algorithms

**Use Cases**:
- Convergence testing
- Worst-case performance
- Maze-specific algorithms

---

### complex_512.npz

**Purpose**: Large-scale complex testing  
**Grid Size**: 512×512 (262,144 cells)  
**Obstacles**: 30% random placement  
**Speed Field**: Random (0.4-2.0)  
**Source**: (1, 1)  
**Target**: (510, 510)  
**Seed**: 7

**Speed Distribution**:
- Mean: 1.2
- Std Dev: 0.5
- Range: [0.4, 2.0]

**Characteristics**:
- Large grid (262K cells)
- High obstacle density
- Highly variable costs
- Multiple path options with different costs
- ~4ms GPU solve time
- Most realistic scenario

**Use Cases**:
- Scalability testing
- Real-world scenario simulation
- Performance limits
- Large-scale validation

---

## Dataset Statistics

### Obstacle Density Distribution

```
sparse_128:    ▓░░░░░░░░░  10%
medium_256:    ▓▓░░░░░░░░  20%
gradient_256:  ▓▓░░░░░░░░  20%
maze_511:      ▓▓▓░░░░░░░  30% (walls)
complex_512:   ▓▓▓░░░░░░░  30%
```

### Speed Field Complexity

```
Dataset         Min Speed  Max Speed  Variance  Type
──────────────────────────────────────────────────────
sparse_128      1.0        1.0        0.000     Uniform
medium_256      1.0        1.0        0.000     Uniform
gradient_256    0.7        1.5        0.045     Gradient
maze_511        1.0        1.0        0.000     Uniform
complex_512     0.4        2.0        0.250     Random
```

### Path Characteristics

| Dataset | Straight-line Dist | Optimal Path | Path Ratio | Complexity |
|---------|-------------------|--------------|------------|------------|
| sparse_128 | 177.4 | ~190 | 1.07 | Low |
| medium_256 | 358.7 | ~400 | 1.11 | Medium |
| gradient_256 | 358.7 | ~420 | 1.17 | Medium |
| maze_511 | 719.1 | ~1200 | 1.67 | High |
| complex_512 | 720.0 | ~850 | 1.18 | High |

---

## Loading Datasets

### Python

```python
import numpy as np
from benchmarks.io_utils import load_case

# Load a dataset
case = load_case('cases/synthetic/maze_511.npz')

# Access fields
obstacles = case.obstacles  # (H, W) array
speeds = case.speeds        # (H, W) array
source = case.source        # (x, y) tuple
target = case.target        # (x, y) tuple
metadata = case.metadata    # dict

# Inspect
print(f"Grid size: {obstacles.shape}")
print(f"Free cells: {np.sum(obstacles < 0.5)}")
print(f"Source: {source} → Target: {target}")
```

### Raw NumPy

```python
import numpy as np
import json

# Load directly
data = np.load('cases/synthetic/complex_512.npz', allow_pickle=True)

obstacles = data['obstacles']
speeds = data['speeds']
source = tuple(data['source'])
target = tuple(data['target'])
metadata = json.loads(str(data['metadata']))
```

---

## Generating Custom Datasets

### From Scratch

```python
import numpy as np
from benchmarks.io_utils import BenchmarkCase, save_case

# Create custom grid
size = 256
obstacles = np.random.random((size, size)) < 0.15  # 15% obstacles
speeds = np.ones((size, size), dtype=np.float32)

# Set source and target
source = (10, 10)
target = (size-10, size-10)

# Package
case = BenchmarkCase(
    obstacles=obstacles.astype(np.float32),
    speeds=speeds,
    source=source,
    target=target,
    metadata={
        'dataset': 'Custom',
        'description': 'My test case',
        'created': '2025-11-11'
    }
)

# Save
save_case('cases/custom/my_case.npz', case)
```

### From Image

```python
from PIL import Image
import numpy as np
from benchmarks.io_utils import BenchmarkCase, save_case

# Load map image (black = obstacle, white = free)
img = Image.open('my_map.png').convert('L')
arr = np.array(img, dtype=np.float32) / 255.0

# Convert to obstacles (invert: dark = blocked)
obstacles = 1.0 - arr

# Uniform speeds
speeds = np.ones_like(obstacles)

case = BenchmarkCase(
    obstacles=obstacles,
    speeds=speeds,
    source=(5, 5),
    target=(arr.shape[1]-5, arr.shape[0]-5),
    metadata={'dataset': 'Image', 'source': 'my_map.png'}
)

save_case('cases/custom/from_image.npz', case)
```

---

## External Datasets

### MovingAI Benchmarks

**Source**: http://movingai.com/benchmarks/

**Download**:
```bash
# Download maps (example)
wget http://movingai.com/benchmarks/grids/mazes.zip
unzip mazes.zip -d raw/movingai/
```

**Convert**:
```bash
python -m benchmarks.prepare_datasets movingai \
    --map raw/movingai/mazes/maze512-32-0.map \
    --scen raw/movingai/mazes/maze512-32-0.map.scen \
    --output cases/movingai/ \
    --limit 50
```

### CMAP Maze Benchmark

**Source**: http://mazebenchmark.github.io

**Generate**:
```bash
# Clone CMAP repo
git clone https://github.com/jvkersch/maze-benchmark.git

# Generate maze
python maze-benchmark/maze_generator.py \
    --size 256 \
    --connectivity 0.3 \
    --output raw/cmap/maze_256_c03.npy
```

**Convert**:
```bash
python -m benchmarks.prepare_datasets cmap \
    --input raw/cmap/maze_256_c03.npy \
    --output cases/cmap/maze_256_c03.npz \
    --connectivity 0.3
```

---

## Dataset Validation

Validate datasets before benchmarking:

```python
from benchmarks.io_utils import load_case
import numpy as np

def validate_case(path):
    """Validate a benchmark case."""
    case = load_case(path)
    
    # Check dimensions
    assert case.obstacles.shape == case.speeds.shape, "Shape mismatch"
    
    # Check data types
    assert case.obstacles.dtype == np.float32, "Wrong obstacles dtype"
    assert case.speeds.dtype == np.float32, "Wrong speeds dtype"
    
    # Check ranges
    assert np.all((case.obstacles >= 0) & (case.obstacles <= 1)), "Obstacles out of range"
    assert np.all(case.speeds > 0), "Invalid speeds"
    
    # Check endpoints
    sx, sy = case.source
    tx, ty = case.target
    H, W = case.obstacles.shape
    assert 0 <= sx < W and 0 <= sy < H, "Source out of bounds"
    assert 0 <= tx < W and 0 <= ty < H, "Target out of bounds"
    assert case.obstacles[sy, sx] < 0.5, "Source is blocked"
    assert case.obstacles[ty, tx] < 0.5, "Target is blocked"
    
    print(f"✓ {path} is valid")

# Validate all
import glob
for path in glob.glob('cases/synthetic/*.npz'):
    validate_case(path)
```

---

## Citation

If you use these datasets in research, please cite:

```bibtex
@misc{optical_neuromorphic_datasets_2025,
  title={Optical Neuromorphic Eikonal Solver Benchmark Datasets},
  author={[Authors]},
  year={2025},
  publisher={Zenodo/OpenML/Kaggle},
  doi={[DOI]},
  url={[URL]}
}
```

---

## License

All datasets are released under **CC BY 4.0** (Creative Commons Attribution).

You are free to:
- Share: copy and redistribute
- Adapt: remix, transform, and build upon

Under the terms:
- Attribution: credit must be given

---

## Contact

Questions about datasets:
- Email: [contact]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

---

**Dataset Version**: 1.0  
**Last Updated**: November 2025

