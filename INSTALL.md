# Installation Guide

This guide provides detailed installation instructions for the Optical Neuromorphic Eikonal Solver on different platforms.

## Quick Install (All Platforms)

```bash
pip install moderngl glfw numpy
```

Then run:

```bash
python quantum_eikonal_solver.py
```

## Platform-Specific Instructions

### Windows

#### Prerequisites

1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **GPU Drivers**: Ensure you have up-to-date drivers
   - **NVIDIA**: [GeForce Drivers](https://www.nvidia.com/Download/index.aspx)
   - **AMD**: [Radeon Software](https://www.amd.com/en/support)
   - **Intel**: [Graphics Drivers](https://www.intel.com/content/www/us/en/download-center/home.html)

#### Installation

```powershell
# Open PowerShell or Command Prompt

# Install dependencies
pip install numpy moderngl glfw

# Verify installation
python -c "import moderngl; print(moderngl.create_standalone_context())"
```

#### Troubleshooting Windows

**Issue**: `ModuleNotFoundError: No module named 'moderngl'`
```powershell
python -m pip install --upgrade pip
pip install moderngl --no-cache-dir
```

**Issue**: GLFW window doesn't open
- Install Microsoft Visual C++ Redistributable: [Download](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Update GPU drivers

**Issue**: OpenGL version error
```
OpenGL 4.3 required, but X.X found
```
- Update GPU drivers to the latest version
- Check if your GPU supports OpenGL 4.3+
- Try: `python quantum_eikonal_solver.py` with `--gl-version 3.3` (if fallback supported)

---

### Linux (Ubuntu/Debian)

#### Prerequisites

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install python3 python3-pip python3-dev
sudo apt install libglfw3 libglfw3-dev
sudo apt install libgl1-mesa-dev libglu1-mesa-dev

# For NVIDIA GPUs
sudo apt install nvidia-driver-XXX  # Replace XXX with latest version number

# For AMD GPUs  
sudo apt install mesa-vulkan-drivers mesa-va-drivers
```

#### Installation

```bash
# Install Python dependencies
pip3 install numpy moderngl glfw

# Verify installation
python3 -c "import moderngl; ctx = moderngl.create_standalone_context(); print(f'OpenGL {ctx.version_code}')"
```

#### Troubleshooting Linux

**Issue**: `libGL error: failed to load driver`
```bash
# Install mesa drivers
sudo apt install mesa-utils

# Test OpenGL
glxinfo | grep "OpenGL version"
```

**Issue**: Permission denied for GPU access
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and back in
```

**Issue**: Headless server (no display)
```bash
# Install Xvfb (virtual framebuffer)
sudo apt install xvfb

# Run with virtual display
xvfb-run -a python3 quantum_eikonal_solver.py
```

---

### macOS

#### Prerequisites

1. **Python 3.8+**: Install via [Homebrew](https://brew.sh/)
   ```bash
   brew install python@3.10
   ```

2. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

#### Installation

```bash
# Install dependencies
pip3 install numpy moderngl glfw

# Verify installation
python3 -c "import moderngl; print('ModernGL installed successfully')"
```

#### macOS Notes

- **M1/M2 Macs**: Use native ARM Python (via Homebrew)
- **OpenGL 4.1 Max**: macOS caps OpenGL at 4.1 (solver may need adjustments)
- **Metal Backend**: Consider using MoltenVK for better performance

#### Troubleshooting macOS

**Issue**: `dyld: Library not loaded`
```bash
# Reinstall dependencies
brew reinstall glfw

# Update pip and reinstall Python packages
pip3 install --upgrade pip
pip3 install --force-reinstall moderngl glfw
```

**Issue**: Performance issues on M1/M2
- Ensure you're using native ARM Python, not x86_64 under Rosetta
- Check architecture: `python3 -c "import platform; print(platform.machine())"`
- Should print `arm64`, not `x86_64`

---

## Docker Installation

For a reproducible environment:

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install numpy moderngl glfw

# Copy solver code
COPY . /app
WORKDIR /app

# Run with virtual framebuffer
CMD ["xvfb-run", "-a", "python", "quantum_eikonal_solver.py"]
```

Build and run:

```bash
docker build -t eikonal-solver .
docker run --rm --gpus all eikonal-solver
```

---

## Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Create conda environment
conda create -n eikonal python=3.10

# Activate
conda activate eikonal

# Install dependencies
pip install moderngl glfw numpy
```

---

## Verification

After installation, verify everything works:

```bash
# Test import
python -c "import moderngl, glfw, numpy; print('✓ All imports successful')"

# Test OpenGL context
python -c "import moderngl; ctx = moderngl.create_standalone_context(); print(f'✓ OpenGL {ctx.version_code}')"

# Run quick test
python quantum_eikonal_solver.py --test
```

Expected output:
```
✓ All imports successful
✓ OpenGL 430
✓ Solver initialized
✓ Test passed
```

---

## Common Issues

### Issue: "No module named 'moderngl'"

**Solution**:
```bash
pip install --upgrade pip
pip install moderngl
```

### Issue: "Failed to create OpenGL context"

**Causes**:
1. Outdated GPU drivers → Update drivers
2. GPU doesn't support OpenGL 4.3 → Check GPU specs
3. Running over SSH without X11 forwarding → Use Xvfb or EGL

**Solution for headless servers**:
```bash
# Install EGL
pip install moderngl[egl]

# Use EGL backend
export MODERNGL_PLATFORM=egl
python quantum_eikonal_solver.py
```

### Issue: "GLFW error: X11: The DISPLAY environment variable is missing"

**Solution (Linux headless)**:
```bash
# Use Xvfb
xvfb-run -a python quantum_eikonal_solver.py

# Or set display
export DISPLAY=:0
python quantum_eikonal_solver.py
```

### Issue: Performance is slow

**Possible causes**:
1. Running on integrated GPU instead of dedicated GPU
2. Power saving mode enabled
3. Thermal throttling

**Solutions**:
```bash
# Force discrete GPU (NVIDIA)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python quantum_eikonal_solver.py

# Check GPU usage
nvidia-smi  # For NVIDIA
radeontop   # For AMD
```

---

## Hardware Requirements

### Minimum

- **GPU**: OpenGL 3.3 capable (may have reduced functionality)
- **RAM**: 4 GB
- **Storage**: 100 MB
- **Python**: 3.8+

### Recommended

- **GPU**: OpenGL 4.3+ capable (NVIDIA GTX 600+, AMD GCN+, Intel HD 4000+)
- **RAM**: 8 GB
- **Storage**: 500 MB (for datasets)
- **Python**: 3.10+

### Optimal

- **GPU**: Dedicated GPU with 2+ GB VRAM (NVIDIA RTX, AMD RX 5000+)
- **RAM**: 16 GB
- **Storage**: 1 GB
- **Python**: 3.10+

---

## GPU Compatibility

| GPU Family | OpenGL Version | Compatibility |
|------------|----------------|---------------|
| NVIDIA GTX 900+ | 4.5+ | ✅ Full support |
| NVIDIA GTX 600-900 | 4.3-4.4 | ✅ Full support |
| NVIDIA GTX 400-500 | 4.1-4.2 | ⚠️ Limited (may work with fallback) |
| AMD RX 5000+ | 4.6 | ✅ Full support |
| AMD RX 400-500 | 4.5 | ✅ Full support |
| AMD GCN (2012+) | 4.3+ | ✅ Full support |
| Intel HD 4000+ | 4.0-4.6 | ✅ Usually works |
| Intel integrated | 3.3+ | ⚠️ May work (slower) |
| Apple M1/M2 | 4.1 | ⚠️ Limited by macOS |

---

## Next Steps

After successful installation:

1. **Run the interactive demo**: `python quantum_eikonal_solver.py`
2. **Generate datasets**: `python -m benchmarks.prepare_datasets synthetic --output cases/synthetic`
3. **Run benchmarks**: `python benchmarks/generate_results.py --cases cases/synthetic --output results/bench.csv`
4. **Read the documentation**: Check [README.md](README.md) and [PAPER.md](PAPER.md)

---

## Getting Help

If you encounter issues:

1. Check this guide's **Troubleshooting** sections
2. Search [GitHub Issues](https://github.com/[username]/[repo]/issues)
3. Open a new issue with:
   - Your OS and version
   - Python version (`python --version`)
   - GPU model and driver version
   - Complete error message
   - Steps to reproduce

---

**Happy pathfinding! 🚀**

