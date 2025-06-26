# Installation

## Overview

Installation guide for Inferneo.

## System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- NVIDIA GPU (recommended)

## Installation Methods

### Using pip

```bash
pip install inferneo
```

### Using conda

```bash
conda install -c conda-forge inferneo
```

### Using Docker

```bash
docker pull inferneo/inferneo
```

## GPU Support

### CUDA Installation

Install CUDA toolkit for GPU acceleration.

### PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

Test your installation:

```bash
python -c "import inferneo; print('Inferneo installed successfully!')"
```

## Next Steps

- [Getting Started](getting-started.md)
- [User Guide](user-guide/)
- [Examples](examples/) 