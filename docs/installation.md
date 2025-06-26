# Installation

This guide covers all the different ways to install Inferneo on various platforms and environments.

## Prerequisites

Before installing Inferneo, make sure you have:

- **Python 3.8+** installed
- **CUDA 11.8+** (for GPU acceleration)
- **NVIDIA GPU drivers** (for GPU acceleration)
- **Git** (for development installation)

## Installation Methods

### 1. PyPI Installation (Recommended)

The easiest way to install Inferneo is through PyPI:

```bash
pip install inferneo
```

For GPU support with CUDA:

```bash
pip install inferneo[gpu]
```

### 2. Conda Installation

Using conda-forge:

```bash
conda install -c conda-forge inferneo
```

### 3. Docker Installation

Pull the official Docker image:

```bash
docker pull inferneo/inferneo:latest
```

Run the container:

```bash
docker run -it --gpus all -p 8000:8000 inferneo/inferneo:latest \
  inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### 4. Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/inferneo/inferneo.git
cd inferneo
pip install -e .
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y python3-pip python3-dev build-essential

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

# Install Inferneo
pip install inferneo[gpu]
```

### CentOS/RHEL

```bash
# Install EPEL repository
sudo yum install -y epel-release

# Install system dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-pip python3-devel

# Install CUDA (if not already installed)
sudo yum install -y cuda-toolkit

# Install Inferneo
pip install inferneo[gpu]
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install Inferneo (CPU only on macOS)
pip install inferneo
```

### Windows

```bash
# Install Python from python.org or Microsoft Store
# Install Visual Studio Build Tools
# Install CUDA Toolkit from NVIDIA

# Install Inferneo
pip install inferneo[gpu]
```

## Cloud Platform Installation

### AWS EC2

```bash
# Launch an instance with NVIDIA GPU
# Connect to your instance

# Install CUDA and Inferneo
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

pip install inferneo[gpu]
```

### Google Cloud Platform

```bash
# Create a VM with NVIDIA GPU
# Connect to your VM

# Install CUDA and Inferneo
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

pip install inferneo[gpu]
```

### Azure

```bash
# Create a VM with NVIDIA GPU
# Connect to your VM

# Install CUDA and Inferneo
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

pip install inferneo[gpu]
```

## Container Installation

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  inferneo:
    image: inferneo/inferneo:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["inferneo", "serve", "--model", "meta-llama/Llama-2-7b-chat-hf"]
```

Run with:

```bash
docker-compose up -d
```

### Kubernetes

Create a deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inferneo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inferneo
  template:
    metadata:
      labels:
        app: inferneo
    spec:
      containers:
      - name: inferneo
        image: inferneo/inferneo:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        command: ["inferneo", "serve", "--model", "meta-llama/Llama-2-7b-chat-hf"]
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Verification

After installation, verify that Inferneo is working correctly:

```bash
# Check installation
python -c "import inferneo; print(inferneo.__version__)"

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Start the server
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --port 8000
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Make sure CUDA is properly installed and in your PATH
2. **GPU memory errors**: Reduce batch size or use a smaller model
3. **Import errors**: Ensure all dependencies are installed correctly
4. **Permission errors**: Use `sudo` or install in a virtual environment

### Getting Help

If you encounter issues:

1. Check the **[FAQ](community/faq.md)**
2. Search **[GitHub Issues](https://github.com/inferneo/inferneo/issues)**
3. Join our **[Discord Community](https://discord.gg/inferneo)**
4. Create a new issue with detailed error information

## Next Steps

Now that you have Inferneo installed:

- **[Getting Started](getting-started.md)** - Quick start guide
- **[User Guide](user-guide/)** - Comprehensive usage documentation
- **[Examples](examples/)** - Code examples and tutorials 