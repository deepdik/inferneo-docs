# Getting Started

Welcome to Inferneo! This guide will help you get up and running with Inferneo in just a few minutes.

## What is Inferneo?

Inferneo is a high-performance inference server that makes it easy to serve large language models and other AI models in production. It's designed to be:

- **Fast**: Optimized for maximum throughput and minimum latency
- **Scalable**: Support for distributed inference and horizontal scaling
- **Easy to use**: Simple APIs and comprehensive tooling
- **Production ready**: Built-in monitoring, health checks, and enterprise features

## Quick Start

### 1. Installation

Install Inferneo using pip:

```bash
pip install inferneo
```

Or using conda:

```bash
conda install -c conda-forge inferneo
```

### 2. Start the Server

Launch Inferneo with a Hugging Face model:

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### 3. Make Your First Request

Use the Python client:

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
```

Or use the REST API:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for model storage

### Recommended Requirements
- **GPU**: NVIDIA A100, H100, or RTX 4090
- **RAM**: 64GB+ system memory
- **Storage**: NVMe SSD with 500GB+ free space
- **Network**: 10Gbps+ for distributed inference

## Supported Platforms

- **Operating Systems**: Linux (Ubuntu 20.04+), Windows 10+, macOS 12+
- **Cloud Platforms**: AWS, Google Cloud, Azure, DigitalOcean
- **Container Platforms**: Docker, Kubernetes, Docker Compose
- **Architectures**: x86_64, ARM64 (Apple Silicon)

## Next Steps

Now that you have Inferneo running, explore these resources:

- **[Installation Guide](installation.md)** - Detailed installation instructions
- **[User Guide](user-guide/)** - Comprehensive usage documentation
- **[Examples](examples/)** - Code examples and tutorials
- **[API Reference](api-reference/)** - Complete API documentation

## Getting Help

If you run into any issues:

1. Check the **[FAQ](community/faq.md)** for common solutions
2. Search existing **[GitHub Issues](https://github.com/inferneo/inferneo/issues)**
3. Join our **[Discord Community](https://discord.gg/inferneo)**
4. Create a new issue with detailed information about your problem

## What's Next?

Ready to dive deeper? Here are some recommended next steps:

- **Learn about model loading**: [Model Loading Guide](user-guide/model-loading.md)
- **Explore batching**: [Batching Guide](user-guide/batching.md)
- **Set up distributed inference**: [Distributed Inference](user-guide/distributed-inference.md)
- **Deploy to production**: [Production Deployment](developer-guide/production-deployment.md) 