# Getting Started

## Overview

Quick setup guide for Inferneo.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional)
- Docker (optional)

## Installation

### Using pip

```bash
pip install inferneo
```

### Using Docker

```bash
docker pull inferneo/inferneo
```

## Quick Start

### Start the server

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### Make your first request

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", 
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

## Next Steps

- **Explore batching**: [Batching Guide](user-guide/batching.md)
- **Set up distributed inference**: [Distributed Inference](user-guide/distributed-inference.md)
- **Deploy to production**: Production deployment guide 