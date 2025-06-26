# Batching

## Overview

Guide for optimizing performance with request batching.

## Basic Batching

### Enable Batching

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --max-batch-size 32
```

### Batch Requests

```python
import requests

# Multiple requests in batch
requests_data = [
    {"model": "meta-llama/Llama-2-7b-chat-hf", "messages": [{"role": "user", "content": "Hello 1"}]},
    {"model": "meta-llama/Llama-2-7b-chat-hf", "messages": [{"role": "user", "content": "Hello 2"}]},
    {"model": "meta-llama/Llama-2-7b-chat-hf", "messages": [{"role": "user", "content": "Hello 3"}]}
]

response = requests.post("http://localhost:8000/v1/chat/completions", json=requests_data)
```

## Configuration

### Batch Size Tuning

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --max-batch-size 16
```

### Concurrent Requests

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --max-concurrent-requests 100
```

## Next Steps

- [Streaming](streaming.md)
- [Quantization](quantization.md)
- [Distributed Inference](distributed-inference.md)
- Performance Tuning guide 