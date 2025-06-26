# Quickstart

## Overview

Quick start guide for using Inferneo.

## Basic Usage

### Start the server

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### Make a request

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

- [Batching](batching.md)
- [Streaming](streaming.md)
- [Distributed Inference](distributed-inference.md) 