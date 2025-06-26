# Streaming

## Overview

Guide for implementing real-time response streaming.

## Basic Streaming

### Enable Streaming

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", 
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### WebSocket Streaming

```python
import websockets
import json

async def stream_response():
    async with websockets.connect('ws://localhost:8000/ws') as websocket:
        await websocket.send(json.dumps({
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }))
        
        async for message in websocket:
            data = json.loads(message)
            print(data['content'], end='')
```

## Configuration

### Stream Settings

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --enable-streaming
```

## Next Steps

- [Batching](batching.md)
- [Quantization](quantization.md)
- [Distributed Inference](distributed-inference.md)
- Performance Tuning guide 