# WebSocket API Reference

This page documents the WebSocket API endpoints provided by Inferneo for real-time streaming.

## Base URL

```
ws://localhost:8000
```

## Endpoints

### Chat Completion Streaming

```
ws://localhost:8000/v1/chat/completions
```
- Send a JSON message with:
  - `model`: Model ID
  - `messages`: List of chat messages
  - `max_tokens`: Maximum tokens
  - `stream`: true
- Receive streaming responses as JSON objects with incremental content.

### Text Completion Streaming

```
ws://localhost:8000/v1/completions
```
- Send a JSON message with:
  - `model`: Model ID
  - `prompt`: Prompt string
  - `max_tokens`: Maximum tokens
  - `stream`: true
- Receive streaming responses as JSON objects with incremental content.

## Example Usage

```python
import asyncio
import websockets
import json

async def stream_chat():
    uri = "ws://localhost:8000/v1/chat/completions"
    async with websockets.connect(uri) as websocket:
        message = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
            "max_tokens": 100
        }
        await websocket.send(json.dumps(message))
        async for response in websocket:
            data = json.loads(response)
            print(data)

asyncio.run(stream_chat())
```

## Error Handling
- Standard WebSocket close codes
- JSON error messages

## Next Steps
- **[REST API](rest-api.md)**
- **[Configuration](configuration.md)** 