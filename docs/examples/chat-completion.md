# Chat Completion Example

This example demonstrates how to use Inferneo for chat-based completions.

## Python Client Example

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you give me an example?"}
]

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=messages,
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## REST API Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 150
  }'
```

## Streaming Example

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=messages,
    max_tokens=150,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Next Steps
- **[Streaming](../user-guide/streaming.md)**
- **[Batching](../user-guide/batching.md)** 