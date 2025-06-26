# Python Client API Reference

This page documents the Python client for Inferneo.

## Installation

```bash
pip install inferneo
```

## Initialization

```python
from inferneo import InferneoClient
client = InferneoClient("http://localhost:8000")
```

## Methods

### Completions

```python
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Explain AI",
    max_tokens=100
)
print(response.choices[0].text)
```

### Chat Completions

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=messages,
    max_tokens=150
)
print(response.choices[0].message.content)
```

### Embeddings

```python
response = client.embeddings.create(
    model="meta-llama/Llama-2-7b-embeddings",
    input=["What is AI?", "Explain deep learning."]
)
print(response.data[0]["embedding"])
```

### Vision

```python
with open("cat.jpg", "rb") as f:
    image_bytes = f.read()
response = client.vision.create(
    model="openai/clip-vit-base-patch16",
    image=image_bytes
)
print(response.choices[0].text)
```

### Multimodal

```python
with open("dog.jpg", "rb") as f:
    image_bytes = f.read()
response = client.multimodal.create(
    model="openai/blip-2",
    prompt="Describe the image.",
    image=image_bytes
)
print(response.choices[0].text)
```

### List Models

```python
models = client.models.list()
print([model.id for model in models.data])
```

## Error Handling

- All methods raise exceptions on network or API errors.

## Next Steps
- **[REST API](rest-api.md)**
- **[WebSocket API](websocket-api.md)** 