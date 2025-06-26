# Text Generation Example

## Overview

Example for text generation with Inferneo.

## Basic Usage

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

- [Embeddings Example](embeddings.md)
- [Vision Models Example](vision-models.md)
- [Multimodal Example](multimodal.md) 