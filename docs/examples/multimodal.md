# Multimodal Example

This example demonstrates how to use Inferneo for multimodal inference (text + image).

## Python Client Example

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Load image
with open("dog.jpg", "rb") as f:
    image_bytes = f.read()

# Multimodal prompt
prompt = "Describe the image and its mood."

response = client.multimodal.create(
    model="openai/blip-2",
    prompt=prompt,
    image=image_bytes
)

print("Response:", response.choices[0].text)
```

## REST API Example

```bash
curl -X POST http://localhost:8000/v1/multimodal \
  -H "Content-Type: application/json" \
  --data-binary @dog.jpg \
  -H "Content-Type: application/octet-stream" \
  -H "Prompt: Describe the image and its mood." \
  -H "Model: openai/blip-2"
```

## Use Cases
- Visual question answering
- Image-text retrieval
- Multimodal chatbots

## Next Steps
- **[Vision Models](vision-models.md)**
- **[Text Generation](text-generation.md)** 