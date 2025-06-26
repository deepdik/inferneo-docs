# Vision Models Example

This example demonstrates how to use Inferneo for vision model inference (e.g., image captioning, classification).

## Python Client Example

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Load image
with open("cat.jpg", "rb") as f:
    image_bytes = f.read()

# Run vision model
response = client.vision.create(
    model="openai/clip-vit-base-patch16",
    image=image_bytes
)

print("Prediction:", response.choices[0].text)
```

## REST API Example

```bash
curl -X POST http://localhost:8000/v1/vision \
  -H "Content-Type: application/json" \
  --data-binary @cat.jpg \
  -H "Content-Type: application/octet-stream" \
  -H "Model: openai/clip-vit-base-patch16"
```

## Use Cases
- Image captioning
- Image classification
- Multimodal search

## Next Steps
- **[Embeddings](embeddings.md)**
- **[Multimodal](multimodal.md)** 