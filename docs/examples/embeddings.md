# Embeddings Example

This example demonstrates how to use Inferneo to generate embeddings for text data.

## Python Client Example

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

texts = [
    "What is machine learning?",
    "Explain deep learning.",
    "Describe neural networks."
]

# Generate embeddings
response = client.embeddings.create(
    model="meta-llama/Llama-2-7b-embeddings",
    input=texts
)

# Access embeddings
for i, embedding in enumerate(response.data):
    print(f"Text {i+1} embedding: {embedding['embedding'][:5]} ...")
```

## REST API Example

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-embeddings",
    "input": [
      "What is machine learning?",
      "Explain deep learning."
    ]
  }'
```

## Use Cases
- Semantic search
- Clustering
- Similarity comparison

## Next Steps
- **[Text Generation](text-generation.md)**
- **[Vision Models](vision-models.md)** 