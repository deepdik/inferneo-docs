# Custom Models

This page explains how to add and serve custom models with Inferneo.

## Adding a Custom Model

1. **Prepare your model** in a supported format (e.g., HuggingFace Transformers, ONNX)
2. **Place the model files** in a directory accessible to the server
3. **Update the configuration** to include your model

## Example: HuggingFace Model

```bash
inferneo serve --model my-org/my-custom-model
```

## Example: ONNX Model

```bash
inferneo serve --model /path/to/model.onnx
```

## Model Configuration

You can specify model parameters in `config.yaml`:

```yaml
model: my-org/my-custom-model
max_model_len: 4096
gpu_memory_utilization: 0.9
```

## Pre/Post-Processing

- Implement custom pre-processing or post-processing using the plugin system

## Testing Your Model

- Use the Python client or REST API to send test requests

## Next Steps
- **[Performance Tuning](performance-tuning.md)**
- **[Contributing](contributing.md)** 