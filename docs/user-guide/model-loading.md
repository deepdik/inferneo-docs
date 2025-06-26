# Model Loading

## Overview

Guide for loading models in Inferneo.

## Supported Formats

- Hugging Face Transformers
- ONNX
- TorchScript
- TensorRT

## Basic Loading

### Hugging Face Models

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### Local Models

```bash
inferneo serve --model /path/to/local/model
```

## Configuration

### Memory Settings

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --gpu-memory-utilization 0.8
```

### Multiple Models

```bash
inferneo serve --model model1 --model model2
```

## Next Steps

- [Batching](batching.md)
- [Streaming](streaming.md)
- [Quantization](quantization.md)
- [Distributed Inference](distributed-inference.md) 