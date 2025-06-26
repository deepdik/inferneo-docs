# Model Loading

This guide covers how to load different types of models in Inferneo, including supported formats, loading strategies, and best practices.

## Supported Model Formats

### Hugging Face Transformers

Inferneo provides seamless integration with Hugging Face Transformers models:

```bash
# Load a Hugging Face model
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

**Supported model types:**
- **Causal Language Models**: GPT, Llama, Mistral, etc.
- **Sequence-to-Sequence**: T5, BART, etc.
- **Encoder Models**: BERT, RoBERTa, etc.
- **Multi-modal Models**: Vision-Language models

### Custom Model Formats

Inferneo supports various model formats for deployment:

#### ONNX Models

```bash
# Load ONNX model
inferneo serve --model path/to/model.onnx --model-type onnx
```

#### TorchScript Models

```bash
# Load TorchScript model
inferneo serve --model path/to/model.pt --model-type torchscript
```

#### TensorRT Models

```bash
# Load TensorRT model
inferneo serve --model path/to/model.trt --model-type tensorrt
```

## Loading Strategies

### Single Model Loading

```bash
# Basic single model loading
inferneo serve --model meta-llama/Llama-2-7b-chat-hf
```

### Multiple Model Loading

```bash
# Load multiple models simultaneously
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --model meta-llama/Llama-2-13b-chat-hf \
  --model microsoft/DialoGPT-medium
```

### Lazy Loading

```bash
# Enable lazy loading for memory efficiency
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --lazy-loading
```

## Model Configuration

### Model Parameters

```bash
# Configure model parameters
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-model-len 4096 \
  --max-batch-size 32 \
  --max-concurrent-requests 100
```

### Memory Management

```bash
# Configure GPU memory usage
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048
```

### Quantization

```bash
# Load quantized model for memory efficiency
inferneo serve \
  --model TheBloke/Llama-2-7B-Chat-GGUF \
  --quantization awq \
  --dtype half
```

## Advanced Loading Options

### Custom Tokenizers

```python
from inferneo import InferneoClient

# Load model with custom tokenizer
client = InferneoClient("http://localhost:8000")

# Use custom tokenizer configuration
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    tokenizer_config={
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>"
    }
)
```

### Model Adapters

```bash
# Load model with LoRA adapters
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --adapter path/to/adapter \
  --adapter-type lora
```

### Custom Model Classes

```python
# Define custom model class
from inferneo.models import BaseModel

class CustomModel(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Custom initialization
    
    def generate(self, inputs, **kwargs):
        # Custom generation logic
        pass

# Load custom model
inferneo serve --model-class CustomModel --model path/to/model
```

## Model Validation

### Health Checks

```python
# Check model health
client = InferneoClient("http://localhost:8000")

# Get model info
model_info = client.models.get("meta-llama/Llama-2-7b-chat-hf")
print(f"Model loaded: {model_info.id}")
print(f"Parameters: {model_info.parameters}")
print(f"Format: {model_info.format}")

# Test model inference
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Test message"}],
    max_tokens=10
)
print("Model is working correctly!")
```

### Performance Testing

```python
import time
import statistics

def benchmark_model(client, model_id, num_requests=100):
    latencies = []
    
    for _ in range(num_requests):
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50
        )
        end_time = time.time()
        latencies.append(end_time - start_time)
    
    return {
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "p95_latency": statistics.quantiles(latencies, n=20)[18],
        "p99_latency": statistics.quantiles(latencies, n=100)[98]
    }

# Run benchmark
results = benchmark_model(client, "meta-llama/Llama-2-7b-chat-hf")
print(f"Performance results: {results}")
```

## Error Handling

### Common Loading Errors

```python
from inferneo import InferneoError

try:
    # Attempt to load model
    response = client.chat.completions.create(
        model="invalid-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except InferneoError as e:
    if "model not found" in str(e):
        print("Model not loaded or not found")
    elif "out of memory" in str(e):
        print("GPU memory insufficient")
    elif "invalid format" in str(e):
        print("Model format not supported")
    else:
        print(f"Unknown error: {e}")
```

### Recovery Strategies

```python
def load_model_with_fallback(client, primary_model, fallback_models):
    """Try to load primary model, fallback to alternatives if needed."""
    
    for model in [primary_model] + fallback_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print(f"Successfully loaded model: {model}")
            return model
        except InferneoError as e:
            print(f"Failed to load {model}: {e}")
            continue
    
    raise Exception("No models could be loaded")

# Usage
try:
    working_model = load_model_with_fallback(
        client,
        "meta-llama/Llama-2-70b-chat-hf",
        ["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
    )
except Exception as e:
    print(f"All models failed to load: {e}")
```

## Best Practices

### 1. Model Selection
- **Choose appropriate model size** for your hardware
- **Consider quantization** for memory efficiency
- **Test performance** before production deployment

### 2. Memory Management
- **Monitor GPU memory usage** during loading
- **Use lazy loading** for multiple models
- **Implement proper cleanup** when switching models

### 3. Error Handling
- **Always handle loading errors** gracefully
- **Implement fallback strategies** for critical applications
- **Log loading events** for debugging

### 4. Performance Optimization
- **Pre-warm models** for consistent latency
- **Use appropriate batch sizes** for your use case
- **Monitor and optimize** memory usage

## Configuration Examples

### Production Configuration

```bash
# Production-ready configuration
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-batch-size 16 \
  --max-concurrent-requests 50 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enable-metrics \
  --log-level info
```

### Development Configuration

```bash
# Development configuration
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-batch-size 4 \
  --max-concurrent-requests 10 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --log-level debug
```

### Multi-GPU Configuration

```bash
# Multi-GPU setup
inferneo serve \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --max-batch-size 8 \
  --max-concurrent-requests 20
```

## Next Steps

Now that you understand model loading:

- **[Batching](batching.md)** - Optimize throughput with request batching
- **[Streaming](streaming.md)** - Implement real-time response streaming
- **[Quantization](quantization.md)** - Reduce memory usage with model quantization
- **[Distributed Inference](distributed-inference.md)** - Scale across multiple GPUs 