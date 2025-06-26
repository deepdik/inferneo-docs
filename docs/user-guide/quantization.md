# Quantization

This guide covers how to use quantization techniques in Inferneo to reduce model size and improve inference speed.

## Overview

Quantization reduces the precision of model weights and activations, providing:

- **Reduced memory usage** for larger models
- **Faster inference** with optimized operations
- **Lower hardware requirements** for deployment
- **Cost savings** in cloud deployments

## Supported Quantization Methods

### AWQ (Activation-aware Weight Quantization)

AWQ is an advanced quantization method that considers activation distributions:

```bash
# Load model with AWQ quantization
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --quantization awq
```

### GPTQ (Gradient-based Post-training Quantization)

GPTQ provides high-quality quantization with minimal accuracy loss:

```bash
# Load model with GPTQ quantization
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --quantization gptq
```

### SqueezeLLM

SqueezeLLM offers efficient quantization for large language models:

```bash
# Load model with SqueezeLLM quantization
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --quantization squeezellm
```

## Quantization Levels

### 4-bit Quantization

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Use 4-bit quantized model
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf-awq",  # AWQ quantized model
    prompt="Explain machine learning",
    max_tokens=100
)
```

### 8-bit Quantization

```python
# Use 8-bit quantized model
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf-gptq",  # GPTQ quantized model
    prompt="Explain deep learning",
    max_tokens=100
)
```

## Model Loading with Quantization

### Server Configuration

```yaml
# config.yaml with quantization
model: meta-llama/Llama-2-7b-chat-hf
quantization: awq
gpu_memory_utilization: 0.9
max_model_len: 4096
```

```bash
# Start server with quantization
inferneo serve --config config.yaml
```

### Python Client Usage

```python
# Connect to quantized model
client = InferneoClient("http://localhost:8000")

# List available models
models = client.models.list()
print("Available models:", [model.id for model in models.data])

# Use quantized model
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf-awq",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

## Performance Comparison

### Memory Usage Comparison

```python
import psutil
import time

def benchmark_memory_usage(model_id, client):
    """Benchmark memory usage for different model configurations."""
    
    # Get initial memory usage
    initial_memory = psutil.virtual_memory().used
    
    # Load model and measure memory
    start_time = time.time()
    
    response = client.completions.create(
        model=model_id,
        prompt="Test prompt for memory measurement",
        max_tokens=10
    )
    
    end_time = time.time()
    final_memory = psutil.virtual_memory().used
    
    memory_used = final_memory - initial_memory
    inference_time = end_time - start_time
    
    return {
        "model": model_id,
        "memory_used_mb": memory_used / (1024 * 1024),
        "inference_time_ms": inference_time * 1000
    }

# Compare different quantization methods
models_to_test = [
    "meta-llama/Llama-2-7b-chat-hf",           # FP16
    "meta-llama/Llama-2-7b-chat-hf-awq",       # AWQ 4-bit
    "meta-llama/Llama-2-7b-chat-hf-gptq",      # GPTQ 4-bit
    "meta-llama/Llama-2-7b-chat-hf-squeezellm" # SqueezeLLM
]

results = []
for model_id in models_to_test:
    try:
        result = benchmark_memory_usage(model_id, client)
        results.append(result)
        print(f"Model: {result['model']}")
        print(f"Memory: {result['memory_used_mb']:.1f} MB")
        print(f"Time: {result['inference_time_ms']:.1f} ms")
        print("-" * 40)
    except Exception as e:
        print(f"Error with {model_id}: {e}")
```

### Throughput Comparison

```python
def benchmark_throughput(model_id, client, num_requests=100):
    """Benchmark throughput for different models."""
    
    prompts = [f"Test prompt {i}" for i in range(num_requests)]
    
    start_time = time.time()
    
    responses = []
    for prompt in prompts:
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50
            )
            responses.append(response)
        except Exception as e:
            print(f"Error: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_requests = len(responses)
    throughput = successful_requests / total_time
    
    return {
        "model": model_id,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "total_time": total_time,
        "throughput_rps": throughput
    }

# Benchmark throughput
for model_id in models_to_test:
    try:
        result = benchmark_throughput(model_id, client)
        print(f"Model: {result['model']}")
        print(f"Throughput: {result['throughput_rps']:.2f} requests/second")
        print(f"Success rate: {result['successful_requests']}/{result['total_requests']}")
        print("-" * 40)
    except Exception as e:
        print(f"Error with {model_id}: {e}")
```

## Quality Assessment

### Accuracy Comparison

```python
def evaluate_quality(model_id, client, test_prompts):
    """Evaluate quality of quantized models."""
    
    results = []
    
    for prompt in test_prompts:
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            results.append({
                "prompt": prompt,
                "response": response.choices[0].text,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "prompt": prompt,
                "response": None,
                "status": "error",
                "error": str(e)
            })
    
    return results

# Test prompts for quality evaluation
test_prompts = [
    "Explain the concept of machine learning in simple terms.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the process of training a neural network.",
    "How does backpropagation work in neural networks?",
    "What is the role of activation functions in neural networks?"
]

# Evaluate different models
for model_id in models_to_test:
    print(f"\nEvaluating {model_id}:")
    print("=" * 50)
    
    results = evaluate_quality(model_id, client, test_prompts)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    # Show sample responses
    for i, result in enumerate(results[:2]):  # Show first 2 responses
        if result["status"] == "success":
            print(f"\nPrompt {i+1}: {result['prompt']}")
            print(f"Response: {result['response'][:100]}...")
```

## Custom Quantization

### Quantization Configuration

```python
# Advanced quantization settings
quantization_config = {
    "method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "scale": True
}

# Use custom quantization
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Test prompt",
    max_tokens=100,
    quantization=quantization_config
)
```

### Quantization Parameters

```python
# GPTQ specific parameters
gptq_config = {
    "method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": True,
    "static_groups": False
}

# AWQ specific parameters
awq_config = {
    "method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "scale": True
}

# SqueezeLLM specific parameters
squeezellm_config = {
    "method": "squeezellm",
    "bits": 4,
    "group_size": 128
}
```

## Hardware Considerations

### GPU Memory Requirements

```python
def check_gpu_memory_requirements(model_id):
    """Check GPU memory requirements for different models."""
    
    # Approximate memory requirements (varies by implementation)
    memory_requirements = {
        "meta-llama/Llama-2-7b-chat-hf": "14 GB",           # FP16
        "meta-llama/Llama-2-7b-chat-hf-awq": "4 GB",        # AWQ 4-bit
        "meta-llama/Llama-2-7b-chat-hf-gptq": "4 GB",       # GPTQ 4-bit
        "meta-llama/Llama-2-7b-chat-hf-squeezellm": "4 GB"  # SqueezeLLM 4-bit
    }
    
    return memory_requirements.get(model_id, "Unknown")

# Check requirements
for model_id in models_to_test:
    memory_req = check_gpu_memory_requirements(model_id)
    print(f"{model_id}: {memory_req}")
```

### CPU vs GPU Quantization

```python
# CPU quantization (slower but more memory efficient)
cpu_config = {
    "device": "cpu",
    "quantization": "awq",
    "bits": 4
}

# GPU quantization (faster but requires more memory)
gpu_config = {
    "device": "cuda",
    "quantization": "awq",
    "bits": 4
}
```

## Best Practices

### Choosing Quantization Method

```python
def choose_quantization_method(use_case, hardware_constraints):
    """Choose the best quantization method based on requirements."""
    
    recommendations = {
        "high_accuracy": {
            "method": "gptq",
            "bits": 4,
            "reason": "Best accuracy preservation"
        },
        "memory_constrained": {
            "method": "awq",
            "bits": 4,
            "reason": "Lowest memory usage"
        },
        "speed_optimized": {
            "method": "squeezellm",
            "bits": 4,
            "reason": "Fastest inference"
        },
        "balanced": {
            "method": "awq",
            "bits": 4,
            "reason": "Good balance of speed and accuracy"
        }
    }
    
    return recommendations.get(use_case, recommendations["balanced"])

# Usage examples
print("High accuracy use case:", choose_quantization_method("high_accuracy", {}))
print("Memory constrained:", choose_quantization_method("memory_constrained", {}))
print("Speed optimized:", choose_quantization_method("speed_optimized", {}))
```

### Quality vs Performance Trade-offs

```python
def evaluate_trade_offs(model_id, client):
    """Evaluate quality vs performance trade-offs."""
    
    # Measure performance
    perf_result = benchmark_throughput(model_id, client, num_requests=50)
    
    # Measure quality
    quality_result = evaluate_quality(model_id, client, test_prompts[:5])
    quality_score = sum(1 for r in quality_result if r["status"] == "success") / len(quality_result)
    
    return {
        "model": model_id,
        "throughput_rps": perf_result["throughput_rps"],
        "quality_score": quality_score,
        "efficiency_score": perf_result["throughput_rps"] * quality_score
    }

# Evaluate trade-offs
trade_off_results = []
for model_id in models_to_test:
    try:
        result = evaluate_trade_offs(model_id, client)
        trade_off_results.append(result)
    except Exception as e:
        print(f"Error evaluating {model_id}: {e}")

# Sort by efficiency score
trade_off_results.sort(key=lambda x: x["efficiency_score"], reverse=True)

print("Model Efficiency Ranking:")
for i, result in enumerate(trade_off_results):
    print(f"{i+1}. {result['model']}")
    print(f"   Throughput: {result['throughput_rps']:.2f} RPS")
    print(f"   Quality: {result['quality_score']:.2f}")
    print(f"   Efficiency: {result['efficiency_score']:.2f}")
    print()
```

## Troubleshooting

### Common Issues

```python
def troubleshoot_quantization_issues(model_id, client):
    """Troubleshoot common quantization issues."""
    
    issues = []
    
    # Check if model loads
    try:
        response = client.completions.create(
            model=model_id,
            prompt="Test",
            max_tokens=10
        )
    except Exception as e:
        issues.append(f"Model loading failed: {e}")
    
    # Check memory usage
    try:
        import psutil
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 90:
            issues.append(f"High memory usage: {memory_usage}%")
    except:
        pass
    
    # Check GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            if gpu_memory > 10:  # More than 10GB
                issues.append(f"High GPU memory usage: {gpu_memory:.1f} GB")
    except:
        pass
    
    return issues

# Troubleshoot issues
for model_id in models_to_test:
    print(f"\nTroubleshooting {model_id}:")
    issues = troubleshoot_quantization_issues(model_id, client)
    
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No issues detected")
```

## Next Steps

- **[Model Loading](model-loading.md)** - Learn about different model loading strategies
- **[Performance Tuning](developer-guide/performance-tuning.md)** - Advanced optimization techniques
- **[Hardware Optimization](developer-guide/hardware-optimization.md)** - Optimize for specific hardware 