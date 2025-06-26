# Quantization

Quantization is a technique that reduces the memory footprint and computational requirements of large language models by using lower precision data types.

## Overview

Quantization converts model weights from high precision (typically FP16 or FP32) to lower precision formats (INT8, INT4, etc.), significantly reducing memory usage while maintaining reasonable accuracy.

## Supported Quantization Methods

### AWQ (Activation-aware Weight Quantization)

AWQ is an efficient quantization method that considers activation statistics during quantization:

```python
from inferneo import Inferneo

# Load model with AWQ quantization
client = Inferneo("http://localhost:8000")

# Use AWQ quantized model
response = client.generate(
    "Explain quantum computing",
    model="meta-llama/Llama-2-7b-chat-hf-awq"
)
```

### GPTQ (Gradient-based Post-training Quantization)

GPTQ provides high-quality quantization with minimal accuracy loss:

```python
from inferneo import Inferneo

# Load GPTQ quantized model
client = Inferneo("http://localhost:8000")

# Use GPTQ quantized model
response = client.generate(
    "Write a story about AI",
    model="meta-llama/Llama-2-7b-chat-hf-gptq"
)
```

### SqueezeLLM

SqueezeLLM offers efficient quantization with sparsity:

```python
from inferneo import Inferneo

# Load SqueezeLLM quantized model
client = Inferneo("http://localhost:8000")

# Use SqueezeLLM quantized model
response = client.generate(
    "Explain machine learning",
    model="meta-llama/Llama-2-7b-chat-hf-squeezellm"
)
```

## Quantization Configuration

### Server Configuration

Configure quantization settings in your server configuration:

```yaml
# config.yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  quantization: "awq"
  quantization_config:
    bits: 4
    group_size: 128
    zero_point: true
    scale: true
```

### Dynamic Quantization

Apply quantization dynamically at runtime:

```python
from inferneo import Inferneo

client = Inferneo("http://localhost:8000")

# Configure quantization parameters
client.set_quantization_config(
    method="awq",
    bits=4,
    group_size=128,
    zero_point=True,
    scale=True
)

# Generate with quantization
response = client.generate("Explain neural networks")
```

## Performance Comparison

### Memory Usage Comparison

```python
import psutil
import time
from inferneo import Inferneo

def benchmark_memory_usage():
    client = Inferneo("http://localhost:8000")
    
    # Test different quantization methods
    models = [
        "meta-llama/Llama-2-7b-chat-hf",  # No quantization
        "meta-llama/Llama-2-7b-chat-hf-awq",  # AWQ
        "meta-llama/Llama-2-7b-chat-hf-gptq",  # GPTQ
        "meta-llama/Llama-2-7b-chat-hf-squeezellm"  # SqueezeLLM
    ]
    
    results = {}
    
    for model in models:
        print(f"Testing {model}...")
        
        # Measure memory before
        memory_before = psutil.virtual_memory().used
        
        # Load model
        start_time = time.time()
        response = client.generate("Test prompt", model=model)
        load_time = time.time() - start_time
        
        # Measure memory after
        memory_after = psutil.virtual_memory().used
        memory_used = memory_after - memory_before
        
        results[model] = {
            "memory_mb": memory_used / (1024 * 1024),
            "load_time": load_time
        }
    
    # Print results
    print("\nMemory Usage Comparison:")
    print("-" * 60)
    for model, metrics in results.items():
        print(f"{model}: {metrics['memory_mb']:.1f} MB, {metrics['load_time']:.2f}s")
    
    return results

# Run benchmark
benchmark_memory_usage()
```

### Speed vs Accuracy Trade-off

```python
from inferneo import Inferneo
import time
import numpy as np

def benchmark_speed_accuracy():
    client = Inferneo("http://localhost:8000")
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing",
        "What is machine learning?",
        "Describe neural networks",
        "How does backpropagation work?",
        "Explain the transformer architecture"
    ]
    
    models = [
        ("FP16", "meta-llama/Llama-2-7b-chat-hf"),
        ("AWQ", "meta-llama/Llama-2-7b-chat-hf-awq"),
        ("GPTQ", "meta-llama/Llama-2-7b-chat-hf-gptq")
    ]
    
    results = {}
    
    for model_name, model_path in models:
        print(f"Testing {model_name}...")
        
        times = []
        responses = []
        
        for prompt in test_prompts:
            start_time = time.time()
            response = client.generate(prompt, model=model_path)
            end_time = time.time()
            
            times.append(end_time - start_time)
            responses.append(response.generated_text)
        
        results[model_name] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "responses": responses
        }
    
    # Print results
    print("\nSpeed Comparison:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics['avg_time']:.3f}s ± {metrics['std_time']:.3f}s")
    
    return results

# Run benchmark
benchmark_speed_accuracy()
```

## Custom Quantization

### Quantizing Your Own Model

```python
from inferneo import Inferneo
import torch

def quantize_model(model_path, output_path, method="awq"):
    """
    Quantize a model using the specified method.
    """
    client = Inferneo("http://localhost:8000")
    
    # Load the original model
    print(f"Loading model from {model_path}...")
    
    # Configure quantization
    quantization_config = {
        "method": method,
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "scale": True
    }
    
    # Apply quantization
    print(f"Applying {method} quantization...")
    client.quantize_model(
        model_path=model_path,
        output_path=output_path,
        config=quantization_config
    )
    
    print(f"Quantized model saved to {output_path}")

# Usage
quantize_model(
    model_path="./my-model",
    output_path="./my-model-awq",
    method="awq"
)
```

### Quantization with Custom Parameters

```python
from inferneo import Inferneo

def custom_quantization():
    client = Inferneo("http://localhost:8000")
    
    # Custom AWQ configuration
    awq_config = {
        "method": "awq",
        "bits": 4,
        "group_size": 64,  # Smaller group size
        "zero_point": True,
        "scale": True,
        "act_order": True,  # Activation ordering
        "true_sequential": True
    }
    
    # Custom GPTQ configuration
    gptq_config = {
        "method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
        "static_groups": False,
        "sym": False,  # Asymmetric quantization
        "true_sequential": True
    }
    
    # Apply custom quantization
    client.set_quantization_config(**awq_config)
    
    # Test generation
    response = client.generate("Test prompt with custom quantization")
    print(response.generated_text)

custom_quantization()
```

## Memory Optimization

### Memory-Efficient Loading

```python
from inferneo import Inferneo
import gc

def memory_efficient_loading():
    client = Inferneo("http://localhost:8000")
    
    # Configure for memory efficiency
    client.set_config(
        max_model_len=2048,  # Reduce context length
        gpu_memory_utilization=0.8,  # Limit GPU memory usage
        swap_space=4  # Enable swap space
    )
    
    # Load quantized model
    response = client.generate(
        "Explain the benefits of quantization",
        model="meta-llama/Llama-2-7b-chat-hf-awq"
    )
    
    # Force garbage collection
    gc.collect()
    
    return response

# Usage
response = memory_efficient_loading()
```

### Multi-GPU Quantization

```python
from inferneo import Inferneo

def multi_gpu_quantization():
    client = Inferneo("http://localhost:8000")
    
    # Configure for multi-GPU
    client.set_config(
        tensor_parallel_size=2,  # Use 2 GPUs
        gpu_memory_utilization=0.7,  # Conservative memory usage
        max_model_len=4096
    )
    
    # Load large quantized model
    response = client.generate(
        "Write a comprehensive guide to AI",
        model="meta-llama/Llama-2-70b-chat-hf-awq"
    )
    
    return response

# Usage
response = multi_gpu_quantization()
```

## Accuracy Evaluation

### Quantization Impact Assessment

```python
from inferneo import Inferneo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_quantization_impact():
    client = Inferneo("http://localhost:8000")
    
    # Test prompts for evaluation
    evaluation_prompts = [
        "Explain the concept of machine learning",
        "What are the advantages of deep learning?",
        "Describe the transformer architecture",
        "How does attention mechanism work?",
        "Explain backpropagation in neural networks"
    ]
    
    # Get responses from different quantization methods
    responses = {}
    
    models = [
        ("FP16", "meta-llama/Llama-2-7b-chat-hf"),
        ("AWQ", "meta-llama/Llama-2-7b-chat-hf-awq"),
        ("GPTQ", "meta-llama/Llama-2-7b-chat-hf-gptq")
    ]
    
    for model_name, model_path in models:
        responses[model_name] = []
        
        for prompt in evaluation_prompts:
            response = client.generate(prompt, model=model_path)
            responses[model_name].append(response.generated_text)
    
    # Calculate similarity scores
    fp16_responses = responses["FP16"]
    
    print("Quantization Impact Analysis:")
    print("-" * 40)
    
    for model_name in ["AWQ", "GPTQ"]:
        similarities = []
        
        for i in range(len(evaluation_prompts)):
            # Simple similarity based on response length and content
            fp16_len = len(fp16_responses[i])
            quantized_len = len(responses[model_name][i])
            
            # Length similarity
            length_sim = min(fp16_len, quantized_len) / max(fp16_len, quantized_len)
            
            # Content similarity (simple word overlap)
            fp16_words = set(fp16_responses[i].lower().split())
            quantized_words = set(responses[model_name][i].lower().split())
            
            if fp16_words and quantized_words:
                content_sim = len(fp16_words & quantized_words) / len(fp16_words | quantized_words)
            else:
                content_sim = 0
            
            # Combined similarity
            combined_sim = (length_sim + content_sim) / 2
            similarities.append(combined_sim)
        
        avg_similarity = np.mean(similarities)
        print(f"{model_name} vs FP16: {avg_similarity:.3f} similarity")
    
    return responses

# Run evaluation
evaluation_results = evaluate_quantization_impact()
```

## Best Practices

### Choosing the Right Quantization Method

1. **AWQ**: Best for general use cases with good accuracy/speed balance
2. **GPTQ**: Best for maximum accuracy preservation
3. **SqueezeLLM**: Best for memory-constrained environments

### Configuration Guidelines

```python
# For production use
production_config = {
    "method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "scale": True
}

# For development/testing
dev_config = {
    "method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": True
}

# For memory-constrained environments
memory_constrained_config = {
    "method": "squeezellm",
    "bits": 4,
    "group_size": 64
}
```

### Monitoring Quantization Performance

```python
import psutil
import time
from inferneo import Inferneo

class QuantizationMonitor:
    def __init__(self):
        self.memory_history = []
        self.latency_history = []
    
    def monitor_performance(self, client, prompt, model):
        # Monitor memory
        memory_before = psutil.virtual_memory().used
        
        # Measure latency
        start_time = time.time()
        response = client.generate(prompt, model=model)
        latency = time.time() - start_time
        
        # Monitor memory after
        memory_after = psutil.virtual_memory().used
        memory_used = memory_after - memory_before
        
        # Record metrics
        self.memory_history.append(memory_used / (1024 * 1024))  # MB
        self.latency_history.append(latency)
        
        return {
            "memory_mb": memory_used / (1024 * 1024),
            "latency_s": latency,
            "response_length": len(response.generated_text)
        }
    
    def get_stats(self):
        return {
            "avg_memory_mb": np.mean(self.memory_history),
            "avg_latency_s": np.mean(self.latency_history),
            "total_requests": len(self.memory_history)
        }

# Usage
monitor = QuantizationMonitor()
client = Inferneo("http://localhost:8000")

# Monitor different models
models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf-awq",
    "meta-llama/Llama-2-7b-chat-hf-gptq"
]

for model in models:
    metrics = monitor.monitor_performance(
        client, 
        "Test prompt", 
        model
    )
    print(f"{model}: {metrics}")

print(f"Overall stats: {monitor.get_stats()}")
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**
```python
# Reduce model size or use more aggressive quantization
client.set_config(
    gpu_memory_utilization=0.6,  # Reduce GPU memory usage
    max_model_len=1024  # Reduce context length
)
```

**Slow Performance**
```python
# Use faster quantization method
client.set_quantization_config(
    method="awq",  # Generally faster than GPTQ
    bits=4,
    group_size=128
)
```

**Accuracy Degradation**
```python
# Use higher precision quantization
client.set_quantization_config(
    method="gptq",  # Better accuracy preservation
    bits=4,
    group_size=128,
    desc_act=True
)
```

For more advanced optimization techniques, see the Performance Tuning guide. 