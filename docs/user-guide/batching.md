# Batching

This guide covers how to optimize inference performance using batching techniques in Inferneo.

## Overview

Batching allows you to process multiple requests together, significantly improving throughput and resource utilization. This is especially important for:

- **High-throughput applications** requiring maximum efficiency
- **Batch processing** of large datasets
- **Resource optimization** to maximize GPU utilization
- **Cost reduction** by processing more requests per unit time

## Basic Batching

### Simple Request Batching

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Batch multiple prompts
prompts = [
    "Explain machine learning",
    "What is deep learning?",
    "Describe neural networks",
    "How does backpropagation work?"
]

# Process all prompts in a single batch
responses = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompts,
    max_tokens=100,
    temperature=0.7
)

# Access individual responses
for i, response in enumerate(responses.choices):
    print(f"Prompt {i+1}: {response.text}")
```

### Chat Completion Batching

```python
# Batch chat conversations
conversations = [
    [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
        {"role": "user", "content": "Give me an example"}
    ],
    [
        {"role": "user", "content": "Explain machine learning"},
        {"role": "user", "content": "What are the types?"}
    ],
    [
        {"role": "user", "content": "How do neural networks work?"}
    ]
]

responses = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=conversations,
    max_tokens=150,
    temperature=0.7
)

for i, response in enumerate(responses.choices):
    print(f"Conversation {i+1}: {response.message.content}")
```

## Advanced Batching Strategies

### Dynamic Batching

```python
import time
from typing import List, Dict

class DynamicBatcher:
    def __init__(self, client, model_id, max_batch_size=32, max_wait_time=0.1):
        self.client = client
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    def add_request(self, prompt: str) -> Dict:
        """Add a request to the batch."""
        request = {
            "prompt": prompt,
            "timestamp": time.time(),
            "future": None
        }
        self.pending_requests.append(request)
        
        # Check if we should process the batch
        if self.should_process_batch():
            self.process_batch()
        
        return request
    
    def should_process_batch(self) -> bool:
        """Determine if we should process the current batch."""
        current_time = time.time()
        
        # Process if batch is full
        if len(self.pending_requests) >= self.max_batch_size:
            return True
        
        # Process if enough time has passed
        if current_time - self.last_batch_time >= self.max_wait_time:
            return True
        
        return False
    
    def process_batch(self):
        """Process the current batch of requests."""
        if not self.pending_requests:
            return
        
        # Extract prompts
        prompts = [req["prompt"] for req in self.pending_requests]
        
        # Process batch
        try:
            responses = self.client.completions.create(
                model=self.model_id,
                prompt=prompts,
                max_tokens=100,
                temperature=0.7
            )
            
            # Assign responses back to requests
            for i, req in enumerate(self.pending_requests):
                req["response"] = responses.choices[i].text
                
        except Exception as e:
            # Handle errors
            for req in self.pending_requests:
                req["error"] = str(e)
        
        # Clear processed requests
        self.pending_requests = []
        self.last_batch_time = time.time()

# Usage
batcher = DynamicBatcher(client, "meta-llama/Llama-2-7b-chat-hf")

# Add requests
batcher.add_request("Explain quantum computing")
batcher.add_request("What is blockchain?")
batcher.add_request("Describe cloud computing")

# Force processing
batcher.process_batch()
```

## Batch Size Optimization

### Finding Optimal Batch Size

```python
import time
import matplotlib.pyplot as plt

def benchmark_batch_sizes(client, model_id, test_prompts, batch_sizes):
    """Benchmark different batch sizes to find optimal performance."""
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        total_time = 0
        total_requests = 0
        
        # Process test prompts in batches
        for i in range(0, len(test_prompts), batch_size):
            batch = test_prompts[i:i + batch_size]
            
            start_time = time.time()
            
            try:
                responses = client.completions.create(
                    model=model_id,
                    prompt=batch,
                    max_tokens=100,
                    temperature=0.7
                )
                
                end_time = time.time()
                batch_time = end_time - start_time
                
                total_time += batch_time
                total_requests += len(batch)
                
            except Exception as e:
                print(f"Error with batch size {batch_size}: {e}")
                continue
        
        # Calculate metrics
        if total_requests > 0:
            avg_time_per_request = total_time / total_requests
            requests_per_second = total_requests / total_time
            
            results[batch_size] = {
                "avg_time_per_request": avg_time_per_request,
                "requests_per_second": requests_per_second,
                "total_time": total_time,
                "total_requests": total_requests
            }
    
    return results

def plot_batch_performance(results):
    """Plot batch size performance results."""
    batch_sizes = list(results.keys())
    throughput = [results[bs]["requests_per_second"] for bs in batch_sizes]
    latency = [results[bs]["avg_time_per_request"] for bs in batch_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput plot
    ax1.plot(batch_sizes, throughput, 'bo-')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Requests per Second')
    ax1.set_title('Throughput vs Batch Size')
    ax1.grid(True)
    
    # Latency plot
    ax2.plot(batch_sizes, latency, 'ro-')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Average Time per Request (seconds)')
    ax2.set_title('Latency vs Batch Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage
test_prompts = ["Test prompt"] * 100  # 100 identical prompts for testing
batch_sizes = [1, 2, 4, 8, 16, 32, 64]

results = benchmark_batch_sizes(client, "meta-llama/Llama-2-7b-chat-hf", test_prompts, batch_sizes)
plot_batch_performance(results)

# Find optimal batch size
optimal_batch_size = max(results.keys(), key=lambda x: results[x]["requests_per_second"])
print(f"Optimal batch size: {optimal_batch_size}")
```

### Adaptive Batch Sizing

```python
class AdaptiveBatcher:
    def __init__(self, client, model_id, initial_batch_size=8):
        self.client = client
        self.model_id = model_id
        self.batch_size = initial_batch_size
        self.performance_history = []
        self.min_batch_size = 1
        self.max_batch_size = 64
    
    def update_batch_size(self, batch_time, num_requests):
        """Update batch size based on performance."""
        if num_requests == 0:
            return
        
        throughput = num_requests / batch_time
        self.performance_history.append({
            "batch_size": self.batch_size,
            "throughput": throughput,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Adjust batch size based on performance trend
        if len(self.performance_history) >= 3:
            recent_throughput = self.performance_history[-1]["throughput"]
            previous_throughput = self.performance_history[-2]["throughput"]
            
            if recent_throughput > previous_throughput:
                # Performance improving, try larger batch
                self.batch_size = min(self.batch_size * 2, self.max_batch_size)
            elif recent_throughput < previous_throughput * 0.9:
                # Performance degrading, reduce batch size
                self.batch_size = max(self.batch_size // 2, self.min_batch_size)
    
    def process_batch(self, prompts):
        """Process a batch with adaptive sizing."""
        if not prompts:
            return []
        
        # Take current batch size
        batch = prompts[:self.batch_size]
        
        start_time = time.time()
        
        try:
            responses = self.client.completions.create(
                model=self.model_id,
                prompt=batch,
                max_tokens=100,
                temperature=0.7
            )
            
            end_time = time.time()
            batch_time = end_time - start_time
            
            # Update batch size based on performance
            self.update_batch_size(batch_time, len(batch))
            
            return responses.choices
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Reduce batch size on error
            self.batch_size = max(self.batch_size // 2, self.min_batch_size)
            return []

# Usage
adaptive_batcher = AdaptiveBatcher(client, "meta-llama/Llama-2-7b-chat-hf")

prompts = ["Test prompt"] * 50
responses = adaptive_batcher.process_batch(prompts)
```

## Memory-Efficient Batching

### Streaming Batch Processing

```python
def stream_batch_process(prompts, client, model_id, batch_size=16):
    """Process batches with streaming to handle large datasets."""
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        try:
            responses = client.completions.create(
                model=model_id,
                prompt=batch,
                max_tokens=100,
                temperature=0.7
            )
            
            # Yield results immediately
            for response in responses.choices:
                yield response.text
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # Yield None for failed requests
            for _ in batch:
                yield None

# Usage
prompts = ["Prompt " + str(i) for i in range(1000)]

for i, response in enumerate(stream_batch_process(prompts, client, "meta-llama/Llama-2-7b-chat-hf")):
    if response:
        print(f"Response {i}: {response[:50]}...")
    else:
        print(f"Request {i} failed")
```

### Chunked Processing

```python
def chunked_batch_process(file_path, client, model_id, chunk_size=1000, batch_size=32):
    """Process large files in chunks with batching."""
    
    results = []
    
    with open(file_path, 'r') as f:
        chunk = []
        
        for line in f:
            chunk.append(line.strip())
            
            if len(chunk) >= chunk_size:
                # Process chunk in batches
                chunk_results = []
                
                for i in range(0, len(chunk), batch_size):
                    batch = chunk[i:i + batch_size]
                    
                    try:
                        responses = client.completions.create(
                            model=model_id,
                            prompt=batch,
                            max_tokens=100,
                            temperature=0.7
                        )
                        
                        chunk_results.extend([choice.text for choice in responses.choices])
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        chunk_results.extend([None] * len(batch))
                
                results.extend(chunk_results)
                chunk = []
        
        # Process remaining chunk
        if chunk:
            chunk_results = []
            for i in range(0, len(chunk), batch_size):
                batch = chunk[i:i + batch_size]
                
                try:
                    responses = client.completions.create(
                        model=model_id,
                        prompt=batch,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    chunk_results.extend([choice.text for choice in responses.choices])
                    
                except Exception as e:
                    print(f"Error processing final batch: {e}")
                    chunk_results.extend([None] * len(batch))
            
            results.extend(chunk_results)
    
    return results

# Usage
results = chunked_batch_process("large_dataset.txt", client, "meta-llama/Llama-2-7b-chat-hf")
```

## Error Handling in Batching

### Robust Batch Processing

```python
def robust_batch_process(prompts, client, model_id, batch_size=16, max_retries=3):
    """Process batches with robust error handling and retries."""
    
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = [None] * len(batch)
        
        for attempt in range(max_retries):
            try:
                responses = client.completions.create(
                    model=model_id,
                    prompt=batch,
                    max_tokens=100,
                    temperature=0.7
                )
                
                # Assign successful responses
                for j, response in enumerate(responses.choices):
                    batch_results[j] = response.text
                
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for batch {i//batch_size}: {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed, keep None values
                    pass
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)
        
        results.extend(batch_results)
    
    return results

# Usage
prompts = ["Prompt " + str(i) for i in range(100)]
results = robust_batch_process(prompts, client, "meta-llama/Llama-2-7b-chat-hf")

# Check success rate
success_count = sum(1 for r in results if r is not None)
print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
```

## Best Practices

1. **Start Small**: Begin with small batch sizes and increase gradually
2. **Monitor Performance**: Track throughput and latency to find optimal batch size
3. **Handle Errors**: Implement retry logic and graceful error handling
4. **Memory Management**: Use streaming for large datasets to avoid memory issues
5. **Adaptive Sizing**: Adjust batch size based on performance and load
6. **Priority Handling**: Implement priority-based batching for time-sensitive requests

## Next Steps

- **[Performance Tuning](developer-guide/performance-tuning.md)** - Advanced optimization techniques
- **[Online Serving](online-serving.md)** - Real-time inference with batching
- **[Offline Inference](offline-inference.md)** - Batch processing for offline workflows 