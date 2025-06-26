# Offline Inference

This guide covers how to use Inferneo for offline inference tasks.

## Overview

Offline inference allows you to process batches of data without real-time constraints, making it ideal for:

- **Batch processing** of large datasets
- **Model evaluation** and testing
- **Data preprocessing** and analysis
- **Research and development** workflows

## Basic Usage

### Single File Processing

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Process a single file
with open("input.txt", "r") as f:
    content = f.read()

response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=content,
    max_tokens=200
)

with open("output.txt", "w") as f:
    f.write(response.choices[0].text)
```

### Batch File Processing

```python
import os
from pathlib import Path

def process_directory(input_dir, output_dir, model_id):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for file_path in input_path.glob("*.txt"):
        with open(file_path, "r") as f:
            content = f.read()
        
        response = client.completions.create(
            model=model_id,
            prompt=content,
            max_tokens=200
        )
        
        output_file = output_path / f"{file_path.stem}_processed.txt"
        with open(output_file, "w") as f:
            f.write(response.choices[0].text)

# Usage
process_directory("input_files", "output_files", "meta-llama/Llama-2-7b-chat-hf")
```

## Advanced Batch Processing

### Parallel Processing

```python
import concurrent.futures
from typing import List, Dict

def process_batch(prompts: List[str], model_id: str) -> List[Dict]:
    """Process a batch of prompts in parallel."""
    
    def process_single(prompt):
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=100
            )
            return {
                "prompt": prompt,
                "response": response.choices[0].text,
                "status": "success"
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "response": None,
                "status": "error",
                "error": str(e)
            }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single, prompts))
    
    return results

# Usage
prompts = [
    "Explain machine learning",
    "What is deep learning?",
    "Describe neural networks"
]

results = process_batch(prompts, "meta-llama/Llama-2-7b-chat-hf")
```

### Memory-Efficient Processing

```python
def process_large_dataset(file_path: str, batch_size: int = 100):
    """Process large datasets in memory-efficient batches."""
    
    results = []
    
    with open(file_path, "r") as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            
            if len(batch) >= batch_size:
                # Process batch
                batch_results = process_batch(batch, "meta-llama/Llama-2-7b-chat-hf")
                results.extend(batch_results)
                batch = []
        
        # Process remaining items
        if batch:
            batch_results = process_batch(batch, "meta-llama/Llama-2-7b-chat-hf")
            results.extend(batch_results)
    
    return results
```

## Data Formats

### CSV Processing

```python
import pandas as pd

def process_csv(input_file: str, output_file: str, prompt_column: str):
    """Process CSV files with prompts."""
    
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Process each row
    responses = []
    for _, row in df.iterrows():
        prompt = row[prompt_column]
        response = client.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            prompt=prompt,
            max_tokens=100
        )
        responses.append(response.choices[0].text)
    
    # Add responses to dataframe
    df['response'] = responses
    
    # Save output
    df.to_csv(output_file, index=False)
    return df

# Usage
process_csv("input.csv", "output.csv", "question")
```

### JSON Processing

```python
import json

def process_json_file(input_file: str, output_file: str):
    """Process JSON files with structured data."""
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        prompt = item.get("prompt", "")
        response = client.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            prompt=prompt,
            max_tokens=100
        )
        
        processed_item = {
            **item,
            "response": response.choices[0].text,
            "model": "meta-llama/Llama-2-7b-chat-hf"
        }
        processed_data.append(processed_item)
    
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    return processed_data
```

## Performance Optimization

### Batch Size Tuning

```python
def find_optimal_batch_size(test_prompts: List[str], model_id: str):
    """Find the optimal batch size for your hardware."""
    
    batch_sizes = [1, 4, 8, 16, 32]
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # Process test batch
        process_batch(test_prompts[:batch_size], model_id)
        
        end_time = time.time()
        throughput = batch_size / (end_time - start_time)
        
        results[batch_size] = {
            "time": end_time - start_time,
            "throughput": throughput
        }
    
    return results

# Usage
test_prompts = ["Test prompt"] * 32
optimal_batch = find_optimal_batch_size(test_prompts, "meta-llama/Llama-2-7b-chat-hf")
print(f"Optimal batch size: {max(optimal_batch, key=lambda x: optimal_batch[x]['throughput'])}")
```

### Memory Management

```python
import gc

def process_with_memory_management(prompts: List[str], model_id: str):
    """Process with explicit memory management."""
    
    results = []
    batch_size = 16
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Process batch
        batch_results = process_batch(batch, model_id)
        results.extend(batch_results)
        
        # Clear memory
        gc.collect()
        
        # Optional: Add delay to prevent overwhelming the server
        time.sleep(0.1)
    
    return results
```

## Error Handling

### Robust Processing

```python
def robust_batch_processing(prompts: List[str], model_id: str, max_retries: int = 3):
    """Process with robust error handling and retries."""
    
    results = []
    
    for i, prompt in enumerate(prompts):
        for attempt in range(max_retries):
            try:
                response = client.completions.create(
                    model=model_id,
                    prompt=prompt,
                    max_tokens=100
                )
                
                results.append({
                    "index": i,
                    "prompt": prompt,
                    "response": response.choices[0].text,
                    "status": "success",
                    "attempts": attempt + 1
                })
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    results.append({
                        "index": i,
                        "prompt": prompt,
                        "response": None,
                        "status": "failed",
                        "error": str(e),
                        "attempts": max_retries
                    })
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    return results
```

## Monitoring and Logging

### Progress Tracking

```python
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_with_progress(prompts: List[str], model_id: str):
    """Process with progress tracking and logging."""
    
    results = []
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=100
            )
            
            results.append({
                "index": i,
                "prompt": prompt,
                "response": response.choices[0].text
            })
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(prompts)} prompts")
                
        except Exception as e:
            logger.error(f"Error processing prompt {i}: {e}")
            results.append({
                "index": i,
                "prompt": prompt,
                "response": None,
                "error": str(e)
            })
    
    return results
```

## Best Practices

1. **Batch Processing**: Group requests to maximize throughput
2. **Error Handling**: Implement retries and graceful error handling
3. **Memory Management**: Process large datasets in chunks
4. **Progress Tracking**: Monitor progress for long-running tasks
5. **Resource Management**: Clean up resources and manage connections
6. **Logging**: Log important events and errors for debugging

## Next Steps

- **[Online Serving](online-serving.md)** - Learn about real-time inference
- **[Batching](batching.md)** - Optimize batch processing performance
- **Performance Tuning** - Advanced optimization techniques 