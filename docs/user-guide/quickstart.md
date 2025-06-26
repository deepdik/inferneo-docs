# Quickstart Guide

This guide will walk you through the essential concepts and common use cases for Inferneo.

## Basic Concepts

### Server Architecture

Inferneo follows a client-server architecture:

- **Inferneo Server**: Handles model loading, inference, and request management
- **Client**: Sends requests to the server and receives responses
- **Models**: AI models loaded into memory for inference

### Key Components

- **Model Engine**: Core inference engine with optimizations
- **Request Scheduler**: Manages request queuing and batching
- **Memory Manager**: Handles GPU memory allocation and optimization
- **API Server**: REST and WebSocket endpoints

## Starting the Server

### Basic Server Start

```bash
# Start with a Hugging Face model
inferneo serve --model meta-llama/Llama-2-7b-chat-hf

# Start with custom port
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --port 8080

# Start with specific GPU
inferneo serve --model meta-llama/Llama-2-7b-chat-hf --gpu 0
```

### Advanced Server Configuration

```bash
# Start with multiple models
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --model meta-llama/Llama-2-13b-chat-hf \
  --port 8000

# Start with custom configuration
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-batch-size 32 \
  --max-concurrent-requests 100 \
  --tensor-parallel-size 2
```

## Using the Python Client

### Basic Usage

```python
from inferneo import InferneoClient

# Create client
client = InferneoClient("http://localhost:8000")

# Text completion
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Chat Completion

```python
# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
# Streaming chat completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Using the REST API

### Text Completion

```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completion

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## Common Use Cases

### 1. Text Generation

```python
# Generate creative text
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Write a short story about a robot learning to paint:",
    max_tokens=200,
    temperature=0.8,
    top_p=0.9
)
```

### 2. Code Generation

```python
# Generate Python code
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a Python programming expert."},
        {"role": "user", "content": "Write a function to sort a list of dictionaries by a specific key"}
    ],
    max_tokens=150,
    temperature=0.3
)
```

### 3. Question Answering

```python
# Answer questions
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "Answer questions accurately and concisely."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50,
    temperature=0.1
)
```

### 4. Translation

```python
# Translate text
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": "Translate 'Hello, how are you?' to Spanish"}
    ],
    max_tokens=50,
    temperature=0.3
)
```

## Configuration Options

### Server Configuration

```bash
# Performance tuning
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --max-batch-size 16 \
  --max-concurrent-requests 50 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# Multi-GPU setup
inferneo serve \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

### Client Configuration

```python
# Configure client
client = InferneoClient(
    "http://localhost:8000",
    timeout=30,
    max_retries=3
)

# Batch requests
responses = []
for prompt in prompts:
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=100
    )
    responses.append(response)
```

## Monitoring and Debugging

### Server Status

```bash
# Check server health
curl http://localhost:8000/health

# Get server metrics
curl http://localhost:8000/metrics
```

### Python Monitoring

```python
# Get server info
info = client.info()
print(f"Models loaded: {info.models}")
print(f"GPU memory: {info.gpu_memory}")

# Get model info
model_info = client.models.get("meta-llama/Llama-2-7b-chat-hf")
print(f"Model parameters: {model_info.parameters}")
print(f"Model format: {model_info.format}")
```

## Best Practices

### 1. Model Selection
- Choose models appropriate for your use case
- Consider model size vs. performance trade-offs
- Use quantized models for better memory efficiency

### 2. Request Optimization
- Use appropriate batch sizes
- Implement request queuing for high load
- Use streaming for long responses

### 3. Resource Management
- Monitor GPU memory usage
- Implement proper error handling
- Use connection pooling for multiple clients

### 4. Security
- Implement authentication for production
- Use HTTPS for secure communication
- Validate input data

## Next Steps

Now that you understand the basics:

- **[Model Loading](model-loading.md)** - Learn about different model formats and loading strategies
- **[Batching](batching.md)** - Optimize throughput with request batching
- **[Streaming](streaming.md)** - Implement real-time response streaming
- **[Distributed Inference](distributed-inference.md)** - Scale across multiple GPUs and machines 