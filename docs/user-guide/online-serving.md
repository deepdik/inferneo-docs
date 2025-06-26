# Online Serving

This guide covers how to deploy and use Inferneo for online serving with real-time inference capabilities.

## Overview

Online serving enables real-time inference for applications that require immediate responses, such as:

- **Chat applications** and conversational AI
- **Web services** and APIs
- **Real-time decision making** systems
- **Interactive applications** and demos

## Server Setup

### Starting the Server

```bash
# Basic server startup
inferneo serve --model meta-llama/Llama-2-7b-chat-hf

# With custom configuration
inferneo serve \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

### Configuration Options

```yaml
# config.yaml
model: meta-llama/Llama-2-7b-chat-hf
host: 0.0.0.0
port: 8000
max_model_len: 4096
gpu_memory_utilization: 0.9
tensor_parallel_size: 1
max_num_batched_tokens: 4096
max_num_seqs: 256
```

```bash
# Using configuration file
inferneo serve --config config.yaml
```

## Client Usage

### Python Client

```python
from inferneo import InferneoClient

# Initialize client
client = InferneoClient("http://localhost:8000")

# Basic completion
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Explain quantum computing",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Chat Completions

```python
# Chat completion with conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
    {"role": "user", "content": "Can you give me an example?"}
]

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=messages,
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
# Streaming for real-time responses
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    max_tokens=200,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## REST API

### Basic Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "Explain artificial intelligence",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

### Advanced API Usage

```python
import requests
import json

# Function to make API calls
def inferneo_api_call(prompt, model="meta-llama/Llama-2-7b-chat-hf"):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Usage
result = inferneo_api_call("Explain quantum computing")
print(result["choices"][0]["text"])
```

## WebSocket API

### Real-time Communication

```python
import asyncio
import websockets
import json

async def chat_with_websocket():
    uri = "ws://localhost:8000/v1/chat/completions"
    
    async with websockets.connect(uri) as websocket:
        # Send message
        message = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        }
        
        await websocket.send(json.dumps(message))
        
        # Receive streaming response
        async for response in websocket:
            data = json.loads(response)
            if "choices" in data and len(data["choices"]) > 0:
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)

# Run the async function
asyncio.run(chat_with_websocket())
```

## Performance Optimization

### Request Batching

```python
# Batch multiple requests for better throughput
def batch_requests(prompts, model_id):
    batch_data = {
        "model": model_id,
        "prompts": prompts,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = requests.post(
        "http://localhost:8000/v1/completions/batch",
        json=batch_data
    )
    
    return response.json()

# Usage
prompts = [
    "Explain machine learning",
    "What is deep learning?",
    "Describe neural networks"
]

results = batch_requests(prompts, "meta-llama/Llama-2-7b-chat-hf")
```

### Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create session with connection pooling
session = requests.Session()

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Use session for multiple requests
def make_request(prompt):
    data = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": prompt,
        "max_tokens": 100
    }
    
    response = session.post(
        "http://localhost:8000/v1/completions",
        json=data
    )
    return response.json()
```

## Load Balancing

### Multiple Server Instances

```python
import random
from typing import List

class LoadBalancedClient:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current_server = 0
    
    def get_next_server(self):
        # Round-robin load balancing
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        return server
    
    def request(self, prompt):
        server = self.get_next_server()
        url = f"{server}/v1/completions"
        
        data = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "prompt": prompt,
            "max_tokens": 100
        }
        
        response = requests.post(url, json=data)
        return response.json()

# Usage
servers = [
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
]

client = LoadBalancedClient(servers)
result = client.request("Explain AI")
```

## Monitoring and Health Checks

### Health Monitoring

```python
import time
import requests

def monitor_server_health(server_url, interval=30):
    """Monitor server health and performance."""
    
    while True:
        try:
            # Health check
            health_response = requests.get(f"{server_url}/health")
            health_status = health_response.json()
            
            # Performance metrics
            metrics_response = requests.get(f"{server_url}/metrics")
            metrics = metrics_response.json()
            
            print(f"Health: {health_status['status']}")
            print(f"Active requests: {metrics.get('active_requests', 0)}")
            print(f"Queue size: {metrics.get('queue_size', 0)}")
            print(f"GPU utilization: {metrics.get('gpu_utilization', 0)}%")
            
        except Exception as e:
            print(f"Error monitoring server: {e}")
        
        time.sleep(interval)

# Usage
monitor_server_health("http://localhost:8000")
```

### Error Handling

```python
def robust_request(prompt, max_retries=3):
    """Make requests with robust error handling."""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/completions",
                json={
                    "model": "meta-llama/Llama-2-7b-chat-hf",
                    "prompt": prompt,
                    "max_tokens": 100
                },
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            else:
                time.sleep(2 ** attempt)  # Exponential backoff
```

## Security Considerations

### Authentication

```python
# API key authentication
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers=headers,
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Hello",
        "max_tokens": 100
    }
)
```

### Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def can_make_request(self):
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False

# Usage
rate_limiter = RateLimiter(max_requests=10, time_window=60)  # 10 requests per minute

def rate_limited_request(prompt):
    if rate_limiter.can_make_request():
        return make_request(prompt)
    else:
        raise Exception("Rate limit exceeded")
```

## Deployment Examples

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["inferneo", "serve", "--model", "meta-llama/Llama-2-7b-chat-hf", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t inferneo-server .
docker run -p 8000:8000 --gpus all inferneo-server
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inferneo-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inferneo-server
  template:
    metadata:
      labels:
        app: inferneo-server
    spec:
      containers:
      - name: inferneo
        image: inferneo-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "4"
```

## Best Practices

1. **Connection Management**: Use connection pooling for multiple requests
2. **Error Handling**: Implement retries and graceful error handling
3. **Monitoring**: Monitor server health and performance metrics
4. **Load Balancing**: Distribute load across multiple server instances
5. **Security**: Implement authentication and rate limiting
6. **Caching**: Cache frequently requested responses when appropriate

## Next Steps

- **[Offline Inference](offline-inference.md)** - Learn about batch processing
- **[Batching](batching.md)** - Optimize request batching
- **Performance Tuning** - Advanced optimization techniques 