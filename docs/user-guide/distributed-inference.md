# Distributed Inference

Distributed inference allows you to scale Inferneo across multiple machines and GPUs to handle high-throughput workloads and large models.

## Overview

Distributed inference enables horizontal scaling by distributing model layers, attention heads, and requests across multiple nodes, providing:

- **Higher throughput** for production workloads
- **Larger model support** beyond single GPU memory limits
- **Fault tolerance** with redundant nodes
- **Cost optimization** through efficient resource utilization

## Architecture

### Tensor Parallelism

Tensor parallelism splits model layers across multiple GPUs:

```python
from inferneo import Inferneo

# Configure tensor parallelism across 4 GPUs
client = Inferneo("http://localhost:8000")

client.set_config(
    tensor_parallel_size=4,  # Split across 4 GPUs
    gpu_memory_utilization=0.8,
    max_model_len=4096
)

# Load large model with tensor parallelism
response = client.generate(
    "Explain distributed computing",
    model="meta-llama/Llama-2-70b-chat-hf"
)
```

### Pipeline Parallelism

Pipeline parallelism distributes model layers across different machines:

```python
from inferneo import Inferneo

# Configure pipeline parallelism
client = Inferneo("http://localhost:8000")

client.set_config(
    pipeline_parallel_size=2,  # 2 pipeline stages
    tensor_parallel_size=2,    # 2 GPUs per stage
    max_model_len=4096
)

# Load model with pipeline parallelism
response = client.generate(
    "Write a comprehensive guide to AI",
    model="meta-llama/Llama-2-70b-chat-hf"
)
```

## Multi-Node Setup

### Coordinator Configuration

```yaml
# coordinator.yaml
coordinator:
  host: "0.0.0.0"
  port: 8000
  workers:
    - host: "192.168.1.10"
      port: 8001
      gpus: [0, 1]
    - host: "192.168.1.11"
      port: 8002
      gpus: [0, 1]
    - host: "192.168.1.12"
      port: 8003
      gpus: [0, 1]

model:
  name: "meta-llama/Llama-2-70b-chat-hf"
  tensor_parallel_size: 6
  pipeline_parallel_size: 3
  max_model_len: 4096
```

### Worker Configuration

```yaml
# worker.yaml
worker:
  host: "0.0.0.0"
  port: 8001
  coordinator_host: "192.168.1.10"
  coordinator_port: 8000
  gpu_ids: [0, 1]
  model_cache_dir: "/path/to/model/cache"
```

### Starting Distributed Cluster

```bash
# Start coordinator
inferneo serve --config coordinator.yaml --role coordinator

# Start workers (on different machines)
inferneo serve --config worker.yaml --role worker
```

## Client Configuration

### Connecting to Distributed Cluster

```python
from inferneo import Inferneo

# Connect to coordinator
client = Inferneo("http://192.168.1.10:8000")

# Configure for distributed inference
client.set_config(
    tensor_parallel_size=6,
    pipeline_parallel_size=3,
    max_model_len=4096,
    gpu_memory_utilization=0.8
)

# Generate with distributed model
response = client.generate(
    "Explain the benefits of distributed computing",
    model="meta-llama/Llama-2-70b-chat-hf"
)
```

### Load Balancing

```python
from inferneo import Inferneo
import random

class LoadBalancedClient:
    def __init__(self, coordinator_hosts):
        self.coordinators = coordinator_hosts
        self.clients = [Inferneo(f"http://{host}") for host in coordinator_hosts]
    
    def generate(self, prompt, **kwargs):
        # Simple round-robin load balancing
        client = random.choice(self.clients)
        return client.generate(prompt, **kwargs)
    
    def generate_batch(self, prompts, **kwargs):
        # Distribute batch across coordinators
        results = []
        for i, prompt in enumerate(prompts):
            client = self.clients[i % len(self.clients)]
            result = client.generate(prompt, **kwargs)
            results.append(result)
        return results

# Usage
coordinator_hosts = [
    "192.168.1.10:8000",
    "192.168.1.11:8000",
    "192.168.1.12:8000"
]

lb_client = LoadBalancedClient(coordinator_hosts)
response = lb_client.generate("Explain load balancing")
```

## Performance Optimization

### Optimal Configuration

```python
from inferneo import Inferneo
import time
import psutil

def benchmark_distributed_configs():
    client = Inferneo("http://localhost:8000")
    
    # Test different configurations
    configs = [
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
        {"tensor_parallel_size": 4, "pipeline_parallel_size": 1},
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 2},
        {"tensor_parallel_size": 4, "pipeline_parallel_size": 2}
    ]
    
    test_prompt = "Write a detailed explanation of distributed systems"
    results = {}
    
    for config in configs:
        print(f"Testing config: {config}")
        
        # Apply configuration
        client.set_config(**config)
        
        # Measure performance
        start_time = time.time()
        response = client.generate(test_prompt)
        latency = time.time() - start_time
        
        # Measure memory usage
        memory_usage = psutil.virtual_memory().percent
        
        results[str(config)] = {
            "latency": latency,
            "memory_usage": memory_usage,
            "response_length": len(response.generated_text)
        }
    
    # Print results
    print("\nPerformance Comparison:")
    print("-" * 60)
    for config, metrics in results.items():
        print(f"{config}: {metrics['latency']:.3f}s, {metrics['memory_usage']:.1f}% memory")
    
    return results

# Run benchmark
benchmark_distributed_configs()
```

### Memory Management

```python
from inferneo import Inferneo
import gc

def memory_efficient_distributed():
    client = Inferneo("http://localhost:8000")
    
    # Configure for memory efficiency
    client.set_config(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.7,  # Conservative memory usage
        max_model_len=2048,  # Reduce context length
        swap_space=8  # Enable swap space
    )
    
    # Generate with memory monitoring
    import psutil
    memory_before = psutil.virtual_memory().used
    
    response = client.generate("Explain memory management in distributed systems")
    
    memory_after = psutil.virtual_memory().used
    memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
    
    print(f"Memory used: {memory_used:.1f} MB")
    
    # Force garbage collection
    gc.collect()
    
    return response

# Usage
response = memory_efficient_distributed()
```

## Fault Tolerance

### Health Monitoring

```python
import requests
import time
from inferneo import Inferneo

class FaultTolerantClient:
    def __init__(self, coordinator_hosts):
        self.coordinators = coordinator_hosts
        self.healthy_coordinators = coordinator_hosts.copy()
        self.clients = {}
        self._init_clients()
    
    def _init_clients(self):
        for host in self.coordinators:
            try:
                self.clients[host] = Inferneo(f"http://{host}")
                # Test connection
                self.clients[host].generate("test", max_tokens=1)
            except Exception as e:
                print(f"Failed to connect to {host}: {e}")
                if host in self.healthy_coordinators:
                    self.healthy_coordinators.remove(host)
    
    def _check_health(self, host):
        try:
            response = requests.get(f"http://{host}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt, **kwargs):
        # Try healthy coordinators
        for host in self.healthy_coordinators:
            try:
                return self.clients[host].generate(prompt, **kwargs)
            except Exception as e:
                print(f"Error with {host}: {e}")
                if not self._check_health(host):
                    self.healthy_coordinators.remove(host)
        
        # If all failed, try to reconnect
        self._init_clients()
        if self.healthy_coordinators:
            return self.generate(prompt, **kwargs)
        else:
            raise Exception("All coordinators are unavailable")

# Usage
coordinator_hosts = [
    "192.168.1.10:8000",
    "192.168.1.11:8000",
    "192.168.1.12:8000"
]

ft_client = FaultTolerantClient(coordinator_hosts)
response = ft_client.generate("Explain fault tolerance")
```

### Automatic Failover

```python
from inferneo import Inferneo
import asyncio
import aiohttp

class AutoFailoverClient:
    def __init__(self, coordinator_hosts):
        self.coordinators = coordinator_hosts
        self.current_coordinator = 0
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate(self, prompt, **kwargs):
        session = await self._get_session()
        
        for attempt in range(len(self.coordinators)):
            coordinator = self.coordinators[self.current_coordinator]
            
            try:
                async with session.post(
                    f"http://{coordinator}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": kwargs.get("max_tokens", 100),
                        "temperature": kwargs.get("temperature", 0.7)
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["text"]
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
            except Exception as e:
                print(f"Failed with {coordinator}: {e}")
                # Move to next coordinator
                self.current_coordinator = (self.current_coordinator + 1) % len(self.coordinators)
        
        raise Exception("All coordinators failed")
    
    async def close(self):
        if self.session:
            await self.session.close()

# Usage
async def main():
    coordinator_hosts = [
        "192.168.1.10:8000",
        "192.168.1.11:8000",
        "192.168.1.12:8000"
    ]
    
    client = AutoFailoverClient(coordinator_hosts)
    
    try:
        response = await client.generate("Explain automatic failover")
        print(response)
    finally:
        await client.close()

asyncio.run(main())
```

## Monitoring and Metrics

### Performance Metrics

```python
import time
import psutil
from collections import defaultdict
from inferneo import Inferneo

class DistributedMetrics:
    def __init__(self):
        self.latency_history = defaultdict(list)
        self.throughput_history = []
        self.memory_history = []
        self.error_count = defaultdict(int)
    
    def record_request(self, coordinator, latency, success=True):
        self.latency_history[coordinator].append(latency)
        
        if not success:
            self.error_count[coordinator] += 1
    
    def record_throughput(self, requests_per_second):
        self.throughput_history.append(requests_per_second)
    
    def record_memory(self, memory_mb):
        self.memory_history.append(memory_mb)
    
    def get_stats(self):
        stats = {}
        
        # Latency stats per coordinator
        for coordinator, latencies in self.latency_history.items():
            if latencies:
                stats[f"{coordinator}_avg_latency"] = sum(latencies) / len(latencies)
                stats[f"{coordinator}_min_latency"] = min(latencies)
                stats[f"{coordinator}_max_latency"] = max(latencies)
        
        # Overall stats
        if self.throughput_history:
            stats["avg_throughput"] = sum(self.throughput_history) / len(self.throughput_history)
        
        if self.memory_history:
            stats["avg_memory"] = sum(self.memory_history) / len(self.memory_history)
        
        # Error rates
        total_requests = sum(len(latencies) for latencies in self.latency_history.values())
        total_errors = sum(self.error_count.values())
        if total_requests > 0:
            stats["error_rate"] = total_errors / total_requests
        
        return stats

# Usage
metrics = DistributedMetrics()
client = Inferneo("http://localhost:8000")

# Simulate requests and record metrics
for i in range(100):
    start_time = time.time()
    try:
        response = client.generate(f"Test prompt {i}")
        latency = time.time() - start_time
        metrics.record_request("coordinator1", latency, success=True)
    except Exception as e:
        latency = time.time() - start_time
        metrics.record_request("coordinator1", latency, success=False)
    
    # Record memory usage
    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
    metrics.record_memory(memory_mb)

print("Metrics:", metrics.get_stats())
```

### Resource Monitoring

```python
import psutil
import GPUtil
from inferneo import Inferneo

def monitor_distributed_resources():
    client = Inferneo("http://localhost:8000")
    
    # Monitor system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Monitor GPU resources
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": gpu.memoryUtil * 100,
                "temperature": gpu.temperature,
                "load": gpu.load * 100
            })
    except:
        pass
    
    # Monitor network
    network = psutil.net_io_counters()
    
    # Monitor disk
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "gpu_info": gpu_info,
        "network_bytes_sent": network.bytes_sent,
        "network_bytes_recv": network.bytes_recv,
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3)
    }

# Continuous monitoring
import time

def continuous_monitoring(duration_seconds=300):
    start_time = time.time()
    metrics_history = []
    
    while time.time() - start_time < duration_seconds:
        metrics = monitor_distributed_resources()
        metrics_history.append(metrics)
        
        print(f"CPU: {metrics['cpu_percent']:.1f}%")
        print(f"Memory: {metrics['memory_percent']:.1f}%")
        print(f"GPU Memory: {[gpu['memory_percent'] for gpu in metrics['gpu_info']]}")
        print("-" * 40)
        
        time.sleep(10)  # Monitor every 10 seconds
    
    return metrics_history

# Run monitoring
# metrics_history = continuous_monitoring(60)  # Monitor for 1 minute
```

## Best Practices

### Configuration Guidelines

```python
# For high-throughput production
production_config = {
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 2,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 4096,
    "max_batch_size": 32
}

# For memory-constrained environments
memory_constrained_config = {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "gpu_memory_utilization": 0.6,
    "max_model_len": 2048,
    "max_batch_size": 16
}

# For development/testing
dev_config = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "gpu_memory_utilization": 0.7,
    "max_model_len": 1024,
    "max_batch_size": 8
}
```

### Network Optimization

```python
# Optimize network settings for distributed inference
import socket

def optimize_network():
    # Set socket options for better performance
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Set buffer sizes
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB
    
    return sock

# Configure client with optimized network settings
client = Inferneo("http://localhost:8000")
client.set_network_config(
    timeout=30,
    max_retries=3,
    connection_pool_size=10
)
```

## Troubleshooting

### Common Issues

**Network Connectivity**
```python
# Test network connectivity between nodes
import subprocess

def test_connectivity(hosts):
    for host in hosts:
        try:
            result = subprocess.run(
                ["ping", "-c", "3", host.split(":")[0]], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"✓ {host} is reachable")
            else:
                print(f"✗ {host} is not reachable")
        except Exception as e:
            print(f"✗ Error testing {host}: {e}")

# Test connectivity
hosts = ["192.168.1.10:8000", "192.168.1.11:8000", "192.168.1.12:8000"]
test_connectivity(hosts)
```

**Memory Issues**
```python
# Monitor and resolve memory issues
def resolve_memory_issues():
    client = Inferneo("http://localhost:8000")
    
    # Reduce memory usage
    client.set_config(
        gpu_memory_utilization=0.6,  # Reduce GPU memory usage
        max_model_len=1024,  # Reduce context length
        max_batch_size=8  # Reduce batch size
    )
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Monitor memory
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        print("Warning: High memory usage detected")
        return False
    
    return True
```

**Load Balancing Issues**
```python
# Implement health checks for load balancer
def health_check(host):
    try:
        response = requests.get(f"http://{host}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def update_load_balancer(hosts):
    healthy_hosts = [host for host in hosts if health_check(host)]
    if not healthy_hosts:
        raise Exception("No healthy hosts available")
    return healthy_hosts
```

For more advanced distributed computing techniques, see the Performance Tuning guide. 