# Distributed Inference

This guide covers how to scale Inferneo for distributed inference across multiple GPUs or nodes.

## Overview

Distributed inference enables you to:
- Serve larger models that don't fit on a single GPU
- Increase throughput by parallelizing requests
- Achieve high availability and fault tolerance

## Multi-GPU Inference

### Tensor Parallelism

Inferneo supports tensor parallelism for large models:

```bash
inferneo serve --model meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4
```

- `--tensor-parallel-size`: Number of GPUs to split the model across

### Pipeline Parallelism

Pipeline parallelism splits model layers across devices (planned for future releases).

## Multi-Node Inference

### Launching on Multiple Nodes

Use a cluster manager (e.g., Kubernetes, SLURM) to launch Inferneo on multiple nodes. Each node should be configured with the same model and appropriate parallelism settings.

## Load Balancing

Deploy a load balancer (e.g., NGINX, HAProxy) in front of multiple Inferneo instances to distribute requests.

## Example: Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inferneo-distributed
spec:
  replicas: 4
  selector:
    matchLabels:
      app: inferneo
  template:
    metadata:
      labels:
        app: inferneo
    spec:
      containers:
      - name: inferneo
        image: inferneo-server:latest
        args: ["serve", "--model", "meta-llama/Llama-2-70b-chat-hf", "--tensor-parallel-size", "4"]
        resources:
          limits:
            nvidia.com/gpu: 4
```

## Monitoring and Scaling

- Use Prometheus/Grafana for monitoring
- Use Kubernetes HPA for autoscaling

## Best Practices

- Use high-speed interconnects (NVLink, InfiniBand) for multi-GPU/multi-node
- Monitor GPU utilization and memory
- Test failover and recovery

## Next Steps
- **[Performance Tuning](performance-tuning.md)**
- **[Batching](batching.md)**
- **[Online Serving](online-serving.md)** 