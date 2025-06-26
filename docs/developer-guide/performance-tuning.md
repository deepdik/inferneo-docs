# Performance Tuning

This page provides tips for optimizing Inferneo's performance.

## Hardware Recommendations
- Use GPUs with high memory and compute
- Prefer NVLink or InfiniBand for multi-GPU/multi-node

## Model Optimization
- Use quantized models (AWQ, GPTQ, SqueezeLLM)
- Reduce max sequence length if not needed
- Use batching to maximize throughput

## Server Configuration
- Tune `max_num_batched_tokens` and `max_num_seqs`
- Adjust `gpu_memory_utilization` for your hardware
- Use multiple server instances for load balancing

## Monitoring
- Use Prometheus/Grafana for metrics
- Monitor GPU utilization and memory

## Troubleshooting
- Check logs for errors
- Profile slow requests
- Test with different batch sizes

## Next Steps
- **[Batching](../user-guide/batching.md)**
- **[Distributed Inference](../user-guide/distributed-inference.md)** 