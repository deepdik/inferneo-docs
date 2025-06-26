# Configuration Reference

This page documents the configuration options for Inferneo.

## Configuration File

Inferneo can be configured using a YAML file or command-line arguments.

### Example config.yaml

```yaml
model: meta-llama/Llama-2-7b-chat-hf
host: 0.0.0.0
port: 8000
max_model_len: 4096
gpu_memory_utilization: 0.9
tensor_parallel_size: 1
max_num_batched_tokens: 4096
max_num_seqs: 256
quantization: awq
```

## Command-Line Arguments

- `--model`: Model ID
- `--host`: Host address
- `--port`: Port number
- `--max-model-len`: Maximum model context length
- `--gpu-memory-utilization`: Fraction of GPU memory to use
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--max-num-batched-tokens`: Max tokens per batch
- `--max-num-seqs`: Max sequences per batch
- `--quantization`: Quantization method (awq, gptq, squeezellm)

## Environment Variables

- `INFERNEO_MODEL`
- `INFERNEO_HOST`
- `INFERNEO_PORT`
- `INFERNEO_QUANTIZATION`

## Next Steps
- **[REST API](rest-api.md)**
- **[WebSocket API](websocket-api.md)** 