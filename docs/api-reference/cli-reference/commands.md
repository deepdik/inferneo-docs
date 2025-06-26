# CLI Commands Reference

This page documents the Inferneo command-line interface (CLI) commands.

## inferneo serve

Start the inference server.

```bash
inferneo serve --model meta-llama/Llama-2-7b-chat-hf [OPTIONS]
```

**Options:**
- `--model`: Model ID to load
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port (default: 8000)
- `--max-model-len`: Maximum context length
- `--gpu-memory-utilization`: Fraction of GPU memory to use
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--quantization`: Quantization method (awq, gptq, squeezellm)
- `--config`: Path to config file

## inferneo models

List available models.

```bash
inferneo models
```

## inferneo info

Show server and environment info.

```bash
inferneo info
```

## inferneo version

Show CLI version.

```bash
inferneo version
```

## Next Steps
- **[Environment Variables](environment-variables.md)** 