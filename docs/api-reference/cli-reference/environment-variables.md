# CLI Environment Variables Reference

This page documents the environment variables for Inferneo CLI and server.

## Environment Variables

- `INFERNEO_MODEL`: Default model ID
- `INFERNEO_HOST`: Default host address
- `INFERNEO_PORT`: Default port
- `INFERNEO_QUANTIZATION`: Default quantization method
- `INFERNEO_CONFIG`: Path to config file
- `INFERNEO_LOG_LEVEL`: Logging level (e.g., INFO, DEBUG)
- `INFERNEO_API_KEY`: API key for authentication (if required)

## Usage Example

```bash
export INFERNEO_MODEL=meta-llama/Llama-2-7b-chat-hf
export INFERNEO_HOST=0.0.0.0
export INFERNEO_PORT=8000
export INFERNEO_QUANTIZATION=awq
```

## Next Steps
- **[Commands](commands.md)** 