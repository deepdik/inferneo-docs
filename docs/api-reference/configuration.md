# Configuration

Inferneo can be configured through various methods including configuration files, environment variables, and command-line arguments.

## Configuration File

### YAML Configuration

Create a `config.yaml` file in your project directory:

```yaml
# Server configuration
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  max_request_size: "10MB"
  timeout: 300

# Model configuration
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  max_model_len: 4096
  gpu_memory_utilization: 0.8
  swap_space: 4
  quantization: "awq"
  quantization_config:
    bits: 4
    group_size: 128
    zero_point: true
    scale: true

# Inference configuration
inference:
  max_batch_size: 32
  max_batch_tokens: 4096
  max_num_seqs: 256
  max_num_batched_tokens: 4096
  max_paddings: 256
  max_num_seqs_per_prompt: 1
  max_num_batched_tokens_per_prompt: 2048
  max_num_seqs_per_batch: 32
  max_num_batched_tokens_per_batch: 4096

# Performance configuration
performance:
  max_concurrent_requests: 100
  max_concurrent_streaming_requests: 50
  max_concurrent_batched_requests: 10
  max_concurrent_batched_streaming_requests: 5
  max_concurrent_batched_tokens: 4096
  max_concurrent_batched_streaming_tokens: 2048

# Logging configuration
logging:
  level: "INFO"
  format: "json"
  file: "inferneo.log"
  max_size: "100MB"
  backup_count: 5

# Security configuration
security:
  api_key: "your-api-key"
  allowed_origins: ["*"]
  rate_limit:
    requests_per_minute: 60
    tokens_per_minute: 150000
    requests_per_day: 3500

# Monitoring configuration
monitoring:
  enabled: true
  metrics_port: 8001
  health_check_interval: 30
  prometheus_enabled: true
```

### JSON Configuration

Alternatively, use JSON format:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"],
    "max_request_size": "10MB",
    "timeout": 300
  },
  "model": {
    "name": "meta-llama/Llama-2-7b-chat-hf",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.8,
    "swap_space": 4,
    "quantization": "awq",
    "quantization_config": {
      "bits": 4,
      "group_size": 128,
      "zero_point": true,
      "scale": true
    }
  },
  "inference": {
    "max_batch_size": 32,
    "max_batch_tokens": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 4096,
    "max_paddings": 256
  },
  "performance": {
    "max_concurrent_requests": 100,
    "max_concurrent_streaming_requests": 50,
    "max_concurrent_batched_requests": 10
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "inferneo.log"
  },
  "security": {
    "api_key": "your-api-key",
    "allowed_origins": ["*"],
    "rate_limit": {
      "requests_per_minute": 60,
      "tokens_per_minute": 150000,
      "requests_per_day": 3500
    }
  },
  "monitoring": {
    "enabled": true,
    "metrics_port": 8001,
    "health_check_interval": 30,
    "prometheus_enabled": true
  }
}
```

## Environment Variables

Configure Inferneo using environment variables:

```bash
# Server configuration
export INFERNEO_HOST="0.0.0.0"
export INFERNEO_PORT=8000
export INFERNEO_CORS_ORIGINS="*"
export INFERNEO_MAX_REQUEST_SIZE="10MB"
export INFERNEO_TIMEOUT=300

# Model configuration
export INFERNEO_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export INFERNEO_TENSOR_PARALLEL_SIZE=1
export INFERNEO_PIPELINE_PARALLEL_SIZE=1
export INFERNEO_MAX_MODEL_LEN=4096
export INFERNEO_GPU_MEMORY_UTILIZATION=0.8
export INFERNEO_SWAP_SPACE=4
export INFERNEO_QUANTIZATION="awq"
export INFERNEO_QUANTIZATION_BITS=4
export INFERNEO_QUANTIZATION_GROUP_SIZE=128

# Inference configuration
export INFERNEO_MAX_BATCH_SIZE=32
export INFERNEO_MAX_BATCH_TOKENS=4096
export INFERNEO_MAX_NUM_SEQS=256
export INFERNEO_MAX_NUM_BATCHED_TOKENS=4096

# Performance configuration
export INFERNEO_MAX_CONCURRENT_REQUESTS=100
export INFERNEO_MAX_CONCURRENT_STREAMING_REQUESTS=50
export INFERNEO_MAX_CONCURRENT_BATCHED_REQUESTS=10

# Logging configuration
export INFERNEO_LOG_LEVEL="INFO"
export INFERNEO_LOG_FORMAT="json"
export INFERNEO_LOG_FILE="inferneo.log"

# Security configuration
export INFERNEO_API_KEY="your-api-key"
export INFERNEO_ALLOWED_ORIGINS="*"
export INFERNEO_RATE_LIMIT_REQUESTS_PER_MINUTE=60
export INFERNEO_RATE_LIMIT_TOKENS_PER_MINUTE=150000
export INFERNEO_RATE_LIMIT_REQUESTS_PER_DAY=3500

# Monitoring configuration
export INFERNEO_MONITORING_ENABLED=true
export INFERNEO_METRICS_PORT=8001
export INFERNEO_HEALTH_CHECK_INTERVAL=30
export INFERNEO_PROMETHEUS_ENABLED=true
```

## Command-Line Arguments

Start Inferneo with command-line arguments:

```bash
# Basic server start
inferneo serve --model meta-llama/Llama-2-7b-chat-hf

# With configuration file
inferneo serve --config config.yaml

# With command-line arguments
inferneo serve \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --quantization awq \
  --max-batch-size 32 \
  --max-concurrent-requests 100

# With environment variables
INFERNEO_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf" \
INFERNEO_TENSOR_PARALLEL_SIZE=1 \
INFERNEO_MAX_MODEL_LEN=4096 \
inferneo serve
```

## Configuration Options

### Server Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | string | "0.0.0.0" | Server host address |
| `port` | integer | 8000 | Server port |
| `cors_origins` | array | ["*"] | Allowed CORS origins |
| `max_request_size` | string | "10MB" | Maximum request size |
| `timeout` | integer | 300 | Request timeout in seconds |

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | required | Model name or path |
| `tensor_parallel_size` | integer | 1 | Number of GPUs for tensor parallelism |
| `pipeline_parallel_size` | integer | 1 | Number of pipeline stages |
| `max_model_len` | integer | 4096 | Maximum sequence length |
| `gpu_memory_utilization` | float | 0.8 | GPU memory utilization ratio |
| `swap_space` | integer | 4 | Swap space in GB |
| `quantization` | string | null | Quantization method (awq, gptq, squeezellm) |
| `quantization_config` | object | {} | Quantization parameters |

### Inference Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_batch_size` | integer | 32 | Maximum batch size |
| `max_batch_tokens` | integer | 4096 | Maximum tokens per batch |
| `max_num_seqs` | integer | 256 | Maximum number of sequences |
| `max_num_batched_tokens` | integer | 4096 | Maximum batched tokens |
| `max_paddings` | integer | 256 | Maximum padding tokens |

### Performance Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_concurrent_requests` | integer | 100 | Maximum concurrent requests |
| `max_concurrent_streaming_requests` | integer | 50 | Maximum streaming requests |
| `max_concurrent_batched_requests` | integer | 10 | Maximum batched requests |
| `max_concurrent_batched_streaming_requests` | integer | 5 | Maximum batched streaming requests |

### Logging Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | "INFO" | Log level (DEBUG, INFO, WARNING, ERROR) |
| `format` | string | "json" | Log format (json, text) |
| `file` | string | null | Log file path |
| `max_size` | string | "100MB" | Maximum log file size |
| `backup_count` | integer | 5 | Number of backup files |

### Security Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | string | null | API key for authentication |
| `allowed_origins` | array | ["*"] | Allowed origins for CORS |
| `rate_limit` | object | {} | Rate limiting configuration |

### Monitoring Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | true | Enable monitoring |
| `metrics_port` | integer | 8001 | Metrics server port |
| `health_check_interval` | integer | 30 | Health check interval in seconds |
| `prometheus_enabled` | boolean | true | Enable Prometheus metrics |

## Configuration Validation

### Schema Validation

Inferneo validates configuration using JSON Schema:

```python
from inferneo import validate_config

# Validate configuration
config = {
    "model": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "tensor_parallel_size": 1
    }
}

try:
    validate_config(config)
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Configuration Testing

Test your configuration before deployment:

```bash
# Validate configuration file
inferneo validate-config --config config.yaml

# Test configuration with dry run
inferneo serve --config config.yaml --dry-run

# Check configuration and exit
inferneo serve --config config.yaml --check-config
```

## Configuration Examples

### Development Configuration

```yaml
# config-dev.yaml
server:
  host: "localhost"
  port: 8000
  cors_origins: ["http://localhost:3000"]

model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  tensor_parallel_size: 1
  max_model_len: 2048
  gpu_memory_utilization: 0.7

inference:
  max_batch_size: 8
  max_batch_tokens: 2048

logging:
  level: "DEBUG"
  format: "text"

security:
  api_key: "dev-api-key"
```

### Production Configuration

```yaml
# config-prod.yaml
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["https://yourdomain.com"]
  max_request_size: "50MB"
  timeout: 600

model:
  name: "meta-llama/Llama-2-70b-chat-hf"
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  max_model_len: 8192
  gpu_memory_utilization: 0.9
  quantization: "awq"
  quantization_config:
    bits: 4
    group_size: 128

inference:
  max_batch_size: 64
  max_batch_tokens: 8192
  max_num_seqs: 512

performance:
  max_concurrent_requests: 200
  max_concurrent_streaming_requests: 100

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/inferneo.log"

security:
  api_key: "${INFERNEO_API_KEY}"
  rate_limit:
    requests_per_minute: 120
    tokens_per_minute: 300000

monitoring:
  enabled: true
  metrics_port: 8001
  prometheus_enabled: true
```

### Multi-GPU Configuration

```yaml
# config-multi-gpu.yaml
model:
  name: "meta-llama/Llama-2-70b-chat-hf"
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  max_model_len: 8192
  gpu_memory_utilization: 0.85

inference:
  max_batch_size: 128
  max_batch_tokens: 16384

performance:
  max_concurrent_requests: 500
  max_concurrent_streaming_requests: 200
  max_concurrent_batched_requests: 50

monitoring:
  enabled: true
  metrics_port: 8001
  health_check_interval: 15
```

### Memory-Constrained Configuration

```yaml
# config-memory-constrained.yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  tensor_parallel_size: 1
  max_model_len: 1024
  gpu_memory_utilization: 0.6
  quantization: "awq"
  quantization_config:
    bits: 4
    group_size: 64

inference:
  max_batch_size: 4
  max_batch_tokens: 1024
  max_num_seqs: 64

performance:
  max_concurrent_requests: 20
  max_concurrent_streaming_requests: 10
```

## Configuration Management

### Environment-Specific Configuration

```python
import os
from pathlib import Path

def load_config():
    env = os.getenv("INFERNEO_ENV", "development")
    config_path = Path(f"config-{env}.yaml")
    
    if config_path.exists():
        return load_yaml_config(config_path)
    else:
        return load_default_config()

# Usage
config = load_config()
```

### Configuration Inheritance

```yaml
# base.yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  tensor_parallel_size: 1
  max_model_len: 4096

# development.yaml
extends: base.yaml

server:
  host: "localhost"

model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  max_model_len: 2048

# production.yaml
extends: base.yaml

model:
  name: "meta-llama/Llama-2-70b-chat-hf"
  tensor_parallel_size: 4
  max_model_len: 8192
```

### Dynamic Configuration

```python
from inferneo import InferneoClient

# Update configuration at runtime
client = InferneoClient("http://localhost:8000")

# Update model configuration
client.update_config({
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9
})

# Update inference configuration
client.update_inference_config({
    "max_batch_size": 64,
    "max_batch_tokens": 8192
})
```

## Troubleshooting

### Common Configuration Issues

**Memory Issues**
```yaml
# Reduce memory usage
model:
  gpu_memory_utilization: 0.6
  max_model_len: 1024
  quantization: "awq"

inference:
  max_batch_size: 4
  max_batch_tokens: 1024
```

**Performance Issues**
```yaml
# Optimize for performance
model:
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.9

inference:
  max_batch_size: 64
  max_batch_tokens: 8192

performance:
  max_concurrent_requests: 200
```

**Network Issues**
```yaml
# Configure for network stability
server:
  timeout: 600
  max_request_size: "50MB"

performance:
  max_concurrent_requests: 50
  max_concurrent_streaming_requests: 25
```

For more information about server configuration, see the server configuration guide. 