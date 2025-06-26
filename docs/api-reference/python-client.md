# Python Client API Reference

The Inferneo Python client provides a simple and intuitive interface for interacting with Inferneo servers.

## Installation

```bash
pip install inferneo
```

## Quick Start

```python
from inferneo import InferneoClient

# Create client
client = InferneoClient("http://localhost:8000")

# Make a request
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Client Configuration

### InferneoClient

```python
InferneoClient(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None
)
```

**Parameters:**

- `base_url` (str): The base URL of the Inferneo server
- `api_key` (str, optional): API key for authentication
- `timeout` (float): Request timeout in seconds
- `max_retries` (int): Maximum number of retry attempts
- `headers` (dict, optional): Additional headers to include in requests

**Example:**

```python
client = InferneoClient(
    base_url="https://api.inferneo.ai",
    api_key="your-api-key",
    timeout=60.0,
    max_retries=5
)
```

## Completions API

### Text Completions

```python
client.completions.create(
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stream: bool = False,
    **kwargs
) -> CompletionResponse
```

**Parameters:**

- `model` (str): The model to use for completion
- `prompt` (str): The input prompt
- `max_tokens` (int, optional): Maximum number of tokens to generate
- `temperature` (float, optional): Controls randomness (0.0 to 2.0)
- `top_p` (float, optional): Nucleus sampling parameter (0.0 to 1.0)
- `top_k` (int, optional): Top-k sampling parameter
- `frequency_penalty` (float, optional): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (float, optional): Presence penalty (-2.0 to 2.0)
- `stop` (str or list, optional): Stop sequences
- `stream` (bool): Whether to stream the response

**Example:**

```python
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="The future of artificial intelligence is",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

print(response.choices[0].text)
```

### Streaming Completions

```python
stream = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Write a story about a robot:",
    stream=True
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="")
```

## Chat Completions API

### Chat Completions

```python
client.chat.completions.create(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stream: bool = False,
    **kwargs
) -> ChatCompletionResponse
```

**Parameters:**

- `model` (str): The model to use for completion
- `messages` (list): List of message dictionaries with 'role' and 'content'
- `max_tokens` (int, optional): Maximum number of tokens to generate
- `temperature` (float, optional): Controls randomness (0.0 to 2.0)
- `top_p` (float, optional): Nucleus sampling parameter (0.0 to 1.0)
- `top_k` (int, optional): Top-k sampling parameter
- `frequency_penalty` (float, optional): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (float, optional): Presence penalty (-2.0 to 2.0)
- `stop` (str or list, optional): Stop sequences
- `stream` (bool): Whether to stream the response

**Example:**

```python
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

### Streaming Chat Completions

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Models API

### List Models

```python
client.models.list() -> List[Model]
```

**Example:**

```python
models = client.models.list()
for model in models:
    print(f"Model: {model.id}")
    print(f"Parameters: {model.parameters}")
    print(f"Format: {model.format}")
```

### Get Model

```python
client.models.get(model_id: str) -> Model
```

**Example:**

```python
model = client.models.get("meta-llama/Llama-2-7b-chat-hf")
print(f"Model ID: {model.id}")
print(f"Parameters: {model.parameters}")
print(f"Format: {model.format}")
```

## Server Information

### Get Server Info

```python
client.info() -> ServerInfo
```

**Example:**

```python
info = client.info()
print(f"Server version: {info.version}")
print(f"Models loaded: {info.models}")
print(f"GPU memory: {info.gpu_memory}")
```

## Error Handling

### Exception Types

```python
from inferneo import InferneoError, InferneoAPIError, InferneoTimeoutError

try:
    response = client.chat.completions.create(
        model="invalid-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except InferneoAPIError as e:
    print(f"API Error: {e.message}")
    print(f"Status Code: {e.status_code}")
except InferneoTimeoutError as e:
    print(f"Timeout Error: {e.message}")
except InferneoError as e:
    print(f"General Error: {e.message}")
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_request_with_retry(client, messages):
    return client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=messages
    )
```

## Advanced Usage

### Batch Processing

```python
import asyncio
from inferneo import AsyncInferneoClient

async def batch_process(client, prompts):
    tasks = []
    for prompt in prompts:
        task = client.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            prompt=prompt,
            max_tokens=100
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    return responses

# Usage
async_client = AsyncInferneoClient("http://localhost:8000")
prompts = ["Hello", "How are you?", "What is AI?"]
responses = await batch_process(async_client, prompts)
```

### Custom Headers

```python
client = InferneoClient(
    base_url="http://localhost:8000",
    headers={
        "User-Agent": "MyApp/1.0",
        "X-Custom-Header": "custom-value"
    }
)
```

### Request Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

client = InferneoClient("http://localhost:8000")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Response Objects

### CompletionResponse

```python
class CompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class CompletionChoice:
    text: str
    index: int
    logprobs: Optional[LogProbs]
    finish_reason: str
```

### ChatCompletionResponse

```python
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ChatCompletionChoice:
    message: ChatMessage
    index: int
    finish_reason: str

class ChatMessage:
    role: str
    content: str
```

### Model

```python
class Model:
    id: str
    object: str
    created: int
    parameters: int
    format: str
```

### ServerInfo

```python
class ServerInfo:
    version: str
    models: List[str]
    gpu_memory: Dict[str, float]
    uptime: float
```

## Best Practices

1. **Connection Pooling**: Reuse client instances for multiple requests
2. **Error Handling**: Always handle exceptions appropriately
3. **Streaming**: Use streaming for long responses to improve user experience
4. **Batching**: Group requests when possible for better performance
5. **Monitoring**: Monitor response times and error rates
6. **Caching**: Cache responses when appropriate
7. **Rate Limiting**: Implement rate limiting for high-volume applications 