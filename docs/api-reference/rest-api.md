# REST API

Inferneo provides a comprehensive REST API for text generation, chat completions, and model management.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

Most endpoints require API key authentication:

```bash
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/v1/models
```

## Models

### List Models

Get a list of available models.

**Endpoint:** `GET /models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-2-7b-chat-hf",
      "object": "model",
      "created": 1640995200,
      "owned_by": "inferneo"
    }
  ]
}
```

### Get Model

Get information about a specific model.

**Endpoint:** `GET /models/{model_id}`

**Response:**
```json
{
  "id": "meta-llama/Llama-2-7b-chat-hf",
  "object": "model",
  "created": 1640995200,
  "owned_by": "inferneo",
  "permission": [],
  "root": "meta-llama/Llama-2-7b-chat-hf",
  "parent": null
}
```

## Completions

### Create Completion

Generate text completions.

**Endpoint:** `POST /completions`

**Request Body:**
```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "prompt": "Explain quantum computing",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n", "END"],
  "stream": false
}
```

**Response:**
```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1640995200,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "choices": [
    {
      "text": "Quantum computing is a revolutionary technology...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 100,
    "total_tokens": 104
  }
}
```

### Streaming Completions

Get streaming completions for real-time text generation.

**Request Body:**
```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "prompt": "Write a story about AI",
  "max_tokens": 200,
  "temperature": 0.8,
  "stream": true
}
```

**Response (Server-Sent Events):**
```
data: {"id":"cmpl-123","object":"text_completion","created":1640995200,"choices":[{"text":"Once","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-123","object":"text_completion","created":1640995200,"choices":[{"text":" upon","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-123","object":"text_completion","created":1640995200,"choices":[{"text":" a","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-123","object":"text_completion","created":1640995200,"choices":[{"text":" time","index":0,"logprobs":null,"finish_reason":null}]}

data: [DONE]
```

## Chat Completions

### Create Chat Completion

Generate chat-style completions with conversation history.

**Endpoint:** `POST /chat/completions`

**Request Body:**
```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking! How can I help you today?"
    },
    {
      "role": "user",
      "content": "Can you explain machine learning?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1640995200,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 150,
    "total_tokens": 195
  }
}
```

### Streaming Chat Completions

Get streaming chat completions.

**Request Body:**
```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a story"
    }
  ],
  "max_tokens": 200,
  "stream": true
}
```

**Response (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","object":"chat.completion","created":1640995200,"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion","created":1640995200,"choices":[{"delta":{"content":"Once"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion","created":1640995200,"choices":[{"delta":{"content":" upon"},"index":0,"finish_reason":null}]}

data: [DONE]
```

## Embeddings

### Create Embeddings

Generate embeddings for text.

**Endpoint:** `POST /embeddings`

**Request Body:**
```json
{
  "model": "text-embedding-ada-002",
  "input": "This is a sample text for embedding",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | The model to use for generation |
| `max_tokens` | integer | 16 | Maximum number of tokens to generate |
| `temperature` | number | 1.0 | Controls randomness (0.0 = deterministic, 2.0 = very random) |
| `top_p` | number | 1.0 | Nucleus sampling parameter |
| `top_k` | integer | -1 | Top-k sampling parameter |
| `frequency_penalty` | number | 0.0 | Penalty for frequent tokens |
| `presence_penalty` | number | 0.0 | Penalty for new tokens |
| `stop` | array | null | Stop sequences |
| `stream` | boolean | false | Whether to stream the response |
| `logprobs` | integer | null | Number of log probabilities to return |
| `echo` | boolean | false | Echo the prompt in the response |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `best_of` | integer | 1 | Number of best completions to return |
| `logit_bias` | object | {} | Bias for specific tokens |
| `suffix` | string | null | Suffix to append to completion |
| `user` | string | null | User identifier for abuse monitoring |

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `invalid_request_error` | Invalid request parameters |
| `model_not_found` | Model not found |
| `rate_limit_exceeded` | Rate limit exceeded |
| `insufficient_quota` | Insufficient quota |
| `server_error` | Internal server error |

## Rate Limiting

Rate limits are applied per API key:

- **Requests per minute:** 60
- **Tokens per minute:** 150,000
- **Requests per day:** 3,500

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995260
```

## Examples

### Python Client

```python
import requests
import json

# Base configuration
base_url = "http://localhost:8000/v1"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

# Text completion
def create_completion(prompt, model="meta-llama/Llama-2-7b-chat-hf"):
    url = f"{base_url}/completions"
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Chat completion
def create_chat_completion(messages, model="meta-llama/Llama-2-7b-chat-hf"):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Usage
completion = create_completion("Explain quantum computing")
print(completion["choices"][0]["text"])

chat_messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
chat_response = create_chat_completion(chat_messages)
print(chat_response["choices"][0]["message"]["content"])
```

### JavaScript Client

```javascript
// Base configuration
const baseUrl = 'http://localhost:8000/v1';
const headers = {
    'Authorization': 'Bearer your-api-key',
    'Content-Type': 'application/json'
};

// Text completion
async function createCompletion(prompt, model = 'meta-llama/Llama-2-7b-chat-hf') {
    const response = await fetch(`${baseUrl}/completions`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            model: model,
            prompt: prompt,
            max_tokens: 100,
            temperature: 0.7
        })
    });
    
    return await response.json();
}

// Chat completion
async function createChatCompletion(messages, model = 'meta-llama/Llama-2-7b-chat-hf') {
    const response = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            model: model,
            messages: messages,
            max_tokens: 150,
            temperature: 0.7
        })
    });
    
    return await response.json();
}

// Usage
createCompletion("Explain quantum computing")
    .then(response => console.log(response.choices[0].text));

const messages = [
    {role: "user", content: "Hello, how are you?"}
];
createChatCompletion(messages)
    .then(response => console.log(response.choices[0].message.content));
```

### cURL Examples

```bash
# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'

# Streaming completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "max_tokens": 200,
    "stream": true
  }'
```

## WebSocket API

For real-time streaming, you can also use the WebSocket API:

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

ws.onopen = function() {
    ws.send(JSON.stringify({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: [
            {role: 'user', content: 'Tell me a story'}
        ],
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.choices && data.choices[0].delta.content) {
        console.log(data.choices[0].delta.content);
    }
};
```

For more information about the Python client, see the [Python Client](python-client.md) documentation. 