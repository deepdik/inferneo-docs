# WebSocket API

Inferneo provides a WebSocket API for real-time streaming and bidirectional communication.

## Connection

### WebSocket URL

```
ws://localhost:8000/v1/chat/completions
```

### Connection Setup

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

ws.onopen = function() {
    console.log('Connected to Inferneo WebSocket API');
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('WebSocket connection closed:', event.code, event.reason);
};
```

## Authentication

### API Key Authentication

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

ws.onopen = function() {
    // Send authentication message
    ws.send(JSON.stringify({
        type: 'auth',
        api_key: 'your-api-key'
    }));
};
```

### Token-based Authentication

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions?token=your-token');
```

## Chat Completions

### Basic Chat Completion

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

ws.onopen = function() {
    // Send chat completion request
    ws.send(JSON.stringify({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: [
            {role: 'user', content: 'Hello, how are you?'}
        ],
        max_tokens: 100,
        temperature: 0.7,
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.choices && data.choices[0].delta.content) {
        console.log(data.choices[0].delta.content);
    }
    
    if (data.choices && data.choices[0].finish_reason) {
        console.log('Generation completed');
        ws.close();
    }
};
```

### Streaming Chat with History

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

const conversation = [
    {role: 'system', content: 'You are a helpful assistant.'},
    {role: 'user', content: 'What is machine learning?'},
    {role: 'assistant', content: 'Machine learning is a subset of artificial intelligence...'},
    {role: 'user', content: 'Can you give me an example?'}
];

ws.onopen = function() {
    ws.send(JSON.stringify({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: conversation,
        max_tokens: 150,
        temperature: 0.7,
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.choices && data.choices[0].delta.content) {
        process.stdout.write(data.choices[0].delta.content);
    }
    
    if (data.choices && data.choices[0].finish_reason) {
        console.log('\nGeneration completed');
        ws.close();
    }
};
```

## Text Completions

### Streaming Text Completion

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/completions');

ws.onopen = function() {
    ws.send(JSON.stringify({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        prompt: 'Explain quantum computing',
        max_tokens: 200,
        temperature: 0.8,
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.choices && data.choices[0].text) {
        process.stdout.write(data.choices[0].text);
    }
    
    if (data.choices && data.choices[0].finish_reason) {
        console.log('\nCompletion finished');
        ws.close();
    }
};
```

## Advanced Features

### Multiple Concurrent Requests

```javascript
class InferneoWebSocketClient {
    constructor(url) {
        this.url = url;
        this.connections = new Map();
        this.requestId = 0;
    }
    
    async sendRequest(request) {
        const id = ++this.requestId;
        
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(this.url);
            const response = {text: '', id: id};
            
            ws.onopen = function() {
                ws.send(JSON.stringify({
                    ...request,
                    request_id: id
                }));
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.choices && data.choices[0].delta.content) {
                    response.text += data.choices[0].delta.content;
                }
                
                if (data.choices && data.choices[0].finish_reason) {
                    resolve(response);
                    ws.close();
                }
            };
            
            ws.onerror = function(error) {
                reject(error);
            };
            
            this.connections.set(id, ws);
        });
    }
    
    closeAll() {
        this.connections.forEach(ws => ws.close());
        this.connections.clear();
    }
}

// Usage
const client = new InferneoWebSocketClient('ws://localhost:8000/v1/chat/completions');

// Send multiple requests
const requests = [
    {
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: [{role: 'user', content: 'Explain AI'}],
        max_tokens: 100,
        stream: true
    },
    {
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: [{role: 'user', content: 'What is ML?'}],
        max_tokens: 100,
        stream: true
    }
];

Promise.all(requests.map(req => client.sendRequest(req)))
    .then(responses => {
        responses.forEach(res => console.log(`Response ${res.id}:`, res.text));
    })
    .finally(() => client.closeAll());
```

### Error Handling

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

ws.onopen = function() {
    ws.send(JSON.stringify({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        messages: [{role: 'user', content: 'Hello'}],
        max_tokens: 100,
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Check for errors
    if (data.error) {
        console.error('API Error:', data.error);
        ws.close();
        return;
    }
    
    if (data.choices && data.choices[0].delta.content) {
        console.log(data.choices[0].delta.content);
    }
    
    if (data.choices && data.choices[0].finish_reason) {
        console.log('Completed');
        ws.close();
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    if (event.code !== 1000) {
        console.error('Connection closed unexpectedly:', event.code, event.reason);
    }
};
```

### Reconnection Logic

```javascript
class ReconnectingWebSocket {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            maxReconnectAttempts: 5,
            reconnectInterval: 1000,
            ...options
        };
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        this.messageQueue = [];
        this.connect();
    }
    
    connect() {
        if (this.isConnecting) return;
        
        this.isConnecting = true;
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            console.log('Connected to WebSocket');
            
            // Send queued messages
            while (this.messageQueue.length > 0) {
                this.ws.send(this.messageQueue.shift());
            }
        };
        
        this.ws.onclose = (event) => {
            this.isConnecting = false;
            
            if (event.code !== 1000 && this.reconnectAttempts < this.options.maxReconnectAttempts) {
                console.log(`Connection closed, attempting to reconnect (${this.reconnectAttempts + 1}/${this.options.maxReconnectAttempts})`);
                this.reconnectAttempts++;
                
                setTimeout(() => {
                    this.connect();
                }, this.options.reconnectInterval * this.reconnectAttempts);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
        } else {
            this.messageQueue.push(typeof data === 'string' ? data : JSON.stringify(data));
        }
    }
    
    close() {
        if (this.ws) {
            this.ws.close(1000);
        }
    }
}

// Usage
const ws = new ReconnectingWebSocket('ws://localhost:8000/v1/chat/completions');

ws.ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.choices && data.choices[0].delta.content) {
        console.log(data.choices[0].delta.content);
    }
};

// Send message (will be queued if not connected)
ws.send({
    model: 'meta-llama/Llama-2-7b-chat-hf',
    messages: [{role: 'user', content: 'Hello'}],
    max_tokens: 100,
    stream: true
});
```

## Python WebSocket Client

```python
import asyncio
import websockets
import json

class InferneoWebSocketClient:
    def __init__(self, url):
        self.url = url
        self.websocket = None
    
    async def connect(self):
        self.websocket = await websockets.connect(self.url)
    
    async def send_request(self, request):
        if not self.websocket:
            await self.connect()
        
        await self.websocket.send(json.dumps(request))
        
        full_response = ""
        async for message in self.websocket:
            data = json.loads(message)
            
            if "error" in data:
                raise Exception(f"API Error: {data['error']}")
            
            if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                content = data["choices"][0]["delta"]["content"]
                full_response += content
                print(content, end="", flush=True)
            
            if "choices" in data and data["choices"][0].get("finish_reason"):
                print()  # New line
                break
        
        return full_response
    
    async def close(self):
        if self.websocket:
            await self.websocket.close()

# Usage
async def main():
    client = InferneoWebSocketClient("ws://localhost:8000/v1/chat/completions")
    
    try:
        request = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [
                {"role": "user", "content": "Tell me a story"}
            ],
            "max_tokens": 200,
            "temperature": 0.8,
            "stream": True
        }
        
        response = await client.send_request(request)
        print(f"\nFull response: {response}")
        
    finally:
        await client.close()

# Run the async function
asyncio.run(main())
```

## Message Format

### Request Format

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n", "END"],
  "stream": true,
  "request_id": "unique-request-id"
}
```

### Response Format

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion.chunk",
  "created": 1640995200,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## Best Practices

### Connection Management

```javascript
// Keep connections alive with ping/pong
const ws = new WebSocket('ws://localhost:8000/v1/chat/completions');

// Send ping every 30 seconds
const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: 'ping'}));
    }
}, 30000);

ws.onclose = function() {
    clearInterval(pingInterval);
};
```

### Rate Limiting

```javascript
class RateLimitedWebSocket {
    constructor(url, rateLimit = 10) { // 10 requests per second
        this.url = url;
        this.rateLimit = rateLimit;
        this.requestQueue = [];
        this.lastRequestTime = 0;
        this.connect();
    }
    
    async sendRequest(request) {
        return new Promise((resolve) => {
            this.requestQueue.push({request, resolve});
            this.processQueue();
        });
    }
    
    async processQueue() {
        if (this.requestQueue.length === 0) return;
        
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minInterval = 1000 / this.rateLimit;
        
        if (timeSinceLastRequest < minInterval) {
            setTimeout(() => this.processQueue(), minInterval - timeSinceLastRequest);
            return;
        }
        
        const {request, resolve} = this.requestQueue.shift();
        this.lastRequestTime = now;
        
        // Send the request
        this.ws.send(JSON.stringify(request));
        resolve();
    }
}
```

### Error Recovery

```javascript
class ResilientWebSocket {
    constructor(url) {
        this.url = url;
        this.maxRetries = 3;
        this.retryDelay = 1000;
        this.connect();
    }
    
    async sendWithRetry(request, retries = 0) {
        try {
            return await this.sendRequest(request);
        } catch (error) {
            if (retries < this.maxRetries) {
                console.log(`Retry ${retries + 1}/${this.maxRetries}`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * (retries + 1)));
                return this.sendWithRetry(request, retries + 1);
            } else {
                throw error;
            }
        }
    }
}
```

For more information about the REST API, see the [REST API](rest-api.md) documentation. 