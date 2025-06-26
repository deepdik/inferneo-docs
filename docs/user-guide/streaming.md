# Streaming

This guide covers how to use streaming responses in Inferneo for real-time text generation.

## Overview

Streaming allows you to receive generated text in real-time as it's being produced, providing:

- **Immediate feedback** to users
- **Better user experience** for long responses
- **Interactive applications** like chat interfaces
- **Progress monitoring** during generation

## Basic Streaming

### Chat Completion Streaming

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Stream chat completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    max_tokens=200,
    stream=True
)

# Process streaming response
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Text Completion Streaming

```python
# Stream text completion
stream = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Explain quantum computing in detail:",
    max_tokens=300,
    temperature=0.7,
    stream=True
)

# Process streaming response
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)
```

## Advanced Streaming

### Custom Stream Processing

```python
def process_stream_with_metadata(stream):
    """Process stream with additional metadata."""
    
    full_response = ""
    tokens_generated = 0
    start_time = time.time()
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            tokens_generated += 1
            
            # Print with metadata
            print(f"[Token {tokens_generated}] {content}", end="", flush=True)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"\n\nGeneration completed:")
    print(f"Total tokens: {tokens_generated}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_generated/generation_time:.2f}")
    
    return full_response

# Usage
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Explain machine learning step by step."}],
    max_tokens=200,
    stream=True
)

response = process_stream_with_metadata(stream)
```

### Streaming with Callbacks

```python
class StreamingCallback:
    def __init__(self):
        self.full_response = ""
        self.tokens_received = 0
        self.start_time = time.time()
    
    def on_token(self, token):
        """Called for each token received."""
        self.full_response += token
        self.tokens_received += 1
        print(token, end="", flush=True)
    
    def on_complete(self):
        """Called when streaming is complete."""
        end_time = time.time()
        generation_time = end_time - self.start_time
        
        print(f"\n\nStreaming completed:")
        print(f"Total tokens: {self.tokens_received}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Tokens per second: {self.tokens_received/generation_time:.2f}")

def stream_with_callback(client, messages, callback):
    """Stream with custom callback handling."""
    
    stream = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=messages,
        max_tokens=200,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            callback.on_token(chunk.choices[0].delta.content)
    
    callback.on_complete()
    return callback.full_response

# Usage
callback = StreamingCallback()
messages = [{"role": "user", "content": "Write a poem about technology."}]

response = stream_with_callback(client, messages, callback)
```

## WebSocket Streaming

### Real-time WebSocket Communication

```python
import asyncio
import websockets
import json

async def stream_with_websocket():
    """Stream using WebSocket connection."""
    
    uri = "ws://localhost:8000/v1/chat/completions"
    
    async with websockets.connect(uri) as websocket:
        # Send streaming request
        message = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Tell me a story."}],
            "stream": True,
            "max_tokens": 200
        }
        
        await websocket.send(json.dumps(message))
        
        # Receive streaming response
        full_response = ""
        
        async for response in websocket:
            data = json.loads(response)
            
            if "choices" in data and len(data["choices"]) > 0:
                delta = data["choices"][0].get("delta", {})
                
                if "content" in delta:
                    content = delta["content"]
                    full_response += content
                    print(content, end="", flush=True)
                
                # Check if streaming is complete
                if data["choices"][0].get("finish_reason") is not None:
                    break
        
        print(f"\n\nFull response: {full_response}")

# Run the async function
asyncio.run(stream_with_websocket())
```

## Interactive Streaming

### Chat Interface with Streaming

```python
def interactive_chat():
    """Interactive chat interface with streaming."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    print("Chat with Inferneo (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Stream response
        print("Assistant: ", end="", flush=True)
        
        try:
            stream = client.chat.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                stream=True
            )
            
            assistant_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_response += content
                    print(content, end="", flush=True)
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            print(f"\nError: {e}")

# Usage
interactive_chat()
```

### Progress Bar Streaming

```python
from tqdm import tqdm

def stream_with_progress_bar(prompt, max_tokens=200):
    """Stream with progress bar showing generation progress."""
    
    stream = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True
    )
    
    full_response = ""
    tokens_generated = 0
    
    # Create progress bar
    with tqdm(total=max_tokens, desc="Generating", unit="tokens") as pbar:
        for chunk in stream:
            if chunk.choices[0].text is not None:
                token = chunk.choices[0].text
                full_response += token
                tokens_generated += 1
                pbar.update(1)
                
                # Update progress bar description
                pbar.set_description(f"Generated {tokens_generated} tokens")
    
    return full_response

# Usage
response = stream_with_progress_bar("Explain the concept of neural networks.")
print(f"\nFinal response: {response}")
```

## Error Handling in Streaming

### Robust Streaming

```python
def robust_streaming(prompt, max_retries=3):
    """Stream with error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            stream = client.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
                stream=True
            )
            
            full_response = ""
            
            for chunk in stream:
                if chunk.choices[0].text is not None:
                    token = chunk.choices[0].text
                    full_response += token
                    print(token, end="", flush=True)
            
            return full_response
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                print("All retry attempts failed")
                return None
            else:
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

# Usage
response = robust_streaming("Write a short story about space exploration.")
```

## Performance Monitoring

### Streaming Performance Metrics

```python
import time
from collections import deque

class StreamingMetrics:
    def __init__(self):
        self.start_time = None
        self.tokens_received = 0
        self.token_times = deque(maxlen=100)
        self.last_token_time = None
    
    def start(self):
        """Start timing the stream."""
        self.start_time = time.time()
        self.last_token_time = self.start_time
    
    def on_token(self, token):
        """Record token timing."""
        current_time = time.time()
        
        if self.last_token_time is not None:
            token_interval = current_time - self.last_token_time
            self.token_times.append(token_interval)
        
        self.tokens_received += 1
        self.last_token_time = current_time
    
    def get_metrics(self):
        """Get streaming performance metrics."""
        if self.start_time is None:
            return None
        
        total_time = time.time() - self.start_time
        
        metrics = {
            "total_tokens": self.tokens_received,
            "total_time": total_time,
            "tokens_per_second": self.tokens_received / total_time if total_time > 0 else 0,
            "average_token_interval": sum(self.token_times) / len(self.token_times) if self.token_times else 0
        }
        
        return metrics

def stream_with_metrics(prompt):
    """Stream with performance monitoring."""
    
    metrics = StreamingMetrics()
    metrics.start()
    
    stream = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True
    )
    
    full_response = ""
    
    for chunk in stream:
        if chunk.choices[0].text is not None:
            token = chunk.choices[0].text
            full_response += token
            metrics.on_token(token)
            print(token, end="", flush=True)
    
    # Print metrics
    performance_metrics = metrics.get_metrics()
    print(f"\n\nPerformance Metrics:")
    print(f"Total tokens: {performance_metrics['total_tokens']}")
    print(f"Total time: {performance_metrics['total_time']:.2f} seconds")
    print(f"Tokens per second: {performance_metrics['tokens_per_second']:.2f}")
    print(f"Average token interval: {performance_metrics['average_token_interval']*1000:.2f} ms")
    
    return full_response

# Usage
response = stream_with_metrics("Explain the benefits of streaming responses.")
```

## Best Practices

1. **Immediate Feedback**: Use streaming for better user experience
2. **Error Handling**: Implement robust error handling for network issues
3. **Progress Indicators**: Show progress bars or indicators for long generations
4. **Performance Monitoring**: Track streaming performance metrics
5. **Resource Management**: Close streams properly to free resources
6. **User Experience**: Provide visual feedback during streaming

## Next Steps

- **[Online Serving](online-serving.md)** - Real-time inference with streaming
- **[Chat Completions](examples/chat-completion.md)** - Advanced chat applications
- **[Performance Tuning](developer-guide/performance-tuning.md)** - Optimize streaming performance 