# REST API Reference

This page documents the REST API endpoints provided by Inferneo.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

```
GET /health
```
- Returns server health status.

### List Models

```
GET /v1/models
```
- Returns a list of available models.

### Text Completion

```
POST /v1/completions
```
- Request body:
  - `model`: Model ID
  - `prompt`: Prompt string or list
  - `max_tokens`: Maximum tokens to generate
  - `temperature`: Sampling temperature
- Response: Generated text(s)

### Chat Completion

```
POST /v1/chat/completions
```
- Request body:
  - `model`: Model ID
  - `messages`: List of chat messages
  - `max_tokens`: Maximum tokens
- Response: Chat completion

### Embeddings

```
POST /v1/embeddings
```
- Request body:
  - `model`: Model ID
  - `input`: List of texts
- Response: Embeddings

### Vision

```
POST /v1/vision
```
- Request body:
  - `model`: Model ID
  - `image`: Image bytes
- Response: Vision model output

### Multimodal

```
POST /v1/multimodal
```
- Request body:
  - `model`: Model ID
  - `prompt`: Text prompt
  - `image`: Image bytes
- Response: Multimodal output

## Error Handling

- Standard HTTP error codes
- JSON error messages

## Next Steps
- **[WebSocket API](websocket-api.md)**
- **[Configuration](configuration.md)** 