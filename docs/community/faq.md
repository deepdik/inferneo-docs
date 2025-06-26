# FAQ

Frequently Asked Questions about Inferneo.

## General

**Q: What is Inferneo?**
A: Inferneo is a high-performance inference server for LLMs and multimodal models.

**Q: What models are supported?**
A: Most HuggingFace Transformers, ONNX, and select vision/multimodal models.

**Q: Is GPU required?**
A: GPU is recommended for best performance, but CPU is supported for smaller models.

## Usage

**Q: How do I install Inferneo?**
A: See the [Installation Guide](../installation.md).

**Q: How do I run the server?**
A: Use `inferneo serve --model <model_id>`.

**Q: How do I use quantization?**
A: Pass `--quantization awq` or `gptq` or `squeezellm` to the server.

## Troubleshooting

**Q: The server won't start. What should I check?**
A: Ensure your model path is correct and you have enough GPU memory.

**Q: How do I report a bug?**
A: Open a GitHub Issue with details and logs.

## Next Steps
- **[Roadmap](roadmap.md)**
- **[Releases](releases.md)** 