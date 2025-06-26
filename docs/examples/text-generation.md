# Text Generation Examples

This guide provides comprehensive examples of text generation tasks using Inferneo.

## Basic Text Completion

### Simple Prompt Completion

```python
from inferneo import InferneoClient

client = InferneoClient("http://localhost:8000")

# Basic text completion
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="The future of artificial intelligence is",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Creative Writing

```python
# Creative story generation
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Write a short story about a robot who discovers emotions:",
    max_tokens=300,
    temperature=0.8,
    top_p=0.9,
    frequency_penalty=0.1
)

print(response.choices[0].text)
```

### Technical Writing

```python
# Technical documentation
response = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt="Explain how machine learning algorithms work:",
    max_tokens=200,
    temperature=0.3,
    top_p=0.8
)

print(response.choices[0].text)
```

## Advanced Text Generation

### Multi-Step Generation

```python
def generate_story_outline():
    # Step 1: Generate story outline
    outline_response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt="Create a 5-point outline for a science fiction story:",
        max_tokens=150,
        temperature=0.7
    )
    
    outline = outline_response.choices[0].text
    
    # Step 2: Generate story from outline
    story_response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=f"Write a story based on this outline:\n{outline}",
        max_tokens=500,
        temperature=0.8
    )
    
    return story_response.choices[0].text

story = generate_story_outline()
print(story)
```

### Conditional Generation

```python
def generate_content_by_style(style, topic):
    style_prompts = {
        "formal": "Write a formal academic essay about",
        "casual": "Write a casual blog post about",
        "technical": "Write a technical explanation of",
        "creative": "Write a creative story about"
    }
    
    prompt = f"{style_prompts[style]} {topic}:"
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7 if style == "creative" else 0.3
    )
    
    return response.choices[0].text

# Generate different styles
formal_text = generate_content_by_style("formal", "climate change")
casual_text = generate_content_by_style("casual", "climate change")
technical_text = generate_content_by_style("technical", "climate change")
creative_text = generate_content_by_style("creative", "climate change")
```

## Specialized Generation Tasks

### Code Generation

```python
def generate_python_function(description):
    prompt = f"""
Write a Python function that {description}.
Include proper docstring and type hints.
"""
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        stop=["\n\n\n", "```"]
    )
    
    return response.choices[0].text

# Generate different functions
sort_function = generate_python_function("sorts a list of dictionaries by a specific key")
file_function = generate_python_function("reads a CSV file and returns a pandas DataFrame")
api_function = generate_python_function("makes an HTTP GET request and handles errors")
```

### Poetry Generation

```python
def generate_poem(theme, style="free verse"):
    prompt = f"""
Write a {style} poem about {theme}.
Make it creative and evocative.
"""
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=150,
        temperature=0.9,
        top_p=0.95
    )
    
    return response.choices[0].text

# Generate different poems
nature_poem = generate_poem("nature", "haiku")
love_poem = generate_poem("love", "sonnet")
city_poem = generate_poem("city life", "free verse")
```

### Email Generation

```python
def generate_email(recipient, purpose, tone="professional"):
    tone_prompts = {
        "professional": "Write a professional business email",
        "friendly": "Write a friendly and casual email",
        "formal": "Write a formal and respectful email"
    }
    
    prompt = f"""
{tone_prompts[tone]} to {recipient} about {purpose}.
Keep it concise and clear.
"""
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=150,
        temperature=0.4
    )
    
    return response.choices[0].text

# Generate different emails
business_email = generate_email("a client", "project update", "professional")
friendly_email = generate_email("a colleague", "lunch invitation", "friendly")
formal_email = generate_email("a professor", "research collaboration", "formal")
```

## Batch Generation

### Multiple Prompts

```python
def batch_generate(prompts, model="meta-llama/Llama-2-7b-chat-hf"):
    responses = []
    
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        responses.append(response.choices[0].text)
    
    return responses

# Example prompts
prompts = [
    "Write a one-sentence summary of machine learning",
    "Write a one-sentence summary of artificial intelligence",
    "Write a one-sentence summary of deep learning",
    "Write a one-sentence summary of neural networks"
]

summaries = batch_generate(prompts)
for i, summary in enumerate(summaries):
    print(f"{i+1}. {summary}")
```

### Template-Based Generation

```python
def generate_from_template(template, variables):
    # Replace placeholders in template
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{key}}}", value)
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    
    return response.choices[0].text

# Template examples
product_description_template = """
Write a compelling product description for {product_name}.
Target audience: {audience}
Key features: {features}
Tone: {tone}
"""

# Generate multiple product descriptions
products = [
    {
        "product_name": "Smart Home Assistant",
        "audience": "tech-savvy homeowners",
        "features": "voice control, automation, security",
        "tone": "innovative and helpful"
    },
    {
        "product_name": "Organic Coffee Beans",
        "audience": "coffee enthusiasts",
        "features": "single origin, fair trade, premium quality",
        "tone": "premium and authentic"
    }
]

for product in products:
    description = generate_from_template(product_description_template, product)
    print(f"\n{product['product_name']}:\n{description}")
```

## Streaming Generation

### Real-Time Text Generation

```python
def stream_generation(prompt, model="meta-llama/Llama-2-7b-chat-hf"):
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True
    )
    
    print("Generating text...")
    for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="", flush=True)
    print("\n")

# Stream different types of content
stream_generation("Write a creative story about time travel:")
stream_generation("Explain quantum computing in simple terms:")
stream_generation("Write a motivational speech about perseverance:")
```

## Quality Control

### Content Filtering

```python
def generate_with_filtering(prompt, content_type="general"):
    # Add content type context
    context_prompts = {
        "professional": "Write in a professional, business-appropriate tone: ",
        "educational": "Write in an educational, informative tone: ",
        "creative": "Write in a creative, imaginative tone: ",
        "technical": "Write in a technical, precise tone: "
    }
    
    full_prompt = context_prompts.get(content_type, "") + prompt
    
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        prompt=full_prompt,
        max_tokens=150,
        temperature=0.7,
        # Add stop sequences for better control
        stop=["\n\n", "---", "###"]
    )
    
    return response.choices[0].text

# Generate with different content types
professional_text = generate_with_filtering("project management", "professional")
educational_text = generate_with_filtering("solar system", "educational")
creative_text = generate_with_filtering("magical forest", "creative")
technical_text = generate_with_filtering("API design", "technical")
```

## Best Practices

### 1. Prompt Engineering
- Be specific and clear in your prompts
- Use context to guide the generation
- Experiment with different prompt formats

### 2. Parameter Tuning
- Use lower temperature (0.1-0.3) for factual content
- Use higher temperature (0.7-0.9) for creative content
- Adjust top_p and top_k for better control

### 3. Error Handling
```python
from inferneo import InferneoError

def safe_generate(prompt):
    try:
        response = client.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text
    except InferneoError as e:
        print(f"Generation failed: {e}")
        return None
```

### 4. Content Validation
```python
def validate_generated_content(text, min_length=10, max_length=1000):
    if len(text) < min_length:
        return False, "Text too short"
    if len(text) > max_length:
        return False, "Text too long"
    if not text.strip():
        return False, "Empty text"
    return True, "Valid content"

# Usage
is_valid, message = validate_generated_content(generated_text)
if not is_valid:
    print(f"Content validation failed: {message}")
```

## Next Steps

Explore more advanced text generation techniques:

- **[Chat Completion Examples](chat-completion.md)** - Interactive conversation examples
- **[Embeddings Examples](embeddings.md)** - Text embedding and similarity
- **[Vision Models Examples](vision-models.md)** - Multi-modal generation
- **[Performance Optimization](user-guide/performance-tuning.md)** - Speed up your generation 