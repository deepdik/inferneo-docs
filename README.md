# Inferneo Documentation

This repository contains the official documentation for [Inferneo](https://github.com/inferneo/inferneo), a high-performance inference server for machine learning models.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/inferneo/docs.git
   cd docs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the documentation server:**
   ```bash
   mkdocs serve
   ```

4. **Open your browser:**
   Navigate to [http://localhost:8000](http://localhost:8000) to view the documentation.

## 📚 Documentation Structure

```
docs/
├── index.md                 # Homepage
├── getting-started.md       # Quick start guide
├── installation.md          # Installation instructions
├── user-guide/             # User documentation
│   ├── quickstart.md
│   ├── offline-inference.md
│   ├── online-serving.md
│   ├── model-loading.md
│   ├── batching.md
│   ├── streaming.md
│   ├── quantization.md
│   └── distributed-inference.md
├── examples/               # Code examples
│   ├── text-generation.md
│   ├── chat-completion.md
│   ├── embeddings.md
│   ├── vision-models.md
│   └── multimodal.md
├── api-reference/          # API documentation
│   ├── python-client.md
│   ├── rest-api.md
│   ├── websocket-api.md
│   └── configuration.md
├── cli-reference/          # CLI documentation
│   ├── commands.md
│   └── environment-variables.md
├── developer-guide/        # Developer documentation
│   ├── architecture.md
│   ├── contributing.md
│   ├── custom-models.md
│   └── performance-tuning.md
└── community/             # Community resources
    ├── roadmap.md
    ├── releases.md
    └── faq.md
```

## 🛠️ Development

### Local Development

1. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the development server:**
   ```bash
   mkdocs serve
   ```

3. **Build the documentation:**
   ```bash
   mkdocs build
   ```

### Adding New Content

1. **Create a new markdown file** in the appropriate directory
2. **Add the page to navigation** in `mkdocs.yml`
3. **Follow the existing style** and formatting conventions
4. **Test locally** before submitting changes

### Style Guide

- Use **Markdown** for all content
- Follow the **existing formatting** patterns
- Include **code examples** where appropriate
- Use **admonitions** for notes, warnings, and tips
- Keep **sections organized** and well-structured

## 🎨 Customization

### Theme Configuration

The documentation uses Material for MkDocs with custom styling:

- **Primary color**: Indigo (#667eea)
- **Custom CSS**: `stylesheets/extra.css`
- **JavaScript**: `javascripts/mathjax.js`

### Adding Custom Styles

1. **Edit `stylesheets/extra.css`** for custom CSS
2. **Edit `javascripts/mathjax.js`** for MathJax configuration
3. **Update `mkdocs.yml`** for theme settings

## 📖 Content Guidelines

### Writing Style

- **Clear and concise** language
- **Step-by-step instructions** for complex tasks
- **Code examples** for all major features
- **Screenshots and diagrams** when helpful
- **Cross-references** to related content

### Code Examples

- **Use realistic examples** that users can run
- **Include error handling** where appropriate
- **Add comments** to explain complex code
- **Test all examples** before publishing

### Documentation Types

- **Tutorials**: Step-by-step guides for common tasks
- **How-to guides**: Instructions for specific tasks
- **Reference**: Complete API and configuration documentation
- **Explanation**: Conceptual information and background

## 🔧 Configuration

### MkDocs Configuration

The main configuration file is `mkdocs.yml`:

```yaml
site_name: Inferneo
site_description: Fast and efficient inference server for machine learning models
theme:
  name: material
  # ... theme configuration
nav:
  # ... navigation structure
plugins:
  # ... enabled plugins
```

### Key Features

- **Search functionality** with instant search
- **Dark/light mode** toggle
- **Mobile-responsive** design
- **Git revision dates** on pages
- **Code syntax highlighting**
- **Mathematical notation** support
- **Admonitions** for notes and warnings

## 🚀 Deployment

### GitHub Pages

1. **Build the documentation:**
   ```bash
   mkdocs build
   ```

2. **Deploy to GitHub Pages:**
   ```bash
   mkdocs gh-deploy
   ```

### Other Platforms

- **Netlify**: Connect repository and set build command to `mkdocs build`
- **Vercel**: Similar to Netlify setup
- **Custom server**: Upload `site/` directory to your web server

## 🤝 Contributing

We welcome contributions to improve the documentation!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test locally** with `mkdocs serve`
5. **Submit a pull request**

### Contribution Guidelines

- **Follow the style guide**
- **Test all changes locally**
- **Update navigation** if adding new pages
- **Include examples** for new features
- **Update related pages** when making changes

## 📞 Support

### Getting Help

- **GitHub Issues**: [Create an issue](https://github.com/inferneo/docs/issues)
- **Discord**: [Join our community](https://discord.gg/inferneo)
- **Email**: docs@inferneo.ai

### Reporting Problems

When reporting issues, please include:

- **Description** of the problem
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, browser, etc.)
- **Screenshots** if applicable

## 📄 License

This documentation is licensed under the [Apache 2.0 License](LICENSE).

## 🙏 Acknowledgments

- **Material for MkDocs** for the excellent documentation theme
- **vLLM** for inspiration and reference
- **The Inferneo community** for feedback and contributions

---

**Built with ❤️ by the Inferneo team** 