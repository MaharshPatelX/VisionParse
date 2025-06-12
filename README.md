# VLM Parser

> **AI-powered screenshot analysis tool** that combines YOLO object detection with Vision Language Models to extract UI elements with structured data.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 What It Does

**Input:** Screenshot image  
**Output:** Structured JSON with UI elements, coordinates, and descriptions

```json
{
  "elements": [
    {
      "id": 1,
      "bbox": [100, 50, 200, 100],
      "confidence": 0.85,
      "name": "Login Button",
      "type": "Button",
      "use": "User authentication",
      "description": "Blue login button in top-right corner"
    }
  ]
}
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **YOLO Detection** | Fast UI element detection with precise bounding boxes |
| 🤖 **Multi-VLM Support** | GPT-4o, Claude, Gemini, Ollama (local models) |
| 🔓 **No Model Restrictions** | Use any model name - future-proof design |
| 📊 **Structured Output** | JSON with coordinates, types, and descriptions |
| 🖼️ **Visual Annotations** | Saves annotated images with numbered elements |
| 🚀 **Production Ready** | CLI tool, Python API, batch processing |
| 🏠 **Local Models** | Private analysis with Ollama (free & offline) |

## 🚀 Quick Start

### 1. Installation
```bash
git clone <repository>
cd myomni
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Analysis
```bash
# Interactive mode (recommended for first time)
python main.py

# Direct CLI
python cli.py screenshot.png --vlm gpt4o

# With custom model
python cli.py screenshot.png --vlm openai --model gpt-4-turbo-preview
```

## 📖 Usage Guide

### Command Line Interface
```bash
# Basic usage
python cli.py image.png --vlm gpt4o

# Batch processing
python cli.py *.png --batch --output results/

# Custom model
python cli.py image.png --vlm claude --model claude-3-opus-latest

# Local model (free)
python cli.py image.png --vlm ollama --model llava:latest

# JSON output only
python cli.py image.png --json-only
```

### Python API
```python
from src.vlm_parser import VLMParser

# Initialize parser
parser = VLMParser(vlm_type='gpt4o')

# Analyze single image
results = parser.analyze('screenshot.png')
print(f"Found {len(results['elements'])} UI elements")

# Batch processing
results = parser.analyze_batch(['img1.png', 'img2.png'])

# Custom model
parser = VLMParser(vlm_type='openai', model='gpt-4-turbo-preview')
results = parser.analyze('screenshot.png')
```

### Interactive Mode
```bash
python main.py
# Follow the prompts to:
# 1. Choose VLM provider
# 2. Enter image path
# 3. Get instant analysis
```

## 🏠 Local Models (Free & Private)

Run vision models locally with **zero API costs**:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a vision model
ollama pull llava:latest         # 4.5GB - Best overall
ollama pull minicpm-v:latest     # 2.8GB - Fast & efficient  
ollama pull moondream:latest     # 1.7GB - Lightweight

# 3. Start Ollama
ollama serve

# 4. Use with VLM Parser
python main.py  # Choose option 4 (Ollama)
```

**Benefits:** Free, private, offline, no API limits

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key

# Default settings
VLM_PROVIDER=gpt4o
```

### Config File (config.json)
```json
{
  "vlm_type": "gpt4o",
  "default_models": {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-flash",
    "ollama": "llava:latest"
  },
  "yolo_config": {
    "box_threshold": 0.05,
    "iou_threshold": 0.1
  }
}
```

## 🎯 Model Flexibility

**✅ No Restrictions** - Use any model name without code changes:

```bash
# Latest GPT models (even future ones)
--model gpt-4-turbo-preview
--model gpt-5-ultra

# Custom Claude models
--model claude-3-opus-latest
--model claude-4-sonnet

# Google's newest models  
--model gemini-2.0-flash-exp
--model gemini-3.0-ultra

# Local Ollama models
--model llava:13b
--model custom-vision-model:latest
```

## 📁 Project Structure

```
myomni/
├── src/                    # Core modules
│   ├── vlm_parser.py      # Main API class
│   ├── yolo_detector.py   # YOLO detection engine
│   ├── vlm_clients.py     # Multi-VLM integrations
│   └── __init__.py        # Package exports
├── cli.py                 # Command line interface
├── main.py               # Interactive mode
├── config.json           # Configuration settings
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create from .env.example)
├── .env.example         # Template for API keys
└── weights/             # YOLO model weights
    └── icon_detect/
        └── model.pt     # Pre-trained YOLO model
```

## 📋 Requirements

- **Python:** 3.8 or higher
- **YOLO Model:** Place model weights in `weights/icon_detect/model.pt`
- **API Keys:** At least one VLM provider API key (or use Ollama locally)
- **Memory:** 4GB RAM minimum (8GB recommended for local models)

## 🔍 Output Examples

### Console Output
```
🔍 Analyzing screenshot.png...
✅ YOLO detected 3 UI elements
🤖 Analyzing with GPT-4o...

Element 1: Login Button
  📍 Location: (892, 24) to (956, 44)
  🎯 Confidence: 0.91
  📝 Blue authentication button in header

Element 2: Search Field  
  📍 Location: (300, 100) to (600, 130)
  🎯 Confidence: 0.87
  📝 Text input for search queries

💾 Results saved to: screenshot_results.json
🖼️ Annotated image: screenshot_analyzed.png
```

### JSON Output
```json
{
  "success": true,
  "elements": [
    {
      "id": 1,
      "bbox": [892, 24, 956, 44],
      "confidence": 0.91,
      "name": "Login Button",
      "type": "Button", 
      "use": "User authentication",
      "description": "Blue authentication button in header"
    }
  ],
  "yolo_detections": 3,
  "vlm_type": "gpt4o",
  "model": "gpt-4o"
}
```

## 🛠️ Advanced Usage

### Batch Processing
```python
from src.vlm_parser import VLMParser

parser = VLMParser(vlm_type='gpt4o')
results = parser.analyze_batch([
    'screen1.png', 'screen2.png', 'screen3.png'
], output_dir='results/')

print(f"Processed {results['summary']['successful']} images")
```

### Custom YOLO Settings
```python
parser = VLMParser(
    vlm_type='gpt4o',
    confidence_threshold=0.3,  # Higher threshold
    iou_threshold=0.5,         # Different IoU
    yolo_model_path='custom_model.pt'
)
```

### Export Results
```python
# Export to CSV
parser.export_results(results, 'output.csv', format='csv')

# Export to JSON  
parser.export_results(results, 'output.json', format='json')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/myomni/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/myomni/discussions)
- **Email:** support@vlmparser.com

---

**Made with ❤️ for the AI community**