# VisionParse

> **AI-powered screenshot analysis tool** that combines YOLO object detection with Vision Language Models to extract UI elements with structured data. Now powered by **LlamaIndex** for unified multimodal AI integration.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LlamaIndex](https://img.shields.io/badge/Powered%20by-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)

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
| 🤖 **LlamaIndex Integration** | Unified interface for GPT-4o, Claude, Gemini, Ollama |
| 🔓 **No Model Restrictions** | Use any model name - future-proof design |
| 📊 **Structured Output** | JSON with coordinates, types, and descriptions |
| 🖼️ **Visual Annotations** | Saves annotated images with numbered elements |
| 🚀 **Production Ready** | Python library, batch processing |
| 🏠 **Local Models** | Private analysis with Ollama (free & offline) |

## 🔥 What's New: LlamaIndex Integration

VisionParse now uses **LlamaIndex** for all VLM interactions, providing:

- **Unified API**: Single interface for all providers instead of multiple libraries
- **Better Error Handling**: Built-in retry logic and timeout management
- **Cleaner Code**: Simplified provider implementations
- **Future-Proof**: Easy to add new models as LlamaIndex supports them

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/MaharshPatelX/VisionParse.git
cd VisionParse
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key
```

### 3. Use as Library
```python
from VisionParse import VisionParse

# Initialize with provider + model
parser = VisionParse(provider='openai', model='gpt-4o')
results = parser.analyze('screenshot.png')

print(f"Found {len(results['elements'])} UI elements")
```

### 4. Test the Setup
```bash
python test.py
```

## 📖 Usage Guide

### Python Library
```python
from VisionParse import VisionParse

# Initialize parser with provider + model
parser = VisionParse(provider='openai', model='gpt-4o')

# Analyze single image
results = parser.analyze('screenshot.png')
print(f"Found {len(results['elements'])} UI elements")

# Different providers
parser = VisionParse(provider='anthropic', model='claude-3-5-sonnet')
parser = VisionParse(provider='google', model='gemini-2.0-flash-exp')
parser = VisionParse(provider='ollama', model='llava:latest')

# Batch processing
results = parser.analyze_batch(['img1.png', 'img2.png'])

# Custom thresholds
parser = VisionParse(
    provider='openai', 
    model='gpt-4o',
    confidence_threshold=0.3,
    iou_threshold=0.5
)
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

# 4. Use with VisionParse
parser = VisionParse(provider='ollama', model='llava:latest')
results = parser.analyze('screenshot.png')
```

**Benefits:** Free, private, offline, no API limits

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key

# Default settings (optional)
VLM_PROVIDER=openai
```

### Config File (config.json)
```json
{
  "provider": "openai",
  "default_models": {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet",
    "google": "gemini-2.0-flash-exp",
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
# Latest OpenAI models (2025)
--model gpt-4.1
--model gpt-4.1-mini
--model gpt-4o

# Latest Anthropic Claude models (2025)
--model claude-4-opus
--model claude-4-sonnet
--model claude-3.5-sonnet

# Latest Google Gemini models (2025)
--model gemini-2.5-pro
--model gemini-2.0-flash

# Local Ollama models
--model llava:latest
--model minicpm-v:latest
--model moondream:latest
```

## 📁 Project Structure

```
VisionParse/
├── VisionParse/            # Main package
│   ├── src/               # Core modules
│   │   ├── vlm_parser.py  # Main API class
│   │   ├── yolo_detector.py # YOLO detection engine
│   │   ├── vlm_clients.py # LlamaIndex VLM integrations
│   │   └── __init__.py    # Module exports
│   └── __init__.py        # Package exports
├── test.py               # Test script for LlamaIndex integration
├── config.json           # Configuration settings
├── requirements.txt      # Python dependencies (now with LlamaIndex)
├── .env                  # API keys (create from .env.example)
├── weights/             # YOLO model weights
│   └── icon_detect/
│       └── model.pt     # Pre-trained YOLO model
└── README.md            # This file
```

## 📋 Requirements

- **Python:** 3.8 or higher
- **LlamaIndex:** Core and provider packages
- **YOLO Model:** Place model weights in `weights/icon_detect/model.pt`
- **API Keys:** At least one VLM provider API key (or use Ollama locally)
- **Memory:** 4GB RAM minimum (8GB recommended for local models)

## 🔍 Dependencies

### Core ML Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
Pillow>=9.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

### LlamaIndex LLM Clients (Unified Interface)
```
llama-index-core>=0.11.0
llama-index-llms-openai>=0.2.0
llama-index-llms-anthropic>=0.3.0
llama-index-llms-gemini>=0.2.0
llama-index-llms-ollama>=0.3.0
```

## 🔍 Output Examples

### Python Output
```python
results = parser.analyze('screenshot.png')
print(f"Found {len(results['elements'])} UI elements")

for element in results['elements']:
    print(f"- {element['name']}: {element['description']}")

# Output:
# Found 3 UI elements
# - Login Button: Blue authentication button in header
# - Search Field: Text input for search queries
# - Menu Icon: Navigation menu toggle
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
  "vlm_type": "openai",
  "model": "gpt-4o"
}
```

## 📚 Library Usage

### Installation as Library
```bash
# Install from local directory
cd VisionParse
pip install -e .

# Or install normally
pip install .

# Install directly from GitHub
pip install git+https://github.com/MaharshPatelX/VisionParse.git
```

### Integration in Your Application
```python
import os
from VisionParse import VisionParse, VisionParseError

class ScreenshotAnalyzer:
    def __init__(self, provider='openai', model='gpt-4o'):
        self.parser = VisionParse(
            provider=provider,
            model=model,
            verbose=False
        )
    
    def analyze_ui(self, image_path):
        """Analyze UI elements in screenshot"""
        try:
            results = self.parser.analyze(image_path)
            
            if results['success']:
                return {
                    'elements': results['elements'],
                    'count': len(results['elements']),
                    'confidence_avg': sum(el['confidence'] for el in results['elements']) / len(results['elements'])
                }
            else:
                return {'error': results['error']}
                
        except VisionParseError as e:
            return {'error': str(e)}

# Usage
analyzer = ScreenshotAnalyzer(provider='anthropic', model='claude-3-5-sonnet')
ui_elements = analyzer.analyze_ui('app_screenshot.png')
```

## 🛠️ Advanced Usage

### Batch Processing
```python
parser = VisionParse(provider='openai', model='gpt-4o')
results = parser.analyze_batch([
    'screen1.png', 'screen2.png', 'screen3.png'
], output_dir='results/')

print(f"Processed {results['summary']['successful']} images")
```

### Custom YOLO Settings
```python
# Dynamic threshold adjustment
parser = VisionParse(provider='openai', model='gpt-4o')

# More sensitive detection
parser.confidence_threshold = 0.1
parser.iou_threshold = 0.3
results = parser.analyze('screenshot.png')

# Stricter detection
parser.confidence_threshold = 0.7
parser.iou_threshold = 0.8
results = parser.analyze('screenshot.png')
```

### Export Results
```python
# Export to CSV
parser.export_results(results, 'output.csv', format='csv')

# Export to JSON  
parser.export_results(results, 'output.json', format='json')
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test.py
```

The test will:
- ✅ Check LlamaIndex package installations
- ✅ Verify API key configuration from .env
- ✅ Test all provider integrations
- ✅ Validate VisionParse initialization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `python test.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues:** [GitHub Issues](https://github.com/MaharshPatelX/VisionParse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/MaharshPatelX/VisionParse/discussions)

## 🏆 Powered By

- **[LlamaIndex](https://www.llamaindex.ai/)** - Unified multimodal AI framework
- **[Ultralytics YOLO](https://ultralytics.com/)** - Object detection
- **[OpenAI](https://openai.com/)** - GPT-4o vision models
- **[Anthropic](https://anthropic.com/)** - Claude vision models  
- **[Google](https://ai.google.dev/)** - Gemini vision models
- **[Ollama](https://ollama.com/)** - Local model inference

---

**Made with ❤️ for the AI community** | **Now with LlamaIndex integration for better multimodal AI**