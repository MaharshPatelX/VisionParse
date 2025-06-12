# VisionParse

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
cd visionparse
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
visionparse screenshot.png --vlm gpt4o

# With custom model
visionparse screenshot.png --vlm openai --model gpt-4.1
```

## 📖 Usage Guide

### Command Line Interface
```bash
# Basic usage
visionparse image.png --vlm gpt4o

# Batch processing
visionparse *.png --batch --output results/

# Custom model
visionparse image.png --vlm claude --model claude-4-opus

# Local model (free)
visionparse image.png --vlm ollama --model llava:latest

# JSON output only
visionparse image.png --json-only
```

### Python API
```python
from src import VisionParse

# Initialize parser
parser = VisionParse(vlm_type='gpt4o')

# Analyze single image
results = parser.analyze('screenshot.png')
print(f"Found {len(results['elements'])} UI elements")

# Batch processing
results = parser.analyze_batch(['img1.png', 'img2.png'])

# Custom model
parser = VisionParse(vlm_type='openai', model='gpt-4.1')
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

# 4. Use with VisionParse
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
    "openai": "gpt-4.1",
    "anthropic": "claude-4-opus",
    "google": "gemini-2.5-flash",
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
--model gpt-4.1-nano
--model gpt-4o

# Latest Anthropic Claude models (2025)
--model claude-4-opus
--model claude-4-sonnet
--model claude-3.7-sonnet
--model claude-3.5-sonnet
--model claude-3.5-haiku

# Latest Google Gemini models (2025)
--model gemini-2.5-pro
--model gemini-2.5-flash
--model gemini-2.0-flash
--model gemini-2.0-pro

# Local Ollama models
--model llava:latest
--model minicpm-v:latest
--model moondream:latest
```

## 📁 Project Structure

```
visionparse/
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

## 📚 Library Usage

VisionParse can be easily integrated into your Python applications as a library.

### Installation as Library
```bash
# Install from local directory
cd visionparse
pip install -e .

# Or install normally
pip install .

# Future: Install from GitHub
pip install git+https://github.com/yourusername/visionparse.git
```

### Basic Library Usage
```python
from src import VisionParse

# Simple usage
parser = VisionParse(vlm_type='gpt4o')
results = parser.analyze('screenshot.png')

# Access results
print(f"Found {len(results['elements'])} UI elements")
for element in results['elements']:
    print(f"- {element['name']}: {element['description']}")
```

### Integration in Your Application
```python
import os
from src import VisionParse, VisionParseError

class ScreenshotAnalyzer:
    def __init__(self, vlm_type='gpt4o', model=None):
        # Load API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        
        self.parser = VisionParse(
            vlm_type=vlm_type,
            model=model,
            api_key=api_key,
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
    
    def get_clickable_elements(self, image_path):
        """Get only clickable UI elements"""
        results = self.analyze_ui(image_path)
        
        if 'elements' in results:
            clickable_types = ['Button', 'Link', 'Icon', 'Menu']
            return [
                el for el in results['elements'] 
                if el['type'] in clickable_types
            ]
        
        return []

# Usage in your app
analyzer = ScreenshotAnalyzer(vlm_type='claude', model='claude-4-opus')
ui_elements = analyzer.analyze_ui('app_screenshot.png')
clickable_elements = analyzer.get_clickable_elements('app_screenshot.png')
```

### Environment Configuration
```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

VISIONPARSE_CONFIG = {
    'vlm_type': os.getenv('VLM_PROVIDER', 'gpt4o'),
    'api_key': os.getenv('OPENAI_API_KEY'),
    'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.05')),
    'iou_threshold': float(os.getenv('IOU_THRESHOLD', '0.1'))
}

# Use in your application
from src import VisionParse
parser = VisionParse(**VISIONPARSE_CONFIG)
```

## 🛠️ Advanced Usage

### Batch Processing
```python
from src import VisionParse

parser = VisionParse(vlm_type='gpt4o')
results = parser.analyze_batch([
    'screen1.png', 'screen2.png', 'screen3.png'
], output_dir='results/')

print(f"Processed {results['summary']['successful']} images")
```

### Custom YOLO Settings
```python
parser = VisionParse(
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

- **Issues:** [GitHub Issues](https://github.com/yourusername/visionparse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/visionparse/discussions)

---

**Made with ❤️ for the AI community**