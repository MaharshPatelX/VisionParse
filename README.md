# VisionParse

> **Python library** for AI-powered screenshot analysis that combines YOLO object detection with Vision Language Models to extract UI elements with structured data. Powered by **LlamaIndex** for unified multimodal AI integration.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LlamaIndex](https://img.shields.io/badge/Powered%20by-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)

## Installation

```bash
pip install git+https://github.com/MaharshPatelX/VisionParse.git
```
```bash
mkdir weights\icon_detect

curl -o weights\icon_detect\model.pt https://github.com/MaharshPatelX/VisionParse/blob/main/weights/icon_detect/model.pt
curl -o weights\icon_detect\model.yaml https://github.com/MaharshPatelX/VisionParse/blob/main/weights/icon_detect/model.yaml
curl -o weights\icon_detect\train_args.yaml https://github.com/MaharshPatelX/VisionParse/blob/main/weights/icon_detect/train_args.yaml
```

## Features

- **YOLO Detection**: Fast UI element detection with precise bounding boxes
- **LlamaIndex Integration**: Unified interface for GPT-4o, Claude, Gemini, Ollama
- **Structured Output**: JSON with coordinates, types, and descriptions
- **Production Ready**: Python library with batch processing support
- **Local Models**: Private analysis with Ollama (free & offline)

## Usage

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

## Configuration

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"
```

## Output Format

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

## Requirements

- **Python:** 3.8 or higher
- **API Keys:** At least one VLM provider API key (or use Ollama locally)
- **Memory:** 4GB RAM minimum (8GB recommended for local models)

## Integration Example

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues:** [GitHub Issues](https://github.com/MaharshPatelX/VisionParse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/MaharshPatelX/VisionParse/discussions)
