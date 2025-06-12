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

curl -o weights\icon_detect\model.pt https://raw.githubusercontent.com/MaharshPatelX/VisionParse/refs/heads/main/weights/icon_detect/model.pt
curl -o weights\icon_detect\model.yaml https://raw.githubusercontent.com/MaharshPatelX/VisionParse/refs/heads/main/weights/icon_detect/model.yaml
curl -o weights\icon_detect\train_args.yaml https://raw.githubusercontent.com/MaharshPatelX/VisionParse/refs/heads/main/weights/icon_detect/train_args.yaml

```

## Features

- **YOLO Detection**: Fast UI element detection with precise bounding boxes
- **LlamaIndex Integration**: Unified interface for GPT-4o, Claude, Gemini, Ollama
- **Structured Output**: JSON with coordinates, types, and descriptions
- **Production Ready**: Python library with batch processing support
- **Local Models**: Private analysis with Ollama (free & offline)
- **Model Validation**: Strict validation ensures only supported models are used

## Supported Models

VisionParse enforces strict model validation. Only these models are supported:

### OpenAI
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.1`
- `gpt-4.1-mini`

### Anthropic
- `claude-3-5-sonnet-latest`
- `claude-3-7-sonnet-latest`
- `claude-sonnet-4-0`
- `claude-opus-4-0`

### Google
- `gemini-2.5-flash-preview-05-20`
- `gemini-2.0-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Ollama (Local)
- `gemma3:4b`
- `qwen2.5vl:3b`

## Usage

```python
from VisionParse import VisionParse

# Initialize parser with provider + model
parser = VisionParse(provider='openai', model='gpt-4o')

# Analyze single image
results = parser.analyze('screenshot.png')
print(f"Found {len(results['elements'])} UI elements")

# Different providers with supported models
parser = VisionParse(provider='anthropic', model='claude-3-5-sonnet-latest')
parser = VisionParse(provider='google', model='gemini-1.5-pro')
parser = VisionParse(provider='ollama', model='qwen2.5vl:3b')

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

## Error Handling

VisionParse includes comprehensive error handling for unsupported models:

```python
from VisionParse import VisionParse, VisionParseError

try:
    # This will raise an error - unsupported model
    parser = VisionParse(provider='openai', model='gpt-3.5-turbo')
except VisionParseError as e:
    print(f"Error: {e}")
    # Output: "Unsupported model 'gpt-3.5-turbo' for provider 'openai'. Supported models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini']"

try:
    # This will also raise an error - unsupported provider
    parser = VisionParse(provider='huggingface', model='any-model')
except VisionParseError as e:
    print(f"Error: {e}")
    # Output: "Unsupported provider 'huggingface'. Supported providers: ['openai', 'anthropic', 'google', 'ollama']"
```

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

# Usage with supported models
analyzer = ScreenshotAnalyzer(provider='anthropic', model='claude-3-5-sonnet-latest')
ui_elements = analyzer.analyze_ui('app_screenshot.png')
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues:** [GitHub Issues](https://github.com/MaharshPatelX/VisionParse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/MaharshPatelX/VisionParse/discussions)
