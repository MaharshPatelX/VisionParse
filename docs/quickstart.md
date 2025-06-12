# Quick Start Guide

Get up and running with VisionParse in 5 minutes!

## 🎯 What You'll Learn

- How to analyze your first screenshot
- Basic VLM provider setup
- Understanding the output
- Next steps

## 📋 Before You Start

Make sure you have:
- ✅ Python 3.8+ installed
- ✅ VisionParse installed ([Installation Guide](installation.md))
- ✅ At least one API key configured
- ✅ A screenshot to analyze

## 🚀 Your First Analysis

### Step 1: Set Up API Key

Choose one VLM provider and add your API key to `.env`:

```bash
# For OpenAI (recommended for beginners)
OPENAI_API_KEY=sk-your-openai-key

# For Claude
ANTHROPIC_API_KEY=sk-ant-your-key

# For Gemini  
GOOGLE_API_KEY=your-google-key
```

### Step 2: Interactive Mode (Easiest)

```bash
# Start interactive mode
python main.py
```

Follow the prompts:
1. Choose your VLM provider (1-4)
2. Enter path to your screenshot
3. Wait for analysis
4. View results!

### Step 3: Command Line (Direct)

```bash
# Analyze with OpenAI GPT-4o
visionparse screenshot.png --vlm openai --model gpt-4o

# Analyze with Claude
visionparse screenshot.png --vlm anthropic --model claude-3-5-sonnet

# Analyze with Gemini
visionparse screenshot.png --vlm google --model gemini-2.0-flash-exp

# Local analysis (no API key needed)
visionparse screenshot.png --vlm ollama --model llava:latest
```

### Step 4: Python API (Programmatic)

```python
from VisionParse import VisionParse

# Initialize parser with provider + model
parser = VisionParse(provider='openai', model='gpt-4o')

# Analyze screenshot
results = parser.analyze('screenshot.png')

# Print results
print(f"Found {len(results['elements'])} UI elements:")
for element in results['elements']:
    print(f"- {element['name']}: {element['description']}")
```

## 📊 Understanding the Output

### Console Output
```
🔍 Analyzing screenshot.png...
✅ YOLO detected 3 UI elements
🤖 Analyzing with GPT-4o...

Element 1: Login Button
  📍 Location: (892, 24) to (956, 44)
  🎯 Confidence: 0.91
  📝 Blue authentication button in header

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

### Files Created
- `screenshot_analyzed.png` - Annotated image with numbered boxes
- `screenshot_results.json` - Detailed analysis data

## ⚙️ Customizing Detection

### Adjust Sensitivity

```python
from VisionParse import VisionParse

# More sensitive detection (finds more elements)
parser = VisionParse(
    provider='openai',
    model='gpt-4o',
    confidence_threshold=0.1,  # Lower = more detections
    iou_threshold=0.3          # Lower = less filtering
)

# Stricter detection (fewer, more confident elements)
parser = VisionParse(
    provider='openai',
    model='gpt-4o', 
    confidence_threshold=0.5,  # Higher = fewer detections
    iou_threshold=0.7          # Higher = more filtering
)
```

### Try Different Models

```bash
# Latest GPT models
visionparse image.png --vlm openai --model gpt-4o
visionparse image.png --vlm openai --model gpt-4o-mini

# Claude models
visionparse image.png --vlm anthropic --model claude-3-5-sonnet
visionparse image.png --vlm anthropic --model claude-3-haiku

# Gemini models
visionparse image.png --vlm google --model gemini-2.0-flash-exp
visionparse image.png --vlm google --model gemini-1.5-flash
```

## 🏠 Free Local Analysis

### Set Up Ollama (No API Keys Required!)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve

# Download vision model (choose one)
ollama pull llava:latest      # Best overall (4.5GB)
ollama pull minicpm-v:latest  # Fast (2.8GB)
ollama pull moondream:latest  # Lightweight (1.7GB)

# Use with VisionParse
visionparse screenshot.png --vlm ollama --model llava:latest
```

**Benefits of Local Models:**
- 🆓 Completely free
- 🔒 Private (data never leaves your machine)
- 📶 Works offline
- 🚀 No API rate limits

## 🔄 Next Steps

### 1. Process Multiple Images
```bash
# Batch processing
visionparse *.png --batch --output batch_results/
```

### 2. Integrate into Your App
```python
from src import VisionParse

class UIAnalyzer:
    def __init__(self):
        self.parser = VisionParse(vlm_type='gpt4o')
    
    def find_buttons(self, image_path):
        results = self.parser.analyze(image_path)
        return [el for el in results['elements'] if el['type'] == 'Button']

analyzer = UIAnalyzer()
buttons = analyzer.find_buttons('screenshot.png')
```

### 3. Export Data
```python
# Export to CSV for analysis
parser.export_results(results, 'analysis.csv', format='csv')

# Export to JSON for processing
parser.export_results(results, 'data.json', format='json')
```

## 🛠️ Troubleshooting Quick Fixes

**No elements detected?**
- Lower confidence threshold: `confidence_threshold=0.1`
- Check if image is clear and contains UI elements

**Too many false detections?**
- Raise confidence threshold: `confidence_threshold=0.3`
- Use stricter IoU threshold: `iou_threshold=0.7`

**API errors?**
- Check your `.env` file has correct API keys
- Verify API key has sufficient credits/quota

**Import errors?**
- Make sure you're in the VisionParse directory
- Run `pip install -e .` again

## 📚 Learn More

- [API Reference](api-reference.md) - Complete function documentation
- [Configuration Guide](configuration.md) - All settings explained
- [Examples](../examples/) - Real-world use cases
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## 💡 Pro Tips

1. **Start with interactive mode** to understand the workflow
2. **Use local models** for testing and development
3. **Adjust thresholds** based on your image type
4. **Try different VLM providers** to compare results
5. **Save configurations** in environment variables for reuse

Happy analyzing! 🚀