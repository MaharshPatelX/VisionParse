# Configuration Guide

Complete guide to configuring VisionParse for your needs.

## 📋 Configuration Methods

VisionParse supports multiple configuration methods (in order of priority):

1. **Direct parameters** in VisionParse constructor
2. **Environment variables** (.env file)
3. **Configuration file** (config.json)
4. **Default values**

## 🔧 Environment Variables (.env)

### API Keys
```bash
# VLM Provider API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key

# Default VLM provider
VLM_PROVIDER=openai
```

### YOLO Configuration
```bash
# YOLO Detection Thresholds
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.1

# Custom YOLO model path
YOLO_MODEL_PATH=weights/icon_detect/model.pt
```

### Ollama Configuration
```bash
# Ollama local server settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60
```

### Example .env file
```bash
# VisionParse Configuration

# Primary VLM Provider
VLM_PROVIDER=openai

# API Keys (add only the ones you need)
OPENAI_API_KEY=sk-proj-abc123...
ANTHROPIC_API_KEY=sk-ant-def456...
GOOGLE_API_KEY=AIza789...

# YOLO Detection Settings
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.1
YOLO_MODEL_PATH=weights/icon_detect/model.pt

# Ollama Settings (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60

# Logging
VERBOSE=false
```

## 📄 Configuration File (config.json)

### Complete Configuration
```json
{
  "provider": "openai",
  "default_models": {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet",
    "google": "gemini-2.0-flash-exp",
    "ollama": "llava:latest"
  },
  "yolo_model_path": "weights/icon_detect/model.pt",
  "confidence_threshold": 0.05,
  "yolo_config": {
    "box_threshold": 0.05,
    "iou_threshold": 0.1
  },
  "ollama_config": {
    "base_url": "http://localhost:11434",
    "timeout": 60
  }
}
```

### Minimal Configuration
```json
{
  "provider": "openai",
  "confidence_threshold": 0.05,
  "yolo_config": {
    "iou_threshold": 0.1
  }
}
```

## 🎛️ Direct Configuration

### Basic Setup
```python
from VisionParse import VisionParse

# Minimal configuration
parser = VisionParse(provider='openai', model='gpt-4o')

# Custom configuration
parser = VisionParse(
    provider='anthropic',
    model='claude-3-5-sonnet',
    api_key='sk-ant-your-key',
    confidence_threshold=0.3,
    iou_threshold=0.5,
    verbose=True
)
```

### Runtime Configuration Changes
```python
parser = VisionParse(provider='openai', model='gpt-4o')

# Change detection sensitivity
parser.confidence_threshold = 0.1  # More sensitive
parser.iou_threshold = 0.3         # Less filtering

# Switch models
parser.provider = 'anthropic'
parser.model = 'claude-3-5-sonnet'
parser.api_key = 'sk-ant-new-key'
```

## 🤖 VLM Provider Configuration

### OpenAI/GPT Configuration
```python
# Environment variables
OPENAI_API_KEY=sk-your-key
VLM_PROVIDER=openai

# Code configuration
parser = VisionParse(
    provider='openai',
    model='gpt-4o',           # Latest GPT model
    api_key='sk-your-key'
)

# Available models (examples)
models = [
    'gpt-4o',          # Flagship multimodal
    'gpt-4o-mini',     # Fast and efficient
    'gpt-4-turbo',     # Previous generation
    'gpt-4',           # Classic GPT-4
    'o1',              # Reasoning model
]
```

### Anthropic/Claude Configuration
```python
# Environment variables
ANTHROPIC_API_KEY=sk-ant-your-key
VLM_PROVIDER=anthropic

# Code configuration
parser = VisionParse(
    provider='anthropic',
    model='claude-3-5-sonnet',
    api_key='sk-ant-your-key'
)

# Available models (examples)
models = [
    'claude-3-5-sonnet-20241022',  # Latest Sonnet
    'claude-3-5-haiku-20241022',   # Fast and efficient
    'claude-3-opus-20240229',      # Most capable
    'claude-3-sonnet-20240229',    # Balanced
    'claude-3-haiku-20240307'      # Fastest
]
```

### Google/Gemini Configuration
```python
# Environment variables
GOOGLE_API_KEY=your-google-key
VLM_PROVIDER=google

# Code configuration
parser = VisionParse(
    vlm_type='google',
    model='gemini-2.5-pro',
    api_key='your-google-key'
)

# Available models (examples)
models = [
    'gemini-2.5-pro',      # Latest with thinking
    'gemini-2.5-flash',    # Low latency
    'gemini-2.0-flash',    # Enhanced performance
    'gemini-2.0-pro'       # Best for coding
]
```

### Ollama/Local Configuration
```python
# Environment variables
OLLAMA_BASE_URL=http://localhost:11434
VLM_PROVIDER=ollama

# Code configuration
parser = VisionParse(
    vlm_type='ollama',
    model='llava:latest'
    # No API key needed for local models
)

# Available models (install with ollama pull)
models = [
    'llava:latest',        # Best overall (4.5GB)
    'llava:13b',          # Larger model (7GB)
    'llava:7b',           # Balanced (3.8GB)
    'minicpm-v:latest',   # Fast & efficient (2.8GB)
    'moondream:latest',   # Lightweight (1.7GB)
    'cogvlm:latest',      # Advanced vision (14GB)
    'qwen-vl:latest'      # Multilingual (4.2GB)
]
```

## 🎯 YOLO Detection Configuration

### Threshold Explanation

**Confidence Threshold** (`confidence_threshold`):
- **Range**: 0.0 - 1.0
- **Default**: 0.05
- **Lower values**: More detections (more sensitive)
- **Higher values**: Fewer detections (more confident)

**IoU Threshold** (`iou_threshold`):
- **Range**: 0.0 - 1.0  
- **Default**: 0.1
- **Lower values**: Less overlap filtering (more detections)
- **Higher values**: More overlap filtering (fewer duplicates)

### Tuning for Different Use Cases

```python
# Web UI analysis (many small elements)
parser = VisionParse(
    vlm_type='gpt4o',
    confidence_threshold=0.1,  # More sensitive
    iou_threshold=0.3          # Allow some overlap
)

# Desktop app analysis (fewer, larger elements)
parser = VisionParse(
    vlm_type='gpt4o',
    confidence_threshold=0.3,  # More selective
    iou_threshold=0.7          # Strict overlap filtering
)

# Mobile app analysis (touch targets)
parser = VisionParse(
    vlm_type='gpt4o',
    confidence_threshold=0.2,  # Balanced
    iou_threshold=0.5          # Moderate filtering
)

# High-resolution screenshots
parser = VisionParse(
    vlm_type='gpt4o',
    confidence_threshold=0.05, # Very sensitive
    iou_threshold=0.2          # Minimal filtering
)
```

### Dynamic Threshold Adjustment

```python
parser = VisionParse(vlm_type='gpt4o')

def analyze_with_adaptive_thresholds(image_path):
    """Try different thresholds and pick best result"""
    
    threshold_configs = [
        (0.05, 0.1),  # Very sensitive
        (0.1, 0.3),   # Moderate
        (0.3, 0.5),   # Conservative
    ]
    
    best_result = None
    best_count = 0
    
    for conf_thresh, iou_thresh in threshold_configs:
        parser.confidence_threshold = conf_thresh
        parser.iou_threshold = iou_thresh
        
        result = parser.analyze(image_path)
        element_count = len(result['elements'])
        
        # Pick configuration with reasonable element count
        if 5 <= element_count <= 20 and element_count > best_count:
            best_result = result
            best_count = element_count
    
    return best_result or result  # Return last result if none optimal
```

## 🔐 Security Configuration

### API Key Management

**Never hardcode API keys in your code!**

```python
# ❌ Bad: Hardcoded API key
parser = VisionParse(
    vlm_type='openai',
    api_key='sk-your-actual-key-here'  # Don't do this!
)

# ✅ Good: Environment variable
import os
parser = VisionParse(
    vlm_type='openai',
    api_key=os.getenv('OPENAI_API_KEY')
)

# ✅ Better: Use .env file
from dotenv import load_dotenv
load_dotenv()

parser = VisionParse(vlm_type='openai')  # Automatically loads from .env
```

### File Permissions

```bash
# Secure your .env file
chmod 600 .env

# Secure config files
chmod 600 config.json
```

### Gitignore Setup

Make sure your `.gitignore` includes:
```
# Environment variables
.env
.env.local
.env.*.local

# Config with secrets
config_local.json
**/api_keys.json
```

## 📊 Performance Configuration

### Memory Optimization

```python
# For memory-constrained environments
parser = VisionParse(
    vlm_type='gpt4o',
    verbose=False,              # Reduce logging overhead
    confidence_threshold=0.3,   # Fewer detections = less memory
)

# Clear results after processing to free memory
results = parser.analyze('image.png')
process_results(results)
del results
```

### Speed Optimization

```python
# For faster processing
parser = VisionParse(
    vlm_type='openai',
    model='gpt-4.1-nano',      # Fastest model
    confidence_threshold=0.2,   # Balanced detection
    iou_threshold=0.5
)

# Use local models for no API latency
parser = VisionParse(
    vlm_type='ollama',
    model='moondream:latest'    # Lightweight local model
)
```

### Batch Processing Configuration

```python
# Optimized for batch processing
parser = VisionParse(
    vlm_type='gpt4o',
    verbose=False  # Reduce log noise for batch jobs
)

# Process in chunks to manage memory
def process_large_batch(image_paths, chunk_size=10):
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i + chunk_size]
        results = parser.analyze_batch(chunk)
        yield results
```

## 🐛 Debugging Configuration

### Verbose Logging

```python
# Enable detailed logging
parser = VisionParse(
    vlm_type='gpt4o',
    verbose=True
)

# Or set logging level manually
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Configuration

```python
# Configuration for testing
test_config = {
    'vlm_type': 'ollama',         # Free local testing
    'model': 'moondream:latest',  # Fast model
    'confidence_threshold': 0.1,  # Sensitive for test images
    'iou_threshold': 0.3,
    'verbose': True
}

parser = VisionParse(**test_config)
```

## 📋 Configuration Validation

### Validate Your Setup

```python
from src import VisionParse, VisionParseError

def validate_configuration():
    """Test if VisionParse is configured correctly"""
    try:
        # Test basic initialization
        parser = VisionParse(vlm_type='gpt4o')
        print("✅ VisionParse initialized successfully")
        
        # Test API key (if not using Ollama)
        if parser.vlm_type != 'ollama' and not parser.api_key:
            print("⚠️  Warning: No API key configured")
        else:
            print("✅ API key configured")
        
        # Test YOLO model
        import os
        if os.path.exists(parser.yolo_model_path):
            print("✅ YOLO model found")
        else:
            print(f"❌ YOLO model not found: {parser.yolo_model_path}")
        
        return True
        
    except VisionParseError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# Run validation
if __name__ == "__main__":
    validate_configuration()
```

## 📚 Configuration Examples

### Development Setup
```python
# .env for development
VLM_PROVIDER=ollama
CONFIDENCE_THRESHOLD=0.1
IOU_THRESHOLD=0.3
VERBOSE=true

# Code
parser = VisionParse(
    vlm_type='ollama',
    model='moondream:latest',
    verbose=True
)
```

### Production Setup
```python
# .env for production
VLM_PROVIDER=gpt4o
OPENAI_API_KEY=sk-real-key
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.1
VERBOSE=false

# Code with error handling
try:
    parser = VisionParse(
        vlm_type=os.getenv('VLM_PROVIDER'),
        api_key=os.getenv('OPENAI_API_KEY'),
        verbose=False
    )
except VisionParseError as e:
    logger.error(f"VisionParse configuration failed: {e}")
    sys.exit(1)
```

### Multi-Environment Setup
```python
import os

# config/development.py
DEV_CONFIG = {
    'vlm_type': 'ollama',
    'model': 'llava:latest',
    'confidence_threshold': 0.1,
    'verbose': True
}

# config/production.py  
PROD_CONFIG = {
    'vlm_type': 'gpt4o',
    'api_key': os.getenv('OPENAI_API_KEY'),
    'confidence_threshold': 0.05,
    'verbose': False
}

# main.py
config = DEV_CONFIG if os.getenv('ENV') == 'dev' else PROD_CONFIG
parser = VisionParse(**config)
```

This configuration guide should cover all your VisionParse setup needs. For specific issues, check the [Troubleshooting Guide](troubleshooting.md).