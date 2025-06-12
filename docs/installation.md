# Installation Guide

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning repository)
- **4GB RAM minimum** (8GB recommended for local models)
- **GPU optional** (for faster YOLO inference)

## 🚀 Installation Methods

### Method 1: Clone and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/MaharshPatelX/VisionParse.git
cd VisionParse

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .
```

### Method 2: Direct GitHub Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/MaharshPatelX/VisionParse.git
```

### Method 3: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv visionparse-env

# Activate virtual environment
# On Windows:
visionparse-env\Scripts\activate
# On macOS/Linux:
source visionparse-env/bin/activate

# Install VisionParse
git clone https://github.com/MaharshPatelX/VisionParse.git
cd VisionParse
pip install -r requirements.txt
pip install -e .
```

## 🔧 Environment Setup

### 1. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

Add your API keys to `.env`:
```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-key

# Anthropic API Key  
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Google API Key
GOOGLE_API_KEY=your-google-api-key

# Default VLM Provider
VLM_PROVIDER=gpt4o
```

### 2. Download YOLO Model

The YOLO model weights should be placed in:
```
VisionParse/
└── weights/
    └── icon_detect/
        └── model.pt
```

If you don't have the model weights, contact the repository maintainer or check the releases page.

## 🏠 Local Models Setup (Optional)

### Install Ollama for Free Local Models

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows, download from https://ollama.ai

# Start Ollama service
ollama serve

# Pull vision models
ollama pull llava:latest      # 4.5GB - Best overall
ollama pull minicpm-v:latest  # 2.8GB - Fast & efficient
ollama pull moondream:latest  # 1.7GB - Lightweight
```

## ✅ Verify Installation

### Test Basic Installation
```bash
# Test CLI installation
visionparse --help

# Test Python import
python -c "from src import VisionParse; print('✅ VisionParse installed successfully!')"
```

### Test with Sample Analysis
```bash
# Interactive mode
python main.py

# CLI mode (if you have a sample image)
visionparse sample.png --vlm gpt4o
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'src'**
```bash
# Make sure you're in the VisionParse directory
cd VisionParse
pip install -e .
```

**2. YOLO Model Not Found**
```bash
# Check if model file exists
ls weights/icon_detect/model.pt
# If missing, download from releases or contact maintainer
```

**3. API Key Errors**
```bash
# Check if .env file exists and has correct keys
cat .env
# Make sure no extra spaces or quotes around keys
```

**4. Permission Errors (Linux/macOS)**
```bash
# Use --user flag if needed
pip install --user -r requirements.txt
```

**5. Ollama Connection Issues**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama if needed
ollama serve
```

### Dependencies Issues

**PyTorch Installation:**
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**OpenCV Issues:**
```bash
# If opencv-python fails
pip install opencv-python-headless
```

## 🔄 Updating VisionParse

```bash
# If installed from Git
cd VisionParse
git pull origin main
pip install -r requirements.txt

# If installed via pip
pip install --upgrade git+https://github.com/MaharshPatelX/VisionParse.git
```

## 🐳 Docker Installation (Coming Soon)

```bash
# Pull Docker image (when available)
docker pull maharshpatelx/visionparse:latest

# Run container
docker run -it --rm -v $(pwd):/workspace maharshpatelx/visionparse:latest
```

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/MaharshPatelX/VisionParse/issues)
3. Create a new issue with detailed error information
4. Join [GitHub Discussions](https://github.com/MaharshPatelX/VisionParse/discussions)