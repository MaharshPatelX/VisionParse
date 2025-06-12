# API Reference

Complete documentation for VisionParse Python API.

## 📚 Table of Contents

- [VisionParse Class](#visionparse-class)
- [Exceptions](#exceptions)
- [Utility Functions](#utility-functions)
- [Type Definitions](#type-definitions)
- [Examples](#examples)

## VisionParse Class

### Constructor

```python
VisionParse(
    config_path: Optional[str] = None,
    vlm_type: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    confidence_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    yolo_model_path: Optional[str] = None,
    verbose: bool = False
)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration JSON file
- `vlm_type` (str, optional): VLM provider ('gpt4o', 'openai', 'claude', 'google', 'ollama', or any custom provider)
- `model` (str, optional): Specific model name (e.g., 'gpt-4.1', 'claude-4-opus')
- `api_key` (str, optional): API key for VLM provider
- `confidence_threshold` (float): YOLO confidence threshold (0.0-1.0, default: 0.05)
- `iou_threshold` (float): YOLO IoU threshold (0.0-1.0, default: 0.1)
- `yolo_model_path` (str, optional): Path to custom YOLO model weights
- `verbose` (bool): Enable verbose logging (default: False)

**Example:**
```python
from src import VisionParse

# Basic initialization
parser = VisionParse(vlm_type='gpt4o')

# Advanced initialization
parser = VisionParse(
    vlm_type='claude',
    model='claude-4-opus',
    api_key='sk-ant-your-key',
    confidence_threshold=0.3,
    iou_threshold=0.5,
    verbose=True
)
```

### Methods

#### analyze()

Analyze a single screenshot and extract UI elements.

```python
analyze(
    image_path: Union[str, Path],
    save_annotated_image: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `image_path` (str | Path): Path to screenshot image
- `save_annotated_image` (bool): Whether to save annotated image (default: True)
- `output_dir` (str, optional): Output directory for results

**Returns:**
```python
{
    "success": bool,
    "elements": List[Dict],  # UI elements found
    "image_path": str,
    "annotated_image_path": str,  # Path to annotated image
    "yolo_detections": int,       # Number of YOLO detections
    "vlm_type": str,
    "model": str,
    "confidence_threshold": float,
    "iou_threshold": float
}
```

**Example:**
```python
parser = VisionParse(vlm_type='gpt4o')
results = parser.analyze('screenshot.png')

if results['success']:
    print(f"Found {len(results['elements'])} elements")
    for element in results['elements']:
        print(f"- {element['name']}: {element['description']}")
else:
    print(f"Analysis failed: {results['error']}")
```

#### analyze_batch()

Analyze multiple images in batch.

```python
analyze_batch(
    image_paths: List[Union[str, Path]],
    save_annotated_images: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `image_paths` (List[str | Path]): List of image paths
- `save_annotated_images` (bool): Whether to save annotated images
- `output_dir` (str, optional): Output directory for results

**Returns:**
```python
{
    "success": bool,
    "results": List[Dict],  # Individual analysis results
    "summary": {
        "total": int,
        "successful": int, 
        "failed": int
    },
    "vlm_type": str,
    "model": str
}
```

**Example:**
```python
image_paths = ['screen1.png', 'screen2.png', 'screen3.png']
batch_results = parser.analyze_batch(image_paths, output_dir='results/')

summary = batch_results['summary']
print(f"Processed {summary['total']} images: {summary['successful']} successful, {summary['failed']} failed")
```

#### export_results()

Export analysis results to file.

```python
export_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'json'
) -> str
```

**Parameters:**
- `results` (Dict): Analysis results from analyze() or analyze_batch()
- `output_path` (str | Path): Output file path
- `format` (str): Export format ('json' or 'csv')

**Returns:**
- `str`: Path to exported file

**Example:**
```python
results = parser.analyze('screenshot.png')

# Export to JSON
json_path = parser.export_results(results, 'analysis.json')

# Export to CSV  
csv_path = parser.export_results(results, 'analysis.csv', format='csv')
```

### Properties

#### Configurable Attributes

These can be modified after initialization:

```python
parser = VisionParse(vlm_type='gpt4o')

# Modify detection sensitivity
parser.confidence_threshold = 0.3  # Higher = fewer detections
parser.iou_threshold = 0.7         # Higher = more overlap filtering

# Change model
parser.model = 'gpt-4.1-mini'
parser.api_key = 'new-api-key'
```

## Exceptions

### VisionParseError

Custom exception for VisionParse-specific errors.

```python
from src import VisionParseError

try:
    parser = VisionParse(vlm_type='invalid')
    results = parser.analyze('nonexistent.png')
except VisionParseError as e:
    print(f"VisionParse error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

**Common VisionParseError scenarios:**
- YOLO model file not found
- Invalid API key
- Invalid threshold values (not between 0-1)
- Image file not found

## Utility Functions

### detect_ui_elements()

Low-level YOLO detection function.

```python
from src import detect_ui_elements

detected_boxes = detect_ui_elements(
    image_path='screenshot.png',
    model_path='weights/icon_detect/model.pt',
    confidence_threshold=0.05,
    iou_threshold=0.1
)
```

### draw_boxes_on_image()

Draw bounding boxes on image.

```python
from src import draw_boxes_on_image

annotated_image = draw_boxes_on_image(
    image_path='screenshot.png',
    boxes=detected_boxes,
    output_path='annotated.png'
)
```

### batch_analyze_regions()

Analyze cropped regions with VLM.

```python
from src import batch_analyze_regions

results = batch_analyze_regions(
    cropped_regions=regions,
    vlm_type='gpt4o',
    api_key='your-key'
)
```

## Type Definitions

### Element Structure

Each detected UI element has this structure:

```python
{
    "id": int,                    # Unique identifier
    "bbox": [int, int, int, int], # [x1, y1, x2, y2] coordinates
    "confidence": float,          # YOLO confidence (0.0-1.0)
    "name": str,                  # Element name (e.g., "Login Button")
    "type": str,                  # Element type (e.g., "Button", "Icon")
    "use": str,                   # What it does (e.g., "User authentication")
    "description": str            # Detailed description
}
```

### Configuration Structure

Configuration file format:

```json
{
    "vlm_type": "gpt4o",
    "default_models": {
        "openai": "gpt-4.1",
        "anthropic": "claude-4-opus", 
        "google": "gemini-2.5-flash",
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

## Examples

### Basic Usage

```python
from src import VisionParse, VisionParseError

# Simple analysis
parser = VisionParse(vlm_type='gpt4o')
results = parser.analyze('screenshot.png')

print(f"Success: {results['success']}")
print(f"Elements found: {len(results['elements'])}")
```

### Advanced Configuration

```python
# Custom configuration
parser = VisionParse(
    vlm_type='claude',
    model='claude-4-opus',
    confidence_threshold=0.2,
    iou_threshold=0.6,
    verbose=True
)

try:
    results = parser.analyze(
        image_path='complex_ui.png',
        save_annotated_image=True,
        output_dir='analysis_output/'
    )
    
    # Filter specific elements
    buttons = [el for el in results['elements'] if el['type'] == 'Button']
    print(f"Found {len(buttons)} buttons")
    
except VisionParseError as e:
    print(f"Analysis failed: {e}")
```

### Dynamic Threshold Adjustment

```python
parser = VisionParse(vlm_type='gpt4o')

# Try different sensitivity levels
for confidence in [0.1, 0.3, 0.5]:
    parser.confidence_threshold = confidence
    results = parser.analyze(f'test_image.png')
    print(f"Confidence {confidence}: {len(results['elements'])} elements")
```

### Batch Processing with Error Handling

```python
import glob
from pathlib import Path

parser = VisionParse(vlm_type='gpt4o')
image_files = glob.glob('screenshots/*.png')

batch_results = parser.analyze_batch(
    image_paths=image_files,
    output_dir='batch_analysis/'
)

# Process results
for result in batch_results['results']:
    image_name = Path(result['image_path']).name
    
    if result['success']:
        element_count = len(result['elements'])
        print(f"✅ {image_name}: {element_count} elements")
    else:
        print(f"❌ {image_name}: {result['error']}")
```

### Integration with Web Framework

```python
from flask import Flask, request, jsonify
from src import VisionParse

app = Flask(__name__)
parser = VisionParse(vlm_type='gpt4o')

@app.route('/analyze', methods=['POST'])
def analyze_screenshot():
    try:
        # Get uploaded file
        file = request.files['image']
        file_path = f'uploads/{file.filename}'
        file.save(file_path)
        
        # Analyze with VisionParse
        results = parser.analyze(file_path)
        
        return jsonify({
            'success': results['success'],
            'elements': results['elements'],
            'count': len(results['elements'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Performance Tips

1. **Reuse parser instances** - Creating VisionParse objects is expensive
2. **Adjust thresholds** based on your image types
3. **Use batch processing** for multiple images
4. **Cache API keys** in environment variables
5. **Use local models** for development to avoid API costs

## Version Compatibility

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **OpenCV**: 4.8+
- **Pillow**: 9.0+

For the latest compatibility information, see [requirements.txt](../requirements.txt).