"""
VisionParse: AI-Powered UI Element Detection and Analysis

A powerful Python library for analyzing user interface elements using YOLO object detection
and Vision Language Models (VLMs) from multiple providers.

Example usage:
    from VisionParse import VisionParse
    
    # Initialize with provider + model
    parser = VisionParse(provider='openai', model='gpt-4o')
    
    # Analyze screenshot
    results = parser.analyze('screenshot.png')
    
    # Process results
    print(f"Found {len(results['elements'])} UI elements")
"""

# Version info
__version__ = "1.0.0"
__author__ = "VisionParse Team"
__license__ = "MIT"

# Import main classes and functions from src
from .src.vlm_parser import VisionParse, VisionParseError
from .src.yolo_detector import detect_ui_elements, draw_boxes_on_image
from .src.vlm_clients import batch_analyze_regions

# Public API
__all__ = [
    "VisionParse",
    "VisionParseError", 
    "detect_ui_elements",
    "draw_boxes_on_image",
    "batch_analyze_regions",
    "__version__"
]