"""
VLM Parser - Screenshot Analysis Tool
"""

from .vlm_parser import VLMParser, VLMParserError
from .yolo_detector import detect_ui_elements, draw_boxes_on_image
from .vlm_clients import batch_analyze_regions

__version__ = "1.0.0"
__all__ = ["VLMParser", "VLMParserError", "detect_ui_elements", "draw_boxes_on_image", "batch_analyze_regions"]