"""
VisionParse - Production Ready Python API
A comprehensive tool for analyzing screenshots and extracting UI elements using VLM models
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import traceback

from .yolo_detector import detect_ui_elements, draw_boxes_on_image, crop_image_regions
from .vlm_clients import batch_analyze_regions, get_available_ollama_models

class VisionParseError(Exception):
    """Custom exception for VisionParse errors"""
    pass

class VisionParse:
    """
    Production-ready VisionParse for UI element analysis
    
    Features:
    - Multiple VLM providers (OpenAI, Claude, Gemini, Ollama)
    - YOLO-based UI detection
    - Configurable thresholds and parameters
    - Batch processing support
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        yolo_model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        verbose: bool = False,
        # Backward compatibility
        vlm_type: Optional[str] = None
    ):
        """
        Initialize VisionParse
        
        Args:
            provider: VLM provider (openai, anthropic, google, ollama)
            model: Specific model name (gpt-4o, claude-3-5-sonnet, gemini-2.0-flash-exp, etc.)
            api_key: API key for VLM provider
            confidence_threshold: YOLO confidence threshold (0.0-1.0)
            iou_threshold: YOLO IoU threshold (0.0-1.0)
            yolo_model_path: Path to custom YOLO model
            config_path: Path to configuration file
            verbose: Enable verbose logging
            vlm_type: Deprecated, use provider instead
        """
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Handle provider/vlm_type (backward compatibility)
        self.provider = provider or vlm_type or self._get_provider()
        self.vlm_type = self.provider  # Keep for backward compatibility
        
        # Set parameters
        self.api_key = api_key or self._get_api_key()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.yolo_model_path = yolo_model_path or self.config.get('yolo_model_path', 'weights/icon_detect/model.pt')
        
        # Set model after provider is set
        self.model = model or self._get_default_model()
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info(f"VisionParse initialized with {self.provider.upper()}")
        if self.model:
            self.logger.info(f"Using model: {self.model}")
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.debug(f"Loaded config from {config_path}")
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'provider': 'openai',
            'vlm_type': 'openai',  # Backward compatibility
            'default_models': {
                'openai': 'gpt-4o',
                'anthropic': 'claude-3-5-sonnet',
                'google': 'gemini-2.0-flash-exp',
                'ollama': 'llava:latest'
            },
            'yolo_model_path': 'weights/icon_detect/model.pt',
            'confidence_threshold': 0.05,
            'yolo_config': {
                'box_threshold': 0.05,
                'iou_threshold': 0.1
            },
            'ollama_config': {
                'base_url': 'http://localhost:11434',
                'timeout': 60
            }
        }
    
    def _get_provider(self) -> str:
        """Get VLM provider with fallback logic"""
        # Check for environment variables
        for env_var in ['VLM_PROVIDER', 'VLM_TYPE', 'PROVIDER']:
            if os.getenv(env_var):
                return os.getenv(env_var).lower()
        
        # Use config default
        return self.config.get('provider', self.config.get('vlm_type', 'openai'))
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources - supports any VLM type"""
        # 1. Provided directly
        if hasattr(self, 'api_key') and self.api_key:
            return self.api_key
        
        # 2. Environment variables - check common patterns
        provider_lower = self.provider.lower()
        
        # Try standard environment variable patterns
        env_var_candidates = [
            f"{self.provider.upper()}_API_KEY",
            f"{self.provider.upper()}_KEY",
        ]
        
        # Add pattern-based candidates
        if any(pattern in provider_lower for pattern in ['gpt', 'openai']):
            env_var_candidates.extend(['OPENAI_API_KEY', 'OPENAI_KEY'])
        elif any(pattern in provider_lower for pattern in ['claude', 'anthropic']):
            env_var_candidates.extend(['ANTHROPIC_API_KEY', 'ANTHROPIC_KEY'])
        elif any(pattern in provider_lower for pattern in ['gemini', 'google']):
            env_var_candidates.extend(['GOOGLE_API_KEY', 'GOOGLE_KEY'])
        elif any(pattern in provider_lower for pattern in ['ollama', 'local']):
            return None  # No API key needed for local models
        
        # Check environment variables
        for env_var in env_var_candidates:
            if os.getenv(env_var):
                return os.getenv(env_var)
        
        # 3. Config file
        api_keys = self.config.get('api_keys', {})
        
        # Try direct key lookup first
        if self.provider in api_keys:
            return api_keys[self.provider]
        
        # Pattern-based key mapping
        if any(pattern in provider_lower for pattern in ['gpt', 'openai']):
            return api_keys.get('openai')
        elif any(pattern in provider_lower for pattern in ['claude', 'anthropic']):
            return api_keys.get('anthropic')
        elif any(pattern in provider_lower for pattern in ['gemini', 'google']):
            return api_keys.get('google')
        
        return None
    
    def _validate_config(self):
        """Validate configuration and requirements"""
        # Check YOLO model
        if not os.path.exists(self.yolo_model_path):
            raise VisionParseError(f"YOLO model not found: {self.yolo_model_path}")
        
        # Check API key for non-local models
        if not any(keyword in self.provider.lower() for keyword in ['ollama', 'local']) and not self.api_key:
            raise VisionParseError(f"API key required for {self.provider.upper()}")
        
        # Validate thresholds
        if not 0 <= self.confidence_threshold <= 1:
            raise VisionParseError("Confidence threshold must be between 0 and 1")
        
        if not 0 <= self.iou_threshold <= 1:
            raise VisionParseError("IoU threshold must be between 0 and 1")
    
    def _get_default_model(self) -> str:
        """Get default model for the current provider"""
        # Allow any model name - provide sensible defaults based on provider patterns
        default_models = self.config.get('default_models', {})
        
        # Check for exact matches first
        if self.provider in default_models:
            return default_models[self.provider]
        
        # Pattern-based fallbacks for common providers
        if 'gpt' in self.provider.lower() or 'openai' in self.provider.lower():
            return default_models.get('openai', 'gpt-4o')
        elif 'claude' in self.provider.lower() or 'anthropic' in self.provider.lower():
            return default_models.get('anthropic', 'claude-3-5-sonnet')
        elif 'gemini' in self.provider.lower() or 'google' in self.provider.lower():
            return default_models.get('google', 'gemini-2.0-flash-exp')
        elif 'ollama' in self.provider.lower():
            return default_models.get('ollama', 'llava:latest')
        else:
            # Default fallback for any unknown provider
            return 'gpt-4o'
    
    def analyze(
        self,
        image_path: Union[str, Path],
        save_annotated_image: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a screenshot and extract UI elements
        
        Args:
            image_path: Path to screenshot image
            save_annotated_image: Whether to save annotated image
            output_dir: Output directory for results
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise VisionParseError(f"Image not found: {image_path}")
            
            self.logger.info(f"Analyzing image: {image_path.name}")
            
            # Step 1: YOLO Detection
            self.logger.debug("Running YOLO detection...")
            detected_boxes = detect_ui_elements(
                image_path=str(image_path),
                model_path=self.yolo_model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold
            )
            
            if not detected_boxes:
                self.logger.warning("No UI elements detected")
                return {
                    'success': True,
                    'elements': [],
                    'image_path': str(image_path),
                    'yolo_detections': 0
                }
            
            self.logger.info(f"YOLO detected {len(detected_boxes)} UI elements")
            
            # Step 2: Crop regions for VLM analysis
            self.logger.debug("Cropping regions for VLM analysis...")
            cropped_regions = crop_image_regions(str(image_path), detected_boxes)
            
            # Step 3: VLM Analysis
            self.logger.info(f"Analyzing with {self.provider.upper()}...")
            results = batch_analyze_regions(
                cropped_regions=cropped_regions,
                vlm_type=self.provider,  # Keep vlm_type for backward compatibility
                api_key=self.api_key,
                model=self.model
            )
            
            # Step 4: Save annotated image if requested
            annotated_image_path = None
            if save_annotated_image:
                if output_dir:
                    output_path = Path(output_dir) / f"{image_path.stem}_analyzed{image_path.suffix}"
                else:
                    output_path = image_path.parent / f"{image_path.stem}_analyzed{image_path.suffix}"
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                draw_boxes_on_image(str(image_path), detected_boxes, str(output_path))
                annotated_image_path = str(output_path)
                self.logger.debug(f"Annotated image saved: {output_path}")
            
            return {
                'success': True,
                'elements': results,
                'image_path': str(image_path),
                'annotated_image_path': annotated_image_path,
                'yolo_detections': len(detected_boxes),
                'vlm_type': self.vlm_type,
                'model': self.model,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            }
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'image_path': str(image_path)
            }
    
    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_annotated_images: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple images in batch
        
        Args:
            image_paths: List of image paths
            save_annotated_images: Whether to save annotated images
            output_dir: Output directory for results
        
        Returns:
            Dictionary containing batch results
        """
        self.logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            result = self.analyze(
                image_path=image_path,
                save_annotated_image=save_annotated_images,
                output_dir=output_dir
            )
            
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
                self.logger.error(f"Failed to process {image_path}: {result.get('error', 'Unknown error')}")
        
        self.logger.info(f"Batch complete: {successful} successful, {failed} failed")
        
        return {
            'success': True,
            'results': results,
            'summary': {
                'total': len(image_paths),
                'successful': successful,
                'failed': failed
            },
            'vlm_type': self.vlm_type,
            'model': self.model
        }
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> str:
        """
        Export results to file
        
        Args:
            results: Analysis results
            output_path: Output file path
            format: Export format ('json', 'csv')
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format.lower() == 'csv':
            import csv
            
            with open(output_path, 'w', newline='') as f:
                if 'elements' in results:
                    elements = results['elements']
                    if elements:
                        writer = csv.DictWriter(f, fieldnames=elements[0].keys())
                        writer.writeheader()
                        writer.writerows(elements)
                else:
                    # Batch results
                    fieldnames = ['image_path', 'success', 'elements_count', 'error']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results.get('results', []):
                        row = {
                            'image_path': result.get('image_path', ''),
                            'success': result.get('success', False),
                            'elements_count': len(result.get('elements', [])),
                            'error': result.get('error', '')
                        }
                        writer.writerow(row)
        
        else:
            raise VisionParseError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Results exported to: {output_path}")
        return str(output_path)