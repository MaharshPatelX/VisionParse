#!/usr/bin/env python3
"""
VLM Parser CLI Tool - Production Ready Interface
Analyze screenshots and extract UI elements with VLM models
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any

from src.vlm_parser import VLMParser

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def validate_image_path(path: str) -> Path:
    """Validate image path exists and is an image file"""
    image_path = Path(path)
    if not image_path.exists():
        raise argparse.ArgumentTypeError(f"Image file not found: {path}")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    if image_path.suffix.lower() not in valid_extensions:
        raise argparse.ArgumentTypeError(f"Invalid image format: {path}")
    
    return image_path

def validate_confidence(value: str) -> float:
    """Validate confidence threshold is between 0 and 1"""
    try:
        conf = float(value)
        if not 0 <= conf <= 1:
            raise argparse.ArgumentTypeError("Confidence must be between 0 and 1")
        return conf
    except ValueError:
        raise argparse.ArgumentTypeError("Confidence must be a number")

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="VLM Parser - Analyze screenshots and extract UI elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s screenshot.png                     # Interactive mode
  %(prog)s screenshot.png --vlm gpt4o         # Use GPT-4o
  %(prog)s screenshot.png --vlm ollama --model llava:latest
  %(prog)s screenshot.png --output results/   # Custom output directory
  %(prog)s screenshot.png --json-only         # Only output JSON
  %(prog)s *.png --batch                      # Process multiple images
        """
    )

    # Required arguments
    parser.add_argument(
        'images',
        nargs='+',
        type=validate_image_path,
        help='Path to screenshot image(s) to analyze'
    )

    # VLM Configuration
    vlm_group = parser.add_argument_group('VLM Configuration')
    vlm_group.add_argument(
        '--vlm', '--provider',
        help='VLM provider to use - supports any provider name (if not specified, will prompt)'
    )
    vlm_group.add_argument(
        '--model',
        help='Specific model to use (e.g., gpt-4o-mini, llava:latest)'
    )
    vlm_group.add_argument(
        '--api-key',
        help='API key for VLM provider (overrides config file)'
    )

    # YOLO Configuration
    yolo_group = parser.add_argument_group('YOLO Detection')
    yolo_group.add_argument(
        '--confidence', '--conf',
        type=validate_confidence,
        default=0.05,
        help='YOLO confidence threshold (0-1, default: 0.05)'
    )
    yolo_group.add_argument(
        '--iou-threshold',
        type=validate_confidence,
        default=0.1,
        help='YOLO IoU threshold (0-1, default: 0.1)'
    )
    yolo_group.add_argument(
        '--yolo-model',
        help='Path to custom YOLO model'
    )

    # Output Configuration
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory (default: same as input)'
    )
    output_group.add_argument(
        '--json-only',
        action='store_true',
        help='Only output JSON results (no console output)'
    )
    output_group.add_argument(
        '--no-image',
        action='store_true',
        help="Don't save annotated image"
    )
    output_group.add_argument(
        '--pretty-json',
        action='store_true',
        help='Pretty print JSON output'
    )

    # Processing Options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images in batch mode'
    )
    proc_group.add_argument(
        '--config',
        type=Path,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    proc_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    proc_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )

    return parser

def print_results(results: List[Dict[str, Any]], quiet: bool = False, json_only: bool = False):
    """Print analysis results to console"""
    if quiet or json_only:
        return

    print("\n📋 ANALYSIS RESULTS:")
    print("=" * 60)
    
    for result in results:
        bbox = result['bbox']
        print(f"Element {result['id']}:")
        print(f"  📍 Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
        print(f"  🎯 Confidence: {result['confidence']:.3f}")
        print(f"  🏷️  Name: {result['name']}")
        print(f"  🔧 Type: {result['type']}")
        print(f"  💡 Use: {result['use']}")
        print(f"  📝 Description: {result['description']}")
        print("-" * 50)

def save_results(results: List[Dict[str, Any]], image_path: Path, output_dir: Optional[Path], 
                pretty_json: bool = False, quiet: bool = False):
    """Save results to JSON file"""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{image_path.stem}_results.json"
    else:
        json_path = image_path.parent / f"{image_path.stem}_results.json"
    
    indent = 2 if pretty_json else None
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=indent)
    
    if not quiet:
        logging.info(f"Results saved to: {json_path}")
    
    return json_path

def process_single_image(parser_instance: VLMParser, image_path: Path, args) -> Dict[str, Any]:
    """Process a single image and return results"""
    try:
        if not args.quiet:
            logging.info(f"Processing: {image_path.name}")
        
        # Run analysis
        results = parser_instance.analyze(
            image_path=str(image_path),
            save_annotated_image=not args.no_image,
            output_dir=str(args.output) if args.output else None
        )
        
        # Print results to console
        print_results(results['elements'], args.quiet, args.json_only)
        
        # Save JSON results
        json_path = save_results(
            results['elements'], 
            image_path, 
            args.output,
            args.pretty_json,
            args.quiet
        )
        
        # Output JSON if requested
        if args.json_only:
            print(json.dumps(results['elements'], indent=2 if args.pretty_json else None))
        
        return {
            'success': True,
            'image': str(image_path),
            'elements_count': len(results['elements']),
            'json_path': str(json_path),
            'annotated_path': results.get('annotated_image_path')
        }
        
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {str(e)}")
        return {
            'success': False,
            'image': str(image_path),
            'error': str(e)
        }

def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if args.quiet and args.verbose:
        logging.warning("Both --quiet and --verbose specified, using verbose mode")
        args.quiet = False
    
    try:
        # Initialize VLM Parser
        config_path = args.config if args.config.exists() else None
        
        parser_instance = VLMParser(
            config_path=str(config_path) if config_path else None,
            vlm_type=args.vlm,
            model=args.model,
            api_key=args.api_key,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            yolo_model_path=args.yolo_model
        )
        
        # Process images
        if args.batch and len(args.images) > 1:
            # Batch processing
            if not args.quiet:
                logging.info(f"Processing {len(args.images)} images in batch mode...")
            
            results = []
            for image_path in args.images:
                result = process_single_image(parser_instance, image_path, args)
                results.append(result)
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            if not args.quiet:
                logging.info(f"Batch complete: {successful}/{len(results)} images processed successfully")
        
        else:
            # Single image processing
            if len(args.images) > 1 and not args.batch:
                logging.warning("Multiple images provided without --batch flag. Processing first image only.")
            
            result = process_single_image(parser_instance, args.images[0], args)
            
            if result['success'] and not args.quiet:
                logging.info(f"Analysis complete: {result['elements_count']} elements found")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()