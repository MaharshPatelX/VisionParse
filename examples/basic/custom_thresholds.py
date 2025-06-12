#!/usr/bin/env python3
"""
VisionParse Basic Example: Custom YOLO Thresholds

This example demonstrates how to adjust YOLO detection thresholds
to customize sensitivity and filtering for different types of images.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from VisionParse import VisionParse, VisionParseError


def threshold_comparison_example():
    """Compare results with different threshold settings"""
    
    print("🎯 VisionParse Threshold Comparison Example")
    print("=" * 50)
    
    # Sample image path
    image_path = "sample_screenshot.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("Please provide a valid screenshot for threshold testing")
        return
    
    try:
        # Define different threshold configurations
        threshold_configs = [
            {
                'name': 'Very Sensitive',
                'confidence': 0.05,
                'iou': 0.1,
                'description': 'Detects many elements, may include false positives'
            },
            {
                'name': 'Moderate',
                'confidence': 0.2,
                'iou': 0.4,
                'description': 'Balanced detection with reasonable filtering'
            },
            {
                'name': 'Conservative',
                'confidence': 0.4,
                'iou': 0.7,
                'description': 'Only confident detections, may miss some elements'
            },
            {
                'name': 'Very Strict',
                'confidence': 0.6,
                'iou': 0.8,
                'description': 'Only very confident detections'
            }
        ]
        
        print(f"🔍 Testing image: {image_path}")
        print(f"📊 Comparing {len(threshold_configs)} threshold configurations...\n")
        
        # Initialize parser
        parser = VisionParse(provider='openai', model='gpt-4o')
        
        results_comparison = []
        
        for config in threshold_configs:
            print(f"🧪 Testing: {config['name']}")
            print(f"   Confidence: {config['confidence']}")
            print(f"   IoU: {config['iou']}")
            print(f"   Description: {config['description']}")
            
            # Update thresholds
            parser.confidence_threshold = config['confidence']
            parser.iou_threshold = config['iou']
            
            # Analyze with current settings
            results = parser.analyze(
                image_path,
                save_annotated_image=True
            )
            
            if results['success']:
                element_count = len(results['elements'])
                yolo_count = results['yolo_detections']
                
                print(f"   📈 Results: {element_count} elements (YOLO: {yolo_count})")
                
                # Calculate confidence statistics
                if results['elements']:
                    confidences = [el['confidence'] for el in results['elements']]
                    avg_confidence = sum(confidences) / len(confidences)
                    min_confidence = min(confidences)
                    max_confidence = max(confidences)
                    
                    print(f"   🎯 Confidence: avg={avg_confidence:.3f}, min={min_confidence:.3f}, max={max_confidence:.3f}")
                
                results_comparison.append({
                    'config': config,
                    'element_count': element_count,
                    'yolo_count': yolo_count,
                    'elements': results['elements']
                })
                
            else:
                print(f"   ❌ Analysis failed: {results.get('error')}")
                
            print()
        
        # Analysis summary
        if results_comparison:
            print("📊 Comparison Summary:")
            print("-" * 60)
            print(f"{'Configuration':<15} {'Elements':<10} {'YOLO':<8} {'Avg Conf':<10}")
            print("-" * 60)
            
            for result in results_comparison:
                config_name = result['config']['name']
                element_count = result['element_count']
                yolo_count = result['yolo_count']
                
                if result['elements']:
                    avg_conf = sum(el['confidence'] for el in result['elements']) / len(result['elements'])
                    avg_conf_str = f"{avg_conf:.3f}"
                else:
                    avg_conf_str = "N/A"
                
                print(f"{config_name:<15} {element_count:<10} {yolo_count:<8} {avg_conf_str:<10}")
            
            print("-" * 60)
            
            # Recommendations
            print("\n💡 Recommendations:")
            
            # Find best balanced result
            if len(results_comparison) >= 2:
                middle_result = results_comparison[1]  # Usually the moderate setting
                print(f"   🎯 For balanced results: {middle_result['config']['name']}")
                print(f"      Confidence: {middle_result['config']['confidence']}")
                print(f"      IoU: {middle_result['config']['iou']}")
            
            # Find result with most elements
            max_elements_result = max(results_comparison, key=lambda x: x['element_count'])
            print(f"   🔍 For maximum detection: {max_elements_result['config']['name']}")
            print(f"      Found {max_elements_result['element_count']} elements")
            
            # Find most confident result
            confident_results = [r for r in results_comparison if r['elements']]
            if confident_results:
                most_confident = max(confident_results, 
                                   key=lambda x: sum(el['confidence'] for el in x['elements']) / len(x['elements']))
                avg_conf = sum(el['confidence'] for el in most_confident['elements']) / len(most_confident['elements'])
                print(f"   ⭐ For high confidence: {most_confident['config']['name']}")
                print(f"      Average confidence: {avg_conf:.3f}")
        
    except VisionParseError as e:
        print(f"❌ VisionParse Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def adaptive_threshold_example():
    """Automatically find optimal thresholds for an image"""
    
    print("🔄 Adaptive Threshold Example")
    print("=" * 35)
    
    image_path = "sample_screenshot.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    try:
        parser = VisionParse(provider='openai', model='gpt-4o')
        
        def find_optimal_thresholds(image_path, target_elements=10):
            """Find thresholds that produce close to target number of elements"""
            
            print(f"🎯 Finding optimal thresholds for ~{target_elements} elements")
            
            # Test different configurations
            test_configs = [
                (0.05, 0.1), (0.1, 0.2), (0.15, 0.3), (0.2, 0.4),
                (0.25, 0.5), (0.3, 0.6), (0.4, 0.7), (0.5, 0.8)
            ]
            
            best_config = None
            best_score = float('inf')
            all_results = []
            
            for conf_thresh, iou_thresh in test_configs:
                parser.confidence_threshold = conf_thresh
                parser.iou_threshold = iou_thresh
                
                results = parser.analyze(image_path, save_annotated_image=False)
                
                if results['success']:
                    element_count = len(results['elements'])
                    score = abs(element_count - target_elements)
                    
                    all_results.append({
                        'conf': conf_thresh,
                        'iou': iou_thresh,
                        'elements': element_count,
                        'score': score
                    })
                    
                    print(f"   Conf: {conf_thresh:.2f}, IoU: {iou_thresh:.2f} → {element_count} elements")
                    
                    if score < best_score:
                        best_score = score
                        best_config = (conf_thresh, iou_thresh, element_count)
            
            return best_config, all_results
        
        # Find optimal configuration
        optimal_config, all_results = find_optimal_thresholds(image_path)
        
        if optimal_config:
            conf, iou, count = optimal_config
            print(f"\n🎯 Optimal Configuration Found:")
            print(f"   Confidence Threshold: {conf}")
            print(f"   IoU Threshold: {iou}")
            print(f"   Elements Detected: {count}")
            
            # Apply optimal settings and get final result
            parser.confidence_threshold = conf
            parser.iou_threshold = iou
            
            final_results = parser.analyze(image_path, save_annotated_image=True)
            
            if final_results['success']:
                print(f"\n✅ Final analysis with optimal settings:")
                print(f"   📊 Elements: {len(final_results['elements'])}")
                print(f"   🖼️  Annotated image: {final_results.get('annotated_image_path')}")
        
    except Exception as e:
        print(f"❌ Error in adaptive threshold: {e}")


def image_type_specific_thresholds():
    """Show recommended thresholds for different image types"""
    
    print("📱 Image Type Specific Thresholds")
    print("=" * 40)
    
    # Predefined configurations for different scenarios
    configurations = {
        'web_ui': {
            'name': 'Web Interface',
            'confidence': 0.1,
            'iou': 0.3,
            'description': 'Many small clickable elements, buttons, links'
        },
        'desktop_app': {
            'name': 'Desktop Application',
            'confidence': 0.2,
            'iou': 0.5,
            'description': 'Larger UI elements, menus, toolbars'
        },
        'mobile_app': {
            'name': 'Mobile Application',
            'confidence': 0.15,
            'iou': 0.4,
            'description': 'Touch-friendly elements, bottom navigation'
        },
        'complex_ui': {
            'name': 'Complex Interface',
            'confidence': 0.05,
            'iou': 0.2,
            'description': 'Dense UIs with many overlapping elements'
        },
        'simple_ui': {
            'name': 'Simple Interface',
            'confidence': 0.3,
            'iou': 0.6,
            'description': 'Clean UIs with few, distinct elements'
        }
    }
    
    print("💡 Recommended threshold configurations:")
    print()
    
    for config_id, config in configurations.items():
        print(f"🏷️  {config['name']}:")
        print(f"   Confidence: {config['confidence']}")
        print(f"   IoU: {config['iou']}")
        print(f"   Use case: {config['description']}")
        print(f"   Code: parser.confidence_threshold = {config['confidence']}")
        print(f"         parser.iou_threshold = {config['iou']}")
        print()
    
    # Example of using specific configuration
    print("📝 Example usage:")
    print("```python")
    print("# For mobile app analysis")
    print("parser = VisionParse(provider='openai', model='gpt-4o')")
    print("parser.confidence_threshold = 0.15")
    print("parser.iou_threshold = 0.4")
    print("results = parser.analyze('mobile_screenshot.png')")
    print("```")


if __name__ == "__main__":
    print("VisionParse Threshold Configuration Examples\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--compare":
            threshold_comparison_example()
        elif command == "--adaptive":
            adaptive_threshold_example()
        elif command == "--types":
            image_type_specific_thresholds()
        else:
            print("Usage:")
            print("  python custom_thresholds.py --compare    # Compare different thresholds")
            print("  python custom_thresholds.py --adaptive   # Find optimal thresholds")
            print("  python custom_thresholds.py --types      # Show type-specific configs")
    else:
        # Run all examples
        threshold_comparison_example()
        print("\n" + "="*60 + "\n")
        adaptive_threshold_example()
        print("\n" + "="*60 + "\n")
        image_type_specific_thresholds()
    
    print("\n📚 Next steps:")
    print("- Experiment with different threshold values for your specific images")
    print("- Check examples/advanced/ for more sophisticated usage")
    print("- Try examples/providers/ to compare different VLM providers")