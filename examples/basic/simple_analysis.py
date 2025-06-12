#!/usr/bin/env python3
"""
VisionParse Basic Example: Simple Screenshot Analysis

This example demonstrates the most basic usage of VisionParse
to analyze a single screenshot and get UI elements.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import VisionParse, VisionParseError


def simple_analysis_example():
    """Basic screenshot analysis example"""
    
    print("🚀 VisionParse Simple Analysis Example")
    print("=" * 40)
    
    try:
        # Initialize VisionParse with default settings
        parser = VisionParse(vlm_type='gpt4o')
        print(f"✅ VisionParse initialized with {parser.vlm_type}")
        
        # Sample image path (you can change this)
        image_path = "sample_screenshot.png"
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            print("Please provide a valid screenshot path or use the sample images from examples/sample_images/")
            return
        
        print(f"🔍 Analyzing: {image_path}")
        
        # Analyze the screenshot
        results = parser.analyze(image_path)
        
        # Check if analysis was successful
        if results['success']:
            print(f"✅ Analysis completed successfully!")
            print(f"📊 Found {len(results['elements'])} UI elements")
            print(f"🤖 Used model: {results['model']}")
            print(f"🎯 YOLO detections: {results['yolo_detections']}")
            
            # Display results
            if results['elements']:
                print("\n📋 Detected UI Elements:")
                print("-" * 50)
                
                for i, element in enumerate(results['elements'], 1):
                    print(f"\nElement {i}:")
                    print(f"  📛 Name: {element['name']}")
                    print(f"  🔧 Type: {element['type']}")
                    print(f"  📍 Location: {element['bbox']}")
                    print(f"  🎯 Confidence: {element['confidence']:.2f}")
                    print(f"  💡 Use: {element['use']}")
                    print(f"  📝 Description: {element['description']}")
            
            # Show output files
            if results.get('annotated_image_path'):
                print(f"\n🖼️  Annotated image saved: {results['annotated_image_path']}")
            
            print(f"\n💾 Full results available in JSON format")
            
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except VisionParseError as e:
        print(f"❌ VisionParse Error: {e}")
        print("\n💡 Tips:")
        print("- Make sure your API key is set in .env file")
        print("- Check if the YOLO model exists in weights/icon_detect/model.pt")
        print("- Verify the image path is correct")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("- Check the troubleshooting guide for help")


def quick_test():
    """Quick test with sample data"""
    print("\n🧪 Quick Test Mode")
    print("=" * 20)
    
    try:
        # Test initialization only
        parser = VisionParse(vlm_type='gpt4o')
        print("✅ VisionParse can be initialized")
        
        # Check API key
        if parser.api_key:
            print("✅ API key is configured")
        else:
            print("⚠️  No API key found - consider using Ollama for local testing")
        
        # Check YOLO model
        if os.path.exists(parser.yolo_model_path):
            print("✅ YOLO model found")
        else:
            print(f"❌ YOLO model not found: {parser.yolo_model_path}")
            
    except Exception as e:
        print(f"❌ Setup issue: {e}")


if __name__ == "__main__":
    # You can run this script directly
    print("VisionParse Basic Analysis Example\n")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            quick_test()
        else:
            # Use provided image path
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                simple_analysis_example()
            else:
                print(f"❌ Image not found: {image_path}")
    else:
        # Run with default settings
        simple_analysis_example()
    
    print("\n📚 Next steps:")
    print("- Try examples/basic/batch_processing.py for multiple images")
    print("- Check examples/advanced/ for more complex use cases")
    print("- See examples/providers/ for different VLM providers")