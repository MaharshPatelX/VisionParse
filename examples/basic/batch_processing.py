#!/usr/bin/env python3
"""
VisionParse Basic Example: Batch Processing

This example shows how to process multiple screenshots at once
and get aggregated results.
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from VisionParse import VisionParse, VisionParseError


def batch_processing_example():
    """Process multiple images in batch"""
    
    print("📦 VisionParse Batch Processing Example")
    print("=" * 45)
    
    try:
        # Initialize VisionParse
        parser = VisionParse(
            provider='openai',
            model='gpt-4o',
            verbose=True  # Enable verbose output for batch processing
        )
        print(f"✅ VisionParse initialized with {parser.provider}")
        
        # Find image files to process
        image_patterns = [
            "screenshots/*.png",
            "screenshots/*.jpg", 
            "examples/sample_images/*.png",
            "*.png",
            "*.jpg"
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern))
        
        # Remove duplicates and check files exist
        image_files = list(set([f for f in image_files if os.path.exists(f)]))
        
        if not image_files:
            print("⚠️  No image files found!")
            print("Please add some screenshots to process:")
            print("- Create a 'screenshots/' folder with PNG/JPG files")
            print("- Use the sample images in examples/sample_images/")
            print("- Place images in the current directory")
            return
        
        print(f"🔍 Found {len(image_files)} images to process:")
        for img in image_files:
            print(f"  - {img}")
        
        # Create output directory for batch results
        output_dir = "batch_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🚀 Starting batch analysis...")
        print(f"📁 Output directory: {output_dir}")
        
        # Process images in batch
        batch_results = parser.analyze_batch(
            image_paths=image_files,
            save_annotated_images=True,
            output_dir=output_dir
        )
        
        # Display batch summary
        summary = batch_results['summary']
        print(f"\n📊 Batch Processing Summary:")
        print(f"  Total images: {summary['total']}")
        print(f"  ✅ Successful: {summary['successful']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  🤖 Provider used: {batch_results.get('provider', 'N/A')}")
        print(f"  🔧 Model: {batch_results['model']}")
        
        # Analyze results in detail
        successful_results = []
        failed_results = []
        total_elements = 0
        
        for result in batch_results['results']:
            image_name = Path(result['image_path']).name
            
            if result['success']:
                element_count = len(result['elements'])
                total_elements += element_count
                successful_results.append((image_name, element_count, result))
                print(f"  ✅ {image_name}: {element_count} elements")
            else:
                failed_results.append((image_name, result['error']))
                print(f"  ❌ {image_name}: {result['error']}")
        
        # Show detailed statistics
        if successful_results:
            print(f"\n📈 Analysis Statistics:")
            print(f"  Total UI elements found: {total_elements}")
            print(f"  Average elements per image: {total_elements / len(successful_results):.1f}")
            
            # Find image with most/least elements
            max_elements = max(successful_results, key=lambda x: x[1])
            min_elements = min(successful_results, key=lambda x: x[1])
            print(f"  Most elements: {max_elements[0]} ({max_elements[1]} elements)")
            print(f"  Least elements: {min_elements[0]} ({min_elements[1]} elements)")
        
        # Show common element types
        if successful_results:
            element_types = {}
            for _, _, result in successful_results:
                for element in result['elements']:
                    elem_type = element['type']
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            if element_types:
                print(f"\n🏷️  Common Element Types:")
                sorted_types = sorted(element_types.items(), key=lambda x: x[1], reverse=True)
                for elem_type, count in sorted_types[:5]:  # Top 5
                    print(f"  {elem_type}: {count} occurrences")
        
        # Export aggregated results
        export_path = f"{output_dir}/batch_summary.json"
        parser.export_results(batch_results, export_path, format='json')
        print(f"\n💾 Batch results exported to: {export_path}")
        
        # Show files created
        print(f"\n📁 Files created in {output_dir}/:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
            
    except VisionParseError as e:
        print(f"❌ VisionParse Error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def process_directory(directory_path):
    """Process all images in a specific directory"""
    
    print(f"📂 Processing directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    try:
        parser = VisionParse(provider='openai', model='gpt-4o')
        
        # Find images in directory
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory_path, ext)
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            print(f"⚠️  No images found in {directory_path}")
            return
        
        print(f"🔍 Found {len(image_files)} images")
        
        # Create output directory
        output_dir = f"{directory_path}_analysis"
        
        # Process batch
        results = parser.analyze_batch(
            image_paths=image_files,
            output_dir=output_dir
        )
        
        print(f"✅ Processed {results['summary']['successful']} images")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing directory: {e}")


def selective_batch_processing():
    """Process only images that match certain criteria"""
    
    print("🎯 Selective Batch Processing Example")
    print("=" * 40)
    
    try:
        parser = VisionParse(provider='openai', model='gpt-4o')
        
        # Find all images
        all_images = glob.glob("**/*.png", recursive=True) + glob.glob("**/*.jpg", recursive=True)
        
        if not all_images:
            print("⚠️  No images found for selective processing")
            return
        
        # Filter criteria (customize as needed)
        def should_process_image(image_path):
            """Determine if image should be processed"""
            file_size = os.path.getsize(image_path)
            filename = os.path.basename(image_path).lower()
            
            # Example criteria:
            # - File size between 50KB and 10MB
            # - Filename contains certain keywords
            # - File is recent (modify as needed)
            
            size_ok = 50_000 < file_size < 10_000_000  # 50KB - 10MB
            name_ok = any(keyword in filename for keyword in ['screenshot', 'ui', 'interface', 'app'])
            
            return size_ok or name_ok
        
        # Filter images
        selected_images = [img for img in all_images if should_process_image(img)]
        
        print(f"📋 Selected {len(selected_images)} images out of {len(all_images)} total:")
        for img in selected_images:
            print(f"  - {img}")
        
        if not selected_images:
            print("⚠️  No images met the selection criteria")
            return
        
        # Process selected images
        results = parser.analyze_batch(
            image_paths=selected_images,
            output_dir="selective_analysis"
        )
        
        print(f"\n✅ Selective processing complete!")
        print(f"📊 Results: {results['summary']}")
        
    except Exception as e:
        print(f"❌ Error in selective processing: {e}")


if __name__ == "__main__":
    print("VisionParse Batch Processing Examples\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--directory" and len(sys.argv) > 2:
            # Process specific directory
            directory = sys.argv[2]
            process_directory(directory)
            
        elif command == "--selective":
            # Selective processing
            selective_batch_processing()
            
        else:
            print("Usage:")
            print("  python batch_processing.py                    # Process found images")
            print("  python batch_processing.py --directory DIR    # Process specific directory")
            print("  python batch_processing.py --selective        # Selective processing")
    else:
        # Default batch processing
        batch_processing_example()
    
    print("\n📚 Next steps:")
    print("- Try examples/basic/custom_thresholds.py to adjust detection sensitivity")
    print("- Check examples/advanced/library_integration.py for app integration")
    print("- Explore examples/use_cases/ for specific applications")