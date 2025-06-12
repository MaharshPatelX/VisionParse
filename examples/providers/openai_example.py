#!/usr/bin/env python3
"""
VisionParse Provider Example: OpenAI/GPT Models

This example demonstrates using different OpenAI models with VisionParse
and shows the latest GPT model capabilities.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from VisionParse import VisionParse, VisionParseError


def openai_models_showcase():
    """Showcase different OpenAI models"""
    
    print("🤖 OpenAI/GPT Models Showcase")
    print("=" * 35)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return
    
    print("✅ OpenAI API key found")
    
    # Sample image
    image_path = "sample_screenshot.png"
    if not os.path.exists(image_path):
        print(f"⚠️  Sample image not found: {image_path}")
        print("Please provide a screenshot to analyze")
        return
    
    # OpenAI models to test (as of 2025)
    openai_models = [
        {
            'name': 'GPT-4.1 (Latest Flagship)',
            'model_id': 'gpt-4.1',
            'description': 'Latest flagship model with superior coding and instruction following',
            'features': ['1M token context', 'Advanced reasoning', 'Best coding performance']
        },
        {
            'name': 'GPT-4.1 Mini (Fast & Efficient)',
            'model_id': 'gpt-4.1-mini',
            'description': 'Significant performance leap, beats GPT-4o in many benchmarks',
            'features': ['50% lower latency', '83% cost reduction', 'Strong performance']
        },
        {
            'name': 'GPT-4.1 Nano (Fastest)',
            'model_id': 'gpt-4.1-nano',
            'description': 'Fastest and cheapest model with exceptional performance',
            'features': ['1M token context', 'Ultra-fast responses', 'Most cost-effective']
        },
        {
            'name': 'GPT-4o (Multimodal)',
            'model_id': 'gpt-4o',
            'description': 'Integrates text and images, enhanced vision capabilities',
            'features': ['Native multimodal', 'Strong vision', 'Proven reliability']
        }
    ]
    
    print(f"🔍 Testing {len(openai_models)} OpenAI models with: {image_path}\n")
    
    results_comparison = []
    
    for model_info in openai_models:
        print(f"🧪 Testing: {model_info['name']}")
        print(f"   Model ID: {model_info['model_id']}")
        print(f"   Description: {model_info['description']}")
        print(f"   Features: {', '.join(model_info['features'])}")
        
        try:
            # Initialize parser with specific model
            parser = VisionParse(
                provider='openai',
                model=model_info['model_id'],
                api_key=api_key
            )
            
            # Analyze image
            results = parser.analyze(image_path, save_annotated_image=False)
            
            if results['success']:
                element_count = len(results['elements'])
                print(f"   ✅ Success: Found {element_count} UI elements")
                
                # Analyze quality of descriptions
                if results['elements']:
                    descriptions = [el['description'] for el in results['elements']]
                    avg_desc_length = sum(len(desc) for desc in descriptions) / len(descriptions)
                    print(f"   📝 Average description length: {avg_desc_length:.1f} characters")
                    
                    # Show sample descriptions
                    print(f"   💡 Sample descriptions:")
                    for i, element in enumerate(results['elements'][:2]):  # Show first 2
                        print(f"      {i+1}. {element['name']}: {element['description'][:100]}...")
                
                results_comparison.append({
                    'model': model_info,
                    'success': True,
                    'element_count': element_count,
                    'results': results
                })
                
            else:
                print(f"   ❌ Failed: {results.get('error')}")
                results_comparison.append({
                    'model': model_info,
                    'success': False,
                    'error': results.get('error')
                })
                
        except VisionParseError as e:
            print(f"   ❌ VisionParse Error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected Error: {e}")
        
        print()
    
    # Comparison summary
    successful_results = [r for r in results_comparison if r['success']]
    
    if successful_results:
        print("📊 Model Comparison Summary:")
        print("-" * 80)
        print(f"{'Model':<25} {'Elements':<10} {'Performance':<15}")
        print("-" * 80)
        
        for result in successful_results:
            model_name = result['model']['name'].split('(')[0].strip()
            element_count = result['element_count']
            
            # Simple performance indicator
            if element_count > 10:
                performance = "High Detection"
            elif element_count > 5:
                performance = "Moderate"
            else:
                performance = "Conservative"
            
            print(f"{model_name:<25} {element_count:<10} {performance:<15}")
        
        print("-" * 80)
        
        # Recommendations
        print("\n💡 Model Recommendations:")
        
        if len(successful_results) > 1:
            # Find model with most elements
            max_elements = max(successful_results, key=lambda x: x['element_count'])
            print(f"   🔍 Best detection: {max_elements['model']['name']}")
            print(f"      Found {max_elements['element_count']} elements")
            
        print(f"   ⚡ Fastest: GPT-4.1 Nano (lowest latency)")
        print(f"   💰 Most cost-effective: GPT-4.1 Nano")
        print(f"   🎯 Best overall: GPT-4.1 (flagship performance)")
        print(f"   🏃 Balanced: GPT-4.1 Mini (speed + performance)")


def openai_advanced_features():
    """Demonstrate advanced OpenAI-specific features"""
    
    print("\n🚀 OpenAI Advanced Features")
    print("=" * 35)
    
    try:
        # Use latest GPT-4.1 model
        parser = VisionParse(
            provider='openai',
            model='gpt-4.1',
            verbose=True
        )
        
        image_path = "sample_screenshot.png"
        if not os.path.exists(image_path):
            print("⚠️  No sample image for advanced features demo")
            return
        
        print("🧠 Using GPT-4.1 with advanced analysis...")
        
        # Custom analysis with detailed prompting
        results = parser.analyze(image_path)
        
        if results['success'] and results['elements']:
            print(f"✅ Advanced analysis complete: {len(results['elements'])} elements")
            
            # Analyze element quality
            elements = results['elements']
            
            # Group by element type
            element_types = {}
            for element in elements:
                elem_type = element['type']
                if elem_type not in element_types:
                    element_types[elem_type] = []
                element_types[elem_type].append(element)
            
            print(f"\n📊 Element Analysis:")
            for elem_type, type_elements in element_types.items():
                print(f"   {elem_type}: {len(type_elements)} elements")
                
                # Show most confident element of this type
                if type_elements:
                    best_element = max(type_elements, key=lambda x: x['confidence'])
                    print(f"      Best: {best_element['name']} (confidence: {best_element['confidence']:.3f})")
            
            # Find interactive elements
            interactive_types = ['Button', 'Link', 'Icon', 'Menu', 'Input']
            interactive_elements = [el for el in elements if el['type'] in interactive_types]
            
            print(f"\n🖱️  Interactive Elements: {len(interactive_elements)}")
            for elem in interactive_elements[:3]:  # Show top 3
                print(f"   • {elem['name']}: {elem['use']}")
            
        else:
            print("❌ Advanced analysis failed")
            
    except Exception as e:
        print(f"❌ Error in advanced features: {e}")


def cost_comparison():
    """Compare costs of different OpenAI models"""
    
    print("\n💰 OpenAI Model Cost Comparison")
    print("=" * 40)
    
    # Approximate pricing (as of 2025)
    model_pricing = {
        'gpt-4.1': {
            'input': 30.0,   # $ per 1M tokens
            'output': 60.0,
            'description': 'Premium flagship model'
        },
        'gpt-4.1-mini': {
            'input': 5.0,    # 83% cost reduction vs GPT-4o
            'output': 15.0,
            'description': 'Best performance per dollar'
        },
        'gpt-4.1-nano': {
            'input': 2.0,    # Most cost-effective
            'output': 8.0,
            'description': 'Ultra cost-effective'
        },
        'gpt-4o': {
            'input': 15.0,
            'output': 30.0,
            'description': 'Balanced multimodal'
        }
    }
    
    print("📋 Estimated costs for UI analysis (per 1M tokens):")
    print("-" * 60)
    print(f"{'Model':<15} {'Input':<8} {'Output':<8} {'Note':<20}")
    print("-" * 60)
    
    for model, pricing in model_pricing.items():
        input_cost = f"${pricing['input']:.1f}"
        output_cost = f"${pricing['output']:.1f}"
        print(f"{model:<15} {input_cost:<8} {output_cost:<8} {pricing['description']:<20}")
    
    print("-" * 60)
    print("\n💡 Cost Tips:")
    print("   • GPT-4.1 Nano: Best for high-volume analysis")
    print("   • GPT-4.1 Mini: Best performance per dollar")
    print("   • GPT-4.1: Use for complex/critical analysis")
    print("   • Image analysis typically uses ~500-2000 tokens per request")


def error_handling_example():
    """Demonstrate error handling with OpenAI API"""
    
    print("\n🛠️  Error Handling Example")
    print("=" * 30)
    
    try:
        # Test with invalid model
        print("🧪 Testing with invalid model...")
        parser = VisionParse(
            provider='openai',
            model='invalid-model-name'
        )
        
        results = parser.analyze('sample_screenshot.png')
        
        if not results['success']:
            print(f"✅ Error handled gracefully: {results.get('error')}")
        
    except Exception as e:
        print(f"✅ Exception caught: {e}")
    
    try:
        # Test with invalid API key
        print("\n🧪 Testing with invalid API key...")
        parser = VisionParse(
            provider='openai',
            model='gpt-4o',
            api_key='invalid-key'
        )
        
        results = parser.analyze('sample_screenshot.png')
        
        if not results['success']:
            print(f"✅ API key error handled: {results.get('error')}")
            
    except Exception as e:
        print(f"✅ API error caught: {e}")


if __name__ == "__main__":
    print("VisionParse OpenAI/GPT Provider Examples\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--models":
            openai_models_showcase()
        elif command == "--advanced":
            openai_advanced_features()
        elif command == "--costs":
            cost_comparison()
        elif command == "--errors":
            error_handling_example()
        else:
            print("Usage:")
            print("  python openai_example.py --models     # Compare different models")
            print("  python openai_example.py --advanced   # Advanced features demo")
            print("  python openai_example.py --costs      # Cost comparison")
            print("  python openai_example.py --errors     # Error handling demo")
    else:
        # Run main showcase
        openai_models_showcase()
        openai_advanced_features()
        cost_comparison()
    
    print("\n📚 Next steps:")
    print("- Try examples/providers/claude_example.py for Anthropic models")
    print("- Check examples/providers/ollama_example.py for free local models")
    print("- Explore examples/use_cases/ for specific applications")