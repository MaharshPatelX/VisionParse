#!/usr/bin/env python3

import os
import json
from src.yolo_detector import detect_ui_elements, draw_boxes_on_image, crop_image_regions
from src.vlm_clients import batch_analyze_regions

def load_config():
    """Load configuration from config.json"""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print("❌ config.json not found!")
        print("Creating sample config.json file...")
        
        sample_config = {
            "vlm_type": "gpt4o",
            "api_keys": {
                "openai": "your_openai_api_key_here",
                "anthropic": "your_anthropic_api_key_here", 
                "google": "your_google_api_key_here"
            },
            "yolo_model_path": "../weights/icon_detect/model.pt",
            "confidence_threshold": 0.3,
            "available_models": {
                "openai": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4-vision-preview"],
                "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                "google": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
                "ollama": ["llava:latest", "llava:7b", "minicpm-v:latest"]
            },
            "note": "You can use any model name, not just the ones listed above"
        }
        
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print("✅ Sample config.json created. Please add your API keys and run again.")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_api_key(config, vlm_type):
    """Get API key for specified VLM type - supports any VLM type"""
    vlm_mapping = {
        'gpt4o': 'openai',
        'claude': 'anthropic', 
        'gemini': 'google',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'google': 'google'
    }
    
    # First try direct mapping
    key_name = vlm_mapping.get(vlm_type.lower())
    if key_name:
        return config['api_keys'].get(key_name)
    
    # For unknown VLM types, try pattern matching
    vlm_lower = vlm_type.lower()
    if 'gpt' in vlm_lower or 'openai' in vlm_lower:
        return config['api_keys'].get('openai')
    elif 'claude' in vlm_lower or 'anthropic' in vlm_lower:
        return config['api_keys'].get('anthropic')
    elif 'gemini' in vlm_lower or 'google' in vlm_lower:
        return config['api_keys'].get('google')
    
    # If no pattern matches, try to find the key in config
    if vlm_type in config.get('api_keys', {}):
        return config['api_keys'][vlm_type]
    
    return None

def main():
    """Main function to run VisionParse"""
    print("🚀 VisionParse - Simple Screenshot Analyzer")
    print("📌 Supports any VLM provider and model name")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    # Get screenshot path from user
    screenshot_path = input("📸 Enter screenshot path: ").strip()
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    # Get VLM choice
    print("\n🤖 Choose VLM:")
    print("1. OpenAI (GPT models)")
    print("2. Claude (Anthropic)")  
    print("3. Gemini (Google)")
    print("4. Ollama (Local models)")
    print("Or enter any VLM type name directly")
    
    choice = input("Enter choice (1-4) or VLM name: ").strip().lower()
    
    # Basic mapping for numbered choices, but accept any VLM name
    if choice == '1':
        vlm_type = 'gpt4o'
    elif choice == '2':
        vlm_type = 'claude'
    elif choice == '3':
        vlm_type = 'gemini'
    elif choice == '4':
        vlm_type = 'ollama'
    elif choice in ['gpt4o', 'openai']:
        vlm_type = 'gpt4o'
    elif choice in ['claude', 'anthropic']:
        vlm_type = 'claude'
    elif choice in ['gemini', 'google']:
        vlm_type = 'gemini'
    elif choice in ['ollama', 'local']:
        vlm_type = 'ollama'
    else:
        # Accept any VLM type name
        vlm_type = choice if choice else config.get('vlm_type', 'gpt4o')
    
    # Get model choice - accept any model name
    model_name = None
    
    # Show available models if configured, but always allow custom input
    available_models = []
    if vlm_type in ['gpt4o', 'openai']:
        available_models = config.get('available_models', {}).get('openai', ['gpt-4o'])
        print(f"\n📋 Suggested OpenAI models:")
    elif vlm_type in ['claude', 'anthropic']:
        available_models = config.get('available_models', {}).get('anthropic', ['claude-3-5-sonnet-20241022'])
        print(f"\n📋 Suggested Claude models:")
    elif vlm_type in ['gemini', 'google']:
        available_models = config.get('available_models', {}).get('google', ['gemini-1.5-flash'])
        print(f"\n📋 Suggested Gemini models:")
    elif vlm_type == 'ollama':
        # Try to get available models from Ollama
        try:
            from vlm_clients import get_available_ollama_models
            available_models = get_available_ollama_models(
                config.get('ollama_config', {}).get('base_url', 'http://localhost:11434')
            )
            print(f"\n📋 Available Ollama models:")
        except Exception as e:
            print(f"⚠️  Could not detect Ollama models: {e}")
            available_models = config.get('available_models', {}).get('ollama', ['llava:latest'])
            print(f"\n📋 Default Ollama models:")
    else:
        # For unknown VLM types, check if there are configured models
        available_models = config.get('available_models', {}).get(vlm_type, [])
        if available_models:
            print(f"\n📋 Available {vlm_type} models:")
        else:
            print(f"\n📋 Enter model name for {vlm_type}:")
    
    # Display available models if any
    if available_models:
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        model_choice = input(f"Choose model (1-{len(available_models)}) or enter any model name: ").strip()
        
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
            model_name = available_models[int(model_choice) - 1]
        else:
            model_name = model_choice
    else:
        model_choice = input("Enter model name: ").strip()
        model_name = model_choice
    
    # Set defaults if no model specified
    if not model_name:
        default_models = {
            'gpt4o': 'gpt-4o',
            'openai': 'gpt-4o', 
            'claude': 'claude-3-5-sonnet-20241022',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'gemini': 'gemini-1.5-flash',
            'google': 'gemini-1.5-flash',
            'ollama': 'llava:latest'
        }
        model_name = default_models.get(vlm_type) or config.get('model_name')
    
    # Get API key (skip for Ollama and other local models)
    api_key = None
    if vlm_type != 'ollama' and 'local' not in vlm_type.lower():
        api_key = get_api_key(config, vlm_type)
        if not api_key or any(placeholder in api_key for placeholder in ['your_', '_key_here']):
            print(f"❌ Please add your {vlm_type.upper()} API key to config.json")
            print(f"💡 For local models, use 'ollama' or include 'local' in the VLM type name")
            return
    
    print(f"✅ Using {vlm_type.upper()}")
    if model_name:
        print(f"✅ Model: {model_name}")
    print(f"✅ Screenshot: {screenshot_path}")
    
    try:
        # Step 1: YOLO Detection
        print("\n🎯 Step 1: Detecting UI elements with YOLO...")
        detected_boxes = detect_ui_elements(
            screenshot_path, 
            config['yolo_model_path'], 
            confidence_threshold=config.get('yolo_config', {}).get('box_threshold', config['confidence_threshold']),
            iou_threshold=config.get('yolo_config', {}).get('iou_threshold', 0.1),
            text_scale=0.4
        )
        
        if not detected_boxes:
            print("❌ No UI elements detected!")
            return
        
        print(f"✅ Found {len(detected_boxes)} UI elements")
        
        # Step 2: Crop regions
        print("\n✂️  Step 2: Cropping regions for analysis...")
        cropped_regions = crop_image_regions(screenshot_path, detected_boxes)
        
        # Step 3: VLM Analysis
        print(f"\n🧠 Step 3: Analyzing with {vlm_type.upper()}...")
        results = batch_analyze_regions(cropped_regions, vlm_type, api_key, model_name)
        
        # Step 4: Display Results
        print("\n📋 RESULTS:")
        print("=" * 60)
        
        for result in results:
            bbox = result['bbox']
            print(f"Element {result['id']}:")
            print(f"  📍 Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
            print(f"  🎯 Confidence: {result['confidence']:.2f}")
            print(f"  🏷️  Name: {result['name']}")
            print(f"  🔧 Type: {result['type']}")
            print(f"  💡 Use: {result['use']}")
            print(f"  📝 Description: {result['description']}")
            print("-" * 50)
        
        # Step 5: Save visualized image
        output_path = screenshot_path.replace('.', '_analyzed.')
        print(f"\n💾 Saving analyzed image to: {output_path}")
        draw_boxes_on_image(screenshot_path, detected_boxes, output_path)
        
        # Save JSON results
        json_output = screenshot_path.replace('.', '_results.').split('.')[0] + '.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(json_output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Results saved to: {json_output}")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()