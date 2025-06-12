import base64
import io
import os
from PIL import Image
import requests
import json

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    if isinstance(image, str):
        # If it's a path, load the image
        image = Image.open(image)
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def analyze_with_openai(image, api_key, model="gpt-4o", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using OpenAI models - Updated for 2024 API"""
    try:
        from openai import OpenAI
        
        # Initialize client with latest API standards
        client = OpenAI(
            api_key=api_key,
            timeout=30.0,  # Add timeout for better error handling
        )
        
        # Encode image
        base64_image = encode_image_to_base64(image)
        
        # Use the model name directly - no restrictions
        actual_model = model
        
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "auto"  # Auto-select detail level
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,  # Increased for better descriptions
            temperature=0.1,  # Lower temperature for consistent results
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error with OpenAI {model}: {str(e)}"

def analyze_with_claude(image, api_key, model="claude-3-5-sonnet-20241022", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Anthropic Claude - Updated for 2024 models"""
    try:
        import anthropic
        
        # Initialize client with latest API standards
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=30.0,
        )
        
        # Encode image
        base64_image = encode_image_to_base64(image)
        
        # Use the model name directly - no restrictions
        actual_model = model
        
        response = client.messages.create(
            model=actual_model,
            max_tokens=300,  # Increased for better descriptions
            temperature=0.1,  # Lower temperature for consistent results
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text.strip()
    
    except Exception as e:
        return f"Error with Claude {model}: {str(e)}"

def analyze_with_gemini(image, api_key, model="gemini-2.0-flash-exp", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Google Gemini - Updated for 2024/2025 models"""
    try:
        import google.generativeai as genai
        from google.generativeai import GenerationConfig
        
        # Configure API key
        genai.configure(api_key=api_key)
        
        # Use the model name directly - no restrictions
        actual_model = model
        
        # Initialize model with generation config
        gen_model = genai.GenerativeModel(
            model_name=actual_model,
            generation_config=GenerationConfig(
                temperature=0.1,  # Lower temperature for consistent results
                top_p=0.95,
                top_k=40,
                max_output_tokens=300,  # Increased for better descriptions
            )
        )
        
        # Convert PIL to format Gemini expects
        if isinstance(image, str):
            image = Image.open(image)
        
        # Generate content with multimodal prompt
        response = gen_model.generate_content([prompt, image])
        
        # Handle response with safety checks
        if response.parts:
            return response.text.strip()
        else:
            # Check if blocked for safety reasons
            if hasattr(response, 'prompt_feedback'):
                return f"Response blocked due to safety filters: {response.prompt_feedback}"
            return "No response generated"
    
    except Exception as e:
        return f"Error with Gemini {model}: {str(e)}"

def analyze_with_ollama(image, model="llava:latest", base_url="http://localhost:11434", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Ollama local models - 2024 implementation"""
    try:
        import requests
        import json
        
        # Convert PIL image to base64
        base64_image = encode_image_to_base64(image)
        
        # Prepare the request payload for Ollama API
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 300  # Max tokens
            }
        }
        
        # Make request to Ollama API
        response = requests.post(
            f"{base_url}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60  # Longer timeout for local models
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return f"Ollama API error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on http://localhost:11434"
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out. The model might be too large or slow."
    except Exception as e:
        return f"Error with Ollama {model}: {str(e)}"

def get_available_ollama_models(base_url="http://localhost:11434"):
    """Get list of available Ollama models"""
    try:
        import requests
        
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            # Filter for vision-capable models (common naming patterns)
            vision_models = [m for m in models if any(keyword in m.lower() for keyword in 
                           ["llava", "vision", "minicpm", "moondream", "cogvlm", "qwen-vl"])]
            return vision_models if vision_models else models[:5]  # Return first 5 if no vision models found
        else:
            return ["llava:latest", "llava:7b", "minicpm-v:latest"]  # Default fallback
    except:
        return ["llava:latest", "llava:7b", "minicpm-v:latest"]  # Default fallback

def get_vlm_analysis(image, vlm_type, api_key, model=None, prompt="Describe this UI element in detail. What is its function?"):
    """
    Get analysis from specified VLM - supports any VLM type and model
    
    Args:
        image: PIL Image or path to image
        vlm_type: Any VLM type name
        api_key: API key for the service (if needed)
        model: Specific model name (optional)
        prompt: Custom prompt for analysis
    
    Returns:
        String description of the UI element
    """
    vlm_type = vlm_type.lower()
    
    # Route to appropriate function based on VLM type patterns
    if any(keyword in vlm_type for keyword in ['gpt', 'openai']):
        model = model or "gpt-4o"  # Default model
        return analyze_with_openai(image, api_key, model, prompt)
    elif any(keyword in vlm_type for keyword in ['claude', 'anthropic']):
        model = model or "claude-3-5-sonnet-20241022"  # Default model
        return analyze_with_claude(image, api_key, model, prompt)
    elif any(keyword in vlm_type for keyword in ['gemini', 'google']):
        model = model or "gemini-1.5-flash"  # Default model
        return analyze_with_gemini(image, api_key, model, prompt)
    elif any(keyword in vlm_type for keyword in ['ollama', 'local']):
        model = model or "llava:latest"  # Default model
        return analyze_with_ollama(image, model, prompt=prompt)
    else:
        # For unknown VLM types, try to determine the provider
        # Check if it looks like an OpenAI model name
        if any(pattern in vlm_type for pattern in ['gpt', 'davinci', 'curie', 'babbage', 'ada']):
            return analyze_with_openai(image, api_key, model or vlm_type, prompt)
        # Check if it looks like a Claude model name
        elif any(pattern in vlm_type for pattern in ['claude', 'sonnet', 'haiku', 'opus']):
            return analyze_with_claude(image, api_key, model or vlm_type, prompt)
        # Check if it looks like a Gemini model name
        elif any(pattern in vlm_type for pattern in ['gemini', 'palm', 'bison']):
            return analyze_with_gemini(image, api_key, model or vlm_type, prompt)
        # Check if it looks like a local/Ollama model name
        elif any(pattern in vlm_type for pattern in ['llava', 'mistral', 'llama', 'vicuna', 'alpaca']):
            return analyze_with_ollama(image, model or vlm_type, prompt=prompt)
        else:
            # Default to OpenAI if cannot determine - pass the vlm_type as model name
            print(f"⚠️  Unknown VLM type '{vlm_type}', defaulting to OpenAI API")
            return analyze_with_openai(image, api_key, model or vlm_type, prompt)

def parse_vlm_response(response_text):
    """Parse VLM response to extract structured information"""
    try:
        # Try to extract structured data from response
        import re
        
        # Initialize default values
        parsed = {
            'name': 'Unknown',
            'type': 'Icon',
            'use': 'Unknown function',
            'description': response_text.strip()
        }
        
        # Extract name - prioritize structured format
        name_patterns = [
            r'Name:\s*([^\n\r]+)',  # After "Name:" label (structured response)
            r'(?:called|named)\s+"?([^".\n]+)"?',  # "called" or "named"
            r'"([^"]+)"',  # Text in quotes
            r'launch(?:es)?\s+(?:the\s+)?([A-Z][^,.\n]+?)(?:\s+(?:application|app))',  # After "launch"
            r'(?:this\s+is\s+)?([A-Z][A-Za-z\s]+?)(?:\s+(?:icon|button|application))',  # Simple pattern
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                name_candidate = match.group(1).strip()
                # Clean up common prefixes/suffixes
                name_candidate = re.sub(r'^(?:the\s+|a\s+|an\s+)', '', name_candidate, flags=re.IGNORECASE)
                name_candidate = re.sub(r'\s+(?:icon|button|app|application)$', '', name_candidate, flags=re.IGNORECASE)
                # Remove brackets if present
                name_candidate = re.sub(r'[\[\]]', '', name_candidate)
                if len(name_candidate) > 1 and len(name_candidate) < 50:
                    parsed['name'] = name_candidate
                    break
        
        # Extract type - prioritize structured format
        type_patterns = [
            r'Type:\s*([^\n\r]+)',  # After "Type:" label (structured response)
            r'\b(icon|button|text field|input field|dropdown|menu|checkbox|radio button|slider|toggle|switch|tab|link|image|label)\b',
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                type_text = match.group(1).strip()
                # Remove brackets if present
                type_text = re.sub(r'[\[\]]', '', type_text)
                parsed['type'] = type_text.title()
                break
        
        # Extract use/function - prioritize structured format
        use_patterns = [
            r'Use:\s*([^\n\r]+)',  # After "Use:" label (structured response)
            r'(?:used to|to|for)\s+([^.]+)',
            r'(?:allows|enables)\s+(?:you to\s+)?([^.]+)',
            r'(?:function is to|purpose is to)\s+([^.]+)',
        ]
        
        for pattern in use_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                use_text = match.group(1).strip()
                # Remove brackets if present
                use_text = re.sub(r'[\[\]]', '', use_text)
                if len(use_text) > 5:  # Avoid very short matches
                    parsed['use'] = use_text
                    break
        
        # Extract description - look for structured response first
        desc_patterns = [
            r'Description:\s*([^\n\r]+)',  # After "Description:" label (structured response)
            r'(?:Brief description|description):\s*([^\n.]+)',  # Alternative format
        ]
        
        description_found = False
        for pattern in desc_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                desc_text = match.group(1).strip()
                # Remove brackets if present
                desc_text = re.sub(r'[\[\]]', '', desc_text)
                if len(desc_text) > 5:  # Avoid very short matches
                    parsed['description'] = desc_text[:100] + ("..." if len(desc_text) > 100 else "")
                    description_found = True
                    break
        
        # Fallback: use first meaningful sentence
        if not description_found:
            sentences = response_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and not sentence.isdigit():  # Avoid numbers like "1"
                    parsed['description'] = sentence[:100] + ("..." if len(sentence) > 100 else "")
                    break
            else:
                # Last fallback: create description from type and name
                parsed['description'] = f"{parsed['type']} for {parsed['name']}"
        
        return parsed
        
    except Exception as e:
        # Fallback to simple parsing
        return {
            'name': 'Unknown',
            'type': 'Icon',
            'use': 'Unknown function',
            'description': response_text.strip()[:100]
        }

def batch_analyze_regions(cropped_regions, vlm_type, api_key, model=None):
    """Analyze multiple cropped regions with VLM and extract structured data"""
    results = []
    
    # Enhanced prompt to get structured information
    prompt = """Analyze this UI element and provide the following information:

Name: [Just the application/element name, e.g., "Google Chrome", "Submit Button", "Search Field"]
Type: [UI element type: Icon, Button, Text Field, Menu, Dropdown, etc.]
Use: [What it does when clicked/used]
Description: [Brief description of the element's purpose]

Be specific and concise. Provide only the requested information."""
    
    for region in cropped_regions:
        print(f"Analyzing region {region['id']}...")
        
        # Get VLM analysis
        response = get_vlm_analysis(
            region['image'], 
            vlm_type, 
            api_key,
            model, 
            prompt
        )
        
        # Parse response to extract structured data
        parsed_data = parse_vlm_response(response)
        
        # Build final result
        results.append({
            'id': region['id'],
            'bbox': region['bbox'],
            'confidence': region['confidence'],
            'name': parsed_data['name'],
            'type': parsed_data['type'],
            'use': parsed_data['use'],
            'description': parsed_data['description']
        })
    
    return results