import base64
import io
import os
from PIL import Image
import re
from typing import Dict, List, Optional, Any

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
    """Analyze image using OpenAI models via LlamaIndex"""
    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.schema import ImageDocument
        from llama_index.core.multi_modal_llms import MultiModalLLM
        from llama_index.multi_modal_llms.openai import OpenAIMultiModal
        
        # Initialize OpenAI multimodal LLM
        llm = OpenAIMultiModal(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=300
        )
        
        # Convert PIL image to ImageDocument
        if isinstance(image, str):
            image = Image.open(image)
        
        # Create image document
        image_doc = ImageDocument(image=image)
        
        # Generate response
        response = llm.complete(
            prompt=prompt,
            image_documents=[image_doc]
        )
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error with OpenAI {model}: {str(e)}"

def analyze_with_claude(image, api_key, model="claude-3-5-sonnet-20241022", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Anthropic Claude via LlamaIndex"""
    try:
        from llama_index.llms.anthropic import Anthropic
        from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
        from llama_index.core.schema import ImageDocument
        
        # Initialize Anthropic multimodal LLM
        llm = AnthropicMultiModal(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=300
        )
        
        # Convert PIL image to ImageDocument
        if isinstance(image, str):
            image = Image.open(image)
        
        # Create image document
        image_doc = ImageDocument(image=image)
        
        # Generate response
        response = llm.complete(
            prompt=prompt,
            image_documents=[image_doc]
        )
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error with Claude {model}: {str(e)}"

def analyze_with_gemini(image, api_key, model="gemini-2.0-flash-exp", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Google Gemini via LlamaIndex"""
    try:
        from llama_index.llms.gemini import Gemini
        from llama_index.multi_modal_llms.gemini import GeminiMultiModal
        from llama_index.core.schema import ImageDocument
        
        # Initialize Gemini multimodal LLM
        llm = GeminiMultiModal(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=300
        )
        
        # Convert PIL image to ImageDocument
        if isinstance(image, str):
            image = Image.open(image)
        
        # Create image document
        image_doc = ImageDocument(image=image)
        
        # Generate response
        response = llm.complete(
            prompt=prompt,
            image_documents=[image_doc]
        )
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error with Gemini {model}: {str(e)}"

def analyze_with_ollama(image, model="llava:latest", base_url="http://localhost:11434", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Ollama local models via LlamaIndex"""
    try:
        from llama_index.llms.ollama import Ollama
        from llama_index.multi_modal_llms.ollama import OllamaMultiModal
        from llama_index.core.schema import ImageDocument
        
        # Initialize Ollama multimodal LLM
        llm = OllamaMultiModal(
            model=model,
            base_url=base_url,
            temperature=0.1,
            context_window=4096,
            request_timeout=60.0
        )
        
        # Convert PIL image to ImageDocument
        if isinstance(image, str):
            image = Image.open(image)
        
        # Create image document
        image_doc = ImageDocument(image=image)
        
        # Generate response
        response = llm.complete(
            prompt=prompt,
            image_documents=[image_doc]
        )
        
        return response.text.strip()
    
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
    Get analysis from specified VLM using LlamaIndex - supports any VLM type and model
    
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
        model = model or "gemini-2.0-flash-exp"  # Default model
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
    """Analyze multiple cropped regions with VLM using LlamaIndex and extract structured data"""
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
        
        # Get VLM analysis using LlamaIndex
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