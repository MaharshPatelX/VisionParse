import base64
import io
import os
import logging
from PIL import Image
import re
from typing import Dict, List, Optional, Any

# Disable LlamaIndex HTTP request logging
logging.getLogger("llama_index.core.base.llms.base").setLevel(logging.WARNING)
logging.getLogger("llama_index.llms.google_genai").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Supported models for each provider
SUPPORTED_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "anthropic": ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-0"],
    "google": ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "ollama": ["gemma3:4b", "qwen2.5vl:3b"]
}

def validate_model(model, provider):
    """Validate if model is supported for the given provider"""
    provider = provider.lower()
    
    # Check if provider exists in supported models
    if provider not in SUPPORTED_MODELS:
        supported_providers = list(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported_providers}")
    
    # Check if model is supported for this provider
    if model not in SUPPORTED_MODELS[provider]:
        supported_models = SUPPORTED_MODELS[provider]
        raise ValueError(f"Unsupported model '{model}' for provider '{provider}'. Supported models: {supported_models}")
    
    return True

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
        # Validate model
        validate_model(model, "openai")
        from llama_index.multi_modal_llms.openai import OpenAIMultiModal
        from llama_index.core.schema import ImageDocument
        import io
        import base64
        
        # Initialize OpenAI multimodal LLM
        llm = OpenAIMultiModal(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=300
        )
        
        # Convert PIL image to bytes for ImageDocument
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Create image document with bytes
        image_doc = ImageDocument(image=img_bytes)
        
        # Generate response
        response = llm.complete(
            prompt=prompt,
            image_documents=[image_doc]
        )
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error with OpenAI {model}: {str(e)}"

def analyze_with_claude(image, api_key, model="claude-sonnet-4-0", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Anthropic Claude via LlamaIndex with Claude 4 workaround"""
    try:
        # Validate model
        validate_model(model, "anthropic")
        from llama_index.llms.anthropic import Anthropic
        from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
        import tempfile
        import os
        
        # Initialize Anthropic LLM with Claude 4 support
        llm = Anthropic(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=300
        )
        
        # Handle image input - save to temporary file if PIL Image
        if isinstance(image, str):
            # Already a file path
            image_path = image
        else:
            # PIL Image - save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()  # Close file handle to prevent Windows file lock issues
            image.save(temp_file.name, format='PNG')
            image_path = temp_file.name
        
        try:
            # Create message with image and text using new Claude 4 compatible format
            messages = [
                ChatMessage(
                    role="user",
                    blocks=[
                        ImageBlock(path=image_path),
                        TextBlock(text=prompt),
                    ],
                )
            ]
            
            # Get response
            response = llm.chat(messages)
            return response.message.content.strip()
            
        finally:
            # Clean up temporary file if we created one
            if not isinstance(image, str) and os.path.exists(image_path):
                os.unlink(image_path)
    
    except Exception as e:
        return f"Error with Claude {model}: {str(e)}"

def analyze_with_gemini(image, api_key, model="gemini-1.5-pro", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Google Gemini via LlamaIndex"""
    try:
        # Validate model
        validate_model(model, "google")
        from llama_index.llms.google_genai import GoogleGenAI
        from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
        import tempfile
        import os
        
        # Initialize Google Gemini LLM
        llm = GoogleGenAI(
            model=model,
            api_key=api_key
        )
        
        # Handle image input - save to temporary file if PIL Image
        if isinstance(image, str):
            # Already a file path
            image_path = image
        else:
            # PIL Image - save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()  # Close file handle to prevent Windows file lock issues
            image.save(temp_file.name, format='PNG')
            image_path = temp_file.name
        
        try:
            # Create message with image and text
            messages = [
                ChatMessage(
                    role="user",
                    blocks=[
                        TextBlock(text=prompt),
                        ImageBlock(path=image_path),
                    ],
                )
            ]
            
            # Get response
            response = llm.chat(messages)
            return response.message.content.strip()
            
        finally:
            # Clean up temporary file if we created one
            if not isinstance(image, str) and os.path.exists(image_path):
                os.unlink(image_path)
    
    except Exception as e:
        return f"Error with Gemini {model}: {str(e)}"

def analyze_with_ollama(image, model="qwen2.5vl:3b", base_url="http://localhost:11434", prompt="Describe this UI element in detail. What is its function?"):
    """Analyze image using Ollama local models via LlamaIndex"""
    try:
        # Validate model
        validate_model(model, "ollama")
        from llama_index.llms.ollama import Ollama
        from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
        import tempfile
        import os
        
        # Initialize Ollama LLM
        llm = Ollama(
            model=model,
            request_timeout=120
        )
        
        # Handle image input - save to temporary file if PIL Image
        if isinstance(image, str):
            # Already a file path
            image_path = image
        else:
            # PIL Image - save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()  # Close file handle to prevent Windows file lock issues
            image.save(temp_file.name, format='PNG')
            image_path = temp_file.name
        
        try:
            # Create message with image and text
            messages = [
                ChatMessage(
                    role="user",
                    blocks=[
                        TextBlock(text=prompt),
                        ImageBlock(path=image_path),
                    ],
                )
            ]
            
            # Get response
            response = llm.chat(messages)
            return response.message.content.strip()
            
        finally:
            # Clean up temporary file if we created one
            if not isinstance(image, str) and os.path.exists(image_path):
                os.unlink(image_path)
    
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
        model = model or "claude-3-5-sonnet-latest"  # Default model
        return analyze_with_claude(image, api_key, model, prompt)
    elif any(keyword in vlm_type for keyword in ['gemini', 'google']):
        model = model or "gemini-1.5-pro"  # Default model
        return analyze_with_gemini(image, api_key, model, prompt)
    elif any(keyword in vlm_type for keyword in ['ollama', 'local']):
        model = model or "qwen2.5vl:3b"  # Default model
        return analyze_with_ollama(image, model, prompt=prompt)
    else:
        # For unknown VLM types, reject with supported options
        supported_providers = list(SUPPORTED_MODELS.keys())
        return f"Error: Unsupported VLM type '{vlm_type}'. Supported providers: {supported_providers}. Use format: 'provider' with supported model."

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
    from tqdm import tqdm
    
    results = []
    
    # Enhanced prompt to get structured information
    prompt = """Analyze this UI element and provide the following information:

Name: [Just the application/element name, e.g., "Google Chrome", "Submit Button", "Search Field"]
Type: [UI element type: Icon, Button, Text Field, Menu, Dropdown, etc.]
Use: [What it does when clicked/used]
Description: [Brief description of the element's purpose]

Be specific and concise. Provide only the requested information."""
    
    # Use tqdm progress bar for analyzing regions
    for region in tqdm(cropped_regions, desc="Analyzing UI elements", unit="element"):
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