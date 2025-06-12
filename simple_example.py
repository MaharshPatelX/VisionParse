#!/usr/bin/env python3
"""
Simple VisionParse Example with New Provider + Model Syntax
"""

from VisionParse import VisionParse

def main():
    """Simple example using the new provider + model syntax"""
    
    # Example 1: OpenAI GPT-4o
    parser = VisionParse(
        provider='openai',
        model='gpt-4o'
    )
    results = parser.analyze('screenshot.png')
    
    # Example 2: Anthropic Claude-3.5-Sonnet  
    parser = VisionParse(
        provider='anthropic',
        model='claude-3-5-sonnet'
    )
    results = parser.analyze('screenshot.png')
    
    # Example 3: Google Gemini-2.0-Flash-Exp
    parser = VisionParse(
        provider='google', 
        model='gemini-2.0-flash-exp'
    )
    results = parser.analyze('screenshot.png')
    
    # Example 4: Ollama (Local)
    parser = VisionParse(
        provider='ollama',
        model='llava:latest'
    )
    results = parser.analyze('screenshot.png')
    
    # Example 5: Any custom provider/model
    parser = VisionParse(
        provider='custom-provider',
        model='custom-model-name'
    )
    results = parser.analyze('screenshot.png')
    
    print(f"Analysis complete: {len(results['elements'])} elements found")

if __name__ == "__main__":
    main()