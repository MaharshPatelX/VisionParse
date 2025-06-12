#!/usr/bin/env python3
"""
VisionParse Setup Script: Ollama Installation and Configuration

This script helps users set up Ollama for local VLM analysis without API costs.
Automatically downloads and configures vision models for VisionParse.
"""

import os
import sys
import subprocess
import platform
import time
import requests
from pathlib import Path


class OllamaSetup:
    """Automated Ollama setup for VisionParse"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.base_url = "http://localhost:11434"
        self.recommended_models = [
            {
                'name': 'llava:latest',
                'size': '4.5GB',
                'description': 'Best overall vision model',
                'recommended': True
            },
            {
                'name': 'minicpm-v:latest', 
                'size': '2.8GB',
                'description': 'Fast and efficient model',
                'recommended': True
            },
            {
                'name': 'moondream:latest',
                'size': '1.7GB', 
                'description': 'Lightweight model for testing',
                'recommended': False
            },
            {
                'name': 'cogvlm:latest',
                'size': '14GB',
                'description': 'Advanced vision capabilities (large)',
                'recommended': False
            }
        ]
    
    def print_header(self):
        """Print setup header"""
        print("🏠 VisionParse Ollama Setup")
        print("=" * 35)
        print("This script will help you set up Ollama for free, local VLM analysis.")
        print("No API keys required - everything runs on your machine!\n")
    
    def check_ollama_installed(self):
        """Check if Ollama is already installed"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Ollama is already installed: {version}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("❌ Ollama not found on system")
        return False
    
    def install_ollama(self):
        """Install Ollama based on the operating system"""
        print("📦 Installing Ollama...")
        
        if self.system == "linux" or self.system == "darwin":  # macOS
            print("🐧 Detected Linux/macOS - using official installer")
            try:
                # Download and run official installer
                cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
                print(f"Running: {cmd}")
                
                result = subprocess.run(cmd, shell=True, check=True)
                print("✅ Ollama installed successfully!")
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print("Please install manually from https://ollama.ai")
                return False
                
        elif self.system == "windows":
            print("🪟 Detected Windows")
            print("Please download and install Ollama manually:")
            print("1. Go to https://ollama.ai")
            print("2. Download the Windows installer")
            print("3. Run the installer")
            print("4. Restart your terminal")
            print("5. Run this script again")
            return False
        
        else:
            print(f"❓ Unsupported system: {self.system}")
            print("Please install Ollama manually from https://ollama.ai")
            return False
    
    def check_ollama_running(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service is running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("❌ Ollama service is not running")
        return False
    
    def start_ollama_service(self):
        """Start Ollama service"""
        print("🚀 Starting Ollama service...")
        
        try:
            if self.system == "windows":
                # On Windows, Ollama usually starts automatically
                print("On Windows, Ollama should start automatically after installation.")
                print("If not, please start it manually from the Start menu.")
            else:
                # On Linux/macOS, start the service
                subprocess.Popen(['ollama', 'serve'], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            print("⏳ Waiting for service to start...")
            for i in range(10):
                if self.check_ollama_running():
                    return True
                time.sleep(2)
                print("   Still waiting...")
            
            print("❌ Failed to start Ollama service")
            print("Please try starting it manually: ollama serve")
            return False
            
        except Exception as e:
            print(f"❌ Error starting service: {e}")
            return False
    
    def list_available_models(self):
        """List currently installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                if models:
                    print("📋 Currently installed models:")
                    for model in models:
                        name = model.get('name', 'Unknown')
                        size = model.get('size', 0)
                        size_gb = size / (1024**3) if size > 0 else 0
                        print(f"   • {name} ({size_gb:.1f}GB)")
                    return [model['name'] for model in models]
                else:
                    print("📋 No models currently installed")
                    return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def pull_model(self, model_name):
        """Download a specific model"""
        print(f"📥 Downloading {model_name}...")
        print("This may take several minutes depending on model size and internet speed.")
        
        try:
            # Use ollama pull command
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Show progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    if "pulling" in line.lower():
                        print(f"   📦 {line}")
                    elif "success" in line.lower():
                        print(f"   ✅ {line}")
                    elif "%" in line:
                        print(f"   ⏳ {line}")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ Successfully downloaded {model_name}")
                return True
            else:
                print(f"❌ Failed to download {model_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            return False
    
    def test_model(self, model_name):
        """Test a model with a simple query"""
        print(f"🧪 Testing {model_name}...")
        
        try:
            # Simple test query
            payload = {
                "model": model_name,
                "prompt": "Describe what you see in this image: a red button.",
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                if response_text:
                    print(f"✅ {model_name} is working!")
                    print(f"   Sample response: {response_text[:100]}...")
                    return True
                else:
                    print(f"⚠️  {model_name} responded but with empty content")
                    return False
            else:
                print(f"❌ {model_name} test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return False
    
    def interactive_model_selection(self):
        """Interactive model selection and installation"""
        print("\n🎯 Model Selection")
        print("Choose which vision models to install:")
        print()
        
        for i, model in enumerate(self.recommended_models, 1):
            status = "⭐ RECOMMENDED" if model['recommended'] else "Optional"
            print(f"{i}. {model['name']} ({model['size']}) - {status}")
            print(f"   {model['description']}")
            print()
        
        print("0. Install all recommended models")
        print("q. Skip model installation")
        print()
        
        while True:
            choice = input("Enter your choice (0-{}, q): ".format(len(self.recommended_models))).strip().lower()
            
            if choice == 'q':
                print("Skipping model installation")
                return []
            
            elif choice == '0':
                # Install all recommended
                selected = [m for m in self.recommended_models if m['recommended']]
                print(f"Installing {len(selected)} recommended models...")
                return selected
            
            elif choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.recommended_models):
                    selected_model = self.recommended_models[choice_num - 1]
                    print(f"Installing {selected_model['name']}...")
                    return [selected_model]
            
            print("Invalid choice. Please try again.")
    
    def create_visionparse_test(self):
        """Create a test script for VisionParse + Ollama"""
        test_script = '''#!/usr/bin/env python3
"""
VisionParse + Ollama Test Script
Quick test to verify everything is working together.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from VisionParse import VisionParse
    
    print("🧪 Testing VisionParse with Ollama...")
    
    # Initialize with Ollama
    parser = VisionParse(vlm_type='ollama', model='llava:latest')
    print("✅ VisionParse initialized with Ollama")
    
    # Test with a sample image (you'll need to provide one)
    # results = parser.analyze('sample_image.png')
    
    print("🎉 Setup successful! VisionParse is ready to use with Ollama")
    print()
    print("Next steps:")
    print("1. Take a screenshot and save it as 'test_screenshot.png'")
    print("2. Run: python main.py")
    print("3. Choose option 4 (Ollama)")
    print("4. Enter the screenshot path")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check your VisionParse installation")

'''
        
        script_path = Path("test_ollama_setup.py")
        with open(script_path, 'w') as f:
            f.write(test_script)
        
        # Make executable on Unix systems
        if self.system != "windows":
            os.chmod(script_path, 0o755)
        
        print(f"📝 Created test script: {script_path}")
        return script_path
    
    def run_full_setup(self):
        """Run the complete Ollama setup process"""
        self.print_header()
        
        # Step 1: Check/Install Ollama
        if not self.check_ollama_installed():
            if not self.install_ollama():
                return False
        
        # Step 2: Start service
        if not self.check_ollama_running():
            if not self.start_ollama_service():
                return False
        
        # Step 3: List existing models
        installed_models = self.list_available_models()
        
        # Step 4: Install new models
        if not installed_models or input("\nInstall additional models? (y/n): ").lower().startswith('y'):
            selected_models = self.interactive_model_selection()
            
            for model in selected_models:
                if self.pull_model(model['name']):
                    # Test the model
                    self.test_model(model['name'])
        
        # Step 5: Create test script
        print("\n🔧 Creating test configuration...")
        test_script = self.create_visionparse_test()
        
        # Step 6: Final instructions
        print("\n🎉 Ollama Setup Complete!")
        print("=" * 30)
        print("Your local VLM setup is ready! Here's what you can do:")
        print()
        print("🚀 Quick Start:")
        print("   python main.py  # Choose option 4 (Ollama)")
        print()
        print("🐍 Python API:")
        print("   from src import VisionParse")
        print("   parser = VisionParse(vlm_type='ollama', model='llava:latest')")
        print("   results = parser.analyze('screenshot.png')")
        print()
        print("⚡ CLI:")
        print("   visionparse screenshot.png --vlm ollama --model llava:latest")
        print()
        print(f"🧪 Test script: python {test_script}")
        print()
        print("💡 Benefits of local models:")
        print("   • Completely free (no API costs)")
        print("   • Private (data never leaves your machine)")
        print("   • Works offline")
        print("   • No rate limits")
        
        return True


def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        # Just check status
        setup = OllamaSetup()
        print("🔍 Checking Ollama Status")
        print("=" * 25)
        
        if setup.check_ollama_installed():
            if setup.check_ollama_running():
                models = setup.list_available_models()
                if models:
                    print(f"✅ Ollama is ready with {len(models)} models")
                else:
                    print("⚠️  Ollama is running but no models installed")
            else:
                print("⚠️  Ollama installed but not running")
        else:
            print("❌ Ollama not installed")
    
    else:
        # Run full setup
        setup = OllamaSetup()
        try:
            setup.run_full_setup()
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")


if __name__ == "__main__":
    main()