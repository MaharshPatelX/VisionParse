#!/usr/bin/env python3
"""
VisionParse Installation Validator

This script validates that VisionParse is properly installed and configured.
Checks dependencies, API keys, YOLO models, and performs test runs.
"""

import os
import sys
import importlib
import subprocess
import tempfile
from pathlib import Path
import json


class VisionParseValidator:
    """Comprehensive installation validator for VisionParse"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
        # Add project root to path
        self.project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(self.project_root))
    
    def print_header(self):
        """Print validation header"""
        print("🔍 VisionParse Installation Validator")
        print("=" * 40)
        print("This script will check if VisionParse is properly installed and configured.\n")
    
    def check_python_version(self):
        """Check Python version compatibility"""
        self.total_checks += 1
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            self.success_count += 1
        else:
            error = f"Python {version.major}.{version.minor} is not supported. Requires Python 3.8+"
            print(f"   ❌ {error}")
            self.errors.append(error)
    
    def check_dependencies(self):
        """Check required Python packages"""
        self.total_checks += 1
        print("📦 Checking Python dependencies...")
        
        required_packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('ultralytics', 'Ultralytics YOLO'),
            ('PIL', 'Pillow'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('requests', 'Requests'),
            ('dotenv', 'Python-dotenv')
        ]
        
        missing_packages = []
        for package, name in required_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {name}")
            except ImportError:
                print(f"   ❌ {name} not found")
                missing_packages.append(name)
        
        if not missing_packages:
            self.success_count += 1
        else:
            error = f"Missing packages: {', '.join(missing_packages)}"
            self.errors.append(error)
            print(f"   💡 Install with: pip install -r requirements.txt")
    
    def check_visionparse_import(self):
        """Check if VisionParse can be imported"""
        self.total_checks += 1
        print("🧩 Checking VisionParse import...")
        
        try:
            from src import VisionParse, VisionParseError
            print("   ✅ VisionParse imports successfully")
            self.success_count += 1
            return True
        except ImportError as e:
            error = f"Cannot import VisionParse: {e}"
            print(f"   ❌ {error}")
            self.errors.append(error)
            return False
    
    def check_yolo_model(self):
        """Check if YOLO model exists"""
        self.total_checks += 1
        print("🎯 Checking YOLO model...")
        
        model_paths = [
            "weights/icon_detect/model.pt",
            "../weights/icon_detect/model.pt",
            "model.pt"
        ]
        
        for model_path in model_paths:
            full_path = self.project_root / model_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ YOLO model found: {model_path} ({size_mb:.1f}MB)")
                self.success_count += 1
                return True
        
        error = "YOLO model not found in expected locations"
        print(f"   ❌ {error}")
        print("   💡 Expected locations:")
        for path in model_paths:
            print(f"      - {path}")
        self.errors.append(error)
        return False
    
    def check_api_keys(self):
        """Check API key configuration"""
        self.total_checks += 1
        print("🔑 Checking API keys...")
        
        # Check .env file
        env_file = self.project_root / ".env"
        config_file = self.project_root / "config.json"
        
        found_keys = {}
        
        # Check environment variables
        api_keys = {
            'OPENAI_API_KEY': 'OpenAI',
            'ANTHROPIC_API_KEY': 'Anthropic',
            'GOOGLE_API_KEY': 'Google'
        }
        
        for env_var, service in api_keys.items():
            if os.getenv(env_var):
                found_keys[service] = 'Environment'
                print(f"   ✅ {service} API key found in environment")
        
        # Check .env file
        if env_file.exists():
            print(f"   ✅ .env file exists")
            # Could parse .env file here if needed
        else:
            print(f"   ⚠️  .env file not found")
            self.warnings.append(".env file not found - consider creating one for API keys")
        
        # Check config.json
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                api_keys_config = config.get('api_keys', {})
                for service, key in api_keys_config.items():
                    if key and key.strip():
                        found_keys[service] = 'Config file'
                        print(f"   ✅ {service} API key found in config")
            except Exception:
                pass
        
        if found_keys:
            print(f"   📊 Found {len(found_keys)} API key(s)")
            self.success_count += 1
        else:
            warning = "No API keys found - Ollama (local) models recommended for testing"
            print(f"   ⚠️  {warning}")
            self.warnings.append(warning)
    
    def check_ollama_available(self):
        """Check if Ollama is available for local models"""
        self.total_checks += 1
        print("🏠 Checking Ollama (local models)...")
        
        try:
            # Check if ollama command exists
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   ✅ Ollama installed: {result.stdout.strip()}")
                
                # Check if service is running
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        if models:
                            print(f"   ✅ Ollama running with {len(models)} model(s)")
                        else:
                            print(f"   ⚠️  Ollama running but no models installed")
                            self.warnings.append("Ollama has no vision models - run scripts/setup_ollama.py")
                    else:
                        print(f"   ⚠️  Ollama installed but not running")
                        self.warnings.append("Ollama not running - start with 'ollama serve'")
                except requests.exceptions.RequestException:
                    print(f"   ⚠️  Ollama installed but service not accessible")
                    self.warnings.append("Ollama service not running")
                
                self.success_count += 1
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            warning = "Ollama not installed - consider installing for free local models"
            print(f"   ⚠️  {warning}")
            self.warnings.append(warning)
            return False
    
    def test_basic_initialization(self):
        """Test basic VisionParse initialization"""
        self.total_checks += 1
        print("🧪 Testing VisionParse initialization...")
        
        try:
            from src import VisionParse, VisionParseError
            
            # Test with Ollama (doesn't require API key)
            try:
                parser = VisionParse(vlm_type='ollama', model='llava:latest')
                print("   ✅ VisionParse initializes with Ollama")
                self.success_count += 1
                return True
            except VisionParseError as e:
                if "YOLO model not found" in str(e):
                    print("   ⚠️  VisionParse fails due to missing YOLO model")
                    self.warnings.append("YOLO model required for initialization")
                else:
                    print(f"   ❌ VisionParse initialization failed: {e}")
                    self.errors.append(f"Initialization error: {e}")
                return False
                
        except Exception as e:
            error = f"Unexpected error during initialization: {e}"
            print(f"   ❌ {error}")
            self.errors.append(error)
            return False
    
    def test_cli_availability(self):
        """Test if CLI commands are available"""
        self.total_checks += 1
        print("⚡ Testing CLI availability...")
        
        cli_commands = ['visionparse', 'vision-parse', 'vparse']
        
        for cmd in cli_commands:
            try:
                result = subprocess.run([cmd, '--help'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"   ✅ CLI command '{cmd}' available")
                    self.success_count += 1
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        warning = "CLI commands not installed - run 'pip install -e .' to enable"
        print(f"   ⚠️  {warning}")
        self.warnings.append(warning)
    
    def check_project_structure(self):
        """Check if project structure is correct"""
        self.total_checks += 1
        print("📁 Checking project structure...")
        
        required_paths = [
            "src/",
            "src/__init__.py",
            "src/vlm_parser.py",
            "src/yolo_detector.py", 
            "src/vlm_clients.py",
            "requirements.txt",
            "main.py",
            "cli.py"
        ]
        
        missing_paths = []
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                missing_paths.append(path)
        
        if not missing_paths:
            print("   ✅ Project structure is complete")
            self.success_count += 1
        else:
            error = f"Missing files/folders: {', '.join(missing_paths)}"
            print(f"   ❌ {error}")
            self.errors.append(error)
    
    def run_comprehensive_test(self):
        """Run a comprehensive test if possible"""
        if self.errors:
            print("⏭️  Skipping comprehensive test due to previous errors")
            return
        
        self.total_checks += 1
        print("🔬 Running comprehensive test...")
        
        try:
            from src import VisionParse
            
            # Create a simple test image
            from PIL import Image, ImageDraw
            
            # Create test image
            img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple button-like rectangle
            draw.rectangle([50, 50, 150, 100], fill='blue', outline='black')
            draw.text((75, 70), "Button", fill='white')
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                temp_image = tmp.name
            
            try:
                # Test with Ollama if available
                parser = VisionParse(vlm_type='ollama', model='llava:latest')
                results = parser.analyze(temp_image, save_annotated_image=False)
                
                if results.get('success'):
                    print("   ✅ End-to-end test successful!")
                    print(f"      Found {len(results.get('elements', []))} elements")
                    self.success_count += 1
                else:
                    print(f"   ⚠️  Test completed but analysis failed: {results.get('error')}")
                    self.warnings.append("Analysis failed - may need API key or model setup")
                    
            except Exception as e:
                print(f"   ⚠️  Test failed: {e}")
                self.warnings.append(f"Comprehensive test failed: {e}")
            
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_image)
                except:
                    pass
                    
        except Exception as e:
            print(f"   ❌ Could not run comprehensive test: {e}")
            self.errors.append(f"Test setup failed: {e}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("📊 Validation Summary")
        print("=" * 50)
        
        success_rate = (self.success_count / self.total_checks * 100) if self.total_checks > 0 else 0
        
        print(f"✅ Successful checks: {self.success_count}/{self.total_checks} ({success_rate:.1f}%)")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        print("\n💡 Next Steps:")
        
        if self.errors:
            print("   🔧 Fix the errors above before using VisionParse")
            if any("YOLO model" in error for error in self.errors):
                print("   📥 Download YOLO model weights to weights/icon_detect/model.pt")
            if any("Missing packages" in error for error in self.errors):
                print("   📦 Install dependencies: pip install -r requirements.txt")
        
        elif success_rate >= 80:
            print("   🎉 VisionParse is ready to use!")
            print("   🚀 Try: python main.py")
            print("   ⚡ Or: visionparse screenshot.png --vlm gpt4o")
            
            if not any("API key" in warning for warning in self.warnings):
                print("   🔑 Set up API keys in .env for cloud models")
            
            if any("Ollama" in warning for warning in self.warnings):
                print("   🏠 Run scripts/setup_ollama.py for free local models")
        
        else:
            print("   ⚠️  Several issues found - please address them before use")
            print("   📚 Check the installation guide: docs/installation.md")
    
    def run_all_checks(self):
        """Run all validation checks"""
        self.print_header()
        
        self.check_python_version()
        self.check_dependencies() 
        self.check_visionparse_import()
        self.check_project_structure()
        self.check_yolo_model()
        self.check_api_keys()
        self.check_ollama_available()
        self.test_basic_initialization()
        self.test_cli_availability()
        self.run_comprehensive_test()
        
        self.print_summary()
        
        # Return success status
        return len(self.errors) == 0 and self.success_count >= self.total_checks * 0.8


def main():
    """Main validation function"""
    validator = VisionParseValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick check mode
        print("🏃 Quick Validation Check")
        print("=" * 25)
        
        validator.check_python_version()
        validator.check_visionparse_import()
        validator.check_yolo_model()
        
        if validator.errors:
            print("\n❌ Quick check failed - run full validation for details")
            return False
        else:
            print("\n✅ Quick check passed!")
            return True
    
    else:
        # Full validation
        success = validator.run_all_checks()
        return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)