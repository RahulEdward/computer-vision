#!/usr/bin/env python3
"""
Computer Genie Setup Script
Automatic installation and configuration for any system
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class ComputerGenieSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.setup_dir = Path(__file__).parent
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 60)
        print("🤖 COMPUTER GENIE SETUP")
        print("=" * 60)
        print(f"🖥️  System: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Setup Directory: {self.setup_dir}")
        print("=" * 60)
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        if self.python_version < (3, 8):
            print("❌ Python 3.8+ required. Current version:", sys.version)
            return False
        print("✅ Python version compatible")
        return True
        
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.setup_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True)
            
            # Install dependencies only (package structure already exists)
            print("✅ Dependencies installed successfully")
            
            print("✅ Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python dependencies: {e}")
            return False
            
    def install_tesseract_windows(self):
        """Install Tesseract on Windows"""
        print("\n🔧 Installing Tesseract OCR on Windows...")
        
        # Try winget first
        try:
            result = subprocess.run(["winget", "install", "UB-Mannheim.TesseractOCR"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Tesseract installed via winget")
                return True
        except FileNotFoundError:
            pass
            
        # Try chocolatey
        try:
            result = subprocess.run(["choco", "install", "tesseract"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Tesseract installed via chocolatey")
                return True
        except FileNotFoundError:
            pass
            
        print("⚠️  Please install Tesseract manually:")
        print("   Download: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
        
    def install_tesseract_macos(self):
        """Install Tesseract on macOS"""
        print("\n🔧 Installing Tesseract OCR on macOS...")
        try:
            subprocess.run(["brew", "install", "tesseract"], check=True, capture_output=True)
            print("✅ Tesseract installed via Homebrew")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Please install Homebrew and run: brew install tesseract")
            return False
            
    def install_tesseract_linux(self):
        """Install Tesseract on Linux"""
        print("\n🔧 Installing Tesseract OCR on Linux...")
        
        # Try apt (Ubuntu/Debian)
        try:
            subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
            subprocess.run(["sudo", "apt", "install", "-y", "tesseract-ocr"], 
                         check=True, capture_output=True)
            print("✅ Tesseract installed via apt")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        # Try yum (CentOS/RHEL)
        try:
            subprocess.run(["sudo", "yum", "install", "-y", "tesseract"], 
                         check=True, capture_output=True)
            print("✅ Tesseract installed via yum")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        print("⚠️  Please install Tesseract manually for your Linux distribution")
        return False
        
    def install_tesseract(self):
        """Install Tesseract OCR based on system"""
        if self.system == "windows":
            return self.install_tesseract_windows()
        elif self.system == "darwin":
            return self.install_tesseract_macos()
        elif self.system == "linux":
            return self.install_tesseract_linux()
        else:
            print(f"⚠️  Unsupported system: {self.system}")
            return False
            
    def verify_tesseract(self):
        """Verify Tesseract installation"""
        print("\n🔍 Verifying Tesseract installation...")
        try:
            result = subprocess.run(["tesseract", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ Tesseract verified: {version_line}")
                return True
        except FileNotFoundError:
            pass
            
        print("❌ Tesseract not found in PATH")
        return False
        
    def create_example_scripts(self):
        """Create example scripts"""
        print("\n📝 Creating example scripts...")
        
        # Quick test script
        test_script = self.setup_dir / "quick_test.py"
        test_content = '''#!/usr/bin/env python3
"""Quick test script for Computer Genie"""

import asyncio
from computer_genie.vision.screenshort import Screenshot
from computer_genie.vision import OCR

async def quick_test():
    print("🤖 Computer Genie Quick Test")
    print("=" * 40)
    
    # Test screenshot
    screenshot = Screenshot()
    image = await screenshot.capture()
    
    if image:
        print("✅ Screenshot: OK")
        print(f"📏 Size: {image.size[0]} x {image.size[1]}")
        
        # Test OCR
        import numpy as np
        image_array = np.array(image)
        ocr = OCR()
        text = await ocr.extract_text(image_array)
        
        if "not available" not in text.lower():
            print("✅ OCR: OK")
            print(f"📖 Text found: {len(text.split())} words")
        else:
            print("⚠️  OCR: Tesseract not available")
    else:
        print("❌ Screenshot: Failed")
    
    screenshot.close()
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(quick_test())
'''
        
        with open(test_script, 'w', encoding='utf-8') as f:
            f.write(test_content)
            
        print("✅ Example scripts created")
        
    def setup_environment(self):
        """Setup environment variables and paths"""
        print("\n🔧 Setting up environment...")
        
        # Add Tesseract to PATH if on Windows
        if self.system == "windows":
            tesseract_paths = [
                "C:\\Program Files\\Tesseract-OCR",
                "C:\\Program Files (x86)\\Tesseract-OCR"
            ]
            
            for path in tesseract_paths:
                if os.path.exists(path):
                    current_path = os.environ.get("PATH", "")
                    if path not in current_path:
                        print(f"📍 Adding to PATH: {path}")
                        # Note: This only affects current session
                        os.environ["PATH"] = f"{current_path};{path}"
                    break
                    
        print("✅ Environment setup complete")
        
    def run_tests(self):
        """Run basic tests"""
        print("\n🧪 Running basic tests...")
        
        try:
            # Test CLI
            result = subprocess.run([sys.executable, "-m", "computer_genie.cli", "--version"], 
                                  capture_output=True, text=True, cwd=str(self.setup_dir))
            if result.returncode == 0:
                print("✅ CLI test passed")
            else:
                print("⚠️  CLI test failed")
                
            # Test quick script
            quick_test_script = self.setup_dir / "quick_test.py"
            if quick_test_script.exists():
                print("🔍 Running quick test...")
                result = subprocess.run([sys.executable, str(quick_test_script)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Quick test passed")
                    print(result.stdout)
                else:
                    print("⚠️  Quick test had issues")
                    print(result.stderr)
                    
        except Exception as e:
            print(f"⚠️  Test error: {e}")
            
    def print_success_message(self):
        """Print success message with usage instructions"""
        print("\n" + "=" * 60)
        print("🎉 COMPUTER GENIE SETUP COMPLETE!")
        print("=" * 60)
        print("\n📋 USAGE:")
        print("1. CLI Commands:")
        print("   genie --help")
        print("   genie vision 'take a screenshot'")
        print("   genie interactive")
        print("\n2. Python Scripts:")
        print("   python quick_test.py")
        print("   python view_screenshot.py")
        print("\n3. Documentation:")
        print("   docs/tutorials/quick_start.md")
        print("   docs/tutorials/cli_usage.md")
        print("\n🚀 Happy automating!")
        print("=" * 60)
        
    def run_setup(self):
        """Run complete setup process"""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_python_version():
            return False
            
        # Install dependencies
        if not self.install_python_dependencies():
            return False
            
        # Install Tesseract
        tesseract_ok = self.install_tesseract()
        
        # Setup environment
        self.setup_environment()
        
        # Verify Tesseract
        if tesseract_ok:
            self.verify_tesseract()
            
        # Create examples
        self.create_example_scripts()
        
        # Run tests
        self.run_tests()
        
        # Success message
        self.print_success_message()
        
        return True

def main():
    """Main setup function"""
    setup = ComputerGenieSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup completed with some issues")
            print("Please check the messages above and install missing components manually")
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()