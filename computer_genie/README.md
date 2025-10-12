# 🤖 Computer Genie

<div align="center">

![Computer Genie Logo](https://img.shields.io/badge/Computer%20Genie-AI%20Vision%20%26%20Automation-blue?style=for-the-badge&logo=robot)

**Enterprise-Grade AI-Powered Computer Vision and Automation Platform**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square)](https://github.com/your-username/computer_genie)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange?style=flat-square)](https://pypi.org/project/computer-genie)

</div>

---

## 🏢 **Built by Abhishek Technologies Pvt Ltd**

Computer Genie is a cutting-edge Python framework that empowers AI agents to seamlessly interact with computer systems through advanced computer vision, intelligent screenshot analysis, OCR text recognition, and sophisticated automation capabilities. Designed for enterprise applications, research institutions, and developers building next-generation AI-powered automation solutions.

### 🎯 **Why Choose Computer Genie?**

**Abhishek Technologies Pvt Ltd** brings years of expertise in AI and automation to deliver a robust, scalable, and enterprise-ready solution. Our team of experienced engineers and AI specialists have crafted Computer Genie to meet the demanding requirements of modern businesses and research organizations.

**Key Advantages:**
- ⚡ **High Performance**: Optimized for speed and efficiency
- 🔒 **Enterprise Security**: Built with security best practices
- 🌐 **Cross-Platform**: Works seamlessly across Windows, macOS, and Linux
- 📈 **Scalable Architecture**: From single scripts to enterprise deployments
- 🛠️ **Professional Support**: Backed by Abhishek Technologies' expert team
- 📚 **Comprehensive Documentation**: Detailed guides and examples

## ✨ Core Features & Capabilities

### 🔍 **Vision & Screenshot**
- **High-quality screenshots** with customizable regions
- **Real-time screen capture** for monitoring and automation
- **Cross-platform support** (Windows, macOS, Linux)
- **Multiple image formats** (PNG, JPEG, etc.)

### 📖 **OCR Text Reading**
- **Tesseract OCR integration** for text extraction
- **Multi-language support** for text recognition
- **Automatic text detection** from screenshots and images
- **Software name and UI element detection**

### 🤖 **AI Agents**
- **Vision Agent**: Screenshot analysis and visual tasks
- **Web Agent**: Browser automation and web interaction
- **Android Agent**: Mobile device automation
- **Interactive Mode**: Real-time AI assistance

### 🖥️ **CLI Interface**
- **Simple commands** for quick automation
- **Interactive mode** for real-time assistance
- **Batch processing** for multiple tasks
- **Comprehensive help system**

### 🔧 **Developer Tools**
- **Python API** for programmatic access
- **Async/await support** for modern Python
- **Extensible architecture** for custom agents
- **Comprehensive logging** and error handling

## 🚀 Installation & Quick Start

### 📦 **Production Installation** (Recommended)

```bash
# Install from PyPI (Coming Soon)
pip install computer-genie

# Or install from source
git clone https://github.com/abhishek-tech/computer_genie.git
cd computer_genie
pip install .
```

### 🔧 **Development Setup**

```bash
# Clone the repository
git clone https://github.com/abhishek-tech/computer_genie.git
cd computer_genie

# Run automatic setup (Windows)
.\install.bat

# Run automatic setup (macOS/Linux)
./install.sh

# Or use Python setup
python setup.py
```

**The automated setup includes:**
- ✅ Python dependencies installation
- ✅ Tesseract OCR configuration
- ✅ Environment setup and validation
- ✅ Example scripts creation
- ✅ Comprehensive testing suite
- ✅ Performance optimization

### 🏢 **Enterprise Installation**

For enterprise deployments, contact **Abhishek Technologies Pvt Ltd** for:
- 🔐 **Custom security configurations**
- 📊 **Performance optimization**
- 🛠️ **Professional installation support**
- 📞 **24/7 technical assistance**
- 📋 **Compliance and audit support**

**Enterprise Contact**: enterprise@abhishektech.com

### 🐳 **Docker Installation**

```bash
# Build Docker image
docker build -t computer-genie .

# Run with Docker Compose
docker-compose up -d
```

### 3. **Quick Test**

```bash
# Test CLI
genie --version
genie vision "take a screenshot and describe what you see"

# Test Python API
python quick_test.py

# Interactive mode
genie interactive
```

## 📖 Usage Examples

### 🖥️ **CLI Usage**

```bash
# Take a screenshot
genie vision "take a screenshot"

# Read text from screen
genie vision "read all text on the screen"

# Web automation
genie web "navigate to google.com and search for python"

# Android automation
genie android "take a screenshot of the home screen"

# Interactive mode
genie interactive --agent vision
```

### 🐍 **Python API**

```python
import asyncio
from computer_genie.vision.screenshort import Screenshot
from computer_genie.vision import OCR

async def main():
    # Take a screenshot
    screenshot = Screenshot()
    image = await screenshot.capture()
    
    # Save screenshot
    image.save("my_screenshot.png")
    
    # Extract text using OCR
    import numpy as np
    image_array = np.array(image)
    ocr = OCR()
    text = await ocr.extract_text(image_array)
    
    print("Text found:", text)
    
    # Cleanup
    screenshot.close()

# Run the example
asyncio.run(main())
```

### 🔄 **Interactive Mode**

```bash
genie interactive --agent vision

# In interactive mode:
> help                          # Show available commands
> screenshot                    # Take a screenshot
> read_text                     # Read text from screen
> find_element "button"         # Find UI elements
> exit                          # Exit interactive mode
```

## 📁 Project Structure

```
computer_genie/
├── 📁 computer_genie/          # Main package
│   ├── 📁 cli/                 # Command-line interface
│   ├── 📁 core/                # Core functionality
│   ├── 📁 vision/              # Vision and OCR modules
│   ├── 📁 tools/               # Automation tools
│   └── 📁 utils/               # Utility functions
├── 📁 docs/                    # Documentation
│   ├── 📁 api/                 # API reference
│   ├── 📁 tutorials/           # Tutorials and guides
│   └── 📁 examples/            # Example code
├── 📁 examples/                # Example scripts
├── 📁 tests/                   # Test suite
├── 📄 setup.py                 # Automatic setup script
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md               # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set Tesseract path manually
export TESSERACT_CMD="/usr/local/bin/tesseract"

# Optional: Set default screenshot directory
export GENIE_SCREENSHOT_DIR="/path/to/screenshots"

# Optional: Enable debug logging
export GENIE_DEBUG=1
```

### Configuration File

Create `~/.computer_genie/config.yaml`:

```yaml
# Screenshot settings
screenshot:
  format: "png"
  quality: 95
  default_path: "~/screenshots"

# OCR settings
ocr:
  language: "eng"
  tesseract_config: "--psm 6"

# Logging settings
logging:
  level: "INFO"
  file: "~/.computer_genie/logs/genie.log"
```

## 📚 Documentation

### 📖 **Tutorials**
- [Quick Start Guide](docs/tutorials/quick_start.md) - Get started in 5 minutes
- [CLI Usage Guide](docs/tutorials/cli_usage.md) - Complete CLI reference
- [Python API Guide](docs/tutorials/python_api.md) - Programming with Computer Genie

### 🔍 **API Reference**
- [CLI Reference](docs/api/cli_reference.md) - All CLI commands and options
- [Python API Reference](docs/api/python_api.md) - Complete API documentation
- [Agent Reference](docs/api/agents.md) - Vision, Web, and Android agents

### 💡 **Examples**
- [Basic Examples](examples/) - Simple usage examples
- [Advanced Examples](examples/advanced/) - Complex automation scenarios
- [Integration Examples](examples/integration/) - Using with other tools

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/computer_genie.git
cd computer_genie

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=computer_genie
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 **Bug Reports**: Found a bug? Let us know!
- 💡 **Feature Requests**: Have an idea? We'd love to hear it!
- 📝 **Documentation**: Help improve our docs
- 🔧 **Code**: Submit pull requests for fixes and features
- 🧪 **Testing**: Help us test on different platforms

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space

### Dependencies
- **Core**: Pillow, numpy, asyncio
- **OCR**: Tesseract OCR engine
- **CLI**: Click, colorama
- **Optional**: OpenCV (for advanced image processing)

## 🔍 Troubleshooting

### Common Issues

#### ❌ "OCR not available - Tesseract not installed"
```bash
# Solution: Install Tesseract OCR
# Windows:
winget install UB-Mannheim.TesseractOCR

# macOS:
brew install tesseract

# Linux:
sudo apt install tesseract-ocr
```

#### ❌ "Permission denied" errors
```bash
# Solution: Run with appropriate permissions
# Windows: Run as Administrator
# macOS/Linux: Use sudo if needed
sudo python setup.py
```

#### ❌ "Module not found" errors
```bash
# Solution: Reinstall in development mode
pip uninstall computer-genie
pip install -e .
```

### Getting Help
- 📖 Check our [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/your-username/computer_genie/issues)
- 💬 [Discussions](https://github.com/your-username/computer_genie/discussions)
- 📧 Email: support@computergenie.dev

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tesseract OCR** - For excellent text recognition capabilities
- **Pillow** - For advanced image processing functionality
- **OpenCV** - For computer vision algorithms
- **PyAutoGUI** - For cross-platform automation
- **Click & Typer** - For beautiful CLI interfaces
- **Open Source Community** - For continuous innovation and support

## 🔗 Links & Resources

- 🌐 **Official Website**: [computergenie.dev](https://computergenie.dev)
- 📚 **Documentation**: [docs.computergenie.dev](https://docs.computergenie.dev)
- 🐙 **GitHub Repository**: [github.com/abhishek-tech/computer_genie](https://github.com/abhishek-tech/computer_genie)
- 📦 **PyPI Package**: [pypi.org/project/computer-genie](https://pypi.org/project/computer-genie)
- 📧 **Support Email**: support@abhishektech.com
- 💼 **Enterprise Solutions**: enterprise@abhishektech.com

---

<div align="center">

## 🏢 **Abhishek Technologies Pvt Ltd**

**Leading Innovation in AI & Automation Solutions**

[![Company Website](https://img.shields.io/badge/Website-abhishektech.com-blue?style=for-the-badge)](https://abhishektech.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/company/abhishek-technologies)
[![Email](https://img.shields.io/badge/Email-Contact%20Us-red?style=for-the-badge&logo=gmail)](mailto:info@abhishektech.com)

**Specialized in:**
- 🤖 AI & Machine Learning Solutions
- 🔍 Computer Vision & Image Processing
- 🚀 Automation & RPA Development
- 📱 Enterprise Software Solutions
- ☁️ Cloud & DevOps Services

---

**Made with ❤️ by Abhishek Technologies Pvt Ltd**

*Empowering businesses with cutting-edge AI technology*

**© 2024 Abhishek Technologies Pvt Ltd. All rights reserved.**

</div>#   g e i n e - v i s i o n 
 
 