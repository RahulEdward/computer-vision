#!/bin/bash
# Computer Genie Quick Installation Script for macOS and Linux
# Run this script to automatically install Computer Genie

echo "========================================"
echo "🤖 COMPUTER GENIE INSTALLER"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python3"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

echo "✅ Python found"
python3 --version

echo
echo "📦 Installing Computer Genie..."
echo

# Run the Python setup script
python3 setup.py

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "🎉 INSTALLATION COMPLETE!"
    echo "========================================"
    echo
    echo "Try these commands:"
    echo "  genie --help"
    echo "  genie vision 'take a screenshot'"
    echo "  python3 quick_test.py"
    echo
else
    echo
    echo "❌ Installation failed"
    echo "Please check the error messages above"
    echo
fi