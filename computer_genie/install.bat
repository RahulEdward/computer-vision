@echo off
REM Computer Genie Quick Installation Script for Windows
REM Run this script to automatically install Computer Genie

echo ========================================
echo 🤖 COMPUTER GENIE INSTALLER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo 📦 Installing Computer Genie...
echo.

REM Run the Python setup script
python setup.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo 🎉 INSTALLATION COMPLETE!
    echo ========================================
    echo.
    echo Try these commands:
    echo   genie --help
    echo   genie vision "take a screenshot"
    echo   python quick_test.py
    echo.
) else (
    echo.
    echo ❌ Installation failed
    echo Please check the error messages above
    echo.
)

pause