# Computer Genie CLI Usage Guide

This guide covers how to use the Computer Genie command-line interface (CLI) for automating computer tasks using AI.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Commands Overview](#commands-overview)
4. [Vision Agent](#vision-agent)
5. [Android Agent](#android-agent)
6. [Web Agent](#web-agent)
7. [Interactive Mode](#interactive-mode)
8. [Error Handling](#error-handling)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Installation

Make sure Computer Genie is properly installed:

```bash
pip install -e .
```

Verify the installation:

```bash
genie --version
```

## Basic Usage

The Computer Genie CLI follows this general pattern:

```bash
genie <command> [options] <prompt>
```

Get help for any command:

```bash
genie --help
genie <command> --help
```

## Commands Overview

Computer Genie provides four main commands:

| Command | Description | Use Case |
|---------|-------------|----------|
| `vision` | Desktop automation using computer vision | Automate desktop applications, take screenshots, analyze screen content |
| `android` | Android device automation | Automate mobile apps, interact with Android UI |
| `web` | Web browser automation | Automate web interactions, scrape data, test websites |
| `interactive` | Start interactive mode | Continuous interaction with any agent type |

## Vision Agent

The vision agent uses computer vision to interact with your desktop.

### Basic Syntax

```bash
genie vision [options] "<prompt>"
```

### Options

- `--model <model_name>`: Specify the model to use (default: "genie")
- `--config <config_file>`: Path to configuration file

### Examples

```bash
# Take a screenshot and describe what's visible
genie vision "describe what you see on the screen"

# Take a screenshot for analysis
genie vision "take a screenshot"

# Find and interact with UI elements
genie vision "click on the start button"

# Search for specific elements
genie vision "find all buttons on the screen"
```

### Capabilities

- **Screenshot capture**: Automatically captures screen content
- **Element detection**: Finds buttons, text fields, and other UI elements
- **OCR text extraction**: Reads text from the screen (requires Tesseract)
- **Click automation**: Clicks on specified elements
- **Screen analysis**: Provides descriptions of screen content

### Requirements

- **Optional**: Tesseract OCR for text extraction
  - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`

## Android Agent

The Android agent automates interactions with Android devices.

### Basic Syntax

```bash
genie android [options] "<prompt>"
```

### Options

- `--device <device_id>`: Specify Android device ID (optional)

### Examples

```bash
# Interact with Android apps
genie android "open the settings app"

# Navigate through UI
genie android "scroll down and find the wifi settings"

# Input text
genie android "type 'hello world' in the search box"
```

### Requirements

- **ADB (Android Debug Bridge)**: Must be installed and in PATH
- **USB Debugging**: Must be enabled on the Android device
- **Device Connection**: Android device must be connected via USB or WiFi

### Setup

1. Enable Developer Options on your Android device
2. Enable USB Debugging
3. Install ADB tools
4. Connect device and authorize debugging

## Web Agent

The Web agent automates browser interactions.

### Basic Syntax

```bash
genie web [options] "<prompt>"
```

### Options

- `--browser <browser_name>`: Browser to use (default: "chrome")
- `--url <starting_url>`: URL to navigate to initially

### Examples

```bash
# Navigate and interact with websites
genie web --url "https://example.com" "click on the login button"

# Search and extract information
genie web "search for 'python tutorials' on Google"

# Fill forms
genie web "fill in the contact form with my details"
```

### Requirements

- **WebDriver**: Appropriate driver for your browser (ChromeDriver, GeckoDriver, etc.)
- **Browser**: Supported browser installed (Chrome, Firefox, Safari, Edge)

## Interactive Mode

Interactive mode allows continuous interaction with an agent without restarting.

### Basic Syntax

```bash
genie interactive [options]
```

### Options

- `--agent <agent_type>`: Agent type to use (vision, android, web) - default: vision

### Usage

```bash
# Start interactive mode with vision agent
genie interactive --agent vision

# Start interactive mode with web agent
genie interactive --agent web
```

### Interactive Commands

Once in interactive mode, you can use these special commands:

- `help`: Show available commands
- `status`: Show current agent status
- `clear`: Clear the screen
- `quit`, `exit`, or `q`: Exit interactive mode
- Any other text: Send as prompt to the agent

### Example Session

```
$ genie interactive --agent vision
Starting Computer Genie v1.0.0 in interactive mode
Agent type: vision
Type 'help' for commands, 'quit' to exit

genie> help
Available commands:
  help - Show this help
  quit/exit/q - Exit interactive mode
  status - Show agent status
  clear - Clear screen
  Any other text will be sent to the agent

genie> take a screenshot
Result: None

genie> status
Agent: vision
Status: Active

genie> quit
Goodbye!
```

## Error Handling

The CLI provides comprehensive error handling with helpful messages:

### Common Error Types

1. **Missing Prompt**: When no prompt is provided
   ```
   Error: No prompt provided. Use 'genie vision --help' for usage.
   ```

2. **Missing Dependencies**: When required libraries aren't installed
   ```
   Error: Missing required dependency - tesseract is not installed or it's not in your PATH.
   ```

3. **Permission Errors**: When lacking necessary permissions
   ```
   Error: Permission denied. Please check file permissions.
   ```

4. **Connection Errors**: When unable to connect to devices/services
   ```
   Error: Could not connect to Android device. Check if device is connected and ADB is working.
   ```

### Exit Codes

- `0`: Success
- `1`: General error
- `130`: Interrupted by user (Ctrl+C)

## Troubleshooting

### Vision Agent Issues

**Problem**: "OCR not available" warning
**Solution**: Install Tesseract OCR (optional, agent will work without it)

**Problem**: Screenshot fails
**Solution**: Check screen permissions on macOS/Linux

### Android Agent Issues

**Problem**: "Could not connect to Android device"
**Solutions**:
- Ensure USB debugging is enabled
- Check ADB connection: `adb devices`
- Try different USB cable/port
- Restart ADB: `adb kill-server && adb start-server`

### Web Agent Issues

**Problem**: "Missing required dependency for web automation"
**Solutions**:
- Install Selenium: `pip install selenium`
- Download appropriate WebDriver (ChromeDriver, etc.)
- Ensure WebDriver is in PATH

**Problem**: Browser doesn't start
**Solutions**:
- Check browser installation
- Update WebDriver to match browser version
- Check browser permissions

### General Issues

**Problem**: Command not found
**Solution**: Ensure Computer Genie is properly installed: `pip install -e .`

**Problem**: Import errors
**Solution**: Check all dependencies are installed: `pip install -r requirements.txt`

## Examples

### Desktop Automation

```bash
# Take and analyze a screenshot
genie vision "describe what applications are currently open"

# Automate file operations
genie vision "open the file explorer and navigate to Documents"

# Interact with applications
genie vision "open notepad and type 'Hello World'"
```

### Mobile Automation

```bash
# App navigation
genie android "open Instagram and go to my profile"

# Settings automation
genie android "turn on airplane mode"

# Text input
genie android "send a message saying 'Hello' to John"
```

### Web Automation

```bash
# Research and data collection
genie web --url "https://news.ycombinator.com" "find the top 3 stories"

# Form automation
genie web --url "https://forms.example.com" "fill out the contact form"

# E-commerce automation
genie web --url "https://shop.example.com" "search for wireless headphones under $100"
```

### Interactive Workflows

```bash
# Start a research session
genie interactive --agent web
# Then interactively navigate websites, extract information, etc.

# Desktop automation session
genie interactive --agent vision
# Then interactively automate desktop tasks
```

## Best Practices

1. **Be Specific**: Provide clear, specific prompts for better results
2. **Test First**: Try simple commands before complex automation
3. **Use Interactive Mode**: For multi-step workflows, interactive mode is more efficient
4. **Check Dependencies**: Ensure all required tools are installed
5. **Handle Errors**: Pay attention to error messages for troubleshooting guidance

## Getting Help

- Use `genie --help` for general help
- Use `genie <command> --help` for command-specific help
- Check the logs for detailed error information
- Refer to the examples in the `examples/` directory

For more advanced usage and API documentation, see the other guides in this documentation.