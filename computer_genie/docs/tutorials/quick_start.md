# Computer Genie Quick Start Guide

Get up and running with Computer Genie in 5 minutes!

## 1. Installation

```bash
# Clone or navigate to the Computer Genie directory
cd computer_genie

# Install the package
pip install -e .

# Verify installation
genie --version
```

## 2. Your First Command

Let's start with a simple vision command:

```bash
genie vision "take a screenshot"
```

This command will:
- Initialize the vision agent
- Capture a screenshot of your current screen
- Provide basic screen information

## 3. Try Interactive Mode

For a more interactive experience:

```bash
genie interactive --agent vision
```

Then try these commands:
- `help` - See available commands
- `status` - Check agent status
- `take a screenshot` - Capture screen
- `quit` - Exit interactive mode

## 4. Explore Different Agents

### Vision Agent (Desktop Automation)
```bash
genie vision "describe what you see on the screen"
```

### Web Agent (Browser Automation)
```bash
genie web --url "https://example.com" "describe this webpage"
```

### Android Agent (Mobile Automation)
```bash
genie android "take a screenshot of the current screen"
```

## 5. Common Use Cases

### Desktop Tasks
```bash
# Analyze your desktop
genie vision "what applications are currently open?"

# Find UI elements
genie vision "find all buttons on the screen"
```

### Web Tasks
```bash
# Research
genie web --url "https://news.ycombinator.com" "what are the top stories?"

# Navigation
genie web --url "https://github.com" "go to the trending repositories"
```

### Mobile Tasks
```bash
# App interaction
genie android "open the settings app"

# UI navigation
genie android "scroll down to find more options"
```

## 6. Getting Help

- `genie --help` - General help
- `genie vision --help` - Vision agent help
- `genie interactive --help` - Interactive mode help

## 7. What's Next?

- Read the [full CLI usage guide](cli_usage.md) for detailed documentation
- Check out the `examples/` directory for more complex examples
- Explore the interactive mode for multi-step workflows

## Troubleshooting

### "OCR not available" warning
This is normal! The vision agent works without OCR, but you can install Tesseract for text extraction:
- Windows: Download from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

### Command not found
Make sure you're in the right directory and have installed the package:
```bash
pip install -e .
```

### Permission errors
On macOS/Linux, you might need to grant screen recording permissions for the vision agent.

## Need More Help?

- Check the detailed [CLI usage guide](cli_usage.md)
- Look at the examples in `examples/README.md`
- Review error messages - they often contain helpful hints!

Happy automating! 🤖