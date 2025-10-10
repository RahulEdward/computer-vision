# Computer Genie Examples

This directory contains example scripts that demonstrate how to use Computer Genie's various features and capabilities.

## Available Examples

### 1. CLI Example (`cli_example.py`)
Demonstrates how to use the Computer Genie command-line interface.

**Usage:**
```bash
python cli_example.py
```

This script will show you:
- How to display help for different commands
- Available CLI options and arguments
- Example commands you can try

### 2. Basic Vision Example (`basic_vision_example.py`)
Shows how to use the Vision Agent programmatically in Python.

**Usage:**
```bash
python basic_vision_example.py
```

This script demonstrates:
- Taking screenshots
- Analyzing screen content
- Finding UI elements
- Performing basic actions

## Getting Started

1. **Install Computer Genie** (if not already installed):
   ```bash
   pip install -e .
   ```

2. **Run the CLI example** to see available commands:
   ```bash
   python examples/cli_example.py
   ```

3. **Try basic CLI commands**:
   ```bash
   # Show help
   genie --help
   
   # Show version
   genie --version
   
   # Use vision agent
   genie vision "take a screenshot"
   
   # Start interactive mode
   genie interactive
   ```

4. **Run the vision example** to see programmatic usage:
   ```bash
   python examples/basic_vision_example.py
   ```

## Common Use Cases

### Taking Screenshots
```bash
genie vision "take a screenshot and save it"
```

### Finding Elements
```bash
genie vision "find all buttons on the screen"
```

### Web Automation
```bash
genie web "navigate to google.com and search for python"
```

### Android Automation
```bash
genie android "open the settings app"
```

### Interactive Mode
```bash
genie interactive
```

## Configuration

Computer Genie can be configured using environment variables or a `.env` file. See the main README for configuration options.

## Documentation

For comprehensive documentation, see:

- **[Quick Start Guide](../docs/tutorials/quick_start.md)** - Get up and running quickly
- **[CLI Usage Guide](../docs/tutorials/cli_usage.md)** - Detailed CLI documentation
- **[CLI Reference](../docs/api/cli_reference.md)** - Complete command reference

## Troubleshooting

If you encounter issues:

1. **Check your Python version** (3.8+ required)
2. **Verify installation**: `genie --version`
3. **Check logs** in `~/.computer_genie/logs/`
4. **Enable debug mode**: Set `GENIE_DEBUG=true`
5. **See the [troubleshooting section](../docs/tutorials/cli_usage.md#troubleshooting)** in the CLI guide

## Contributing

Feel free to add more examples! Follow the existing patterns and include:
- Clear documentation
- Error handling
- Logging
- Comments explaining the code