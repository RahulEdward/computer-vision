# Computer Genie CLI Reference

Complete reference for all Computer Genie CLI commands, options, and parameters.

## Global Options

These options are available for all commands:

### `--version`
Display the version of Computer Genie and exit.

```bash
genie --version
```

**Output**: `Computer Genie 1.0.0`

### `--help`
Display help information and exit.

```bash
genie --help
```

## Commands

### `genie vision`

Run the vision agent for desktop automation using computer vision.

#### Syntax
```bash
genie vision [OPTIONS] PROMPT
```

#### Arguments

##### `PROMPT` (required)
The instruction or task for the vision agent to perform.

- **Type**: String
- **Required**: Yes
- **Example**: `"take a screenshot"`

#### Options

##### `--model MODEL`
Specify the model to use for vision processing.

- **Type**: String
- **Default**: `"genie"`
- **Example**: `--model custom_model`

##### `--config CONFIG_FILE`
Path to a configuration file.

- **Type**: Path
- **Default**: None
- **Example**: `--config /path/to/config.yaml`

#### Examples

```bash
# Basic screenshot
genie vision "take a screenshot"

# Screen analysis
genie vision "describe what you see on the screen"

# UI interaction
genie vision "click on the start button"

# With custom model
genie vision --model advanced "find all text on screen"
```

#### Exit Codes
- `0`: Success
- `1`: Error (missing prompt, agent failure, etc.)
- `130`: Interrupted by user

---

### `genie android`

Run the Android agent for mobile device automation.

#### Syntax
```bash
genie android [OPTIONS] PROMPT
```

#### Arguments

##### `PROMPT` (required)
The instruction or task for the Android agent to perform.

- **Type**: String
- **Required**: Yes
- **Example**: `"open settings app"`

#### Options

##### `--device DEVICE_ID`
Specify the Android device ID to use.

- **Type**: String
- **Default**: Default device (first available)
- **Example**: `--device emulator-5554`

#### Examples

```bash
# Basic app interaction
genie android "open the camera app"

# UI navigation
genie android "scroll down and find wifi settings"

# With specific device
genie android --device emulator-5554 "take a screenshot"
```

#### Prerequisites
- ADB (Android Debug Bridge) installed and in PATH
- Android device with USB debugging enabled
- Device connected via USB or WiFi

#### Exit Codes
- `0`: Success
- `1`: Error (missing prompt, connection failure, etc.)
- `130`: Interrupted by user

---

### `genie web`

Run the web agent for browser automation.

#### Syntax
```bash
genie web [OPTIONS] PROMPT
```

#### Arguments

##### `PROMPT` (required)
The instruction or task for the web agent to perform.

- **Type**: String
- **Required**: Yes
- **Example**: `"navigate to google.com"`

#### Options

##### `--browser BROWSER`
Specify the browser to use.

- **Type**: String
- **Default**: `"chrome"`
- **Choices**: `chrome`, `firefox`, `safari`, `edge`
- **Example**: `--browser firefox`

##### `--url URL`
Starting URL to navigate to before executing the prompt.

- **Type**: URL
- **Default**: None
- **Example**: `--url https://example.com`

#### Examples

```bash
# Basic web interaction
genie web "search for python tutorials on Google"

# With starting URL
genie web --url "https://github.com" "find trending repositories"

# With specific browser
genie web --browser firefox --url "https://example.com" "click the login button"
```

#### Prerequisites
- Selenium WebDriver
- Appropriate browser driver (ChromeDriver, GeckoDriver, etc.)
- Supported browser installed

#### Exit Codes
- `0`: Success
- `1`: Error (missing prompt, browser failure, etc.)
- `130`: Interrupted by user

---

### `genie interactive`

Start interactive mode for continuous interaction with an agent.

#### Syntax
```bash
genie interactive [OPTIONS]
```

#### Options

##### `--agent AGENT_TYPE`
Specify the type of agent to use in interactive mode.

- **Type**: String
- **Default**: `"vision"`
- **Choices**: `vision`, `android`, `web`
- **Example**: `--agent web`

#### Interactive Commands

Once in interactive mode, these special commands are available:

##### `help`
Display available interactive commands.

##### `status`
Show current agent status and information.

##### `clear`
Clear the terminal screen.

##### `quit`, `exit`, `q`
Exit interactive mode.

#### Examples

```bash
# Start with vision agent (default)
genie interactive

# Start with web agent
genie interactive --agent web

# Start with Android agent
genie interactive --agent android
```

#### Interactive Session Example

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

genie> status
Agent: vision
Status: Active

genie> take a screenshot
Result: None

genie> quit
Goodbye!
```

#### Exit Codes
- `0`: Success
- `1`: Error (agent initialization failure, etc.)
- `130`: Interrupted by user

---

## Error Handling

### Common Error Messages

#### Missing Prompt
```
Error: No prompt provided. Use 'genie <command> --help' for usage.
```
**Solution**: Provide a prompt string as an argument.

#### Empty Prompt
```
Error: Empty prompt provided.
```
**Solution**: Provide a non-empty prompt string.

#### Missing Dependencies
```
Error: Missing required dependency - <dependency_name>
```
**Solution**: Install the missing dependency as indicated.

#### Permission Denied
```
Error: Permission denied. Please check file permissions.
```
**Solution**: Check and adjust file/system permissions as needed.

#### Connection Errors
```
Error: Could not connect to <service/device>
```
**Solution**: Check connection, ensure service is running, verify configuration.

### Exit Code Reference

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Command completed successfully |
| 1 | General Error | Command failed due to an error |
| 130 | User Interrupt | Command was interrupted by user (Ctrl+C) |

### Logging

All commands provide detailed logging. Log levels and output can be controlled through:

- Environment variables
- Configuration files
- Command-line options (where available)

Logs include:
- Timestamp
- Component name
- Log level (INFO, WARNING, ERROR)
- Message

Example log output:
```
2025-10-07 13:04:22,297 - computer_genie.core.agent - INFO - Starting Vision Agent session
2025-10-07 13:04:23,062 - computer_genie.models.genie_model - INFO - Screenshot captured: (1920, 1080)
```

---

## Environment Variables

### `GENIE_LOG_LEVEL`
Set the logging level for Computer Genie.

- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Default**: `INFO`
- **Example**: `export GENIE_LOG_LEVEL=DEBUG`

### `GENIE_CONFIG_PATH`
Default path for configuration files.

- **Type**: Path
- **Default**: `~/.config/computer_genie/`
- **Example**: `export GENIE_CONFIG_PATH=/custom/config/path`

---

## Configuration Files

Configuration files can be specified using the `--config` option (where available) or placed in the default configuration directory.

### Format
Configuration files should be in YAML format.

### Example Configuration
```yaml
vision:
  model: "genie"
  ocr_enabled: true
  screenshot_quality: "high"

android:
  default_device: "emulator-5554"
  timeout: 30

web:
  default_browser: "chrome"
  headless: false
  timeout: 30
```

---

## Integration

### Shell Integration

Add Computer Genie to your shell for easier access:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias cv="genie vision"
alias ca="genie android"
alias cw="genie web"
alias ci="genie interactive"
```

### Scripting

Computer Genie can be used in scripts:

```bash
#!/bin/bash
# Automated screenshot script
genie vision "take a screenshot" || echo "Screenshot failed"
```

### Exit Code Handling

```bash
if genie vision "check if application is running"; then
    echo "Application is running"
else
    echo "Application check failed"
fi
```