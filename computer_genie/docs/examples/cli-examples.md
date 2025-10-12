# 🖥️ Computer Genie CLI Examples

<div align="center">

![CLI Examples](https://img.shields.io/badge/CLI-Examples-blue?style=for-the-badge&logo=terminal)

**Practical Command-Line Interface Examples**

</div>

---

## 📋 **Table of Contents**

1. [Basic Commands](#basic-commands)
2. [Vision Commands](#vision-commands)
3. [File Processing](#file-processing)
4. [Automation Tasks](#automation-tasks)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)

---

## 🚀 **Basic Commands**

### **Check Version and Help**
```bash
# Check Computer Genie version
genie --version

# Get general help
genie --help

# Get help for specific command
genie vision --help
```

### **System Information**
```bash
# Check system requirements
genie system --check

# View configuration
genie config --show

# Test installation
genie test --quick
```

---

## 👁️ **Vision Commands**

### **Screenshot Operations**
```bash
# Take a screenshot
genie vision "take a screenshot"

# Take screenshot of specific area
genie vision "take screenshot of the top-left corner"

# Take screenshot and save with custom name
genie vision "take a screenshot and save as desktop_capture.png"

# Take screenshot of specific window
genie vision "take screenshot of the browser window"
```

### **Screen Analysis**
```bash
# Analyze current screen
genie vision "what do you see on the screen?"

# Find specific elements
genie vision "find the login button"

# Count elements
genie vision "how many buttons are visible?"

# Describe screen layout
genie vision "describe the layout of this page"
```

### **Text Recognition**
```bash
# Extract all text from screen
genie vision "extract all text from the screen"

# Find specific text
genie vision "find the word 'Submit' on the screen"

# Read text from specific area
genie vision "read the text in the top navigation bar"

# Extract text from image file
genie vision "extract text from document.png"
```

---

## 📁 **File Processing**

### **Image Analysis**
```bash
# Analyze image content
genie vision "analyze the content of image.jpg"

# Extract text from image
genie vision "extract text from invoice.pdf"

# Identify objects in image
genie vision "what objects are in photo.png?"

# Compare two images
genie vision "compare image1.jpg and image2.jpg"
```

### **Document Processing**
```bash
# Process PDF document
genie vision "process document.pdf and extract key information"

# Extract form data
genie vision "extract form fields from form.pdf"

# Analyze invoice
genie vision "extract invoice details from invoice.pdf"

# Process multiple files
genie vision "process all PDF files in the documents folder"
```

### **Batch Operations**
```bash
# Process multiple images
genie vision "extract text from all images in the folder"

# Batch rename files
genie vision "rename all screenshots with timestamp"

# Convert image formats
genie vision "convert all PNG files to JPG"

# Organize files by content
genie vision "organize images by content type"
```

---

## 🤖 **Automation Tasks**

### **Mouse and Keyboard**
```bash
# Click on element
genie vision "click on the Submit button"

# Type text
genie vision "type 'Hello World' in the text field"

# Press keyboard shortcuts
genie vision "press Ctrl+C to copy"

# Scroll page
genie vision "scroll down to see more content"
```

### **Form Automation**
```bash
# Fill login form
genie vision "fill login form with username 'john' and password 'secret'"

# Submit form
genie vision "fill the contact form and submit it"

# Select dropdown options
genie vision "select 'United States' from the country dropdown"

# Upload file
genie vision "upload the file resume.pdf to the file input"
```

### **Web Automation**
```bash
# Navigate to website
genie vision "open browser and go to google.com"

# Search on website
genie vision "search for 'computer vision' on Google"

# Download file
genie vision "download the PDF from the current page"

# Take screenshot of webpage
genie vision "take screenshot of the entire webpage"
```

---

## ⚙️ **Configuration**

### **Model Selection**
```bash
# Use specific model
genie vision --model gpt-4-vision "analyze this image"

# Use local model
genie vision --model local "extract text from screen"

# List available models
genie config --list-models
```

### **Output Options**
```bash
# Save output to file
genie vision "take screenshot" --output screenshot.png

# Set output format
genie vision "extract text" --format json

# Verbose output
genie vision "analyze screen" --verbose

# Quiet mode
genie vision "take screenshot" --quiet
```

### **Performance Settings**
```bash
# Set timeout
genie vision "analyze complex image" --timeout 60

# Set quality
genie vision "take screenshot" --quality high

# Enable caching
genie vision "process document" --cache

# Parallel processing
genie vision "process multiple files" --parallel 4
```

---

## 🔧 **Advanced Usage**

### **Scripting and Automation**
```bash
# Run script file
genie script automation_script.txt

# Execute multiple commands
genie vision "take screenshot" && genie vision "extract text"

# Conditional execution
genie vision "if login button exists, click it"

# Loop operations
genie vision "repeat: take screenshot every 5 seconds for 1 minute"
```

### **Integration with Other Tools**
```bash
# Pipe output to other commands
genie vision "extract text" | grep "important"

# Use with curl for API calls
genie vision "take screenshot" | curl -X POST -F "image=@-" api.example.com

# Combine with system commands
genie vision "take screenshot" && open screenshot.png

# Use in shell scripts
#!/bin/bash
result=$(genie vision "check if page loaded")
if [[ $result == *"loaded"* ]]; then
    echo "Page is ready"
fi
```

### **Error Handling**
```bash
# Retry on failure
genie vision "click submit button" --retry 3

# Ignore errors
genie vision "optional task" --ignore-errors

# Set error handling
genie vision "critical task" --on-error stop

# Debug mode
genie vision "troublesome command" --debug
```

---

## 📊 **Real-World Examples**

### **Daily Automation Tasks**
```bash
# Morning routine
genie vision "take screenshot of desktop for daily backup"
genie vision "check email notifications"
genie vision "open calendar and check today's meetings"

# Work automation
genie vision "fill timesheet with 8 hours for today"
genie vision "download daily reports from dashboard"
genie vision "backup important files to cloud"

# System maintenance
genie vision "take screenshot of system performance"
genie vision "check disk space and clean temporary files"
genie vision "update software if notifications exist"
```

### **Data Entry Automation**
```bash
# Invoice processing
genie vision "extract invoice number, date, and amount from invoice.pdf"
genie vision "enter invoice data into accounting system"
genie vision "mark invoice as processed"

# Customer data entry
genie vision "extract customer information from form.pdf"
genie vision "enter customer data into CRM system"
genie vision "send confirmation email to customer"

# Inventory management
genie vision "scan barcode and update inventory"
genie vision "check stock levels and reorder if needed"
genie vision "generate inventory report"
```

### **Quality Assurance**
```bash
# Website testing
genie vision "test login functionality on website"
genie vision "verify all links work correctly"
genie vision "check responsive design on mobile view"

# Application testing
genie vision "test all buttons in the application"
genie vision "verify form validation works correctly"
genie vision "check error handling for invalid inputs"

# Visual regression testing
genie vision "compare current page with baseline screenshot"
genie vision "detect visual differences in UI elements"
genie vision "generate visual testing report"
```

---

## 🎯 **Tips and Best Practices**

### **Command Optimization**
```bash
# Use specific descriptions for better accuracy
# Good: "click the blue Submit button in the bottom right"
# Better than: "click submit"

# Combine related operations
genie vision "take screenshot, extract text, and save both to files"

# Use timeouts for slow operations
genie vision "wait for page to load completely" --timeout 30
```

### **Error Prevention**
```bash
# Verify before action
genie vision "check if login button exists before clicking"

# Use safe mode for critical operations
genie vision "safely delete temporary files" --safe-mode

# Backup before changes
genie vision "backup current state before making changes"
```

### **Performance Tips**
```bash
# Use caching for repeated operations
genie vision "analyze document.pdf" --cache

# Process in parallel when possible
genie vision "process all images" --parallel

# Use appropriate quality settings
genie vision "take screenshot" --quality medium  # Faster than high
```

---

## 🔍 **Troubleshooting CLI Issues**

### **Common Problems**
```bash
# Command not found
# Solution: Check if Computer Genie is properly installed
pip install computer-genie
genie --version

# Permission denied
# Solution: Run with appropriate permissions or check file access
sudo genie vision "system-level task"  # Linux/Mac
# Or run PowerShell as Administrator on Windows

# Timeout errors
# Solution: Increase timeout or check system performance
genie vision "slow operation" --timeout 120

# Model not available
# Solution: Check available models and internet connection
genie config --list-models
genie vision "task" --model alternative-model
```

### **Debug Commands**
```bash
# Enable debug logging
genie vision "problematic command" --debug --verbose

# Check system status
genie system --status

# Test basic functionality
genie test --basic

# Validate configuration
genie config --validate
```

---

## 📞 **Getting Help**

### **Built-in Help**
```bash
# General help
genie --help

# Command-specific help
genie vision --help
genie config --help
genie system --help

# List all available commands
genie --list-commands

# Show examples
genie --examples
```

### **Support Resources**
- 📚 **Documentation**: [docs.computer-genie.com](https://docs.computer-genie.com)
- 💬 **Community**: [community.computer-genie.com](https://community.computer-genie.com)
- 📧 **Support**: support@abhishektech.com

---

**© 2024 Abhishek Technologies Pvt Ltd. All rights reserved.**

*For more CLI examples and updates, visit [cli.computer-genie.com](https://cli.computer-genie.com)*