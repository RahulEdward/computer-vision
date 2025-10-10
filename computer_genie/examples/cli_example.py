#!/usr/bin/env python3
"""
CLI Example for Computer Genie

This example shows how to use the Computer Genie CLI commands
to interact with different agents.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and display the result"""
    print(f"\n{'='*50}")
    print(f"Example: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        print(f"Exit code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Command timed out after 30 seconds")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Main example function"""
    print("Computer Genie CLI Examples")
    print("This script demonstrates various CLI commands")
    
    # Example 1: Show help
    run_command("genie --help", "Display main help")
    
    # Example 2: Show version
    run_command("genie --version", "Display version")
    
    # Example 3: Vision agent help
    run_command("genie vision --help", "Vision agent help")
    
    # Example 4: Android agent help
    run_command("genie android --help", "Android agent help")
    
    # Example 5: Web agent help
    run_command("genie web --help", "Web agent help")
    
    # Example 6: Interactive mode help
    run_command("genie interactive --help", "Interactive mode help")
    
    print(f"\n{'='*50}")
    print("CLI Examples completed!")
    print("You can now try running these commands yourself:")
    print("  genie vision 'take a screenshot'")
    print("  genie android 'open settings'")
    print("  genie web 'navigate to google.com'")
    print("  genie interactive")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()