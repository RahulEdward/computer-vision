"""Main CLI entry point for Computer Genie"""

import asyncio
import argparse
import sys
from typing import Optional
from pathlib import Path

from computer_genie import __version__
from computer_genie.core.agent import VisionAgent, AndroidAgent, WebAgent
from computer_genie.config import config
from computer_genie.utils import setup_logger

logger = setup_logger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        prog="genie",
        description="Computer Genie - AI-powered computer automation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"Computer Genie {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Vision command
    vision_parser = subparsers.add_parser("vision", help="Run vision agent")
    vision_parser.add_argument("--model", default="genie", help="Model to use")
    vision_parser.add_argument("--config", help="Config file path")
    vision_parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    
    # Android command
    android_parser = subparsers.add_parser("android", help="Run Android agent")
    android_parser.add_argument("--device", help="Android device ID")
    android_parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Run web agent")
    web_parser.add_argument("--browser", default="chrome", help="Browser to use")
    web_parser.add_argument("--url", help="Starting URL")
    web_parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument("--agent", choices=["vision", "android", "web"], 
                                   default="vision", help="Agent type to use")
    
    return parser

async def run_vision_agent(args) -> int:
    """Run the vision agent"""
    try:
        # Validate arguments
        if not args.prompt:
            print("Error: No prompt provided. Use 'genie vision --help' for usage.")
            return 1
        
        if len(args.prompt.strip()) == 0:
            print("Error: Empty prompt provided.")
            return 1
        
        logger.info(f"Initializing vision agent with model: {args.model}")
        agent = VisionAgent(model_name=args.model)
        
        async with agent:
            logger.info(f"Executing prompt: {args.prompt}")
            result = await agent.act(args.prompt)
            print(f"Result: {result}")
            logger.info("Vision agent completed successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: Missing required dependency. Please check installation: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: Required file not found: {e}")
        return 1
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        print(f"Error: Permission denied. Please check file permissions: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Vision agent interrupted by user")
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Vision agent failed: {e}", exc_info=True)
        print(f"Error: Vision agent failed - {e}")
        return 1

async def run_android_agent(args) -> int:
    """Run the Android agent"""
    try:
        # Validate arguments
        if not args.prompt:
            print("Error: No prompt provided. Use 'genie android --help' for usage.")
            return 1
        
        if len(args.prompt.strip()) == 0:
            print("Error: Empty prompt provided.")
            return 1
        
        logger.info(f"Initializing Android agent with device: {args.device or 'default'}")
        agent = AndroidAgent()
        
        async with agent:
            logger.info(f"Executing prompt: {args.prompt}")
            result = await agent.act(args.prompt)
            print(f"Result: {result}")
            logger.info("Android agent completed successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: Missing required dependency for Android automation: {e}")
        print("Hint: You may need to install ADB or Android development tools")
        return 1
    except ConnectionError as e:
        logger.error(f"Android connection failed: {e}")
        print(f"Error: Could not connect to Android device: {e}")
        print("Hint: Check if device is connected and ADB is working")
        return 1
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        print(f"Error: Permission denied for Android operations: {e}")
        print("Hint: Enable USB debugging and grant permissions")
        return 1
    except KeyboardInterrupt:
        logger.info("Android agent interrupted by user")
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Android agent failed: {e}", exc_info=True)
        print(f"Error: Android agent failed - {e}")
        return 1

async def run_web_agent(args) -> int:
    """Run the web agent"""
    try:
        # Validate arguments
        if not args.prompt:
            print("Error: No prompt provided. Use 'genie web --help' for usage.")
            return 1
        
        if len(args.prompt.strip()) == 0:
            print("Error: Empty prompt provided.")
            return 1
        
        # Validate URL if provided
        if args.url and not (args.url.startswith('http://') or args.url.startswith('https://')):
            print(f"Warning: URL '{args.url}' doesn't start with http:// or https://")
        
        logger.info(f"Initializing Web agent with browser: {args.browser}")
        agent = WebAgent()
        
        async with agent:
            if args.url:
                logger.info(f"Navigating to: {args.url}")
                await agent.navigate(args.url)
            
            logger.info(f"Executing prompt: {args.prompt}")
            result = await agent.act(args.prompt)
            print(f"Result: {result}")
            logger.info("Web agent completed successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: Missing required dependency for web automation: {e}")
        print("Hint: You may need to install Selenium WebDriver or browser drivers")
        return 1
    except ConnectionError as e:
        logger.error(f"Web connection failed: {e}")
        print(f"Error: Could not connect to website: {e}")
        print("Hint: Check your internet connection and URL")
        return 1
    except TimeoutError as e:
        logger.error(f"Web operation timed out: {e}")
        print(f"Error: Web operation timed out: {e}")
        print("Hint: The website may be slow or unresponsive")
        return 1
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        print(f"Error: Permission denied for web operations: {e}")
        print("Hint: Check browser permissions or run with appropriate privileges")
        return 1
    except KeyboardInterrupt:
        logger.info("Web agent interrupted by user")
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Web agent failed: {e}", exc_info=True)
        print(f"Error: Web agent failed - {e}")
        return 1

async def run_interactive(args) -> int:
    """Run interactive mode"""
    print(f"Starting Computer Genie v{__version__} in interactive mode")
    print(f"Agent type: {args.agent}")
    print("Type 'help' for commands, 'quit' to exit")
    
    # Create appropriate agent
    try:
        logger.info(f"Initializing {args.agent} agent for interactive mode")
        if args.agent == "vision":
            agent = VisionAgent()
        elif args.agent == "android":
            agent = AndroidAgent()
        elif args.agent == "web":
            agent = WebAgent()
        else:
            print(f"Error: Unknown agent type: {args.agent}")
            print("Available agent types: vision, android, web")
            return 1
    except ImportError as e:
        logger.error(f"Failed to initialize {args.agent} agent: {e}")
        print(f"Error: Could not initialize {args.agent} agent - missing dependencies: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize {args.agent} agent: {e}")
        print(f"Error: Could not initialize {args.agent} agent: {e}")
        return 1
    
    try:
        async with agent:
            logger.info("Interactive mode started successfully")
            while True:
                try:
                    prompt = input("\ngenie> ").strip()
                    
                    if not prompt:
                        continue
                    elif prompt.lower() in ["quit", "exit", "q"]:
                        logger.info("User requested exit from interactive mode")
                        break
                    elif prompt.lower() == "help":
                        print("Available commands:")
                        print("  help - Show this help")
                        print("  quit/exit/q - Exit interactive mode")
                        print("  status - Show agent status")
                        print("  clear - Clear screen")
                        print("  Any other text will be sent to the agent")
                        continue
                    elif prompt.lower() == "status":
                        print(f"Agent: {args.agent}")
                        print(f"Status: Active")
                        continue
                    elif prompt.lower() == "clear":
                        import os
                        os.system('cls' if os.name == 'nt' else 'clear')
                        continue
                    
                    # Validate prompt
                    if len(prompt) > 1000:
                        print("Warning: Prompt is very long. Consider breaking it into smaller parts.")
                    
                    # Send prompt to agent
                    logger.info(f"Processing user prompt: {prompt[:50]}...")
                    result = await agent.act(prompt)
                    print(f"Result: {result}")
                    
                except KeyboardInterrupt:
                    print("\nUse 'quit' to exit")
                except EOFError:
                    print("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    print(f"Error: {e}")
                    print("Type 'help' for available commands")
        
        print("Goodbye!")
        logger.info("Interactive mode ended")
        return 0
        
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
        logger.info("Interactive mode interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}", exc_info=True)
        print(f"Error: Interactive mode failed - {e}")
        return 1

async def async_main() -> int:
    """Async main function"""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 0
        
        logger.info(f"Starting Computer Genie CLI with command: {args.command}")
        
        # Route to appropriate handler
        if args.command == "vision":
            return await run_vision_agent(args)
        elif args.command == "android":
            return await run_android_agent(args)
        elif args.command == "web":
            return await run_web_agent(args)
        elif args.command == "interactive":
            return await run_interactive(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            print(f"Error: Unknown command '{args.command}'")
            print("Available commands: vision, android, web, interactive")
            print("Use 'genie --help' for more information")
            return 1
            
    except SystemExit as e:
        # Handle argparse exits (like --help, --version)
        return e.code if e.code is not None else 0
    except Exception as e:
        logger.error(f"Error in async_main: {e}", exc_info=True)
        print(f"Error: Unexpected error in command processing - {e}")
        return 1

def main() -> int:
    """Main entry point"""
    try:
        # Set up basic logging configuration if not already configured
        import logging
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        logger.info("Computer Genie CLI starting")
        result = asyncio.run(async_main())
        logger.info(f"Computer Genie CLI finished with exit code: {result}")
        return result
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logger.info("CLI interrupted by user")
        return 130
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: Missing required dependencies - {e}")
        print("Please check your installation and try again")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        print(f"Error: Unexpected error - {e}")
        print("Please check the logs for more details")
        return 1

# Alias for entry point compatibility
app = main

if __name__ == "__main__":
    sys.exit(main())