#!/usr/bin/env python3
"""
Basic Vision Agent Example

This example demonstrates how to use the Computer Genie Vision Agent
to perform basic computer vision tasks like taking screenshots,
detecting elements, and performing actions.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import computer_genie
sys.path.insert(0, str(Path(__file__).parent.parent))

from computer_genie import VisionAgent
from computer_genie.config import config
from computer_genie.utils import setup_logger

async def main():
    """Main example function"""
    
    # Setup logging
    logger = setup_logger("vision_example", config.log_level)
    logger.info("Starting Vision Agent Example")
    
    try:
        # Initialize the vision agent
        agent = VisionAgent()
        logger.info("Vision agent initialized successfully")
        
        # Example 1: Take a screenshot
        logger.info("Taking a screenshot...")
        screenshot = await agent.screenshot()
        logger.info(f"Screenshot taken: {screenshot.size if screenshot else 'Failed'}")
        
        # Example 2: Describe what's on the screen
        logger.info("Analyzing screen content...")
        description = await agent.describe_screen()
        logger.info(f"Screen description: {description}")
        
        # Example 3: Find an element (example: looking for a button)
        logger.info("Looking for clickable elements...")
        elements = await agent.find_elements("button")
        logger.info(f"Found {len(elements)} button elements")
        
        # Example 4: Perform a simple action
        if elements:
            logger.info("Clicking on the first button found...")
            result = await agent.click(elements[0])
            logger.info(f"Click result: {result}")
        
        logger.info("Vision agent example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in vision agent example: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Run the example
    exit_code = asyncio.run(main())
    sys.exit(exit_code)