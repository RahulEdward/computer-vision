#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Ensure we can import the local package without installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from computer_genie import VisionAgent
from computer_genie.exceptions import ElementNotFoundError, ActionFailedError


async def main() -> int:
    agent = VisionAgent()
    try:
        async with agent:
            # Try clicking a generic "button" element
            try:
                await agent.click("button")
                print("Clicked on 'button'.")
            except ElementNotFoundError:
                print("No 'button' element found on screen. Skipping click.")
            except ActionFailedError as e:
                print(f"Click action failed: {e}. Skipping click.")

            # Type some text wherever the focus is
            try:
                await agent.type("Hello World")
                print("Typed 'Hello World'.")
            except ActionFailedError as e:
                print(f"Type action failed: {e}.")

            # Ask the agent to describe the current screen
            try:
                result = await agent.get("What's on screen?")
                print(f"Screen description: {result}")
            except ActionFailedError as e:
                print(f"Get action failed: {e}.")

            # Execute a higher-level instruction
            try:
                await agent.act("Complete the form")
                print("Act instruction executed: 'Complete the form'.")
            except ActionFailedError as e:
                print(f"Act action failed: {e}.")

        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
