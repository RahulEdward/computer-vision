"""Core Vision Agent implementation"""

import asyncio
import logging
from typing import Optional, Union, Any, Dict, List, Type
from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image
import numpy as np

from computer_genie.config import config
from computer_genie.types import (
    Point, Rectangle, ScreenInfo, MessageParam, 
    ActSettings, ResponseSchema, Locator
)
from computer_genie.models import ModelRegistry, get_model
from computer_genie.tools import OSController, Browser, Clipboard
from computer_genie.vision import Screenshot, OCR, ElementDetector
from computer_genie.locators import LocatorResolver
from computer_genie.reporting import Reporter, ReporterRegistry
from computer_genie.utils import setup_logger, retry_with_backoff
from computer_genie.exceptions import (
    GenieException, ModelNotFoundError, 
    ElementNotFoundError, ActionFailedError
)

logger = setup_logger(__name__)

class VisionAgent:
    """Main Vision Agent for computer automation"""
    
    def __init__(
        self,
        model: Optional[Union[str, Dict[str, str]]] = None,
        models: Optional[ModelRegistry] = None,
        display: int = 1,
        reporters: Optional[List[Reporter]] = None,
        log_level: int = logging.INFO,
        secure_mode: bool = False,
        cache_enabled: bool = True,
        **kwargs
    ):
        """Initialize Vision Agent
        
        Args:
            model: Model name or dict of model mappings
            models: Custom model registry
            display: Display number for multi-monitor setups
            reporters: List of reporters for logging actions
            log_level: Logging level
            secure_mode: Enable secure mode with restrictions
            cache_enabled: Enable caching for better performance
        """
        self.model = model or config.default_model
        self.models = models or ModelRegistry.default()
        self.display = display
        self.reporters = reporters or []
        self.log_level = log_level
        self.secure_mode = secure_mode
        self.cache_enabled = cache_enabled
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize vision components
        self.screenshot = Screenshot(display=display)
        self.ocr = OCR()
        self.element_detector = ElementDetector()
        self.locator_resolver = LocatorResolver()
        
        # Session management
        self._session_id = None
        self._context = {}
        self._action_history = []
        
        # Setup logging
        logging.getLogger("computer_genie").setLevel(log_level)
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all tools"""
        return {
            "os": OSController(display=self.display),
            "browser": Browser(),
            "clipboard": Clipboard(),
        }
    
    @retry_with_backoff(max_retries=3)
    async def click(
        self,
        locator: Union[str, Locator],
        model: Optional[str] = None,
        timeout: float = 10.0,
        **kwargs
    ) -> Point:
        """Click on an element
        
        Args:
            locator: Element locator (text, element, xpath, etc.)
            model: Model to use for element detection
            timeout: Maximum time to wait for element
            
        Returns:
            Point where click occurred
        """
        try:
            # Take screenshot
            screenshot = await self.screenshot.capture()
            
            # Report action start
            self._report_action("click", {"locator": str(locator)}, screenshot)
            
            # Resolve locator to coordinates
            model_name = model or self._get_model_for_action("locate")
            model_instance = self._get_model_instance(model_name)
            
            point = await model_instance.locate(
                locator=locator,
                image=screenshot,
                timeout=timeout
            )
            
            if point is None:
                raise ElementNotFoundError(f"Could not find element: {locator}")
            
            # Perform click
            await self.tools["os"].click(point.x, point.y)
            
            # Log action
            self._action_history.append({
                "action": "click",
                "locator": str(locator),
                "point": point,
                "model": model_name
            })
            
            logger.info(f"Clicked at {point}")
            return point
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            raise ActionFailedError(f"Failed to click: {e}")
    
    async def type(
        self,
        text: str,
        clear_first: bool = False,
        **kwargs
    ) -> None:
        """Type text at current location
        
        Args:
            text: Text to type
            clear_first: Clear existing text first
        """
        try:
            self._report_action("type", {"text": text, "clear_first": clear_first})
            
            if clear_first:
                await self.tools["os"].keyboard_tap("a", modifier_keys=["control"])
                await self.tools["os"].keyboard_tap("delete")
            
            await self.tools["os"].type_text(text)
            
            self._action_history.append({
                "action": "type",
                "text": text,
                "clear_first": clear_first
            })
            
            logger.info(f"Typed text: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Type failed: {e}")
            raise ActionFailedError(f"Failed to type: {e}")
    
    async def get(
        self,
        query: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        response_schema: Optional[Type[ResponseSchema]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[str, ResponseSchema]:
        """Extract information from screen
        
        Args:
            query: Question about screen content
            image: Optional image to analyze instead of screenshot
            response_schema: Pydantic model for structured response
            model: Model to use for extraction
            
        Returns:
            Extracted information as string or structured data
        """
        try:
            # Get image
            if image is None:
                image = await self.screenshot.capture()
            elif isinstance(image, (str, Path)):
                image = Image.open(image)
            
            self._report_action("get", {"query": query}, image)
            
            # Get model
            model_name = model or self._get_model_for_action("get")
            model_instance = self._get_model_instance(model_name)
            
            # Extract information
            result = await model_instance.get(
                query=query,
                image=image,
                response_schema=response_schema
            )
            
            self._action_history.append({
                "action": "get",
                "query": query,
                "result": str(result)[:100],
                "model": model_name
            })
            
            logger.info(f"Extracted: {str(result)[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Get failed: {e}")
            raise ActionFailedError(f"Failed to get information: {e}")
    
    async def act(
        self,
        instruction: str,
        model: Optional[str] = None,
        max_steps: int = 50,
        settings: Optional[ActSettings] = None,
        **kwargs
    ) -> None:
        """Execute complex instruction autonomously
        
        Args:
            instruction: Natural language instruction
            model: Model to use for planning and execution
            max_steps: Maximum steps to execute
            settings: Additional settings for execution
        """
        try:
            self._report_action("act", {"instruction": instruction})
            
            # Get model
            model_name = model or self._get_model_for_action("act")
            model_instance = self._get_model_instance(model_name)
            
            # Create execution context
            context = {
                "instruction": instruction,
                "tools": self.tools,
                "screenshot_fn": self.screenshot.capture,
                "max_steps": max_steps,
                "history": self._action_history
            }
            
            # Execute instruction
            await model_instance.act(
                messages=[{"role": "user", "content": instruction}],
                context=context,
                settings=settings
            )
            
            self._action_history.append({
                "action": "act",
                "instruction": instruction,
                "model": model_name
            })
            
            logger.info(f"Completed instruction: {instruction}")
            
        except Exception as e:
            logger.error(f"Act failed: {e}")
            raise ActionFailedError(f"Failed to execute instruction: {e}")
    
    async def locate(
        self,
        locator: Union[str, Locator],
        model: Optional[str] = None,
        all_matches: bool = False,
        **kwargs
    ) -> Union[Point, List[Point]]:
        """Locate element(s) on screen
        
        Args:
            locator: Element locator
            model: Model to use
            all_matches: Return all matching elements
            
        Returns:
            Point or list of points
        """
        screenshot = await self.screenshot.capture()
        model_name = model or self._get_model_for_action("locate")
        model_instance = self._get_model_instance(model_name)
        
        points = await model_instance.locate_all(
            locator=locator,
            image=screenshot
        )
        
        if not points:
            raise ElementNotFoundError(f"Element not found: {locator}")
        
        return points if all_matches else points[0]
    
    async def mouse_move(
        self,
        locator: Union[str, Locator, Point],
        model: Optional[str] = None,
        **kwargs
    ) -> Point:
        """Move mouse to element or position
        
        Args:
            locator: Target location
            model: Model to use
            
        Returns:
            Target point
        """
        if isinstance(locator, Point):
            point = locator
        else:
            point = await self.locate(locator, model=model)
        
        await self.tools["os"].mouse_move(point.x, point.y)
        
        self._action_history.append({
            "action": "mouse_move",
            "point": point
        })
        
        return point
    
    async def keyboard(
        self,
        key: str,
        modifier_keys: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Send keyboard input
        
        Args:
            key: Key to press
            modifier_keys: Modifier keys (ctrl, shift, alt)
        """
        await self.tools["os"].keyboard_tap(key, modifier_keys=modifier_keys)
        
        self._action_history.append({
            "action": "keyboard",
            "key": key,
            "modifiers": modifier_keys
        })
    
    async def screenshot(
        self,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Image.Image:
        """Take screenshot
        
        Args:
            save_path: Optional path to save screenshot
            
        Returns:
            Screenshot as PIL Image
        """
        image = await self.screenshot.capture()
        
        if save_path:
            image.save(save_path)
            logger.info(f"Screenshot saved to {save_path}")
        
        return image
    
    async def wait(
        self,
        locator: Union[str, Locator],
        timeout: float = 10.0,
        model: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Wait for element to appear
        
        Args:
            locator: Element to wait for
            timeout: Maximum wait time
            model: Model to use
            
        Returns:
            True if element found, False if timeout
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                await self.locate(locator, model=model)
                return True
            except ElementNotFoundError:
                await asyncio.sleep(0.5)
        
        return False
    
    def _get_model_for_action(self, action: str) -> str:
        """Get model name for action"""
        if isinstance(self.model, dict):
            return self.model.get(action, config.default_model)
        return self.model
    
    def _get_model_instance(self, model_name: str):
        """Get model instance from registry"""
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model not found: {model_name}")
        return self.models[model_name]
    
    def _report_action(self, action: str, params: Dict, image: Optional[Image.Image] = None):
        """Report action to all reporters"""
        for reporter in self.reporters:
            reporter.add_action(action, params, image)
    
    async def __aenter__(self):
        """Async context manager entry"""
        logger.info("Starting Vision Agent session")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Generate reports
        for reporter in self.reporters:
            await reporter.generate()
        
        # Cleanup
        await self.cleanup()
        
        logger.info("Vision Agent session ended")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Close tools
        for tool in self.tools.values():
            if hasattr(tool, "close"):
                await tool.close()
    
    # Synchronous wrapper for backwards compatibility
    def __enter__(self):
        """Sync context manager entry"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.__aenter__())
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.__aexit__(exc_type, exc_val, exc_tb))


class AndroidAgent(VisionAgent):
    """Android-specific automation agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Android-specific initialization
        
    async def swipe(self, start: Point, end: Point, duration: float = 0.5):
        """Swipe gesture on Android device"""
        # Placeholder implementation
        pass
    
    async def tap(self, point: Point):
        """Tap gesture on Android device"""
        # Placeholder implementation
        pass

class WebAgent(VisionAgent):
    """Web-specific automation agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Web-specific initialization
        
    async def navigate(self, url: str):
        """Navigate to URL"""
        await self.tools["browser"].navigate(url)
    
    async def find_element(self, selector: str):
        """Find web element by CSS selector"""
        # Placeholder implementation
        pass