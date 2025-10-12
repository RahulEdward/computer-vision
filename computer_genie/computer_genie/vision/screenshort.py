"""Screenshot capture functionality"""

import asyncio
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np
import mss
import platform

from computer_genie.utils import setup_logger
from computer_genie.types import Rectangle

logger = setup_logger(__name__)

class Screenshot:
    """Screenshot capture with multi-monitor support"""
    
    def __init__(self, display: int = 1):
        self.display = display
        self.sct = mss.mss()
        self.platform = platform.system().lower()
        
    async def capture(
        self,
        region: Optional[Rectangle] = None,
        display: Optional[int] = None
    ) -> Image.Image:
        """Capture screenshot
        
        Args:
            region: Optional region to capture
            display: Display number (overrides default)
            
        Returns:
            Screenshot as PIL Image
        """
        display_num = display or self.display
        
        # Get monitor info
        if display_num == 0:
            # All monitors combined
            monitor = self.sct.monitors[0]
        else:
            # Specific monitor
            if display_num <= len(self.sct.monitors) - 1:
                monitor = self.sct.monitors[display_num]
            else:
                logger.warning(f"Display {display_num} not found, using primary")
                monitor = self.sct.monitors[1]
        
        # Apply region if specified
        if region:
            monitor = {
                "left": monitor["left"] + region.x,
                "top": monitor["top"] + region.y,
                "width": region.width,
                "height": region.height
            }
        
        # Capture screenshot
        screenshot = self.sct.grab(monitor)
        
        # Convert to PIL Image
        image = Image.frombytes(
            "RGB",
            (screenshot.width, screenshot.height),
            screenshot.rgb
        )
        
        logger.debug(f"Captured screenshot: {image.size}")
        return image
    
    async def capture_numpy(
        self,
        region: Optional[Rectangle] = None,
        display: Optional[int] = None
    ) -> np.ndarray:
        """Capture screenshot as numpy array
        
        Args:
            region: Optional region to capture
            display: Display number
            
        Returns:
            Screenshot as numpy array (BGR format)
        """
        image = await self.capture(region, display)
        return np.array(image)[:, :, ::-1]  # RGB to BGR
    
    def get_displays(self) -> list:
        """Get list of available displays"""
        return self.sct.monitors
    
    def get_display_size(self, display: Optional[int] = None) -> Tuple[int, int]:
        """Get display size
        
        Args:
            display: Display number
            
        Returns:
            Width and height tuple
        """
        display_num = display or self.display
        
        if display_num <= len(self.sct.monitors) - 1:
            monitor = self.sct.monitors[display_num]
            return (monitor["width"], monitor["height"])
        
        return (1920, 1080)  # Default fallback
    
    async def capture_multiple(
        self,
        regions: list[Rectangle],
        display: Optional[int] = None
    ) -> list[Image.Image]:
        """Capture multiple regions efficiently
        
        Args:
            regions: List of regions to capture
            display: Display number
            
        Returns:
            List of PIL Images
        """
        # Capture full screen once
        full_screenshot = await self.capture(display=display)
        
        # Crop regions
        images = []
        for region in regions:
            cropped = full_screenshot.crop((
                region.x,
                region.y,
                region.x + region.width,
                region.y + region.height
            ))
            images.append(cropped)
        
        return images
    
    def close(self):
        """Close screenshot context"""
        self.sct.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()