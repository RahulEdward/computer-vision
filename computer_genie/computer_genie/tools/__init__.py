"""Tools for Computer Genie automation"""

from .base import BaseTool
from .os_controller import OSController
from .clipboard import Clipboard

# Browser tool placeholder
class Browser:
    """Browser automation tool"""
    
    def __init__(self):
        pass
    
    async def navigate(self, url: str):
        """Navigate to URL"""
        pass
    
    async def click(self, selector: str):
        """Click element by selector"""
        pass
    
    async def type(self, selector: str, text: str):
        """Type text into element"""
        pass
    
    async def close(self):
        """Close browser"""
        pass

__all__ = ['BaseTool', 'OSController', 'Browser', 'Clipboard']