"""Base tool classes for Computer Genie"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def initialize(self):
        """Initialize the tool"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup tool resources"""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the tool's main functionality"""
        pass

__all__ = ['BaseTool']