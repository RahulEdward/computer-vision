"""Base model classes"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Type, Any, Dict
from PIL import Image
import numpy as np

from computer_genie.types import (
    Point, MessageParam, ResponseSchema,
    Locator, ActSettings
)

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def initialize(self):
        """Initialize the model"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup model resources"""
        pass

class ActModel(BaseModel):
    """Model for performing actions"""
    
    @abstractmethod
    async def act(
        self,
        messages: List[MessageParam],
        context: Dict[str, Any],
        settings: Optional[ActSettings] = None
    ) -> None:
        """Perform an action based on messages and context"""
        pass

class GetModel(BaseModel):
    """Model for getting information from images"""
    
    @abstractmethod
    async def get(
        self,
        query: str,
        image: Union[Image.Image, np.ndarray],
        response_schema: Optional[Type[ResponseSchema]] = None
    ) -> Union[str, ResponseSchema]:
        """Get information from image based on query"""
        pass

class LocateModel(BaseModel):
    """Model for locating elements in images"""
    
    @abstractmethod
    async def locate(
        self,
        locator: Union[str, Locator],
        image: Union[Image.Image, np.ndarray],
        timeout: float = 10.0
    ) -> Optional[Point]:
        """Locate element in image"""
        pass
    
    async def locate_all(
        self,
        locator: Union[str, Locator],
        image: Union[Image.Image, np.ndarray]
    ) -> List[Point]:
        """Locate all matching elements in image"""
        point = await self.locate(locator, image)
        return [point] if point else []

__all__ = ['BaseModel', 'ActModel', 'GetModel', 'LocateModel']