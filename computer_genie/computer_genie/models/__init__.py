"""Model registry and factory for Computer Genie"""

from typing import Dict, Any, Optional
from .base import BaseModel, ActModel, GetModel, LocateModel

class ModelRegistry:
    """Registry for managing models"""
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
    
    def register(self, name: str, model: BaseModel):
        """Register a model"""
        self._models[name] = model
    
    def get(self, name: str) -> BaseModel:
        """Get a model by name"""
        return self._models.get(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if model exists"""
        return name in self._models
    
    def __getitem__(self, name: str) -> BaseModel:
        """Get model using bracket notation"""
        return self._models[name]
    
    @classmethod
    def default(cls) -> 'ModelRegistry':
        """Create default registry with basic models"""
        registry = cls()
        # Add placeholder models - these would be replaced with actual implementations
        from .genie_model import GenieVisionModel
        registry.register("genie-vision", GenieVisionModel())
        return registry

def get_model(name: str, registry: Optional[ModelRegistry] = None) -> BaseModel:
    """Get model from registry"""
    if registry is None:
        registry = ModelRegistry.default()
    return registry.get(name)

__all__ = ['ModelRegistry', 'get_model', 'BaseModel', 'ActModel', 'GetModel', 'LocateModel']