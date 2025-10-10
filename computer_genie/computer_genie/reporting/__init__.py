"""Reporting components for Computer Genie"""

from typing import Dict, Any, Optional, List
from PIL import Image

class Reporter:
    """Base reporter class"""
    
    def __init__(self, name: str):
        self.name = name
        self.actions = []
    
    def add_action(self, action: str, params: Dict[str, Any], image: Optional[Image.Image] = None):
        """Add action to report"""
        self.actions.append({
            "action": action,
            "params": params,
            "image": image
        })
    
    async def generate(self):
        """Generate report"""
        pass

class ReporterRegistry:
    """Registry for reporters"""
    
    def __init__(self):
        self._reporters: Dict[str, Reporter] = {}
    
    def register(self, name: str, reporter: Reporter):
        """Register a reporter"""
        self._reporters[name] = reporter
    
    def get(self, name: str) -> Optional[Reporter]:
        """Get reporter by name"""
        return self._reporters.get(name)
    
    def get_all(self) -> List[Reporter]:
        """Get all reporters"""
        return list(self._reporters.values())

__all__ = ['Reporter', 'ReporterRegistry']