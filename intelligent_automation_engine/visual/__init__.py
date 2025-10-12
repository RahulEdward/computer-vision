"""
Visual Programming Interface Package

Provides drag-and-drop visual programming capabilities for creating
automation workflows through an intuitive graphical interface.
"""

from .interface_builder import VisualInterfaceBuilder
from .automation_blocks import AutomationBlock, BlockType, BlockCategory
from .flow_manager import FlowManager
from .canvas_renderer import CanvasRenderer

__all__ = [
    'VisualInterfaceBuilder',
    'AutomationBlock',
    'BlockType',
    'BlockCategory',
    'FlowManager',
    'CanvasRenderer'
]

__version__ = "1.0.0"