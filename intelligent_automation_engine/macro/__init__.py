"""
Smart Macro Recording Package

This package provides intelligent macro recording capabilities that can:
- Record user actions and generalize them into reusable patterns
- Detect repetitive patterns and optimize them
- Handle dynamic content and variable elements
- Create robust automation workflows from demonstrations
"""

from .smart_macro_recorder import SmartMacroRecorder
from .pattern_detector import PatternDetector
from .action_generalizer import ActionGeneralizer
from .recording_session import RecordingSession, RecordedAction

__all__ = [
    'SmartMacroRecorder',
    'PatternDetector', 
    'ActionGeneralizer',
    'RecordingSession',
    'RecordedAction'
]

__version__ = "1.0.0"