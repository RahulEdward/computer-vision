"""
Offline Multilingual Adaptive System
ऑफलाइन बहुभाषी अनुकूली प्रणाली

A comprehensive system that works offline, supports 100+ languages, 
and adapts to user preferences for computer vision and AI tasks.

Key Features:
- Complete offline functionality
- Support for 100+ languages
- Adaptive user preference learning
- Lightweight model compression
- Cross-platform compatibility
- Real-time language detection and switching
"""

from .offline_model_manager import OfflineModelManager
from .multilingual_processor import MultilingualProcessor
from .user_preference_engine import UserPreferenceEngine
from .language_detector import LanguageDetector
from .model_compressor import ModelCompressor
from .adaptive_interface import AdaptiveInterface
from .offline_storage import OfflineStorage

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"

__all__ = [
    'OfflineModelManager',
    'MultilingualProcessor', 
    'UserPreferenceEngine',
    'LanguageDetector',
    'ModelCompressor',
    'AdaptiveInterface',
    'OfflineStorage'
]