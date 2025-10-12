"""
Semantic Understanding Module

This module provides semantic understanding of application states and transitions,
enabling the automation engine to understand context, predict state changes,
and make intelligent decisions based on application behavior.

Key Components:
- StateManager: Tracks and manages application states
- TransitionAnalyzer: Analyzes state transitions and patterns
- ContextAwareEngine: Provides context-aware automation decisions
- ApplicationProfiler: Profiles applications to understand their behavior
- SemanticValidator: Validates automation actions against expected states
"""

from .state_manager import StateManager, ApplicationState, StateType
from .transition_analyzer import TransitionAnalyzer, StateTransition, TransitionPattern
from .context_engine import ContextAwareEngine, ContextInfo, DecisionContext
from .application_profiler import ApplicationProfiler, ApplicationProfile
from .semantic_validator import SemanticValidator, ValidationResult, ValidationRule

__all__ = [
    'StateManager',
    'ApplicationState', 
    'StateType',
    'TransitionAnalyzer',
    'StateTransition',
    'TransitionPattern',
    'ContextAwareEngine',
    'ContextInfo',
    'DecisionContext',
    'ApplicationProfiler',
    'ApplicationProfile',
    'SemanticValidator',
    'ValidationResult',
    'ValidationRule'
]

__version__ = "1.0.0"
__author__ = "Intelligent Automation Engine"