"""Intelligent Automation Engine

A comprehensive automation framework that provides intelligent task execution,
workflow management, and adaptive automation capabilities.

Key Features:
- Intelligent workflow orchestration
- Adaptive task execution
- Machine learning integration
- Real-time monitoring and analytics
- Extensible plugin architecture
- Multi-modal automation support
- Time-based scheduling and automation

Core Modules:
- core: Core automation engine and workflow management
- ml: Machine learning and AI capabilities
- monitoring: Real-time monitoring and analytics
- plugins: Extensible plugin system
- utils: Utility functions and helpers
- parallel: Parallel execution and task management
- scheduling: Time-based scheduling and automation
"""

from .core import (
    AutomationEngine,
)

# Removed imports for non-existent modules: ml, monitoring, plugins, utils

from .parallel import (
    ExecutionPlanner,
    DependencyAnalyzer,
    TaskScheduler,
    ResourceBalancer,
    SynchronizationManager
)

from .scheduling import (
    CronScheduler,
    NaturalLanguageScheduler,
    ScheduleOptimizer,
    TimeManager,
    EventScheduler
)

__version__ = "1.0.0"

__all__ = [
    # Core components
    'AutomationEngine',
    

    
    # Parallel execution components
    'ExecutionPlanner',
    'DependencyAnalyzer',
    'TaskScheduler',
    'ResourceBalancer',
    'SynchronizationManager',
    
    # Scheduling components
    'CronScheduler',
    'NaturalLanguageScheduler',
    'ScheduleOptimizer',
    'TimeManager',
    'EventScheduler'
]