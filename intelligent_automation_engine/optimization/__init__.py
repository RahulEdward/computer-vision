"""Loop Detection and Optimization Package

This package provides comprehensive loop detection, pattern optimization, and performance
enhancement capabilities for automation workflows.
"""

from .loop_detector import (
    LoopDetector,
    LoopType,
    PatternType,
    DetectionStrategy,
    ActionStep,
    Pattern,
    Loop,
    DetectionResult,
    LoopContext
)

from .pattern_optimizer import (
    PatternOptimizer,
    OptimizationType,
    OptimizationStrategy,
    OptimizationPriority,
    OptimizationRule,
    OptimizationCandidate,
    OptimizationPlan,
    OptimizationResult,
    OptimizationContext
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    MetricType,
    AnalysisType,
    AlertSeverity,
    PerformanceMetric,
    PerformanceProfile,
    PerformanceAlert,
    PerformanceThreshold,
    PerformanceReport,
    AnalysisContext
)

from .cache_manager import (
    CacheManager,
    CacheType,
    EvictionPolicy,
    CacheStrategy,
    CacheStatus,
    CacheKey,
    CacheEntry,
    CacheConfiguration,
    CacheStatistics,
    CacheEvent
)

from .resource_optimizer import (
    ResourceOptimizer,
    ResourceType,
    OptimizationStrategy as ResourceOptimizationStrategy,
    ResourcePriority,
    ThrottleMode,
    ResourceLimit,
    ResourceUsage,
    OptimizationAction,
    ResourcePool,
    OptimizationPlan as ResourceOptimizationPlan,
    ResourceProfile
)

__all__ = [
    # Loop Detection
    'LoopDetector',
    'LoopType',
    'PatternType', 
    'DetectionStrategy',
    'ActionStep',
    'Pattern',
    'Loop',
    'DetectionResult',
    'LoopContext',
    
    # Pattern Optimization
    'PatternOptimizer',
    'OptimizationType',
    'OptimizationStrategy',
    'OptimizationPriority',
    'OptimizationRule',
    'OptimizationCandidate',
    'OptimizationPlan',
    'OptimizationResult',
    'OptimizationContext',
    
    # Performance Analysis
    'PerformanceAnalyzer',
    'MetricType',
    'AnalysisType',
    'AlertSeverity',
    'PerformanceMetric',
    'PerformanceProfile',
    'PerformanceAlert',
    'PerformanceThreshold',
    'PerformanceReport',
    'AnalysisContext',
    
    # Cache Management
    'CacheManager',
    'CacheType',
    'EvictionPolicy',
    'CacheStrategy',
    'CacheStatus',
    'CacheKey',
    'CacheEntry',
    'CacheConfiguration',
    'CacheStatistics',
    'CacheEvent',
    
    # Resource Optimization
    'ResourceOptimizer',
    'ResourceType',
    'ResourceOptimizationStrategy',
    'ResourcePriority',
    'ThrottleMode',
    'ResourceLimit',
    'ResourceUsage',
    'OptimizationAction',
    'ResourcePool',
    'ResourceOptimizationPlan',
    'ResourceProfile'
]