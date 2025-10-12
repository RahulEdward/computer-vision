"""Parallel Execution Package

This package provides comprehensive parallel execution planning and management
capabilities for optimizing automation workflows through intelligent task
scheduling, dependency analysis, and resource coordination.

Key Components:
- ExecutionPlanner: Creates optimized execution plans for parallel task execution
- DependencyAnalyzer: Analyzes task dependencies and identifies optimization opportunities
- TaskScheduler: Manages task scheduling and execution across multiple workers
- ResourceBalancer: Optimizes resource allocation and load distribution
- SynchronizationManager: Coordinates parallel execution and manages shared resources

Features:
- Intelligent dependency analysis and critical path identification
- Multiple execution strategies (sequential, parallel, pipeline, adaptive)
- Resource-aware scheduling and load balancing
- Deadlock detection and resolution
- Performance monitoring and optimization
- Comprehensive synchronization primitives
"""

from .execution_planner import (
    ExecutionPlanner,
    ExecutionStrategy,
    ParallelizationMode,
    OptimizationObjective,
    ExecutionStage,
    ExecutionMetrics,
    ExecutionPlan,
    PlanningContext
)

from .dependency_analyzer import (
    DependencyAnalyzer,
    DependencyType,
    AnalysisType,
    DependencyStrength,
    Dependency,
    DependencyPath,
    DependencyCluster,
    AnalysisResult,
    DependencyGraph
)

from .task_scheduler import (
    TaskScheduler,
    SchedulingAlgorithm,
    TaskState,
    SchedulingPolicy,
    Task,
    TaskDependency,
    ScheduledTask,
    WorkerNode,
    SchedulingQueue,
    SchedulingMetrics
)

from .resource_balancer import (
    ResourceBalancer,
    BalancingStrategy,
    LoadMetric,
    BalancingObjective,
    ResourceNode,
    LoadMetrics,
    BalancingRule,
    BalancingDecision,
    BalancingPlan
)

from .synchronization_manager import (
    SynchronizationManager,
    SynchronizationType,
    SynchronizationScope,
    LockType,
    SynchronizationState,
    SynchronizationPrimitive,
    SynchronizationRequest,
    SynchronizationEvent,
    DeadlockInfo,
    SynchronizationMetrics
)

__all__ = [
    # Core classes
    'ExecutionPlanner',
    'DependencyAnalyzer', 
    'TaskScheduler',
    'ResourceBalancer',
    'SynchronizationManager',
    
    # Execution planner
    'ExecutionStrategy',
    'ParallelizationMode',
    'OptimizationObjective',
    'ExecutionStage',
    'ExecutionMetrics',
    'ExecutionPlan',
    'PlanningContext',
    
    # Dependency analyzer
    'DependencyType',
    'AnalysisType',
    'DependencyStrength',
    'Dependency',
    'DependencyPath',
    'DependencyCluster',
    'AnalysisResult',
    'DependencyGraph',
    
    # Task scheduler
    'SchedulingAlgorithm',
    'TaskState',
    'SchedulingPolicy',
    'Task',
    'TaskDependency',
    'ScheduledTask',
    'WorkerNode',
    'SchedulingQueue',
    'SchedulingMetrics',
    
    # Resource balancer
    'BalancingStrategy',
    'LoadMetric',
    'BalancingObjective',
    'ResourceNode',
    'LoadMetrics',
    'BalancingRule',
    'BalancingDecision',
    'BalancingPlan',
    
    # Synchronization manager
    'SynchronizationType',
    'SynchronizationScope',
    'LockType',
    'SynchronizationState',
    'SynchronizationPrimitive',
    'SynchronizationRequest',
    'SynchronizationEvent',
    'DeadlockInfo',
    'SynchronizationMetrics'
]