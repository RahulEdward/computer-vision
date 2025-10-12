"""Intelligent Planning System

This package provides intelligent planning capabilities for automation workflows,
including goal-oriented planning, path optimization, task decomposition, and resource management.

Key Components:
- GoalPlanner: Analyzes automation goals and creates optimal execution paths
- PathOptimizer: Optimizes execution paths for efficiency and reliability  
- TaskDecomposer: Breaks down complex goals into manageable sub-tasks
- ResourceManager: Manages computational resources and task scheduling
- PlanValidator: Validates plans for feasibility, safety, and compliance

Data Structures:
- Goal, SubGoal, Task: Goal and task representations
- ExecutionPath, PathNode: Path and execution structures
- TaskHierarchy, TaskDependency: Task organization structures
- Resource, ResourcePool: Resource management structures
- ValidationRule, ValidationReport: Plan validation structures
"""

from .goal_planner import (
    GoalPlanner,
    Goal,
    SubGoal,
    Task,
    PlanningContext,
    PlanningResult,
    GoalType,
    GoalStatus,
    Priority
)

from .path_optimizer import (
    PathOptimizer,
    PathNode,
    ExecutionPath,
    PathMetrics,
    OptimizationResult,
    OptimizationStrategy,
    PathType,
    NodeType
)

from .task_decomposer import (
    TaskDecomposer,
    TaskDependency,
    TaskHierarchy,
    DecompositionResult,
    DecompositionStrategy,
    TaskComplexity,
    DependencyType
)

from .resource_manager import (
    ResourceManager,
    ResourceSpec,
    Resource,
    ResourceAllocation,
    TaskSchedule,
    ResourcePool,
    ResourceType,
    ResourceStatus,
    AllocationStrategy,
    SchedulingPolicy
)

from .plan_validator import (
    PlanValidator,
    ValidationRule,
    ValidationIssue,
    ValidationContext,
    ValidationReport,
    PlanValidationResult,
    ValidationLevel,
    ValidationCategory,
    ValidationSeverity,
    ValidationStatus
)

__all__ = [
    # Main classes
    'GoalPlanner',
    'PathOptimizer', 
    'TaskDecomposer',
    'ResourceManager',
    'PlanValidator',
    
    # Goal planning
    'Goal',
    'SubGoal',
    'Task',
    'PlanningContext',
    'PlanningResult',
    'GoalType',
    'GoalStatus',
    'Priority',
    
    # Path optimization
    'PathNode',
    'ExecutionPath',
    'PathMetrics',
    'OptimizationResult',
    'OptimizationStrategy',
    'PathType',
    'NodeType',
    
    # Task decomposition
    'TaskDependency',
    'TaskHierarchy',
    'DecompositionResult',
    'DecompositionStrategy',
    'TaskComplexity',
    'DependencyType',
    
    # Resource management
    'ResourceSpec',
    'Resource',
    'ResourceAllocation',
    'TaskSchedule',
    'ResourcePool',
    'ResourceType',
    'ResourceStatus',
    'AllocationStrategy',
    'SchedulingPolicy',
    
    # Plan validation
    'ValidationRule',
    'ValidationIssue',
    'ValidationContext',
    'ValidationReport',
    'PlanValidationResult',
    'ValidationLevel',
    'ValidationCategory',
    'ValidationSeverity',
    'ValidationStatus'
]