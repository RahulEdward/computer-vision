"""
Execution Planner for Parallel Task Optimization

This module provides comprehensive execution planning capabilities for optimizing
parallel task execution in automation workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
import uuid
import logging
import threading
import concurrent.futures
from collections import defaultdict, deque
import networkx as nx
import statistics

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for parallel planning"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    OPTIMIZED = "optimized"


class ParallelizationMode(Enum):
    """Parallelization modes"""
    TASK_LEVEL = "task_level"
    DATA_LEVEL = "data_level"
    PIPELINE_LEVEL = "pipeline_level"
    HYBRID_LEVEL = "hybrid_level"


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_RESOURCES = "minimize_resources"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"


@dataclass
class ExecutionStage:
    """Execution stage in a parallel plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Stage identification
    name: str = ""
    description: str = ""
    stage_number: int = 0
    
    # Stage tasks
    task_ids: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    # Stage dependencies
    depends_on: List[str] = field(default_factory=list)  # Stage IDs
    blocks: List[str] = field(default_factory=list)  # Stage IDs
    
    # Stage execution
    execution_mode: ParallelizationMode = ParallelizationMode.TASK_LEVEL
    max_parallelism: int = 0  # 0 = unlimited
    
    # Stage timing
    estimated_duration: timedelta = timedelta(0)
    actual_duration: timedelta = timedelta(0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Stage resources
    required_resources: Dict[str, float] = field(default_factory=dict)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    
    # Stage status
    status: str = "pending"  # pending, ready, running, completed, failed
    progress: float = 0.0
    
    # Stage metadata
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Metrics for execution performance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timing metrics
    total_execution_time: timedelta = timedelta(0)
    parallel_execution_time: timedelta = timedelta(0)
    sequential_execution_time: timedelta = timedelta(0)
    
    # Parallelization metrics
    parallelization_ratio: float = 0.0  # Parallel time / Total time
    speedup_factor: float = 1.0  # Sequential time / Parallel time
    efficiency: float = 0.0  # Speedup / Number of processors
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    resource_efficiency: float = 0.0
    
    # Task metrics
    total_tasks: int = 0
    parallel_tasks: int = 0
    sequential_tasks: int = 0
    failed_tasks: int = 0
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    data_throughput: float = 0.0
    
    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    
    # Optimization metrics
    optimization_score: float = 0.0
    bottleneck_stages: List[str] = field(default_factory=list)
    
    # Metadata
    measurement_time: datetime = field(default_factory=datetime.now)
    measurement_duration: timedelta = timedelta(0)


@dataclass
class ExecutionPlan:
    """Comprehensive execution plan for parallel tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Plan identification
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    # Plan strategy
    strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL
    parallelization_mode: ParallelizationMode = ParallelizationMode.TASK_LEVEL
    optimization_objective: OptimizationObjective = OptimizationObjective.MINIMIZE_TIME
    
    # Plan structure
    stages: List[ExecutionStage] = field(default_factory=list)
    stage_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Plan execution
    execution_order: List[str] = field(default_factory=list)  # Stage IDs
    critical_path: List[str] = field(default_factory=list)  # Stage IDs
    
    # Plan resources
    total_resources: Dict[str, float] = field(default_factory=dict)
    peak_resources: Dict[str, float] = field(default_factory=dict)
    
    # Plan timing
    estimated_total_time: timedelta = timedelta(0)
    estimated_parallel_time: timedelta = timedelta(0)
    actual_total_time: timedelta = timedelta(0)
    
    # Plan constraints
    max_parallelism: int = 0  # 0 = unlimited
    resource_limits: Dict[str, float] = field(default_factory=dict)
    time_limit: Optional[timedelta] = None
    
    # Plan status
    status: str = "draft"  # draft, validated, executing, completed, failed
    progress: float = 0.0
    
    # Plan results
    execution_metrics: Optional[ExecutionMetrics] = None
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    
    # Plan metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    last_modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class PlanningContext:
    """Context for execution planning"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Context identification
    name: str = ""
    description: str = ""
    
    # Available resources
    available_cpu_cores: int = 1
    available_memory: float = 0.0  # bytes
    available_storage: float = 0.0  # bytes
    available_network_bandwidth: float = 0.0  # bytes/sec
    
    # Resource constraints
    max_cpu_usage: float = 1.0  # 100%
    max_memory_usage: float = 0.8  # 80%
    max_storage_usage: float = 0.9  # 90%
    
    # Execution constraints
    max_parallel_tasks: int = 0  # 0 = unlimited
    max_execution_time: Optional[timedelta] = None
    priority_weights: Dict[str, float] = field(default_factory=dict)
    
    # Environment information
    execution_environment: str = "local"  # local, cloud, distributed
    platform_capabilities: Dict[str, Any] = field(default_factory=dict)
    
    # Planning preferences
    optimization_preferences: List[OptimizationObjective] = field(default_factory=list)
    risk_tolerance: float = 0.5  # 0 = conservative, 1 = aggressive
    
    # Historical data
    historical_performance: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class ExecutionPlanner:
    """Comprehensive execution planner for parallel task optimization"""
    
    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL):
        self.strategy = strategy
        
        # Planning data
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.planning_contexts: Dict[str, PlanningContext] = {}
        
        # Task and dependency data
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.dependency_graph: nx.DiGraph = nx.DiGraph()
        
        # Execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance data
        self.performance_profiles: Dict[str, Dict[str, Any]] = {}
        self.optimization_cache: Dict[str, Any] = {}
        
        # Planning statistics
        self.planner_stats = {
            'total_plans_created': 0,
            'total_plans_executed': 0,
            'average_speedup': 1.0,
            'average_efficiency': 0.0,
            'successful_executions': 0
        }
        
        logger.info(f"Execution planner initialized with {strategy.value} strategy")
    
    def register_task(self, task_id: str, task_info: Dict[str, Any]):
        """Register a task for planning"""
        try:
            self.task_registry[task_id] = {
                'id': task_id,
                'name': task_info.get('name', task_id),
                'description': task_info.get('description', ''),
                'estimated_duration': task_info.get('estimated_duration', timedelta(seconds=1)),
                'resource_requirements': task_info.get('resource_requirements', {}),
                'dependencies': task_info.get('dependencies', []),
                'parallelizable': task_info.get('parallelizable', True),
                'priority': task_info.get('priority', 0),
                'tags': task_info.get('tags', []),
                'registered_at': datetime.now()
            }
            
            # Update dependency graph
            self._update_dependency_graph(task_id, task_info.get('dependencies', []))
            
            logger.debug(f"Registered task: {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to register task: {e}")
    
    def add_dependency(self, task_id: str, depends_on: str, dependency_type: str = "finish_to_start"):
        """Add a dependency between tasks"""
        try:
            if task_id not in self.task_registry or depends_on not in self.task_registry:
                logger.error(f"Tasks not found for dependency: {task_id} -> {depends_on}")
                return
            
            # Add to dependency graph
            self.dependency_graph.add_edge(depends_on, task_id, type=dependency_type)
            
            # Update task registry
            if 'dependencies' not in self.task_registry[task_id]:
                self.task_registry[task_id]['dependencies'] = []
            
            if depends_on not in self.task_registry[task_id]['dependencies']:
                self.task_registry[task_id]['dependencies'].append(depends_on)
            
            logger.debug(f"Added dependency: {task_id} depends on {depends_on}")
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
    
    def analyze_dependencies(self, tasks: List, dependencies: List) -> Dict[str, Any]:
        """Analyze task dependencies and return analysis results"""
        try:
            # Register tasks first
            for task in tasks:
                self.register_task(task.id, {
                    'name': task.name,
                    'description': getattr(task, 'description', ''),
                    'function': getattr(task, 'function', None),
                    'dependencies': getattr(task, 'dependencies', [])
                })
            
            # Add dependencies
            for dep in dependencies:
                source_id = getattr(dep, 'source_task_id', '')
                target_id = getattr(dep, 'target_task_id', '')
                dep_type = getattr(dep, 'dependency_type', 'finish_to_start')
                
                if source_id and target_id:
                    self.add_dependency(target_id, source_id, dep_type)
            
            # Get task IDs
            task_ids = [task.id for task in tasks]
            
            # Perform dependency analysis
            analysis = self._analyze_task_dependencies(task_ids)
            
            # Create dependency graph representation
            dependency_graph = {}
            for task_id in task_ids:
                if task_id in self.task_registry:
                    dependency_graph[task_id] = self.task_registry[task_id].get('dependencies', [])
            
            # Find critical path (longest path through dependencies)
            critical_path = []
            if analysis['dependency_levels']:
                # Simple critical path: tasks that are in the longest dependency chain
                max_level = max(range(len(analysis['dependency_levels'])), 
                              key=lambda i: len(analysis['dependency_levels'][i]))
                critical_path = analysis['dependency_levels'][max_level]
            
            # Identify parallel opportunities
            parallel_opportunities = []
            for level in analysis['dependency_levels']:
                if len(level) > 1:
                    parallel_opportunities.extend(level)
            
            # Identify bottlenecks (tasks with many dependents)
            bottlenecks = []
            for task_id in task_ids:
                dependents = [t for t in task_ids 
                            if task_id in self.task_registry.get(t, {}).get('dependencies', [])]
                if len(dependents) > 1:
                    bottlenecks.append(task_id)
            
            return {
                'dependency_graph': dependency_graph,
                'critical_path': critical_path,
                'parallel_opportunities': parallel_opportunities,
                'bottlenecks': bottlenecks,
                'dependency_levels': analysis['dependency_levels'],
                'total_levels': analysis['total_levels'],
                'max_parallelism': analysis['max_parallelism']
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return {
                'dependency_graph': {},
                'critical_path': [],
                'parallel_opportunities': [],
                'bottlenecks': [],
                'dependency_levels': [],
                'total_levels': 0,
                'max_parallelism': 0
            }
    
    def create_execution_plan(self, task_ids: List[str], 
                            context: Optional[PlanningContext] = None) -> ExecutionPlan:
        """Create an optimized execution plan for the given tasks"""
        try:
            # Use default context if none provided
            if context is None:
                context = self._create_default_context()
            
            # Create plan
            plan = ExecutionPlan(
                name=f"Execution Plan {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Parallel execution plan for {len(task_ids)} tasks",
                strategy=self.strategy
            )
            
            # Validate tasks
            valid_tasks = self._validate_tasks(task_ids)
            if not valid_tasks:
                logger.error("No valid tasks found for planning")
                return plan
            
            # Analyze dependencies
            dependency_analysis = self._analyze_task_dependencies(valid_tasks)
            
            # Create execution stages
            stages = self._create_execution_stages(valid_tasks, dependency_analysis, context)
            plan.stages = stages
            
            # Calculate stage dependencies
            plan.stage_dependencies = self._calculate_stage_dependencies(stages)
            
            # Determine execution order
            plan.execution_order = self._determine_execution_order(stages, plan.stage_dependencies)
            
            # Find critical path
            plan.critical_path = self._find_critical_path(stages, plan.stage_dependencies)
            
            # Optimize plan
            self._optimize_execution_plan(plan, context)
            
            # Calculate estimates
            plan.estimated_total_time = self._estimate_total_execution_time(plan)
            plan.estimated_parallel_time = self._estimate_parallel_execution_time(plan)
            
            # Store plan
            self.execution_plans[plan.id] = plan
            self.planning_contexts[plan.id] = context
            
            # Update statistics
            self.planner_stats['total_plans_created'] += 1
            
            logger.info(f"Created execution plan with {len(stages)} stages")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            return ExecutionPlan()
    
    def execute_plan(self, plan_id: str, executor: Optional[concurrent.futures.Executor] = None) -> bool:
        """Execute a parallel execution plan"""
        try:
            if plan_id not in self.execution_plans:
                logger.error(f"Execution plan not found: {plan_id}")
                return False
            
            plan = self.execution_plans[plan_id]
            context = self.planning_contexts.get(plan_id)
            
            # Use default executor if none provided
            if executor is None:
                max_workers = context.available_cpu_cores if context else 4
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                should_shutdown = True
            else:
                should_shutdown = False
            
            try:
                plan.status = "executing"
                start_time = datetime.now()
                
                # Track execution
                execution_info = {
                    'plan_id': plan_id,
                    'start_time': start_time,
                    'executor': executor,
                    'stage_futures': {},
                    'completed_stages': set(),
                    'failed_stages': set()
                }
                self.active_executions[plan_id] = execution_info
                
                # Execute stages in order
                success = self._execute_stages(plan, executor, execution_info)
                
                # Calculate metrics
                end_time = datetime.now()
                plan.actual_total_time = end_time - start_time
                plan.execution_metrics = self._calculate_execution_metrics(plan, execution_info)
                
                # Update plan status
                plan.status = "completed" if success else "failed"
                plan.progress = 1.0 if success else plan.progress
                
                # Store execution history
                self.execution_history.append({
                    'plan_id': plan_id,
                    'success': success,
                    'execution_time': plan.actual_total_time,
                    'metrics': plan.execution_metrics,
                    'timestamp': end_time
                })
                
                # Update statistics
                self.planner_stats['total_plans_executed'] += 1
                if success:
                    self.planner_stats['successful_executions'] += 1
                
                # Update performance profiles
                self._update_performance_profiles(plan, execution_info)
                
                logger.info(f"Executed plan {plan_id}: {'Success' if success else 'Failed'}")
                return success
                
            finally:
                # Cleanup
                if plan_id in self.active_executions:
                    del self.active_executions[plan_id]
                
                if should_shutdown:
                    executor.shutdown(wait=True)
            
        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            return False
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get execution plan status"""
        try:
            if plan_id not in self.execution_plans:
                return {'error': 'Plan not found'}
            
            plan = self.execution_plans[plan_id]
            execution_info = self.active_executions.get(plan_id, {})
            
            status = {
                'plan_id': plan_id,
                'name': plan.name,
                'status': plan.status,
                'progress': plan.progress,
                'total_stages': len(plan.stages),
                'completed_stages': len(execution_info.get('completed_stages', set())),
                'failed_stages': len(execution_info.get('failed_stages', set())),
                'estimated_time': plan.estimated_total_time,
                'actual_time': plan.actual_total_time,
                'current_stage': None
            }
            
            # Find current stage
            if plan.status == "executing":
                for stage in plan.stages:
                    if stage.status == "running":
                        status['current_stage'] = {
                            'id': stage.id,
                            'name': stage.name,
                            'progress': stage.progress
                        }
                        break
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get plan status: {e}")
            return {'error': str(e)}
    
    def optimize_plan(self, plan_id: str, objective: OptimizationObjective) -> bool:
        """Optimize an existing execution plan"""
        try:
            if plan_id not in self.execution_plans:
                logger.error(f"Execution plan not found: {plan_id}")
                return False
            
            plan = self.execution_plans[plan_id]
            context = self.planning_contexts.get(plan_id)
            
            if not context:
                context = self._create_default_context()
            
            # Update optimization objective
            plan.optimization_objective = objective
            
            # Re-optimize plan
            self._optimize_execution_plan(plan, context)
            
            # Recalculate estimates
            plan.estimated_total_time = self._estimate_total_execution_time(plan)
            plan.estimated_parallel_time = self._estimate_parallel_execution_time(plan)
            
            # Update modification time
            plan.last_modified = datetime.now()
            
            logger.info(f"Optimized plan {plan_id} for {objective.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize plan: {e}")
            return False
    
    def get_optimization_recommendations(self, plan_id: str) -> List[str]:
        """Get optimization recommendations for a plan"""
        recommendations = []
        
        try:
            if plan_id not in self.execution_plans:
                return ["Plan not found"]
            
            plan = self.execution_plans[plan_id]
            
            # Analyze plan structure
            if len(plan.stages) > 10:
                recommendations.append("Consider consolidating stages to reduce overhead")
            
            # Check parallelization opportunities
            sequential_stages = [s for s in plan.stages if len(s.parallel_groups) <= 1]
            if len(sequential_stages) > len(plan.stages) * 0.7:
                recommendations.append("Increase parallelization to improve performance")
            
            # Check resource utilization
            if plan.execution_metrics:
                if plan.execution_metrics.cpu_utilization < 0.5:
                    recommendations.append("Low CPU utilization - consider increasing parallelism")
                
                if plan.execution_metrics.efficiency < 0.6:
                    recommendations.append("Low parallel efficiency - review task dependencies")
            
            # Check critical path
            if len(plan.critical_path) > len(plan.stages) * 0.8:
                recommendations.append("Long critical path - optimize bottleneck stages")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Plan appears well-optimized")
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            recommendations.append(f"Error analyzing plan: {e}")
        
        return recommendations
    
    def get_planner_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planner statistics"""
        try:
            stats = dict(self.planner_stats)
            
            # Add derived statistics
            if stats['total_plans_executed'] > 0:
                stats['success_rate'] = stats['successful_executions'] / stats['total_plans_executed']
            else:
                stats['success_rate'] = 0.0
            
            # Add current state
            stats.update({
                'registered_tasks': len(self.task_registry),
                'active_plans': len(self.execution_plans),
                'active_executions': len(self.active_executions),
                'performance_profiles': len(self.performance_profiles),
                'dependency_graph_nodes': self.dependency_graph.number_of_nodes(),
                'dependency_graph_edges': self.dependency_graph.number_of_edges()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get planner statistics: {e}")
            return {}
    
    # Internal methods
    def _create_default_context(self) -> PlanningContext:
        """Create a default planning context"""
        import psutil
        
        return PlanningContext(
            name="Default Context",
            description="Default planning context",
            available_cpu_cores=psutil.cpu_count(),
            available_memory=psutil.virtual_memory().total,
            optimization_preferences=[OptimizationObjective.MINIMIZE_TIME]
        )
    
    def _update_dependency_graph(self, task_id: str, dependencies: List[str]):
        """Update the dependency graph with task dependencies"""
        try:
            # Add task node
            self.dependency_graph.add_node(task_id)
            
            # Add dependency edges
            for dep in dependencies:
                if dep in self.task_registry:
                    self.dependency_graph.add_edge(dep, task_id)
            
        except Exception as e:
            logger.error(f"Failed to update dependency graph: {e}")
    
    def _validate_tasks(self, task_ids: List[str]) -> List[str]:
        """Validate and filter task IDs"""
        valid_tasks = []
        
        for task_id in task_ids:
            if task_id in self.task_registry:
                valid_tasks.append(task_id)
            else:
                logger.warning(f"Task not found in registry: {task_id}")
        
        return valid_tasks
    
    def _analyze_task_dependencies(self, task_ids: List[str]) -> Dict[str, Any]:
        """Analyze task dependencies for planning"""
        try:
            # Create subgraph for the given tasks
            subgraph = self.dependency_graph.subgraph(task_ids)
            
            # Find independent tasks (no dependencies)
            independent_tasks = [task for task in task_ids 
                               if subgraph.in_degree(task) == 0]
            
            # Find dependent tasks
            dependent_tasks = [task for task in task_ids 
                             if subgraph.in_degree(task) > 0]
            
            # Calculate levels (topological ordering)
            levels = []
            remaining_tasks = set(task_ids)
            
            while remaining_tasks:
                # Find tasks with no remaining dependencies
                ready_tasks = []
                for task in remaining_tasks:
                    deps = set(subgraph.predecessors(task))
                    if deps.issubset(set(task_ids) - remaining_tasks):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Handle circular dependencies
                    ready_tasks = [list(remaining_tasks)[0]]
                
                levels.append(ready_tasks)
                remaining_tasks -= set(ready_tasks)
            
            return {
                'independent_tasks': independent_tasks,
                'dependent_tasks': dependent_tasks,
                'dependency_levels': levels,
                'total_levels': len(levels),
                'max_parallelism': max(len(level) for level in levels) if levels else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze task dependencies: {e}")
            return {
                'independent_tasks': task_ids,
                'dependent_tasks': [],
                'dependency_levels': [task_ids],
                'total_levels': 1,
                'max_parallelism': len(task_ids)
            }
    
    def _create_execution_stages(self, task_ids: List[str], dependency_analysis: Dict[str, Any], 
                               context: PlanningContext) -> List[ExecutionStage]:
        """Create execution stages from task dependencies"""
        stages = []
        
        try:
            dependency_levels = dependency_analysis.get('dependency_levels', [task_ids])
            
            for i, level_tasks in enumerate(dependency_levels):
                # Create stage for this level
                stage = ExecutionStage(
                    name=f"Stage {i + 1}",
                    description=f"Execution stage {i + 1} with {len(level_tasks)} tasks",
                    stage_number=i + 1,
                    task_ids=level_tasks
                )
                
                # Determine parallelization
                if len(level_tasks) > 1 and self.strategy != ExecutionStrategy.SEQUENTIAL:
                    # Group tasks for parallel execution
                    max_parallel = context.max_parallel_tasks or len(level_tasks)
                    parallel_groups = self._create_parallel_groups(level_tasks, max_parallel)
                    stage.parallel_groups = parallel_groups
                    stage.execution_mode = ParallelizationMode.TASK_LEVEL
                else:
                    # Sequential execution
                    stage.parallel_groups = [[task] for task in level_tasks]
                    stage.execution_mode = ParallelizationMode.TASK_LEVEL
                
                # Set dependencies on previous stages
                if i > 0:
                    stage.depends_on = [stages[i-1].id]
                
                # Calculate resource requirements
                stage.required_resources = self._calculate_stage_resources(level_tasks)
                
                # Estimate duration
                stage.estimated_duration = self._estimate_stage_duration(level_tasks, stage.parallel_groups)
                
                stages.append(stage)
            
        except Exception as e:
            logger.error(f"Failed to create execution stages: {e}")
        
        return stages
    
    def _create_parallel_groups(self, task_ids: List[str], max_parallel: int) -> List[List[str]]:
        """Create parallel groups from task IDs"""
        if max_parallel <= 0 or max_parallel >= len(task_ids):
            return [task_ids]  # Single group with all tasks
        
        # Split tasks into groups
        groups = []
        group_size = len(task_ids) // max_parallel
        remainder = len(task_ids) % max_parallel
        
        start_idx = 0
        for i in range(max_parallel):
            # Add one extra task to first 'remainder' groups
            current_group_size = group_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_group_size
            
            if start_idx < len(task_ids):
                groups.append(task_ids[start_idx:end_idx])
                start_idx = end_idx
        
        return [group for group in groups if group]  # Remove empty groups
    
    def _calculate_stage_dependencies(self, stages: List[ExecutionStage]) -> Dict[str, List[str]]:
        """Calculate dependencies between stages"""
        dependencies = {}
        
        for stage in stages:
            dependencies[stage.id] = stage.depends_on.copy()
        
        return dependencies
    
    def _determine_execution_order(self, stages: List[ExecutionStage], 
                                 dependencies: Dict[str, List[str]]) -> List[str]:
        """Determine execution order for stages"""
        try:
            # Simple topological sort
            order = []
            remaining_stages = {s.id: s for s in stages}
            
            while remaining_stages:
                # Find stages with no dependencies
                ready_stages = []
                for stage_id in remaining_stages:
                    deps = dependencies.get(stage_id, [])
                    if all(dep not in remaining_stages for dep in deps):
                        ready_stages.append(stage_id)
                
                if not ready_stages:
                    # Break circular dependencies by stage number
                    ready_stages = [min(remaining_stages.keys(), 
                                      key=lambda x: remaining_stages[x].stage_number)]
                
                # Add ready stages to order (sorted by stage number)
                ready_stages.sort(key=lambda x: remaining_stages[x].stage_number)
                for stage_id in ready_stages:
                    order.append(stage_id)
                    del remaining_stages[stage_id]
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to determine execution order: {e}")
            return [s.id for s in stages]
    
    def _find_critical_path(self, stages: List[ExecutionStage], 
                          dependencies: Dict[str, List[str]]) -> List[str]:
        """Find the critical path through the execution stages"""
        try:
            # Build stage lookup
            stage_lookup = {s.id: s for s in stages}
            
            # Calculate longest path (critical path)
            def calculate_longest_path(stage_id: str, memo: Dict[str, float]) -> float:
                if stage_id in memo:
                    return memo[stage_id]
                
                stage = stage_lookup.get(stage_id)
                if not stage:
                    return 0.0
                
                # Base case: no dependencies
                deps = dependencies.get(stage_id, [])
                if not deps:
                    memo[stage_id] = stage.estimated_duration.total_seconds()
                    return memo[stage_id]
                
                # Recursive case: max of dependency paths + current stage
                max_dep_time = max(calculate_longest_path(dep, memo) for dep in deps)
                memo[stage_id] = max_dep_time + stage.estimated_duration.total_seconds()
                return memo[stage_id]
            
            # Calculate longest paths for all stages
            memo = {}
            for stage in stages:
                calculate_longest_path(stage.id, memo)
            
            # Find the critical path by backtracking from the longest path
            critical_path = []
            
            # Start with the stage that has the longest total path
            current_stage_id = max(memo.keys(), key=lambda x: memo[x])
            
            while current_stage_id:
                critical_path.append(current_stage_id)
                
                # Find the dependency with the longest path
                deps = dependencies.get(current_stage_id, [])
                if deps:
                    current_stage_id = max(deps, key=lambda x: memo.get(x, 0))
                else:
                    current_stage_id = None
            
            # Reverse to get correct order
            critical_path.reverse()
            return critical_path
            
        except Exception as e:
            logger.error(f"Failed to find critical path: {e}")
            return [s.id for s in stages]
    
    def _optimize_execution_plan(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize the execution plan based on objectives"""
        try:
            objective = plan.optimization_objective
            
            if objective == OptimizationObjective.MINIMIZE_TIME:
                self._optimize_for_time(plan, context)
            elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                self._optimize_for_throughput(plan, context)
            elif objective == OptimizationObjective.MINIMIZE_RESOURCES:
                self._optimize_for_resources(plan, context)
            elif objective == OptimizationObjective.BALANCE_LOAD:
                self._optimize_for_load_balance(plan, context)
            elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
                self._optimize_for_efficiency(plan, context)
            
        except Exception as e:
            logger.error(f"Failed to optimize execution plan: {e}")
    
    def _optimize_for_time(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize plan to minimize execution time"""
        try:
            # Increase parallelism where possible
            for stage in plan.stages:
                if len(stage.task_ids) > 1:
                    # Maximize parallel groups
                    max_parallel = context.max_parallel_tasks or len(stage.task_ids)
                    stage.parallel_groups = [stage.task_ids[:max_parallel]]
                    if len(stage.task_ids) > max_parallel:
                        remaining = stage.task_ids[max_parallel:]
                        stage.parallel_groups.extend([[task] for task in remaining])
            
        except Exception as e:
            logger.error(f"Failed to optimize for time: {e}")
    
    def _optimize_for_throughput(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize plan to maximize throughput"""
        try:
            # Balance parallel groups for consistent throughput
            for stage in plan.stages:
                if len(stage.task_ids) > 1:
                    # Create balanced groups
                    num_groups = min(context.available_cpu_cores, len(stage.task_ids))
                    stage.parallel_groups = self._create_parallel_groups(stage.task_ids, num_groups)
            
        except Exception as e:
            logger.error(f"Failed to optimize for throughput: {e}")
    
    def _optimize_for_resources(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize plan to minimize resource usage"""
        try:
            # Reduce parallelism to save resources
            for stage in plan.stages:
                # Limit parallel groups
                max_groups = max(1, context.available_cpu_cores // 2)
                if len(stage.parallel_groups) > max_groups:
                    # Consolidate groups
                    consolidated = []
                    tasks_per_group = len(stage.task_ids) // max_groups
                    
                    for i in range(max_groups):
                        start_idx = i * tasks_per_group
                        end_idx = start_idx + tasks_per_group
                        if i == max_groups - 1:  # Last group gets remaining tasks
                            end_idx = len(stage.task_ids)
                        
                        group_tasks = stage.task_ids[start_idx:end_idx]
                        if group_tasks:
                            consolidated.append(group_tasks)
                    
                    stage.parallel_groups = consolidated
            
        except Exception as e:
            logger.error(f"Failed to optimize for resources: {e}")
    
    def _optimize_for_load_balance(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize plan for load balancing"""
        try:
            # Balance task distribution across stages
            for stage in plan.stages:
                if len(stage.task_ids) > 1:
                    # Create evenly sized groups
                    num_groups = context.available_cpu_cores
                    stage.parallel_groups = self._create_balanced_groups(stage.task_ids, num_groups)
            
        except Exception as e:
            logger.error(f"Failed to optimize for load balance: {e}")
    
    def _optimize_for_efficiency(self, plan: ExecutionPlan, context: PlanningContext):
        """Optimize plan for maximum efficiency"""
        try:
            # Balance between parallelism and overhead
            for stage in plan.stages:
                if len(stage.task_ids) > 1:
                    # Optimal group size based on task count and available cores
                    optimal_groups = min(context.available_cpu_cores, 
                                       max(1, len(stage.task_ids) // 2))
                    stage.parallel_groups = self._create_parallel_groups(stage.task_ids, optimal_groups)
            
        except Exception as e:
            logger.error(f"Failed to optimize for efficiency: {e}")
    
    def _create_balanced_groups(self, task_ids: List[str], num_groups: int) -> List[List[str]]:
        """Create balanced groups for load distribution"""
        if num_groups <= 0 or num_groups >= len(task_ids):
            return [[task] for task in task_ids]
        
        groups = [[] for _ in range(num_groups)]
        
        # Distribute tasks round-robin
        for i, task_id in enumerate(task_ids):
            groups[i % num_groups].append(task_id)
        
        return [group for group in groups if group]
    
    def _calculate_stage_resources(self, task_ids: List[str]) -> Dict[str, float]:
        """Calculate resource requirements for a stage"""
        resources = defaultdict(float)
        
        try:
            for task_id in task_ids:
                task_info = self.task_registry.get(task_id, {})
                task_resources = task_info.get('resource_requirements', {})
                
                for resource, amount in task_resources.items():
                    resources[resource] += amount
            
        except Exception as e:
            logger.error(f"Failed to calculate stage resources: {e}")
        
        return dict(resources)
    
    def _estimate_stage_duration(self, task_ids: List[str], parallel_groups: List[List[str]]) -> timedelta:
        """Estimate duration for a stage"""
        try:
            if not parallel_groups:
                return timedelta(0)
            
            # Calculate duration for each parallel group
            group_durations = []
            
            for group in parallel_groups:
                # For parallel execution, duration is the max of tasks in the group
                group_duration = timedelta(0)
                for task_id in group:
                    task_info = self.task_registry.get(task_id, {})
                    task_duration = task_info.get('estimated_duration', timedelta(seconds=1))
                    group_duration = max(group_duration, task_duration)
                
                group_durations.append(group_duration)
            
            # Total stage duration is sum of group durations (sequential groups)
            return sum(group_durations, timedelta(0))
            
        except Exception as e:
            logger.error(f"Failed to estimate stage duration: {e}")
            return timedelta(seconds=len(task_ids))
    
    def _estimate_total_execution_time(self, plan: ExecutionPlan) -> timedelta:
        """Estimate total execution time for the plan"""
        try:
            # Sum durations along the critical path
            total_time = timedelta(0)
            
            for stage_id in plan.critical_path:
                stage = next((s for s in plan.stages if s.id == stage_id), None)
                if stage:
                    total_time += stage.estimated_duration
            
            return total_time
            
        except Exception as e:
            logger.error(f"Failed to estimate total execution time: {e}")
            return timedelta(0)
    
    def _estimate_parallel_execution_time(self, plan: ExecutionPlan) -> timedelta:
        """Estimate parallel execution time"""
        try:
            # Calculate time if all parallelizable tasks run in parallel
            max_stage_time = timedelta(0)
            
            for stage in plan.stages:
                max_stage_time = max(max_stage_time, stage.estimated_duration)
            
            return max_stage_time
            
        except Exception as e:
            logger.error(f"Failed to estimate parallel execution time: {e}")
            return timedelta(0)
    
    def _execute_stages(self, plan: ExecutionPlan, executor: concurrent.futures.Executor, 
                       execution_info: Dict[str, Any]) -> bool:
        """Execute stages according to the plan"""
        try:
            completed_stages = execution_info['completed_stages']
            failed_stages = execution_info['failed_stages']
            
            for stage_id in plan.execution_order:
                stage = next((s for s in plan.stages if s.id == stage_id), None)
                if not stage:
                    continue
                
                # Check if dependencies are satisfied
                if not all(dep in completed_stages for dep in stage.depends_on):
                    logger.warning(f"Dependencies not satisfied for stage: {stage_id}")
                    failed_stages.add(stage_id)
                    continue
                
                # Execute stage
                stage.status = "running"
                stage.start_time = datetime.now()
                
                success = self._execute_stage(stage, executor)
                
                stage.end_time = datetime.now()
                stage.actual_duration = stage.end_time - stage.start_time
                
                if success:
                    stage.status = "completed"
                    stage.progress = 1.0
                    completed_stages.add(stage_id)
                else:
                    stage.status = "failed"
                    failed_stages.add(stage_id)
                
                # Update plan progress
                plan.progress = len(completed_stages) / len(plan.stages)
            
            # Return success if more stages completed than failed
            return len(completed_stages) > len(failed_stages)
            
        except Exception as e:
            logger.error(f"Failed to execute stages: {e}")
            return False
    
    def _execute_stage(self, stage: ExecutionStage, executor: concurrent.futures.Executor) -> bool:
        """Execute a single stage"""
        try:
            # Submit parallel groups for execution
            group_futures = []
            
            for group in stage.parallel_groups:
                # Submit group for parallel execution
                future = executor.submit(self._execute_task_group, group)
                group_futures.append(future)
            
            # Wait for all groups to complete
            success_count = 0
            for future in concurrent.futures.as_completed(group_futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    if result:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Task group execution failed: {e}")
            
            # Stage succeeds if majority of groups succeed
            return success_count > len(group_futures) / 2
            
        except Exception as e:
            logger.error(f"Failed to execute stage: {e}")
            return False
    
    def _execute_task_group(self, task_ids: List[str]) -> bool:
        """Execute a group of tasks"""
        try:
            # Simulate task execution
            success_count = 0
            
            for task_id in task_ids:
                try:
                    # Simulate task work
                    task_info = self.task_registry.get(task_id, {})
                    duration = task_info.get('estimated_duration', timedelta(seconds=1))
                    
                    # Simulate work (in real implementation, this would call actual task)
                    import time
                    time.sleep(min(duration.total_seconds(), 0.1))  # Cap simulation time
                    
                    success_count += 1
                    logger.debug(f"Executed task: {task_id}")
                    
                except Exception as e:
                    logger.error(f"Task execution failed: {task_id} - {e}")
            
            return success_count > len(task_ids) / 2
            
        except Exception as e:
            logger.error(f"Failed to execute task group: {e}")
            return False
    
    def _calculate_execution_metrics(self, plan: ExecutionPlan, 
                                   execution_info: Dict[str, Any]) -> ExecutionMetrics:
        """Calculate execution metrics for the plan"""
        try:
            metrics = ExecutionMetrics()
            
            # Basic metrics
            metrics.total_execution_time = plan.actual_total_time
            metrics.total_tasks = sum(len(stage.task_ids) for stage in plan.stages)
            metrics.parallel_tasks = sum(
                len([task for group in stage.parallel_groups for task in group])
                for stage in plan.stages if len(stage.parallel_groups) > 1
            )
            metrics.sequential_tasks = metrics.total_tasks - metrics.parallel_tasks
            
            # Calculate speedup and efficiency
            if plan.estimated_total_time.total_seconds() > 0:
                metrics.speedup_factor = (
                    plan.estimated_total_time.total_seconds() / 
                    plan.actual_total_time.total_seconds()
                )
            
            if metrics.parallel_tasks > 0:
                metrics.efficiency = metrics.speedup_factor / metrics.parallel_tasks
            
            # Success rate
            completed_stages = len(execution_info.get('completed_stages', set()))
            total_stages = len(plan.stages)
            if total_stages > 0:
                metrics.success_rate = completed_stages / total_stages
                metrics.error_rate = 1.0 - metrics.success_rate
            
            # Throughput
            if plan.actual_total_time.total_seconds() > 0:
                metrics.tasks_per_second = metrics.total_tasks / plan.actual_total_time.total_seconds()
            
            # Optimization score (composite metric)
            metrics.optimization_score = (
                metrics.speedup_factor * 0.4 +
                metrics.efficiency * 0.3 +
                metrics.success_rate * 0.3
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate execution metrics: {e}")
            return ExecutionMetrics()
    
    def _update_performance_profiles(self, plan: ExecutionPlan, execution_info: Dict[str, Any]):
        """Update performance profiles based on execution results"""
        try:
            # Update task performance profiles
            for stage in plan.stages:
                for task_id in stage.task_ids:
                    if task_id not in self.performance_profiles:
                        self.performance_profiles[task_id] = {
                            'executions': [],
                            'average_duration': timedelta(0),
                            'success_rate': 0.0
                        }
                    
                    profile = self.performance_profiles[task_id]
                    
                    # Add execution record
                    profile['executions'].append({
                        'duration': stage.actual_duration,
                        'success': stage.status == "completed",
                        'timestamp': stage.end_time or datetime.now()
                    })
                    
                    # Update averages (keep last 100 executions)
                    profile['executions'] = profile['executions'][-100:]
                    
                    # Recalculate averages
                    if profile['executions']:
                        durations = [e['duration'] for e in profile['executions']]
                        profile['average_duration'] = sum(durations, timedelta(0)) / len(durations)
                        
                        successes = sum(1 for e in profile['executions'] if e['success'])
                        profile['success_rate'] = successes / len(profile['executions'])
            
            # Update planner statistics
            if plan.execution_metrics:
                self.planner_stats['average_speedup'] = (
                    (self.planner_stats['average_speedup'] * (self.planner_stats['total_plans_executed'] - 1) +
                     plan.execution_metrics.speedup_factor) / self.planner_stats['total_plans_executed']
                )
                
                self.planner_stats['average_efficiency'] = (
                    (self.planner_stats['average_efficiency'] * (self.planner_stats['total_plans_executed'] - 1) +
                     plan.execution_metrics.efficiency) / self.planner_stats['total_plans_executed']
                )
            
        except Exception as e:
            logger.error(f"Failed to update performance profiles: {e}")