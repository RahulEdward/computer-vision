"""
Dependency Analyzer for Parallel Execution

This module provides comprehensive dependency analysis capabilities for
optimizing parallel task execution and identifying execution bottlenecks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
import uuid
import logging
import networkx as nx
from collections import defaultdict, deque
import itertools

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies between tasks"""
    FINISH_TO_START = "finish_to_start"  # Task B starts after Task A finishes
    START_TO_START = "start_to_start"    # Task B starts when Task A starts
    FINISH_TO_FINISH = "finish_to_finish"  # Task B finishes when Task A finishes
    START_TO_FINISH = "start_to_finish"  # Task B finishes when Task A starts
    DATA_DEPENDENCY = "data_dependency"  # Task B needs data from Task A
    RESOURCE_DEPENDENCY = "resource_dependency"  # Tasks share exclusive resources
    CONDITIONAL_DEPENDENCY = "conditional_dependency"  # Dependency based on conditions


class AnalysisType(Enum):
    """Types of dependency analysis"""
    BASIC = "basic"
    CRITICAL_PATH = "critical_path"
    BOTTLENECK = "bottleneck"
    PARALLELIZATION = "parallelization"
    OPTIMIZATION = "optimization"
    CIRCULAR = "circular"
    TRANSITIVE = "transitive"
    COMPREHENSIVE = "comprehensive"


class DependencyStrength(Enum):
    """Strength of dependencies"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


@dataclass
class Dependency:
    """Represents a dependency between tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Dependency identification
    source_task: str = ""  # Task that must complete first
    target_task: str = ""  # Task that depends on source
    
    # Dependency properties
    dependency_type: DependencyType = DependencyType.FINISH_TO_START
    strength: DependencyStrength = DependencyStrength.MODERATE
    
    # Dependency constraints
    delay: timedelta = timedelta(0)  # Minimum delay between tasks
    condition: Optional[str] = None  # Conditional dependency
    
    # Dependency metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyPath:
    """Represents a path through dependencies"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Path identification
    name: str = ""
    description: str = ""
    
    # Path structure
    task_sequence: List[str] = field(default_factory=list)
    dependency_sequence: List[str] = field(default_factory=list)  # Dependency IDs
    
    # Path metrics
    total_duration: timedelta = timedelta(0)
    critical_path: bool = False
    parallelizable: bool = True
    
    # Path analysis
    bottlenecks: List[str] = field(default_factory=list)  # Task IDs
    optimization_potential: float = 0.0  # 0-1 scale
    
    # Path metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyCluster:
    """Represents a cluster of related dependencies"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Cluster identification
    name: str = ""
    description: str = ""
    
    # Cluster structure
    task_ids: List[str] = field(default_factory=list)
    dependency_ids: List[str] = field(default_factory=list)
    
    # Cluster properties
    cluster_type: str = "general"  # general, sequential, parallel, mixed
    parallelization_potential: float = 0.0  # 0-1 scale
    
    # Cluster metrics
    internal_dependencies: int = 0
    external_dependencies: int = 0
    complexity_score: float = 0.0
    
    # Cluster analysis
    optimization_opportunities: List[str] = field(default_factory=list)
    recommended_strategy: str = ""
    
    # Cluster metadata
    created_at: datetime = field(default_factory=datetime.now)
    analysis_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Results of dependency analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Analysis identification
    analysis_type: AnalysisType = AnalysisType.BASIC
    analysis_name: str = ""
    
    # Analysis results
    critical_paths: List[DependencyPath] = field(default_factory=list)
    dependency_clusters: List[DependencyCluster] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)  # Task IDs
    has_circular_dependencies: bool = False
    circular_dependency_chains: List[List[str]] = field(default_factory=list)
    
    # Analysis metrics
    total_tasks: int = 0
    total_dependencies: int = 0
    parallelization_ratio: float = 0.0
    complexity_score: float = 0.0
    
    # Optimization insights
    optimization_opportunities: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    potential_speedup: float = 1.0
    
    # Analysis metadata
    analysis_duration: timedelta = timedelta(0)
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_path(self) -> List[str]:
        """Get the task sequence of the first critical path (for backward compatibility)"""
        if self.critical_paths:
            return self.critical_paths[0].task_sequence
        return []
    
    @property
    def total_duration(self) -> timedelta:
        """Get total duration from critical path"""
        if self.critical_paths:
            return self.critical_paths[0].total_duration
        return timedelta(0)


@dataclass
class DependencyGraph:
    """Comprehensive dependency graph representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Graph identification
    name: str = ""
    description: str = ""
    
    # Graph structure
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    
    # Graph properties
    is_acyclic: bool = True
    has_cycles: List[List[str]] = field(default_factory=list)  # Circular dependencies
    
    # Graph metrics
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    
    # Graph analysis
    topological_order: List[str] = field(default_factory=list)
    strongly_connected_components: List[List[str]] = field(default_factory=list)
    
    # Graph metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


class DependencyAnalyzer:
    """Comprehensive dependency analyzer for parallel execution optimization"""
    
    def __init__(self):
        # Core data structures
        self.dependency_graph = DependencyGraph()
        self.nx_graph = nx.DiGraph()  # NetworkX graph for advanced algorithms
        
        # Analysis cache
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.path_cache: Dict[str, List[DependencyPath]] = {}
        
        # Performance tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_analyses': 0,
            'average_analysis_time': timedelta(0),
            'cache_hit_rate': 0.0
        }
        
        logger.info("Dependency analyzer initialized")
    
    def add_task(self, task_id: str, task_info: Dict[str, Any]):
        """Add a task to the dependency graph"""
        try:
            self.dependency_graph.tasks[task_id] = {
                'id': task_id,
                'name': task_info.get('name', task_id),
                'description': task_info.get('description', ''),
                'estimated_duration': task_info.get('estimated_duration', timedelta(seconds=1)),
                'resource_requirements': task_info.get('resource_requirements', {}),
                'parallelizable': task_info.get('parallelizable', True),
                'priority': task_info.get('priority', 0),
                'added_at': datetime.now()
            }
            
            # Add to NetworkX graph
            self.nx_graph.add_node(task_id, **self.dependency_graph.tasks[task_id])
            
            # Update graph metrics
            self._update_graph_metrics()
            
            logger.debug(f"Added task to dependency graph: {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
    
    def add_dependency(self, source_task: str, target_task: str, 
                      dependency_type: DependencyType = DependencyType.FINISH_TO_START,
                      **kwargs) -> str:
        """Add a dependency between tasks"""
        try:
            # Validate tasks exist
            if source_task not in self.dependency_graph.tasks:
                logger.error(f"Source task not found: {source_task}")
                return ""
            
            if target_task not in self.dependency_graph.tasks:
                logger.error(f"Target task not found: {target_task}")
                return ""
            
            # Create dependency
            dependency = Dependency(
                source_task=source_task,
                target_task=target_task,
                dependency_type=dependency_type,
                **kwargs
            )
            
            # Add to dependency graph
            self.dependency_graph.dependencies[dependency.id] = dependency
            
            # Add to NetworkX graph
            self.nx_graph.add_edge(
                source_task, 
                target_task,
                dependency_id=dependency.id,
                dependency_type=dependency_type.value,
                weight=kwargs.get('delay', timedelta(0)).total_seconds()
            )
            
            # Check for cycles
            self._check_for_cycles()
            
            # Update graph metrics
            self._update_graph_metrics()
            
            # Clear relevant caches
            self._clear_analysis_cache()
            
            logger.debug(f"Added dependency: {source_task} -> {target_task}")
            return dependency.id
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return ""
    
    def remove_dependency(self, dependency_id: str) -> bool:
        """Remove a dependency from the graph"""
        try:
            if dependency_id not in self.dependency_graph.dependencies:
                logger.error(f"Dependency not found: {dependency_id}")
                return False
            
            dependency = self.dependency_graph.dependencies[dependency_id]
            
            # Remove from NetworkX graph
            if self.nx_graph.has_edge(dependency.source_task, dependency.target_task):
                self.nx_graph.remove_edge(dependency.source_task, dependency.target_task)
            
            # Remove from dependency graph
            del self.dependency_graph.dependencies[dependency_id]
            
            # Update graph metrics
            self._update_graph_metrics()
            
            # Clear relevant caches
            self._clear_analysis_cache()
            
            logger.debug(f"Removed dependency: {dependency_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False
    
    def analyze_dependencies(self, analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
                           task_subset: Optional[List[str]] = None) -> AnalysisResult:
        """Perform comprehensive dependency analysis"""
        try:
            start_time = datetime.now()
            
            # Check cache
            cache_key = f"{analysis_type.value}_{hash(tuple(sorted(task_subset or [])))}"
            if cache_key in self.analysis_cache:
                self.performance_metrics['cache_hit_rate'] += 1
                return self.analysis_cache[cache_key]
            
            # Create analysis result
            result = AnalysisResult(
                analysis_type=analysis_type,
                analysis_name=f"{analysis_type.value.title()} Analysis"
            )
            
            # Determine tasks to analyze
            if task_subset:
                tasks_to_analyze = [t for t in task_subset if t in self.dependency_graph.tasks]
            else:
                tasks_to_analyze = list(self.dependency_graph.tasks.keys())
            
            result.total_tasks = len(tasks_to_analyze)
            result.total_dependencies = len(self.dependency_graph.dependencies)
            
            # Perform specific analysis
            if analysis_type in [AnalysisType.BASIC, AnalysisType.COMPREHENSIVE]:
                self._perform_basic_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.CRITICAL_PATH, AnalysisType.COMPREHENSIVE]:
                self._perform_critical_path_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.BOTTLENECK, AnalysisType.COMPREHENSIVE]:
                self._perform_bottleneck_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.PARALLELIZATION, AnalysisType.COMPREHENSIVE]:
                self._perform_parallelization_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.OPTIMIZATION, AnalysisType.COMPREHENSIVE]:
                self._perform_optimization_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.CIRCULAR, AnalysisType.COMPREHENSIVE]:
                self._perform_circular_dependency_analysis(result, tasks_to_analyze)
            
            if analysis_type in [AnalysisType.TRANSITIVE, AnalysisType.COMPREHENSIVE]:
                self._perform_transitive_analysis(result, tasks_to_analyze)
            
            # Calculate analysis duration
            end_time = datetime.now()
            result.analysis_duration = end_time - start_time
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            # Update performance metrics
            self._update_performance_metrics(result.analysis_duration)
            
            # Store in history
            self.analysis_history.append({
                'analysis_type': analysis_type.value,
                'task_count': len(tasks_to_analyze),
                'duration': result.analysis_duration,
                'timestamp': end_time
            })
            
            logger.info(f"Completed {analysis_type.value} analysis in {result.analysis_duration}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return AnalysisResult()
    
    def find_critical_path(self, start_tasks: Optional[List[str]] = None,
                          end_tasks: Optional[List[str]] = None) -> List[DependencyPath]:
        """Find critical paths in the dependency graph"""
        try:
            # Use cache if available
            cache_key = f"critical_path_{hash(tuple(sorted(start_tasks or [])))}"
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]
            
            critical_paths = []
            
            # Determine start and end tasks
            if not start_tasks:
                start_tasks = [task for task in self.dependency_graph.tasks.keys()
                             if self.nx_graph.in_degree(task) == 0]
            
            if not end_tasks:
                end_tasks = [task for task in self.dependency_graph.tasks.keys()
                           if self.nx_graph.out_degree(task) == 0]
            
            # Find longest paths from each start to each end
            for start_task in start_tasks:
                for end_task in end_tasks:
                    try:
                        # Find longest path (critical path)
                        path = self._find_longest_path(start_task, end_task)
                        if path:
                            critical_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
            # Sort by duration (longest first)
            critical_paths.sort(key=lambda p: p.total_duration, reverse=True)
            
            # Mark the longest path as critical
            if critical_paths:
                critical_paths[0].critical_path = True
            
            # Cache result
            self.path_cache[cache_key] = critical_paths
            
            return critical_paths
            
        except Exception as e:
            logger.error(f"Failed to find critical path: {e}")
            return []
    
    def identify_bottlenecks(self, threshold: float = 0.8) -> List[str]:
        """Identify bottleneck tasks in the dependency graph"""
        try:
            bottlenecks = []
            
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(self.nx_graph)
            
            # Find tasks with high centrality
            max_centrality = max(centrality.values()) if centrality else 0
            threshold_value = max_centrality * threshold
            
            for task, centrality_value in centrality.items():
                if centrality_value >= threshold_value:
                    bottlenecks.append(task)
            
            # Also consider tasks with high in-degree or out-degree
            for task in self.dependency_graph.tasks.keys():
                in_degree = self.nx_graph.in_degree(task)
                out_degree = self.nx_graph.out_degree(task)
                
                if in_degree > 3 or out_degree > 3:  # Arbitrary threshold
                    if task not in bottlenecks:
                        bottlenecks.append(task)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to identify bottlenecks: {e}")
            return []
    
    def find_parallelizable_groups(self) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel"""
        try:
            parallelizable_groups = []
            
            # Get topological ordering
            try:
                topo_order = list(nx.topological_sort(self.nx_graph))
            except nx.NetworkXError:
                # Graph has cycles, use partial ordering
                topo_order = list(self.dependency_graph.tasks.keys())
            
            # Group tasks by dependency level
            levels = []
            remaining_tasks = set(topo_order)
            
            while remaining_tasks:
                # Find tasks with no dependencies in remaining set
                current_level = []
                for task in list(remaining_tasks):
                    predecessors = set(self.nx_graph.predecessors(task))
                    if predecessors.issubset(set(topo_order) - remaining_tasks):
                        current_level.append(task)
                
                if not current_level:
                    # Handle remaining tasks (possibly due to cycles)
                    current_level = [list(remaining_tasks)[0]]
                
                levels.append(current_level)
                remaining_tasks -= set(current_level)
            
            # Filter levels with multiple tasks (parallelizable)
            for level in levels:
                if len(level) > 1:
                    # Further filter by parallelizable flag
                    parallelizable_tasks = [
                        task for task in level
                        if self.dependency_graph.tasks.get(task, {}).get('parallelizable', True)
                    ]
                    if len(parallelizable_tasks) > 1:
                        parallelizable_groups.append(parallelizable_tasks)
            
            return parallelizable_groups
            
        except Exception as e:
            logger.error(f"Failed to find parallelizable groups: {e}")
            return []
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        try:
            cycles = []
            
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(self.nx_graph))
            
            # Filter SCCs with more than one node (actual cycles)
            for scc in sccs:
                if len(scc) > 1:
                    cycles.append(list(scc))
            
            # Update graph state
            self.dependency_graph.has_cycles = cycles
            self.dependency_graph.is_acyclic = len(cycles) == 0
            
            return cycles
            
        except Exception as e:
            logger.error(f"Failed to detect circular dependencies: {e}")
            return []
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations for the dependency graph"""
        suggestions = []
        
        try:
            # Analyze current state
            bottlenecks = self.identify_bottlenecks()
            parallelizable_groups = self.find_parallelizable_groups()
            cycles = self.detect_circular_dependencies()
            
            # Suggest based on analysis
            if cycles:
                suggestions.append(f"Remove {len(cycles)} circular dependencies to enable proper scheduling")
            
            if bottlenecks:
                suggestions.append(f"Optimize {len(bottlenecks)} bottleneck tasks: {', '.join(bottlenecks[:3])}")
            
            if parallelizable_groups:
                total_parallel_tasks = sum(len(group) for group in parallelizable_groups)
                suggestions.append(f"Leverage {total_parallel_tasks} parallelizable tasks across {len(parallelizable_groups)} groups")
            
            # Check for long sequential chains
            critical_paths = self.find_critical_path()
            if critical_paths and len(critical_paths[0].task_sequence) > 10:
                suggestions.append("Break down long sequential chains to improve parallelization")
            
            # Check for resource conflicts
            resource_conflicts = self._detect_resource_conflicts()
            if resource_conflicts:
                suggestions.append(f"Resolve {len(resource_conflicts)} resource conflicts")
            
            # Check for optimization opportunities
            if len(self.dependency_graph.tasks) > 20:
                density = self.dependency_graph.density
                if density > 0.5:
                    suggestions.append("High dependency density - consider task consolidation")
                elif density < 0.1:
                    suggestions.append("Low dependency density - consider adding more parallelism")
            
            if not suggestions:
                suggestions.append("Dependency graph appears well-optimized")
            
        except Exception as e:
            logger.error(f"Failed to suggest optimizations: {e}")
            suggestions.append(f"Error analyzing graph: {e}")
        
        return suggestions
    
    def get_dependency_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dependency statistics"""
        try:
            stats = {
                # Basic counts
                'total_tasks': len(self.dependency_graph.tasks),
                'total_dependencies': len(self.dependency_graph.dependencies),
                'graph_density': self.dependency_graph.density,
                
                # Graph properties
                'is_acyclic': self.dependency_graph.is_acyclic,
                'has_cycles': len(self.dependency_graph.has_cycles) > 0,
                'cycle_count': len(self.dependency_graph.has_cycles),
                
                # Dependency types
                'dependency_types': {},
                'dependency_strengths': {},
                
                # Task properties
                'parallelizable_tasks': 0,
                'sequential_tasks': 0,
                
                # Graph metrics
                'average_in_degree': 0.0,
                'average_out_degree': 0.0,
                'max_in_degree': 0,
                'max_out_degree': 0,
                
                # Performance metrics
                'total_analyses': self.performance_metrics['total_analyses'],
                'average_analysis_time': self.performance_metrics['average_analysis_time'],
                'cache_hit_rate': self.performance_metrics.get('cache_hit_rate', 0.0)
            }
            
            # Calculate dependency type distribution
            for dep in self.dependency_graph.dependencies.values():
                dep_type = dep.dependency_type.value
                stats['dependency_types'][dep_type] = stats['dependency_types'].get(dep_type, 0) + 1
                
                strength = dep.strength.value
                stats['dependency_strengths'][strength] = stats['dependency_strengths'].get(strength, 0) + 1
            
            # Calculate task properties
            for task_info in self.dependency_graph.tasks.values():
                if task_info.get('parallelizable', True):
                    stats['parallelizable_tasks'] += 1
                else:
                    stats['sequential_tasks'] += 1
            
            # Calculate degree statistics
            if self.nx_graph.nodes():
                in_degrees = [self.nx_graph.in_degree(node) for node in self.nx_graph.nodes()]
                out_degrees = [self.nx_graph.out_degree(node) for node in self.nx_graph.nodes()]
                
                stats['average_in_degree'] = sum(in_degrees) / len(in_degrees)
                stats['average_out_degree'] = sum(out_degrees) / len(out_degrees)
                stats['max_in_degree'] = max(in_degrees)
                stats['max_out_degree'] = max(out_degrees)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get dependency statistics: {e}")
            return {}
    
    # Internal methods
    def _update_graph_metrics(self):
        """Update graph metrics"""
        try:
            self.dependency_graph.node_count = len(self.dependency_graph.tasks)
            self.dependency_graph.edge_count = len(self.dependency_graph.dependencies)
            
            # Calculate density
            n = self.dependency_graph.node_count
            if n > 1:
                max_edges = n * (n - 1)
                self.dependency_graph.density = self.dependency_graph.edge_count / max_edges
            else:
                self.dependency_graph.density = 0.0
            
            # Update modification time
            self.dependency_graph.last_modified = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update graph metrics: {e}")
    
    def _check_for_cycles(self):
        """Check for cycles in the dependency graph"""
        try:
            cycles = self.detect_circular_dependencies()
            self.dependency_graph.has_cycles = cycles
            self.dependency_graph.is_acyclic = len(cycles) == 0
            
        except Exception as e:
            logger.error(f"Failed to check for cycles: {e}")
    
    def _clear_analysis_cache(self):
        """Clear analysis cache when graph changes"""
        self.analysis_cache.clear()
        self.path_cache.clear()
    
    def _perform_basic_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform basic dependency analysis"""
        try:
            # Calculate basic metrics
            result.complexity_score = self._calculate_complexity_score(tasks)
            result.parallelization_ratio = self._calculate_parallelization_ratio(tasks)
            
            # Identify basic patterns
            clusters = self._identify_dependency_clusters(tasks)
            result.dependency_clusters = clusters
            
        except Exception as e:
            logger.error(f"Failed to perform basic analysis: {e}")
    
    def _perform_critical_path_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform critical path analysis"""
        try:
            # Find critical paths
            critical_paths = self.find_critical_path()
            result.critical_paths = critical_paths
            
            # Calculate potential speedup
            if critical_paths:
                longest_path = critical_paths[0]
                total_sequential_time = longest_path.total_duration
                
                # Estimate parallel time
                parallelizable_groups = self.find_parallelizable_groups()
                if parallelizable_groups:
                    # Simplified speedup calculation
                    max_parallel_time = max(
                        sum(self.dependency_graph.tasks.get(task, {}).get('estimated_duration', timedelta(seconds=1))
                            for task in group)
                        for group in parallelizable_groups
                    )
                    
                    if max_parallel_time.total_seconds() > 0:
                        result.potential_speedup = total_sequential_time.total_seconds() / max_parallel_time.total_seconds()
            
        except Exception as e:
            logger.error(f"Failed to perform critical path analysis: {e}")
    
    def _perform_bottleneck_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform bottleneck analysis"""
        try:
            bottlenecks = self.identify_bottlenecks()
            result.bottlenecks = bottlenecks
            
            # Add bottleneck-specific recommendations
            if bottlenecks:
                result.recommended_actions.extend([
                    f"Optimize bottleneck task: {task}" for task in bottlenecks[:3]
                ])
            
        except Exception as e:
            logger.error(f"Failed to perform bottleneck analysis: {e}")
    
    def _perform_parallelization_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform parallelization analysis"""
        try:
            parallelizable_groups = self.find_parallelizable_groups()
            
            # Create clusters for parallelizable groups
            for i, group in enumerate(parallelizable_groups):
                cluster = DependencyCluster(
                    name=f"Parallel Group {i + 1}",
                    description=f"Group of {len(group)} parallelizable tasks",
                    task_ids=group,
                    cluster_type="parallel",
                    parallelization_potential=1.0
                )
                result.dependency_clusters.append(cluster)
            
            # Add parallelization recommendations
            if parallelizable_groups:
                total_parallel_tasks = sum(len(group) for group in parallelizable_groups)
                result.optimization_opportunities.append(
                    f"Execute {total_parallel_tasks} tasks in parallel across {len(parallelizable_groups)} groups"
                )
            
        except Exception as e:
            logger.error(f"Failed to perform parallelization analysis: {e}")
    
    def _perform_optimization_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform optimization analysis"""
        try:
            # Get optimization suggestions
            suggestions = self.suggest_optimizations()
            result.optimization_opportunities.extend(suggestions)
            
            # Analyze optimization potential
            result.recommended_actions.extend([
                "Consider task consolidation for small tasks",
                "Implement resource pooling for shared resources",
                "Add caching for repeated computations"
            ])
            
        except Exception as e:
            logger.error(f"Failed to perform optimization analysis: {e}")
    
    def _perform_circular_dependency_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform circular dependency analysis"""
        try:
            cycles = self.detect_circular_dependencies()
            
            if cycles:
                result.recommended_actions.extend([
                    f"Break circular dependency in tasks: {', '.join(cycle[:3])}" 
                    for cycle in cycles[:3]
                ])
            
        except Exception as e:
            logger.error(f"Failed to perform circular dependency analysis: {e}")
    
    def _perform_transitive_analysis(self, result: AnalysisResult, tasks: List[str]):
        """Perform transitive dependency analysis"""
        try:
            # Find transitive dependencies that could be optimized
            transitive_deps = self._find_transitive_dependencies(tasks)
            
            if transitive_deps:
                result.optimization_opportunities.append(
                    f"Optimize {len(transitive_deps)} transitive dependencies"
                )
            
        except Exception as e:
            logger.error(f"Failed to perform transitive analysis: {e}")
    
    def _calculate_complexity_score(self, tasks: List[str]) -> float:
        """Calculate complexity score for the dependency graph"""
        try:
            if not tasks:
                return 0.0
            
            # Factors contributing to complexity
            task_count = len(tasks)
            dependency_count = len(self.dependency_graph.dependencies)
            
            # Normalize by task count
            if task_count == 0:
                return 0.0
            
            dependency_ratio = dependency_count / task_count
            density = self.dependency_graph.density
            
            # Complexity increases with more dependencies and higher density
            complexity = (dependency_ratio * 0.6 + density * 0.4) * min(task_count / 10, 1.0)
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate complexity score: {e}")
            return 0.0
    
    def _calculate_parallelization_ratio(self, tasks: List[str]) -> float:
        """Calculate parallelization ratio"""
        try:
            if not tasks:
                return 0.0
            
            parallelizable_groups = self.find_parallelizable_groups()
            parallel_tasks = sum(len(group) for group in parallelizable_groups)
            
            return parallel_tasks / len(tasks)
            
        except Exception as e:
            logger.error(f"Failed to calculate parallelization ratio: {e}")
            return 0.0
    
    def _identify_dependency_clusters(self, tasks: List[str]) -> List[DependencyCluster]:
        """Identify clusters of related dependencies"""
        clusters = []
        
        try:
            # Use community detection algorithms
            if len(tasks) > 3:
                # Convert to undirected graph for community detection
                undirected = self.nx_graph.to_undirected()
                
                # Simple clustering based on connected components
                components = list(nx.connected_components(undirected))
                
                for i, component in enumerate(components):
                    if len(component) > 1:
                        cluster = DependencyCluster(
                            name=f"Cluster {i + 1}",
                            description=f"Connected component with {len(component)} tasks",
                            task_ids=list(component),
                            cluster_type="connected"
                        )
                        clusters.append(cluster)
            
        except Exception as e:
            logger.error(f"Failed to identify dependency clusters: {e}")
        
        return clusters
    
    def _find_longest_path(self, start_task: str, end_task: str) -> Optional[DependencyPath]:
        """Find the longest path between two tasks"""
        try:
            # Use modified Dijkstra's algorithm for longest path
            # (negate weights and find shortest path)
            
            # Create a copy of the graph with negated weights
            graph_copy = self.nx_graph.copy()
            for u, v, data in graph_copy.edges(data=True):
                weight = data.get('weight', 1)
                graph_copy[u][v]['weight'] = -weight
            
            try:
                # Find shortest path in negated graph (= longest path in original)
                path_nodes = nx.shortest_path(graph_copy, start_task, end_task, weight='weight')
                
                # Calculate total duration
                total_duration = timedelta(0)
                for task in path_nodes:
                    task_info = self.dependency_graph.tasks.get(task, {})
                    duration = task_info.get('estimated_duration', timedelta(seconds=1))
                    total_duration += duration
                
                # Create dependency path
                dependency_path = DependencyPath(
                    name=f"Path from {start_task} to {end_task}",
                    description=f"Dependency path with {len(path_nodes)} tasks",
                    task_sequence=path_nodes,
                    total_duration=total_duration
                )
                
                return dependency_path
                
            except nx.NetworkXNoPath:
                return None
            
        except Exception as e:
            logger.error(f"Failed to find longest path: {e}")
            return None
    
    def _detect_resource_conflicts(self) -> List[Dict[str, Any]]:
        """Detect resource conflicts between tasks"""
        conflicts = []
        
        try:
            # Group tasks by resource requirements
            resource_tasks = defaultdict(list)
            
            for task_id, task_info in self.dependency_graph.tasks.items():
                resources = task_info.get('resource_requirements', {})
                for resource in resources:
                    resource_tasks[resource].append(task_id)
            
            # Find potential conflicts (tasks using same exclusive resources)
            for resource, task_list in resource_tasks.items():
                if len(task_list) > 1:
                    # Check if tasks can run in parallel
                    parallelizable_groups = self.find_parallelizable_groups()
                    for group in parallelizable_groups:
                        conflicting_tasks = [t for t in task_list if t in group]
                        if len(conflicting_tasks) > 1:
                            conflicts.append({
                                'resource': resource,
                                'conflicting_tasks': conflicting_tasks,
                                'conflict_type': 'parallel_resource_conflict'
                            })
            
        except Exception as e:
            logger.error(f"Failed to detect resource conflicts: {e}")
        
        return conflicts
    
    def _find_transitive_dependencies(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Find transitive dependencies that could be optimized"""
        transitive_deps = []
        
        try:
            # Find all paths of length > 2
            for task in tasks:
                # Get all reachable tasks
                reachable = nx.descendants(self.nx_graph, task)
                
                # Check for direct dependencies that could be transitive
                direct_successors = set(self.nx_graph.successors(task))
                
                for successor in direct_successors:
                    # Check if there's an alternative path
                    successor_reachable = nx.descendants(self.nx_graph, successor)
                    indirect_paths = reachable.intersection(successor_reachable)
                    
                    if indirect_paths:
                        transitive_deps.append({
                            'source': task,
                            'target': successor,
                            'alternative_paths': list(indirect_paths)
                        })
            
        except Exception as e:
            logger.error(f"Failed to find transitive dependencies: {e}")
        
        return transitive_deps
    
    def _update_performance_metrics(self, analysis_duration: timedelta):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_analyses'] += 1
            
            # Update average analysis time
            total_analyses = self.performance_metrics['total_analyses']
            current_avg = self.performance_metrics['average_analysis_time']
            
            new_avg_seconds = (
                (current_avg.total_seconds() * (total_analyses - 1) + analysis_duration.total_seconds()) 
                / total_analyses
            )
            self.performance_metrics['average_analysis_time'] = timedelta(seconds=new_avg_seconds)
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def analyze_dependencies(self, tasks, dependencies) -> AnalysisResult:
        """Analyze dependencies for given tasks and dependencies (test-compatible method)"""
        try:
            # Clear existing data
            self.dependency_graph.tasks.clear()
            self.dependency_graph.dependencies.clear()
            self.nx_graph.clear()
            
            # Add tasks to the analyzer
            for task in tasks:
                task_info = {
                    'name': task.name,
                    'description': task.description,
                    'estimated_duration': task.estimated_duration or timedelta(seconds=1),
                    'parallelizable': True,
                    'priority': 0
                }
                self.add_task(task.id, task_info)
            
            # Add dependencies to the analyzer
            for dep in dependencies:
                # Handle different dependency formats
                if hasattr(dep, 'source_task_id') and hasattr(dep, 'target_task_id'):
                    source = dep.source_task_id
                    target = dep.target_task_id
                    dep_type_raw = getattr(dep, 'dependency_type', 'finish_to_start')
                else:
                    # Handle TaskDependency format from tests
                    source = dep.source_task_id if hasattr(dep, 'source_task_id') else getattr(dep, 'dependent_task_id', '')
                    target = dep.target_task_id if hasattr(dep, 'target_task_id') else getattr(dep, 'dependency_task_id', '')
                    dep_type_raw = getattr(dep, 'dependency_type', 'finish_to_start')
                
                # Convert string dependency type to enum
                if isinstance(dep_type_raw, str):
                    dep_type = DependencyType.FINISH_TO_START  # default
                    for dt in DependencyType:
                        if dt.value == dep_type_raw:
                            dep_type = dt
                            break
                else:
                    dep_type = dep_type_raw
                
                if source and target:
                    self.add_dependency(source, target, dep_type)
            
            # Create analysis result
            result = AnalysisResult(
                analysis_type=AnalysisType.COMPREHENSIVE,
                analysis_name="Dependency Analysis"
            )
            
            result.total_tasks = len(tasks)
            result.total_dependencies = len(dependencies)
            
            # Check for circular dependencies
            try:
                cycles = list(nx.simple_cycles(self.nx_graph))
                result.has_circular_dependencies = len(cycles) > 0
                result.circular_dependency_chains = cycles
            except Exception as e:
                logger.warning(f"Failed to detect cycles: {e}")
                result.has_circular_dependencies = False
                result.circular_dependency_chains = []
            
            # Find critical paths
            try:
                if not result.has_circular_dependencies:
                    # Find longest path (critical path)
                    if self.nx_graph.nodes():
                        # Get topological order
                        topo_order = list(nx.topological_sort(self.nx_graph))
                        
                        # Calculate longest path
                        distances = {}
                        for node in topo_order:
                            distances[node] = 0
                            for pred in self.nx_graph.predecessors(node):
                                task_duration = self.dependency_graph.tasks.get(pred, {}).get('estimated_duration', timedelta(seconds=1))
                                distances[node] = max(distances[node], distances.get(pred, 0) + task_duration.total_seconds())
                        
                        # Create critical path
                        if distances:
                            max_node = max(distances, key=distances.get)
                            critical_path = DependencyPath(
                                name="Critical Path",
                                description="Longest execution path",
                                task_sequence=[max_node],  # Simplified for now
                                total_duration=timedelta(seconds=distances[max_node]),
                                critical_path=True
                            )
                            result.critical_paths = [critical_path]
            except Exception as e:
                logger.warning(f"Failed to calculate critical path: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            # Return a basic result on error
            return AnalysisResult(
                analysis_type=AnalysisType.BASIC,
                analysis_name="Failed Analysis",
                total_tasks=len(tasks) if tasks else 0,
                total_dependencies=len(dependencies) if dependencies else 0
            )