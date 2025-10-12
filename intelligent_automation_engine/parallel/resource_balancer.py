"""
Resource Balancer for Parallel Execution

This module provides comprehensive resource balancing capabilities for
optimizing resource allocation and load distribution in parallel execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import threading
import statistics
import math
from collections import defaultdict, deque
import concurrent.futures
import time

logger = logging.getLogger(__name__)


class BalancingStrategy(Enum):
    """Resource balancing strategies"""
    ROUND_ROBIN = "round_robin"  # Distribute tasks evenly
    LEAST_LOADED = "least_loaded"  # Assign to least loaded resource
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weight-based distribution
    RESOURCE_AWARE = "resource_aware"  # Consider resource requirements
    PERFORMANCE_BASED = "performance_based"  # Based on historical performance
    ADAPTIVE = "adaptive"  # Dynamically adapt strategy
    LOCALITY_AWARE = "locality_aware"  # Consider data locality
    COST_OPTIMIZED = "cost_optimized"  # Minimize resource costs


class LoadMetric(Enum):
    """Metrics for measuring resource load"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    TASK_COUNT = "task_count"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"
    CUSTOM = "custom"


class BalancingObjective(Enum):
    """Objectives for resource balancing"""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_FAIRNESS = "maximize_fairness"
    MINIMIZE_ENERGY = "minimize_energy"


@dataclass
class ResourceNode:
    """Represents a resource node in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Node identification
    name: str = ""
    description: str = ""
    node_type: str = "worker"
    
    # Resource capabilities
    cpu_cores: int = 1
    memory_gb: float = 1.0
    disk_gb: float = 10.0
    network_mbps: float = 100.0
    
    # Current resource usage
    cpu_usage: float = 0.0  # 0.0 to 1.0
    memory_usage: float = 0.0  # 0.0 to 1.0
    disk_usage: float = 0.0  # 0.0 to 1.0
    network_usage: float = 0.0  # 0.0 to 1.0
    
    # Node state
    is_active: bool = True
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    current_tasks: int = 0
    max_tasks: int = 10
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: timedelta = timedelta(0)
    
    # Load balancing
    weight: float = 1.0
    priority: int = 0
    cost_per_hour: float = 0.0
    
    # Node metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadMetrics:
    """Load metrics for a resource node"""
    node_id: str = ""
    
    # Basic metrics
    cpu_load: float = 0.0
    memory_load: float = 0.0
    disk_load: float = 0.0
    network_load: float = 0.0
    
    # Task metrics
    task_count: int = 0
    queue_length: int = 0
    
    # Performance metrics
    response_time: timedelta = timedelta(0)
    throughput: float = 0.0
    error_rate: float = 0.0
    
    # Composite metrics
    overall_load: float = 0.0
    load_score: float = 0.0
    health_score: float = 1.0
    
    # Measurement metadata
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_window: timedelta = timedelta(minutes=1)


@dataclass
class BalancingRule:
    """Rule for resource balancing decisions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Rule identification
    name: str = ""
    description: str = ""
    
    # Rule conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    
    # Rule actions
    strategy: BalancingStrategy = BalancingStrategy.LEAST_LOADED
    target_nodes: List[str] = field(default_factory=list)
    weight_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Rule properties
    priority: int = 0
    is_active: bool = True
    
    # Rule metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    application_count: int = 0


@dataclass
class BalancingDecision:
    """Represents a balancing decision"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Decision context
    task_id: Optional[str] = None
    source_node: Optional[str] = None
    target_node: str = ""
    
    # Decision rationale
    strategy_used: BalancingStrategy = BalancingStrategy.LEAST_LOADED
    load_metrics: Dict[str, LoadMetrics] = field(default_factory=dict)
    decision_factors: Dict[str, float] = field(default_factory=dict)
    
    # Decision outcome
    expected_improvement: float = 0.0
    actual_improvement: Optional[float] = None
    confidence: float = 0.0
    
    # Decision metadata
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: timedelta = timedelta(0)


@dataclass
class BalancingPlan:
    """Plan for resource balancing operations"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Plan identification
    name: str = ""
    description: str = ""
    
    # Plan content
    decisions: List[BalancingDecision] = field(default_factory=list)
    migrations: List[Dict[str, Any]] = field(default_factory=list)
    weight_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Plan execution
    execution_order: List[str] = field(default_factory=list)
    estimated_duration: timedelta = timedelta(0)
    estimated_improvement: float = 0.0
    
    # Plan metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    status: str = "pending"


class ResourceBalancer:
    """Comprehensive resource balancer for parallel execution optimization"""
    
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.LEAST_LOADED,
                 objective: BalancingObjective = BalancingObjective.MINIMIZE_LATENCY):
        self.strategy = strategy
        self.objective = objective
        
        # Core data structures
        self.nodes: Dict[str, ResourceNode] = {}
        self.load_metrics: Dict[str, LoadMetrics] = {}
        self.balancing_rules: Dict[str, BalancingRule] = {}
        
        # Balancing state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Decision tracking
        self.decisions: List[BalancingDecision] = []
        self.decision_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.balancing_stats = {
            'total_decisions': 0,
            'successful_balancing': 0,
            'failed_balancing': 0,
            'average_decision_time': timedelta(0),
            'total_improvement': 0.0
        }
        
        # Configuration
        self.config = {
            'monitoring_interval': timedelta(seconds=30),
            'load_threshold': 0.8,
            'imbalance_threshold': 0.3,
            'min_improvement': 0.05,
            'decision_timeout': timedelta(seconds=5)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Resource balancer initialized with {strategy.value} strategy")
    
    def add_node(self, node: ResourceNode) -> bool:
        """Add a resource node to the balancer"""
        try:
            with self.lock:
                if node.id in self.nodes:
                    logger.warning(f"Node already exists: {node.id}")
                    return False
                
                self.nodes[node.id] = node
                
                # Initialize load metrics
                self.load_metrics[node.id] = LoadMetrics(node_id=node.id)
                
                logger.debug(f"Added node: {node.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a resource node from the balancer"""
        try:
            with self.lock:
                if node_id not in self.nodes:
                    logger.warning(f"Node not found: {node_id}")
                    return False
                
                node = self.nodes[node_id]
                
                # Check if node has active tasks
                if node.current_tasks > 0:
                    logger.warning(f"Node has active tasks: {node_id}")
                    return False
                
                del self.nodes[node_id]
                if node_id in self.load_metrics:
                    del self.load_metrics[node_id]
                
                logger.debug(f"Removed node: {node_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove node: {e}")
            return False
    
    def update_node_load(self, node_id: str, metrics: LoadMetrics) -> bool:
        """Update load metrics for a node"""
        try:
            with self.lock:
                if node_id not in self.nodes:
                    logger.warning(f"Node not found: {node_id}")
                    return False
                
                metrics.node_id = node_id
                metrics.timestamp = datetime.now()
                
                # Calculate composite metrics
                self._calculate_composite_metrics(metrics)
                
                self.load_metrics[node_id] = metrics
                
                # Update node state
                node = self.nodes[node_id]
                node.cpu_usage = metrics.cpu_load
                node.memory_usage = metrics.memory_load
                node.disk_usage = metrics.disk_load
                node.network_usage = metrics.network_load
                node.current_tasks = metrics.task_count
                node.last_heartbeat = datetime.now()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update node load: {e}")
            return False
    
    def select_node(self, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select the best node for a task based on balancing strategy"""
        try:
            with self.lock:
                start_time = datetime.now()
                
                # Get available nodes
                available_nodes = [
                    node for node in self.nodes.values()
                    if node.is_active and node.is_healthy and node.current_tasks < node.max_tasks
                ]
                
                if not available_nodes:
                    return None
                
                # Apply balancing strategy
                selected_node = self._apply_balancing_strategy(available_nodes, task_requirements)
                
                if selected_node:
                    # Create decision record
                    decision = BalancingDecision(
                        target_node=selected_node.id,
                        strategy_used=self.strategy,
                        load_metrics=dict(self.load_metrics),
                        execution_time=datetime.now() - start_time
                    )
                    
                    self.decisions.append(decision)
                    self.decision_history.append(decision)
                    
                    # Update statistics
                    self.balancing_stats['total_decisions'] += 1
                    
                    logger.debug(f"Selected node: {selected_node.id}")
                    return selected_node.id
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to select node: {e}")
            return None
    
    def start_monitoring(self) -> bool:
        """Start load monitoring and automatic balancing"""
        try:
            with self.lock:
                if self.is_monitoring:
                    logger.warning("Monitoring is already running")
                    return False
                
                self.is_monitoring = True
                self.stop_event.clear()
                self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitor_thread.start()
                
                logger.info("Started resource balancing monitoring")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop load monitoring"""
        try:
            with self.lock:
                if not self.is_monitoring:
                    logger.warning("Monitoring is not running")
                    return False
                
                self.is_monitoring = False
                self.stop_event.set()
                
                if self.monitor_thread:
                    self.monitor_thread.join(timeout=10)
                
                logger.info("Stopped resource balancing monitoring")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def add_balancing_rule(self, rule: BalancingRule) -> bool:
        """Add a balancing rule"""
        try:
            with self.lock:
                self.balancing_rules[rule.id] = rule
                
                logger.debug(f"Added balancing rule: {rule.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add balancing rule: {e}")
            return False
    
    def create_balancing_plan(self, target_nodes: Optional[List[str]] = None) -> BalancingPlan:
        """Create a comprehensive balancing plan"""
        try:
            with self.lock:
                plan = BalancingPlan(
                    name=f"Balancing Plan {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description="Automated resource balancing plan"
                )
                
                # Analyze current load distribution
                load_analysis = self._analyze_load_distribution()
                
                # Identify imbalances
                imbalances = self._identify_imbalances(load_analysis)
                
                # Generate balancing decisions
                for imbalance in imbalances:
                    decision = self._generate_balancing_decision(imbalance)
                    if decision:
                        plan.decisions.append(decision)
                
                # Calculate execution order
                plan.execution_order = self._calculate_execution_order(plan.decisions)
                
                # Estimate plan impact
                plan.estimated_improvement = self._estimate_plan_improvement(plan)
                plan.estimated_duration = self._estimate_plan_duration(plan)
                
                return plan
                
        except Exception as e:
            logger.error(f"Failed to create balancing plan: {e}")
            return BalancingPlan()
    
    def execute_balancing_plan(self, plan: BalancingPlan) -> bool:
        """Execute a balancing plan"""
        try:
            with self.lock:
                plan.status = "executing"
                
                for decision_id in plan.execution_order:
                    decision = next((d for d in plan.decisions if d.id == decision_id), None)
                    if not decision:
                        continue
                    
                    # Execute balancing decision
                    success = self._execute_balancing_decision(decision)
                    if not success:
                        logger.warning(f"Failed to execute decision: {decision_id}")
                        plan.status = "partial"
                        return False
                
                plan.status = "completed"
                
                logger.info(f"Executed balancing plan: {plan.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to execute balancing plan: {e}")
            plan.status = "failed"
            return False
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across nodes"""
        try:
            with self.lock:
                distribution = {
                    'nodes': {},
                    'summary': {
                        'total_nodes': len(self.nodes),
                        'active_nodes': 0,
                        'average_load': 0.0,
                        'load_variance': 0.0,
                        'imbalance_score': 0.0
                    }
                }
                
                loads = []
                active_count = 0
                
                for node_id, node in self.nodes.items():
                    if not node.is_active:
                        continue
                    
                    active_count += 1
                    metrics = self.load_metrics.get(node_id, LoadMetrics())
                    
                    node_info = {
                        'node_id': node_id,
                        'name': node.name,
                        'load_score': metrics.load_score,
                        'overall_load': metrics.overall_load,
                        'cpu_load': metrics.cpu_load,
                        'memory_load': metrics.memory_load,
                        'task_count': metrics.task_count,
                        'health_score': metrics.health_score,
                        'is_healthy': node.is_healthy
                    }
                    
                    distribution['nodes'][node_id] = node_info
                    loads.append(metrics.overall_load)
                
                # Calculate summary statistics
                if loads:
                    distribution['summary']['active_nodes'] = active_count
                    distribution['summary']['average_load'] = statistics.mean(loads)
                    distribution['summary']['load_variance'] = statistics.variance(loads) if len(loads) > 1 else 0.0
                    distribution['summary']['imbalance_score'] = self._calculate_imbalance_score(loads)
                
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get load distribution: {e}")
            return {}
    
    def get_balancing_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for improving load balancing"""
        try:
            with self.lock:
                recommendations = []
                
                # Analyze load distribution
                distribution = self.get_load_distribution()
                
                if not distribution.get('nodes'):
                    return recommendations
                
                # Check for overloaded nodes
                for node_id, node_info in distribution['nodes'].items():
                    if node_info['overall_load'] > self.config['load_threshold']:
                        recommendations.append({
                            'type': 'scale_out',
                            'priority': 'high',
                            'node_id': node_id,
                            'description': f"Node {node_info['name']} is overloaded ({node_info['overall_load']:.2%})",
                            'suggested_action': 'Add more capacity or redistribute tasks'
                        })
                
                # Check for imbalance
                imbalance_score = distribution['summary']['imbalance_score']
                if imbalance_score > self.config['imbalance_threshold']:
                    recommendations.append({
                        'type': 'rebalance',
                        'priority': 'medium',
                        'description': f"Load imbalance detected (score: {imbalance_score:.2f})",
                        'suggested_action': 'Redistribute tasks across nodes'
                    })
                
                # Check for underutilized nodes
                avg_load = distribution['summary']['average_load']
                for node_id, node_info in distribution['nodes'].items():
                    if node_info['overall_load'] < avg_load * 0.5 and avg_load > 0.3:
                        recommendations.append({
                            'type': 'scale_in',
                            'priority': 'low',
                            'node_id': node_id,
                            'description': f"Node {node_info['name']} is underutilized ({node_info['overall_load']:.2%})",
                            'suggested_action': 'Consider consolidating tasks or reducing capacity'
                        })
                
                # Check for unhealthy nodes
                for node_id, node_info in distribution['nodes'].items():
                    if not node_info['is_healthy'] or node_info['health_score'] < 0.7:
                        recommendations.append({
                            'type': 'health_issue',
                            'priority': 'high',
                            'node_id': node_id,
                            'description': f"Node {node_info['name']} has health issues (score: {node_info['health_score']:.2f})",
                            'suggested_action': 'Investigate and resolve health issues'
                        })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Failed to get balancing recommendations: {e}")
            return []
    
    def get_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive balancing statistics"""
        try:
            with self.lock:
                stats = dict(self.balancing_stats)
                
                # Add current state
                stats.update({
                    'total_nodes': len(self.nodes),
                    'active_nodes': sum(1 for n in self.nodes.values() if n.is_active),
                    'healthy_nodes': sum(1 for n in self.nodes.values() if n.is_healthy),
                    'is_monitoring': self.is_monitoring,
                    'strategy': self.strategy.value,
                    'objective': self.objective.value
                })
                
                # Add recent decision statistics
                recent_decisions = [d for d in self.decision_history if 
                                 datetime.now() - d.timestamp < timedelta(hours=1)]
                
                if recent_decisions:
                    stats['recent_decisions'] = len(recent_decisions)
                    stats['recent_avg_decision_time'] = sum(
                        (d.execution_time for d in recent_decisions), timedelta(0)
                    ) / len(recent_decisions)
                
                # Add load distribution summary
                distribution = self.get_load_distribution()
                if distribution.get('summary'):
                    stats.update(distribution['summary'])
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get balancing statistics: {e}")
            return {}
    
    # Internal methods
    def _monitoring_loop(self):
        """Main monitoring loop for automatic balancing"""
        try:
            while self.is_monitoring and not self.stop_event.is_set():
                try:
                    # Collect load metrics
                    self._collect_load_metrics()
                    
                    # Check for balancing triggers
                    if self._should_trigger_balancing():
                        # Create and execute balancing plan
                        plan = self.create_balancing_plan()
                        if plan.decisions:
                            self.execute_balancing_plan(plan)
                    
                    # Apply balancing rules
                    self._apply_balancing_rules()
                    
                    # Sleep until next monitoring cycle
                    self.stop_event.wait(self.config['monitoring_interval'].total_seconds())
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Longer sleep on error
            
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
        finally:
            logger.debug("Monitoring loop ended")
    
    def _apply_balancing_strategy(self, available_nodes: List[ResourceNode], 
                                task_requirements: Optional[Dict[str, Any]] = None) -> Optional[ResourceNode]:
        """Apply the configured balancing strategy"""
        try:
            if self.strategy == BalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_nodes)
            
            elif self.strategy == BalancingStrategy.LEAST_LOADED:
                return self._least_loaded_selection(available_nodes)
            
            elif self.strategy == BalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_nodes)
            
            elif self.strategy == BalancingStrategy.RESOURCE_AWARE:
                return self._resource_aware_selection(available_nodes, task_requirements)
            
            elif self.strategy == BalancingStrategy.PERFORMANCE_BASED:
                return self._performance_based_selection(available_nodes)
            
            elif self.strategy == BalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_nodes, task_requirements)
            
            else:
                # Default to least loaded
                return self._least_loaded_selection(available_nodes)
            
        except Exception as e:
            logger.error(f"Failed to apply balancing strategy: {e}")
            return available_nodes[0] if available_nodes else None
    
    def _round_robin_selection(self, nodes: List[ResourceNode]) -> Optional[ResourceNode]:
        """Round-robin node selection"""
        if not nodes:
            return None
        
        # Simple round-robin based on total decisions
        index = self.balancing_stats['total_decisions'] % len(nodes)
        return nodes[index]
    
    def _least_loaded_selection(self, nodes: List[ResourceNode]) -> Optional[ResourceNode]:
        """Select the least loaded node"""
        if not nodes:
            return None
        
        min_load = float('inf')
        selected_node = None
        
        for node in nodes:
            metrics = self.load_metrics.get(node.id, LoadMetrics())
            load = metrics.overall_load
            
            if load < min_load:
                min_load = load
                selected_node = node
        
        return selected_node or nodes[0]
    
    def _weighted_round_robin_selection(self, nodes: List[ResourceNode]) -> Optional[ResourceNode]:
        """Weighted round-robin selection based on node weights"""
        if not nodes:
            return None
        
        # Calculate cumulative weights
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return self._round_robin_selection(nodes)
        
        # Select based on weight distribution
        target = (self.balancing_stats['total_decisions'] % int(total_weight * 100)) / 100.0
        cumulative = 0.0
        
        for node in nodes:
            cumulative += node.weight
            if cumulative >= target:
                return node
        
        return nodes[-1]  # Fallback
    
    def _resource_aware_selection(self, nodes: List[ResourceNode], 
                                task_requirements: Optional[Dict[str, Any]] = None) -> Optional[ResourceNode]:
        """Resource-aware node selection"""
        if not nodes:
            return None
        
        if not task_requirements:
            return self._least_loaded_selection(nodes)
        
        best_score = -1
        selected_node = None
        
        for node in nodes:
            score = self._calculate_resource_fit_score(node, task_requirements)
            if score > best_score:
                best_score = score
                selected_node = node
        
        return selected_node or nodes[0]
    
    def _performance_based_selection(self, nodes: List[ResourceNode]) -> Optional[ResourceNode]:
        """Performance-based node selection"""
        if not nodes:
            return None
        
        best_score = -1
        selected_node = None
        
        for node in nodes:
            # Calculate performance score
            total_tasks = node.completed_tasks + node.failed_tasks
            success_rate = node.completed_tasks / total_tasks if total_tasks > 0 else 1.0
            
            # Consider response time (lower is better)
            response_penalty = node.average_response_time.total_seconds() / 60.0  # Normalize to minutes
            
            # Calculate composite score
            score = success_rate - response_penalty * 0.1
            
            if score > best_score:
                best_score = score
                selected_node = node
        
        return selected_node or nodes[0]
    
    def _adaptive_selection(self, nodes: List[ResourceNode], 
                          task_requirements: Optional[Dict[str, Any]] = None) -> Optional[ResourceNode]:
        """Adaptive selection that changes strategy based on conditions"""
        if not nodes:
            return None
        
        # Analyze current system state
        distribution = self.get_load_distribution()
        imbalance_score = distribution['summary'].get('imbalance_score', 0.0)
        avg_load = distribution['summary'].get('average_load', 0.0)
        
        # Choose strategy based on conditions
        if imbalance_score > self.config['imbalance_threshold']:
            # High imbalance - use least loaded
            return self._least_loaded_selection(nodes)
        elif avg_load > self.config['load_threshold']:
            # High overall load - use performance-based
            return self._performance_based_selection(nodes)
        elif task_requirements:
            # Normal conditions with requirements - use resource-aware
            return self._resource_aware_selection(nodes, task_requirements)
        else:
            # Normal conditions - use weighted round-robin
            return self._weighted_round_robin_selection(nodes)
    
    def _calculate_composite_metrics(self, metrics: LoadMetrics):
        """Calculate composite metrics from basic metrics"""
        try:
            # Calculate overall load as weighted average
            weights = {
                'cpu': 0.4,
                'memory': 0.3,
                'disk': 0.1,
                'network': 0.1,
                'tasks': 0.1
            }
            
            # Normalize task count (assuming max 100 tasks)
            task_load = min(metrics.task_count / 100.0, 1.0)
            
            metrics.overall_load = (
                weights['cpu'] * metrics.cpu_load +
                weights['memory'] * metrics.memory_load +
                weights['disk'] * metrics.disk_load +
                weights['network'] * metrics.network_load +
                weights['tasks'] * task_load
            )
            
            # Calculate load score (inverse of load for scoring)
            metrics.load_score = max(0.0, 1.0 - metrics.overall_load)
            
            # Calculate health score
            metrics.health_score = max(0.0, 1.0 - metrics.error_rate)
            
        except Exception as e:
            logger.error(f"Failed to calculate composite metrics: {e}")
    
    def _calculate_resource_fit_score(self, node: ResourceNode, 
                                    task_requirements: Dict[str, Any]) -> float:
        """Calculate how well a node fits task requirements"""
        try:
            score = 0.0
            
            # Check CPU requirement
            cpu_req = task_requirements.get('cpu_cores', 0)
            if cpu_req > 0:
                available_cpu = node.cpu_cores * (1.0 - node.cpu_usage)
                if available_cpu >= cpu_req:
                    score += 0.3 * (available_cpu / cpu_req)
                else:
                    return 0.0  # Cannot satisfy requirement
            
            # Check memory requirement
            memory_req = task_requirements.get('memory_gb', 0)
            if memory_req > 0:
                available_memory = node.memory_gb * (1.0 - node.memory_usage)
                if available_memory >= memory_req:
                    score += 0.3 * (available_memory / memory_req)
                else:
                    return 0.0  # Cannot satisfy requirement
            
            # Check disk requirement
            disk_req = task_requirements.get('disk_gb', 0)
            if disk_req > 0:
                available_disk = node.disk_gb * (1.0 - node.disk_usage)
                if available_disk >= disk_req:
                    score += 0.2 * (available_disk / disk_req)
                else:
                    return 0.0  # Cannot satisfy requirement
            
            # Check network requirement
            network_req = task_requirements.get('network_mbps', 0)
            if network_req > 0:
                available_network = node.network_mbps * (1.0 - node.network_usage)
                if available_network >= network_req:
                    score += 0.2 * (available_network / network_req)
                else:
                    return 0.0  # Cannot satisfy requirement
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate resource fit score: {e}")
            return 0.0
    
    def _collect_load_metrics(self):
        """Collect load metrics from all nodes"""
        try:
            # In a real implementation, this would collect actual metrics
            # For now, we'll simulate metric collection
            current_time = datetime.now()
            
            for node_id, node in self.nodes.items():
                if not node.is_active:
                    continue
                
                # Simulate metric collection
                metrics = LoadMetrics(
                    node_id=node_id,
                    cpu_load=node.cpu_usage,
                    memory_load=node.memory_usage,
                    disk_load=node.disk_usage,
                    network_load=node.network_usage,
                    task_count=node.current_tasks,
                    timestamp=current_time
                )
                
                self._calculate_composite_metrics(metrics)
                self.load_metrics[node_id] = metrics
            
        except Exception as e:
            logger.error(f"Failed to collect load metrics: {e}")
    
    def _should_trigger_balancing(self) -> bool:
        """Check if automatic balancing should be triggered"""
        try:
            distribution = self.get_load_distribution()
            
            # Check imbalance threshold
            imbalance_score = distribution['summary'].get('imbalance_score', 0.0)
            if imbalance_score > self.config['imbalance_threshold']:
                return True
            
            # Check for overloaded nodes
            for node_info in distribution['nodes'].values():
                if node_info['overall_load'] > self.config['load_threshold']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check balancing triggers: {e}")
            return False
    
    def _apply_balancing_rules(self):
        """Apply active balancing rules"""
        try:
            for rule in self.balancing_rules.values():
                if not rule.is_active:
                    continue
                
                # Check rule conditions
                if self._check_rule_conditions(rule):
                    self._execute_balancing_rule(rule)
            
        except Exception as e:
            logger.error(f"Failed to apply balancing rules: {e}")
    
    def _check_rule_conditions(self, rule: BalancingRule) -> bool:
        """Check if rule conditions are met"""
        try:
            # Simple condition checking - can be extended
            for condition, value in rule.conditions.items():
                if condition == 'min_imbalance':
                    distribution = self.get_load_distribution()
                    imbalance = distribution['summary'].get('imbalance_score', 0.0)
                    if imbalance < value:
                        return False
                
                elif condition == 'max_load':
                    distribution = self.get_load_distribution()
                    avg_load = distribution['summary'].get('average_load', 0.0)
                    if avg_load > value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check rule conditions: {e}")
            return False
    
    def _execute_balancing_rule(self, rule: BalancingRule):
        """Execute a balancing rule"""
        try:
            # Apply weight adjustments
            for node_id, adjustment in rule.weight_adjustments.items():
                if node_id in self.nodes:
                    self.nodes[node_id].weight = max(0.1, self.nodes[node_id].weight + adjustment)
            
            # Update rule statistics
            rule.last_applied = datetime.now()
            rule.application_count += 1
            
            logger.debug(f"Applied balancing rule: {rule.id}")
            
        except Exception as e:
            logger.error(f"Failed to execute balancing rule: {e}")
    
    def _analyze_load_distribution(self) -> Dict[str, Any]:
        """Analyze current load distribution"""
        try:
            analysis = {
                'nodes': {},
                'clusters': [],
                'bottlenecks': [],
                'recommendations': []
            }
            
            # Analyze each node
            for node_id, metrics in self.load_metrics.items():
                node_analysis = {
                    'load_level': 'normal',
                    'bottleneck_resources': [],
                    'efficiency_score': metrics.load_score
                }
                
                # Identify load level
                if metrics.overall_load > 0.8:
                    node_analysis['load_level'] = 'high'
                elif metrics.overall_load < 0.2:
                    node_analysis['load_level'] = 'low'
                
                # Identify bottleneck resources
                if metrics.cpu_load > 0.8:
                    node_analysis['bottleneck_resources'].append('cpu')
                if metrics.memory_load > 0.8:
                    node_analysis['bottleneck_resources'].append('memory')
                if metrics.disk_load > 0.8:
                    node_analysis['bottleneck_resources'].append('disk')
                if metrics.network_load > 0.8:
                    node_analysis['bottleneck_resources'].append('network')
                
                analysis['nodes'][node_id] = node_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze load distribution: {e}")
            return {}
    
    def _identify_imbalances(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify load imbalances from analysis"""
        try:
            imbalances = []
            
            # Find overloaded and underloaded nodes
            overloaded = []
            underloaded = []
            
            for node_id, node_analysis in analysis['nodes'].items():
                if node_analysis['load_level'] == 'high':
                    overloaded.append(node_id)
                elif node_analysis['load_level'] == 'low':
                    underloaded.append(node_id)
            
            # Create imbalance records
            for overloaded_node in overloaded:
                for underloaded_node in underloaded:
                    imbalances.append({
                        'type': 'load_imbalance',
                        'source_node': overloaded_node,
                        'target_node': underloaded_node,
                        'severity': 'medium'
                    })
            
            return imbalances
            
        except Exception as e:
            logger.error(f"Failed to identify imbalances: {e}")
            return []
    
    def _generate_balancing_decision(self, imbalance: Dict[str, Any]) -> Optional[BalancingDecision]:
        """Generate a balancing decision for an imbalance"""
        try:
            decision = BalancingDecision(
                source_node=imbalance.get('source_node'),
                target_node=imbalance['target_node'],
                strategy_used=self.strategy
            )
            
            # Calculate expected improvement
            if decision.source_node and decision.target_node:
                source_metrics = self.load_metrics.get(decision.source_node)
                target_metrics = self.load_metrics.get(decision.target_node)
                
                if source_metrics and target_metrics:
                    load_diff = source_metrics.overall_load - target_metrics.overall_load
                    decision.expected_improvement = load_diff * 0.5  # Assume 50% of difference
                    decision.confidence = min(0.9, load_diff)
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to generate balancing decision: {e}")
            return None
    
    def _calculate_execution_order(self, decisions: List[BalancingDecision]) -> List[str]:
        """Calculate optimal execution order for decisions"""
        try:
            # Simple ordering by expected improvement (highest first)
            sorted_decisions = sorted(decisions, key=lambda d: d.expected_improvement, reverse=True)
            return [d.id for d in sorted_decisions]
            
        except Exception as e:
            logger.error(f"Failed to calculate execution order: {e}")
            return [d.id for d in decisions]
    
    def _estimate_plan_improvement(self, plan: BalancingPlan) -> float:
        """Estimate the improvement from executing a plan"""
        try:
            total_improvement = sum(d.expected_improvement for d in plan.decisions)
            return total_improvement
            
        except Exception as e:
            logger.error(f"Failed to estimate plan improvement: {e}")
            return 0.0
    
    def _estimate_plan_duration(self, plan: BalancingPlan) -> timedelta:
        """Estimate the duration to execute a plan"""
        try:
            # Simple estimation: 1 second per decision
            return timedelta(seconds=len(plan.decisions))
            
        except Exception as e:
            logger.error(f"Failed to estimate plan duration: {e}")
            return timedelta(0)
    
    def _execute_balancing_decision(self, decision: BalancingDecision) -> bool:
        """Execute a single balancing decision"""
        try:
            # In a real implementation, this would perform actual load balancing
            # For now, we'll simulate the execution
            
            if decision.target_node in self.nodes:
                # Simulate load redistribution
                target_node = self.nodes[decision.target_node]
                target_metrics = self.load_metrics.get(decision.target_node)
                
                if target_metrics:
                    # Simulate load increase on target
                    target_metrics.overall_load = min(1.0, target_metrics.overall_load + 0.1)
                    target_node.current_tasks += 1
                
                # Update statistics
                self.balancing_stats['successful_balancing'] += 1
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute balancing decision: {e}")
            self.balancing_stats['failed_balancing'] += 1
            return False
    
    def _calculate_imbalance_score(self, loads: List[float]) -> float:
        """Calculate imbalance score from load values"""
        try:
            if len(loads) < 2:
                return 0.0
            
            # Use coefficient of variation as imbalance measure
            mean_load = statistics.mean(loads)
            if mean_load == 0:
                return 0.0
            
            std_dev = statistics.stdev(loads)
            return std_dev / mean_load
            
        except Exception as e:
            logger.error(f"Failed to calculate imbalance score: {e}")
            return 0.0