"""
Resource Optimizer for Automation Workflow Efficiency

This module provides comprehensive resource optimization capabilities including
CPU, memory, I/O, and network resource management for automation workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
import uuid
import logging
import threading
import psutil
import time
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    THREADS = "threads"
    FILE_HANDLES = "file_handles"
    NETWORK_CONNECTIONS = "network_connections"


class OptimizationStrategy(Enum):
    """Resource optimization strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class ResourcePriority(Enum):
    """Resource allocation priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ThrottleMode(Enum):
    """Resource throttling modes"""
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"
    ADAPTIVE = "adaptive"


@dataclass
class ResourceLimit:
    """Resource usage limit configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Limit identification
    resource_type: ResourceType = ResourceType.CPU
    name: str = ""
    description: str = ""
    
    # Limit values
    soft_limit: float = 0.0  # Warning threshold
    hard_limit: float = 0.0  # Maximum allowed
    target_limit: float = 0.0  # Optimal target
    
    # Limit scope
    applies_to: List[str] = field(default_factory=list)  # Component IDs
    component_types: List[str] = field(default_factory=list)
    
    # Limit behavior
    throttle_mode: ThrottleMode = ThrottleMode.SOFT
    auto_scale: bool = True
    priority: ResourcePriority = ResourcePriority.NORMAL
    
    # Timing
    evaluation_window: timedelta = timedelta(minutes=1)
    cooldown_period: timedelta = timedelta(minutes=5)
    
    # Configuration
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceUsage:
    """Resource usage measurement"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Usage identification
    resource_type: ResourceType = ResourceType.CPU
    component_id: str = ""
    component_type: str = ""
    
    # Usage data
    current_usage: float = 0.0
    peak_usage: float = 0.0
    average_usage: float = 0.0
    
    # Usage context
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_duration: timedelta = timedelta(seconds=1)
    
    # System context
    total_available: float = 0.0
    system_usage: float = 0.0
    
    # Metadata
    unit: str = ""
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Resource optimization action"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Action identification
    action_type: str = ""  # throttle, scale, migrate, cache, etc.
    target_resource: ResourceType = ResourceType.CPU
    target_component: str = ""
    
    # Action parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_impact: float = 0.0  # Expected resource savings
    confidence: float = 0.0  # Confidence in the action
    
    # Action timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Action status
    status: str = "pending"  # pending, executing, completed, failed
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    
    # Action impact
    actual_impact: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    
    # Rollback information
    rollback_possible: bool = True
    rollback_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePool:
    """Resource pool for allocation management"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Pool identification
    name: str = ""
    resource_type: ResourceType = ResourceType.CPU
    description: str = ""
    
    # Pool capacity
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    reserved_capacity: float = 0.0
    
    # Pool allocation
    allocations: Dict[str, float] = field(default_factory=dict)  # component_id -> amount
    allocation_history: List[Tuple[str, float, datetime]] = field(default_factory=list)
    
    # Pool configuration
    max_allocation_per_component: float = 0.0
    min_available_threshold: float = 0.1  # 10% minimum available
    auto_rebalance: bool = True
    
    # Pool statistics
    utilization_rate: float = 0.0
    peak_utilization: float = 0.0
    allocation_count: int = 0
    
    # Pool metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_rebalanced: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationPlan:
    """Comprehensive resource optimization plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Plan identification
    name: str = ""
    description: str = ""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # Plan scope
    target_components: List[str] = field(default_factory=list)
    target_resources: List[ResourceType] = field(default_factory=list)
    
    # Plan actions
    actions: List[OptimizationAction] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # action_id -> dependencies
    
    # Plan objectives
    target_savings: Dict[ResourceType, float] = field(default_factory=dict)
    priority_weights: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Plan execution
    execution_order: List[str] = field(default_factory=list)  # action IDs
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    # Plan status
    status: str = "draft"  # draft, approved, executing, completed, failed
    progress: float = 0.0
    
    # Plan results
    actual_savings: Dict[ResourceType, float] = field(default_factory=dict)
    success_rate: float = 0.0
    
    # Plan metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    estimated_duration: timedelta = timedelta(0)
    actual_duration: timedelta = timedelta(0)


@dataclass
class ResourceProfile:
    """Resource usage profile for a component"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Profile identification
    component_id: str = ""
    component_type: str = ""
    name: str = ""
    
    # Resource usage patterns
    usage_history: Dict[ResourceType, List[ResourceUsage]] = field(default_factory=dict)
    
    # Usage statistics
    average_usage: Dict[ResourceType, float] = field(default_factory=dict)
    peak_usage: Dict[ResourceType, float] = field(default_factory=dict)
    baseline_usage: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Usage patterns
    usage_trends: Dict[ResourceType, str] = field(default_factory=dict)  # increasing, decreasing, stable
    peak_times: List[datetime] = field(default_factory=list)
    low_usage_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    
    # Optimization potential
    optimization_score: float = 0.0
    bottleneck_resources: List[ResourceType] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    # Profile metadata
    sample_count: int = 0
    analysis_period: timedelta = timedelta(0)
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


class ResourceOptimizer:
    """Comprehensive resource optimizer for automation workflows"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        
        # Resource monitoring
        self.resource_usage: Dict[str, List[ResourceUsage]] = defaultdict(list)
        self.resource_profiles: Dict[str, ResourceProfile] = {}
        
        # Resource management
        self.resource_limits: Dict[str, ResourceLimit] = {}
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        
        # Optimization
        self.optimization_plans: Dict[str, OptimizationPlan] = {}
        self.optimization_history: List[OptimizationAction] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Statistics
        self.optimizer_stats = {
            'total_optimizations': 0,
            'total_resource_savings': defaultdict(float),
            'average_optimization_impact': 0.0,
            'successful_optimizations': 0
        }
        
        # Initialize system resource pools
        self._initialize_system_pools()
        
        logger.info(f"Resource optimizer initialized with {strategy.value} strategy")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("Resource monitoring already active")
                return
            
            self.monitoring_interval = interval
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"Started resource monitoring with {interval}s interval")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        try:
            if not self.monitoring_active:
                logger.warning("Resource monitoring not active")
                return
            
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10.0)
            
            logger.info("Stopped resource monitoring")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    def collect_resource_usage(self, component_id: str, component_type: str = ""):
        """Collect current resource usage for a component"""
        try:
            # Get system resource usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Create resource usage records
            usage_records = []
            
            # CPU usage
            cpu_record = ResourceUsage(
                resource_type=ResourceType.CPU,
                component_id=component_id,
                component_type=component_type,
                current_usage=cpu_usage,
                total_available=100.0,
                system_usage=cpu_usage,
                unit="percent"
            )
            usage_records.append(cpu_record)
            
            # Memory usage
            memory_record = ResourceUsage(
                resource_type=ResourceType.MEMORY,
                component_id=component_id,
                component_type=component_type,
                current_usage=memory_info.used,
                total_available=memory_info.total,
                system_usage=memory_info.percent,
                unit="bytes"
            )
            usage_records.append(memory_record)
            
            # Disk I/O
            if disk_io:
                disk_record = ResourceUsage(
                    resource_type=ResourceType.DISK_IO,
                    component_id=component_id,
                    component_type=component_type,
                    current_usage=disk_io.read_bytes + disk_io.write_bytes,
                    unit="bytes"
                )
                usage_records.append(disk_record)
            
            # Network I/O
            if network_io:
                network_record = ResourceUsage(
                    resource_type=ResourceType.NETWORK_IO,
                    component_id=component_id,
                    component_type=component_type,
                    current_usage=network_io.bytes_sent + network_io.bytes_recv,
                    unit="bytes"
                )
                usage_records.append(network_record)
            
            # Store usage records
            for record in usage_records:
                self.resource_usage[component_id].append(record)
                self._update_resource_profile(record)
                self._check_resource_limits(record)
            
            # Cleanup old records
            self._cleanup_old_usage_data()
            
        except Exception as e:
            logger.error(f"Failed to collect resource usage: {e}")
    
    def add_resource_limit(self, limit: ResourceLimit):
        """Add a resource limit"""
        try:
            self.resource_limits[limit.id] = limit
            logger.info(f"Added resource limit: {limit.name}")
            
        except Exception as e:
            logger.error(f"Failed to add resource limit: {e}")
    
    def create_optimization_plan(self, target_components: Optional[List[str]] = None,
                               target_resources: Optional[List[ResourceType]] = None) -> OptimizationPlan:
        """Create a resource optimization plan"""
        try:
            plan = OptimizationPlan(
                name=f"Optimization Plan {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Automated resource optimization plan",
                strategy=self.strategy,
                target_components=target_components or [],
                target_resources=target_resources or list(ResourceType)
            )
            
            # Analyze current resource usage
            optimization_opportunities = self._analyze_optimization_opportunities(
                target_components, target_resources
            )
            
            # Generate optimization actions
            actions = self._generate_optimization_actions(optimization_opportunities)
            plan.actions = actions
            
            # Calculate dependencies
            plan.dependencies = self._calculate_action_dependencies(actions)
            
            # Determine execution order
            plan.execution_order = self._determine_execution_order(actions, plan.dependencies)
            
            # Estimate impact
            plan.target_savings = self._estimate_optimization_impact(actions)
            
            # Store plan
            self.optimization_plans[plan.id] = plan
            
            logger.info(f"Created optimization plan with {len(actions)} actions")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create optimization plan: {e}")
            return OptimizationPlan()
    
    def execute_optimization_plan(self, plan_id: str) -> bool:
        """Execute an optimization plan"""
        try:
            if plan_id not in self.optimization_plans:
                logger.error(f"Optimization plan not found: {plan_id}")
                return False
            
            plan = self.optimization_plans[plan_id]
            plan.status = "executing"
            
            start_time = datetime.now()
            successful_actions = 0
            
            # Execute actions in order
            for action_id in plan.execution_order:
                action = next((a for a in plan.actions if a.id == action_id), None)
                if not action:
                    continue
                
                # Check dependencies
                if not self._check_action_dependencies(action, plan):
                    logger.warning(f"Dependencies not met for action: {action_id}")
                    continue
                
                # Execute action
                success = self._execute_optimization_action(action)
                if success:
                    successful_actions += 1
                
                # Update plan progress
                plan.progress = successful_actions / len(plan.actions)
            
            # Update plan status
            plan.status = "completed" if successful_actions > 0 else "failed"
            plan.success_rate = successful_actions / len(plan.actions)
            plan.actual_duration = datetime.now() - start_time
            
            # Calculate actual savings
            plan.actual_savings = self._calculate_actual_savings(plan)
            
            # Update statistics
            self.optimizer_stats['total_optimizations'] += 1
            if plan.success_rate > 0.5:
                self.optimizer_stats['successful_optimizations'] += 1
            
            logger.info(f"Executed optimization plan: {plan.success_rate:.1%} success rate")
            return plan.success_rate > 0.5
            
        except Exception as e:
            logger.error(f"Failed to execute optimization plan: {e}")
            return False
    
    def get_resource_profile(self, component_id: str) -> Optional[ResourceProfile]:
        """Get resource profile for a component"""
        return self.resource_profiles.get(component_id)
    
    def get_optimization_recommendations(self, component_id: Optional[str] = None) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        try:
            # Get profiles to analyze
            profiles_to_analyze = []
            if component_id:
                profile = self.resource_profiles.get(component_id)
                if profile:
                    profiles_to_analyze.append(profile)
            else:
                profiles_to_analyze = list(self.resource_profiles.values())
            
            # Analyze each profile
            for profile in profiles_to_analyze:
                # Check for high resource usage
                for resource_type, usage in profile.average_usage.items():
                    if resource_type == ResourceType.CPU and usage > 80.0:
                        recommendations.append(f"High CPU usage in {profile.name}: Consider optimization")
                    elif resource_type == ResourceType.MEMORY and usage > 0.8:
                        recommendations.append(f"High memory usage in {profile.name}: Consider caching optimization")
                
                # Check for bottlenecks
                if profile.bottleneck_resources:
                    resource_names = [r.value for r in profile.bottleneck_resources]
                    recommendations.append(f"Resource bottlenecks in {profile.name}: {', '.join(resource_names)}")
                
                # Add specific optimization opportunities
                recommendations.extend(profile.optimization_opportunities)
            
            # General recommendations
            if not recommendations:
                recommendations.append("No specific optimization opportunities identified")
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
        
        return recommendations
    
    def allocate_resource(self, resource_type: ResourceType, component_id: str, 
                         amount: float) -> bool:
        """Allocate resources from a pool"""
        try:
            if resource_type not in self.resource_pools:
                logger.error(f"Resource pool not found: {resource_type.value}")
                return False
            
            pool = self.resource_pools[resource_type]
            
            # Check if allocation is possible
            if pool.available_capacity < amount:
                logger.warning(f"Insufficient {resource_type.value} capacity")
                return False
            
            # Check per-component limit
            if pool.max_allocation_per_component > 0:
                current_allocation = pool.allocations.get(component_id, 0.0)
                if current_allocation + amount > pool.max_allocation_per_component:
                    logger.warning(f"Per-component allocation limit exceeded")
                    return False
            
            # Perform allocation
            pool.allocations[component_id] = pool.allocations.get(component_id, 0.0) + amount
            pool.available_capacity -= amount
            pool.allocation_count += 1
            
            # Record allocation history
            pool.allocation_history.append((component_id, amount, datetime.now()))
            
            # Update utilization
            pool.utilization_rate = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
            pool.peak_utilization = max(pool.peak_utilization, pool.utilization_rate)
            
            logger.debug(f"Allocated {amount} {resource_type.value} to {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate resource: {e}")
            return False
    
    def deallocate_resource(self, resource_type: ResourceType, component_id: str, 
                           amount: Optional[float] = None) -> bool:
        """Deallocate resources from a pool"""
        try:
            if resource_type not in self.resource_pools:
                logger.error(f"Resource pool not found: {resource_type.value}")
                return False
            
            pool = self.resource_pools[resource_type]
            
            if component_id not in pool.allocations:
                logger.warning(f"No allocation found for component: {component_id}")
                return False
            
            # Determine amount to deallocate
            current_allocation = pool.allocations[component_id]
            dealloc_amount = amount if amount is not None else current_allocation
            dealloc_amount = min(dealloc_amount, current_allocation)
            
            # Perform deallocation
            pool.allocations[component_id] -= dealloc_amount
            if pool.allocations[component_id] <= 0:
                del pool.allocations[component_id]
            
            pool.available_capacity += dealloc_amount
            
            # Update utilization
            pool.utilization_rate = (pool.total_capacity - pool.available_capacity) / pool.total_capacity
            
            logger.debug(f"Deallocated {dealloc_amount} {resource_type.value} from {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deallocate resource: {e}")
            return False
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        try:
            stats = {
                'total_components_monitored': len(self.resource_profiles),
                'total_resource_records': sum(len(records) for records in self.resource_usage.values()),
                'active_resource_limits': len([l for l in self.resource_limits.values() if l.enabled]),
                'optimization_plans': len(self.optimization_plans),
                'total_optimizations': self.optimizer_stats['total_optimizations'],
                'successful_optimizations': self.optimizer_stats['successful_optimizations'],
                'monitoring_active': self.monitoring_active,
                'resource_pools': {}
            }
            
            # Add resource pool statistics
            for resource_type, pool in self.resource_pools.items():
                stats['resource_pools'][resource_type.value] = {
                    'total_capacity': pool.total_capacity,
                    'available_capacity': pool.available_capacity,
                    'utilization_rate': pool.utilization_rate,
                    'peak_utilization': pool.peak_utilization,
                    'active_allocations': len(pool.allocations)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get resource statistics: {e}")
            return {}
    
    # Internal methods
    def _initialize_system_pools(self):
        """Initialize system resource pools"""
        try:
            # CPU pool
            cpu_count = psutil.cpu_count()
            self.resource_pools[ResourceType.CPU] = ResourcePool(
                name="System CPU Pool",
                resource_type=ResourceType.CPU,
                total_capacity=cpu_count * 100.0,  # 100% per core
                available_capacity=cpu_count * 100.0,
                max_allocation_per_component=50.0  # 50% max per component
            )
            
            # Memory pool
            memory_info = psutil.virtual_memory()
            self.resource_pools[ResourceType.MEMORY] = ResourcePool(
                name="System Memory Pool",
                resource_type=ResourceType.MEMORY,
                total_capacity=memory_info.total,
                available_capacity=memory_info.available,
                max_allocation_per_component=memory_info.total * 0.3  # 30% max per component
            )
            
            # Thread pool
            self.resource_pools[ResourceType.THREADS] = ResourcePool(
                name="System Thread Pool",
                resource_type=ResourceType.THREADS,
                total_capacity=1000.0,  # Reasonable thread limit
                available_capacity=1000.0,
                max_allocation_per_component=100.0  # 100 threads max per component
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize system pools: {e}")
    
    def _monitoring_worker(self):
        """Worker thread for resource monitoring"""
        try:
            while self.monitoring_active:
                # Collect system-wide resource usage
                self.collect_resource_usage("system", "system")
                
                # Check for optimization opportunities
                if len(self.resource_profiles) > 0:
                    self._check_optimization_triggers()
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"Monitoring worker failed: {e}")
        finally:
            self.monitoring_active = False
    
    def _update_resource_profile(self, usage: ResourceUsage):
        """Update resource profile with new usage data"""
        try:
            component_id = usage.component_id
            
            # Get or create profile
            if component_id not in self.resource_profiles:
                self.resource_profiles[component_id] = ResourceProfile(
                    component_id=component_id,
                    component_type=usage.component_type,
                    name=f"{usage.component_type}_{component_id[:8]}"
                )
            
            profile = self.resource_profiles[component_id]
            
            # Add usage to history
            if usage.resource_type not in profile.usage_history:
                profile.usage_history[usage.resource_type] = []
            
            profile.usage_history[usage.resource_type].append(usage)
            profile.sample_count += 1
            profile.last_updated = datetime.now()
            
            # Update statistics
            self._update_profile_statistics(profile, usage.resource_type)
            
            # Analyze optimization potential
            self._analyze_profile_optimization_potential(profile)
            
        except Exception as e:
            logger.error(f"Failed to update resource profile: {e}")
    
    def _update_profile_statistics(self, profile: ResourceProfile, resource_type: ResourceType):
        """Update profile statistics for a resource type"""
        try:
            usage_history = profile.usage_history.get(resource_type, [])
            if not usage_history:
                return
            
            # Calculate statistics
            current_values = [u.current_usage for u in usage_history[-20:]]  # Last 20 measurements
            
            if current_values:
                profile.average_usage[resource_type] = statistics.mean(current_values)
                profile.peak_usage[resource_type] = max(current_values)
                
                # Calculate baseline (minimum stable usage)
                if len(current_values) >= 10:
                    sorted_values = sorted(current_values)
                    profile.baseline_usage[resource_type] = statistics.mean(sorted_values[:5])  # Bottom 25%
                
                # Analyze trend
                if len(current_values) >= 5:
                    first_half = statistics.mean(current_values[:len(current_values)//2])
                    second_half = statistics.mean(current_values[len(current_values)//2:])
                    
                    if second_half > first_half * 1.1:
                        profile.usage_trends[resource_type] = "increasing"
                    elif second_half < first_half * 0.9:
                        profile.usage_trends[resource_type] = "decreasing"
                    else:
                        profile.usage_trends[resource_type] = "stable"
            
        except Exception as e:
            logger.error(f"Failed to update profile statistics: {e}")
    
    def _analyze_profile_optimization_potential(self, profile: ResourceProfile):
        """Analyze optimization potential for a profile"""
        try:
            optimization_score = 0.0
            bottlenecks = []
            opportunities = []
            
            # Check each resource type
            for resource_type, avg_usage in profile.average_usage.items():
                # Normalize usage for comparison
                if resource_type == ResourceType.CPU:
                    normalized_usage = avg_usage / 100.0
                elif resource_type == ResourceType.MEMORY:
                    total_memory = psutil.virtual_memory().total
                    normalized_usage = avg_usage / total_memory
                else:
                    normalized_usage = 0.5  # Default moderate usage
                
                # Check for high usage (potential bottleneck)
                if normalized_usage > 0.8:
                    bottlenecks.append(resource_type)
                    optimization_score += 0.3
                    opportunities.append(f"High {resource_type.value} usage - consider optimization")
                
                # Check for inefficient usage patterns
                baseline = profile.baseline_usage.get(resource_type, 0)
                peak = profile.peak_usage.get(resource_type, avg_usage)
                
                if peak > baseline * 3:  # High variance
                    optimization_score += 0.2
                    opportunities.append(f"Variable {resource_type.value} usage - consider load balancing")
                
                # Check trends
                trend = profile.usage_trends.get(resource_type, "stable")
                if trend == "increasing":
                    optimization_score += 0.1
                    opportunities.append(f"Increasing {resource_type.value} usage trend")
            
            # Update profile
            profile.optimization_score = min(optimization_score, 1.0)
            profile.bottleneck_resources = bottlenecks
            profile.optimization_opportunities = opportunities
            
        except Exception as e:
            logger.error(f"Failed to analyze optimization potential: {e}")
    
    def _check_resource_limits(self, usage: ResourceUsage):
        """Check usage against configured limits"""
        try:
            for limit in self.resource_limits.values():
                if not limit.enabled:
                    continue
                
                # Check if limit applies
                if limit.resource_type != usage.resource_type:
                    continue
                
                if limit.applies_to and usage.component_id not in limit.applies_to:
                    continue
                
                if limit.component_types and usage.component_type not in limit.component_types:
                    continue
                
                # Normalize usage for comparison
                normalized_usage = self._normalize_usage_value(usage)
                
                # Check limits
                if normalized_usage > limit.hard_limit:
                    self._handle_limit_violation(usage, limit, "hard")
                elif normalized_usage > limit.soft_limit:
                    self._handle_limit_violation(usage, limit, "soft")
            
        except Exception as e:
            logger.error(f"Failed to check resource limits: {e}")
    
    def _normalize_usage_value(self, usage: ResourceUsage) -> float:
        """Normalize usage value for limit comparison"""
        if usage.resource_type == ResourceType.CPU:
            return usage.current_usage  # Already in percentage
        elif usage.resource_type == ResourceType.MEMORY:
            if usage.total_available > 0:
                return (usage.current_usage / usage.total_available) * 100.0
            return 0.0
        else:
            return usage.current_usage
    
    def _handle_limit_violation(self, usage: ResourceUsage, limit: ResourceLimit, violation_type: str):
        """Handle resource limit violation"""
        try:
            logger.warning(f"{violation_type.upper()} limit violation: {usage.component_id} "
                          f"{usage.resource_type.value} usage {usage.current_usage}")
            
            # Apply throttling if configured
            if limit.throttle_mode != ThrottleMode.NONE:
                self._apply_throttling(usage, limit, violation_type)
            
            # Trigger optimization if auto-scale is enabled
            if limit.auto_scale:
                self._trigger_auto_optimization(usage, limit)
            
        except Exception as e:
            logger.error(f"Failed to handle limit violation: {e}")
    
    def _apply_throttling(self, usage: ResourceUsage, limit: ResourceLimit, violation_type: str):
        """Apply resource throttling"""
        try:
            # Create throttling action
            action = OptimizationAction(
                action_type="throttle",
                target_resource=usage.resource_type,
                target_component=usage.component_id,
                parameters={
                    'throttle_mode': limit.throttle_mode.value,
                    'target_usage': limit.target_limit,
                    'violation_type': violation_type
                }
            )
            
            # Execute throttling
            self._execute_optimization_action(action)
            
        except Exception as e:
            logger.error(f"Failed to apply throttling: {e}")
    
    def _trigger_auto_optimization(self, usage: ResourceUsage, limit: ResourceLimit):
        """Trigger automatic optimization"""
        try:
            # Create optimization plan for the component
            plan = self.create_optimization_plan(
                target_components=[usage.component_id],
                target_resources=[usage.resource_type]
            )
            
            # Execute plan if it has actions
            if plan.actions:
                self.execute_optimization_plan(plan.id)
            
        except Exception as e:
            logger.error(f"Failed to trigger auto optimization: {e}")
    
    def _analyze_optimization_opportunities(self, target_components: Optional[List[str]], 
                                          target_resources: Optional[List[ResourceType]]) -> List[Dict[str, Any]]:
        """Analyze optimization opportunities"""
        opportunities = []
        
        try:
            # Get profiles to analyze
            profiles_to_analyze = []
            if target_components:
                for comp_id in target_components:
                    if comp_id in self.resource_profiles:
                        profiles_to_analyze.append(self.resource_profiles[comp_id])
            else:
                profiles_to_analyze = list(self.resource_profiles.values())
            
            # Analyze each profile
            for profile in profiles_to_analyze:
                for resource_type, avg_usage in profile.average_usage.items():
                    # Skip if not in target resources
                    if target_resources and resource_type not in target_resources:
                        continue
                    
                    # Check for optimization opportunities
                    if profile.optimization_score > 0.3:
                        opportunities.append({
                            'type': 'high_usage_optimization',
                            'component_id': profile.component_id,
                            'resource_type': resource_type,
                            'current_usage': avg_usage,
                            'optimization_score': profile.optimization_score,
                            'opportunities': profile.optimization_opportunities
                        })
            
        except Exception as e:
            logger.error(f"Failed to analyze optimization opportunities: {e}")
        
        return opportunities
    
    def _generate_optimization_actions(self, opportunities: List[Dict[str, Any]]) -> List[OptimizationAction]:
        """Generate optimization actions from opportunities"""
        actions = []
        
        try:
            for opp in opportunities:
                if opp['type'] == 'high_usage_optimization':
                    # Generate appropriate actions based on resource type
                    resource_type = opp['resource_type']
                    component_id = opp['component_id']
                    
                    if resource_type == ResourceType.CPU:
                        # CPU optimization actions
                        actions.append(OptimizationAction(
                            action_type="cpu_throttle",
                            target_resource=resource_type,
                            target_component=component_id,
                            parameters={'target_usage': 70.0},
                            expected_impact=0.2,
                            confidence=0.8
                        ))
                    
                    elif resource_type == ResourceType.MEMORY:
                        # Memory optimization actions
                        actions.append(OptimizationAction(
                            action_type="memory_cache_optimize",
                            target_resource=resource_type,
                            target_component=component_id,
                            parameters={'cache_size_reduction': 0.3},
                            expected_impact=0.3,
                            confidence=0.7
                        ))
                    
                    # General optimization action
                    actions.append(OptimizationAction(
                        action_type="resource_rebalance",
                        target_resource=resource_type,
                        target_component=component_id,
                        parameters={'rebalance_factor': 0.8},
                        expected_impact=0.15,
                        confidence=0.6
                    ))
            
        except Exception as e:
            logger.error(f"Failed to generate optimization actions: {e}")
        
        return actions
    
    def _calculate_action_dependencies(self, actions: List[OptimizationAction]) -> Dict[str, List[str]]:
        """Calculate dependencies between optimization actions"""
        dependencies = {}
        
        try:
            # Simple dependency logic: same component actions should be sequential
            component_actions = defaultdict(list)
            for action in actions:
                component_actions[action.target_component].append(action.id)
            
            # Create dependencies within each component
            for comp_id, action_ids in component_actions.items():
                if len(action_ids) > 1:
                    for i in range(1, len(action_ids)):
                        dependencies[action_ids[i]] = [action_ids[i-1]]
            
        except Exception as e:
            logger.error(f"Failed to calculate action dependencies: {e}")
        
        return dependencies
    
    def _determine_execution_order(self, actions: List[OptimizationAction], 
                                  dependencies: Dict[str, List[str]]) -> List[str]:
        """Determine execution order for optimization actions"""
        try:
            # Simple topological sort
            order = []
            remaining_actions = {a.id: a for a in actions}
            
            while remaining_actions:
                # Find actions with no dependencies
                ready_actions = []
                for action_id in remaining_actions:
                    deps = dependencies.get(action_id, [])
                    if all(dep not in remaining_actions for dep in deps):
                        ready_actions.append(action_id)
                
                if not ready_actions:
                    # Break circular dependencies by picking highest confidence action
                    ready_actions = [max(remaining_actions.keys(), 
                                       key=lambda x: remaining_actions[x].confidence)]
                
                # Add ready actions to order
                for action_id in ready_actions:
                    order.append(action_id)
                    del remaining_actions[action_id]
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to determine execution order: {e}")
            return [a.id for a in actions]
    
    def _estimate_optimization_impact(self, actions: List[OptimizationAction]) -> Dict[ResourceType, float]:
        """Estimate optimization impact by resource type"""
        impact = defaultdict(float)
        
        try:
            for action in actions:
                impact[action.target_resource] += action.expected_impact
            
        except Exception as e:
            logger.error(f"Failed to estimate optimization impact: {e}")
        
        return dict(impact)
    
    def _check_action_dependencies(self, action: OptimizationAction, plan: OptimizationPlan) -> bool:
        """Check if action dependencies are satisfied"""
        try:
            dependencies = plan.dependencies.get(action.id, [])
            
            for dep_id in dependencies:
                dep_action = next((a for a in plan.actions if a.id == dep_id), None)
                if not dep_action or dep_action.status != "completed":
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check action dependencies: {e}")
            return False
    
    def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute a single optimization action"""
        try:
            action.status = "executing"
            action.executed_at = datetime.now()
            
            # Simulate action execution based on type
            success = True
            
            if action.action_type == "cpu_throttle":
                # Simulate CPU throttling
                logger.info(f"Throttling CPU for {action.target_component}")
                action.actual_impact = action.expected_impact * 0.8
                
            elif action.action_type == "memory_cache_optimize":
                # Simulate memory optimization
                logger.info(f"Optimizing memory cache for {action.target_component}")
                action.actual_impact = action.expected_impact * 0.9
                
            elif action.action_type == "resource_rebalance":
                # Simulate resource rebalancing
                logger.info(f"Rebalancing resources for {action.target_component}")
                action.actual_impact = action.expected_impact * 0.7
                
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                success = False
            
            # Update action status
            action.status = "completed" if success else "failed"
            action.completed_at = datetime.now()
            
            # Store in history
            self.optimization_history.append(action)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute optimization action: {e}")
            action.status = "failed"
            action.error_message = str(e)
            return False
    
    def _calculate_actual_savings(self, plan: OptimizationPlan) -> Dict[ResourceType, float]:
        """Calculate actual resource savings from executed plan"""
        savings = defaultdict(float)
        
        try:
            for action in plan.actions:
                if action.status == "completed":
                    savings[action.target_resource] += action.actual_impact
            
        except Exception as e:
            logger.error(f"Failed to calculate actual savings: {e}")
        
        return dict(savings)
    
    def _check_optimization_triggers(self):
        """Check for automatic optimization triggers"""
        try:
            # Check for high resource usage across profiles
            high_usage_components = []
            
            for profile in self.resource_profiles.values():
                if profile.optimization_score > 0.7:  # High optimization potential
                    high_usage_components.append(profile.component_id)
            
            # Trigger optimization if multiple components have high usage
            if len(high_usage_components) >= 3:
                logger.info(f"Triggering automatic optimization for {len(high_usage_components)} components")
                plan = self.create_optimization_plan(target_components=high_usage_components)
                if plan.actions:
                    self.execute_optimization_plan(plan.id)
            
        except Exception as e:
            logger.error(f"Failed to check optimization triggers: {e}")
    
    def _cleanup_old_usage_data(self):
        """Clean up old resource usage data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data
            
            for component_id in list(self.resource_usage.keys()):
                # Filter out old usage records
                self.resource_usage[component_id] = [
                    usage for usage in self.resource_usage[component_id]
                    if usage.timestamp >= cutoff_time
                ]
                
                # Remove empty entries
                if not self.resource_usage[component_id]:
                    del self.resource_usage[component_id]
            
        except Exception as e:
            logger.error(f"Failed to cleanup old usage data: {e}")