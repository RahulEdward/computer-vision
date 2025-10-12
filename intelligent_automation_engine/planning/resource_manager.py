"""
Resource Management System

This module manages computational resources, schedules tasks, and optimizes
resource allocation for efficient automation execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from datetime import datetime, timedelta
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from queue import PriorityQueue, Queue
import psutil
import time

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


class ResourceStatus(Enum):
    """Resource availability status"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    DEADLINE_AWARE = "deadline_aware"


class SchedulingPolicy(Enum):
    """Task scheduling policies"""
    FIFO = "fifo"                    # First In, First Out
    LIFO = "lifo"                    # Last In, First Out
    SJF = "sjf"                      # Shortest Job First
    PRIORITY = "priority"            # Priority-based
    ROUND_ROBIN = "round_robin"      # Round Robin
    DEADLINE = "deadline"            # Earliest Deadline First
    FAIR_SHARE = "fair_share"        # Fair Share
    ADAPTIVE = "adaptive"            # Adaptive scheduling


@dataclass
class ResourceSpec:
    """Specification for a resource requirement"""
    resource_type: ResourceType = ResourceType.CPU
    amount: float = 0.0
    unit: str = ""
    
    # Resource constraints
    min_amount: float = 0.0
    max_amount: float = float('inf')
    preferred_amount: float = 0.0
    
    # Quality requirements
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Temporal requirements
    duration: Optional[timedelta] = None
    deadline: Optional[datetime] = None
    priority: int = 0
    
    # Flexibility
    is_flexible: bool = True
    can_be_shared: bool = True
    can_be_preempted: bool = False
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Resource:
    """Represents a computational resource"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    resource_type: ResourceType = ResourceType.CPU
    
    # Capacity
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    allocated_capacity: float = 0.0
    reserved_capacity: float = 0.0
    
    # Status
    status: ResourceStatus = ResourceStatus.AVAILABLE
    health_score: float = 1.0
    performance_score: float = 1.0
    
    # Capabilities
    capabilities: Dict[str, Any] = field(default_factory=dict)
    supported_operations: List[str] = field(default_factory=list)
    
    # Constraints
    max_concurrent_tasks: int = 1
    current_task_count: int = 0
    allocation_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    utilization_history: List[Tuple[datetime, float]] = field(default_factory=list)
    performance_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Metadata
    location: str = ""
    cost_per_unit: float = 0.0
    maintenance_schedule: List[Tuple[datetime, datetime]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceAllocation:
    """Represents an allocation of resources to a task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    resource_id: str = ""
    
    # Allocation details
    allocated_amount: float = 0.0
    allocation_start: datetime = field(default_factory=datetime.now)
    allocation_end: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    # Status
    status: str = "pending"  # pending, active, completed, failed, cancelled
    priority: int = 0
    
    # Performance
    actual_usage: float = 0.0
    efficiency: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Constraints
    can_be_preempted: bool = False
    can_be_migrated: bool = False
    exclusive_access: bool = False
    
    # Metadata
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIRST_FIT
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TaskSchedule:
    """Represents a scheduled task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    
    # Scheduling
    scheduled_start: datetime = field(default_factory=datetime.now)
    estimated_duration: timedelta = timedelta(0)
    deadline: Optional[datetime] = None
    priority: int = 0
    
    # Resource requirements
    resource_requirements: List[ResourceSpec] = field(default_factory=list)
    allocated_resources: List[str] = field(default_factory=list)  # Resource IDs
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    dependents: List[str] = field(default_factory=list)    # Task IDs
    
    # Status
    status: str = "scheduled"  # scheduled, ready, running, completed, failed, cancelled
    progress: float = 0.0
    
    # Execution
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FIFO
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ResourcePool:
    """Represents a pool of similar resources"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    resource_type: ResourceType = ResourceType.CPU
    
    # Pool composition
    resource_ids: List[str] = field(default_factory=list)
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    
    # Pool policies
    allocation_strategy: AllocationStrategy = AllocationStrategy.LOAD_BALANCED
    load_balancing_enabled: bool = True
    auto_scaling_enabled: bool = False
    
    # Constraints
    max_pool_size: int = 100
    min_pool_size: int = 1
    scaling_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring
    utilization_target: float = 0.8
    current_utilization: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class ResourceManager:
    """Manages computational resources and task scheduling"""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.schedules: Dict[str, TaskSchedule] = {}
        
        # Scheduling queues
        self.scheduling_queue = PriorityQueue()
        self.ready_queue = Queue()
        self.running_tasks: Dict[str, TaskSchedule] = {}
        
        # Resource monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 5  # seconds
        self.monitoring_thread = None
        
        # Allocation strategies
        self.allocation_strategies = {
            AllocationStrategy.FIRST_FIT: self._allocate_first_fit,
            AllocationStrategy.BEST_FIT: self._allocate_best_fit,
            AllocationStrategy.WORST_FIT: self._allocate_worst_fit,
            AllocationStrategy.ROUND_ROBIN: self._allocate_round_robin,
            AllocationStrategy.PRIORITY_BASED: self._allocate_priority_based,
            AllocationStrategy.LOAD_BALANCED: self._allocate_load_balanced,
            AllocationStrategy.DEADLINE_AWARE: self._allocate_deadline_aware
        }
        
        # Scheduling policies
        self.scheduling_policies = {
            SchedulingPolicy.FIFO: self._schedule_fifo,
            SchedulingPolicy.LIFO: self._schedule_lifo,
            SchedulingPolicy.SJF: self._schedule_sjf,
            SchedulingPolicy.PRIORITY: self._schedule_priority,
            SchedulingPolicy.ROUND_ROBIN: self._schedule_round_robin,
            SchedulingPolicy.DEADLINE: self._schedule_deadline,
            SchedulingPolicy.FAIR_SHARE: self._schedule_fair_share,
            SchedulingPolicy.ADAPTIVE: self._schedule_adaptive
        }
        
        # System resources
        self._initialize_system_resources()
        
        # Statistics
        self.resource_stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_utilization': 0.0,
            'peak_utilization': 0.0,
            'total_tasks_scheduled': 0,
            'completed_tasks': 0
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def register_resource(self, resource: Resource) -> bool:
        """Register a new resource"""
        try:
            self.resources[resource.id] = resource
            logger.info(f"Resource registered: {resource.name} ({resource.resource_type.value})")
            return True
        except Exception as e:
            logger.error(f"Failed to register resource: {e}")
            return False
    
    def create_resource_pool(self, name: str, resource_type: ResourceType,
                           resource_ids: List[str], allocation_strategy: AllocationStrategy = AllocationStrategy.LOAD_BALANCED) -> str:
        """Create a resource pool"""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            resource_ids=resource_ids.copy(),
            allocation_strategy=allocation_strategy
        )
        
        # Calculate total capacity
        total_capacity = 0.0
        available_capacity = 0.0
        
        for resource_id in resource_ids:
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                total_capacity += resource.total_capacity
                available_capacity += resource.available_capacity
        
        pool.total_capacity = total_capacity
        pool.available_capacity = available_capacity
        
        self.resource_pools[pool.id] = pool
        logger.info(f"Resource pool created: {name} with {len(resource_ids)} resources")
        
        return pool.id
    
    def allocate_resources(self, task_id: str, resource_requirements: List[ResourceSpec],
                          strategy: AllocationStrategy = AllocationStrategy.BEST_FIT) -> Dict[str, Any]:
        """Allocate resources for a task"""
        try:
            allocator = self.allocation_strategies.get(strategy, self._allocate_best_fit)
            allocation_result = allocator(task_id, resource_requirements)
            
            if allocation_result['success']:
                # Create allocation records
                for allocation_info in allocation_result['allocations']:
                    allocation = ResourceAllocation(
                        task_id=task_id,
                        resource_id=allocation_info['resource_id'],
                        allocated_amount=allocation_info['amount'],
                        allocation_strategy=strategy,
                        priority=allocation_info.get('priority', 0)
                    )
                    
                    self.allocations[allocation.id] = allocation
                    
                    # Update resource availability
                    resource = self.resources[allocation_info['resource_id']]
                    resource.allocated_capacity += allocation_info['amount']
                    resource.available_capacity -= allocation_info['amount']
                    resource.current_task_count += 1
                    resource.last_updated = datetime.now()
                
                self.resource_stats['total_allocations'] += 1
                self.resource_stats['successful_allocations'] += 1
                
                logger.info(f"Resources allocated for task {task_id}")
            else:
                self.resource_stats['failed_allocations'] += 1
                logger.warning(f"Resource allocation failed for task {task_id}: {allocation_result.get('reason', 'Unknown')}")
            
            return allocation_result
            
        except Exception as e:
            self.resource_stats['failed_allocations'] += 1
            logger.error(f"Resource allocation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def schedule_task(self, task_id: str, resource_requirements: List[ResourceSpec],
                     estimated_duration: timedelta, deadline: Optional[datetime] = None,
                     priority: int = 0, dependencies: List[str] = None,
                     scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY) -> str:
        """Schedule a task for execution"""
        schedule = TaskSchedule(
            task_id=task_id,
            resource_requirements=resource_requirements,
            estimated_duration=estimated_duration,
            deadline=deadline,
            priority=priority,
            dependencies=dependencies or [],
            scheduling_policy=scheduling_policy
        )
        
        self.schedules[schedule.id] = schedule
        
        # Add to scheduling queue with priority
        priority_score = self._calculate_scheduling_priority(schedule)
        self.scheduling_queue.put((priority_score, schedule.id))
        
        self.resource_stats['total_tasks_scheduled'] += 1
        logger.info(f"Task {task_id} scheduled with priority {priority}")
        
        return schedule.id
    
    def release_resources(self, task_id: str) -> bool:
        """Release resources allocated to a task"""
        try:
            released_allocations = []
            
            for allocation_id, allocation in self.allocations.items():
                if allocation.task_id == task_id and allocation.status == 'active':
                    # Update resource availability
                    resource = self.resources[allocation.resource_id]
                    resource.allocated_capacity -= allocation.allocated_amount
                    resource.available_capacity += allocation.allocated_amount
                    resource.current_task_count -= 1
                    resource.last_updated = datetime.now()
                    
                    # Mark allocation as completed
                    allocation.status = 'completed'
                    allocation.allocation_end = datetime.now()
                    allocation.last_updated = datetime.now()
                    
                    released_allocations.append(allocation_id)
            
            logger.info(f"Released {len(released_allocations)} resource allocations for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to release resources for task {task_id}: {e}")
            return False
    
    def get_resource_utilization(self, resource_id: Optional[str] = None) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        if resource_id:
            resource = self.resources.get(resource_id)
            if not resource:
                return {'error': 'Resource not found'}
            
            utilization = (resource.allocated_capacity / resource.total_capacity) if resource.total_capacity > 0 else 0
            
            return {
                'resource_id': resource_id,
                'utilization': utilization,
                'allocated_capacity': resource.allocated_capacity,
                'available_capacity': resource.available_capacity,
                'total_capacity': resource.total_capacity,
                'current_tasks': resource.current_task_count,
                'status': resource.status.value
            }
        else:
            # Overall utilization
            total_capacity = sum(r.total_capacity for r in self.resources.values())
            total_allocated = sum(r.allocated_capacity for r in self.resources.values())
            
            overall_utilization = (total_allocated / total_capacity) if total_capacity > 0 else 0
            
            return {
                'overall_utilization': overall_utilization,
                'total_resources': len(self.resources),
                'active_allocations': len([a for a in self.allocations.values() if a.status == 'active']),
                'total_capacity': total_capacity,
                'allocated_capacity': total_allocated,
                'available_capacity': total_capacity - total_allocated
            }
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize current resource allocations"""
        optimization_results = {
            'optimizations_applied': 0,
            'efficiency_improvement': 0.0,
            'recommendations': []
        }
        
        # Analyze current allocations
        inefficient_allocations = self._identify_inefficient_allocations()
        
        # Apply optimizations
        for allocation_id in inefficient_allocations:
            if self._optimize_allocation(allocation_id):
                optimization_results['optimizations_applied'] += 1
        
        # Generate recommendations
        optimization_results['recommendations'] = self._generate_optimization_recommendations()
        
        return optimization_results
    
    def predict_resource_needs(self, time_horizon: timedelta) -> Dict[str, Any]:
        """Predict future resource needs"""
        # Analyze historical patterns
        historical_data = self._analyze_historical_usage()
        
        # Predict future demand
        predicted_demand = self._predict_demand(historical_data, time_horizon)
        
        # Identify potential bottlenecks
        bottlenecks = self._identify_potential_bottlenecks(predicted_demand)
        
        return {
            'time_horizon': time_horizon,
            'predicted_demand': predicted_demand,
            'potential_bottlenecks': bottlenecks,
            'recommendations': self._generate_capacity_recommendations(predicted_demand, bottlenecks)
        }
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if not self.monitoring_enabled or self.monitoring_thread:
            return
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
        logger.info("Resource monitoring stopped")
    
    # Allocation strategies
    def _allocate_first_fit(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """First-fit allocation strategy"""
        allocations = []
        
        for req in requirements:
            for resource in self.resources.values():
                if (resource.resource_type == req.resource_type and 
                    resource.available_capacity >= req.amount and
                    resource.status == ResourceStatus.AVAILABLE):
                    
                    allocations.append({
                        'resource_id': resource.id,
                        'amount': req.amount,
                        'requirement': req
                    })
                    break
            else:
                return {'success': False, 'reason': f'No available resource for {req.resource_type.value}'}
        
        return {'success': True, 'allocations': allocations}
    
    def _allocate_best_fit(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Best-fit allocation strategy"""
        allocations = []
        
        for req in requirements:
            best_resource = None
            best_fit_score = float('inf')
            
            for resource in self.resources.values():
                if (resource.resource_type == req.resource_type and 
                    resource.available_capacity >= req.amount and
                    resource.status == ResourceStatus.AVAILABLE):
                    
                    # Calculate fit score (prefer resources with capacity closest to requirement)
                    fit_score = resource.available_capacity - req.amount
                    if fit_score < best_fit_score:
                        best_fit_score = fit_score
                        best_resource = resource
            
            if best_resource:
                allocations.append({
                    'resource_id': best_resource.id,
                    'amount': req.amount,
                    'requirement': req
                })
            else:
                return {'success': False, 'reason': f'No available resource for {req.resource_type.value}'}
        
        return {'success': True, 'allocations': allocations}
    
    def _allocate_worst_fit(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Worst-fit allocation strategy"""
        allocations = []
        
        for req in requirements:
            worst_resource = None
            worst_fit_score = -1
            
            for resource in self.resources.values():
                if (resource.resource_type == req.resource_type and 
                    resource.available_capacity >= req.amount and
                    resource.status == ResourceStatus.AVAILABLE):
                    
                    # Calculate fit score (prefer resources with most available capacity)
                    fit_score = resource.available_capacity - req.amount
                    if fit_score > worst_fit_score:
                        worst_fit_score = fit_score
                        worst_resource = resource
            
            if worst_resource:
                allocations.append({
                    'resource_id': worst_resource.id,
                    'amount': req.amount,
                    'requirement': req
                })
            else:
                return {'success': False, 'reason': f'No available resource for {req.resource_type.value}'}
        
        return {'success': True, 'allocations': allocations}
    
    def _allocate_round_robin(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Round-robin allocation strategy"""
        # Simplified implementation
        return self._allocate_first_fit(task_id, requirements)
    
    def _allocate_priority_based(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Priority-based allocation strategy"""
        # Simplified implementation
        return self._allocate_best_fit(task_id, requirements)
    
    def _allocate_load_balanced(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Load-balanced allocation strategy"""
        allocations = []
        
        for req in requirements:
            best_resource = None
            lowest_utilization = float('inf')
            
            for resource in self.resources.values():
                if (resource.resource_type == req.resource_type and 
                    resource.available_capacity >= req.amount and
                    resource.status == ResourceStatus.AVAILABLE):
                    
                    # Calculate current utilization
                    utilization = resource.allocated_capacity / resource.total_capacity if resource.total_capacity > 0 else 0
                    
                    if utilization < lowest_utilization:
                        lowest_utilization = utilization
                        best_resource = resource
            
            if best_resource:
                allocations.append({
                    'resource_id': best_resource.id,
                    'amount': req.amount,
                    'requirement': req
                })
            else:
                return {'success': False, 'reason': f'No available resource for {req.resource_type.value}'}
        
        return {'success': True, 'allocations': allocations}
    
    def _allocate_deadline_aware(self, task_id: str, requirements: List[ResourceSpec]) -> Dict[str, Any]:
        """Deadline-aware allocation strategy"""
        # Simplified implementation
        return self._allocate_best_fit(task_id, requirements)
    
    # Scheduling policies
    def _schedule_fifo(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """FIFO scheduling policy"""
        return sorted(tasks, key=lambda t: t.created_at)
    
    def _schedule_lifo(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """LIFO scheduling policy"""
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def _schedule_sjf(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Shortest Job First scheduling policy"""
        return sorted(tasks, key=lambda t: t.estimated_duration)
    
    def _schedule_priority(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Priority-based scheduling policy"""
        return sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    def _schedule_round_robin(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Round-robin scheduling policy"""
        # Simplified implementation
        return tasks
    
    def _schedule_deadline(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Earliest Deadline First scheduling policy"""
        return sorted(tasks, key=lambda t: t.deadline or datetime.max)
    
    def _schedule_fair_share(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Fair share scheduling policy"""
        # Simplified implementation
        return tasks
    
    def _schedule_adaptive(self, tasks: List[TaskSchedule]) -> List[TaskSchedule]:
        """Adaptive scheduling policy"""
        # Simplified implementation
        return self._schedule_priority(tasks)
    
    # Helper methods
    def _initialize_system_resources(self) -> None:
        """Initialize system resources"""
        # CPU resource
        cpu_resource = Resource(
            name="System CPU",
            resource_type=ResourceType.CPU,
            total_capacity=psutil.cpu_count(),
            available_capacity=psutil.cpu_count(),
            capabilities={'cores': psutil.cpu_count(), 'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0}
        )
        self.register_resource(cpu_resource)
        
        # Memory resource
        memory_info = psutil.virtual_memory()
        memory_resource = Resource(
            name="System Memory",
            resource_type=ResourceType.MEMORY,
            total_capacity=memory_info.total / (1024**3),  # GB
            available_capacity=memory_info.available / (1024**3),  # GB
            capabilities={'total_gb': memory_info.total / (1024**3)}
        )
        self.register_resource(memory_resource)
        
        # Disk resource
        disk_info = psutil.disk_usage('/')
        disk_resource = Resource(
            name="System Disk",
            resource_type=ResourceType.DISK,
            total_capacity=disk_info.total / (1024**3),  # GB
            available_capacity=disk_info.free / (1024**3),  # GB
            capabilities={'total_gb': disk_info.total / (1024**3)}
        )
        self.register_resource(disk_resource)
    
    def _calculate_scheduling_priority(self, schedule: TaskSchedule) -> int:
        """Calculate scheduling priority for a task"""
        # Higher priority number = higher priority in queue (negative for min-heap)
        base_priority = -schedule.priority
        
        # Adjust for deadline urgency
        if schedule.deadline:
            time_to_deadline = (schedule.deadline - datetime.now()).total_seconds()
            if time_to_deadline > 0:
                urgency_factor = 1.0 / time_to_deadline
                base_priority -= int(urgency_factor * 1000)
        
        return base_priority
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                self._update_resource_metrics()
                self._update_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_resource_metrics(self) -> None:
        """Update resource utilization metrics"""
        current_time = datetime.now()
        
        for resource in self.resources.values():
            if resource.total_capacity > 0:
                utilization = resource.allocated_capacity / resource.total_capacity
                resource.utilization_history.append((current_time, utilization))
                
                # Keep only recent history
                cutoff_time = current_time - timedelta(hours=24)
                resource.utilization_history = [
                    (time, util) for time, util in resource.utilization_history
                    if time > cutoff_time
                ]
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        # Update overall utilization
        total_capacity = sum(r.total_capacity for r in self.resources.values())
        total_allocated = sum(r.allocated_capacity for r in self.resources.values())
        
        if total_capacity > 0:
            current_utilization = total_allocated / total_capacity
            self.resource_stats['average_utilization'] = current_utilization
            self.resource_stats['peak_utilization'] = max(
                self.resource_stats['peak_utilization'],
                current_utilization
            )
    
    def _identify_inefficient_allocations(self) -> List[str]:
        """Identify inefficient resource allocations"""
        inefficient = []
        
        for allocation_id, allocation in self.allocations.items():
            if allocation.status == 'active':
                # Check if allocation is underutilized
                if allocation.actual_usage < allocation.allocated_amount * 0.5:
                    inefficient.append(allocation_id)
        
        return inefficient
    
    def _optimize_allocation(self, allocation_id: str) -> bool:
        """Optimize a specific allocation"""
        # Simplified implementation
        return True
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check for underutilized resources
        for resource in self.resources.values():
            if resource.total_capacity > 0:
                utilization = resource.allocated_capacity / resource.total_capacity
                if utilization < 0.3:
                    recommendations.append(f"Resource {resource.name} is underutilized ({utilization:.1%})")
        
        return recommendations
    
    def _analyze_historical_usage(self) -> Dict[str, Any]:
        """Analyze historical resource usage patterns"""
        return {'patterns': []}  # Simplified
    
    def _predict_demand(self, historical_data: Dict[str, Any], time_horizon: timedelta) -> Dict[str, float]:
        """Predict future resource demand"""
        return {'cpu': 0.7, 'memory': 0.6}  # Simplified
    
    def _identify_potential_bottlenecks(self, predicted_demand: Dict[str, float]) -> List[str]:
        """Identify potential resource bottlenecks"""
        bottlenecks = []
        
        for resource_type, demand in predicted_demand.items():
            if demand > 0.9:
                bottlenecks.append(resource_type)
        
        return bottlenecks
    
    def _generate_capacity_recommendations(self, predicted_demand: Dict[str, float], bottlenecks: List[str]) -> List[str]:
        """Generate capacity planning recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            recommendations.append(f"Consider increasing {bottleneck} capacity")
        
        return recommendations
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource management statistics"""
        return {
            'total_resources': len(self.resources),
            'total_pools': len(self.resource_pools),
            'active_allocations': len([a for a in self.allocations.values() if a.status == 'active']),
            'scheduled_tasks': len(self.schedules),
            'running_tasks': len(self.running_tasks),
            'resource_stats': self.resource_stats.copy()
        }