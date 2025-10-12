"""
Task Scheduler for Parallel Execution

This module provides comprehensive task scheduling capabilities for
optimizing parallel task execution with various scheduling algorithms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import threading
import heapq
import queue
from collections import defaultdict, deque
import concurrent.futures
import time

logger = logging.getLogger(__name__)


class SchedulingAlgorithm(Enum):
    """Scheduling algorithms for task execution"""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    SHORTEST_JOB_FIRST = "shortest_job_first"  # SJF
    LONGEST_JOB_FIRST = "longest_job_first"  # LJF
    ROUND_ROBIN = "round_robin"  # Round Robin
    FAIR_SHARE = "fair_share"  # Fair Share scheduling
    DEADLINE_FIRST = "deadline_first"  # Earliest Deadline First
    CRITICAL_PATH = "critical_path"  # Critical Path scheduling
    RESOURCE_AWARE = "resource_aware"  # Resource-aware scheduling


class TaskState(Enum):
    """States of scheduled tasks"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class SchedulingPolicy:
    """Scheduling policies"""
    PREEMPTIVE = "preemptive"  # Tasks can be interrupted
    NON_PREEMPTIVE = "non_preemptive"  # Tasks run to completion
    COOPERATIVE = "cooperative"  # Tasks yield voluntarily


@dataclass
class Task:
    """Base task class for parallel execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    function: Optional[callable] = None
    estimated_duration: Optional[timedelta] = None
    
    def execute(self) -> Any:
        """Execute the task - to be implemented by subclasses"""
        if self.function:
            return self.function()
        pass


@dataclass
class TaskDependency:
    """Represents a dependency between tasks"""
    source_task_id: str
    target_task_id: str
    dependency_type: str = "sequential"  # sequential, parallel, conditional
    
    
@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Task identification
    name: str = ""
    description: str = ""
    task_function: Optional[Callable] = None
    
    # Task properties
    priority: int = 0  # Higher values = higher priority
    estimated_duration: timedelta = timedelta(seconds=1)
    actual_duration: timedelta = timedelta(0)
    
    # Task constraints
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Task execution
    state: TaskState = TaskState.PENDING
    assigned_worker: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Task results
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Task metadata
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerNode:
    """Represents a worker node for task execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Worker identification
    name: str = ""
    description: str = ""
    
    # Worker capabilities
    max_concurrent_tasks: int = 1
    current_task_count: int = 0
    available_resources: Dict[str, float] = field(default_factory=dict)
    
    # Worker state
    is_active: bool = True
    is_busy: bool = False
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Worker performance
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: timedelta = timedelta(0)
    
    # Worker metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingQueue:
    """Represents a scheduling queue"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Queue identification
    name: str = ""
    description: str = ""
    
    # Queue configuration
    algorithm: SchedulingAlgorithm = SchedulingAlgorithm.FIFO
    policy: SchedulingPolicy = SchedulingPolicy.NON_PREEMPTIVE
    max_size: int = 0  # 0 = unlimited
    
    # Queue state
    tasks: List[ScheduledTask] = field(default_factory=list)
    priority_queue: List[Tuple[int, str]] = field(default_factory=list)  # (priority, task_id)
    
    # Queue metrics
    total_enqueued: int = 0
    total_dequeued: int = 0
    average_wait_time: timedelta = timedelta(0)
    
    # Queue metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class SchedulingMetrics:
    """Metrics for scheduling performance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Latency metrics
    average_wait_time: timedelta = timedelta(0)
    average_execution_time: timedelta = timedelta(0)
    average_turnaround_time: timedelta = timedelta(0)
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    worker_utilization: float = 0.0
    
    # Quality metrics
    success_rate: float = 0.0
    deadline_miss_rate: float = 0.0
    
    # Efficiency metrics
    scheduling_overhead: timedelta = timedelta(0)
    context_switch_overhead: timedelta = timedelta(0)
    
    # Measurement metadata
    measurement_start: datetime = field(default_factory=datetime.now)
    measurement_duration: timedelta = timedelta(0)


class TaskScheduler:
    """Comprehensive task scheduler for parallel execution optimization"""
    
    def __init__(self, algorithm: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY,
                 policy: SchedulingPolicy = SchedulingPolicy.NON_PREEMPTIVE):
        self.algorithm = algorithm
        self.policy = policy
        
        # Core data structures
        self.tasks: Dict[str, ScheduledTask] = {}
        self.workers: Dict[str, WorkerNode] = {}
        self.queues: Dict[str, SchedulingQueue] = {}
        
        # Default queue
        self.default_queue = SchedulingQueue(
            name="Default Queue",
            description="Default scheduling queue",
            algorithm=algorithm,
            policy=policy
        )
        self.queues[self.default_queue.id] = self.default_queue
        
        # Scheduling state
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Execution management
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        
        # Performance tracking
        self.metrics = SchedulingMetrics()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Scheduling statistics
        self.scheduler_stats = {
            'total_tasks_scheduled': 0,
            'total_tasks_completed': 0,
            'total_scheduling_decisions': 0,
            'average_scheduling_time': timedelta(0)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Task scheduler initialized with {algorithm.value} algorithm")
    
    def add_worker(self, worker_id: str, max_concurrent_tasks: int = 1,
                   available_resources: Optional[Dict[str, float]] = None) -> bool:
        """Add a worker node to the scheduler"""
        try:
            with self.lock:
                if worker_id in self.workers:
                    logger.warning(f"Worker already exists: {worker_id}")
                    return False
                
                worker = WorkerNode(
                    id=worker_id,
                    name=f"Worker {worker_id}",
                    max_concurrent_tasks=max_concurrent_tasks,
                    available_resources=available_resources or {}
                )
                
                self.workers[worker_id] = worker
                
                logger.debug(f"Added worker: {worker_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add worker: {e}")
            return False
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker node from the scheduler"""
        try:
            with self.lock:
                if worker_id not in self.workers:
                    logger.warning(f"Worker not found: {worker_id}")
                    return False
                
                worker = self.workers[worker_id]
                
                # Check if worker has running tasks
                if worker.current_task_count > 0:
                    logger.warning(f"Worker has running tasks: {worker_id}")
                    return False
                
                del self.workers[worker_id]
                
                logger.debug(f"Removed worker: {worker_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove worker: {e}")
            return False
    
    def schedule_task(self, task: ScheduledTask, queue_id: Optional[str] = None) -> bool:
        """Schedule a task for execution"""
        try:
            with self.lock:
                # Use default queue if none specified
                if queue_id is None:
                    queue_id = self.default_queue.id
                
                if queue_id not in self.queues:
                    logger.error(f"Queue not found: {queue_id}")
                    return False
                
                queue = self.queues[queue_id]
                
                # Check queue capacity
                if queue.max_size > 0 and len(queue.tasks) >= queue.max_size:
                    logger.warning(f"Queue is full: {queue_id}")
                    return False
                
                # Add task to scheduler
                self.tasks[task.id] = task
                task.scheduled_at = datetime.now()
                task.state = TaskState.READY
                
                # Add to queue based on algorithm
                self._enqueue_task(queue, task)
                
                # Update statistics
                self.scheduler_stats['total_tasks_scheduled'] += 1
                queue.total_enqueued += 1
                queue.last_modified = datetime.now()
                
                logger.debug(f"Scheduled task: {task.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return False
    
    def start_scheduler(self, max_workers: int = 4) -> bool:
        """Start the task scheduler"""
        try:
            with self.lock:
                if self.is_running:
                    logger.warning("Scheduler is already running")
                    return False
                
                # Create thread pool executor
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                
                # Start scheduler thread
                self.is_running = True
                self.stop_event.clear()
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
                self.scheduler_thread.start()
                
                # Start metrics collection
                self.metrics.measurement_start = datetime.now()
                
                logger.info(f"Started task scheduler with {max_workers} workers")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False
    
    def stop_scheduler(self, wait: bool = True) -> bool:
        """Stop the task scheduler"""
        try:
            with self.lock:
                if not self.is_running:
                    logger.warning("Scheduler is not running")
                    return False
                
                # Signal stop
                self.is_running = False
                self.stop_event.set()
                
                # Wait for scheduler thread
                if wait and self.scheduler_thread:
                    self.scheduler_thread.join(timeout=10)
                
                # Shutdown executor
                if self.executor:
                    self.executor.shutdown(wait=wait)
                    self.executor = None
                
                # Cancel running tasks
                for future in self.running_tasks.values():
                    future.cancel()
                self.running_tasks.clear()
                
                # Update metrics
                self.metrics.measurement_duration = datetime.now() - self.metrics.measurement_start
                
                logger.info("Stopped task scheduler")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task"""
        try:
            with self.lock:
                if task_id not in self.tasks:
                    logger.warning(f"Task not found: {task_id}")
                    return False
                
                task = self.tasks[task_id]
                
                if task.state == TaskState.COMPLETED:
                    logger.warning(f"Task already completed: {task_id}")
                    return False
                
                # Cancel running task
                if task_id in self.running_tasks:
                    future = self.running_tasks[task_id]
                    future.cancel()
                    del self.running_tasks[task_id]
                
                # Remove from queues
                for queue in self.queues.values():
                    queue.tasks = [t for t in queue.tasks if t.id != task_id]
                    queue.priority_queue = [(p, tid) for p, tid in queue.priority_queue if tid != task_id]
                    heapq.heapify(queue.priority_queue)
                
                # Update task state
                task.state = TaskState.CANCELLED
                task.end_time = datetime.now()
                
                logger.debug(f"Cancelled task: {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a scheduled task"""
        try:
            with self.lock:
                if task_id not in self.tasks:
                    return {'error': 'Task not found'}
                
                task = self.tasks[task_id]
                
                status = {
                    'task_id': task_id,
                    'name': task.name,
                    'state': task.state.value,
                    'priority': task.priority,
                    'estimated_duration': task.estimated_duration,
                    'actual_duration': task.actual_duration,
                    'created_at': task.created_at,
                    'scheduled_at': task.scheduled_at,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'assigned_worker': task.assigned_worker,
                    'retry_count': task.retry_count,
                    'dependencies': task.dependencies
                }
                
                # Add queue information
                for queue_id, queue in self.queues.items():
                    if any(t.id == task_id for t in queue.tasks):
                        status['queue_id'] = queue_id
                        status['queue_position'] = next(
                            i for i, t in enumerate(queue.tasks) if t.id == task_id
                        )
                        break
                
                return status
                
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {'error': str(e)}
    
    def get_queue_status(self, queue_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a scheduling queue"""
        try:
            with self.lock:
                if queue_id is None:
                    queue_id = self.default_queue.id
                
                if queue_id not in self.queues:
                    return {'error': 'Queue not found'}
                
                queue = self.queues[queue_id]
                
                status = {
                    'queue_id': queue_id,
                    'name': queue.name,
                    'algorithm': queue.algorithm.value,
                    'policy': queue.policy.value,
                    'current_size': len(queue.tasks),
                    'max_size': queue.max_size,
                    'total_enqueued': queue.total_enqueued,
                    'total_dequeued': queue.total_dequeued,
                    'average_wait_time': queue.average_wait_time,
                    'created_at': queue.created_at,
                    'last_modified': queue.last_modified
                }
                
                # Add task breakdown by state
                task_states = defaultdict(int)
                for task in queue.tasks:
                    task_states[task.state.value] += 1
                status['task_states'] = dict(task_states)
                
                return status
                
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {'error': str(e)}
    
    def get_worker_status(self, worker_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of worker(s)"""
        try:
            with self.lock:
                if worker_id is not None:
                    # Get specific worker status
                    if worker_id not in self.workers:
                        return {'error': 'Worker not found'}
                    
                    worker = self.workers[worker_id]
                    
                    return {
                        'worker_id': worker_id,
                        'name': worker.name,
                        'is_active': worker.is_active,
                        'is_busy': worker.is_busy,
                        'max_concurrent_tasks': worker.max_concurrent_tasks,
                        'current_task_count': worker.current_task_count,
                        'available_resources': worker.available_resources,
                        'completed_tasks': worker.completed_tasks,
                        'failed_tasks': worker.failed_tasks,
                        'average_task_duration': worker.average_task_duration,
                        'last_heartbeat': worker.last_heartbeat,
                        'created_at': worker.created_at
                    }
                else:
                    # Get all workers status
                    workers_status = []
                    for wid in self.workers:
                        worker_status = self.get_worker_status(wid)
                        if 'error' not in worker_status:
                            workers_status.append(worker_status)
                    
                    return workers_status
                
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            return {'error': str(e)}
    
    def get_scheduling_metrics(self) -> SchedulingMetrics:
        """Get current scheduling metrics"""
        try:
            with self.lock:
                # Update current metrics
                self._update_metrics()
                return self.metrics
                
        except Exception as e:
            logger.error(f"Failed to get scheduling metrics: {e}")
            return SchedulingMetrics()
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        try:
            with self.lock:
                stats = dict(self.scheduler_stats)
                
                # Add current state
                stats.update({
                    'total_tasks': len(self.tasks),
                    'total_workers': len(self.workers),
                    'total_queues': len(self.queues),
                    'running_tasks': len(self.running_tasks),
                    'is_running': self.is_running
                })
                
                # Add task state breakdown
                task_states = defaultdict(int)
                for task in self.tasks.values():
                    task_states[task.state.value] += 1
                stats['task_states'] = dict(task_states)
                
                # Add worker utilization
                if self.workers:
                    active_workers = sum(1 for w in self.workers.values() if w.is_active)
                    busy_workers = sum(1 for w in self.workers.values() if w.is_busy)
                    stats['active_workers'] = active_workers
                    stats['busy_workers'] = busy_workers
                    stats['worker_utilization'] = busy_workers / active_workers if active_workers > 0 else 0.0
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get scheduler statistics: {e}")
            return {}
    
    # Internal methods
    def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Make scheduling decisions
                    self._make_scheduling_decisions()
                    
                    # Update worker heartbeats
                    self._update_worker_heartbeats()
                    
                    # Clean up completed tasks
                    self._cleanup_completed_tasks()
                    
                    # Update metrics
                    self._update_metrics()
                    
                    # Sleep briefly to avoid busy waiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(1)  # Longer sleep on error
            
        except Exception as e:
            logger.error(f"Scheduler loop failed: {e}")
        finally:
            logger.debug("Scheduler loop ended")
    
    def _make_scheduling_decisions(self):
        """Make scheduling decisions for ready tasks"""
        try:
            with self.lock:
                if not self.executor:
                    return
                
                # Find available workers
                available_workers = [
                    worker for worker in self.workers.values()
                    if worker.is_active and worker.current_task_count < worker.max_concurrent_tasks
                ]
                
                if not available_workers:
                    return
                
                # Process each queue
                for queue in self.queues.values():
                    if not queue.tasks:
                        continue
                    
                    # Get next task based on algorithm
                    task = self._dequeue_task(queue)
                    if not task:
                        continue
                    
                    # Check dependencies
                    if not self._check_dependencies(task):
                        # Re-queue task if dependencies not met
                        self._enqueue_task(queue, task)
                        continue
                    
                    # Find suitable worker
                    worker = self._find_suitable_worker(task, available_workers)
                    if not worker:
                        # Re-queue task if no suitable worker
                        self._enqueue_task(queue, task)
                        continue
                    
                    # Assign task to worker
                    self._assign_task_to_worker(task, worker)
                    
                    # Remove worker from available list
                    if worker.current_task_count >= worker.max_concurrent_tasks:
                        available_workers.remove(worker)
                    
                    # Update statistics
                    self.scheduler_stats['total_scheduling_decisions'] += 1
                    
                    if not available_workers:
                        break
                
        except Exception as e:
            logger.error(f"Failed to make scheduling decisions: {e}")
    
    def _enqueue_task(self, queue: SchedulingQueue, task: ScheduledTask):
        """Add task to queue based on scheduling algorithm"""
        try:
            if queue.algorithm == SchedulingAlgorithm.FIFO:
                queue.tasks.append(task)
            
            elif queue.algorithm == SchedulingAlgorithm.LIFO:
                queue.tasks.insert(0, task)
            
            elif queue.algorithm == SchedulingAlgorithm.PRIORITY:
                # Use priority queue
                heapq.heappush(queue.priority_queue, (-task.priority, task.id))
                queue.tasks.append(task)
            
            elif queue.algorithm == SchedulingAlgorithm.SHORTEST_JOB_FIRST:
                # Insert in sorted order by duration
                duration = task.estimated_duration.total_seconds()
                insert_pos = 0
                for i, t in enumerate(queue.tasks):
                    if t.estimated_duration.total_seconds() > duration:
                        insert_pos = i
                        break
                    insert_pos = i + 1
                queue.tasks.insert(insert_pos, task)
            
            elif queue.algorithm == SchedulingAlgorithm.LONGEST_JOB_FIRST:
                # Insert in reverse sorted order by duration
                duration = task.estimated_duration.total_seconds()
                insert_pos = 0
                for i, t in enumerate(queue.tasks):
                    if t.estimated_duration.total_seconds() < duration:
                        insert_pos = i
                        break
                    insert_pos = i + 1
                queue.tasks.insert(insert_pos, task)
            
            elif queue.algorithm == SchedulingAlgorithm.DEADLINE_FIRST:
                # Insert in sorted order by deadline
                if task.deadline:
                    insert_pos = 0
                    for i, t in enumerate(queue.tasks):
                        if t.deadline and t.deadline > task.deadline:
                            insert_pos = i
                            break
                        insert_pos = i + 1
                    queue.tasks.insert(insert_pos, task)
                else:
                    queue.tasks.append(task)
            
            else:
                # Default to FIFO
                queue.tasks.append(task)
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
    
    def _dequeue_task(self, queue: SchedulingQueue) -> Optional[ScheduledTask]:
        """Remove and return next task from queue"""
        try:
            if not queue.tasks:
                return None
            
            if queue.algorithm == SchedulingAlgorithm.PRIORITY:
                # Use priority queue
                if queue.priority_queue:
                    _, task_id = heapq.heappop(queue.priority_queue)
                    # Find and remove task from tasks list
                    for i, task in enumerate(queue.tasks):
                        if task.id == task_id:
                            return queue.tasks.pop(i)
                # Fallback to first task
                return queue.tasks.pop(0) if queue.tasks else None
            
            else:
                # For other algorithms, task order is maintained in tasks list
                return queue.tasks.pop(0)
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are satisfied"""
        try:
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    dep_task = self.tasks[dep_id]
                    if dep_task.state != TaskState.COMPLETED:
                        return False
                else:
                    # Dependency not found - assume it's external and satisfied
                    continue
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check dependencies: {e}")
            return False
    
    def _find_suitable_worker(self, task: ScheduledTask, available_workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Find a suitable worker for the task"""
        try:
            # Simple resource-aware assignment
            for worker in available_workers:
                # Check resource requirements
                can_handle = True
                for resource, required in task.resource_requirements.items():
                    available = worker.available_resources.get(resource, 0)
                    if available < required:
                        can_handle = False
                        break
                
                if can_handle:
                    return worker
            
            # If no worker can handle resource requirements, return first available
            return available_workers[0] if available_workers else None
            
        except Exception as e:
            logger.error(f"Failed to find suitable worker: {e}")
            return None
    
    def _assign_task_to_worker(self, task: ScheduledTask, worker: WorkerNode):
        """Assign a task to a worker"""
        try:
            # Update task state
            task.state = TaskState.RUNNING
            task.assigned_worker = worker.id
            task.start_time = datetime.now()
            
            # Update worker state
            worker.current_task_count += 1
            worker.is_busy = worker.current_task_count >= worker.max_concurrent_tasks
            
            # Reserve resources
            for resource, required in task.resource_requirements.items():
                if resource in worker.available_resources:
                    worker.available_resources[resource] -= required
            
            # Submit task for execution
            if self.executor and task.task_function:
                future = self.executor.submit(self._execute_task, task)
                self.running_tasks[task.id] = future
                
                # Add completion callback
                future.add_done_callback(lambda f: self._task_completed(task.id, f))
            
            logger.debug(f"Assigned task {task.id} to worker {worker.id}")
            
        except Exception as e:
            logger.error(f"Failed to assign task to worker: {e}")
    
    def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a task"""
        try:
            if task.task_function:
                # Execute the actual task function
                result = task.task_function()
                return result
            else:
                # Simulate task execution
                time.sleep(min(task.estimated_duration.total_seconds(), 0.1))
                return f"Task {task.id} completed"
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.id} - {e}")
            raise
    
    def _task_completed(self, task_id: str, future: concurrent.futures.Future):
        """Handle task completion"""
        try:
            with self.lock:
                if task_id not in self.tasks:
                    return
                
                task = self.tasks[task_id]
                worker = self.workers.get(task.assigned_worker) if task.assigned_worker else None
                
                # Update task state
                task.end_time = datetime.now()
                task.actual_duration = task.end_time - (task.start_time or task.end_time)
                
                try:
                    # Get task result
                    task.result = future.result()
                    task.state = TaskState.COMPLETED
                    
                    # Update statistics
                    self.scheduler_stats['total_tasks_completed'] += 1
                    self.metrics.completed_tasks += 1
                    
                    if worker:
                        worker.completed_tasks += 1
                    
                except Exception as e:
                    # Task failed
                    task.error = e
                    task.state = TaskState.FAILED
                    self.metrics.failed_tasks += 1
                    
                    if worker:
                        worker.failed_tasks += 1
                    
                    # Retry if possible
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.state = TaskState.READY
                        # Re-schedule task
                        self.schedule_task(task)
                
                # Update worker state
                if worker:
                    worker.current_task_count = max(0, worker.current_task_count - 1)
                    worker.is_busy = worker.current_task_count >= worker.max_concurrent_tasks
                    
                    # Release resources
                    for resource, required in task.resource_requirements.items():
                        if resource in worker.available_resources:
                            worker.available_resources[resource] += required
                    
                    # Update worker performance metrics
                    self._update_worker_performance(worker, task)
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                logger.debug(f"Task completed: {task_id} - {task.state.value}")
                
        except Exception as e:
            logger.error(f"Failed to handle task completion: {e}")
    
    def _update_worker_heartbeats(self):
        """Update worker heartbeats"""
        try:
            current_time = datetime.now()
            
            for worker in self.workers.values():
                # Simple heartbeat - just update timestamp
                # In a real implementation, this would check actual worker health
                worker.last_heartbeat = current_time
            
        except Exception as e:
            logger.error(f"Failed to update worker heartbeats: {e}")
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks"""
        try:
            current_time = datetime.now()
            cleanup_threshold = timedelta(hours=1)  # Keep completed tasks for 1 hour
            
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED] and
                    task.end_time and current_time - task.end_time > cleanup_threshold):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
            
            if tasks_to_remove:
                logger.debug(f"Cleaned up {len(tasks_to_remove)} old tasks")
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed tasks: {e}")
    
    def _update_metrics(self):
        """Update scheduling metrics"""
        try:
            current_time = datetime.now()
            measurement_duration = current_time - self.metrics.measurement_start
            
            if measurement_duration.total_seconds() > 0:
                # Update throughput
                self.metrics.tasks_per_second = self.metrics.completed_tasks / measurement_duration.total_seconds()
            
            # Update success rate
            total_finished = self.metrics.completed_tasks + self.metrics.failed_tasks
            if total_finished > 0:
                self.metrics.success_rate = self.metrics.completed_tasks / total_finished
            
            # Update worker utilization
            if self.workers:
                busy_workers = sum(1 for w in self.workers.values() if w.is_busy)
                active_workers = sum(1 for w in self.workers.values() if w.is_active)
                if active_workers > 0:
                    self.metrics.worker_utilization = busy_workers / active_workers
            
            # Update wait times (simplified calculation)
            wait_times = []
            for task in self.tasks.values():
                if task.start_time and task.scheduled_at:
                    wait_time = task.start_time - task.scheduled_at
                    wait_times.append(wait_time)
            
            if wait_times:
                total_wait = sum(wait_times, timedelta(0))
                self.metrics.average_wait_time = total_wait / len(wait_times)
            
            # Update execution times
            execution_times = []
            for task in self.tasks.values():
                if task.actual_duration.total_seconds() > 0:
                    execution_times.append(task.actual_duration)
            
            if execution_times:
                total_execution = sum(execution_times, timedelta(0))
                self.metrics.average_execution_time = total_execution / len(execution_times)
            
            # Update measurement duration
            self.metrics.measurement_duration = measurement_duration
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _update_worker_performance(self, worker: WorkerNode, task: ScheduledTask):
        """Update worker performance metrics"""
        try:
            # Update average task duration
            total_tasks = worker.completed_tasks + worker.failed_tasks
            if total_tasks > 0:
                current_avg = worker.average_task_duration.total_seconds()
                new_duration = task.actual_duration.total_seconds()
                
                new_avg_seconds = (current_avg * (total_tasks - 1) + new_duration) / total_tasks
                worker.average_task_duration = timedelta(seconds=new_avg_seconds)
            
        except Exception as e:
            logger.error(f"Failed to update worker performance: {e}")