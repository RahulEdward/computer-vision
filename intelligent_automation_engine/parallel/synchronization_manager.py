"""
Synchronization Manager for Parallel Execution

This module provides comprehensive synchronization capabilities for
coordinating parallel task execution and managing shared resources.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import threading
import asyncio
import queue
import time
from collections import defaultdict, deque
import concurrent.futures
import weakref

logger = logging.getLogger(__name__)


class SynchronizationType(Enum):
    """Types of synchronization primitives"""
    MUTEX = "mutex"  # Mutual exclusion
    SEMAPHORE = "semaphore"  # Counting semaphore
    BARRIER = "barrier"  # Synchronization barrier
    CONDITION = "condition"  # Condition variable
    EVENT = "event"  # Event signaling
    LOCK = "lock"  # Read-write lock
    QUEUE = "queue"  # Synchronized queue
    CHANNEL = "channel"  # Message channel


class SynchronizationScope(Enum):
    """Scope of synchronization"""
    LOCAL = "local"  # Within single process
    PROCESS = "process"  # Across processes
    DISTRIBUTED = "distributed"  # Across network
    GLOBAL = "global"  # System-wide


class LockType(Enum):
    """Types of locks"""
    EXCLUSIVE = "exclusive"  # Exclusive/write lock
    SHARED = "shared"  # Shared/read lock
    UPGRADEABLE = "upgradeable"  # Upgradeable read lock


class SynchronizationState(Enum):
    """States of synchronization objects"""
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    WAITING = "waiting"
    RELEASED = "released"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SynchronizationPrimitive:
    """Base class for synchronization primitives"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Primitive identification
    name: str = ""
    description: str = ""
    sync_type: SynchronizationType = SynchronizationType.MUTEX
    scope: SynchronizationScope = SynchronizationScope.LOCAL
    
    # Primitive state
    state: SynchronizationState = SynchronizationState.AVAILABLE
    owner: Optional[str] = None
    waiters: List[str] = field(default_factory=list)
    
    # Primitive configuration
    max_count: int = 1  # For semaphores
    timeout: Optional[timedelta] = None
    is_reentrant: bool = False
    
    # Primitive statistics
    acquisition_count: int = 0
    total_wait_time: timedelta = timedelta(0)
    max_wait_time: timedelta = timedelta(0)
    
    # Primitive metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronizationRequest:
    """Request for synchronization primitive access"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request identification
    requester_id: str = ""
    primitive_id: str = ""
    
    # Request details
    lock_type: LockType = LockType.EXCLUSIVE
    timeout: Optional[timedelta] = None
    priority: int = 0
    
    # Request state
    state: SynchronizationState = SynchronizationState.WAITING
    granted_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    
    # Request metadata
    requested_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronizationEvent:
    """Event in synchronization system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Event identification
    event_type: str = ""
    primitive_id: str = ""
    requester_id: str = ""
    
    # Event details
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Event metadata
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "info"


@dataclass
class DeadlockInfo:
    """Information about detected deadlock"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Deadlock details
    involved_tasks: List[str] = field(default_factory=list)
    involved_primitives: List[str] = field(default_factory=list)
    dependency_cycle: List[Tuple[str, str]] = field(default_factory=list)
    
    # Deadlock resolution
    resolution_strategy: str = ""
    victim_task: Optional[str] = None
    
    # Deadlock metadata
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution_time: timedelta = timedelta(0)


@dataclass
class SynchronizationMetrics:
    """Metrics for synchronization performance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Throughput metrics
    acquisitions_per_second: float = 0.0
    releases_per_second: float = 0.0
    
    # Latency metrics
    average_wait_time: timedelta = timedelta(0)
    average_hold_time: timedelta = timedelta(0)
    max_wait_time: timedelta = timedelta(0)
    
    # Contention metrics
    contention_rate: float = 0.0
    deadlock_count: int = 0
    timeout_count: int = 0
    
    # Efficiency metrics
    utilization_rate: float = 0.0
    fairness_index: float = 0.0
    
    # Measurement metadata
    measurement_start: datetime = field(default_factory=datetime.now)
    measurement_duration: timedelta = timedelta(0)


class SynchronizationManager:
    """Comprehensive synchronization manager for parallel execution coordination"""
    
    def __init__(self):
        # Core data structures
        self.primitives: Dict[str, SynchronizationPrimitive] = {}
        self.requests: Dict[str, SynchronizationRequest] = {}
        self.events: deque = deque(maxlen=1000)
        
        # Synchronization objects
        self.mutexes: Dict[str, threading.RLock] = {}
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.barriers: Dict[str, threading.Barrier] = {}
        self.conditions: Dict[str, threading.Condition] = {}
        self.events_sync: Dict[str, threading.Event] = {}
        self.queues: Dict[str, queue.Queue] = {}
        
        # Deadlock detection
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.deadlock_detector_thread: Optional[threading.Thread] = None
        self.deadlock_detection_enabled = True
        
        # Performance tracking
        self.metrics = SynchronizationMetrics()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Manager state
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Configuration
        self.config = {
            'deadlock_detection_interval': timedelta(seconds=10),
            'default_timeout': timedelta(seconds=30),
            'max_wait_queue_size': 100,
            'enable_fairness': True,
            'enable_priority': True
        }
        
        # Statistics
        self.sync_stats = {
            'total_primitives': 0,
            'total_acquisitions': 0,
            'total_releases': 0,
            'total_timeouts': 0,
            'total_deadlocks': 0
        }
        
        # Thread safety
        self.manager_lock = threading.RLock()
        
        logger.info("Synchronization manager initialized")
    
    def create_mutex(self, name: str, is_reentrant: bool = True) -> str:
        """Create a mutex synchronization primitive"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Mutex: {name}",
                    sync_type=SynchronizationType.MUTEX,
                    is_reentrant=is_reentrant
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                if is_reentrant:
                    self.mutexes[primitive.id] = threading.RLock()
                else:
                    self.mutexes[primitive.id] = threading.Lock()
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("mutex_created", primitive.id, "", f"Created mutex: {name}")
                
                logger.debug(f"Created mutex: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create mutex: {e}")
            return ""
    
    def create_semaphore(self, name: str, max_count: int = 1) -> str:
        """Create a semaphore synchronization primitive"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Semaphore: {name}",
                    sync_type=SynchronizationType.SEMAPHORE,
                    max_count=max_count
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                self.semaphores[primitive.id] = threading.Semaphore(max_count)
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("semaphore_created", primitive.id, "", f"Created semaphore: {name}")
                
                logger.debug(f"Created semaphore: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create semaphore: {e}")
            return ""
    
    def create_barrier(self, name: str, party_count: int) -> str:
        """Create a barrier synchronization primitive"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Barrier: {name}",
                    sync_type=SynchronizationType.BARRIER,
                    max_count=party_count
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                self.barriers[primitive.id] = threading.Barrier(party_count)
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("barrier_created", primitive.id, "", f"Created barrier: {name}")
                
                logger.debug(f"Created barrier: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create barrier: {e}")
            return ""
    
    def create_condition(self, name: str, mutex_id: Optional[str] = None) -> str:
        """Create a condition variable synchronization primitive"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Condition: {name}",
                    sync_type=SynchronizationType.CONDITION
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                if mutex_id and mutex_id in self.mutexes:
                    lock = self.mutexes[mutex_id]
                else:
                    lock = threading.RLock()
                
                self.conditions[primitive.id] = threading.Condition(lock)
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("condition_created", primitive.id, "", f"Created condition: {name}")
                
                logger.debug(f"Created condition: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create condition: {e}")
            return ""
    
    def create_event(self, name: str) -> str:
        """Create an event synchronization primitive"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Event: {name}",
                    sync_type=SynchronizationType.EVENT
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                self.events_sync[primitive.id] = threading.Event()
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("event_created", primitive.id, "", f"Created event: {name}")
                
                logger.debug(f"Created event: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return ""
    
    def create_queue(self, name: str, max_size: int = 0) -> str:
        """Create a synchronized queue"""
        try:
            with self.manager_lock:
                primitive = SynchronizationPrimitive(
                    name=name,
                    description=f"Queue: {name}",
                    sync_type=SynchronizationType.QUEUE,
                    max_count=max_size
                )
                
                self.primitives[primitive.id] = primitive
                
                # Create actual synchronization object
                self.queues[primitive.id] = queue.Queue(maxsize=max_size)
                
                self.sync_stats['total_primitives'] += 1
                
                self._log_event("queue_created", primitive.id, "", f"Created queue: {name}")
                
                logger.debug(f"Created queue: {name} ({primitive.id})")
                return primitive.id
                
        except Exception as e:
            logger.error(f"Failed to create queue: {e}")
            return ""
    
    def acquire(self, primitive_id: str, requester_id: str, 
               timeout: Optional[timedelta] = None, 
               lock_type: LockType = LockType.EXCLUSIVE) -> bool:
        """Acquire a synchronization primitive"""
        try:
            with self.manager_lock:
                if primitive_id not in self.primitives:
                    logger.error(f"Primitive not found: {primitive_id}")
                    return False
                
                primitive = self.primitives[primitive_id]
                
                # Create acquisition request
                request = SynchronizationRequest(
                    requester_id=requester_id,
                    primitive_id=primitive_id,
                    lock_type=lock_type,
                    timeout=timeout or self.config['default_timeout']
                )
                
                self.requests[request.id] = request
                
                # Update dependency graph for deadlock detection
                self._update_dependency_graph(requester_id, primitive_id, "acquire")
                
                # Log acquisition attempt
                self._log_event("acquire_attempt", primitive_id, requester_id, 
                              f"Attempting to acquire {primitive.sync_type.value}")
                
                # Attempt acquisition
                success = self._perform_acquisition(primitive, request)
                
                if success:
                    # Update primitive state
                    primitive.state = SynchronizationState.ACQUIRED
                    primitive.owner = requester_id
                    primitive.acquisition_count += 1
                    primitive.last_accessed = datetime.now()
                    
                    # Update request state
                    request.state = SynchronizationState.ACQUIRED
                    request.granted_at = datetime.now()
                    
                    # Update statistics
                    self.sync_stats['total_acquisitions'] += 1
                    
                    self._log_event("acquire_success", primitive_id, requester_id, 
                                  f"Successfully acquired {primitive.sync_type.value}")
                else:
                    # Update request state
                    request.state = SynchronizationState.TIMEOUT
                    
                    # Update statistics
                    self.sync_stats['total_timeouts'] += 1
                    
                    self._log_event("acquire_timeout", primitive_id, requester_id, 
                                  f"Timeout acquiring {primitive.sync_type.value}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to acquire primitive: {e}")
            return False
    
    def release(self, primitive_id: str, requester_id: str) -> bool:
        """Release a synchronization primitive"""
        try:
            with self.manager_lock:
                if primitive_id not in self.primitives:
                    logger.error(f"Primitive not found: {primitive_id}")
                    return False
                
                primitive = self.primitives[primitive_id]
                
                # Check ownership
                if primitive.owner != requester_id:
                    logger.warning(f"Requester {requester_id} does not own primitive {primitive_id}")
                    return False
                
                # Perform release
                success = self._perform_release(primitive, requester_id)
                
                if success:
                    # Update primitive state
                    primitive.state = SynchronizationState.RELEASED
                    primitive.owner = None
                    primitive.last_accessed = datetime.now()
                    
                    # Update dependency graph
                    self._update_dependency_graph(requester_id, primitive_id, "release")
                    
                    # Update statistics
                    self.sync_stats['total_releases'] += 1
                    
                    self._log_event("release_success", primitive_id, requester_id, 
                                  f"Successfully released {primitive.sync_type.value}")
                else:
                    self._log_event("release_failed", primitive_id, requester_id, 
                                  f"Failed to release {primitive.sync_type.value}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to release primitive: {e}")
            return False
    
    def wait_barrier(self, primitive_id: str, requester_id: str, 
                    timeout: Optional[timedelta] = None) -> bool:
        """Wait at a barrier synchronization point"""
        try:
            if primitive_id not in self.barriers:
                logger.error(f"Barrier not found: {primitive_id}")
                return False
            
            barrier = self.barriers[primitive_id]
            timeout_seconds = timeout.total_seconds() if timeout else None
            
            try:
                barrier.wait(timeout=timeout_seconds)
                
                self._log_event("barrier_wait_success", primitive_id, requester_id, 
                              "Successfully waited at barrier")
                return True
                
            except threading.BrokenBarrierError:
                self._log_event("barrier_broken", primitive_id, requester_id, 
                              "Barrier was broken")
                return False
                
        except Exception as e:
            logger.error(f"Failed to wait at barrier: {e}")
            return False
    
    def signal_event(self, primitive_id: str, requester_id: str) -> bool:
        """Signal an event"""
        try:
            if primitive_id not in self.events_sync:
                logger.error(f"Event not found: {primitive_id}")
                return False
            
            event = self.events_sync[primitive_id]
            event.set()
            
            self._log_event("event_signaled", primitive_id, requester_id, 
                          "Event signaled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to signal event: {e}")
            return False
    
    def wait_event(self, primitive_id: str, requester_id: str, 
                  timeout: Optional[timedelta] = None) -> bool:
        """Wait for an event"""
        try:
            if primitive_id not in self.events_sync:
                logger.error(f"Event not found: {primitive_id}")
                return False
            
            event = self.events_sync[primitive_id]
            timeout_seconds = timeout.total_seconds() if timeout else None
            
            success = event.wait(timeout=timeout_seconds)
            
            if success:
                self._log_event("event_wait_success", primitive_id, requester_id, 
                              "Successfully waited for event")
            else:
                self._log_event("event_wait_timeout", primitive_id, requester_id, 
                              "Timeout waiting for event")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to wait for event: {e}")
            return False
    
    def put_queue(self, primitive_id: str, item: Any, requester_id: str, 
                 timeout: Optional[timedelta] = None) -> bool:
        """Put an item into a synchronized queue"""
        try:
            if primitive_id not in self.queues:
                logger.error(f"Queue not found: {primitive_id}")
                return False
            
            queue_obj = self.queues[primitive_id]
            timeout_seconds = timeout.total_seconds() if timeout else None
            
            try:
                queue_obj.put(item, timeout=timeout_seconds)
                
                self._log_event("queue_put_success", primitive_id, requester_id, 
                              "Successfully put item in queue")
                return True
                
            except queue.Full:
                self._log_event("queue_put_full", primitive_id, requester_id, 
                              "Queue is full")
                return False
                
        except Exception as e:
            logger.error(f"Failed to put item in queue: {e}")
            return False
    
    def get_queue(self, primitive_id: str, requester_id: str, 
                 timeout: Optional[timedelta] = None) -> Tuple[bool, Any]:
        """Get an item from a synchronized queue"""
        try:
            if primitive_id not in self.queues:
                logger.error(f"Queue not found: {primitive_id}")
                return False, None
            
            queue_obj = self.queues[primitive_id]
            timeout_seconds = timeout.total_seconds() if timeout else None
            
            try:
                item = queue_obj.get(timeout=timeout_seconds)
                
                self._log_event("queue_get_success", primitive_id, requester_id, 
                              "Successfully got item from queue")
                return True, item
                
            except queue.Empty:
                self._log_event("queue_get_empty", primitive_id, requester_id, 
                              "Queue is empty")
                return False, None
                
        except Exception as e:
            logger.error(f"Failed to get item from queue: {e}")
            return False, None
    
    def start_deadlock_detection(self) -> bool:
        """Start deadlock detection"""
        try:
            with self.manager_lock:
                if self.deadlock_detector_thread and self.deadlock_detector_thread.is_alive():
                    logger.warning("Deadlock detection is already running")
                    return False
                
                self.deadlock_detection_enabled = True
                self.stop_event.clear()
                self.deadlock_detector_thread = threading.Thread(
                    target=self._deadlock_detection_loop, daemon=True
                )
                self.deadlock_detector_thread.start()
                
                logger.info("Started deadlock detection")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start deadlock detection: {e}")
            return False
    
    def stop_deadlock_detection(self) -> bool:
        """Stop deadlock detection"""
        try:
            with self.manager_lock:
                self.deadlock_detection_enabled = False
                self.stop_event.set()
                
                if self.deadlock_detector_thread:
                    self.deadlock_detector_thread.join(timeout=10)
                
                logger.info("Stopped deadlock detection")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop deadlock detection: {e}")
            return False
    
    def get_primitive_status(self, primitive_id: str) -> Dict[str, Any]:
        """Get status of a synchronization primitive"""
        try:
            with self.manager_lock:
                if primitive_id not in self.primitives:
                    return {'error': 'Primitive not found'}
                
                primitive = self.primitives[primitive_id]
                
                status = {
                    'primitive_id': primitive_id,
                    'name': primitive.name,
                    'type': primitive.sync_type.value,
                    'scope': primitive.scope.value,
                    'state': primitive.state.value,
                    'owner': primitive.owner,
                    'waiters': len(primitive.waiters),
                    'acquisition_count': primitive.acquisition_count,
                    'total_wait_time': primitive.total_wait_time,
                    'max_wait_time': primitive.max_wait_time,
                    'created_at': primitive.created_at,
                    'last_accessed': primitive.last_accessed
                }
                
                # Add type-specific information
                if primitive.sync_type == SynchronizationType.SEMAPHORE:
                    status['max_count'] = primitive.max_count
                    if primitive_id in self.semaphores:
                        # Note: Can't get current count from threading.Semaphore
                        status['available_permits'] = 'unknown'
                
                elif primitive.sync_type == SynchronizationType.QUEUE:
                    if primitive_id in self.queues:
                        queue_obj = self.queues[primitive_id]
                        status['queue_size'] = queue_obj.qsize()
                        status['max_size'] = primitive.max_count
                
                return status
                
        except Exception as e:
            logger.error(f"Failed to get primitive status: {e}")
            return {'error': str(e)}
    
    def get_synchronization_metrics(self) -> SynchronizationMetrics:
        """Get current synchronization metrics"""
        try:
            with self.manager_lock:
                # Update current metrics
                self._update_metrics()
                return self.metrics
                
        except Exception as e:
            logger.error(f"Failed to get synchronization metrics: {e}")
            return SynchronizationMetrics()
    
    def get_synchronization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization statistics"""
        try:
            with self.manager_lock:
                stats = dict(self.sync_stats)
                
                # Add current state
                stats.update({
                    'active_primitives': len(self.primitives),
                    'active_requests': len(self.requests),
                    'deadlock_detection_enabled': self.deadlock_detection_enabled
                })
                
                # Add primitive type breakdown
                primitive_types = defaultdict(int)
                for primitive in self.primitives.values():
                    primitive_types[primitive.sync_type.value] += 1
                stats['primitive_types'] = dict(primitive_types)
                
                # Add state breakdown
                primitive_states = defaultdict(int)
                for primitive in self.primitives.values():
                    primitive_states[primitive.state.value] += 1
                stats['primitive_states'] = dict(primitive_states)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get synchronization statistics: {e}")
            return {}
    
    def cleanup_primitives(self, max_age: timedelta = timedelta(hours=1)) -> int:
        """Clean up old unused primitives"""
        try:
            with self.manager_lock:
                current_time = datetime.now()
                primitives_to_remove = []
                
                for primitive_id, primitive in self.primitives.items():
                    # Check if primitive is old and unused
                    age = current_time - primitive.last_accessed
                    if (age > max_age and 
                        primitive.state in [SynchronizationState.AVAILABLE, SynchronizationState.RELEASED] and
                        not primitive.waiters):
                        primitives_to_remove.append(primitive_id)
                
                # Remove old primitives
                for primitive_id in primitives_to_remove:
                    self._remove_primitive(primitive_id)
                
                logger.debug(f"Cleaned up {len(primitives_to_remove)} old primitives")
                return len(primitives_to_remove)
                
        except Exception as e:
            logger.error(f"Failed to cleanup primitives: {e}")
            return 0
    
    # Internal methods
    def _perform_acquisition(self, primitive: SynchronizationPrimitive, 
                           request: SynchronizationRequest) -> bool:
        """Perform the actual acquisition of a synchronization primitive"""
        try:
            timeout_seconds = request.timeout.total_seconds() if request.timeout else None
            
            if primitive.sync_type == SynchronizationType.MUTEX:
                if primitive.id in self.mutexes:
                    mutex = self.mutexes[primitive.id]
                    return mutex.acquire(timeout=timeout_seconds)
            
            elif primitive.sync_type == SynchronizationType.SEMAPHORE:
                if primitive.id in self.semaphores:
                    semaphore = self.semaphores[primitive.id]
                    return semaphore.acquire(timeout=timeout_seconds)
            
            elif primitive.sync_type == SynchronizationType.CONDITION:
                if primitive.id in self.conditions:
                    condition = self.conditions[primitive.id]
                    with condition:
                        return condition.wait(timeout=timeout_seconds)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to perform acquisition: {e}")
            return False
    
    def _perform_release(self, primitive: SynchronizationPrimitive, requester_id: str) -> bool:
        """Perform the actual release of a synchronization primitive"""
        try:
            if primitive.sync_type == SynchronizationType.MUTEX:
                if primitive.id in self.mutexes:
                    mutex = self.mutexes[primitive.id]
                    mutex.release()
                    return True
            
            elif primitive.sync_type == SynchronizationType.SEMAPHORE:
                if primitive.id in self.semaphores:
                    semaphore = self.semaphores[primitive.id]
                    semaphore.release()
                    return True
            
            elif primitive.sync_type == SynchronizationType.CONDITION:
                if primitive.id in self.conditions:
                    condition = self.conditions[primitive.id]
                    with condition:
                        condition.notify_all()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to perform release: {e}")
            return False
    
    def _update_dependency_graph(self, requester_id: str, primitive_id: str, action: str):
        """Update dependency graph for deadlock detection"""
        try:
            if action == "acquire":
                # Add dependency: requester waits for primitive
                self.dependency_graph[requester_id].add(primitive_id)
            elif action == "release":
                # Remove dependency: requester no longer waits for primitive
                if requester_id in self.dependency_graph:
                    self.dependency_graph[requester_id].discard(primitive_id)
                    if not self.dependency_graph[requester_id]:
                        del self.dependency_graph[requester_id]
            
        except Exception as e:
            logger.error(f"Failed to update dependency graph: {e}")
    
    def _deadlock_detection_loop(self):
        """Main loop for deadlock detection"""
        try:
            while self.deadlock_detection_enabled and not self.stop_event.is_set():
                try:
                    # Detect deadlocks
                    deadlocks = self._detect_deadlocks()
                    
                    # Resolve detected deadlocks
                    for deadlock in deadlocks:
                        self._resolve_deadlock(deadlock)
                    
                    # Sleep until next detection cycle
                    self.stop_event.wait(self.config['deadlock_detection_interval'].total_seconds())
                    
                except Exception as e:
                    logger.error(f"Error in deadlock detection loop: {e}")
                    time.sleep(60)  # Longer sleep on error
            
        except Exception as e:
            logger.error(f"Deadlock detection loop failed: {e}")
        finally:
            logger.debug("Deadlock detection loop ended")
    
    def _detect_deadlocks(self) -> List[DeadlockInfo]:
        """Detect deadlocks in the dependency graph"""
        try:
            deadlocks = []
            
            # Use DFS to detect cycles in dependency graph
            visited = set()
            rec_stack = set()
            
            def dfs(node, path):
                if node in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]
                    
                    deadlock = DeadlockInfo(
                        involved_tasks=cycle,
                        dependency_cycle=[(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
                    )
                    deadlocks.append(deadlock)
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in self.dependency_graph.get(node, []):
                    dfs(neighbor, path + [neighbor])
                
                rec_stack.remove(node)
            
            # Check all nodes
            for node in self.dependency_graph:
                if node not in visited:
                    dfs(node, [node])
            
            if deadlocks:
                logger.warning(f"Detected {len(deadlocks)} deadlocks")
            
            return deadlocks
            
        except Exception as e:
            logger.error(f"Failed to detect deadlocks: {e}")
            return []
    
    def _resolve_deadlock(self, deadlock: DeadlockInfo):
        """Resolve a detected deadlock"""
        try:
            # Simple resolution strategy: abort the youngest task
            if deadlock.involved_tasks:
                # Find the task with the most recent request
                youngest_task = None
                latest_time = datetime.min
                
                for task_id in deadlock.involved_tasks:
                    for request in self.requests.values():
                        if (request.requester_id == task_id and 
                            request.requested_at > latest_time):
                            latest_time = request.requested_at
                            youngest_task = task_id
                
                if youngest_task:
                    deadlock.victim_task = youngest_task
                    deadlock.resolution_strategy = "abort_youngest"
                    
                    # Abort the victim task (in practice, this would notify the task)
                    self._abort_task(youngest_task)
                    
                    deadlock.resolved_at = datetime.now()
                    deadlock.resolution_time = deadlock.resolved_at - deadlock.detected_at
                    
                    self.sync_stats['total_deadlocks'] += 1
                    
                    self._log_event("deadlock_resolved", "", youngest_task, 
                                  f"Resolved deadlock by aborting task {youngest_task}")
            
        except Exception as e:
            logger.error(f"Failed to resolve deadlock: {e}")
    
    def _abort_task(self, task_id: str):
        """Abort a task to resolve deadlock"""
        try:
            # Remove task from dependency graph
            if task_id in self.dependency_graph:
                del self.dependency_graph[task_id]
            
            # Remove task's requests
            requests_to_remove = []
            for request_id, request in self.requests.items():
                if request.requester_id == task_id:
                    requests_to_remove.append(request_id)
            
            for request_id in requests_to_remove:
                del self.requests[request_id]
            
            logger.debug(f"Aborted task: {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to abort task: {e}")
    
    def _remove_primitive(self, primitive_id: str):
        """Remove a synchronization primitive"""
        try:
            # Remove from primitives
            if primitive_id in self.primitives:
                del self.primitives[primitive_id]
            
            # Remove from synchronization objects
            if primitive_id in self.mutexes:
                del self.mutexes[primitive_id]
            if primitive_id in self.semaphores:
                del self.semaphores[primitive_id]
            if primitive_id in self.barriers:
                del self.barriers[primitive_id]
            if primitive_id in self.conditions:
                del self.conditions[primitive_id]
            if primitive_id in self.events_sync:
                del self.events_sync[primitive_id]
            if primitive_id in self.queues:
                del self.queues[primitive_id]
            
            # Remove related requests
            requests_to_remove = []
            for request_id, request in self.requests.items():
                if request.primitive_id == primitive_id:
                    requests_to_remove.append(request_id)
            
            for request_id in requests_to_remove:
                del self.requests[request_id]
            
        except Exception as e:
            logger.error(f"Failed to remove primitive: {e}")
    
    def _update_metrics(self):
        """Update synchronization metrics"""
        try:
            current_time = datetime.now()
            measurement_duration = current_time - self.metrics.measurement_start
            
            if measurement_duration.total_seconds() > 0:
                # Update throughput
                self.metrics.acquisitions_per_second = (
                    self.sync_stats['total_acquisitions'] / measurement_duration.total_seconds()
                )
                self.metrics.releases_per_second = (
                    self.sync_stats['total_releases'] / measurement_duration.total_seconds()
                )
            
            # Update wait times
            wait_times = []
            for request in self.requests.values():
                if request.granted_at and request.requested_at:
                    wait_time = request.granted_at - request.requested_at
                    wait_times.append(wait_time)
            
            if wait_times:
                total_wait = sum(wait_times, timedelta(0))
                self.metrics.average_wait_time = total_wait / len(wait_times)
                self.metrics.max_wait_time = max(wait_times)
            
            # Update contention rate
            total_requests = len(self.requests)
            if total_requests > 0:
                waiting_requests = sum(1 for r in self.requests.values() 
                                     if r.state == SynchronizationState.WAITING)
                self.metrics.contention_rate = waiting_requests / total_requests
            
            # Update other metrics
            self.metrics.deadlock_count = self.sync_stats['total_deadlocks']
            self.metrics.timeout_count = self.sync_stats['total_timeouts']
            self.metrics.measurement_duration = measurement_duration
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _log_event(self, event_type: str, primitive_id: str, requester_id: str, description: str):
        """Log a synchronization event"""
        try:
            event = SynchronizationEvent(
                event_type=event_type,
                primitive_id=primitive_id,
                requester_id=requester_id,
                description=description
            )
            
            self.events.append(event)
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")