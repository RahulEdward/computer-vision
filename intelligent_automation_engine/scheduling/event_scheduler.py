"""
Event Scheduler for Intelligent Event Management

This module provides comprehensive event scheduling capabilities including
event-driven automation, triggers, and reactive scheduling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta, timezone
import uuid
import logging
import threading
import asyncio
import queue
from collections import defaultdict, deque
import json
import weakref

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events"""
    SYSTEM = "system"
    USER = "user"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    EXTERNAL = "external"
    INTERNAL = "internal"
    WEBHOOK = "webhook"
    API = "api"


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


class EventStatus(Enum):
    """Event status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TriggerType(Enum):
    """Types of triggers"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    THRESHOLD_BASED = "threshold_based"
    PATTERN_BASED = "pattern_based"
    MANUAL = "manual"


class SchedulerState(Enum):
    """Scheduler states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class Event:
    """Represents an event in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Event identification
    name: str = ""
    event_type: EventType = EventType.USER
    source: str = ""
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    # Event properties
    priority: EventPriority = EventPriority.MEDIUM
    status: EventStatus = EventStatus.PENDING
    
    # Event relationships
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    
    # Event processing
    handler: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[timedelta] = None
    
    # Event result
    result: Any = None
    error_message: str = ""
    processing_time: timedelta = timedelta(0)


@dataclass
class EventTrigger:
    """Represents an event trigger"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Trigger identification
    name: str = ""
    description: str = ""
    trigger_type: TriggerType = TriggerType.TIME_BASED
    
    # Trigger configuration
    condition: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Trigger timing
    schedule: Optional[str] = None  # Cron expression or time specification
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    
    # Trigger actions
    actions: List[Dict[str, Any]] = field(default_factory=list)
    event_template: Optional[Dict[str, Any]] = None
    
    # Trigger state
    is_enabled: bool = True
    is_active: bool = False
    run_count: int = 0
    
    # Trigger metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class EventHandler:
    """Represents an event handler"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Handler identification
    name: str = ""
    description: str = ""
    
    # Handler configuration
    event_types: List[EventType] = field(default_factory=list)
    event_patterns: List[str] = field(default_factory=list)
    
    # Handler function
    handler_function: Optional[Callable] = None
    handler_module: Optional[str] = None
    handler_class: Optional[str] = None
    
    # Handler properties
    is_async: bool = False
    timeout: Optional[timedelta] = None
    max_concurrent: int = 1
    
    # Handler state
    is_enabled: bool = True
    current_executions: int = 0
    total_executions: int = 0
    
    # Handler metadata
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0


@dataclass
class EventSubscription:
    """Represents an event subscription"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Subscription identification
    subscriber_id: str = ""
    subscriber_name: str = ""
    
    # Subscription configuration
    event_types: List[EventType] = field(default_factory=list)
    event_patterns: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Subscription delivery
    delivery_method: str = "callback"  # callback, webhook, queue
    delivery_endpoint: Optional[str] = None
    delivery_options: Dict[str, Any] = field(default_factory=dict)
    
    # Subscription state
    is_active: bool = True
    delivery_count: int = 0
    last_delivery: Optional[datetime] = None
    
    # Subscription metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class EventMetrics:
    """Event processing metrics"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metrics identification
    period_start: datetime = field(default_factory=datetime.now)
    period_end: Optional[datetime] = None
    
    # Event counts
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    cancelled_events: int = 0
    
    # Processing metrics
    average_processing_time: timedelta = timedelta(0)
    max_processing_time: timedelta = timedelta(0)
    min_processing_time: timedelta = timedelta(0)
    
    # Queue metrics
    max_queue_size: int = 0
    average_queue_size: float = 0.0
    queue_overflow_count: int = 0
    
    # Handler metrics
    active_handlers: int = 0
    handler_executions: Dict[str, int] = field(default_factory=dict)
    handler_failures: Dict[str, int] = field(default_factory=dict)
    
    # Trigger metrics
    active_triggers: int = 0
    trigger_executions: Dict[str, int] = field(default_factory=dict)
    trigger_failures: Dict[str, int] = field(default_factory=dict)


class EventScheduler:
    """Comprehensive event scheduler for intelligent event management"""
    
    def __init__(self, max_queue_size: int = 10000, max_workers: int = 10):
        # Core configuration
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.timezone = timezone.utc
        
        # Event management
        self.events: Dict[str, Event] = {}
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self.event_history: deque = deque(maxlen=10000)
        
        # Triggers and handlers
        self.triggers: Dict[str, EventTrigger] = {}
        self.handlers: Dict[str, EventHandler] = {}
        self.subscriptions: Dict[str, EventSubscription] = {}
        
        # Scheduler state
        self.state = SchedulerState.STOPPED
        self.worker_threads: List[threading.Thread] = []
        self.trigger_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Event processing
        self.event_processors: Dict[str, Callable] = {}
        self.event_filters: List[Callable] = []
        self.event_transformers: List[Callable] = []
        
        # Metrics and monitoring
        self.current_metrics = EventMetrics()
        self.metrics_history: deque = deque(maxlen=100)
        
        # Statistics
        self.scheduler_stats = {
            'total_events_processed': 0,
            'total_events_failed': 0,
            'total_triggers_executed': 0,
            'total_handlers_executed': 0,
            'uptime': timedelta(0),
            'start_time': None
        }
        
        # Thread safety
        self.scheduler_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Async support
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_tasks: Set[asyncio.Task] = set()
        
        logger.info("Event scheduler initialized")
    
    def start(self) -> bool:
        """Start the event scheduler"""
        try:
            with self.scheduler_lock:
                if self.state != SchedulerState.STOPPED:
                    logger.warning("Scheduler is already running or starting")
                    return False
                
                self.state = SchedulerState.STARTING
                self.shutdown_event.clear()
                
                # Start worker threads
                for i in range(self.max_workers):
                    worker = threading.Thread(
                        target=self._worker_loop,
                        name=f"EventWorker-{i}",
                        daemon=True
                    )
                    worker.start()
                    self.worker_threads.append(worker)
                
                # Start trigger monitoring thread
                self.trigger_thread = threading.Thread(
                    target=self._trigger_loop,
                    name="TriggerMonitor",
                    daemon=True
                )
                self.trigger_thread.start()
                
                # Start metrics collection thread
                self.metrics_thread = threading.Thread(
                    target=self._metrics_loop,
                    name="MetricsCollector",
                    daemon=True
                )
                self.metrics_thread.start()
                
                # Update state and statistics
                self.state = SchedulerState.RUNNING
                self.scheduler_stats['start_time'] = datetime.now(self.timezone)
                
                logger.info("Event scheduler started successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start event scheduler: {e}")
            self.state = SchedulerState.ERROR
            return False
    
    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the event scheduler"""
        try:
            with self.scheduler_lock:
                if self.state == SchedulerState.STOPPED:
                    return True
                
                self.state = SchedulerState.STOPPING
                
                # Signal shutdown
                self.shutdown_event.set()
                
                # Wait for threads to finish
                all_threads = self.worker_threads + [self.trigger_thread, self.metrics_thread]
                for thread in all_threads:
                    if thread and thread.is_alive():
                        thread.join(timeout=timeout / len(all_threads))
                
                # Cancel async tasks
                if self.async_loop and not self.async_loop.is_closed():
                    for task in self.async_tasks:
                        if not task.done():
                            task.cancel()
                
                # Update state and statistics
                self.state = SchedulerState.STOPPED
                if self.scheduler_stats['start_time']:
                    self.scheduler_stats['uptime'] += (
                        datetime.now(self.timezone) - self.scheduler_stats['start_time']
                    )
                
                # Clear threads
                self.worker_threads.clear()
                self.trigger_thread = None
                self.metrics_thread = None
                
                logger.info("Event scheduler stopped successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop event scheduler: {e}")
            return False
    
    def schedule_event(self, event: Event) -> bool:
        """Schedule an event for processing"""
        try:
            with self.scheduler_lock:
                if self.state != SchedulerState.RUNNING:
                    logger.warning("Scheduler is not running")
                    return False
                
                # Apply filters
                for filter_func in self.event_filters:
                    if not filter_func(event):
                        logger.debug(f"Event {event.id} filtered out")
                        return False
                
                # Apply transformers
                for transformer in self.event_transformers:
                    event = transformer(event)
                
                # Store event
                self.events[event.id] = event
                
                # Add to queue with priority
                priority = -event.priority.value  # Negative for max-heap behavior
                try:
                    self.event_queue.put_nowait((priority, event.created_at, event.id))
                    logger.debug(f"Scheduled event: {event.id}")
                    return True
                except queue.Full:
                    logger.error("Event queue is full")
                    self.current_metrics.queue_overflow_count += 1
                    return False
                
        except Exception as e:
            logger.error(f"Failed to schedule event: {e}")
            return False
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event"""
        try:
            with self.scheduler_lock:
                if event_id not in self.events:
                    return False
                
                event = self.events[event_id]
                if event.status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED]:
                    return False
                
                event.status = EventStatus.CANCELLED
                logger.debug(f"Cancelled event: {event_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cancel event: {e}")
            return False
    
    def add_trigger(self, trigger: EventTrigger) -> bool:
        """Add an event trigger"""
        try:
            with self.scheduler_lock:
                self.triggers[trigger.id] = trigger
                
                # Calculate next run time if scheduled
                if trigger.schedule and trigger.is_enabled:
                    trigger.next_run = self._calculate_next_run(trigger.schedule)
                
                logger.debug(f"Added trigger: {trigger.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add trigger: {e}")
            return False
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove an event trigger"""
        try:
            with self.scheduler_lock:
                if trigger_id in self.triggers:
                    del self.triggers[trigger_id]
                    logger.debug(f"Removed trigger: {trigger_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove trigger: {e}")
            return False
    
    def add_handler(self, handler: EventHandler) -> bool:
        """Add an event handler"""
        try:
            with self.scheduler_lock:
                self.handlers[handler.id] = handler
                logger.debug(f"Added handler: {handler.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add handler: {e}")
            return False
    
    def remove_handler(self, handler_id: str) -> bool:
        """Remove an event handler"""
        try:
            with self.scheduler_lock:
                if handler_id in self.handlers:
                    del self.handlers[handler_id]
                    logger.debug(f"Removed handler: {handler_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove handler: {e}")
            return False
    
    def subscribe(self, subscription: EventSubscription) -> bool:
        """Add an event subscription"""
        try:
            with self.scheduler_lock:
                self.subscriptions[subscription.id] = subscription
                logger.debug(f"Added subscription: {subscription.subscriber_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add subscription: {e}")
            return False
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove an event subscription"""
        try:
            with self.scheduler_lock:
                if subscription_id in self.subscriptions:
                    del self.subscriptions[subscription_id]
                    logger.debug(f"Removed subscription: {subscription_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove subscription: {e}")
            return False
    
    def get_event_status(self, event_id: str) -> Optional[EventStatus]:
        """Get status of an event"""
        try:
            with self.scheduler_lock:
                if event_id in self.events:
                    return self.events[event_id].status
                return None
                
        except Exception as e:
            logger.error(f"Failed to get event status: {e}")
            return None
    
    def get_scheduler_metrics(self) -> EventMetrics:
        """Get current scheduler metrics"""
        try:
            with self.scheduler_lock:
                # Update current metrics
                self.current_metrics.period_end = datetime.now(self.timezone)
                self.current_metrics.active_handlers = len([h for h in self.handlers.values() if h.is_enabled])
                self.current_metrics.active_triggers = len([t for t in self.triggers.values() if t.is_enabled])
                
                return self.current_metrics
                
        except Exception as e:
            logger.error(f"Failed to get scheduler metrics: {e}")
            return EventMetrics()
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        try:
            with self.scheduler_lock:
                stats = dict(self.scheduler_stats)
                
                # Add current state
                stats.update({
                    'state': self.state.value,
                    'queue_size': self.event_queue.qsize(),
                    'active_events': len([e for e in self.events.values() 
                                        if e.status == EventStatus.PROCESSING]),
                    'total_triggers': len(self.triggers),
                    'total_handlers': len(self.handlers),
                    'total_subscriptions': len(self.subscriptions),
                    'worker_threads': len(self.worker_threads)
                })
                
                # Calculate uptime
                if self.scheduler_stats['start_time'] and self.state == SchedulerState.RUNNING:
                    current_uptime = datetime.now(self.timezone) - self.scheduler_stats['start_time']
                    stats['current_uptime'] = current_uptime
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get scheduler statistics: {e}")
            return {}
    
    # Internal methods
    def _worker_loop(self):
        """Main worker loop for processing events"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get event from queue with timeout
                    priority, created_at, event_id = self.event_queue.get(timeout=1.0)
                    
                    if event_id not in self.events:
                        continue
                    
                    event = self.events[event_id]
                    
                    # Check if event is still valid
                    if event.status != EventStatus.PENDING:
                        continue
                    
                    # Process event
                    self._process_event(event)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}")
                    
        except Exception as e:
            logger.error(f"Worker loop failed: {e}")
    
    def _process_event(self, event: Event):
        """Process a single event"""
        try:
            start_time = datetime.now(self.timezone)
            event.status = EventStatus.PROCESSING
            event.processed_at = start_time
            
            # Find matching handlers
            matching_handlers = self._find_matching_handlers(event)
            
            if not matching_handlers:
                logger.warning(f"No handlers found for event: {event.id}")
                event.status = EventStatus.COMPLETED
                return
            
            # Execute handlers
            for handler in matching_handlers:
                try:
                    if handler.current_executions >= handler.max_concurrent:
                        logger.warning(f"Handler {handler.name} at max concurrency")
                        continue
                    
                    handler.current_executions += 1
                    
                    # Execute handler
                    if handler.is_async:
                        self._execute_async_handler(handler, event)
                    else:
                        self._execute_sync_handler(handler, event)
                    
                    handler.total_executions += 1
                    self.current_metrics.handler_executions[handler.id] = (
                        self.current_metrics.handler_executions.get(handler.id, 0) + 1
                    )
                    
                except Exception as e:
                    logger.error(f"Handler {handler.name} failed: {e}")
                    event.error_message = str(e)
                    self.current_metrics.handler_failures[handler.id] = (
                        self.current_metrics.handler_failures.get(handler.id, 0) + 1
                    )
                finally:
                    handler.current_executions -= 1
            
            # Update event status
            if not event.error_message:
                event.status = EventStatus.COMPLETED
            else:
                event.status = EventStatus.FAILED
                
                # Retry if configured
                if event.retry_count < event.max_retries:
                    event.retry_count += 1
                    event.status = EventStatus.RETRYING
                    # Re-queue for retry
                    self.schedule_event(event)
            
            # Calculate processing time
            end_time = datetime.now(self.timezone)
            event.processing_time = end_time - start_time
            
            # Update metrics
            self.current_metrics.processed_events += 1
            if event.status == EventStatus.FAILED:
                self.current_metrics.failed_events += 1
            
            # Update statistics
            self.scheduler_stats['total_events_processed'] += 1
            if event.status == EventStatus.FAILED:
                self.scheduler_stats['total_events_failed'] += 1
            
            # Notify subscribers
            self._notify_subscribers(event)
            
            # Move to history
            self.event_history.append(event)
            
        except Exception as e:
            logger.error(f"Failed to process event {event.id}: {e}")
            event.status = EventStatus.FAILED
            event.error_message = str(e)
    
    def _find_matching_handlers(self, event: Event) -> List[EventHandler]:
        """Find handlers that match the event"""
        try:
            matching_handlers = []
            
            for handler in self.handlers.values():
                if not handler.is_enabled:
                    continue
                
                # Check event type match
                if handler.event_types and event.event_type not in handler.event_types:
                    continue
                
                # Check pattern match
                if handler.event_patterns:
                    pattern_match = False
                    for pattern in handler.event_patterns:
                        if self._match_pattern(event, pattern):
                            pattern_match = True
                            break
                    if not pattern_match:
                        continue
                
                matching_handlers.append(handler)
            
            # Sort by priority
            matching_handlers.sort(key=lambda h: h.priority, reverse=True)
            return matching_handlers
            
        except Exception as e:
            logger.error(f"Failed to find matching handlers: {e}")
            return []
    
    def _execute_sync_handler(self, handler: EventHandler, event: Event):
        """Execute synchronous handler"""
        try:
            if handler.handler_function:
                result = handler.handler_function(event)
                event.result = result
            else:
                logger.warning(f"Handler {handler.name} has no function")
                
        except Exception as e:
            logger.error(f"Sync handler execution failed: {e}")
            raise
    
    def _execute_async_handler(self, handler: EventHandler, event: Event):
        """Execute asynchronous handler"""
        try:
            if not self.async_loop:
                self.async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.async_loop)
            
            if handler.handler_function:
                task = self.async_loop.create_task(handler.handler_function(event))
                self.async_tasks.add(task)
                task.add_done_callback(self.async_tasks.discard)
            else:
                logger.warning(f"Async handler {handler.name} has no function")
                
        except Exception as e:
            logger.error(f"Async handler execution failed: {e}")
            raise
    
    def _trigger_loop(self):
        """Monitor and execute triggers"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    current_time = datetime.now(self.timezone)
                    
                    for trigger in list(self.triggers.values()):
                        if not trigger.is_enabled or not trigger.is_active:
                            continue
                        
                        # Check if trigger should fire
                        if trigger.next_run and current_time >= trigger.next_run:
                            self._execute_trigger(trigger)
                    
                    # Sleep for a short interval
                    self.shutdown_event.wait(1.0)
                    
                except Exception as e:
                    logger.error(f"Error in trigger loop: {e}")
                    
        except Exception as e:
            logger.error(f"Trigger loop failed: {e}")
    
    def _execute_trigger(self, trigger: EventTrigger):
        """Execute a trigger"""
        try:
            # Create event from trigger
            if trigger.event_template:
                event_data = dict(trigger.event_template)
            else:
                event_data = {}
            
            event = Event(
                name=f"Triggered: {trigger.name}",
                event_type=EventType.TRIGGERED,
                source=f"trigger:{trigger.id}",
                data=event_data,
                metadata={'trigger_id': trigger.id}
            )
            
            # Schedule the event
            if self.schedule_event(event):
                trigger.run_count += 1
                trigger.last_run = datetime.now(self.timezone)
                
                # Calculate next run
                if trigger.schedule:
                    trigger.next_run = self._calculate_next_run(trigger.schedule)
                
                # Update metrics
                self.current_metrics.trigger_executions[trigger.id] = (
                    self.current_metrics.trigger_executions.get(trigger.id, 0) + 1
                )
                self.scheduler_stats['total_triggers_executed'] += 1
                
                logger.debug(f"Executed trigger: {trigger.name}")
            else:
                self.current_metrics.trigger_failures[trigger.id] = (
                    self.current_metrics.trigger_failures.get(trigger.id, 0) + 1
                )
                
        except Exception as e:
            logger.error(f"Failed to execute trigger {trigger.name}: {e}")
            self.current_metrics.trigger_failures[trigger.id] = (
                self.current_metrics.trigger_failures.get(trigger.id, 0) + 1
            )
    
    def _notify_subscribers(self, event: Event):
        """Notify event subscribers"""
        try:
            for subscription in self.subscriptions.values():
                if not subscription.is_active:
                    continue
                
                # Check if subscription matches event
                if not self._subscription_matches_event(subscription, event):
                    continue
                
                # Deliver event to subscriber
                self._deliver_event_to_subscriber(subscription, event)
                
        except Exception as e:
            logger.error(f"Failed to notify subscribers: {e}")
    
    def _subscription_matches_event(self, subscription: EventSubscription, event: Event) -> bool:
        """Check if subscription matches event"""
        try:
            # Check event type
            if subscription.event_types and event.event_type not in subscription.event_types:
                return False
            
            # Check patterns
            if subscription.event_patterns:
                pattern_match = False
                for pattern in subscription.event_patterns:
                    if self._match_pattern(event, pattern):
                        pattern_match = True
                        break
                if not pattern_match:
                    return False
            
            # Check filters
            for filter_key, filter_value in subscription.filters.items():
                if filter_key in event.data:
                    if event.data[filter_key] != filter_value:
                        return False
                elif filter_key in event.metadata:
                    if event.metadata[filter_key] != filter_value:
                        return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check subscription match: {e}")
            return False
    
    def _deliver_event_to_subscriber(self, subscription: EventSubscription, event: Event):
        """Deliver event to subscriber"""
        try:
            if subscription.delivery_method == "callback":
                # Callback delivery (placeholder)
                pass
            elif subscription.delivery_method == "webhook":
                # Webhook delivery (placeholder)
                pass
            elif subscription.delivery_method == "queue":
                # Queue delivery (placeholder)
                pass
            
            subscription.delivery_count += 1
            subscription.last_delivery = datetime.now(self.timezone)
            
        except Exception as e:
            logger.error(f"Failed to deliver event to subscriber: {e}")
    
    def _match_pattern(self, event: Event, pattern: str) -> bool:
        """Check if event matches pattern"""
        try:
            # Simple pattern matching (can be enhanced)
            if pattern == "*":
                return True
            elif pattern.startswith("name:"):
                return event.name == pattern[5:]
            elif pattern.startswith("source:"):
                return event.source == pattern[7:]
            else:
                return pattern in event.name or pattern in event.source
                
        except Exception as e:
            logger.error(f"Failed to match pattern: {e}")
            return False
    
    def _calculate_next_run(self, schedule: str) -> Optional[datetime]:
        """Calculate next run time from schedule"""
        try:
            # This is a simplified implementation
            # In practice, this would parse cron expressions or other schedule formats
            
            current_time = datetime.now(self.timezone)
            
            # Simple interval parsing
            if schedule.endswith('s'):
                seconds = int(schedule[:-1])
                return current_time + timedelta(seconds=seconds)
            elif schedule.endswith('m'):
                minutes = int(schedule[:-1])
                return current_time + timedelta(minutes=minutes)
            elif schedule.endswith('h'):
                hours = int(schedule[:-1])
                return current_time + timedelta(hours=hours)
            elif schedule.endswith('d'):
                days = int(schedule[:-1])
                return current_time + timedelta(days=days)
            else:
                # Default to 1 hour
                return current_time + timedelta(hours=1)
                
        except Exception as e:
            logger.error(f"Failed to calculate next run: {e}")
            return None
    
    def _metrics_loop(self):
        """Collect and update metrics"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Update queue metrics
                    current_queue_size = self.event_queue.qsize()
                    if current_queue_size > self.current_metrics.max_queue_size:
                        self.current_metrics.max_queue_size = current_queue_size
                    
                    # Calculate average queue size
                    self.current_metrics.average_queue_size = (
                        (self.current_metrics.average_queue_size + current_queue_size) / 2
                    )
                    
                    # Sleep for metrics collection interval
                    self.shutdown_event.wait(10.0)
                    
                except Exception as e:
                    logger.error(f"Error in metrics loop: {e}")
                    
        except Exception as e:
            logger.error(f"Metrics loop failed: {e}")