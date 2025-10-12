"""
Distributed task queue system using Apache Kafka for handling millions of concurrent automations.
Provides high-throughput, fault-tolerant task processing with horizontal scaling.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = KafkaConsumer = KafkaAdminClient = NewTopic = None

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class VisionTask:
    """Vision automation task definition."""
    id: str
    type: str  # 'click', 'type', 'get', 'act', 'screenshot'
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 30
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.scheduled_at:
            data['scheduled_at'] = self.scheduled_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisionTask':
        """Create task from dictionary."""
        data['priority'] = TaskPriority(data['priority'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('scheduled_at'):
            data['scheduled_at'] = datetime.fromisoformat(data['scheduled_at'])
        return cls(**data)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    worker_id: str = ""
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['completed_at'] = self.completed_at.isoformat()
        return data


class KafkaTaskQueue:
    """High-performance distributed task queue using Apache Kafka."""
    
    def __init__(self, 
                 bootstrap_servers: List[str] = None,
                 task_topic: str = "vision_tasks",
                 result_topic: str = "vision_results",
                 dead_letter_topic: str = "vision_dead_letter",
                 consumer_group: str = "vision_workers"):
        
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka not available. Install with: pip install kafka-python")
        
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self.task_topic = task_topic
        self.result_topic = result_topic
        self.dead_letter_topic = dead_letter_topic
        self.consumer_group = consumer_group
        
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.admin_client: Optional[KafkaAdminClient] = None
        
        self.worker_id = str(uuid.uuid4())
        self.is_running = False
        self.task_handlers: Dict[str, Callable] = {}
        self.metrics = {
            "tasks_produced": 0,
            "tasks_consumed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time_ms": 0,
            "throughput_tasks_per_second": 0
        }
        
        self._last_metrics_update = time.time()
        self._processing_times: List[float] = []
    
    async def initialize(self):
        """Initialize Kafka connections and create topics."""
        logger.info("Initializing Kafka task queue...")
        
        try:
            # Create admin client
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"admin_{self.worker_id}"
            )
            
            # Create topics if they don't exist
            await self._create_topics()
            
            # Initialize producer with optimized settings
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # Performance optimizations
                batch_size=16384,  # 16KB batches
                linger_ms=10,      # Wait up to 10ms to batch messages
                compression_type='snappy',  # Fast compression
                acks='1',          # Wait for leader acknowledgment
                retries=3,
                max_in_flight_requests_per_connection=5,
                buffer_memory=33554432,  # 32MB buffer
            )
            
            logger.info("Kafka task queue initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    async def _create_topics(self):
        """Create Kafka topics with optimal configurations."""
        topics = [
            NewTopic(
                name=self.task_topic,
                num_partitions=12,  # High parallelism
                replication_factor=1,  # Adjust based on cluster size
                topic_configs={
                    'retention.ms': str(24 * 60 * 60 * 1000),  # 24 hours
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            ),
            NewTopic(
                name=self.result_topic,
                num_partitions=12,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'compression.type': 'snappy'
                }
            ),
            NewTopic(
                name=self.dead_letter_topic,
                num_partitions=4,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(30 * 24 * 60 * 60 * 1000),  # 30 days
                    'compression.type': 'snappy'
                }
            )
        ]
        
        try:
            # Check existing topics
            existing_topics = self.admin_client.list_topics()
            topics_to_create = [
                topic for topic in topics 
                if topic.name not in existing_topics
            ]
            
            if topics_to_create:
                self.admin_client.create_topics(topics_to_create)
                logger.info(f"Created topics: {[t.name for t in topics_to_create]}")
            
        except Exception as e:
            logger.warning(f"Topic creation warning: {e}")
    
    async def submit_task(self, task: VisionTask) -> bool:
        """Submit a task to the queue."""
        if not self.producer:
            logger.error("Producer not initialized")
            return False
        
        try:
            # Determine partition key for load balancing
            partition_key = f"{task.type}_{task.priority.value}"
            
            # Send task to Kafka
            future = self.producer.send(
                self.task_topic,
                value=task.to_dict(),
                key=partition_key
            )
            
            # Wait for acknowledgment (non-blocking)
            record_metadata = future.get(timeout=1)
            
            self.metrics["tasks_produced"] += 1
            
            logger.debug(f"Task {task.id} submitted to partition {record_metadata.partition}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.id}: {e}")
            return False
    
    async def submit_batch_tasks(self, tasks: List[VisionTask]) -> int:
        """Submit multiple tasks efficiently."""
        if not self.producer:
            logger.error("Producer not initialized")
            return 0
        
        submitted = 0
        
        try:
            # Send all tasks
            futures = []
            for task in tasks:
                partition_key = f"{task.type}_{task.priority.value}"
                future = self.producer.send(
                    self.task_topic,
                    value=task.to_dict(),
                    key=partition_key
                )
                futures.append(future)
            
            # Flush to ensure all messages are sent
            self.producer.flush()
            
            # Wait for all acknowledgments
            for future in futures:
                try:
                    future.get(timeout=1)
                    submitted += 1
                except Exception as e:
                    logger.error(f"Failed to submit task: {e}")
            
            self.metrics["tasks_produced"] += submitted
            logger.info(f"Submitted {submitted}/{len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Batch task submission failed: {e}")
        
        return submitted
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def start_worker(self):
        """Start consuming and processing tasks."""
        if not KAFKA_AVAILABLE:
            logger.error("Cannot start worker: Kafka not available")
            return
        
        logger.info(f"Starting worker {self.worker_id}")
        
        # Initialize consumer with optimized settings
        self.consumer = KafkaConsumer(
            self.task_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            # Performance optimizations
            fetch_min_bytes=1024,      # Minimum 1KB per fetch
            fetch_max_wait_ms=500,     # Wait up to 500ms for data
            max_partition_fetch_bytes=1048576,  # 1MB per partition
            auto_offset_reset='latest',
            enable_auto_commit=False,  # Manual commit for reliability
            max_poll_records=100,      # Process up to 100 records per poll
        )
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if message_batch:
                    await self._process_message_batch(message_batch)
                    
                    # Commit offsets after successful processing
                    self.consumer.commit()
                
                # Update metrics periodically
                await self._update_metrics()
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
            logger.info(f"Worker {self.worker_id} stopped")
    
    async def _process_message_batch(self, message_batch):
        """Process a batch of messages efficiently."""
        tasks_to_process = []
        
        # Collect all tasks from the batch
        for topic_partition, messages in message_batch.items():
            for message in messages:
                try:
                    task = VisionTask.from_dict(message.value)
                    tasks_to_process.append(task)
                except Exception as e:
                    logger.error(f"Failed to deserialize task: {e}")
        
        # Process tasks concurrently
        if tasks_to_process:
            await self._process_tasks_concurrently(tasks_to_process)
    
    async def _process_tasks_concurrently(self, tasks: List[VisionTask]):
        """Process multiple tasks concurrently."""
        # Limit concurrency to prevent resource exhaustion
        semaphore = asyncio.Semaphore(10)
        
        async def process_single_task(task):
            async with semaphore:
                await self._process_task(task)
        
        # Process all tasks concurrently
        await asyncio.gather(*[process_single_task(task) for task in tasks])
    
    async def _process_task(self, task: VisionTask):
        """Process a single task."""
        start_time = time.time()
        result = None
        
        try:
            self.metrics["tasks_consumed"] += 1
            
            # Check if task is scheduled for future execution
            if task.scheduled_at and task.scheduled_at > datetime.utcnow():
                # Re-queue for later processing
                await self.submit_task(task)
                return
            
            # Get handler for task type
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.type}")
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(task.payload),
                    timeout=task.timeout_seconds
                )
                
                # Create success result
                execution_time = (time.time() - start_time) * 1000
                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    execution_time_ms=execution_time,
                    worker_id=self.worker_id
                )
                
                self.metrics["tasks_completed"] += 1
                self._processing_times.append(execution_time)
                
            except asyncio.TimeoutError:
                raise Exception(f"Task timed out after {task.timeout_seconds} seconds")
            
        except Exception as e:
            # Create failure result
            execution_time = (time.time() - start_time) * 1000
            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
                worker_id=self.worker_id
            )
            
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.id} failed: {e}")
            
            # Handle retries
            if hasattr(task, 'retry_count'):
                task.retry_count += 1
            else:
                task.retry_count = 1
            
            if task.retry_count <= task.max_retries:
                # Re-queue for retry with exponential backoff
                task.scheduled_at = datetime.utcnow() + timedelta(
                    seconds=2 ** task.retry_count
                )
                await self.submit_task(task)
                task_result.status = TaskStatus.RETRYING
            else:
                # Send to dead letter queue
                await self._send_to_dead_letter_queue(task, str(e))
        
        # Publish result
        await self._publish_result(task_result)
    
    async def _publish_result(self, result: TaskResult):
        """Publish task result."""
        if not self.producer:
            return
        
        try:
            self.producer.send(
                self.result_topic,
                value=result.to_dict(),
                key=result.task_id
            )
        except Exception as e:
            logger.error(f"Failed to publish result for task {result.task_id}: {e}")
    
    async def _send_to_dead_letter_queue(self, task: VisionTask, error: str):
        """Send failed task to dead letter queue."""
        if not self.producer:
            return
        
        try:
            dead_letter_data = {
                "task": task.to_dict(),
                "error": error,
                "failed_at": datetime.utcnow().isoformat(),
                "worker_id": self.worker_id
            }
            
            self.producer.send(
                self.dead_letter_topic,
                value=dead_letter_data,
                key=task.id
            )
            
            logger.warning(f"Task {task.id} sent to dead letter queue")
            
        except Exception as e:
            logger.error(f"Failed to send task {task.id} to dead letter queue: {e}")
    
    async def _update_metrics(self):
        """Update performance metrics."""
        current_time = time.time()
        
        # Update every 10 seconds
        if current_time - self._last_metrics_update >= 10:
            # Calculate average processing time
            if self._processing_times:
                self.metrics["avg_processing_time_ms"] = sum(self._processing_times) / len(self._processing_times)
                
                # Keep only recent processing times (last 1000)
                if len(self._processing_times) > 1000:
                    self._processing_times = self._processing_times[-1000:]
            
            # Calculate throughput
            time_diff = current_time - self._last_metrics_update
            tasks_processed = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            self.metrics["throughput_tasks_per_second"] = tasks_processed / time_diff if time_diff > 0 else 0
            
            self._last_metrics_update = current_time
    
    async def stop_worker(self):
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.is_running = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "target_throughput": 10000,  # 10K tasks per second target
            "performance_ratio": self.metrics["throughput_tasks_per_second"] / 10000
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        if self.admin_client:
            self.admin_client.close()


class TaskScheduler:
    """Advanced task scheduler with priority queues and load balancing."""
    
    def __init__(self, queue: KafkaTaskQueue):
        self.queue = queue
        self.scheduled_tasks: Dict[str, VisionTask] = {}
        self.is_running = False
    
    async def schedule_task(self, task: VisionTask, delay_seconds: float = 0):
        """Schedule a task for future execution."""
        if delay_seconds > 0:
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        
        self.scheduled_tasks[task.id] = task
        await self.queue.submit_task(task)
    
    async def schedule_recurring_task(self, task: VisionTask, interval_seconds: float):
        """Schedule a recurring task."""
        # This would be implemented with a separate scheduler service
        # For now, just schedule the first execution
        await self.schedule_task(task)
    
    async def start_scheduler(self):
        """Start the task scheduler."""
        self.is_running = True
        
        while self.is_running:
            # Process scheduled tasks
            current_time = datetime.utcnow()
            ready_tasks = [
                task for task in self.scheduled_tasks.values()
                if task.scheduled_at and task.scheduled_at <= current_time
            ]
            
            for task in ready_tasks:
                await self.queue.submit_task(task)
                del self.scheduled_tasks[task.id]
            
            await asyncio.sleep(1)  # Check every second
    
    async def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False


# Factory functions for easy integration
async def create_kafka_queue(bootstrap_servers: List[str] = None) -> KafkaTaskQueue:
    """Create and initialize Kafka task queue."""
    queue = KafkaTaskQueue(bootstrap_servers)
    await queue.initialize()
    return queue


async def create_task_scheduler(queue: KafkaTaskQueue) -> TaskScheduler:
    """Create task scheduler."""
    return TaskScheduler(queue)