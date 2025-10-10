"""
Intelligent batching system for grouping similar operations for bulk processing.
Optimizes throughput by reducing overhead and enabling vectorized operations.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be batched."""
    SCREENSHOT = "screenshot"
    ELEMENT_DETECTION = "element_detection"
    OCR = "ocr"
    IMAGE_PROCESSING = "image_processing"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    CUSTOM = "custom"


class BatchStrategy(Enum):
    """Batching strategies."""
    TIME_BASED = "time_based"          # Batch by time window
    SIZE_BASED = "size_based"          # Batch by number of operations
    SIMILARITY_BASED = "similarity_based"  # Batch by operation similarity
    ADAPTIVE = "adaptive"              # Adaptive strategy based on performance
    PRIORITY_BASED = "priority_based"  # Batch by priority levels


class Priority(Enum):
    """Operation priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class BatchableOperation:
    """Represents an operation that can be batched."""
    id: str
    operation_type: OperationType
    priority: Priority
    parameters: Dict[str, Any]
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = None
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    similarity_hash: Optional[str] = None
    estimated_duration_ms: float = 100.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate similarity hash after initialization."""
        if self.similarity_hash is None:
            self.similarity_hash = self._calculate_similarity_hash()
    
    def _calculate_similarity_hash(self) -> str:
        """Calculate hash for operation similarity."""
        # Create a hash based on operation type and key parameters
        hash_data = f"{self.operation_type.value}"
        
        # Add relevant parameters for similarity comparison
        if self.operation_type == OperationType.ELEMENT_DETECTION:
            hash_data += f"_{self.parameters.get('selector', '')}"
            hash_data += f"_{self.parameters.get('detection_method', '')}"
        elif self.operation_type == OperationType.IMAGE_PROCESSING:
            hash_data += f"_{self.parameters.get('operation', '')}"
            hash_data += f"_{self.parameters.get('kernel_size', '')}"
        elif self.operation_type == OperationType.OCR:
            hash_data += f"_{self.parameters.get('language', '')}"
            hash_data += f"_{self.parameters.get('mode', '')}"
        
        return hashlib.md5(hash_data.encode()).hexdigest()[:8]
    
    def is_similar_to(self, other: 'BatchableOperation') -> bool:
        """Check if this operation is similar to another."""
        if self.operation_type != other.operation_type:
            return False
        
        if self.similarity_hash == other.similarity_hash:
            return True
        
        # Additional similarity checks based on operation type
        if self.operation_type == OperationType.ELEMENT_DETECTION:
            return (
                self.parameters.get('detection_method') == other.parameters.get('detection_method') and
                self.parameters.get('confidence_threshold') == other.parameters.get('confidence_threshold')
            )
        elif self.operation_type == OperationType.IMAGE_PROCESSING:
            return (
                self.parameters.get('operation') == other.parameters.get('operation') and
                self.parameters.get('kernel_size') == other.parameters.get('kernel_size')
            )
        
        return False
    
    def can_batch_with(self, other: 'BatchableOperation') -> bool:
        """Check if this operation can be batched with another."""
        # Check basic compatibility
        if not self.is_similar_to(other):
            return False
        
        # Check priority compatibility (only batch similar priorities)
        priority_diff = abs(self.priority.value - other.priority.value)
        if priority_diff > 1:
            return False
        
        # Check timeout compatibility
        time_diff = abs(self.timestamp - other.timestamp)
        min_timeout = min(self.timeout, other.timeout)
        if time_diff > min_timeout * 0.5:  # Don't batch if time difference is too large
            return False
        
        return True


@dataclass
class Batch:
    """Represents a batch of similar operations."""
    id: str
    operations: List[BatchableOperation] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_size: int = 10
    max_wait_time_ms: float = 100.0
    estimated_duration_ms: float = 0.0
    
    def add_operation(self, operation: BatchableOperation) -> bool:
        """Add operation to batch if compatible."""
        if len(self.operations) >= self.max_size:
            return False
        
        if self.operations and not self.operations[0].can_batch_with(operation):
            return False
        
        self.operations.append(operation)
        self.estimated_duration_ms += operation.estimated_duration_ms
        return True
    
    def is_ready(self) -> bool:
        """Check if batch is ready for execution."""
        if not self.operations:
            return False
        
        # Check size threshold
        if len(self.operations) >= self.max_size:
            return True
        
        # Check time threshold
        elapsed_ms = (time.time() - self.created_at) * 1000
        if elapsed_ms >= self.max_wait_time_ms:
            return True
        
        # Check if any operation is about to timeout
        current_time = time.time()
        for op in self.operations:
            time_since_created = current_time - op.timestamp
            if time_since_created >= op.timeout * 0.8:  # 80% of timeout
                return True
        
        return False
    
    def get_priority(self) -> Priority:
        """Get the highest priority in the batch."""
        if not self.operations:
            return Priority.LOW
        
        return min(op.priority for op in self.operations)
    
    def get_operation_type(self) -> Optional[OperationType]:
        """Get the operation type of the batch."""
        if not self.operations:
            return None
        return self.operations[0].operation_type


class BatchPerformanceAnalyzer:
    """Analyzes batch performance and optimizes parameters."""
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "avg_batch_size": 0.0,
            "avg_execution_time_ms": 0.0,
            "avg_wait_time_ms": 0.0,
            "throughput_ops_per_sec": 0.0,
            "efficiency_score": 0.0
        }
        
        # Adaptive parameters
        self.optimal_batch_sizes: Dict[OperationType, int] = defaultdict(lambda: 5)
        self.optimal_wait_times: Dict[OperationType, float] = defaultdict(lambda: 50.0)
    
    def record_execution(self, batch: Batch, execution_time_ms: float, success_rate: float):
        """Record batch execution for analysis."""
        wait_time_ms = (time.time() - batch.created_at) * 1000
        
        record = {
            "batch_id": batch.id,
            "operation_type": batch.get_operation_type(),
            "batch_size": len(batch.operations),
            "execution_time_ms": execution_time_ms,
            "wait_time_ms": wait_time_ms,
            "success_rate": success_rate,
            "efficiency": len(batch.operations) / (execution_time_ms + wait_time_ms),
            "timestamp": time.time()
        }
        
        self.execution_history.append(record)
        
        # Keep only recent history (last 1000 executions)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        self._update_metrics()
        self._update_optimal_parameters()
    
    def _update_metrics(self):
        """Update performance metrics."""
        if not self.execution_history:
            return
        
        recent_history = self.execution_history[-100:]  # Last 100 executions
        
        self.performance_metrics["avg_batch_size"] = np.mean([r["batch_size"] for r in recent_history])
        self.performance_metrics["avg_execution_time_ms"] = np.mean([r["execution_time_ms"] for r in recent_history])
        self.performance_metrics["avg_wait_time_ms"] = np.mean([r["wait_time_ms"] for r in recent_history])
        
        # Calculate throughput
        total_ops = sum(r["batch_size"] for r in recent_history)
        total_time_s = sum(r["execution_time_ms"] + r["wait_time_ms"] for r in recent_history) / 1000
        self.performance_metrics["throughput_ops_per_sec"] = total_ops / max(total_time_s, 1)
        
        # Calculate efficiency score
        avg_efficiency = np.mean([r["efficiency"] for r in recent_history])
        avg_success_rate = np.mean([r["success_rate"] for r in recent_history])
        self.performance_metrics["efficiency_score"] = avg_efficiency * avg_success_rate
    
    def _update_optimal_parameters(self):
        """Update optimal batching parameters based on performance."""
        if len(self.execution_history) < 10:
            return
        
        # Group by operation type
        by_type = defaultdict(list)
        for record in self.execution_history[-100:]:
            if record["operation_type"]:
                by_type[record["operation_type"]].append(record)
        
        for op_type, records in by_type.items():
            if len(records) < 5:
                continue
            
            # Find optimal batch size (maximize efficiency)
            batch_sizes = [r["batch_size"] for r in records]
            efficiencies = [r["efficiency"] for r in records]
            
            # Group by batch size and calculate average efficiency
            size_efficiency = defaultdict(list)
            for size, eff in zip(batch_sizes, efficiencies):
                size_efficiency[size].append(eff)
            
            best_size = max(size_efficiency.keys(), 
                          key=lambda s: np.mean(size_efficiency[s]))
            self.optimal_batch_sizes[op_type] = best_size
            
            # Find optimal wait time (balance between latency and throughput)
            wait_times = [r["wait_time_ms"] for r in records]
            
            # Target wait time that maximizes throughput while keeping latency reasonable
            optimal_wait = np.percentile(wait_times, 75)  # 75th percentile
            self.optimal_wait_times[op_type] = min(optimal_wait, 200.0)  # Cap at 200ms
    
    def get_optimal_batch_size(self, operation_type: OperationType) -> int:
        """Get optimal batch size for operation type."""
        return self.optimal_batch_sizes[operation_type]
    
    def get_optimal_wait_time(self, operation_type: OperationType) -> float:
        """Get optimal wait time for operation type."""
        return self.optimal_wait_times[operation_type]
    
    def should_force_execution(self, batch: Batch) -> bool:
        """Determine if batch should be forced to execute."""
        if not batch.operations:
            return False
        
        # Force execution if any operation is critical priority
        if any(op.priority == Priority.CRITICAL for op in batch.operations):
            return True
        
        # Force execution if wait time exceeds optimal threshold
        wait_time_ms = (time.time() - batch.created_at) * 1000
        optimal_wait = self.get_optimal_wait_time(batch.get_operation_type())
        
        return wait_time_ms >= optimal_wait * 1.5


class IntelligentBatcher:
    """Intelligent batching system for operation optimization."""
    
    def __init__(self, 
                 default_strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
                 max_concurrent_batches: int = 10,
                 enable_performance_analysis: bool = True):
        
        self.default_strategy = default_strategy
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_performance_analysis = enable_performance_analysis
        
        # Batch management
        self.pending_operations: deque = deque()
        self.active_batches: Dict[str, Batch] = {}
        self.executing_batches: Set[str] = set()
        
        # Performance analysis
        self.performance_analyzer = BatchPerformanceAnalyzer() if enable_performance_analysis else None
        
        # Operation handlers
        self.operation_handlers: Dict[OperationType, Callable] = {}
        
        # Statistics
        self.stats = {
            "operations_submitted": 0,
            "operations_completed": 0,
            "batches_created": 0,
            "batches_executed": 0,
            "avg_batch_size": 0.0,
            "total_processing_time_ms": 0.0,
            "avg_latency_ms": 0.0
        }
        
        # Background task for batch processing
        self._batch_processor_task = None
        self._running = False
    
    async def start(self):
        """Start the batch processor."""
        if self._running:
            return
        
        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info("Intelligent Batcher started")
    
    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Execute remaining batches
        await self._execute_all_pending_batches()
        logger.info("Intelligent Batcher stopped")
    
    def register_handler(self, operation_type: OperationType, handler: Callable):
        """Register handler for operation type."""
        self.operation_handlers[operation_type] = handler
    
    async def submit_operation(self, 
                             operation_type: OperationType,
                             parameters: Dict[str, Any],
                             priority: Priority = Priority.MEDIUM,
                             timeout: float = 30.0) -> Any:
        """Submit operation for batched execution."""
        
        operation_id = f"{operation_type.value}_{int(time.time() * 1000000)}"
        future = asyncio.Future()
        
        operation = BatchableOperation(
            id=operation_id,
            operation_type=operation_type,
            priority=priority,
            parameters=parameters,
            future=future,
            timeout=timeout
        )
        
        self.pending_operations.append(operation)
        self.stats["operations_submitted"] += 1
        
        # Trigger immediate processing for critical operations
        if priority == Priority.CRITICAL:
            asyncio.create_task(self._process_pending_operations())
        
        return await future
    
    async def _batch_processor_loop(self):
        """Main batch processing loop."""
        while self._running:
            try:
                await self._process_pending_operations()
                await self._check_ready_batches()
                await asyncio.sleep(0.01)  # 10ms processing interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_pending_operations(self):
        """Process pending operations and create batches."""
        if not self.pending_operations:
            return
        
        # Group operations by similarity and priority
        operation_groups = self._group_operations()
        
        for group in operation_groups:
            if len(self.active_batches) >= self.max_concurrent_batches:
                break
            
            batch = self._create_batch_for_group(group)
            if batch:
                self.active_batches[batch.id] = batch
                self.stats["batches_created"] += 1
    
    def _group_operations(self) -> List[List[BatchableOperation]]:
        """Group pending operations by similarity."""
        if not self.pending_operations:
            return []
        
        # Convert deque to list for processing
        operations = list(self.pending_operations)
        self.pending_operations.clear()
        
        # Sort by priority and timestamp
        operations.sort(key=lambda op: (op.priority.value, op.timestamp))
        
        groups = []
        used_operations = set()
        
        for i, op in enumerate(operations):
            if op.id in used_operations:
                continue
            
            # Start new group with this operation
            group = [op]
            used_operations.add(op.id)
            
            # Find similar operations to add to group
            for j, other_op in enumerate(operations[i+1:], i+1):
                if other_op.id in used_operations:
                    continue
                
                if op.can_batch_with(other_op) and len(group) < self._get_max_batch_size(op.operation_type):
                    group.append(other_op)
                    used_operations.add(other_op.id)
            
            groups.append(group)
        
        return groups
    
    def _create_batch_for_group(self, operations: List[BatchableOperation]) -> Optional[Batch]:
        """Create batch for operation group."""
        if not operations:
            return None
        
        batch_id = f"batch_{int(time.time() * 1000000)}"
        op_type = operations[0].operation_type
        
        batch = Batch(
            id=batch_id,
            strategy=self.default_strategy,
            max_size=self._get_max_batch_size(op_type),
            max_wait_time_ms=self._get_max_wait_time(op_type)
        )
        
        for operation in operations:
            if not batch.add_operation(operation):
                # Return remaining operations to pending queue
                remaining_ops = operations[operations.index(operation):]
                self.pending_operations.extendleft(reversed(remaining_ops))
                break
        
        return batch if batch.operations else None
    
    async def _check_ready_batches(self):
        """Check and execute ready batches."""
        ready_batches = []
        
        for batch_id, batch in list(self.active_batches.items()):
            if batch_id in self.executing_batches:
                continue
            
            should_execute = (
                batch.is_ready() or
                (self.performance_analyzer and self.performance_analyzer.should_force_execution(batch))
            )
            
            if should_execute:
                ready_batches.append(batch)
                del self.active_batches[batch_id]
        
        # Execute ready batches
        for batch in ready_batches:
            asyncio.create_task(self._execute_batch(batch))
    
    async def _execute_batch(self, batch: Batch):
        """Execute a batch of operations."""
        batch_id = batch.id
        self.executing_batches.add(batch_id)
        
        start_time = time.time()
        successful_operations = 0
        
        try:
            operation_type = batch.get_operation_type()
            if operation_type not in self.operation_handlers:
                raise ValueError(f"No handler registered for {operation_type}")
            
            handler = self.operation_handlers[operation_type]
            
            # Prepare batch parameters
            batch_parameters = [op.parameters for op in batch.operations]
            
            # Execute batch
            results = await handler(batch_parameters)
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results] * len(batch.operations)
            
            # Set results for each operation
            for operation, result in zip(batch.operations, results):
                if operation.future and not operation.future.done():
                    operation.future.set_result(result)
                    successful_operations += 1
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            
            # Set exception for all operations
            for operation in batch.operations:
                if operation.future and not operation.future.done():
                    operation.future.set_exception(e)
        
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            success_rate = successful_operations / len(batch.operations) if batch.operations else 0
            
            # Update statistics
            self.stats["batches_executed"] += 1
            self.stats["operations_completed"] += successful_operations
            self.stats["total_processing_time_ms"] += execution_time_ms
            
            # Update averages
            if self.stats["batches_executed"] > 0:
                self.stats["avg_batch_size"] = (
                    self.stats["operations_completed"] / self.stats["batches_executed"]
                )
            
            # Record performance
            if self.performance_analyzer:
                self.performance_analyzer.record_execution(batch, execution_time_ms, success_rate)
            
            # Calculate latency
            for operation in batch.operations:
                latency_ms = (time.time() - operation.timestamp) * 1000
                current_avg = self.stats["avg_latency_ms"]
                count = self.stats["operations_completed"]
                self.stats["avg_latency_ms"] = ((current_avg * (count - 1)) + latency_ms) / count
            
            self.executing_batches.discard(batch_id)
    
    async def _execute_all_pending_batches(self):
        """Execute all pending batches during shutdown."""
        # Process any remaining pending operations
        await self._process_pending_operations()
        
        # Execute all active batches
        execution_tasks = []
        for batch in list(self.active_batches.values()):
            if batch.id not in self.executing_batches:
                execution_tasks.append(self._execute_batch(batch))
        
        if execution_tasks:
            await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        self.active_batches.clear()
    
    def _get_max_batch_size(self, operation_type: OperationType) -> int:
        """Get maximum batch size for operation type."""
        if self.performance_analyzer:
            return self.performance_analyzer.get_optimal_batch_size(operation_type)
        
        # Default batch sizes
        default_sizes = {
            OperationType.SCREENSHOT: 3,
            OperationType.ELEMENT_DETECTION: 10,
            OperationType.OCR: 5,
            OperationType.IMAGE_PROCESSING: 8,
            OperationType.CLICK: 5,
            OperationType.TYPE: 3,
            OperationType.SCROLL: 2,
            OperationType.WAIT: 1,
            OperationType.CUSTOM: 5
        }
        
        return default_sizes.get(operation_type, 5)
    
    def _get_max_wait_time(self, operation_type: OperationType) -> float:
        """Get maximum wait time for operation type."""
        if self.performance_analyzer:
            return self.performance_analyzer.get_optimal_wait_time(operation_type)
        
        # Default wait times (ms)
        default_wait_times = {
            OperationType.SCREENSHOT: 50.0,
            OperationType.ELEMENT_DETECTION: 100.0,
            OperationType.OCR: 150.0,
            OperationType.IMAGE_PROCESSING: 200.0,
            OperationType.CLICK: 30.0,
            OperationType.TYPE: 50.0,
            OperationType.SCROLL: 30.0,
            OperationType.WAIT: 10.0,
            OperationType.CUSTOM: 100.0
        }
        
        return default_wait_times.get(operation_type, 100.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get batcher performance statistics."""
        stats = {
            **self.stats,
            "active_batches": len(self.active_batches),
            "executing_batches": len(self.executing_batches),
            "pending_operations": len(self.pending_operations),
            "throughput_ops_per_sec": 0.0,
            "efficiency_score": 0.0
        }
        
        # Calculate throughput
        if self.stats["total_processing_time_ms"] > 0:
            stats["throughput_ops_per_sec"] = (
                self.stats["operations_completed"] / 
                (self.stats["total_processing_time_ms"] / 1000)
            )
        
        # Calculate efficiency score
        if self.stats["operations_submitted"] > 0:
            completion_rate = self.stats["operations_completed"] / self.stats["operations_submitted"]
            latency_score = max(0, 1 - (self.stats["avg_latency_ms"] / 1000))  # Penalize high latency
            stats["efficiency_score"] = completion_rate * latency_score
        
        # Add performance analyzer stats
        if self.performance_analyzer:
            stats["performance_analysis"] = self.performance_analyzer.performance_metrics
            stats["optimal_parameters"] = {
                "batch_sizes": dict(self.performance_analyzer.optimal_batch_sizes),
                "wait_times": dict(self.performance_analyzer.optimal_wait_times)
            }
        
        return stats
    
    async def benchmark_batching(self, operation_type: OperationType, 
                                test_operations: List[Dict[str, Any]], 
                                iterations: int = 5) -> Dict[str, Any]:
        """Benchmark batching performance."""
        
        # Mock handler for testing
        async def mock_handler(batch_params: List[Dict[str, Any]]) -> List[Any]:
            # Simulate processing time based on batch size
            processing_time = len(batch_params) * 0.01  # 10ms per operation
            await asyncio.sleep(processing_time)
            return [f"result_{i}" for i in range(len(batch_params))]
        
        # Register mock handler
        original_handler = self.operation_handlers.get(operation_type)
        self.operation_handlers[operation_type] = mock_handler
        
        try:
            # Benchmark with batching
            batched_times = []
            for _ in range(iterations):
                start_time = time.time()
                
                tasks = []
                for params in test_operations:
                    task = self.submit_operation(operation_type, params)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                batched_times.append((time.time() - start_time) * 1000)
            
            # Benchmark without batching (sequential)
            sequential_times = []
            for _ in range(iterations):
                start_time = time.time()
                
                for params in test_operations:
                    await mock_handler([params])
                
                sequential_times.append((time.time() - start_time) * 1000)
            
            # Calculate results
            avg_batched_time = np.mean(batched_times)
            avg_sequential_time = np.mean(sequential_times)
            speedup = avg_sequential_time / avg_batched_time
            
            return {
                "test_operations_count": len(test_operations),
                "iterations": iterations,
                "batched_performance": {
                    "avg_time_ms": avg_batched_time,
                    "min_time_ms": np.min(batched_times),
                    "max_time_ms": np.max(batched_times)
                },
                "sequential_performance": {
                    "avg_time_ms": avg_sequential_time,
                    "min_time_ms": np.min(sequential_times),
                    "max_time_ms": np.max(sequential_times)
                },
                "improvement": {
                    "speedup_factor": speedup,
                    "efficiency_gain_percent": ((speedup - 1) * 100),
                    "target_achieved": speedup >= 2.0  # Target: 2x speedup
                },
                "current_stats": self.get_performance_stats()
            }
        
        finally:
            # Restore original handler
            if original_handler:
                self.operation_handlers[operation_type] = original_handler
            else:
                self.operation_handlers.pop(operation_type, None)


# Factory function for easy integration
def create_intelligent_batcher(strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
                             max_concurrent_batches: int = 10) -> IntelligentBatcher:
    """Create intelligent batcher instance."""
    return IntelligentBatcher(strategy, max_concurrent_batches)