"""
Adaptive Circuit Breaker Pattern Implementation

Provides intelligent circuit breaking with adaptive thresholds based on system load,
preventing cascade failures and ensuring system stability.
"""

import asyncio
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import deque
import psutil
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0
    adaptive_threshold: bool = True
    load_factor_weight: float = 0.3
    min_requests: int = 10


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    load_average: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CallResult:
    """Result of a circuit breaker call"""
    success: bool
    duration: float
    error: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)


class SystemLoadMonitor:
    """Monitors system load and provides adaptive thresholds"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self, interval: float):
        """Continuous monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
                
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Calculate load average (Windows approximation)
            load_avg = cpu_percent / 100.0
            
            # Calculate disk and network utilization
            disk_percent = 0.0
            network_percent = 0.0
            
            if hasattr(psutil, 'disk_usage'):
                disk_usage = psutil.disk_usage('/')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_percent=disk_percent,
                network_io_percent=network_percent,
                load_average=load_avg
            )
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0)
            
    def get_current_load_factor(self) -> float:
        """Calculate current system load factor (0.0 to 1.0)"""
        with self._lock:
            if not self.metrics_history:
                return 0.5  # Default moderate load
                
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
            
            # Weighted average of different metrics
            cpu_weight = 0.4
            memory_weight = 0.3
            disk_weight = 0.2
            network_weight = 0.1
            
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_io_percent for m in recent_metrics) / len(recent_metrics)
            avg_network = sum(m.network_io_percent for m in recent_metrics) / len(recent_metrics)
            
            load_factor = (
                (avg_cpu / 100.0) * cpu_weight +
                (avg_memory / 100.0) * memory_weight +
                (avg_disk / 100.0) * disk_weight +
                (avg_network / 100.0) * network_weight
            )
            
            return min(1.0, max(0.0, load_factor))
            
    def get_adaptive_threshold(self, base_threshold: int) -> int:
        """Calculate adaptive threshold based on system load"""
        load_factor = self.get_current_load_factor()
        
        # Lower threshold when system is under high load
        # Higher threshold when system is performing well
        if load_factor > 0.8:  # High load
            multiplier = 0.5
        elif load_factor > 0.6:  # Medium-high load
            multiplier = 0.7
        elif load_factor > 0.4:  # Medium load
            multiplier = 1.0
        elif load_factor > 0.2:  # Low-medium load
            multiplier = 1.3
        else:  # Low load
            multiplier = 1.5
            
        return max(1, int(base_threshold * multiplier))


class AdaptiveCircuitBreaker:
    """Adaptive circuit breaker with system load awareness"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig, 
                 load_monitor: Optional[SystemLoadMonitor] = None):
        self.name = name
        self.config = config
        self.load_monitor = load_monitor or SystemLoadMonitor()
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.call_history: deque = deque(maxlen=1000)
        
        self._lock = threading.Lock()
        self._state_change_callbacks: List[Callable] = []
        
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self._state_change_callbacks.append(callback)
        
    def _notify_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        """Notify callbacks of state change"""
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
                
    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state"""
        old_state = self.state
        self.state = new_state
        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
        self._notify_state_change(old_state, new_state)
        
    def _get_effective_threshold(self) -> int:
        """Get effective failure threshold based on system load"""
        if self.config.adaptive_threshold and self.load_monitor:
            return self.load_monitor.get_adaptive_threshold(self.config.failure_threshold)
        return self.config.failure_threshold
        
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.state == CircuitBreakerState.OPEN and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
        
    def _record_success(self, duration: float):
        """Record successful call"""
        with self._lock:
            result = CallResult(success=True, duration=duration)
            self.call_history.append(result)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitBreakerState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                
    def _record_failure(self, error: Exception, duration: float):
        """Record failed call"""
        with self._lock:
            result = CallResult(success=False, duration=duration, error=error)
            self.call_history.append(result)
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            effective_threshold = self._get_effective_threshold()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= effective_threshold:
                    self._change_state(CircuitBreakerState.OPEN)
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._change_state(CircuitBreakerState.OPEN)
                self.success_count = 0
                
    @asynccontextmanager
    async def call(self):
        """Context manager for circuit breaker calls"""
        # Check if we should attempt reset
        if self._should_attempt_reset():
            with self._lock:
                if self.state == CircuitBreakerState.OPEN:
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                    
        # Check current state
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
            
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self._record_success(duration)
        except Exception as e:
            duration = time.time() - start_time
            self._record_failure(e, duration)
            raise
            
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        async with self.call():
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection"""
        # For sync calls, we need to handle the async context manager differently
        if self._should_attempt_reset():
            with self._lock:
                if self.state == CircuitBreakerState.OPEN:
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                    
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
            
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self._record_success(duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self._record_failure(e, duration)
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            recent_calls = list(self.call_history)[-100:]  # Last 100 calls
            
            if recent_calls:
                success_rate = sum(1 for call in recent_calls if call.success) / len(recent_calls)
                avg_duration = sum(call.duration for call in recent_calls) / len(recent_calls)
            else:
                success_rate = 1.0
                avg_duration = 0.0
                
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'success_rate': success_rate,
                'avg_duration': avg_duration,
                'total_calls': len(self.call_history),
                'effective_threshold': self._get_effective_threshold(),
                'system_load_factor': self.load_monitor.get_current_load_factor() if self.load_monitor else 0.0
            }


class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self, load_monitor: Optional[SystemLoadMonitor] = None):
        self.load_monitor = load_monitor or SystemLoadMonitor()
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self._lock = threading.Lock()
        
    async def start(self):
        """Start the circuit breaker manager"""
        await self.load_monitor.start_monitoring()
        
    async def stop(self):
        """Stop the circuit breaker manager"""
        await self.load_monitor.stop_monitoring()
        
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> AdaptiveCircuitBreaker:
        """Create a new circuit breaker"""
        if config is None:
            config = CircuitBreakerConfig()
            
        with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
                
            circuit_breaker = AdaptiveCircuitBreaker(name, config, self.load_monitor)
            self.circuit_breakers[name] = circuit_breaker
            return circuit_breaker
            
    def get_circuit_breaker(self, name: str) -> Optional[AdaptiveCircuitBreaker]:
        """Get existing circuit breaker"""
        return self.circuit_breakers.get(name)
        
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove circuit breaker"""
        with self._lock:
            return self.circuit_breakers.pop(name, None) is not None
            
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
        
    def reset_all(self):
        """Reset all circuit breakers to closed state"""
        with self._lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker._change_state(CircuitBreakerState.CLOSED)
                circuit_breaker.failure_count = 0
                circuit_breaker.success_count = 0


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None, 
                   manager: Optional[CircuitBreakerManager] = None):
    """Decorator to add circuit breaker protection to functions"""
    def decorator(func):
        nonlocal manager
        if manager is None:
            manager = CircuitBreakerManager()
            
        cb = manager.create_circuit_breaker(name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await cb.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return cb.call_sync(func, *args, **kwargs)
            return sync_wrapper
            
    return decorator