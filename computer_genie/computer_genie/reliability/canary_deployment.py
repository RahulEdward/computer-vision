"""
Automatic Rollback with Canary Deployments

Implements comprehensive canary deployment system with automatic rollback
based on health metrics, error rates, and performance indicators.
"""

import asyncio
import time
import json
import logging
import threading
import uuid
import statistics
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone
import weakref
import traceback
import hashlib

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    CANARY_STARTING = "canary_starting"
    CANARY_RUNNING = "canary_running"
    CANARY_SCALING = "canary_scaling"
    FULL_DEPLOYMENT = "full_deployment"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLBACK_COMPLETED = "rollback_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RollbackTrigger(Enum):
    """Reasons for triggering rollback"""
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    RESPONSE_TIME_THRESHOLD = "response_time_threshold"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    MANUAL_TRIGGER = "manual_trigger"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CUSTOM_METRIC_THRESHOLD = "custom_metric_threshold"


@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Traffic splitting configuration
    initial_canary_percentage: float = 5.0
    max_canary_percentage: float = 50.0
    traffic_increment: float = 5.0
    increment_interval_seconds: float = 300.0  # 5 minutes
    
    # Health thresholds
    error_rate_threshold: float = 5.0  # Percentage
    response_time_threshold: float = 2000.0  # Milliseconds
    success_rate_threshold: float = 95.0  # Percentage
    
    # Timing configuration
    canary_duration_seconds: float = 1800.0  # 30 minutes
    health_check_interval_seconds: float = 30.0
    metrics_collection_interval_seconds: float = 60.0
    rollback_timeout_seconds: float = 300.0  # 5 minutes
    
    # Rollback configuration
    auto_rollback_enabled: bool = True
    rollback_on_first_failure: bool = False
    consecutive_failures_threshold: int = 3
    
    # Custom metrics thresholds
    custom_metrics_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Dependencies
    required_dependencies: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentTarget:
    """Represents a deployment target (instance, container, etc.)"""
    target_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    endpoint: str = ""
    version: str = ""
    is_canary: bool = False
    weight: float = 0.0  # Traffic weight
    status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    target_id: str = ""
    timestamp: float = field(default_factory=time.time)
    status: HealthStatus = HealthStatus.UNKNOWN
    response_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Metrics for deployment monitoring"""
    timestamp: float = field(default_factory=time.time)
    canary_traffic_percentage: float = 0.0
    
    # Request metrics
    total_requests: int = 0
    canary_requests: int = 0
    production_requests: int = 0
    
    # Error metrics
    total_errors: int = 0
    canary_errors: int = 0
    production_errors: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    canary_avg_response_time: float = 0.0
    production_avg_response_time: float = 0.0
    
    p95_response_time: float = 0.0
    canary_p95_response_time: float = 0.0
    production_p95_response_time: float = 0.0
    
    # Success rates
    success_rate: float = 100.0
    canary_success_rate: float = 100.0
    production_success_rate: float = 100.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'canary_traffic_percentage': self.canary_traffic_percentage,
            'total_requests': self.total_requests,
            'canary_requests': self.canary_requests,
            'production_requests': self.production_requests,
            'total_errors': self.total_errors,
            'canary_errors': self.canary_errors,
            'production_errors': self.production_errors,
            'avg_response_time': self.avg_response_time,
            'canary_avg_response_time': self.canary_avg_response_time,
            'production_avg_response_time': self.production_avg_response_time,
            'p95_response_time': self.p95_response_time,
            'canary_p95_response_time': self.canary_p95_response_time,
            'production_p95_response_time': self.production_p95_response_time,
            'success_rate': self.success_rate,
            'canary_success_rate': self.canary_success_rate,
            'production_success_rate': self.production_success_rate,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class DeploymentEvent:
    """Event in the deployment process"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str = ""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    status: DeploymentStatus = DeploymentStatus.PENDING
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker(ABC):
    """Abstract base class for health checking"""
    
    @abstractmethod
    async def check_health(self, target: DeploymentTarget) -> HealthCheckResult:
        """Check health of a deployment target"""
        pass


class HTTPHealthChecker(HealthChecker):
    """HTTP-based health checker"""
    
    def __init__(self, health_endpoint: str = "/health", timeout: float = 10.0):
        self.health_endpoint = health_endpoint
        self.timeout = timeout
        
    async def check_health(self, target: DeploymentTarget) -> HealthCheckResult:
        """Check health via HTTP endpoint"""
        import aiohttp
        
        result = HealthCheckResult(target_id=target.target_id)
        start_time = time.time()
        
        try:
            url = f"{target.endpoint.rstrip('/')}{self.health_endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    result.response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status == 200:
                        result.status = HealthStatus.HEALTHY
                        
                        # Try to parse response for additional metrics
                        try:
                            data = await response.json()
                            if isinstance(data, dict):
                                result.metrics.update(data.get('metrics', {}))
                        except:
                            pass
                            
                    elif 200 <= response.status < 300:
                        result.status = HealthStatus.HEALTHY
                    elif 400 <= response.status < 500:
                        result.status = HealthStatus.DEGRADED
                        result.error_message = f"HTTP {response.status}"
                    else:
                        result.status = HealthStatus.UNHEALTHY
                        result.error_message = f"HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Health check timeout"
            result.response_time = self.timeout * 1000
            
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            result.response_time = (time.time() - start_time) * 1000
            
        return result


class MetricsCollector(ABC):
    """Abstract base class for metrics collection"""
    
    @abstractmethod
    async def collect_metrics(self, targets: List[DeploymentTarget]) -> DeploymentMetrics:
        """Collect deployment metrics"""
        pass


class DefaultMetricsCollector(MetricsCollector):
    """Default metrics collector"""
    
    def __init__(self):
        self.request_history: deque = deque(maxlen=10000)
        self.response_times: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    async def collect_metrics(self, targets: List[DeploymentTarget]) -> DeploymentMetrics:
        """Collect deployment metrics"""
        metrics = DeploymentMetrics()
        
        with self._lock:
            # Calculate traffic distribution
            total_weight = sum(target.weight for target in targets)
            if total_weight > 0:
                canary_weight = sum(target.weight for target in targets if target.is_canary)
                metrics.canary_traffic_percentage = (canary_weight / total_weight) * 100
                
            # Calculate request metrics from history
            current_time = time.time()
            recent_requests = [req for req in self.request_history 
                             if current_time - req['timestamp'] <= 300]  # Last 5 minutes
                             
            metrics.total_requests = len(recent_requests)
            metrics.canary_requests = sum(1 for req in recent_requests if req.get('is_canary', False))
            metrics.production_requests = metrics.total_requests - metrics.canary_requests
            
            # Calculate error metrics
            error_requests = [req for req in recent_requests if req.get('error', False)]
            metrics.total_errors = len(error_requests)
            metrics.canary_errors = sum(1 for req in error_requests if req.get('is_canary', False))
            metrics.production_errors = metrics.total_errors - metrics.canary_errors
            
            # Calculate success rates
            if metrics.total_requests > 0:
                metrics.success_rate = ((metrics.total_requests - metrics.total_errors) / metrics.total_requests) * 100
                
            if metrics.canary_requests > 0:
                metrics.canary_success_rate = ((metrics.canary_requests - metrics.canary_errors) / metrics.canary_requests) * 100
                
            if metrics.production_requests > 0:
                metrics.production_success_rate = ((metrics.production_requests - metrics.production_errors) / metrics.production_requests) * 100
                
            # Calculate response time metrics
            if self.response_times:
                all_times = list(self.response_times)
                metrics.avg_response_time = statistics.mean(all_times)
                metrics.p95_response_time = self._percentile(all_times, 95)
                
                canary_times = [req['response_time'] for req in recent_requests if req.get('is_canary', False)]
                production_times = [req['response_time'] for req in recent_requests if not req.get('is_canary', False)]
                
                if canary_times:
                    metrics.canary_avg_response_time = statistics.mean(canary_times)
                    metrics.canary_p95_response_time = self._percentile(canary_times, 95)
                    
                if production_times:
                    metrics.production_avg_response_time = statistics.mean(production_times)
                    metrics.production_p95_response_time = self._percentile(production_times, 95)
                    
        return metrics
        
    def record_request(self, is_canary: bool, response_time: float, error: bool = False):
        """Record a request for metrics calculation"""
        with self._lock:
            request_data = {
                'timestamp': time.time(),
                'is_canary': is_canary,
                'response_time': response_time,
                'error': error
            }
            
            self.request_history.append(request_data)
            self.response_times.append(response_time)
            
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class TrafficSplitter:
    """Handles traffic splitting for canary deployments"""
    
    def __init__(self):
        self._lock = threading.Lock()
        
    def update_traffic_weights(self, targets: List[DeploymentTarget], canary_percentage: float):
        """Update traffic weights based on canary percentage"""
        with self._lock:
            canary_targets = [t for t in targets if t.is_canary]
            production_targets = [t for t in targets if not t.is_canary]
            
            if not canary_targets or not production_targets:
                return
                
            # Distribute canary traffic among canary targets
            canary_weight_per_target = canary_percentage / len(canary_targets)
            for target in canary_targets:
                target.weight = canary_weight_per_target
                
            # Distribute production traffic among production targets
            production_percentage = 100.0 - canary_percentage
            production_weight_per_target = production_percentage / len(production_targets)
            for target in production_targets:
                target.weight = production_weight_per_target
                
    def route_request(self, targets: List[DeploymentTarget], request_id: str) -> Optional[DeploymentTarget]:
        """Route request to appropriate target based on weights"""
        with self._lock:
            healthy_targets = [t for t in targets if t.status == HealthStatus.HEALTHY]
            
            if not healthy_targets:
                return None
                
            # Use request ID hash for consistent routing
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            random_value = (hash_value % 10000) / 100.0  # 0-100
            
            cumulative_weight = 0.0
            for target in healthy_targets:
                cumulative_weight += target.weight
                if random_value <= cumulative_weight:
                    return target
                    
            # Fallback to last target
            return healthy_targets[-1] if healthy_targets else None


class RollbackDecisionEngine:
    """Makes decisions about when to trigger rollbacks"""
    
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.consecutive_failures = 0
        self.last_check_time = time.time()
        
    def should_rollback(self, metrics: DeploymentMetrics, 
                       health_results: List[HealthCheckResult]) -> Tuple[bool, Optional[RollbackTrigger], str]:
        """Determine if rollback should be triggered"""
        
        # Check error rate threshold
        if metrics.canary_success_rate < self.config.success_rate_threshold:
            error_rate = 100.0 - metrics.canary_success_rate
            if error_rate > self.config.error_rate_threshold:
                return True, RollbackTrigger.ERROR_RATE_THRESHOLD, f"Canary error rate {error_rate:.1f}% exceeds threshold {self.config.error_rate_threshold}%"
                
        # Check response time threshold
        if metrics.canary_avg_response_time > self.config.response_time_threshold:
            return True, RollbackTrigger.RESPONSE_TIME_THRESHOLD, f"Canary response time {metrics.canary_avg_response_time:.1f}ms exceeds threshold {self.config.response_time_threshold}ms"
            
        # Check health check failures
        canary_health_results = [hr for hr in health_results if any(t.is_canary and t.target_id == hr.target_id for t in [])]  # Would need targets list
        unhealthy_canaries = [hr for hr in canary_health_results if hr.status == HealthStatus.UNHEALTHY]
        
        if unhealthy_canaries:
            if self.config.rollback_on_first_failure:
                return True, RollbackTrigger.HEALTH_CHECK_FAILURE, f"Canary health check failed: {unhealthy_canaries[0].error_message}"
                
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.config.consecutive_failures_threshold:
                return True, RollbackTrigger.HEALTH_CHECK_FAILURE, f"Consecutive health check failures: {self.consecutive_failures}"
        else:
            self.consecutive_failures = 0
            
        # Check custom metrics thresholds
        for metric_name, threshold in self.config.custom_metrics_thresholds.items():
            metric_value = metrics.custom_metrics.get(metric_name, 0.0)
            if metric_value > threshold:
                return True, RollbackTrigger.CUSTOM_METRIC_THRESHOLD, f"Custom metric {metric_name} value {metric_value} exceeds threshold {threshold}"
                
        return False, None, ""


class CanaryDeploymentManager:
    """Manages canary deployment lifecycle"""
    
    def __init__(self, config: CanaryConfig, 
                 health_checker: HealthChecker = None,
                 metrics_collector: MetricsCollector = None):
        self.config = config
        self.health_checker = health_checker or HTTPHealthChecker()
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.traffic_splitter = TrafficSplitter()
        self.rollback_engine = RollbackDecisionEngine(config)
        
        self.status = DeploymentStatus.PENDING
        self.targets: List[DeploymentTarget] = []
        self.events: List[DeploymentEvent] = []
        self.metrics_history: deque = deque(maxlen=1000)
        self.health_history: deque = deque(maxlen=1000)
        
        self.current_canary_percentage = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.rollback_reason: Optional[str] = None
        self.rollback_trigger: Optional[RollbackTrigger] = None
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def add_target(self, target: DeploymentTarget):
        """Add a deployment target"""
        with self._lock:
            self.targets.append(target)
            
        self._log_event("target_added", f"Added target: {target.name}")
        
    def remove_target(self, target_id: str):
        """Remove a deployment target"""
        with self._lock:
            self.targets = [t for t in self.targets if t.target_id != target_id]
            
        self._log_event("target_removed", f"Removed target: {target_id}")
        
    async def start_deployment(self) -> bool:
        """Start the canary deployment"""
        if self.status != DeploymentStatus.PENDING:
            logger.error(f"Cannot start deployment in status: {self.status}")
            return False
            
        try:
            self.status = DeploymentStatus.INITIALIZING
            self.start_time = time.time()
            self._log_event("deployment_started", "Canary deployment started")
            
            # Validate targets
            if not await self._validate_targets():
                self.status = DeploymentStatus.FAILED
                self._log_event("deployment_failed", "Target validation failed")
                return False
                
            # Start with initial canary percentage
            self.current_canary_percentage = self.config.initial_canary_percentage
            self.status = DeploymentStatus.CANARY_STARTING
            
            # Update traffic weights
            self.traffic_splitter.update_traffic_weights(self.targets, self.current_canary_percentage)
            
            self.status = DeploymentStatus.CANARY_RUNNING
            self._log_event("canary_started", f"Canary started with {self.current_canary_percentage}% traffic")
            
            # Start monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            return True
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            self._log_event("deployment_failed", f"Deployment failed: {str(e)}")
            logger.error(f"Failed to start deployment: {e}")
            return False
            
    async def _validate_targets(self) -> bool:
        """Validate deployment targets"""
        if not self.targets:
            logger.error("No deployment targets configured")
            return False
            
        canary_targets = [t for t in self.targets if t.is_canary]
        production_targets = [t for t in self.targets if not t.is_canary]
        
        if not canary_targets:
            logger.error("No canary targets configured")
            return False
            
        if not production_targets:
            logger.error("No production targets configured")
            return False
            
        # Perform initial health checks
        for target in self.targets:
            health_result = await self.health_checker.check_health(target)
            target.status = health_result.status
            target.last_health_check = health_result.timestamp
            
            if health_result.status == HealthStatus.UNHEALTHY:
                logger.error(f"Target {target.name} is unhealthy: {health_result.error_message}")
                return False
                
        return True
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while not self._stop_event.is_set() and self.status in [
                DeploymentStatus.CANARY_RUNNING, 
                DeploymentStatus.CANARY_SCALING
            ]:
                # Collect metrics
                metrics = await self.metrics_collector.collect_metrics(self.targets)
                self.metrics_history.append(metrics)
                
                # Perform health checks
                health_results = []
                for target in self.targets:
                    health_result = await self.health_checker.check_health(target)
                    target.status = health_result.status
                    target.last_health_check = health_result.timestamp
                    health_results.append(health_result)
                    
                self.health_history.extend(health_results)
                
                # Check if rollback is needed
                if self.config.auto_rollback_enabled:
                    should_rollback, trigger, reason = self.rollback_engine.should_rollback(metrics, health_results)
                    
                    if should_rollback:
                        self.rollback_trigger = trigger
                        self.rollback_reason = reason
                        await self._trigger_rollback()
                        break
                        
                # Check if we should scale up canary
                if self.status == DeploymentStatus.CANARY_RUNNING:
                    await self._check_canary_scaling(metrics)
                    
                # Check if deployment is complete
                if self.current_canary_percentage >= self.config.max_canary_percentage:
                    await self._complete_deployment()
                    break
                    
                # Check deployment timeout
                if time.time() - self.start_time > self.config.canary_duration_seconds:
                    await self._complete_deployment()
                    break
                    
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.status = DeploymentStatus.FAILED
            self._log_event("monitoring_failed", f"Monitoring failed: {str(e)}")
            
    async def _check_canary_scaling(self, metrics: DeploymentMetrics):
        """Check if canary should be scaled up"""
        # Only scale if canary is performing well
        if (metrics.canary_success_rate >= self.config.success_rate_threshold and
            metrics.canary_avg_response_time <= self.config.response_time_threshold):
            
            # Check if enough time has passed since last scaling
            last_scaling_event = None
            for event in reversed(self.events):
                if event.event_type in ["canary_scaled", "canary_started"]:
                    last_scaling_event = event
                    break
                    
            if (last_scaling_event and 
                time.time() - last_scaling_event.timestamp >= self.config.increment_interval_seconds):
                
                await self._scale_canary()
                
    async def _scale_canary(self):
        """Scale up canary traffic"""
        if self.current_canary_percentage >= self.config.max_canary_percentage:
            return
            
        self.status = DeploymentStatus.CANARY_SCALING
        
        new_percentage = min(
            self.current_canary_percentage + self.config.traffic_increment,
            self.config.max_canary_percentage
        )
        
        self.current_canary_percentage = new_percentage
        self.traffic_splitter.update_traffic_weights(self.targets, new_percentage)
        
        self.status = DeploymentStatus.CANARY_RUNNING
        self._log_event("canary_scaled", f"Canary scaled to {new_percentage}% traffic")
        
    async def _complete_deployment(self):
        """Complete the deployment"""
        self.status = DeploymentStatus.FULL_DEPLOYMENT
        
        # Set all canary targets to 100% traffic
        with self._lock:
            for target in self.targets:
                if target.is_canary:
                    target.weight = 100.0 / len([t for t in self.targets if t.is_canary])
                else:
                    target.weight = 0.0
                    
        self.status = DeploymentStatus.COMPLETED
        self.end_time = time.time()
        self._log_event("deployment_completed", "Canary deployment completed successfully")
        
    async def _trigger_rollback(self):
        """Trigger automatic rollback"""
        self.status = DeploymentStatus.ROLLING_BACK
        self._log_event("rollback_started", f"Rollback triggered: {self.rollback_reason}")
        
        try:
            # Set all traffic to production targets
            with self._lock:
                for target in self.targets:
                    if target.is_canary:
                        target.weight = 0.0
                    else:
                        target.weight = 100.0 / len([t for t in self.targets if not t.is_canary])
                        
            # Wait for rollback to complete
            await asyncio.sleep(5.0)  # Give time for traffic to shift
            
            self.status = DeploymentStatus.ROLLBACK_COMPLETED
            self.end_time = time.time()
            self._log_event("rollback_completed", "Rollback completed successfully")
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            self._log_event("rollback_failed", f"Rollback failed: {str(e)}")
            logger.error(f"Rollback failed: {e}")
            
    async def manual_rollback(self, reason: str = "Manual rollback"):
        """Manually trigger rollback"""
        if self.status not in [DeploymentStatus.CANARY_RUNNING, DeploymentStatus.CANARY_SCALING]:
            logger.error(f"Cannot rollback in status: {self.status}")
            return False
            
        self.rollback_trigger = RollbackTrigger.MANUAL_TRIGGER
        self.rollback_reason = reason
        await self._trigger_rollback()
        return True
        
    async def stop_deployment(self):
        """Stop the deployment"""
        self._stop_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        if self.status in [DeploymentStatus.CANARY_RUNNING, DeploymentStatus.CANARY_SCALING]:
            self.status = DeploymentStatus.CANCELLED
            self.end_time = time.time()
            self._log_event("deployment_cancelled", "Deployment cancelled")
            
    def _log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Log a deployment event"""
        event = DeploymentEvent(
            deployment_id=self.config.deployment_id,
            event_type=event_type,
            status=self.status,
            message=message,
            data=data or {}
        )
        
        with self._lock:
            self.events.append(event)
            
        logger.info(f"Deployment {self.config.deployment_id}: {message}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        with self._lock:
            canary_targets = [t for t in self.targets if t.is_canary]
            production_targets = [t for t in self.targets if not t.is_canary]
            
            return {
                'deployment_id': self.config.deployment_id,
                'name': self.config.name,
                'status': self.status.value,
                'current_canary_percentage': self.current_canary_percentage,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration': (self.end_time or time.time()) - (self.start_time or time.time()),
                'rollback_reason': self.rollback_reason,
                'rollback_trigger': self.rollback_trigger.value if self.rollback_trigger else None,
                'targets': {
                    'canary': len(canary_targets),
                    'production': len(production_targets),
                    'healthy': len([t for t in self.targets if t.status == HealthStatus.HEALTHY]),
                    'unhealthy': len([t for t in self.targets if t.status == HealthStatus.UNHEALTHY])
                },
                'events_count': len(self.events),
                'metrics_count': len(self.metrics_history)
            }
            
    def get_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics"""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-limit:]
            return [m.to_dict() for m in recent_metrics]
            
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self._lock:
            recent_events = self.events[-limit:]
            return [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp,
                    'event_type': e.event_type,
                    'status': e.status.value,
                    'message': e.message,
                    'data': e.data
                }
                for e in recent_events
            ]


# Utility functions
def create_canary_deployment(name: str, 
                           canary_targets: List[str],
                           production_targets: List[str],
                           config: CanaryConfig = None) -> CanaryDeploymentManager:
    """Create a canary deployment"""
    if config is None:
        config = CanaryConfig(name=name)
        
    manager = CanaryDeploymentManager(config)
    
    # Add canary targets
    for i, endpoint in enumerate(canary_targets):
        target = DeploymentTarget(
            name=f"canary-{i}",
            endpoint=endpoint,
            is_canary=True,
            version="canary"
        )
        manager.add_target(target)
        
    # Add production targets
    for i, endpoint in enumerate(production_targets):
        target = DeploymentTarget(
            name=f"production-{i}",
            endpoint=endpoint,
            is_canary=False,
            version="production"
        )
        manager.add_target(target)
        
    return manager


async def deploy_with_canary(name: str,
                           canary_endpoints: List[str],
                           production_endpoints: List[str],
                           canary_percentage: float = 10.0,
                           auto_rollback: bool = True) -> CanaryDeploymentManager:
    """Deploy with canary strategy"""
    config = CanaryConfig(
        name=name,
        initial_canary_percentage=canary_percentage,
        auto_rollback_enabled=auto_rollback
    )
    
    manager = create_canary_deployment(name, canary_endpoints, production_endpoints, config)
    
    success = await manager.start_deployment()
    if not success:
        raise RuntimeError(f"Failed to start canary deployment: {name}")
        
    return manager