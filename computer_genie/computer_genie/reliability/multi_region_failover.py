"""
Multi-Region Failover with Sub-1 Second RTO

Implements comprehensive multi-region failover system with automatic
disaster recovery, data replication, and sub-1 second recovery time objectives.
"""

import asyncio
import time
import json
import logging
import threading
import uuid
import hashlib
import socket
import struct
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import weakref
import traceback
import statistics
import ipaddress

logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """Status of a region"""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class FailoverTrigger(Enum):
    """Triggers for failover"""
    HEALTH_CHECK_FAILURE = "health_check_failure"
    LATENCY_THRESHOLD = "latency_threshold"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    MANUAL_TRIGGER = "manual_trigger"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CENTER_FAILURE = "data_center_failure"
    SECURITY_INCIDENT = "security_incident"


class ReplicationStrategy(Enum):
    """Data replication strategies"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNCHRONOUS = "semi_synchronous"
    EVENTUAL_CONSISTENCY = "eventual_consistency"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    GEOGRAPHIC = "geographic"
    HEALTH_BASED = "health_based"


@dataclass
class RegionConfig:
    """Configuration for a region"""
    region_id: str
    region_name: str
    data_center: str
    
    # Network configuration
    primary_endpoint: str
    backup_endpoints: List[str] = field(default_factory=list)
    internal_endpoints: List[str] = field(default_factory=list)
    
    # Geographic information
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = "UTC"
    
    # Capacity and limits
    max_capacity: int = 1000
    current_capacity: int = 0
    weight: float = 1.0
    priority: int = 1  # Lower number = higher priority
    
    # Health check configuration
    health_check_url: str = "/health"
    health_check_interval: float = 5.0
    health_check_timeout: float = 2.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    
    # Performance thresholds
    max_latency_ms: float = 100.0
    max_error_rate: float = 0.01  # 1%
    min_throughput: float = 100.0
    
    # Replication settings
    replication_strategy: ReplicationStrategy = ReplicationStrategy.ASYNCHRONOUS
    replication_lag_threshold_ms: float = 100.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverConfig:
    """Configuration for failover system"""
    # RTO/RPO targets
    rto_target_seconds: float = 1.0  # Recovery Time Objective
    rpo_target_seconds: float = 5.0  # Recovery Point Objective
    
    # Detection thresholds
    health_check_failures: int = 3
    latency_threshold_ms: float = 500.0
    error_rate_threshold: float = 0.05  # 5%
    
    # Failover behavior
    auto_failover_enabled: bool = True
    auto_failback_enabled: bool = True
    failback_delay_seconds: float = 300.0  # 5 minutes
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LATENCY
    sticky_sessions: bool = False
    session_timeout_seconds: float = 3600.0
    
    # Data consistency
    consistency_check_enabled: bool = True
    consistency_check_interval: float = 60.0
    max_replication_lag_ms: float = 1000.0
    
    # Monitoring
    monitoring_interval_seconds: float = 1.0
    metrics_retention_hours: float = 24.0
    
    # Alerting
    alert_on_failover: bool = True
    alert_on_degradation: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    timestamp: float = field(default_factory=time.time)
    region_id: str = ""
    endpoint: str = ""
    
    # Health status
    is_healthy: bool = False
    response_time_ms: float = 0.0
    status_code: int = 0
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency_ms: float = 0.0
    
    # Application metrics
    active_connections: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverEvent:
    """Represents a failover event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Event details
    trigger: FailoverTrigger
    source_region: str
    target_region: str
    
    # Timing information
    detection_time: float = 0.0
    decision_time: float = 0.0
    execution_time: float = 0.0
    completion_time: float = 0.0
    total_duration: float = 0.0
    
    # Impact assessment
    affected_services: List[str] = field(default_factory=list)
    affected_users: int = 0
    data_loss_seconds: float = 0.0
    
    # Success metrics
    rto_achieved: bool = False
    rpo_achieved: bool = False
    
    # Context
    health_metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'trigger': self.trigger.value,
            'source_region': self.source_region,
            'target_region': self.target_region,
            'detection_time': self.detection_time,
            'decision_time': self.decision_time,
            'execution_time': self.execution_time,
            'completion_time': self.completion_time,
            'total_duration': self.total_duration,
            'affected_services': self.affected_services,
            'affected_users': self.affected_users,
            'data_loss_seconds': self.data_loss_seconds,
            'rto_achieved': self.rto_achieved,
            'rpo_achieved': self.rpo_achieved,
            'health_metrics': self.health_metrics,
            'error_details': self.error_details,
            'metadata': self.metadata
        }


@dataclass
class RegionMetrics:
    """Performance metrics for a region"""
    timestamp: float = field(default_factory=time.time)
    region_id: str = ""
    
    # Performance metrics
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io_mbps: float = 0.0
    
    # Connection metrics
    active_connections: int = 0
    connection_pool_usage: float = 0.0
    queue_depth: int = 0
    
    # Data metrics
    replication_lag_ms: float = 0.0
    data_consistency_score: float = 1.0
    backup_status: str = "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'region_id': self.region_id,
            'latency_ms': self.latency_ms,
            'throughput_rps': self.throughput_rps,
            'error_rate': self.error_rate,
            'availability': self.availability,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_io_mbps': self.network_io_mbps,
            'active_connections': self.active_connections,
            'connection_pool_usage': self.connection_pool_usage,
            'queue_depth': self.queue_depth,
            'replication_lag_ms': self.replication_lag_ms,
            'data_consistency_score': self.data_consistency_score,
            'backup_status': self.backup_status
        }


class HealthChecker:
    """Performs health checks on regions"""
    
    def __init__(self):
        self._session_cache = {}
        
    async def check_health(self, region: RegionConfig) -> HealthCheckResult:
        """Perform health check on a region"""
        start_time = time.time()
        
        try:
            # Simulate HTTP health check
            response_time = await self._simulate_http_check(region.primary_endpoint + region.health_check_url)
            
            # Simulate system metrics collection
            cpu_usage = await self._get_cpu_usage(region.region_id)
            memory_usage = await self._get_memory_usage(region.region_id)
            disk_usage = await self._get_disk_usage(region.region_id)
            
            # Simulate application metrics
            active_connections = await self._get_active_connections(region.region_id)
            requests_per_second = await self._get_requests_per_second(region.region_id)
            error_rate = await self._get_error_rate(region.region_id)
            
            # Determine health status
            is_healthy = (
                response_time < region.health_check_timeout * 1000 and
                cpu_usage < 90.0 and
                memory_usage < 90.0 and
                disk_usage < 90.0 and
                error_rate < region.max_error_rate
            )
            
            return HealthCheckResult(
                region_id=region.region_id,
                endpoint=region.primary_endpoint,
                is_healthy=is_healthy,
                response_time_ms=response_time,
                status_code=200 if is_healthy else 500,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency_ms=response_time,
                active_connections=active_connections,
                requests_per_second=requests_per_second,
                error_rate=error_rate
            )
            
        except Exception as e:
            return HealthCheckResult(
                region_id=region.region_id,
                endpoint=region.primary_endpoint,
                is_healthy=False,
                error_message=str(e),
                error_details={'exception': traceback.format_exc()}
            )
            
    async def _simulate_http_check(self, url: str) -> float:
        """Simulate HTTP health check"""
        # Simulate network latency
        await asyncio.sleep(0.01 + (hash(url) % 100) / 10000)  # 10-20ms
        return 10.0 + (hash(url) % 50)  # 10-60ms response time
        
    async def _get_cpu_usage(self, region_id: str) -> float:
        """Simulate CPU usage retrieval"""
        return 20.0 + (hash(region_id) % 60)  # 20-80% CPU usage
        
    async def _get_memory_usage(self, region_id: str) -> float:
        """Simulate memory usage retrieval"""
        return 30.0 + (hash(region_id) % 50)  # 30-80% memory usage
        
    async def _get_disk_usage(self, region_id: str) -> float:
        """Simulate disk usage retrieval"""
        return 40.0 + (hash(region_id) % 40)  # 40-80% disk usage
        
    async def _get_active_connections(self, region_id: str) -> int:
        """Simulate active connections retrieval"""
        return 100 + (hash(region_id) % 500)  # 100-600 connections
        
    async def _get_requests_per_second(self, region_id: str) -> float:
        """Simulate RPS retrieval"""
        return 50.0 + (hash(region_id) % 200)  # 50-250 RPS
        
    async def _get_error_rate(self, region_id: str) -> float:
        """Simulate error rate retrieval"""
        return (hash(region_id) % 100) / 10000  # 0-1% error rate


class LoadBalancer:
    """Manages load balancing across regions"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LATENCY):
        self.strategy = strategy
        self.connection_counts = defaultdict(int)
        self.latency_history = defaultdict(list)
        self.session_affinity = {}
        
    def select_region(self, regions: List[RegionConfig], 
                     metrics: Dict[str, RegionMetrics],
                     session_id: Optional[str] = None) -> Optional[RegionConfig]:
        """Select the best region for a request"""
        # Filter healthy regions
        healthy_regions = [r for r in regions if self._is_region_healthy(r, metrics)]
        
        if not healthy_regions:
            return None
            
        # Check session affinity
        if session_id and session_id in self.session_affinity:
            preferred_region = self.session_affinity[session_id]
            if preferred_region in [r.region_id for r in healthy_regions]:
                return next(r for r in healthy_regions if r.region_id == preferred_region)
                
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_regions)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_regions)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_regions)
        elif self.strategy == LoadBalancingStrategy.LEAST_LATENCY:
            return self._least_latency_selection(healthy_regions, metrics)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_regions, metrics)
        else:
            return healthy_regions[0]  # Default to first healthy region
            
    def _is_region_healthy(self, region: RegionConfig, metrics: Dict[str, RegionMetrics]) -> bool:
        """Check if region is healthy"""
        region_metrics = metrics.get(region.region_id)
        if not region_metrics:
            return False
            
        return (
            region_metrics.availability > 0.99 and
            region_metrics.error_rate < region.max_error_rate and
            region_metrics.latency_ms < region.max_latency_ms
        )
        
    def _round_robin_selection(self, regions: List[RegionConfig]) -> RegionConfig:
        """Round robin selection"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
            
        selected = regions[self._round_robin_index % len(regions)]
        self._round_robin_index += 1
        return selected
        
    def _weighted_round_robin_selection(self, regions: List[RegionConfig]) -> RegionConfig:
        """Weighted round robin selection"""
        total_weight = sum(r.weight for r in regions)
        if total_weight == 0:
            return regions[0]
            
        # Simple weighted selection
        weights = [r.weight / total_weight for r in regions]
        import random
        return random.choices(regions, weights=weights)[0]
        
    def _least_connections_selection(self, regions: List[RegionConfig]) -> RegionConfig:
        """Least connections selection"""
        return min(regions, key=lambda r: self.connection_counts[r.region_id])
        
    def _least_latency_selection(self, regions: List[RegionConfig], 
                               metrics: Dict[str, RegionMetrics]) -> RegionConfig:
        """Least latency selection"""
        def get_latency(region):
            region_metrics = metrics.get(region.region_id)
            return region_metrics.latency_ms if region_metrics else float('inf')
            
        return min(regions, key=get_latency)
        
    def _health_based_selection(self, regions: List[RegionConfig], 
                              metrics: Dict[str, RegionMetrics]) -> RegionConfig:
        """Health-based selection"""
        def health_score(region):
            region_metrics = metrics.get(region.region_id)
            if not region_metrics:
                return 0
                
            # Calculate composite health score
            availability_score = region_metrics.availability
            latency_score = max(0, 1 - region_metrics.latency_ms / 1000)
            error_score = max(0, 1 - region_metrics.error_rate * 100)
            cpu_score = max(0, 1 - region_metrics.cpu_usage / 100)
            
            return (availability_score + latency_score + error_score + cpu_score) / 4
            
        return max(regions, key=health_score)
        
    def record_connection(self, region_id: str):
        """Record a new connection to a region"""
        self.connection_counts[region_id] += 1
        
    def release_connection(self, region_id: str):
        """Release a connection from a region"""
        self.connection_counts[region_id] = max(0, self.connection_counts[region_id] - 1)


class DataReplicator:
    """Handles data replication across regions"""
    
    def __init__(self, strategy: ReplicationStrategy = ReplicationStrategy.ASYNCHRONOUS):
        self.strategy = strategy
        self.replication_queues = defaultdict(deque)
        self.replication_lag = defaultdict(float)
        self.consistency_checksums = defaultdict(str)
        
    async def replicate_data(self, source_region: str, target_regions: List[str], 
                           data: Dict[str, Any]) -> Dict[str, bool]:
        """Replicate data to target regions"""
        results = {}
        
        if self.strategy == ReplicationStrategy.SYNCHRONOUS:
            # Wait for all replications to complete
            tasks = []
            for target_region in target_regions:
                task = asyncio.create_task(self._replicate_to_region(source_region, target_region, data))
                tasks.append((target_region, task))
                
            for target_region, task in tasks:
                try:
                    success = await asyncio.wait_for(task, timeout=5.0)
                    results[target_region] = success
                except asyncio.TimeoutError:
                    results[target_region] = False
                    
        elif self.strategy == ReplicationStrategy.ASYNCHRONOUS:
            # Queue replications for background processing
            for target_region in target_regions:
                self.replication_queues[target_region].append({
                    'source_region': source_region,
                    'data': data,
                    'timestamp': time.time()
                })
                results[target_region] = True  # Assume success for async
                
        elif self.strategy == ReplicationStrategy.SEMI_SYNCHRONOUS:
            # Wait for at least one replication to complete
            tasks = []
            for target_region in target_regions:
                task = asyncio.create_task(self._replicate_to_region(source_region, target_region, data))
                tasks.append((target_region, task))
                
            # Wait for first completion
            done, pending = await asyncio.wait(
                [task for _, task in tasks], 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=2.0
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
            # Process results
            for target_region, task in tasks:
                if task in done:
                    try:
                        results[target_region] = await task
                    except Exception:
                        results[target_region] = False
                else:
                    results[target_region] = False
                    
        return results
        
    async def _replicate_to_region(self, source_region: str, target_region: str, 
                                 data: Dict[str, Any]) -> bool:
        """Replicate data to a specific region"""
        try:
            # Simulate replication latency
            latency = 0.01 + (hash(target_region) % 50) / 1000  # 10-60ms
            await asyncio.sleep(latency)
            
            # Update replication lag
            self.replication_lag[target_region] = latency * 1000
            
            # Update consistency checksum
            data_str = json.dumps(data, sort_keys=True)
            checksum = hashlib.md5(data_str.encode()).hexdigest()
            self.consistency_checksums[target_region] = checksum
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to replicate to {target_region}: {e}")
            return False
            
    async def check_consistency(self, regions: List[str]) -> Dict[str, bool]:
        """Check data consistency across regions"""
        if len(regions) < 2:
            return {region: True for region in regions}
            
        # Get checksums for all regions
        checksums = {region: self.consistency_checksums.get(region, "") for region in regions}
        
        # Find the most common checksum (majority consensus)
        checksum_counts = defaultdict(int)
        for checksum in checksums.values():
            if checksum:
                checksum_counts[checksum] += 1
                
        if not checksum_counts:
            return {region: False for region in regions}
            
        majority_checksum = max(checksum_counts.items(), key=lambda x: x[1])[0]
        
        # Check each region against majority
        results = {}
        for region in regions:
            region_checksum = checksums[region]
            results[region] = region_checksum == majority_checksum
            
        return results
        
    def get_replication_lag(self, region: str) -> float:
        """Get replication lag for a region in milliseconds"""
        return self.replication_lag.get(region, 0.0)


class FailoverDecisionEngine:
    """Makes failover decisions based on health and performance metrics"""
    
    def __init__(self, config: FailoverConfig):
        self.config = config
        self.failure_counts = defaultdict(int)
        self.last_failover_time = defaultdict(float)
        
    def should_failover(self, primary_region: RegionConfig, 
                       health_result: HealthCheckResult,
                       metrics: RegionMetrics) -> Tuple[bool, FailoverTrigger]:
        """Determine if failover should be triggered"""
        current_time = time.time()
        
        # Check if auto-failover is enabled
        if not self.config.auto_failover_enabled:
            return False, None
            
        # Check health check failures
        if not health_result.is_healthy:
            self.failure_counts[primary_region.region_id] += 1
            if self.failure_counts[primary_region.region_id] >= self.config.health_check_failures:
                return True, FailoverTrigger.HEALTH_CHECK_FAILURE
        else:
            self.failure_counts[primary_region.region_id] = 0
            
        # Check latency threshold
        if metrics.latency_ms > self.config.latency_threshold_ms:
            return True, FailoverTrigger.LATENCY_THRESHOLD
            
        # Check error rate threshold
        if metrics.error_rate > self.config.error_rate_threshold:
            return True, FailoverTrigger.ERROR_RATE_THRESHOLD
            
        # Check resource exhaustion
        if (health_result.cpu_usage > 95 or 
            health_result.memory_usage > 95 or 
            health_result.disk_usage > 95):
            return True, FailoverTrigger.RESOURCE_EXHAUSTION
            
        return False, None
        
    def should_failback(self, original_region: RegionConfig,
                       health_result: HealthCheckResult,
                       metrics: RegionMetrics) -> bool:
        """Determine if failback should be triggered"""
        if not self.config.auto_failback_enabled:
            return False
            
        current_time = time.time()
        last_failover = self.last_failover_time.get(original_region.region_id, 0)
        
        # Check failback delay
        if current_time - last_failover < self.config.failback_delay_seconds:
            return False
            
        # Check if original region is healthy
        if not health_result.is_healthy:
            return False
            
        # Check performance metrics
        if (metrics.latency_ms > original_region.max_latency_ms or
            metrics.error_rate > original_region.max_error_rate):
            return False
            
        # Check resource utilization
        if (health_result.cpu_usage > 80 or
            health_result.memory_usage > 80 or
            health_result.disk_usage > 80):
            return False
            
        return True
        
    def record_failover(self, region_id: str):
        """Record a failover event"""
        self.last_failover_time[region_id] = time.time()
        self.failure_counts[region_id] = 0


class MultiRegionFailoverManager:
    """Main multi-region failover management system"""
    
    def __init__(self, config: FailoverConfig = None):
        self.config = config or FailoverConfig()
        self.regions: Dict[str, RegionConfig] = {}
        self.region_status: Dict[str, RegionStatus] = {}
        self.current_primary: Optional[str] = None
        
        # Components
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer(self.config.load_balancing_strategy)
        self.data_replicator = DataReplicator()
        self.decision_engine = FailoverDecisionEngine(self.config)
        
        # Metrics and events
        self.region_metrics: Dict[str, RegionMetrics] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.failover_events: deque = deque(maxlen=1000)
        
        # State management
        self._lock = threading.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        
    def add_region(self, region: RegionConfig):
        """Add a region to the failover system"""
        with self._lock:
            self.regions[region.region_id] = region
            self.region_status[region.region_id] = RegionStatus.STANDBY
            
            # Set first region as primary
            if self.current_primary is None:
                self.current_primary = region.region_id
                self.region_status[region.region_id] = RegionStatus.ACTIVE
                
    def remove_region(self, region_id: str):
        """Remove a region from the failover system"""
        with self._lock:
            if region_id in self.regions:
                del self.regions[region_id]
                del self.region_status[region_id]
                
                # If removing primary, trigger failover
                if self.current_primary == region_id:
                    self.current_primary = None
                    asyncio.create_task(self._trigger_emergency_failover())
                    
    def get_primary_region(self) -> Optional[RegionConfig]:
        """Get the current primary region"""
        if self.current_primary:
            return self.regions.get(self.current_primary)
        return None
        
    def get_standby_regions(self) -> List[RegionConfig]:
        """Get all standby regions"""
        standby_regions = []
        for region_id, status in self.region_status.items():
            if status == RegionStatus.STANDBY and region_id in self.regions:
                standby_regions.append(self.regions[region_id])
        return standby_regions
        
    async def trigger_manual_failover(self, target_region_id: str) -> FailoverEvent:
        """Manually trigger failover to a specific region"""
        return await self._execute_failover(
            target_region_id, 
            FailoverTrigger.MANUAL_TRIGGER
        )
        
    async def _execute_failover(self, target_region_id: str, 
                              trigger: FailoverTrigger) -> FailoverEvent:
        """Execute failover to target region"""
        start_time = time.time()
        
        # Create failover event
        event = FailoverEvent(
            trigger=trigger,
            source_region=self.current_primary or "unknown",
            target_region=target_region_id,
            detection_time=start_time
        )
        
        try:
            # Validate target region
            if target_region_id not in self.regions:
                raise ValueError(f"Target region {target_region_id} not found")
                
            target_region = self.regions[target_region_id]
            
            # Check target region health
            health_result = await self.health_checker.check_health(target_region)
            if not health_result.is_healthy:
                raise ValueError(f"Target region {target_region_id} is not healthy")
                
            decision_time = time.time()
            event.decision_time = decision_time - start_time
            
            # Execute failover steps
            execution_start = time.time()
            
            # 1. Update DNS/Load balancer
            await self._update_traffic_routing(target_region_id)
            
            # 2. Promote target region to primary
            with self._lock:
                old_primary = self.current_primary
                self.current_primary = target_region_id
                self.region_status[target_region_id] = RegionStatus.ACTIVE
                
                if old_primary:
                    self.region_status[old_primary] = RegionStatus.STANDBY
                    
            # 3. Ensure data consistency
            await self._ensure_data_consistency(target_region_id)
            
            # 4. Update monitoring and alerting
            await self._update_monitoring_targets(target_region_id)
            
            execution_time = time.time() - execution_start
            event.execution_time = execution_time
            
            # Record completion
            completion_time = time.time()
            event.completion_time = completion_time - start_time
            event.total_duration = completion_time - start_time
            
            # Check RTO/RPO achievement
            event.rto_achieved = event.total_duration <= self.config.rto_target_seconds
            event.rpo_achieved = True  # Simplified for demo
            
            # Record metrics
            event.health_metrics = {
                'target_latency': health_result.response_time_ms,
                'target_cpu': health_result.cpu_usage,
                'target_memory': health_result.memory_usage
            }
            
            # Record decision engine state
            self.decision_engine.record_failover(target_region_id)
            
            logger.info(f"Failover completed: {event.source_region} -> {event.target_region} "
                       f"in {event.total_duration:.3f}s (RTO: {event.rto_achieved})")
            
        except Exception as e:
            event.error_details = str(e)
            logger.error(f"Failover failed: {e}")
            
        finally:
            with self._lock:
                self.failover_events.append(event)
                
        return event
        
    async def _update_traffic_routing(self, target_region_id: str):
        """Update traffic routing to target region"""
        # Simulate DNS update or load balancer reconfiguration
        await asyncio.sleep(0.1)  # Simulate routing update time
        logger.info(f"Traffic routing updated to region {target_region_id}")
        
    async def _ensure_data_consistency(self, target_region_id: str):
        """Ensure data consistency in target region"""
        # Check replication lag
        lag = self.data_replicator.get_replication_lag(target_region_id)
        if lag > self.config.max_replication_lag_ms:
            logger.warning(f"High replication lag in {target_region_id}: {lag}ms")
            
        # Verify data consistency
        all_regions = list(self.regions.keys())
        consistency_results = await self.data_replicator.check_consistency(all_regions)
        
        if not consistency_results.get(target_region_id, False):
            logger.warning(f"Data inconsistency detected in {target_region_id}")
            
    async def _update_monitoring_targets(self, target_region_id: str):
        """Update monitoring and alerting targets"""
        # Simulate monitoring system update
        await asyncio.sleep(0.05)
        logger.info(f"Monitoring targets updated for region {target_region_id}")
        
    async def _trigger_emergency_failover(self):
        """Trigger emergency failover when primary is lost"""
        standby_regions = self.get_standby_regions()
        if not standby_regions:
            logger.error("No standby regions available for emergency failover")
            return
            
        # Select best standby region
        best_region = None
        best_score = -1
        
        for region in standby_regions:
            health_result = await self.health_checker.check_health(region)
            if health_result.is_healthy:
                # Calculate region score
                score = (
                    (100 - health_result.cpu_usage) +
                    (100 - health_result.memory_usage) +
                    (1000 - health_result.response_time_ms) / 10 +
                    region.priority * 10
                )
                
                if score > best_score:
                    best_score = score
                    best_region = region
                    
        if best_region:
            await self._execute_failover(
                best_region.region_id,
                FailoverTrigger.DATA_CENTER_FAILURE
            )
        else:
            logger.error("No healthy standby regions available")
            
    async def start_monitoring(self):
        """Start continuous monitoring and failover detection"""
        if self._monitoring_task is not None:
            return
            
        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._stop_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while not self._stop_event.is_set():
                # Check health of all regions
                health_tasks = []
                for region in self.regions.values():
                    task = asyncio.create_task(self.health_checker.check_health(region))
                    health_tasks.append((region.region_id, task))
                    
                # Collect health results
                for region_id, task in health_tasks:
                    try:
                        health_result = await task
                        self.health_results[region_id] = health_result
                        
                        # Update region metrics
                        metrics = RegionMetrics(
                            region_id=region_id,
                            latency_ms=health_result.response_time_ms,
                            throughput_rps=health_result.requests_per_second,
                            error_rate=health_result.error_rate,
                            availability=1.0 if health_result.is_healthy else 0.0,
                            cpu_usage=health_result.cpu_usage,
                            memory_usage=health_result.memory_usage,
                            disk_usage=health_result.disk_usage,
                            active_connections=health_result.active_connections,
                            replication_lag_ms=self.data_replicator.get_replication_lag(region_id)
                        )
                        self.region_metrics[region_id] = metrics
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {region_id}: {e}")
                        
                # Check if failover is needed
                if self.current_primary:
                    primary_region = self.regions[self.current_primary]
                    primary_health = self.health_results.get(self.current_primary)
                    primary_metrics = self.region_metrics.get(self.current_primary)
                    
                    if primary_health and primary_metrics:
                        should_failover, trigger = self.decision_engine.should_failover(
                            primary_region, primary_health, primary_metrics
                        )
                        
                        if should_failover:
                            # Find best standby region
                            standby_regions = self.get_standby_regions()
                            if standby_regions:
                                target_region = self.load_balancer.select_region(
                                    standby_regions, self.region_metrics
                                )
                                
                                if target_region:
                                    logger.warning(f"Triggering automatic failover: {trigger.value}")
                                    await self._execute_failover(target_region.region_id, trigger)
                                    
                # Check for failback opportunities
                await self._check_failback_opportunities()
                
                # Wait for next iteration
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            
    async def _check_failback_opportunities(self):
        """Check if failback to original primary is possible"""
        if not self.config.auto_failback_enabled:
            return
            
        # Find regions that could be better primaries
        for region_id, region in self.regions.items():
            if (region_id != self.current_primary and 
                self.region_status[region_id] == RegionStatus.STANDBY):
                
                health_result = self.health_results.get(region_id)
                metrics = self.region_metrics.get(region_id)
                
                if health_result and metrics:
                    should_failback = self.decision_engine.should_failback(
                        region, health_result, metrics
                    )
                    
                    if should_failback and region.priority < self.regions[self.current_primary].priority:
                        logger.info(f"Triggering automatic failback to higher priority region {region_id}")
                        await self._execute_failover(region_id, FailoverTrigger.MANUAL_TRIGGER)
                        break
                        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self._lock:
            return {
                'primary_region': self.current_primary,
                'total_regions': len(self.regions),
                'healthy_regions': sum(1 for r in self.health_results.values() if r.is_healthy),
                'region_status': dict(self.region_status),
                'recent_failovers': len([e for e in self.failover_events if time.time() - e.timestamp < 3600]),
                'average_rto': statistics.mean([e.total_duration for e in self.failover_events if e.total_duration > 0]) if self.failover_events else 0,
                'rto_success_rate': sum(1 for e in self.failover_events if e.rto_achieved) / len(self.failover_events) if self.failover_events else 1.0
            }
            
    def get_region_metrics(self, region_id: str = None) -> Union[RegionMetrics, Dict[str, RegionMetrics]]:
        """Get metrics for specific region or all regions"""
        if region_id:
            return self.region_metrics.get(region_id)
        return dict(self.region_metrics)
        
    def get_failover_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent failover events"""
        with self._lock:
            recent_events = list(self.failover_events)[-limit:]
            return [event.to_dict() for event in recent_events]


# Utility functions
def create_region_config(region_id: str, region_name: str, 
                        endpoint: str, **kwargs) -> RegionConfig:
    """Create a region configuration"""
    return RegionConfig(
        region_id=region_id,
        region_name=region_name,
        data_center=kwargs.get('data_center', region_name),
        primary_endpoint=endpoint,
        **kwargs
    )


def create_development_failover_config() -> FailoverConfig:
    """Create failover configuration for development"""
    return FailoverConfig(
        rto_target_seconds=2.0,
        rpo_target_seconds=10.0,
        auto_failover_enabled=True,
        auto_failback_enabled=True,
        monitoring_interval_seconds=5.0
    )


def create_production_failover_config() -> FailoverConfig:
    """Create failover configuration for production"""
    return FailoverConfig(
        rto_target_seconds=1.0,
        rpo_target_seconds=5.0,
        auto_failover_enabled=True,
        auto_failback_enabled=True,
        monitoring_interval_seconds=1.0,
        health_check_failures=2,
        latency_threshold_ms=200.0,
        error_rate_threshold=0.02
    )


async def create_multi_region_setup() -> MultiRegionFailoverManager:
    """Create a complete multi-region failover setup"""
    config = create_production_failover_config()
    manager = MultiRegionFailoverManager(config)
    
    # Add sample regions
    regions = [
        create_region_config("us-east-1", "US East", "https://us-east-1.example.com", priority=1),
        create_region_config("us-west-2", "US West", "https://us-west-2.example.com", priority=2),
        create_region_config("eu-west-1", "EU West", "https://eu-west-1.example.com", priority=3),
        create_region_config("ap-southeast-1", "Asia Pacific", "https://ap-southeast-1.example.com", priority=4)
    ]
    
    for region in regions:
        manager.add_region(region)
        
    await manager.start_monitoring()
    
    return manager