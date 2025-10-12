"""
Chaos Engineering Tools for Automatic Resilience Testing

Implements comprehensive chaos engineering capabilities to test system resilience
through controlled failure injection and automated testing scenarios.
"""

import asyncio
import random
import time
import threading
import psutil
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import subprocess
import socket
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    PROCESS_KILL = "process_kill"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATABASE_SLOW = "database_slow"
    TIMEOUT_INJECTION = "timeout_injection"
    EXCEPTION_INJECTION = "exception_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FailureConfig:
    """Configuration for failure injection"""
    failure_type: FailureType
    duration: float
    intensity: float = 1.0  # 0.0 to 1.0
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a chaos experiment"""
    experiment_id: str
    status: ExperimentStatus
    start_time: float
    end_time: Optional[float] = None
    failures_injected: List[FailureConfig] = field(default_factory=list)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    recovery_time: Optional[float] = None
    success_rate: float = 0.0
    error_messages: List[str] = field(default_factory=list)


@dataclass
class ResilienceMetrics:
    """Metrics for measuring system resilience"""
    availability: float
    response_time_p95: float
    error_rate: float
    recovery_time: float
    throughput: float
    timestamp: float = field(default_factory=time.time)


class FailureInjector(ABC):
    """Abstract base class for failure injectors"""
    
    @abstractmethod
    async def inject_failure(self, config: FailureConfig) -> bool:
        """Inject a specific type of failure"""
        pass
        
    @abstractmethod
    async def stop_failure(self, config: FailureConfig) -> bool:
        """Stop the injected failure"""
        pass
        
    @abstractmethod
    def is_supported(self, failure_type: FailureType) -> bool:
        """Check if this injector supports the failure type"""
        pass


class NetworkFailureInjector(FailureInjector):
    """Injects network-related failures"""
    
    def __init__(self):
        self.active_failures: Set[str] = set()
        
    def is_supported(self, failure_type: FailureType) -> bool:
        return failure_type in [
            FailureType.NETWORK_LATENCY,
            FailureType.NETWORK_PARTITION,
            FailureType.NETWORK_PACKET_LOSS
        ]
        
    async def inject_failure(self, config: FailureConfig) -> bool:
        """Inject network failure"""
        try:
            if config.failure_type == FailureType.NETWORK_LATENCY:
                return await self._inject_latency(config)
            elif config.failure_type == FailureType.NETWORK_PARTITION:
                return await self._inject_partition(config)
            elif config.failure_type == FailureType.NETWORK_PACKET_LOSS:
                return await self._inject_packet_loss(config)
            return False
        except Exception as e:
            logger.error(f"Failed to inject network failure: {e}")
            return False
            
    async def stop_failure(self, config: FailureConfig) -> bool:
        """Stop network failure"""
        try:
            failure_id = f"{config.failure_type.value}_{config.target}"
            if failure_id in self.active_failures:
                self.active_failures.remove(failure_id)
                # In a real implementation, this would clean up network rules
                logger.info(f"Stopped network failure: {failure_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop network failure: {e}")
            return False
            
    async def _inject_latency(self, config: FailureConfig) -> bool:
        """Inject network latency"""
        delay_ms = int(config.intensity * 1000)  # Convert to milliseconds
        target = config.target or "0.0.0.0/0"
        
        # Simulate latency injection (in real implementation, use tc or similar)
        logger.info(f"Injecting {delay_ms}ms latency to {target}")
        failure_id = f"{config.failure_type.value}_{target}"
        self.active_failures.add(failure_id)
        return True
        
    async def _inject_partition(self, config: FailureConfig) -> bool:
        """Inject network partition"""
        target = config.target or "external"
        
        # Simulate network partition (in real implementation, use iptables)
        logger.info(f"Creating network partition for {target}")
        failure_id = f"{config.failure_type.value}_{target}"
        self.active_failures.add(failure_id)
        return True
        
    async def _inject_packet_loss(self, config: FailureConfig) -> bool:
        """Inject packet loss"""
        loss_percent = int(config.intensity * 100)
        target = config.target or "0.0.0.0/0"
        
        # Simulate packet loss (in real implementation, use tc)
        logger.info(f"Injecting {loss_percent}% packet loss to {target}")
        failure_id = f"{config.failure_type.value}_{target}"
        self.active_failures.add(failure_id)
        return True


class ResourceFailureInjector(FailureInjector):
    """Injects resource-related failures"""
    
    def __init__(self):
        self.stress_processes: Dict[str, subprocess.Popen] = {}
        
    def is_supported(self, failure_type: FailureType) -> bool:
        return failure_type in [
            FailureType.CPU_STRESS,
            FailureType.MEMORY_STRESS,
            FailureType.DISK_STRESS,
            FailureType.RESOURCE_EXHAUSTION
        ]
        
    async def inject_failure(self, config: FailureConfig) -> bool:
        """Inject resource failure"""
        try:
            if config.failure_type == FailureType.CPU_STRESS:
                return await self._inject_cpu_stress(config)
            elif config.failure_type == FailureType.MEMORY_STRESS:
                return await self._inject_memory_stress(config)
            elif config.failure_type == FailureType.DISK_STRESS:
                return await self._inject_disk_stress(config)
            elif config.failure_type == FailureType.RESOURCE_EXHAUSTION:
                return await self._inject_resource_exhaustion(config)
            return False
        except Exception as e:
            logger.error(f"Failed to inject resource failure: {e}")
            return False
            
    async def stop_failure(self, config: FailureConfig) -> bool:
        """Stop resource failure"""
        try:
            failure_id = f"{config.failure_type.value}_{config.target}"
            if failure_id in self.stress_processes:
                process = self.stress_processes.pop(failure_id)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                logger.info(f"Stopped resource failure: {failure_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop resource failure: {e}")
            return False
            
    async def _inject_cpu_stress(self, config: FailureConfig) -> bool:
        """Inject CPU stress"""
        cpu_cores = int(psutil.cpu_count() * config.intensity)
        logger.info(f"Starting CPU stress on {cpu_cores} cores")
        
        # Simulate CPU stress (in real implementation, use stress-ng or similar)
        failure_id = f"{config.failure_type.value}_cpu"
        # self.stress_processes[failure_id] = process
        return True
        
    async def _inject_memory_stress(self, config: FailureConfig) -> bool:
        """Inject memory stress"""
        memory_mb = int(psutil.virtual_memory().total * config.intensity / (1024 * 1024))
        logger.info(f"Starting memory stress: {memory_mb}MB")
        
        # Simulate memory stress
        failure_id = f"{config.failure_type.value}_memory"
        return True
        
    async def _inject_disk_stress(self, config: FailureConfig) -> bool:
        """Inject disk I/O stress"""
        logger.info(f"Starting disk I/O stress with intensity {config.intensity}")
        
        # Simulate disk stress
        failure_id = f"{config.failure_type.value}_disk"
        return True
        
    async def _inject_resource_exhaustion(self, config: FailureConfig) -> bool:
        """Inject resource exhaustion"""
        logger.info("Starting resource exhaustion simulation")
        
        # Simulate resource exhaustion
        failure_id = f"{config.failure_type.value}_resources"
        return True


class ApplicationFailureInjector(FailureInjector):
    """Injects application-level failures"""
    
    def __init__(self):
        self.injected_failures: Set[str] = set()
        
    def is_supported(self, failure_type: FailureType) -> bool:
        return failure_type in [
            FailureType.PROCESS_KILL,
            FailureType.SERVICE_UNAVAILABLE,
            FailureType.DATABASE_SLOW,
            FailureType.TIMEOUT_INJECTION,
            FailureType.EXCEPTION_INJECTION
        ]
        
    async def inject_failure(self, config: FailureConfig) -> bool:
        """Inject application failure"""
        try:
            if config.failure_type == FailureType.PROCESS_KILL:
                return await self._inject_process_kill(config)
            elif config.failure_type == FailureType.SERVICE_UNAVAILABLE:
                return await self._inject_service_unavailable(config)
            elif config.failure_type == FailureType.DATABASE_SLOW:
                return await self._inject_database_slow(config)
            elif config.failure_type == FailureType.TIMEOUT_INJECTION:
                return await self._inject_timeout(config)
            elif config.failure_type == FailureType.EXCEPTION_INJECTION:
                return await self._inject_exception(config)
            return False
        except Exception as e:
            logger.error(f"Failed to inject application failure: {e}")
            return False
            
    async def stop_failure(self, config: FailureConfig) -> bool:
        """Stop application failure"""
        try:
            failure_id = f"{config.failure_type.value}_{config.target}"
            if failure_id in self.injected_failures:
                self.injected_failures.remove(failure_id)
                logger.info(f"Stopped application failure: {failure_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop application failure: {e}")
            return False
            
    async def _inject_process_kill(self, config: FailureConfig) -> bool:
        """Kill specified process"""
        process_name = config.target or "target_process"
        logger.info(f"Killing process: {process_name}")
        
        # In real implementation, find and kill the process
        failure_id = f"{config.failure_type.value}_{process_name}"
        self.injected_failures.add(failure_id)
        return True
        
    async def _inject_service_unavailable(self, config: FailureConfig) -> bool:
        """Make service unavailable"""
        service_name = config.target or "target_service"
        logger.info(f"Making service unavailable: {service_name}")
        
        failure_id = f"{config.failure_type.value}_{service_name}"
        self.injected_failures.add(failure_id)
        return True
        
    async def _inject_database_slow(self, config: FailureConfig) -> bool:
        """Inject database slowness"""
        db_name = config.target or "database"
        delay_ms = int(config.intensity * 1000)
        logger.info(f"Injecting {delay_ms}ms delay to database: {db_name}")
        
        failure_id = f"{config.failure_type.value}_{db_name}"
        self.injected_failures.add(failure_id)
        return True
        
    async def _inject_timeout(self, config: FailureConfig) -> bool:
        """Inject timeout failures"""
        target = config.target or "requests"
        timeout_ms = int(config.intensity * 1000)
        logger.info(f"Injecting {timeout_ms}ms timeouts to {target}")
        
        failure_id = f"{config.failure_type.value}_{target}"
        self.injected_failures.add(failure_id)
        return True
        
    async def _inject_exception(self, config: FailureConfig) -> bool:
        """Inject random exceptions"""
        target = config.target or "application"
        error_rate = config.intensity * 100
        logger.info(f"Injecting {error_rate}% error rate to {target}")
        
        failure_id = f"{config.failure_type.value}_{target}"
        self.injected_failures.add(failure_id)
        return True


class ChaosExperiment:
    """Represents a chaos engineering experiment"""
    
    def __init__(self, experiment_id: str, name: str, description: str):
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.failures: List[FailureConfig] = []
        self.duration: float = 300.0  # 5 minutes default
        self.steady_state_hypothesis: Optional[Callable] = None
        self.rollback_strategy: Optional[Callable] = None
        
    def add_failure(self, failure_config: FailureConfig):
        """Add failure to experiment"""
        self.failures.append(failure_config)
        
    def set_steady_state_hypothesis(self, hypothesis: Callable):
        """Set steady state hypothesis function"""
        self.steady_state_hypothesis = hypothesis
        
    def set_rollback_strategy(self, strategy: Callable):
        """Set rollback strategy function"""
        self.rollback_strategy = strategy


class ResilienceValidator:
    """Validates system resilience during experiments"""
    
    def __init__(self):
        self.baseline_metrics: Optional[ResilienceMetrics] = None
        
    async def collect_baseline_metrics(self) -> ResilienceMetrics:
        """Collect baseline system metrics"""
        # Simulate metric collection
        metrics = ResilienceMetrics(
            availability=99.9,
            response_time_p95=100.0,
            error_rate=0.1,
            recovery_time=0.0,
            throughput=1000.0
        )
        self.baseline_metrics = metrics
        return metrics
        
    async def collect_current_metrics(self) -> ResilienceMetrics:
        """Collect current system metrics"""
        # Simulate metric collection during experiment
        return ResilienceMetrics(
            availability=random.uniform(95.0, 99.9),
            response_time_p95=random.uniform(100.0, 500.0),
            error_rate=random.uniform(0.1, 5.0),
            recovery_time=random.uniform(0.0, 30.0),
            throughput=random.uniform(500.0, 1000.0)
        )
        
    def validate_steady_state(self, current_metrics: ResilienceMetrics) -> bool:
        """Validate if system is in steady state"""
        if not self.baseline_metrics:
            return True
            
        # Check if metrics are within acceptable thresholds
        availability_ok = current_metrics.availability >= self.baseline_metrics.availability * 0.95
        response_time_ok = current_metrics.response_time_p95 <= self.baseline_metrics.response_time_p95 * 2.0
        error_rate_ok = current_metrics.error_rate <= self.baseline_metrics.error_rate * 5.0
        
        return availability_ok and response_time_ok and error_rate_ok
        
    def calculate_resilience_score(self, metrics: ResilienceMetrics) -> float:
        """Calculate overall resilience score (0-100)"""
        if not self.baseline_metrics:
            return 100.0
            
        # Weighted scoring
        availability_score = min(100, (metrics.availability / self.baseline_metrics.availability) * 100)
        response_time_score = min(100, (self.baseline_metrics.response_time_p95 / metrics.response_time_p95) * 100)
        error_rate_score = min(100, (self.baseline_metrics.error_rate / max(metrics.error_rate, 0.01)) * 100)
        recovery_score = max(0, 100 - metrics.recovery_time * 2)  # Penalty for slow recovery
        
        return (availability_score * 0.3 + response_time_score * 0.25 + 
                error_rate_score * 0.25 + recovery_score * 0.2)


class ChaosRunner:
    """Orchestrates chaos engineering experiments"""
    
    def __init__(self):
        self.injectors: List[FailureInjector] = [
            NetworkFailureInjector(),
            ResourceFailureInjector(),
            ApplicationFailureInjector()
        ]
        self.validator = ResilienceValidator()
        self.active_experiments: Dict[str, ExperimentResult] = {}
        self._lock = threading.Lock()
        
    def register_injector(self, injector: FailureInjector):
        """Register a custom failure injector"""
        self.injectors.append(injector)
        
    def _get_injector(self, failure_type: FailureType) -> Optional[FailureInjector]:
        """Get appropriate injector for failure type"""
        for injector in self.injectors:
            if injector.is_supported(failure_type):
                return injector
        return None
        
    async def run_experiment(self, experiment: ChaosExperiment) -> ExperimentResult:
        """Run a chaos engineering experiment"""
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            status=ExperimentStatus.RUNNING,
            start_time=time.time()
        )
        
        with self._lock:
            self.active_experiments[experiment.experiment_id] = result
            
        try:
            # Collect baseline metrics
            logger.info(f"Starting experiment: {experiment.name}")
            result.metrics_before = await self.validator.collect_baseline_metrics()
            
            # Verify steady state before experiment
            if experiment.steady_state_hypothesis:
                if not await experiment.steady_state_hypothesis():
                    raise Exception("System not in steady state before experiment")
                    
            # Inject failures
            injected_failures = []
            for failure_config in experiment.failures:
                injector = self._get_injector(failure_config.failure_type)
                if injector:
                    success = await injector.inject_failure(failure_config)
                    if success:
                        injected_failures.append((injector, failure_config))
                        result.failures_injected.append(failure_config)
                        logger.info(f"Injected failure: {failure_config.failure_type.value}")
                    else:
                        logger.error(f"Failed to inject: {failure_config.failure_type.value}")
                        
            # Monitor system during experiment
            start_monitoring = time.time()
            monitoring_interval = 5.0  # 5 seconds
            steady_state_violations = 0
            
            while time.time() - start_monitoring < experiment.duration:
                await asyncio.sleep(monitoring_interval)
                
                current_metrics = await self.validator.collect_current_metrics()
                if not self.validator.validate_steady_state(current_metrics):
                    steady_state_violations += 1
                    logger.warning(f"Steady state violation #{steady_state_violations}")
                    
                    # If too many violations, consider stopping experiment
                    if steady_state_violations >= 5:
                        logger.error("Too many steady state violations, stopping experiment")
                        break
                        
            # Stop all injected failures
            for injector, failure_config in injected_failures:
                await injector.stop_failure(failure_config)
                logger.info(f"Stopped failure: {failure_config.failure_type.value}")
                
            # Wait for system recovery
            recovery_start = time.time()
            max_recovery_time = 60.0  # 1 minute max
            
            while time.time() - recovery_start < max_recovery_time:
                await asyncio.sleep(2.0)
                current_metrics = await self.validator.collect_current_metrics()
                if self.validator.validate_steady_state(current_metrics):
                    result.recovery_time = time.time() - recovery_start
                    break
                    
            # Collect final metrics
            result.metrics_after = await self.validator.collect_current_metrics()
            result.success_rate = self.validator.calculate_resilience_score(result.metrics_after)
            result.status = ExperimentStatus.COMPLETED
            result.end_time = time.time()
            
            logger.info(f"Experiment completed: {experiment.name}, Score: {result.success_rate:.1f}")
            
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_messages.append(str(e))
            result.end_time = time.time()
            logger.error(f"Experiment failed: {experiment.name}, Error: {e}")
            
            # Emergency cleanup
            for injector in self.injectors:
                for failure_config in experiment.failures:
                    if injector.is_supported(failure_config.failure_type):
                        try:
                            await injector.stop_failure(failure_config)
                        except Exception as cleanup_error:
                            logger.error(f"Cleanup failed: {cleanup_error}")
                            
        return result
        
    async def run_continuous_testing(self, experiments: List[ChaosExperiment], 
                                   interval: float = 3600.0) -> None:
        """Run continuous chaos testing"""
        logger.info("Starting continuous chaos testing")
        
        while True:
            try:
                for experiment in experiments:
                    result = await self.run_experiment(experiment)
                    
                    # Log results
                    if result.status == ExperimentStatus.COMPLETED:
                        logger.info(f"Continuous test passed: {experiment.name}")
                    else:
                        logger.error(f"Continuous test failed: {experiment.name}")
                        
                    # Wait between experiments
                    await asyncio.sleep(60.0)
                    
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous testing: {e}")
                await asyncio.sleep(300.0)  # Wait 5 minutes before retry
                
    def get_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get result of specific experiment"""
        return self.active_experiments.get(experiment_id)
        
    def get_all_results(self) -> Dict[str, ExperimentResult]:
        """Get all experiment results"""
        return dict(self.active_experiments)
        
    async def emergency_stop_all(self):
        """Emergency stop all running experiments"""
        logger.warning("Emergency stop triggered for all experiments")
        
        for result in self.active_experiments.values():
            if result.status == ExperimentStatus.RUNNING:
                result.status = ExperimentStatus.CANCELLED
                result.end_time = time.time()
                
        # Stop all injectors
        for injector in self.injectors:
            if hasattr(injector, 'active_failures'):
                injector.active_failures.clear()
            if hasattr(injector, 'stress_processes'):
                for process in injector.stress_processes.values():
                    try:
                        process.terminate()
                    except:
                        pass
                injector.stress_processes.clear()
            if hasattr(injector, 'injected_failures'):
                injector.injected_failures.clear()


# Predefined experiment templates
def create_network_resilience_experiment() -> ChaosExperiment:
    """Create network resilience experiment"""
    experiment = ChaosExperiment(
        "network_resilience",
        "Network Resilience Test",
        "Tests system resilience to network failures"
    )
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.NETWORK_LATENCY,
        duration=120.0,
        intensity=0.5,
        target="external_api"
    ))
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.NETWORK_PACKET_LOSS,
        duration=60.0,
        intensity=0.1,
        target="database"
    ))
    
    return experiment


def create_resource_exhaustion_experiment() -> ChaosExperiment:
    """Create resource exhaustion experiment"""
    experiment = ChaosExperiment(
        "resource_exhaustion",
        "Resource Exhaustion Test",
        "Tests system behavior under resource pressure"
    )
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.CPU_STRESS,
        duration=180.0,
        intensity=0.8
    ))
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.MEMORY_STRESS,
        duration=120.0,
        intensity=0.6
    ))
    
    return experiment


def create_application_failure_experiment() -> ChaosExperiment:
    """Create application failure experiment"""
    experiment = ChaosExperiment(
        "application_failure",
        "Application Failure Test",
        "Tests application-level failure handling"
    )
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.SERVICE_UNAVAILABLE,
        duration=90.0,
        intensity=1.0,
        target="auth_service"
    ))
    
    experiment.add_failure(FailureConfig(
        failure_type=FailureType.DATABASE_SLOW,
        duration=150.0,
        intensity=0.7,
        target="main_database"
    ))
    
    return experiment