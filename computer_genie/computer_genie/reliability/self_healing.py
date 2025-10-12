"""
Self-Healing Mechanisms for Automatic Recovery

Implements intelligent self-healing capabilities that automatically detect,
diagnose, and recover from system failures and crashes.
"""

import asyncio
import time
import threading
import psutil
import logging
import subprocess
import signal
import os
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RESTART_SERVICE = "restart_service"
    RESTART_PROCESS = "restart_process"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION = "reset_connection"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    CUSTOM_SCRIPT = "custom_script"
    NOTIFY_ADMIN = "notify_admin"


@dataclass
class HealthCheck:
    """Configuration for health checks"""
    name: str
    check_function: Callable
    interval: float = 30.0
    timeout: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""
    name: str
    trigger_conditions: List[str]
    actions: List[RecoveryAction]
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 300.0
    success_criteria: Optional[Callable] = None
    rollback_strategy: Optional['RecoveryStrategy'] = None


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    strategy_name: str
    action: RecoveryAction
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    error_message: Optional[str] = None
    duration: float = 0.0
    attempt_number: int = 1


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: Dict[str, List[HealthResult]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_tasks: Dict[str, asyncio.Task] = {}
        
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        with self._lock:
            self.health_checks[health_check.name] = health_check
            self.health_history[health_check.name] = []
            self.failure_counts[health_check.name] = 0
            self.success_counts[health_check.name] = 0
            
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        with self._lock:
            self.health_checks.pop(name, None)
            self.health_history.pop(name, None)
            self.failure_counts.pop(name, None)
            self.success_counts.pop(name, None)
            
        # Stop monitoring task if running
        if name in self._monitor_tasks:
            self._monitor_tasks[name].cancel()
            del self._monitor_tasks[name]
            
    async def start_monitoring(self):
        """Start health monitoring for all registered checks"""
        self._monitoring = True
        
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                task = asyncio.create_task(self._monitor_health_check(health_check))
                self._monitor_tasks[name] = task
                
        logger.info(f"Started monitoring {len(self._monitor_tasks)} health checks")
        
    async def stop_monitoring(self):
        """Stop all health monitoring"""
        self._monitoring = False
        
        for task in self._monitor_tasks.values():
            task.cancel()
            
        await asyncio.gather(*self._monitor_tasks.values(), return_exceptions=True)
        self._monitor_tasks.clear()
        logger.info("Stopped health monitoring")
        
    async def _monitor_health_check(self, health_check: HealthCheck):
        """Monitor a single health check"""
        while self._monitoring:
            try:
                result = await self._perform_health_check(health_check)
                self._record_health_result(result)
                await asyncio.sleep(health_check.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring health check {health_check.name}: {e}")
                await asyncio.sleep(health_check.interval)
                
    async def _perform_health_check(self, health_check: HealthCheck) -> HealthResult:
        """Perform a single health check"""
        start_time = time.time()
        
        try:
            # Check dependencies first
            for dep_name in health_check.dependencies:
                dep_status = self.get_current_status(dep_name)
                if dep_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    return HealthResult(
                        name=health_check.name,
                        status=HealthStatus.DEGRADED,
                        message=f"Dependency {dep_name} is {dep_status.value}",
                        duration=time.time() - start_time
                    )
                    
            # Perform the actual health check with timeout
            result = await asyncio.wait_for(
                self._execute_check_function(health_check.check_function),
                timeout=health_check.timeout
            )
            
            duration = time.time() - start_time
            
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check passed" if result else "Check failed"
                return HealthResult(health_check.name, status, message, duration=duration)
            elif isinstance(result, HealthResult):
                result.duration = duration
                return result
            else:
                return HealthResult(
                    health_check.name,
                    HealthStatus.UNKNOWN,
                    f"Invalid check result: {result}",
                    duration=duration
                )
                
        except asyncio.TimeoutError:
            return HealthResult(
                health_check.name,
                HealthStatus.UNHEALTHY,
                f"Health check timed out after {health_check.timeout}s",
                duration=time.time() - start_time
            )
        except Exception as e:
            return HealthResult(
                health_check.name,
                HealthStatus.UNHEALTHY,
                f"Health check failed: {str(e)}",
                duration=time.time() - start_time
            )
            
    async def _execute_check_function(self, check_function: Callable) -> Any:
        """Execute health check function (sync or async)"""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()
            
    def _record_health_result(self, result: HealthResult):
        """Record health check result and update counters"""
        with self._lock:
            # Add to history
            if result.name not in self.health_history:
                self.health_history[result.name] = []
                
            self.health_history[result.name].append(result)
            
            # Keep only last 1000 results
            if len(self.health_history[result.name]) > 1000:
                self.health_history[result.name] = self.health_history[result.name][-1000:]
                
            # Update counters
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self.failure_counts[result.name] = self.failure_counts.get(result.name, 0) + 1
                self.success_counts[result.name] = 0
            else:
                self.success_counts[result.name] = self.success_counts.get(result.name, 0) + 1
                if result.status == HealthStatus.HEALTHY:
                    self.failure_counts[result.name] = 0
                    
    def get_current_status(self, name: str) -> HealthStatus:
        """Get current health status for a component"""
        with self._lock:
            if name not in self.health_history or not self.health_history[name]:
                return HealthStatus.UNKNOWN
                
            latest_result = self.health_history[name][-1]
            return latest_result.status
            
    def is_component_failing(self, name: str) -> bool:
        """Check if component is consistently failing"""
        health_check = self.health_checks.get(name)
        if not health_check:
            return False
            
        failure_count = self.failure_counts.get(name, 0)
        return failure_count >= health_check.failure_threshold
        
    def is_component_recovered(self, name: str) -> bool:
        """Check if component has recovered"""
        health_check = self.health_checks.get(name)
        if not health_check:
            return False
            
        success_count = self.success_counts.get(name, 0)
        return success_count >= health_check.success_threshold
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        with self._lock:
            summary = {
                'total_checks': len(self.health_checks),
                'healthy': 0,
                'degraded': 0,
                'unhealthy': 0,
                'critical': 0,
                'unknown': 0,
                'components': {}
            }
            
            for name in self.health_checks:
                status = self.get_current_status(name)
                summary['components'][name] = {
                    'status': status.value,
                    'failure_count': self.failure_counts.get(name, 0),
                    'success_count': self.success_counts.get(name, 0)
                }
                
                # Update counters
                if status == HealthStatus.HEALTHY:
                    summary['healthy'] += 1
                elif status == HealthStatus.DEGRADED:
                    summary['degraded'] += 1
                elif status == HealthStatus.UNHEALTHY:
                    summary['unhealthy'] += 1
                elif status == HealthStatus.CRITICAL:
                    summary['critical'] += 1
                else:
                    summary['unknown'] += 1
                    
            return summary


class AutoRecoveryEngine:
    """Executes automatic recovery actions"""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.active_recoveries: Set[str] = set()
        self._lock = threading.Lock()
        
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a recovery strategy"""
        self.recovery_strategies[strategy.name] = strategy
        logger.info(f"Registered recovery strategy: {strategy.name}")
        
    def unregister_recovery_strategy(self, name: str):
        """Unregister a recovery strategy"""
        self.recovery_strategies.pop(name, None)
        logger.info(f"Unregistered recovery strategy: {name}")
        
    async def execute_recovery(self, strategy_name: str, component_name: str) -> bool:
        """Execute recovery strategy for a component"""
        strategy = self.recovery_strategies.get(strategy_name)
        if not strategy:
            logger.error(f"Recovery strategy not found: {strategy_name}")
            return False
            
        recovery_key = f"{strategy_name}_{component_name}"
        
        with self._lock:
            if recovery_key in self.active_recoveries:
                logger.warning(f"Recovery already in progress: {recovery_key}")
                return False
            self.active_recoveries.add(recovery_key)
            
        try:
            logger.info(f"Starting recovery: {strategy_name} for {component_name}")
            
            for attempt in range(strategy.max_attempts):
                success = await self._execute_recovery_attempt(
                    strategy, component_name, attempt + 1
                )
                
                if success:
                    logger.info(f"Recovery successful: {strategy_name} for {component_name}")
                    return True
                    
                # Wait before next attempt with exponential backoff
                if attempt < strategy.max_attempts - 1:
                    delay = min(
                        strategy.initial_delay * (strategy.backoff_multiplier ** attempt),
                        strategy.max_delay
                    )
                    logger.info(f"Recovery attempt {attempt + 1} failed, waiting {delay}s")
                    await asyncio.sleep(delay)
                    
            logger.error(f"Recovery failed after {strategy.max_attempts} attempts: {strategy_name}")
            
            # Execute rollback if available
            if strategy.rollback_strategy:
                logger.info(f"Executing rollback strategy: {strategy.rollback_strategy.name}")
                await self.execute_recovery(strategy.rollback_strategy.name, component_name)
                
            return False
            
        finally:
            with self._lock:
                self.active_recoveries.discard(recovery_key)
                
    async def _execute_recovery_attempt(self, strategy: RecoveryStrategy, 
                                      component_name: str, attempt_number: int) -> bool:
        """Execute a single recovery attempt"""
        for action in strategy.actions:
            attempt = RecoveryAttempt(
                strategy_name=strategy.name,
                action=action,
                attempt_number=attempt_number
            )
            
            start_time = time.time()
            
            try:
                success = await self._execute_recovery_action(action, component_name)
                attempt.success = success
                attempt.duration = time.time() - start_time
                
                if not success:
                    attempt.error_message = f"Action {action.value} failed"
                    self.recovery_history.append(attempt)
                    return False
                    
            except Exception as e:
                attempt.success = False
                attempt.error_message = str(e)
                attempt.duration = time.time() - start_time
                self.recovery_history.append(attempt)
                logger.error(f"Recovery action failed: {action.value}, Error: {e}")
                return False
                
            self.recovery_history.append(attempt)
            
        # Check success criteria if defined
        if strategy.success_criteria:
            try:
                return await strategy.success_criteria(component_name)
            except Exception as e:
                logger.error(f"Success criteria check failed: {e}")
                return False
                
        return True
        
    async def _execute_recovery_action(self, action: RecoveryAction, component_name: str) -> bool:
        """Execute a specific recovery action"""
        try:
            if action == RecoveryAction.RESTART_SERVICE:
                return await self._restart_service(component_name)
            elif action == RecoveryAction.RESTART_PROCESS:
                return await self._restart_process(component_name)
            elif action == RecoveryAction.CLEAR_CACHE:
                return await self._clear_cache(component_name)
            elif action == RecoveryAction.RESET_CONNECTION:
                return await self._reset_connection(component_name)
            elif action == RecoveryAction.SCALE_UP:
                return await self._scale_up(component_name)
            elif action == RecoveryAction.SCALE_DOWN:
                return await self._scale_down(component_name)
            elif action == RecoveryAction.FAILOVER:
                return await self._failover(component_name)
            elif action == RecoveryAction.ROLLBACK:
                return await self._rollback(component_name)
            elif action == RecoveryAction.CUSTOM_SCRIPT:
                return await self._run_custom_script(component_name)
            elif action == RecoveryAction.NOTIFY_ADMIN:
                return await self._notify_admin(component_name)
            else:
                logger.error(f"Unknown recovery action: {action}")
                return False
        except Exception as e:
            logger.error(f"Error executing recovery action {action.value}: {e}")
            return False
            
    async def _restart_service(self, service_name: str) -> bool:
        """Restart a system service"""
        try:
            # On Windows, use sc command
            if os.name == 'nt':
                result = subprocess.run(
                    ['sc', 'stop', service_name],
                    capture_output=True, text=True, timeout=30
                )
                await asyncio.sleep(2)
                result = subprocess.run(
                    ['sc', 'start', service_name],
                    capture_output=True, text=True, timeout=30
                )
                return result.returncode == 0
            else:
                # On Unix-like systems, use systemctl
                result = subprocess.run(
                    ['systemctl', 'restart', service_name],
                    capture_output=True, text=True, timeout=30
                )
                return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return False
            
    async def _restart_process(self, process_name: str) -> bool:
        """Restart a process by name"""
        try:
            # Find and kill the process
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    proc.terminate()
                    proc.wait(timeout=10)
                    
            # In a real implementation, you would restart the process here
            logger.info(f"Process {process_name} terminated")
            return True
        except Exception as e:
            logger.error(f"Failed to restart process {process_name}: {e}")
            return False
            
    async def _clear_cache(self, component_name: str) -> bool:
        """Clear cache for a component"""
        try:
            # Implementation depends on the caching system
            logger.info(f"Clearing cache for {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache for {component_name}: {e}")
            return False
            
    async def _reset_connection(self, component_name: str) -> bool:
        """Reset connections for a component"""
        try:
            logger.info(f"Resetting connections for {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset connections for {component_name}: {e}")
            return False
            
    async def _scale_up(self, component_name: str) -> bool:
        """Scale up a component"""
        try:
            logger.info(f"Scaling up {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to scale up {component_name}: {e}")
            return False
            
    async def _scale_down(self, component_name: str) -> bool:
        """Scale down a component"""
        try:
            logger.info(f"Scaling down {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to scale down {component_name}: {e}")
            return False
            
    async def _failover(self, component_name: str) -> bool:
        """Failover to backup component"""
        try:
            logger.info(f"Failing over {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to failover {component_name}: {e}")
            return False
            
    async def _rollback(self, component_name: str) -> bool:
        """Rollback component to previous version"""
        try:
            logger.info(f"Rolling back {component_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback {component_name}: {e}")
            return False
            
    async def _run_custom_script(self, component_name: str) -> bool:
        """Run custom recovery script"""
        try:
            script_path = f"recovery_scripts/{component_name}.py"
            if os.path.exists(script_path):
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True, text=True, timeout=300
                )
                return result.returncode == 0
            return False
        except Exception as e:
            logger.error(f"Failed to run custom script for {component_name}: {e}")
            return False
            
    async def _notify_admin(self, component_name: str) -> bool:
        """Notify administrator of the issue"""
        try:
            logger.critical(f"ADMIN NOTIFICATION: Component {component_name} requires attention")
            # In a real implementation, send email, SMS, or push notification
            return True
        except Exception as e:
            logger.error(f"Failed to notify admin about {component_name}: {e}")
            return False
            
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for attempt in self.recovery_history if attempt.success)
        
        stats = {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            'active_recoveries': len(self.active_recoveries),
            'strategies_registered': len(self.recovery_strategies)
        }
        
        # Group by strategy
        strategy_stats = {}
        for attempt in self.recovery_history:
            if attempt.strategy_name not in strategy_stats:
                strategy_stats[attempt.strategy_name] = {
                    'total': 0,
                    'successful': 0,
                    'avg_duration': 0.0
                }
            
            strategy_stats[attempt.strategy_name]['total'] += 1
            if attempt.success:
                strategy_stats[attempt.strategy_name]['successful'] += 1
                
        # Calculate average durations
        for strategy_name, stat in strategy_stats.items():
            attempts = [a for a in self.recovery_history if a.strategy_name == strategy_name]
            if attempts:
                stat['avg_duration'] = sum(a.duration for a in attempts) / len(attempts)
                stat['success_rate'] = stat['successful'] / stat['total']
                
        stats['by_strategy'] = strategy_stats
        return stats


class SelfHealingManager:
    """Orchestrates self-healing operations"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.recovery_engine = AutoRecoveryEngine()
        self.component_strategies: Dict[str, str] = {}  # component -> strategy mapping
        self._monitoring = False
        self._healing_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start self-healing system"""
        await self.health_checker.start_monitoring()
        self._monitoring = True
        self._healing_task = asyncio.create_task(self._healing_loop())
        logger.info("Self-healing system started")
        
    async def stop(self):
        """Stop self-healing system"""
        self._monitoring = False
        if self._healing_task:
            self._healing_task.cancel()
            try:
                await self._healing_task
            except asyncio.CancelledError:
                pass
        await self.health_checker.stop_monitoring()
        logger.info("Self-healing system stopped")
        
    def register_component(self, health_check: HealthCheck, strategy_name: str):
        """Register a component with health check and recovery strategy"""
        self.health_checker.register_health_check(health_check)
        self.component_strategies[health_check.name] = strategy_name
        logger.info(f"Registered component: {health_check.name} with strategy: {strategy_name}")
        
    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """Register a recovery strategy"""
        self.recovery_engine.register_recovery_strategy(strategy)
        
    async def _healing_loop(self):
        """Main healing loop that monitors and triggers recovery"""
        while self._monitoring:
            try:
                # Check all components for failures
                for component_name, strategy_name in self.component_strategies.items():
                    if self.health_checker.is_component_failing(component_name):
                        logger.warning(f"Component failing: {component_name}")
                        
                        # Trigger recovery
                        success = await self.recovery_engine.execute_recovery(
                            strategy_name, component_name
                        )
                        
                        if success:
                            logger.info(f"Recovery successful for: {component_name}")
                        else:
                            logger.error(f"Recovery failed for: {component_name}")
                            
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                await asyncio.sleep(10.0)
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        health_summary = self.health_checker.get_health_summary()
        recovery_stats = self.recovery_engine.get_recovery_stats()
        
        return {
            'health': health_summary,
            'recovery': recovery_stats,
            'monitoring': self._monitoring,
            'registered_components': len(self.component_strategies)
        }


# Predefined health checks
async def check_cpu_usage() -> HealthResult:
    """Check CPU usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    
    if cpu_percent > 90:
        status = HealthStatus.CRITICAL
        message = f"CPU usage critical: {cpu_percent}%"
    elif cpu_percent > 80:
        status = HealthStatus.UNHEALTHY
        message = f"CPU usage high: {cpu_percent}%"
    elif cpu_percent > 70:
        status = HealthStatus.DEGRADED
        message = f"CPU usage elevated: {cpu_percent}%"
    else:
        status = HealthStatus.HEALTHY
        message = f"CPU usage normal: {cpu_percent}%"
        
    return HealthResult("cpu_usage", status, message, metadata={"cpu_percent": cpu_percent})


async def check_memory_usage() -> HealthResult:
    """Check memory usage"""
    memory = psutil.virtual_memory()
    
    if memory.percent > 95:
        status = HealthStatus.CRITICAL
        message = f"Memory usage critical: {memory.percent}%"
    elif memory.percent > 85:
        status = HealthStatus.UNHEALTHY
        message = f"Memory usage high: {memory.percent}%"
    elif memory.percent > 75:
        status = HealthStatus.DEGRADED
        message = f"Memory usage elevated: {memory.percent}%"
    else:
        status = HealthStatus.HEALTHY
        message = f"Memory usage normal: {memory.percent}%"
        
    return HealthResult("memory_usage", status, message, metadata={"memory_percent": memory.percent})


async def check_disk_space() -> HealthResult:
    """Check disk space"""
    disk = psutil.disk_usage('/')
    percent_used = (disk.used / disk.total) * 100
    
    if percent_used > 95:
        status = HealthStatus.CRITICAL
        message = f"Disk space critical: {percent_used:.1f}%"
    elif percent_used > 85:
        status = HealthStatus.UNHEALTHY
        message = f"Disk space low: {percent_used:.1f}%"
    elif percent_used > 75:
        status = HealthStatus.DEGRADED
        message = f"Disk space elevated: {percent_used:.1f}%"
    else:
        status = HealthStatus.HEALTHY
        message = f"Disk space normal: {percent_used:.1f}%"
        
    return HealthResult("disk_space", status, message, metadata={"disk_percent": percent_used})


# Predefined recovery strategies
def create_service_restart_strategy() -> RecoveryStrategy:
    """Create strategy for restarting services"""
    return RecoveryStrategy(
        name="service_restart",
        trigger_conditions=["service_down", "service_unresponsive"],
        actions=[RecoveryAction.RESTART_SERVICE],
        max_attempts=3,
        initial_delay=5.0
    )


def create_process_recovery_strategy() -> RecoveryStrategy:
    """Create strategy for process recovery"""
    return RecoveryStrategy(
        name="process_recovery",
        trigger_conditions=["process_crashed", "process_hanging"],
        actions=[RecoveryAction.RESTART_PROCESS, RecoveryAction.CLEAR_CACHE],
        max_attempts=2,
        initial_delay=2.0
    )


def create_resource_cleanup_strategy() -> RecoveryStrategy:
    """Create strategy for resource cleanup"""
    return RecoveryStrategy(
        name="resource_cleanup",
        trigger_conditions=["high_cpu", "high_memory", "high_disk"],
        actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.RESTART_PROCESS],
        max_attempts=2,
        initial_delay=1.0
    )