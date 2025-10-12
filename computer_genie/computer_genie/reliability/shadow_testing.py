"""
Shadow Testing Infrastructure for Safe Production Testing

Implements comprehensive shadow testing to safely test new code versions
in production by running them alongside the current version without affecting users.
"""

import asyncio
import time
import json
import logging
import threading
import uuid
import copy
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
import pickle

logger = logging.getLogger(__name__)


class ShadowTestStatus(Enum):
    """Shadow test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ComparisonResult(Enum):
    """Result of comparing primary and shadow responses"""
    MATCH = "match"
    MISMATCH = "mismatch"
    SHADOW_ERROR = "shadow_error"
    PRIMARY_ERROR = "primary_error"
    BOTH_ERROR = "both_error"
    TIMEOUT = "timeout"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies for shadow testing"""
    PERCENTAGE = "percentage"
    USER_BASED = "user_based"
    FEATURE_FLAG = "feature_flag"
    CANARY = "canary"
    A_B_TEST = "a_b_test"


@dataclass
class ShadowTestConfig:
    """Configuration for shadow testing"""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    enabled: bool = True
    traffic_percentage: float = 10.0  # Percentage of traffic to shadow test
    max_concurrent_tests: int = 100
    timeout_seconds: float = 30.0
    compare_responses: bool = True
    log_mismatches: bool = True
    alert_on_errors: bool = True
    error_threshold: float = 5.0  # Error rate threshold (%)
    performance_threshold: float = 2.0  # Performance degradation threshold (multiplier)
    sample_rate: float = 1.0  # Sampling rate for logging
    split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.PERCENTAGE
    target_users: Set[str] = field(default_factory=set)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowRequest:
    """Request data for shadow testing"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = "GET"
    path: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'method': self.method,
            'path': self.path,
            'headers': self.headers,
            'query_params': self.query_params,
            'body': self.body.decode('utf-8') if self.body else None,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ShadowResponse:
    """Response data from shadow testing"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'response_id': self.response_id,
            'request_id': self.request_id,
            'status_code': self.status_code,
            'headers': self.headers,
            'body': self.body.decode('utf-8') if self.body else None,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'error': self.error,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ShadowTestResult:
    """Result of a shadow test execution"""
    test_id: str = ""
    request_id: str = ""
    primary_response: Optional[ShadowResponse] = None
    shadow_response: Optional[ShadowResponse] = None
    comparison_result: ComparisonResult = ComparisonResult.MATCH
    differences: List[str] = field(default_factory=list)
    performance_ratio: float = 1.0  # Shadow time / Primary time
    status: ShadowTestStatus = ShadowTestStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get test duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_id': self.test_id,
            'request_id': self.request_id,
            'primary_response': self.primary_response.to_dict() if self.primary_response else None,
            'shadow_response': self.shadow_response.to_dict() if self.shadow_response else None,
            'comparison_result': self.comparison_result.value,
            'differences': self.differences,
            'performance_ratio': self.performance_ratio,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class ResponseComparator(ABC):
    """Abstract base class for response comparison"""
    
    @abstractmethod
    def compare(self, primary: ShadowResponse, shadow: ShadowResponse) -> Tuple[ComparisonResult, List[str]]:
        """Compare two responses and return result with differences"""
        pass


class DefaultResponseComparator(ResponseComparator):
    """Default response comparator"""
    
    def __init__(self, ignore_headers: Set[str] = None, tolerance: float = 0.001):
        self.ignore_headers = ignore_headers or {'date', 'server', 'x-request-id', 'x-trace-id'}
        self.tolerance = tolerance
        
    def compare(self, primary: ShadowResponse, shadow: ShadowResponse) -> Tuple[ComparisonResult, List[str]]:
        """Compare two responses"""
        differences = []
        
        # Check for errors
        if primary.error and shadow.error:
            return ComparisonResult.BOTH_ERROR, ["Both responses had errors"]
        elif primary.error:
            return ComparisonResult.PRIMARY_ERROR, [f"Primary error: {primary.error}"]
        elif shadow.error:
            return ComparisonResult.SHADOW_ERROR, [f"Shadow error: {shadow.error}"]
            
        # Compare status codes
        if primary.status_code != shadow.status_code:
            differences.append(f"Status code: {primary.status_code} vs {shadow.status_code}")
            
        # Compare headers (excluding ignored ones)
        primary_headers = {k.lower(): v for k, v in primary.headers.items() 
                          if k.lower() not in self.ignore_headers}
        shadow_headers = {k.lower(): v for k, v in shadow.headers.items() 
                         if k.lower() not in self.ignore_headers}
                         
        for key in set(primary_headers.keys()) | set(shadow_headers.keys()):
            if primary_headers.get(key) != shadow_headers.get(key):
                differences.append(f"Header {key}: {primary_headers.get(key)} vs {shadow_headers.get(key)}")
                
        # Compare response bodies
        if primary.body != shadow.body:
            # Try to parse as JSON for better comparison
            try:
                primary_json = json.loads(primary.body.decode('utf-8')) if primary.body else None
                shadow_json = json.loads(shadow.body.decode('utf-8')) if shadow.body else None
                
                if primary_json != shadow_json:
                    differences.append("Response body JSON differs")
                    
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to byte comparison
                differences.append("Response body differs")
                
        # Check performance difference
        if shadow.execution_time > primary.execution_time * (1 + self.tolerance):
            ratio = shadow.execution_time / primary.execution_time if primary.execution_time > 0 else float('inf')
            differences.append(f"Performance degradation: {ratio:.2f}x slower")
            
        return ComparisonResult.MATCH if not differences else ComparisonResult.MISMATCH, differences


class ShadowExecutor(ABC):
    """Abstract base class for shadow execution"""
    
    @abstractmethod
    async def execute_primary(self, request: ShadowRequest) -> ShadowResponse:
        """Execute request on primary system"""
        pass
        
    @abstractmethod
    async def execute_shadow(self, request: ShadowRequest) -> ShadowResponse:
        """Execute request on shadow system"""
        pass


class HTTPShadowExecutor(ShadowExecutor):
    """HTTP-based shadow executor"""
    
    def __init__(self, primary_base_url: str, shadow_base_url: str):
        self.primary_base_url = primary_base_url.rstrip('/')
        self.shadow_base_url = shadow_base_url.rstrip('/')
        
    async def execute_primary(self, request: ShadowRequest) -> ShadowResponse:
        """Execute request on primary system"""
        return await self._execute_request(self.primary_base_url, request, "primary")
        
    async def execute_shadow(self, request: ShadowRequest) -> ShadowResponse:
        """Execute request on shadow system"""
        return await self._execute_request(self.shadow_base_url, request, "shadow")
        
    async def _execute_request(self, base_url: str, request: ShadowRequest, system: str) -> ShadowResponse:
        """Execute HTTP request"""
        import aiohttp
        
        start_time = time.time()
        response = ShadowResponse(request_id=request.request_id)
        
        try:
            url = f"{base_url}{request.path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers=request.headers,
                    params=request.query_params,
                    data=request.body
                ) as resp:
                    response.status_code = resp.status
                    response.headers = dict(resp.headers)
                    response.body = await resp.read()
                    
        except Exception as e:
            response.error = str(e)
            logger.error(f"Error executing {system} request: {e}")
            
        response.execution_time = time.time() - start_time
        response.metadata['system'] = system
        
        return response


class FunctionShadowExecutor(ShadowExecutor):
    """Function-based shadow executor for testing function calls"""
    
    def __init__(self, primary_func: Callable, shadow_func: Callable):
        self.primary_func = primary_func
        self.shadow_func = shadow_func
        
    async def execute_primary(self, request: ShadowRequest) -> ShadowResponse:
        """Execute primary function"""
        return await self._execute_function(self.primary_func, request, "primary")
        
    async def execute_shadow(self, request: ShadowRequest) -> ShadowResponse:
        """Execute shadow function"""
        return await self._execute_function(self.shadow_func, request, "shadow")
        
    async def _execute_function(self, func: Callable, request: ShadowRequest, system: str) -> ShadowResponse:
        """Execute function with request data"""
        start_time = time.time()
        response = ShadowResponse(request_id=request.request_id)
        
        try:
            # Convert request to function arguments
            args = request.metadata.get('args', [])
            kwargs = request.metadata.get('kwargs', {})
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Convert result to response
            response.status_code = 200
            response.body = json.dumps(result).encode('utf-8') if result is not None else b''
            
        except Exception as e:
            response.error = str(e)
            response.status_code = 500
            logger.error(f"Error executing {system} function: {e}")
            
        response.execution_time = time.time() - start_time
        response.metadata['system'] = system
        
        return response


class TrafficSplitter:
    """Handles traffic splitting for shadow testing"""
    
    def __init__(self, config: ShadowTestConfig):
        self.config = config
        
    def should_shadow_test(self, request: ShadowRequest) -> bool:
        """Determine if request should be shadow tested"""
        if not self.config.enabled:
            return False
            
        if self.config.split_strategy == TrafficSplitStrategy.PERCENTAGE:
            return self._percentage_split(request)
        elif self.config.split_strategy == TrafficSplitStrategy.USER_BASED:
            return self._user_based_split(request)
        elif self.config.split_strategy == TrafficSplitStrategy.FEATURE_FLAG:
            return self._feature_flag_split(request)
        else:
            return self._percentage_split(request)
            
    def _percentage_split(self, request: ShadowRequest) -> bool:
        """Split traffic based on percentage"""
        # Use request ID hash for consistent splitting
        hash_value = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < self.config.traffic_percentage
        
    def _user_based_split(self, request: ShadowRequest) -> bool:
        """Split traffic based on user ID"""
        if not request.user_id:
            return False
            
        if self.config.target_users:
            return request.user_id in self.config.target_users
            
        # Use user ID hash for consistent splitting
        hash_value = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < self.config.traffic_percentage
        
    def _feature_flag_split(self, request: ShadowRequest) -> bool:
        """Split traffic based on feature flags"""
        for flag, enabled in self.config.feature_flags.items():
            if enabled and request.metadata.get(f'feature_{flag}'):
                return True
        return False


class ShadowTestMetrics:
    """Collects and analyzes shadow test metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.results: deque = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.performance_samples = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
    def add_result(self, result: ShadowTestResult):
        """Add test result to metrics"""
        with self._lock:
            self.results.append(result)
            
            if result.comparison_result != ComparisonResult.MATCH:
                self.error_counts[result.comparison_result] += 1
                
            if result.primary_response and result.shadow_response:
                self.performance_samples.append(result.performance_ratio)
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            if not self.results:
                return {}
                
            total_tests = len(self.results)
            matches = sum(1 for r in self.results if r.comparison_result == ComparisonResult.MATCH)
            
            # Calculate error rates
            error_rates = {}
            for error_type, count in self.error_counts.items():
                error_rates[error_type.value] = (count / total_tests) * 100
                
            # Calculate performance metrics
            performance_metrics = {}
            if self.performance_samples:
                performance_metrics = {
                    'mean_ratio': statistics.mean(self.performance_samples),
                    'median_ratio': statistics.median(self.performance_samples),
                    'p95_ratio': self._percentile(self.performance_samples, 95),
                    'p99_ratio': self._percentile(self.performance_samples, 99)
                }
                
            # Calculate recent trends
            recent_results = list(self.results)[-100:]  # Last 100 results
            recent_matches = sum(1 for r in recent_results if r.comparison_result == ComparisonResult.MATCH)
            recent_match_rate = (recent_matches / len(recent_results)) * 100 if recent_results else 0
            
            return {
                'total_tests': total_tests,
                'match_rate': (matches / total_tests) * 100,
                'recent_match_rate': recent_match_rate,
                'error_rates': error_rates,
                'performance_metrics': performance_metrics,
                'window_size': self.window_size,
                'timestamp': time.time()
            }
            
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
            
    def should_alert(self, config: ShadowTestConfig) -> Tuple[bool, List[str]]:
        """Check if metrics warrant an alert"""
        metrics = self.get_metrics()
        alerts = []
        
        if not metrics:
            return False, alerts
            
        # Check error rate threshold
        total_error_rate = 100 - metrics['match_rate']
        if total_error_rate > config.error_threshold:
            alerts.append(f"Error rate {total_error_rate:.1f}% exceeds threshold {config.error_threshold}%")
            
        # Check performance threshold
        perf_metrics = metrics.get('performance_metrics', {})
        if perf_metrics.get('mean_ratio', 1.0) > config.performance_threshold:
            alerts.append(f"Performance ratio {perf_metrics['mean_ratio']:.2f} exceeds threshold {config.performance_threshold}")
            
        return len(alerts) > 0, alerts


class ShadowTestManager:
    """Manages shadow testing operations"""
    
    def __init__(self, executor: ShadowExecutor, 
                 comparator: ResponseComparator = None,
                 config: ShadowTestConfig = None):
        self.executor = executor
        self.comparator = comparator or DefaultResponseComparator()
        self.config = config or ShadowTestConfig()
        self.traffic_splitter = TrafficSplitter(self.config)
        self.metrics = ShadowTestMetrics()
        self.active_tests: Dict[str, ShadowTestResult] = {}
        self.completed_tests: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tests)
        
    async def test_request(self, request: ShadowRequest) -> Optional[ShadowTestResult]:
        """Test a request with shadow testing"""
        # Check if request should be shadow tested
        if not self.traffic_splitter.should_shadow_test(request):
            return None
            
        # Check sampling rate
        if self.config.sample_rate < 1.0:
            hash_value = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
            if (hash_value % 100) / 100.0 > self.config.sample_rate:
                return None
                
        # Create test result
        test_result = ShadowTestResult(
            test_id=self.config.test_id,
            request_id=request.request_id,
            status=ShadowTestStatus.PENDING
        )
        
        with self._lock:
            self.active_tests[request.request_id] = test_result
            
        try:
            async with self._semaphore:
                await self._execute_shadow_test(request, test_result)
                
        except Exception as e:
            test_result.status = ShadowTestStatus.FAILED
            test_result.error_message = str(e)
            logger.error(f"Shadow test failed: {e}")
            
        finally:
            test_result.end_time = time.time()
            
            with self._lock:
                self.active_tests.pop(request.request_id, None)
                self.completed_tests.append(test_result)
                
            # Update metrics
            self.metrics.add_result(test_result)
            
            # Check for alerts
            if self.config.alert_on_errors:
                should_alert, alerts = self.metrics.should_alert(self.config)
                if should_alert:
                    await self._send_alerts(alerts, test_result)
                    
        return test_result
        
    async def _execute_shadow_test(self, request: ShadowRequest, test_result: ShadowTestResult):
        """Execute shadow test"""
        test_result.status = ShadowTestStatus.RUNNING
        
        # Execute primary and shadow requests concurrently
        primary_task = asyncio.create_task(self.executor.execute_primary(request))
        shadow_task = asyncio.create_task(self.executor.execute_shadow(request))
        
        try:
            # Wait for both with timeout
            primary_response, shadow_response = await asyncio.wait_for(
                asyncio.gather(primary_task, shadow_task, return_exceptions=True),
                timeout=self.config.timeout_seconds
            )
            
            # Handle exceptions
            if isinstance(primary_response, Exception):
                test_result.primary_response = ShadowResponse(
                    request_id=request.request_id,
                    error=str(primary_response)
                )
            else:
                test_result.primary_response = primary_response
                
            if isinstance(shadow_response, Exception):
                test_result.shadow_response = ShadowResponse(
                    request_id=request.request_id,
                    error=str(shadow_response)
                )
            else:
                test_result.shadow_response = shadow_response
                
            # Compare responses if enabled
            if self.config.compare_responses and test_result.primary_response and test_result.shadow_response:
                comparison_result, differences = self.comparator.compare(
                    test_result.primary_response,
                    test_result.shadow_response
                )
                
                test_result.comparison_result = comparison_result
                test_result.differences = differences
                
                # Calculate performance ratio
                if (test_result.primary_response.execution_time > 0 and 
                    test_result.shadow_response.execution_time > 0):
                    test_result.performance_ratio = (
                        test_result.shadow_response.execution_time / 
                        test_result.primary_response.execution_time
                    )
                    
            test_result.status = ShadowTestStatus.COMPLETED
            
        except asyncio.TimeoutError:
            test_result.status = ShadowTestStatus.TIMEOUT
            test_result.error_message = f"Test timed out after {self.config.timeout_seconds}s"
            
            # Cancel pending tasks
            if not primary_task.done():
                primary_task.cancel()
            if not shadow_task.done():
                shadow_task.cancel()
                
    async def _send_alerts(self, alerts: List[str], test_result: ShadowTestResult):
        """Send alerts for test failures"""
        alert_data = {
            'test_id': self.config.test_id,
            'test_name': self.config.name,
            'alerts': alerts,
            'test_result': test_result.to_dict(),
            'metrics': self.metrics.get_metrics(),
            'timestamp': time.time()
        }
        
        logger.warning(f"Shadow test alerts: {alerts}")
        
        # Here you could integrate with alerting systems like:
        # - Slack/Teams notifications
        # - Email alerts
        # - PagerDuty
        # - Custom webhook endpoints
        
    def get_test_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent test results"""
        with self._lock:
            results = list(self.completed_tests)[-limit:]
            return [result.to_dict() for result in results]
            
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get currently active tests"""
        with self._lock:
            return [result.to_dict() for result in self.active_tests.values()]
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get shadow testing metrics"""
        return self.metrics.get_metrics()
        
    def update_config(self, config: ShadowTestConfig):
        """Update shadow test configuration"""
        self.config = config
        self.traffic_splitter = TrafficSplitter(config)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tests)
        
    async def stop_all_tests(self):
        """Stop all active tests"""
        with self._lock:
            for test_result in self.active_tests.values():
                test_result.status = ShadowTestStatus.CANCELLED
                test_result.end_time = time.time()
                
            self.active_tests.clear()


# Utility functions and decorators
def shadow_test(config: ShadowTestConfig = None):
    """Decorator for shadow testing functions"""
    def decorator(primary_func: Callable):
        def wrapper(*args, **kwargs):
            # This would integrate with the shadow testing infrastructure
            # For now, just execute the primary function
            return primary_func(*args, **kwargs)
        return wrapper
    return decorator


async def create_shadow_test_suite(test_configs: List[ShadowTestConfig],
                                 executor: ShadowExecutor) -> Dict[str, ShadowTestManager]:
    """Create a suite of shadow tests"""
    managers = {}
    
    for config in test_configs:
        manager = ShadowTestManager(executor, config=config)
        managers[config.test_id] = manager
        
    return managers


def create_http_shadow_test(name: str, primary_url: str, shadow_url: str,
                          traffic_percentage: float = 10.0) -> ShadowTestManager:
    """Create an HTTP shadow test"""
    config = ShadowTestConfig(
        name=name,
        traffic_percentage=traffic_percentage
    )
    
    executor = HTTPShadowExecutor(primary_url, shadow_url)
    return ShadowTestManager(executor, config=config)


def create_function_shadow_test(name: str, primary_func: Callable, shadow_func: Callable,
                              traffic_percentage: float = 10.0) -> ShadowTestManager:
    """Create a function shadow test"""
    config = ShadowTestConfig(
        name=name,
        traffic_percentage=traffic_percentage
    )
    
    executor = FunctionShadowExecutor(primary_func, shadow_func)
    return ShadowTestManager(executor, config=config)