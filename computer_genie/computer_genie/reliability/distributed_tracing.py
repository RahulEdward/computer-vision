"""
Distributed Tracing with OpenTelemetry

Implements comprehensive distributed tracing system for monitoring
and debugging distributed systems with OpenTelemetry integration.
"""

import asyncio
import time
import json
import logging
import threading
import uuid
import traceback
import inspect
import functools
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
import weakref

# OpenTelemetry imports (would be installed via pip)
try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Mock OpenTelemetry classes for development
    OPENTELEMETRY_AVAILABLE = False
    
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()
            
    class MockSpan:
        def __init__(self):
            self.span_id = str(uuid.uuid4())
            self.trace_id = str(uuid.uuid4())
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def set_attribute(self, key, value):
            pass
            
        def set_status(self, status):
            pass
            
        def record_exception(self, exception):
            pass
            
        def add_event(self, name, attributes=None):
            pass

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Span kinds for different types of operations"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class TraceStatus(Enum):
    """Trace status"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SamplingStrategy(Enum):
    """Sampling strategies"""
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    PROBABILISTIC = "probabilistic"
    RATE_LIMITED = "rate_limited"
    PARENT_BASED = "parent_based"


@dataclass
class TraceConfig:
    """Configuration for distributed tracing"""
    service_name: str = "computer-genie"
    service_version: str = "1.0.0"
    environment: str = "development"
    
    # Sampling configuration
    sampling_strategy: SamplingStrategy = SamplingStrategy.PROBABILISTIC
    sampling_rate: float = 0.1  # 10% sampling
    max_traces_per_second: int = 100
    
    # Export configuration
    jaeger_endpoint: Optional[str] = None
    zipkin_endpoint: Optional[str] = None
    console_export: bool = True
    
    # Instrumentation configuration
    auto_instrument_requests: bool = True
    auto_instrument_database: bool = True
    auto_instrument_async_http: bool = True
    
    # Custom configuration
    custom_attributes: Dict[str, str] = field(default_factory=dict)
    resource_attributes: Dict[str, str] = field(default_factory=dict)
    
    # Performance configuration
    max_span_attributes: int = 128
    max_span_events: int = 128
    max_span_links: int = 128
    span_timeout_seconds: float = 300.0  # 5 minutes


@dataclass
class SpanInfo:
    """Information about a span"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: TraceStatus = TraceStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    
    # Attributes and tags
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Events and logs
    events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    
    # Error information
    error: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Baggage and context
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value,
            'kind': self.kind.value,
            'attributes': self.attributes,
            'tags': self.tags,
            'events': self.events,
            'logs': self.logs,
            'error': self.error,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'stack_trace': self.stack_trace,
            'baggage': self.baggage
        }


@dataclass
class TraceMetrics:
    """Metrics for trace analysis"""
    timestamp: float = field(default_factory=time.time)
    
    # Span metrics
    total_spans: int = 0
    error_spans: int = 0
    timeout_spans: int = 0
    
    # Performance metrics
    avg_duration: float = 0.0
    p50_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    
    # Service metrics
    services_count: int = 0
    operations_count: int = 0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Throughput metrics
    traces_per_second: float = 0.0
    spans_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'total_spans': self.total_spans,
            'error_spans': self.error_spans,
            'timeout_spans': self.timeout_spans,
            'avg_duration': self.avg_duration,
            'p50_duration': self.p50_duration,
            'p95_duration': self.p95_duration,
            'p99_duration': self.p99_duration,
            'services_count': self.services_count,
            'operations_count': self.operations_count,
            'error_rate': self.error_rate,
            'timeout_rate': self.timeout_rate,
            'traces_per_second': self.traces_per_second,
            'spans_per_second': self.spans_per_second
        }


class SpanExporter(ABC):
    """Abstract base class for span exporters"""
    
    @abstractmethod
    async def export_spans(self, spans: List[SpanInfo]) -> bool:
        """Export spans to external system"""
        pass
        
    @abstractmethod
    async def shutdown(self):
        """Shutdown the exporter"""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Console span exporter for debugging"""
    
    def __init__(self, pretty_print: bool = True):
        self.pretty_print = pretty_print
        
    async def export_spans(self, spans: List[SpanInfo]) -> bool:
        """Export spans to console"""
        try:
            for span in spans:
                if self.pretty_print:
                    print(json.dumps(span.to_dict(), indent=2))
                else:
                    print(json.dumps(span.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Failed to export spans to console: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown console exporter"""
        pass


class JaegerSpanExporter(SpanExporter):
    """Jaeger span exporter"""
    
    def __init__(self, endpoint: str, service_name: str):
        self.endpoint = endpoint
        self.service_name = service_name
        self.session = None
        
    async def export_spans(self, spans: List[SpanInfo]) -> bool:
        """Export spans to Jaeger"""
        try:
            import aiohttp
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Convert spans to Jaeger format
            jaeger_spans = []
            for span in spans:
                jaeger_span = {
                    'traceID': span.trace_id,
                    'spanID': span.span_id,
                    'parentSpanID': span.parent_span_id,
                    'operationName': span.operation_name,
                    'startTime': int(span.start_time * 1000000),  # microseconds
                    'duration': int((span.duration or 0) * 1000000),  # microseconds
                    'tags': [{'key': k, 'value': v} for k, v in span.tags.items()],
                    'process': {
                        'serviceName': span.service_name or self.service_name,
                        'tags': []
                    }
                }
                jaeger_spans.append(jaeger_span)
                
            # Send to Jaeger
            payload = {'spans': jaeger_spans}
            async with self.session.post(self.endpoint, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown Jaeger exporter"""
        if self.session:
            await self.session.close()


class TraceSampler:
    """Handles trace sampling decisions"""
    
    def __init__(self, config: TraceConfig):
        self.config = config
        self.traces_this_second = 0
        self.last_second = int(time.time())
        self._lock = threading.Lock()
        
    def should_sample(self, trace_id: str, operation_name: str, 
                     parent_sampled: Optional[bool] = None) -> bool:
        """Determine if trace should be sampled"""
        
        if self.config.sampling_strategy == SamplingStrategy.ALWAYS_ON:
            return True
        elif self.config.sampling_strategy == SamplingStrategy.ALWAYS_OFF:
            return False
        elif self.config.sampling_strategy == SamplingStrategy.PARENT_BASED:
            return parent_sampled if parent_sampled is not None else self._probabilistic_sample(trace_id)
        elif self.config.sampling_strategy == SamplingStrategy.RATE_LIMITED:
            return self._rate_limited_sample()
        else:  # PROBABILISTIC
            return self._probabilistic_sample(trace_id)
            
    def _probabilistic_sample(self, trace_id: str) -> bool:
        """Probabilistic sampling based on trace ID"""
        # Use trace ID hash for consistent sampling
        hash_value = hash(trace_id) % 10000
        threshold = self.config.sampling_rate * 10000
        return hash_value < threshold
        
    def _rate_limited_sample(self) -> bool:
        """Rate-limited sampling"""
        with self._lock:
            current_second = int(time.time())
            
            if current_second != self.last_second:
                self.traces_this_second = 0
                self.last_second = current_second
                
            if self.traces_this_second < self.config.max_traces_per_second:
                self.traces_this_second += 1
                return True
                
            return False


class SpanContext:
    """Manages span context and propagation"""
    
    def __init__(self):
        self._current_span: Optional[SpanInfo] = None
        self._span_stack: List[SpanInfo] = []
        self._baggage: Dict[str, str] = {}
        
    def get_current_span(self) -> Optional[SpanInfo]:
        """Get current active span"""
        return self._current_span
        
    def set_current_span(self, span: Optional[SpanInfo]):
        """Set current active span"""
        self._current_span = span
        
    def push_span(self, span: SpanInfo):
        """Push span onto stack"""
        if self._current_span:
            self._span_stack.append(self._current_span)
        self._current_span = span
        
    def pop_span(self) -> Optional[SpanInfo]:
        """Pop span from stack"""
        current = self._current_span
        
        if self._span_stack:
            self._current_span = self._span_stack.pop()
        else:
            self._current_span = None
            
        return current
        
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value"""
        return self._baggage.get(key)
        
    def set_baggage(self, key: str, value: str):
        """Set baggage value"""
        self._baggage[key] = value
        
    def get_all_baggage(self) -> Dict[str, str]:
        """Get all baggage"""
        return self._baggage.copy()
        
    def inject_headers(self, headers: Dict[str, str]):
        """Inject trace context into headers"""
        if self._current_span:
            headers['X-Trace-Id'] = self._current_span.trace_id
            headers['X-Span-Id'] = self._current_span.span_id
            
        for key, value in self._baggage.items():
            headers[f'X-Baggage-{key}'] = value
            
    def extract_headers(self, headers: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """Extract trace context from headers"""
        trace_id = headers.get('X-Trace-Id')
        span_id = headers.get('X-Span-Id')
        
        # Extract baggage
        for key, value in headers.items():
            if key.startswith('X-Baggage-'):
                baggage_key = key[10:]  # Remove 'X-Baggage-' prefix
                self._baggage[baggage_key] = value
                
        if trace_id and span_id:
            return trace_id, span_id
            
        return None


class DistributedTracer:
    """Main distributed tracing implementation"""
    
    def __init__(self, config: TraceConfig):
        self.config = config
        self.sampler = TraceSampler(config)
        self.exporters: List[SpanExporter] = []
        self.active_spans: Dict[str, SpanInfo] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self.metrics_history: deque = deque(maxlen=1000)
        
        self._context = threading.local()
        self._lock = threading.Lock()
        self._export_task: Optional[asyncio.Task] = None
        self._export_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        
        # Initialize exporters
        self._initialize_exporters()
        
        # Start export task
        self._start_export_task()
        
    def _initialize_exporters(self):
        """Initialize span exporters"""
        if self.config.console_export:
            self.exporters.append(ConsoleSpanExporter())
            
        if self.config.jaeger_endpoint:
            self.exporters.append(JaegerSpanExporter(
                self.config.jaeger_endpoint,
                self.config.service_name
            ))
            
        if self.config.zipkin_endpoint:
            # Would implement ZipkinSpanExporter
            pass
            
    def _start_export_task(self):
        """Start background export task"""
        self._export_task = asyncio.create_task(self._export_loop())
        
    async def _export_loop(self):
        """Background loop for exporting spans"""
        batch = []
        batch_size = 100
        batch_timeout = 5.0
        last_export = time.time()
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for spans or timeout
                    span = await asyncio.wait_for(
                        self._export_queue.get(),
                        timeout=batch_timeout
                    )
                    batch.append(span)
                    
                    # Export when batch is full or timeout reached
                    if (len(batch) >= batch_size or 
                        time.time() - last_export >= batch_timeout):
                        
                        if batch:
                            await self._export_batch(batch)
                            batch.clear()
                            last_export = time.time()
                            
                except asyncio.TimeoutError:
                    # Export any pending spans
                    if batch:
                        await self._export_batch(batch)
                        batch.clear()
                        last_export = time.time()
                        
        except Exception as e:
            logger.error(f"Error in export loop: {e}")
            
    async def _export_batch(self, spans: List[SpanInfo]):
        """Export a batch of spans"""
        for exporter in self.exporters:
            try:
                await exporter.export_spans(spans)
            except Exception as e:
                logger.error(f"Failed to export spans with {type(exporter).__name__}: {e}")
                
    def _get_context(self) -> SpanContext:
        """Get thread-local span context"""
        if not hasattr(self._context, 'span_context'):
            self._context.span_context = SpanContext()
        return self._context.span_context
        
    def start_span(self, operation_name: str, 
                   parent_span: Optional[SpanInfo] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   attributes: Dict[str, Any] = None,
                   tags: Dict[str, str] = None) -> SpanInfo:
        """Start a new span"""
        
        context = self._get_context()
        current_span = parent_span or context.get_current_span()
        
        # Generate trace and span IDs
        if current_span:
            trace_id = current_span.trace_id
            parent_span_id = current_span.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
            
        span_id = str(uuid.uuid4())
        
        # Check sampling
        parent_sampled = current_span is not None if current_span else None
        if not self.sampler.should_sample(trace_id, operation_name, parent_sampled):
            # Return a no-op span
            return SpanInfo(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                service_name=self.config.service_name,
                kind=kind
            )
            
        # Create span
        span = SpanInfo(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.config.service_name,
            kind=kind,
            attributes=attributes or {},
            tags=tags or {},
            baggage=context.get_all_baggage()
        )
        
        # Add default attributes
        span.attributes.update({
            'service.name': self.config.service_name,
            'service.version': self.config.service_version,
            'environment': self.config.environment
        })
        
        # Add custom attributes
        span.attributes.update(self.config.custom_attributes)
        
        # Store active span
        with self._lock:
            self.active_spans[span_id] = span
            
        return span
        
    def finish_span(self, span: SpanInfo, 
                   status: TraceStatus = TraceStatus.OK,
                   error: Optional[Exception] = None):
        """Finish a span"""
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        span.status = status
        
        # Handle error
        if error:
            span.status = TraceStatus.ERROR
            span.error = str(error)
            span.exception_type = type(error).__name__
            span.exception_message = str(error)
            span.stack_trace = traceback.format_exc()
            
        # Remove from active spans
        with self._lock:
            self.active_spans.pop(span.span_id, None)
            self.completed_spans.append(span)
            
        # Queue for export
        try:
            self._export_queue.put_nowait(span)
        except asyncio.QueueFull:
            logger.warning("Export queue full, dropping span")
            
    @contextmanager
    def span(self, operation_name: str, **kwargs):
        """Context manager for spans"""
        context = self._get_context()
        span = self.start_span(operation_name, **kwargs)
        
        context.push_span(span)
        
        try:
            yield span
        except Exception as e:
            self.finish_span(span, TraceStatus.ERROR, e)
            raise
        else:
            self.finish_span(span, TraceStatus.OK)
        finally:
            context.pop_span()
            
    @asynccontextmanager
    async def async_span(self, operation_name: str, **kwargs):
        """Async context manager for spans"""
        context = self._get_context()
        span = self.start_span(operation_name, **kwargs)
        
        context.push_span(span)
        
        try:
            yield span
        except Exception as e:
            self.finish_span(span, TraceStatus.ERROR, e)
            raise
        else:
            self.finish_span(span, TraceStatus.OK)
        finally:
            context.pop_span()
            
    def trace_function(self, operation_name: str = None, 
                      kind: SpanKind = SpanKind.INTERNAL,
                      attributes: Dict[str, Any] = None):
        """Decorator for tracing functions"""
        def decorator(func):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__qualname__}"
                
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.async_span(operation_name, kind=kind, attributes=attributes) as span:
                        # Add function attributes
                        span.attributes.update({
                            'function.name': func.__name__,
                            'function.module': func.__module__,
                            'function.args_count': len(args),
                            'function.kwargs_count': len(kwargs)
                        })
                        
                        return await func(*args, **kwargs)
                        
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.span(operation_name, kind=kind, attributes=attributes) as span:
                        # Add function attributes
                        span.attributes.update({
                            'function.name': func.__name__,
                            'function.module': func.__module__,
                            'function.args_count': len(args),
                            'function.kwargs_count': len(kwargs)
                        })
                        
                        return func(*args, **kwargs)
                        
                return sync_wrapper
                
        return decorator
        
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to current span"""
        context = self._get_context()
        current_span = context.get_current_span()
        
        if current_span:
            event = {
                'name': name,
                'timestamp': time.time(),
                'attributes': attributes or {}
            }
            current_span.events.append(event)
            
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        context = self._get_context()
        current_span = context.get_current_span()
        
        if current_span:
            current_span.attributes[key] = value
            
    def set_tag(self, key: str, value: str):
        """Set tag on current span"""
        context = self._get_context()
        current_span = context.get_current_span()
        
        if current_span:
            current_span.tags[key] = value
            
    def log(self, message: str):
        """Add log to current span"""
        context = self._get_context()
        current_span = context.get_current_span()
        
        if current_span:
            log_entry = f"{time.time()}: {message}"
            current_span.logs.append(log_entry)
            
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        context = self._get_context()
        current_span = context.get_current_span()
        return current_span.trace_id if current_span else None
        
    def get_span_id(self) -> Optional[str]:
        """Get current span ID"""
        context = self._get_context()
        current_span = context.get_current_span()
        return current_span.span_id if current_span else None
        
    def inject_headers(self, headers: Dict[str, str]):
        """Inject trace context into headers"""
        context = self._get_context()
        context.inject_headers(headers)
        
    def extract_headers(self, headers: Dict[str, str]) -> Optional[SpanInfo]:
        """Extract trace context from headers and create span"""
        context = self._get_context()
        trace_context = context.extract_headers(headers)
        
        if trace_context:
            trace_id, parent_span_id = trace_context
            
            # Create a span with extracted context
            span = SpanInfo(
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name="extracted_span",
                service_name=self.config.service_name
            )
            
            return span
            
        return None
        
    def get_metrics(self) -> TraceMetrics:
        """Get current trace metrics"""
        with self._lock:
            completed_spans_list = list(self.completed_spans)
            
        if not completed_spans_list:
            return TraceMetrics()
            
        # Calculate metrics
        total_spans = len(completed_spans_list)
        error_spans = sum(1 for span in completed_spans_list if span.status == TraceStatus.ERROR)
        timeout_spans = sum(1 for span in completed_spans_list if span.status == TraceStatus.TIMEOUT)
        
        durations = [span.duration for span in completed_spans_list if span.duration is not None]
        
        metrics = TraceMetrics(
            total_spans=total_spans,
            error_spans=error_spans,
            timeout_spans=timeout_spans,
            error_rate=(error_spans / total_spans) * 100 if total_spans > 0 else 0.0,
            timeout_rate=(timeout_spans / total_spans) * 100 if total_spans > 0 else 0.0,
            services_count=len(set(span.service_name for span in completed_spans_list)),
            operations_count=len(set(span.operation_name for span in completed_spans_list))
        )
        
        if durations:
            durations.sort()
            metrics.avg_duration = sum(durations) / len(durations)
            metrics.p50_duration = durations[int(len(durations) * 0.5)]
            metrics.p95_duration = durations[int(len(durations) * 0.95)]
            metrics.p99_duration = durations[int(len(durations) * 0.99)]
            
        # Calculate throughput (last minute)
        current_time = time.time()
        recent_spans = [span for span in completed_spans_list 
                       if current_time - span.start_time <= 60]
                       
        metrics.traces_per_second = len(set(span.trace_id for span in recent_spans)) / 60.0
        metrics.spans_per_second = len(recent_spans) / 60.0
        
        return metrics
        
    async def shutdown(self):
        """Shutdown the tracer"""
        self._shutdown_event.set()
        
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
                
        # Shutdown exporters
        for exporter in self.exporters:
            await exporter.shutdown()


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def initialize_tracing(config: TraceConfig = None) -> DistributedTracer:
    """Initialize global tracing"""
    global _global_tracer
    
    if config is None:
        config = TraceConfig()
        
    _global_tracer = DistributedTracer(config)
    
    # Auto-instrument if enabled
    if OPENTELEMETRY_AVAILABLE:
        if config.auto_instrument_requests:
            RequestsInstrumentor().instrument()
            
        if config.auto_instrument_async_http:
            AioHttpClientInstrumentor().instrument()
            
        if config.auto_instrument_database:
            SQLAlchemyInstrumentor().instrument()
            
    return _global_tracer


def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer"""
    return _global_tracer


# Convenience functions
def trace(operation_name: str = None, **kwargs):
    """Decorator for tracing functions"""
    tracer = get_tracer()
    if tracer:
        return tracer.trace_function(operation_name, **kwargs)
    else:
        # No-op decorator
        def decorator(func):
            return func
        return decorator


def start_span(operation_name: str, **kwargs) -> Optional[SpanInfo]:
    """Start a span"""
    tracer = get_tracer()
    if tracer:
        return tracer.start_span(operation_name, **kwargs)
    return None


def finish_span(span: SpanInfo, **kwargs):
    """Finish a span"""
    tracer = get_tracer()
    if tracer:
        tracer.finish_span(span, **kwargs)


def add_event(name: str, attributes: Dict[str, Any] = None):
    """Add event to current span"""
    tracer = get_tracer()
    if tracer:
        tracer.add_event(name, attributes)


def set_attribute(key: str, value: Any):
    """Set attribute on current span"""
    tracer = get_tracer()
    if tracer:
        tracer.set_attribute(key, value)


def set_tag(key: str, value: str):
    """Set tag on current span"""
    tracer = get_tracer()
    if tracer:
        tracer.set_tag(key, value)


def log(message: str):
    """Add log to current span"""
    tracer = get_tracer()
    if tracer:
        tracer.log(message)


# Utility functions
def create_jaeger_config(endpoint: str = "http://localhost:14268/api/traces",
                        service_name: str = "computer-genie") -> TraceConfig:
    """Create configuration for Jaeger"""
    return TraceConfig(
        service_name=service_name,
        jaeger_endpoint=endpoint,
        console_export=False,
        sampling_strategy=SamplingStrategy.PROBABILISTIC,
        sampling_rate=0.1
    )


def create_development_config(service_name: str = "computer-genie") -> TraceConfig:
    """Create configuration for development"""
    return TraceConfig(
        service_name=service_name,
        environment="development",
        console_export=True,
        sampling_strategy=SamplingStrategy.ALWAYS_ON
    )


def create_production_config(service_name: str = "computer-genie",
                           jaeger_endpoint: str = None) -> TraceConfig:
    """Create configuration for production"""
    return TraceConfig(
        service_name=service_name,
        environment="production",
        console_export=False,
        jaeger_endpoint=jaeger_endpoint,
        sampling_strategy=SamplingStrategy.PROBABILISTIC,
        sampling_rate=0.01,  # 1% sampling in production
        max_traces_per_second=1000
    )