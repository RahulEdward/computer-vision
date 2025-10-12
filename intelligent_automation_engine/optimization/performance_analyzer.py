"""
Performance Analyzer for Automation Workflow Monitoring

This module provides comprehensive performance analysis, monitoring, and
profiling capabilities for automation workflows and patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
import uuid
import logging
import statistics
from collections import defaultdict, deque
import threading
import time

from .loop_detector import ActionStep, Pattern, Loop

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    RESOURCE_UTILIZATION = "resource_utilization"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_LENGTH = "queue_length"


class AnalysisType(Enum):
    """Types of performance analysis"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    BOTTLENECK_ANALYSIS = "bottleneck_analysis"


class AlertSeverity(Enum):
    """Severity levels for performance alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metric identification
    name: str = ""
    type: MetricType = MetricType.EXECUTION_TIME
    category: str = ""
    
    # Measurement data
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context information
    source_id: str = ""  # ID of the action, pattern, or loop
    source_type: str = ""  # action, pattern, loop
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality indicators
    confidence: float = 1.0
    accuracy: float = 1.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PerformanceProfile:
    """Performance profile for an automation component"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Profile identification
    component_id: str = ""
    component_type: str = ""  # action, pattern, loop, workflow
    name: str = ""
    
    # Performance statistics
    metrics: List[PerformanceMetric] = field(default_factory=list)
    
    # Statistical summaries
    average_execution_time: timedelta = timedelta(0)
    min_execution_time: timedelta = timedelta(0)
    max_execution_time: timedelta = timedelta(0)
    median_execution_time: timedelta = timedelta(0)
    
    # Resource usage
    average_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    average_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    
    # Reliability metrics
    success_rate: float = 1.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Performance trends
    performance_trend: str = "stable"  # improving, degrading, stable, volatile
    trend_confidence: float = 0.0
    
    # Benchmarking
    baseline_performance: Optional['PerformanceProfile'] = None
    performance_delta: float = 0.0
    
    # Analysis metadata
    sample_count: int = 0
    analysis_period: timedelta = timedelta(0)
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations or anomalies"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Alert identification
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    
    # Alert trigger
    metric_type: MetricType = MetricType.EXECUTION_TIME
    threshold_value: float = 0.0
    actual_value: float = 0.0
    component_id: str = ""
    
    # Alert timing
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    duration: timedelta = timedelta(0)
    
    # Alert context
    context: Dict[str, Any] = field(default_factory=dict)
    related_metrics: List[str] = field(default_factory=list)  # Metric IDs
    
    # Alert management
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: Optional[datetime] = None
    
    # Resolution
    resolved: bool = False
    resolution_notes: str = ""
    auto_resolved: bool = False
    
    # Metadata
    tags: List[str] = field(default_factory=list)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Threshold identification
    name: str = ""
    description: str = ""
    metric_type: MetricType = MetricType.EXECUTION_TIME
    
    # Threshold values
    warning_threshold: float = 0.0
    error_threshold: float = 0.0
    critical_threshold: float = 0.0
    
    # Threshold conditions
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window: timedelta = timedelta(minutes=5)
    min_samples: int = 3
    
    # Scope
    applies_to: List[str] = field(default_factory=list)  # Component IDs or patterns
    component_types: List[str] = field(default_factory=list)
    
    # Configuration
    enabled: bool = True
    auto_resolve: bool = True
    notification_enabled: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Report identification
    title: str = ""
    description: str = ""
    analysis_type: AnalysisType = AnalysisType.BATCH
    
    # Analysis scope
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    components_analyzed: List[str] = field(default_factory=list)
    
    # Performance profiles
    profiles: List[PerformanceProfile] = field(default_factory=list)
    
    # Key findings
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)
    performance_improvements: List[str] = field(default_factory=list)
    
    # Alerts and issues
    alerts_generated: List[str] = field(default_factory=list)  # Alert IDs
    issues_identified: List[str] = field(default_factory=list)
    
    # Report metadata
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = ""
    report_format: str = "json"  # json, html, pdf
    
    # Quality indicators
    data_completeness: float = 1.0
    analysis_confidence: float = 1.0


@dataclass
class AnalysisContext:
    """Context for performance analysis operations"""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Analysis configuration
    analysis_type: AnalysisType = AnalysisType.BATCH
    time_window: timedelta = timedelta(hours=1)
    sample_rate: float = 1.0  # 1.0 = 100% sampling
    
    # Metric selection
    metric_types: List[MetricType] = field(default_factory=list)
    include_all_metrics: bool = True
    
    # Analysis options
    enable_trend_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_bottleneck_analysis: bool = True
    enable_comparative_analysis: bool = False
    
    # Filtering options
    component_filters: List[str] = field(default_factory=list)
    tag_filters: List[str] = field(default_factory=list)
    min_sample_count: int = 10
    
    # Output options
    generate_report: bool = True
    include_recommendations: bool = True
    alert_on_issues: bool = True
    
    # Performance settings
    max_analysis_time: timedelta = timedelta(minutes=10)
    parallel_processing: bool = True
    cache_results: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for automation workflows"""
    
    def __init__(self):
        # Metric storage
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.profiles: Dict[str, PerformanceProfile] = {}
        
        # Alerting system
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alert_handlers: List[Callable] = []
        
        # Analysis history
        self.analysis_history: List[PerformanceReport] = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Performance statistics
        self.analyzer_stats = {
            'total_metrics_collected': 0,
            'total_analyses_performed': 0,
            'total_alerts_generated': 0,
            'average_analysis_time': timedelta(0)
        }
        
        # Configuration
        self.default_context = AnalysisContext()
        self.auto_profile_updates = True
        self.metric_retention_period = timedelta(days=30)
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        logger.info("Performance analyzer initialized")
    
    def collect_metric(self, metric: PerformanceMetric):
        """Collect a performance metric"""
        try:
            # Store metric
            self.metrics[metric.source_id].append(metric)
            
            # Update statistics
            self.analyzer_stats['total_metrics_collected'] += 1
            
            # Update profile if auto-update is enabled
            if self.auto_profile_updates:
                self._update_profile(metric)
            
            # Check thresholds for alerts
            self._check_thresholds(metric)
            
            # Clean old metrics
            self._cleanup_old_metrics()
            
            logger.debug(f"Collected metric: {metric.name} = {metric.value} {metric.unit}")
            
        except Exception as e:
            logger.error(f"Failed to collect metric: {e}")
    
    def collect_action_metrics(self, action: ActionStep):
        """Collect metrics from an action step"""
        try:
            # Execution time metric
            exec_metric = PerformanceMetric(
                name="execution_time",
                type=MetricType.EXECUTION_TIME,
                value=action.duration.total_seconds(),
                unit="seconds",
                source_id=action.id,
                source_type="action",
                context=action.context.copy()
            )
            self.collect_metric(exec_metric)
            
            # Success rate metric
            success_metric = PerformanceMetric(
                name="success_rate",
                type=MetricType.SUCCESS_RATE,
                value=1.0 if action.success else 0.0,
                unit="ratio",
                source_id=action.id,
                source_type="action",
                context=action.context.copy()
            )
            self.collect_metric(success_metric)
            
            # Retry count metric
            if action.retry_count > 0:
                retry_metric = PerformanceMetric(
                    name="retry_count",
                    type=MetricType.ERROR_RATE,
                    value=action.retry_count,
                    unit="count",
                    source_id=action.id,
                    source_type="action",
                    context=action.context.copy()
                )
                self.collect_metric(retry_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect action metrics: {e}")
    
    def collect_pattern_metrics(self, pattern: Pattern):
        """Collect metrics from a pattern"""
        try:
            # Pattern frequency metric
            freq_metric = PerformanceMetric(
                name="pattern_frequency",
                type=MetricType.THROUGHPUT,
                value=pattern.frequency,
                unit="occurrences",
                source_id=pattern.id,
                source_type="pattern",
                context={'pattern_type': pattern.type.value}
            )
            self.collect_metric(freq_metric)
            
            # Average duration metric
            duration_metric = PerformanceMetric(
                name="average_duration",
                type=MetricType.EXECUTION_TIME,
                value=pattern.average_duration.total_seconds(),
                unit="seconds",
                source_id=pattern.id,
                source_type="pattern",
                context={'pattern_type': pattern.type.value}
            )
            self.collect_metric(duration_metric)
            
            # Success rate metric
            success_metric = PerformanceMetric(
                name="success_rate",
                type=MetricType.SUCCESS_RATE,
                value=pattern.success_rate,
                unit="ratio",
                source_id=pattern.id,
                source_type="pattern",
                context={'pattern_type': pattern.type.value}
            )
            self.collect_metric(success_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect pattern metrics: {e}")
    
    def collect_loop_metrics(self, loop: Loop):
        """Collect metrics from a loop"""
        try:
            # Loop iteration count metric
            iter_metric = PerformanceMetric(
                name="iteration_count",
                type=MetricType.THROUGHPUT,
                value=loop.total_iterations,
                unit="iterations",
                source_id=loop.id,
                source_type="loop",
                context={'loop_type': loop.type.value}
            )
            self.collect_metric(iter_metric)
            
            # Average iteration time metric
            time_metric = PerformanceMetric(
                name="average_iteration_time",
                type=MetricType.EXECUTION_TIME,
                value=loop.average_iteration_time.total_seconds(),
                unit="seconds",
                source_id=loop.id,
                source_type="loop",
                context={'loop_type': loop.type.value}
            )
            self.collect_metric(time_metric)
            
            # Efficiency score metric
            efficiency_metric = PerformanceMetric(
                name="efficiency_score",
                type=MetricType.RESOURCE_UTILIZATION,
                value=loop.efficiency_score,
                unit="score",
                source_id=loop.id,
                source_type="loop",
                context={'loop_type': loop.type.value}
            )
            self.collect_metric(efficiency_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect loop metrics: {e}")
    
    def analyze_performance(self, context: Optional[AnalysisContext] = None) -> PerformanceReport:
        """Perform comprehensive performance analysis"""
        try:
            start_time = datetime.now()
            context = context or self.default_context
            
            # Initialize report
            report = PerformanceReport(
                title=f"Performance Analysis {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                description="Automated performance analysis",
                analysis_type=context.analysis_type,
                start_time=start_time - context.time_window,
                end_time=start_time
            )
            
            # Get metrics for analysis
            analysis_metrics = self._get_metrics_for_analysis(context)
            
            # Generate performance profiles
            profiles = self._generate_performance_profiles(analysis_metrics, context)
            report.profiles = profiles
            
            # Perform different types of analysis
            if context.enable_trend_analysis:
                trends = self._analyze_trends(analysis_metrics, context)
                report.trends = trends
            
            if context.enable_anomaly_detection:
                anomalies = self._detect_anomalies(analysis_metrics, context)
                report.anomalies = anomalies
            
            if context.enable_bottleneck_analysis:
                bottlenecks = self._analyze_bottlenecks(analysis_metrics, context)
                report.bottlenecks = bottlenecks
            
            # Generate performance summary
            report.performance_summary = self._generate_performance_summary(profiles)
            
            # Generate recommendations
            if context.include_recommendations:
                report.optimization_recommendations = self._generate_optimization_recommendations(profiles)
                report.performance_improvements = self._generate_performance_improvements(profiles)
            
            # Calculate quality indicators
            report.data_completeness = self._calculate_data_completeness(analysis_metrics)
            report.analysis_confidence = self._calculate_analysis_confidence(profiles)
            
            # Update statistics
            analysis_time = datetime.now() - start_time
            self._update_analysis_statistics(analysis_time)
            
            # Store report
            self.analysis_history.append(report)
            
            logger.info(f"Performance analysis completed in {analysis_time}")
            return report
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return PerformanceReport()
    
    def start_real_time_monitoring(self, interval: float = 5.0):
        """Start real-time performance monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("Real-time monitoring already active")
                return
            
            self.monitoring_interval = interval
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"Started real-time monitoring with {interval}s interval")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def stop_real_time_monitoring(self):
        """Stop real-time performance monitoring"""
        try:
            if not self.monitoring_active:
                logger.warning("Real-time monitoring not active")
                return
            
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10.0)
            
            logger.info("Stopped real-time monitoring")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    def add_performance_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold for alerting"""
        try:
            self.thresholds[threshold.id] = threshold
            logger.info(f"Added performance threshold: {threshold.name}")
            
        except Exception as e:
            logger.error(f"Failed to add threshold: {e}")
    
    def remove_performance_threshold(self, threshold_id: str) -> bool:
        """Remove a performance threshold"""
        try:
            if threshold_id in self.thresholds:
                del self.thresholds[threshold_id]
                logger.info(f"Removed performance threshold: {threshold_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove threshold: {e}")
            return False
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add an alert handler function"""
        try:
            self.alert_handlers.append(handler)
            logger.info("Added alert handler")
            
        except Exception as e:
            logger.error(f"Failed to add alert handler: {e}")
    
    def get_performance_profile(self, component_id: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a component"""
        return self.profiles.get(component_id)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "") -> bool:
        """Acknowledge a performance alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Acknowledged alert: {alert_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve a performance alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                alert.duration = alert.resolved_at - alert.triggered_at
                logger.info(f"Resolved alert: {alert_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    # Internal methods
    def _update_profile(self, metric: PerformanceMetric):
        """Update performance profile with new metric"""
        try:
            component_id = metric.source_id
            
            # Get or create profile
            if component_id not in self.profiles:
                self.profiles[component_id] = PerformanceProfile(
                    component_id=component_id,
                    component_type=metric.source_type,
                    name=f"{metric.source_type}_{component_id[:8]}"
                )
            
            profile = self.profiles[component_id]
            profile.metrics.append(metric)
            profile.sample_count += 1
            profile.last_updated = datetime.now()
            
            # Update statistical summaries
            self._update_profile_statistics(profile)
            
        except Exception as e:
            logger.error(f"Failed to update profile: {e}")
    
    def _update_profile_statistics(self, profile: PerformanceProfile):
        """Update statistical summaries for a profile"""
        try:
            # Get execution time metrics
            exec_times = [
                m.value for m in profile.metrics 
                if m.type == MetricType.EXECUTION_TIME
            ]
            
            if exec_times:
                profile.average_execution_time = timedelta(seconds=statistics.mean(exec_times))
                profile.min_execution_time = timedelta(seconds=min(exec_times))
                profile.max_execution_time = timedelta(seconds=max(exec_times))
                profile.median_execution_time = timedelta(seconds=statistics.median(exec_times))
            
            # Get success rate metrics
            success_rates = [
                m.value for m in profile.metrics 
                if m.type == MetricType.SUCCESS_RATE
            ]
            
            if success_rates:
                profile.success_rate = statistics.mean(success_rates)
                profile.error_rate = 1.0 - profile.success_rate
            
            # Get memory usage metrics
            memory_usage = [
                m.value for m in profile.metrics 
                if m.type == MetricType.MEMORY_USAGE
            ]
            
            if memory_usage:
                profile.average_memory_usage = statistics.mean(memory_usage)
                profile.peak_memory_usage = max(memory_usage)
            
            # Get CPU usage metrics
            cpu_usage = [
                m.value for m in profile.metrics 
                if m.type == MetricType.CPU_USAGE
            ]
            
            if cpu_usage:
                profile.average_cpu_usage = statistics.mean(cpu_usage)
                profile.peak_cpu_usage = max(cpu_usage)
            
            # Analyze performance trend
            profile.performance_trend = self._analyze_performance_trend(profile)
            
        except Exception as e:
            logger.error(f"Failed to update profile statistics: {e}")
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check metric against configured thresholds"""
        try:
            for threshold in self.thresholds.values():
                if not threshold.enabled:
                    continue
                
                # Check if threshold applies to this metric
                if threshold.metric_type != metric.type:
                    continue
                
                if threshold.applies_to and metric.source_id not in threshold.applies_to:
                    continue
                
                if threshold.component_types and metric.source_type not in threshold.component_types:
                    continue
                
                # Evaluate threshold
                violation = self._evaluate_threshold(metric, threshold)
                
                if violation:
                    self._generate_alert(metric, threshold, violation)
            
        except Exception as e:
            logger.error(f"Failed to check thresholds: {e}")
    
    def _evaluate_threshold(self, metric: PerformanceMetric, 
                           threshold: PerformanceThreshold) -> Optional[str]:
        """Evaluate if metric violates threshold"""
        try:
            value = metric.value
            operator = threshold.comparison_operator
            
            # Check critical threshold
            if threshold.critical_threshold > 0:
                if self._compare_values(value, threshold.critical_threshold, operator):
                    return "critical"
            
            # Check error threshold
            if threshold.error_threshold > 0:
                if self._compare_values(value, threshold.error_threshold, operator):
                    return "error"
            
            # Check warning threshold
            if threshold.warning_threshold > 0:
                if self._compare_values(value, threshold.warning_threshold, operator):
                    return "warning"
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate threshold: {e}")
            return None
    
    def _compare_values(self, value: float, threshold: float, operator: str) -> bool:
        """Compare values using specified operator"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            return False
    
    def _generate_alert(self, metric: PerformanceMetric, 
                       threshold: PerformanceThreshold, 
                       severity: str):
        """Generate performance alert"""
        try:
            alert = PerformanceAlert(
                title=f"{threshold.name} Threshold Violation",
                description=f"Metric {metric.name} exceeded {severity} threshold",
                severity=AlertSeverity(severity),
                metric_type=metric.type,
                threshold_value=getattr(threshold, f"{severity}_threshold"),
                actual_value=metric.value,
                component_id=metric.source_id,
                context=metric.context.copy(),
                related_metrics=[metric.id]
            )
            
            # Store alert
            self.alerts[alert.id] = alert
            
            # Update statistics
            self.analyzer_stats['total_alerts_generated'] += 1
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            logger.warning(f"Generated {severity} alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
    
    def _monitoring_worker(self):
        """Worker thread for real-time monitoring"""
        try:
            while self.monitoring_active:
                # Perform real-time analysis
                context = AnalysisContext(
                    analysis_type=AnalysisType.REAL_TIME,
                    time_window=timedelta(seconds=self.monitoring_interval * 2),
                    generate_report=False
                )
                
                self.analyze_performance(context)
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"Monitoring worker failed: {e}")
        finally:
            self.monitoring_active = False
    
    def _get_metrics_for_analysis(self, context: AnalysisContext) -> List[PerformanceMetric]:
        """Get metrics for analysis based on context"""
        analysis_metrics = []
        
        try:
            cutoff_time = datetime.now() - context.time_window
            
            for component_id, metrics in self.metrics.items():
                # Apply component filters
                if context.component_filters and component_id not in context.component_filters:
                    continue
                
                # Filter by time window
                recent_metrics = [
                    m for m in metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                # Apply metric type filters
                if not context.include_all_metrics and context.metric_types:
                    recent_metrics = [
                        m for m in recent_metrics 
                        if m.type in context.metric_types
                    ]
                
                # Apply tag filters
                if context.tag_filters:
                    recent_metrics = [
                        m for m in recent_metrics 
                        if any(tag in m.tags for tag in context.tag_filters)
                    ]
                
                # Check minimum sample count
                if len(recent_metrics) >= context.min_sample_count:
                    analysis_metrics.extend(recent_metrics)
            
        except Exception as e:
            logger.error(f"Failed to get metrics for analysis: {e}")
        
        return analysis_metrics
    
    def _generate_performance_profiles(self, metrics: List[PerformanceMetric], 
                                     context: AnalysisContext) -> List[PerformanceProfile]:
        """Generate performance profiles from metrics"""
        profiles = []
        
        try:
            # Group metrics by component
            component_metrics = defaultdict(list)
            for metric in metrics:
                component_metrics[metric.source_id].append(metric)
            
            # Generate profile for each component
            for component_id, comp_metrics in component_metrics.items():
                if len(comp_metrics) < context.min_sample_count:
                    continue
                
                profile = PerformanceProfile(
                    component_id=component_id,
                    component_type=comp_metrics[0].source_type,
                    name=f"{comp_metrics[0].source_type}_{component_id[:8]}",
                    metrics=comp_metrics,
                    sample_count=len(comp_metrics),
                    analysis_period=context.time_window
                )
                
                # Update profile statistics
                self._update_profile_statistics(profile)
                
                profiles.append(profile)
            
        except Exception as e:
            logger.error(f"Failed to generate performance profiles: {e}")
        
        return profiles
    
    def _analyze_trends(self, metrics: List[PerformanceMetric], 
                       context: AnalysisContext) -> List[Dict[str, Any]]:
        """Analyze performance trends"""
        trends = []
        
        try:
            # Group metrics by type and component
            metric_groups = defaultdict(lambda: defaultdict(list))
            for metric in metrics:
                metric_groups[metric.type][metric.source_id].append(metric)
            
            # Analyze trends for each group
            for metric_type, components in metric_groups.items():
                for component_id, comp_metrics in components.items():
                    if len(comp_metrics) < 5:  # Need minimum data points
                        continue
                    
                    # Sort by timestamp
                    comp_metrics.sort(key=lambda m: m.timestamp)
                    
                    # Calculate trend
                    values = [m.value for m in comp_metrics]
                    trend_direction = self._calculate_trend_direction(values)
                    trend_strength = self._calculate_trend_strength(values)
                    
                    if abs(trend_strength) > 0.1:  # Significant trend
                        trends.append({
                            'component_id': component_id,
                            'metric_type': metric_type.value,
                            'trend_direction': trend_direction,
                            'trend_strength': trend_strength,
                            'data_points': len(values),
                            'time_span': comp_metrics[-1].timestamp - comp_metrics[0].timestamp
                        })
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
        
        return trends
    
    def _detect_anomalies(self, metrics: List[PerformanceMetric], 
                         context: AnalysisContext) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []
        
        try:
            # Group metrics by type and component
            metric_groups = defaultdict(lambda: defaultdict(list))
            for metric in metrics:
                metric_groups[metric.type][metric.source_id].append(metric)
            
            # Detect anomalies for each group
            for metric_type, components in metric_groups.items():
                for component_id, comp_metrics in components.items():
                    if len(comp_metrics) < 10:  # Need sufficient data
                        continue
                    
                    values = [m.value for m in comp_metrics]
                    anomaly_indices = self._detect_statistical_anomalies(values)
                    
                    for idx in anomaly_indices:
                        anomalies.append({
                            'component_id': component_id,
                            'metric_type': metric_type.value,
                            'anomaly_value': values[idx],
                            'timestamp': comp_metrics[idx].timestamp,
                            'severity': self._calculate_anomaly_severity(values, idx),
                            'context': comp_metrics[idx].context
                        })
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
        
        return anomalies
    
    def _analyze_bottlenecks(self, metrics: List[PerformanceMetric], 
                           context: AnalysisContext) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        try:
            # Group execution time metrics by component
            exec_metrics = defaultdict(list)
            for metric in metrics:
                if metric.type == MetricType.EXECUTION_TIME:
                    exec_metrics[metric.source_id].append(metric)
            
            # Find components with high execution times
            for component_id, comp_metrics in exec_metrics.items():
                if len(comp_metrics) < 5:
                    continue
                
                values = [m.value for m in comp_metrics]
                avg_time = statistics.mean(values)
                max_time = max(values)
                
                # Consider as bottleneck if average time is high
                if avg_time > 5.0:  # 5 seconds threshold
                    bottlenecks.append({
                        'component_id': component_id,
                        'component_type': comp_metrics[0].source_type,
                        'average_time': avg_time,
                        'max_time': max_time,
                        'sample_count': len(values),
                        'severity': 'high' if avg_time > 10.0 else 'medium'
                    })
            
        except Exception as e:
            logger.error(f"Failed to analyze bottlenecks: {e}")
        
        return bottlenecks
    
    def _generate_performance_summary(self, profiles: List[PerformanceProfile]) -> Dict[str, Any]:
        """Generate performance summary from profiles"""
        summary = {
            'total_components': len(profiles),
            'average_execution_time': timedelta(0),
            'overall_success_rate': 0.0,
            'total_samples': 0,
            'performance_distribution': {}
        }
        
        try:
            if not profiles:
                return summary
            
            # Calculate averages
            total_exec_time = sum((p.average_execution_time for p in profiles), timedelta(0))
            summary['average_execution_time'] = total_exec_time / len(profiles)
            
            total_success_rate = sum(p.success_rate for p in profiles)
            summary['overall_success_rate'] = total_success_rate / len(profiles)
            
            summary['total_samples'] = sum(p.sample_count for p in profiles)
            
            # Performance distribution
            fast_components = len([p for p in profiles if p.average_execution_time < timedelta(seconds=1)])
            medium_components = len([p for p in profiles if timedelta(seconds=1) <= p.average_execution_time < timedelta(seconds=5)])
            slow_components = len([p for p in profiles if p.average_execution_time >= timedelta(seconds=5)])
            
            summary['performance_distribution'] = {
                'fast': fast_components,
                'medium': medium_components,
                'slow': slow_components
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
        
        return summary
    
    def _generate_optimization_recommendations(self, profiles: List[PerformanceProfile]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            # Find slow components
            slow_components = [p for p in profiles if p.average_execution_time > timedelta(seconds=5)]
            if slow_components:
                recommendations.append(f"Optimize {len(slow_components)} slow-performing components")
            
            # Find components with low success rates
            unreliable_components = [p for p in profiles if p.success_rate < 0.9]
            if unreliable_components:
                recommendations.append(f"Improve reliability of {len(unreliable_components)} components")
            
            # Find components with high resource usage
            resource_intensive = [p for p in profiles if p.peak_memory_usage > 1000.0]
            if resource_intensive:
                recommendations.append(f"Optimize memory usage for {len(resource_intensive)} components")
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
        
        return recommendations
    
    def _generate_performance_improvements(self, profiles: List[PerformanceProfile]) -> List[str]:
        """Generate performance improvement suggestions"""
        improvements = []
        
        try:
            # Suggest caching for frequently used components
            frequent_components = [p for p in profiles if p.sample_count > 100]
            if frequent_components:
                improvements.append("Consider implementing caching for frequently used components")
            
            # Suggest parallelization for independent operations
            improvements.append("Consider parallelizing independent operations")
            
            # Suggest batch processing for similar operations
            improvements.append("Consider batch processing for similar operations")
            
        except Exception as e:
            logger.error(f"Failed to generate performance improvements: {e}")
        
        return improvements
    
    # Helper methods
    def _analyze_performance_trend(self, profile: PerformanceProfile) -> str:
        """Analyze performance trend for a profile"""
        try:
            if len(profile.metrics) < 5:
                return "insufficient_data"
            
            # Get recent execution times
            exec_times = [
                m.value for m in profile.metrics[-20:]  # Last 20 measurements
                if m.type == MetricType.EXECUTION_TIME
            ]
            
            if len(exec_times) < 5:
                return "insufficient_data"
            
            # Calculate trend
            trend_strength = self._calculate_trend_strength(exec_times)
            
            if trend_strength > 0.1:
                return "degrading"
            elif trend_strength < -0.1:
                return "improving"
            elif abs(trend_strength) < 0.05:
                return "stable"
            else:
                return "volatile"
                
        except Exception as e:
            logger.error(f"Failed to analyze performance trend: {e}")
            return "unknown"
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "unknown"
        
        # Simple linear trend calculation
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simplified trend strength calculation
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        if first_half == 0:
            return 0.0
        
        return (second_half - first_half) / first_half
    
    def _detect_statistical_anomalies(self, values: List[float]) -> List[int]:
        """Detect statistical anomalies using simple outlier detection"""
        if len(values) < 10:
            return []
        
        # Calculate statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        # Find outliers (values beyond 2 standard deviations)
        anomalies = []
        for i, value in enumerate(values):
            if abs(value - mean_val) > 2 * std_val:
                anomalies.append(i)
        
        return anomalies
    
    def _calculate_anomaly_severity(self, values: List[float], anomaly_index: int) -> str:
        """Calculate severity of an anomaly"""
        if len(values) < 10:
            return "low"
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        anomaly_value = values[anomaly_index]
        
        deviation = abs(anomaly_value - mean_val) / std_val
        
        if deviation > 3:
            return "critical"
        elif deviation > 2.5:
            return "high"
        elif deviation > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_data_completeness(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate data completeness score"""
        if not metrics:
            return 0.0
        
        # Simple completeness calculation based on metric availability
        metric_types = set(m.type for m in metrics)
        expected_types = {MetricType.EXECUTION_TIME, MetricType.SUCCESS_RATE}
        
        return len(metric_types.intersection(expected_types)) / len(expected_types)
    
    def _calculate_analysis_confidence(self, profiles: List[PerformanceProfile]) -> float:
        """Calculate analysis confidence score"""
        if not profiles:
            return 0.0
        
        # Base confidence on sample sizes
        total_samples = sum(p.sample_count for p in profiles)
        avg_samples = total_samples / len(profiles)
        
        # Higher sample count = higher confidence
        confidence = min(avg_samples / 100.0, 1.0)
        return confidence
    
    def _update_analysis_statistics(self, analysis_time: timedelta):
        """Update analysis performance statistics"""
        self.analyzer_stats['total_analyses_performed'] += 1
        
        # Update average analysis time
        current_avg = self.analyzer_stats['average_analysis_time']
        new_avg = (current_avg * (self.analyzer_stats['total_analyses_performed'] - 1) + 
                  analysis_time) / self.analyzer_stats['total_analyses_performed']
        self.analyzer_stats['average_analysis_time'] = new_avg
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        try:
            cutoff_time = datetime.now() - self.metric_retention_period
            
            for component_id in list(self.metrics.keys()):
                # Filter out old metrics
                self.metrics[component_id] = [
                    m for m in self.metrics[component_id] 
                    if m.timestamp >= cutoff_time
                ]
                
                # Remove empty entries
                if not self.metrics[component_id]:
                    del self.metrics[component_id]
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds"""
        # Execution time threshold
        exec_threshold = PerformanceThreshold(
            name="Execution Time Threshold",
            description="Alert when execution time is too high",
            metric_type=MetricType.EXECUTION_TIME,
            warning_threshold=5.0,  # 5 seconds
            error_threshold=10.0,   # 10 seconds
            critical_threshold=30.0, # 30 seconds
            comparison_operator=">",
            component_types=["action", "pattern", "loop"]
        )
        self.add_performance_threshold(exec_threshold)
        
        # Success rate threshold
        success_threshold = PerformanceThreshold(
            name="Success Rate Threshold",
            description="Alert when success rate is too low",
            metric_type=MetricType.SUCCESS_RATE,
            warning_threshold=0.9,  # 90%
            error_threshold=0.8,    # 80%
            critical_threshold=0.7, # 70%
            comparison_operator="<",
            component_types=["action", "pattern", "loop"]
        )
        self.add_performance_threshold(success_threshold)
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get analyzer performance statistics"""
        return {
            'total_metrics_collected': self.analyzer_stats['total_metrics_collected'],
            'total_analyses_performed': self.analyzer_stats['total_analyses_performed'],
            'total_alerts_generated': self.analyzer_stats['total_alerts_generated'],
            'average_analysis_time': self.analyzer_stats['average_analysis_time'],
            'active_profiles': len(self.profiles),
            'active_alerts': len(self.get_active_alerts()),
            'configured_thresholds': len(self.thresholds),
            'monitoring_active': self.monitoring_active,
            'history_size': len(self.analysis_history)
        }