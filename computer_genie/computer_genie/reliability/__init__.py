"""
Computer Genie Fault-Tolerant Reliability System

A comprehensive fault-tolerant reliability system designed to guarantee 99.99% uptime
with automatic disaster recovery and sub-1 second recovery time objectives (RTO).

This module provides 10 integrated components for comprehensive fault tolerance:
1. Circuit Breaker Pattern - Adaptive thresholds based on system load
2. Chaos Engineering - Automated resilience testing
3. Self-Healing Mechanisms - Auto-recovery from crashes
4. Byzantine Fault Tolerance - Distributed consensus
5. Event Sourcing - Complete audit trail and time-travel debugging
6. Shadow Testing - Safe production testing
7. Canary Deployment - Automatic rollback
8. Distributed Tracing - OpenTelemetry integration
9. Predictive Failure Detection - ML-based anomaly detection
10. Multi-Region Failover - Sub-1 second RTO
"""

# Circuit Breaker Pattern
from .circuit_breaker import (
    CircuitBreakerState,
    CircuitBreakerConfig,
    SystemMetrics,
    CallResult,
    SystemLoadMonitor,
    AdaptiveCircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    circuit_breaker,
)

# Chaos Engineering
from .chaos_engineering import (
    FailureType,
    ExperimentStatus,
    FailureConfig,
    ExperimentResult,
    ResilienceMetrics,
    FailureInjector,
    NetworkFailureInjector,
    ResourceFailureInjector,
    ApplicationFailureInjector,
    ChaosExperiment,
    ResilienceValidator,
    ChaosRunner,
    create_network_failure_experiment,
    create_resource_failure_experiment,
    create_application_failure_experiment,
)

# Self-Healing Mechanisms
from .self_healing import (
    HealthStatus,
    RecoveryAction,
    HealthCheck,
    HealthResult,
    RecoveryStrategy,
    RecoveryAttempt,
    HealthChecker,
    AutoRecoveryEngine,
    SelfHealingManager,
    create_cpu_health_check,
    create_memory_health_check,
    create_disk_health_check,
    create_service_restart_strategy,
    create_process_recovery_strategy,
    create_resource_cleanup_strategy,
)

# Byzantine Fault Tolerance
from .byzantine_fault_tolerance import (
    NodeState,
    MessageType,
    ConsensusPhase,
    ConsensusMessage,
    Proposal,
    NodeInfo,
    ByzantineDetector,
    MessageSigner,
    PBFTConsensus,
    ByzantineFaultToleranceManager,
    create_test_nodes,
    create_production_nodes,
)

# Event Sourcing
from .event_sourcing import (
    EventType,
    EventSeverity,
    Event,
    Snapshot,
    EventStore,
    SQLiteEventStore,
    EventProjection,
    AggregateProjection,
    SystemMetricsProjection,
    EventSourcingManager,
)

# Shadow Testing
from .shadow_testing import (
    ShadowTestStatus,
    ComparisonResult,
    TrafficSplittingStrategy,
    ShadowTestConfig,
    ShadowRequest,
    ShadowResponse,
    ShadowTestResult,
    ResponseComparator,
    DefaultResponseComparator,
    ShadowExecutor,
    HTTPShadowExecutor,
    FunctionShadowExecutor,
    TrafficSplitter,
    ShadowTestMetrics,
    ShadowTestManager,
    create_http_shadow_test,
    create_function_shadow_test,
)

# Canary Deployment
from .canary_deployment import (
    DeploymentStatus,
    HealthStatus as CanaryHealthStatus,
    RollbackTrigger,
    CanaryConfig,
    DeploymentTarget,
    HealthCheckResult as CanaryHealthCheckResult,
    DeploymentMetrics,
    DeploymentEvent,
    HealthChecker as CanaryHealthChecker,
    MetricsCollector,
    HTTPHealthChecker,
    DefaultMetricsCollector,
    TrafficSplitter as CanaryTrafficSplitter,
    RollbackDecisionEngine,
    CanaryDeploymentManager,
    create_canary_config,
    create_deployment_with_canary,
)

# Distributed Tracing
from .distributed_tracing import (
    SpanKind,
    TraceStatus,
    SamplingStrategy,
    TraceConfig,
    SpanInfo,
    TraceMetrics,
    SpanExporter,
    ConsoleSpanExporter,
    JaegerSpanExporter,
    TraceSampler,
    SpanContext,
    DistributedTracer,
    create_development_trace_config,
    create_production_trace_config,
)

# Predictive Failure Detection
from .predictive_failure import (
    AnomalyType,
    PredictionConfidence,
    ModelType,
    MetricData,
    AnomalyDetectionConfig,
    AnomalyPrediction,
    ModelPerformance,
    FeatureExtractor,
    AnomalyDetectionModel,
    IsolationForestModel,
    OneClassSVMModel,
    StatisticalAnomalyModel,
    AnomalyClassifier,
    PredictiveFailureDetector,
    create_development_config,
    create_production_config,
)

# Multi-Region Failover
from .multi_region_failover import (
    RegionStatus,
    FailoverTrigger,
    ReplicationStrategy,
    LoadBalancingStrategy,
    RegionConfig,
    FailoverConfig,
    HealthCheckResult as FailoverHealthCheckResult,
    FailoverEvent,
    RegionMetrics,
    HealthChecker as FailoverHealthChecker,
    LoadBalancer,
    DataReplicator,
    FailoverDecisionEngine,
    MultiRegionFailoverManager,
    create_region_config,
    create_development_failover_config,
    create_production_failover_config,
)

# Version information
__version__ = "1.0.0"
__author__ = "Computer Genie Team"
__description__ = "Fault-Tolerant Reliability System for 99.99% Uptime"

# Convenience imports for common use cases
__all__ = [
    # Circuit Breaker
    "CircuitBreakerState",
    "CircuitBreakerConfig", 
    "AdaptiveCircuitBreaker",
    "CircuitBreakerManager",
    "circuit_breaker",
    
    # Chaos Engineering
    "ChaosRunner",
    "ChaosExperiment",
    "create_network_failure_experiment",
    "create_resource_failure_experiment",
    "create_application_failure_experiment",
    
    # Self-Healing
    "SelfHealingManager",
    "create_cpu_health_check",
    "create_memory_health_check",
    "create_disk_health_check",
    "create_service_restart_strategy",
    
    # Byzantine Fault Tolerance
    "ByzantineFaultToleranceManager",
    "create_test_nodes",
    "create_production_nodes",
    
    # Event Sourcing
    "EventSourcingManager",
    "Event",
    "EventType",
    "EventSeverity",
    
    # Shadow Testing
    "ShadowTestManager",
    "create_http_shadow_test",
    "create_function_shadow_test",
    
    # Canary Deployment
    "CanaryDeploymentManager",
    "create_canary_config",
    "create_deployment_with_canary",
    
    # Distributed Tracing
    "DistributedTracer",
    "create_development_trace_config",
    "create_production_trace_config",
    
    # Predictive Failure Detection
    "PredictiveFailureDetector",
    "create_development_config",
    "create_production_config",
    
    # Multi-Region Failover
    "MultiRegionFailoverManager",
    "create_region_config",
    "create_development_failover_config",
    "create_production_failover_config",
]

# Quick start configuration
def create_reliability_system():
    """
    Create a pre-configured reliability system with all components.
    
    Returns:
        dict: Dictionary containing all initialized reliability components
    """
    return {
        'circuit_breaker': CircuitBreakerManager(),
        'chaos_runner': ChaosRunner(),
        'self_healing': SelfHealingManager(),
        'byzantine_ft': ByzantineFaultToleranceManager(create_test_nodes(4)),
        'event_sourcing': EventSourcingManager(),
        'shadow_testing': ShadowTestManager(),
        'canary_deployment': CanaryDeploymentManager(create_canary_config("default")),
        'distributed_tracing': DistributedTracer(create_development_trace_config()),
        'predictive_failure': PredictiveFailureDetector(create_development_config()),
        'multi_region_failover': MultiRegionFailoverManager(),
    }

def get_system_info():
    """Get information about the reliability system."""
    return {
        'version': __version__,
        'components': 10,
        'uptime_guarantee': '99.99%',
        'rto_target': '<1 second',
        'rpo_target': '<5 seconds',
        'features': [
            'Adaptive Circuit Breakers',
            'Chaos Engineering',
            'Self-Healing Mechanisms', 
            'Byzantine Fault Tolerance',
            'Event Sourcing',
            'Shadow Testing',
            'Canary Deployments',
            'Distributed Tracing',
            'Predictive Failure Detection',
            'Multi-Region Failover',
        ]
    }