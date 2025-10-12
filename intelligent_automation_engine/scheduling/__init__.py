"""
Time-based Scheduling Package for Intelligent Automation Engine

This package provides comprehensive time-based scheduling capabilities including:
- Cron expression parsing and scheduling
- Natural language time expression parsing
- Schedule optimization and conflict resolution
- Time management and timezone handling
- Event-driven scheduling and automation

Key Components:
- CronScheduler: Advanced cron expression parsing and scheduling
- NaturalLanguageScheduler: Human-friendly time expression parsing
- ScheduleOptimizer: Intelligent schedule management and optimization
- TimeManager: Comprehensive time management and calculations
- EventScheduler: Event-driven automation and reactive scheduling

Features:
- Multiple cron expression formats and extensions
- Natural language parsing (e.g., "every Monday at 9 AM")
- Schedule conflict detection and resolution
- Resource-aware scheduling optimization
- Timezone-aware time calculations
- Business hours and holiday handling
- Event triggers and reactive automation
- Performance monitoring and metrics
"""

from .cron_scheduler import (
    CronScheduler,
    CronFieldType,
    ScheduleStatus,
    ScheduleType,
    CronField,
    CronExpression,
    ScheduleEntry,
    ScheduleResult
)

from .natural_language_scheduler import (
    NaturalLanguageScheduler,
    TimeUnit,
    ScheduleFrequency,
    TimeReference,
    DayOfWeek,
    TimeExpression,
    ParsedSchedule,
    ParsingRule
)

from .schedule_optimizer import (
    ScheduleOptimizer,
    OptimizationObjective,
    ConflictType,
    OptimizationStrategy,
    SchedulePriority,
    ScheduleConflict,
    OptimizationConstraint,
    OptimizationResult,
    ResourceAllocation,
    OptimizationPlan
)

from .time_manager import (
    TimeManager,
    TimeUnit as TimeManagerTimeUnit,
    TimeFormat,
    TimeZoneType,
    BusinessDayRule,
    TimeRange,
    TimeCalculation,
    BusinessHours,
    TimeZoneInfo,
    TimeConversion
)

from .event_scheduler import (
    EventScheduler,
    EventType,
    EventPriority,
    EventStatus,
    TriggerType,
    SchedulerState,
    Event,
    EventTrigger,
    EventHandler,
    EventSubscription,
    EventMetrics
)

__all__ = [
    # CronScheduler
    'CronScheduler',
    'CronFieldType',
    'ScheduleStatus',
    'ScheduleType',
    'CronField',
    'CronExpression',
    'ScheduleEntry',
    'ScheduleResult',
    
    # NaturalLanguageScheduler
    'NaturalLanguageScheduler',
    'TimeUnit',
    'ScheduleFrequency',
    'TimeReference',
    'DayOfWeek',
    'TimeExpression',
    'ParsedSchedule',
    'ParsingRule',
    
    # ScheduleOptimizer
    'ScheduleOptimizer',
    'OptimizationObjective',
    'ConflictType',
    'OptimizationStrategy',
    'SchedulePriority',
    'ScheduleConflict',
    'OptimizationConstraint',
    'OptimizationResult',
    'ResourceAllocation',
    'OptimizationPlan',
    
    # TimeManager
    'TimeManager',
    'TimeManagerTimeUnit',
    'TimeFormat',
    'TimeZoneType',
    'BusinessDayRule',
    'TimeRange',
    'TimeCalculation',
    'BusinessHours',
    'TimeZoneInfo',
    'TimeConversion',
    
    # EventScheduler
    'EventScheduler',
    'EventType',
    'EventPriority',
    'EventStatus',
    'TriggerType',
    'SchedulerState',
    'Event',
    'EventTrigger',
    'EventHandler',
    'EventSubscription',
    'EventMetrics'
]