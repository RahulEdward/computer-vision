"""
Cron Scheduler for Time-Based Automation

This module provides comprehensive cron expression parsing and scheduling
capabilities for automation workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union, Tuple
from datetime import datetime, timedelta, timezone
import uuid
import logging
import re
import calendar
import threading
import time
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)


class CronFieldType(Enum):
    """Types of cron fields"""
    SECOND = "second"  # 0-59
    MINUTE = "minute"  # 0-59
    HOUR = "hour"  # 0-23
    DAY = "day"  # 1-31
    MONTH = "month"  # 1-12
    WEEKDAY = "weekday"  # 0-6 (Sunday=0)
    YEAR = "year"  # 1970-3000


class ScheduleStatus(Enum):
    """Status of scheduled tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ScheduleType(Enum):
    """Types of schedules"""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"
    CONDITIONAL = "conditional"


@dataclass
class CronField:
    """Represents a single field in a cron expression"""
    field_type: CronFieldType
    raw_value: str = "*"
    values: Set[int] = field(default_factory=set)
    ranges: List[Tuple[int, int]] = field(default_factory=list)
    step: Optional[int] = None
    is_wildcard: bool = False
    is_last: bool = False  # For 'L' in day field
    is_weekday: bool = False  # For 'W' in day field
    is_hash: bool = False  # For '#' in weekday field
    hash_value: Optional[int] = None


@dataclass
class CronExpression:
    """Parsed cron expression"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Expression components
    raw_expression: str = ""
    fields: Dict[CronFieldType, CronField] = field(default_factory=dict)
    
    # Expression metadata
    description: str = ""
    timezone: Optional[timezone] = None
    is_valid: bool = False
    
    # Expression statistics
    next_runs: List[datetime] = field(default_factory=list)
    last_calculated: Optional[datetime] = None
    
    # Expression context
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleEntry:
    """Entry in the schedule"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Schedule identification
    name: str = ""
    description: str = ""
    
    # Schedule configuration
    cron_expression: CronExpression = field(default_factory=CronExpression)
    schedule_type: ScheduleType = ScheduleType.CRON
    timezone: Optional[timezone] = None
    
    # Schedule execution
    task_function: Optional[Callable] = None
    task_args: Tuple = field(default_factory=tuple)
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Schedule state
    status: ScheduleStatus = ScheduleStatus.PENDING
    is_enabled: bool = True
    is_recurring: bool = True
    
    # Schedule timing
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    
    # Schedule metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleResult:
    """Result of schedule execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Result identification
    schedule_id: str = ""
    execution_id: str = ""
    
    # Result details
    status: ScheduleStatus = ScheduleStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: timedelta = timedelta(0)
    
    # Result data
    result: Any = None
    error: Optional[str] = None
    output: str = ""
    
    # Result metadata
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class CronParseError(Exception):
    """Exception raised when parsing cron expressions fails"""
    pass


class CronScheduler:
    """Comprehensive cron scheduler for time-based automation"""
    
    def __init__(self, timezone: Optional[timezone] = None):
        # Core data structures
        self.schedules: Dict[str, ScheduleEntry] = {}
        self.results: deque = deque(maxlen=1000)
        self.execution_queue: List[Tuple[datetime, str]] = []  # Priority queue
        
        # Scheduler configuration
        self.timezone = timezone or timezone.utc
        self.is_running = False
        self.check_interval = timedelta(seconds=1)
        
        # Scheduler threads
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Cron field definitions
        self.field_ranges = {
            CronFieldType.SECOND: (0, 59),
            CronFieldType.MINUTE: (0, 59),
            CronFieldType.HOUR: (0, 23),
            CronFieldType.DAY: (1, 31),
            CronFieldType.MONTH: (1, 12),
            CronFieldType.WEEKDAY: (0, 6),
            CronFieldType.YEAR: (1970, 3000)
        }
        
        # Month and weekday names
        self.month_names = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        self.weekday_names = {
            'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6
        }
        
        # Statistics
        self.scheduler_stats = {
            'total_schedules': 0,
            'active_schedules': 0,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'skipped_executions': 0
        }
        
        # Thread safety
        self.scheduler_lock = threading.RLock()
        
        logger.info("Cron scheduler initialized")
    
    def parse_cron_expression(self, expression: str, timezone: Optional[timezone] = None) -> CronExpression:
        """Parse a cron expression into structured format"""
        try:
            # Clean and validate expression
            expression = expression.strip()
            if not expression:
                raise CronParseError("Empty cron expression")
            
            # Split expression into fields
            parts = expression.split()
            
            # Support different cron formats
            if len(parts) == 5:
                # Standard cron: minute hour day month weekday
                field_types = [CronFieldType.MINUTE, CronFieldType.HOUR, 
                              CronFieldType.DAY, CronFieldType.MONTH, CronFieldType.WEEKDAY]
            elif len(parts) == 6:
                # Extended cron: second minute hour day month weekday
                field_types = [CronFieldType.SECOND, CronFieldType.MINUTE, CronFieldType.HOUR,
                              CronFieldType.DAY, CronFieldType.MONTH, CronFieldType.WEEKDAY]
            elif len(parts) == 7:
                # Full cron: second minute hour day month weekday year
                field_types = [CronFieldType.SECOND, CronFieldType.MINUTE, CronFieldType.HOUR,
                              CronFieldType.DAY, CronFieldType.MONTH, CronFieldType.WEEKDAY, CronFieldType.YEAR]
            else:
                raise CronParseError(f"Invalid cron expression format: {expression}")
            
            # Parse each field
            fields = {}
            for i, (part, field_type) in enumerate(zip(parts, field_types)):
                fields[field_type] = self._parse_cron_field(part, field_type)
            
            # Create cron expression object
            cron_expr = CronExpression(
                raw_expression=expression,
                fields=fields,
                timezone=timezone or self.timezone,
                is_valid=True,
                description=self._generate_description(fields)
            )
            
            # Calculate next runs
            cron_expr.next_runs = self._calculate_next_runs(cron_expr, count=10)
            cron_expr.last_calculated = datetime.now()
            
            logger.debug(f"Parsed cron expression: {expression}")
            return cron_expr
            
        except Exception as e:
            logger.error(f"Failed to parse cron expression '{expression}': {e}")
            raise CronParseError(f"Failed to parse cron expression: {e}")
    
    def add_schedule(self, name: str, cron_expression: str, task_function: Callable,
                    task_args: Tuple = (), task_kwargs: Dict[str, Any] = None,
                    description: str = "", timezone: Optional[timezone] = None,
                    max_runs: Optional[int] = None, tags: List[str] = None) -> str:
        """Add a new cron schedule"""
        try:
            with self.scheduler_lock:
                # Parse cron expression
                parsed_expr = self.parse_cron_expression(cron_expression, timezone)
                
                # Create schedule entry
                schedule = ScheduleEntry(
                    name=name,
                    description=description,
                    cron_expression=parsed_expr,
                    timezone=timezone or self.timezone,
                    task_function=task_function,
                    task_args=task_args,
                    task_kwargs=task_kwargs or {},
                    max_runs=max_runs,
                    tags=tags or []
                )
                
                # Calculate next run
                schedule.next_run = self._get_next_run_time(parsed_expr)
                
                # Add to schedules
                self.schedules[schedule.id] = schedule
                
                # Update execution queue
                if schedule.next_run:
                    heapq.heappush(self.execution_queue, (schedule.next_run, schedule.id))
                
                # Update statistics
                self.scheduler_stats['total_schedules'] += 1
                self.scheduler_stats['active_schedules'] += 1
                
                logger.info(f"Added schedule: {name} ({schedule.id})")
                return schedule.id
                
        except Exception as e:
            logger.error(f"Failed to add schedule: {e}")
            return ""
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule"""
        try:
            with self.scheduler_lock:
                if schedule_id not in self.schedules:
                    logger.warning(f"Schedule not found: {schedule_id}")
                    return False
                
                # Remove from schedules
                schedule = self.schedules.pop(schedule_id)
                
                # Remove from execution queue
                self.execution_queue = [(time, sid) for time, sid in self.execution_queue 
                                      if sid != schedule_id]
                heapq.heapify(self.execution_queue)
                
                # Update statistics
                if schedule.is_enabled:
                    self.scheduler_stats['active_schedules'] -= 1
                
                logger.info(f"Removed schedule: {schedule.name} ({schedule_id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove schedule: {e}")
            return False
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule"""
        try:
            with self.scheduler_lock:
                if schedule_id not in self.schedules:
                    logger.warning(f"Schedule not found: {schedule_id}")
                    return False
                
                schedule = self.schedules[schedule_id]
                if not schedule.is_enabled:
                    schedule.is_enabled = True
                    schedule.updated_at = datetime.now()
                    
                    # Recalculate next run
                    schedule.next_run = self._get_next_run_time(schedule.cron_expression)
                    
                    # Add to execution queue
                    if schedule.next_run:
                        heapq.heappush(self.execution_queue, (schedule.next_run, schedule_id))
                    
                    # Update statistics
                    self.scheduler_stats['active_schedules'] += 1
                    
                    logger.info(f"Enabled schedule: {schedule.name} ({schedule_id})")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to enable schedule: {e}")
            return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule"""
        try:
            with self.scheduler_lock:
                if schedule_id not in self.schedules:
                    logger.warning(f"Schedule not found: {schedule_id}")
                    return False
                
                schedule = self.schedules[schedule_id]
                if schedule.is_enabled:
                    schedule.is_enabled = False
                    schedule.updated_at = datetime.now()
                    
                    # Remove from execution queue
                    self.execution_queue = [(time, sid) for time, sid in self.execution_queue 
                                          if sid != schedule_id]
                    heapq.heapify(self.execution_queue)
                    
                    # Update statistics
                    self.scheduler_stats['active_schedules'] -= 1
                    
                    logger.info(f"Disabled schedule: {schedule.name} ({schedule_id})")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to disable schedule: {e}")
            return False
    
    def start_scheduler(self) -> bool:
        """Start the cron scheduler"""
        try:
            with self.scheduler_lock:
                if self.is_running:
                    logger.warning("Scheduler is already running")
                    return False
                
                self.is_running = True
                self.stop_event.clear()
                
                # Start scheduler thread
                self.scheduler_thread = threading.Thread(
                    target=self._scheduler_loop, daemon=True
                )
                self.scheduler_thread.start()
                
                logger.info("Started cron scheduler")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False
    
    def stop_scheduler(self) -> bool:
        """Stop the cron scheduler"""
        try:
            with self.scheduler_lock:
                if not self.is_running:
                    logger.warning("Scheduler is not running")
                    return False
                
                self.is_running = False
                self.stop_event.set()
                
                # Wait for scheduler thread to finish
                if self.scheduler_thread:
                    self.scheduler_thread.join(timeout=10)
                
                logger.info("Stopped cron scheduler")
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False
    
    def get_schedule_status(self, schedule_id: str) -> Dict[str, Any]:
        """Get status of a schedule"""
        try:
            with self.scheduler_lock:
                if schedule_id not in self.schedules:
                    return {'error': 'Schedule not found'}
                
                schedule = self.schedules[schedule_id]
                
                return {
                    'schedule_id': schedule_id,
                    'name': schedule.name,
                    'description': schedule.description,
                    'cron_expression': schedule.cron_expression.raw_expression,
                    'status': schedule.status.value,
                    'is_enabled': schedule.is_enabled,
                    'is_recurring': schedule.is_recurring,
                    'next_run': schedule.next_run,
                    'last_run': schedule.last_run,
                    'run_count': schedule.run_count,
                    'max_runs': schedule.max_runs,
                    'created_at': schedule.created_at,
                    'updated_at': schedule.updated_at,
                    'tags': schedule.tags
                }
                
        except Exception as e:
            logger.error(f"Failed to get schedule status: {e}")
            return {'error': str(e)}
    
    def get_next_runs(self, schedule_id: str, count: int = 10) -> List[datetime]:
        """Get next run times for a schedule"""
        try:
            with self.scheduler_lock:
                if schedule_id not in self.schedules:
                    return []
                
                schedule = self.schedules[schedule_id]
                return self._calculate_next_runs(schedule.cron_expression, count)
                
        except Exception as e:
            logger.error(f"Failed to get next runs: {e}")
            return []
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        try:
            with self.scheduler_lock:
                stats = dict(self.scheduler_stats)
                
                # Add current state
                stats.update({
                    'is_running': self.is_running,
                    'pending_executions': len(self.execution_queue),
                    'total_results': len(self.results)
                })
                
                # Add schedule status breakdown
                status_counts = defaultdict(int)
                for schedule in self.schedules.values():
                    status_counts[schedule.status.value] += 1
                stats['schedule_status_counts'] = dict(status_counts)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get scheduler statistics: {e}")
            return {}
    
    # Internal methods
    def _parse_cron_field(self, field_value: str, field_type: CronFieldType) -> CronField:
        """Parse a single cron field"""
        try:
            field = CronField(field_type=field_type, raw_value=field_value)
            min_val, max_val = self.field_ranges[field_type]
            
            # Handle wildcard
            if field_value == '*':
                field.is_wildcard = True
                field.values = set(range(min_val, max_val + 1))
                return field
            
            # Handle special characters
            if 'L' in field_value and field_type == CronFieldType.DAY:
                field.is_last = True
                return field
            
            if 'W' in field_value and field_type == CronFieldType.DAY:
                field.is_weekday = True
                # Extract day number before 'W'
                day_str = field_value.replace('W', '')
                if day_str.isdigit():
                    field.values.add(int(day_str))
                return field
            
            if '#' in field_value and field_type == CronFieldType.WEEKDAY:
                field.is_hash = True
                parts = field_value.split('#')
                if len(parts) == 2:
                    field.values.add(int(parts[0]))
                    field.hash_value = int(parts[1])
                return field
            
            # Handle step values
            if '/' in field_value:
                range_part, step_part = field_value.split('/', 1)
                field.step = int(step_part)
                field_value = range_part
            
            # Handle ranges and lists
            values = set()
            for part in field_value.split(','):
                if '-' in part:
                    # Range
                    start_str, end_str = part.split('-', 1)
                    start = self._parse_field_value(start_str, field_type)
                    end = self._parse_field_value(end_str, field_type)
                    
                    if field.step:
                        values.update(range(start, end + 1, field.step))
                    else:
                        values.update(range(start, end + 1))
                    
                    field.ranges.append((start, end))
                else:
                    # Single value
                    value = self._parse_field_value(part, field_type)
                    values.add(value)
            
            # Apply step to all values if specified
            if field.step and not field.ranges:
                stepped_values = set()
                for value in sorted(values):
                    if (value - min_val) % field.step == 0:
                        stepped_values.add(value)
                values = stepped_values
            
            field.values = values
            return field
            
        except Exception as e:
            raise CronParseError(f"Failed to parse field '{field_value}': {e}")
    
    def _parse_field_value(self, value_str: str, field_type: CronFieldType) -> int:
        """Parse a single field value (number or name)"""
        try:
            # Try parsing as number
            if value_str.isdigit():
                return int(value_str)
            
            # Try parsing as name
            value_lower = value_str.lower()
            
            if field_type == CronFieldType.MONTH and value_lower in self.month_names:
                return self.month_names[value_lower]
            
            if field_type == CronFieldType.WEEKDAY and value_lower in self.weekday_names:
                return self.weekday_names[value_lower]
            
            raise ValueError(f"Invalid field value: {value_str}")
            
        except Exception as e:
            raise CronParseError(f"Failed to parse field value '{value_str}': {e}")
    
    def _generate_description(self, fields: Dict[CronFieldType, CronField]) -> str:
        """Generate human-readable description of cron expression"""
        try:
            parts = []
            
            # Handle common patterns
            minute_field = fields.get(CronFieldType.MINUTE)
            hour_field = fields.get(CronFieldType.HOUR)
            day_field = fields.get(CronFieldType.DAY)
            month_field = fields.get(CronFieldType.MONTH)
            weekday_field = fields.get(CronFieldType.WEEKDAY)
            
            # Every minute
            if (minute_field and minute_field.is_wildcard and
                hour_field and hour_field.is_wildcard):
                return "Every minute"
            
            # Every hour
            if (minute_field and len(minute_field.values) == 1 and 0 in minute_field.values and
                hour_field and hour_field.is_wildcard):
                return "Every hour"
            
            # Daily at specific time
            if (minute_field and len(minute_field.values) == 1 and
                hour_field and len(hour_field.values) == 1 and
                day_field and day_field.is_wildcard):
                minute = list(minute_field.values)[0]
                hour = list(hour_field.values)[0]
                return f"Daily at {hour:02d}:{minute:02d}"
            
            # Weekly on specific day
            if (weekday_field and len(weekday_field.values) == 1 and
                day_field and day_field.is_wildcard):
                weekday = list(weekday_field.values)[0]
                weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 
                               'Thursday', 'Friday', 'Saturday']
                return f"Weekly on {weekday_names[weekday]}"
            
            # Monthly
            if (day_field and len(day_field.values) == 1 and
                month_field and month_field.is_wildcard):
                day = list(day_field.values)[0]
                return f"Monthly on day {day}"
            
            return "Custom schedule"
            
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return "Custom schedule"
    
    def _calculate_next_runs(self, cron_expr: CronExpression, count: int = 10) -> List[datetime]:
        """Calculate next run times for a cron expression"""
        try:
            next_runs = []
            current_time = datetime.now(cron_expr.timezone)
            
            # Start from next minute
            next_time = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            attempts = 0
            max_attempts = count * 100  # Prevent infinite loops
            
            while len(next_runs) < count and attempts < max_attempts:
                if self._matches_cron_expression(next_time, cron_expr):
                    next_runs.append(next_time)
                
                next_time += timedelta(minutes=1)
                attempts += 1
            
            return next_runs
            
        except Exception as e:
            logger.error(f"Failed to calculate next runs: {e}")
            return []
    
    def _get_next_run_time(self, cron_expr: CronExpression) -> Optional[datetime]:
        """Get the next run time for a cron expression"""
        try:
            next_runs = self._calculate_next_runs(cron_expr, count=1)
            return next_runs[0] if next_runs else None
            
        except Exception as e:
            logger.error(f"Failed to get next run time: {e}")
            return None
    
    def _matches_cron_expression(self, dt: datetime, cron_expr: CronExpression) -> bool:
        """Check if a datetime matches a cron expression"""
        try:
            fields = cron_expr.fields
            
            # Check each field
            for field_type, field in fields.items():
                if not self._matches_field(dt, field):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to match cron expression: {e}")
            return False
    
    def _matches_field(self, dt: datetime, field: CronField) -> bool:
        """Check if a datetime matches a cron field"""
        try:
            if field.is_wildcard:
                return True
            
            # Get the appropriate value from datetime
            if field.field_type == CronFieldType.SECOND:
                value = dt.second
            elif field.field_type == CronFieldType.MINUTE:
                value = dt.minute
            elif field.field_type == CronFieldType.HOUR:
                value = dt.hour
            elif field.field_type == CronFieldType.DAY:
                if field.is_last:
                    # Last day of month
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    return dt.day == last_day
                elif field.is_weekday:
                    # Nearest weekday to specified day
                    target_day = list(field.values)[0] if field.values else dt.day
                    return self._is_nearest_weekday(dt, target_day)
                else:
                    value = dt.day
            elif field.field_type == CronFieldType.MONTH:
                value = dt.month
            elif field.field_type == CronFieldType.WEEKDAY:
                if field.is_hash:
                    # Nth occurrence of weekday in month
                    weekday = list(field.values)[0]
                    occurrence = field.hash_value
                    return self._is_nth_weekday(dt, weekday, occurrence)
                else:
                    value = dt.weekday()
                    # Convert Python weekday (0=Monday) to cron weekday (0=Sunday)
                    value = (value + 1) % 7
            elif field.field_type == CronFieldType.YEAR:
                value = dt.year
            else:
                return True
            
            return value in field.values
            
        except Exception as e:
            logger.error(f"Failed to match field: {e}")
            return False
    
    def _is_nearest_weekday(self, dt: datetime, target_day: int) -> bool:
        """Check if date is nearest weekday to target day"""
        try:
            # Get the target date
            target_date = dt.replace(day=target_day)
            
            # If target is a weekday, check if it's the same day
            if target_date.weekday() < 5:  # Monday=0, Friday=4
                return dt.day == target_day
            
            # Find nearest weekday
            if target_date.weekday() == 5:  # Saturday
                # Check Friday (previous day)
                friday = target_date - timedelta(days=1)
                if friday.month == dt.month:
                    return dt.day == friday.day
                # Otherwise check Monday (next weekday)
                monday = target_date + timedelta(days=2)
                if monday.month == dt.month:
                    return dt.day == monday.day
            
            elif target_date.weekday() == 6:  # Sunday
                # Check Monday (next day)
                monday = target_date + timedelta(days=1)
                if monday.month == dt.month:
                    return dt.day == monday.day
                # Otherwise check Friday (previous weekday)
                friday = target_date - timedelta(days=2)
                if friday.month == dt.month:
                    return dt.day == friday.day
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check nearest weekday: {e}")
            return False
    
    def _is_nth_weekday(self, dt: datetime, weekday: int, occurrence: int) -> bool:
        """Check if date is the nth occurrence of weekday in month"""
        try:
            # Convert cron weekday to Python weekday
            python_weekday = (weekday - 1) % 7
            
            if dt.weekday() != python_weekday:
                return False
            
            # Count occurrences of this weekday in the month
            first_day = dt.replace(day=1)
            days_in_month = calendar.monthrange(dt.year, dt.month)[1]
            
            occurrence_count = 0
            for day in range(1, days_in_month + 1):
                check_date = dt.replace(day=day)
                if check_date.weekday() == python_weekday:
                    occurrence_count += 1
                    if check_date.day == dt.day:
                        return occurrence_count == occurrence
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check nth weekday: {e}")
            return False
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    current_time = datetime.now(self.timezone)
                    
                    # Process due schedules
                    self._process_due_schedules(current_time)
                    
                    # Sleep until next check
                    self.stop_event.wait(self.check_interval.total_seconds())
                    
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(60)  # Longer sleep on error
            
        except Exception as e:
            logger.error(f"Scheduler loop failed: {e}")
        finally:
            logger.debug("Scheduler loop ended")
    
    def _process_due_schedules(self, current_time: datetime):
        """Process schedules that are due for execution"""
        try:
            with self.scheduler_lock:
                # Get due schedules
                due_schedules = []
                while (self.execution_queue and 
                       self.execution_queue[0][0] <= current_time):
                    _, schedule_id = heapq.heappop(self.execution_queue)
                    if schedule_id in self.schedules:
                        due_schedules.append(schedule_id)
                
                # Execute due schedules
                for schedule_id in due_schedules:
                    self._execute_schedule(schedule_id, current_time)
            
        except Exception as e:
            logger.error(f"Failed to process due schedules: {e}")
    
    def _execute_schedule(self, schedule_id: str, execution_time: datetime):
        """Execute a scheduled task"""
        try:
            schedule = self.schedules.get(schedule_id)
            if not schedule or not schedule.is_enabled:
                return
            
            # Check if max runs reached
            if schedule.max_runs and schedule.run_count >= schedule.max_runs:
                schedule.status = ScheduleStatus.COMPLETED
                return
            
            # Create execution result
            result = ScheduleResult(
                schedule_id=schedule_id,
                execution_id=str(uuid.uuid4()),
                start_time=execution_time
            )
            
            try:
                # Update schedule status
                schedule.status = ScheduleStatus.RUNNING
                schedule.last_run = execution_time
                schedule.run_count += 1
                
                # Execute task
                if schedule.task_function:
                    task_result = schedule.task_function(*schedule.task_args, **schedule.task_kwargs)
                    result.result = task_result
                    result.status = ScheduleStatus.COMPLETED
                else:
                    result.status = ScheduleStatus.SKIPPED
                
                # Update statistics
                if result.status == ScheduleStatus.COMPLETED:
                    self.scheduler_stats['successful_executions'] += 1
                else:
                    self.scheduler_stats['skipped_executions'] += 1
                
            except Exception as e:
                result.status = ScheduleStatus.FAILED
                result.error = str(e)
                self.scheduler_stats['failed_executions'] += 1
                logger.error(f"Schedule execution failed: {e}")
            
            finally:
                # Update result timing
                result.end_time = datetime.now(self.timezone)
                result.duration = result.end_time - result.start_time
                
                # Update schedule status
                if schedule.is_recurring and result.status != ScheduleStatus.FAILED:
                    schedule.status = ScheduleStatus.PENDING
                    
                    # Schedule next run
                    next_run = self._get_next_run_time(schedule.cron_expression)
                    if next_run:
                        schedule.next_run = next_run
                        heapq.heappush(self.execution_queue, (next_run, schedule_id))
                else:
                    schedule.status = ScheduleStatus.COMPLETED
                
                # Store result
                self.results.append(result)
                self.scheduler_stats['total_executions'] += 1
                
                logger.debug(f"Executed schedule: {schedule.name} ({schedule_id})")
            
        except Exception as e:
            logger.error(f"Failed to execute schedule: {e}")