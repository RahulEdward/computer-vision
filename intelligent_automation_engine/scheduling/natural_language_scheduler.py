"""
Natural Language Scheduler for Human-Friendly Time Expressions

This module provides natural language parsing for scheduling automation
tasks using human-readable time expressions.
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
from collections import defaultdict

logger = logging.getLogger(__name__)


class TimeUnit(Enum):
    """Time units for scheduling"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class ScheduleFrequency(Enum):
    """Frequency types for schedules"""
    ONCE = "once"
    RECURRING = "recurring"
    CONDITIONAL = "conditional"


class TimeReference(Enum):
    """Time reference points"""
    NOW = "now"
    TODAY = "today"
    TOMORROW = "tomorrow"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    NEXT_WEEK = "next_week"
    THIS_MONTH = "this_month"
    NEXT_MONTH = "next_month"
    THIS_YEAR = "this_year"
    NEXT_YEAR = "next_year"


class DayOfWeek(Enum):
    """Days of the week"""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class TimeExpression:
    """Parsed time expression"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Expression components
    raw_expression: str = ""
    normalized_expression: str = ""
    
    # Time components
    frequency: ScheduleFrequency = ScheduleFrequency.ONCE
    interval: Optional[int] = None
    unit: Optional[TimeUnit] = None
    
    # Specific time components
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None
    
    # Date components
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    weekday: Optional[DayOfWeek] = None
    
    # Reference points
    reference: Optional[TimeReference] = None
    offset: timedelta = timedelta(0)
    
    # Expression metadata
    confidence: float = 0.0
    is_valid: bool = False
    ambiguity_warnings: List[str] = field(default_factory=list)
    
    # Calculated times
    next_execution: Optional[datetime] = None
    calculated_times: List[datetime] = field(default_factory=list)
    
    # Expression context
    timezone: Optional[timezone] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSchedule:
    """Parsed natural language schedule"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Schedule identification
    name: str = ""
    description: str = ""
    
    # Time expression
    time_expression: TimeExpression = field(default_factory=TimeExpression)
    
    # Schedule configuration
    task_function: Optional[Callable] = None
    task_args: Tuple = field(default_factory=tuple)
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Schedule state
    is_enabled: bool = True
    is_recurring: bool = False
    
    # Schedule timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_executions: Optional[int] = None
    execution_count: int = 0
    
    # Schedule metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingRule:
    """Rule for parsing natural language expressions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Rule identification
    name: str = ""
    description: str = ""
    
    # Rule pattern
    pattern: str = ""
    regex: Optional[re.Pattern] = None
    
    # Rule configuration
    priority: int = 0
    is_enabled: bool = True
    
    # Rule processing
    processor: Optional[Callable] = None
    examples: List[str] = field(default_factory=list)
    
    # Rule metadata
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0


class NaturalLanguageParseError(Exception):
    """Exception raised when parsing natural language fails"""
    pass


class NaturalLanguageScheduler:
    """Natural language scheduler for human-friendly time expressions"""
    
    def __init__(self, timezone: Optional[timezone] = None):
        # Core data structures
        self.schedules: Dict[str, ParsedSchedule] = {}
        self.parsing_rules: List[ParsingRule] = []
        
        # Scheduler configuration
        self.timezone = timezone or timezone.utc
        self.default_confidence_threshold = 0.7
        
        # Language patterns
        self.time_patterns = self._initialize_time_patterns()
        self.number_words = self._initialize_number_words()
        self.time_keywords = self._initialize_time_keywords()
        
        # Statistics
        self.parsing_stats = {
            'total_parses': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'average_confidence': 0.0
        }
        
        # Thread safety
        self.scheduler_lock = threading.RLock()
        
        # Initialize default parsing rules
        self._initialize_parsing_rules()
        
        logger.info("Natural language scheduler initialized")
    
    def parse_natural_language(self, expression: str, timezone: Optional[timezone] = None) -> TimeExpression:
        """Parse a natural language time expression"""
        try:
            with self.scheduler_lock:
                # Clean and normalize expression
                normalized = self._normalize_expression(expression)
                
                # Create time expression object
                time_expr = TimeExpression(
                    raw_expression=expression,
                    normalized_expression=normalized,
                    timezone=timezone or self.timezone
                )
                
                # Try parsing with different rules
                best_result = None
                best_confidence = 0.0
                
                for rule in sorted(self.parsing_rules, key=lambda r: r.priority, reverse=True):
                    if not rule.is_enabled:
                        continue
                    
                    try:
                        result = self._apply_parsing_rule(normalized, rule)
                        if result and result.confidence > best_confidence:
                            best_result = result
                            best_confidence = result.confidence
                            
                            # Stop if we have high confidence
                            if best_confidence >= 0.9:
                                break
                    
                    except Exception as e:
                        logger.debug(f"Rule {rule.name} failed: {e}")
                        continue
                
                # Use best result
                if best_result and best_confidence >= self.default_confidence_threshold:
                    time_expr = best_result
                    time_expr.is_valid = True
                    
                    # Calculate execution times
                    time_expr.calculated_times = self._calculate_execution_times(time_expr)
                    if time_expr.calculated_times:
                        time_expr.next_execution = time_expr.calculated_times[0]
                
                # Update statistics
                self.parsing_stats['total_parses'] += 1
                if time_expr.is_valid:
                    self.parsing_stats['successful_parses'] += 1
                else:
                    self.parsing_stats['failed_parses'] += 1
                
                # Update average confidence
                total_confidence = (self.parsing_stats['average_confidence'] * 
                                  (self.parsing_stats['total_parses'] - 1) + time_expr.confidence)
                self.parsing_stats['average_confidence'] = total_confidence / self.parsing_stats['total_parses']
                
                logger.debug(f"Parsed expression: {expression} -> {time_expr.confidence:.2f}")
                return time_expr
                
        except Exception as e:
            logger.error(f"Failed to parse natural language: {e}")
            raise NaturalLanguageParseError(f"Failed to parse expression: {e}")
    
    def add_schedule(self, name: str, expression: str, task_function: Callable,
                    task_args: Tuple = (), task_kwargs: Dict[str, Any] = None,
                    description: str = "", timezone: Optional[timezone] = None,
                    max_executions: Optional[int] = None, tags: List[str] = None) -> str:
        """Add a new natural language schedule"""
        try:
            with self.scheduler_lock:
                # Parse time expression
                time_expr = self.parse_natural_language(expression, timezone)
                
                if not time_expr.is_valid:
                    raise NaturalLanguageParseError(f"Could not parse expression: {expression}")
                
                # Create schedule
                schedule = ParsedSchedule(
                    name=name,
                    description=description,
                    time_expression=time_expr,
                    task_function=task_function,
                    task_args=task_args,
                    task_kwargs=task_kwargs or {},
                    is_recurring=(time_expr.frequency == ScheduleFrequency.RECURRING),
                    max_executions=max_executions,
                    tags=tags or []
                )
                
                # Set timing
                if time_expr.next_execution:
                    schedule.start_time = time_expr.next_execution
                
                # Add to schedules
                self.schedules[schedule.id] = schedule
                
                logger.info(f"Added natural language schedule: {name} ({schedule.id})")
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
                
                schedule = self.schedules.pop(schedule_id)
                logger.info(f"Removed schedule: {schedule.name} ({schedule_id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove schedule: {e}")
            return False
    
    def get_schedule_suggestions(self, partial_expression: str) -> List[str]:
        """Get suggestions for completing a partial expression"""
        try:
            suggestions = []
            partial_lower = partial_expression.lower().strip()
            
            # Common patterns
            common_patterns = [
                "every day at 9am",
                "every monday at 2pm",
                "every hour",
                "every 30 minutes",
                "tomorrow at noon",
                "next week",
                "in 5 minutes",
                "at midnight",
                "every weekday at 8:30am",
                "every weekend",
                "first monday of every month",
                "last day of every month"
            ]
            
            # Filter suggestions based on partial input
            for pattern in common_patterns:
                if not partial_lower or pattern.startswith(partial_lower):
                    suggestions.append(pattern)
            
            # Add time-based suggestions
            if "at" in partial_lower:
                time_suggestions = [
                    "at 9am", "at 2pm", "at noon", "at midnight",
                    "at 8:30am", "at 5:30pm", "at 10:15am"
                ]
                suggestions.extend([s for s in time_suggestions if s.startswith(partial_lower)])
            
            # Add frequency suggestions
            if "every" in partial_lower:
                frequency_suggestions = [
                    "every day", "every hour", "every minute",
                    "every monday", "every weekend", "every month"
                ]
                suggestions.extend([s for s in frequency_suggestions if s.startswith(partial_lower)])
            
            return suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    def validate_expression(self, expression: str) -> Dict[str, Any]:
        """Validate a natural language expression"""
        try:
            time_expr = self.parse_natural_language(expression)
            
            return {
                'is_valid': time_expr.is_valid,
                'confidence': time_expr.confidence,
                'warnings': time_expr.ambiguity_warnings,
                'next_execution': time_expr.next_execution,
                'frequency': time_expr.frequency.value if time_expr.frequency else None,
                'normalized': time_expr.normalized_expression
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'warnings': [str(e)],
                'next_execution': None,
                'frequency': None,
                'normalized': expression
            }
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        try:
            with self.scheduler_lock:
                stats = dict(self.parsing_stats)
                
                # Add rule statistics
                rule_stats = []
                for rule in self.parsing_rules:
                    rule_stats.append({
                        'name': rule.name,
                        'usage_count': rule.usage_count,
                        'success_rate': rule.success_rate,
                        'is_enabled': rule.is_enabled
                    })
                
                stats['rule_statistics'] = rule_stats
                stats['total_schedules'] = len(self.schedules)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get parsing statistics: {e}")
            return {}
    
    # Internal methods
    def _normalize_expression(self, expression: str) -> str:
        """Normalize a natural language expression"""
        try:
            # Convert to lowercase
            normalized = expression.lower().strip()
            
            # Replace common variations
            replacements = {
                r'\bat\b': 'at',
                r'\bevery\b': 'every',
                r'\bin\b': 'in',
                r'\bon\b': 'on',
                r'\bafter\b': 'after',
                r'\bbefore\b': 'before',
                r'\bdaily\b': 'every day',
                r'\bweekly\b': 'every week',
                r'\bmonthly\b': 'every month',
                r'\byearly\b': 'every year',
                r'\bhourly\b': 'every hour',
                r'\bminutes?\b': 'minute',
                r'\bhours?\b': 'hour',
                r'\bdays?\b': 'day',
                r'\bweeks?\b': 'week',
                r'\bmonths?\b': 'month',
                r'\byears?\b': 'year',
                r'\bseconds?\b': 'second'
            }
            
            for pattern, replacement in replacements.items():
                normalized = re.sub(pattern, replacement, normalized)
            
            # Replace number words with digits
            for word, number in self.number_words.items():
                normalized = re.sub(rf'\b{word}\b', str(number), normalized)
            
            # Clean up extra spaces
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize expression: {e}")
            return expression.lower().strip()
    
    def _initialize_time_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize time parsing patterns"""
        patterns = {
            'time_12h': re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', re.IGNORECASE),
            'time_24h': re.compile(r'(\d{1,2}):(\d{2})'),
            'time_special': re.compile(r'\b(noon|midnight|morning|afternoon|evening|night)\b'),
            'interval': re.compile(r'every\s+(\d+)\s+(minute|hour|day|week|month|year)s?'),
            'frequency': re.compile(r'every\s+(minute|hour|day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend)'),
            'relative': re.compile(r'(in|after)\s+(\d+)\s+(minute|hour|day|week|month|year)s?'),
            'absolute': re.compile(r'(tomorrow|today|yesterday|next\s+week|next\s+month|next\s+year)'),
            'weekday': re.compile(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'),
            'ordinal': re.compile(r'\b(first|second|third|fourth|fifth|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'),
            'date': re.compile(r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?'),
            'month_day': re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b')
        }
        
        return patterns
    
    def _initialize_number_words(self) -> Dict[str, int]:
        """Initialize number word mappings"""
        return {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60
        }
    
    def _initialize_time_keywords(self) -> Dict[str, Any]:
        """Initialize time-related keywords"""
        return {
            'weekdays': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            'months': ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december'],
            'time_units': ['second', 'minute', 'hour', 'day', 'week', 'month', 'year'],
            'special_times': {
                'noon': (12, 0),
                'midnight': (0, 0),
                'morning': (9, 0),
                'afternoon': (14, 0),
                'evening': (18, 0),
                'night': (21, 0)
            },
            'ordinals': ['first', 'second', 'third', 'fourth', 'fifth', 'last']
        }
    
    def _initialize_parsing_rules(self):
        """Initialize default parsing rules"""
        try:
            # Rule 1: Every X time unit (e.g., "every 5 minutes")
            self.parsing_rules.append(ParsingRule(
                name="interval_rule",
                description="Parse interval expressions like 'every 5 minutes'",
                pattern=r'every\s+(\d+)\s+(minute|hour|day|week|month|year)s?',
                priority=90,
                processor=self._process_interval_rule,
                examples=["every 5 minutes", "every 2 hours", "every day"]
            ))
            
            # Rule 2: Every time unit (e.g., "every day")
            self.parsing_rules.append(ParsingRule(
                name="frequency_rule",
                description="Parse frequency expressions like 'every day'",
                pattern=r'every\s+(minute|hour|day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend)',
                priority=85,
                processor=self._process_frequency_rule,
                examples=["every day", "every monday", "every hour"]
            ))
            
            # Rule 3: At specific time (e.g., "at 2pm")
            self.parsing_rules.append(ParsingRule(
                name="time_rule",
                description="Parse time expressions like 'at 2pm'",
                pattern=r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
                priority=80,
                processor=self._process_time_rule,
                examples=["at 2pm", "at 9:30am", "at noon"]
            ))
            
            # Rule 4: Relative time (e.g., "in 5 minutes")
            self.parsing_rules.append(ParsingRule(
                name="relative_rule",
                description="Parse relative time expressions like 'in 5 minutes'",
                pattern=r'(in|after)\s+(\d+)\s+(minute|hour|day|week|month|year)s?',
                priority=75,
                processor=self._process_relative_rule,
                examples=["in 5 minutes", "after 2 hours", "in 1 day"]
            ))
            
            # Rule 5: Absolute references (e.g., "tomorrow")
            self.parsing_rules.append(ParsingRule(
                name="absolute_rule",
                description="Parse absolute time references like 'tomorrow'",
                pattern=r'(tomorrow|today|yesterday|next\s+week|next\s+month|next\s+year)',
                priority=70,
                processor=self._process_absolute_rule,
                examples=["tomorrow", "next week", "today"]
            ))
            
            # Compile regex patterns
            for rule in self.parsing_rules:
                if rule.pattern:
                    rule.regex = re.compile(rule.pattern, re.IGNORECASE)
            
            logger.debug(f"Initialized {len(self.parsing_rules)} parsing rules")
            
        except Exception as e:
            logger.error(f"Failed to initialize parsing rules: {e}")
    
    def _apply_parsing_rule(self, expression: str, rule: ParsingRule) -> Optional[TimeExpression]:
        """Apply a parsing rule to an expression"""
        try:
            if not rule.regex or not rule.processor:
                return None
            
            match = rule.regex.search(expression)
            if not match:
                return None
            
            # Apply processor
            result = rule.processor(expression, match)
            
            # Update rule statistics
            rule.usage_count += 1
            if result and result.confidence > 0.5:
                rule.success_rate = ((rule.success_rate * (rule.usage_count - 1)) + 1.0) / rule.usage_count
            else:
                rule.success_rate = (rule.success_rate * (rule.usage_count - 1)) / rule.usage_count
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply parsing rule {rule.name}: {e}")
            return None
    
    def _process_interval_rule(self, expression: str, match: re.Match) -> TimeExpression:
        """Process interval rule (e.g., 'every 5 minutes')"""
        try:
            interval = int(match.group(1))
            unit_str = match.group(2)
            
            time_expr = TimeExpression(
                raw_expression=expression,
                normalized_expression=expression,
                frequency=ScheduleFrequency.RECURRING,
                interval=interval,
                unit=TimeUnit(unit_str),
                confidence=0.9,
                timezone=self.timezone
            )
            
            return time_expr
            
        except Exception as e:
            logger.error(f"Failed to process interval rule: {e}")
            return None
    
    def _process_frequency_rule(self, expression: str, match: re.Match) -> TimeExpression:
        """Process frequency rule (e.g., 'every day')"""
        try:
            unit_str = match.group(1)
            
            time_expr = TimeExpression(
                raw_expression=expression,
                normalized_expression=expression,
                frequency=ScheduleFrequency.RECURRING,
                interval=1,
                confidence=0.85,
                timezone=self.timezone
            )
            
            # Handle special cases
            if unit_str in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                time_expr.unit = TimeUnit.WEEK
                time_expr.weekday = DayOfWeek[unit_str.upper()]
            elif unit_str == 'weekday':
                time_expr.unit = TimeUnit.DAY
                time_expr.ambiguity_warnings.append("Weekday schedule will run Monday-Friday")
            elif unit_str == 'weekend':
                time_expr.unit = TimeUnit.WEEK
                time_expr.ambiguity_warnings.append("Weekend schedule will run Saturday-Sunday")
            else:
                time_expr.unit = TimeUnit(unit_str)
            
            return time_expr
            
        except Exception as e:
            logger.error(f"Failed to process frequency rule: {e}")
            return None
    
    def _process_time_rule(self, expression: str, match: re.Match) -> TimeExpression:
        """Process time rule (e.g., 'at 2pm')"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)
            
            # Convert to 24-hour format
            if period:
                if period.lower() == 'pm' and hour != 12:
                    hour += 12
                elif period.lower() == 'am' and hour == 12:
                    hour = 0
            
            time_expr = TimeExpression(
                raw_expression=expression,
                normalized_expression=expression,
                frequency=ScheduleFrequency.ONCE,
                hour=hour,
                minute=minute,
                confidence=0.8,
                timezone=self.timezone
            )
            
            # Check for recurring context
            if 'every' in expression or 'daily' in expression:
                time_expr.frequency = ScheduleFrequency.RECURRING
                time_expr.unit = TimeUnit.DAY
                time_expr.interval = 1
            
            return time_expr
            
        except Exception as e:
            logger.error(f"Failed to process time rule: {e}")
            return None
    
    def _process_relative_rule(self, expression: str, match: re.Match) -> TimeExpression:
        """Process relative rule (e.g., 'in 5 minutes')"""
        try:
            direction = match.group(1)  # 'in' or 'after'
            amount = int(match.group(2))
            unit_str = match.group(3)
            
            time_expr = TimeExpression(
                raw_expression=expression,
                normalized_expression=expression,
                frequency=ScheduleFrequency.ONCE,
                interval=amount,
                unit=TimeUnit(unit_str),
                confidence=0.9,
                timezone=self.timezone
            )
            
            # Calculate offset
            if unit_str == 'minute':
                time_expr.offset = timedelta(minutes=amount)
            elif unit_str == 'hour':
                time_expr.offset = timedelta(hours=amount)
            elif unit_str == 'day':
                time_expr.offset = timedelta(days=amount)
            elif unit_str == 'week':
                time_expr.offset = timedelta(weeks=amount)
            elif unit_str == 'month':
                time_expr.offset = timedelta(days=amount * 30)  # Approximate
            elif unit_str == 'year':
                time_expr.offset = timedelta(days=amount * 365)  # Approximate
            
            return time_expr
            
        except Exception as e:
            logger.error(f"Failed to process relative rule: {e}")
            return None
    
    def _process_absolute_rule(self, expression: str, match: re.Match) -> TimeExpression:
        """Process absolute rule (e.g., 'tomorrow')"""
        try:
            reference_str = match.group(1)
            
            time_expr = TimeExpression(
                raw_expression=expression,
                normalized_expression=expression,
                frequency=ScheduleFrequency.ONCE,
                confidence=0.8,
                timezone=self.timezone
            )
            
            # Set reference and offset
            if reference_str == 'today':
                time_expr.reference = TimeReference.TODAY
            elif reference_str == 'tomorrow':
                time_expr.reference = TimeReference.TOMORROW
                time_expr.offset = timedelta(days=1)
            elif reference_str == 'yesterday':
                time_expr.reference = TimeReference.YESTERDAY
                time_expr.offset = timedelta(days=-1)
            elif 'next week' in reference_str:
                time_expr.reference = TimeReference.NEXT_WEEK
                time_expr.offset = timedelta(weeks=1)
            elif 'next month' in reference_str:
                time_expr.reference = TimeReference.NEXT_MONTH
                time_expr.offset = timedelta(days=30)  # Approximate
            elif 'next year' in reference_str:
                time_expr.reference = TimeReference.NEXT_YEAR
                time_expr.offset = timedelta(days=365)  # Approximate
            
            return time_expr
            
        except Exception as e:
            logger.error(f"Failed to process absolute rule: {e}")
            return None
    
    def _calculate_execution_times(self, time_expr: TimeExpression, count: int = 10) -> List[datetime]:
        """Calculate execution times for a time expression"""
        try:
            execution_times = []
            current_time = datetime.now(time_expr.timezone)
            
            if time_expr.frequency == ScheduleFrequency.ONCE:
                # One-time execution
                if time_expr.reference:
                    base_time = current_time + time_expr.offset
                else:
                    base_time = current_time + time_expr.offset
                
                # Apply specific time if provided
                if time_expr.hour is not None:
                    base_time = base_time.replace(
                        hour=time_expr.hour,
                        minute=time_expr.minute or 0,
                        second=time_expr.second or 0,
                        microsecond=0
                    )
                
                execution_times.append(base_time)
            
            elif time_expr.frequency == ScheduleFrequency.RECURRING:
                # Recurring execution
                next_time = current_time
                
                for _ in range(count):
                    # Calculate next execution based on unit
                    if time_expr.unit == TimeUnit.MINUTE:
                        next_time += timedelta(minutes=time_expr.interval or 1)
                    elif time_expr.unit == TimeUnit.HOUR:
                        next_time += timedelta(hours=time_expr.interval or 1)
                    elif time_expr.unit == TimeUnit.DAY:
                        next_time += timedelta(days=time_expr.interval or 1)
                    elif time_expr.unit == TimeUnit.WEEK:
                        next_time += timedelta(weeks=time_expr.interval or 1)
                    elif time_expr.unit == TimeUnit.MONTH:
                        next_time += timedelta(days=(time_expr.interval or 1) * 30)
                    elif time_expr.unit == TimeUnit.YEAR:
                        next_time += timedelta(days=(time_expr.interval or 1) * 365)
                    
                    # Apply specific time if provided
                    if time_expr.hour is not None:
                        next_time = next_time.replace(
                            hour=time_expr.hour,
                            minute=time_expr.minute or 0,
                            second=time_expr.second or 0,
                            microsecond=0
                        )
                    
                    execution_times.append(next_time)
            
            return execution_times
            
        except Exception as e:
            logger.error(f"Failed to calculate execution times: {e}")
            return []