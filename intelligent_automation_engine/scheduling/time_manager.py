"""
Time Manager for Intelligent Time Management

This module provides comprehensive time management capabilities including
timezone handling, time calculations, and temporal operations for scheduling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Union, Tuple, Callable
from datetime import datetime, timedelta, timezone, date, time
import uuid
import logging
import threading
import pytz
from collections import defaultdict, deque
import calendar
import re

logger = logging.getLogger(__name__)


class TimeUnit(Enum):
    """Time units for calculations"""
    MICROSECOND = "microsecond"
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TimeFormat(Enum):
    """Time format types"""
    ISO_8601 = "iso_8601"
    RFC_3339 = "rfc_3339"
    UNIX_TIMESTAMP = "unix_timestamp"
    HUMAN_READABLE = "human_readable"
    CUSTOM = "custom"


class TimeZoneType(Enum):
    """Timezone types"""
    UTC = "utc"
    LOCAL = "local"
    FIXED_OFFSET = "fixed_offset"
    NAMED = "named"


class BusinessDayRule(Enum):
    """Business day rules"""
    WEEKDAYS_ONLY = "weekdays_only"
    EXCLUDE_HOLIDAYS = "exclude_holidays"
    CUSTOM_SCHEDULE = "custom_schedule"


@dataclass
class TimeRange:
    """Represents a time range"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Time range definition
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Time range properties
    timezone: Optional[timezone] = None
    is_all_day: bool = False
    is_recurring: bool = False
    
    # Recurrence information
    recurrence_pattern: Optional[str] = None
    recurrence_end: Optional[datetime] = None
    
    # Time range metadata
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.end_time is None and self.duration is not None:
            self.end_time = self.start_time + self.duration
        elif self.duration is None and self.end_time is not None:
            self.duration = self.end_time - self.start_time


@dataclass
class TimeCalculation:
    """Represents a time calculation result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Calculation identification
    operation: str = ""
    operands: List[Any] = field(default_factory=list)
    
    # Calculation result
    result: Any = None
    result_type: str = ""
    
    # Calculation metadata
    calculated_at: datetime = field(default_factory=datetime.now)
    calculation_time: timedelta = timedelta(0)
    precision: str = "second"
    
    # Error information
    success: bool = True
    error_message: str = ""


@dataclass
class BusinessHours:
    """Represents business hours configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Business hours definition
    name: str = ""
    timezone: timezone = timezone.utc
    
    # Weekly schedule (0=Monday, 6=Sunday)
    weekly_schedule: Dict[int, List[Tuple[time, time]]] = field(default_factory=dict)
    
    # Holiday configuration
    holidays: List[date] = field(default_factory=list)
    holiday_rule: BusinessDayRule = BusinessDayRule.EXCLUDE_HOLIDAYS
    
    # Special dates
    special_hours: Dict[date, List[Tuple[time, time]]] = field(default_factory=dict)
    closed_dates: List[date] = field(default_factory=list)
    
    # Business hours metadata
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class TimeZoneInfo:
    """Timezone information"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timezone identification
    name: str = ""
    abbreviation: str = ""
    timezone_type: TimeZoneType = TimeZoneType.UTC
    
    # Timezone details
    utc_offset: timedelta = timedelta(0)
    dst_offset: Optional[timedelta] = None
    is_dst_active: bool = False
    
    # Timezone metadata
    country: str = ""
    region: str = ""
    city: str = ""
    
    # Timezone object
    timezone_obj: Optional[timezone] = None


@dataclass
class TimeConversion:
    """Time conversion result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Conversion identification
    source_time: datetime = field(default_factory=datetime.now)
    source_timezone: Optional[timezone] = None
    target_timezone: Optional[timezone] = None
    
    # Conversion result
    converted_time: Optional[datetime] = None
    conversion_offset: timedelta = timedelta(0)
    
    # Conversion metadata
    converted_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""


class TimeManager:
    """Comprehensive time manager for intelligent time management"""
    
    def __init__(self, default_timezone: Optional[timezone] = None):
        # Core configuration
        self.default_timezone = default_timezone or timezone.utc
        self.business_hours: Dict[str, BusinessHours] = {}
        self.time_zones: Dict[str, TimeZoneInfo] = {}
        
        # Time calculations cache
        self.calculation_cache: Dict[str, TimeCalculation] = {}
        self.calculation_history: deque = deque(maxlen=1000)
        
        # Time formats
        self.time_formats = {
            TimeFormat.ISO_8601: "%Y-%m-%dT%H:%M:%S%z",
            TimeFormat.RFC_3339: "%Y-%m-%dT%H:%M:%S%z",
            TimeFormat.UNIX_TIMESTAMP: "timestamp",
            TimeFormat.HUMAN_READABLE: "%Y-%m-%d %H:%M:%S %Z",
            TimeFormat.CUSTOM: None
        }
        
        # Time unit conversions (to seconds)
        self.time_unit_seconds = {
            TimeUnit.MICROSECOND: 0.000001,
            TimeUnit.MILLISECOND: 0.001,
            TimeUnit.SECOND: 1,
            TimeUnit.MINUTE: 60,
            TimeUnit.HOUR: 3600,
            TimeUnit.DAY: 86400,
            TimeUnit.WEEK: 604800,
            TimeUnit.MONTH: 2629746,  # Average month
            TimeUnit.QUARTER: 7889238,  # Average quarter
            TimeUnit.YEAR: 31556952   # Average year
        }
        
        # Statistics
        self.time_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'total_conversions': 0,
            'successful_conversions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self.time_lock = threading.RLock()
        
        # Initialize common timezones
        self._initialize_common_timezones()
        
        logger.info("Time manager initialized")
    
    def get_current_time(self, timezone_obj: Optional[timezone] = None) -> datetime:
        """Get current time in specified timezone"""
        try:
            tz = timezone_obj or self.default_timezone
            return datetime.now(tz)
            
        except Exception as e:
            logger.error(f"Failed to get current time: {e}")
            return datetime.now(self.default_timezone)
    
    def convert_timezone(self, dt: datetime, target_timezone: timezone) -> TimeConversion:
        """Convert datetime to target timezone"""
        try:
            with self.time_lock:
                conversion = TimeConversion(
                    source_time=dt,
                    source_timezone=dt.tzinfo,
                    target_timezone=target_timezone
                )
                
                # Perform conversion
                if dt.tzinfo is None:
                    # Assume source is in default timezone
                    dt = dt.replace(tzinfo=self.default_timezone)
                
                converted = dt.astimezone(target_timezone)
                conversion.converted_time = converted
                conversion.conversion_offset = converted.utcoffset() - dt.utcoffset()
                conversion.success = True
                
                # Update statistics
                self.time_stats['total_conversions'] += 1
                self.time_stats['successful_conversions'] += 1
                
                logger.debug(f"Converted time from {dt.tzinfo} to {target_timezone}")
                return conversion
                
        except Exception as e:
            logger.error(f"Failed to convert timezone: {e}")
            conversion.success = False
            conversion.error_message = str(e)
            return conversion
    
    def format_time(self, dt: datetime, format_type: TimeFormat, custom_format: Optional[str] = None) -> str:
        """Format datetime according to specified format"""
        try:
            if format_type == TimeFormat.UNIX_TIMESTAMP:
                return str(int(dt.timestamp()))
            elif format_type == TimeFormat.CUSTOM and custom_format:
                return dt.strftime(custom_format)
            else:
                format_string = self.time_formats.get(format_type)
                if format_string:
                    return dt.strftime(format_string)
                else:
                    return dt.isoformat()
                    
        except Exception as e:
            logger.error(f"Failed to format time: {e}")
            return dt.isoformat()
    
    def parse_time(self, time_string: str, format_type: TimeFormat = TimeFormat.ISO_8601, 
                   custom_format: Optional[str] = None, timezone_obj: Optional[timezone] = None) -> Optional[datetime]:
        """Parse time string to datetime"""
        try:
            if format_type == TimeFormat.UNIX_TIMESTAMP:
                timestamp = float(time_string)
                dt = datetime.fromtimestamp(timestamp, tz=timezone_obj or self.default_timezone)
                return dt
            elif format_type == TimeFormat.CUSTOM and custom_format:
                dt = datetime.strptime(time_string, custom_format)
                if timezone_obj:
                    dt = dt.replace(tzinfo=timezone_obj)
                return dt
            else:
                # Try common formats
                formats_to_try = [
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%H:%M:%S",
                    "%H:%M"
                ]
                
                for fmt in formats_to_try:
                    try:
                        dt = datetime.strptime(time_string, fmt)
                        if timezone_obj and dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone_obj)
                        return dt
                    except ValueError:
                        continue
                
                # If all formats fail, try dateutil parser
                try:
                    from dateutil import parser
                    dt = parser.parse(time_string)
                    if timezone_obj and dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone_obj)
                    return dt
                except ImportError:
                    logger.warning("dateutil not available for flexible parsing")
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse time: {e}")
            return None
    
    def calculate_duration(self, start_time: datetime, end_time: datetime, 
                          unit: TimeUnit = TimeUnit.SECOND) -> TimeCalculation:
        """Calculate duration between two times"""
        try:
            with self.time_lock:
                calculation = TimeCalculation(
                    operation="duration",
                    operands=[start_time, end_time, unit]
                )
                
                # Calculate duration
                duration = end_time - start_time
                total_seconds = duration.total_seconds()
                
                # Convert to requested unit
                unit_seconds = self.time_unit_seconds[unit]
                result = total_seconds / unit_seconds
                
                calculation.result = result
                calculation.result_type = f"float ({unit.value})"
                calculation.success = True
                
                # Cache and store
                self._store_calculation(calculation)
                
                logger.debug(f"Calculated duration: {result} {unit.value}")
                return calculation
                
        except Exception as e:
            logger.error(f"Failed to calculate duration: {e}")
            calculation.success = False
            calculation.error_message = str(e)
            return calculation
    
    def add_time(self, base_time: datetime, amount: float, unit: TimeUnit) -> TimeCalculation:
        """Add time amount to base time"""
        try:
            with self.time_lock:
                calculation = TimeCalculation(
                    operation="add",
                    operands=[base_time, amount, unit]
                )
                
                # Convert amount to timedelta
                unit_seconds = self.time_unit_seconds[unit]
                total_seconds = amount * unit_seconds
                delta = timedelta(seconds=total_seconds)
                
                # Add to base time
                result = base_time + delta
                
                calculation.result = result
                calculation.result_type = "datetime"
                calculation.success = True
                
                # Cache and store
                self._store_calculation(calculation)
                
                logger.debug(f"Added {amount} {unit.value} to {base_time}")
                return calculation
                
        except Exception as e:
            logger.error(f"Failed to add time: {e}")
            calculation.success = False
            calculation.error_message = str(e)
            return calculation
    
    def subtract_time(self, base_time: datetime, amount: float, unit: TimeUnit) -> TimeCalculation:
        """Subtract time amount from base time"""
        try:
            with self.time_lock:
                calculation = TimeCalculation(
                    operation="subtract",
                    operands=[base_time, amount, unit]
                )
                
                # Convert amount to timedelta
                unit_seconds = self.time_unit_seconds[unit]
                total_seconds = amount * unit_seconds
                delta = timedelta(seconds=total_seconds)
                
                # Subtract from base time
                result = base_time - delta
                
                calculation.result = result
                calculation.result_type = "datetime"
                calculation.success = True
                
                # Cache and store
                self._store_calculation(calculation)
                
                logger.debug(f"Subtracted {amount} {unit.value} from {base_time}")
                return calculation
                
        except Exception as e:
            logger.error(f"Failed to subtract time: {e}")
            calculation.success = False
            calculation.error_message = str(e)
            return calculation
    
    def create_time_range(self, start_time: datetime, end_time: Optional[datetime] = None, 
                         duration: Optional[timedelta] = None, **kwargs) -> TimeRange:
        """Create a time range"""
        try:
            time_range = TimeRange(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                **kwargs
            )
            
            logger.debug(f"Created time range: {time_range.start_time} to {time_range.end_time}")
            return time_range
            
        except Exception as e:
            logger.error(f"Failed to create time range: {e}")
            return TimeRange(start_time=start_time)
    
    def check_time_overlap(self, range1: TimeRange, range2: TimeRange) -> bool:
        """Check if two time ranges overlap"""
        try:
            if not range1.end_time or not range2.end_time:
                return False
            
            return (range1.start_time < range2.end_time and 
                   range2.start_time < range1.end_time)
                   
        except Exception as e:
            logger.error(f"Failed to check time overlap: {e}")
            return False
    
    def get_time_intersection(self, range1: TimeRange, range2: TimeRange) -> Optional[TimeRange]:
        """Get intersection of two time ranges"""
        try:
            if not self.check_time_overlap(range1, range2):
                return None
            
            start_time = max(range1.start_time, range2.start_time)
            end_time = min(range1.end_time, range2.end_time)
            
            return TimeRange(
                start_time=start_time,
                end_time=end_time,
                name=f"Intersection of {range1.name} and {range2.name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get time intersection: {e}")
            return None
    
    def add_business_hours(self, business_hours: BusinessHours) -> bool:
        """Add business hours configuration"""
        try:
            with self.time_lock:
                self.business_hours[business_hours.id] = business_hours
                logger.debug(f"Added business hours: {business_hours.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add business hours: {e}")
            return False
    
    def is_business_time(self, dt: datetime, business_hours_id: str) -> bool:
        """Check if datetime falls within business hours"""
        try:
            if business_hours_id not in self.business_hours:
                return False
            
            bh = self.business_hours[business_hours_id]
            
            # Convert to business hours timezone
            if dt.tzinfo != bh.timezone:
                dt = dt.astimezone(bh.timezone)
            
            # Check if date is closed
            if dt.date() in bh.closed_dates:
                return False
            
            # Check if date is a holiday
            if (bh.holiday_rule == BusinessDayRule.EXCLUDE_HOLIDAYS and 
                dt.date() in bh.holidays):
                return False
            
            # Check special hours
            if dt.date() in bh.special_hours:
                hours = bh.special_hours[dt.date()]
                return self._is_time_in_hours(dt.time(), hours)
            
            # Check regular weekly schedule
            weekday = dt.weekday()  # 0=Monday, 6=Sunday
            if weekday in bh.weekly_schedule:
                hours = bh.weekly_schedule[weekday]
                return self._is_time_in_hours(dt.time(), hours)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check business time: {e}")
            return False
    
    def get_next_business_time(self, dt: datetime, business_hours_id: str) -> Optional[datetime]:
        """Get next business time after given datetime"""
        try:
            if business_hours_id not in self.business_hours:
                return None
            
            bh = self.business_hours[business_hours_id]
            
            # Convert to business hours timezone
            if dt.tzinfo != bh.timezone:
                dt = dt.astimezone(bh.timezone)
            
            # Start from next minute
            current = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # Search for next business time (limit search to 30 days)
            for _ in range(30 * 24 * 60):  # 30 days in minutes
                if self.is_business_time(current, business_hours_id):
                    return current
                current += timedelta(minutes=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next business time: {e}")
            return None
    
    def calculate_business_duration(self, start_time: datetime, end_time: datetime, 
                                  business_hours_id: str, unit: TimeUnit = TimeUnit.HOUR) -> TimeCalculation:
        """Calculate duration considering only business hours"""
        try:
            with self.time_lock:
                calculation = TimeCalculation(
                    operation="business_duration",
                    operands=[start_time, end_time, business_hours_id, unit]
                )
                
                if business_hours_id not in self.business_hours:
                    calculation.success = False
                    calculation.error_message = "Business hours not found"
                    return calculation
                
                bh = self.business_hours[business_hours_id]
                
                # Convert times to business hours timezone
                if start_time.tzinfo != bh.timezone:
                    start_time = start_time.astimezone(bh.timezone)
                if end_time.tzinfo != bh.timezone:
                    end_time = end_time.astimezone(bh.timezone)
                
                # Calculate business duration
                total_minutes = 0
                current = start_time.replace(second=0, microsecond=0)
                
                while current < end_time:
                    if self.is_business_time(current, business_hours_id):
                        total_minutes += 1
                    current += timedelta(minutes=1)
                
                # Convert to requested unit
                total_seconds = total_minutes * 60
                unit_seconds = self.time_unit_seconds[unit]
                result = total_seconds / unit_seconds
                
                calculation.result = result
                calculation.result_type = f"float ({unit.value})"
                calculation.success = True
                
                # Cache and store
                self._store_calculation(calculation)
                
                logger.debug(f"Calculated business duration: {result} {unit.value}")
                return calculation
                
        except Exception as e:
            logger.error(f"Failed to calculate business duration: {e}")
            calculation.success = False
            calculation.error_message = str(e)
            return calculation
    
    def get_timezone_info(self, timezone_name: str) -> Optional[TimeZoneInfo]:
        """Get timezone information"""
        try:
            if timezone_name in self.time_zones:
                return self.time_zones[timezone_name]
            
            # Try to create timezone info from pytz
            try:
                tz = pytz.timezone(timezone_name)
                now = datetime.now(tz)
                
                info = TimeZoneInfo(
                    name=timezone_name,
                    timezone_type=TimeZoneType.NAMED,
                    utc_offset=now.utcoffset(),
                    is_dst_active=bool(now.dst()),
                    timezone_obj=tz
                )
                
                if now.dst():
                    info.dst_offset = now.dst()
                
                # Cache the info
                self.time_zones[timezone_name] = info
                return info
                
            except pytz.UnknownTimeZoneError:
                logger.warning(f"Unknown timezone: {timezone_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get timezone info: {e}")
            return None
    
    def get_time_statistics(self) -> Dict[str, Any]:
        """Get time management statistics"""
        try:
            with self.time_lock:
                stats = dict(self.time_stats)
                
                # Add current state
                stats.update({
                    'business_hours_count': len(self.business_hours),
                    'cached_timezones': len(self.time_zones),
                    'calculation_cache_size': len(self.calculation_cache),
                    'calculation_history_size': len(self.calculation_history)
                })
                
                # Add success rates
                if stats['total_calculations'] > 0:
                    stats['calculation_success_rate'] = (
                        stats['successful_calculations'] / stats['total_calculations']
                    )
                
                if stats['total_conversions'] > 0:
                    stats['conversion_success_rate'] = (
                        stats['successful_conversions'] / stats['total_conversions']
                    )
                
                # Add cache hit rate
                total_cache_requests = stats['cache_hits'] + stats['cache_misses']
                if total_cache_requests > 0:
                    stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get time statistics: {e}")
            return {}
    
    # Internal methods
    def _initialize_common_timezones(self):
        """Initialize common timezone information"""
        try:
            common_timezones = [
                'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific',
                'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Rome',
                'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Kolkata', 'Australia/Sydney'
            ]
            
            for tz_name in common_timezones:
                self.get_timezone_info(tz_name)
                
        except Exception as e:
            logger.error(f"Failed to initialize common timezones: {e}")
    
    def _is_time_in_hours(self, time_obj: time, hours: List[Tuple[time, time]]) -> bool:
        """Check if time falls within any of the hour ranges"""
        try:
            for start_time, end_time in hours:
                if start_time <= time_obj <= end_time:
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to check time in hours: {e}")
            return False
    
    def _store_calculation(self, calculation: TimeCalculation):
        """Store calculation in cache and history"""
        try:
            # Generate cache key
            cache_key = f"{calculation.operation}_{hash(str(calculation.operands))}"
            
            # Store in cache
            self.calculation_cache[cache_key] = calculation
            
            # Store in history
            self.calculation_history.append(calculation)
            
            # Update statistics
            self.time_stats['total_calculations'] += 1
            if calculation.success:
                self.time_stats['successful_calculations'] += 1
                
        except Exception as e:
            logger.error(f"Failed to store calculation: {e}")
    
    def _get_cached_calculation(self, operation: str, operands: List[Any]) -> Optional[TimeCalculation]:
        """Get cached calculation if available"""
        try:
            cache_key = f"{operation}_{hash(str(operands))}"
            
            if cache_key in self.calculation_cache:
                self.time_stats['cache_hits'] += 1
                return self.calculation_cache[cache_key]
            else:
                self.time_stats['cache_misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached calculation: {e}")
            self.time_stats['cache_misses'] += 1
            return None