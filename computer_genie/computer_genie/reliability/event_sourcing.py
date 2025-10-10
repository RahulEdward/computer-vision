"""
Event Sourcing for Complete Audit Trail and Time-Travel Debugging

Implements comprehensive event sourcing to capture all system state changes,
enabling complete audit trails, time-travel debugging, and system replay.
"""

import asyncio
import time
import json
import logging
import threading
import sqlite3
import pickle
import gzip
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Type, Iterator
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import uuid
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import weakref

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    USER_ACTION = "user_action"
    API_CALL = "api_call"
    STATE_CHANGE = "state_change"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    CONFIGURATION_CHANGE = "configuration_change"
    DEPLOYMENT_EVENT = "deployment_event"
    HEALTH_CHECK = "health_check"
    RECOVERY_ACTION = "recovery_action"


class EventSeverity(Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    """Base event class for event sourcing"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.STATE_CHANGE
    timestamp: float = field(default_factory=time.time)
    aggregate_id: str = ""
    aggregate_type: str = ""
    event_version: int = 1
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
            
    def _calculate_checksum(self) -> str:
        """Calculate event checksum for integrity verification"""
        event_data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'aggregate_id': self.aggregate_id,
            'aggregate_type': self.aggregate_type,
            'event_version': self.event_version,
            'data': self.data
        }
        
        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()
        
    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        # Convert enum values
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = EventType(data['event_type'])
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = EventSeverity(data['severity'])
            
        return cls(**data)


@dataclass
class Snapshot:
    """Aggregate snapshot for performance optimization"""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str = ""
    aggregate_type: str = ""
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    state: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
            
    def _calculate_checksum(self) -> str:
        """Calculate snapshot checksum"""
        snapshot_data = {
            'aggregate_id': self.aggregate_id,
            'aggregate_type': self.aggregate_type,
            'version': self.version,
            'state': self.state
        }
        
        snapshot_json = json.dumps(snapshot_data, sort_keys=True)
        return hashlib.sha256(snapshot_json.encode()).hexdigest()
        
    def verify_integrity(self) -> bool:
        """Verify snapshot integrity"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum


class EventStore(ABC):
    """Abstract event store interface"""
    
    @abstractmethod
    async def append_event(self, event: Event) -> bool:
        """Append an event to the store"""
        pass
        
    @abstractmethod
    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for an aggregate"""
        pass
        
    @abstractmethod
    async def get_events_by_type(self, event_type: EventType, 
                                from_timestamp: float = 0.0,
                                to_timestamp: Optional[float] = None) -> List[Event]:
        """Get events by type within time range"""
        pass
        
    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> bool:
        """Save aggregate snapshot"""
        pass
        
    @abstractmethod
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for aggregate"""
        pass
        
    @abstractmethod
    async def get_all_events(self, from_timestamp: float = 0.0,
                           to_timestamp: Optional[float] = None,
                           limit: Optional[int] = None) -> List[Event]:
        """Get all events within time range"""
        pass


class SQLiteEventStore(EventStore):
    """SQLite-based event store implementation"""
    
    def __init__(self, db_path: str = "event_store.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    aggregate_type TEXT NOT NULL,
                    event_version INTEGER NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    correlation_id TEXT,
                    causation_id TEXT,
                    severity TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT (julianday('now'))
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    aggregate_id TEXT NOT NULL,
                    aggregate_type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    state TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT (julianday('now'))
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, event_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_aggregate ON snapshots(aggregate_id, version DESC)")
            
            conn.commit()
            
    async def append_event(self, event: Event) -> bool:
        """Append an event to the store"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO events (
                            event_id, event_type, timestamp, aggregate_id, aggregate_type,
                            event_version, user_id, session_id, correlation_id, causation_id,
                            severity, data, metadata, checksum
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp,
                        event.aggregate_id,
                        event.aggregate_type,
                        event.event_version,
                        event.user_id,
                        event.session_id,
                        event.correlation_id,
                        event.causation_id,
                        event.severity.value,
                        json.dumps(event.data),
                        json.dumps(event.metadata),
                        event.checksum
                    ))
                    conn.commit()
                    
            logger.debug(f"Event appended: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return False
            
    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for an aggregate"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM events 
                        WHERE aggregate_id = ? AND event_version >= ?
                        ORDER BY event_version ASC
                    """, (aggregate_id, from_version))
                    
                    events = []
                    for row in cursor.fetchall():
                        event_data = dict(row)
                        event_data['data'] = json.loads(event_data['data'])
                        event_data['metadata'] = json.loads(event_data['metadata'])
                        events.append(Event.from_dict(event_data))
                        
                    return events
                    
        except Exception as e:
            logger.error(f"Failed to get events for aggregate {aggregate_id}: {e}")
            return []
            
    async def get_events_by_type(self, event_type: EventType, 
                                from_timestamp: float = 0.0,
                                to_timestamp: Optional[float] = None) -> List[Event]:
        """Get events by type within time range"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    if to_timestamp is None:
                        to_timestamp = time.time()
                        
                    cursor = conn.execute("""
                        SELECT * FROM events 
                        WHERE event_type = ? AND timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp ASC
                    """, (event_type.value, from_timestamp, to_timestamp))
                    
                    events = []
                    for row in cursor.fetchall():
                        event_data = dict(row)
                        event_data['data'] = json.loads(event_data['data'])
                        event_data['metadata'] = json.loads(event_data['metadata'])
                        events.append(Event.from_dict(event_data))
                        
                    return events
                    
        except Exception as e:
            logger.error(f"Failed to get events by type {event_type}: {e}")
            return []
            
    async def save_snapshot(self, snapshot: Snapshot) -> bool:
        """Save aggregate snapshot"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Delete old snapshots for this aggregate
                    conn.execute("""
                        DELETE FROM snapshots 
                        WHERE aggregate_id = ? AND version < ?
                    """, (snapshot.aggregate_id, snapshot.version))
                    
                    # Insert new snapshot
                    conn.execute("""
                        INSERT OR REPLACE INTO snapshots (
                            snapshot_id, aggregate_id, aggregate_type, version,
                            timestamp, state, checksum
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot.snapshot_id,
                        snapshot.aggregate_id,
                        snapshot.aggregate_type,
                        snapshot.version,
                        snapshot.timestamp,
                        json.dumps(snapshot.state),
                        snapshot.checksum
                    ))
                    conn.commit()
                    
            logger.debug(f"Snapshot saved: {snapshot.snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
            
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for aggregate"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT * FROM snapshots 
                        WHERE aggregate_id = ?
                        ORDER BY version DESC
                        LIMIT 1
                    """, (aggregate_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        snapshot_data = dict(row)
                        snapshot_data['state'] = json.loads(snapshot_data['state'])
                        return Snapshot(**snapshot_data)
                        
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get snapshot for aggregate {aggregate_id}: {e}")
            return None
            
    async def get_all_events(self, from_timestamp: float = 0.0,
                           to_timestamp: Optional[float] = None,
                           limit: Optional[int] = None) -> List[Event]:
        """Get all events within time range"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    if to_timestamp is None:
                        to_timestamp = time.time()
                        
                    query = """
                        SELECT * FROM events 
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp ASC
                    """
                    params = [from_timestamp, to_timestamp]
                    
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                        
                    cursor = conn.execute(query, params)
                    
                    events = []
                    for row in cursor.fetchall():
                        event_data = dict(row)
                        event_data['data'] = json.loads(event_data['data'])
                        event_data['metadata'] = json.loads(event_data['metadata'])
                        events.append(Event.from_dict(event_data))
                        
                    return events
                    
        except Exception as e:
            logger.error(f"Failed to get all events: {e}")
            return []


class EventProjection(ABC):
    """Abstract base class for event projections"""
    
    @abstractmethod
    async def handle_event(self, event: Event):
        """Handle an event and update projection"""
        pass
        
    @abstractmethod
    async def rebuild(self, events: List[Event]):
        """Rebuild projection from events"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current projection state"""
        pass


class AggregateProjection(EventProjection):
    """Projection that maintains aggregate state"""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.state: Dict[str, Any] = {}
        self.version = 0
        self.last_updated = time.time()
        
    async def handle_event(self, event: Event):
        """Handle an event and update state"""
        if event.aggregate_id != self.aggregate_id:
            return
            
        # Apply event to state
        self._apply_event(event)
        self.version = event.event_version
        self.last_updated = time.time()
        
    def _apply_event(self, event: Event):
        """Apply event to aggregate state"""
        # Default implementation - merge event data into state
        if event.data:
            self.state.update(event.data)
            
    async def rebuild(self, events: List[Event]):
        """Rebuild projection from events"""
        self.state = {}
        self.version = 0
        
        for event in events:
            if event.aggregate_id == self.aggregate_id:
                self._apply_event(event)
                self.version = max(self.version, event.event_version)
                
        self.last_updated = time.time()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current projection state"""
        return {
            'aggregate_id': self.aggregate_id,
            'state': self.state.copy(),
            'version': self.version,
            'last_updated': self.last_updated
        }


class SystemMetricsProjection(EventProjection):
    """Projection for system metrics and statistics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'errors_count': 0,
            'last_error': None,
            'performance_metrics': [],
            'uptime_start': time.time()
        }
        
    async def handle_event(self, event: Event):
        """Handle an event and update metrics"""
        self.metrics['total_events'] += 1
        self.metrics['events_by_type'][event.event_type.value] += 1
        self.metrics['events_by_severity'][event.severity.value] += 1
        
        if event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
            self.metrics['errors_count'] += 1
            self.metrics['last_error'] = {
                'timestamp': event.timestamp,
                'event_id': event.event_id,
                'message': event.data.get('message', 'Unknown error')
            }
            
        if event.event_type == EventType.PERFORMANCE_METRIC:
            self.metrics['performance_metrics'].append({
                'timestamp': event.timestamp,
                'metric': event.data
            })
            
            # Keep only last 1000 performance metrics
            if len(self.metrics['performance_metrics']) > 1000:
                self.metrics['performance_metrics'] = self.metrics['performance_metrics'][-1000:]
                
    async def rebuild(self, events: List[Event]):
        """Rebuild projection from events"""
        self.metrics = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'errors_count': 0,
            'last_error': None,
            'performance_metrics': [],
            'uptime_start': time.time()
        }
        
        for event in events:
            await self.handle_event(event)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current projection state"""
        current_time = time.time()
        uptime = current_time - self.metrics['uptime_start']
        
        return {
            'total_events': self.metrics['total_events'],
            'events_by_type': dict(self.metrics['events_by_type']),
            'events_by_severity': dict(self.metrics['events_by_severity']),
            'errors_count': self.metrics['errors_count'],
            'last_error': self.metrics['last_error'],
            'uptime_seconds': uptime,
            'events_per_second': self.metrics['total_events'] / uptime if uptime > 0 else 0,
            'recent_performance_metrics': self.metrics['performance_metrics'][-10:]  # Last 10 metrics
        }


class EventSourcingManager:
    """Manages event sourcing operations"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.projections: Dict[str, EventProjection] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.aggregate_cache: Dict[str, AggregateProjection] = {}
        self.snapshot_frequency = 100  # Create snapshot every 100 events
        self._lock = threading.Lock()
        
        # Built-in projections
        self.system_metrics = SystemMetricsProjection()
        self.register_projection("system_metrics", self.system_metrics)
        
    async def publish_event(self, event: Event) -> bool:
        """Publish an event to the event store"""
        # Verify event integrity
        if not event.verify_integrity():
            logger.error(f"Event integrity check failed: {event.event_id}")
            return False
            
        # Store event
        success = await self.event_store.append_event(event)
        
        if success:
            # Update projections
            await self._update_projections(event)
            
            # Call event handlers
            await self._call_event_handlers(event)
            
            # Check if snapshot is needed
            await self._check_snapshot_needed(event)
            
            logger.debug(f"Event published: {event.event_id}")
            
        return success
        
    async def _update_projections(self, event: Event):
        """Update all projections with the new event"""
        for projection in self.projections.values():
            try:
                await projection.handle_event(event)
            except Exception as e:
                logger.error(f"Error updating projection: {e}")
                
    async def _call_event_handlers(self, event: Event):
        """Call registered event handlers"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                
    async def _check_snapshot_needed(self, event: Event):
        """Check if a snapshot should be created"""
        if event.aggregate_id and event.event_version % self.snapshot_frequency == 0:
            await self.create_snapshot(event.aggregate_id)
            
    async def create_snapshot(self, aggregate_id: str) -> bool:
        """Create a snapshot for an aggregate"""
        try:
            # Get aggregate projection
            projection = await self.get_aggregate_projection(aggregate_id)
            
            if projection:
                snapshot = Snapshot(
                    aggregate_id=aggregate_id,
                    aggregate_type="generic",  # Could be determined from events
                    version=projection.version,
                    state=projection.state
                )
                
                return await self.event_store.save_snapshot(snapshot)
                
        except Exception as e:
            logger.error(f"Failed to create snapshot for {aggregate_id}: {e}")
            
        return False
        
    async def get_aggregate_projection(self, aggregate_id: str) -> Optional[AggregateProjection]:
        """Get or create aggregate projection"""
        with self._lock:
            if aggregate_id in self.aggregate_cache:
                return self.aggregate_cache[aggregate_id]
                
        # Create new projection
        projection = AggregateProjection(aggregate_id)
        
        # Try to load from snapshot
        snapshot = await self.event_store.get_snapshot(aggregate_id)
        if snapshot and snapshot.verify_integrity():
            projection.state = snapshot.state.copy()
            projection.version = snapshot.version
            from_version = snapshot.version + 1
        else:
            from_version = 0
            
        # Load events since snapshot
        events = await self.event_store.get_events(aggregate_id, from_version)
        
        for event in events:
            await projection.handle_event(event)
            
        with self._lock:
            self.aggregate_cache[aggregate_id] = projection
            
        return projection
        
    def register_projection(self, name: str, projection: EventProjection):
        """Register a projection"""
        self.projections[name] = projection
        logger.info(f"Registered projection: {name}")
        
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered event handler for: {event_type.value}")
        
    async def rebuild_projection(self, projection_name: str, 
                               from_timestamp: float = 0.0) -> bool:
        """Rebuild a projection from events"""
        projection = self.projections.get(projection_name)
        if not projection:
            logger.error(f"Projection not found: {projection_name}")
            return False
            
        try:
            events = await self.event_store.get_all_events(from_timestamp)
            await projection.rebuild(events)
            logger.info(f"Rebuilt projection: {projection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild projection {projection_name}: {e}")
            return False
            
    async def time_travel_debug(self, aggregate_id: str, 
                              target_timestamp: float) -> Optional[Dict[str, Any]]:
        """Debug aggregate state at a specific point in time"""
        try:
            # Get all events for the aggregate up to the target time
            all_events = await self.event_store.get_events(aggregate_id)
            target_events = [e for e in all_events if e.timestamp <= target_timestamp]
            
            if not target_events:
                return None
                
            # Rebuild state up to target time
            projection = AggregateProjection(aggregate_id)
            await projection.rebuild(target_events)
            
            return {
                'aggregate_id': aggregate_id,
                'target_timestamp': target_timestamp,
                'target_datetime': datetime.fromtimestamp(target_timestamp, timezone.utc).isoformat(),
                'state': projection.state,
                'version': projection.version,
                'events_count': len(target_events),
                'last_event': target_events[-1].to_dict() if target_events else None
            }
            
        except Exception as e:
            logger.error(f"Time travel debug failed for {aggregate_id}: {e}")
            return None
            
    async def get_audit_trail(self, aggregate_id: str, 
                            from_timestamp: float = 0.0,
                            to_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get complete audit trail for an aggregate"""
        try:
            events = await self.event_store.get_events(aggregate_id)
            
            if to_timestamp is None:
                to_timestamp = time.time()
                
            filtered_events = [
                e for e in events 
                if from_timestamp <= e.timestamp <= to_timestamp
            ]
            
            audit_trail = []
            for event in filtered_events:
                audit_entry = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'datetime': datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'severity': event.severity.value,
                    'data': event.data,
                    'metadata': event.metadata
                }
                audit_trail.append(audit_entry)
                
            return audit_trail
            
        except Exception as e:
            logger.error(f"Failed to get audit trail for {aggregate_id}: {e}")
            return []
            
    async def replay_events(self, from_timestamp: float, 
                          to_timestamp: Optional[float] = None,
                          event_types: Optional[List[EventType]] = None) -> Dict[str, Any]:
        """Replay events for analysis"""
        try:
            if to_timestamp is None:
                to_timestamp = time.time()
                
            events = await self.event_store.get_all_events(from_timestamp, to_timestamp)
            
            if event_types:
                events = [e for e in events if e.event_type in event_types]
                
            # Analyze events
            analysis = {
                'total_events': len(events),
                'time_range': {
                    'from': from_timestamp,
                    'to': to_timestamp,
                    'duration_seconds': to_timestamp - from_timestamp
                },
                'events_by_type': defaultdict(int),
                'events_by_severity': defaultdict(int),
                'unique_aggregates': set(),
                'unique_users': set(),
                'timeline': []
            }
            
            for event in events:
                analysis['events_by_type'][event.event_type.value] += 1
                analysis['events_by_severity'][event.severity.value] += 1
                analysis['unique_aggregates'].add(event.aggregate_id)
                if event.user_id:
                    analysis['unique_users'].add(event.user_id)
                    
                analysis['timeline'].append({
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'aggregate_id': event.aggregate_id,
                    'user_id': event.user_id
                })
                
            # Convert sets to lists for JSON serialization
            analysis['unique_aggregates'] = list(analysis['unique_aggregates'])
            analysis['unique_users'] = list(analysis['unique_users'])
            analysis['events_by_type'] = dict(analysis['events_by_type'])
            analysis['events_by_severity'] = dict(analysis['events_by_severity'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            return {}
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall event sourcing system status"""
        return {
            'projections_count': len(self.projections),
            'cached_aggregates': len(self.aggregate_cache),
            'event_handlers': {
                event_type.value: len(handlers) 
                for event_type, handlers in self.event_handlers.items()
            },
            'system_metrics': self.system_metrics.get_state(),
            'snapshot_frequency': self.snapshot_frequency
        }


# Utility functions for creating common events
def create_user_action_event(user_id: str, action: str, data: Dict[str, Any],
                           aggregate_id: str = "", session_id: Optional[str] = None) -> Event:
    """Create a user action event"""
    return Event(
        event_type=EventType.USER_ACTION,
        aggregate_id=aggregate_id,
        aggregate_type="user_session",
        user_id=user_id,
        session_id=session_id,
        data={'action': action, **data},
        metadata={'source': 'user_interface'}
    )


def create_error_event(error_message: str, error_type: str, 
                      aggregate_id: str = "", severity: EventSeverity = EventSeverity.ERROR) -> Event:
    """Create an error event"""
    return Event(
        event_type=EventType.ERROR_OCCURRED,
        aggregate_id=aggregate_id,
        severity=severity,
        data={
            'error_message': error_message,
            'error_type': error_type
        },
        metadata={'source': 'system'}
    )


def create_performance_event(metric_name: str, value: float, unit: str,
                           aggregate_id: str = "") -> Event:
    """Create a performance metric event"""
    return Event(
        event_type=EventType.PERFORMANCE_METRIC,
        aggregate_id=aggregate_id,
        data={
            'metric_name': metric_name,
            'value': value,
            'unit': unit
        },
        metadata={'source': 'performance_monitor'}
    )