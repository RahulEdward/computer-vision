"""
Cache Manager for Automation Workflow Optimization

This module provides intelligent caching capabilities to optimize automation
workflows by storing and reusing computation results, patterns, and data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Callable, Union, Generic, TypeVar
from datetime import datetime, timedelta
import uuid
import logging
import hashlib
import pickle
import threading
import weakref
from collections import OrderedDict, defaultdict
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheType(Enum):
    """Types of cache storage"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Cache strategies for different scenarios"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"
    REFRESH_AHEAD = "refresh_ahead"


class CacheStatus(Enum):
    """Cache entry status"""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class CacheKey:
    """Cache key with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Key components
    namespace: str = ""
    primary_key: str = ""
    secondary_keys: List[str] = field(default_factory=list)
    
    # Key metadata
    hash_value: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # Key properties
    case_sensitive: bool = True
    include_version: bool = False
    version: str = "1.0"
    
    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash for the cache key"""
        key_parts = [self.namespace, self.primary_key] + self.secondary_keys
        
        if not self.case_sensitive:
            key_parts = [part.lower() for part in key_parts]
        
        if self.include_version:
            key_parts.append(self.version)
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def __str__(self) -> str:
        return f"{self.namespace}:{self.primary_key}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CacheKey):
            return False
        return self.hash_value == other.hash_value
    
    def __hash__(self) -> int:
        return hash(self.hash_value)


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata and value"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Entry identification
    key: CacheKey = field(default_factory=CacheKey)
    
    # Entry data
    value: Optional[T] = None
    serialized_value: Optional[bytes] = None
    
    # Entry metadata
    size: int = 0  # Size in bytes
    priority: int = 0  # Higher = more important
    tags: List[str] = field(default_factory=list)
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl: Optional[timedelta] = None
    
    # Usage statistics
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    # Entry status
    status: CacheStatus = CacheStatus.VALID
    error_message: str = ""
    
    # Dependency tracking
    dependencies: List[str] = field(default_factory=list)  # Other cache keys
    dependents: List[str] = field(default_factory=list)    # Keys that depend on this
    
    # Validation
    checksum: str = ""
    validator: Optional[Callable[[T], bool]] = None
    
    def __post_init__(self):
        if self.ttl and not self.expires_at:
            self.expires_at = self.created_at + self.ttl
        
        if self.value is not None and not self.serialized_value:
            self.serialized_value = self._serialize_value()
            self.size = len(self.serialized_value)
            self.checksum = self._calculate_checksum()
    
    def _serialize_value(self) -> bytes:
        """Serialize the cache value"""
        try:
            return pickle.dumps(self.value)
        except Exception as e:
            logger.error(f"Failed to serialize cache value: {e}")
            return b""
    
    def _deserialize_value(self) -> Optional[T]:
        """Deserialize the cache value"""
        try:
            if self.serialized_value:
                return pickle.loads(self.serialized_value)
            return None
        except Exception as e:
            logger.error(f"Failed to deserialize cache value: {e}")
            return None
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        if self.serialized_value:
            return hashlib.md5(self.serialized_value).hexdigest()
        return ""
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def is_valid(self) -> bool:
        """Check if cache entry is valid"""
        if self.status != CacheStatus.VALID:
            return False
        
        if self.is_expired():
            return False
        
        # Validate checksum
        if self.checksum and self.serialized_value:
            current_checksum = hashlib.md5(self.serialized_value).hexdigest()
            if current_checksum != self.checksum:
                return False
        
        # Custom validation
        if self.validator and self.value is not None:
            try:
                return self.validator(self.value)
            except Exception:
                return False
        
        return True
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_value(self) -> Optional[T]:
        """Get the cache value, deserializing if necessary"""
        if self.value is not None:
            return self.value
        
        if self.serialized_value:
            self.value = self._deserialize_value()
            return self.value
        
        return None


@dataclass
class CacheConfiguration:
    """Cache configuration settings"""
    # Basic settings
    name: str = "default"
    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000  # Maximum number of entries
    max_memory: int = 100 * 1024 * 1024  # 100MB in bytes
    
    # Eviction settings
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    eviction_threshold: float = 0.8  # Evict when 80% full
    
    # TTL settings
    default_ttl: Optional[timedelta] = None
    max_ttl: Optional[timedelta] = None
    
    # Strategy settings
    cache_strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    
    # Performance settings
    enable_compression: bool = False
    enable_encryption: bool = False
    enable_statistics: bool = True
    
    # Persistence settings
    persist_to_disk: bool = False
    disk_path: str = ""
    auto_save_interval: timedelta = timedelta(minutes=5)
    
    # Concurrency settings
    thread_safe: bool = True
    max_concurrent_operations: int = 100
    
    # Validation settings
    enable_integrity_checks: bool = True
    enable_dependency_tracking: bool = True


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    # Basic metrics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Performance metrics
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    average_response_time: timedelta = timedelta(0)
    
    # Size metrics
    current_size: int = 0
    current_memory_usage: int = 0
    max_size_reached: int = 0
    
    # Operation metrics
    total_puts: int = 0
    total_gets: int = 0
    total_deletes: int = 0
    total_evictions: int = 0
    
    # Error metrics
    total_errors: int = 0
    serialization_errors: int = 0
    validation_errors: int = 0
    
    # Timing metrics
    last_reset: datetime = field(default_factory=datetime.now)
    uptime: timedelta = timedelta(0)
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
            self.miss_rate = self.cache_misses / self.total_requests
        
        self.uptime = datetime.now() - self.last_reset


@dataclass
class CacheEvent:
    """Cache event for monitoring and debugging"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Event identification
    event_type: str = ""  # hit, miss, put, delete, evict, expire
    cache_name: str = ""
    
    # Event data
    key: Optional[CacheKey] = None
    entry_id: str = ""
    
    # Event timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta = timedelta(0)
    
    # Event context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event result
    success: bool = True
    error_message: str = ""


class CacheManager:
    """Intelligent cache manager for automation workflows"""
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        self.config = config or CacheConfiguration()
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}  # hash -> entry
        self.key_index: Dict[CacheKey, str] = {}  # key -> hash
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> hashes
        
        # Access tracking for LRU/LFU
        self.access_order = OrderedDict()  # hash -> timestamp
        self.access_frequency: Dict[str, int] = defaultdict(int)
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics and monitoring
        self.statistics = CacheStatistics()
        self.events: List[CacheEvent] = []
        self.event_handlers: List[Callable[[CacheEvent], None]] = []
        
        # Thread safety
        self.lock = threading.RLock() if self.config.thread_safe else None
        
        # Background tasks
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_active = False
        
        # Start background cleanup if needed
        if self.config.default_ttl or self.config.eviction_policy == EvictionPolicy.TTL:
            self.start_cleanup_worker()
        
        logger.info(f"Cache manager initialized: {self.config.name}")
    
    def put(self, key: CacheKey, value: Any, ttl: Optional[timedelta] = None, 
            priority: int = 0, tags: Optional[List[str]] = None) -> bool:
        """Store a value in the cache"""
        try:
            start_time = datetime.now()
            
            with self._get_lock():
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl or self.config.default_ttl,
                    priority=priority,
                    tags=tags or []
                )
                
                # Check if we need to evict entries
                if self._should_evict():
                    self._evict_entries()
                
                # Store entry
                hash_key = key.hash_value
                self.entries[hash_key] = entry
                self.key_index[key] = hash_key
                
                # Update indexes
                self._update_access_tracking(hash_key)
                self._update_tag_index(entry)
                
                # Update statistics
                self.statistics.total_puts += 1
                self.statistics.current_size += 1
                self.statistics.current_memory_usage += entry.size
                
                # Record event
                duration = datetime.now() - start_time
                self._record_event("put", key, entry.id, duration, True)
                
                logger.debug(f"Cached value for key: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to put cache entry: {e}")
            self._record_event("put", key, "", datetime.now() - start_time, False, str(e))
            return False
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve a value from the cache"""
        try:
            start_time = datetime.now()
            
            with self._get_lock():
                hash_key = key.hash_value
                
                # Check if entry exists
                if hash_key not in self.entries:
                    self.statistics.cache_misses += 1
                    self.statistics.total_requests += 1
                    self._record_event("miss", key, "", datetime.now() - start_time, True)
                    return None
                
                entry = self.entries[hash_key]
                
                # Check if entry is valid
                if not entry.is_valid():
                    # Remove invalid entry
                    self._remove_entry(hash_key)
                    self.statistics.cache_misses += 1
                    self.statistics.total_requests += 1
                    self._record_event("miss", key, entry.id, datetime.now() - start_time, True)
                    return None
                
                # Update access tracking
                entry.touch()
                self._update_access_tracking(hash_key)
                
                # Update statistics
                self.statistics.cache_hits += 1
                self.statistics.total_requests += 1
                entry.hit_count += 1
                
                # Record event
                duration = datetime.now() - start_time
                self._record_event("hit", key, entry.id, duration, True)
                
                value = entry.get_value()
                logger.debug(f"Cache hit for key: {key}")
                return value
                
        except Exception as e:
            logger.error(f"Failed to get cache entry: {e}")
            self._record_event("get", key, "", datetime.now() - start_time, False, str(e))
            return None
    
    def delete(self, key: CacheKey) -> bool:
        """Delete a value from the cache"""
        try:
            start_time = datetime.now()
            
            with self._get_lock():
                hash_key = key.hash_value
                
                if hash_key not in self.entries:
                    return False
                
                entry = self.entries[hash_key]
                self._remove_entry(hash_key)
                
                # Update statistics
                self.statistics.total_deletes += 1
                
                # Record event
                duration = datetime.now() - start_time
                self._record_event("delete", key, entry.id, duration, True)
                
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
            self._record_event("delete", key, "", datetime.now() - start_time, False, str(e))
            return False
    
    def exists(self, key: CacheKey) -> bool:
        """Check if a key exists in the cache"""
        with self._get_lock():
            hash_key = key.hash_value
            
            if hash_key not in self.entries:
                return False
            
            entry = self.entries[hash_key]
            return entry.is_valid()
    
    def clear(self) -> bool:
        """Clear all entries from the cache"""
        try:
            with self._get_lock():
                self.entries.clear()
                self.key_index.clear()
                self.tag_index.clear()
                self.access_order.clear()
                self.access_frequency.clear()
                self.dependency_graph.clear()
                
                # Reset statistics
                self.statistics.current_size = 0
                self.statistics.current_memory_usage = 0
                
                logger.info("Cache cleared")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        try:
            with self._get_lock():
                if tag not in self.tag_index:
                    return 0
                
                hash_keys = list(self.tag_index[tag])
                count = 0
                
                for hash_key in hash_keys:
                    if hash_key in self.entries:
                        self._remove_entry(hash_key)
                        count += 1
                
                logger.info(f"Invalidated {count} entries with tag: {tag}")
                return count
                
        except Exception as e:
            logger.error(f"Failed to invalidate by tag: {e}")
            return 0
    
    def invalidate_dependencies(self, key: CacheKey) -> int:
        """Invalidate all entries that depend on the given key"""
        try:
            with self._get_lock():
                hash_key = key.hash_value
                
                if hash_key not in self.dependency_graph:
                    return 0
                
                dependent_keys = list(self.dependency_graph[hash_key])
                count = 0
                
                for dep_key in dependent_keys:
                    if dep_key in self.entries:
                        self._remove_entry(dep_key)
                        count += 1
                
                logger.info(f"Invalidated {count} dependent entries")
                return count
                
        except Exception as e:
            logger.error(f"Failed to invalidate dependencies: {e}")
            return 0
    
    def get_by_pattern(self, pattern: str) -> List[Tuple[CacheKey, Any]]:
        """Get all entries matching a key pattern"""
        results = []
        
        try:
            with self._get_lock():
                for key, hash_key in self.key_index.items():
                    if self._matches_pattern(str(key), pattern):
                        entry = self.entries.get(hash_key)
                        if entry and entry.is_valid():
                            value = entry.get_value()
                            results.append((key, value))
                
        except Exception as e:
            logger.error(f"Failed to get by pattern: {e}")
        
        return results
    
    def get_by_tag(self, tag: str) -> List[Tuple[CacheKey, Any]]:
        """Get all entries with a specific tag"""
        results = []
        
        try:
            with self._get_lock():
                if tag in self.tag_index:
                    for hash_key in self.tag_index[tag]:
                        entry = self.entries.get(hash_key)
                        if entry and entry.is_valid():
                            # Find the key for this hash
                            key = next((k for k, h in self.key_index.items() if h == hash_key), None)
                            if key:
                                value = entry.get_value()
                                results.append((key, value))
                
        except Exception as e:
            logger.error(f"Failed to get by tag: {e}")
        
        return results
    
    def set_dependency(self, dependent_key: CacheKey, dependency_key: CacheKey):
        """Set a dependency relationship between cache entries"""
        try:
            with self._get_lock():
                dep_hash = dependency_key.hash_value
                dependent_hash = dependent_key.hash_value
                
                self.dependency_graph[dep_hash].add(dependent_hash)
                
                # Update entry dependency lists
                if dependent_hash in self.entries:
                    self.entries[dependent_hash].dependencies.append(dep_hash)
                
                if dep_hash in self.entries:
                    self.entries[dep_hash].dependents.append(dependent_hash)
                
        except Exception as e:
            logger.error(f"Failed to set dependency: {e}")
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache performance statistics"""
        with self._get_lock():
            self.statistics.calculate_derived_metrics()
            return self.statistics
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        with self._get_lock():
            stats = self.get_statistics()
            
            return {
                'name': self.config.name,
                'type': self.config.cache_type.value,
                'size': stats.current_size,
                'max_size': self.config.max_size,
                'memory_usage': stats.current_memory_usage,
                'max_memory': self.config.max_memory,
                'hit_rate': stats.hit_rate,
                'miss_rate': stats.miss_rate,
                'total_requests': stats.total_requests,
                'eviction_policy': self.config.eviction_policy.value,
                'uptime': stats.uptime,
                'active_tags': len(self.tag_index),
                'dependencies': len(self.dependency_graph)
            }
    
    def add_event_handler(self, handler: Callable[[CacheEvent], None]):
        """Add an event handler for cache events"""
        self.event_handlers.append(handler)
    
    def start_cleanup_worker(self):
        """Start background cleanup worker"""
        if self.cleanup_active:
            return
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self.cleanup_thread.start()
        
        logger.info("Started cache cleanup worker")
    
    def stop_cleanup_worker(self):
        """Stop background cleanup worker"""
        self.cleanup_active = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("Stopped cache cleanup worker")
    
    # Internal methods
    def _get_lock(self):
        """Get thread lock if thread safety is enabled"""
        if self.lock:
            return self.lock
        else:
            # Return a dummy context manager
            class DummyLock:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyLock()
    
    def _should_evict(self) -> bool:
        """Check if cache eviction is needed"""
        size_threshold = self.config.max_size * self.config.eviction_threshold
        memory_threshold = self.config.max_memory * self.config.eviction_threshold
        
        return (self.statistics.current_size >= size_threshold or 
                self.statistics.current_memory_usage >= memory_threshold)
    
    def _evict_entries(self):
        """Evict entries based on eviction policy"""
        try:
            target_size = int(self.config.max_size * 0.7)  # Evict to 70% capacity
            
            if self.config.eviction_policy == EvictionPolicy.LRU:
                self._evict_lru(target_size)
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                self._evict_lfu(target_size)
            elif self.config.eviction_policy == EvictionPolicy.FIFO:
                self._evict_fifo(target_size)
            elif self.config.eviction_policy == EvictionPolicy.TTL:
                self._evict_expired()
            elif self.config.eviction_policy == EvictionPolicy.PRIORITY_BASED:
                self._evict_by_priority(target_size)
            elif self.config.eviction_policy == EvictionPolicy.SIZE_BASED:
                self._evict_by_size(target_size)
            
        except Exception as e:
            logger.error(f"Failed to evict entries: {e}")
    
    def _evict_lru(self, target_size: int):
        """Evict least recently used entries"""
        while self.statistics.current_size > target_size and self.access_order:
            hash_key, _ = self.access_order.popitem(last=False)
            if hash_key in self.entries:
                self._remove_entry(hash_key)
                self.statistics.total_evictions += 1
    
    def _evict_lfu(self, target_size: int):
        """Evict least frequently used entries"""
        # Sort by access frequency
        sorted_entries = sorted(
            self.access_frequency.items(),
            key=lambda x: x[1]
        )
        
        for hash_key, _ in sorted_entries:
            if self.statistics.current_size <= target_size:
                break
            
            if hash_key in self.entries:
                self._remove_entry(hash_key)
                self.statistics.total_evictions += 1
    
    def _evict_fifo(self, target_size: int):
        """Evict first in, first out entries"""
        # Sort by creation time
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].created_at
        )
        
        for hash_key, _ in sorted_entries:
            if self.statistics.current_size <= target_size:
                break
            
            self._remove_entry(hash_key)
            self.statistics.total_evictions += 1
    
    def _evict_expired(self):
        """Evict expired entries"""
        expired_keys = []
        
        for hash_key, entry in self.entries.items():
            if entry.is_expired():
                expired_keys.append(hash_key)
        
        for hash_key in expired_keys:
            self._remove_entry(hash_key)
            self.statistics.total_evictions += 1
    
    def _evict_by_priority(self, target_size: int):
        """Evict entries by priority (lowest first)"""
        # Sort by priority (ascending)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].priority
        )
        
        for hash_key, _ in sorted_entries:
            if self.statistics.current_size <= target_size:
                break
            
            self._remove_entry(hash_key)
            self.statistics.total_evictions += 1
    
    def _evict_by_size(self, target_size: int):
        """Evict largest entries first"""
        # Sort by size (descending)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].size,
            reverse=True
        )
        
        for hash_key, _ in sorted_entries:
            if self.statistics.current_size <= target_size:
                break
            
            self._remove_entry(hash_key)
            self.statistics.total_evictions += 1
    
    def _remove_entry(self, hash_key: str):
        """Remove an entry from all indexes"""
        if hash_key not in self.entries:
            return
        
        entry = self.entries[hash_key]
        
        # Remove from main storage
        del self.entries[hash_key]
        
        # Remove from key index
        key_to_remove = None
        for key, h in self.key_index.items():
            if h == hash_key:
                key_to_remove = key
                break
        
        if key_to_remove:
            del self.key_index[key_to_remove]
        
        # Remove from tag index
        for tag in entry.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(hash_key)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Remove from access tracking
        self.access_order.pop(hash_key, None)
        self.access_frequency.pop(hash_key, None)
        
        # Remove from dependency graph
        self.dependency_graph.pop(hash_key, None)
        for deps in self.dependency_graph.values():
            deps.discard(hash_key)
        
        # Update statistics
        self.statistics.current_size -= 1
        self.statistics.current_memory_usage -= entry.size
    
    def _update_access_tracking(self, hash_key: str):
        """Update access tracking for LRU/LFU policies"""
        # Update LRU tracking
        self.access_order[hash_key] = datetime.now()
        
        # Update LFU tracking
        self.access_frequency[hash_key] += 1
    
    def _update_tag_index(self, entry: CacheEntry):
        """Update tag index with entry tags"""
        hash_key = entry.key.hash_value
        
        for tag in entry.tags:
            self.tag_index[tag].add(hash_key)
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern (simple wildcard matching)"""
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
    
    def _record_event(self, event_type: str, key: Optional[CacheKey], 
                     entry_id: str, duration: timedelta, 
                     success: bool, error_message: str = ""):
        """Record a cache event"""
        try:
            event = CacheEvent(
                event_type=event_type,
                cache_name=self.config.name,
                key=key,
                entry_id=entry_id,
                duration=duration,
                success=success,
                error_message=error_message
            )
            
            # Store event (with size limit)
            self.events.append(event)
            if len(self.events) > 1000:  # Keep last 1000 events
                self.events = self.events[-1000:]
            
            # Notify handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup"""
        try:
            while self.cleanup_active:
                with self._get_lock():
                    # Remove expired entries
                    self._evict_expired()
                    
                    # Check if eviction is needed
                    if self._should_evict():
                        self._evict_entries()
                
                # Sleep for cleanup interval
                import time
                time.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Cleanup worker failed: {e}")
        finally:
            self.cleanup_active = False


# Utility functions for creating cache keys
def create_cache_key(namespace: str, primary_key: str, 
                    secondary_keys: Optional[List[str]] = None,
                    version: str = "1.0") -> CacheKey:
    """Create a cache key with the given parameters"""
    return CacheKey(
        namespace=namespace,
        primary_key=primary_key,
        secondary_keys=secondary_keys or [],
        version=version,
        include_version=True
    )


def create_action_cache_key(action_type: str, action_id: str, 
                           parameters: Optional[Dict[str, Any]] = None) -> CacheKey:
    """Create a cache key for an action"""
    secondary_keys = []
    
    if parameters:
        # Sort parameters for consistent key generation
        sorted_params = sorted(parameters.items())
        param_string = json.dumps(sorted_params, sort_keys=True)
        param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
        secondary_keys.append(param_hash)
    
    return create_cache_key("actions", f"{action_type}:{action_id}", secondary_keys)


def create_pattern_cache_key(pattern_type: str, pattern_signature: str) -> CacheKey:
    """Create a cache key for a pattern"""
    return create_cache_key("patterns", f"{pattern_type}:{pattern_signature}")


def create_result_cache_key(operation: str, inputs: Dict[str, Any]) -> CacheKey:
    """Create a cache key for operation results"""
    # Create deterministic hash from inputs
    input_string = json.dumps(inputs, sort_keys=True)
    input_hash = hashlib.md5(input_string.encode()).hexdigest()[:12]
    
    return create_cache_key("results", operation, [input_hash])