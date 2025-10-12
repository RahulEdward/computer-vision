"""
Local Storage System for User Preferences and Model Adaptations
उपयोगकर्ता प्राथमिकताओं और मॉडल अनुकूलन के लिए स्थानीय भंडारण प्रणाली

Comprehensive local storage system for managing user preferences, model adaptations,
language-specific customizations, and offline data persistence.
"""

import os
import json
import sqlite3
import pickle
import hashlib
import shutil
import threading
import time
import gzip
import lzma
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import torch

# Optional imports for advanced storage
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class StorageType(Enum):
    """Storage backend types"""
    JSON = "json"
    SQLITE = "sqlite"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    LMDB = "lmdb"
    BINARY = "binary"


class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZSTD = "zstd"


class DataCategory(Enum):
    """Data categories for organization"""
    USER_PREFERENCES = "user_preferences"
    MODEL_ADAPTATIONS = "model_adaptations"
    LANGUAGE_MODELS = "language_models"
    CACHE_DATA = "cache_data"
    INTERACTION_HISTORY = "interaction_history"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_CONFIG = "system_config"


class AccessPattern(Enum):
    """Data access patterns for optimization"""
    FREQUENT_READ = "frequent_read"
    FREQUENT_WRITE = "frequent_write"
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    APPEND_ONLY = "append_only"
    READ_ONLY = "read_only"


@dataclass
class StorageConfig:
    """Configuration for storage system"""
    base_path: str
    storage_type: StorageType = StorageType.SQLITE
    compression: CompressionType = CompressionType.GZIP
    max_cache_size_mb: int = 1024
    auto_backup: bool = True
    backup_interval_hours: int = 24
    encryption_enabled: bool = False
    sync_enabled: bool = False
    max_file_size_mb: int = 100
    cleanup_old_data: bool = True
    retention_days: int = 365


@dataclass
class DataEntry:
    """Data entry with metadata"""
    key: str
    value: Any
    category: DataCategory
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferenceData:
    """User preference data structure"""
    user_id: str
    language_preferences: Dict[str, float]  # language -> preference score
    interface_settings: Dict[str, Any]
    model_preferences: Dict[str, Any]
    interaction_patterns: Dict[str, Any]
    accessibility_settings: Dict[str, Any]
    privacy_settings: Dict[str, Any]
    created_at: float
    updated_at: float


@dataclass
class ModelAdaptationData:
    """Model adaptation data structure"""
    model_id: str
    language_code: str
    adaptation_type: str  # fine_tuning, prompt_engineering, etc.
    adaptation_data: bytes  # Serialized model weights/parameters
    performance_metrics: Dict[str, float]
    training_metadata: Dict[str, Any]
    created_at: float
    file_size_bytes: int


class StorageBackend:
    """Base class for storage backends"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def store(self, key: str, value: Any, category: DataCategory) -> bool:
        """Store data"""
        raise NotImplementedError
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data"""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete data"""
        raise NotImplementedError
    
    def list_keys(self, category: Optional[DataCategory] = None) -> List[str]:
        """List all keys"""
        raise NotImplementedError
    
    def cleanup(self) -> None:
        """Cleanup storage"""
        raise NotImplementedError


class SQLiteBackend(StorageBackend):
    """SQLite storage backend"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.db_path = Path(config.base_path) / "storage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    category TEXT,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    size_bytes INTEGER,
                    checksum TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON data_entries(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON data_entries(timestamp)
            """)
            
            conn.commit()
    
    def store(self, key: str, value: Any, category: DataCategory) -> bool:
        """Store data in SQLite"""
        try:
            with self._lock:
                # Serialize value
                if self.config.compression == CompressionType.GZIP:
                    serialized = gzip.compress(pickle.dumps(value))
                elif self.config.compression == CompressionType.LZMA:
                    serialized = lzma.compress(pickle.dumps(value))
                else:
                    serialized = pickle.dumps(value)
                
                # Calculate metadata
                size_bytes = len(serialized)
                checksum = hashlib.md5(serialized).hexdigest()
                timestamp = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO data_entries 
                        (key, value, category, timestamp, size_bytes, checksum, last_accessed, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (key, serialized, category.value, timestamp, size_bytes, 
                         checksum, timestamp, json.dumps({})))
                    
                    conn.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from SQLite"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT value, access_count FROM data_entries WHERE key = ?
                    """, (key,))
                    
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    
                    serialized_value, access_count = row
                    
                    # Update access statistics
                    conn.execute("""
                        UPDATE data_entries 
                        SET access_count = ?, last_accessed = ?
                        WHERE key = ?
                    """, (access_count + 1, time.time(), key))
                    
                    conn.commit()
                
                # Deserialize value
                if self.config.compression == CompressionType.GZIP:
                    value = pickle.loads(gzip.decompress(serialized_value))
                elif self.config.compression == CompressionType.LZMA:
                    value = pickle.loads(lzma.decompress(serialized_value))
                else:
                    value = pickle.loads(serialized_value)
                
                return value
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from SQLite"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM data_entries WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self.logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    def list_keys(self, category: Optional[DataCategory] = None) -> List[str]:
        """List keys from SQLite"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    if category:
                        cursor = conn.execute(
                            "SELECT key FROM data_entries WHERE category = ?", 
                            (category.value,)
                        )
                    else:
                        cursor = conn.execute("SELECT key FROM data_entries")
                    
                    return [row[0] for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []
    
    def cleanup(self) -> None:
        """Cleanup old data"""
        if not self.config.cleanup_old_data:
            return
        
        try:
            cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM data_entries WHERE timestamp < ?", 
                        (cutoff_time,)
                    )
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        self.logger.info(f"Cleaned up {cursor.rowcount} old entries")
                        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class HDF5Backend(StorageBackend):
    """HDF5 storage backend for large numerical data"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 backend")
        
        self.file_path = Path(config.base_path) / "storage.h5"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def store(self, key: str, value: Any, category: DataCategory) -> bool:
        """Store data in HDF5"""
        try:
            with self._lock:
                with h5py.File(self.file_path, 'a') as f:
                    # Create group for category if it doesn't exist
                    if category.value not in f:
                        f.create_group(category.value)
                    
                    group = f[category.value]
                    
                    # Delete existing key if present
                    if key in group:
                        del group[key]
                    
                    # Store based on data type
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().numpy()
                        
                        dataset = group.create_dataset(key, data=value, compression='gzip')
                        dataset.attrs['timestamp'] = time.time()
                        dataset.attrs['dtype'] = str(value.dtype)
                        
                    else:
                        # For non-array data, serialize and store as bytes
                        serialized = pickle.dumps(value)
                        dataset = group.create_dataset(key, data=np.frombuffer(serialized, dtype=np.uint8))
                        dataset.attrs['timestamp'] = time.time()
                        dataset.attrs['is_pickled'] = True
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from HDF5"""
        try:
            with self._lock:
                with h5py.File(self.file_path, 'r') as f:
                    # Search for key in all groups
                    for group_name in f.keys():
                        group = f[group_name]
                        if key in group:
                            dataset = group[key]
                            
                            if dataset.attrs.get('is_pickled', False):
                                # Deserialize pickled data
                                data_bytes = dataset[:].tobytes()
                                return pickle.loads(data_bytes)
                            else:
                                # Return numpy array
                                return dataset[:]
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from HDF5"""
        try:
            with self._lock:
                with h5py.File(self.file_path, 'a') as f:
                    for group_name in f.keys():
                        group = f[group_name]
                        if key in group:
                            del group[key]
                            return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    def list_keys(self, category: Optional[DataCategory] = None) -> List[str]:
        """List keys from HDF5"""
        try:
            with self._lock:
                with h5py.File(self.file_path, 'r') as f:
                    keys = []
                    
                    if category:
                        if category.value in f:
                            keys.extend(list(f[category.value].keys()))
                    else:
                        for group_name in f.keys():
                            keys.extend(list(f[group_name].keys()))
                    
                    return keys
                    
        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []
    
    def cleanup(self) -> None:
        """Cleanup HDF5 file"""
        # HDF5 doesn't support in-place deletion efficiently
        # This would require rewriting the entire file
        pass


class FileSystemBackend(StorageBackend):
    """File system storage backend"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def _get_file_path(self, key: str, category: DataCategory) -> Path:
        """Get file path for key"""
        category_dir = self.base_path / category.value
        category_dir.mkdir(exist_ok=True)
        
        # Use hash to avoid filesystem limitations
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return category_dir / f"{key_hash}.dat"
    
    def store(self, key: str, value: Any, category: DataCategory) -> bool:
        """Store data to file system"""
        try:
            with self._lock:
                file_path = self._get_file_path(key, category)
                
                # Serialize data
                serialized = pickle.dumps(value)
                
                # Apply compression
                if self.config.compression == CompressionType.GZIP:
                    serialized = gzip.compress(serialized)
                elif self.config.compression == CompressionType.LZMA:
                    serialized = lzma.compress(serialized)
                
                # Write to file
                with open(file_path, 'wb') as f:
                    f.write(serialized)
                
                # Store metadata
                metadata = {
                    'key': key,
                    'category': category.value,
                    'timestamp': time.time(),
                    'size_bytes': len(serialized),
                    'compression': self.config.compression.value
                }
                
                metadata_path = file_path.with_suffix('.meta')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from file system"""
        try:
            with self._lock:
                # Search in all categories
                for category in DataCategory:
                    file_path = self._get_file_path(key, category)
                    
                    if file_path.exists():
                        # Read metadata
                        metadata_path = file_path.with_suffix('.meta')
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            compression = CompressionType(metadata.get('compression', 'none'))
                        else:
                            compression = self.config.compression
                        
                        # Read and decompress data
                        with open(file_path, 'rb') as f:
                            data = f.read()
                        
                        if compression == CompressionType.GZIP:
                            data = gzip.decompress(data)
                        elif compression == CompressionType.LZMA:
                            data = lzma.decompress(data)
                        
                        return pickle.loads(data)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from file system"""
        try:
            with self._lock:
                deleted = False
                
                for category in DataCategory:
                    file_path = self._get_file_path(key, category)
                    metadata_path = file_path.with_suffix('.meta')
                    
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                    
                    if metadata_path.exists():
                        metadata_path.unlink()
                
                return deleted
                
        except Exception as e:
            self.logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    def list_keys(self, category: Optional[DataCategory] = None) -> List[str]:
        """List keys from file system"""
        try:
            with self._lock:
                keys = []
                
                categories = [category] if category else list(DataCategory)
                
                for cat in categories:
                    category_dir = self.base_path / cat.value
                    if category_dir.exists():
                        for meta_file in category_dir.glob('*.meta'):
                            try:
                                with open(meta_file, 'r') as f:
                                    metadata = json.load(f)
                                keys.append(metadata['key'])
                            except:
                                continue
                
                return keys
                
        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []
    
    def cleanup(self) -> None:
        """Cleanup old files"""
        if not self.config.cleanup_old_data:
            return
        
        try:
            cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
            
            with self._lock:
                for category_dir in self.base_path.iterdir():
                    if category_dir.is_dir():
                        for meta_file in category_dir.glob('*.meta'):
                            try:
                                with open(meta_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                if metadata.get('timestamp', 0) < cutoff_time:
                                    # Delete data and metadata files
                                    data_file = meta_file.with_suffix('.dat')
                                    if data_file.exists():
                                        data_file.unlink()
                                    meta_file.unlink()
                                    
                            except:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class LocalStorageManager:
    """
    Comprehensive local storage manager
    
    Features:
    - Multiple storage backends (SQLite, HDF5, FileSystem)
    - Automatic compression and encryption
    - User preference management
    - Model adaptation storage
    - Performance optimization
    - Automatic backup and cleanup
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage backend
        self.backend = self._create_backend()
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_access_times = {}
        self.cache_size_bytes = 0
        
        # Background tasks
        self.cleanup_thread = None
        self.backup_thread = None
        
        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._start_background_tasks()
        
        self.logger.info("LocalStorageManager initialized")
    
    def _create_backend(self) -> StorageBackend:
        """Create storage backend based on configuration"""
        if self.config.storage_type == StorageType.SQLITE:
            return SQLiteBackend(self.config)
        elif self.config.storage_type == StorageType.HDF5:
            return HDF5Backend(self.config)
        else:
            return FileSystemBackend(self.config)
    
    def store_user_preferences(self, user_id: str, preferences: UserPreferenceData) -> bool:
        """Store user preferences"""
        key = f"user_prefs_{user_id}"
        success = self.backend.store(key, preferences, DataCategory.USER_PREFERENCES)
        
        if success:
            self.stats['writes'] += 1
            # Update cache
            self._update_cache(key, preferences)
        
        return success
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferenceData]:
        """Get user preferences"""
        key = f"user_prefs_{user_id}"
        
        # Check cache first
        if key in self.cache:
            self.stats['cache_hits'] += 1
            self.cache_access_times[key] = time.time()
            return self.cache[key]
        
        # Load from storage
        preferences = self.backend.retrieve(key)
        self.stats['reads'] += 1
        
        if preferences:
            self.stats['cache_misses'] += 1
            self._update_cache(key, preferences)
        
        return preferences
    
    def store_model_adaptation(self, model_id: str, language_code: str, 
                              adaptation_data: ModelAdaptationData) -> bool:
        """Store model adaptation data"""
        key = f"model_adapt_{model_id}_{language_code}"
        success = self.backend.store(key, adaptation_data, DataCategory.MODEL_ADAPTATIONS)
        
        if success:
            self.stats['writes'] += 1
        
        return success
    
    def get_model_adaptation(self, model_id: str, language_code: str) -> Optional[ModelAdaptationData]:
        """Get model adaptation data"""
        key = f"model_adapt_{model_id}_{language_code}"
        
        adaptation_data = self.backend.retrieve(key)
        if adaptation_data:
            self.stats['reads'] += 1
        
        return adaptation_data
    
    def store_language_model(self, language_code: str, model_data: bytes) -> bool:
        """Store language-specific model data"""
        key = f"lang_model_{language_code}"
        success = self.backend.store(key, model_data, DataCategory.LANGUAGE_MODELS)
        
        if success:
            self.stats['writes'] += 1
        
        return success
    
    def get_language_model(self, language_code: str) -> Optional[bytes]:
        """Get language-specific model data"""
        key = f"lang_model_{language_code}"
        
        model_data = self.backend.retrieve(key)
        if model_data:
            self.stats['reads'] += 1
        
        return model_data
    
    def store_interaction_history(self, user_id: str, interactions: List[Dict[str, Any]]) -> bool:
        """Store user interaction history"""
        key = f"interactions_{user_id}"
        success = self.backend.store(key, interactions, DataCategory.INTERACTION_HISTORY)
        
        if success:
            self.stats['writes'] += 1
        
        return success
    
    def get_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interaction history"""
        key = f"interactions_{user_id}"
        
        interactions = self.backend.retrieve(key)
        if interactions:
            self.stats['reads'] += 1
            return interactions
        
        return []
    
    def store_performance_metrics(self, model_id: str, metrics: Dict[str, Any]) -> bool:
        """Store performance metrics"""
        key = f"metrics_{model_id}_{int(time.time())}"
        success = self.backend.store(key, metrics, DataCategory.PERFORMANCE_METRICS)
        
        if success:
            self.stats['writes'] += 1
        
        return success
    
    def get_performance_metrics(self, model_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics for a model"""
        all_keys = self.backend.list_keys(DataCategory.PERFORMANCE_METRICS)
        model_keys = [k for k in all_keys if k.startswith(f"metrics_{model_id}_")]
        
        # Sort by timestamp (newest first)
        model_keys.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
        
        metrics = []
        for key in model_keys[:limit]:
            metric_data = self.backend.retrieve(key)
            if metric_data:
                metrics.append(metric_data)
        
        if metrics:
            self.stats['reads'] += len(metrics)
        
        return metrics
    
    def cache_data(self, key: str, value: Any, category: DataCategory) -> bool:
        """Cache frequently accessed data"""
        success = self.backend.store(key, value, DataCategory.CACHE_DATA)
        
        if success:
            self.stats['writes'] += 1
            self._update_cache(key, value)
        
        return success
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data"""
        # Check memory cache first
        if key in self.cache:
            self.stats['cache_hits'] += 1
            self.cache_access_times[key] = time.time()
            return self.cache[key]
        
        # Check storage cache
        data = self.backend.retrieve(key)
        if data:
            self.stats['reads'] += 1
            self.stats['cache_misses'] += 1
            self._update_cache(key, data)
        
        return data
    
    def _update_cache(self, key: str, value: Any):
        """Update memory cache"""
        # Estimate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        # Check cache size limit
        max_cache_bytes = self.config.max_cache_size_mb * 1024 * 1024
        
        while self.cache_size_bytes + size_bytes > max_cache_bytes and self.cache:
            # Remove least recently used item
            lru_key = min(self.cache_access_times.keys(), 
                         key=lambda k: self.cache_access_times[k])
            
            if lru_key in self.cache:
                del self.cache[lru_key]
                del self.cache_access_times[lru_key]
                # Approximate size reduction
                self.cache_size_bytes -= size_bytes // 2
        
        # Add to cache
        self.cache[key] = value
        self.cache_access_times[key] = time.time()
        self.cache_size_bytes += size_bytes
    
    def list_user_preferences(self) -> List[str]:
        """List all user IDs with stored preferences"""
        keys = self.backend.list_keys(DataCategory.USER_PREFERENCES)
        user_ids = []
        
        for key in keys:
            if key.startswith("user_prefs_"):
                user_id = key[11:]  # Remove "user_prefs_" prefix
                user_ids.append(user_id)
        
        return user_ids
    
    def list_model_adaptations(self) -> List[Tuple[str, str]]:
        """List all model adaptations (model_id, language_code)"""
        keys = self.backend.list_keys(DataCategory.MODEL_ADAPTATIONS)
        adaptations = []
        
        for key in keys:
            if key.startswith("model_adapt_"):
                parts = key[12:].split('_', 1)  # Remove "model_adapt_" prefix
                if len(parts) == 2:
                    model_id, language_code = parts
                    adaptations.append((model_id, language_code))
        
        return adaptations
    
    def list_language_models(self) -> List[str]:
        """List all stored language models"""
        keys = self.backend.list_keys(DataCategory.LANGUAGE_MODELS)
        languages = []
        
        for key in keys:
            if key.startswith("lang_model_"):
                language_code = key[11:]  # Remove "lang_model_" prefix
                languages.append(language_code)
        
        return languages
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user"""
        success = True
        
        # Delete user preferences
        pref_key = f"user_prefs_{user_id}"
        success &= self.backend.delete(pref_key)
        
        # Delete interaction history
        history_key = f"interactions_{user_id}"
        success &= self.backend.delete(history_key)
        
        # Remove from cache
        if pref_key in self.cache:
            del self.cache[pref_key]
            del self.cache_access_times[pref_key]
        
        if history_key in self.cache:
            del self.cache[history_key]
            del self.cache_access_times[history_key]
        
        return success
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data"""
        data = {}
        
        # Get user preferences
        preferences = self.get_user_preferences(user_id)
        if preferences:
            data['preferences'] = asdict(preferences)
        
        # Get interaction history
        interactions = self.get_interaction_history(user_id)
        if interactions:
            data['interactions'] = interactions
        
        return data
    
    def import_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Import user data"""
        success = True
        
        # Import preferences
        if 'preferences' in data:
            prefs = UserPreferenceData(**data['preferences'])
            success &= self.store_user_preferences(user_id, prefs)
        
        # Import interactions
        if 'interactions' in data:
            success &= self.store_interaction_history(user_id, data['interactions'])
        
        return success
    
    def backup_data(self, backup_path: str) -> bool:
        """Create backup of all data"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy storage files
            source_dir = Path(self.config.base_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_dir / f"backup_{timestamp}"
            
            shutil.copytree(source_dir, backup_subdir)
            
            self.logger.info(f"Backup created at {backup_subdir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restore_data(self, backup_path: str) -> bool:
        """Restore data from backup"""
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                return False
            
            # Stop background tasks
            self._stop_background_tasks()
            
            # Clear current data
            source_dir = Path(self.config.base_path)
            if source_dir.exists():
                shutil.rmtree(source_dir)
            
            # Restore from backup
            shutil.copytree(backup_dir, source_dir)
            
            # Reinitialize backend
            self.backend = self._create_backend()
            
            # Clear cache
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_size_bytes = 0
            
            # Restart background tasks
            self._start_background_tasks()
            
            self.logger.info(f"Data restored from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = self.stats.copy()
        
        # Add cache statistics
        stats.update({
            'cache_size_items': len(self.cache),
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            )
        })
        
        # Add storage size information
        try:
            storage_path = Path(self.config.base_path)
            if storage_path.exists():
                total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
                stats['storage_size_mb'] = total_size / (1024 * 1024)
            else:
                stats['storage_size_mb'] = 0
        except:
            stats['storage_size_mb'] = 0
        
        return stats
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.config.auto_backup:
            self.backup_thread = threading.Thread(target=self._backup_worker, daemon=True)
            self.backup_thread.start()
        
        if self.config.cleanup_old_data:
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
    
    def _stop_background_tasks(self):
        """Stop background tasks"""
        # Background threads are daemon threads and will stop automatically
        pass
    
    def _backup_worker(self):
        """Background backup worker"""
        while True:
            try:
                time.sleep(self.config.backup_interval_hours * 3600)
                
                backup_dir = Path(self.config.base_path).parent / "backups"
                self.backup_data(str(backup_dir))
                
            except Exception as e:
                self.logger.error(f"Background backup failed: {e}")
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(24 * 3600)  # Run daily
                self.backend.cleanup()
                
            except Exception as e:
                self.logger.error(f"Background cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_background_tasks()


if __name__ == "__main__":
    # Example usage
    
    # Create storage configuration
    config = StorageConfig(
        base_path="./local_storage",
        storage_type=StorageType.SQLITE,
        compression=CompressionType.GZIP,
        max_cache_size_mb=512,
        auto_backup=True
    )
    
    # Initialize storage manager
    storage = LocalStorageManager(config)
    
    # Store user preferences
    user_prefs = UserPreferenceData(
        user_id="user123",
        language_preferences={"en": 1.0, "es": 0.8, "fr": 0.6},
        interface_settings={"theme": "dark", "font_size": 14},
        model_preferences={"speed": "fast", "accuracy": "high"},
        interaction_patterns={"avg_session_length": 300},
        accessibility_settings={"high_contrast": False},
        privacy_settings={"data_sharing": False},
        created_at=time.time(),
        updated_at=time.time()
    )
    
    storage.store_user_preferences("user123", user_prefs)
    
    # Retrieve user preferences
    retrieved_prefs = storage.get_user_preferences("user123")
    print(f"Retrieved preferences for user: {retrieved_prefs.user_id}")
    
    # Store model adaptation
    adaptation_data = ModelAdaptationData(
        model_id="multilingual_bert",
        language_code="hi",
        adaptation_type="fine_tuning",
        adaptation_data=b"model_weights_data",
        performance_metrics={"accuracy": 0.95, "f1_score": 0.93},
        training_metadata={"epochs": 10, "learning_rate": 0.001},
        created_at=time.time(),
        file_size_bytes=1024
    )
    
    storage.store_model_adaptation("multilingual_bert", "hi", adaptation_data)
    
    # Get storage statistics
    stats = storage.get_storage_stats()
    print(f"Storage stats: {stats}")