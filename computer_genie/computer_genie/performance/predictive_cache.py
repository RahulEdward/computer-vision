"""
ML-based predictive caching and smart prefetching system.
Learns user patterns to preload likely-needed resources.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class UserAction:
    """Represents a user action for pattern learning."""
    timestamp: float
    action_type: str  # 'click', 'type', 'get', 'act'
    target: str
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""
    screen_hash: str = ""
    success: bool = True
    response_time_ms: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    ttl_seconds: float = 3600.0  # 1 hour default
    prediction_score: float = 0.0
    user_pattern_match: bool = False


@dataclass
class PredictionFeatures:
    """Features for ML prediction."""
    time_of_day: float
    day_of_week: int
    session_duration: float
    actions_in_session: int
    last_action_type: str
    target_similarity: float
    screen_context_hash: str
    user_behavior_cluster: int = -1


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for prediction."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.action_history: deque = deque(maxlen=max_history)
        self.user_sessions: Dict[str, List[UserAction]] = defaultdict(list)
        self.behavior_clusters: Optional[KMeans] = None
        self.action_predictor: Optional[RandomForestClassifier] = None
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_scaler = StandardScaler()
        
        # Pattern detection
        self.common_sequences: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.user_preferences: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.temporal_patterns: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # hour -> action_type -> count
        
        # Model training state
        self.model_trained = False
        self.last_training_time = 0
        self.training_interval = 3600  # Retrain every hour
    
    def record_action(self, action: UserAction):
        """Record user action for pattern learning."""
        self.action_history.append(action)
        self.user_sessions[action.session_id].append(action)
        
        # Update patterns
        self._update_patterns(action)
        
        # Trigger model retraining if needed
        current_time = time.time()
        if (current_time - self.last_training_time > self.training_interval and 
            len(self.action_history) >= 100):
            asyncio.create_task(self._retrain_models())
    
    def _update_patterns(self, action: UserAction):
        """Update behavior patterns with new action."""
        # Update user preferences
        self.user_preferences[action.user_id][action.action_type] += 1.0
        self.user_preferences[action.user_id][action.target] += 0.5
        
        # Update temporal patterns
        hour = int(time.localtime(action.timestamp).tm_hour)
        self.temporal_patterns[hour][action.action_type] += 1
        
        # Update sequence patterns
        if action.session_id in self.user_sessions:
            session_actions = self.user_sessions[action.session_id]
            if len(session_actions) >= 3:
                # Extract last 3 action types as sequence
                sequence = tuple(a.action_type for a in session_actions[-3:])
                self.common_sequences[sequence] += 1
    
    async def _retrain_models(self):
        """Retrain ML models with latest data."""
        try:
            logger.info("Retraining user behavior models...")
            
            if len(self.action_history) < 50:
                return
            
            # Prepare training data
            features, labels = self._prepare_training_data()
            
            if len(features) == 0:
                return
            
            # Train behavior clustering
            if len(features) >= 10:
                self.behavior_clusters = KMeans(n_clusters=min(5, len(features) // 10), random_state=42)
                cluster_features = self.feature_scaler.fit_transform(features)
                self.behavior_clusters.fit(cluster_features)
            
            # Train action predictor
            if len(set(labels)) > 1:  # Need multiple classes
                self.action_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
                self.action_predictor.fit(features, labels)
            
            self.model_trained = True
            self.last_training_time = time.time()
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[str]]:
        """Prepare training data from action history."""
        features = []
        labels = []
        
        for i, action in enumerate(self.action_history):
            if i == 0:
                continue  # Skip first action (no previous context)
            
            prev_action = self.action_history[i - 1]
            
            # Extract features
            feature_vector = [
                action.timestamp % 86400,  # Time of day in seconds
                time.localtime(action.timestamp).tm_wday,  # Day of week
                action.response_time_ms,
                len(action.target),
                1.0 if action.success else 0.0,
                hash(action.screen_hash) % 1000,  # Screen context
                hash(prev_action.action_type) % 100,  # Previous action type
            ]
            
            features.append(feature_vector)
            labels.append(action.action_type)
        
        return features, labels
    
    def predict_next_actions(self, current_action: UserAction, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next likely actions."""
        predictions = []
        
        # Sequence-based prediction
        if current_action.session_id in self.user_sessions:
            session_actions = self.user_sessions[current_action.session_id]
            if len(session_actions) >= 2:
                recent_sequence = tuple(a.action_type for a in session_actions[-2:])
                
                # Find sequences that start with recent actions
                for sequence, count in self.common_sequences.items():
                    if len(sequence) > 2 and sequence[:-1] == recent_sequence:
                        next_action = sequence[-1]
                        confidence = count / sum(self.common_sequences.values())
                        predictions.append((next_action, confidence))
        
        # ML-based prediction
        if self.model_trained and self.action_predictor:
            try:
                features = self._extract_prediction_features(current_action)
                probabilities = self.action_predictor.predict_proba([features])[0]
                classes = self.action_predictor.classes_
                
                for cls, prob in zip(classes, probabilities):
                    predictions.append((cls, prob))
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Temporal pattern prediction
        hour = int(time.localtime(current_action.timestamp).tm_hour)
        if hour in self.temporal_patterns:
            total_actions = sum(self.temporal_patterns[hour].values())
            for action_type, count in self.temporal_patterns[hour].items():
                confidence = count / total_actions
                predictions.append((action_type, confidence * 0.3))  # Lower weight for temporal
        
        # User preference prediction
        if current_action.user_id in self.user_preferences:
            user_prefs = self.user_preferences[current_action.user_id]
            total_prefs = sum(user_prefs.values())
            for action_type, count in user_prefs.items():
                confidence = count / total_prefs
                predictions.append((action_type, confidence * 0.2))  # Lower weight for preferences
        
        # Aggregate and rank predictions
        prediction_scores = defaultdict(float)
        for action_type, confidence in predictions:
            prediction_scores[action_type] += confidence
        
        # Sort by confidence and return top-k
        sorted_predictions = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def _extract_prediction_features(self, action: UserAction) -> List[float]:
        """Extract features for ML prediction."""
        return [
            action.timestamp % 86400,  # Time of day
            time.localtime(action.timestamp).tm_wday,  # Day of week
            len(self.user_sessions.get(action.session_id, [])),  # Session length
            len(action.target),  # Target length
            hash(action.screen_hash) % 1000,  # Screen context
            self.user_preferences[action.user_id].get(action.action_type, 0),  # User preference
        ]
    
    def get_user_behavior_cluster(self, user_id: str) -> int:
        """Get user's behavior cluster."""
        if not self.model_trained or not self.behavior_clusters:
            return -1
        
        try:
            # Get user's recent actions
            user_actions = [a for a in self.action_history if a.user_id == user_id]
            if len(user_actions) < 5:
                return -1
            
            # Extract features for clustering
            features = []
            for action in user_actions[-10:]:  # Last 10 actions
                feature_vector = self._extract_prediction_features(action)
                features.append(feature_vector)
            
            # Average features
            avg_features = np.mean(features, axis=0)
            scaled_features = self.feature_scaler.transform([avg_features])
            
            return self.behavior_clusters.predict(scaled_features)[0]
        
        except Exception as e:
            logger.warning(f"Behavior clustering failed: {e}")
            return -1


class SmartCache:
    """Intelligent cache with ML-based prefetching."""
    
    def __init__(self, max_size_mb: int = 1024, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        
        # LRU tracking
        self.access_order: deque = deque()
        
        # Prefetch queue
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_tasks: Set[asyncio.Task] = set()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "prefetch_hits": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str).encode())
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.stats["total_requests"] += 1
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > entry.ttl_seconds:
                await self._evict(key)
                self.stats["misses"] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Check if this was a prefetch hit
            if entry.prediction_score > 0:
                self.stats["prefetch_hits"] += 1
            
            self.stats["hits"] += 1
            return entry.value
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: float = 3600.0, 
                  prediction_score: float = 0.0, user_pattern_match: bool = False):
        """Set value in cache."""
        size_bytes = self._calculate_size(value)
        
        # Check if we need to evict entries
        while (len(self.cache) >= self.max_entries or 
               self.current_size_bytes + size_bytes > self.max_size_bytes):
            if not await self._evict_lru():
                break  # No more entries to evict
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds,
            prediction_score=prediction_score,
            user_pattern_match=user_pattern_match
        )
        
        # Remove old entry if exists
        if key in self.cache:
            await self._evict(key)
        
        # Add new entry
        self.cache[key] = entry
        self.current_size_bytes += size_bytes
        self.access_order.append(key)
    
    async def _evict(self, key: str):
        """Evict specific cache entry."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            
            self.stats["evictions"] += 1
    
    async def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self.access_order:
            return False
        
        # Find LRU entry with lowest prediction score
        candidates = []
        for key in list(self.access_order)[:10]:  # Check first 10 LRU entries
            if key in self.cache:
                entry = self.cache[key]
                candidates.append((key, entry.prediction_score, entry.last_access))
        
        if not candidates:
            return False
        
        # Sort by prediction score (ascending) then by last access (ascending)
        candidates.sort(key=lambda x: (x[1], x[2]))
        key_to_evict = candidates[0][0]
        
        await self._evict(key_to_evict)
        return True
    
    async def prefetch(self, key: str, fetch_func: callable, prediction_score: float = 0.5):
        """Prefetch data based on prediction."""
        if key in self.cache:
            return  # Already cached
        
        # Add to prefetch queue
        await self.prefetch_queue.put((key, fetch_func, prediction_score))
    
    async def _prefetch_worker(self):
        """Worker to process prefetch queue."""
        while True:
            try:
                key, fetch_func, prediction_score = await self.prefetch_queue.get()
                
                # Skip if already cached
                if key in self.cache:
                    self.prefetch_queue.task_done()
                    continue
                
                # Fetch data
                try:
                    value = await fetch_func()
                    await self.set(key, value, prediction_score=prediction_score)
                    logger.debug(f"Prefetched: {key}")
                except Exception as e:
                    logger.warning(f"Prefetch failed for {key}: {e}")
                
                self.prefetch_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
    
    def start_prefetch_workers(self, num_workers: int = 3):
        """Start prefetch worker tasks."""
        for _ in range(num_workers):
            task = asyncio.create_task(self._prefetch_worker())
            self.prefetch_tasks.add(task)
    
    async def stop_prefetch_workers(self):
        """Stop prefetch worker tasks."""
        for task in self.prefetch_tasks:
            task.cancel()
        
        await asyncio.gather(*self.prefetch_tasks, return_exceptions=True)
        self.prefetch_tasks.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats["hits"] / max(self.stats["total_requests"], 1)
        prefetch_effectiveness = self.stats["prefetch_hits"] / max(self.stats["hits"], 1)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "prefetch_effectiveness": prefetch_effectiveness,
            "cache_size_mb": self.current_size_bytes / (1024 * 1024),
            "cache_entries": len(self.cache),
            "avg_entry_size_kb": (self.current_size_bytes / len(self.cache) / 1024) if self.cache else 0
        }


class PredictiveCacheManager:
    """Main predictive cache manager."""
    
    def __init__(self, cache_size_mb: int = 1024, max_entries: int = 10000):
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.cache = SmartCache(cache_size_mb, max_entries)
        
        # Resource-specific caches
        self.screenshot_cache = SmartCache(512, 1000)  # Screenshots
        self.element_cache = SmartCache(256, 5000)     # UI elements
        self.model_cache = SmartCache(256, 100)        # ML models
        
        # Prefetch strategies
        self.prefetch_strategies = {
            "sequence_based": True,
            "temporal_based": True,
            "similarity_based": True,
            "user_cluster_based": True
        }
        
        # Performance tracking
        self.performance_metrics = {
            "cache_hit_rate": 0.0,
            "prefetch_accuracy": 0.0,
            "avg_response_time_ms": 0.0,
            "memory_efficiency": 0.0
        }
    
    async def initialize(self):
        """Initialize the cache manager."""
        self.cache.start_prefetch_workers(3)
        self.screenshot_cache.start_prefetch_workers(2)
        self.element_cache.start_prefetch_workers(2)
        self.model_cache.start_prefetch_workers(1)
        
        logger.info("Predictive cache manager initialized")
    
    async def record_user_action(self, action: UserAction):
        """Record user action and trigger predictive prefetching."""
        self.behavior_analyzer.record_action(action)
        
        # Predict next actions and prefetch
        await self._trigger_predictive_prefetch(action)
    
    async def _trigger_predictive_prefetch(self, current_action: UserAction):
        """Trigger predictive prefetching based on current action."""
        try:
            # Get predictions
            predictions = self.behavior_analyzer.predict_next_actions(current_action, top_k=5)
            
            for predicted_action, confidence in predictions:
                if confidence > 0.3:  # Only prefetch high-confidence predictions
                    await self._prefetch_for_action(predicted_action, current_action, confidence)
        
        except Exception as e:
            logger.warning(f"Predictive prefetch failed: {e}")
    
    async def _prefetch_for_action(self, predicted_action: str, current_action: UserAction, confidence: float):
        """Prefetch resources for predicted action."""
        # Prefetch screenshot analysis for predicted click/type actions
        if predicted_action in ['click', 'type'] and current_action.screen_hash:
            prefetch_key = f"screen_analysis_{current_action.screen_hash}"
            
            async def fetch_screen_analysis():
                # Simulate screen analysis
                return {
                    "elements": [],
                    "analysis_time": time.time(),
                    "predicted": True
                }
            
            await self.screenshot_cache.prefetch(prefetch_key, fetch_screen_analysis, confidence)
        
        # Prefetch similar targets based on user history
        if predicted_action == current_action.action_type:
            similar_targets = self._find_similar_targets(current_action.target, current_action.user_id)
            
            for target, similarity in similar_targets[:3]:  # Top 3 similar targets
                prefetch_key = f"element_{target}_{current_action.screen_hash}"
                
                async def fetch_element_info():
                    return {
                        "target": target,
                        "similarity": similarity,
                        "predicted": True,
                        "timestamp": time.time()
                    }
                
                await self.element_cache.prefetch(prefetch_key, fetch_element_info, confidence * similarity)
    
    def _find_similar_targets(self, target: str, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar targets based on user history."""
        user_actions = [a for a in self.behavior_analyzer.action_history if a.user_id == user_id]
        
        if not user_actions:
            return []
        
        # Extract unique targets
        targets = list(set(a.target for a in user_actions if a.target != target))
        
        if not targets:
            return []
        
        try:
            # Use TF-IDF for text similarity
            all_targets = [target] + targets
            tfidf_matrix = self.behavior_analyzer.text_vectorizer.fit_transform(all_targets)
            
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Combine with frequency weights
            target_frequencies = defaultdict(int)
            for action in user_actions:
                target_frequencies[action.target] += 1
            
            weighted_similarities = []
            for i, sim in enumerate(similarities):
                tgt = targets[i]
                frequency_weight = target_frequencies[tgt] / len(user_actions)
                weighted_sim = sim * 0.7 + frequency_weight * 0.3
                weighted_similarities.append((tgt, weighted_sim))
            
            # Sort by weighted similarity
            weighted_similarities.sort(key=lambda x: x[1], reverse=True)
            return weighted_similarities[:top_k]
        
        except Exception as e:
            logger.warning(f"Target similarity calculation failed: {e}")
            return []
    
    async def get_cached_result(self, cache_type: str, key: str) -> Optional[Any]:
        """Get result from appropriate cache."""
        cache_map = {
            "screenshot": self.screenshot_cache,
            "element": self.element_cache,
            "model": self.model_cache,
            "general": self.cache
        }
        
        cache = cache_map.get(cache_type, self.cache)
        return await cache.get(key)
    
    async def cache_result(self, cache_type: str, key: str, value: Any, 
                          ttl_seconds: float = 3600.0, prediction_score: float = 0.0):
        """Cache result in appropriate cache."""
        cache_map = {
            "screenshot": self.screenshot_cache,
            "element": self.element_cache,
            "model": self.model_cache,
            "general": self.cache
        }
        
        cache = cache_map.get(cache_type, self.cache)
        await cache.set(key, value, ttl_seconds, prediction_score)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Aggregate cache statistics
        all_caches = [self.cache, self.screenshot_cache, self.element_cache, self.model_cache]
        total_stats = {
            "hits": sum(c.stats["hits"] for c in all_caches),
            "misses": sum(c.stats["misses"] for c in all_caches),
            "prefetch_hits": sum(c.stats["prefetch_hits"] for c in all_caches),
            "total_requests": sum(c.stats["total_requests"] for c in all_caches)
        }
        
        # Calculate metrics
        hit_rate = total_stats["hits"] / max(total_stats["total_requests"], 1)
        prefetch_accuracy = total_stats["prefetch_hits"] / max(total_stats["hits"], 1)
        
        return {
            "cache_performance": {
                "overall_hit_rate": hit_rate,
                "prefetch_accuracy": prefetch_accuracy,
                "total_requests": total_stats["total_requests"],
                "cache_efficiency": hit_rate * prefetch_accuracy
            },
            "individual_caches": {
                "general": self.cache.get_stats(),
                "screenshot": self.screenshot_cache.get_stats(),
                "element": self.element_cache.get_stats(),
                "model": self.model_cache.get_stats()
            },
            "behavior_analysis": {
                "actions_recorded": len(self.behavior_analyzer.action_history),
                "unique_users": len(self.behavior_analyzer.user_preferences),
                "behavior_patterns": len(self.behavior_analyzer.common_sequences),
                "model_trained": self.behavior_analyzer.model_trained
            },
            "target_performance": {
                "target_response_time_ms": 100,
                "cache_hit_target": 0.8,
                "prefetch_accuracy_target": 0.6,
                "performance_score": min(hit_rate / 0.8, 1.0) * min(prefetch_accuracy / 0.6, 1.0)
            }
        }
    
    async def cleanup(self):
        """Clean up cache manager resources."""
        await self.cache.stop_prefetch_workers()
        await self.screenshot_cache.stop_prefetch_workers()
        await self.element_cache.stop_prefetch_workers()
        await self.model_cache.stop_prefetch_workers()


# Factory function for easy integration
def create_predictive_cache_manager(cache_size_mb: int = 1024, max_entries: int = 10000) -> PredictiveCacheManager:
    """Create predictive cache manager."""
    return PredictiveCacheManager(cache_size_mb, max_entries)