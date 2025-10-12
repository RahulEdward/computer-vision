"""
User Preference Engine
उपयोगकर्ता प्राथमिकता इंजन

Advanced adaptive user preference learning and personalization system.
Learns from user interactions, adapts to preferences, and provides personalized experiences.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sqlite3
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
from pathlib import Path
import pickle
import uuid

# Optional imports for advanced features
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class InteractionType(Enum):
    """Types of user interactions"""
    CLICK = "click"
    VIEW = "view"
    SEARCH = "search"
    SELECTION = "selection"
    RATING = "rating"
    FEEDBACK = "feedback"
    LANGUAGE_SWITCH = "language_switch"
    MODEL_CHOICE = "model_choice"
    SETTING_CHANGE = "setting_change"
    TASK_COMPLETION = "task_completion"


class PreferenceCategory(Enum):
    """Categories of user preferences"""
    LANGUAGE = "language"
    INTERFACE = "interface"
    MODEL = "model"
    PERFORMANCE = "performance"
    CONTENT = "content"
    BEHAVIOR = "behavior"
    ACCESSIBILITY = "accessibility"
    PRIVACY = "privacy"


class LearningMode(Enum):
    """Learning modes for preference adaptation"""
    PASSIVE = "passive"  # Learn from implicit interactions
    ACTIVE = "active"    # Ask for explicit feedback
    HYBRID = "hybrid"    # Combination of both


@dataclass
class UserInteraction:
    """Represents a single user interaction"""
    interaction_id: str
    user_id: str
    timestamp: float
    interaction_type: InteractionType
    context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    duration: Optional[float] = None
    success: Optional[bool] = None


@dataclass
class UserPreference:
    """Represents a learned user preference"""
    preference_id: str
    user_id: str
    category: PreferenceCategory
    key: str
    value: Any
    confidence: float
    last_updated: float
    frequency: int = 1
    context_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Comprehensive user profile"""
    user_id: str
    created_at: float
    last_active: float
    preferences: Dict[str, UserPreference] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)  # Interaction IDs
    language_preferences: List[str] = field(default_factory=list)
    model_preferences: Dict[str, float] = field(default_factory=dict)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    personalization_level: float = 0.5  # 0.0 = minimal, 1.0 = maximum
    privacy_settings: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PreferenceConfig:
    """Configuration for preference learning"""
    learning_mode: LearningMode = LearningMode.HYBRID
    min_interactions_for_learning: int = 5
    confidence_threshold: float = 0.7
    max_history_size: int = 10000
    preference_decay_rate: float = 0.95
    adaptation_rate: float = 0.1
    clustering_enabled: bool = True
    privacy_mode: bool = False
    auto_save_interval: int = 300  # seconds


class PreferenceLearningModel(nn.Module):
    """Neural network for learning user preferences"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.preference_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        preference_score = self.preference_predictor(encoded)
        return encoded, preference_score


class UserPreferenceEngine:
    """
    Advanced user preference learning and adaptation system
    
    Features:
    - Implicit and explicit preference learning
    - Real-time adaptation to user behavior
    - Privacy-preserving preference storage
    - Multi-dimensional preference modeling
    - Context-aware recommendations
    - Cross-session preference persistence
    """
    
    def __init__(self, config: Optional[PreferenceConfig] = None, 
                 storage_path: Optional[str] = None):
        self.config = config or PreferenceConfig()
        self.storage_path = Path(storage_path) if storage_path else Path("user_preferences")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # User data storage
        self.users: Dict[str, UserProfile] = {}
        self.interactions: Dict[str, UserInteraction] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Learning models
        self.preference_model = None
        self.feature_scaler = None
        if HAS_SKLEARN:
            self.feature_scaler = StandardScaler()
        
        # Threading for background processing
        self._processing_lock = threading.RLock()
        self._save_timer = None
        
        # Initialize storage
        self._init_storage()
        self._load_user_data()
        
        # Start auto-save timer
        self._start_auto_save()
        
        self.logger.info("UserPreferenceEngine initialized")
    
    def _init_storage(self):
        """Initialize local storage for user preferences"""
        self.db_path = self.storage_path / "preferences.db"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    created_at REAL,
                    last_active REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp REAL,
                    interaction_data TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    preference_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    category TEXT,
                    key TEXT,
                    value TEXT,
                    confidence REAL,
                    last_updated REAL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            conn.commit()
    
    def _load_user_data(self):
        """Load existing user data from storage"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Load users
                cursor.execute("SELECT user_id, profile_data FROM users")
                for user_id, profile_data in cursor.fetchall():
                    try:
                        profile_dict = json.loads(profile_data)
                        # Convert dict back to UserProfile
                        profile = UserProfile(**profile_dict)
                        self.users[user_id] = profile
                    except Exception as e:
                        self.logger.warning(f"Failed to load user {user_id}: {e}")
                
                self.logger.info(f"Loaded {len(self.users)} user profiles")
        
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
    
    def _start_auto_save(self):
        """Start automatic saving of user data"""
        def auto_save():
            self.save_user_data()
            self._save_timer = threading.Timer(self.config.auto_save_interval, auto_save)
            self._save_timer.start()
        
        self._save_timer = threading.Timer(self.config.auto_save_interval, auto_save)
        self._save_timer.start()
    
    def create_user(self, user_id: Optional[str] = None, 
                   initial_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user profile"""
        if not user_id:
            user_id = str(uuid.uuid4())
        
        if user_id in self.users:
            self.logger.warning(f"User {user_id} already exists")
            return user_id
        
        # Create user profile
        profile = UserProfile(
            user_id=user_id,
            created_at=time.time(),
            last_active=time.time(),
            privacy_settings={
                'data_collection': True,
                'personalization': True,
                'analytics': True,
                'cross_session': True
            }
        )
        
        # Add initial preferences if provided
        if initial_preferences:
            for key, value in initial_preferences.items():
                self.set_preference(user_id, PreferenceCategory.BEHAVIOR, key, value)
        
        with self._processing_lock:
            self.users[user_id] = profile
        
        self.logger.info(f"Created user profile for {user_id}")
        return user_id
    
    def record_interaction(self, user_id: str, interaction_type: InteractionType,
                          context: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
                          session_id: Optional[str] = None, duration: Optional[float] = None,
                          success: Optional[bool] = None) -> str:
        """Record a user interaction"""
        if user_id not in self.users:
            self.create_user(user_id)
        
        interaction_id = str(uuid.uuid4())
        interaction = UserInteraction(
            interaction_id=interaction_id,
            user_id=user_id,
            timestamp=time.time(),
            interaction_type=interaction_type,
            context=context,
            metadata=metadata or {},
            session_id=session_id,
            duration=duration,
            success=success
        )
        
        with self._processing_lock:
            # Store interaction
            self.interactions[interaction_id] = interaction
            
            # Update user profile
            user = self.users[user_id]
            user.last_active = interaction.timestamp
            user.interaction_history.append(interaction_id)
            
            # Limit history size
            if len(user.interaction_history) > self.config.max_history_size:
                old_interaction_id = user.interaction_history.pop(0)
                if old_interaction_id in self.interactions:
                    del self.interactions[old_interaction_id]
        
        # Learn from interaction
        self._learn_from_interaction(interaction)
        
        return interaction_id
    
    def _learn_from_interaction(self, interaction: UserInteraction):
        """Learn preferences from user interaction"""
        user = self.users[interaction.user_id]
        
        # Language preference learning
        if 'language' in interaction.context:
            self._update_language_preference(user, interaction.context['language'])
        
        # Model preference learning
        if 'model' in interaction.context:
            self._update_model_preference(user, interaction.context['model'], interaction.success)
        
        # Interface preference learning
        if interaction.interaction_type == InteractionType.SETTING_CHANGE:
            self._update_interface_preferences(user, interaction.context)
        
        # Performance preference learning
        if interaction.duration is not None:
            self._update_performance_preferences(user, interaction)
        
        # Content preference learning
        if 'content_type' in interaction.context:
            self._update_content_preferences(user, interaction)
    
    def _update_language_preference(self, user: UserProfile, language: str):
        """Update language preferences based on usage"""
        if language not in user.language_preferences:
            user.language_preferences.append(language)
        else:
            # Move to front (most recently used)
            user.language_preferences.remove(language)
            user.language_preferences.insert(0, language)
        
        # Keep only top 10 languages
        user.language_preferences = user.language_preferences[:10]
        
        # Set as preference
        self.set_preference(user.user_id, PreferenceCategory.LANGUAGE, 
                          'primary_language', language, confidence=0.8)
    
    def _update_model_preference(self, user: UserProfile, model: str, success: Optional[bool]):
        """Update model preferences based on usage and success"""
        if model not in user.model_preferences:
            user.model_preferences[model] = 0.5
        
        # Adjust preference based on success
        if success is not None:
            adjustment = 0.1 if success else -0.1
            user.model_preferences[model] = max(0.0, min(1.0, 
                user.model_preferences[model] + adjustment))
        else:
            # Slight positive adjustment for usage
            user.model_preferences[model] = min(1.0, 
                user.model_preferences[model] + 0.02)
    
    def _update_interface_preferences(self, user: UserProfile, context: Dict[str, Any]):
        """Update interface preferences from setting changes"""
        for key, value in context.items():
            if key.startswith('ui_') or key.startswith('interface_'):
                self.set_preference(user.user_id, PreferenceCategory.INTERFACE, 
                                  key, value, confidence=0.9)
    
    def _update_performance_preferences(self, user: UserProfile, interaction: UserInteraction):
        """Update performance preferences based on interaction duration"""
        if 'task_type' in interaction.context:
            task_type = interaction.context['task_type']
            
            # Track average duration for task types
            if 'task_durations' not in user.usage_patterns:
                user.usage_patterns['task_durations'] = {}
            
            if task_type not in user.usage_patterns['task_durations']:
                user.usage_patterns['task_durations'][task_type] = []
            
            user.usage_patterns['task_durations'][task_type].append(interaction.duration)
            
            # Keep only recent durations
            if len(user.usage_patterns['task_durations'][task_type]) > 50:
                user.usage_patterns['task_durations'][task_type] = \
                    user.usage_patterns['task_durations'][task_type][-50:]
    
    def _update_content_preferences(self, user: UserProfile, interaction: UserInteraction):
        """Update content preferences based on interaction"""
        content_type = interaction.context.get('content_type')
        if content_type:
            # Track content type preferences
            if 'content_preferences' not in user.usage_patterns:
                user.usage_patterns['content_preferences'] = defaultdict(int)
            
            user.usage_patterns['content_preferences'][content_type] += 1
    
    def set_preference(self, user_id: str, category: PreferenceCategory, 
                      key: str, value: Any, confidence: float = 1.0,
                      context_conditions: Optional[Dict[str, Any]] = None) -> str:
        """Explicitly set a user preference"""
        if user_id not in self.users:
            self.create_user(user_id)
        
        preference_id = f"{user_id}_{category.value}_{key}"
        preference = UserPreference(
            preference_id=preference_id,
            user_id=user_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            last_updated=time.time(),
            context_conditions=context_conditions or {}
        )
        
        with self._processing_lock:
            user = self.users[user_id]
            user.preferences[preference_id] = preference
        
        return preference_id
    
    def get_preference(self, user_id: str, category: PreferenceCategory, 
                      key: str, default: Any = None, 
                      context: Optional[Dict[str, Any]] = None) -> Any:
        """Get a user preference value"""
        if user_id not in self.users:
            return default
        
        preference_id = f"{user_id}_{category.value}_{key}"
        user = self.users[user_id]
        
        if preference_id not in user.preferences:
            return default
        
        preference = user.preferences[preference_id]
        
        # Check context conditions
        if context and preference.context_conditions:
            for cond_key, cond_value in preference.context_conditions.items():
                if context.get(cond_key) != cond_value:
                    return default
        
        return preference.value
    
    def get_recommendations(self, user_id: str, category: Optional[PreferenceCategory] = None,
                           context: Optional[Dict[str, Any]] = None, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get personalized recommendations for the user"""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        recommendations = []
        
        # Language recommendations
        if not category or category == PreferenceCategory.LANGUAGE:
            for lang in user.language_preferences[:3]:
                recommendations.append({
                    'type': 'language',
                    'value': lang,
                    'confidence': 0.8,
                    'reason': 'frequently_used'
                })
        
        # Model recommendations
        if not category or category == PreferenceCategory.MODEL:
            sorted_models = sorted(user.model_preferences.items(), 
                                 key=lambda x: x[1], reverse=True)
            for model, score in sorted_models[:3]:
                if score > 0.6:
                    recommendations.append({
                        'type': 'model',
                        'value': model,
                        'confidence': score,
                        'reason': 'high_success_rate'
                    })
        
        # Content recommendations
        if not category or category == PreferenceCategory.CONTENT:
            content_prefs = user.usage_patterns.get('content_preferences', {})
            sorted_content = sorted(content_prefs.items(), 
                                  key=lambda x: x[1], reverse=True)
            for content_type, count in sorted_content[:3]:
                recommendations.append({
                    'type': 'content',
                    'value': content_type,
                    'confidence': min(1.0, count / 10),
                    'reason': 'frequently_accessed'
                })
        
        return recommendations[:limit]
    
    def adapt_interface(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get adaptive interface settings for the user"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        adaptations = {}
        
        # Language adaptation
        primary_lang = self.get_preference(user_id, PreferenceCategory.LANGUAGE, 
                                         'primary_language', 'en')
        adaptations['language'] = primary_lang
        
        # Theme adaptation
        theme = self.get_preference(user_id, PreferenceCategory.INTERFACE, 
                                  'theme', 'auto')
        adaptations['theme'] = theme
        
        # Complexity adaptation based on usage patterns
        interaction_count = len(user.interaction_history)
        if interaction_count < 10:
            adaptations['complexity'] = 'beginner'
        elif interaction_count < 100:
            adaptations['complexity'] = 'intermediate'
        else:
            adaptations['complexity'] = 'advanced'
        
        # Performance adaptation
        if 'task_durations' in user.usage_patterns:
            avg_durations = {}
            for task_type, durations in user.usage_patterns['task_durations'].items():
                avg_durations[task_type] = sum(durations) / len(durations)
            
            # Suggest faster models for users who prefer speed
            if avg_durations:
                overall_avg = sum(avg_durations.values()) / len(avg_durations)
                if overall_avg < 5.0:  # Fast user
                    adaptations['performance_preference'] = 'speed'
                elif overall_avg > 15.0:  # Patient user
                    adaptations['performance_preference'] = 'quality'
                else:
                    adaptations['performance_preference'] = 'balanced'
        
        return adaptations
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user behavior and preferences"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        insights = {
            'profile_age_days': (time.time() - user.created_at) / 86400,
            'total_interactions': len(user.interaction_history),
            'preferred_languages': user.language_preferences[:5],
            'top_models': sorted(user.model_preferences.items(), 
                               key=lambda x: x[1], reverse=True)[:3],
            'personalization_level': user.personalization_level,
            'last_active_hours_ago': (time.time() - user.last_active) / 3600
        }
        
        # Usage patterns
        if 'content_preferences' in user.usage_patterns:
            insights['content_distribution'] = dict(user.usage_patterns['content_preferences'])
        
        # Interaction patterns
        recent_interactions = user.interaction_history[-50:]  # Last 50 interactions
        interaction_types = defaultdict(int)
        for interaction_id in recent_interactions:
            if interaction_id in self.interactions:
                interaction = self.interactions[interaction_id]
                interaction_types[interaction.interaction_type.value] += 1
        
        insights['interaction_patterns'] = dict(interaction_types)
        
        return insights
    
    def export_user_data(self, user_id: str, include_interactions: bool = False) -> Dict[str, Any]:
        """Export user data for backup or transfer"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        data = {
            'profile': asdict(user),
            'preferences': {pid: asdict(pref) for pid, pref in user.preferences.items()}
        }
        
        if include_interactions:
            user_interactions = []
            for interaction_id in user.interaction_history:
                if interaction_id in self.interactions:
                    interaction = self.interactions[interaction_id]
                    user_interactions.append(asdict(interaction))
            data['interactions'] = user_interactions
        
        return data
    
    def import_user_data(self, user_data: Dict[str, Any]) -> str:
        """Import user data from backup"""
        profile_data = user_data.get('profile', {})
        if not profile_data:
            raise ValueError("Invalid user data: missing profile")
        
        user_id = profile_data['user_id']
        
        # Create user profile
        profile = UserProfile(**profile_data)
        
        # Import preferences
        preferences_data = user_data.get('preferences', {})
        for pref_id, pref_data in preferences_data.items():
            # Convert category and enum values
            pref_data['category'] = PreferenceCategory(pref_data['category'])
            preference = UserPreference(**pref_data)
            profile.preferences[pref_id] = preference
        
        # Import interactions if provided
        interactions_data = user_data.get('interactions', [])
        for interaction_data in interactions_data:
            interaction_data['interaction_type'] = InteractionType(interaction_data['interaction_type'])
            interaction = UserInteraction(**interaction_data)
            self.interactions[interaction.interaction_id] = interaction
        
        with self._processing_lock:
            self.users[user_id] = profile
        
        self.logger.info(f"Imported user data for {user_id}")
        return user_id
    
    def save_user_data(self):
        """Save user data to persistent storage"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Save users
                for user_id, profile in self.users.items():
                    profile_data = json.dumps(asdict(profile), default=str)
                    cursor.execute("""
                        INSERT OR REPLACE INTO users 
                        (user_id, profile_data, created_at, last_active)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, profile_data, profile.created_at, profile.last_active))
                
                # Save recent interactions
                recent_time = time.time() - 86400 * 7  # Last 7 days
                for interaction in self.interactions.values():
                    if interaction.timestamp > recent_time:
                        interaction_data = json.dumps(asdict(interaction), default=str)
                        cursor.execute("""
                            INSERT OR REPLACE INTO interactions
                            (interaction_id, user_id, timestamp, interaction_data)
                            VALUES (?, ?, ?, ?)
                        """, (interaction.interaction_id, interaction.user_id, 
                             interaction.timestamp, interaction_data))
                
                conn.commit()
                self.logger.info(f"Saved data for {len(self.users)} users")
        
        except Exception as e:
            self.logger.error(f"Failed to save user data: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old interaction data"""
        cutoff_time = time.time() - (days_to_keep * 86400)
        
        with self._processing_lock:
            # Remove old interactions
            old_interactions = [
                iid for iid, interaction in self.interactions.items()
                if interaction.timestamp < cutoff_time
            ]
            
            for iid in old_interactions:
                del self.interactions[iid]
            
            # Update user interaction histories
            for user in self.users.values():
                user.interaction_history = [
                    iid for iid in user.interaction_history
                    if iid in self.interactions
                ]
        
        self.logger.info(f"Cleaned up {len(old_interactions)} old interactions")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_interactions = len(self.interactions)
        active_users = len([u for u in self.users.values() 
                           if time.time() - u.last_active < 86400])
        
        return {
            'total_users': len(self.users),
            'active_users_24h': active_users,
            'total_interactions': total_interactions,
            'avg_interactions_per_user': total_interactions / max(1, len(self.users)),
            'storage_path': str(self.storage_path),
            'learning_mode': self.config.learning_mode.value
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_save_timer') and self._save_timer:
            self._save_timer.cancel()
        self.save_user_data()


# Example usage
if __name__ == "__main__":
    # Initialize preference engine
    engine = UserPreferenceEngine()
    
    # Create a test user
    user_id = engine.create_user()
    print(f"Created user: {user_id}")
    
    # Record some interactions
    engine.record_interaction(
        user_id, InteractionType.LANGUAGE_SWITCH,
        {'language': 'hi', 'previous_language': 'en'}
    )
    
    engine.record_interaction(
        user_id, InteractionType.MODEL_CHOICE,
        {'model': 'fast_model', 'task_type': 'translation'},
        success=True
    )
    
    engine.record_interaction(
        user_id, InteractionType.SETTING_CHANGE,
        {'ui_theme': 'dark', 'interface_complexity': 'advanced'}
    )
    
    # Get recommendations
    recommendations = engine.get_recommendations(user_id)
    print(f"Recommendations: {recommendations}")
    
    # Get adaptive interface
    adaptations = engine.adapt_interface(user_id)
    print(f"Interface adaptations: {adaptations}")
    
    # Get user insights
    insights = engine.get_user_insights(user_id)
    print(f"User insights: {insights}")
    
    # Get system stats
    stats = engine.get_system_stats()
    print(f"System stats: {stats}")