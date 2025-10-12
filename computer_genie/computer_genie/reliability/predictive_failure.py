"""
Predictive Failure Detection using Anomaly Detection ML Models

Implements comprehensive predictive failure detection system using
machine learning models for anomaly detection and failure prediction.
"""

import asyncio
import time
import json
import logging
import threading
import uuid
import pickle
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import weakref
import traceback
import statistics
import math

# ML imports (would be installed via pip)
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
    # Mock classes for development
    class IsolationForest:
        def __init__(self, **kwargs):
            pass
        def fit(self, X):
            pass
        def predict(self, X):
            return np.ones(len(X))
        def decision_function(self, X):
            return np.zeros(len(X))
            
    class OneClassSVM:
        def __init__(self, **kwargs):
            pass
        def fit(self, X):
            pass
        def predict(self, X):
            return np.ones(len(X))
        def decision_function(self, X):
            return np.zeros(len(X))
            
    class StandardScaler:
        def __init__(self):
            pass
        def fit(self, X):
            pass
        def transform(self, X):
            return X
        def fit_transform(self, X):
            return X

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    LATENCY_INCREASE = "latency_increase"
    THROUGHPUT_DROP = "throughput_drop"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    NETWORK_CONGESTION = "network_congestion"
    DATABASE_SLOWDOWN = "database_slowdown"
    DEPENDENCY_FAILURE = "dependency_failure"
    SECURITY_BREACH = "security_breach"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(Enum):
    """Types of ML models"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    STATISTICAL = "statistical"


@dataclass
class MetricData:
    """Represents a metric data point"""
    timestamp: float = field(default_factory=time.time)
    metric_name: str = ""
    value: float = 0.0
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    model_type: ModelType = ModelType.ISOLATION_FOREST
    
    # Data collection
    metrics_window_size: int = 1000
    feature_window_size: int = 100
    sampling_interval_seconds: float = 60.0
    
    # Model parameters
    contamination_rate: float = 0.1  # Expected anomaly rate
    n_estimators: int = 100
    max_samples: int = 256
    random_state: int = 42
    
    # Thresholds
    anomaly_threshold: float = -0.5
    confidence_threshold: float = 0.7
    
    # Training
    retrain_interval_hours: float = 24.0
    min_training_samples: int = 1000
    validation_split: float = 0.2
    
    # Features
    feature_names: List[str] = field(default_factory=lambda: [
        'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
        'response_time', 'error_rate', 'throughput', 'queue_size'
    ])
    
    # Alerting
    alert_on_anomaly: bool = True
    alert_threshold: float = 0.8
    cooldown_period_seconds: float = 300.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPrediction:
    """Represents an anomaly prediction"""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Prediction details
    is_anomaly: bool = False
    anomaly_type: Optional[AnomalyType] = None
    confidence: PredictionConfidence = PredictionConfidence.LOW
    score: float = 0.0
    probability: float = 0.0
    
    # Time prediction
    predicted_failure_time: Optional[float] = None
    time_to_failure: Optional[float] = None
    
    # Affected metrics
    affected_metrics: List[str] = field(default_factory=list)
    metric_values: Dict[str, float] = field(default_factory=dict)
    
    # Model information
    model_type: ModelType = ModelType.ISOLATION_FOREST
    model_version: str = "1.0"
    
    # Context
    features: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    severity: str = "low"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type.value if self.anomaly_type else None,
            'confidence': self.confidence.value,
            'score': self.score,
            'probability': self.probability,
            'predicted_failure_time': self.predicted_failure_time,
            'time_to_failure': self.time_to_failure,
            'affected_metrics': self.affected_metrics,
            'metric_values': self.metric_values,
            'model_type': self.model_type.value,
            'model_version': self.model_version,
            'features': self.features,
            'feature_names': self.feature_names,
            'recommendations': self.recommendations,
            'severity': self.severity,
            'metadata': self.metadata
        }


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    timestamp: float = field(default_factory=time.time)
    model_type: ModelType = ModelType.ISOLATION_FOREST
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Anomaly detection metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Performance metrics
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_size: int = 0
    
    # Data metrics
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'model_type': self.model_type.value,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_size': self.model_size,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'feature_count': self.feature_count
        }


class FeatureExtractor:
    """Extracts features from metric data"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def extract_features(self, metrics: List[MetricData]) -> Tuple[List[float], List[str]]:
        """Extract features from metrics"""
        if not metrics:
            return [], []
            
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics[-self.window_size:]:
            metric_groups[metric.metric_name].append(metric.value)
            
        features = []
        feature_names = []
        
        for metric_name, values in metric_groups.items():
            if not values:
                continue
                
            # Statistical features
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values)
            ])
            
            feature_names.extend([
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_min",
                f"{metric_name}_max",
                f"{metric_name}_median"
            ])
            
            # Trend features
            if len(values) > 1:
                # Linear trend
                x = np.arange(len(values))
                trend = np.polyfit(x, values, 1)[0]
                features.append(trend)
                feature_names.append(f"{metric_name}_trend")
                
                # Rate of change
                rate_of_change = (values[-1] - values[0]) / len(values)
                features.append(rate_of_change)
                feature_names.append(f"{metric_name}_rate_of_change")
                
                # Volatility
                volatility = np.std(np.diff(values))
                features.append(volatility)
                feature_names.append(f"{metric_name}_volatility")
            else:
                features.extend([0.0, 0.0, 0.0])
                feature_names.extend([
                    f"{metric_name}_trend",
                    f"{metric_name}_rate_of_change",
                    f"{metric_name}_volatility"
                ])
                
            # Percentile features
            if len(values) >= 10:
                features.extend([
                    np.percentile(values, 25),
                    np.percentile(values, 75),
                    np.percentile(values, 95)
                ])
                feature_names.extend([
                    f"{metric_name}_p25",
                    f"{metric_name}_p75",
                    f"{metric_name}_p95"
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
                feature_names.extend([
                    f"{metric_name}_p25",
                    f"{metric_name}_p75",
                    f"{metric_name}_p95"
                ])
                
        return features, feature_names


class AnomalyDetectionModel(ABC):
    """Abstract base class for anomaly detection models"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ModelPerformance:
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        pass
        
    @abstractmethod
    def save_model(self, path: str):
        """Save model to file"""
        pass
        
    @abstractmethod
    def load_model(self, path: str):
        """Load model from file"""
        pass


class IsolationForestModel(AnomalyDetectionModel):
    """Isolation Forest anomaly detection model"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination_rate,
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            random_state=config.random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ModelPerformance:
        """Train the Isolation Forest model"""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Evaluate on training data
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Calculate performance metrics
        performance = ModelPerformance(
            model_type=ModelType.ISOLATION_FOREST,
            training_time=training_time,
            training_samples=len(X),
            feature_count=X.shape[1] if len(X.shape) > 1 else 1
        )
        
        if y is not None:
            # Calculate accuracy metrics if ground truth is available
            y_pred = (predictions == -1).astype(int)
            performance.true_positives = np.sum((y == 1) & (y_pred == 1))
            performance.false_positives = np.sum((y == 0) & (y_pred == 1))
            performance.true_negatives = np.sum((y == 0) & (y_pred == 0))
            performance.false_negatives = np.sum((y == 1) & (y_pred == 0))
            
            if performance.true_positives + performance.false_positives > 0:
                performance.precision = performance.true_positives / (performance.true_positives + performance.false_positives)
            if performance.true_positives + performance.false_negatives > 0:
                performance.recall = performance.true_positives / (performance.true_positives + performance.false_negatives)
            if performance.precision + performance.recall > 0:
                performance.f1_score = 2 * (performance.precision * performance.recall) / (performance.precision + performance.recall)
                
            total_samples = len(y)
            performance.accuracy = (performance.true_positives + performance.true_negatives) / total_samples
            
        return performance
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to binary predictions (1 = anomaly, 0 = normal)
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, scores
        
    def save_model(self, path: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']


class OneClassSVMModel(AnomalyDetectionModel):
    """One-Class SVM anomaly detection model"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.model = OneClassSVM(
            nu=config.contamination_rate,
            kernel='rbf',
            gamma='scale'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ModelPerformance:
        """Train the One-Class SVM model"""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Evaluate on training data
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Calculate performance metrics
        performance = ModelPerformance(
            model_type=ModelType.ONE_CLASS_SVM,
            training_time=training_time,
            training_samples=len(X),
            feature_count=X.shape[1] if len(X.shape) > 1 else 1
        )
        
        return performance
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to binary predictions (1 = anomaly, 0 = normal)
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, scores
        
    def save_model(self, path: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']


class StatisticalAnomalyModel(AnomalyDetectionModel):
    """Statistical anomaly detection model using Z-score and IQR"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.feature_stats = {}
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ModelPerformance:
        """Train the statistical model"""
        start_time = time.time()
        
        # Calculate statistics for each feature
        self.feature_stats = {}
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            
            self.feature_stats[i] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'median': np.median(feature_data),
                'q1': np.percentile(feature_data, 25),
                'q3': np.percentile(feature_data, 75),
                'iqr': np.percentile(feature_data, 75) - np.percentile(feature_data, 25)
            }
            
        self.is_trained = True
        training_time = time.time() - start_time
        
        # Calculate performance metrics
        performance = ModelPerformance(
            model_type=ModelType.STATISTICAL,
            training_time=training_time,
            training_samples=len(X),
            feature_count=X.shape[1] if len(X.shape) > 1 else 1
        )
        
        return performance
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using statistical methods"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        predictions = []
        scores = []
        
        for sample in X:
            anomaly_scores = []
            
            for i, value in enumerate(sample):
                stats = self.feature_stats[i]
                
                # Z-score method
                z_score = abs((value - stats['mean']) / (stats['std'] + 1e-8))
                
                # IQR method
                iqr_score = 0
                if value < stats['q1'] - 1.5 * stats['iqr'] or value > stats['q3'] + 1.5 * stats['iqr']:
                    iqr_score = 1
                    
                # Combined score
                combined_score = max(z_score / 3.0, iqr_score)  # Normalize z-score
                anomaly_scores.append(combined_score)
                
            # Overall anomaly score
            overall_score = np.mean(anomaly_scores)
            scores.append(overall_score)
            
            # Binary prediction
            is_anomaly = overall_score > self.config.anomaly_threshold
            predictions.append(1 if is_anomaly else 0)
            
        return np.array(predictions), np.array(scores)
        
    def save_model(self, path: str):
        """Save model to file"""
        model_data = {
            'feature_stats': self.feature_stats,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.feature_stats = model_data['feature_stats']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']


class AnomalyClassifier:
    """Classifies anomalies into specific types"""
    
    def __init__(self):
        self.classification_rules = {
            AnomalyType.CPU_SPIKE: lambda features, names: self._check_cpu_spike(features, names),
            AnomalyType.MEMORY_LEAK: lambda features, names: self._check_memory_leak(features, names),
            AnomalyType.DISK_FULL: lambda features, names: self._check_disk_full(features, names),
            AnomalyType.LATENCY_INCREASE: lambda features, names: self._check_latency_increase(features, names),
            AnomalyType.ERROR_SPIKE: lambda features, names: self._check_error_spike(features, names),
            AnomalyType.THROUGHPUT_DROP: lambda features, names: self._check_throughput_drop(features, names)
        }
        
    def classify_anomaly(self, features: List[float], feature_names: List[str]) -> Optional[AnomalyType]:
        """Classify anomaly type based on features"""
        feature_dict = dict(zip(feature_names, features))
        
        # Check each anomaly type
        for anomaly_type, rule_func in self.classification_rules.items():
            if rule_func(feature_dict, feature_names):
                return anomaly_type
                
        return None
        
    def _check_cpu_spike(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for CPU spike anomaly"""
        cpu_features = [name for name in names if 'cpu' in name.lower()]
        if not cpu_features:
            return False
            
        for feature_name in cpu_features:
            if 'mean' in feature_name and features.get(feature_name, 0) > 80:
                return True
            if 'max' in feature_name and features.get(feature_name, 0) > 95:
                return True
                
        return False
        
    def _check_memory_leak(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for memory leak anomaly"""
        memory_features = [name for name in names if 'memory' in name.lower()]
        if not memory_features:
            return False
            
        for feature_name in memory_features:
            if 'trend' in feature_name and features.get(feature_name, 0) > 0.1:
                return True
            if 'mean' in feature_name and features.get(feature_name, 0) > 85:
                return True
                
        return False
        
    def _check_disk_full(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for disk full anomaly"""
        disk_features = [name for name in names if 'disk' in name.lower()]
        if not disk_features:
            return False
            
        for feature_name in disk_features:
            if 'mean' in feature_name and features.get(feature_name, 0) > 90:
                return True
                
        return False
        
    def _check_latency_increase(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for latency increase anomaly"""
        latency_features = [name for name in names if any(term in name.lower() for term in ['response_time', 'latency'])]
        if not latency_features:
            return False
            
        for feature_name in latency_features:
            if 'trend' in feature_name and features.get(feature_name, 0) > 0.05:
                return True
            if 'p95' in feature_name and features.get(feature_name, 0) > 1000:  # 1 second
                return True
                
        return False
        
    def _check_error_spike(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for error spike anomaly"""
        error_features = [name for name in names if 'error' in name.lower()]
        if not error_features:
            return False
            
        for feature_name in error_features:
            if 'mean' in feature_name and features.get(feature_name, 0) > 5:  # 5% error rate
                return True
                
        return False
        
    def _check_throughput_drop(self, features: Dict[str, float], names: List[str]) -> bool:
        """Check for throughput drop anomaly"""
        throughput_features = [name for name in names if 'throughput' in name.lower()]
        if not throughput_features:
            return False
            
        for feature_name in throughput_features:
            if 'trend' in feature_name and features.get(feature_name, 0) < -0.1:
                return True
                
        return False


class PredictiveFailureDetector:
    """Main predictive failure detection system"""
    
    def __init__(self, config: AnomalyDetectionConfig = None):
        self.config = config or AnomalyDetectionConfig()
        self.feature_extractor = FeatureExtractor(self.config.feature_window_size)
        self.anomaly_classifier = AnomalyClassifier()
        
        # Initialize model based on configuration
        if self.config.model_type == ModelType.ISOLATION_FOREST:
            self.model = IsolationForestModel(self.config)
        elif self.config.model_type == ModelType.ONE_CLASS_SVM:
            self.model = OneClassSVMModel(self.config)
        elif self.config.model_type == ModelType.STATISTICAL:
            self.model = StatisticalAnomalyModel(self.config)
        else:
            self.model = IsolationForestModel(self.config)  # Default
            
        # Data storage
        self.metrics_data: deque = deque(maxlen=self.config.metrics_window_size)
        self.predictions_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=100)
        
        # State
        self.is_trained = False
        self.last_training_time = 0
        self.last_alert_time = 0
        
        self._lock = threading.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        
    def add_metric(self, metric: MetricData):
        """Add a metric data point"""
        with self._lock:
            self.metrics_data.append(metric)
            
    def add_metrics(self, metrics: List[MetricData]):
        """Add multiple metric data points"""
        with self._lock:
            self.metrics_data.extend(metrics)
            
    async def train_model(self, force_retrain: bool = False) -> Optional[ModelPerformance]:
        """Train the anomaly detection model"""
        current_time = time.time()
        
        # Check if retraining is needed
        if (not force_retrain and 
            self.is_trained and 
            current_time - self.last_training_time < self.config.retrain_interval_hours * 3600):
            return None
            
        with self._lock:
            metrics_list = list(self.metrics_data)
            
        if len(metrics_list) < self.config.min_training_samples:
            logger.warning(f"Not enough training samples: {len(metrics_list)} < {self.config.min_training_samples}")
            return None
            
        try:
            # Extract features
            features_list = []
            feature_names = None
            
            # Process metrics in windows
            window_size = self.config.feature_window_size
            for i in range(window_size, len(metrics_list), window_size // 2):
                window_metrics = metrics_list[i-window_size:i]
                features, names = self.feature_extractor.extract_features(window_metrics)
                
                if features:
                    features_list.append(features)
                    if feature_names is None:
                        feature_names = names
                        
            if not features_list:
                logger.error("No features extracted from metrics")
                return None
                
            # Convert to numpy array
            X = np.array(features_list)
            
            # Train model
            performance = self.model.train(X)
            
            self.is_trained = True
            self.last_training_time = current_time
            
            with self._lock:
                self.performance_history.append(performance)
                
            logger.info(f"Model trained successfully. Performance: {performance.to_dict()}")
            return performance
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return None
            
    async def predict_anomalies(self, recent_metrics: List[MetricData] = None) -> List[AnomalyPrediction]:
        """Predict anomalies from recent metrics"""
        if not self.is_trained:
            logger.warning("Model not trained, cannot make predictions")
            return []
            
        # Use provided metrics or recent metrics from storage
        if recent_metrics is None:
            with self._lock:
                recent_metrics = list(self.metrics_data)[-self.config.feature_window_size:]
                
        if len(recent_metrics) < self.config.feature_window_size:
            logger.warning(f"Not enough recent metrics for prediction: {len(recent_metrics)}")
            return []
            
        try:
            # Extract features
            features, feature_names = self.feature_extractor.extract_features(recent_metrics)
            
            if not features:
                logger.warning("No features extracted for prediction")
                return []
                
            # Make prediction
            X = np.array([features])
            predictions, scores = self.model.predict(X)
            
            # Create prediction objects
            prediction_results = []
            
            for i, (is_anomaly, score) in enumerate(zip(predictions, scores)):
                prediction = AnomalyPrediction(
                    is_anomaly=bool(is_anomaly),
                    score=float(score),
                    features=features,
                    feature_names=feature_names,
                    model_type=self.config.model_type,
                    probability=self._score_to_probability(score)
                )
                
                # Classify anomaly type if detected
                if prediction.is_anomaly:
                    prediction.anomaly_type = self.anomaly_classifier.classify_anomaly(features, feature_names)
                    prediction.confidence = self._calculate_confidence(score)
                    prediction.recommendations = self._generate_recommendations(prediction.anomaly_type)
                    prediction.severity = self._calculate_severity(score, prediction.anomaly_type)
                    
                    # Predict failure time
                    prediction.predicted_failure_time, prediction.time_to_failure = self._predict_failure_time(
                        recent_metrics, prediction.anomaly_type
                    )
                    
                    # Extract affected metrics
                    prediction.affected_metrics = self._identify_affected_metrics(features, feature_names, score)
                    prediction.metric_values = {
                        metric.metric_name: metric.value 
                        for metric in recent_metrics[-10:]  # Last 10 metrics
                    }
                    
                prediction_results.append(prediction)
                
            # Store predictions
            with self._lock:
                self.predictions_history.extend(prediction_results)
                
            return prediction_results
            
        except Exception as e:
            logger.error(f"Failed to predict anomalies: {e}")
            return []
            
    def _score_to_probability(self, score: float) -> float:
        """Convert anomaly score to probability"""
        # Sigmoid transformation
        return 1.0 / (1.0 + math.exp(-score))
        
    def _calculate_confidence(self, score: float) -> PredictionConfidence:
        """Calculate confidence level based on score"""
        abs_score = abs(score)
        
        if abs_score > 2.0:
            return PredictionConfidence.CRITICAL
        elif abs_score > 1.0:
            return PredictionConfidence.HIGH
        elif abs_score > 0.5:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
            
    def _generate_recommendations(self, anomaly_type: Optional[AnomalyType]) -> List[str]:
        """Generate recommendations based on anomaly type"""
        if anomaly_type is None:
            return ["Monitor system closely", "Check system logs"]
            
        recommendations = {
            AnomalyType.CPU_SPIKE: [
                "Check for runaway processes",
                "Scale up CPU resources",
                "Optimize CPU-intensive operations"
            ],
            AnomalyType.MEMORY_LEAK: [
                "Restart affected services",
                "Check for memory leaks in application code",
                "Monitor memory usage patterns"
            ],
            AnomalyType.DISK_FULL: [
                "Clean up temporary files",
                "Archive old logs",
                "Add more disk space"
            ],
            AnomalyType.LATENCY_INCREASE: [
                "Check network connectivity",
                "Optimize database queries",
                "Scale up infrastructure"
            ],
            AnomalyType.ERROR_SPIKE: [
                "Check application logs",
                "Verify external dependencies",
                "Review recent deployments"
            ],
            AnomalyType.THROUGHPUT_DROP: [
                "Check system capacity",
                "Monitor resource utilization",
                "Scale up infrastructure"
            ]
        }
        
        return recommendations.get(anomaly_type, ["Monitor system closely"])
        
    def _calculate_severity(self, score: float, anomaly_type: Optional[AnomalyType]) -> str:
        """Calculate severity level"""
        abs_score = abs(score)
        
        # Base severity on score
        if abs_score > 2.0:
            base_severity = "critical"
        elif abs_score > 1.0:
            base_severity = "high"
        elif abs_score > 0.5:
            base_severity = "medium"
        else:
            base_severity = "low"
            
        # Adjust based on anomaly type
        critical_types = [AnomalyType.DISK_FULL, AnomalyType.MEMORY_LEAK, AnomalyType.SECURITY_BREACH]
        if anomaly_type in critical_types and base_severity != "critical":
            if base_severity == "high":
                return "critical"
            elif base_severity == "medium":
                return "high"
                
        return base_severity
        
    def _predict_failure_time(self, metrics: List[MetricData], 
                            anomaly_type: Optional[AnomalyType]) -> Tuple[Optional[float], Optional[float]]:
        """Predict when failure might occur"""
        if not metrics or anomaly_type is None:
            return None, None
            
        try:
            # Simple trend-based prediction
            current_time = time.time()
            
            # Get relevant metric values
            relevant_metrics = []
            for metric in metrics[-50:]:  # Last 50 metrics
                if self._is_relevant_metric(metric.metric_name, anomaly_type):
                    relevant_metrics.append((metric.timestamp, metric.value))
                    
            if len(relevant_metrics) < 10:
                return None, None
                
            # Calculate trend
            times = [m[0] for m in relevant_metrics]
            values = [m[1] for m in relevant_metrics]
            
            if len(set(values)) < 2:  # No variation
                return None, None
                
            # Linear regression for trend
            x = np.array(times)
            y = np.array(values)
            
            if len(x) > 1:
                trend = np.polyfit(x, y, 1)[0]
                
                # Predict failure based on threshold
                threshold = self._get_failure_threshold(anomaly_type)
                current_value = values[-1]
                
                if trend > 0 and current_value < threshold:
                    # Increasing trend
                    time_to_threshold = (threshold - current_value) / trend
                    failure_time = current_time + time_to_threshold
                    return failure_time, time_to_threshold
                elif trend < 0 and current_value > threshold:
                    # Decreasing trend
                    time_to_threshold = (current_value - threshold) / abs(trend)
                    failure_time = current_time + time_to_threshold
                    return failure_time, time_to_threshold
                    
        except Exception as e:
            logger.error(f"Failed to predict failure time: {e}")
            
        return None, None
        
    def _is_relevant_metric(self, metric_name: str, anomaly_type: AnomalyType) -> bool:
        """Check if metric is relevant for anomaly type"""
        relevance_map = {
            AnomalyType.CPU_SPIKE: ['cpu'],
            AnomalyType.MEMORY_LEAK: ['memory'],
            AnomalyType.DISK_FULL: ['disk'],
            AnomalyType.LATENCY_INCREASE: ['response_time', 'latency'],
            AnomalyType.ERROR_SPIKE: ['error'],
            AnomalyType.THROUGHPUT_DROP: ['throughput', 'requests']
        }
        
        relevant_terms = relevance_map.get(anomaly_type, [])
        return any(term in metric_name.lower() for term in relevant_terms)
        
    def _get_failure_threshold(self, anomaly_type: AnomalyType) -> float:
        """Get failure threshold for anomaly type"""
        thresholds = {
            AnomalyType.CPU_SPIKE: 95.0,
            AnomalyType.MEMORY_LEAK: 95.0,
            AnomalyType.DISK_FULL: 95.0,
            AnomalyType.LATENCY_INCREASE: 5000.0,  # 5 seconds
            AnomalyType.ERROR_SPIKE: 50.0,  # 50% error rate
            AnomalyType.THROUGHPUT_DROP: 0.0  # Complete drop
        }
        
        return thresholds.get(anomaly_type, 100.0)
        
    def _identify_affected_metrics(self, features: List[float], 
                                 feature_names: List[str], score: float) -> List[str]:
        """Identify which metrics are most affected"""
        if not features or not feature_names:
            return []
            
        # Simple approach: identify metrics with highest feature values
        affected = []
        
        for i, (feature_value, feature_name) in enumerate(zip(features, feature_names)):
            # Look for high values in statistical features
            if any(stat in feature_name for stat in ['mean', 'max', 'p95']):
                if abs(feature_value) > 1.0:  # Threshold for significance
                    metric_name = feature_name.split('_')[0]
                    if metric_name not in affected:
                        affected.append(metric_name)
                        
        return affected[:5]  # Return top 5 affected metrics
        
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self._monitoring_task is not None:
            return
            
        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._stop_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while not self._stop_event.is_set():
                # Check if model needs retraining
                if not self.is_trained or self._should_retrain():
                    await self.train_model()
                    
                # Make predictions if model is trained
                if self.is_trained:
                    predictions = await self.predict_anomalies()
                    
                    # Handle alerts
                    for prediction in predictions:
                        if prediction.is_anomaly and self._should_alert(prediction):
                            await self._send_alert(prediction)
                            
                # Wait for next iteration
                await asyncio.sleep(self.config.sampling_interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            
    def _should_retrain(self) -> bool:
        """Check if model should be retrained"""
        current_time = time.time()
        return (current_time - self.last_training_time > 
                self.config.retrain_interval_hours * 3600)
                
    def _should_alert(self, prediction: AnomalyPrediction) -> bool:
        """Check if alert should be sent"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_alert_time < self.config.cooldown_period_seconds:
            return False
            
        # Check alert threshold
        if prediction.probability < self.config.alert_threshold:
            return False
            
        return True
        
    async def _send_alert(self, prediction: AnomalyPrediction):
        """Send alert for anomaly prediction"""
        self.last_alert_time = time.time()
        
        alert_message = f"""
        ANOMALY DETECTED
        
        Type: {prediction.anomaly_type.value if prediction.anomaly_type else 'Unknown'}
        Confidence: {prediction.confidence.value}
        Severity: {prediction.severity}
        Score: {prediction.score:.3f}
        Probability: {prediction.probability:.3f}
        
        Affected Metrics: {', '.join(prediction.affected_metrics)}
        
        Recommendations:
        {chr(10).join(f'- {rec}' for rec in prediction.recommendations)}
        
        Time to Failure: {prediction.time_to_failure:.1f} seconds
        """ if prediction.time_to_failure else ""
        
        logger.warning(alert_message)
        
        # Here you would integrate with alerting systems
        # (email, Slack, PagerDuty, etc.)
        
    def get_model_performance(self) -> Optional[ModelPerformance]:
        """Get latest model performance"""
        with self._lock:
            if self.performance_history:
                return self.performance_history[-1]
        return None
        
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        with self._lock:
            recent = list(self.predictions_history)[-limit:]
            return [p.to_dict() for p in recent]
            
    def save_model(self, path: str):
        """Save trained model"""
        if self.is_trained:
            self.model.save_model(path)
        else:
            raise ValueError("No trained model to save")
            
    def load_model(self, path: str):
        """Load trained model"""
        self.model.load_model(path)
        self.is_trained = True


# Utility functions
def create_development_config() -> AnomalyDetectionConfig:
    """Create configuration for development"""
    return AnomalyDetectionConfig(
        model_type=ModelType.STATISTICAL,
        sampling_interval_seconds=30.0,
        contamination_rate=0.05,
        alert_on_anomaly=True
    )


def create_production_config() -> AnomalyDetectionConfig:
    """Create configuration for production"""
    return AnomalyDetectionConfig(
        model_type=ModelType.ISOLATION_FOREST,
        sampling_interval_seconds=60.0,
        contamination_rate=0.01,
        retrain_interval_hours=12.0,
        alert_on_anomaly=True,
        alert_threshold=0.9
    )


async def create_predictive_detector(config: AnomalyDetectionConfig = None) -> PredictiveFailureDetector:
    """Create and initialize predictive failure detector"""
    if config is None:
        config = create_production_config()
        
    detector = PredictiveFailureDetector(config)
    await detector.start_monitoring()
    
    return detector