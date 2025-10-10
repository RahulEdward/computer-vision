"""
Offline Model Manager
ऑफलाइन मॉडल प्रबंधक

Manages offline AI models for computer vision and NLP tasks across 100+ languages.
Handles model loading, caching, and efficient resource management.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading
import time
from collections import defaultdict

# Optional dependencies
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


class ModelType(Enum):
    """Supported model types for offline execution"""
    VISION_CLASSIFICATION = "vision_classification"
    OBJECT_DETECTION = "object_detection"
    TEXT_CLASSIFICATION = "text_classification"
    LANGUAGE_TRANSLATION = "language_translation"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "ner"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"


class ModelFormat(Enum):
    """Supported model formats for offline deployment"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    QUANTIZED = "quantized"
    COMPRESSED = "compressed"


@dataclass
class ModelConfig:
    """Configuration for offline models"""
    model_id: str
    model_type: ModelType
    model_format: ModelFormat
    languages: List[str]
    file_path: str
    model_size: int  # in bytes
    memory_requirement: int  # in MB
    inference_time: float  # in milliseconds
    accuracy: float
    compression_ratio: float = 1.0
    quantization_bits: int = 32
    supports_gpu: bool = True
    supports_cpu: bool = True
    min_hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflineSystemConfig:
    """Configuration for the offline system"""
    models_directory: str = "./offline_models"
    cache_directory: str = "./model_cache"
    max_cache_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    max_concurrent_models: int = 5
    auto_cleanup: bool = True
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_compression: bool = True
    supported_languages: List[str] = field(default_factory=lambda: [
        # Major world languages (100+ languages)
        'en', 'zh', 'hi', 'es', 'fr', 'ar', 'bn', 'ru', 'pt', 'id',
        'ur', 'de', 'ja', 'sw', 'mr', 'te', 'tr', 'ta', 'vi', 'ko',
        'it', 'th', 'gu', 'kn', 'ml', 'or', 'pa', 'as', 'ne', 'si',
        'my', 'km', 'lo', 'ka', 'am', 'ti', 'om', 'so', 'zu', 'xh',
        'af', 'st', 'tn', 'ss', 've', 'ts', 'nr', 'nso', 'lg', 'ak',
        'tw', 'ee', 'ha', 'ig', 'yo', 'ff', 'wo', 'sn', 'ny', 'mg',
        'rw', 'rn', 'ki', 'lu', 'ln', 'kg', 'sw', 'bem', 'loz', 'ndc',
        'kj', 'hz', 'ng', 'ii', 'za', 'ug', 'bo', 'dz', 'mn', 'ky',
        'kk', 'uz', 'tk', 'tg', 'az', 'hy', 'ka', 'eu', 'ca', 'gl',
        'cy', 'ga', 'gd', 'br', 'co', 'mt', 'is', 'fo', 'no', 'da',
        'sv', 'fi', 'et', 'lv', 'lt', 'pl', 'cs', 'sk', 'sl', 'hr',
        'sr', 'bs', 'mk', 'bg', 'ro', 'hu', 'sq', 'el', 'he', 'yi'
    ])


class OfflineModelManager:
    """
    Comprehensive offline model manager for multilingual AI systems
    
    Features:
    - Supports 100+ languages
    - Multiple model formats (PyTorch, ONNX, TensorRT)
    - Intelligent caching and memory management
    - Model compression and quantization
    - Hardware-aware optimization
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[OfflineSystemConfig] = None):
        self.config = config or OfflineSystemConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize directories
        self._setup_directories()
        
        # Model registry and cache
        self.model_registry: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.cache_usage: Dict[str, float] = {}  # Track last access time
        
        # Threading and synchronization
        self._lock = threading.RLock()
        self._loading_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Hardware detection
        self.device = self._detect_optimal_device()
        self.hardware_info = self._get_hardware_info()
        
        # Load existing model registry
        self._load_model_registry()
        
        self.logger.info(f"OfflineModelManager initialized with {len(self.config.supported_languages)} languages")
    
    def _setup_directories(self):
        """Setup required directories for offline operation"""
        for directory in [self.config.models_directory, self.config.cache_directory]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _detect_optimal_device(self) -> torch.device:
        """Detect the optimal device for model execution"""
        if self.config.enable_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information"""
        info = {
            'device': str(self.device),
            'cpu_count': os.cpu_count(),
            'available_memory': self._get_available_memory(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory,
                'gpu_name': torch.cuda.get_device_name(0)
            })
        
        return info
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes"""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            return 8 * 1024 * 1024 * 1024  # Default 8GB
    
    def register_model(self, model_config: ModelConfig) -> bool:
        """Register a new model in the offline system"""
        try:
            with self._lock:
                # Validate model file exists
                if not os.path.exists(model_config.file_path):
                    self.logger.error(f"Model file not found: {model_config.file_path}")
                    return False
                
                # Validate languages are supported
                unsupported_langs = set(model_config.languages) - set(self.config.supported_languages)
                if unsupported_langs:
                    self.logger.warning(f"Unsupported languages: {unsupported_langs}")
                
                # Add to registry
                self.model_registry[model_config.model_id] = model_config
                self._save_model_registry()
                
                self.logger.info(f"Registered model: {model_config.model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register model {model_config.model_id}: {e}")
            return False
    
    def load_model(self, model_id: str, language: Optional[str] = None) -> Optional[Any]:
        """Load a model for offline inference"""
        if model_id not in self.model_registry:
            self.logger.error(f"Model not found in registry: {model_id}")
            return None
        
        model_config = self.model_registry[model_id]
        
        # Check language support
        if language and language not in model_config.languages:
            self.logger.error(f"Language {language} not supported by model {model_id}")
            return None
        
        # Check if already loaded
        cache_key = f"{model_id}_{language}" if language else model_id
        if cache_key in self.loaded_models:
            self.cache_usage[cache_key] = time.time()
            return self.loaded_models[cache_key]
        
        # Load model with thread safety
        with self._loading_locks[cache_key]:
            if cache_key in self.loaded_models:  # Double-check after acquiring lock
                return self.loaded_models[cache_key]
            
            try:
                model = self._load_model_by_format(model_config)
                if model is not None:
                    self.loaded_models[cache_key] = model
                    self.cache_usage[cache_key] = time.time()
                    self._manage_cache()
                    
                    self.logger.info(f"Loaded model: {cache_key}")
                    return model
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {cache_key}: {e}")
                return None
    
    def _load_model_by_format(self, model_config: ModelConfig) -> Optional[Any]:
        """Load model based on its format"""
        if model_config.model_format == ModelFormat.PYTORCH:
            return self._load_pytorch_model(model_config)
        elif model_config.model_format == ModelFormat.ONNX and HAS_ONNX:
            return self._load_onnx_model(model_config)
        elif model_config.model_format == ModelFormat.TENSORRT and HAS_TENSORRT:
            return self._load_tensorrt_model(model_config)
        else:
            self.logger.error(f"Unsupported model format: {model_config.model_format}")
            return None
    
    def _load_pytorch_model(self, model_config: ModelConfig) -> Optional[torch.nn.Module]:
        """Load PyTorch model"""
        try:
            model = torch.load(model_config.file_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            return None
    
    def _load_onnx_model(self, model_config: ModelConfig) -> Optional[Any]:
        """Load ONNX model"""
        try:
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(model_config.file_path, providers=providers)
            return session
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            return None
    
    def _load_tensorrt_model(self, model_config: ModelConfig) -> Optional[Any]:
        """Load TensorRT model"""
        # TensorRT implementation would go here
        self.logger.warning("TensorRT loading not implemented yet")
        return None
    
    def _manage_cache(self):
        """Manage model cache to stay within memory limits"""
        if len(self.loaded_models) <= self.config.max_concurrent_models:
            return
        
        # Sort by last access time and remove oldest
        sorted_models = sorted(
            self.cache_usage.items(),
            key=lambda x: x[1]
        )
        
        models_to_remove = len(self.loaded_models) - self.config.max_concurrent_models
        for cache_key, _ in sorted_models[:models_to_remove]:
            if cache_key in self.loaded_models:
                del self.loaded_models[cache_key]
                del self.cache_usage[cache_key]
                self.logger.info(f"Removed model from cache: {cache_key}")
    
    def get_supported_languages(self, model_type: Optional[ModelType] = None) -> List[str]:
        """Get list of supported languages, optionally filtered by model type"""
        if model_type is None:
            return self.config.supported_languages.copy()
        
        supported_langs = set()
        for model_config in self.model_registry.values():
            if model_config.model_type == model_type:
                supported_langs.update(model_config.languages)
        
        return list(supported_langs)
    
    def get_available_models(self, language: Optional[str] = None, 
                           model_type: Optional[ModelType] = None) -> List[ModelConfig]:
        """Get available models, optionally filtered by language and type"""
        models = []
        for model_config in self.model_registry.values():
            if language and language not in model_config.languages:
                continue
            if model_type and model_config.model_type != model_type:
                continue
            models.append(model_config)
        
        return models
    
    def unload_model(self, model_id: str, language: Optional[str] = None):
        """Unload a specific model from memory"""
        cache_key = f"{model_id}_{language}" if language else model_id
        
        with self._lock:
            if cache_key in self.loaded_models:
                del self.loaded_models[cache_key]
                if cache_key in self.cache_usage:
                    del self.cache_usage[cache_key]
                self.logger.info(f"Unloaded model: {cache_key}")
    
    def clear_cache(self):
        """Clear all loaded models from memory"""
        with self._lock:
            self.loaded_models.clear()
            self.cache_usage.clear()
            self.logger.info("Cleared all models from cache")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and resource usage"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_count': len(self.model_registry),
            'supported_languages': len(self.config.supported_languages),
            'hardware_info': self.hardware_info,
            'cache_usage': len(self.loaded_models),
            'max_cache': self.config.max_concurrent_models
        }
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_file = os.path.join(self.config.models_directory, "model_registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_id, config_data in data.items():
                        config_data['model_type'] = ModelType(config_data['model_type'])
                        config_data['model_format'] = ModelFormat(config_data['model_format'])
                        self.model_registry[model_id] = ModelConfig(**config_data)
                
                self.logger.info(f"Loaded {len(self.model_registry)} models from registry")
            except Exception as e:
                self.logger.error(f"Failed to load model registry: {e}")
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_file = os.path.join(self.config.models_directory, "model_registry.json")
        try:
            data = {}
            for model_id, config in self.model_registry.items():
                config_dict = {
                    'model_id': config.model_id,
                    'model_type': config.model_type.value,
                    'model_format': config.model_format.value,
                    'languages': config.languages,
                    'file_path': config.file_path,
                    'model_size': config.model_size,
                    'memory_requirement': config.memory_requirement,
                    'inference_time': config.inference_time,
                    'accuracy': config.accuracy,
                    'compression_ratio': config.compression_ratio,
                    'quantization_bits': config.quantization_bits,
                    'supports_gpu': config.supports_gpu,
                    'supports_cpu': config.supports_cpu,
                    'min_hardware_requirements': config.min_hardware_requirements,
                    'metadata': config.metadata
                }
                data[model_id] = config_dict
            
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize offline model manager
    config = OfflineSystemConfig()
    manager = OfflineModelManager(config)
    
    # Print system status
    status = manager.get_system_status()
    print("Offline Model Manager Status:")
    print(f"- Supported Languages: {status['supported_languages']}")
    print(f"- Hardware: {status['hardware_info']['device']}")
    print(f"- Available Models: {status['model_count']}")