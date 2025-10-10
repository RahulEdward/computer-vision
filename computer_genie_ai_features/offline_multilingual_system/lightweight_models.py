"""
Lightweight Offline Language Models
हल्के ऑफ़लाइन भाषा मॉडल

Efficient lightweight language models optimized for offline execution across 100+ languages.
Includes model compression, quantization, and adaptive loading strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import gzip
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import threading
import time
import hashlib
from collections import defaultdict, OrderedDict
import math

# Optional imports for model optimization
try:
    import torch.quantization as quant
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

try:
    from torch.jit import script, trace
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


class ModelSize(Enum):
    """Model size categories"""
    NANO = "nano"      # <1MB
    MICRO = "micro"    # 1-5MB
    SMALL = "small"    # 5-20MB
    MEDIUM = "medium"  # 20-100MB
    LARGE = "large"    # 100MB+


class ModelType(Enum):
    """Types of language models"""
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SENTIMENT = "sentiment"
    NER = "ner"  # Named Entity Recognition
    POS = "pos"  # Part of Speech
    SUMMARIZATION = "summarization"


class CompressionMethod(Enum):
    """Model compression methods"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    LOW_RANK = "low_rank"
    HUFFMAN = "huffman"


@dataclass
class ModelConfig:
    """Configuration for lightweight models"""
    model_id: str
    model_type: ModelType
    languages: List[str]
    size_category: ModelSize
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    compression_methods: List[CompressionMethod] = field(default_factory=list)
    quantization_bits: int = 8
    max_sequence_length: int = 512
    supports_streaming: bool = True
    memory_footprint_mb: float = 0.0


@dataclass
class ModelMetadata:
    """Metadata for model management"""
    model_id: str
    version: str
    created_at: float
    file_size_bytes: int
    checksum: str
    languages: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    compression_ratio: float = 1.0
    load_time_ms: float = 0.0
    inference_time_ms: float = 0.0


class CompactEmbedding(nn.Module):
    """Compact embedding layer with shared parameters"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_languages: int, shared_ratio: float = 0.8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_languages = num_languages
        
        # Shared embeddings across languages
        shared_dim = int(embedding_dim * shared_ratio)
        lang_specific_dim = embedding_dim - shared_dim
        
        self.shared_embeddings = nn.Embedding(vocab_size, shared_dim)
        self.lang_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, lang_specific_dim) 
            for _ in range(num_languages)
        ])
        
        # Language ID embedding
        self.lang_id_embedding = nn.Embedding(num_languages, embedding_dim)
    
    def forward(self, input_ids: torch.Tensor, language_id: int = 0):
        shared_emb = self.shared_embeddings(input_ids)
        lang_emb = self.lang_embeddings[language_id](input_ids)
        lang_id_emb = self.lang_id_embedding(torch.tensor(language_id))
        
        # Combine embeddings
        combined = torch.cat([shared_emb, lang_emb], dim=-1)
        combined = combined + lang_id_emb.unsqueeze(0).unsqueeze(0)
        
        return combined


class LightweightTransformer(nn.Module):
    """Lightweight transformer model for multilingual tasks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Compact embedding layer
        self.embeddings = CompactEmbedding(
            config.vocab_size, 
            config.embedding_dim,
            len(config.languages)
        )
        
        # Lightweight transformer layers
        self.layers = nn.ModuleList([
            LightweightTransformerLayer(config.embedding_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # Output heads for different tasks
        self.output_heads = nn.ModuleDict()
        self._init_output_heads()
    
    def _init_output_heads(self):
        """Initialize task-specific output heads"""
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.output_heads['classification'] = nn.Linear(self.config.embedding_dim, 2)
        elif self.config.model_type == ModelType.SENTIMENT:
            self.output_heads['sentiment'] = nn.Linear(self.config.embedding_dim, 3)
        elif self.config.model_type == ModelType.NER:
            self.output_heads['ner'] = nn.Linear(self.config.embedding_dim, 9)  # BIO tags
        elif self.config.model_type == ModelType.POS:
            self.output_heads['pos'] = nn.Linear(self.config.embedding_dim, 17)  # Universal POS tags
    
    def forward(self, input_ids: torch.Tensor, language_id: int = 0, 
                attention_mask: Optional[torch.Tensor] = None):
        # Embeddings
        x = self.embeddings(input_ids, language_id)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_task_output(self, hidden_states: torch.Tensor, task: str):
        """Get output for specific task"""
        if task in self.output_heads:
            return self.output_heads[task](hidden_states)
        return hidden_states


class LightweightTransformerLayer(nn.Module):
    """Lightweight transformer layer with efficient attention"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention with reduced complexity
        self.attention = LightweightAttention(embedding_dim, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class LightweightAttention(nn.Module):
    """Efficient attention mechanism with reduced complexity"""
    
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        
        return self.out_proj(attn_output)


class ModelCompressor:
    """Model compression utilities"""
    
    @staticmethod
    def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
        """Quantize model to reduce size"""
        if not HAS_QUANTIZATION:
            logging.warning("Quantization not available, returning original model")
            return model
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse modules if possible
        try:
            model = torch.quantization.fuse_modules(model, [['linear', 'relu']])
        except:
            pass
        
        # Prepare and convert
        model_prepared = torch.quantization.prepare(model)
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def prune_model(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Prune model weights to reduce size"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Simple magnitude-based pruning
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
        
        return model
    
    @staticmethod
    def compress_embeddings(embeddings: nn.Embedding, rank: int) -> nn.Module:
        """Compress embeddings using low-rank approximation"""
        weight = embeddings.weight.data
        U, S, V = torch.svd(weight)
        
        # Keep top-k singular values
        U_compressed = U[:, :rank]
        S_compressed = S[:rank]
        V_compressed = V[:, :rank]
        
        # Create compressed embedding
        compressed = nn.Sequential(
            nn.Linear(embeddings.num_embeddings, rank, bias=False),
            nn.Linear(rank, embeddings.embedding_dim, bias=False)
        )
        
        compressed[0].weight.data = (U_compressed * S_compressed).T
        compressed[1].weight.data = V_compressed.T
        
        return compressed


class LightweightModelManager:
    """
    Manager for lightweight offline language models
    
    Features:
    - Efficient model loading and caching
    - Dynamic model compression
    - Memory-aware model management
    - Language-specific model routing
    - Performance optimization
    """
    
    def __init__(self, models_dir: str = "lightweight_models", 
                 max_memory_mb: int = 500):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.max_memory_mb = max_memory_mb
        
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.loaded_models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        
        # Language mappings
        self.language_models: Dict[str, List[str]] = defaultdict(list)
        
        # Memory management
        self.model_usage: OrderedDict = OrderedDict()
        self.current_memory_mb = 0.0
        
        # Threading
        self._lock = threading.RLock()
        
        # Load existing models
        self._discover_models()
        
        self.logger.info(f"LightweightModelManager initialized with {len(self.model_configs)} models")
    
    def _discover_models(self):
        """Discover existing models in the models directory"""
        for model_file in self.models_dir.glob("*.json"):
            try:
                with open(model_file, 'r') as f:
                    config_data = json.load(f)
                
                config = ModelConfig(**config_data)
                self.model_configs[config.model_id] = config
                
                # Map languages to models
                for lang in config.languages:
                    self.language_models[lang].append(config.model_id)
                
                # Load metadata if available
                metadata_file = model_file.with_suffix('.metadata.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata_data = json.load(f)
                    self.model_metadata[config.model_id] = ModelMetadata(**metadata_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to load model config {model_file}: {e}")
    
    def create_model(self, config: ModelConfig) -> str:
        """Create a new lightweight model"""
        model = LightweightTransformer(config)
        
        # Apply compression if specified
        if CompressionMethod.QUANTIZATION in config.compression_methods:
            model = ModelCompressor.quantize_model(model, config.quantization_bits)
        
        if CompressionMethod.PRUNING in config.compression_methods:
            model = ModelCompressor.prune_model(model)
        
        # Save model
        model_path = self.models_dir / f"{config.model_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = self.models_dir / f"{config.model_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=config.model_id,
            version="1.0",
            created_at=time.time(),
            file_size_bytes=model_path.stat().st_size,
            checksum=self._calculate_checksum(model_path),
            languages=config.languages
        )
        
        # Save metadata
        metadata_path = self.models_dir / f"{config.model_id}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        # Update internal state
        with self._lock:
            self.model_configs[config.model_id] = config
            self.model_metadata[config.model_id] = metadata
            
            for lang in config.languages:
                self.language_models[lang].append(config.model_id)
        
        self.logger.info(f"Created model {config.model_id}")
        return config.model_id
    
    def load_model(self, model_id: str, force_reload: bool = False) -> Optional[nn.Module]:
        """Load a model into memory"""
        if model_id not in self.model_configs:
            self.logger.error(f"Model {model_id} not found")
            return None
        
        with self._lock:
            # Check if already loaded
            if model_id in self.loaded_models and not force_reload:
                self._update_usage(model_id)
                return self.loaded_models[model_id]
            
            # Check memory constraints
            config = self.model_configs[model_id]
            if not self._can_load_model(config):
                self._free_memory_for_model(config)
            
            # Load model
            start_time = time.time()
            model_path = self.models_dir / f"{model_id}.pt"
            
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            try:
                # Create model instance
                model = LightweightTransformer(config)
                
                # Load state dict
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                
                # Apply JIT compilation if available
                if HAS_JIT:
                    try:
                        model = torch.jit.script(model)
                    except:
                        pass  # JIT compilation failed, use regular model
                
                # Store in cache
                self.loaded_models[model_id] = model
                self.current_memory_mb += config.memory_footprint_mb
                self._update_usage(model_id)
                
                # Update metadata
                load_time = (time.time() - start_time) * 1000
                if model_id in self.model_metadata:
                    self.model_metadata[model_id].load_time_ms = load_time
                
                self.logger.info(f"Loaded model {model_id} in {load_time:.1f}ms")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {e}")
                return None
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                if model_id in self.model_configs:
                    self.current_memory_mb -= self.model_configs[model_id].memory_footprint_mb
                if model_id in self.model_usage:
                    del self.model_usage[model_id]
                self.logger.info(f"Unloaded model {model_id}")
    
    def get_model_for_language(self, language: str, model_type: ModelType) -> Optional[nn.Module]:
        """Get the best model for a specific language and task"""
        if language not in self.language_models:
            # Try fallback to English or similar languages
            fallback_langs = ['en', 'es', 'fr', 'de', 'zh']
            for fallback in fallback_langs:
                if fallback in self.language_models:
                    language = fallback
                    break
            else:
                self.logger.warning(f"No models available for language {language}")
                return None
        
        # Find best model for the task
        candidate_models = []
        for model_id in self.language_models[language]:
            config = self.model_configs[model_id]
            if config.model_type == model_type:
                candidate_models.append((model_id, config))
        
        if not candidate_models:
            self.logger.warning(f"No {model_type.value} models for language {language}")
            return None
        
        # Select best model (smallest size for now)
        best_model_id = min(candidate_models, key=lambda x: x[1].memory_footprint_mb)[0]
        
        return self.load_model(best_model_id)
    
    def _can_load_model(self, config: ModelConfig) -> bool:
        """Check if model can be loaded within memory constraints"""
        return (self.current_memory_mb + config.memory_footprint_mb) <= self.max_memory_mb
    
    def _free_memory_for_model(self, config: ModelConfig):
        """Free memory to make space for new model"""
        required_memory = config.memory_footprint_mb
        freed_memory = 0.0
        
        # Unload least recently used models
        while freed_memory < required_memory and self.model_usage:
            lru_model_id = next(iter(self.model_usage))
            lru_config = self.model_configs[lru_model_id]
            freed_memory += lru_config.memory_footprint_mb
            self.unload_model(lru_model_id)
    
    def _update_usage(self, model_id: str):
        """Update model usage for LRU tracking"""
        if model_id in self.model_usage:
            del self.model_usage[model_id]
        self.model_usage[model_id] = time.time()
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive model information"""
        if model_id not in self.model_configs:
            return None
        
        config = self.model_configs[model_id]
        metadata = self.model_metadata.get(model_id)
        
        info = {
            'model_id': model_id,
            'type': config.model_type.value,
            'languages': config.languages,
            'size_category': config.size_category.value,
            'memory_footprint_mb': config.memory_footprint_mb,
            'is_loaded': model_id in self.loaded_models,
            'compression_methods': [m.value for m in config.compression_methods]
        }
        
        if metadata:
            info.update({
                'file_size_bytes': metadata.file_size_bytes,
                'created_at': metadata.created_at,
                'load_time_ms': metadata.load_time_ms,
                'performance_metrics': metadata.performance_metrics
            })
        
        return info
    
    def list_models(self, language: Optional[str] = None, 
                   model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List available models with optional filtering"""
        models = []
        
        for model_id, config in self.model_configs.items():
            # Apply filters
            if language and language not in config.languages:
                continue
            if model_type and config.model_type != model_type:
                continue
            
            model_info = self.get_model_info(model_id)
            if model_info:
                models.append(model_info)
        
        return sorted(models, key=lambda x: x['memory_footprint_mb'])
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_models': len(self.model_configs),
            'loaded_models': len(self.loaded_models),
            'current_memory_mb': self.current_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': self.current_memory_mb / self.max_memory_mb,
            'supported_languages': len(self.language_models),
            'models_by_type': {
                model_type.value: len([
                    c for c in self.model_configs.values() 
                    if c.model_type == model_type
                ]) for model_type in ModelType
            }
        }
    
    def optimize_models(self):
        """Optimize loaded models for better performance"""
        for model_id, model in self.loaded_models.items():
            try:
                # Apply torch.jit optimization if not already done
                if not isinstance(model, torch.jit.ScriptModule) and HAS_JIT:
                    optimized = torch.jit.script(model)
                    self.loaded_models[model_id] = optimized
                    self.logger.info(f"Optimized model {model_id} with JIT")
            except Exception as e:
                self.logger.warning(f"Failed to optimize model {model_id}: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            self.loaded_models.clear()
            self.model_usage.clear()
            self.current_memory_mb = 0.0


# Example usage and model creation
def create_sample_models(manager: LightweightModelManager):
    """Create sample lightweight models for different languages and tasks"""
    
    # Multilingual sentiment analysis model
    sentiment_config = ModelConfig(
        model_id="multilingual_sentiment_nano",
        model_type=ModelType.SENTIMENT,
        languages=["en", "es", "fr", "de", "it", "pt", "hi", "zh", "ja", "ko"],
        size_category=ModelSize.NANO,
        vocab_size=8000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        compression_methods=[CompressionMethod.QUANTIZATION, CompressionMethod.PRUNING],
        memory_footprint_mb=0.8
    )
    
    # Multilingual NER model
    ner_config = ModelConfig(
        model_id="multilingual_ner_micro",
        model_type=ModelType.NER,
        languages=["en", "es", "fr", "de", "hi", "ar", "zh", "ja"],
        size_category=ModelSize.MICRO,
        vocab_size=12000,
        embedding_dim=192,
        hidden_dim=384,
        num_layers=3,
        compression_methods=[CompressionMethod.QUANTIZATION],
        memory_footprint_mb=2.5
    )
    
    # Language-specific embedding models
    embedding_configs = []
    major_languages = [
        ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"),
        ("hi", "Hindi"), ("ar", "Arabic"), ("zh", "Chinese"), ("ja", "Japanese"),
        ("ko", "Korean"), ("ru", "Russian"), ("pt", "Portuguese"), ("it", "Italian")
    ]
    
    for lang_code, lang_name in major_languages:
        config = ModelConfig(
            model_id=f"embedding_{lang_code}_nano",
            model_type=ModelType.EMBEDDING,
            languages=[lang_code],
            size_category=ModelSize.NANO,
            vocab_size=5000,
            embedding_dim=96,
            hidden_dim=192,
            num_layers=1,
            compression_methods=[CompressionMethod.QUANTIZATION, CompressionMethod.LOW_RANK],
            memory_footprint_mb=0.3
        )
        embedding_configs.append(config)
    
    # Create all models
    all_configs = [sentiment_config, ner_config] + embedding_configs
    
    for config in all_configs:
        try:
            manager.create_model(config)
        except Exception as e:
            logging.error(f"Failed to create model {config.model_id}: {e}")


if __name__ == "__main__":
    # Initialize model manager
    manager = LightweightModelManager(max_memory_mb=100)
    
    # Create sample models
    create_sample_models(manager)
    
    # Test model loading
    print("Available models:")
    models = manager.list_models()
    for model in models[:5]:  # Show first 5
        print(f"- {model['model_id']}: {model['languages']} ({model['memory_footprint_mb']:.1f}MB)")
    
    # Test language-specific model loading
    sentiment_model = manager.get_model_for_language("hi", ModelType.SENTIMENT)
    if sentiment_model:
        print("Successfully loaded Hindi sentiment model")
    
    # Get system stats
    stats = manager.get_system_stats()
    print(f"\nSystem Stats:")
    print(f"- Total models: {stats['total_models']}")
    print(f"- Memory usage: {stats['current_memory_mb']:.1f}/{stats['max_memory_mb']}MB")
    print(f"- Supported languages: {stats['supported_languages']}")
    
    # Cleanup
    manager.cleanup()