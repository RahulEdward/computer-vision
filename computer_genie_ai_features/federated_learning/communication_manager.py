"""
Communication Manager for Federated Learning
फेडरेटेड लर्निंग के लिए कम्युनिकेशन मैनेजर

Handles efficient communication, compression, and data transfer
between federated learning clients and server.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
import gzip
import pickle
import zlib
import lz4.frame
import brotli
from collections import defaultdict
import hashlib
import base64
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


class CompressionMethod(Enum):
    """Compression methods for model updates"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    BROTLI = "brotli"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"


class QuantizationMethod(Enum):
    """Quantization methods"""
    UNIFORM = "uniform"
    NON_UNIFORM = "non_uniform"
    STOCHASTIC = "stochastic"
    DYNAMIC = "dynamic"


@dataclass
class CommunicationConfig:
    """Configuration for communication manager"""
    # Compression settings
    compression_method: CompressionMethod = CompressionMethod.GZIP
    compression_level: int = 6
    enable_quantization: bool = True
    quantization_bits: int = 8
    quantization_method: QuantizationMethod = QuantizationMethod.UNIFORM
    
    # Sparsification settings
    enable_sparsification: bool = True
    sparsification_ratio: float = 0.1  # Keep top 10%
    sparsification_threshold: float = 1e-4
    
    # Low-rank approximation
    enable_low_rank: bool = False
    low_rank_ratio: float = 0.5
    
    # Communication optimization
    enable_delta_compression: bool = True
    enable_adaptive_compression: bool = True
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    
    # Error correction
    enable_error_correction: bool = True
    checksum_algorithm: str = "sha256"
    
    # Batching and buffering
    enable_batching: bool = True
    batch_size: int = 32
    buffer_size: int = 1024
    
    # Async communication
    enable_async: bool = True
    max_concurrent_transfers: int = 10
    timeout_seconds: int = 300
    
    # Monitoring
    enable_monitoring: bool = True
    log_compression_stats: bool = True


class CompressionStats:
    """Track compression statistics"""
    
    def __init__(self):
        self.total_original_size = 0
        self.total_compressed_size = 0
        self.compression_times = []
        self.decompression_times = []
        self.compression_ratios = []
        self.method_stats = defaultdict(lambda: {'count': 0, 'total_ratio': 0.0})
    
    def add_compression_stat(self, original_size: int, compressed_size: int,
                           compression_time: float, method: str) -> None:
        """Add compression statistics"""
        self.total_original_size += original_size
        self.total_compressed_size += compressed_size
        self.compression_times.append(compression_time)
        
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        self.compression_ratios.append(ratio)
        
        # Method-specific stats
        self.method_stats[method]['count'] += 1
        self.method_stats[method]['total_ratio'] += ratio
    
    def add_decompression_stat(self, decompression_time: float) -> None:
        """Add decompression statistics"""
        self.decompression_times.append(decompression_time)
    
    def get_average_compression_ratio(self) -> float:
        """Get average compression ratio"""
        return self.total_compressed_size / self.total_original_size if self.total_original_size > 0 else 1.0
    
    def get_average_compression_time(self) -> float:
        """Get average compression time"""
        return np.mean(self.compression_times) if self.compression_times else 0.0
    
    def get_average_decompression_time(self) -> float:
        """Get average decompression time"""
        return np.mean(self.decompression_times) if self.decompression_times else 0.0
    
    def get_method_performance(self, method: str) -> Dict[str, float]:
        """Get performance stats for specific method"""
        if method in self.method_stats:
            stats = self.method_stats[method]
            avg_ratio = stats['total_ratio'] / stats['count'] if stats['count'] > 0 else 1.0
            return {
                'count': stats['count'],
                'average_ratio': avg_ratio
            }
        return {'count': 0, 'average_ratio': 1.0}


class ModelQuantizer:
    """Quantize model parameters for compression"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def quantize_model(self, model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Quantize model parameters"""
        quantized_model = {}
        quantization_info = {}
        
        for name, param in model_state.items():
            if param.dtype == torch.float32 or param.dtype == torch.float64:
                quantized_param, quant_info = self._quantize_tensor(param, name)
                quantized_model[name] = quantized_param
                quantization_info[name] = quant_info
            else:
                # Keep non-float parameters as is
                quantized_model[name] = param
                quantization_info[name] = {'method': 'none'}
        
        return quantized_model, quantization_info
    
    def dequantize_model(self, quantized_model: Dict[str, Any], 
                        quantization_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Dequantize model parameters"""
        dequantized_model = {}
        
        for name, quantized_param in quantized_model.items():
            if name in quantization_info:
                quant_info = quantization_info[name]
                if quant_info['method'] != 'none':
                    dequantized_param = self._dequantize_tensor(quantized_param, quant_info)
                    dequantized_model[name] = dequantized_param
                else:
                    dequantized_model[name] = quantized_param
            else:
                dequantized_model[name] = quantized_param
        
        return dequantized_model
    
    def _quantize_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize a single tensor"""
        if self.config.quantization_method == QuantizationMethod.UNIFORM:
            return self._uniform_quantization(tensor)
        elif self.config.quantization_method == QuantizationMethod.STOCHASTIC:
            return self._stochastic_quantization(tensor)
        elif self.config.quantization_method == QuantizationMethod.DYNAMIC:
            return self._dynamic_quantization(tensor)
        else:
            return self._uniform_quantization(tensor)
    
    def _uniform_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Uniform quantization"""
        # Calculate quantization parameters
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Number of quantization levels
        num_levels = 2 ** self.config.quantization_bits
        
        # Quantization scale and zero point
        scale = (max_val - min_val) / (num_levels - 1)
        zero_point = -min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, num_levels - 1)
        
        # Convert to appropriate integer type
        if self.config.quantization_bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif self.config.quantization_bits <= 16:
            quantized = quantized.to(torch.int16)
        else:
            quantized = quantized.to(torch.int32)
        
        quant_info = {
            'method': 'uniform',
            'scale': scale,
            'zero_point': zero_point,
            'min_val': min_val,
            'max_val': max_val,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'bits': self.config.quantization_bits
        }
        
        return quantized, quant_info
    
    def _stochastic_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Stochastic quantization"""
        # Similar to uniform but with stochastic rounding
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        num_levels = 2 ** self.config.quantization_bits
        scale = (max_val - min_val) / (num_levels - 1)
        zero_point = -min_val / scale
        
        # Stochastic rounding
        continuous = tensor / scale + zero_point
        floor_vals = torch.floor(continuous)
        prob = continuous - floor_vals
        
        # Stochastic decision
        random_vals = torch.rand_like(prob)
        quantized = torch.where(random_vals < prob, floor_vals + 1, floor_vals)
        quantized = torch.clamp(quantized, 0, num_levels - 1)
        
        if self.config.quantization_bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif self.config.quantization_bits <= 16:
            quantized = quantized.to(torch.int16)
        else:
            quantized = quantized.to(torch.int32)
        
        quant_info = {
            'method': 'stochastic',
            'scale': scale,
            'zero_point': zero_point,
            'min_val': min_val,
            'max_val': max_val,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'bits': self.config.quantization_bits
        }
        
        return quantized, quant_info
    
    def _dynamic_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Dynamic quantization based on tensor statistics"""
        # Use percentiles for more robust quantization
        percentile_low = 1.0
        percentile_high = 99.0
        
        min_val = torch.quantile(tensor, percentile_low / 100.0).item()
        max_val = torch.quantile(tensor, percentile_high / 100.0).item()
        
        # Clamp outliers
        clamped_tensor = torch.clamp(tensor, min_val, max_val)
        
        # Apply uniform quantization to clamped tensor
        return self._uniform_quantization(clamped_tensor)
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor, 
                          quant_info: Dict[str, Any]) -> torch.Tensor:
        """Dequantize a tensor"""
        # Convert to float
        float_tensor = quantized_tensor.float()
        
        # Dequantize
        scale = quant_info['scale']
        zero_point = quant_info['zero_point']
        
        dequantized = (float_tensor - zero_point) * scale
        
        # Restore original shape and dtype
        original_shape = quant_info['shape']
        original_dtype = getattr(torch, quant_info['dtype'].split('.')[-1])
        
        return dequantized.reshape(original_shape).to(original_dtype)


class ModelSparsifier:
    """Sparsify model parameters for compression"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def sparsify_model(self, model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Sparsify model parameters"""
        sparse_model = {}
        sparsification_info = {}
        
        for name, param in model_state.items():
            sparse_param, sparse_info = self._sparsify_tensor(param, name)
            sparse_model[name] = sparse_param
            sparsification_info[name] = sparse_info
        
        return sparse_model, sparsification_info
    
    def densify_model(self, sparse_model: Dict[str, Any], 
                     sparsification_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert sparse model back to dense"""
        dense_model = {}
        
        for name, sparse_param in sparse_model.items():
            if name in sparsification_info:
                sparse_info = sparsification_info[name]
                dense_param = self._densify_tensor(sparse_param, sparse_info)
                dense_model[name] = dense_param
            else:
                dense_model[name] = sparse_param
        
        return dense_model
    
    def _sparsify_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Sparsify a single tensor"""
        # Calculate threshold for sparsification
        abs_tensor = torch.abs(tensor)
        
        if self.config.sparsification_ratio < 1.0:
            # Keep top k% of values
            k = int(tensor.numel() * self.config.sparsification_ratio)
            threshold = torch.topk(abs_tensor.flatten(), k).values[-1].item()
        else:
            threshold = self.config.sparsification_threshold
        
        # Create mask
        mask = abs_tensor >= threshold
        
        # Get sparse representation
        indices = torch.nonzero(mask, as_tuple=False)
        values = tensor[mask]
        
        sparse_param = {
            'indices': indices,
            'values': values,
            'density': mask.float().mean().item()
        }
        
        sparse_info = {
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'threshold': threshold,
            'original_size': tensor.numel(),
            'sparse_size': len(values)
        }
        
        return sparse_param, sparse_info
    
    def _densify_tensor(self, sparse_param: Dict[str, Any], 
                       sparse_info: Dict[str, Any]) -> torch.Tensor:
        """Convert sparse representation back to dense tensor"""
        shape = sparse_info['shape']
        dtype = getattr(torch, sparse_info['dtype'].split('.')[-1])
        
        # Create dense tensor
        dense_tensor = torch.zeros(shape, dtype=dtype)
        
        # Fill in sparse values
        indices = sparse_param['indices']
        values = sparse_param['values']
        
        if len(indices) > 0:
            # Convert indices to tuple for advanced indexing
            idx_tuple = tuple(indices.t())
            dense_tensor[idx_tuple] = values
        
        return dense_tensor


class LowRankApproximator:
    """Low-rank approximation for model compression"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def approximate_model(self, model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply low-rank approximation to model"""
        approximated_model = {}
        approximation_info = {}
        
        for name, param in model_state.items():
            if len(param.shape) >= 2:  # Only apply to matrices
                approx_param, approx_info = self._approximate_tensor(param, name)
                approximated_model[name] = approx_param
                approximation_info[name] = approx_info
            else:
                # Keep vectors as is
                approximated_model[name] = param
                approximation_info[name] = {'method': 'none'}
        
        return approximated_model, approximation_info
    
    def reconstruct_model(self, approximated_model: Dict[str, Any], 
                         approximation_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Reconstruct model from low-rank approximation"""
        reconstructed_model = {}
        
        for name, approx_param in approximated_model.items():
            if name in approximation_info:
                approx_info = approximation_info[name]
                if approx_info['method'] != 'none':
                    reconstructed_param = self._reconstruct_tensor(approx_param, approx_info)
                    reconstructed_model[name] = reconstructed_param
                else:
                    reconstructed_model[name] = approx_param
            else:
                reconstructed_model[name] = approx_param
        
        return reconstructed_model
    
    def _approximate_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Apply SVD-based low-rank approximation"""
        # Reshape to 2D if needed
        original_shape = tensor.shape
        if len(tensor.shape) > 2:
            tensor_2d = tensor.reshape(tensor.shape[0], -1)
        else:
            tensor_2d = tensor
        
        # Perform SVD
        U, S, V = torch.svd(tensor_2d)
        
        # Determine rank
        total_rank = min(tensor_2d.shape)
        target_rank = int(total_rank * self.config.low_rank_ratio)
        target_rank = max(1, min(target_rank, total_rank))
        
        # Truncate
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        V_truncated = V[:, :target_rank]
        
        approx_param = {
            'U': U_truncated,
            'S': S_truncated,
            'V': V_truncated
        }
        
        approx_info = {
            'method': 'svd',
            'original_shape': original_shape,
            'original_rank': total_rank,
            'target_rank': target_rank,
            'dtype': str(tensor.dtype)
        }
        
        return approx_param, approx_info
    
    def _reconstruct_tensor(self, approx_param: Dict[str, torch.Tensor], 
                           approx_info: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct tensor from low-rank approximation"""
        U = approx_param['U']
        S = approx_param['S']
        V = approx_param['V']
        
        # Reconstruct
        reconstructed_2d = U @ torch.diag(S) @ V.t()
        
        # Reshape to original shape
        original_shape = approx_info['original_shape']
        reconstructed = reconstructed_2d.reshape(original_shape)
        
        # Restore dtype
        original_dtype = getattr(torch, approx_info['dtype'].split('.')[-1])
        return reconstructed.to(original_dtype)


class CommunicationManager:
    """Main communication manager"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.quantizer = ModelQuantizer(config) if config.enable_quantization else None
        self.sparsifier = ModelSparsifier(config) if config.enable_sparsification else None
        self.low_rank_approximator = LowRankApproximator(config) if config.enable_low_rank else None
        
        # Statistics
        self.stats = CompressionStats()
        
        # Delta compression state
        self.previous_model = None
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_transfers)
    
    def compress_model(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress model for transmission"""
        start_time = time.time()
        
        # Start with original model
        compressed_data = {'model': model_state}
        compression_info = {'methods': []}
        
        # Apply delta compression if enabled
        if self.config.enable_delta_compression and self.previous_model is not None:
            delta_model = self._compute_delta(model_state, self.previous_model)
            compressed_data['model'] = delta_model
            compression_info['methods'].append('delta')
            compression_info['is_delta'] = True
        else:
            compression_info['is_delta'] = False
        
        # Apply quantization
        if self.quantizer:
            quantized_model, quant_info = self.quantizer.quantize_model(compressed_data['model'])
            compressed_data['model'] = quantized_model
            compressed_data['quantization_info'] = quant_info
            compression_info['methods'].append('quantization')
        
        # Apply sparsification
        if self.sparsifier:
            sparse_model, sparse_info = self.sparsifier.sparsify_model(compressed_data['model'])
            compressed_data['model'] = sparse_model
            compressed_data['sparsification_info'] = sparse_info
            compression_info['methods'].append('sparsification')
        
        # Apply low-rank approximation
        if self.low_rank_approximator:
            approx_model, approx_info = self.low_rank_approximator.approximate_model(compressed_data['model'])
            compressed_data['model'] = approx_model
            compressed_data['approximation_info'] = approx_info
            compression_info['methods'].append('low_rank')
        
        # Apply standard compression
        serialized_data = pickle.dumps(compressed_data)
        original_size = len(serialized_data)
        
        if self.config.compression_method != CompressionMethod.NONE:
            compressed_bytes = self._apply_standard_compression(serialized_data)
            compression_info['methods'].append(self.config.compression_method.value)
        else:
            compressed_bytes = serialized_data
        
        compressed_size = len(compressed_bytes)
        compression_time = time.time() - start_time
        
        # Update statistics
        self.stats.add_compression_stat(
            original_size, compressed_size, compression_time,
            '+'.join(compression_info['methods'])
        )
        
        # Store for delta compression
        self.previous_model = model_state.copy()
        
        # Prepare final compressed data
        final_data = {
            'compressed_model': base64.b64encode(compressed_bytes).decode('utf-8'),
            'compression_info': compression_info,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'checksum': self._calculate_checksum(compressed_bytes)
        }
        
        if self.config.log_compression_stats:
            ratio = compressed_size / original_size
            self.logger.info(f"Model compressed: {original_size} -> {compressed_size} bytes (ratio: {ratio:.3f})")
        
        return final_data
    
    def decompress_model(self, compressed_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress model from transmission"""
        start_time = time.time()
        
        # Extract compressed bytes
        compressed_bytes = base64.b64decode(compressed_data['compressed_model'].encode('utf-8'))
        compression_info = compressed_data['compression_info']
        
        # Verify checksum
        if self.config.enable_error_correction:
            expected_checksum = compressed_data.get('checksum')
            actual_checksum = self._calculate_checksum(compressed_bytes)
            if expected_checksum != actual_checksum:
                raise ValueError("Checksum mismatch - data corruption detected")
        
        # Apply standard decompression
        if self.config.compression_method.value in compression_info['methods']:
            decompressed_bytes = self._apply_standard_decompression(compressed_bytes)
        else:
            decompressed_bytes = compressed_bytes
        
        # Deserialize
        decompressed_data = pickle.loads(decompressed_bytes)
        model_state = decompressed_data['model']
        
        # Reverse low-rank approximation
        if 'low_rank' in compression_info['methods']:
            approx_info = decompressed_data['approximation_info']
            model_state = self.low_rank_approximator.reconstruct_model(model_state, approx_info)
        
        # Reverse sparsification
        if 'sparsification' in compression_info['methods']:
            sparse_info = decompressed_data['sparsification_info']
            model_state = self.sparsifier.densify_model(model_state, sparse_info)
        
        # Reverse quantization
        if 'quantization' in compression_info['methods']:
            quant_info = decompressed_data['quantization_info']
            model_state = self.quantizer.dequantize_model(model_state, quant_info)
        
        # Reverse delta compression
        if compression_info.get('is_delta', False) and self.previous_model is not None:
            model_state = self._apply_delta(model_state, self.previous_model)
        
        decompression_time = time.time() - start_time
        self.stats.add_decompression_stat(decompression_time)
        
        return model_state
    
    def _compute_delta(self, current_model: Dict[str, torch.Tensor], 
                      previous_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute delta between current and previous model"""
        delta_model = {}
        
        for name, current_param in current_model.items():
            if name in previous_model:
                delta_model[name] = current_param - previous_model[name]
            else:
                delta_model[name] = current_param
        
        return delta_model
    
    def _apply_delta(self, delta_model: Dict[str, torch.Tensor], 
                    previous_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply delta to previous model"""
        current_model = {}
        
        for name, delta_param in delta_model.items():
            if name in previous_model:
                current_model[name] = previous_model[name] + delta_param
            else:
                current_model[name] = delta_param
        
        return current_model
    
    def _apply_standard_compression(self, data: bytes) -> bytes:
        """Apply standard compression algorithm"""
        if self.config.compression_method == CompressionMethod.GZIP:
            return gzip.compress(data, compresslevel=self.config.compression_level)
        elif self.config.compression_method == CompressionMethod.ZLIB:
            return zlib.compress(data, level=self.config.compression_level)
        elif self.config.compression_method == CompressionMethod.LZ4:
            return lz4.frame.compress(data, compression_level=self.config.compression_level)
        elif self.config.compression_method == CompressionMethod.BROTLI:
            return brotli.compress(data, quality=self.config.compression_level)
        else:
            return data
    
    def _apply_standard_decompression(self, compressed_data: bytes) -> bytes:
        """Apply standard decompression algorithm"""
        if self.config.compression_method == CompressionMethod.GZIP:
            return gzip.decompress(compressed_data)
        elif self.config.compression_method == CompressionMethod.ZLIB:
            return zlib.decompress(compressed_data)
        elif self.config.compression_method == CompressionMethod.LZ4:
            return lz4.frame.decompress(compressed_data)
        elif self.config.compression_method == CompressionMethod.BROTLI:
            return brotli.decompress(compressed_data)
        else:
            return compressed_data
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for error detection"""
        if self.config.checksum_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.config.checksum_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            'average_compression_ratio': self.stats.get_average_compression_ratio(),
            'average_compression_time': self.stats.get_average_compression_time(),
            'average_decompression_time': self.stats.get_average_decompression_time(),
            'total_original_size': self.stats.total_original_size,
            'total_compressed_size': self.stats.total_compressed_size,
            'total_savings': self.stats.total_original_size - self.stats.total_compressed_size,
            'method_stats': dict(self.stats.method_stats)
        }
    
    def optimize_compression_settings(self) -> None:
        """Automatically optimize compression settings based on performance"""
        if not self.config.enable_adaptive_compression:
            return
        
        # Analyze recent performance
        recent_ratios = self.stats.compression_ratios[-10:] if len(self.stats.compression_ratios) >= 10 else []
        recent_times = self.stats.compression_times[-10:] if len(self.stats.compression_times) >= 10 else []
        
        if recent_ratios and recent_times:
            avg_ratio = np.mean(recent_ratios)
            avg_time = np.mean(recent_times)
            
            # Adjust compression level based on performance
            if avg_ratio > 0.8 and avg_time > 1.0:  # Poor compression, slow
                self.config.compression_level = max(1, self.config.compression_level - 1)
            elif avg_ratio < 0.5 and avg_time < 0.1:  # Good compression, fast
                self.config.compression_level = min(9, self.config.compression_level + 1)
            
            self.logger.debug(f"Adjusted compression level to {self.config.compression_level}")


# Example usage
if __name__ == "__main__":
    # Create communication configuration
    config = CommunicationConfig(
        compression_method=CompressionMethod.GZIP,
        enable_quantization=True,
        enable_sparsification=True,
        enable_delta_compression=True
    )
    
    # Create communication manager
    comm_manager = CommunicationManager(config)
    
    print("Communication manager created successfully!")
    print(f"Configuration: {config}")
    
    # Test compression with dummy model
    dummy_model = {
        'layer1.weight': torch.randn(100, 50),
        'layer1.bias': torch.randn(100),
        'layer2.weight': torch.randn(10, 100),
        'layer2.bias': torch.randn(10)
    }
    
    # Compress model
    compressed_data = comm_manager.compress_model(dummy_model)
    print(f"Original size: {compressed_data['original_size']} bytes")
    print(f"Compressed size: {compressed_data['compressed_size']} bytes")
    print(f"Compression ratio: {compressed_data['compressed_size'] / compressed_data['original_size']:.3f}")
    
    # Decompress model
    decompressed_model = comm_manager.decompress_model(compressed_data)
    print("Model decompressed successfully!")
    
    # Check compression stats
    stats = comm_manager.get_compression_stats()
    print(f"Compression statistics: {stats}")