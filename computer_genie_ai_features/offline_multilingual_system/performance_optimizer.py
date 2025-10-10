"""
Performance Optimization for Offline Execution
ऑफ़लाइन निष्पादन के लिए प्रदर्शन अनुकूलन

Advanced performance optimization system for running multilingual AI models
efficiently on various hardware configurations offline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
import platform
import json
import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import gc
import warnings
from collections import defaultdict, deque
import math

# Optional imports for hardware optimization
try:
    import torch.backends.cudnn as cudnn
    HAS_CUDNN = True
except ImportError:
    HAS_CUDNN = False

try:
    import torch.backends.mkldnn as mkldnn
    HAS_MKLDNN = True
except ImportError:
    HAS_MKLDNN = False

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


class HardwareType(Enum):
    """Hardware types for optimization"""
    CPU_INTEL = "cpu_intel"
    CPU_AMD = "cpu_amd"
    CPU_ARM = "cpu_arm"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    GPU_INTEL = "gpu_intel"
    MOBILE = "mobile"
    EDGE = "edge"
    EMBEDDED = "embedded"


class OptimizationLevel(Enum):
    """Optimization levels"""
    CONSERVATIVE = "conservative"  # Minimal optimizations, maximum compatibility
    BALANCED = "balanced"         # Balance between performance and compatibility
    AGGRESSIVE = "aggressive"     # Maximum performance, may reduce compatibility
    CUSTOM = "custom"            # User-defined optimizations


class ExecutionMode(Enum):
    """Execution modes"""
    INFERENCE = "inference"
    TRAINING = "training"
    STREAMING = "streaming"
    BATCH = "batch"
    REAL_TIME = "real_time"


class MemoryStrategy(Enum):
    """Memory management strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    STREAMING = "streaming"


@dataclass
class HardwareProfile:
    """Hardware profile for optimization"""
    hardware_type: HardwareType
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_gb: float
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: str = ""
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_avx: bool = False
    supports_avx512: bool = False
    cache_size_mb: int = 0
    memory_bandwidth_gbps: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    optimization_level: OptimizationLevel
    execution_mode: ExecutionMode
    memory_strategy: MemoryStrategy
    target_latency_ms: float = 100.0
    target_throughput: float = 10.0  # inferences per second
    max_memory_usage_mb: int = 1024
    enable_mixed_precision: bool = True
    enable_graph_optimization: bool = True
    enable_kernel_fusion: bool = True
    enable_memory_pooling: bool = True
    batch_size: int = 1
    num_threads: int = 0  # 0 = auto-detect
    use_gpu: bool = False
    gpu_device_id: int = 0


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    latency_ms: float
    throughput_fps: float
    memory_usage_mb: float
    cpu_utilization: float
    gpu_utilization: float = 0.0
    energy_consumption_watts: float = 0.0
    model_accuracy: float = 0.0
    optimization_overhead_ms: float = 0.0


class HardwareDetector:
    """Automatic hardware detection and profiling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_hardware(self) -> HardwareProfile:
        """Detect and profile current hardware"""
        # CPU information
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()
        gpu_info = self._get_gpu_info()
        
        # Determine hardware type
        hardware_type = self._determine_hardware_type(cpu_info)
        
        profile = HardwareProfile(
            hardware_type=hardware_type,
            cpu_cores=cpu_info['cores'],
            cpu_frequency=cpu_info['frequency'],
            memory_gb=memory_info['total_gb'],
            gpu_memory_gb=gpu_info.get('memory_gb', 0.0),
            gpu_compute_capability=gpu_info.get('compute_capability', ''),
            supports_fp16=self._check_fp16_support(),
            supports_int8=self._check_int8_support(),
            supports_avx=self._check_avx_support(),
            supports_avx512=self._check_avx512_support(),
            cache_size_mb=cpu_info.get('cache_mb', 0),
            memory_bandwidth_gbps=memory_info.get('bandwidth_gbps', 0.0)
        )
        
        self.logger.info(f"Detected hardware: {hardware_type.value}")
        return profile
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            
            return {
                'cores': cpu_count,
                'frequency': cpu_freq.max / 1000 if cpu_freq else 2.0,  # Convert to GHz
                'brand': platform.processor(),
                'architecture': platform.machine()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get CPU info: {e}")
            return {'cores': 4, 'frequency': 2.0, 'brand': 'Unknown', 'architecture': 'x86_64'}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory info: {e}")
            return {'total_gb': 8.0, 'available_gb': 4.0}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        gpu_info = {}
        
        if torch.cuda.is_available():
            try:
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info['compute_capability'] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
                gpu_info['name'] = torch.cuda.get_device_name(0)
            except Exception as e:
                self.logger.warning(f"Failed to get GPU info: {e}")
        
        return gpu_info
    
    def _determine_hardware_type(self, cpu_info: Dict[str, Any]) -> HardwareType:
        """Determine hardware type from CPU info"""
        brand = cpu_info.get('brand', '').lower()
        arch = cpu_info.get('architecture', '').lower()
        
        if torch.cuda.is_available():
            return HardwareType.GPU_NVIDIA
        elif 'intel' in brand:
            return HardwareType.CPU_INTEL
        elif 'amd' in brand:
            return HardwareType.CPU_AMD
        elif 'arm' in arch or 'aarch64' in arch:
            return HardwareType.CPU_ARM
        else:
            return HardwareType.CPU_INTEL  # Default fallback
    
    def _check_fp16_support(self) -> bool:
        """Check if hardware supports FP16"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)[0] >= 7  # Volta and newer
        return False
    
    def _check_int8_support(self) -> bool:
        """Check if hardware supports INT8"""
        return torch.backends.quantized.engine in ['fbgemm', 'qnnpack']
    
    def _check_avx_support(self) -> bool:
        """Check if CPU supports AVX instructions"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx' in info.get('flags', [])
        except:
            return True  # Assume support for modern CPUs
    
    def _check_avx512_support(self) -> bool:
        """Check if CPU supports AVX-512 instructions"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx512f' in info.get('flags', [])
        except:
            return False


class ModelOptimizer:
    """Model-level optimizations for different hardware"""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.hardware_profile = hardware_profile
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self, model: nn.Module, 
                      config: OptimizationConfig) -> nn.Module:
        """Apply comprehensive model optimizations"""
        optimized_model = model
        
        # Apply optimizations based on hardware and config
        if config.enable_mixed_precision and self.hardware_profile.supports_fp16:
            optimized_model = self._apply_mixed_precision(optimized_model)
        
        if config.enable_graph_optimization:
            optimized_model = self._apply_graph_optimization(optimized_model)
        
        if config.enable_kernel_fusion:
            optimized_model = self._apply_kernel_fusion(optimized_model)
        
        # Hardware-specific optimizations
        optimized_model = self._apply_hardware_optimizations(optimized_model, config)
        
        return optimized_model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision optimization"""
        if self.hardware_profile.supports_fp16:
            model = model.half()
            self.logger.info("Applied FP16 mixed precision")
        return model
    
    def _apply_graph_optimization(self, model: nn.Module) -> nn.Module:
        """Apply graph-level optimizations"""
        try:
            # TorchScript optimization
            model.eval()
            example_input = torch.randn(1, 512)  # Adjust based on model
            traced_model = torch.jit.trace(model, example_input)
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            self.logger.info("Applied graph optimization")
            return optimized_model
        except Exception as e:
            self.logger.warning(f"Graph optimization failed: {e}")
            return model
    
    def _apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Apply kernel fusion optimizations"""
        try:
            # Fuse common patterns
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    # Fuse Conv-BN-ReLU patterns
                    fused_module = self._fuse_conv_bn_relu(module)
                    if fused_module is not module:
                        setattr(model, name, fused_module)
            
            self.logger.info("Applied kernel fusion")
        except Exception as e:
            self.logger.warning(f"Kernel fusion failed: {e}")
        
        return model
    
    def _fuse_conv_bn_relu(self, module: nn.Sequential) -> nn.Module:
        """Fuse Conv-BatchNorm-ReLU patterns"""
        # Simplified fusion - in practice, use torch.quantization.fuse_modules
        return module
    
    def _apply_hardware_optimizations(self, model: nn.Module, 
                                    config: OptimizationConfig) -> nn.Module:
        """Apply hardware-specific optimizations"""
        if self.hardware_profile.hardware_type == HardwareType.CPU_INTEL:
            return self._optimize_for_intel_cpu(model, config)
        elif self.hardware_profile.hardware_type == HardwareType.GPU_NVIDIA:
            return self._optimize_for_nvidia_gpu(model, config)
        elif self.hardware_profile.hardware_type == HardwareType.CPU_ARM:
            return self._optimize_for_arm_cpu(model, config)
        else:
            return model
    
    def _optimize_for_intel_cpu(self, model: nn.Module, 
                               config: OptimizationConfig) -> nn.Module:
        """Optimize for Intel CPU"""
        try:
            if HAS_IPEX:
                model = ipex.optimize(model)
                self.logger.info("Applied Intel Extension optimizations")
            
            if HAS_MKLDNN:
                torch.backends.mkldnn.enabled = True
                self.logger.info("Enabled MKL-DNN backend")
        
        except Exception as e:
            self.logger.warning(f"Intel CPU optimization failed: {e}")
        
        return model
    
    def _optimize_for_nvidia_gpu(self, model: nn.Module, 
                                config: OptimizationConfig) -> nn.Module:
        """Optimize for NVIDIA GPU"""
        try:
            if HAS_CUDNN:
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.logger.info("Optimized cuDNN settings")
            
            # Move model to GPU
            if config.use_gpu:
                model = model.cuda(config.gpu_device_id)
        
        except Exception as e:
            self.logger.warning(f"NVIDIA GPU optimization failed: {e}")
        
        return model
    
    def _optimize_for_arm_cpu(self, model: nn.Module, 
                             config: OptimizationConfig) -> nn.Module:
        """Optimize for ARM CPU"""
        try:
            # ARM-specific optimizations
            torch.set_num_threads(min(4, self.hardware_profile.cpu_cores))
            self.logger.info("Applied ARM CPU optimizations")
        
        except Exception as e:
            self.logger.warning(f"ARM CPU optimization failed: {e}")
        
        return model


class MemoryManager:
    """Advanced memory management for offline execution"""
    
    def __init__(self, strategy: MemoryStrategy, max_memory_mb: int):
        self.strategy = strategy
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.allocated_memory = 0
        self.peak_memory = 0
        self.memory_pool = {}
        
        # Memory optimization settings
        self._configure_memory_settings()
    
    def _configure_memory_settings(self):
        """Configure memory optimization settings"""
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            # Aggressive memory optimization
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory-efficient attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
        
        elif self.strategy == MemoryStrategy.CONSERVATIVE:
            # Conservative memory usage
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    
    def allocate_tensor(self, shape: Tuple[int, ...], 
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor with memory management"""
        size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        size_mb = size_bytes / (1024 * 1024)
        
        # Check memory constraints
        if self.allocated_memory + size_mb > self.max_memory_mb:
            self._free_memory(size_mb)
        
        # Allocate tensor
        tensor = torch.empty(shape, dtype=dtype)
        self.allocated_memory += size_mb
        self.peak_memory = max(self.peak_memory, self.allocated_memory)
        
        return tensor
    
    def _free_memory(self, required_mb: float):
        """Free memory to meet requirements"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        # Update allocated memory tracking
        current_memory = self._get_current_memory_usage()
        self.allocated_memory = current_memory
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize memory layout for better cache performance"""
        # Reorder parameters for better memory locality
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Ensure contiguous memory layout
                module.weight.data = module.weight.data.contiguous()
            
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.contiguous()
        
        return model
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing to reduce memory usage"""
        try:
            from torch.utils.checkpoint import checkpoint
            
            # Apply checkpointing to transformer layers
            for name, module in model.named_modules():
                if 'transformer' in name.lower() or 'layer' in name.lower():
                    # Wrap module with checkpointing
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(original_forward, *args, **kwargs)
                    
                    module.forward = checkpointed_forward
            
            self.logger.info("Enabled gradient checkpointing")
        
        except Exception as e:
            self.logger.warning(f"Gradient checkpointing failed: {e}")
        
        return model


class InferenceEngine:
    """Optimized inference engine for offline execution"""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig,
                 hardware_profile: HardwareProfile):
        self.model = model
        self.config = config
        self.hardware_profile = hardware_profile
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.memory_manager = MemoryManager(
            config.memory_strategy, 
            config.max_memory_usage_mb
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics(0, 0, 0, 0)
        
        # Threading setup
        self._setup_threading()
        
        # Warm up model
        self._warmup_model()
    
    def _setup_threading(self):
        """Setup optimal threading configuration"""
        if self.config.num_threads == 0:
            # Auto-detect optimal thread count
            if self.hardware_profile.hardware_type in [HardwareType.CPU_INTEL, HardwareType.CPU_AMD]:
                num_threads = min(self.hardware_profile.cpu_cores, 8)
            else:
                num_threads = min(self.hardware_profile.cpu_cores, 4)
        else:
            num_threads = self.config.num_threads
        
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        self.logger.info(f"Set thread count to {num_threads}")
    
    def _warmup_model(self):
        """Warm up model for optimal performance"""
        self.model.eval()
        
        # Create dummy input for warmup
        dummy_input = torch.randn(
            self.config.batch_size, 
            512,  # Adjust based on model input size
            dtype=torch.float16 if self.hardware_profile.supports_fp16 else torch.float32
        )
        
        if self.config.use_gpu and torch.cuda.is_available():
            dummy_input = dummy_input.cuda(self.config.gpu_device_id)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        
        # Clear cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model warmup completed")
    
    def infer(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, PerformanceMetrics]:
        """Perform optimized inference"""
        start_time = time.time()
        
        # Prepare input
        if self.config.use_gpu and torch.cuda.is_available():
            input_data = input_data.cuda(self.config.gpu_device_id)
        
        # Memory tracking
        memory_before = self.memory_manager._get_current_memory_usage()
        
        # Inference
        with torch.no_grad():
            if self.hardware_profile.supports_fp16 and self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(input_data)
            else:
                output = self.model(input_data)
        
        # Calculate metrics
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        memory_after = self.memory_manager._get_current_memory_usage()
        memory_used = memory_after - memory_before
        
        metrics = PerformanceMetrics(
            latency_ms=inference_time,
            throughput_fps=1000 / inference_time if inference_time > 0 else 0,
            memory_usage_mb=memory_used,
            cpu_utilization=psutil.cpu_percent(),
            gpu_utilization=self._get_gpu_utilization()
        )
        
        # Update performance history
        self.performance_history.append(metrics)
        self.current_metrics = metrics
        
        return output, metrics
    
    def batch_infer(self, input_batch: List[torch.Tensor]) -> Tuple[List[torch.Tensor], PerformanceMetrics]:
        """Perform batch inference with optimization"""
        # Stack inputs into batch
        batch_tensor = torch.stack(input_batch)
        
        # Perform inference
        output, metrics = self.infer(batch_tensor)
        
        # Split output back to list
        output_list = torch.unbind(output, dim=0)
        
        # Adjust metrics for batch
        metrics.throughput_fps *= len(input_batch)
        
        return list(output_list), metrics
    
    def stream_infer(self, input_stream: Iterator[torch.Tensor]) -> Iterator[Tuple[torch.Tensor, PerformanceMetrics]]:
        """Perform streaming inference"""
        for input_data in input_stream:
            yield self.infer(input_data)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.config.gpu_device_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return utilization.gpu
            except:
                return 0.0
        return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.performance_history:
            return {}
        
        latencies = [m.latency_ms for m in self.performance_history]
        throughputs = [m.throughput_fps for m in self.performance_history]
        memory_usage = [m.memory_usage_mb for m in self.performance_history]
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_throughput_fps': np.mean(throughputs),
            'max_throughput_fps': np.max(throughputs),
            'avg_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage),
            'total_inferences': len(self.performance_history)
        }


class PerformanceOptimizer:
    """
    Comprehensive performance optimization system
    
    Features:
    - Automatic hardware detection and profiling
    - Model optimization for different hardware
    - Memory management and optimization
    - Real-time performance monitoring
    - Adaptive optimization based on workload
    """
    
    def __init__(self, models_dir: str = "optimized_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.hardware_detector = HardwareDetector()
        self.hardware_profile = self.hardware_detector.detect_hardware()
        self.model_optimizer = ModelOptimizer(self.hardware_profile)
        
        # Optimization cache
        self.optimization_cache = {}
        self.performance_profiles = {}
        
        self.logger.info("PerformanceOptimizer initialized")
    
    def optimize_model_for_hardware(self, model: nn.Module, 
                                   config: OptimizationConfig) -> nn.Module:
        """Optimize model for current hardware"""
        # Check cache first
        cache_key = self._get_cache_key(model, config)
        if cache_key in self.optimization_cache:
            self.logger.info("Using cached optimization")
            return self.optimization_cache[cache_key]
        
        # Apply optimizations
        optimized_model = self.model_optimizer.optimize_model(model, config)
        
        # Cache result
        self.optimization_cache[cache_key] = optimized_model
        
        return optimized_model
    
    def create_inference_engine(self, model: nn.Module, 
                               config: OptimizationConfig) -> InferenceEngine:
        """Create optimized inference engine"""
        optimized_model = self.optimize_model_for_hardware(model, config)
        return InferenceEngine(optimized_model, config, self.hardware_profile)
    
    def benchmark_model(self, model: nn.Module, 
                       input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance"""
        # Test different configurations
        configs = [
            OptimizationConfig(OptimizationLevel.CONSERVATIVE, ExecutionMode.INFERENCE, MemoryStrategy.CONSERVATIVE),
            OptimizationConfig(OptimizationLevel.BALANCED, ExecutionMode.INFERENCE, MemoryStrategy.BALANCED),
            OptimizationConfig(OptimizationLevel.AGGRESSIVE, ExecutionMode.INFERENCE, MemoryStrategy.AGGRESSIVE)
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            try:
                # Create inference engine
                engine = self.create_inference_engine(model, config)
                
                # Benchmark
                dummy_input = torch.randn(input_shape)
                latencies = []
                
                for _ in range(num_iterations):
                    _, metrics = engine.infer(dummy_input)
                    latencies.append(metrics.latency_ms)
                
                results[config.optimization_level.value] = {
                    'avg_latency_ms': np.mean(latencies),
                    'std_latency_ms': np.std(latencies),
                    'min_latency_ms': np.min(latencies),
                    'max_latency_ms': np.max(latencies),
                    'throughput_fps': 1000 / np.mean(latencies)
                }
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {config.optimization_level.value}: {e}")
        
        return results
    
    def auto_tune_config(self, model: nn.Module, 
                        input_shape: Tuple[int, ...],
                        target_latency_ms: float = 100.0) -> OptimizationConfig:
        """Automatically tune optimization configuration"""
        best_config = None
        best_latency = float('inf')
        
        # Test different configurations
        test_configs = self._generate_test_configs()
        
        for config in test_configs:
            try:
                engine = self.create_inference_engine(model, config)
                dummy_input = torch.randn(input_shape)
                
                # Test performance
                _, metrics = engine.infer(dummy_input)
                
                if metrics.latency_ms < target_latency_ms and metrics.latency_ms < best_latency:
                    best_latency = metrics.latency_ms
                    best_config = config
                
            except Exception as e:
                self.logger.warning(f"Config test failed: {e}")
        
        if best_config is None:
            # Fallback to conservative config
            best_config = OptimizationConfig(
                OptimizationLevel.CONSERVATIVE,
                ExecutionMode.INFERENCE,
                MemoryStrategy.CONSERVATIVE
            )
        
        self.logger.info(f"Auto-tuned config: {best_config.optimization_level.value}")
        return best_config
    
    def _generate_test_configs(self) -> List[OptimizationConfig]:
        """Generate test configurations for auto-tuning"""
        configs = []
        
        for opt_level in OptimizationLevel:
            if opt_level == OptimizationLevel.CUSTOM:
                continue
            
            for mem_strategy in MemoryStrategy:
                config = OptimizationConfig(
                    optimization_level=opt_level,
                    execution_mode=ExecutionMode.INFERENCE,
                    memory_strategy=mem_strategy,
                    enable_mixed_precision=self.hardware_profile.supports_fp16,
                    use_gpu=torch.cuda.is_available()
                )
                configs.append(config)
        
        return configs
    
    def _get_cache_key(self, model: nn.Module, config: OptimizationConfig) -> str:
        """Generate cache key for optimization"""
        model_hash = hashlib.md5(str(model).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
        return f"{model_hash}_{config_hash}"
    
    def save_optimization_profile(self, model_id: str, config: OptimizationConfig,
                                 metrics: PerformanceMetrics):
        """Save optimization profile for future use"""
        profile = {
            'model_id': model_id,
            'hardware_profile': self.hardware_profile.__dict__,
            'optimization_config': config.__dict__,
            'performance_metrics': metrics.__dict__,
            'timestamp': time.time()
        }
        
        profile_path = self.models_dir / f"{model_id}_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
    
    def load_optimization_profile(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load saved optimization profile"""
        profile_path = self.models_dir / f"{model_id}_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system optimization recommendations"""
        recommendations = {
            'hardware_profile': self.hardware_profile.__dict__,
            'recommended_settings': {},
            'potential_improvements': []
        }
        
        # Memory recommendations
        if self.hardware_profile.memory_gb < 8:
            recommendations['potential_improvements'].append(
                "Consider upgrading RAM to at least 8GB for better performance"
            )
        
        # CPU recommendations
        if self.hardware_profile.cpu_cores < 4:
            recommendations['potential_improvements'].append(
                "Consider upgrading to a CPU with at least 4 cores"
            )
        
        # GPU recommendations
        if self.hardware_profile.gpu_memory_gb == 0:
            recommendations['potential_improvements'].append(
                "Consider adding a dedicated GPU for faster inference"
            )
        
        # Optimization settings
        recommendations['recommended_settings'] = {
            'optimization_level': OptimizationLevel.BALANCED.value,
            'memory_strategy': MemoryStrategy.BALANCED.value,
            'enable_mixed_precision': self.hardware_profile.supports_fp16,
            'num_threads': min(self.hardware_profile.cpu_cores, 8)
        }
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Create example model
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Auto-tune configuration
    config = optimizer.auto_tune_config(model, (1, 512))
    
    # Create optimized inference engine
    engine = optimizer.create_inference_engine(model, config)
    
    # Test inference
    test_input = torch.randn(1, 512)
    output, metrics = engine.infer(test_input)
    
    print(f"Inference latency: {metrics.latency_ms:.2f}ms")
    print(f"Throughput: {metrics.throughput_fps:.2f} FPS")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
    
    # Get system recommendations
    recommendations = optimizer.get_system_recommendations()
    print("\nSystem Recommendations:")
    for improvement in recommendations['potential_improvements']:
        print(f"- {improvement}")