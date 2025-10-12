"""
Hardware-Aware Neural Architecture Search
हार्डवेयर-जागरूक न्यूरल आर्किटेक्चर खोज

Implementation of hardware-aware NAS including latency prediction, memory estimation,
energy consumption modeling, and multi-objective optimization for different hardware platforms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import time
import logging
from collections import defaultdict
import pickle

from .search_space import SearchSpace, Operation, OperationType


class HardwareType(Enum):
    """Types of target hardware"""
    GPU_V100 = "gpu_v100"
    GPU_RTX3080 = "gpu_rtx3080"
    GPU_T4 = "gpu_t4"
    CPU_INTEL = "cpu_intel"
    CPU_ARM = "cpu_arm"
    MOBILE_ANDROID = "mobile_android"
    MOBILE_IOS = "mobile_ios"
    EDGE_JETSON = "edge_jetson"
    EDGE_RPI = "edge_rpi"
    TPU_V3 = "tpu_v3"
    FPGA = "fpga"


class MetricType(Enum):
    """Types of hardware metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ENERGY = "energy"
    FLOPS = "flops"
    PARAMETERS = "parameters"
    MODEL_SIZE = "model_size"


@dataclass
class HardwareConstraints:
    """Hardware constraints for architecture search"""
    # Latency constraints (ms)
    max_latency: float = 100.0
    target_latency: float = 50.0
    
    # Memory constraints (MB)
    max_memory: float = 1000.0
    target_memory: float = 500.0
    
    # Energy constraints (mJ)
    max_energy: float = 1000.0
    target_energy: float = 500.0
    
    # Model size constraints (MB)
    max_model_size: float = 100.0
    target_model_size: float = 50.0
    
    # Throughput constraints (samples/sec)
    min_throughput: float = 10.0
    target_throughput: float = 100.0
    
    # Hardware-specific constraints
    hardware_type: HardwareType = HardwareType.GPU_V100
    batch_size: int = 1
    input_resolution: Tuple[int, int] = (224, 224)
    precision: str = "fp32"  # fp32, fp16, int8


@dataclass
class HardwareProfile:
    """Hardware performance profile"""
    hardware_type: HardwareType
    
    # Operation latencies (ms)
    operation_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Memory usage per operation (MB)
    operation_memory: Dict[str, float] = field(default_factory=dict)
    
    # Energy consumption per operation (mJ)
    operation_energy: Dict[str, float] = field(default_factory=dict)
    
    # Hardware specifications
    memory_bandwidth: float = 1000.0  # GB/s
    compute_capability: float = 100.0  # TFLOPS
    power_consumption: float = 250.0  # W
    
    # Optimization parameters
    parallel_efficiency: float = 0.8
    memory_efficiency: float = 0.9
    cache_hit_ratio: float = 0.95


class OperationProfiler:
    """Profiles individual operations on target hardware"""
    
    def __init__(self, hardware_type: HardwareType, device: str = "cuda"):
        self.hardware_type = hardware_type
        self.device = device
        self.profile_cache = {}
        
        # Load or initialize hardware profiles
        self.hardware_profiles = self._load_hardware_profiles()
    
    def _load_hardware_profiles(self) -> Dict[HardwareType, HardwareProfile]:
        """Load pre-computed hardware profiles"""
        profiles = {}
        
        # GPU V100 profile
        profiles[HardwareType.GPU_V100] = HardwareProfile(
            hardware_type=HardwareType.GPU_V100,
            operation_latencies={
                "conv_3x3": 0.5,
                "conv_5x5": 1.2,
                "conv_7x7": 2.1,
                "conv_1x1": 0.2,
                "dwise_conv_3x3": 0.3,
                "dwise_conv_5x5": 0.7,
                "max_pool_3x3": 0.1,
                "avg_pool_3x3": 0.1,
                "skip_connect": 0.01,
                "zero": 0.0
            },
            operation_memory={
                "conv_3x3": 10.0,
                "conv_5x5": 25.0,
                "conv_7x7": 49.0,
                "conv_1x1": 1.0,
                "dwise_conv_3x3": 3.0,
                "dwise_conv_5x5": 7.5,
                "max_pool_3x3": 0.5,
                "avg_pool_3x3": 0.5,
                "skip_connect": 0.1,
                "zero": 0.0
            },
            operation_energy={
                "conv_3x3": 5.0,
                "conv_5x5": 12.0,
                "conv_7x7": 21.0,
                "conv_1x1": 1.0,
                "dwise_conv_3x3": 2.0,
                "dwise_conv_5x5": 4.5,
                "max_pool_3x3": 0.5,
                "avg_pool_3x3": 0.5,
                "skip_connect": 0.1,
                "zero": 0.0
            },
            memory_bandwidth=900.0,
            compute_capability=125.0,
            power_consumption=300.0
        )
        
        # Mobile Android profile
        profiles[HardwareType.MOBILE_ANDROID] = HardwareProfile(
            hardware_type=HardwareType.MOBILE_ANDROID,
            operation_latencies={
                "conv_3x3": 5.0,
                "conv_5x5": 12.0,
                "conv_7x7": 21.0,
                "conv_1x1": 2.0,
                "dwise_conv_3x3": 3.0,
                "dwise_conv_5x5": 7.0,
                "max_pool_3x3": 1.0,
                "avg_pool_3x3": 1.0,
                "skip_connect": 0.1,
                "zero": 0.0
            },
            operation_memory={
                "conv_3x3": 5.0,
                "conv_5x5": 12.5,
                "conv_7x7": 24.5,
                "conv_1x1": 0.5,
                "dwise_conv_3x3": 1.5,
                "dwise_conv_5x5": 3.75,
                "max_pool_3x3": 0.25,
                "avg_pool_3x3": 0.25,
                "skip_connect": 0.05,
                "zero": 0.0
            },
            operation_energy={
                "conv_3x3": 10.0,
                "conv_5x5": 24.0,
                "conv_7x7": 42.0,
                "conv_1x1": 2.0,
                "dwise_conv_3x3": 4.0,
                "dwise_conv_5x5": 9.0,
                "max_pool_3x3": 1.0,
                "avg_pool_3x3": 1.0,
                "skip_connect": 0.2,
                "zero": 0.0
            },
            memory_bandwidth=25.0,
            compute_capability=1.0,
            power_consumption=5.0
        )
        
        # Add more hardware profiles as needed
        profiles[HardwareType.CPU_INTEL] = self._create_cpu_profile()
        profiles[HardwareType.EDGE_JETSON] = self._create_edge_profile()
        
        return profiles
    
    def _create_cpu_profile(self) -> HardwareProfile:
        """Create CPU hardware profile"""
        return HardwareProfile(
            hardware_type=HardwareType.CPU_INTEL,
            operation_latencies={
                "conv_3x3": 10.0,
                "conv_5x5": 25.0,
                "conv_7x7": 45.0,
                "conv_1x1": 3.0,
                "dwise_conv_3x3": 6.0,
                "dwise_conv_5x5": 15.0,
                "max_pool_3x3": 2.0,
                "avg_pool_3x3": 2.0,
                "skip_connect": 0.1,
                "zero": 0.0
            },
            memory_bandwidth=100.0,
            compute_capability=2.0,
            power_consumption=65.0
        )
    
    def _create_edge_profile(self) -> HardwareProfile:
        """Create edge device profile"""
        return HardwareProfile(
            hardware_type=HardwareType.EDGE_JETSON,
            operation_latencies={
                "conv_3x3": 8.0,
                "conv_5x5": 20.0,
                "conv_7x7": 35.0,
                "conv_1x1": 2.5,
                "dwise_conv_3x3": 4.0,
                "dwise_conv_5x5": 10.0,
                "max_pool_3x3": 1.5,
                "avg_pool_3x3": 1.5,
                "skip_connect": 0.1,
                "zero": 0.0
            },
            memory_bandwidth=50.0,
            compute_capability=5.0,
            power_consumption=15.0
        )
    
    def profile_operation(
        self, 
        operation: Operation, 
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile single operation"""
        cache_key = f"{operation.config.operation_type.value}_{input_shape}"
        
        if cache_key in self.profile_cache:
            return self.profile_cache[cache_key]
        
        # Get hardware profile
        profile = self.hardware_profiles.get(self.hardware_type)
        if profile is None:
            # Fallback to GPU V100 profile
            profile = self.hardware_profiles[HardwareType.GPU_V100]
        
        # Get base metrics from profile
        op_name = operation.config.operation_type.value
        base_latency = profile.operation_latencies.get(op_name, 1.0)
        base_memory = profile.operation_memory.get(op_name, 1.0)
        base_energy = profile.operation_energy.get(op_name, 1.0)
        
        # Scale based on input shape and channels
        batch_size, channels, height, width = input_shape
        scale_factor = (channels * height * width) / (64 * 32 * 32)  # Normalize to reference
        
        metrics = {
            'latency': base_latency * scale_factor,
            'memory': base_memory * scale_factor,
            'energy': base_energy * scale_factor,
            'flops': self._estimate_flops(operation, input_shape),
            'parameters': self._count_parameters(operation)
        }
        
        # Cache results
        self.profile_cache[cache_key] = metrics
        
        return metrics
    
    def _estimate_flops(self, operation: Operation, input_shape: Tuple[int, ...]) -> float:
        """Estimate FLOPs for operation"""
        batch_size, in_channels, height, width = input_shape
        out_channels = operation.config.channels
        kernel_size = operation.config.kernel_size
        
        if operation.config.operation_type in [OperationType.CONV_3X3, OperationType.CONV_5X5, OperationType.CONV_7X7, OperationType.CONV_1X1]:
            # Standard convolution FLOPs
            flops = batch_size * out_channels * height * width * in_channels * kernel_size * kernel_size
        elif operation.config.operation_type in [OperationType.DWISE_CONV_3X3, OperationType.DWISE_CONV_5X5]:
            # Depthwise convolution FLOPs
            flops = batch_size * in_channels * height * width * kernel_size * kernel_size
            flops += batch_size * out_channels * height * width * in_channels  # Pointwise
        elif operation.config.operation_type in [OperationType.MAX_POOL_3X3, OperationType.AVG_POOL_3X3]:
            # Pooling FLOPs
            flops = batch_size * in_channels * height * width * kernel_size * kernel_size
        else:
            # Default minimal FLOPs
            flops = batch_size * in_channels * height * width
        
        return flops / 1e9  # Convert to GFLOPs
    
    def _count_parameters(self, operation: Operation) -> int:
        """Count parameters in operation"""
        # This would count actual parameters in the operation
        # For now, return estimated count
        if hasattr(operation, 'operation') and hasattr(operation.operation, 'parameters'):
            return sum(p.numel() for p in operation.operation.parameters())
        else:
            # Estimate based on operation type
            channels = operation.config.channels
            kernel_size = operation.config.kernel_size
            
            if operation.config.operation_type in [OperationType.CONV_3X3, OperationType.CONV_5X5, OperationType.CONV_7X7, OperationType.CONV_1X1]:
                return channels * channels * kernel_size * kernel_size
            elif operation.config.operation_type in [OperationType.DWISE_CONV_3X3, OperationType.DWISE_CONV_5X5]:
                return channels * kernel_size * kernel_size + channels * channels
            else:
                return 0


class HardwarePredictor:
    """Predicts hardware metrics for complete architectures"""
    
    def __init__(self, profiler: OperationProfiler):
        self.profiler = profiler
        self.prediction_cache = {}
    
    def predict_metrics(
        self, 
        architecture: Dict[str, Any],
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ) -> Dict[str, float]:
        """Predict hardware metrics for architecture"""
        arch_key = json.dumps(architecture, sort_keys=True)
        
        if arch_key in self.prediction_cache:
            return self.prediction_cache[arch_key]
        
        # Initialize metrics
        total_metrics = {
            'latency': 0.0,
            'memory': 0.0,
            'energy': 0.0,
            'flops': 0.0,
            'parameters': 0,
            'model_size': 0.0
        }
        
        # Simulate forward pass through architecture
        current_shape = input_shape
        
        # Process stem
        stem_metrics = self._predict_stem_metrics(current_shape)
        for key in total_metrics:
            total_metrics[key] += stem_metrics.get(key, 0)
        
        # Update shape after stem
        current_shape = (current_shape[0], 16, current_shape[2], current_shape[3])
        
        # Process cells
        num_cells = architecture.get('num_cells', 8)
        
        for cell_idx in range(num_cells):
            # Determine cell type
            is_reduction = cell_idx in [num_cells // 3, 2 * num_cells // 3]
            cell_type = 'reduction_cell' if is_reduction else 'normal_cell'
            
            cell_arch = architecture.get(cell_type, {})
            cell_metrics = self._predict_cell_metrics(cell_arch, current_shape)
            
            for key in total_metrics:
                total_metrics[key] += cell_metrics.get(key, 0)
            
            # Update shape after cell
            if is_reduction:
                current_shape = (
                    current_shape[0], 
                    min(current_shape[1] * 2, 512),
                    current_shape[2] // 2,
                    current_shape[3] // 2
                )
        
        # Process classifier
        classifier_metrics = self._predict_classifier_metrics(current_shape)
        for key in total_metrics:
            total_metrics[key] += classifier_metrics.get(key, 0)
        
        # Calculate derived metrics
        total_metrics['throughput'] = 1000.0 / total_metrics['latency'] if total_metrics['latency'] > 0 else float('inf')
        total_metrics['model_size'] = total_metrics['parameters'] * 4 / (1024 * 1024)  # MB for fp32
        
        # Cache results
        self.prediction_cache[arch_key] = total_metrics
        
        return total_metrics
    
    def _predict_stem_metrics(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Predict metrics for stem network"""
        # Simplified stem: 3x3 conv + BN + ReLU
        from .search_space import OperationConfig
        
        stem_op_config = OperationConfig(
            operation_type=OperationType.CONV_3X3,
            channels=16,
            kernel_size=3
        )
        
        stem_op = Operation(stem_op_config)
        return self.profiler.profile_operation(stem_op, input_shape)
    
    def _predict_cell_metrics(
        self, 
        cell_architecture: Dict[str, Any], 
        input_shape: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """Predict metrics for single cell"""
        cell_metrics = {
            'latency': 0.0,
            'memory': 0.0,
            'energy': 0.0,
            'flops': 0.0,
            'parameters': 0
        }
        
        # Get operations from cell architecture
        operations = cell_architecture.get('operations', [])
        
        for op_name in operations:
            # Create operation
            op_type = OperationType(op_name) if isinstance(op_name, str) else op_name
            
            from .search_space import OperationConfig
            op_config = OperationConfig(
                operation_type=op_type,
                channels=input_shape[1],
                kernel_size=3
            )
            
            operation = Operation(op_config)
            op_metrics = self.profiler.profile_operation(operation, input_shape)
            
            # Accumulate metrics
            for key in cell_metrics:
                cell_metrics[key] += op_metrics.get(key, 0)
        
        return cell_metrics
    
    def _predict_classifier_metrics(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Predict metrics for classifier"""
        # Global average pooling + linear layer
        batch_size, channels, height, width = input_shape
        
        # GAP has minimal cost
        gap_metrics = {
            'latency': 0.1,
            'memory': 0.1,
            'energy': 0.1,
            'flops': batch_size * channels * height * width / 1e9,
            'parameters': 0
        }
        
        # Linear layer
        num_classes = 10  # Assume 10 classes
        linear_params = channels * num_classes
        
        linear_metrics = {
            'latency': 0.1,
            'memory': linear_params * 4 / (1024 * 1024),  # MB
            'energy': 0.1,
            'flops': batch_size * linear_params / 1e9,
            'parameters': linear_params
        }
        
        # Combine metrics
        total_metrics = {}
        for key in gap_metrics:
            total_metrics[key] = gap_metrics[key] + linear_metrics[key]
        
        return total_metrics


class HardwareAwareOptimizer:
    """Multi-objective optimizer for hardware-aware NAS"""
    
    def __init__(
        self, 
        predictor: HardwarePredictor,
        constraints: HardwareConstraints
    ):
        self.predictor = predictor
        self.constraints = constraints
        
        # Pareto frontier for multi-objective optimization
        self.pareto_frontier = []
        self.dominated_solutions = []
    
    def evaluate_architecture(
        self, 
        architecture: Dict[str, Any],
        accuracy: float
    ) -> Dict[str, float]:
        """Evaluate architecture with hardware constraints"""
        # Predict hardware metrics
        hw_metrics = self.predictor.predict_metrics(architecture)
        
        # Calculate constraint violations
        violations = self._calculate_violations(hw_metrics)
        
        # Calculate multi-objective score
        objectives = {
            'accuracy': accuracy,
            'latency': hw_metrics['latency'],
            'memory': hw_metrics['memory'],
            'energy': hw_metrics['energy'],
            'model_size': hw_metrics['model_size'],
            'violations': sum(violations.values())
        }
        
        # Calculate weighted score
        score = self._calculate_weighted_score(objectives)
        
        return {
            'score': score,
            'objectives': objectives,
            'hw_metrics': hw_metrics,
            'violations': violations
        }
    
    def _calculate_violations(self, hw_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate constraint violations"""
        violations = {}
        
        # Latency violation
        if hw_metrics['latency'] > self.constraints.max_latency:
            violations['latency'] = hw_metrics['latency'] - self.constraints.max_latency
        else:
            violations['latency'] = 0.0
        
        # Memory violation
        if hw_metrics['memory'] > self.constraints.max_memory:
            violations['memory'] = hw_metrics['memory'] - self.constraints.max_memory
        else:
            violations['memory'] = 0.0
        
        # Energy violation
        if hw_metrics['energy'] > self.constraints.max_energy:
            violations['energy'] = hw_metrics['energy'] - self.constraints.max_energy
        else:
            violations['energy'] = 0.0
        
        # Model size violation
        if hw_metrics['model_size'] > self.constraints.max_model_size:
            violations['model_size'] = hw_metrics['model_size'] - self.constraints.max_model_size
        else:
            violations['model_size'] = 0.0
        
        return violations
    
    def _calculate_weighted_score(self, objectives: Dict[str, float]) -> float:
        """Calculate weighted multi-objective score"""
        # Normalize objectives
        normalized = {}
        
        # Accuracy (higher is better)
        normalized['accuracy'] = objectives['accuracy']
        
        # Latency (lower is better, normalize by target)
        normalized['latency'] = max(0, 1.0 - objectives['latency'] / self.constraints.target_latency)
        
        # Memory (lower is better, normalize by target)
        normalized['memory'] = max(0, 1.0 - objectives['memory'] / self.constraints.target_memory)
        
        # Energy (lower is better, normalize by target)
        normalized['energy'] = max(0, 1.0 - objectives['energy'] / self.constraints.target_energy)
        
        # Model size (lower is better, normalize by target)
        normalized['model_size'] = max(0, 1.0 - objectives['model_size'] / self.constraints.target_model_size)
        
        # Violations (lower is better)
        violation_penalty = objectives['violations'] * 10.0  # Heavy penalty
        
        # Weighted combination
        weights = {
            'accuracy': 0.4,
            'latency': 0.2,
            'memory': 0.15,
            'energy': 0.15,
            'model_size': 0.1
        }
        
        score = sum(weights[key] * normalized[key] for key in weights)
        score -= violation_penalty  # Subtract penalty
        
        return max(0.0, score)  # Ensure non-negative
    
    def update_pareto_frontier(
        self, 
        architecture: Dict[str, Any], 
        objectives: Dict[str, float]
    ):
        """Update Pareto frontier with new solution"""
        # Check if this solution dominates any existing solutions
        dominated_indices = []
        is_dominated = False
        
        for i, (existing_arch, existing_obj) in enumerate(self.pareto_frontier):
            if self._dominates(objectives, existing_obj):
                dominated_indices.append(i)
            elif self._dominates(existing_obj, objectives):
                is_dominated = True
                break
        
        # If not dominated, add to frontier
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(dominated_indices):
                self.dominated_solutions.append(self.pareto_frontier.pop(i))
            
            # Add new solution
            self.pareto_frontier.append((architecture, objectives))
    
    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2 (Pareto dominance)"""
        # For accuracy: higher is better
        # For others: lower is better
        
        better_in_any = False
        
        # Accuracy (maximize)
        if obj1['accuracy'] > obj2['accuracy']:
            better_in_any = True
        elif obj1['accuracy'] < obj2['accuracy']:
            return False
        
        # Latency (minimize)
        if obj1['latency'] < obj2['latency']:
            better_in_any = True
        elif obj1['latency'] > obj2['latency']:
            return False
        
        # Memory (minimize)
        if obj1['memory'] < obj2['memory']:
            better_in_any = True
        elif obj1['memory'] > obj2['memory']:
            return False
        
        # Energy (minimize)
        if obj1['energy'] < obj2['energy']:
            better_in_any = True
        elif obj1['energy'] > obj2['energy']:
            return False
        
        return better_in_any
    
    def get_best_architectures(self, top_k: int = 5) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        """Get top-k architectures from Pareto frontier"""
        # Sort by weighted score
        scored_frontier = []
        for arch, obj in self.pareto_frontier:
            score = self._calculate_weighted_score(obj)
            scored_frontier.append((arch, obj, score))
        
        scored_frontier.sort(key=lambda x: x[2], reverse=True)
        
        return [(arch, obj) for arch, obj, score in scored_frontier[:top_k]]


# Example usage
if __name__ == "__main__":
    # Create hardware constraints
    constraints = HardwareConstraints(
        max_latency=50.0,
        max_memory=500.0,
        max_energy=500.0,
        hardware_type=HardwareType.MOBILE_ANDROID,
        batch_size=1,
        input_resolution=(224, 224)
    )
    
    # Create profiler and predictor
    profiler = OperationProfiler(HardwareType.MOBILE_ANDROID)
    predictor = HardwarePredictor(profiler)
    
    # Create optimizer
    optimizer = HardwareAwareOptimizer(predictor, constraints)
    
    print("Hardware-Aware NAS Implementation Created Successfully!")
    print(f"Target hardware: {constraints.hardware_type.value}")
    print(f"Max latency: {constraints.max_latency} ms")
    print(f"Max memory: {constraints.max_memory} MB")
    print(f"Max energy: {constraints.max_energy} mJ")
    
    # Test architecture evaluation
    sample_arch = {
        'normal_cell': {'operations': ['conv_3x3', 'dwise_conv_3x3']},
        'reduction_cell': {'operations': ['conv_3x3', 'max_pool_3x3']},
        'num_cells': 8,
        'channels': 16
    }
    
    hw_metrics = predictor.predict_metrics(sample_arch)
    print(f"\nSample architecture metrics:")
    print(f"Latency: {hw_metrics['latency']:.2f} ms")
    print(f"Memory: {hw_metrics['memory']:.2f} MB")
    print(f"Energy: {hw_metrics['energy']:.2f} mJ")
    print(f"FLOPs: {hw_metrics['flops']:.2f} G")
    print(f"Parameters: {hw_metrics['parameters']:,}")
    
    print("Hardware-aware NAS implementation completed!")