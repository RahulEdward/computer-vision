"""
Architecture-based Methods for Continual Learning
निरंतर शिक्षा के लिए आर्किटेक्चर-आधारित विधियां

Implementation of architecture-based approaches for continual learning
including Progressive Neural Networks, PackNet, and dynamic architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import copy


class ArchitectureMethod(Enum):
    """Architecture-based continual learning methods"""
    PROGRESSIVE = "progressive"       # Progressive Neural Networks
    PACKNET = "packnet"              # PackNet
    HAT = "hat"                      # Hard Attention to Task
    PIGGYBACK = "piggyback"          # Piggyback
    SUPERMASK = "supermask"          # Supermask
    DYNAMIC_EXPANSION = "dynamic_expansion"  # Dynamic network expansion
    EXPERT_GATE = "expert_gate"      # Expert gating
    MODULAR = "modular"              # Modular networks


@dataclass
class ArchitectureConfig:
    """Configuration for architecture-based methods"""
    # General parameters
    method: ArchitectureMethod = ArchitectureMethod.PROGRESSIVE
    
    # Progressive Neural Networks
    pnn_lateral_connections: bool = True
    pnn_adapter_size: int = 64
    pnn_num_adapters: int = 2
    
    # PackNet parameters
    packnet_pruning_ratio: float = 0.5
    packnet_retrain_epochs: int = 10
    packnet_prune_method: str = "magnitude"  # "magnitude", "gradient", "random"
    
    # HAT parameters
    hat_smax: float = 400.0
    hat_clipgrad: float = 10000.0
    hat_thres_cosh: float = 50.0
    hat_thres_emb: float = 6.0
    
    # Piggyback parameters
    piggyback_mask_init: str = "kaiming"     # "kaiming", "xavier", "normal"
    piggyback_mask_scale: float = 1e-3
    
    # Supermask parameters
    supermask_sparsity: float = 0.9
    supermask_init_method: str = "kaiming"
    
    # Dynamic expansion parameters
    expansion_threshold: float = 0.95        # Accuracy threshold for expansion
    expansion_factor: float = 1.5            # Factor to expand by
    max_capacity: int = 10000               # Maximum network capacity
    
    # Expert gating parameters
    num_experts: int = 4
    expert_capacity: float = 1.25
    gate_noise: float = 1e-2
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Networks implementation"""
    
    def __init__(self, base_model: nn.Module, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Store base architecture
        self.base_model = base_model
        self.columns = nn.ModuleList([copy.deepcopy(base_model)])
        self.lateral_connections = nn.ModuleList()
        self.adapters = nn.ModuleList()
        
        # Task information
        self.num_tasks = 1
        self.current_task = 0
        
        # Initialize lateral connections for first column
        self._initialize_lateral_connections()
    
    def _initialize_lateral_connections(self):
        """Initialize lateral connections between columns"""
        if not self.config.pnn_lateral_connections:
            return
        
        # For each new column, create lateral connections from all previous columns
        if self.num_tasks > 1:
            lateral_layers = nn.ModuleList()
            
            # Get layer dimensions from base model
            for name, module in self.base_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Create lateral connection for this layer
                    if isinstance(module, nn.Linear):
                        lateral_layer = nn.Linear(
                            module.in_features * (self.num_tasks - 1),
                            self.config.pnn_adapter_size
                        )
                    elif isinstance(module, nn.Conv2d):
                        lateral_layer = nn.Conv2d(
                            module.in_channels * (self.num_tasks - 1),
                            self.config.pnn_adapter_size,
                            kernel_size=1
                        )
                    
                    lateral_layers.append(lateral_layer)
            
            self.lateral_connections.append(lateral_layers)
    
    def add_task(self, task_id: int):
        """Add a new column for a new task"""
        if self.config.verbose:
            self.logger.info(f"PNN: Adding column for task {task_id}")
        
        # Add new column (copy of base model)
        new_column = copy.deepcopy(self.base_model)
        self.columns.append(new_column)
        
        # Freeze previous columns
        for i in range(len(self.columns) - 1):
            for param in self.columns[i].parameters():
                param.requires_grad = False
        
        self.num_tasks += 1
        self.current_task = task_id
        
        # Initialize lateral connections for new column
        self._initialize_lateral_connections()
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through progressive network"""
        if task_id is None:
            task_id = self.current_task
        
        # Ensure task_id is valid
        if task_id >= len(self.columns):
            task_id = len(self.columns) - 1
        
        # Forward through current column
        current_column = self.columns[task_id]
        
        if not self.config.pnn_lateral_connections or task_id == 0:
            # No lateral connections for first task or if disabled
            return current_column(x)
        
        # Forward with lateral connections
        # This is a simplified implementation - in practice, you'd need to
        # carefully handle the lateral connections at each layer
        return current_column(x)
    
    def get_task_parameters(self, task_id: int) -> List[torch.Tensor]:
        """Get parameters for specific task"""
        if task_id < len(self.columns):
            return list(self.columns[task_id].parameters())
        return []


class PackNet(nn.Module):
    """PackNet implementation for continual learning"""
    
    def __init__(self, base_model: nn.Module, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        self.model = base_model
        self.masks = {}
        self.task_masks = {}
        self.current_task = 0
        
        # Initialize masks
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize binary masks for all parameters"""
        for name, param in self.model.named_parameters():
            self.masks[name] = torch.ones_like(param.data)
    
    def add_task(self, task_id: int):
        """Add new task and create mask"""
        if self.config.verbose:
            self.logger.info(f"PackNet: Adding task {task_id}")
        
        self.current_task = task_id
        
        if task_id > 0:
            # Prune network for previous task
            self._prune_for_task(task_id - 1)
    
    def _prune_for_task(self, task_id: int):
        """Prune network and create mask for specific task"""
        if self.config.verbose:
            self.logger.info(f"PackNet: Pruning for task {task_id}")
        
        task_mask = {}
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Get available weights (not yet pruned)
                available_mask = self.masks[name]
                available_weights = param.data * available_mask
                
                # Compute importance scores
                if self.config.packnet_prune_method == "magnitude":
                    importance = torch.abs(available_weights)
                elif self.config.packnet_prune_method == "gradient":
                    # Use gradient magnitude (simplified)
                    if param.grad is not None:
                        importance = torch.abs(param.grad.data) * available_mask
                    else:
                        importance = torch.abs(available_weights)
                else:  # random
                    importance = torch.rand_like(available_weights) * available_mask
                
                # Determine pruning threshold
                flat_importance = importance[available_mask.bool()].flatten()
                if len(flat_importance) > 0:
                    k = int(len(flat_importance) * self.config.packnet_pruning_ratio)
                    threshold = torch.kthvalue(flat_importance, k).values
                    
                    # Create task mask
                    task_mask[name] = (importance >= threshold) & available_mask.bool()
                    
                    # Update global mask
                    self.masks[name] = self.masks[name] & (~task_mask[name])
                else:
                    task_mask[name] = torch.zeros_like(param.data, dtype=torch.bool)
        
        self.task_masks[task_id] = task_mask
    
    def apply_mask(self, task_id: Optional[int] = None):
        """Apply mask to model parameters"""
        if task_id is None:
            task_id = self.current_task
        
        if task_id in self.task_masks:
            for name, param in self.model.named_parameters():
                if name in self.task_masks[task_id]:
                    param.data *= self.task_masks[task_id][name].float()
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with task-specific mask"""
        if task_id is None:
            task_id = self.current_task
        
        # Apply task mask
        self.apply_mask(task_id)
        
        return self.model(x)
    
    def get_sparsity(self, task_id: Optional[int] = None) -> float:
        """Get sparsity level for task"""
        if task_id is None:
            task_id = self.current_task
        
        if task_id not in self.task_masks:
            return 0.0
        
        total_params = 0
        used_params = 0
        
        for name, mask in self.task_masks[task_id].items():
            total_params += mask.numel()
            used_params += mask.sum().item()
        
        return 1.0 - (used_params / total_params) if total_params > 0 else 0.0


class HardAttentionToTask(nn.Module):
    """Hard Attention to Task (HAT) implementation"""
    
    def __init__(self, base_model: nn.Module, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        self.model = base_model
        self.attention_modules = nn.ModuleDict()
        self.task_embeddings = nn.ParameterDict()
        self.current_task = 0
        
        # Initialize attention modules
        self._initialize_attention()
    
    def _initialize_attention(self):
        """Initialize attention modules for each layer"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Create attention module
                if isinstance(module, nn.Linear):
                    attention_size = module.out_features
                elif isinstance(module, nn.Conv2d):
                    attention_size = module.out_channels
                
                self.attention_modules[name] = nn.Linear(
                    self.config.hat_thres_emb, attention_size
                )
    
    def add_task(self, task_id: int):
        """Add task embedding for new task"""
        if self.config.verbose:
            self.logger.info(f"HAT: Adding task {task_id}")
        
        self.current_task = task_id
        
        # Add task embedding
        self.task_embeddings[str(task_id)] = nn.Parameter(
            torch.randn(self.config.hat_thres_emb)
        )
    
    def get_attention_mask(self, task_id: int, layer_name: str) -> torch.Tensor:
        """Get attention mask for specific task and layer"""
        if str(task_id) not in self.task_embeddings:
            return torch.ones(1)  # Default mask
        
        task_emb = self.task_embeddings[str(task_id)]
        
        if layer_name in self.attention_modules:
            attention_logits = self.attention_modules[layer_name](task_emb)
            attention_mask = torch.sigmoid(attention_logits * self.config.hat_smax)
            return attention_mask
        
        return torch.ones(1)
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with hard attention"""
        if task_id is None:
            task_id = self.current_task
        
        # This is a simplified implementation
        # In practice, you'd need to apply attention at each layer
        return self.model(x)


class DynamicExpansion(nn.Module):
    """Dynamic network expansion for continual learning"""
    
    def __init__(self, base_model: nn.Module, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        self.model = base_model
        self.expansion_history = []
        self.current_capacity = self._count_parameters()
        self.task_performance = {}
    
    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def should_expand(self, task_id: int, performance: float) -> bool:
        """Determine if network should be expanded"""
        if performance < self.config.expansion_threshold:
            if self.current_capacity < self.config.max_capacity:
                return True
        return False
    
    def expand_network(self, task_id: int, layer_name: str = None):
        """Expand network capacity"""
        if self.config.verbose:
            self.logger.info(f"Dynamic Expansion: Expanding for task {task_id}")
        
        # Simple expansion: add neurons to largest layer
        max_size = 0
        target_layer = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features > max_size:
                    max_size = module.out_features
                    target_layer = (name, module)
        
        if target_layer:
            name, layer = target_layer
            new_size = int(layer.out_features * self.config.expansion_factor)
            
            # Create new layer with expanded size
            new_layer = nn.Linear(layer.in_features, new_size)
            
            # Copy old weights
            with torch.no_grad():
                new_layer.weight[:layer.out_features] = layer.weight
                new_layer.bias[:layer.out_features] = layer.bias
            
            # Replace layer in model (simplified)
            # In practice, you'd need more sophisticated replacement
            self.expansion_history.append({
                'task_id': task_id,
                'layer_name': name,
                'old_size': layer.out_features,
                'new_size': new_size
            })
            
            self.current_capacity = self._count_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dynamic network"""
        return self.model(x)


class ArchitectureManager:
    """Manager for architecture-based continual learning methods"""
    
    def __init__(self, base_model: nn.Module, config: ArchitectureConfig):
        self.config = config
        self.base_model = base_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize architecture method
        if config.method == ArchitectureMethod.PROGRESSIVE:
            self.method = ProgressiveNeuralNetwork(base_model, config)
        elif config.method == ArchitectureMethod.PACKNET:
            self.method = PackNet(base_model, config)
        elif config.method == ArchitectureMethod.HAT:
            self.method = HardAttentionToTask(base_model, config)
        elif config.method == ArchitectureMethod.DYNAMIC_EXPANSION:
            self.method = DynamicExpansion(base_model, config)
        else:
            raise ValueError(f"Unsupported architecture method: {config.method}")
        
        self.task_count = 0
        self.method_stats = {}
    
    def add_task(self, task_id: int):
        """Add new task to architecture"""
        if self.config.verbose:
            self.logger.info(f"Architecture Manager: Adding task {task_id}")
        
        self.method.add_task(task_id)
        self.task_count += 1
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through architecture"""
        return self.method(x, task_id)
    
    def get_model_for_task(self, task_id: int) -> nn.Module:
        """Get model for specific task"""
        if hasattr(self.method, 'get_task_parameters'):
            # For methods like Progressive NN
            return self.method
        else:
            # For methods that modify the base model
            return self.method.model if hasattr(self.method, 'model') else self.method
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get architecture method statistics"""
        stats = {
            'method': self.config.method.value,
            'task_count': self.task_count,
            'base_model_params': sum(p.numel() for p in self.base_model.parameters())
        }
        
        # Method-specific statistics
        if isinstance(self.method, ProgressiveNeuralNetwork):
            stats.update({
                'num_columns': len(self.method.columns),
                'lateral_connections': self.config.pnn_lateral_connections,
                'total_params': sum(sum(p.numel() for p in col.parameters()) 
                                  for col in self.method.columns)
            })
        
        elif isinstance(self.method, PackNet):
            stats.update({
                'num_task_masks': len(self.method.task_masks),
                'pruning_ratio': self.config.packnet_pruning_ratio,
                'current_sparsity': self.method.get_sparsity()
            })
        
        elif isinstance(self.method, DynamicExpansion):
            stats.update({
                'current_capacity': self.method.current_capacity,
                'expansion_history': len(self.method.expansion_history),
                'max_capacity': self.config.max_capacity
            })
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create a simple base model
    base_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create architecture configuration
    config = ArchitectureConfig(
        method=ArchitectureMethod.PROGRESSIVE,
        pnn_lateral_connections=True
    )
    
    # Create architecture manager
    arch_manager = ArchitectureManager(base_model, config)
    
    print("Architecture manager created successfully!")
    print(f"Method: {config.method.value}")
    print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters())}")