"""
Supernet Training for One-Shot NAS
वन-शॉट NAS के लिए सुपरनेट प्रशिक्षण

Implementation of supernet training strategies including One-Shot NAS, Progressive Training,
Weight Sharing, and various sampling strategies for efficient architecture search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
import json
import time
import logging
from collections import defaultdict, OrderedDict

from .search_space import SearchSpace, Operation, OperationType, OperationConfig


class SupernetTrainingStrategy(Enum):
    """Types of supernet training strategies"""
    ONE_SHOT = "one_shot"
    PROGRESSIVE = "progressive"
    SANDWICH = "sandwich"
    FAIR_NAS = "fair_nas"
    SINGLE_PATH = "single_path"
    UNIFORM_SAMPLING = "uniform_sampling"
    PROGRESSIVE_SHRINKING = "progressive_shrinking"
    BIGNAS = "bignas"
    ONCE_FOR_ALL = "once_for_all"


class SamplingStrategy(Enum):
    """Architecture sampling strategies"""
    UNIFORM = "uniform"
    PROGRESSIVE = "progressive"
    SANDWICH = "sandwich"
    EVOLUTIONARY = "evolutionary"
    FAIR = "fair"
    PRIORITIZED = "prioritized"


@dataclass
class SupernetConfig:
    """Configuration for supernet training"""
    # Training strategy
    training_strategy: SupernetTrainingStrategy = SupernetTrainingStrategy.ONE_SHOT
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    
    # Training parameters
    epochs: int = 120
    warmup_epochs: int = 5
    learning_rate: float = 0.025
    weight_decay: float = 3e-4
    momentum: float = 0.9
    grad_clip: float = 5.0
    
    # Sampling parameters
    num_sample_training: int = 4
    num_sample_inplace: int = 2
    sandwich_rule: bool = True
    
    # Progressive training
    progressive_stages: int = 4
    stage_epochs: int = 30
    
    # Architecture constraints
    min_layers: int = 8
    max_layers: int = 20
    min_channels: int = 16
    max_channels: int = 512
    
    # Hardware-aware training
    latency_constraint: float = 100.0  # ms
    memory_constraint: float = 1000.0  # MB
    energy_constraint: float = 1000.0  # mJ
    
    # Distillation parameters
    teacher_forcing: bool = False
    distillation_alpha: float = 0.5
    temperature: float = 4.0


class MixedOperation(nn.Module):
    """Mixed operation for supernet"""
    
    def __init__(self, operations: List[Operation], operation_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.num_ops = len(operations)
        
        if operation_weights is not None:
            self.register_buffer('operation_weights', operation_weights)
        else:
            self.register_buffer('operation_weights', torch.ones(self.num_ops))
    
    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with weighted operations"""
        if weights is None:
            weights = self.operation_weights
        
        # Apply operations and weight them
        outputs = []
        for i, op in enumerate(self.operations):
            if weights[i] > 0:  # Skip zero-weight operations
                outputs.append(weights[i] * op(x))
        
        if not outputs:
            return torch.zeros_like(x)
        
        return sum(outputs)
    
    def forward_single(self, x: torch.Tensor, op_idx: int) -> torch.Tensor:
        """Forward pass with single operation"""
        return self.operations[op_idx](x)


class SupernetCell(nn.Module):
    """Supernet cell with mixed operations"""
    
    def __init__(
        self, 
        num_nodes: int, 
        operations: List[Operation],
        channels: int,
        stride: int = 1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.stride = stride
        
        # Create mixed operations for each edge
        self.mixed_ops = nn.ModuleDict()
        
        for i in range(num_nodes):
            for j in range(i + 2):  # Connect to previous nodes
                edge_key = f"{j}_{i+2}"
                self.mixed_ops[edge_key] = MixedOperation(operations)
        
        # Preprocessing operations
        self.preprocess = nn.ModuleDict()
        for i in range(2):  # Two input nodes
            self.preprocess[str(i)] = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
    
    def forward(
        self, 
        inputs: List[torch.Tensor], 
        architecture: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through cell"""
        # Preprocess inputs
        states = []
        for i, inp in enumerate(inputs):
            states.append(self.preprocess[str(i)](inp))
        
        # Process intermediate nodes
        for i in range(self.num_nodes):
            node_inputs = []
            
            for j in range(i + 2):
                edge_key = f"{j}_{i+2}"
                
                if architecture is not None:
                    # Single path forward
                    op_idx = architecture.get(edge_key, 0)
                    output = self.mixed_ops[edge_key].forward_single(states[j], op_idx)
                elif weights is not None:
                    # Weighted forward
                    edge_weights = weights.get(edge_key, None)
                    output = self.mixed_ops[edge_key](states[j], edge_weights)
                else:
                    # Uniform forward
                    output = self.mixed_ops[edge_key](states[j])
                
                node_inputs.append(output)
            
            # Combine inputs to this node
            if node_inputs:
                states.append(sum(node_inputs))
            else:
                states.append(torch.zeros_like(states[0]))
        
        # Combine outputs from intermediate nodes
        output_nodes = states[2:]  # Skip input nodes
        if output_nodes:
            return sum(output_nodes)
        else:
            return states[0]


class Supernet(nn.Module):
    """Complete supernet architecture"""
    
    def __init__(
        self, 
        search_space: SearchSpace,
        config: SupernetConfig,
        num_classes: int = 10
    ):
        super().__init__()
        self.search_space = search_space
        self.config = config
        self.num_classes = num_classes
        
        # Build supernet components
        self.stem = self._build_stem()
        self.cells = self._build_cells()
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._build_classifier()
        
        # Architecture sampling
        self.current_architecture = None
        self.architecture_weights = None
    
    def _build_stem(self) -> nn.Module:
        """Build stem network"""
        initial_channels = self.search_space.config.initial_channels
        
        return nn.Sequential(
            nn.Conv2d(3, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_cells(self) -> nn.ModuleList:
        """Build supernet cells"""
        cells = nn.ModuleList()
        
        # Create operations for cells
        operations = []
        for op_type in self.search_space.config.operations:
            op_config = OperationConfig(
                operation_type=op_type,
                channels=self.search_space.config.initial_channels
            )
            operations.append(Operation(op_config))
        
        # Build cells
        num_cells = self.search_space.config.num_cells
        num_nodes = self.search_space.config.num_nodes_per_cell
        
        for i in range(num_cells):
            # Determine if this is a reduction cell
            is_reduction = i in [num_cells // 3, 2 * num_cells // 3]
            stride = 2 if is_reduction else 1
            
            cell = SupernetCell(
                num_nodes=num_nodes,
                operations=operations,
                channels=self.search_space.config.initial_channels,
                stride=stride
            )
            
            cells.append(cell)
        
        return cells
    
    def _build_classifier(self) -> nn.Module:
        """Build classifier head"""
        return nn.Linear(self.search_space.config.initial_channels, self.num_classes)
    
    def forward(
        self, 
        x: torch.Tensor, 
        architecture: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through supernet"""
        # Stem
        x = self.stem(x)
        
        # Initialize cell inputs
        prev_prev, prev = x, x
        
        # Process cells
        for i, cell in enumerate(self.cells):
            # Get architecture for this cell
            cell_arch = None
            cell_weights = None
            
            if architecture is not None:
                cell_type = 'reduction' if i in [len(self.cells) // 3, 2 * len(self.cells) // 3] else 'normal'
                cell_arch = architecture.get(f'{cell_type}_cell', None)
            
            if weights is not None:
                cell_type = 'reduction' if i in [len(self.cells) // 3, 2 * len(self.cells) // 3] else 'normal'
                cell_weights = weights.get(f'{cell_type}_cell', None)
            
            # Forward through cell
            output = cell([prev_prev, prev], cell_arch, cell_weights)
            
            # Update inputs for next cell
            prev_prev, prev = prev, output
        
        # Global pooling and classification
        x = self.global_pooling(prev)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def sample_architecture(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Dict[str, Any]:
        """Sample architecture from supernet"""
        if strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sampling()
        elif strategy == SamplingStrategy.PROGRESSIVE:
            return self._progressive_sampling()
        elif strategy == SamplingStrategy.SANDWICH:
            return self._sandwich_sampling()
        elif strategy == SamplingStrategy.FAIR:
            return self._fair_sampling()
        else:
            return self._uniform_sampling()
    
    def _uniform_sampling(self) -> Dict[str, Any]:
        """Uniform random sampling"""
        architecture = {}
        
        for cell_type in ['normal_cell', 'reduction_cell']:
            cell_arch = {}
            num_nodes = self.search_space.config.num_nodes_per_cell
            
            for i in range(num_nodes):
                for j in range(i + 2):
                    edge_key = f"{j}_{i+2}"
                    # Randomly select operation
                    op_idx = random.randint(0, len(self.search_space.config.operations) - 1)
                    cell_arch[edge_key] = op_idx
            
            architecture[cell_type] = cell_arch
        
        return architecture
    
    def _progressive_sampling(self) -> Dict[str, Any]:
        """Progressive sampling with increasing complexity"""
        # Start with simple operations and gradually include complex ones
        simple_ops = [OperationType.SKIP_CONNECT, OperationType.CONV_3X3]
        complex_ops = list(self.search_space.config.operations)
        
        # Determine sampling probability based on training progress
        # This would be updated during training
        prob_complex = 0.5  # Placeholder
        
        architecture = {}
        
        for cell_type in ['normal_cell', 'reduction_cell']:
            cell_arch = {}
            num_nodes = self.search_space.config.num_nodes_per_cell
            
            for i in range(num_nodes):
                for j in range(i + 2):
                    edge_key = f"{j}_{i+2}"
                    
                    if random.random() < prob_complex:
                        # Sample from all operations
                        op_idx = random.randint(0, len(complex_ops) - 1)
                    else:
                        # Sample from simple operations
                        simple_idx = random.randint(0, len(simple_ops) - 1)
                        op_idx = complex_ops.index(simple_ops[simple_idx])
                    
                    cell_arch[edge_key] = op_idx
            
            architecture[cell_type] = cell_arch
        
        return architecture
    
    def _sandwich_sampling(self) -> Dict[str, Any]:
        """Sandwich sampling (largest, smallest, random)"""
        # This would return different architectures based on sandwich rule
        # For now, return uniform sampling
        return self._uniform_sampling()
    
    def _fair_sampling(self) -> Dict[str, Any]:
        """Fair sampling ensuring all operations are trained equally"""
        # Track operation usage and sample less-used operations more frequently
        # For now, return uniform sampling
        return self._uniform_sampling()


class SupernetTrainer:
    """Trainer for supernet with various strategies"""
    
    def __init__(
        self, 
        supernet: Supernet, 
        config: SupernetConfig,
        device: str = "cuda"
    ):
        self.supernet = supernet
        self.config = config
        self.device = device
        
        # Move model to device
        self.supernet = self.supernet.to(device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.SGD(
            self.supernet.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'architecture_performance': defaultdict(list)
        }
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Train supernet"""
        logging.info(f"Starting supernet training with {self.config.training_strategy.value} strategy")
        
        if self.config.training_strategy == SupernetTrainingStrategy.ONE_SHOT:
            return self._train_one_shot(train_loader, val_loader)
        elif self.config.training_strategy == SupernetTrainingStrategy.PROGRESSIVE:
            return self._train_progressive(train_loader, val_loader)
        elif self.config.training_strategy == SupernetTrainingStrategy.SANDWICH:
            return self._train_sandwich(train_loader, val_loader)
        else:
            return self._train_one_shot(train_loader, val_loader)
    
    def _train_one_shot(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """One-shot supernet training"""
        for epoch in range(self.config.epochs):
            # Training phase
            self.supernet.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Sample architectures for this batch
                architectures = []
                for _ in range(self.config.num_sample_training):
                    arch = self.supernet.sample_architecture(self.config.sampling_strategy)
                    architectures.append(arch)
                
                # Train on sampled architectures
                total_loss = 0.0
                
                for arch in architectures:
                    self.optimizer.zero_grad()
                    
                    output = self.supernet(data, architecture=arch)
                    loss = self.criterion(output, target)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.supernet.parameters(), self.config.grad_clip)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                epoch_loss += total_loss / len(architectures)
                num_batches += 1
                
                # Quick training for demo
                if batch_idx > 10:
                    break
            
            # Validation phase
            val_accuracy = self._validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            avg_loss = epoch_loss / num_batches
            self.training_stats['epoch_losses'].append(avg_loss)
            self.training_stats['epoch_accuracies'].append(val_accuracy)
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        
        return {
            'training_stats': self.training_stats,
            'final_supernet': self.supernet.state_dict()
        }
    
    def _train_progressive(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Progressive supernet training"""
        stages = self.config.progressive_stages
        epochs_per_stage = self.config.epochs // stages
        
        for stage in range(stages):
            logging.info(f"Training stage {stage + 1}/{stages}")
            
            # Adjust sampling strategy for this stage
            stage_config = copy.deepcopy(self.config)
            stage_config.epochs = epochs_per_stage
            
            # Train for this stage
            stage_results = self._train_one_shot(train_loader, val_loader)
            
            # Update training stats
            self.training_stats['epoch_losses'].extend(stage_results['training_stats']['epoch_losses'])
            self.training_stats['epoch_accuracies'].extend(stage_results['training_stats']['epoch_accuracies'])
        
        return {
            'training_stats': self.training_stats,
            'final_supernet': self.supernet.state_dict()
        }
    
    def _train_sandwich(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Sandwich rule training"""
        # Implement sandwich rule: train largest, smallest, and random architectures
        for epoch in range(self.config.epochs):
            self.supernet.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Sandwich sampling: largest, smallest, random
                architectures = [
                    self._sample_largest_architecture(),
                    self._sample_smallest_architecture(),
                    self.supernet.sample_architecture(SamplingStrategy.UNIFORM),
                    self.supernet.sample_architecture(SamplingStrategy.UNIFORM)
                ]
                
                # Train on sandwich architectures
                total_loss = 0.0
                
                for arch in architectures:
                    self.optimizer.zero_grad()
                    
                    output = self.supernet(data, architecture=arch)
                    loss = self.criterion(output, target)
                    
                    loss.backward()
                    
                    if self.config.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.supernet.parameters(), self.config.grad_clip)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                epoch_loss += total_loss / len(architectures)
                num_batches += 1
                
                if batch_idx > 10:
                    break
            
            # Validation
            val_accuracy = self._validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            avg_loss = epoch_loss / num_batches
            self.training_stats['epoch_losses'].append(avg_loss)
            self.training_stats['epoch_accuracies'].append(val_accuracy)
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        
        return {
            'training_stats': self.training_stats,
            'final_supernet': self.supernet.state_dict()
        }
    
    def _sample_largest_architecture(self) -> Dict[str, Any]:
        """Sample largest possible architecture"""
        # Select operations with highest complexity/parameters
        complex_ops = [OperationType.CONV_7X7, OperationType.CONV_5X5, OperationType.DWISE_CONV_5X5]
        
        architecture = {}
        for cell_type in ['normal_cell', 'reduction_cell']:
            cell_arch = {}
            num_nodes = self.supernet.search_space.config.num_nodes_per_cell
            
            for i in range(num_nodes):
                for j in range(i + 2):
                    edge_key = f"{j}_{i+2}"
                    # Select most complex operation available
                    for op in complex_ops:
                        if op in self.supernet.search_space.config.operations:
                            op_idx = self.supernet.search_space.config.operations.index(op)
                            cell_arch[edge_key] = op_idx
                            break
                    else:
                        # Fallback to first operation
                        cell_arch[edge_key] = 0
            
            architecture[cell_type] = cell_arch
        
        return architecture
    
    def _sample_smallest_architecture(self) -> Dict[str, Any]:
        """Sample smallest possible architecture"""
        # Select operations with lowest complexity/parameters
        simple_ops = [OperationType.SKIP_CONNECT, OperationType.CONV_1X1, OperationType.CONV_3X3]
        
        architecture = {}
        for cell_type in ['normal_cell', 'reduction_cell']:
            cell_arch = {}
            num_nodes = self.supernet.search_space.config.num_nodes_per_cell
            
            for i in range(num_nodes):
                for j in range(i + 2):
                    edge_key = f"{j}_{i+2}"
                    # Select simplest operation available
                    for op in simple_ops:
                        if op in self.supernet.search_space.config.operations:
                            op_idx = self.supernet.search_space.config.operations.index(op)
                            cell_arch[edge_key] = op_idx
                            break
                    else:
                        # Fallback to first operation
                        cell_arch[edge_key] = 0
            
            architecture[cell_type] = cell_arch
        
        return architecture
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate supernet with random architectures"""
        self.supernet.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Sample random architecture for validation
                architecture = self.supernet.sample_architecture()
                output = self.supernet(data, architecture=architecture)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx > 5:  # Quick validation
                    break
        
        return correct / total if total > 0 else 0.0
    
    def extract_subnet(self, architecture: Dict[str, Any]) -> nn.Module:
        """Extract subnet from supernet given architecture"""
        # This would create a standalone model with the specified architecture
        # For now, return a simplified version
        class ExtractedSubnet(nn.Module):
            def __init__(self, supernet, architecture):
                super().__init__()
                self.supernet = supernet
                self.architecture = architecture
            
            def forward(self, x):
                return self.supernet(x, architecture=self.architecture)
        
        return ExtractedSubnet(self.supernet, architecture)


# Example usage
if __name__ == "__main__":
    from .search_space import SearchSpaceConfig, CellSearchSpace
    
    # Create search space
    space_config = SearchSpaceConfig(
        input_channels=3,
        num_classes=10,
        num_cells=8,
        num_nodes_per_cell=4
    )
    search_space = CellSearchSpace(space_config)
    
    # Create supernet configuration
    supernet_config = SupernetConfig(
        training_strategy=SupernetTrainingStrategy.ONE_SHOT,
        sampling_strategy=SamplingStrategy.UNIFORM,
        epochs=50,
        num_sample_training=4
    )
    
    # Create supernet
    supernet = Supernet(search_space, supernet_config, num_classes=10)
    
    # Create trainer
    trainer = SupernetTrainer(supernet, supernet_config, device="cpu")
    
    print("Supernet Training Implementation Created Successfully!")
    print(f"Supernet parameters: {sum(p.numel() for p in supernet.parameters()):,}")
    print(f"Training strategy: {supernet_config.training_strategy.value}")
    print(f"Sampling strategy: {supernet_config.sampling_strategy.value}")
    
    # Test architecture sampling
    sample_arch = supernet.sample_architecture()
    print(f"\nSample architecture keys: {list(sample_arch.keys())}")
    
    print("Supernet training implementation completed!")