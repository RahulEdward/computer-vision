"""
Neural Architecture Search Space Definition
न्यूरल आर्किटेक्चर खोज स्थान परिभाषा

Comprehensive search space definitions for neural architecture search including
cell-based, macro, and micro search spaces with various operations and connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from abc import ABC, abstractmethod
import copy
import json


class OperationType(Enum):
    """Types of operations in search space"""
    # Basic operations
    CONV_3X3 = "conv_3x3"
    CONV_5X5 = "conv_5x5"
    CONV_7X7 = "conv_7x7"
    CONV_1X1 = "conv_1x1"
    
    # Depthwise separable convolutions
    DWISE_CONV_3X3 = "dwise_conv_3x3"
    DWISE_CONV_5X5 = "dwise_conv_5x5"
    
    # Dilated convolutions
    DILATED_CONV_3X3 = "dilated_conv_3x3"
    DILATED_CONV_5X5 = "dilated_conv_5x5"
    
    # Pooling operations
    MAX_POOL_3X3 = "max_pool_3x3"
    AVG_POOL_3X3 = "avg_pool_3x3"
    GLOBAL_AVG_POOL = "global_avg_pool"
    
    # Skip connections
    SKIP_CONNECT = "skip_connect"
    IDENTITY = "identity"
    
    # Zero operation
    ZERO = "zero"
    
    # Advanced operations
    SQUEEZE_EXCITE = "squeeze_excite"
    INVERTED_RESIDUAL = "inverted_residual"
    TRANSFORMER_BLOCK = "transformer_block"
    
    # Activation functions
    RELU = "relu"
    SWISH = "swish"
    GELU = "gelu"
    
    # Normalization
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"


class ConnectionType(Enum):
    """Types of connections between nodes"""
    SEQUENTIAL = "sequential"
    RESIDUAL = "residual"
    DENSE = "dense"
    ATTENTION = "attention"
    HIGHWAY = "highway"


@dataclass
class OperationConfig:
    """Configuration for operations"""
    operation_type: OperationType
    channels: int
    stride: int = 1
    kernel_size: int = 3
    dilation: int = 1
    groups: int = 1
    bias: bool = False
    activation: str = "relu"
    normalization: str = "batch_norm"
    dropout: float = 0.0


@dataclass
class SearchSpaceConfig:
    """Configuration for search space"""
    # Basic parameters
    input_channels: int = 3
    num_classes: int = 10
    num_cells: int = 8
    num_nodes_per_cell: int = 4
    
    # Operation parameters
    operations: List[OperationType] = field(default_factory=lambda: [
        OperationType.CONV_3X3,
        OperationType.CONV_5X5,
        OperationType.DWISE_CONV_3X3,
        OperationType.DWISE_CONV_5X5,
        OperationType.MAX_POOL_3X3,
        OperationType.AVG_POOL_3X3,
        OperationType.SKIP_CONNECT,
        OperationType.ZERO
    ])
    
    # Connection parameters
    connections: List[ConnectionType] = field(default_factory=lambda: [
        ConnectionType.SEQUENTIAL,
        ConnectionType.RESIDUAL
    ])
    
    # Channel parameters
    initial_channels: int = 16
    channel_multiplier: float = 2.0
    
    # Search constraints
    max_depth: int = 20
    max_width: int = 512
    max_parameters: int = 10_000_000
    
    # Hardware constraints
    max_latency: float = 100.0  # ms
    max_memory: float = 1000.0  # MB
    max_energy: float = 1000.0  # mJ


class Operation(nn.Module):
    """Base operation class"""
    
    def __init__(self, config: OperationConfig):
        super().__init__()
        self.config = config
        self.operation = self._build_operation()
    
    def _build_operation(self) -> nn.Module:
        """Build the actual operation"""
        op_type = self.config.operation_type
        
        if op_type == OperationType.CONV_3X3:
            return self._conv_operation(3)
        elif op_type == OperationType.CONV_5X5:
            return self._conv_operation(5)
        elif op_type == OperationType.CONV_7X7:
            return self._conv_operation(7)
        elif op_type == OperationType.CONV_1X1:
            return self._conv_operation(1)
        elif op_type == OperationType.DWISE_CONV_3X3:
            return self._depthwise_conv_operation(3)
        elif op_type == OperationType.DWISE_CONV_5X5:
            return self._depthwise_conv_operation(5)
        elif op_type == OperationType.DILATED_CONV_3X3:
            return self._dilated_conv_operation(3)
        elif op_type == OperationType.DILATED_CONV_5X5:
            return self._dilated_conv_operation(5)
        elif op_type == OperationType.MAX_POOL_3X3:
            return self._pooling_operation("max")
        elif op_type == OperationType.AVG_POOL_3X3:
            return self._pooling_operation("avg")
        elif op_type == OperationType.GLOBAL_AVG_POOL:
            return self._global_pooling_operation()
        elif op_type == OperationType.SKIP_CONNECT:
            return self._skip_connection()
        elif op_type == OperationType.IDENTITY:
            return nn.Identity()
        elif op_type == OperationType.ZERO:
            return self._zero_operation()
        elif op_type == OperationType.SQUEEZE_EXCITE:
            return self._squeeze_excite_operation()
        elif op_type == OperationType.INVERTED_RESIDUAL:
            return self._inverted_residual_operation()
        else:
            return nn.Identity()
    
    def _conv_operation(self, kernel_size: int) -> nn.Module:
        """Create convolution operation"""
        padding = kernel_size // 2
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            self.config.channels, self.config.channels,
            kernel_size=kernel_size,
            stride=self.config.stride,
            padding=padding,
            dilation=self.config.dilation,
            groups=self.config.groups,
            bias=self.config.bias
        ))
        
        # Normalization
        if self.config.normalization == "batch_norm":
            layers.append(nn.BatchNorm2d(self.config.channels))
        elif self.config.normalization == "layer_norm":
            layers.append(nn.LayerNorm(self.config.channels))
        elif self.config.normalization == "group_norm":
            layers.append(nn.GroupNorm(8, self.config.channels))
        
        # Activation
        if self.config.activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif self.config.activation == "swish":
            layers.append(nn.SiLU(inplace=True))
        elif self.config.activation == "gelu":
            layers.append(nn.GELU())
        
        # Dropout
        if self.config.dropout > 0:
            layers.append(nn.Dropout2d(self.config.dropout))
        
        return nn.Sequential(*layers)
    
    def _depthwise_conv_operation(self, kernel_size: int) -> nn.Module:
        """Create depthwise separable convolution"""
        padding = kernel_size // 2
        
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                self.config.channels, self.config.channels,
                kernel_size=kernel_size,
                stride=self.config.stride,
                padding=padding,
                groups=self.config.channels,
                bias=False
            ),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(
                self.config.channels, self.config.channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True)
        )
    
    def _dilated_conv_operation(self, kernel_size: int) -> nn.Module:
        """Create dilated convolution"""
        padding = (kernel_size - 1) * self.config.dilation // 2
        
        return nn.Sequential(
            nn.Conv2d(
                self.config.channels, self.config.channels,
                kernel_size=kernel_size,
                stride=self.config.stride,
                padding=padding,
                dilation=self.config.dilation,
                bias=False
            ),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True)
        )
    
    def _pooling_operation(self, pool_type: str) -> nn.Module:
        """Create pooling operation"""
        if pool_type == "max":
            return nn.MaxPool2d(
                kernel_size=3,
                stride=self.config.stride,
                padding=1
            )
        else:
            return nn.AvgPool2d(
                kernel_size=3,
                stride=self.config.stride,
                padding=1
            )
    
    def _global_pooling_operation(self) -> nn.Module:
        """Create global average pooling"""
        return nn.AdaptiveAvgPool2d((1, 1))
    
    def _skip_connection(self) -> nn.Module:
        """Create skip connection"""
        if self.config.stride == 1:
            return nn.Identity()
        else:
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    self.config.channels, self.config.channels,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.config.channels)
            )
    
    def _zero_operation(self) -> nn.Module:
        """Create zero operation"""
        class ZeroOp(nn.Module):
            def __init__(self, stride):
                super().__init__()
                self.stride = stride
            
            def forward(self, x):
                if self.stride == 1:
                    return x.mul(0.0)
                else:
                    return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        
        return ZeroOp(self.config.stride)
    
    def _squeeze_excite_operation(self) -> nn.Module:
        """Create squeeze-and-excitation operation"""
        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y.expand_as(x)
        
        return SEBlock(self.config.channels)
    
    def _inverted_residual_operation(self) -> nn.Module:
        """Create inverted residual operation"""
        expand_ratio = 6
        expanded_channels = self.config.channels * expand_ratio
        
        return nn.Sequential(
            # Expand
            nn.Conv2d(self.config.channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            
            # Depthwise
            nn.Conv2d(
                expanded_channels, expanded_channels, 3,
                stride=self.config.stride, padding=1,
                groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            
            # Project
            nn.Conv2d(expanded_channels, self.config.channels, 1, bias=False),
            nn.BatchNorm2d(self.config.channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.operation(x)


class SearchSpace(ABC):
    """Abstract base class for search spaces"""
    
    def __init__(self, config: SearchSpaceConfig):
        self.config = config
        self.operations = self._build_operations()
        self.connections = self._build_connections()
    
    @abstractmethod
    def _build_operations(self) -> Dict[str, Operation]:
        """Build available operations"""
        pass
    
    @abstractmethod
    def _build_connections(self) -> Dict[str, ConnectionType]:
        """Build available connections"""
        pass
    
    @abstractmethod
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture"""
        pass
    
    @abstractmethod
    def encode_architecture(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Encode architecture to tensor representation"""
        pass
    
    @abstractmethod
    def decode_architecture(self, encoding: torch.Tensor) -> Dict[str, Any]:
        """Decode tensor representation to architecture"""
        pass
    
    def get_search_space_size(self) -> int:
        """Get total search space size"""
        return len(self.operations) ** self.config.num_nodes_per_cell


class CellSearchSpace(SearchSpace):
    """Cell-based search space (like DARTS)"""
    
    def __init__(self, config: SearchSpaceConfig):
        super().__init__(config)
        self.num_intermediate_nodes = config.num_nodes_per_cell
        self.num_input_nodes = 2  # Previous two cells
    
    def _build_operations(self) -> Dict[str, Operation]:
        """Build available operations"""
        operations = {}
        
        for op_type in self.config.operations:
            op_config = OperationConfig(
                operation_type=op_type,
                channels=self.config.initial_channels
            )
            operations[op_type.value] = Operation(op_config)
        
        return operations
    
    def _build_connections(self) -> Dict[str, ConnectionType]:
        """Build available connections"""
        connections = {}
        for conn_type in self.config.connections:
            connections[conn_type.value] = conn_type
        return connections
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random cell architecture"""
        architecture = {
            'normal_cell': self._sample_cell(),
            'reduction_cell': self._sample_cell(),
            'num_cells': self.config.num_cells,
            'channels': self.config.initial_channels
        }
        return architecture
    
    def _sample_cell(self) -> Dict[str, Any]:
        """Sample a single cell"""
        cell = {
            'nodes': [],
            'connections': []
        }
        
        # Sample operations for each intermediate node
        for i in range(self.num_intermediate_nodes):
            node = {
                'id': i + self.num_input_nodes,
                'inputs': [],
                'operations': []
            }
            
            # Each node takes inputs from previous nodes
            num_inputs = min(2, i + self.num_input_nodes)
            input_nodes = np.random.choice(
                i + self.num_input_nodes, 
                size=num_inputs, 
                replace=False
            ).tolist()
            
            for input_node in input_nodes:
                # Sample operation for this edge
                op_type = np.random.choice(list(self.operations.keys()))
                
                node['inputs'].append(input_node)
                node['operations'].append(op_type)
            
            cell['nodes'].append(node)
        
        return cell
    
    def encode_architecture(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Encode architecture to tensor representation"""
        # Create one-hot encoding for operations
        num_ops = len(self.operations)
        encoding = []
        
        for cell_type in ['normal_cell', 'reduction_cell']:
            cell = architecture[cell_type]
            cell_encoding = []
            
            for node in cell['nodes']:
                for op in node['operations']:
                    op_idx = list(self.operations.keys()).index(op)
                    op_encoding = torch.zeros(num_ops)
                    op_encoding[op_idx] = 1.0
                    cell_encoding.append(op_encoding)
            
            # Pad to fixed size
            while len(cell_encoding) < self.num_intermediate_nodes * 2:
                cell_encoding.append(torch.zeros(num_ops))
            
            encoding.extend(cell_encoding)
        
        return torch.stack(encoding).flatten()
    
    def decode_architecture(self, encoding: torch.Tensor) -> Dict[str, Any]:
        """Decode tensor representation to architecture"""
        num_ops = len(self.operations)
        op_names = list(self.operations.keys())
        
        # Reshape encoding
        encoding = encoding.view(-1, num_ops)
        
        architecture = {
            'normal_cell': {'nodes': [], 'connections': []},
            'reduction_cell': {'nodes': [], 'connections': []},
            'num_cells': self.config.num_cells,
            'channels': self.config.initial_channels
        }
        
        # Decode each cell
        idx = 0
        for cell_type in ['normal_cell', 'reduction_cell']:
            for i in range(self.num_intermediate_nodes):
                node = {
                    'id': i + self.num_input_nodes,
                    'inputs': [],
                    'operations': []
                }
                
                # Decode operations (assuming 2 inputs per node)
                for j in range(2):
                    if idx < len(encoding):
                        op_idx = torch.argmax(encoding[idx]).item()
                        op_name = op_names[op_idx]
                        
                        node['inputs'].append(j)
                        node['operations'].append(op_name)
                        idx += 1
                
                architecture[cell_type]['nodes'].append(node)
        
        return architecture


class MacroSearchSpace(SearchSpace):
    """Macro search space for overall architecture"""
    
    def __init__(self, config: SearchSpaceConfig):
        super().__init__(config)
        self.depth_choices = list(range(8, config.max_depth + 1))
        self.width_choices = [16, 32, 64, 128, 256, 512]
    
    def _build_operations(self) -> Dict[str, Operation]:
        """Build available operations"""
        operations = {}
        
        for op_type in self.config.operations:
            for channels in self.width_choices:
                op_config = OperationConfig(
                    operation_type=op_type,
                    channels=channels
                )
                key = f"{op_type.value}_{channels}"
                operations[key] = Operation(op_config)
        
        return operations
    
    def _build_connections(self) -> Dict[str, ConnectionType]:
        """Build available connections"""
        connections = {}
        for conn_type in self.config.connections:
            connections[conn_type.value] = conn_type
        return connections
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random macro architecture"""
        depth = np.random.choice(self.depth_choices)
        
        architecture = {
            'depth': depth,
            'stages': [],
            'global_pool': 'adaptive_avg',
            'classifier': 'linear'
        }
        
        # Sample stages
        current_channels = self.config.initial_channels
        
        for stage_idx in range(4):  # 4 stages typically
            stage_depth = depth // 4 + (1 if stage_idx < depth % 4 else 0)
            
            stage = {
                'depth': stage_depth,
                'channels': current_channels,
                'stride': 2 if stage_idx > 0 else 1,
                'blocks': []
            }
            
            # Sample blocks in this stage
            for block_idx in range(stage_depth):
                block_op = np.random.choice(list(self.config.operations))
                
                block = {
                    'operation': block_op.value,
                    'channels': current_channels,
                    'stride': stage['stride'] if block_idx == 0 else 1
                }
                
                stage['blocks'].append(block)
            
            architecture['stages'].append(stage)
            current_channels = min(current_channels * 2, 512)
        
        return architecture
    
    def encode_architecture(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Encode macro architecture"""
        encoding = []
        
        # Encode depth
        depth_encoding = torch.zeros(len(self.depth_choices))
        depth_idx = self.depth_choices.index(architecture['depth'])
        depth_encoding[depth_idx] = 1.0
        encoding.append(depth_encoding)
        
        # Encode stages
        for stage in architecture['stages']:
            # Encode stage depth
            stage_depth_encoding = torch.zeros(10)  # Max 10 blocks per stage
            if stage['depth'] < 10:
                stage_depth_encoding[stage['depth']] = 1.0
            encoding.append(stage_depth_encoding)
            
            # Encode operations in stage
            for block in stage['blocks']:
                op_encoding = torch.zeros(len(self.config.operations))
                op_idx = [op.value for op in self.config.operations].index(block['operation'])
                op_encoding[op_idx] = 1.0
                encoding.append(op_encoding)
        
        return torch.cat(encoding)
    
    def decode_architecture(self, encoding: torch.Tensor) -> Dict[str, Any]:
        """Decode macro architecture"""
        # This is a simplified decoder
        # In practice, you'd need more sophisticated decoding logic
        return self.sample_architecture()


class MicroSearchSpace(SearchSpace):
    """Micro search space for operation-level search"""
    
    def __init__(self, config: SearchSpaceConfig):
        super().__init__(config)
        self.kernel_sizes = [1, 3, 5, 7]
        self.expansion_ratios = [1, 3, 6]
        self.se_ratios = [0, 0.25]
    
    def _build_operations(self) -> Dict[str, Operation]:
        """Build micro operations"""
        operations = {}
        
        # Build all combinations of micro operations
        for op_type in self.config.operations:
            for kernel_size in self.kernel_sizes:
                for exp_ratio in self.expansion_ratios:
                    for se_ratio in self.se_ratios:
                        op_config = OperationConfig(
                            operation_type=op_type,
                            channels=self.config.initial_channels,
                            kernel_size=kernel_size
                        )
                        
                        key = f"{op_type.value}_k{kernel_size}_e{exp_ratio}_se{se_ratio}"
                        operations[key] = Operation(op_config)
        
        return operations
    
    def _build_connections(self) -> Dict[str, ConnectionType]:
        """Build connections"""
        return super()._build_connections()
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample micro architecture"""
        architecture = {
            'blocks': [],
            'num_blocks': self.config.num_cells
        }
        
        for i in range(self.config.num_cells):
            block = {
                'operation': np.random.choice([op.value for op in self.config.operations]),
                'kernel_size': np.random.choice(self.kernel_sizes),
                'expansion_ratio': np.random.choice(self.expansion_ratios),
                'se_ratio': np.random.choice(self.se_ratios),
                'channels': self.config.initial_channels * (2 ** (i // 2))
            }
            architecture['blocks'].append(block)
        
        return architecture
    
    def encode_architecture(self, architecture: Dict[str, Any]) -> torch.Tensor:
        """Encode micro architecture"""
        encoding = []
        
        for block in architecture['blocks']:
            # Encode operation
            op_encoding = torch.zeros(len(self.config.operations))
            op_idx = [op.value for op in self.config.operations].index(block['operation'])
            op_encoding[op_idx] = 1.0
            encoding.append(op_encoding)
            
            # Encode kernel size
            kernel_encoding = torch.zeros(len(self.kernel_sizes))
            kernel_idx = self.kernel_sizes.index(block['kernel_size'])
            kernel_encoding[kernel_idx] = 1.0
            encoding.append(kernel_encoding)
            
            # Encode expansion ratio
            exp_encoding = torch.zeros(len(self.expansion_ratios))
            exp_idx = self.expansion_ratios.index(block['expansion_ratio'])
            exp_encoding[exp_idx] = 1.0
            encoding.append(exp_encoding)
            
            # Encode SE ratio
            se_encoding = torch.zeros(len(self.se_ratios))
            se_idx = self.se_ratios.index(block['se_ratio'])
            se_encoding[se_idx] = 1.0
            encoding.append(se_encoding)
        
        return torch.cat(encoding)
    
    def decode_architecture(self, encoding: torch.Tensor) -> Dict[str, Any]:
        """Decode micro architecture"""
        # Simplified decoder
        return self.sample_architecture()


# Example usage
if __name__ == "__main__":
    # Create search space configuration
    config = SearchSpaceConfig(
        input_channels=3,
        num_classes=10,
        num_cells=8,
        num_nodes_per_cell=4,
        initial_channels=16
    )
    
    # Test cell search space
    cell_space = CellSearchSpace(config)
    cell_arch = cell_space.sample_architecture()
    cell_encoding = cell_space.encode_architecture(cell_arch)
    
    print("Neural Architecture Search Space Created Successfully!")
    print(f"Cell search space size: {cell_space.get_search_space_size()}")
    print(f"Cell architecture encoding shape: {cell_encoding.shape}")
    
    # Test macro search space
    macro_space = MacroSearchSpace(config)
    macro_arch = macro_space.sample_architecture()
    macro_encoding = macro_space.encode_architecture(macro_arch)
    
    print(f"Macro search space size: {macro_space.get_search_space_size()}")
    print(f"Macro architecture encoding shape: {macro_encoding.shape}")
    
    # Test micro search space
    micro_space = MicroSearchSpace(config)
    micro_arch = micro_space.sample_architecture()
    micro_encoding = micro_space.encode_architecture(micro_arch)
    
    print(f"Micro search space size: {micro_space.get_search_space_size()}")
    print(f"Micro architecture encoding shape: {micro_encoding.shape}")
    
    print("\nSample Cell Architecture:")
    print(f"Normal cell nodes: {len(cell_arch['normal_cell']['nodes'])}")
    print(f"Reduction cell nodes: {len(cell_arch['reduction_cell']['nodes'])}")
    
    print("\nSample Macro Architecture:")
    print(f"Depth: {macro_arch['depth']}")
    print(f"Number of stages: {len(macro_arch['stages'])}")
    
    print("\nSample Micro Architecture:")
    print(f"Number of blocks: {len(micro_arch['blocks'])}")
    print(f"First block: {micro_arch['blocks'][0]}")