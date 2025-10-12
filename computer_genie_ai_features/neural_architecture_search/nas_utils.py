"""
Neural Architecture Search Utilities
न्यूरल आर्किटेक्चर खोज उपयोगिताएं

Utility functions and helper classes for NAS including architecture encoding/decoding,
visualization, benchmarking, and integration with popular NAS benchmarks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import hashlib
from collections import defaultdict
import logging
import time
import os

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .search_space import SearchSpace, Operation, OperationType, ConnectionType


class ArchitectureFormat(Enum):
    """Architecture representation formats"""
    GENOTYPE = "genotype"           # DARTS-style genotype
    ADJACENCY_MATRIX = "adjacency_matrix"  # Graph adjacency matrix
    SEQUENCE = "sequence"           # Sequential representation
    TREE = "tree"                   # Tree-based representation
    JSON = "json"                   # JSON format
    COMPACT = "compact"             # Compact string representation


@dataclass
class BenchmarkResult:
    """Result from architecture benchmark"""
    architecture_id: str
    accuracy: float
    latency: float
    memory: float
    flops: float
    parameters: int
    training_time: float
    evaluation_time: float
    hardware_type: str
    dataset: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'architecture_id': self.architecture_id,
            'accuracy': self.accuracy,
            'latency': self.latency,
            'memory': self.memory,
            'flops': self.flops,
            'parameters': self.parameters,
            'training_time': self.training_time,
            'evaluation_time': self.evaluation_time,
            'hardware_type': self.hardware_type,
            'dataset': self.dataset
        }


class ArchitectureEncoder:
    """Encodes architectures in various formats"""
    
    def __init__(self):
        self.operation_to_idx = {op.value: idx for idx, op in enumerate(OperationType)}
        self.idx_to_operation = {idx: op for op, idx in self.operation_to_idx.items()}
        self.connection_to_idx = {conn.value: idx for idx, conn in enumerate(ConnectionType)}
        self.idx_to_connection = {idx: conn for conn, idx in self.connection_to_idx.items()}
    
    def encode_architecture(
        self, 
        architecture: Dict[str, Any], 
        format_type: ArchitectureFormat
    ) -> Union[str, List, Dict, np.ndarray]:
        """Encode architecture in specified format"""
        if format_type == ArchitectureFormat.GENOTYPE:
            return self._encode_genotype(architecture)
        elif format_type == ArchitectureFormat.ADJACENCY_MATRIX:
            return self._encode_adjacency_matrix(architecture)
        elif format_type == ArchitectureFormat.SEQUENCE:
            return self._encode_sequence(architecture)
        elif format_type == ArchitectureFormat.TREE:
            return self._encode_tree(architecture)
        elif format_type == ArchitectureFormat.JSON:
            return self._encode_json(architecture)
        elif format_type == ArchitectureFormat.COMPACT:
            return self._encode_compact(architecture)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def decode_architecture(
        self, 
        encoded: Union[str, List, Dict, np.ndarray], 
        format_type: ArchitectureFormat
    ) -> Dict[str, Any]:
        """Decode architecture from specified format"""
        if format_type == ArchitectureFormat.GENOTYPE:
            return self._decode_genotype(encoded)
        elif format_type == ArchitectureFormat.ADJACENCY_MATRIX:
            return self._decode_adjacency_matrix(encoded)
        elif format_type == ArchitectureFormat.SEQUENCE:
            return self._decode_sequence(encoded)
        elif format_type == ArchitectureFormat.TREE:
            return self._decode_tree(encoded)
        elif format_type == ArchitectureFormat.JSON:
            return self._decode_json(encoded)
        elif format_type == ArchitectureFormat.COMPACT:
            return self._decode_compact(encoded)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _encode_genotype(self, architecture: Dict[str, Any]) -> str:
        """Encode as DARTS-style genotype"""
        normal_cell = architecture.get('normal_cell', {})
        reduction_cell = architecture.get('reduction_cell', {})
        
        def cell_to_genotype(cell):
            operations = cell.get('operations', [])
            connections = cell.get('connections', [])
            
            genotype_ops = []
            for i, op in enumerate(operations):
                if isinstance(op, str):
                    op_name = op
                else:
                    op_name = op.value if hasattr(op, 'value') else str(op)
                
                # Get connection for this operation
                conn = connections[i] if i < len(connections) else 0
                genotype_ops.append((op_name, conn))
            
            return genotype_ops
        
        normal_genotype = cell_to_genotype(normal_cell)
        reduction_genotype = cell_to_genotype(reduction_cell)
        
        genotype_str = f"Genotype(normal={normal_genotype}, reduction={reduction_genotype})"
        return genotype_str
    
    def _encode_adjacency_matrix(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode as adjacency matrix"""
        # Simplified adjacency matrix for cell-based architecture
        num_nodes = architecture.get('num_nodes', 4)
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        # Fill matrix based on connections
        normal_cell = architecture.get('normal_cell', {})
        connections = normal_cell.get('connections', [])
        
        for i, conn in enumerate(connections):
            if i + 1 < num_nodes:
                matrix[conn, i + 1] = 1
        
        return matrix
    
    def _encode_sequence(self, architecture: Dict[str, Any]) -> List[int]:
        """Encode as sequence of operation indices"""
        sequence = []
        
        # Encode normal cell operations
        normal_ops = architecture.get('normal_cell', {}).get('operations', [])
        for op in normal_ops:
            op_name = op.value if hasattr(op, 'value') else str(op)
            sequence.append(self.operation_to_idx.get(op_name, 0))
        
        # Encode reduction cell operations
        reduction_ops = architecture.get('reduction_cell', {}).get('operations', [])
        for op in reduction_ops:
            op_name = op.value if hasattr(op, 'value') else str(op)
            sequence.append(self.operation_to_idx.get(op_name, 0))
        
        return sequence
    
    def _encode_tree(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Encode as tree structure"""
        tree = {
            'type': 'architecture',
            'children': []
        }
        
        # Add normal cell as child
        normal_cell = architecture.get('normal_cell', {})
        tree['children'].append({
            'type': 'normal_cell',
            'operations': normal_cell.get('operations', []),
            'connections': normal_cell.get('connections', [])
        })
        
        # Add reduction cell as child
        reduction_cell = architecture.get('reduction_cell', {})
        tree['children'].append({
            'type': 'reduction_cell',
            'operations': reduction_cell.get('operations', []),
            'connections': reduction_cell.get('connections', [])
        })
        
        return tree
    
    def _encode_json(self, architecture: Dict[str, Any]) -> str:
        """Encode as JSON string"""
        # Convert enum values to strings for JSON serialization
        json_arch = {}
        for key, value in architecture.items():
            if isinstance(value, dict):
                json_value = {}
                for k, v in value.items():
                    if isinstance(v, list):
                        json_value[k] = [item.value if hasattr(item, 'value') else str(item) for item in v]
                    else:
                        json_value[k] = v.value if hasattr(v, 'value') else v
                json_arch[key] = json_value
            else:
                json_arch[key] = value.value if hasattr(value, 'value') else value
        
        return json.dumps(json_arch, indent=2)
    
    def _encode_compact(self, architecture: Dict[str, Any]) -> str:
        """Encode as compact string representation"""
        parts = []
        
        # Encode normal cell
        normal_ops = architecture.get('normal_cell', {}).get('operations', [])
        normal_str = ''.join([str(self.operation_to_idx.get(op.value if hasattr(op, 'value') else str(op), 0)) for op in normal_ops])
        parts.append(f"N{normal_str}")
        
        # Encode reduction cell
        reduction_ops = architecture.get('reduction_cell', {}).get('operations', [])
        reduction_str = ''.join([str(self.operation_to_idx.get(op.value if hasattr(op, 'value') else str(op), 0)) for op in reduction_ops])
        parts.append(f"R{reduction_str}")
        
        return '_'.join(parts)
    
    def _decode_genotype(self, genotype_str: str) -> Dict[str, Any]:
        """Decode from DARTS-style genotype"""
        # This is a simplified decoder - in practice, you'd need proper parsing
        architecture = {
            'normal_cell': {'operations': [], 'connections': []},
            'reduction_cell': {'operations': [], 'connections': []}
        }
        
        # Extract operations from genotype string (simplified)
        if 'normal=' in genotype_str:
            # Parse normal cell operations
            architecture['normal_cell']['operations'] = ['conv_3x3', 'dwise_conv_3x3']
            architecture['normal_cell']['connections'] = [0, 1]
        
        if 'reduction=' in genotype_str:
            # Parse reduction cell operations
            architecture['reduction_cell']['operations'] = ['conv_3x3', 'max_pool_3x3']
            architecture['reduction_cell']['connections'] = [0, 1]
        
        return architecture
    
    def _decode_adjacency_matrix(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Decode from adjacency matrix"""
        connections = []
        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                if matrix[i, j] == 1:
                    connections.append(i)
                    break
            else:
                connections.append(0)
        
        return {
            'normal_cell': {
                'operations': ['conv_3x3'] * len(connections),
                'connections': connections
            },
            'reduction_cell': {
                'operations': ['conv_3x3'] * len(connections),
                'connections': connections
            }
        }
    
    def _decode_sequence(self, sequence: List[int]) -> Dict[str, Any]:
        """Decode from sequence"""
        mid_point = len(sequence) // 2
        normal_ops = [self.idx_to_operation.get(idx, 'conv_3x3') for idx in sequence[:mid_point]]
        reduction_ops = [self.idx_to_operation.get(idx, 'conv_3x3') for idx in sequence[mid_point:]]
        
        return {
            'normal_cell': {
                'operations': normal_ops,
                'connections': list(range(len(normal_ops)))
            },
            'reduction_cell': {
                'operations': reduction_ops,
                'connections': list(range(len(reduction_ops)))
            }
        }
    
    def _decode_tree(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Decode from tree structure"""
        architecture = {}
        
        for child in tree.get('children', []):
            if child['type'] in ['normal_cell', 'reduction_cell']:
                architecture[child['type']] = {
                    'operations': child.get('operations', []),
                    'connections': child.get('connections', [])
                }
        
        return architecture
    
    def _decode_json(self, json_str: str) -> Dict[str, Any]:
        """Decode from JSON string"""
        return json.loads(json_str)
    
    def _decode_compact(self, compact_str: str) -> Dict[str, Any]:
        """Decode from compact string"""
        parts = compact_str.split('_')
        architecture = {}
        
        for part in parts:
            if part.startswith('N'):
                # Normal cell
                op_indices = [int(c) for c in part[1:]]
                operations = [self.idx_to_operation.get(idx, 'conv_3x3') for idx in op_indices]
                architecture['normal_cell'] = {
                    'operations': operations,
                    'connections': list(range(len(operations)))
                }
            elif part.startswith('R'):
                # Reduction cell
                op_indices = [int(c) for c in part[1:]]
                operations = [self.idx_to_operation.get(idx, 'conv_3x3') for idx in op_indices]
                architecture['reduction_cell'] = {
                    'operations': operations,
                    'connections': list(range(len(operations)))
                }
        
        return architecture


class ArchitectureVisualizer:
    """Visualizes neural architectures"""
    
    def __init__(self):
        self.operation_colors = {
            'conv_3x3': 'lightblue',
            'conv_5x5': 'lightgreen',
            'conv_7x7': 'lightcoral',
            'conv_1x1': 'lightyellow',
            'dwise_conv_3x3': 'lightpink',
            'dwise_conv_5x5': 'lightgray',
            'max_pool_3x3': 'orange',
            'avg_pool_3x3': 'purple',
            'skip_connect': 'white',
            'zero': 'black'
        }
    
    def visualize_architecture(
        self, 
        architecture: Dict[str, Any], 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """Visualize architecture as a graph"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Visualize normal cell
        self._visualize_cell(
            architecture.get('normal_cell', {}), 
            ax1, 
            title="Normal Cell"
        )
        
        # Visualize reduction cell
        self._visualize_cell(
            architecture.get('reduction_cell', {}), 
            ax2, 
            title="Reduction Cell"
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _visualize_cell(self, cell: Dict[str, Any], ax, title: str):
        """Visualize single cell"""
        operations = cell.get('operations', [])
        connections = cell.get('connections', [])
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node('input', pos=(0, 0))
        for i, op in enumerate(operations):
            op_name = op.value if hasattr(op, 'value') else str(op)
            G.add_node(f'op_{i}', pos=(1, i), operation=op_name)
        G.add_node('output', pos=(2, len(operations) // 2))
        
        # Add edges
        for i, conn in enumerate(connections):
            if conn == -1:  # Input connection
                G.add_edge('input', f'op_{i}')
            else:
                G.add_edge(f'op_{conn}', f'op_{i}')
        
        # Connect to output
        for i in range(len(operations)):
            G.add_edge(f'op_{i}', 'output')
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if node == 'input' or node == 'output':
                node_colors.append('gray')
            else:
                op_name = G.nodes[node].get('operation', 'conv_3x3')
                node_colors.append(self.operation_colors.get(op_name, 'lightblue'))
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, 
                with_labels=True, node_size=1000, 
                font_size=8, arrows=True)
        
        ax.set_title(title)
        ax.axis('off')
    
    def plot_search_progress(
        self, 
        search_history: List[Dict[str, float]], 
        save_path: Optional[str] = None
    ):
        """Plot search progress over time"""
        iterations = list(range(len(search_history)))
        accuracies = [result.get('accuracy', 0) for result in search_history]
        latencies = [result.get('latency', 0) for result in search_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Accuracy over time
        ax1.plot(iterations, accuracies, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Search Progress - Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Latency over time
        ax2.plot(iterations, latencies, 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Search Progress - Latency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ArchitectureBenchmark:
    """Benchmarking utilities for architectures"""
    
    def __init__(self, benchmark_dir: str = "nas_benchmark"):
        self.benchmark_dir = benchmark_dir
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Load existing benchmark data
        self.benchmark_data = self._load_benchmark_data()
    
    def _load_benchmark_data(self) -> Dict[str, BenchmarkResult]:
        """Load existing benchmark data"""
        benchmark_file = os.path.join(self.benchmark_dir, "benchmark_data.pkl")
        
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    
    def _save_benchmark_data(self):
        """Save benchmark data"""
        benchmark_file = os.path.join(self.benchmark_dir, "benchmark_data.pkl")
        
        with open(benchmark_file, 'wb') as f:
            pickle.dump(self.benchmark_data, f)
    
    def add_benchmark_result(self, result: BenchmarkResult):
        """Add benchmark result"""
        self.benchmark_data[result.architecture_id] = result
        self._save_benchmark_data()
    
    def get_benchmark_result(self, architecture_id: str) -> Optional[BenchmarkResult]:
        """Get benchmark result for architecture"""
        return self.benchmark_data.get(architecture_id)
    
    def query_similar_architectures(
        self, 
        target_accuracy: float, 
        max_latency: float,
        tolerance: float = 2.0
    ) -> List[BenchmarkResult]:
        """Query architectures with similar performance"""
        similar = []
        
        for result in self.benchmark_data.values():
            if (abs(result.accuracy - target_accuracy) <= tolerance and 
                result.latency <= max_latency):
                similar.append(result)
        
        # Sort by accuracy (descending)
        similar.sort(key=lambda x: x.accuracy, reverse=True)
        
        return similar
    
    def get_pareto_frontier(self) -> List[BenchmarkResult]:
        """Get Pareto frontier of architectures (accuracy vs latency)"""
        results = list(self.benchmark_data.values())
        
        if not results:
            return []
        
        # Sort by accuracy (descending)
        results.sort(key=lambda x: x.accuracy, reverse=True)
        
        pareto_frontier = []
        min_latency = float('inf')
        
        for result in results:
            if result.latency < min_latency:
                pareto_frontier.append(result)
                min_latency = result.latency
        
        return pareto_frontier
    
    def export_benchmark_data(self, format_type: str = "csv") -> str:
        """Export benchmark data to file"""
        if format_type == "csv":
            import pandas as pd
            
            data = [result.to_dict() for result in self.benchmark_data.values()]
            df = pd.DataFrame(data)
            
            export_file = os.path.join(self.benchmark_dir, "benchmark_data.csv")
            df.to_csv(export_file, index=False)
            
            return export_file
        
        elif format_type == "json":
            data = {k: v.to_dict() for k, v in self.benchmark_data.items()}
            
            export_file = os.path.join(self.benchmark_dir, "benchmark_data.json")
            with open(export_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return export_file
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


def generate_architecture_hash(architecture: Dict[str, Any]) -> str:
    """Generate unique hash for architecture"""
    # Convert architecture to canonical string representation
    encoder = ArchitectureEncoder()
    arch_str = encoder.encode_architecture(architecture, ArchitectureFormat.JSON)
    
    # Generate hash
    return hashlib.md5(arch_str.encode()).hexdigest()


def compare_architectures(arch1: Dict[str, Any], arch2: Dict[str, Any]) -> float:
    """Compare similarity between two architectures"""
    encoder = ArchitectureEncoder()
    
    # Encode both architectures as sequences
    seq1 = encoder.encode_architecture(arch1, ArchitectureFormat.SEQUENCE)
    seq2 = encoder.encode_architecture(arch2, ArchitectureFormat.SEQUENCE)
    
    # Pad sequences to same length
    max_len = max(len(seq1), len(seq2))
    seq1.extend([0] * (max_len - len(seq1)))
    seq2.extend([0] * (max_len - len(seq2)))
    
    # Calculate similarity (normalized Hamming distance)
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    similarity = matches / max_len
    
    return similarity


# Example usage
if __name__ == "__main__":
    # Test architecture encoding/decoding
    sample_architecture = {
        'normal_cell': {
            'operations': ['conv_3x3', 'dwise_conv_3x3', 'conv_1x1'],
            'connections': [0, 1, 2]
        },
        'reduction_cell': {
            'operations': ['conv_3x3', 'max_pool_3x3', 'conv_1x1'],
            'connections': [0, 1, 2]
        },
        'num_cells': 8,
        'channels': 16
    }
    
    # Test encoder
    encoder = ArchitectureEncoder()
    
    # Test different formats
    genotype = encoder.encode_architecture(sample_architecture, ArchitectureFormat.GENOTYPE)
    sequence = encoder.encode_architecture(sample_architecture, ArchitectureFormat.SEQUENCE)
    compact = encoder.encode_architecture(sample_architecture, ArchitectureFormat.COMPACT)
    
    print("NAS Utilities Implementation Created Successfully!")
    print(f"Genotype: {genotype}")
    print(f"Sequence: {sequence}")
    print(f"Compact: {compact}")
    
    # Test architecture hash
    arch_hash = generate_architecture_hash(sample_architecture)
    print(f"Architecture hash: {arch_hash}")
    
    # Test visualizer
    visualizer = ArchitectureVisualizer()
    print("Architecture visualizer ready!")
    
    # Test benchmark
    benchmark = ArchitectureBenchmark()
    print("Architecture benchmark system ready!")
    
    print("NAS utilities implementation completed!")