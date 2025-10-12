"""
Neural Architecture Search (NAS) Framework
न्यूरल आर्किटेक्चर खोज ढांचा

Advanced neural architecture search implementation with multiple search strategies,
search spaces, and optimization methods for automated model design.

Features:
- Differentiable Architecture Search (DARTS)
- Progressive Dynamic Hurdles (PDARTS)
- Efficient Neural Architecture Search (ENAS)
- Random Search and Evolutionary Search
- Hardware-Aware NAS
- Multi-Objective NAS
- Supernet Training and One-Shot NAS
- Architecture Performance Prediction
- Search Space Design and Optimization
- Resource-Constrained Architecture Search

Components:
- search_space: Defines the architecture search space
- search_strategies: Various NAS algorithms and strategies
- supernet_trainer: Training supernetworks for one-shot NAS
- architecture_evaluator: Evaluating and ranking architectures
- hardware_predictor: Hardware-aware performance prediction
- multi_objective_optimizer: Multi-objective architecture optimization
"""

from .search_space import (
    SearchSpace,
    CellSearchSpace,
    MacroSearchSpace,
    MicroSearchSpace,
    Operation,
    OperationType,
    ConnectionType,
    OperationConfig,
    SearchSpaceConfig
)

from .search_strategies import (
    SearchStrategy,
    SearchConfig,
    BaseSearchStrategy,
    RandomSearch,
    DARTS,
    EvolutionarySearch,
    RegularizedEvolution,
    ArchitectureEvaluator as StrategyEvaluator
)

from .supernet_training import (
    SupernetTrainer,
    SupernetConfig,
    SupernetTrainingStrategy,
    SamplingStrategy,
    MixedOperation,
    SupernetCell,
    Supernet
)

from .architecture_evaluation import (
    ArchitectureEvaluator,
    EvaluationMetric,
    EvaluationMode,
    EvaluationConfig,
    EvaluationResult,
    ArchitectureTrainer,
    RobustnessEvaluator,
    CalibrationEvaluator
)

from .hardware_aware import (
    HardwareType,
    MetricType,
    HardwareConstraints,
    HardwareProfile,
    OperationProfiler,
    HardwarePredictor,
    HardwareAwareOptimizer
)

from .nas_utils import (
    ArchitectureFormat,
    BenchmarkResult,
    ArchitectureEncoder,
    ArchitectureVisualizer,
    ArchitectureBenchmark
)

__all__ = [
    # Search Space
    'SearchSpace',
    'CellSearchSpace', 
    'MacroSearchSpace',
    'MicroSearchSpace',
    'Operation',
    'OperationType',
    'ConnectionType',
    'OperationConfig',
    'SearchSpaceConfig',
    
    # Search Strategies
    'SearchStrategy',
    'SearchConfig',
    'BaseSearchStrategy',
    'RandomSearch',
    'DARTS',
    'EvolutionarySearch',
    'RegularizedEvolution',
    'StrategyEvaluator',
    
    # Supernet Training
    'SupernetTrainer',
    'SupernetConfig',
    'SupernetTrainingStrategy',
    'SamplingStrategy',
    'MixedOperation',
    'SupernetCell',
    'Supernet',
    
    # Architecture Evaluation
    'ArchitectureEvaluator',
    'EvaluationMetric',
    'EvaluationMode',
    'EvaluationConfig',
    'EvaluationResult',
    'ArchitectureTrainer',
    'RobustnessEvaluator',
    'CalibrationEvaluator',
    
    # Hardware Aware
    'HardwareType',
    'MetricType',
    'HardwareConstraints',
    'HardwareProfile',
    'OperationProfiler',
    'HardwarePredictor',
    'HardwareAwareOptimizer',
    
    # NAS Utilities
    'ArchitectureFormat',
    'BenchmarkResult',
    'ArchitectureEncoder',
    'ArchitectureVisualizer',
    'ArchitectureBenchmark'
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI"
__description__ = "Advanced Neural Architecture Search Framework"