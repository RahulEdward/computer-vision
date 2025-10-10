"""
Continual Learning Framework
निरंतर शिक्षा ढांचा

Advanced continual learning system for handling new tasks and data streams
while preventing catastrophic forgetting and enabling knowledge transfer.

Features:
- Elastic Weight Consolidation (EWC) for parameter importance
- Progressive Neural Networks for task-specific capacity
- Learning without Forgetting (LwF) for knowledge distillation
- Memory-based approaches (Experience Replay, GEM, A-GEM)
- Meta-learning for few-shot adaptation
- Task-incremental and class-incremental learning
- Domain adaptation and transfer learning
- Regularization-based continual learning
- Architecture-based approaches
- Rehearsal and pseudo-rehearsal methods
- Online and offline continual learning
- Multi-task and multi-domain scenarios
"""

from .continual_trainer import (
    ContinualTrainer,
    ContinualConfig,
    TaskInfo,
    LearningStrategy,
    MemoryStrategy
)

from .memory_manager import (
    MemoryManager,
    ExperienceReplay,
    GradientEpisodicMemory,
    MemoryConfig,
    MemoryBuffer
)

from .regularization_methods import (
    RegularizationMethod,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    PackNet,
    RegularizationConfig
)

from .architecture_methods import (
    ProgressiveNeuralNetwork,
    DynamicExpandableNetwork,
    ArchitectureConfig,
    TaskSpecificLayers
)

from .meta_learning import (
    MetaLearner,
    MAML,
    Reptile,
    MetaConfig,
    TaskSampler
)

from .evaluation_metrics import (
    ContinualMetrics,
    ForgettingMeasure,
    TransferMeasure,
    EvaluationConfig
)

__all__ = [
    'ContinualTrainer',
    'ContinualConfig', 
    'TaskInfo',
    'LearningStrategy',
    'MemoryStrategy',
    'MemoryManager',
    'ExperienceReplay',
    'GradientEpisodicMemory',
    'MemoryConfig',
    'MemoryBuffer',
    'RegularizationMethod',
    'ElasticWeightConsolidation',
    'LearningWithoutForgetting',
    'PackNet',
    'RegularizationConfig',
    'ProgressiveNeuralNetwork',
    'DynamicExpandableNetwork',
    'ArchitectureConfig',
    'TaskSpecificLayers',
    'MetaLearner',
    'MAML',
    'Reptile',
    'MetaConfig',
    'TaskSampler',
    'ContinualMetrics',
    'ForgettingMeasure',
    'TransferMeasure',
    'EvaluationConfig'
]