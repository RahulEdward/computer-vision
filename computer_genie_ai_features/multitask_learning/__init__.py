"""
Multi-Task Learning Module for Computer Genie
मल्टी-टास्क लर्निंग मॉड्यूल - Computer Genie के लिए

This module implements multi-task learning capabilities that enable simultaneous:
- Element detection and classification
- Optical Character Recognition (OCR)
- User intent understanding
- Action prediction
- Context awareness

Features:
- Shared feature extraction backbone
- Task-specific heads for different outputs
- Dynamic task weighting and balancing
- Cross-task knowledge transfer
- Efficient multi-task optimization
- Real-time multi-task inference
"""

from .multi_task_model import (
    MultiTaskConfig,
    SharedBackbone,
    TaskHead,
    MultiTaskModel,
    TaskType,
    TaskWeight
)

from .task_balancer import (
    TaskBalancer,
    BalancingStrategy,
    DynamicWeightScheduler,
    TaskPerformanceMonitor
)

from .multi_task_trainer import (
    MultiTaskTrainer,
    MultiTaskLoss,
    TaskSpecificMetrics,
    TrainingConfig
)

from .inference_engine import (
    MultiTaskInferenceEngine,
    InferenceConfig,
    TaskOutput,
    MultiTaskResult
)

__all__ = [
    # Core model components
    'MultiTaskConfig',
    'SharedBackbone', 
    'TaskHead',
    'MultiTaskModel',
    'TaskType',
    'TaskWeight',
    
    # Task balancing
    'TaskBalancer',
    'BalancingStrategy',
    'DynamicWeightScheduler',
    'TaskPerformanceMonitor',
    
    # Training
    'MultiTaskTrainer',
    'MultiTaskLoss',
    'TaskSpecificMetrics',
    'TrainingConfig',
    
    # Inference
    'MultiTaskInferenceEngine',
    'InferenceConfig',
    'TaskOutput',
    'MultiTaskResult'
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"