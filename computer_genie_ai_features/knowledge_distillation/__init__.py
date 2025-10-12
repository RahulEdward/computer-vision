"""
Knowledge Distillation for Model Compression
मॉडल संपीड़न के लिए ज्ञान आसवन

Advanced knowledge distillation framework for creating smaller, efficient models
while maintaining performance through teacher-student learning paradigms.

Features:
- Teacher-Student Training: Traditional knowledge distillation with temperature scaling
- Multi-Teacher Distillation: Learning from multiple expert models
- Self-Distillation: Progressive self-improvement through iterative distillation
- Feature-based Distillation: Intermediate layer knowledge transfer
- Attention Transfer: Distilling attention mechanisms and spatial relationships
- Progressive Distillation: Gradual model compression with staged training
- Online Distillation: Mutual learning between peer networks
- Quantization-aware Distillation: Combined compression and quantization
- Cross-modal Distillation: Knowledge transfer across different modalities
- Adaptive Distillation: Dynamic loss weighting and temperature adjustment
- Structured Distillation: Preserving model architecture relationships
- Ensemble Distillation: Distilling knowledge from model ensembles
"""

from .knowledge_distiller import (
    KnowledgeDistiller,
    DistillationConfig,
    DistillationType,
    TemperatureScheduler
)

from .teacher_student import (
    TeacherStudentTrainer,
    MultiTeacherDistiller,
    SelfDistiller,
    OnlineDistiller
)

from .feature_distillation import (
    FeatureDistiller,
    AttentionTransfer,
    StructuredDistiller,
    LayerMatcher
)

from .distillation_losses import (
    DistillationLoss,
    KLDivergenceLoss,
    AttentionLoss,
    FeatureMatchingLoss,
    RelationLoss
)

__all__ = [
    'KnowledgeDistiller',
    'DistillationConfig',
    'DistillationType',
    'TemperatureScheduler',
    'TeacherStudentTrainer',
    'MultiTeacherDistiller',
    'SelfDistiller',
    'OnlineDistiller',
    'FeatureDistiller',
    'AttentionTransfer',
    'StructuredDistiller',
    'LayerMatcher',
    'DistillationLoss',
    'KLDivergenceLoss',
    'AttentionLoss',
    'FeatureMatchingLoss',
    'RelationLoss'
]