#!/usr/bin/env python3
"""
Few-Shot Learning Module for Computer Genie
==========================================

यह module few-shot learning capabilities provide करता है जो 2-3 examples से custom automation सीखता है।

Features:
- Meta-learning algorithms
- Prototype networks
- Model-agnostic meta-learning (MAML)
- Task adaptation from few examples
- Custom automation generation
- Transfer learning

Author: Computer Genie AI Team
"""

from .meta_learner import MetaLearner, MAMLLearner
from .prototype_network import PrototypeNetwork, RelationNetwork
from .task_adapter import TaskAdapter, CustomAutomationGenerator
from .few_shot_trainer import FewShotTrainer, MetaTrainer
from .data_augmentation import DataAugmenter, ExampleGenerator

__all__ = [
    'MetaLearner',
    'MAMLLearner',
    'PrototypeNetwork', 
    'RelationNetwork',
    'TaskAdapter',
    'CustomAutomationGenerator',
    'FewShotTrainer',
    'MetaTrainer',
    'DataAugmenter',
    'ExampleGenerator'
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"