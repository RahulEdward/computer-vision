#!/usr/bin/env python3
"""
Computer Genie AI Features - Cutting-Edge AI Components
======================================================

यह module Computer Genie के लिए advanced AI features प्रदान करता है:

🧠 Core AI Features:
- Multi-modal transformer models (screen + audio + intent)
- Reinforcement learning agents for user corrections
- Few-shot learning for custom automation
- Neural architecture search for auto-optimization
- Federated learning for privacy-preserving improvements

🔬 Advanced Features:
- Explainable AI with visual attention maps
- Adversarial training for robust detection
- Continual learning without forgetting
- Multi-task learning (detection + OCR + intent)
- Knowledge distillation for smaller models

🌍 Global Features:
- Offline operation support
- 100+ language support
- User preference adaptation
- Privacy-first design

Author: Computer Genie AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"

# Import all AI feature modules
from . import multimodal_transformers
from . import reinforcement_learning
from . import few_shot_learning
from . import neural_architecture_search
from . import federated_learning
from . import explainable_ai
from . import adversarial_training
from . import continual_learning
from . import multitask_learning
from . import knowledge_distillation

__all__ = [
    "multimodal_transformers",
    "reinforcement_learning", 
    "few_shot_learning",
    "neural_architecture_search",
    "federated_learning",
    "explainable_ai",
    "adversarial_training",
    "continual_learning",
    "multitask_learning",
    "knowledge_distillation"
]