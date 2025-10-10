"""
Adversarial Training Module for Computer Genie
कंप्यूटर जीनी के लिए विरोधी प्रशिक्षण मॉड्यूल

This module implements adversarial training techniques to improve model robustness
against adversarial attacks and enhance generalization performance.

Features:
- Adversarial attack generation (FGSM, PGD, C&W, etc.)
- Adversarial training strategies
- Robust optimization techniques
- Defense mechanisms
- Attack detection and mitigation
- Certified defenses
- Adaptive attacks handling
- Robustness evaluation metrics
- Multi-scale adversarial training
- Domain adaptation robustness

मुख्य विशेषताएं:
- विरोधी हमला उत्पादन (FGSM, PGD, C&W, आदि)
- विरोधी प्रशिक्षण रणनीतियां
- मजबूत अनुकूलन तकनीकें
- रक्षा तंत्र
- हमला पहचान और शमन
- प्रमाणित रक्षा
- अनुकूली हमलों का संचालन
- मजबूती मूल्यांकन मेट्रिक्स
- बहु-स्तरीय विरोधी प्रशिक्षण
- डोमेन अनुकूलन मजबूती
"""

from .adversarial_attacks import (
    AttackType,
    AttackConfig,
    AdversarialAttack,
    FGSMAttack,
    PGDAttack,
    CWAttack,
    AutoAttack,
    AttackEvaluator
)

from .adversarial_trainer import (
    TrainingStrategy,
    RobustnessConfig,
    AdversarialTrainer,
    RobustOptimizer,
    DefenseEvaluator
)

from .defense_mechanisms import (
    DefenseType,
    DefenseConfig,
    AdversarialDefense,
    InputTransformation,
    ModelEnsemble,
    CertifiedDefense
)

from .robustness_evaluator import (
    RobustnessMetric,
    EvaluationConfig,
    RobustnessEvaluator,
    AttackSuccess,
    RobustnessReport
)

__all__ = [
    # Attack classes
    'AttackType',
    'AttackConfig', 
    'AdversarialAttack',
    'FGSMAttack',
    'PGDAttack',
    'CWAttack',
    'AutoAttack',
    'AttackEvaluator',
    
    # Training classes
    'TrainingStrategy',
    'RobustnessConfig',
    'AdversarialTrainer',
    'RobustOptimizer',
    'DefenseEvaluator',
    
    # Defense classes
    'DefenseType',
    'DefenseConfig',
    'AdversarialDefense',
    'InputTransformation',
    'ModelEnsemble',
    'CertifiedDefense',
    
    # Evaluation classes
    'RobustnessMetric',
    'EvaluationConfig',
    'RobustnessEvaluator',
    'AttackSuccess',
    'RobustnessReport'
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"
__description__ = "Adversarial training and robustness enhancement for Computer Genie"