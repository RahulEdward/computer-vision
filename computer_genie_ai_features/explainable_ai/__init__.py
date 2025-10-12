"""
Explainable AI Module for Computer Genie
व्याख्यात्मक AI मॉड्यूल - Computer Genie के लिए

This module provides explainable AI capabilities with visual attention maps
and decision reasoning for transparent AI decision-making.

Features:
- Visual attention maps showing where the AI is looking
- Decision reasoning and explanation generation
- Confidence scoring and uncertainty quantification
- Interactive explanation interfaces
- Multi-modal explanation (visual + textual)
- Real-time explanation generation
- User-friendly explanation visualization

मुख्य विशेषताएं:
- दृश्य ध्यान मानचित्र जो दिखाते हैं कि AI कहाँ देख रहा है
- निर्णय तर्क और व्याख्या उत्पादन
- विश्वास स्कोरिंग और अनिश्चितता मापन
- इंटरैक्टिव व्याख्या इंटरफेस
- बहु-मोडल व्याख्या (दृश्य + पाठ्य)
"""

from .attention_visualizer import AttentionVisualizer, AttentionMap
from .explanation_generator import ExplanationGenerator, DecisionExplainer
from .confidence_estimator import ConfidenceEstimator, UncertaintyQuantifier
from .interactive_explainer import InteractiveExplainer, ExplanationInterface

__all__ = [
    'AttentionVisualizer',
    'AttentionMap', 
    'ExplanationGenerator',
    'DecisionExplainer',
    'ConfidenceEstimator',
    'UncertaintyQuantifier',
    'InteractiveExplainer',
    'ExplanationInterface'
]