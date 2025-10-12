"""
Explanation Generator for Computer Genie
व्याख्या जनरेटर - Computer Genie के लिए

Generates human-readable explanations for AI decisions and model predictions
in multiple languages with contextual reasoning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
from datetime import datetime
from enum import Enum
import re


class ExplanationType(Enum):
    """Types of explanations that can be generated"""
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"
    COUNTERFACTUAL = "counterfactual"
    EXAMPLE_BASED = "example_based"
    RULE_BASED = "rule_based"
    ATTENTION_BASED = "attention_based"


class LanguageSupport(Enum):
    """Supported languages for explanations"""
    ENGLISH = "en"
    HINDI = "hi"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation"""
    language: LanguageSupport = LanguageSupport.ENGLISH
    explanation_types: List[ExplanationType] = None
    max_explanation_length: int = 500
    include_confidence_intervals: bool = True
    include_alternative_decisions: bool = True
    technical_level: str = "intermediate"  # "basic", "intermediate", "advanced"
    include_visual_references: bool = True
    personalization_enabled: bool = True
    
    def __post_init__(self):
        if self.explanation_types is None:
            self.explanation_types = [
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.ATTENTION_BASED,
                ExplanationType.DECISION_PATH
            ]


class DecisionExplainer:
    """Explains individual AI decisions with reasoning"""
    
    def __init__(self, config: ExplanationConfig = None):
        self.config = config or ExplanationConfig()
        self.explanation_templates = self._load_explanation_templates()
        self.decision_history = []
        
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different languages"""
        templates = {
            "en": {
                "confidence_high": "I am {confidence:.1%} confident in this decision because",
                "confidence_medium": "I am moderately confident ({confidence:.1%}) in this decision because",
                "confidence_low": "I have low confidence ({confidence:.1%}) in this decision because",
                "feature_importance": "The most important factors were: {features}",
                "attention_focus": "I focused primarily on {regions} in the image",
                "decision_path": "My reasoning process: {steps}",
                "alternative": "Alternatively, I considered {alternative} with {alt_confidence:.1%} confidence",
                "uncertainty": "I'm uncertain about {uncertain_aspects}",
                "element_detection": "I detected a {element_type} element at {location} because {reasoning}",
                "action_recommendation": "I recommend {action} because {justification}"
            },
            "hi": {
                "confidence_high": "मुझे इस निर्णय पर {confidence:.1%} विश्वास है क्योंकि",
                "confidence_medium": "मुझे इस निर्णय पर मध्यम विश्वास ({confidence:.1%}) है क्योंकि",
                "confidence_low": "मुझे इस निर्णय पर कम विश्वास ({confidence:.1%}) है क्योंकि",
                "feature_importance": "सबसे महत्वपूर्ण कारक थे: {features}",
                "attention_focus": "मैंने मुख्य रूप से छवि में {regions} पर ध्यान दिया",
                "decision_path": "मेरी तर्क प्रक्रिया: {steps}",
                "alternative": "वैकल्पिक रूप से, मैंने {alternative} पर {alt_confidence:.1%} विश्वास के साथ विचार किया",
                "uncertainty": "मुझे {uncertain_aspects} के बारे में अनिश्चितता है",
                "element_detection": "मैंने {location} पर एक {element_type} तत्व का पता लगाया क्योंकि {reasoning}",
                "action_recommendation": "मैं {action} की सिफारिश करता हूं क्योंकि {justification}"
            }
        }
        return templates
    
    def explain_prediction(self, prediction: Dict[str, Any], 
                         model_outputs: Dict[str, torch.Tensor],
                         input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanation for a prediction"""
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": prediction.get("confidence", 0.0),
            "explanations": {},
            "language": self.config.language.value
        }
        
        # Generate different types of explanations
        for exp_type in self.config.explanation_types:
            if exp_type == ExplanationType.FEATURE_IMPORTANCE:
                explanation["explanations"]["feature_importance"] = \
                    self._explain_feature_importance(model_outputs, input_data)
            
            elif exp_type == ExplanationType.ATTENTION_BASED:
                explanation["explanations"]["attention"] = \
                    self._explain_attention_patterns(model_outputs)
            
            elif exp_type == ExplanationType.DECISION_PATH:
                explanation["explanations"]["decision_path"] = \
                    self._explain_decision_path(prediction, model_outputs)
            
            elif exp_type == ExplanationType.COUNTERFACTUAL:
                explanation["explanations"]["counterfactual"] = \
                    self._generate_counterfactual_explanation(prediction, input_data)
        
        # Generate natural language summary
        explanation["summary"] = self._generate_natural_language_summary(explanation)
        
        # Store in history
        self.decision_history.append(explanation)
        
        return explanation
    
    def _explain_feature_importance(self, model_outputs: Dict[str, torch.Tensor],
                                  input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain which features were most important for the decision"""
        feature_importance = {}
        
        # Extract feature importance from gradients or attention weights
        if "gradients" in model_outputs:
            gradients = model_outputs["gradients"]
            importance_scores = torch.abs(gradients).mean(dim=0)
            
            # Get top features
            top_k = min(5, len(importance_scores))
            top_indices = torch.topk(importance_scores, top_k).indices
            
            features = []
            for idx in top_indices:
                feature_name = f"feature_{idx.item()}"
                importance = importance_scores[idx].item()
                features.append({
                    "name": feature_name,
                    "importance": float(importance),
                    "description": self._get_feature_description(feature_name, input_data)
                })
            
            feature_importance["top_features"] = features
            feature_importance["explanation"] = self._format_feature_explanation(features)
        
        return feature_importance
    
    def _explain_attention_patterns(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Explain attention patterns in the model"""
        attention_explanation = {}
        
        if "attention_weights" in model_outputs:
            attention = model_outputs["attention_weights"]
            
            # Analyze attention distribution
            attention_stats = {
                "entropy": float(self._calculate_attention_entropy(attention)),
                "focus_concentration": float(self._calculate_focus_concentration(attention)),
                "spatial_distribution": self._analyze_spatial_attention(attention)
            }
            
            attention_explanation["statistics"] = attention_stats
            attention_explanation["interpretation"] = self._interpret_attention_patterns(attention_stats)
        
        return attention_explanation
    
    def _explain_decision_path(self, prediction: Dict[str, Any],
                             model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Explain the decision-making process step by step"""
        decision_path = {
            "steps": [],
            "reasoning_chain": []
        }
        
        # Simulate decision steps (in practice, this would use actual model internals)
        steps = [
            {
                "step": 1,
                "description": "Analyzed visual input and extracted features",
                "confidence": 0.9,
                "key_findings": ["Detected UI elements", "Identified text regions", "Recognized layout patterns"]
            },
            {
                "step": 2,
                "description": "Applied attention mechanism to focus on relevant regions",
                "confidence": 0.85,
                "key_findings": ["Focused on interactive elements", "Prioritized text content", "Considered spatial relationships"]
            },
            {
                "step": 3,
                "description": "Made final prediction based on integrated features",
                "confidence": prediction.get("confidence", 0.0),
                "key_findings": [f"Predicted: {prediction.get('class', 'unknown')}", 
                               f"Confidence: {prediction.get('confidence', 0.0):.2f}"]
            }
        ]
        
        decision_path["steps"] = steps
        decision_path["overall_reasoning"] = self._generate_reasoning_chain(steps)
        
        return decision_path
    
    def _generate_counterfactual_explanation(self, prediction: Dict[str, Any],
                                           input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counterfactual explanations (what would change the decision)"""
        counterfactual = {
            "current_prediction": prediction.get("class", "unknown"),
            "alternative_scenarios": []
        }
        
        # Generate hypothetical scenarios
        scenarios = [
            {
                "change": "If the button was in a different color",
                "predicted_outcome": "Might be classified as inactive",
                "confidence_change": -0.2
            },
            {
                "change": "If the text was in a different language",
                "predicted_outcome": "Language detection would change",
                "confidence_change": -0.1
            },
            {
                "change": "If the element was larger",
                "predicted_outcome": "Higher confidence in detection",
                "confidence_change": 0.15
            }
        ]
        
        counterfactual["alternative_scenarios"] = scenarios
        return counterfactual
    
    def _generate_natural_language_summary(self, explanation: Dict[str, Any]) -> str:
        """Generate a natural language summary of the explanation"""
        lang = self.config.language.value
        templates = self.explanation_templates.get(lang, self.explanation_templates["en"])
        
        confidence = explanation["confidence"]
        prediction_class = explanation["prediction"].get("class", "unknown")
        
        # Choose confidence template
        if confidence > 0.8:
            confidence_text = templates["confidence_high"].format(confidence=confidence)
        elif confidence > 0.5:
            confidence_text = templates["confidence_medium"].format(confidence=confidence)
        else:
            confidence_text = templates["confidence_low"].format(confidence=confidence)
        
        summary_parts = [confidence_text]
        
        # Add feature importance if available
        if "feature_importance" in explanation["explanations"]:
            features = explanation["explanations"]["feature_importance"].get("top_features", [])
            if features:
                feature_names = [f["name"] for f in features[:3]]
                feature_text = templates["feature_importance"].format(
                    features=", ".join(feature_names)
                )
                summary_parts.append(feature_text)
        
        # Add attention explanation if available
        if "attention" in explanation["explanations"]:
            attention_text = templates["attention_focus"].format(
                regions="the most relevant screen areas"
            )
            summary_parts.append(attention_text)
        
        return ". ".join(summary_parts) + "."
    
    def _get_feature_description(self, feature_name: str, input_data: Dict[str, Any]) -> str:
        """Get human-readable description of a feature"""
        # This would be customized based on the actual features used
        feature_descriptions = {
            "color_histogram": "Color distribution in the image",
            "edge_density": "Amount of edges and boundaries",
            "text_regions": "Areas containing text",
            "button_features": "Button-like visual characteristics",
            "spatial_layout": "Arrangement of elements on screen"
        }
        
        return feature_descriptions.get(feature_name, f"Feature: {feature_name}")
    
    def _format_feature_explanation(self, features: List[Dict]) -> str:
        """Format feature importance into readable text"""
        if not features:
            return "No significant features identified."
        
        explanations = []
        for feature in features[:3]:  # Top 3 features
            explanations.append(f"{feature['description']} (importance: {feature['importance']:.2f})")
        
        return "Key factors: " + "; ".join(explanations)
    
    def _calculate_attention_entropy(self, attention: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        attention_flat = attention.flatten()
        attention_prob = torch.softmax(attention_flat, dim=0)
        entropy = -torch.sum(attention_prob * torch.log(attention_prob + 1e-8))
        return entropy.item()
    
    def _calculate_focus_concentration(self, attention: torch.Tensor) -> float:
        """Calculate how concentrated the attention is"""
        attention_flat = attention.flatten()
        max_attention = torch.max(attention_flat)
        mean_attention = torch.mean(attention_flat)
        concentration = max_attention / (mean_attention + 1e-8)
        return concentration.item()
    
    def _analyze_spatial_attention(self, attention: torch.Tensor) -> Dict[str, Any]:
        """Analyze spatial distribution of attention"""
        if attention.dim() >= 2:
            h, w = attention.shape[-2:]
            
            # Find center of attention
            attention_2d = attention.view(-1, h, w).mean(0)
            y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            
            total_attention = torch.sum(attention_2d)
            center_y = torch.sum(y_coords * attention_2d) / total_attention
            center_x = torch.sum(x_coords * attention_2d) / total_attention
            
            return {
                "center_of_attention": [float(center_x), float(center_y)],
                "attention_spread": float(torch.std(attention_2d)),
                "max_attention_location": [int(torch.argmax(attention_2d) % w), 
                                         int(torch.argmax(attention_2d) // w)]
            }
        
        return {}
    
    def _interpret_attention_patterns(self, attention_stats: Dict[str, Any]) -> str:
        """Interpret attention statistics into readable explanation"""
        entropy = attention_stats.get("entropy", 0)
        concentration = attention_stats.get("focus_concentration", 0)
        
        if entropy < 2.0:
            focus_desc = "highly focused"
        elif entropy < 4.0:
            focus_desc = "moderately focused"
        else:
            focus_desc = "broadly distributed"
        
        if concentration > 5.0:
            concentration_desc = "very concentrated"
        elif concentration > 2.0:
            concentration_desc = "somewhat concentrated"
        else:
            concentration_desc = "evenly distributed"
        
        return f"Attention is {focus_desc} and {concentration_desc} across the input."
    
    def _generate_reasoning_chain(self, steps: List[Dict]) -> str:
        """Generate a coherent reasoning chain from decision steps"""
        reasoning_parts = []
        for step in steps:
            step_desc = f"Step {step['step']}: {step['description']}"
            if step['key_findings']:
                findings = ", ".join(step['key_findings'])
                step_desc += f" (Key findings: {findings})"
            reasoning_parts.append(step_desc)
        
        return " → ".join(reasoning_parts)


class ExplanationGenerator:
    """Main class for generating comprehensive explanations"""
    
    def __init__(self, config: ExplanationConfig = None):
        self.config = config or ExplanationConfig()
        self.decision_explainer = DecisionExplainer(config)
        self.explanation_cache = {}
        
    def generate_comprehensive_explanation(self, 
                                         prediction: Dict[str, Any],
                                         model: nn.Module,
                                         input_data: Dict[str, Any],
                                         model_outputs: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """Generate a comprehensive explanation for a model prediction"""
        
        # Extract model outputs if not provided
        if model_outputs is None:
            model_outputs = self._extract_model_outputs(model, input_data)
        
        # Generate explanation using decision explainer
        explanation = self.decision_explainer.explain_prediction(
            prediction, model_outputs, input_data
        )
        
        # Add additional context
        explanation["context"] = {
            "input_type": self._determine_input_type(input_data),
            "model_type": type(model).__name__,
            "explanation_config": {
                "language": self.config.language.value,
                "technical_level": self.config.technical_level,
                "explanation_types": [et.value for et in self.config.explanation_types]
            }
        }
        
        # Add user-specific adaptations if enabled
        if self.config.personalization_enabled:
            explanation = self._personalize_explanation(explanation, input_data)
        
        return explanation
    
    def _extract_model_outputs(self, model: nn.Module, 
                             input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract relevant outputs from model for explanation"""
        outputs = {}
        
        # This would be implemented based on the specific model architecture
        # For now, return empty dict
        return outputs
    
    def _determine_input_type(self, input_data: Dict[str, Any]) -> str:
        """Determine the type of input data"""
        if "image" in input_data:
            return "image"
        elif "text" in input_data:
            return "text"
        elif "audio" in input_data:
            return "audio"
        else:
            return "multimodal"
    
    def _personalize_explanation(self, explanation: Dict[str, Any], 
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize explanation based on user preferences and history"""
        # This would implement personalization logic
        # For now, just add a personalization flag
        explanation["personalized"] = True
        return explanation
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated explanations"""
        return {
            "total_explanations": len(self.decision_explainer.decision_history),
            "average_confidence": np.mean([
                exp["confidence"] for exp in self.decision_explainer.decision_history
            ]) if self.decision_explainer.decision_history else 0.0,
            "language_distribution": self._get_language_distribution(),
            "explanation_types_used": self._get_explanation_types_distribution()
        }
    
    def _get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of explanation languages"""
        lang_counts = {}
        for exp in self.decision_explainer.decision_history:
            lang = exp.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return lang_counts
    
    def _get_explanation_types_distribution(self) -> Dict[str, int]:
        """Get distribution of explanation types used"""
        type_counts = {}
        for exp in self.decision_explainer.decision_history:
            for exp_type in exp.get("explanations", {}).keys():
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        return type_counts


# Example usage
if __name__ == "__main__":
    # Create explanation generator
    config = ExplanationConfig(
        language=LanguageSupport.ENGLISH,
        technical_level="intermediate",
        explanation_types=[
            ExplanationType.FEATURE_IMPORTANCE,
            ExplanationType.ATTENTION_BASED,
            ExplanationType.DECISION_PATH
        ]
    )
    
    generator = ExplanationGenerator(config)
    
    # Example prediction
    prediction = {
        "class": "button",
        "confidence": 0.87,
        "bbox": [100, 200, 150, 230]
    }
    
    # Example input data
    input_data = {
        "image": np.random.rand(224, 224, 3),
        "user_id": "user123"
    }
    
    # Example model outputs
    model_outputs = {
        "attention_weights": torch.rand(1, 8, 196, 196),
        "gradients": torch.rand(512)
    }
    
    # Generate explanation
    explanation = generator.decision_explainer.explain_prediction(
        prediction, model_outputs, input_data
    )
    
    print("Generated Explanation:")
    print(json.dumps(explanation, indent=2, default=str))