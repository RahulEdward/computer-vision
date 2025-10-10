"""
Interactive Explainer for Computer Genie
इंटरैक्टिव व्याख्याकर्ता - Computer Genie के लिए

Provides interactive explanation interfaces that allow users to explore
AI decisions through visual and conversational interfaces.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
import threading
import queue
from datetime import datetime


class InteractionMode(Enum):
    """Types of interaction modes for explanations"""
    VISUAL_EXPLORATION = "visual_exploration"
    CONVERSATIONAL = "conversational"
    COMPARATIVE = "comparative"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION_ANALYSIS = "attention_analysis"


class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class InteractiveConfig:
    """Configuration for interactive explanations"""
    default_mode: InteractionMode = InteractionMode.VISUAL_EXPLORATION
    explanation_level: ExplanationLevel = ExplanationLevel.INTERMEDIATE
    enable_real_time_updates: bool = True
    max_interaction_history: int = 100
    auto_save_interactions: bool = True
    supported_languages: List[str] = None
    ui_theme: str = "light"  # "light", "dark", "auto"
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "hi", "es", "fr", "de"]


class ExplanationInterface:
    """Base class for explanation interfaces"""
    
    def __init__(self, config: InteractiveConfig = None):
        self.config = config or InteractiveConfig()
        self.interaction_history = []
        self.current_explanation = None
        self.user_feedback = []
        
    def display_explanation(self, explanation: Dict[str, Any]) -> None:
        """Display explanation to user"""
        raise NotImplementedError("Subclasses must implement display_explanation")
    
    def handle_user_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interaction and return response"""
        raise NotImplementedError("Subclasses must implement handle_user_interaction")
    
    def update_explanation(self, new_data: Dict[str, Any]) -> None:
        """Update explanation with new data"""
        self.current_explanation = new_data
        if self.config.enable_real_time_updates:
            self.display_explanation(new_data)
    
    def record_interaction(self, interaction_type: str, data: Dict[str, Any]) -> None:
        """Record user interaction for analysis"""
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "data": data,
            "explanation_id": id(self.current_explanation) if self.current_explanation else None
        }
        
        self.interaction_history.append(interaction_record)
        
        # Limit history size
        if len(self.interaction_history) > self.config.max_interaction_history:
            self.interaction_history = self.interaction_history[-self.config.max_interaction_history:]


class VisualExplorationInterface(ExplanationInterface):
    """Visual interface for exploring AI explanations"""
    
    def __init__(self, config: InteractiveConfig = None):
        super().__init__(config)
        self.figure = None
        self.axes = None
        self.attention_overlay = None
        self.confidence_display = None
        
    def create_visual_interface(self, image: np.ndarray, 
                              attention_map: np.ndarray,
                              predictions: Dict[str, Any]) -> None:
        """Create interactive visual interface"""
        self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.figure.suptitle('AI Decision Explanation - Interactive Explorer')
        
        # Original image
        self.axes[0, 0].imshow(image)
        self.axes[0, 0].set_title('Original Image')
        self.axes[0, 0].axis('off')
        
        # Attention heatmap
        im = self.axes[0, 1].imshow(attention_map, cmap='jet', alpha=0.7)
        self.axes[0, 1].set_title('Attention Map')
        self.axes[0, 1].axis('off')
        plt.colorbar(im, ax=self.axes[0, 1])
        
        # Overlay
        overlay = self._create_overlay(image, attention_map)
        self.axes[1, 0].imshow(overlay)
        self.axes[1, 0].set_title('Attention Overlay')
        self.axes[1, 0].axis('off')
        
        # Prediction details
        self._display_prediction_details(predictions)
        
        # Add interactive elements
        self._add_interactive_controls()
        
        plt.tight_layout()
        plt.show()
    
    def _create_overlay(self, image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Create attention overlay on image"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize attention map to match image
        attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create colored heatmap
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        return overlay
    
    def _display_prediction_details(self, predictions: Dict[str, Any]) -> None:
        """Display prediction details in text format"""
        self.axes[1, 1].axis('off')
        
        details_text = f"""
        Prediction: {predictions.get('class', 'Unknown')}
        Confidence: {predictions.get('confidence', 0.0):.2%}
        
        Top Predictions:
        """
        
        # Add top predictions if available
        if 'top_predictions' in predictions:
            for i, (cls, conf) in enumerate(predictions['top_predictions'][:5]):
                details_text += f"  {i+1}. {cls}: {conf:.2%}\n"
        
        # Add feature importance if available
        if 'feature_importance' in predictions:
            details_text += "\nKey Features:\n"
            for feature in predictions['feature_importance'][:3]:
                details_text += f"  • {feature['name']}: {feature['importance']:.3f}\n"
        
        self.axes[1, 1].text(0.05, 0.95, details_text, transform=self.axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    def _add_interactive_controls(self) -> None:
        """Add interactive controls to the interface"""
        # Add sliders for attention threshold
        ax_threshold = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.threshold_slider = Slider(ax_threshold, 'Attention Threshold', 0.0, 1.0, valinit=0.5)
        self.threshold_slider.on_changed(self._update_attention_threshold)
        
        # Add button for different explanation modes
        ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.mode_button = Button(ax_button, 'Switch Mode')
        self.mode_button.on_clicked(self._switch_explanation_mode)
    
    def _update_attention_threshold(self, val: float) -> None:
        """Update attention visualization based on threshold"""
        self.record_interaction("threshold_change", {"new_threshold": val})
        # Implementation would update the visualization
        print(f"Attention threshold updated to: {val:.2f}")
    
    def _switch_explanation_mode(self, event) -> None:
        """Switch between different explanation modes"""
        self.record_interaction("mode_switch", {"event": str(event)})
        print("Switching explanation mode...")
    
    def display_explanation(self, explanation: Dict[str, Any]) -> None:
        """Display explanation in visual format"""
        if 'image' in explanation and 'attention_map' in explanation:
            self.create_visual_interface(
                explanation['image'],
                explanation['attention_map'],
                explanation.get('predictions', {})
            )


class ConversationalInterface(ExplanationInterface):
    """Conversational interface for AI explanations"""
    
    def __init__(self, config: InteractiveConfig = None):
        super().__init__(config)
        self.conversation_history = []
        self.question_templates = self._load_question_templates()
        
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load templates for common questions"""
        return {
            "confidence": [
                "How confident are you in this prediction?",
                "Why are you {confidence_level} confident?",
                "What would make you more confident?"
            ],
            "features": [
                "What features did you focus on?",
                "Why is {feature_name} important?",
                "What if {feature_name} was different?"
            ],
            "alternatives": [
                "What other predictions did you consider?",
                "Why didn't you choose {alternative_class}?",
                "What would change your prediction?"
            ],
            "explanation": [
                "Can you explain this decision?",
                "Walk me through your reasoning",
                "What am I looking at here?"
            ]
        }
    
    def process_user_question(self, question: str, 
                            explanation_context: Dict[str, Any]) -> str:
        """Process user question and generate response"""
        question_lower = question.lower()
        
        # Classify question type
        question_type = self._classify_question(question_lower)
        
        # Generate appropriate response
        response = self._generate_response(question_type, question, explanation_context)
        
        # Record conversation
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_question": question,
            "question_type": question_type,
            "ai_response": response
        })
        
        return response
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked"""
        confidence_keywords = ["confident", "sure", "certain", "trust"]
        feature_keywords = ["feature", "important", "focus", "look", "see"]
        alternative_keywords = ["other", "alternative", "different", "else", "instead"]
        explanation_keywords = ["why", "how", "explain", "reason", "because"]
        
        if any(keyword in question for keyword in confidence_keywords):
            return "confidence"
        elif any(keyword in question for keyword in feature_keywords):
            return "features"
        elif any(keyword in question for keyword in alternative_keywords):
            return "alternatives"
        elif any(keyword in question for keyword in explanation_keywords):
            return "explanation"
        else:
            return "general"
    
    def _generate_response(self, question_type: str, question: str,
                         context: Dict[str, Any]) -> str:
        """Generate response based on question type and context"""
        if question_type == "confidence":
            confidence = context.get("confidence", 0.0)
            if confidence > 0.8:
                return f"I'm very confident ({confidence:.1%}) in this prediction because the key features strongly indicate this classification. The attention patterns show clear focus on relevant elements."
            elif confidence > 0.5:
                return f"I have moderate confidence ({confidence:.1%}) in this prediction. While the main features support this classification, there's some uncertainty due to ambiguous elements."
            else:
                return f"I have low confidence ({confidence:.1%}) in this prediction. The features are not clearly distinguishable, and I would recommend human verification."
        
        elif question_type == "features":
            features = context.get("feature_importance", [])
            if features:
                top_features = features[:3]
                feature_text = ", ".join([f["name"] for f in top_features])
                return f"I focused primarily on these features: {feature_text}. These were the most discriminative elements for making this classification."
            else:
                return "I analyzed various visual features including colors, shapes, textures, and spatial relationships to make this prediction."
        
        elif question_type == "alternatives":
            alternatives = context.get("alternative_predictions", [])
            if alternatives:
                alt_text = ", ".join([f"{alt['class']} ({alt['confidence']:.1%})" for alt in alternatives[:3]])
                return f"I also considered these alternatives: {alt_text}. However, the current prediction had the strongest supporting evidence."
            else:
                return "This was the most likely classification based on the available evidence. Other possibilities had significantly lower confidence scores."
        
        elif question_type == "explanation":
            return self._generate_detailed_explanation(context)
        
        else:
            return "I can explain my confidence level, the features I focused on, alternative predictions I considered, or walk you through my reasoning process. What would you like to know more about?"
    
    def _generate_detailed_explanation(self, context: Dict[str, Any]) -> str:
        """Generate detailed explanation of the decision process"""
        explanation_parts = []
        
        # Add prediction info
        prediction = context.get("prediction", {})
        if prediction:
            explanation_parts.append(f"I predicted '{prediction.get('class', 'unknown')}' with {prediction.get('confidence', 0.0):.1%} confidence.")
        
        # Add reasoning process
        explanation_parts.append("My reasoning process involved:")
        explanation_parts.append("1. Analyzing the visual features and patterns")
        explanation_parts.append("2. Applying attention mechanisms to focus on relevant areas")
        explanation_parts.append("3. Comparing against learned patterns from training data")
        explanation_parts.append("4. Calculating confidence based on feature strength and consistency")
        
        # Add specific details if available
        if "attention_focus" in context:
            explanation_parts.append(f"I focused particularly on {context['attention_focus']} in the image.")
        
        return " ".join(explanation_parts)
    
    def display_explanation(self, explanation: Dict[str, Any]) -> None:
        """Display explanation in conversational format"""
        print("\n=== AI Explanation (Conversational Mode) ===")
        print(f"Prediction: {explanation.get('prediction', {}).get('class', 'Unknown')}")
        print(f"Confidence: {explanation.get('confidence', 0.0):.1%}")
        print("\nAsk me anything about this prediction!")
        print("Examples: 'Why are you confident?', 'What features did you focus on?', 'What alternatives did you consider?'")
        
        # Start interactive conversation
        self._start_conversation_loop(explanation)
    
    def _start_conversation_loop(self, explanation: Dict[str, Any]) -> None:
        """Start interactive conversation loop"""
        print("\nType 'quit' to exit the conversation.")
        
        while True:
            try:
                user_input = input("\nYour question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Feel free to ask more questions anytime.")
                    break
                
                if user_input:
                    response = self.process_user_question(user_input, explanation)
                    print(f"\nAI: {response}")
                
            except KeyboardInterrupt:
                print("\nConversation ended.")
                break
            except Exception as e:
                print(f"\nError processing question: {e}")


class ComparativeInterface(ExplanationInterface):
    """Interface for comparing multiple AI decisions"""
    
    def __init__(self, config: InteractiveConfig = None):
        super().__init__(config)
        self.comparison_data = []
        
    def compare_explanations(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple explanations side by side"""
        if len(explanations) < 2:
            return {"error": "Need at least 2 explanations to compare"}
        
        comparison = {
            "num_explanations": len(explanations),
            "predictions": [exp.get("prediction", {}) for exp in explanations],
            "confidence_comparison": self._compare_confidences(explanations),
            "feature_comparison": self._compare_features(explanations),
            "attention_comparison": self._compare_attention_patterns(explanations),
            "consensus_analysis": self._analyze_consensus(explanations)
        }
        
        return comparison
    
    def _compare_confidences(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare confidence levels across explanations"""
        confidences = [exp.get("confidence", 0.0) for exp in explanations]
        
        return {
            "confidences": confidences,
            "average_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "confidence_agreement": np.std(confidences) < 0.1  # Low std indicates agreement
        }
    
    def _compare_features(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare feature importance across explanations"""
        all_features = {}
        
        for i, exp in enumerate(explanations):
            features = exp.get("feature_importance", [])
            for feature in features:
                feature_name = feature.get("name", f"feature_{i}")
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(feature.get("importance", 0.0))
        
        # Calculate feature consistency
        feature_consistency = {}
        for feature_name, importances in all_features.items():
            feature_consistency[feature_name] = {
                "mean_importance": np.mean(importances),
                "std_importance": np.std(importances),
                "consistency": np.std(importances) < 0.1
            }
        
        return {
            "feature_consistency": feature_consistency,
            "common_features": [f for f, data in feature_consistency.items() 
                              if data["consistency"]],
            "inconsistent_features": [f for f, data in feature_consistency.items() 
                                    if not data["consistency"]]
        }
    
    def _compare_attention_patterns(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare attention patterns across explanations"""
        attention_maps = []
        
        for exp in explanations:
            if "attention_map" in exp:
                attention_maps.append(exp["attention_map"])
        
        if len(attention_maps) < 2:
            return {"error": "Not enough attention maps for comparison"}
        
        # Calculate correlation between attention maps
        correlations = []
        for i in range(len(attention_maps)):
            for j in range(i + 1, len(attention_maps)):
                corr = np.corrcoef(attention_maps[i].flatten(), 
                                 attention_maps[j].flatten())[0, 1]
                correlations.append(corr)
        
        return {
            "attention_correlations": correlations,
            "average_correlation": np.mean(correlations),
            "attention_agreement": np.mean(correlations) > 0.7  # High correlation indicates agreement
        }
    
    def _analyze_consensus(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus across multiple explanations"""
        predictions = [exp.get("prediction", {}).get("class", "unknown") 
                      for exp in explanations]
        
        # Count prediction frequencies
        prediction_counts = {}
        for pred in predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Find consensus
        most_common_pred = max(prediction_counts, key=prediction_counts.get)
        consensus_strength = prediction_counts[most_common_pred] / len(predictions)
        
        return {
            "prediction_distribution": prediction_counts,
            "consensus_prediction": most_common_pred,
            "consensus_strength": consensus_strength,
            "has_strong_consensus": consensus_strength > 0.7,
            "disagreement_level": 1.0 - consensus_strength
        }
    
    def display_explanation(self, explanation: Dict[str, Any]) -> None:
        """Display comparative explanation"""
        if isinstance(explanation, list):
            comparison = self.compare_explanations(explanation)
            print("\n=== Comparative Analysis ===")
            print(json.dumps(comparison, indent=2, default=str))
        else:
            print("Comparative interface requires multiple explanations")


class InteractiveExplainer:
    """Main class for interactive AI explanations"""
    
    def __init__(self, config: InteractiveConfig = None):
        self.config = config or InteractiveConfig()
        self.interfaces = {
            InteractionMode.VISUAL_EXPLORATION: VisualExplorationInterface(config),
            InteractionMode.CONVERSATIONAL: ConversationalInterface(config),
            InteractionMode.COMPARATIVE: ComparativeInterface(config)
        }
        self.current_mode = self.config.default_mode
        self.session_data = []
        
    def explain_interactively(self, explanation_data: Dict[str, Any],
                            mode: InteractionMode = None) -> None:
        """Start interactive explanation session"""
        mode = mode or self.current_mode
        interface = self.interfaces[mode]
        
        print(f"\nStarting interactive explanation in {mode.value} mode...")
        interface.display_explanation(explanation_data)
        
        # Record session
        self.session_data.append({
            "timestamp": datetime.now().isoformat(),
            "mode": mode.value,
            "explanation_id": id(explanation_data)
        })
    
    def switch_mode(self, new_mode: InteractionMode) -> None:
        """Switch to a different interaction mode"""
        self.current_mode = new_mode
        print(f"Switched to {new_mode.value} mode")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about the interactive session"""
        if not self.session_data:
            return {"error": "No session data available"}
        
        mode_counts = {}
        for session in self.session_data:
            mode = session["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return {
            "total_sessions": len(self.session_data),
            "mode_usage": mode_counts,
            "most_used_mode": max(mode_counts, key=mode_counts.get),
            "session_duration": "N/A"  # Would calculate from timestamps
        }


# Example usage
if __name__ == "__main__":
    # Create interactive explainer
    config = InteractiveConfig(
        default_mode=InteractionMode.CONVERSATIONAL,
        explanation_level=ExplanationLevel.INTERMEDIATE
    )
    
    explainer = InteractiveExplainer(config)
    
    # Example explanation data
    explanation_data = {
        "prediction": {"class": "button", "confidence": 0.85},
        "confidence": 0.85,
        "feature_importance": [
            {"name": "color_contrast", "importance": 0.8},
            {"name": "rectangular_shape", "importance": 0.7},
            {"name": "text_content", "importance": 0.6}
        ],
        "attention_map": np.random.rand(224, 224),
        "image": np.random.rand(224, 224, 3),
        "alternative_predictions": [
            {"class": "text_field", "confidence": 0.12},
            {"class": "image", "confidence": 0.03}
        ]
    }
    
    # Start interactive explanation
    explainer.explain_interactively(explanation_data, InteractionMode.CONVERSATIONAL)