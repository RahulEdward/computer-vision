"""
Confidence Estimator for Computer Genie
विश्वास अनुमानकर्ता - Computer Genie के लिए

Provides confidence scoring and uncertainty quantification for AI predictions
to help users understand when the AI is certain vs uncertain about decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from sklearn.calibration import calibration_curve
import warnings


class UncertaintyType(Enum):
    """Types of uncertainty that can be measured"""
    ALEATORIC = "aleatoric"  # Data uncertainty
    EPISTEMIC = "epistemic"  # Model uncertainty
    TOTAL = "total"  # Combined uncertainty


class ConfidenceMethod(Enum):
    """Methods for estimating confidence"""
    SOFTMAX_ENTROPY = "softmax_entropy"
    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    TEMPERATURE_SCALING = "temperature_scaling"
    DEEP_ENSEMBLES = "deep_ensembles"
    BAYESIAN_NEURAL_NETWORK = "bayesian_neural_network"


@dataclass
class ConfidenceConfig:
    """Configuration for confidence estimation"""
    primary_method: ConfidenceMethod = ConfidenceMethod.SOFTMAX_ENTROPY
    secondary_methods: List[ConfidenceMethod] = None
    monte_carlo_samples: int = 100
    temperature_scaling_enabled: bool = True
    calibration_enabled: bool = True
    uncertainty_threshold_low: float = 0.3
    uncertainty_threshold_high: float = 0.7
    ensemble_size: int = 5
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.secondary_methods is None:
            self.secondary_methods = [
                ConfidenceMethod.MONTE_CARLO_DROPOUT,
                ConfidenceMethod.TEMPERATURE_SCALING
            ]


class UncertaintyQuantifier:
    """Quantifies different types of uncertainty in model predictions"""
    
    def __init__(self, config: ConfidenceConfig = None):
        self.config = config or ConfidenceConfig()
        self.calibration_data = []
        self.temperature_parameter = nn.Parameter(torch.ones(1))
        
    def estimate_aleatoric_uncertainty(self, model_outputs: torch.Tensor,
                                     targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Estimate aleatoric (data) uncertainty"""
        if model_outputs.dim() == 1:
            model_outputs = model_outputs.unsqueeze(0)
        
        # For classification, use entropy of predicted probabilities
        if model_outputs.shape[-1] > 1:
            probs = F.softmax(model_outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # Normalize by log of number of classes
            normalized_entropy = entropy / np.log(model_outputs.shape[-1])
            return normalized_entropy
        else:
            # For regression, estimate from prediction variance if available
            return torch.zeros_like(model_outputs.squeeze())
    
    def estimate_epistemic_uncertainty(self, model: nn.Module,
                                     input_data: torch.Tensor,
                                     num_samples: int = None) -> torch.Tensor:
        """Estimate epistemic (model) uncertainty using Monte Carlo Dropout"""
        num_samples = num_samples or self.config.monte_carlo_samples
        
        # Enable dropout for uncertainty estimation
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(input_data)
                predictions.append(pred)
        
        # Return to eval mode
        model.eval()
        
        # Calculate variance across predictions
        predictions_tensor = torch.stack(predictions)
        epistemic_uncertainty = torch.var(predictions_tensor, dim=0)
        
        return epistemic_uncertainty
    
    def estimate_total_uncertainty(self, model: nn.Module,
                                 input_data: torch.Tensor,
                                 num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Estimate total uncertainty (aleatoric + epistemic)"""
        num_samples = num_samples or self.config.monte_carlo_samples
        
        # Get multiple predictions with dropout
        model.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(input_data)
                predictions.append(pred)
        
        model.eval()
        
        predictions_tensor = torch.stack(predictions)
        
        # Calculate different uncertainty measures
        mean_prediction = torch.mean(predictions_tensor, dim=0)
        epistemic = torch.var(predictions_tensor, dim=0)
        
        # Aleatoric uncertainty from mean prediction
        aleatoric = self.estimate_aleatoric_uncertainty(mean_prediction)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
            "mean_prediction": mean_prediction,
            "prediction_std": torch.std(predictions_tensor, dim=0)
        }


class ConfidenceEstimator:
    """Main class for estimating prediction confidence"""
    
    def __init__(self, config: ConfidenceConfig = None):
        self.config = config or ConfidenceConfig()
        self.uncertainty_quantifier = UncertaintyQuantifier(config)
        self.calibration_history = []
        self.confidence_history = []
        
    def estimate_confidence(self, model: nn.Module,
                          input_data: torch.Tensor,
                          prediction: torch.Tensor,
                          method: ConfidenceMethod = None) -> Dict[str, Any]:
        """Estimate confidence using specified method"""
        method = method or self.config.primary_method
        
        confidence_result = {
            "method": method.value,
            "timestamp": torch.tensor(np.datetime64('now').astype(float))
        }
        
        if method == ConfidenceMethod.SOFTMAX_ENTROPY:
            confidence_result.update(self._softmax_entropy_confidence(prediction))
        
        elif method == ConfidenceMethod.MONTE_CARLO_DROPOUT:
            confidence_result.update(self._monte_carlo_confidence(model, input_data))
        
        elif method == ConfidenceMethod.TEMPERATURE_SCALING:
            confidence_result.update(self._temperature_scaling_confidence(prediction))
        
        elif method == ConfidenceMethod.ENSEMBLE_VARIANCE:
            confidence_result.update(self._ensemble_confidence(model, input_data))
        
        else:
            # Fallback to softmax entropy
            confidence_result.update(self._softmax_entropy_confidence(prediction))
        
        # Add uncertainty quantification
        uncertainty_info = self.uncertainty_quantifier.estimate_total_uncertainty(
            model, input_data
        )
        confidence_result["uncertainty"] = uncertainty_info
        
        # Store in history
        self.confidence_history.append(confidence_result)
        
        return confidence_result
    
    def _softmax_entropy_confidence(self, prediction: torch.Tensor) -> Dict[str, Any]:
        """Calculate confidence based on softmax entropy"""
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(0)
        
        # Apply softmax to get probabilities
        probs = F.softmax(prediction, dim=-1)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log(prediction.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - normalized_entropy
        
        # Get top prediction and its probability
        top_prob, top_class = torch.max(probs, dim=-1)
        
        return {
            "confidence_score": float(confidence.mean()),
            "entropy": float(entropy.mean()),
            "normalized_entropy": float(normalized_entropy.mean()),
            "top_class_probability": float(top_prob.mean()),
            "predicted_class": int(top_class.item()) if top_class.numel() == 1 else top_class.tolist(),
            "probability_distribution": probs.squeeze().tolist()
        }
    
    def _monte_carlo_confidence(self, model: nn.Module, 
                              input_data: torch.Tensor) -> Dict[str, Any]:
        """Calculate confidence using Monte Carlo Dropout"""
        uncertainty_info = self.uncertainty_quantifier.estimate_total_uncertainty(
            model, input_data, self.config.monte_carlo_samples
        )
        
        # Calculate confidence from uncertainty
        epistemic_uncertainty = uncertainty_info["epistemic"].mean()
        aleatoric_uncertainty = uncertainty_info["aleatoric"].mean()
        total_uncertainty = uncertainty_info["total"].mean()
        
        # Confidence is inverse of uncertainty
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        return {
            "confidence_score": float(confidence),
            "epistemic_uncertainty": float(epistemic_uncertainty),
            "aleatoric_uncertainty": float(aleatoric_uncertainty),
            "total_uncertainty": float(total_uncertainty),
            "prediction_variance": float(uncertainty_info["prediction_std"].mean()),
            "num_mc_samples": self.config.monte_carlo_samples
        }
    
    def _temperature_scaling_confidence(self, prediction: torch.Tensor) -> Dict[str, Any]:
        """Calculate confidence using temperature scaling"""
        # Apply temperature scaling
        scaled_logits = prediction / self.temperature_parameter
        
        # Get calibrated probabilities
        calibrated_probs = F.softmax(scaled_logits, dim=-1)
        
        # Calculate confidence metrics
        entropy = -torch.sum(calibrated_probs * torch.log(calibrated_probs + 1e-8), dim=-1)
        max_entropy = np.log(prediction.shape[-1])
        confidence = 1.0 - (entropy / max_entropy)
        
        top_prob, top_class = torch.max(calibrated_probs, dim=-1)
        
        return {
            "confidence_score": float(confidence.mean()),
            "calibrated_probability": float(top_prob.mean()),
            "temperature": float(self.temperature_parameter.item()),
            "predicted_class": int(top_class.item()) if top_class.numel() == 1 else top_class.tolist(),
            "calibrated_distribution": calibrated_probs.squeeze().tolist()
        }
    
    def _ensemble_confidence(self, model: nn.Module, 
                           input_data: torch.Tensor) -> Dict[str, Any]:
        """Calculate confidence using ensemble variance (simplified)"""
        # This is a simplified version - in practice, you'd have multiple models
        predictions = []
        
        # Simulate ensemble by adding noise to model parameters
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = param.data.clone()
        
        for i in range(self.config.ensemble_size):
            # Add small noise to parameters
            for name, param in model.named_parameters():
                noise = torch.randn_like(param) * 0.01
                param.data = original_state[name] + noise
            
            with torch.no_grad():
                pred = model(input_data)
                predictions.append(pred)
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_state[name]
        
        # Calculate ensemble statistics
        predictions_tensor = torch.stack(predictions)
        mean_pred = torch.mean(predictions_tensor, dim=0)
        variance = torch.var(predictions_tensor, dim=0)
        
        # Confidence from ensemble agreement
        confidence = 1.0 / (1.0 + variance.mean())
        
        return {
            "confidence_score": float(confidence),
            "ensemble_variance": float(variance.mean()),
            "ensemble_std": float(torch.std(predictions_tensor, dim=0).mean()),
            "ensemble_size": self.config.ensemble_size,
            "prediction_agreement": float(1.0 - variance.mean())
        }
    
    def calibrate_confidence(self, predictions: List[torch.Tensor],
                           true_labels: List[torch.Tensor],
                           confidences: List[float]) -> Dict[str, Any]:
        """Calibrate confidence scores using reliability diagrams"""
        if len(predictions) != len(true_labels) or len(predictions) != len(confidences):
            raise ValueError("Predictions, labels, and confidences must have same length")
        
        # Convert to numpy arrays
        all_confidences = np.array(confidences)
        all_correct = []
        
        for pred, label in zip(predictions, true_labels):
            if pred.dim() > 1:
                pred_class = torch.argmax(pred, dim=-1)
            else:
                pred_class = pred
            
            correct = (pred_class == label).float().mean().item()
            all_correct.append(correct)
        
        all_correct = np.array(all_correct)
        
        # Calculate calibration metrics
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                all_correct, all_confidences, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            calibration_result = {
                "expected_calibration_error": float(ece),
                "maximum_calibration_error": float(mce),
                "reliability_diagram": {
                    "bin_boundaries": mean_predicted_value.tolist(),
                    "bin_accuracies": fraction_of_positives.tolist()
                },
                "is_well_calibrated": ece < 0.1,  # Threshold for good calibration
                "calibration_quality": "good" if ece < 0.05 else "fair" if ece < 0.15 else "poor"
            }
            
        except Exception as e:
            calibration_result = {
                "error": f"Calibration calculation failed: {str(e)}",
                "expected_calibration_error": float('inf'),
                "is_well_calibrated": False
            }
        
        # Store calibration data
        self.calibration_history.append(calibration_result)
        
        return calibration_result
    
    def get_confidence_interpretation(self, confidence_score: float) -> Dict[str, Any]:
        """Interpret confidence score for user understanding"""
        if confidence_score >= self.config.uncertainty_threshold_high:
            level = "high"
            description = "The AI is very confident in this prediction"
            recommendation = "You can trust this prediction with high confidence"
            color_code = "green"
        elif confidence_score >= self.config.uncertainty_threshold_low:
            level = "medium"
            description = "The AI has moderate confidence in this prediction"
            recommendation = "Consider verifying this prediction if it's critical"
            color_code = "yellow"
        else:
            level = "low"
            description = "The AI has low confidence in this prediction"
            recommendation = "This prediction should be verified before acting on it"
            color_code = "red"
        
        return {
            "confidence_level": level,
            "description": description,
            "recommendation": recommendation,
            "color_code": color_code,
            "numerical_score": confidence_score,
            "uncertainty_score": 1.0 - confidence_score
        }
    
    def analyze_confidence_trends(self) -> Dict[str, Any]:
        """Analyze trends in confidence over time"""
        if not self.confidence_history:
            return {"error": "No confidence history available"}
        
        confidence_scores = [
            entry.get("confidence_score", 0.0) 
            for entry in self.confidence_history
        ]
        
        # Calculate trend statistics
        recent_scores = confidence_scores[-10:] if len(confidence_scores) >= 10 else confidence_scores
        
        trend_analysis = {
            "total_predictions": len(confidence_scores),
            "average_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "recent_average": np.mean(recent_scores),
            "confidence_trend": "improving" if len(recent_scores) > 1 and 
                              np.mean(recent_scores) > np.mean(confidence_scores[:-10]) 
                              else "stable",
            "low_confidence_rate": np.mean([s < self.config.uncertainty_threshold_low 
                                          for s in confidence_scores]),
            "high_confidence_rate": np.mean([s > self.config.uncertainty_threshold_high 
                                           for s in confidence_scores])
        }
        
        # Method usage statistics
        method_counts = {}
        for entry in self.confidence_history:
            method = entry.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        trend_analysis["method_usage"] = method_counts
        
        return trend_analysis
    
    def optimize_temperature_parameter(self, validation_predictions: List[torch.Tensor],
                                     validation_labels: List[torch.Tensor],
                                     learning_rate: float = 0.01,
                                     max_iterations: int = 100) -> float:
        """Optimize temperature parameter for better calibration"""
        optimizer = torch.optim.LBFGS([self.temperature_parameter], lr=learning_rate)
        
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            
            for pred, label in zip(validation_predictions, validation_labels):
                scaled_logits = pred / self.temperature_parameter
                loss = F.cross_entropy(scaled_logits, label)
                total_loss += loss
            
            total_loss.backward()
            return total_loss
        
        for _ in range(max_iterations):
            optimizer.step(closure)
        
        return float(self.temperature_parameter.item())


# Example usage and testing
if __name__ == "__main__":
    # Create confidence estimator
    config = ConfidenceConfig(
        primary_method=ConfidenceMethod.MONTE_CARLO_DROPOUT,
        monte_carlo_samples=50,
        uncertainty_threshold_low=0.3,
        uncertainty_threshold_high=0.7
    )
    
    estimator = ConfidenceEstimator(config)
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            return self.fc(self.dropout(x))
    
    model = DummyModel()
    input_data = torch.randn(1, 10)
    prediction = model(input_data)
    
    # Estimate confidence
    confidence_result = estimator.estimate_confidence(model, input_data, prediction)
    print("Confidence Result:")
    print(json.dumps(confidence_result, indent=2, default=str))
    
    # Get confidence interpretation
    interpretation = estimator.get_confidence_interpretation(
        confidence_result["confidence_score"]
    )
    print("\nConfidence Interpretation:")
    print(json.dumps(interpretation, indent=2))