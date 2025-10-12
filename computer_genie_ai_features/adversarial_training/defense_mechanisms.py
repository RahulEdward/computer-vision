"""
Defense Mechanisms for Adversarial Robustness
विरोधी मजबूती के लिए रक्षा तंत्र

Implements various defense mechanisms to protect models against adversarial attacks
and improve overall robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from abc import ABC, abstractmethod


class DefenseType(Enum):
    """Types of defense mechanisms"""
    ADVERSARIAL_TRAINING = "adversarial_training"
    DEFENSIVE_DISTILLATION = "defensive_distillation"
    INPUT_PREPROCESSING = "input_preprocessing"
    DETECTION_BASED = "detection_based"
    RANDOMIZED_SMOOTHING = "randomized_smoothing"
    CERTIFIED_DEFENSE = "certified_defense"
    ENSEMBLE_DEFENSE = "ensemble_defense"
    FEATURE_SQUEEZING = "feature_squeezing"
    GRADIENT_MASKING = "gradient_masking"
    ADVERSARIAL_PURIFICATION = "adversarial_purification"


class PreprocessingType(Enum):
    """Input preprocessing methods"""
    GAUSSIAN_NOISE = "gaussian_noise"
    MEDIAN_FILTER = "median_filter"
    BILATERAL_FILTER = "bilateral_filter"
    JPEG_COMPRESSION = "jpeg_compression"
    BIT_DEPTH_REDUCTION = "bit_depth_reduction"
    TOTAL_VARIANCE_MINIMIZATION = "total_variance_minimization"
    PIXEL_DEFLECTION = "pixel_deflection"
    THERMOMETER_ENCODING = "thermometer_encoding"


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms"""
    defense_type: DefenseType = DefenseType.INPUT_PREPROCESSING
    
    # Input preprocessing parameters
    preprocessing_methods: List[PreprocessingType] = field(default_factory=list)
    noise_std: float = 0.1
    filter_size: int = 3
    jpeg_quality: int = 75
    bit_depth: int = 4
    
    # Defensive distillation parameters
    distillation_temperature: float = 20.0
    teacher_model_path: Optional[str] = None
    
    # Detection parameters
    detection_threshold: float = 0.5
    use_statistical_tests: bool = True
    
    # Randomized smoothing parameters
    smoothing_noise_std: float = 0.25
    smoothing_samples: int = 1000
    smoothing_alpha: float = 0.001
    
    # Ensemble parameters
    ensemble_size: int = 5
    ensemble_diversity_weight: float = 0.1
    
    # Feature squeezing parameters
    squeeze_bit_depth: int = 5
    squeeze_filter_size: int = 2
    
    # Certified defense parameters
    certification_radius: float = 0.5
    certification_confidence: float = 0.95
    
    # General parameters
    device: str = "cuda"
    batch_size: int = 32


class DefenseMechanism(ABC):
    """Abstract base class for defense mechanisms"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply defense mechanism to inputs"""
        pass
    
    @abstractmethod
    def get_defense_info(self) -> Dict[str, Any]:
        """Get information about the defense mechanism"""
        pass


class InputPreprocessingDefense(DefenseMechanism):
    """Input preprocessing defense mechanisms"""
    
    def __init__(self, config: DefenseConfig):
        super().__init__(config)
        self.preprocessing_methods = config.preprocessing_methods
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply preprocessing defenses to inputs"""
        defended_inputs = inputs.clone()
        
        for method in self.preprocessing_methods:
            if method == PreprocessingType.GAUSSIAN_NOISE:
                defended_inputs = self._add_gaussian_noise(defended_inputs)
            elif method == PreprocessingType.MEDIAN_FILTER:
                defended_inputs = self._apply_median_filter(defended_inputs)
            elif method == PreprocessingType.BILATERAL_FILTER:
                defended_inputs = self._apply_bilateral_filter(defended_inputs)
            elif method == PreprocessingType.JPEG_COMPRESSION:
                defended_inputs = self._apply_jpeg_compression(defended_inputs)
            elif method == PreprocessingType.BIT_DEPTH_REDUCTION:
                defended_inputs = self._reduce_bit_depth(defended_inputs)
            elif method == PreprocessingType.TOTAL_VARIANCE_MINIMIZATION:
                defended_inputs = self._apply_tv_minimization(defended_inputs)
            elif method == PreprocessingType.PIXEL_DEFLECTION:
                defended_inputs = self._apply_pixel_deflection(defended_inputs)
            elif method == PreprocessingType.THERMOMETER_ENCODING:
                defended_inputs = self._apply_thermometer_encoding(defended_inputs)
        
        return defended_inputs
    
    def _add_gaussian_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to inputs"""
        noise = torch.randn_like(inputs) * self.config.noise_std
        return torch.clamp(inputs + noise, 0, 1)
    
    def _apply_median_filter(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply median filter to inputs"""
        # Simplified median filter implementation
        kernel_size = self.config.filter_size
        padding = kernel_size // 2
        
        # Apply median filter channel-wise
        filtered = torch.zeros_like(inputs)
        for b in range(inputs.size(0)):
            for c in range(inputs.size(1)):
                channel = inputs[b, c]
                # Pad the channel
                padded = F.pad(channel.unsqueeze(0).unsqueeze(0), 
                              (padding, padding, padding, padding), mode='reflect')
                
                # Extract patches and compute median
                patches = F.unfold(padded, kernel_size, stride=1)
                medians = patches.median(dim=1)[0]
                filtered[b, c] = medians.view(channel.shape)
        
        return filtered
    
    def _apply_bilateral_filter(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filter (simplified implementation)"""
        # Simplified bilateral filter using Gaussian blur
        sigma_spatial = 1.0
        sigma_color = 0.1
        
        # Apply Gaussian blur as approximation
        kernel_size = self.config.filter_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma_spatial ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable convolution
        kernel_x = kernel_1d.view(1, 1, 1, -1).to(inputs.device)
        kernel_y = kernel_1d.view(1, 1, -1, 1).to(inputs.device)
        
        filtered = inputs
        for c in range(inputs.size(1)):
            channel = inputs[:, c:c+1]
            channel = F.conv2d(channel, kernel_x, padding=(0, kernel_size//2))
            channel = F.conv2d(channel, kernel_y, padding=(kernel_size//2, 0))
            filtered[:, c:c+1] = channel
        
        return filtered
    
    def _apply_jpeg_compression(self, inputs: torch.Tensor) -> torch.Tensor:
        """Simulate JPEG compression (simplified)"""
        # Quantization-based approximation of JPEG compression
        quality = self.config.jpeg_quality
        quantization_factor = (100 - quality) / 100.0 * 0.5
        
        # Apply quantization
        quantized = torch.round(inputs / quantization_factor) * quantization_factor
        return torch.clamp(quantized, 0, 1)
    
    def _reduce_bit_depth(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reduce bit depth of inputs"""
        bit_depth = self.config.bit_depth
        levels = 2 ** bit_depth
        
        # Quantize to reduced bit depth
        quantized = torch.round(inputs * (levels - 1)) / (levels - 1)
        return quantized
    
    def _apply_tv_minimization(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply total variation minimization (simplified)"""
        # Simplified TV denoising using gradient penalty
        lambda_tv = 0.1
        
        # Compute total variation
        diff_x = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]
        diff_y = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
        
        tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
        
        # Apply simple smoothing based on TV
        smoothed = inputs - lambda_tv * torch.sign(inputs - 0.5) * tv_loss
        return torch.clamp(smoothed, 0, 1)
    
    def _apply_pixel_deflection(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply pixel deflection defense"""
        deflection_rate = 0.1  # Percentage of pixels to deflect
        window_size = 10
        
        defended = inputs.clone()
        batch_size, channels, height, width = inputs.shape
        
        for b in range(batch_size):
            for c in range(channels):
                # Randomly select pixels to deflect
                num_deflect = int(height * width * deflection_rate)
                deflect_indices = torch.randperm(height * width)[:num_deflect]
                
                for idx in deflect_indices:
                    y, x = idx // width, idx % width
                    
                    # Define window around pixel
                    y_min = max(0, y - window_size // 2)
                    y_max = min(height, y + window_size // 2 + 1)
                    x_min = max(0, x - window_size // 2)
                    x_max = min(width, x + window_size // 2 + 1)
                    
                    # Replace with random pixel from window
                    window = inputs[b, c, y_min:y_max, x_min:x_max]
                    if window.numel() > 0:
                        random_pixel = window.flatten()[torch.randint(0, window.numel(), (1,))]
                        defended[b, c, y, x] = random_pixel
        
        return defended
    
    def _apply_thermometer_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply thermometer encoding"""
        levels = 16  # Number of thermometer levels
        
        # Convert to thermometer encoding and back
        scaled = inputs * (levels - 1)
        indices = torch.floor(scaled).long()
        
        # Create thermometer representation
        thermometer = torch.zeros(*inputs.shape, levels, device=inputs.device)
        for i in range(levels):
            thermometer[..., i] = (indices >= i).float()
        
        # Convert back to regular representation
        decoded = thermometer.sum(dim=-1) / levels
        return decoded
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get preprocessing defense information"""
        return {
            'defense_type': 'input_preprocessing',
            'methods': [method.value for method in self.preprocessing_methods],
            'noise_std': self.config.noise_std,
            'filter_size': self.config.filter_size,
            'jpeg_quality': self.config.jpeg_quality,
            'bit_depth': self.config.bit_depth
        }


class DefensiveDistillationDefense(DefenseMechanism):
    """Defensive distillation defense mechanism"""
    
    def __init__(self, config: DefenseConfig, teacher_model: nn.Module):
        super().__init__(config)
        self.teacher_model = teacher_model
        self.temperature = config.distillation_temperature
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply defensive distillation (returns original inputs as distillation is applied during training)"""
        return inputs
    
    def distill_model(self, student_model: nn.Module, train_loader, num_epochs: int = 10) -> nn.Module:
        """Train student model using defensive distillation"""
        self.teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get teacher predictions with temperature
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs) / self.temperature
                    teacher_probs = F.softmax(teacher_outputs, dim=1)
                
                # Get student predictions with temperature
                student_outputs = student_model(inputs) / self.temperature
                student_log_probs = F.log_softmax(student_outputs, dim=1)
                
                # Distillation loss
                distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                # Standard cross-entropy loss
                standard_outputs = student_model(inputs)
                standard_loss = F.cross_entropy(standard_outputs, targets)
                
                # Combined loss
                total_loss = 0.7 * distillation_loss + 0.3 * standard_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        return student_model
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get distillation defense information"""
        return {
            'defense_type': 'defensive_distillation',
            'temperature': self.temperature,
            'teacher_model': str(type(self.teacher_model).__name__)
        }


class DetectionBasedDefense(DefenseMechanism):
    """Detection-based defense mechanism"""
    
    def __init__(self, config: DefenseConfig):
        super().__init__(config)
        self.threshold = config.detection_threshold
        self.use_statistical_tests = config.use_statistical_tests
        
        # Initialize detection statistics
        self.clean_stats = {'mean': None, 'std': None}
        self.is_calibrated = False
    
    def calibrate(self, clean_inputs: torch.Tensor, model: nn.Module) -> None:
        """Calibrate detector on clean inputs"""
        model.eval()
        
        with torch.no_grad():
            # Extract features for calibration
            features = self._extract_features(clean_inputs, model)
            
            # Compute statistics
            self.clean_stats['mean'] = features.mean(dim=0)
            self.clean_stats['std'] = features.std(dim=0)
            
        self.is_calibrated = True
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Detect and filter adversarial examples"""
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before use")
        
        # Detect adversarial examples
        is_adversarial = self.detect_adversarial(inputs, model)
        
        # Filter out detected adversarial examples (replace with noise or reject)
        defended_inputs = inputs.clone()
        adversarial_mask = is_adversarial.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        # Replace adversarial examples with Gaussian noise
        noise = torch.randn_like(inputs) * 0.1
        defended_inputs = torch.where(adversarial_mask, noise, defended_inputs)
        
        return defended_inputs
    
    def detect_adversarial(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Detect adversarial examples"""
        model.eval()
        
        with torch.no_grad():
            # Extract features
            features = self._extract_features(inputs, model)
            
            # Compute detection scores
            if self.use_statistical_tests:
                scores = self._statistical_detection(features)
            else:
                scores = self._distance_based_detection(features)
            
            # Apply threshold
            is_adversarial = scores > self.threshold
            
        return is_adversarial
    
    def _extract_features(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Extract features for detection"""
        # Use model's intermediate representations
        features = []
        
        def hook_fn(module, input, output):
            features.append(output.view(output.size(0), -1))
        
        # Register hooks on intermediate layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate features
        if features:
            return torch.cat(features, dim=1)
        else:
            return inputs.view(inputs.size(0), -1)
    
    def _statistical_detection(self, features: torch.Tensor) -> torch.Tensor:
        """Statistical-based detection"""
        # Compute Mahalanobis distance
        diff = features - self.clean_stats['mean']
        inv_cov = torch.diag(1.0 / (self.clean_stats['std'] + 1e-8))
        
        mahalanobis_dist = torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
        return mahalanobis_dist
    
    def _distance_based_detection(self, features: torch.Tensor) -> torch.Tensor:
        """Distance-based detection"""
        # Compute L2 distance from clean mean
        diff = features - self.clean_stats['mean']
        l2_dist = torch.norm(diff, dim=1)
        return l2_dist
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get detection defense information"""
        return {
            'defense_type': 'detection_based',
            'threshold': self.threshold,
            'use_statistical_tests': self.use_statistical_tests,
            'is_calibrated': self.is_calibrated
        }


class RandomizedSmoothingDefense(DefenseMechanism):
    """Randomized smoothing defense mechanism"""
    
    def __init__(self, config: DefenseConfig):
        super().__init__(config)
        self.noise_std = config.smoothing_noise_std
        self.num_samples = config.smoothing_samples
        self.alpha = config.smoothing_alpha
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply randomized smoothing (returns original inputs as smoothing is applied during inference)"""
        return inputs
    
    def smooth_predict(self, inputs: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make smoothed predictions with confidence"""
        model.eval()
        batch_size, num_classes = inputs.size(0), self._get_num_classes(model, inputs)
        
        # Sample predictions
        vote_counts = torch.zeros(batch_size, num_classes, device=self.device)
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Add Gaussian noise
                noisy_inputs = inputs + torch.randn_like(inputs) * self.noise_std
                noisy_inputs = torch.clamp(noisy_inputs, 0, 1)
                
                # Get predictions
                outputs = model(noisy_inputs)
                predictions = outputs.argmax(dim=1)
                
                # Count votes
                for i in range(batch_size):
                    vote_counts[i, predictions[i]] += 1
        
        # Get top predictions and confidence
        top_votes, top_classes = vote_counts.max(dim=1)
        confidence = top_votes / self.num_samples
        
        return top_classes, confidence
    
    def certify_robustness(self, inputs: torch.Tensor, model: nn.Module, 
                          radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Certify robustness using randomized smoothing"""
        predictions, confidence = self.smooth_predict(inputs, model)
        
        # Compute certification radius
        from scipy.stats import norm
        certified_radius = self.noise_std * norm.ppf(confidence.cpu().numpy())
        certified_radius = torch.tensor(certified_radius, device=self.device)
        
        # Check if certified radius is greater than attack radius
        is_certified = certified_radius >= radius
        
        return predictions, is_certified
    
    def _get_num_classes(self, model: nn.Module, inputs: torch.Tensor) -> int:
        """Get number of classes from model output"""
        with torch.no_grad():
            sample_output = model(inputs[:1])
            return sample_output.size(1)
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get randomized smoothing defense information"""
        return {
            'defense_type': 'randomized_smoothing',
            'noise_std': self.noise_std,
            'num_samples': self.num_samples,
            'alpha': self.alpha
        }


class EnsembleDefense(DefenseMechanism):
    """Ensemble-based defense mechanism"""
    
    def __init__(self, config: DefenseConfig, models: List[nn.Module]):
        super().__init__(config)
        self.models = models
        self.ensemble_size = len(models)
        self.diversity_weight = config.ensemble_diversity_weight
    
    def defend(self, inputs: torch.Tensor, model: nn.Module = None) -> torch.Tensor:
        """Apply ensemble defense (returns original inputs as ensemble is used during inference)"""
        return inputs
    
    def ensemble_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                predictions.append(F.softmax(outputs, dim=1))
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    
    def diverse_ensemble_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make diversity-aware ensemble predictions"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                predictions.append(F.softmax(outputs, dim=1))
        
        # Compute diversity weights
        diversity_weights = self._compute_diversity_weights(predictions)
        
        # Weighted average
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weighted_predictions.append(pred * diversity_weights[i])
        
        ensemble_pred = torch.stack(weighted_predictions).sum(dim=0)
        return ensemble_pred
    
    def _compute_diversity_weights(self, predictions: List[torch.Tensor]) -> List[float]:
        """Compute diversity-based weights for ensemble members"""
        num_models = len(predictions)
        diversity_matrix = torch.zeros(num_models, num_models)
        
        # Compute pairwise diversity (KL divergence)
        for i in range(num_models):
            for j in range(num_models):
                if i != j:
                    kl_div = F.kl_div(
                        predictions[i].log(), predictions[j], 
                        reduction='batchmean'
                    )
                    diversity_matrix[i, j] = kl_div
        
        # Compute diversity scores
        diversity_scores = diversity_matrix.sum(dim=1)
        
        # Convert to weights (higher diversity = higher weight)
        weights = F.softmax(diversity_scores * self.diversity_weight, dim=0)
        return weights.tolist()
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get ensemble defense information"""
        return {
            'defense_type': 'ensemble_defense',
            'ensemble_size': self.ensemble_size,
            'diversity_weight': self.diversity_weight,
            'model_types': [type(model).__name__ for model in self.models]
        }


class FeatureSqueezingDefense(DefenseMechanism):
    """Feature squeezing defense mechanism"""
    
    def __init__(self, config: DefenseConfig):
        super().__init__(config)
        self.bit_depth = config.squeeze_bit_depth
        self.filter_size = config.squeeze_filter_size
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply feature squeezing"""
        # Apply bit depth reduction
        squeezed = self._reduce_bit_depth(inputs)
        
        # Apply spatial smoothing
        squeezed = self._apply_spatial_smoothing(squeezed)
        
        return squeezed
    
    def _reduce_bit_depth(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reduce bit depth of inputs"""
        levels = 2 ** self.bit_depth
        quantized = torch.round(inputs * (levels - 1)) / (levels - 1)
        return quantized
    
    def _apply_spatial_smoothing(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing filter"""
        # Simple average pooling followed by upsampling
        kernel_size = self.filter_size
        
        # Apply average pooling
        pooled = F.avg_pool2d(inputs, kernel_size, stride=1, padding=kernel_size//2)
        
        return pooled
    
    def detect_squeezing_difference(self, inputs: torch.Tensor, model: nn.Module, 
                                  threshold: float = 0.1) -> torch.Tensor:
        """Detect adversarial examples using squeezing difference"""
        model.eval()
        
        with torch.no_grad():
            # Original predictions
            original_outputs = model(inputs)
            original_probs = F.softmax(original_outputs, dim=1)
            
            # Squeezed predictions
            squeezed_inputs = self.defend(inputs, model)
            squeezed_outputs = model(squeezed_inputs)
            squeezed_probs = F.softmax(squeezed_outputs, dim=1)
            
            # Compute difference
            prob_diff = torch.norm(original_probs - squeezed_probs, dim=1)
            
            # Detect based on threshold
            is_adversarial = prob_diff > threshold
        
        return is_adversarial
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get feature squeezing defense information"""
        return {
            'defense_type': 'feature_squeezing',
            'bit_depth': self.bit_depth,
            'filter_size': self.filter_size
        }


class AdversarialPurificationDefense(DefenseMechanism):
    """Adversarial purification defense using generative models"""
    
    def __init__(self, config: DefenseConfig, purifier_model: nn.Module):
        super().__init__(config)
        self.purifier_model = purifier_model
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply adversarial purification"""
        self.purifier_model.eval()
        
        with torch.no_grad():
            # Purify inputs using generative model
            purified = self.purifier_model(inputs)
            
            # Ensure outputs are in valid range
            purified = torch.clamp(purified, 0, 1)
        
        return purified
    
    def iterative_purification(self, inputs: torch.Tensor, model: nn.Module, 
                             num_iterations: int = 3) -> torch.Tensor:
        """Apply iterative purification"""
        purified = inputs
        
        for _ in range(num_iterations):
            purified = self.defend(purified, model)
        
        return purified
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get purification defense information"""
        return {
            'defense_type': 'adversarial_purification',
            'purifier_model': type(self.purifier_model).__name__
        }


class DefenseEnsemble:
    """Combine multiple defense mechanisms"""
    
    def __init__(self, defenses: List[DefenseMechanism], weights: Optional[List[float]] = None):
        self.defenses = defenses
        self.weights = weights or [1.0 / len(defenses)] * len(defenses)
        
        if len(self.weights) != len(self.defenses):
            raise ValueError("Number of weights must match number of defenses")
    
    def defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply ensemble of defenses"""
        defended_inputs = torch.zeros_like(inputs)
        
        for defense, weight in zip(self.defenses, self.weights):
            defense_output = defense.defend(inputs, model)
            defended_inputs += weight * defense_output
        
        return torch.clamp(defended_inputs, 0, 1)
    
    def adaptive_defend(self, inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Apply adaptive ensemble defense based on input characteristics"""
        # Analyze input characteristics
        input_variance = torch.var(inputs, dim=(2, 3)).mean()
        input_entropy = self._compute_entropy(inputs)
        
        # Adjust weights based on characteristics
        adaptive_weights = self._compute_adaptive_weights(input_variance, input_entropy)
        
        # Apply weighted ensemble
        defended_inputs = torch.zeros_like(inputs)
        for defense, weight in zip(self.defenses, adaptive_weights):
            defense_output = defense.defend(inputs, model)
            defended_inputs += weight * defense_output
        
        return torch.clamp(defended_inputs, 0, 1)
    
    def _compute_entropy(self, inputs: torch.Tensor) -> float:
        """Compute entropy of inputs"""
        # Simplified entropy computation
        hist = torch.histc(inputs.flatten(), bins=256, min=0, max=1)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -torch.sum(hist * torch.log2(hist))
        return entropy.item()
    
    def _compute_adaptive_weights(self, variance: float, entropy: float) -> List[float]:
        """Compute adaptive weights based on input characteristics"""
        # Simple heuristic: high variance/entropy -> more preprocessing
        # Low variance/entropy -> more detection-based defenses
        
        base_weights = self.weights.copy()
        
        # Adjust based on variance and entropy
        for i, defense in enumerate(self.defenses):
            if isinstance(defense, InputPreprocessingDefense):
                base_weights[i] *= (1 + variance + entropy / 10)
            elif isinstance(defense, DetectionBasedDefense):
                base_weights[i] *= (2 - variance - entropy / 10)
        
        # Normalize weights
        total_weight = sum(base_weights)
        return [w / total_weight for w in base_weights]
    
    def get_defense_info(self) -> Dict[str, Any]:
        """Get ensemble defense information"""
        return {
            'defense_type': 'ensemble',
            'num_defenses': len(self.defenses),
            'defense_types': [defense.get_defense_info()['defense_type'] for defense in self.defenses],
            'weights': self.weights
        }


# Example usage
if __name__ == "__main__":
    # Create defense configuration
    config = DefenseConfig(
        defense_type=DefenseType.INPUT_PREPROCESSING,
        preprocessing_methods=[
            PreprocessingType.GAUSSIAN_NOISE,
            PreprocessingType.MEDIAN_FILTER,
            PreprocessingType.BIT_DEPTH_REDUCTION
        ],
        noise_std=0.1,
        filter_size=3,
        bit_depth=5
    )
    
    # Create defense mechanism
    defense = InputPreprocessingDefense(config)
    
    print("Defense mechanisms created successfully!")
    print(f"Defense info: {defense.get_defense_info()}")