"""
Distillation Loss Functions
आसवन हानि फ़ंक्शन

Comprehensive collection of specialized loss functions for knowledge distillation
including temperature-based, attention-based, and structured losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math
from abc import ABC, abstractmethod


class LossType(Enum):
    """Types of distillation losses"""
    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    COSINE = "cosine"
    CORRELATION = "correlation"
    ATTENTION = "attention"
    FEATURE_MAP = "feature_map"
    GRAM_MATRIX = "gram_matrix"
    JACOBIAN = "jacobian"
    MUTUAL_INFORMATION = "mutual_information"
    WASSERSTEIN = "wasserstein"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies"""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    ADAPTIVE = "adaptive"
    CURRICULUM = "curriculum"


@dataclass
class LossConfig:
    """Configuration for distillation losses"""
    # Basic parameters
    temperature: float = 4.0
    alpha: float = 0.7                   # Distillation loss weight
    beta: float = 0.3                    # Task loss weight
    
    # Temperature scheduling
    temperature_schedule: TemperatureSchedule = TemperatureSchedule.CONSTANT
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    
    # Loss-specific parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Adaptive parameters
    adaptive_threshold: float = 0.1
    adaptation_rate: float = 0.01
    
    # Device
    device: str = "cuda"


class DistillationLoss(ABC):
    """Abstract base class for distillation losses"""
    
    def __init__(self, config: LossConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute distillation loss"""
        pass
    
    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Get temperature based on schedule"""
        if self.config.temperature_schedule == TemperatureSchedule.CONSTANT:
            return self.config.temperature
        
        elif self.config.temperature_schedule == TemperatureSchedule.LINEAR_DECAY:
            progress = epoch / total_epochs
            return self.config.max_temperature - progress * (self.config.max_temperature - self.config.min_temperature)
        
        elif self.config.temperature_schedule == TemperatureSchedule.EXPONENTIAL_DECAY:
            decay_rate = math.log(self.config.min_temperature / self.config.max_temperature) / total_epochs
            return self.config.max_temperature * math.exp(decay_rate * epoch)
        
        elif self.config.temperature_schedule == TemperatureSchedule.COSINE_ANNEALING:
            progress = epoch / total_epochs
            return self.config.min_temperature + 0.5 * (self.config.max_temperature - self.config.min_temperature) * (1 + math.cos(math.pi * progress))
        
        else:
            return self.config.temperature


class KLDivergenceLoss(DistillationLoss):
    """KL Divergence loss for knowledge distillation"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, temperature: Optional[float] = None,
                    **kwargs) -> torch.Tensor:
        """Compute KL divergence loss"""
        if temperature is None:
            temperature = self.config.temperature
        
        # Soften predictions
        student_soft = F.log_softmax(student_outputs / temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared
        return kl_loss * (temperature ** 2)


class MSELoss(DistillationLoss):
    """MSE loss for feature distillation"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute MSE loss"""
        return F.mse_loss(student_outputs, teacher_outputs)


class CosineSimilarityLoss(DistillationLoss):
    """Cosine similarity loss"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute cosine similarity loss"""
        # Flatten outputs
        student_flat = student_outputs.view(student_outputs.size(0), -1)
        teacher_flat = teacher_outputs.view(teacher_outputs.size(0), -1)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)
        
        # Return 1 - cosine similarity as loss
        return 1 - cosine_sim.mean()


class CorrelationLoss(DistillationLoss):
    """Correlation-based loss"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute correlation loss"""
        # Flatten outputs
        student_flat = student_outputs.view(student_outputs.size(0), -1)
        teacher_flat = teacher_outputs.view(teacher_outputs.size(0), -1)
        
        # Compute correlation matrix
        student_corr = torch.corrcoef(student_flat.T)
        teacher_corr = torch.corrcoef(teacher_flat.T)
        
        # Handle NaN values
        student_corr = torch.nan_to_num(student_corr, 0.0)
        teacher_corr = torch.nan_to_num(teacher_corr, 0.0)
        
        # Compute MSE between correlation matrices
        return F.mse_loss(student_corr, teacher_corr)


class AttentionLoss(DistillationLoss):
    """Attention-based distillation loss"""
    
    def compute_attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Compute attention map from feature map"""
        if len(feature_map.shape) == 4:  # Conv features
            # Spatial attention
            attention = torch.sum(feature_map.pow(2), dim=1, keepdim=True)
        else:  # FC features
            # Channel attention
            attention = feature_map.pow(2)
        
        return attention
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute attention loss"""
        student_attention = self.compute_attention_map(student_outputs)
        teacher_attention = self.compute_attention_map(teacher_outputs)
        
        # Normalize attention maps
        student_norm = F.normalize(student_attention.view(student_attention.size(0), -1), p=2, dim=1)
        teacher_norm = F.normalize(teacher_attention.view(teacher_attention.size(0), -1), p=2, dim=1)
        
        return F.mse_loss(student_norm, teacher_norm)


class GramMatrixLoss(DistillationLoss):
    """Gram matrix-based loss"""
    
    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        if len(features.shape) == 4:  # Conv features
            batch_size, channels, height, width = features.size()
            features_reshaped = features.view(batch_size, channels, height * width)
            
            # Compute Gram matrix
            gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
            
            # Normalize
            gram = gram / (channels * height * width)
        else:  # FC features
            batch_size, features_dim = features.size()
            features_reshaped = features.view(batch_size, features_dim, 1)
            
            gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
            gram = gram / features_dim
        
        return gram
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute Gram matrix loss"""
        student_gram = self.gram_matrix(student_outputs)
        teacher_gram = self.gram_matrix(teacher_outputs)
        
        return F.mse_loss(student_gram, teacher_gram)


class FocalDistillationLoss(DistillationLoss):
    """Focal loss for distillation"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, temperature: Optional[float] = None,
                    **kwargs) -> torch.Tensor:
        """Compute focal distillation loss"""
        if temperature is None:
            temperature = self.config.temperature
        
        # Soften predictions
        student_soft = F.softmax(student_outputs / temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
        
        # Compute focal weights
        pt = torch.sum(student_soft * teacher_soft, dim=1)
        focal_weight = self.config.focal_alpha * (1 - pt) ** self.config.focal_gamma
        
        # Compute KL divergence
        student_log_soft = F.log_softmax(student_outputs / temperature, dim=1)
        kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
        
        # Apply focal weighting
        focal_loss = focal_weight * kl_loss
        
        return focal_loss.mean() * (temperature ** 2)


class MutualInformationLoss(DistillationLoss):
    """Mutual information-based loss"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute mutual information loss"""
        # Flatten outputs
        student_flat = student_outputs.view(student_outputs.size(0), -1)
        teacher_flat = teacher_outputs.view(teacher_outputs.size(0), -1)
        
        # Compute joint distribution (simplified)
        joint_prob = F.softmax(torch.mm(student_flat, teacher_flat.t()), dim=1)
        
        # Compute marginal distributions
        student_prob = F.softmax(torch.sum(student_flat, dim=1, keepdim=True), dim=0)
        teacher_prob = F.softmax(torch.sum(teacher_flat, dim=1, keepdim=True), dim=0)
        
        # Compute marginal product
        marginal_product = torch.mm(student_prob.t(), teacher_prob)
        
        # Compute KL divergence between joint and marginal product
        mi_loss = F.kl_div(torch.log(joint_prob + 1e-8), marginal_product + 1e-8, reduction='batchmean')
        
        # Return negative MI as loss (we want to maximize MI)
        return -mi_loss


class WassersteinLoss(DistillationLoss):
    """Wasserstein distance-based loss"""
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute Wasserstein loss (simplified)"""
        # Sort outputs to compute Wasserstein distance
        student_sorted, _ = torch.sort(student_outputs, dim=1)
        teacher_sorted, _ = torch.sort(teacher_outputs, dim=1)
        
        # Compute L2 distance between sorted distributions
        wasserstein_dist = torch.mean((student_sorted - teacher_sorted) ** 2)
        
        return wasserstein_dist


class AdaptiveLoss(DistillationLoss):
    """Adaptive loss that adjusts based on performance"""
    
    def __init__(self, config: LossConfig):
        super().__init__(config)
        self.performance_history = []
        self.current_weight = 1.0
    
    def update_weight(self, current_performance: float):
        """Update loss weight based on performance"""
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) > 10:
            # Remove old history
            self.performance_history = self.performance_history[-10:]
            
            # Compute performance trend
            recent_avg = np.mean(self.performance_history[-5:])
            older_avg = np.mean(self.performance_history[:5])
            
            if recent_avg > older_avg + self.config.adaptive_threshold:
                # Performance improving, increase weight
                self.current_weight = min(2.0, self.current_weight + self.config.adaptation_rate)
            elif recent_avg < older_avg - self.config.adaptive_threshold:
                # Performance degrading, decrease weight
                self.current_weight = max(0.1, self.current_weight - self.config.adaptation_rate)
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute adaptive loss"""
        # Use KL divergence as base loss
        kl_loss = KLDivergenceLoss(self.config)
        base_loss = kl_loss.compute_loss(student_outputs, teacher_outputs, targets, **kwargs)
        
        return self.current_weight * base_loss


class CombinedLoss(DistillationLoss):
    """Combined loss function with multiple components"""
    
    def __init__(self, config: LossConfig, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        
        # Initialize individual losses
        self.losses = {
            'kl': KLDivergenceLoss(config),
            'mse': MSELoss(config),
            'cosine': CosineSimilarityLoss(config),
            'attention': AttentionLoss(config),
            'gram': GramMatrixLoss(config)
        }
        
        # Default weights
        self.loss_weights = loss_weights or {
            'kl': 1.0,
            'mse': 0.1,
            'cosine': 0.1,
            'attention': 0.1,
            'gram': 0.1
        }
    
    def compute_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        individual_losses = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.losses.items():
            try:
                loss_value = loss_fn.compute_loss(student_outputs, teacher_outputs, targets, **kwargs)
                individual_losses[loss_name] = loss_value
                total_loss += self.loss_weights[loss_name] * loss_value
            except Exception as e:
                # Skip losses that fail
                individual_losses[loss_name] = torch.tensor(0.0, device=self.device)
        
        individual_losses['total'] = total_loss
        return individual_losses


class DistillationLossManager:
    """Manager for distillation losses"""
    
    def __init__(self, config: LossConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize loss functions
        self.loss_functions = {
            LossType.KL_DIVERGENCE: KLDivergenceLoss(config),
            LossType.MSE: MSELoss(config),
            LossType.COSINE: CosineSimilarityLoss(config),
            LossType.CORRELATION: CorrelationLoss(config),
            LossType.ATTENTION: AttentionLoss(config),
            LossType.GRAM_MATRIX: GramMatrixLoss(config),
            LossType.FOCAL: FocalDistillationLoss(config),
            LossType.MUTUAL_INFORMATION: MutualInformationLoss(config),
            LossType.WASSERSTEIN: WassersteinLoss(config)
        }
        
        # Loss history
        self.loss_history = {loss_type.value: [] for loss_type in LossType}
        
        # Current epoch for temperature scheduling
        self.current_epoch = 0
        self.total_epochs = 100
    
    def compute_loss(self, loss_type: LossType, student_outputs: torch.Tensor,
                    teacher_outputs: torch.Tensor, targets: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Compute specific loss type"""
        if loss_type not in self.loss_functions:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss_fn = self.loss_functions[loss_type]
        
        # Add temperature for temperature-based losses
        if loss_type in [LossType.KL_DIVERGENCE, LossType.FOCAL]:
            temperature = loss_fn.get_temperature(self.current_epoch, self.total_epochs)
            kwargs['temperature'] = temperature
        
        loss = loss_fn.compute_loss(student_outputs, teacher_outputs, targets, **kwargs)
        
        # Store in history
        self.loss_history[loss_type.value].append(loss.item())
        
        return loss
    
    def compute_combined_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                             targets: Optional[torch.Tensor] = None,
                             loss_weights: Optional[Dict[LossType, float]] = None) -> Dict[str, torch.Tensor]:
        """Compute combined loss with multiple components"""
        if loss_weights is None:
            loss_weights = {
                LossType.KL_DIVERGENCE: 1.0,
                LossType.MSE: 0.1,
                LossType.ATTENTION: 0.1
            }
        
        individual_losses = {}
        total_loss = 0.0
        
        for loss_type, weight in loss_weights.items():
            try:
                loss_value = self.compute_loss(loss_type, student_outputs, teacher_outputs, targets)
                individual_losses[loss_type.value] = loss_value
                total_loss += weight * loss_value
            except Exception as e:
                individual_losses[loss_type.value] = torch.tensor(0.0, device=self.device)
        
        individual_losses['total'] = total_loss
        return individual_losses
    
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update current epoch for scheduling"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def get_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get loss statistics"""
        stats = {}
        
        for loss_type, history in self.loss_history.items():
            if history:
                stats[loss_type] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'current': history[-1]
                }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = LossConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        temperature_schedule=TemperatureSchedule.COSINE_ANNEALING
    )
    
    # Create loss manager
    loss_manager = DistillationLossManager(config)
    
    # Test with dummy data
    student_outputs = torch.randn(32, 10)
    teacher_outputs = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    
    # Test individual losses
    kl_loss = loss_manager.compute_loss(LossType.KL_DIVERGENCE, student_outputs, teacher_outputs, targets)
    mse_loss = loss_manager.compute_loss(LossType.MSE, student_outputs, teacher_outputs)
    cosine_loss = loss_manager.compute_loss(LossType.COSINE, student_outputs, teacher_outputs)
    
    print("Distillation Loss Functions Created Successfully!")
    print(f"KL Divergence Loss: {kl_loss.item():.4f}")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Cosine Loss: {cosine_loss.item():.4f}")
    
    # Test combined loss
    combined_losses = loss_manager.compute_combined_loss(
        student_outputs, teacher_outputs, targets,
        loss_weights={
            LossType.KL_DIVERGENCE: 1.0,
            LossType.MSE: 0.1,
            LossType.COSINE: 0.1
        }
    )
    
    print(f"Combined Loss: {combined_losses['total'].item():.4f}")
    
    # Test temperature scheduling
    for epoch in range(5):
        loss_manager.update_epoch(epoch, 10)
        temp_loss = loss_manager.compute_loss(LossType.KL_DIVERGENCE, student_outputs, teacher_outputs, targets)
        print(f"Epoch {epoch}, Temperature Loss: {temp_loss.item():.4f}")
    
    # Get statistics
    stats = loss_manager.get_loss_statistics()
    print("Loss Statistics:", stats)