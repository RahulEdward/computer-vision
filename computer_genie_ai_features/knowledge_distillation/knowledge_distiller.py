"""
Core Knowledge Distillation Framework
मुख्य ज्ञान आसवन ढांचा

Main knowledge distillation implementation with various distillation strategies,
temperature scheduling, and adaptive loss weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DistillationType(Enum):
    """Types of knowledge distillation"""
    RESPONSE_BASED = "response_based"      # Output-based distillation
    FEATURE_BASED = "feature_based"       # Intermediate feature distillation
    RELATION_BASED = "relation_based"     # Relational knowledge distillation
    ATTENTION_BASED = "attention_based"   # Attention transfer
    STRUCTURED = "structured"             # Structured knowledge transfer
    PROGRESSIVE = "progressive"           # Progressive distillation
    ONLINE = "online"                     # Online distillation
    SELF = "self"                        # Self-distillation


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies"""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    ADAPTIVE = "adaptive"
    CURRICULUM = "curriculum"


class LossWeightingStrategy(Enum):
    """Loss weighting strategies"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    CURRICULUM = "curriculum"
    UNCERTAINTY = "uncertainty"
    GRADIENT_BASED = "gradient_based"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    # Distillation type
    distillation_type: DistillationType = DistillationType.RESPONSE_BASED
    
    # Temperature parameters
    temperature: float = 4.0
    temperature_schedule: TemperatureSchedule = TemperatureSchedule.CONSTANT
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    
    # Loss weighting
    alpha: float = 0.7                    # Weight for distillation loss
    beta: float = 0.3                     # Weight for student loss
    loss_weighting: LossWeightingStrategy = LossWeightingStrategy.FIXED
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Feature distillation parameters
    feature_layers: List[str] = field(default_factory=list)
    feature_weight: float = 1.0
    
    # Attention distillation parameters
    attention_layers: List[str] = field(default_factory=list)
    attention_weight: float = 1.0
    
    # Progressive distillation parameters
    progressive_stages: int = 3
    stage_epochs: int = 30
    
    # Adaptive parameters
    adaptation_rate: float = 0.01
    performance_threshold: float = 0.95
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"


class TemperatureScheduler:
    """Temperature scheduler for knowledge distillation"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.current_temperature = config.temperature
        self.initial_temperature = config.temperature
        self.step_count = 0
        
    def step(self, epoch: int, total_epochs: int, 
             performance_metric: Optional[float] = None) -> float:
        """Update temperature based on schedule"""
        self.step_count += 1
        
        if self.config.temperature_schedule == TemperatureSchedule.CONSTANT:
            self.current_temperature = self.config.temperature
            
        elif self.config.temperature_schedule == TemperatureSchedule.LINEAR_DECAY:
            progress = epoch / total_epochs
            self.current_temperature = (
                self.config.max_temperature - 
                progress * (self.config.max_temperature - self.config.min_temperature)
            )
            
        elif self.config.temperature_schedule == TemperatureSchedule.EXPONENTIAL_DECAY:
            decay_rate = 0.95
            self.current_temperature = (
                self.config.temperature * (decay_rate ** epoch)
            )
            self.current_temperature = max(self.current_temperature, self.config.min_temperature)
            
        elif self.config.temperature_schedule == TemperatureSchedule.COSINE_ANNEALING:
            progress = epoch / total_epochs
            self.current_temperature = (
                self.config.min_temperature + 
                0.5 * (self.config.max_temperature - self.config.min_temperature) * 
                (1 + np.cos(np.pi * progress))
            )
            
        elif self.config.temperature_schedule == TemperatureSchedule.ADAPTIVE:
            if performance_metric is not None:
                # Increase temperature if performance is low
                if performance_metric < self.config.performance_threshold:
                    self.current_temperature = min(
                        self.current_temperature * 1.1, 
                        self.config.max_temperature
                    )
                else:
                    self.current_temperature = max(
                        self.current_temperature * 0.95, 
                        self.config.min_temperature
                    )
        
        return self.current_temperature
    
    def get_temperature(self) -> float:
        """Get current temperature"""
        return self.current_temperature


class LossWeightScheduler:
    """Scheduler for adaptive loss weighting"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.current_alpha = config.alpha
        self.current_beta = config.beta
        self.loss_history = []
        
    def update_weights(self, distillation_loss: float, student_loss: float,
                      epoch: int, total_epochs: int) -> Tuple[float, float]:
        """Update loss weights based on strategy"""
        self.loss_history.append({
            'distillation_loss': distillation_loss,
            'student_loss': student_loss,
            'epoch': epoch
        })
        
        if self.config.loss_weighting == LossWeightingStrategy.FIXED:
            return self.config.alpha, self.config.beta
            
        elif self.config.loss_weighting == LossWeightingStrategy.ADAPTIVE:
            # Adapt based on relative loss magnitudes
            if len(self.loss_history) > 10:
                recent_dist_loss = np.mean([h['distillation_loss'] for h in self.loss_history[-10:]])
                recent_student_loss = np.mean([h['student_loss'] for h in self.loss_history[-10:]])
                
                total_loss = recent_dist_loss + recent_student_loss
                if total_loss > 0:
                    self.current_alpha = recent_student_loss / total_loss
                    self.current_beta = recent_dist_loss / total_loss
                    
        elif self.config.loss_weighting == LossWeightingStrategy.CURRICULUM:
            # Start with more emphasis on student loss, gradually increase distillation
            progress = epoch / total_epochs
            self.current_alpha = 0.3 + 0.4 * progress  # 0.3 -> 0.7
            self.current_beta = 0.7 - 0.4 * progress   # 0.7 -> 0.3
            
        elif self.config.loss_weighting == LossWeightingStrategy.UNCERTAINTY:
            # Weight based on loss uncertainty (variance)
            if len(self.loss_history) > 20:
                dist_losses = [h['distillation_loss'] for h in self.loss_history[-20:]]
                student_losses = [h['student_loss'] for h in self.loss_history[-20:]]
                
                dist_var = np.var(dist_losses)
                student_var = np.var(student_losses)
                
                total_var = dist_var + student_var
                if total_var > 0:
                    # Higher weight for more uncertain (variable) loss
                    self.current_alpha = dist_var / total_var
                    self.current_beta = student_var / total_var
        
        # Ensure weights sum to 1
        total_weight = self.current_alpha + self.current_beta
        if total_weight > 0:
            self.current_alpha /= total_weight
            self.current_beta /= total_weight
        
        return self.current_alpha, self.current_beta


class KnowledgeDistiller:
    """Main knowledge distillation framework"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 config: DistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move models to device
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Set teacher to evaluation mode
        self.teacher_model.eval()
        
        # Initialize schedulers
        self.temperature_scheduler = TemperatureScheduler(config)
        self.loss_weight_scheduler = LossWeightScheduler(config)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=config.learning_rate
        )
        
        # Training statistics
        self.training_history = {
            'total_loss': [],
            'distillation_loss': [],
            'student_loss': [],
            'temperature': [],
            'alpha': [],
            'beta': [],
            'accuracy': []
        }
        
        # Feature hooks for intermediate layer distillation
        self.teacher_features = {}
        self.student_features = {}
        self.feature_hooks = []
        
        if config.feature_layers:
            self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register hooks for feature extraction"""
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        # Register hooks for specified layers
        for layer_name in self.config.feature_layers:
            # Teacher hooks
            teacher_layer = self._get_layer_by_name(self.teacher_model, layer_name)
            if teacher_layer is not None:
                hook = teacher_layer.register_forward_hook(get_teacher_hook(layer_name))
                self.feature_hooks.append(hook)
            
            # Student hooks
            student_layer = self._get_layer_by_name(self.student_model, layer_name)
            if student_layer is not None:
                hook = student_layer.register_forward_hook(get_student_hook(layer_name))
                self.feature_hooks.append(hook)
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from model"""
        try:
            layer = model
            for attr in layer_name.split('.'):
                layer = getattr(layer, attr)
            return layer
        except AttributeError:
            self.logger.warning(f"Layer {layer_name} not found in model")
            return None
    
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor, 
                         temperature: float) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        # Soften predictions with temperature
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared (as in original paper)
        return kl_loss * (temperature ** 2)
    
    def feature_distillation_loss(self) -> torch.Tensor:
        """Compute feature-based distillation loss"""
        if not self.config.feature_layers:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        count = 0
        
        for layer_name in self.config.feature_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Ensure same dimensions (may need adaptation layer)
                if teacher_feat.shape != student_feat.shape:
                    # Simple adaptation: average pooling or linear projection
                    if len(teacher_feat.shape) == 4:  # Conv features
                        # Adaptive average pooling
                        student_feat = F.adaptive_avg_pool2d(
                            student_feat, teacher_feat.shape[2:]
                        )
                    elif len(teacher_feat.shape) == 2:  # Linear features
                        # Linear projection (would need to be learned)
                        continue  # Skip for now
                
                # MSE loss between features
                feat_loss = F.mse_loss(student_feat, teacher_feat)
                total_loss += feat_loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor,
                   epoch: int, total_epochs: int) -> Dict[str, float]:
        """Perform one training step"""
        self.student_model.train()
        self.teacher_model.eval()
        
        # Move data to device
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Clear previous features
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch_data)
        
        student_logits = self.student_model(batch_data)
        
        # Get current temperature
        temperature = self.temperature_scheduler.get_temperature()
        
        # Compute losses
        # 1. Distillation loss
        dist_loss = self.distillation_loss(student_logits, teacher_logits, temperature)
        
        # 2. Student loss (cross-entropy with true labels)
        student_loss = F.cross_entropy(student_logits, batch_labels)
        
        # 3. Feature distillation loss
        feature_loss = self.feature_distillation_loss()
        
        # Update loss weights
        alpha, beta = self.loss_weight_scheduler.update_weights(
            dist_loss.item(), student_loss.item(), epoch, total_epochs
        )
        
        # Combined loss
        total_loss = (alpha * dist_loss + 
                     beta * student_loss + 
                     self.config.feature_weight * feature_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            _, predicted = torch.max(student_logits, 1)
            accuracy = (predicted == batch_labels).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': dist_loss.item(),
            'student_loss': student_loss.item(),
            'feature_loss': feature_loss.item(),
            'accuracy': accuracy,
            'temperature': temperature,
            'alpha': alpha,
            'beta': beta
        }
    
    def train(self, train_loader, val_loader=None, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Train student model with knowledge distillation"""
        if epochs is None:
            epochs = self.config.epochs
        
        if self.config.verbose:
            self.logger.info(f"Starting knowledge distillation training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_metrics = {
                'total_loss': [],
                'distillation_loss': [],
                'student_loss': [],
                'feature_loss': [],
                'accuracy': [],
                'temperature': [],
                'alpha': [],
                'beta': []
            }
            
            # Training loop
            for batch_idx, (data, labels) in enumerate(train_loader):
                step_metrics = self.train_step(data, labels, epoch, epochs)
                
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
            
            # Average metrics for epoch
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Update temperature scheduler
            val_accuracy = None
            if val_loader is not None:
                val_accuracy = self.evaluate(val_loader)
                avg_metrics['val_accuracy'] = val_accuracy
            
            self.temperature_scheduler.step(epoch, epochs, val_accuracy)
            
            # Store training history
            for key, value in avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Accuracy: {avg_metrics['accuracy']:.4f}, "
                    f"Temperature: {avg_metrics['temperature']:.2f}"
                )
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)
        
        return self.training_history
    
    def evaluate(self, test_loader) -> float:
        """Evaluate student model"""
        self.student_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.student_model(data)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f'distillation_checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.verbose:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        if self.config.verbose:
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.training_history['total_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Knowledge Distillation Training History', fontsize=16)
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(self.training_history['total_loss'], label='Total Loss')
        ax1.plot(self.training_history['distillation_loss'], label='Distillation Loss')
        ax1.plot(self.training_history['student_loss'], label='Student Loss')
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2 = axes[0, 1]
        ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.training_history:
            ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Temperature curve
        ax3 = axes[1, 0]
        ax3.plot(self.training_history['temperature'])
        ax3.set_title('Temperature Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Temperature')
        ax3.grid(True)
        
        # Loss weights
        ax4 = axes[1, 1]
        ax4.plot(self.training_history['alpha'], label='Alpha (Distillation)')
        ax4.plot(self.training_history['beta'], label='Beta (Student)')
        ax4.set_title('Loss Weights')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Weight')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive distillation statistics"""
        stats = {
            'config': {
                'distillation_type': self.config.distillation_type.value,
                'temperature': self.config.temperature,
                'temperature_schedule': self.config.temperature_schedule.value,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'loss_weighting': self.config.loss_weighting.value
            },
            'training': {
                'epochs_trained': len(self.training_history['total_loss']),
                'final_temperature': self.temperature_scheduler.get_temperature(),
                'final_alpha': self.loss_weight_scheduler.current_alpha,
                'final_beta': self.loss_weight_scheduler.current_beta
            }
        }
        
        # Training statistics
        if self.training_history['total_loss']:
            stats['training'].update({
                'final_loss': self.training_history['total_loss'][-1],
                'best_loss': min(self.training_history['total_loss']),
                'final_accuracy': self.training_history['accuracy'][-1],
                'best_accuracy': max(self.training_history['accuracy']),
                'avg_distillation_loss': np.mean(self.training_history['distillation_loss']),
                'avg_student_loss': np.mean(self.training_history['student_loss'])
            })
        
        # Model statistics
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        stats['model'] = {
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': teacher_params / student_params if student_params > 0 else 0,
            'parameter_reduction': 1 - (student_params / teacher_params) if teacher_params > 0 else 0
        }
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        # Remove feature hooks
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks.clear()
        
        # Clear feature storage
        self.teacher_features.clear()
        self.student_features.clear()


# Example usage
if __name__ == "__main__":
    # Create simple teacher and student models
    teacher = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    student = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Create distillation configuration
    config = DistillationConfig(
        distillation_type=DistillationType.RESPONSE_BASED,
        temperature=4.0,
        temperature_schedule=TemperatureSchedule.COSINE_ANNEALING,
        alpha=0.7,
        beta=0.3,
        epochs=50
    )
    
    # Create knowledge distiller
    distiller = KnowledgeDistiller(teacher, student, config)
    
    print("Knowledge distiller created successfully!")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters())}")
    print(f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x")