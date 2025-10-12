"""
Teacher-Student Training Framework
शिक्षक-छात्र प्रशिक्षण ढांचा

Advanced teacher-student training implementations including multi-teacher distillation,
self-distillation, and online distillation methods.
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
import copy
import matplotlib.pyplot as plt


class MultiTeacherStrategy(Enum):
    """Strategies for multi-teacher distillation"""
    ENSEMBLE = "ensemble"                # Average teacher predictions
    WEIGHTED_ENSEMBLE = "weighted"       # Weighted average based on performance
    ATTENTION_WEIGHTED = "attention"     # Attention-based weighting
    DYNAMIC_SELECTION = "dynamic"        # Dynamic teacher selection
    HIERARCHICAL = "hierarchical"        # Hierarchical teacher arrangement
    COMPETITIVE = "competitive"          # Competitive teacher selection


class OnlineDistillationMode(Enum):
    """Modes for online distillation"""
    MUTUAL = "mutual"                    # Mutual learning between peers
    PROGRESSIVE = "progressive"          # Progressive peer teaching
    COLLABORATIVE = "collaborative"      # Collaborative learning
    COMPETITIVE = "competitive"          # Competitive peer learning


@dataclass
class TeacherStudentConfig:
    """Configuration for teacher-student training"""
    # Basic parameters
    temperature: float = 4.0
    alpha: float = 0.7                   # Distillation loss weight
    beta: float = 0.3                    # Student loss weight
    
    # Multi-teacher parameters
    multi_teacher_strategy: MultiTeacherStrategy = MultiTeacherStrategy.ENSEMBLE
    teacher_weights: Optional[List[float]] = None
    dynamic_weighting: bool = True
    
    # Self-distillation parameters
    self_distill_epochs: int = 10
    self_distill_temperature: float = 3.0
    
    # Online distillation parameters
    online_mode: OnlineDistillationMode = OnlineDistillationMode.MUTUAL
    peer_learning_rate: float = 0.001
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Device and logging
    device: str = "cuda"
    verbose: bool = True


class TeacherStudentTrainer:
    """Basic teacher-student distillation trainer"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: TeacherStudentConfig):
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
            'accuracy': [],
            'teacher_accuracy': []
        }
    
    def distillation_loss(self, student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        temperature = self.config.temperature
        
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (temperature ** 2)
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        self.student_model.train()
        self.teacher_model.eval()
        
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch_data)
        
        student_logits = self.student_model(batch_data)
        
        # Compute losses
        dist_loss = self.distillation_loss(student_logits, teacher_logits)
        student_loss = F.cross_entropy(student_logits, batch_labels)
        
        total_loss = (self.config.alpha * dist_loss + 
                     self.config.beta * student_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracies
        with torch.no_grad():
            _, student_pred = torch.max(student_logits, 1)
            _, teacher_pred = torch.max(teacher_logits, 1)
            
            student_acc = (student_pred == batch_labels).float().mean().item()
            teacher_acc = (teacher_pred == batch_labels).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': dist_loss.item(),
            'student_loss': student_loss.item(),
            'accuracy': student_acc,
            'teacher_accuracy': teacher_acc
        }
    
    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """Train student model"""
        if self.config.verbose:
            self.logger.info(f"Starting teacher-student training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_data, batch_labels in train_loader:
                step_metrics = self.train_step(batch_data, batch_labels)
                
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
            
            # Average metrics for epoch
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Validation
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                avg_metrics['val_accuracy'] = val_acc
            
            # Store history
            for key, value in avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Student Acc: {avg_metrics['accuracy']:.4f}, "
                    f"Teacher Acc: {avg_metrics['teacher_accuracy']:.4f}"
                )
        
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
        
        return correct / total


class MultiTeacherDistiller:
    """Multi-teacher knowledge distillation"""
    
    def __init__(self, teacher_models: List[nn.Module], student_model: nn.Module,
                 config: TeacherStudentConfig):
        self.teacher_models = teacher_models
        self.student_model = student_model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move models to device
        for teacher in self.teacher_models:
            teacher.to(self.device)
            teacher.eval()
        
        self.student_model.to(self.device)
        
        # Initialize teacher weights
        if config.teacher_weights is None:
            self.teacher_weights = [1.0 / len(teacher_models)] * len(teacher_models)
        else:
            self.teacher_weights = config.teacher_weights.copy()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=config.learning_rate
        )
        
        # Attention mechanism for dynamic weighting
        if config.multi_teacher_strategy == MultiTeacherStrategy.ATTENTION_WEIGHTED:
            self.attention_net = nn.Sequential(
                nn.Linear(len(teacher_models), 64),
                nn.ReLU(),
                nn.Linear(64, len(teacher_models)),
                nn.Softmax(dim=1)
            ).to(self.device)
            
            self.attention_optimizer = torch.optim.Adam(
                self.attention_net.parameters(),
                lr=config.learning_rate
            )
        
        # Training statistics
        self.training_history = {
            'total_loss': [],
            'distillation_loss': [],
            'student_loss': [],
            'accuracy': [],
            'teacher_weights': []
        }
        
        # Teacher performance tracking
        self.teacher_performance = [[] for _ in range(len(teacher_models))]
    
    def compute_teacher_ensemble(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """Compute ensemble teacher predictions"""
        teacher_logits = []
        teacher_accuracies = []
        
        with torch.no_grad():
            for teacher in self.teacher_models:
                logits = teacher(batch_data)
                teacher_logits.append(logits)
        
        if self.config.multi_teacher_strategy == MultiTeacherStrategy.ENSEMBLE:
            # Simple average
            ensemble_logits = torch.stack(teacher_logits).mean(dim=0)
            weights = self.teacher_weights
            
        elif self.config.multi_teacher_strategy == MultiTeacherStrategy.WEIGHTED_ENSEMBLE:
            # Weighted average based on performance
            weights = self._update_teacher_weights()
            weighted_logits = []
            
            for i, logits in enumerate(teacher_logits):
                weighted_logits.append(weights[i] * logits)
            
            ensemble_logits = torch.stack(weighted_logits).sum(dim=0)
            
        elif self.config.multi_teacher_strategy == MultiTeacherStrategy.ATTENTION_WEIGHTED:
            # Attention-based weighting
            teacher_features = torch.stack([logits.mean(dim=1) for logits in teacher_logits], dim=1)
            attention_weights = self.attention_net(teacher_features)
            
            weighted_logits = []
            for i, logits in enumerate(teacher_logits):
                weight = attention_weights[:, i:i+1]
                weighted_logits.append(weight * logits)
            
            ensemble_logits = torch.stack(weighted_logits).sum(dim=0)
            weights = attention_weights.mean(dim=0).cpu().numpy().tolist()
            
        elif self.config.multi_teacher_strategy == MultiTeacherStrategy.DYNAMIC_SELECTION:
            # Select best teacher for each sample
            teacher_confidences = []
            for logits in teacher_logits:
                confidence = F.softmax(logits, dim=1).max(dim=1)[0]
                teacher_confidences.append(confidence)
            
            teacher_confidences = torch.stack(teacher_confidences, dim=1)
            best_teachers = teacher_confidences.argmax(dim=1)
            
            ensemble_logits = torch.zeros_like(teacher_logits[0])
            for i, teacher_idx in enumerate(best_teachers):
                ensemble_logits[i] = teacher_logits[teacher_idx][i]
            
            weights = self.teacher_weights
            
        else:
            # Default to simple ensemble
            ensemble_logits = torch.stack(teacher_logits).mean(dim=0)
            weights = self.teacher_weights
        
        return ensemble_logits, weights
    
    def _update_teacher_weights(self) -> List[float]:
        """Update teacher weights based on performance"""
        if not self.config.dynamic_weighting:
            return self.teacher_weights
        
        # Update weights based on recent performance
        if all(len(perf) > 0 for perf in self.teacher_performance):
            recent_performance = [np.mean(perf[-10:]) for perf in self.teacher_performance]
            
            # Softmax normalization
            performance_array = np.array(recent_performance)
            exp_perf = np.exp(performance_array - np.max(performance_array))
            weights = exp_perf / np.sum(exp_perf)
            
            self.teacher_weights = weights.tolist()
        
        return self.teacher_weights
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Perform one training step with multi-teacher distillation"""
        self.student_model.train()
        
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Get ensemble teacher predictions
        teacher_logits, current_weights = self.compute_teacher_ensemble(batch_data)
        
        # Student forward pass
        student_logits = self.student_model(batch_data)
        
        # Compute losses
        dist_loss = self._distillation_loss(student_logits, teacher_logits)
        student_loss = F.cross_entropy(student_logits, batch_labels)
        
        total_loss = (self.config.alpha * dist_loss + 
                     self.config.beta * student_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        if hasattr(self, 'attention_optimizer'):
            self.attention_optimizer.zero_grad()
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        if hasattr(self, 'attention_net'):
            torch.nn.utils.clip_grad_norm_(self.attention_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if hasattr(self, 'attention_optimizer'):
            self.attention_optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            _, predicted = torch.max(student_logits, 1)
            accuracy = (predicted == batch_labels).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': dist_loss.item(),
            'student_loss': student_loss.item(),
            'accuracy': accuracy,
            'teacher_weights': current_weights
        }
    
    def _distillation_loss(self, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss"""
        temperature = self.config.temperature
        
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """Train student with multi-teacher distillation"""
        if self.config.verbose:
            self.logger.info(f"Starting multi-teacher distillation with {len(self.teacher_models)} teachers")
        
        for epoch in range(self.config.epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_data, batch_labels in train_loader:
                step_metrics = self.train_step(batch_data, batch_labels)
                
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
            
            # Average metrics
            avg_metrics = {}
            for key, values in epoch_metrics.items():
                if key == 'teacher_weights':
                    # Average teacher weights
                    avg_weights = np.mean(values, axis=0).tolist()
                    avg_metrics[key] = avg_weights
                else:
                    avg_metrics[key] = np.mean(values)
            
            # Store history
            for key, value in avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Accuracy: {avg_metrics['accuracy']:.4f}, "
                    f"Weights: {[f'{w:.3f}' for w in avg_metrics['teacher_weights']]}"
                )
        
        return self.training_history


class SelfDistiller:
    """Self-distillation implementation"""
    
    def __init__(self, model: nn.Module, config: TeacherStudentConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        self.model.to(self.device)
        
        # Create a copy for teacher
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Training statistics
        self.training_history = {
            'total_loss': [],
            'distillation_loss': [],
            'student_loss': [],
            'accuracy': [],
            'improvement': []
        }
    
    def self_distill_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Perform self-distillation step"""
        self.model.train()
        self.teacher_model.eval()
        
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Teacher predictions (from previous iteration)
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch_data)
        
        # Student predictions (current model)
        student_logits = self.model(batch_data)
        
        # Compute losses
        temperature = self.config.self_distill_temperature
        
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        dist_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        student_loss = F.cross_entropy(student_logits, batch_labels)
        
        total_loss = self.config.alpha * dist_loss + self.config.beta * student_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy and improvement
        with torch.no_grad():
            _, student_pred = torch.max(student_logits, 1)
            _, teacher_pred = torch.max(teacher_logits, 1)
            
            student_acc = (student_pred == batch_labels).float().mean().item()
            teacher_acc = (teacher_pred == batch_labels).float().mean().item()
            improvement = student_acc - teacher_acc
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': dist_loss.item(),
            'student_loss': student_loss.item(),
            'accuracy': student_acc,
            'improvement': improvement
        }
    
    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """Train with self-distillation"""
        if self.config.verbose:
            self.logger.info(f"Starting self-distillation training")
        
        for epoch in range(self.config.epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_data, batch_labels in train_loader:
                step_metrics = self.self_distill_step(batch_data, batch_labels)
                
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
            
            # Average metrics
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Store history
            for key, value in avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Update teacher model periodically
            if (epoch + 1) % self.config.self_distill_epochs == 0:
                self.teacher_model.load_state_dict(self.model.state_dict())
                self.teacher_model.eval()
                
                if self.config.verbose:
                    self.logger.info(f"Updated teacher model at epoch {epoch + 1}")
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Accuracy: {avg_metrics['accuracy']:.4f}, "
                    f"Improvement: {avg_metrics['improvement']:.4f}"
                )
        
        return self.training_history


class OnlineDistiller:
    """Online distillation with peer learning"""
    
    def __init__(self, peer_models: List[nn.Module], config: TeacherStudentConfig):
        self.peer_models = peer_models
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move models to device
        for model in self.peer_models:
            model.to(self.device)
        
        # Initialize optimizers for each peer
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.peer_learning_rate)
            for model in peer_models
        ]
        
        # Training statistics
        self.training_history = {
            'total_loss': [],
            'peer_losses': [[] for _ in range(len(peer_models))],
            'peer_accuracies': [[] for _ in range(len(peer_models))],
            'ensemble_accuracy': []
        }
    
    def mutual_learning_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, Any]:
        """Perform mutual learning step"""
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Set all models to training mode
        for model in self.peer_models:
            model.train()
        
        # Forward pass for all peers
        peer_logits = []
        for model in self.peer_models:
            logits = model(batch_data)
            peer_logits.append(logits)
        
        # Compute ensemble predictions
        ensemble_logits = torch.stack(peer_logits).mean(dim=0)
        
        # Compute losses for each peer
        peer_losses = []
        peer_accuracies = []
        
        for i, (model, optimizer, logits) in enumerate(zip(self.peer_models, self.optimizers, peer_logits)):
            # Distillation from ensemble (excluding self)
            other_logits = [peer_logits[j] for j in range(len(peer_logits)) if j != i]
            if other_logits:
                teacher_logits = torch.stack(other_logits).mean(dim=0)
            else:
                teacher_logits = ensemble_logits
            
            # Compute losses
            temperature = self.config.temperature
            
            student_soft = F.log_softmax(logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
            
            dist_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
            student_loss = F.cross_entropy(logits, batch_labels)
            
            total_loss = self.config.alpha * dist_loss + self.config.beta * student_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == batch_labels).float().mean().item()
            
            peer_losses.append(total_loss.item())
            peer_accuracies.append(accuracy)
        
        # Ensemble accuracy
        with torch.no_grad():
            _, ensemble_pred = torch.max(ensemble_logits, 1)
            ensemble_acc = (ensemble_pred == batch_labels).float().mean().item()
        
        return {
            'total_loss': np.mean(peer_losses),
            'peer_losses': peer_losses,
            'peer_accuracies': peer_accuracies,
            'ensemble_accuracy': ensemble_acc
        }
    
    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """Train with online distillation"""
        if self.config.verbose:
            self.logger.info(f"Starting online distillation with {len(self.peer_models)} peers")
        
        for epoch in range(self.config.epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_data, batch_labels in train_loader:
                step_metrics = self.mutual_learning_step(batch_data, batch_labels)
                
                epoch_metrics['total_loss'].append(step_metrics['total_loss'])
                epoch_metrics['ensemble_accuracy'].append(step_metrics['ensemble_accuracy'])
                
                for i, (loss, acc) in enumerate(zip(step_metrics['peer_losses'], step_metrics['peer_accuracies'])):
                    epoch_metrics[f'peer_{i}_loss'].append(loss)
                    epoch_metrics[f'peer_{i}_accuracy'].append(acc)
            
            # Average metrics
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Store history
            self.training_history['total_loss'].append(avg_metrics['total_loss'])
            self.training_history['ensemble_accuracy'].append(avg_metrics['ensemble_accuracy'])
            
            for i in range(len(self.peer_models)):
                self.training_history['peer_losses'][i].append(avg_metrics[f'peer_{i}_loss'])
                self.training_history['peer_accuracies'][i].append(avg_metrics[f'peer_{i}_accuracy'])
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                peer_accs = [avg_metrics[f'peer_{i}_accuracy'] for i in range(len(self.peer_models))]
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Ensemble Acc: {avg_metrics['ensemble_accuracy']:.4f}, "
                    f"Peer Accs: {[f'{acc:.3f}' for acc in peer_accs]}"
                )
        
        return self.training_history


# Example usage
if __name__ == "__main__":
    # Create models
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
    
    # Create configuration
    config = TeacherStudentConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        epochs=50
    )
    
    # Test different distillation methods
    print("Teacher-Student Training Components Created Successfully!")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters())}")
    
    # Multi-teacher example
    teachers = [teacher, copy.deepcopy(teacher)]
    multi_distiller = MultiTeacherDistiller(teachers, student, config)
    print(f"Multi-teacher distiller created with {len(teachers)} teachers")
    
    # Self-distillation example
    self_distiller = SelfDistiller(copy.deepcopy(student), config)
    print("Self-distillation trainer created")
    
    # Online distillation example
    peers = [copy.deepcopy(student) for _ in range(3)]
    online_distiller = OnlineDistiller(peers, config)
    print(f"Online distillation trainer created with {len(peers)} peers")