"""
Adversarial Training Implementation
विरोधी प्रशिक्षण कार्यान्वयन

Implements adversarial training strategies to improve model robustness
against adversarial attacks and enhance generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import math
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from .adversarial_attacks import (
    AdversarialAttack, FGSMAttack, PGDAttack, CWAttack, 
    AttackConfig, AttackType, NormType
)


class TrainingStrategy(Enum):
    """Adversarial training strategies"""
    STANDARD = "standard"                    # Standard adversarial training
    FAST = "fast"                           # Fast adversarial training (FGSM)
    FREE = "free"                           # Free adversarial training
    TRADES = "trades"                       # TRADES (TRadeoff-inspired Adversarial DEfense)
    MART = "mart"                           # MART (Misclassification Aware adveRsarial Training)
    AWP = "awp"                            # Adversarial Weight Perturbation
    PROGRESSIVE = "progressive"             # Progressive adversarial training
    CURRICULUM = "curriculum"               # Curriculum adversarial training
    ENSEMBLE = "ensemble"                   # Ensemble adversarial training
    SELF_ADAPTIVE = "self_adaptive"         # Self-adaptive adversarial training


class OptimizationMethod(Enum):
    """Optimization methods for robust training"""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    SAM = "sam"                            # Sharpness-Aware Minimization
    ASAM = "asam"                          # Adaptive SAM


@dataclass
class RobustnessConfig:
    """Configuration for adversarial training"""
    # Training strategy
    strategy: TrainingStrategy = TrainingStrategy.STANDARD
    
    # Attack configuration for training
    attack_config: AttackConfig = field(default_factory=lambda: AttackConfig(
        attack_type=AttackType.PGD,
        epsilon=8.0/255.0,
        step_size=2.0/255.0,
        num_steps=10
    ))
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # Optimization
    optimizer_type: OptimizationMethod = OptimizationMethod.SGD
    scheduler_type: str = "cosine"         # cosine, step, multistep
    warmup_epochs: int = 5
    
    # Adversarial training specific
    adversarial_ratio: float = 1.0         # Ratio of adversarial examples
    clean_ratio: float = 0.0               # Ratio of clean examples
    
    # TRADES specific
    trades_beta: float = 6.0               # Trade-off parameter for TRADES
    
    # MART specific
    mart_beta: float = 6.0                 # Trade-off parameter for MART
    
    # AWP specific
    awp_gamma: float = 0.01                # AWP perturbation size
    awp_warmup: int = 10                   # AWP warmup epochs
    
    # Progressive training
    progressive_schedule: List[float] = field(default_factory=lambda: [0.0, 2.0/255.0, 4.0/255.0, 8.0/255.0])
    progressive_epochs: List[int] = field(default_factory=lambda: [10, 30, 60, 100])
    
    # Curriculum training
    curriculum_schedule: str = "linear"     # linear, exponential, cosine
    curriculum_start_epsilon: float = 1.0/255.0
    curriculum_end_epsilon: float = 8.0/255.0
    
    # Ensemble training
    ensemble_size: int = 3
    ensemble_diversity_weight: float = 0.1
    
    # Regularization
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # Evaluation
    eval_attack_config: AttackConfig = field(default_factory=lambda: AttackConfig(
        attack_type=AttackType.PGD,
        epsilon=8.0/255.0,
        step_size=2.0/255.0,
        num_steps=20
    ))
    
    # Checkpointing and logging
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5
    
    # Device and performance
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Advanced options
    use_amp: bool = True                   # Automatic Mixed Precision
    gradient_clipping: float = 0.0         # Gradient clipping threshold
    early_stopping_patience: int = 10
    
    # Monitoring
    track_gradients: bool = False
    track_weights: bool = False
    visualize_training: bool = True


class TrainingStats:
    """Track training statistics and metrics"""
    
    def __init__(self):
        self.epoch_stats = []
        self.batch_stats = []
        self.best_clean_acc = 0.0
        self.best_robust_acc = 0.0
        self.best_epoch = 0
        
        # Loss tracking
        self.clean_losses = []
        self.adversarial_losses = []
        self.total_losses = []
        
        # Accuracy tracking
        self.clean_accuracies = []
        self.robust_accuracies = []
        
        # Gradient and weight statistics
        self.gradient_norms = []
        self.weight_norms = []
        
        # Training time
        self.epoch_times = []
        self.total_training_time = 0.0
    
    def update_epoch(self, epoch: int, clean_loss: float, adv_loss: float,
                    clean_acc: float, robust_acc: float, epoch_time: float) -> None:
        """Update epoch statistics"""
        self.epoch_stats.append({
            'epoch': epoch,
            'clean_loss': clean_loss,
            'adversarial_loss': adv_loss,
            'total_loss': clean_loss + adv_loss,
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'epoch_time': epoch_time
        })
        
        # Update tracking lists
        self.clean_losses.append(clean_loss)
        self.adversarial_losses.append(adv_loss)
        self.total_losses.append(clean_loss + adv_loss)
        self.clean_accuracies.append(clean_acc)
        self.robust_accuracies.append(robust_acc)
        self.epoch_times.append(epoch_time)
        
        # Update best metrics
        if robust_acc > self.best_robust_acc:
            self.best_robust_acc = robust_acc
            self.best_epoch = epoch
        
        if clean_acc > self.best_clean_acc:
            self.best_clean_acc = clean_acc
        
        self.total_training_time += epoch_time
    
    def update_batch(self, batch_idx: int, loss: float, accuracy: float) -> None:
        """Update batch statistics"""
        self.batch_stats.append({
            'batch': batch_idx,
            'loss': loss,
            'accuracy': accuracy
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'total_epochs': len(self.epoch_stats),
            'best_clean_accuracy': self.best_clean_acc,
            'best_robust_accuracy': self.best_robust_acc,
            'best_epoch': self.best_epoch,
            'final_clean_accuracy': self.clean_accuracies[-1] if self.clean_accuracies else 0.0,
            'final_robust_accuracy': self.robust_accuracies[-1] if self.robust_accuracies else 0.0,
            'total_training_time': self.total_training_time,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0.0,
            'convergence_epoch': self._find_convergence_epoch()
        }
    
    def _find_convergence_epoch(self) -> int:
        """Find epoch where training converged"""
        if len(self.robust_accuracies) < 10:
            return len(self.robust_accuracies)
        
        # Look for plateau in robust accuracy
        window_size = 5
        for i in range(window_size, len(self.robust_accuracies)):
            recent_acc = self.robust_accuracies[i-window_size:i]
            if np.std(recent_acc) < 0.01:  # Low variance indicates convergence
                return i - window_size
        
        return len(self.robust_accuracies)


class RobustOptimizer:
    """Robust optimization methods for adversarial training"""
    
    def __init__(self, model: nn.Module, config: RobustnessConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize optimizer
        if config.optimizer_type == OptimizationMethod.SGD:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizationMethod.ADAM:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizationMethod.ADAMW:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizationMethod.SAM:
            self.optimizer = SAMOptimizer(
                model.parameters(),
                base_optimizer=optim.SGD,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        
        # Initialize scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        elif config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif config.scheduler_type == "multistep":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60, 120, 160], gamma=0.2
            )
        else:
            self.scheduler = None
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step"""
        if self.config.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            self.optimizer.step()
    
    def zero_grad(self) -> None:
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def schedule_step(self) -> None:
        """Step learning rate scheduler"""
        if self.scheduler:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class SAMOptimizer:
    """Sharpness-Aware Minimization optimizer"""
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.param_groups = list(params)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho
    
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group:
                if p.grad is None:
                    continue
                p.sub_(p.grad)  # go back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()


class AdversarialTrainer:
    """Main adversarial training class"""
    
    def __init__(self, model: nn.Module, config: RobustnessConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = RobustOptimizer(model, config)
        
        # Initialize attack for training
        self.train_attack = self._create_attack(config.attack_config)
        self.eval_attack = self._create_attack(config.eval_attack_config)
        
        # Initialize statistics
        self.stats = TrainingStats()
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize curriculum/progressive schedules
        self.current_epsilon = config.attack_config.epsilon
        self.epoch = 0
    
    def _create_attack(self, attack_config: AttackConfig) -> AdversarialAttack:
        """Create attack instance based on configuration"""
        if attack_config.attack_type == AttackType.FGSM:
            return FGSMAttack(attack_config)
        elif attack_config.attack_type == AttackType.PGD:
            return PGDAttack(attack_config)
        elif attack_config.attack_type == AttackType.CW:
            return CWAttack(attack_config)
        else:
            return PGDAttack(attack_config)  # Default to PGD
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info(f"Starting adversarial training with strategy: {self.config.strategy}")
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        
        start_time = time.time()
        best_robust_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Update curriculum/progressive schedules
            self._update_training_schedule(epoch)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self._evaluate(val_loader, epoch)
            else:
                val_metrics = {'clean_accuracy': 0.0, 'robust_accuracy': 0.0, 'clean_loss': 0.0}
            
            epoch_time = time.time() - epoch_start_time
            
            # Update statistics
            self.stats.update_epoch(
                epoch, train_metrics['clean_loss'], train_metrics['adversarial_loss'],
                val_metrics['clean_accuracy'], val_metrics['robust_accuracy'], epoch_time
            )
            
            # Learning rate scheduling
            self.optimizer.schedule_step()
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Clean Acc: {val_metrics['clean_accuracy']:.3f}, "
                    f"Val Robust Acc: {val_metrics['robust_accuracy']:.3f}, "
                    f"LR: {self.optimizer.get_lr():.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['robust_accuracy'])
            
            # Early stopping
            if val_metrics['robust_accuracy'] > best_robust_acc:
                best_robust_acc = val_metrics['robust_accuracy']
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        self.stats.total_training_time = total_time
        
        # Final evaluation
        if val_loader is not None:
            final_metrics = self._evaluate(val_loader, self.config.num_epochs)
        else:
            final_metrics = {}
        
        # Generate training report
        training_summary = self.stats.get_summary()
        training_summary.update(final_metrics)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best robust accuracy: {self.stats.best_robust_acc:.3f}")
        self.logger.info(f"Total training time: {total_time:.2f}s")
        
        return training_summary
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_clean_loss = 0.0
        total_adv_loss = 0.0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Calculate loss based on training strategy
            if self.config.strategy == TrainingStrategy.STANDARD:
                loss, clean_loss, adv_loss = self._standard_training_step(inputs, targets)
            elif self.config.strategy == TrainingStrategy.FAST:
                loss, clean_loss, adv_loss = self._fast_training_step(inputs, targets)
            elif self.config.strategy == TrainingStrategy.TRADES:
                loss, clean_loss, adv_loss = self._trades_training_step(inputs, targets)
            elif self.config.strategy == TrainingStrategy.MART:
                loss, clean_loss, adv_loss = self._mart_training_step(inputs, targets)
            elif self.config.strategy == TrainingStrategy.AWP:
                loss, clean_loss, adv_loss = self._awp_training_step(inputs, targets, epoch)
            else:
                loss, clean_loss, adv_loss = self._standard_training_step(inputs, targets)
            
            # Optimization step
            self.optimizer.step(loss)
            
            # Update statistics
            total_clean_loss += clean_loss * batch_size
            total_adv_loss += adv_loss * batch_size
            total_samples += batch_size
            
            # Batch logging
            if batch_idx % self.config.log_interval == 0:
                self.logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)} - "
                    f"Loss: {loss.item():.4f}, Clean: {clean_loss:.4f}, Adv: {adv_loss:.4f}"
                )
        
        return {
            'clean_loss': total_clean_loss / total_samples,
            'adversarial_loss': total_adv_loss / total_samples,
            'total_loss': (total_clean_loss + total_adv_loss) / total_samples
        }
    
    def _standard_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Standard adversarial training step"""
        # Generate adversarial examples
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            adversarial = self.train_attack.attack(self.model, inputs, targets)
            
            # Calculate losses
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(adversarial)
            
            clean_loss = F.cross_entropy(clean_outputs, targets)
            adv_loss = F.cross_entropy(adv_outputs, targets)
            
            # Combine losses
            total_loss = (self.config.clean_ratio * clean_loss + 
                         self.config.adversarial_ratio * adv_loss)
        
        return total_loss, clean_loss.item(), adv_loss.item()
    
    def _fast_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Fast adversarial training (FGSM only)"""
        # Use FGSM for fast training
        fgsm_config = AttackConfig(
            attack_type=AttackType.FGSM,
            epsilon=self.current_epsilon,
            norm_type=self.config.attack_config.norm_type
        )
        fgsm_attack = FGSMAttack(fgsm_config)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            adversarial = fgsm_attack.attack(self.model, inputs, targets)
            
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(adversarial)
            
            clean_loss = F.cross_entropy(clean_outputs, targets)
            adv_loss = F.cross_entropy(adv_outputs, targets)
            
            total_loss = (self.config.clean_ratio * clean_loss + 
                         self.config.adversarial_ratio * adv_loss)
        
        return total_loss, clean_loss.item(), adv_loss.item()
    
    def _trades_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """TRADES training step"""
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            # Clean forward pass
            clean_outputs = self.model(inputs)
            clean_loss = F.cross_entropy(clean_outputs, targets)
            
            # Generate adversarial examples
            adversarial = self.train_attack.attack(self.model, inputs, targets)
            adv_outputs = self.model(adversarial)
            
            # TRADES loss (KL divergence between clean and adversarial outputs)
            kl_loss = F.kl_div(
                F.log_softmax(adv_outputs, dim=1),
                F.softmax(clean_outputs, dim=1),
                reduction='batchmean'
            )
            
            total_loss = clean_loss + self.config.trades_beta * kl_loss
        
        return total_loss, clean_loss.item(), kl_loss.item()
    
    def _mart_training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """MART training step"""
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            # Clean forward pass
            clean_outputs = self.model(inputs)
            clean_loss = F.cross_entropy(clean_outputs, targets)
            
            # Generate adversarial examples
            adversarial = self.train_attack.attack(self.model, inputs, targets)
            adv_outputs = self.model(adversarial)
            
            # MART loss
            clean_probs = F.softmax(clean_outputs, dim=1)
            adv_probs = F.softmax(adv_outputs, dim=1)
            
            # Misclassification-aware term
            clean_preds = clean_outputs.argmax(dim=1)
            correct_mask = (clean_preds == targets).float()
            
            kl_loss = F.kl_div(
                F.log_softmax(adv_outputs, dim=1),
                clean_probs,
                reduction='none'
            ).sum(dim=1)
            
            # Weight KL loss by correctness
            weighted_kl = (kl_loss * correct_mask).mean()
            
            total_loss = clean_loss + self.config.mart_beta * weighted_kl
        
        return total_loss, clean_loss.item(), weighted_kl.item()
    
    def _awp_training_step(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, float, float]:
        """AWP (Adversarial Weight Perturbation) training step"""
        if epoch < self.config.awp_warmup:
            return self._standard_training_step(inputs, targets)
        
        # Standard adversarial training loss
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            adversarial = self.train_attack.attack(self.model, inputs, targets)
            adv_outputs = self.model(adversarial)
            adv_loss = F.cross_entropy(adv_outputs, targets)
        
        # Calculate gradients for AWP
        adv_loss.backward(retain_graph=True)
        
        # Apply weight perturbation
        self._apply_weight_perturbation()
        
        # Calculate loss with perturbed weights
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            perturbed_outputs = self.model(adversarial)
            perturbed_loss = F.cross_entropy(perturbed_outputs, targets)
        
        # Restore original weights
        self._restore_weights()
        
        return perturbed_loss, 0.0, perturbed_loss.item()
    
    def _apply_weight_perturbation(self) -> None:
        """Apply AWP weight perturbation"""
        # Store original weights and apply perturbation
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.original_weights[name] = param.data.clone()
                grad_norm = param.grad.norm()
                if grad_norm > 0:
                    perturbation = self.config.awp_gamma * param.grad / grad_norm
                    param.data.add_(perturbation)
    
    def _restore_weights(self) -> None:
        """Restore original weights after AWP"""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])
    
    def _update_training_schedule(self, epoch: int) -> None:
        """Update curriculum/progressive training schedules"""
        if self.config.strategy == TrainingStrategy.PROGRESSIVE:
            # Progressive adversarial training
            for i, epoch_threshold in enumerate(self.config.progressive_epochs):
                if epoch < epoch_threshold:
                    self.current_epsilon = self.config.progressive_schedule[i]
                    break
            
            # Update attack epsilon
            self.train_attack.config.epsilon = self.current_epsilon
        
        elif self.config.strategy == TrainingStrategy.CURRICULUM:
            # Curriculum adversarial training
            progress = epoch / self.config.num_epochs
            
            if self.config.curriculum_schedule == "linear":
                self.current_epsilon = (
                    self.config.curriculum_start_epsilon + 
                    progress * (self.config.curriculum_end_epsilon - self.config.curriculum_start_epsilon)
                )
            elif self.config.curriculum_schedule == "exponential":
                self.current_epsilon = (
                    self.config.curriculum_start_epsilon * 
                    (self.config.curriculum_end_epsilon / self.config.curriculum_start_epsilon) ** progress
                )
            elif self.config.curriculum_schedule == "cosine":
                self.current_epsilon = (
                    self.config.curriculum_start_epsilon + 
                    0.5 * (self.config.curriculum_end_epsilon - self.config.curriculum_start_epsilon) * 
                    (1 - math.cos(math.pi * progress))
                )
            
            # Update attack epsilon
            self.train_attack.config.epsilon = self.current_epsilon
    
    def _evaluate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total_samples = 0
        total_clean_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Clean evaluation
                clean_outputs = self.model(inputs)
                clean_loss = F.cross_entropy(clean_outputs, targets)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_correct += (clean_preds == targets).sum().item()
                
                # Robust evaluation
                adversarial = self.eval_attack.attack(self.model, inputs, targets)
                robust_outputs = self.model(adversarial)
                robust_preds = robust_outputs.argmax(dim=1)
                robust_correct += (robust_preds == targets).sum().item()
                
                total_samples += batch_size
                total_clean_loss += clean_loss.item() * batch_size
        
        clean_accuracy = clean_correct / total_samples
        robust_accuracy = robust_correct / total_samples
        avg_clean_loss = total_clean_loss / total_samples
        
        return {
            'clean_accuracy': clean_accuracy,
            'robust_accuracy': robust_accuracy,
            'clean_loss': avg_clean_loss
        }
    
    def _save_checkpoint(self, epoch: int, robust_acc: float) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'robust_accuracy': robust_acc,
            'config': self.config,
            'stats': self.stats
        }
        
        checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
    
    def _save_best_model(self) -> None:
        """Save best model"""
        best_model_path = os.path.join(self.config.save_dir, "best_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'stats': self.stats
        }, best_model_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        self.epoch = checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history"""
        if not self.config.visualize_training:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(len(self.stats.clean_accuracies))
        
        # Accuracy plot
        axes[0, 0].plot(epochs, self.stats.clean_accuracies, label='Clean Accuracy', color='blue')
        axes[0, 0].plot(epochs, self.stats.robust_accuracies, label='Robust Accuracy', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(epochs, self.stats.clean_losses, label='Clean Loss', color='blue')
        axes[0, 1].plot(epochs, self.stats.adversarial_losses, label='Adversarial Loss', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        if hasattr(self.optimizer, 'scheduler') and self.optimizer.scheduler:
            lrs = [self.optimizer.get_lr() for _ in epochs]
            axes[1, 0].plot(epochs, lrs, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        # Epoch time plot
        axes[1, 1].plot(epochs, self.stats.epoch_times, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.config.save_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()


class DefenseEvaluator:
    """Evaluate defense mechanisms and robustness"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
    
    def evaluate_robustness(self, model: nn.Module, test_loader: DataLoader,
                          attack_configs: List[AttackConfig]) -> Dict[str, Any]:
        """Comprehensive robustness evaluation"""
        model.eval()
        results = {}
        
        for i, attack_config in enumerate(attack_configs):
            attack_name = f"{attack_config.attack_type.value}_eps_{attack_config.epsilon:.3f}"
            
            # Create attack
            if attack_config.attack_type == AttackType.FGSM:
                attack = FGSMAttack(attack_config)
            elif attack_config.attack_type == AttackType.PGD:
                attack = PGDAttack(attack_config)
            elif attack_config.attack_type == AttackType.CW:
                attack = CWAttack(attack_config)
            else:
                attack = PGDAttack(attack_config)
            
            # Evaluate attack
            clean_correct = 0
            robust_correct = 0
            total_samples = 0
            
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Clean accuracy
                with torch.no_grad():
                    clean_outputs = model(inputs)
                    clean_preds = clean_outputs.argmax(dim=1)
                    clean_correct += (clean_preds == targets).sum().item()
                
                # Robust accuracy
                adversarial = attack.attack(model, inputs, targets)
                with torch.no_grad():
                    robust_outputs = model(adversarial)
                    robust_preds = robust_outputs.argmax(dim=1)
                    robust_correct += (robust_preds == targets).sum().item()
                
                total_samples += inputs.size(0)
            
            clean_acc = clean_correct / total_samples
            robust_acc = robust_correct / total_samples
            
            results[attack_name] = {
                'clean_accuracy': clean_acc,
                'robust_accuracy': robust_acc,
                'attack_success_rate': 1.0 - robust_acc,
                'attack_config': attack_config
            }
            
            self.logger.info(f"Attack {attack_name}: Clean Acc: {clean_acc:.3f}, "
                           f"Robust Acc: {robust_acc:.3f}")
        
        return results
    
    def compare_defenses(self, models: Dict[str, nn.Module], test_loader: DataLoader,
                        attack_configs: List[AttackConfig]) -> Dict[str, Any]:
        """Compare multiple defense methods"""
        comparison_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating defense: {model_name}")
            results = self.evaluate_robustness(model, test_loader, attack_configs)
            comparison_results[model_name] = results
        
        # Generate comparison summary
        summary = {
            'individual_results': comparison_results,
            'best_clean_accuracy': max(
                [(name, np.mean([r['clean_accuracy'] for r in results.values()])) 
                 for name, results in comparison_results.items()],
                key=lambda x: x[1]
            ),
            'best_robust_accuracy': max(
                [(name, np.mean([r['robust_accuracy'] for r in results.values()])) 
                 for name, results in comparison_results.items()],
                key=lambda x: x[1]
            )
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = RobustnessConfig(
        strategy=TrainingStrategy.STANDARD,
        num_epochs=100,
        batch_size=128,
        learning_rate=0.1,
        attack_config=AttackConfig(
            attack_type=AttackType.PGD,
            epsilon=8.0/255.0,
            step_size=2.0/255.0,
            num_steps=10
        )
    )
    
    print("Adversarial trainer created successfully!")
    print(f"Configuration: {config}")
    print(f"Training strategy: {config.strategy}")
    print(f"Attack configuration: {config.attack_config}")