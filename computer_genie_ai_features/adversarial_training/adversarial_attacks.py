"""
Adversarial Attack Generation and Evaluation
विरोधी हमला उत्पादन और मूल्यांकन

Implements various adversarial attack methods for testing model robustness
and generating adversarial examples for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import math
from abc import ABC, abstractmethod
import warnings


class AttackType(Enum):
    """Types of adversarial attacks"""
    FGSM = "fgsm"                    # Fast Gradient Sign Method
    PGD = "pgd"                      # Projected Gradient Descent
    CW = "cw"                        # Carlini & Wagner
    DEEPFOOL = "deepfool"            # DeepFool
    JSMA = "jsma"                    # Jacobian-based Saliency Map
    AUTO_ATTACK = "auto_attack"      # AutoAttack
    SQUARE = "square"                # Square Attack
    BOUNDARY = "boundary"            # Boundary Attack
    SPSA = "spsa"                    # Simultaneous Perturbation SA
    NATURAL_ES = "natural_es"        # Natural Evolution Strategies
    SEMANTIC = "semantic"            # Semantic attacks
    PHYSICAL = "physical"            # Physical world attacks


class NormType(Enum):
    """Norm types for perturbation constraints"""
    L_INF = "linf"                   # L-infinity norm
    L2 = "l2"                        # L2 norm
    L1 = "l1"                        # L1 norm
    L0 = "l0"                        # L0 norm (sparsity)


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    # Attack type and parameters
    attack_type: AttackType = AttackType.PGD
    norm_type: NormType = NormType.L_INF
    epsilon: float = 8.0 / 255.0     # Perturbation budget
    step_size: float = 2.0 / 255.0   # Step size for iterative attacks
    num_steps: int = 10              # Number of attack iterations
    
    # Targeting
    targeted: bool = False           # Targeted vs untargeted attack
    target_class: Optional[int] = None
    
    # Optimization parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    decay_factor: float = 1.0
    
    # Randomization
    random_start: bool = True        # Random initialization
    random_restarts: int = 1         # Number of random restarts
    
    # Constraints
    clip_min: float = 0.0           # Minimum pixel value
    clip_max: float = 1.0           # Maximum pixel value
    
    # C&W specific parameters
    confidence: float = 0.0          # Confidence parameter for C&W
    c_init: float = 1e-3            # Initial c value for C&W
    c_upper: float = 1e10           # Upper bound for c
    binary_search_steps: int = 9     # Binary search iterations
    
    # AutoAttack parameters
    version: str = "standard"        # AutoAttack version
    
    # Evaluation parameters
    batch_size: int = 32
    device: str = "cuda"
    verbose: bool = False
    
    # Advanced options
    use_best_loss: bool = False      # Use best loss during attack
    loss_function: str = "ce"        # Loss function (ce, cw, etc.)
    early_stopping: bool = False     # Early stopping criterion
    
    # Physical attack parameters
    physical_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic attack parameters
    semantic_transformations: List[str] = field(default_factory=list)


class AttackStats:
    """Track attack statistics and performance"""
    
    def __init__(self):
        self.total_samples = 0
        self.successful_attacks = 0
        self.average_perturbation = 0.0
        self.average_confidence_drop = 0.0
        self.attack_times = []
        self.perturbation_norms = []
        self.success_by_class = {}
        self.confidence_changes = []
    
    def update(self, original_preds: torch.Tensor, adversarial_preds: torch.Tensor,
               perturbations: torch.Tensor, attack_time: float,
               original_confidences: torch.Tensor, adversarial_confidences: torch.Tensor) -> None:
        """Update attack statistics"""
        batch_size = original_preds.size(0)
        self.total_samples += batch_size
        
        # Calculate success rate
        successful = (original_preds != adversarial_preds).sum().item()
        self.successful_attacks += successful
        
        # Calculate perturbation statistics
        perturbation_norms = torch.norm(perturbations.view(batch_size, -1), dim=1)
        self.perturbation_norms.extend(perturbation_norms.cpu().numpy())
        self.average_perturbation = np.mean(self.perturbation_norms)
        
        # Calculate confidence changes
        confidence_drops = original_confidences - adversarial_confidences
        self.confidence_changes.extend(confidence_drops.cpu().numpy())
        self.average_confidence_drop = np.mean(self.confidence_changes)
        
        # Track timing
        self.attack_times.append(attack_time)
        
        # Track success by class
        for i in range(batch_size):
            orig_class = original_preds[i].item()
            if orig_class not in self.success_by_class:
                self.success_by_class[orig_class] = {'total': 0, 'successful': 0}
            
            self.success_by_class[orig_class]['total'] += 1
            if original_preds[i] != adversarial_preds[i]:
                self.success_by_class[orig_class]['successful'] += 1
    
    def get_success_rate(self) -> float:
        """Get overall attack success rate"""
        return self.successful_attacks / self.total_samples if self.total_samples > 0 else 0.0
    
    def get_average_time(self) -> float:
        """Get average attack time"""
        return np.mean(self.attack_times) if self.attack_times else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics"""
        return {
            'total_samples': self.total_samples,
            'success_rate': self.get_success_rate(),
            'average_perturbation': self.average_perturbation,
            'average_confidence_drop': self.average_confidence_drop,
            'average_time': self.get_average_time(),
            'success_by_class': {k: v['successful'] / v['total'] if v['total'] > 0 else 0.0 
                               for k, v in self.success_by_class.items()},
            'perturbation_std': np.std(self.perturbation_norms) if self.perturbation_norms else 0.0,
            'confidence_drop_std': np.std(self.confidence_changes) if self.confidence_changes else 0.0
        }


class AdversarialAttack(ABC):
    """Base class for adversarial attacks"""
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stats = AttackStats()
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def attack(self, model: nn.Module, inputs: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples"""
        pass
    
    def _project_perturbation(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Project perturbation to satisfy norm constraints"""
        if self.config.norm_type == NormType.L_INF:
            return torch.clamp(perturbation, -self.config.epsilon, self.config.epsilon)
        elif self.config.norm_type == NormType.L2:
            batch_size = perturbation.size(0)
            perturbation_flat = perturbation.view(batch_size, -1)
            norm = torch.norm(perturbation_flat, dim=1, keepdim=True)
            scaling_factor = torch.min(torch.ones_like(norm), self.config.epsilon / (norm + 1e-8))
            return perturbation * scaling_factor.view(batch_size, *([1] * (len(perturbation.shape) - 1)))
        elif self.config.norm_type == NormType.L1:
            batch_size = perturbation.size(0)
            perturbation_flat = perturbation.view(batch_size, -1)
            norm = torch.norm(perturbation_flat, p=1, dim=1, keepdim=True)
            scaling_factor = torch.min(torch.ones_like(norm), self.config.epsilon / (norm + 1e-8))
            return perturbation * scaling_factor.view(batch_size, *([1] * (len(perturbation.shape) - 1)))
        else:
            return perturbation
    
    def _clip_adversarial(self, adversarial: torch.Tensor, 
                         original: torch.Tensor) -> torch.Tensor:
        """Clip adversarial examples to valid range"""
        # Clip to valid pixel range
        adversarial = torch.clamp(adversarial, self.config.clip_min, self.config.clip_max)
        
        # Ensure perturbation satisfies norm constraint
        perturbation = adversarial - original
        perturbation = self._project_perturbation(perturbation)
        adversarial = original + perturbation
        
        # Final clipping
        return torch.clamp(adversarial, self.config.clip_min, self.config.clip_max)
    
    def _get_loss(self, model: nn.Module, inputs: torch.Tensor, 
                  targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss for attack optimization"""
        outputs = model(inputs)
        
        if self.config.loss_function == "ce":
            if self.config.targeted:
                return -F.cross_entropy(outputs, targets)
            else:
                return F.cross_entropy(outputs, targets)
        elif self.config.loss_function == "cw":
            # Carlini & Wagner loss
            target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
            other_logits = outputs.clone()
            other_logits.scatter_(1, targets.unsqueeze(1), -float('inf'))
            max_other_logits = other_logits.max(1)[0]
            
            if self.config.targeted:
                return torch.clamp(max_other_logits - target_logits + self.config.confidence, min=0)
            else:
                return torch.clamp(target_logits - max_other_logits + self.config.confidence, min=0)
        else:
            return F.cross_entropy(outputs, targets)


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method attack"""
    
    def attack(self, model: nn.Module, inputs: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        start_time = time.time()
        
        model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Enable gradient computation
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        loss = self._get_loss(model, inputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        grad_sign = inputs.grad.sign()
        perturbation = self.config.epsilon * grad_sign
        
        # Create adversarial examples
        adversarial = inputs + perturbation
        adversarial = self._clip_adversarial(adversarial, inputs)
        
        # Update statistics
        with torch.no_grad():
            original_preds = outputs.argmax(dim=1)
            adversarial_outputs = model(adversarial)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            original_confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
            adversarial_confidences = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
            
            self.stats.update(
                original_preds, adversarial_preds, adversarial - inputs,
                time.time() - start_time, original_confidences, adversarial_confidences
            )
        
        return adversarial.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent attack"""
    
    def attack(self, model: nn.Module, inputs: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        start_time = time.time()
        
        model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        best_adversarial = inputs.clone()
        best_loss = float('inf') if not self.config.targeted else float('-inf')
        
        for restart in range(self.config.random_restarts):
            # Initialize adversarial examples
            if self.config.random_start and restart > 0:
                # Random initialization within epsilon ball
                if self.config.norm_type == NormType.L_INF:
                    noise = torch.empty_like(inputs).uniform_(-self.config.epsilon, self.config.epsilon)
                elif self.config.norm_type == NormType.L2:
                    noise = torch.randn_like(inputs)
                    noise = noise / torch.norm(noise.view(inputs.size(0), -1), dim=1, keepdim=True).view(-1, *([1] * (len(inputs.shape) - 1)))
                    noise = noise * self.config.epsilon * torch.rand(inputs.size(0), device=self.device).view(-1, *([1] * (len(inputs.shape) - 1)))
                else:
                    noise = torch.zeros_like(inputs)
                
                adversarial = inputs + noise
                adversarial = self._clip_adversarial(adversarial, inputs)
            else:
                adversarial = inputs.clone()
            
            # Iterative attack
            for step in range(self.config.num_steps):
                adversarial.requires_grad_(True)
                
                # Forward pass
                outputs = model(adversarial)
                loss = self._get_loss(model, adversarial, targets)
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                
                # Update adversarial examples
                with torch.no_grad():
                    if self.config.norm_type == NormType.L_INF:
                        perturbation = self.config.step_size * adversarial.grad.sign()
                    elif self.config.norm_type == NormType.L2:
                        grad_norm = torch.norm(adversarial.grad.view(adversarial.size(0), -1), dim=1, keepdim=True)
                        perturbation = self.config.step_size * adversarial.grad / (grad_norm.view(-1, *([1] * (len(adversarial.shape) - 1))) + 1e-8)
                    else:
                        perturbation = self.config.step_size * adversarial.grad.sign()
                    
                    adversarial = adversarial + perturbation
                    adversarial = self._clip_adversarial(adversarial, inputs)
                
                adversarial = adversarial.detach()
            
            # Check if this restart produced better results
            with torch.no_grad():
                final_outputs = model(adversarial)
                final_loss = self._get_loss(model, adversarial, targets)
                
                if self.config.use_best_loss:
                    if (not self.config.targeted and final_loss < best_loss) or \
                       (self.config.targeted and final_loss > best_loss):
                        best_loss = final_loss
                        best_adversarial = adversarial.clone()
                else:
                    best_adversarial = adversarial.clone()
        
        # Update statistics
        with torch.no_grad():
            original_outputs = model(inputs)
            original_preds = original_outputs.argmax(dim=1)
            adversarial_outputs = model(best_adversarial)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            original_confidences = F.softmax(original_outputs, dim=1).max(dim=1)[0]
            adversarial_confidences = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
            
            self.stats.update(
                original_preds, adversarial_preds, best_adversarial - inputs,
                time.time() - start_time, original_confidences, adversarial_confidences
            )
        
        return best_adversarial


class CWAttack(AdversarialAttack):
    """Carlini & Wagner attack"""
    
    def attack(self, model: nn.Module, inputs: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """Generate C&W adversarial examples"""
        start_time = time.time()
        
        model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        batch_size = inputs.size(0)
        
        # Initialize variables for binary search
        c_lower = torch.zeros(batch_size, device=self.device)
        c_upper = torch.full((batch_size,), self.config.c_upper, device=self.device)
        c = torch.full((batch_size,), self.config.c_init, device=self.device)
        
        best_adversarial = inputs.clone()
        best_perturbation = torch.full_like(inputs, float('inf'))
        
        # Binary search for optimal c
        for binary_step in range(self.config.binary_search_steps):
            # Initialize perturbation variable
            w = torch.zeros_like(inputs, requires_grad=True)
            optimizer = torch.optim.Adam([w], lr=self.config.learning_rate)
            
            # Optimization loop
            for step in range(self.config.num_steps):
                optimizer.zero_grad()
                
                # Convert w to adversarial examples
                adversarial = 0.5 * (torch.tanh(w) + 1) * (self.config.clip_max - self.config.clip_min) + self.config.clip_min
                
                # Calculate losses
                outputs = model(adversarial)
                
                # L2 distance loss
                l2_loss = torch.norm((adversarial - inputs).view(batch_size, -1), dim=1)
                
                # Adversarial loss (C&W formulation)
                target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), -float('inf'))
                max_other_logits = other_logits.max(1)[0]
                
                if self.config.targeted:
                    adv_loss = torch.clamp(max_other_logits - target_logits + self.config.confidence, min=0)
                else:
                    adv_loss = torch.clamp(target_logits - max_other_logits + self.config.confidence, min=0)
                
                # Combined loss
                total_loss = l2_loss + c * adv_loss
                loss = total_loss.sum()
                
                loss.backward()
                optimizer.step()
            
            # Update best adversarial examples and adjust c
            with torch.no_grad():
                final_adversarial = 0.5 * (torch.tanh(w) + 1) * (self.config.clip_max - self.config.clip_min) + self.config.clip_min
                final_outputs = model(final_adversarial)
                final_preds = final_outputs.argmax(dim=1)
                
                # Check success and update c
                for i in range(batch_size):
                    current_perturbation = torch.norm((final_adversarial[i] - inputs[i]).view(-1))
                    
                    if self.config.targeted:
                        success = (final_preds[i] == targets[i])
                    else:
                        success = (final_preds[i] != targets[i])
                    
                    if success and current_perturbation < best_perturbation[i]:
                        best_adversarial[i] = final_adversarial[i]
                        best_perturbation[i] = current_perturbation
                        c_upper[i] = c[i]
                    else:
                        c_lower[i] = c[i]
                    
                    # Update c for next binary search step
                    if c_upper[i] < self.config.c_upper:
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                    else:
                        c[i] = c_lower[i] * 10
        
        # Update statistics
        with torch.no_grad():
            original_outputs = model(inputs)
            original_preds = original_outputs.argmax(dim=1)
            adversarial_outputs = model(best_adversarial)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            original_confidences = F.softmax(original_outputs, dim=1).max(dim=1)[0]
            adversarial_confidences = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
            
            self.stats.update(
                original_preds, adversarial_preds, best_adversarial - inputs,
                time.time() - start_time, original_confidences, adversarial_confidences
            )
        
        return best_adversarial


class AutoAttack(AdversarialAttack):
    """AutoAttack ensemble of attacks"""
    
    def __init__(self, config: AttackConfig):
        super().__init__(config)
        
        # Initialize individual attacks
        self.attacks = []
        
        if config.version == "standard":
            # Standard AutoAttack configuration
            attacks_config = [
                (AttackType.PGD, {"num_steps": 100, "step_size": config.epsilon / 4}),
                (AttackType.PGD, {"num_steps": 100, "step_size": config.epsilon / 4, "targeted": True}),
                (AttackType.CW, {"binary_search_steps": 9}),
                (AttackType.SQUARE, {"num_steps": 5000})
            ]
        else:
            # Custom configuration
            attacks_config = [
                (AttackType.PGD, {"num_steps": 50}),
                (AttackType.CW, {"binary_search_steps": 5})
            ]
        
        for attack_type, params in attacks_config:
            attack_config = AttackConfig(**{**config.__dict__, **params})
            attack_config.attack_type = attack_type
            
            if attack_type == AttackType.PGD:
                self.attacks.append(PGDAttack(attack_config))
            elif attack_type == AttackType.CW:
                self.attacks.append(CWAttack(attack_config))
            elif attack_type == AttackType.FGSM:
                self.attacks.append(FGSMAttack(attack_config))
    
    def attack(self, model: nn.Module, inputs: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """Generate AutoAttack adversarial examples"""
        start_time = time.time()
        
        model.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        best_adversarial = inputs.clone()
        
        # Track which samples are still correctly classified
        with torch.no_grad():
            current_outputs = model(inputs)
            current_preds = current_outputs.argmax(dim=1)
            remaining_mask = (current_preds == targets)
        
        # Apply attacks sequentially
        for i, attack in enumerate(self.attacks):
            if not remaining_mask.any():
                break  # All samples successfully attacked
            
            # Only attack remaining correctly classified samples
            remaining_inputs = inputs[remaining_mask]
            remaining_targets = targets[remaining_mask]
            
            if len(remaining_inputs) > 0:
                # Generate adversarial examples for remaining samples
                adversarial_remaining = attack.attack(model, remaining_inputs, remaining_targets)
                
                # Update best adversarial examples
                best_adversarial[remaining_mask] = adversarial_remaining
                
                # Update remaining mask
                with torch.no_grad():
                    new_outputs = model(adversarial_remaining)
                    new_preds = new_outputs.argmax(dim=1)
                    
                    # Update the mask for samples that are still correctly classified
                    remaining_indices = torch.where(remaining_mask)[0]
                    for j, pred in enumerate(new_preds):
                        if pred != remaining_targets[j]:
                            remaining_mask[remaining_indices[j]] = False
        
        # Update statistics
        with torch.no_grad():
            original_outputs = model(inputs)
            original_preds = original_outputs.argmax(dim=1)
            adversarial_outputs = model(best_adversarial)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            original_confidences = F.softmax(original_outputs, dim=1).max(dim=1)[0]
            adversarial_confidences = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]
            
            self.stats.update(
                original_preds, adversarial_preds, best_adversarial - inputs,
                time.time() - start_time, original_confidences, adversarial_confidences
            )
        
        return best_adversarial


class AttackEvaluator:
    """Evaluate and compare different adversarial attacks"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.attack_results = {}
    
    def evaluate_attack(self, attack: AdversarialAttack, model: nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       attack_name: str) -> Dict[str, Any]:
        """Evaluate a single attack on a dataset"""
        model.eval()
        attack.stats = AttackStats()  # Reset statistics
        
        total_samples = 0
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Generate adversarial examples
                start_time = time.time()
                adversarial = attack.attack(model, inputs, targets)
                attack_time = time.time() - start_time
                
                total_samples += inputs.size(0)
                total_time += attack_time
                
                if attack.config.verbose and batch_idx % 10 == 0:
                    self.logger.info(f"Attack {attack_name}: Batch {batch_idx}, "
                                   f"Success rate: {attack.stats.get_success_rate():.3f}")
        
        # Get final statistics
        stats = attack.stats.get_statistics()
        stats['total_time'] = total_time
        stats['samples_per_second'] = total_samples / total_time if total_time > 0 else 0
        
        self.attack_results[attack_name] = stats
        
        self.logger.info(f"Attack {attack_name} completed:")
        self.logger.info(f"  Success rate: {stats['success_rate']:.3f}")
        self.logger.info(f"  Average perturbation: {stats['average_perturbation']:.6f}")
        self.logger.info(f"  Average time per sample: {stats['average_time']:.4f}s")
        
        return stats
    
    def compare_attacks(self, attacks: Dict[str, AdversarialAttack], 
                       model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Compare multiple attacks"""
        comparison_results = {}
        
        for attack_name, attack in attacks.items():
            results = self.evaluate_attack(attack, model, data_loader, attack_name)
            comparison_results[attack_name] = results
        
        # Generate comparison summary
        summary = {
            'attack_comparison': comparison_results,
            'best_success_rate': max(comparison_results.items(), 
                                   key=lambda x: x[1]['success_rate']),
            'fastest_attack': min(comparison_results.items(), 
                                key=lambda x: x[1]['average_time']),
            'smallest_perturbation': min(comparison_results.items(), 
                                       key=lambda x: x[1]['average_perturbation'])
        }
        
        return summary
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics"""
        return {
            'individual_results': self.attack_results,
            'summary': {
                'total_attacks_evaluated': len(self.attack_results),
                'average_success_rate': np.mean([r['success_rate'] for r in self.attack_results.values()]),
                'average_perturbation': np.mean([r['average_perturbation'] for r in self.attack_results.values()]),
                'average_time': np.mean([r['average_time'] for r in self.attack_results.values()])
            }
        }


# Example usage
if __name__ == "__main__":
    # Create attack configurations
    fgsm_config = AttackConfig(
        attack_type=AttackType.FGSM,
        epsilon=8.0/255.0,
        norm_type=NormType.L_INF
    )
    
    pgd_config = AttackConfig(
        attack_type=AttackType.PGD,
        epsilon=8.0/255.0,
        step_size=2.0/255.0,
        num_steps=10,
        norm_type=NormType.L_INF,
        random_start=True
    )
    
    cw_config = AttackConfig(
        attack_type=AttackType.CW,
        norm_type=NormType.L2,
        confidence=0.0,
        binary_search_steps=9,
        num_steps=1000
    )
    
    # Create attacks
    fgsm_attack = FGSMAttack(fgsm_config)
    pgd_attack = PGDAttack(pgd_config)
    cw_attack = CWAttack(cw_config)
    
    print("Adversarial attacks created successfully!")
    print(f"FGSM config: {fgsm_config}")
    print(f"PGD config: {pgd_config}")
    print(f"C&W config: {cw_config}")