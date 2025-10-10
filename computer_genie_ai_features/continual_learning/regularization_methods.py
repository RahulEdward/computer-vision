"""
Regularization Methods for Continual Learning
निरंतर शिक्षा के लिए नियमितीकरण विधियां

Implementation of various regularization techniques to prevent catastrophic
forgetting in continual learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import defaultdict
from abc import ABC, abstractmethod


class RegularizationType(Enum):
    """Types of regularization methods"""
    EWC = "ewc"                      # Elastic Weight Consolidation
    SI = "si"                        # Synaptic Intelligence
    MAS = "mas"                      # Memory Aware Synapses
    LWF = "lwf"                      # Learning without Forgetting
    PACKNET = "packnet"              # PackNet
    HAT = "hat"                      # Hard Attention to Task
    PIGGYBACK = "piggyback"          # Piggyback
    SUPERMASK = "supermask"          # Supermask
    L2 = "l2"                        # L2 regularization
    DROPOUT = "dropout"              # Dropout regularization


@dataclass
class RegularizationConfig:
    """Configuration for regularization methods"""
    # General parameters
    regularization_type: RegularizationType = RegularizationType.EWC
    lambda_reg: float = 1000.0       # Regularization strength
    
    # EWC parameters
    ewc_alpha: float = 0.9           # EWC decay factor
    ewc_online: bool = True          # Online EWC
    fisher_estimation_samples: int = 1000
    
    # SI parameters
    si_c: float = 0.1                # SI regularization parameter
    si_xi: float = 0.1               # SI damping parameter
    
    # MAS parameters
    mas_alpha: float = 0.5           # MAS importance decay
    
    # LwF parameters
    lwf_temperature: float = 4.0     # Distillation temperature
    lwf_alpha: float = 1.0           # LwF loss weight
    
    # PackNet parameters
    packnet_pruning_ratio: float = 0.5
    packnet_retrain_epochs: int = 10
    
    # HAT parameters
    hat_smax: float = 400.0          # HAT attention sharpness
    hat_clipgrad: float = 10000.0    # HAT gradient clipping
    
    # Dropout parameters
    dropout_rate: float = 0.5
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True


class RegularizationMethod(ABC):
    """Abstract base class for regularization methods"""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Called before starting a new task"""
        pass
    
    @abstractmethod
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Called after completing a task"""
        pass
    
    @abstractmethod
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute regularization penalty"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get regularization statistics"""
        pass


class ElasticWeightConsolidation(RegularizationMethod):
    """Elastic Weight Consolidation (EWC) implementation"""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Initialize for new task"""
        if self.config.verbose:
            self.logger.info(f"EWC: Starting task {task_id}")
        
        # Store current parameters as optimal for previous tasks
        if task_id > 0:
            self.optimal_params[task_id - 1] = {}
            for name, param in model.named_parameters():
                self.optimal_params[task_id - 1][name] = param.data.clone()
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute Fisher Information Matrix after task completion"""
        if self.config.verbose:
            self.logger.info(f"EWC: Computing Fisher Information for task {task_id}")
        
        model.eval()
        fisher_dict = {}
        
        # Initialize Fisher Information
        for name, param in model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # Compute Fisher Information
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            if num_samples >= self.config.fisher_estimation_samples:
                break
            
            data, targets = data.to(self.device), targets.to(self.device)
            model.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher Information
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += data.size(0)
        
        # Normalize Fisher Information
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        # Store Fisher Information
        if self.config.ewc_online and task_id > 0:
            # Online EWC: combine with previous Fisher Information
            if task_id - 1 in self.fisher_information:
                for name in fisher_dict:
                    self.fisher_information[task_id][name] = (
                        self.config.ewc_alpha * self.fisher_information[task_id - 1][name] +
                        (1 - self.config.ewc_alpha) * fisher_dict[name]
                    )
            else:
                self.fisher_information[task_id] = fisher_dict
        else:
            # Standard EWC: store Fisher Information for this task
            self.fisher_information[task_id] = fisher_dict
        
        self.task_count = task_id + 1
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute EWC penalty"""
        penalty = torch.tensor(0.0, device=self.device)
        
        if task_id == 0:
            return penalty
        
        # Compute penalty for all previous tasks
        for prev_task_id in range(task_id):
            if prev_task_id in self.fisher_information and prev_task_id in self.optimal_params:
                for name, param in model.named_parameters():
                    if name in self.fisher_information[prev_task_id] and name in self.optimal_params[prev_task_id]:
                        fisher = self.fisher_information[prev_task_id][name]
                        optimal = self.optimal_params[prev_task_id][name]
                        penalty += (fisher * (param - optimal) ** 2).sum()
        
        return self.config.lambda_reg * penalty
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics"""
        stats = {
            'method': 'EWC',
            'task_count': self.task_count,
            'fisher_tasks': len(self.fisher_information),
            'optimal_param_tasks': len(self.optimal_params),
            'lambda_reg': self.config.lambda_reg,
            'online_ewc': self.config.ewc_online
        }
        
        # Fisher Information statistics
        if self.fisher_information:
            fisher_means = []
            fisher_stds = []
            for task_id, fisher_dict in self.fisher_information.items():
                task_fishers = []
                for name, fisher in fisher_dict.items():
                    task_fishers.append(fisher.mean().item())
                fisher_means.append(np.mean(task_fishers))
                fisher_stds.append(np.std(task_fishers))
            
            stats['fisher_mean'] = np.mean(fisher_means)
            stats['fisher_std'] = np.mean(fisher_stds)
        
        return stats


class SynapticIntelligence(RegularizationMethod):
    """Synaptic Intelligence (SI) implementation"""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.omega = {}
        self.W = {}
        self.prev_params = {}
        self.task_count = 0
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Initialize for new task"""
        if self.config.verbose:
            self.logger.info(f"SI: Starting task {task_id}")
        
        # Initialize omega and W for new parameters
        for name, param in model.named_parameters():
            if name not in self.omega:
                self.omega[name] = torch.zeros_like(param.data)
                self.W[name] = torch.zeros_like(param.data)
            
            # Store previous parameters
            self.prev_params[name] = param.data.clone()
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Update importance weights after task completion"""
        if self.config.verbose:
            self.logger.info(f"SI: Updating importance weights for task {task_id}")
        
        # Update omega (importance weights)
        for name, param in model.named_parameters():
            if name in self.W and name in self.prev_params:
                # Compute parameter change
                param_change = param.data - self.prev_params[name]
                
                # Update omega
                self.omega[name] += self.W[name] / (param_change ** 2 + self.config.si_xi)
                
                # Reset W for next task
                self.W[name].zero_()
        
        self.task_count = task_id + 1
    
    def update_importance(self, model: nn.Module) -> None:
        """Update importance weights during training"""
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.W:
                # Accumulate gradients
                self.W[name] -= param.grad.data * (param.data - self.prev_params[name])
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute SI penalty"""
        penalty = torch.tensor(0.0, device=self.device)
        
        if task_id == 0:
            return penalty
        
        # Compute penalty based on importance weights
        for name, param in model.named_parameters():
            if name in self.omega and name in self.prev_params:
                penalty += (self.omega[name] * (param - self.prev_params[name]) ** 2).sum()
        
        return self.config.lambda_reg * penalty
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SI statistics"""
        stats = {
            'method': 'SI',
            'task_count': self.task_count,
            'lambda_reg': self.config.lambda_reg,
            'si_c': self.config.si_c,
            'si_xi': self.config.si_xi
        }
        
        # Omega statistics
        if self.omega:
            omega_values = []
            for name, omega in self.omega.items():
                omega_values.extend(omega.flatten().tolist())
            
            stats['omega_mean'] = np.mean(omega_values)
            stats['omega_std'] = np.std(omega_values)
            stats['omega_max'] = np.max(omega_values)
            stats['omega_min'] = np.min(omega_values)
        
        return stats


class MemoryAwareSynapses(RegularizationMethod):
    """Memory Aware Synapses (MAS) implementation"""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.importance = {}
        self.optimal_params = {}
        self.task_count = 0
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Initialize for new task"""
        if self.config.verbose:
            self.logger.info(f"MAS: Starting task {task_id}")
        
        # Store optimal parameters from previous task
        if task_id > 0:
            self.optimal_params[task_id - 1] = {}
            for name, param in model.named_parameters():
                self.optimal_params[task_id - 1][name] = param.data.clone()
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute importance weights after task completion"""
        if self.config.verbose:
            self.logger.info(f"MAS: Computing importance weights for task {task_id}")
        
        model.eval()
        importance_dict = {}
        
        # Initialize importance weights
        for name, param in model.named_parameters():
            importance_dict[name] = torch.zeros_like(param.data)
        
        # Compute importance based on gradients
        num_samples = 0
        for data, _ in dataloader:
            data = data.to(self.device)
            model.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Compute gradients w.r.t. outputs (unsupervised)
            for i in range(outputs.size(1)):
                model.zero_grad()
                outputs[:, i].sum().backward(retain_graph=True)
                
                # Accumulate importance
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        importance_dict[name] += param.grad.data.abs()
            
            num_samples += data.size(0)
        
        # Normalize importance
        for name in importance_dict:
            importance_dict[name] /= num_samples
        
        # Update importance with decay
        if task_id == 0:
            self.importance = importance_dict
        else:
            for name in importance_dict:
                if name in self.importance:
                    self.importance[name] = (
                        self.config.mas_alpha * self.importance[name] +
                        (1 - self.config.mas_alpha) * importance_dict[name]
                    )
                else:
                    self.importance[name] = importance_dict[name]
        
        self.task_count = task_id + 1
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute MAS penalty"""
        penalty = torch.tensor(0.0, device=self.device)
        
        if task_id == 0:
            return penalty
        
        # Compute penalty for all previous tasks
        for prev_task_id in range(task_id):
            if prev_task_id in self.optimal_params:
                for name, param in model.named_parameters():
                    if name in self.importance and name in self.optimal_params[prev_task_id]:
                        importance = self.importance[name]
                        optimal = self.optimal_params[prev_task_id][name]
                        penalty += (importance * (param - optimal) ** 2).sum()
        
        return self.config.lambda_reg * penalty
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MAS statistics"""
        stats = {
            'method': 'MAS',
            'task_count': self.task_count,
            'lambda_reg': self.config.lambda_reg,
            'mas_alpha': self.config.mas_alpha
        }
        
        # Importance statistics
        if self.importance:
            importance_values = []
            for name, importance in self.importance.items():
                importance_values.extend(importance.flatten().tolist())
            
            stats['importance_mean'] = np.mean(importance_values)
            stats['importance_std'] = np.std(importance_values)
            stats['importance_max'] = np.max(importance_values)
            stats['importance_min'] = np.min(importance_values)
        
        return stats


class LearningWithoutForgetting(RegularizationMethod):
    """Learning without Forgetting (LwF) implementation"""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.previous_models = {}
        self.task_count = 0
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Store previous model before starting new task"""
        if self.config.verbose:
            self.logger.info(f"LwF: Starting task {task_id}")
        
        if task_id > 0:
            # Store previous model
            import copy
            self.previous_models[task_id - 1] = copy.deepcopy(model)
            self.previous_models[task_id - 1].eval()
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Nothing to do after task for LwF"""
        self.task_count = task_id + 1
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute LwF distillation penalty"""
        penalty = torch.tensor(0.0, device=self.device)
        
        if task_id == 0 or not self.previous_models:
            return penalty
        
        # This method should be called during training with current batch
        # For now, return zero as penalty is computed in training loop
        return penalty
    
    def compute_distillation_loss(self, model: nn.Module, data: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute distillation loss for LwF"""
        if task_id == 0 or not self.previous_models:
            return torch.tensor(0.0, device=self.device)
        
        model.eval()
        distillation_loss = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # Get predictions from previous models
            for prev_task_id, prev_model in self.previous_models.items():
                prev_model.eval()
                prev_outputs = prev_model(data)
                
                # Current model predictions
                model.train()
                current_outputs = model(data)
                
                # Compute distillation loss
                prev_probs = F.softmax(prev_outputs / self.config.lwf_temperature, dim=1)
                current_log_probs = F.log_softmax(current_outputs / self.config.lwf_temperature, dim=1)
                
                kl_loss = F.kl_div(current_log_probs, prev_probs, reduction='batchmean')
                distillation_loss += kl_loss * (self.config.lwf_temperature ** 2)
        
        return self.config.lwf_alpha * distillation_loss
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get LwF statistics"""
        return {
            'method': 'LwF',
            'task_count': self.task_count,
            'previous_models': len(self.previous_models),
            'lwf_temperature': self.config.lwf_temperature,
            'lwf_alpha': self.config.lwf_alpha
        }


class L2Regularization(RegularizationMethod):
    """L2 Regularization implementation"""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.task_count = 0
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Nothing to do before task for L2"""
        pass
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Nothing to do after task for L2"""
        self.task_count = task_id + 1
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute L2 penalty"""
        penalty = torch.tensor(0.0, device=self.device)
        
        for param in model.parameters():
            penalty += torch.norm(param, p=2) ** 2
        
        return self.config.lambda_reg * penalty
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get L2 statistics"""
        return {
            'method': 'L2',
            'task_count': self.task_count,
            'lambda_reg': self.config.lambda_reg
        }


class RegularizationManager:
    """Manager for different regularization methods"""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize regularization method
        if config.regularization_type == RegularizationType.EWC:
            self.method = ElasticWeightConsolidation(config)
        elif config.regularization_type == RegularizationType.SI:
            self.method = SynapticIntelligence(config)
        elif config.regularization_type == RegularizationType.MAS:
            self.method = MemoryAwareSynapses(config)
        elif config.regularization_type == RegularizationType.LWF:
            self.method = LearningWithoutForgetting(config)
        elif config.regularization_type == RegularizationType.L2:
            self.method = L2Regularization(config)
        else:
            raise ValueError(f"Unsupported regularization type: {config.regularization_type}")
        
        self.total_penalties = 0
        self.penalty_history = []
    
    def before_task(self, model: nn.Module, task_id: int) -> None:
        """Called before starting a new task"""
        self.method.before_task(model, task_id)
    
    def after_task(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Called after completing a task"""
        self.method.after_task(model, task_id, dataloader)
    
    def compute_penalty(self, model: nn.Module, task_id: int) -> torch.Tensor:
        """Compute regularization penalty"""
        penalty = self.method.compute_penalty(model, task_id)
        
        self.total_penalties += 1
        self.penalty_history.append(penalty.item())
        
        return penalty
    
    def update_importance(self, model: nn.Module) -> None:
        """Update importance weights (for SI)"""
        if hasattr(self.method, 'update_importance'):
            self.method.update_importance(model)
    
    def compute_distillation_loss(self, model: nn.Module, data: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute distillation loss (for LwF)"""
        if hasattr(self.method, 'compute_distillation_loss'):
            return self.method.compute_distillation_loss(model, data, task_id)
        return torch.tensor(0.0, device=self.config.device)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regularization statistics"""
        method_stats = self.method.get_statistics()
        
        manager_stats = {
            'total_penalties_computed': self.total_penalties,
            'penalty_history_length': len(self.penalty_history),
            'regularization_type': self.config.regularization_type.value
        }
        
        if self.penalty_history:
            manager_stats.update({
                'avg_penalty': np.mean(self.penalty_history),
                'std_penalty': np.std(self.penalty_history),
                'max_penalty': np.max(self.penalty_history),
                'min_penalty': np.min(self.penalty_history)
            })
        
        return {**method_stats, **manager_stats}


# Example usage
if __name__ == "__main__":
    # Create regularization configuration
    config = RegularizationConfig(
        regularization_type=RegularizationType.EWC,
        lambda_reg=1000.0,
        ewc_online=True
    )
    
    # Create regularization manager
    reg_manager = RegularizationManager(config)
    
    print("Regularization manager created successfully!")
    print(f"Method: {config.regularization_type.value}")
    print(f"Lambda: {config.lambda_reg}")