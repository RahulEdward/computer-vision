"""
Meta-Learning for Continual Learning
निरंतर शिक्षा के लिए मेटा-लर्निंग

Implementation of meta-learning approaches for continual learning including
MAML, Reptile, and other gradient-based meta-learning methods.
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


class MetaLearningMethod(Enum):
    """Meta-learning methods for continual learning"""
    MAML = "maml"                    # Model-Agnostic Meta-Learning
    REPTILE = "reptile"              # Reptile
    FOMAML = "fomaml"               # First-Order MAML
    ANIL = "anil"                   # Almost No Inner Loop
    METASGD = "metasgd"             # Meta-SGD
    LEARNINGTOLEARNBYGD = "l2l_gd"  # Learning to Learn by Gradient Descent
    PROTOTYPICAL = "prototypical"    # Prototypical Networks
    MATCHING = "matching"            # Matching Networks


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning methods"""
    # General parameters
    method: MetaLearningMethod = MetaLearningMethod.MAML
    
    # MAML parameters
    inner_lr: float = 0.01           # Inner loop learning rate
    outer_lr: float = 0.001          # Outer loop learning rate
    inner_steps: int = 5             # Number of inner loop steps
    first_order: bool = False        # Use first-order approximation
    
    # Reptile parameters
    reptile_lr: float = 0.001        # Reptile learning rate
    reptile_inner_steps: int = 10    # Inner steps for Reptile
    
    # Meta-SGD parameters
    meta_sgd_lr: float = 0.001       # Meta-SGD learning rate
    learn_inner_lr: bool = True      # Learn inner learning rates
    
    # Prototypical Networks parameters
    proto_distance: str = "euclidean"  # "euclidean", "cosine"
    proto_temperature: float = 1.0
    
    # Task sampling parameters
    n_way: int = 5                   # Number of classes per task
    k_shot: int = 5                  # Number of examples per class
    query_shots: int = 15            # Number of query examples per class
    
    # Training parameters
    meta_batch_size: int = 4         # Number of tasks per meta-batch
    adaptation_steps: int = 5        # Steps for task adaptation
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True


class MetaLearner(ABC):
    """Abstract base class for meta-learning methods"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Statistics
        self.meta_updates = 0
        self.task_adaptations = 0
    
    @abstractmethod
    def meta_train_step(self, tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform one meta-training step"""
        pass
    
    @abstractmethod
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: Optional[int] = None) -> nn.Module:
        """Adapt model to new task"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        pass


class MAML(MetaLearner):
    """Model-Agnostic Meta-Learning implementation"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__(model, config)
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.outer_lr
        )
        
        # Track gradients
        self.meta_gradients = []
    
    def meta_train_step(self, tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform MAML meta-training step"""
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        task_losses = []
        
        for task in tasks:
            # Extract task data
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            query_x = task['query_x'].to(self.device)
            query_y = task['query_y'].to(self.device)
            
            # Inner loop adaptation
            adapted_model = self._inner_loop_adaptation(support_x, support_y)
            
            # Compute query loss with adapted model
            query_pred = adapted_model(query_x)
            query_loss = F.cross_entropy(query_pred, query_y)
            
            total_loss += query_loss
            task_losses.append(query_loss.item())
            
            self.task_adaptations += 1
        
        # Meta-update
        avg_loss = total_loss / len(tasks)
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        self.meta_updates += 1
        
        return {
            'meta_loss': avg_loss.item(),
            'task_losses': task_losses,
            'avg_task_loss': np.mean(task_losses)
        }
    
    def _inner_loop_adaptation(self, support_x: torch.Tensor, 
                              support_y: torch.Tensor) -> nn.Module:
        """Perform inner loop adaptation"""
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        
        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_lr
        )
        
        # Inner loop updates
        for step in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass
            support_pred = adapted_model(support_x)
            support_loss = F.cross_entropy(support_pred, support_y)
            
            # Backward pass
            support_loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: Optional[int] = None) -> nn.Module:
        """Adapt model to new task using MAML"""
        if num_steps is None:
            num_steps = self.config.adaptation_steps
        
        # Create adapted model
        adapted_model = copy.deepcopy(self.model)
        
        # Adaptation optimizer
        optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_lr
        )
        
        # Adaptation steps
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            pred = adapted_model(support_data)
            loss = F.cross_entropy(pred, support_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MAML statistics"""
        return {
            'method': 'MAML',
            'meta_updates': self.meta_updates,
            'task_adaptations': self.task_adaptations,
            'inner_lr': self.config.inner_lr,
            'outer_lr': self.config.outer_lr,
            'inner_steps': self.config.inner_steps,
            'first_order': self.config.first_order
        }


class Reptile(MetaLearner):
    """Reptile meta-learning implementation"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__(model, config)
        
        # Store initial parameters
        self.initial_params = {name: param.clone() 
                              for name, param in self.model.named_parameters()}
    
    def meta_train_step(self, tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform Reptile meta-training step"""
        task_losses = []
        updated_params = []
        
        for task in tasks:
            # Extract task data
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            
            # Create task-specific model
            task_model = copy.deepcopy(self.model)
            task_optimizer = torch.optim.SGD(
                task_model.parameters(), 
                lr=self.config.inner_lr
            )
            
            # Task-specific training
            for step in range(self.config.reptile_inner_steps):
                task_optimizer.zero_grad()
                
                pred = task_model(support_x)
                loss = F.cross_entropy(pred, support_y)
                
                loss.backward()
                task_optimizer.step()
            
            # Store updated parameters
            task_params = {name: param.clone() 
                          for name, param in task_model.named_parameters()}
            updated_params.append(task_params)
            task_losses.append(loss.item())
            
            self.task_adaptations += 1
        
        # Reptile meta-update
        self._reptile_meta_update(updated_params)
        self.meta_updates += 1
        
        return {
            'meta_loss': np.mean(task_losses),
            'task_losses': task_losses,
            'avg_task_loss': np.mean(task_losses)
        }
    
    def _reptile_meta_update(self, updated_params_list: List[Dict[str, torch.Tensor]]):
        """Perform Reptile meta-update"""
        # Compute average of updated parameters
        avg_params = {}
        
        for name, param in self.model.named_parameters():
            avg_update = torch.zeros_like(param.data)
            
            for updated_params in updated_params_list:
                avg_update += updated_params[name] - param.data
            
            avg_update /= len(updated_params_list)
            
            # Update model parameters
            param.data += self.config.reptile_lr * avg_update
    
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: Optional[int] = None) -> nn.Module:
        """Adapt model to new task using Reptile"""
        if num_steps is None:
            num_steps = self.config.adaptation_steps
        
        # Create adapted model
        adapted_model = copy.deepcopy(self.model)
        
        # Adaptation optimizer
        optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_lr
        )
        
        # Adaptation steps
        for step in range(num_steps):
            optimizer.zero_grad()
            
            pred = adapted_model(support_data)
            loss = F.cross_entropy(pred, support_labels)
            
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Reptile statistics"""
        return {
            'method': 'Reptile',
            'meta_updates': self.meta_updates,
            'task_adaptations': self.task_adaptations,
            'reptile_lr': self.config.reptile_lr,
            'inner_steps': self.config.reptile_inner_steps
        }


class PrototypicalNetworks(MetaLearner):
    """Prototypical Networks implementation"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__(model, config)
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.outer_lr
        )
        
        # Feature extractor (assume model outputs features)
        self.feature_extractor = model
    
    def meta_train_step(self, tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform Prototypical Networks meta-training step"""
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        task_losses = []
        
        for task in tasks:
            # Extract task data
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            query_x = task['query_x'].to(self.device)
            query_y = task['query_y'].to(self.device)
            
            # Compute prototypes
            prototypes = self._compute_prototypes(support_x, support_y)
            
            # Compute query loss
            query_loss = self._compute_prototypical_loss(query_x, query_y, prototypes)
            
            total_loss += query_loss
            task_losses.append(query_loss.item())
            
            self.task_adaptations += 1
        
        # Meta-update
        avg_loss = total_loss / len(tasks)
        avg_loss.backward()
        
        self.meta_optimizer.step()
        self.meta_updates += 1
        
        return {
            'meta_loss': avg_loss.item(),
            'task_losses': task_losses,
            'avg_task_loss': np.mean(task_losses)
        }
    
    def _compute_prototypes(self, support_x: torch.Tensor, 
                           support_y: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set"""
        # Extract features
        features = self.feature_extractor(support_x)
        
        # Compute prototypes for each class
        unique_classes = torch.unique(support_y)
        prototypes = []
        
        for class_id in unique_classes:
            class_mask = (support_y == class_id)
            class_features = features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_prototypical_loss(self, query_x: torch.Tensor, 
                                  query_y: torch.Tensor, 
                                  prototypes: torch.Tensor) -> torch.Tensor:
        """Compute prototypical loss"""
        # Extract query features
        query_features = self.feature_extractor(query_x)
        
        # Compute distances to prototypes
        if self.config.proto_distance == "euclidean":
            distances = torch.cdist(query_features, prototypes)
        elif self.config.proto_distance == "cosine":
            # Normalize features
            query_norm = F.normalize(query_features, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())
        else:
            distances = torch.cdist(query_features, prototypes)
        
        # Convert distances to logits
        logits = -distances / self.config.proto_temperature
        
        # Compute cross-entropy loss
        return F.cross_entropy(logits, query_y)
    
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: Optional[int] = None) -> nn.Module:
        """Adapt to task by computing prototypes"""
        # For prototypical networks, adaptation is just computing prototypes
        prototypes = self._compute_prototypes(support_data, support_labels)
        
        # Create a wrapper model that uses prototypes for prediction
        class PrototypicalPredictor(nn.Module):
            def __init__(self, feature_extractor, prototypes, config):
                super().__init__()
                self.feature_extractor = feature_extractor
                self.prototypes = prototypes
                self.config = config
            
            def forward(self, x):
                features = self.feature_extractor(x)
                
                if self.config.proto_distance == "euclidean":
                    distances = torch.cdist(features, self.prototypes)
                elif self.config.proto_distance == "cosine":
                    features_norm = F.normalize(features, dim=1)
                    proto_norm = F.normalize(self.prototypes, dim=1)
                    distances = 1 - torch.mm(features_norm, proto_norm.t())
                else:
                    distances = torch.cdist(features, self.prototypes)
                
                logits = -distances / self.config.proto_temperature
                return logits
        
        return PrototypicalPredictor(self.feature_extractor, prototypes, self.config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Prototypical Networks statistics"""
        return {
            'method': 'Prototypical Networks',
            'meta_updates': self.meta_updates,
            'task_adaptations': self.task_adaptations,
            'proto_distance': self.config.proto_distance,
            'proto_temperature': self.config.proto_temperature
        }


class MetaLearningManager:
    """Manager for meta-learning approaches in continual learning"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize meta-learning method
        if config.method == MetaLearningMethod.MAML:
            self.meta_learner = MAML(model, config)
        elif config.method == MetaLearningMethod.REPTILE:
            self.meta_learner = Reptile(model, config)
        elif config.method == MetaLearningMethod.PROTOTYPICAL:
            self.meta_learner = PrototypicalNetworks(model, config)
        else:
            raise ValueError(f"Unsupported meta-learning method: {config.method}")
        
        # Training statistics
        self.training_history = []
        self.adaptation_history = []
    
    def meta_train(self, task_generator: Callable, num_iterations: int) -> Dict[str, List[float]]:
        """Meta-train the model"""
        if self.config.verbose:
            self.logger.info(f"Starting meta-training for {num_iterations} iterations")
        
        meta_losses = []
        task_losses = []
        
        for iteration in range(num_iterations):
            # Generate meta-batch of tasks
            tasks = []
            for _ in range(self.config.meta_batch_size):
                task = task_generator()
                tasks.append(task)
            
            # Meta-training step
            step_results = self.meta_learner.meta_train_step(tasks)
            
            meta_losses.append(step_results['meta_loss'])
            task_losses.extend(step_results['task_losses'])
            
            # Log progress
            if self.config.verbose and (iteration + 1) % 100 == 0:
                self.logger.info(f"Iteration {iteration + 1}/{num_iterations}, "
                               f"Meta Loss: {step_results['meta_loss']:.4f}")
        
        # Store training history
        self.training_history.append({
            'meta_losses': meta_losses,
            'task_losses': task_losses,
            'num_iterations': num_iterations
        })
        
        return {
            'meta_losses': meta_losses,
            'task_losses': task_losses
        }
    
    def adapt_to_new_task(self, support_data: torch.Tensor, 
                         support_labels: torch.Tensor,
                         num_steps: Optional[int] = None) -> nn.Module:
        """Adapt model to new task"""
        if self.config.verbose:
            self.logger.info("Adapting to new task")
        
        adapted_model = self.meta_learner.adapt_to_task(
            support_data, support_labels, num_steps
        )
        
        # Store adaptation history
        self.adaptation_history.append({
            'support_size': support_data.size(0),
            'num_classes': len(torch.unique(support_labels)),
            'adaptation_steps': num_steps or self.config.adaptation_steps
        })
        
        return adapted_model
    
    def evaluate_adaptation(self, adapted_model: nn.Module, 
                           query_data: torch.Tensor, 
                           query_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate adapted model on query set"""
        adapted_model.eval()
        
        with torch.no_grad():
            query_pred = adapted_model(query_data)
            query_loss = F.cross_entropy(query_pred, query_labels)
            
            # Compute accuracy
            _, predicted = torch.max(query_pred, 1)
            accuracy = (predicted == query_labels).float().mean()
        
        return {
            'query_loss': query_loss.item(),
            'query_accuracy': accuracy.item()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics"""
        meta_stats = self.meta_learner.get_statistics()
        
        manager_stats = {
            'training_sessions': len(self.training_history),
            'total_adaptations': len(self.adaptation_history),
            'config': {
                'method': self.config.method.value,
                'n_way': self.config.n_way,
                'k_shot': self.config.k_shot,
                'meta_batch_size': self.config.meta_batch_size
            }
        }
        
        # Training history statistics
        if self.training_history:
            all_meta_losses = []
            for session in self.training_history:
                all_meta_losses.extend(session['meta_losses'])
            
            manager_stats.update({
                'avg_meta_loss': np.mean(all_meta_losses),
                'std_meta_loss': np.std(all_meta_losses),
                'min_meta_loss': np.min(all_meta_losses),
                'max_meta_loss': np.max(all_meta_losses)
            })
        
        # Adaptation history statistics
        if self.adaptation_history:
            adaptation_steps = [adapt['adaptation_steps'] for adapt in self.adaptation_history]
            manager_stats.update({
                'avg_adaptation_steps': np.mean(adaptation_steps),
                'total_adaptation_steps': sum(adaptation_steps)
            })
        
        return {**meta_stats, **manager_stats}


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)  # Feature extractor
    )
    
    # Create meta-learning configuration
    config = MetaLearningConfig(
        method=MetaLearningMethod.MAML,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        n_way=5,
        k_shot=5
    )
    
    # Create meta-learning manager
    meta_manager = MetaLearningManager(model, config)
    
    print("Meta-learning manager created successfully!")
    print(f"Method: {config.method.value}")
    print(f"N-way: {config.n_way}, K-shot: {config.k_shot}")