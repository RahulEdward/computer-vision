"""
Continual Learning Trainer
निरंतर शिक्षा प्रशिक्षक

Main trainer class for continual learning scenarios with support for
various learning strategies and memory management approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import json
import copy
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns


class LearningStrategy(Enum):
    """Continual learning strategies"""
    NAIVE = "naive"                    # Standard fine-tuning
    EWC = "ewc"                       # Elastic Weight Consolidation
    LWF = "lwf"                       # Learning without Forgetting
    PACKNET = "packnet"               # PackNet pruning
    PROGRESSIVE = "progressive"        # Progressive Neural Networks
    GEM = "gem"                       # Gradient Episodic Memory
    AGEM = "agem"                     # Averaged GEM
    MAS = "mas"                       # Memory Aware Synapses
    SI = "si"                         # Synaptic Intelligence
    REPLAY = "replay"                 # Experience Replay
    ICARL = "icarl"                   # iCaRL
    MAML = "maml"                     # Model-Agnostic Meta-Learning
    REPTILE = "reptile"               # Reptile meta-learning


class MemoryStrategy(Enum):
    """Memory management strategies"""
    NONE = "none"
    RANDOM = "random"
    HERDING = "herding"
    GRADIENT_BASED = "gradient_based"
    UNCERTAINTY_BASED = "uncertainty_based"
    DIVERSITY_BASED = "diversity_based"
    PROTOTYPE_BASED = "prototype_based"


class TaskType(Enum):
    """Types of continual learning tasks"""
    TASK_INCREMENTAL = "task_incremental"      # New tasks with task labels
    CLASS_INCREMENTAL = "class_incremental"    # New classes without task labels
    DOMAIN_INCREMENTAL = "domain_incremental"  # New domains, same classes


@dataclass
class TaskInfo:
    """Information about a learning task"""
    task_id: int
    task_name: str
    num_classes: int
    class_names: List[str]
    task_type: TaskType
    
    # Data information
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    
    # Task-specific parameters
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    
    # Metadata
    description: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ContinualConfig:
    """Configuration for continual learning"""
    # Learning strategy
    strategy: LearningStrategy = LearningStrategy.EWC
    memory_strategy: MemoryStrategy = MemoryStrategy.RANDOM
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    
    # Memory parameters
    memory_size: int = 1000
    memory_per_task: Optional[int] = None
    
    # Strategy-specific parameters
    ewc_lambda: float = 1000.0        # EWC regularization strength
    lwf_alpha: float = 1.0            # LwF distillation weight
    lwf_temperature: float = 4.0      # LwF temperature
    gem_margin: float = 0.5           # GEM margin
    mas_lambda: float = 1.0           # MAS regularization strength
    si_c: float = 0.1                 # SI regularization strength
    
    # Progressive networks
    progressive_columns: int = 3
    progressive_lateral_connections: bool = True
    
    # Meta-learning parameters
    meta_lr: float = 0.01
    meta_batch_size: int = 4
    meta_inner_steps: int = 5
    
    # Evaluation parameters
    eval_frequency: int = 1
    save_checkpoints: bool = True
    checkpoint_dir: str = "./continual_checkpoints"
    
    # Device and performance
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Logging
    verbose: bool = True
    log_dir: str = "./continual_logs"


@dataclass
class TrainingStats:
    """Statistics tracking for continual learning"""
    task_accuracies: Dict[int, List[float]] = field(default_factory=dict)
    task_losses: Dict[int, List[float]] = field(default_factory=dict)
    forgetting_measures: Dict[int, float] = field(default_factory=dict)
    transfer_measures: Dict[int, float] = field(default_factory=dict)
    
    # Memory statistics
    memory_usage: List[int] = field(default_factory=list)
    memory_diversity: List[float] = field(default_factory=list)
    
    # Training time
    task_training_times: Dict[int, float] = field(default_factory=dict)
    total_training_time: float = 0.0
    
    # Model statistics
    model_parameters: List[int] = field(default_factory=list)
    model_size_mb: List[float] = field(default_factory=list)


class ContinualTrainer:
    """Main continual learning trainer"""
    
    def __init__(self, model: nn.Module, config: ContinualConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Task management
        self.tasks: List[TaskInfo] = []
        self.current_task_id = -1
        
        # Memory management
        self.memory_buffer = {}
        self.memory_labels = {}
        
        # Strategy-specific components
        self.fisher_information = {}
        self.optimal_params = {}
        self.importance_weights = {}
        self.old_model = None
        
        # Statistics tracking
        self.stats = TrainingStats()
        
        # Initialize strategy-specific components
        self._initialize_strategy()
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        if self.config.strategy == LearningStrategy.PROGRESSIVE:
            self._initialize_progressive_networks()
        elif self.config.strategy in [LearningStrategy.MAML, LearningStrategy.REPTILE]:
            self._initialize_meta_learning()
    
    def _initialize_progressive_networks(self):
        """Initialize progressive neural networks"""
        # Store original model as first column
        self.progressive_columns = [copy.deepcopy(self.model)]
        self.lateral_connections = {}
    
    def _initialize_meta_learning(self):
        """Initialize meta-learning components"""
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.config.meta_lr)
        self.task_distributions = []
    
    def add_task(self, task_info: TaskInfo, train_loader, val_loader=None, test_loader=None):
        """Add a new task to the continual learning sequence"""
        self.logger.info(f"Adding task {task_info.task_id}: {task_info.task_name}")
        
        # Update task information
        task_info.train_size = len(train_loader.dataset) if train_loader else 0
        task_info.val_size = len(val_loader.dataset) if val_loader else 0
        task_info.test_size = len(test_loader.dataset) if test_loader else 0
        
        self.tasks.append(task_info)
        self.current_task_id = task_info.task_id
        
        # Initialize task-specific statistics
        self.stats.task_accuracies[task_info.task_id] = []
        self.stats.task_losses[task_info.task_id] = []
        
        # Train on the new task
        start_time = time.time()
        self._train_task(task_info, train_loader, val_loader)
        training_time = time.time() - start_time
        
        self.stats.task_training_times[task_info.task_id] = training_time
        self.stats.total_training_time += training_time
        
        # Update memory buffer
        if self.config.memory_strategy != MemoryStrategy.NONE:
            self._update_memory(task_info, train_loader)
        
        # Evaluate on all previous tasks
        if test_loader:
            self._evaluate_all_tasks(test_loader)
        
        # Save checkpoint
        if self.config.save_checkpoints:
            self._save_checkpoint(task_info.task_id)
        
        self.logger.info(f"Task {task_info.task_id} training completed in {training_time:.2f}s")
    
    def _train_task(self, task_info: TaskInfo, train_loader, val_loader=None):
        """Train on a specific task"""
        self.logger.info(f"Training on task {task_info.task_id}")
        
        # Setup optimizer
        optimizer = self._setup_optimizer(task_info)
        scheduler = self._setup_scheduler(optimizer, task_info)
        
        # Strategy-specific preparation
        if self.config.strategy == LearningStrategy.EWC and self.current_task_id > 0:
            self._compute_fisher_information(train_loader)
        elif self.config.strategy == LearningStrategy.LWF and self.current_task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
        elif self.config.strategy == LearningStrategy.PROGRESSIVE:
            self._add_progressive_column(task_info)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(train_loader, optimizer, task_info)
            
            # Validation
            if val_loader and epoch % self.config.eval_frequency == 0:
                val_acc = self._evaluate_task(val_loader, task_info.task_id)
                self.stats.task_accuracies[task_info.task_id].append(val_acc)
            
            self.stats.task_losses[task_info.task_id].append(epoch_loss)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            if self.config.verbose:
                self.logger.info(
                    f"Task {task_info.task_id}, Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Loss: {epoch_loss:.4f}"
                )
    
    def _train_epoch(self, train_loader, optimizer, task_info: TaskInfo) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(inputs, targets, task_info)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(inputs, targets, task_info)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Memory replay
            if self.config.strategy in [LearningStrategy.REPLAY, LearningStrategy.GEM, LearningStrategy.AGEM]:
                if self.memory_buffer and len(self.memory_buffer) > 0:
                    self._replay_memory(optimizer, task_info)
        
        return total_loss / num_batches
    
    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor, task_info: TaskInfo) -> torch.Tensor:
        """Compute loss based on the continual learning strategy"""
        # Base classification loss
        outputs = self.model(inputs)
        base_loss = F.cross_entropy(outputs, targets)
        
        # Strategy-specific regularization
        reg_loss = 0.0
        
        if self.config.strategy == LearningStrategy.EWC and self.current_task_id > 0:
            reg_loss = self._compute_ewc_loss()
        elif self.config.strategy == LearningStrategy.LWF and self.current_task_id > 0:
            reg_loss = self._compute_lwf_loss(inputs)
        elif self.config.strategy == LearningStrategy.MAS and self.current_task_id > 0:
            reg_loss = self._compute_mas_loss()
        elif self.config.strategy == LearningStrategy.SI and self.current_task_id > 0:
            reg_loss = self._compute_si_loss()
        
        return base_loss + reg_loss
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute Elastic Weight Consolidation regularization loss"""
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.config.ewc_lambda * ewc_loss
    
    def _compute_lwf_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Learning without Forgetting distillation loss"""
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            old_outputs = self.old_model(inputs)
        
        new_outputs = self.model(inputs)
        
        # Knowledge distillation loss
        old_probs = F.softmax(old_outputs / self.config.lwf_temperature, dim=1)
        new_log_probs = F.log_softmax(new_outputs / self.config.lwf_temperature, dim=1)
        
        distill_loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        distill_loss *= (self.config.lwf_temperature ** 2)
        
        return self.config.lwf_alpha * distill_loss
    
    def _compute_mas_loss(self) -> torch.Tensor:
        """Compute Memory Aware Synapses regularization loss"""
        mas_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.importance_weights and name in self.optimal_params:
                importance = self.importance_weights[name]
                optimal = self.optimal_params[name]
                mas_loss += (importance * (param - optimal) ** 2).sum()
        
        return self.config.mas_lambda * mas_loss
    
    def _compute_si_loss(self) -> torch.Tensor:
        """Compute Synaptic Intelligence regularization loss"""
        si_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.importance_weights and name in self.optimal_params:
                importance = self.importance_weights[name]
                optimal = self.optimal_params[name]
                si_loss += (importance * (param - optimal) ** 2).sum()
        
        return self.config.si_c * si_loss
    
    def _compute_fisher_information(self, train_loader):
        """Compute Fisher Information Matrix for EWC"""
        self.logger.info("Computing Fisher Information Matrix")
        
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        # Compute Fisher information
        num_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        # Normalize Fisher information
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        self.fisher_information = fisher_info
        
        # Store optimal parameters
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
    
    def _add_progressive_column(self, task_info: TaskInfo):
        """Add a new column for progressive neural networks"""
        if len(self.progressive_columns) >= self.config.progressive_columns:
            self.logger.warning("Maximum number of progressive columns reached")
            return
        
        # Create new column (copy of base model)
        new_column = copy.deepcopy(self.progressive_columns[0])
        
        # Freeze previous columns
        for column in self.progressive_columns:
            for param in column.parameters():
                param.requires_grad = False
        
        # Add lateral connections if enabled
        if self.config.progressive_lateral_connections:
            self._add_lateral_connections(new_column, len(self.progressive_columns))
        
        self.progressive_columns.append(new_column)
        self.model = new_column  # Use new column as current model
    
    def _add_lateral_connections(self, new_column: nn.Module, column_idx: int):
        """Add lateral connections between progressive columns"""
        # This is a simplified implementation
        # In practice, you would need to modify the model architecture
        pass
    
    def _update_memory(self, task_info: TaskInfo, train_loader):
        """Update memory buffer with examples from current task"""
        if self.config.memory_strategy == MemoryStrategy.NONE:
            return
        
        # Determine memory allocation
        if self.config.memory_per_task:
            task_memory_size = self.config.memory_per_task
        else:
            num_tasks = len(self.tasks)
            task_memory_size = self.config.memory_size // num_tasks
        
        # Select examples based on strategy
        if self.config.memory_strategy == MemoryStrategy.RANDOM:
            selected_examples = self._select_random_examples(train_loader, task_memory_size)
        elif self.config.memory_strategy == MemoryStrategy.HERDING:
            selected_examples = self._select_herding_examples(train_loader, task_memory_size)
        elif self.config.memory_strategy == MemoryStrategy.GRADIENT_BASED:
            selected_examples = self._select_gradient_based_examples(train_loader, task_memory_size)
        else:
            selected_examples = self._select_random_examples(train_loader, task_memory_size)
        
        # Update memory buffer
        self.memory_buffer[task_info.task_id] = selected_examples['inputs']
        self.memory_labels[task_info.task_id] = selected_examples['targets']
        
        self.logger.info(f"Updated memory buffer for task {task_info.task_id} with {len(selected_examples['inputs'])} examples")
    
    def _select_random_examples(self, train_loader, num_examples: int) -> Dict[str, torch.Tensor]:
        """Randomly select examples for memory buffer"""
        all_inputs = []
        all_targets = []
        
        for inputs, targets in train_loader:
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Random selection
        indices = torch.randperm(len(all_inputs))[:num_examples]
        
        return {
            'inputs': all_inputs[indices],
            'targets': all_targets[indices]
        }
    
    def _select_herding_examples(self, train_loader, num_examples: int) -> Dict[str, torch.Tensor]:
        """Select examples using herding strategy (class-balanced)"""
        # Get all examples
        all_inputs = []
        all_targets = []
        
        for inputs, targets in train_loader:
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Group by class
        unique_classes = torch.unique(all_targets)
        examples_per_class = num_examples // len(unique_classes)
        
        selected_inputs = []
        selected_targets = []
        
        for class_id in unique_classes:
            class_mask = all_targets == class_id
            class_inputs = all_inputs[class_mask]
            class_targets = all_targets[class_mask]
            
            # Select examples for this class
            if len(class_inputs) <= examples_per_class:
                selected_inputs.append(class_inputs)
                selected_targets.append(class_targets)
            else:
                indices = torch.randperm(len(class_inputs))[:examples_per_class]
                selected_inputs.append(class_inputs[indices])
                selected_targets.append(class_targets[indices])
        
        return {
            'inputs': torch.cat(selected_inputs, dim=0),
            'targets': torch.cat(selected_targets, dim=0)
        }
    
    def _select_gradient_based_examples(self, train_loader, num_examples: int) -> Dict[str, torch.Tensor]:
        """Select examples based on gradient magnitude"""
        self.model.eval()
        
        example_gradients = []
        all_inputs = []
        all_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad_(True)
            
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='none')
            
            for i in range(len(loss)):
                self.model.zero_grad()
                loss[i].backward(retain_graph=True)
                
                # Compute gradient magnitude
                grad_magnitude = torch.norm(inputs.grad[i]).item()
                example_gradients.append(grad_magnitude)
            
            all_inputs.append(inputs.detach().cpu())
            all_targets.append(targets.cpu())
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Select examples with highest gradient magnitudes
        gradient_indices = np.argsort(example_gradients)[-num_examples:]
        
        return {
            'inputs': all_inputs[gradient_indices],
            'targets': all_targets[gradient_indices]
        }
    
    def _replay_memory(self, optimizer, task_info: TaskInfo):
        """Replay examples from memory buffer"""
        if not self.memory_buffer:
            return
        
        # Sample from memory buffer
        memory_inputs = []
        memory_targets = []
        
        for task_id, inputs in self.memory_buffer.items():
            if task_id != task_info.task_id:  # Don't replay current task
                memory_inputs.append(inputs)
                memory_targets.append(self.memory_labels[task_id])
        
        if not memory_inputs:
            return
        
        # Concatenate all memory examples
        memory_inputs = torch.cat(memory_inputs, dim=0)
        memory_targets = torch.cat(memory_targets, dim=0)
        
        # Sample batch
        batch_size = min(self.config.batch_size, len(memory_inputs))
        indices = torch.randperm(len(memory_inputs))[:batch_size]
        
        batch_inputs = memory_inputs[indices].to(self.device)
        batch_targets = memory_targets[indices].to(self.device)
        
        # Forward pass on memory examples
        if self.config.strategy == LearningStrategy.GEM:
            self._gem_replay(batch_inputs, batch_targets, optimizer)
        elif self.config.strategy == LearningStrategy.AGEM:
            self._agem_replay(batch_inputs, batch_targets, optimizer)
        else:
            # Standard replay
            optimizer.zero_grad()
            outputs = self.model(batch_inputs)
            loss = F.cross_entropy(outputs, batch_targets)
            loss.backward()
            optimizer.step()
    
    def _gem_replay(self, memory_inputs: torch.Tensor, memory_targets: torch.Tensor, optimizer):
        """Gradient Episodic Memory replay"""
        # Compute gradients on memory examples
        self.model.zero_grad()
        memory_outputs = self.model(memory_inputs)
        memory_loss = F.cross_entropy(memory_outputs, memory_targets)
        memory_loss.backward()
        
        # Store memory gradients
        memory_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                memory_grads.append(param.grad.data.clone().flatten())
        
        if memory_grads:
            memory_grad_vector = torch.cat(memory_grads)
            
            # Check if current gradients violate memory constraints
            current_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    current_grads.append(param.grad.data.clone().flatten())
            
            if current_grads:
                current_grad_vector = torch.cat(current_grads)
                
                # Project gradients if necessary
                dot_product = torch.dot(current_grad_vector, memory_grad_vector)
                if dot_product < 0:
                    # Project current gradients
                    memory_norm_sq = torch.dot(memory_grad_vector, memory_grad_vector)
                    projection = dot_product / memory_norm_sq
                    projected_grad = current_grad_vector - projection * memory_grad_vector
                    
                    # Update model gradients
                    idx = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param_size = param.grad.numel()
                            param.grad.data = projected_grad[idx:idx+param_size].view(param.grad.shape)
                            idx += param_size
    
    def _agem_replay(self, memory_inputs: torch.Tensor, memory_targets: torch.Tensor, optimizer):
        """Averaged Gradient Episodic Memory replay"""
        # This is a simplified implementation
        # In practice, A-GEM uses a reference gradient computed on a reference batch
        self._gem_replay(memory_inputs, memory_targets, optimizer)
    
    def _setup_optimizer(self, task_info: TaskInfo) -> optim.Optimizer:
        """Setup optimizer for task training"""
        lr = task_info.learning_rate or self.config.learning_rate
        
        if self.config.strategy == LearningStrategy.PROGRESSIVE:
            # Only optimize current column
            params = self.model.parameters()
        else:
            params = self.model.parameters()
        
        return optim.Adam(params, lr=lr)
    
    def _setup_scheduler(self, optimizer, task_info: TaskInfo):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    def _evaluate_task(self, test_loader, task_id: int) -> float:
        """Evaluate model on a specific task"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def _evaluate_all_tasks(self, test_loaders):
        """Evaluate model on all previous tasks"""
        if not isinstance(test_loaders, dict):
            # If single test loader, assume it's for current task
            test_loaders = {self.current_task_id: test_loaders}
        
        for task_id, test_loader in test_loaders.items():
            if task_id <= self.current_task_id:
                accuracy = self._evaluate_task(test_loader, task_id)
                
                # Update statistics
                if task_id not in self.stats.task_accuracies:
                    self.stats.task_accuracies[task_id] = []
                self.stats.task_accuracies[task_id].append(accuracy)
                
                self.logger.info(f"Task {task_id} accuracy: {accuracy:.4f}")
    
    def _save_checkpoint(self, task_id: int):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'task_id': task_id,
            'tasks': self.tasks,
            'config': self.config,
            'stats': self.stats,
            'memory_buffer': self.memory_buffer,
            'memory_labels': self.memory_labels
        }
        
        # Strategy-specific state
        if self.config.strategy == LearningStrategy.EWC:
            checkpoint['fisher_information'] = self.fisher_information
            checkpoint['optimal_params'] = self.optimal_params
        elif self.config.strategy == LearningStrategy.PROGRESSIVE:
            checkpoint['progressive_columns'] = [col.state_dict() for col in self.progressive_columns]
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"task_{task_id}_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_task_id = checkpoint['task_id']
        self.tasks = checkpoint['tasks']
        self.stats = checkpoint['stats']
        self.memory_buffer = checkpoint.get('memory_buffer', {})
        self.memory_labels = checkpoint.get('memory_labels', {})
        
        # Strategy-specific state
        if self.config.strategy == LearningStrategy.EWC:
            self.fisher_information = checkpoint.get('fisher_information', {})
            self.optimal_params = checkpoint.get('optimal_params', {})
        elif self.config.strategy == LearningStrategy.PROGRESSIVE:
            progressive_states = checkpoint.get('progressive_columns', [])
            self.progressive_columns = []
            for state_dict in progressive_states:
                column = copy.deepcopy(self.model)
                column.load_state_dict(state_dict)
                self.progressive_columns.append(column)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def compute_forgetting_measure(self) -> Dict[int, float]:
        """Compute forgetting measure for all tasks"""
        forgetting_measures = {}
        
        for task_id in self.stats.task_accuracies:
            if len(self.stats.task_accuracies[task_id]) > 1:
                max_acc = max(self.stats.task_accuracies[task_id])
                final_acc = self.stats.task_accuracies[task_id][-1]
                forgetting_measures[task_id] = max_acc - final_acc
            else:
                forgetting_measures[task_id] = 0.0
        
        self.stats.forgetting_measures = forgetting_measures
        return forgetting_measures
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves for all tasks"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Task accuracies over time
        for task_id, accuracies in self.stats.task_accuracies.items():
            axes[0, 0].plot(accuracies, label=f'Task {task_id}', marker='o')
        axes[0, 0].set_xlabel('Evaluation Points')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Task Accuracies Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Task losses over time
        for task_id, losses in self.stats.task_losses.items():
            axes[0, 1].plot(losses, label=f'Task {task_id}', marker='o')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Task Losses Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Forgetting measures
        forgetting_measures = self.compute_forgetting_measure()
        if forgetting_measures:
            task_ids = list(forgetting_measures.keys())
            forgetting_values = list(forgetting_measures.values())
            axes[1, 0].bar(task_ids, forgetting_values, alpha=0.7)
            axes[1, 0].set_xlabel('Task ID')
            axes[1, 0].set_ylabel('Forgetting Measure')
            axes[1, 0].set_title('Forgetting Measures by Task')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Memory usage over time
        if self.stats.memory_usage:
            axes[1, 1].plot(self.stats.memory_usage, marker='o')
            axes[1, 1].set_xlabel('Tasks')
            axes[1, 1].set_ylabel('Memory Usage')
            axes[1, 1].set_title('Memory Usage Over Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        forgetting_measures = self.compute_forgetting_measure()
        
        summary = {
            'num_tasks': len(self.tasks),
            'strategy': self.config.strategy.value,
            'memory_strategy': self.config.memory_strategy.value,
            'total_training_time': self.stats.total_training_time,
            'average_forgetting': np.mean(list(forgetting_measures.values())) if forgetting_measures else 0.0,
            'final_accuracies': {
                task_id: accs[-1] if accs else 0.0 
                for task_id, accs in self.stats.task_accuracies.items()
            },
            'memory_size': len(self.memory_buffer) if self.memory_buffer else 0,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create configuration
    config = ContinualConfig(
        strategy=LearningStrategy.EWC,
        memory_strategy=MemoryStrategy.HERDING,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=5,
        memory_size=1000,
        ewc_lambda=1000.0
    )
    
    # Create trainer
    trainer = ContinualTrainer(model, config)
    
    print("Continual learning trainer created successfully!")
    print(f"Strategy: {config.strategy.value}")
    print(f"Memory strategy: {config.memory_strategy.value}")