"""
Memory Management for Continual Learning
निरंतर शिक्षा के लिए मेमोरी प्रबंधन

Advanced memory management system for continual learning with various
replay strategies and memory selection mechanisms.
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
import random
from collections import defaultdict, deque
import heapq
from abc import ABC, abstractmethod


class MemorySelectionStrategy(Enum):
    """Memory selection strategies"""
    RANDOM = "random"
    FIFO = "fifo"                    # First In First Out
    LIFO = "lifo"                    # Last In First Out
    HERDING = "herding"              # Class-balanced herding
    GRADIENT_BASED = "gradient_based" # Based on gradient magnitude
    UNCERTAINTY_BASED = "uncertainty_based"  # Based on prediction uncertainty
    DIVERSITY_BASED = "diversity_based"      # Based on feature diversity
    PROTOTYPE_BASED = "prototype_based"      # Based on class prototypes
    LOSS_BASED = "loss_based"        # Based on loss values
    INFLUENCE_BASED = "influence_based"      # Based on influence functions


class ReplayStrategy(Enum):
    """Replay strategies"""
    STANDARD = "standard"            # Standard experience replay
    BALANCED = "balanced"            # Balanced replay across tasks/classes
    PRIORITIZED = "prioritized"      # Prioritized experience replay
    GRADIENT_BASED = "gradient_based" # Gradient-based replay
    ADVERSARIAL = "adversarial"      # Adversarial replay
    GENERATIVE = "generative"        # Generative replay


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    # Memory size parameters
    total_memory_size: int = 1000
    memory_per_task: Optional[int] = None
    memory_per_class: Optional[int] = None
    
    # Selection strategy
    selection_strategy: MemorySelectionStrategy = MemorySelectionStrategy.RANDOM
    replay_strategy: ReplayStrategy = ReplayStrategy.STANDARD
    
    # Replay parameters
    replay_batch_size: int = 32
    replay_frequency: int = 1
    replay_weight: float = 1.0
    
    # Herding parameters
    herding_method: str = "mean"     # "mean" or "kmeans"
    
    # Gradient-based parameters
    gradient_threshold: float = 0.1
    gradient_accumulation: int = 1
    
    # Uncertainty parameters
    uncertainty_method: str = "entropy"  # "entropy", "variance", "mutual_info"
    uncertainty_samples: int = 10
    
    # Diversity parameters
    diversity_metric: str = "cosine"     # "cosine", "euclidean", "kl_div"
    diversity_threshold: float = 0.8
    
    # Prioritized replay parameters
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_epsilon: float = 1e-6
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True


@dataclass
class MemoryItem:
    """Individual memory item"""
    data: torch.Tensor
    target: torch.Tensor
    task_id: int
    class_id: int
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    uncertainty: Optional[float] = None
    priority: float = 1.0
    
    # Feature representations
    features: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class MemoryBuffer:
    """Base memory buffer class"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Memory storage
        self.memory: List[MemoryItem] = []
        self.task_memory: Dict[int, List[MemoryItem]] = defaultdict(list)
        self.class_memory: Dict[int, List[MemoryItem]] = defaultdict(list)
        
        # Statistics
        self.total_items_seen = 0
        self.total_items_stored = 0
        self.replacement_count = 0
    
    def add_item(self, item: MemoryItem) -> bool:
        """Add item to memory buffer"""
        self.total_items_seen += 1
        
        # Check if memory is full
        if len(self.memory) < self.config.total_memory_size:
            self._insert_item(item)
            return True
        else:
            # Need to replace an item
            return self._replace_item(item)
    
    def _insert_item(self, item: MemoryItem):
        """Insert item into memory"""
        self.memory.append(item)
        self.task_memory[item.task_id].append(item)
        self.class_memory[item.class_id].append(item)
        self.total_items_stored += 1
    
    def _replace_item(self, new_item: MemoryItem) -> bool:
        """Replace an item in memory based on selection strategy"""
        if self.config.selection_strategy == MemorySelectionStrategy.RANDOM:
            return self._replace_random(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.FIFO:
            return self._replace_fifo(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.LIFO:
            return self._replace_lifo(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.HERDING:
            return self._replace_herding(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.GRADIENT_BASED:
            return self._replace_gradient_based(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.UNCERTAINTY_BASED:
            return self._replace_uncertainty_based(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.DIVERSITY_BASED:
            return self._replace_diversity_based(new_item)
        elif self.config.selection_strategy == MemorySelectionStrategy.LOSS_BASED:
            return self._replace_loss_based(new_item)
        else:
            return self._replace_random(new_item)
    
    def _replace_random(self, new_item: MemoryItem) -> bool:
        """Replace random item"""
        idx = random.randint(0, len(self.memory) - 1)
        old_item = self.memory[idx]
        
        # Remove from task and class memory
        self.task_memory[old_item.task_id].remove(old_item)
        self.class_memory[old_item.class_id].remove(old_item)
        
        # Replace with new item
        self.memory[idx] = new_item
        self.task_memory[new_item.task_id].append(new_item)
        self.class_memory[new_item.class_id].append(new_item)
        
        self.replacement_count += 1
        return True
    
    def _replace_fifo(self, new_item: MemoryItem) -> bool:
        """Replace oldest item (FIFO)"""
        # Find oldest item
        oldest_idx = 0
        oldest_time = self.memory[0].timestamp
        
        for i, item in enumerate(self.memory):
            if item.timestamp < oldest_time:
                oldest_time = item.timestamp
                oldest_idx = i
        
        old_item = self.memory[oldest_idx]
        
        # Remove from task and class memory
        self.task_memory[old_item.task_id].remove(old_item)
        self.class_memory[old_item.class_id].remove(old_item)
        
        # Replace with new item
        self.memory[oldest_idx] = new_item
        self.task_memory[new_item.task_id].append(new_item)
        self.class_memory[new_item.class_id].append(new_item)
        
        self.replacement_count += 1
        return True
    
    def _replace_lifo(self, new_item: MemoryItem) -> bool:
        """Replace newest item (LIFO)"""
        # Find newest item
        newest_idx = 0
        newest_time = self.memory[0].timestamp
        
        for i, item in enumerate(self.memory):
            if item.timestamp > newest_time:
                newest_time = item.timestamp
                newest_idx = i
        
        old_item = self.memory[newest_idx]
        
        # Remove from task and class memory
        self.task_memory[old_item.task_id].remove(old_item)
        self.class_memory[old_item.class_id].remove(old_item)
        
        # Replace with new item
        self.memory[newest_idx] = new_item
        self.task_memory[new_item.task_id].append(new_item)
        self.class_memory[new_item.class_id].append(new_item)
        
        self.replacement_count += 1
        return True
    
    def _replace_herding(self, new_item: MemoryItem) -> bool:
        """Replace item using herding strategy"""
        # Find items of the same class
        same_class_items = self.class_memory[new_item.class_id]
        
        if not same_class_items:
            # No items of same class, use random replacement
            return self._replace_random(new_item)
        
        # Compute class mean (simplified herding)
        if new_item.features is not None:
            class_features = [item.features for item in same_class_items if item.features is not None]
            if class_features:
                class_mean = torch.stack(class_features).mean(dim=0)
                
                # Find item farthest from class mean
                max_distance = -1
                farthest_idx = -1
                
                for i, item in enumerate(self.memory):
                    if item.class_id == new_item.class_id and item.features is not None:
                        distance = torch.norm(item.features - class_mean).item()
                        if distance > max_distance:
                            max_distance = distance
                            farthest_idx = i
                
                if farthest_idx >= 0:
                    old_item = self.memory[farthest_idx]
                    
                    # Remove from task and class memory
                    self.task_memory[old_item.task_id].remove(old_item)
                    self.class_memory[old_item.class_id].remove(old_item)
                    
                    # Replace with new item
                    self.memory[farthest_idx] = new_item
                    self.task_memory[new_item.task_id].append(new_item)
                    self.class_memory[new_item.class_id].append(new_item)
                    
                    self.replacement_count += 1
                    return True
        
        # Fallback to random replacement
        return self._replace_random(new_item)
    
    def _replace_gradient_based(self, new_item: MemoryItem) -> bool:
        """Replace item with lowest gradient norm"""
        if new_item.gradient_norm is None:
            return self._replace_random(new_item)
        
        # Find item with lowest gradient norm
        min_gradient = float('inf')
        min_idx = -1
        
        for i, item in enumerate(self.memory):
            if item.gradient_norm is not None and item.gradient_norm < min_gradient:
                min_gradient = item.gradient_norm
                min_idx = i
        
        if min_idx >= 0 and new_item.gradient_norm > min_gradient:
            old_item = self.memory[min_idx]
            
            # Remove from task and class memory
            self.task_memory[old_item.task_id].remove(old_item)
            self.class_memory[old_item.class_id].remove(old_item)
            
            # Replace with new item
            self.memory[min_idx] = new_item
            self.task_memory[new_item.task_id].append(new_item)
            self.class_memory[new_item.class_id].append(new_item)
            
            self.replacement_count += 1
            return True
        
        return False  # Don't add if gradient norm is too low
    
    def _replace_uncertainty_based(self, new_item: MemoryItem) -> bool:
        """Replace item with lowest uncertainty"""
        if new_item.uncertainty is None:
            return self._replace_random(new_item)
        
        # Find item with lowest uncertainty
        min_uncertainty = float('inf')
        min_idx = -1
        
        for i, item in enumerate(self.memory):
            if item.uncertainty is not None and item.uncertainty < min_uncertainty:
                min_uncertainty = item.uncertainty
                min_idx = i
        
        if min_idx >= 0 and new_item.uncertainty > min_uncertainty:
            old_item = self.memory[min_idx]
            
            # Remove from task and class memory
            self.task_memory[old_item.task_id].remove(old_item)
            self.class_memory[old_item.class_id].remove(old_item)
            
            # Replace with new item
            self.memory[min_idx] = new_item
            self.task_memory[new_item.task_id].append(new_item)
            self.class_memory[new_item.class_id].append(new_item)
            
            self.replacement_count += 1
            return True
        
        return False  # Don't add if uncertainty is too low
    
    def _replace_diversity_based(self, new_item: MemoryItem) -> bool:
        """Replace item to maximize diversity"""
        if new_item.features is None:
            return self._replace_random(new_item)
        
        # Compute diversity scores
        diversity_scores = []
        
        for i, item in enumerate(self.memory):
            if item.features is not None:
                # Compute similarity with new item
                if self.config.diversity_metric == "cosine":
                    similarity = F.cosine_similarity(
                        new_item.features.unsqueeze(0), 
                        item.features.unsqueeze(0)
                    ).item()
                elif self.config.diversity_metric == "euclidean":
                    similarity = -torch.norm(new_item.features - item.features).item()
                else:
                    similarity = 0.0
                
                diversity_scores.append((i, similarity))
        
        if diversity_scores:
            # Sort by similarity (ascending for diversity)
            diversity_scores.sort(key=lambda x: x[1])
            
            # Replace most similar item
            most_similar_idx = diversity_scores[0][0]
            old_item = self.memory[most_similar_idx]
            
            # Remove from task and class memory
            self.task_memory[old_item.task_id].remove(old_item)
            self.class_memory[old_item.class_id].remove(old_item)
            
            # Replace with new item
            self.memory[most_similar_idx] = new_item
            self.task_memory[new_item.task_id].append(new_item)
            self.class_memory[new_item.class_id].append(new_item)
            
            self.replacement_count += 1
            return True
        
        return self._replace_random(new_item)
    
    def _replace_loss_based(self, new_item: MemoryItem) -> bool:
        """Replace item with lowest loss"""
        if new_item.loss is None:
            return self._replace_random(new_item)
        
        # Find item with lowest loss
        min_loss = float('inf')
        min_idx = -1
        
        for i, item in enumerate(self.memory):
            if item.loss is not None and item.loss < min_loss:
                min_loss = item.loss
                min_idx = i
        
        if min_idx >= 0 and new_item.loss > min_loss:
            old_item = self.memory[min_idx]
            
            # Remove from task and class memory
            self.task_memory[old_item.task_id].remove(old_item)
            self.class_memory[old_item.class_id].remove(old_item)
            
            # Replace with new item
            self.memory[min_idx] = new_item
            self.task_memory[new_item.task_id].append(new_item)
            self.class_memory[new_item.class_id].append(new_item)
            
            self.replacement_count += 1
            return True
        
        return False  # Don't add if loss is too low
    
    def sample_batch(self, batch_size: int, task_id: Optional[int] = None, 
                    class_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from memory"""
        if not self.memory:
            return torch.empty(0), torch.empty(0)
        
        # Filter memory based on constraints
        if task_id is not None:
            available_items = self.task_memory[task_id]
        elif class_id is not None:
            available_items = self.class_memory[class_id]
        else:
            available_items = self.memory
        
        if not available_items:
            return torch.empty(0), torch.empty(0)
        
        # Sample items
        if self.config.replay_strategy == ReplayStrategy.STANDARD:
            sampled_items = self._sample_standard(available_items, batch_size)
        elif self.config.replay_strategy == ReplayStrategy.BALANCED:
            sampled_items = self._sample_balanced(available_items, batch_size)
        elif self.config.replay_strategy == ReplayStrategy.PRIORITIZED:
            sampled_items = self._sample_prioritized(available_items, batch_size)
        else:
            sampled_items = self._sample_standard(available_items, batch_size)
        
        # Extract data and targets
        data = torch.stack([item.data for item in sampled_items])
        targets = torch.stack([item.target for item in sampled_items])
        
        return data.to(self.device), targets.to(self.device)
    
    def _sample_standard(self, items: List[MemoryItem], batch_size: int) -> List[MemoryItem]:
        """Standard random sampling"""
        return random.sample(items, min(batch_size, len(items)))
    
    def _sample_balanced(self, items: List[MemoryItem], batch_size: int) -> List[MemoryItem]:
        """Balanced sampling across classes"""
        # Group by class
        class_items = defaultdict(list)
        for item in items:
            class_items[item.class_id].append(item)
        
        # Sample equally from each class
        sampled_items = []
        num_classes = len(class_items)
        items_per_class = batch_size // num_classes
        
        for class_id, class_item_list in class_items.items():
            num_to_sample = min(items_per_class, len(class_item_list))
            sampled_items.extend(random.sample(class_item_list, num_to_sample))
        
        # Fill remaining slots randomly
        remaining = batch_size - len(sampled_items)
        if remaining > 0:
            remaining_items = [item for item in items if item not in sampled_items]
            if remaining_items:
                sampled_items.extend(random.sample(remaining_items, min(remaining, len(remaining_items))))
        
        return sampled_items
    
    def _sample_prioritized(self, items: List[MemoryItem], batch_size: int) -> List[MemoryItem]:
        """Prioritized sampling based on priority scores"""
        if not items:
            return []
        
        # Extract priorities
        priorities = [item.priority for item in items]
        priorities = np.array(priorities)
        
        # Apply alpha exponent
        priorities = priorities ** self.config.priority_alpha
        
        # Compute probabilities
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(items), size=min(batch_size, len(items)), 
                                 replace=False, p=probabilities)
        
        return [items[i] for i in indices]
    
    def update_priorities(self, items: List[MemoryItem], losses: List[float]):
        """Update priority scores for prioritized replay"""
        for item, loss in zip(items, losses):
            item.priority = abs(loss) + self.config.priority_epsilon
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory buffer statistics"""
        stats = {
            'total_items_seen': self.total_items_seen,
            'total_items_stored': self.total_items_stored,
            'current_size': len(self.memory),
            'replacement_count': self.replacement_count,
            'memory_utilization': len(self.memory) / self.config.total_memory_size,
            'tasks_in_memory': len(self.task_memory),
            'classes_in_memory': len(self.class_memory)
        }
        
        # Task distribution
        task_distribution = {}
        for task_id, items in self.task_memory.items():
            task_distribution[task_id] = len(items)
        stats['task_distribution'] = task_distribution
        
        # Class distribution
        class_distribution = {}
        for class_id, items in self.class_memory.items():
            class_distribution[class_id] = len(items)
        stats['class_distribution'] = class_distribution
        
        return stats
    
    def clear(self):
        """Clear all memory"""
        self.memory.clear()
        self.task_memory.clear()
        self.class_memory.clear()
        self.total_items_seen = 0
        self.total_items_stored = 0
        self.replacement_count = 0


class ExperienceReplay:
    """Experience Replay implementation"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.buffer = MemoryBuffer(config)
        self.logger = logging.getLogger(__name__)
    
    def store_experience(self, data: torch.Tensor, targets: torch.Tensor, 
                        task_id: int, model: nn.Module = None) -> None:
        """Store experience in memory buffer"""
        batch_size = data.size(0)
        
        for i in range(batch_size):
            # Create memory item
            item = MemoryItem(
                data=data[i].cpu(),
                target=targets[i].cpu(),
                task_id=task_id,
                class_id=targets[i].item()
            )
            
            # Compute additional metadata if model is provided
            if model is not None:
                item = self._compute_metadata(item, model)
            
            # Add to buffer
            self.buffer.add_item(item)
    
    def _compute_metadata(self, item: MemoryItem, model: nn.Module) -> MemoryItem:
        """Compute additional metadata for memory item"""
        model.eval()
        
        with torch.no_grad():
            data = item.data.unsqueeze(0).to(self.config.device)
            target = item.target.unsqueeze(0).to(self.config.device)
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = F.cross_entropy(output, target)
            item.loss = loss.item()
            
            # Compute uncertainty
            if self.config.uncertainty_method == "entropy":
                probs = F.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                item.uncertainty = entropy.item()
            
            # Extract features (assuming model has a feature extractor)
            if hasattr(model, 'features'):
                features = model.features(data)
                item.features = features.squeeze().cpu()
            elif hasattr(model, 'feature_extractor'):
                features = model.feature_extractor(data)
                item.features = features.squeeze().cpu()
        
        return item
    
    def replay(self, batch_size: int, task_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample replay batch"""
        return self.buffer.sample_batch(batch_size, task_id=task_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics"""
        return self.buffer.get_statistics()


class GradientEpisodicMemory:
    """Gradient Episodic Memory (GEM) implementation"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.buffer = MemoryBuffer(config)
        self.reference_gradients = {}
        self.logger = logging.getLogger(__name__)
    
    def store_experience(self, data: torch.Tensor, targets: torch.Tensor, 
                        task_id: int, model: nn.Module) -> None:
        """Store experience and compute reference gradients"""
        # Store in buffer
        batch_size = data.size(0)
        
        for i in range(batch_size):
            item = MemoryItem(
                data=data[i].cpu(),
                target=targets[i].cpu(),
                task_id=task_id,
                class_id=targets[i].item()
            )
            
            # Compute gradient norm
            item = self._compute_gradient_metadata(item, model)
            self.buffer.add_item(item)
        
        # Update reference gradients for this task
        self._update_reference_gradients(task_id, model)
    
    def _compute_gradient_metadata(self, item: MemoryItem, model: nn.Module) -> MemoryItem:
        """Compute gradient-based metadata"""
        model.train()
        
        data = item.data.unsqueeze(0).to(self.config.device)
        target = item.target.unsqueeze(0).to(self.config.device)
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        item.gradient_norm = total_norm ** 0.5
        item.loss = loss.item()
        
        return item
    
    def _update_reference_gradients(self, task_id: int, model: nn.Module):
        """Update reference gradients for task"""
        task_items = self.buffer.task_memory[task_id]
        if not task_items:
            return
        
        # Sample reference batch
        reference_batch_size = min(self.config.replay_batch_size, len(task_items))
        reference_items = random.sample(task_items, reference_batch_size)
        
        # Compute reference gradients
        model.train()
        model.zero_grad()
        
        total_loss = 0.0
        for item in reference_items:
            data = item.data.unsqueeze(0).to(self.config.device)
            target = item.target.unsqueeze(0).to(self.config.device)
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss
        
        # Backward pass
        avg_loss = total_loss / len(reference_items)
        avg_loss.backward()
        
        # Store reference gradients
        reference_grads = []
        for param in model.parameters():
            if param.grad is not None:
                reference_grads.append(param.grad.data.clone().flatten())
        
        if reference_grads:
            self.reference_gradients[task_id] = torch.cat(reference_grads)
    
    def project_gradients(self, model: nn.Module, current_task_id: int) -> bool:
        """Project gradients to satisfy GEM constraints"""
        if not self.reference_gradients:
            return False
        
        # Get current gradients
        current_grads = []
        for param in model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.data.clone().flatten())
        
        if not current_grads:
            return False
        
        current_grad_vector = torch.cat(current_grads)
        
        # Check constraints for all previous tasks
        projection_needed = False
        
        for task_id, reference_grad in self.reference_gradients.items():
            if task_id != current_task_id:
                # Check if constraint is violated
                dot_product = torch.dot(current_grad_vector, reference_grad)
                
                if dot_product < -self.config.gem_margin:
                    # Project gradients
                    reference_norm_sq = torch.dot(reference_grad, reference_grad)
                    projection = (dot_product + self.config.gem_margin) / reference_norm_sq
                    current_grad_vector = current_grad_vector - projection * reference_grad
                    projection_needed = True
        
        # Update model gradients if projection was needed
        if projection_needed:
            idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_size = param.grad.numel()
                    param.grad.data = current_grad_vector[idx:idx+param_size].view(param.grad.shape)
                    idx += param_size
        
        return projection_needed
    
    def replay(self, batch_size: int, exclude_task: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample replay batch excluding specific task"""
        if exclude_task is not None:
            # Sample from all tasks except the excluded one
            available_items = []
            for task_id, items in self.buffer.task_memory.items():
                if task_id != exclude_task:
                    available_items.extend(items)
            
            if not available_items:
                return torch.empty(0), torch.empty(0)
            
            sampled_items = random.sample(available_items, min(batch_size, len(available_items)))
            
            # Extract data and targets
            data = torch.stack([item.data for item in sampled_items])
            targets = torch.stack([item.target for item in sampled_items])
            
            return data.to(self.config.device), targets.to(self.config.device)
        else:
            return self.buffer.sample_batch(batch_size)


class MemoryManager:
    """Main memory manager for continual learning"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory components
        self.experience_replay = ExperienceReplay(config)
        self.gem = GradientEpisodicMemory(config)
        
        # Statistics
        self.total_replays = 0
        self.total_projections = 0
    
    def store_batch(self, data: torch.Tensor, targets: torch.Tensor, 
                   task_id: int, model: nn.Module = None) -> None:
        """Store a batch of experiences"""
        # Store in experience replay buffer
        self.experience_replay.store_experience(data, targets, task_id, model)
        
        # Store in GEM buffer if model is provided
        if model is not None:
            self.gem.store_experience(data, targets, task_id, model)
    
    def get_replay_batch(self, batch_size: int, strategy: str = "standard", 
                        task_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get replay batch based on strategy"""
        self.total_replays += 1
        
        if strategy == "gem":
            return self.gem.replay(batch_size, exclude_task=task_id)
        else:
            return self.experience_replay.replay(batch_size, task_id=task_id)
    
    def apply_gem_projection(self, model: nn.Module, current_task_id: int) -> bool:
        """Apply GEM gradient projection"""
        projection_applied = self.gem.project_gradients(model, current_task_id)
        if projection_applied:
            self.total_projections += 1
        return projection_applied
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        er_stats = self.experience_replay.get_statistics()
        
        stats = {
            'experience_replay': er_stats,
            'total_replays': self.total_replays,
            'total_projections': self.total_projections,
            'gem_reference_tasks': len(self.gem.reference_gradients)
        }
        
        return stats
    
    def clear_memory(self):
        """Clear all memory buffers"""
        self.experience_replay.buffer.clear()
        self.gem.buffer.clear()
        self.gem.reference_gradients.clear()
        self.total_replays = 0
        self.total_projections = 0


# Example usage
if __name__ == "__main__":
    # Create memory configuration
    config = MemoryConfig(
        total_memory_size=1000,
        selection_strategy=MemorySelectionStrategy.HERDING,
        replay_strategy=ReplayStrategy.BALANCED,
        replay_batch_size=32
    )
    
    # Create memory manager
    memory_manager = MemoryManager(config)
    
    print("Memory manager created successfully!")
    print(f"Total memory size: {config.total_memory_size}")
    print(f"Selection strategy: {config.selection_strategy.value}")
    print(f"Replay strategy: {config.replay_strategy.value}")