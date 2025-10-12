"""
Task Balancer for Multi-Task Learning
टास्क बैलेंसर - मल्टी-टास्क लर्निंग के लिए

Implements dynamic task weighting and balancing strategies to ensure
optimal learning across all tasks in the multi-task model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math
from collections import defaultdict, deque
import logging


class BalancingStrategy(Enum):
    """Strategies for balancing multiple tasks"""
    EQUAL_WEIGHTS = "equal_weights"
    UNCERTAINTY_WEIGHTING = "uncertainty_weighting"
    GRADIENT_NORMALIZATION = "gradient_normalization"
    DYNAMIC_WEIGHT_AVERAGE = "dynamic_weight_average"
    PERFORMANCE_BASED = "performance_based"
    CURRICULUM_LEARNING = "curriculum_learning"
    ADAPTIVE_LOSS_SCALING = "adaptive_loss_scaling"


@dataclass
class BalancingConfig:
    """Configuration for task balancing"""
    strategy: BalancingStrategy = BalancingStrategy.DYNAMIC_WEIGHT_AVERAGE
    update_frequency: int = 100  # Update weights every N steps
    temperature: float = 2.0     # Temperature for softmax weighting
    momentum: float = 0.9        # Momentum for weight updates
    min_weight: float = 0.1      # Minimum weight for any task
    max_weight: float = 3.0      # Maximum weight for any task
    
    # Performance-based balancing
    performance_window: int = 1000  # Window for performance tracking
    target_performance: Dict[str, float] = None
    
    # Curriculum learning
    curriculum_schedule: Dict[str, Tuple[int, int]] = None  # (start_epoch, end_epoch)
    
    def __post_init__(self):
        if self.target_performance is None:
            self.target_performance = {
                "element_detection": 0.85,
                "ocr": 0.90,
                "intent_classification": 0.80,
                "action_prediction": 0.75
            }
        
        if self.curriculum_schedule is None:
            self.curriculum_schedule = {
                "element_detection": (0, 10),
                "ocr": (5, 20),
                "intent_classification": (10, 30),
                "action_prediction": (15, 40)
            }


class TaskPerformanceMonitor:
    """Monitors performance of individual tasks"""
    
    def __init__(self, config: BalancingConfig):
        self.config = config
        self.performance_history = defaultdict(lambda: deque(maxlen=config.performance_window))
        self.loss_history = defaultdict(lambda: deque(maxlen=config.performance_window))
        self.gradient_norms = defaultdict(lambda: deque(maxlen=100))
        self.current_epoch = 0
        self.current_step = 0
        
    def update_performance(self, task_name: str, performance: float) -> None:
        """Update performance metric for a task"""
        self.performance_history[task_name].append(performance)
    
    def update_loss(self, task_name: str, loss: float) -> None:
        """Update loss for a task"""
        self.loss_history[task_name].append(loss)
    
    def update_gradient_norm(self, task_name: str, grad_norm: float) -> None:
        """Update gradient norm for a task"""
        self.gradient_norms[task_name].append(grad_norm)
    
    def get_average_performance(self, task_name: str, window: int = None) -> float:
        """Get average performance for a task"""
        if task_name not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history[task_name])
        if not history:
            return 0.0
        
        if window:
            history = history[-window:]
        
        return np.mean(history)
    
    def get_average_loss(self, task_name: str, window: int = None) -> float:
        """Get average loss for a task"""
        if task_name not in self.loss_history:
            return float('inf')
        
        history = list(self.loss_history[task_name])
        if not history:
            return float('inf')
        
        if window:
            history = history[-window:]
        
        return np.mean(history)
    
    def get_gradient_norm(self, task_name: str) -> float:
        """Get recent gradient norm for a task"""
        if task_name not in self.gradient_norms or not self.gradient_norms[task_name]:
            return 1.0
        
        return list(self.gradient_norms[task_name])[-1]
    
    def get_performance_trend(self, task_name: str, window: int = 50) -> float:
        """Get performance trend (positive = improving, negative = degrading)"""
        if task_name not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history[task_name])
        if len(history) < window:
            return 0.0
        
        recent = np.mean(history[-window//2:])
        older = np.mean(history[-window:-window//2])
        
        return recent - older
    
    def step(self) -> None:
        """Increment step counter"""
        self.current_step += 1
    
    def epoch(self) -> None:
        """Increment epoch counter"""
        self.current_epoch += 1


class DynamicWeightScheduler:
    """Schedules task weights dynamically based on various criteria"""
    
    def __init__(self, config: BalancingConfig, task_names: List[str]):
        self.config = config
        self.task_names = task_names
        self.current_weights = {name: 1.0 for name in task_names}
        self.weight_history = defaultdict(list)
        self.monitor = TaskPerformanceMonitor(config)
        
    def compute_uncertainty_weights(self, task_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute weights based on task uncertainty (higher loss = higher weight)"""
        if not task_losses:
            return self.current_weights
        
        # Normalize losses
        loss_values = list(task_losses.values())
        if max(loss_values) == min(loss_values):
            return {name: 1.0 for name in self.task_names}
        
        normalized_losses = {}
        max_loss = max(loss_values)
        min_loss = min(loss_values)
        
        for task_name, loss in task_losses.items():
            normalized_loss = (loss - min_loss) / (max_loss - min_loss)
            # Higher loss gets higher weight
            weight = 1.0 + normalized_loss
            normalized_losses[task_name] = weight
        
        return self._apply_constraints(normalized_losses)
    
    def compute_gradient_normalization_weights(self, gradient_norms: Dict[str, float]) -> Dict[str, float]:
        """Compute weights to normalize gradient magnitudes across tasks"""
        if not gradient_norms:
            return self.current_weights
        
        # Target: make all gradient norms similar
        avg_grad_norm = np.mean(list(gradient_norms.values()))
        
        weights = {}
        for task_name, grad_norm in gradient_norms.items():
            if grad_norm > 0:
                weight = avg_grad_norm / grad_norm
            else:
                weight = 1.0
            weights[task_name] = weight
        
        return self._apply_constraints(weights)
    
    def compute_performance_based_weights(self) -> Dict[str, float]:
        """Compute weights based on performance relative to targets"""
        weights = {}
        
        for task_name in self.task_names:
            current_perf = self.monitor.get_average_performance(task_name, window=100)
            target_perf = self.config.target_performance.get(task_name, 0.8)
            
            # If performance is below target, increase weight
            if current_perf < target_perf:
                performance_gap = target_perf - current_perf
                weight = 1.0 + performance_gap * 2.0  # Scale factor
            else:
                # If performance is above target, slightly decrease weight
                weight = max(0.5, 1.0 - (current_perf - target_perf) * 0.5)
            
            weights[task_name] = weight
        
        return self._apply_constraints(weights)
    
    def compute_curriculum_weights(self) -> Dict[str, float]:
        """Compute weights based on curriculum learning schedule"""
        current_epoch = self.monitor.current_epoch
        weights = {}
        
        for task_name in self.task_names:
            if task_name in self.config.curriculum_schedule:
                start_epoch, end_epoch = self.config.curriculum_schedule[task_name]
                
                if current_epoch < start_epoch:
                    # Task not started yet
                    weight = 0.1
                elif current_epoch > end_epoch:
                    # Task fully active
                    weight = 1.0
                else:
                    # Gradually increase weight
                    progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
                    weight = 0.1 + 0.9 * progress
            else:
                weight = 1.0
            
            weights[task_name] = weight
        
        return self._apply_constraints(weights)
    
    def compute_dynamic_weight_average(self, task_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute weights using Dynamic Weight Average (DWA) method"""
        if not task_losses or len(self.weight_history[list(task_losses.keys())[0]]) < 2:
            return {name: 1.0 for name in self.task_names}
        
        weights = {}
        
        for task_name, current_loss in task_losses.items():
            loss_history = self.monitor.loss_history[task_name]
            
            if len(loss_history) >= 2:
                # Calculate relative decrease rate
                prev_loss = list(loss_history)[-2]
                if prev_loss > 0:
                    rate = current_loss / prev_loss
                else:
                    rate = 1.0
                
                # Apply temperature scaling
                weight = math.exp(rate / self.config.temperature)
            else:
                weight = 1.0
            
            weights[task_name] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight * len(weights) for name, w in weights.items()}
        
        return self._apply_constraints(weights)
    
    def compute_adaptive_loss_scaling(self, task_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute weights using adaptive loss scaling"""
        if not task_losses:
            return self.current_weights
        
        weights = {}
        
        # Calculate loss ratios
        avg_loss = np.mean(list(task_losses.values()))
        
        for task_name, loss in task_losses.items():
            if avg_loss > 0:
                # Scale inversely with loss magnitude
                weight = avg_loss / max(loss, 1e-8)
            else:
                weight = 1.0
            
            weights[task_name] = weight
        
        return self._apply_constraints(weights)
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max constraints and momentum to weights"""
        constrained_weights = {}
        
        for task_name, weight in weights.items():
            # Apply min/max constraints
            weight = max(self.config.min_weight, min(self.config.max_weight, weight))
            
            # Apply momentum
            if task_name in self.current_weights:
                weight = (self.config.momentum * self.current_weights[task_name] + 
                         (1 - self.config.momentum) * weight)
            
            constrained_weights[task_name] = weight
        
        return constrained_weights
    
    def update_weights(self, task_losses: Dict[str, float], 
                      gradient_norms: Dict[str, float] = None) -> Dict[str, float]:
        """Update task weights based on current strategy"""
        if self.monitor.current_step % self.config.update_frequency != 0:
            return self.current_weights
        
        # Update monitoring data
        for task_name, loss in task_losses.items():
            self.monitor.update_loss(task_name, loss)
        
        if gradient_norms:
            for task_name, grad_norm in gradient_norms.items():
                self.monitor.update_gradient_norm(task_name, grad_norm)
        
        # Compute new weights based on strategy
        if self.config.strategy == BalancingStrategy.EQUAL_WEIGHTS:
            new_weights = {name: 1.0 for name in self.task_names}
        
        elif self.config.strategy == BalancingStrategy.UNCERTAINTY_WEIGHTING:
            new_weights = self.compute_uncertainty_weights(task_losses)
        
        elif self.config.strategy == BalancingStrategy.GRADIENT_NORMALIZATION:
            if gradient_norms:
                new_weights = self.compute_gradient_normalization_weights(gradient_norms)
            else:
                new_weights = self.current_weights
        
        elif self.config.strategy == BalancingStrategy.DYNAMIC_WEIGHT_AVERAGE:
            new_weights = self.compute_dynamic_weight_average(task_losses)
        
        elif self.config.strategy == BalancingStrategy.PERFORMANCE_BASED:
            new_weights = self.compute_performance_based_weights()
        
        elif self.config.strategy == BalancingStrategy.CURRICULUM_LEARNING:
            new_weights = self.compute_curriculum_weights()
        
        elif self.config.strategy == BalancingStrategy.ADAPTIVE_LOSS_SCALING:
            new_weights = self.compute_adaptive_loss_scaling(task_losses)
        
        else:
            new_weights = self.current_weights
        
        # Update current weights and history
        self.current_weights = new_weights
        for task_name, weight in new_weights.items():
            self.weight_history[task_name].append(weight)
        
        return new_weights
    
    def step(self) -> None:
        """Increment step counter"""
        self.monitor.step()
    
    def epoch(self) -> None:
        """Increment epoch counter"""
        self.monitor.epoch()
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about weight evolution"""
        stats = {}
        
        for task_name in self.task_names:
            history = self.weight_history[task_name]
            if history:
                stats[task_name] = {
                    "current_weight": self.current_weights[task_name],
                    "average_weight": np.mean(history),
                    "weight_std": np.std(history),
                    "min_weight": min(history),
                    "max_weight": max(history),
                    "weight_trend": np.mean(history[-10:]) - np.mean(history[:10]) if len(history) >= 20 else 0.0
                }
        
        return stats


class TaskBalancer:
    """Main class for balancing multiple tasks in multi-task learning"""
    
    def __init__(self, config: BalancingConfig, task_names: List[str]):
        self.config = config
        self.task_names = task_names
        self.scheduler = DynamicWeightScheduler(config, task_names)
        self.logger = logging.getLogger(__name__)
        
    def balance_losses(self, task_losses: Dict[str, torch.Tensor],
                      gradient_norms: Dict[str, float] = None) -> torch.Tensor:
        """Balance multiple task losses into a single loss"""
        if not task_losses:
            return torch.tensor(0.0)
        
        # Convert tensors to float values for weight computation
        loss_values = {name: loss.item() if isinstance(loss, torch.Tensor) else loss 
                      for name, loss in task_losses.items()}
        
        # Update weights
        weights = self.scheduler.update_weights(loss_values, gradient_norms)
        
        # Compute weighted loss
        total_loss = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        
        for task_name, loss in task_losses.items():
            weight = weights.get(task_name, 1.0)
            weighted_loss = weight * loss
            total_loss = total_loss + weighted_loss
            
            # Log individual weighted losses
            self.logger.debug(f"Task: {task_name}, Loss: {loss.item():.4f}, "
                            f"Weight: {weight:.4f}, Weighted: {weighted_loss.item():.4f}")
        
        return total_loss
    
    def update_performance(self, task_performances: Dict[str, float]) -> None:
        """Update performance metrics for tasks"""
        for task_name, performance in task_performances.items():
            self.scheduler.monitor.update_performance(task_name, performance)
    
    def step(self) -> None:
        """Increment step counter"""
        self.scheduler.step()
    
    def epoch(self) -> None:
        """Increment epoch counter"""
        self.scheduler.epoch()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current task weights"""
        return self.scheduler.current_weights.copy()
    
    def get_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive balancing statistics"""
        weight_stats = self.scheduler.get_weight_statistics()
        
        performance_stats = {}
        for task_name in self.task_names:
            performance_stats[task_name] = {
                "average_performance": self.scheduler.monitor.get_average_performance(task_name),
                "average_loss": self.scheduler.monitor.get_average_loss(task_name),
                "performance_trend": self.scheduler.monitor.get_performance_trend(task_name),
                "gradient_norm": self.scheduler.monitor.get_gradient_norm(task_name)
            }
        
        return {
            "weight_statistics": weight_stats,
            "performance_statistics": performance_stats,
            "current_epoch": self.scheduler.monitor.current_epoch,
            "current_step": self.scheduler.monitor.current_step,
            "balancing_strategy": self.config.strategy.value
        }
    
    def save_state(self, filepath: str) -> None:
        """Save balancer state"""
        state = {
            "config": self.config,
            "current_weights": self.scheduler.current_weights,
            "weight_history": dict(self.scheduler.weight_history),
            "current_epoch": self.scheduler.monitor.current_epoch,
            "current_step": self.scheduler.monitor.current_step
        }
        
        torch.save(state, filepath)
    
    def load_state(self, filepath: str) -> None:
        """Load balancer state"""
        state = torch.load(filepath)
        
        self.scheduler.current_weights = state["current_weights"]
        self.scheduler.weight_history = defaultdict(list, state["weight_history"])
        self.scheduler.monitor.current_epoch = state["current_epoch"]
        self.scheduler.monitor.current_step = state["current_step"]


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = BalancingConfig(
        strategy=BalancingStrategy.DYNAMIC_WEIGHT_AVERAGE,
        update_frequency=50,
        temperature=2.0
    )
    
    # Task names
    task_names = ["element_detection", "ocr", "intent_classification", "action_prediction"]
    
    # Create balancer
    balancer = TaskBalancer(config, task_names)
    
    # Simulate training loop
    for step in range(1000):
        # Simulate task losses
        task_losses = {
            "element_detection": torch.tensor(np.random.uniform(0.1, 1.0)),
            "ocr": torch.tensor(np.random.uniform(0.2, 0.8)),
            "intent_classification": torch.tensor(np.random.uniform(0.15, 0.9)),
            "action_prediction": torch.tensor(np.random.uniform(0.3, 1.2))
        }
        
        # Balance losses
        total_loss = balancer.balance_losses(task_losses)
        
        # Update step
        balancer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Total Loss = {total_loss.item():.4f}")
            print(f"Current Weights: {balancer.get_current_weights()}")
    
    # Get final statistics
    stats = balancer.get_balancing_statistics()
    print("\\nFinal Statistics:")
    print(f"Weight Statistics: {stats['weight_statistics']}")