"""
Aggregation Strategies for Federated Learning
फेडरेटेड लर्निंग के लिए एग्रीगेशन रणनीतियां

Implements various model aggregation strategies for federated learning
including FedAvg, FedProx, FedNova, and adaptive aggregation methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import math
from collections import defaultdict, OrderedDict
import copy


class AggregationMethod(Enum):
    """Aggregation methods for federated learning"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    FEDOPT = "fedopt"
    SCAFFOLD = "scaffold"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    BULYAN = "bulyan"
    ADAPTIVE = "adaptive"


class WeightingScheme(Enum):
    """Weighting schemes for client contributions"""
    UNIFORM = "uniform"
    DATA_SIZE = "data_size"
    LOSS_BASED = "loss_based"
    ACCURACY_BASED = "accuracy_based"
    GRADIENT_NORM = "gradient_norm"
    STALENESS = "staleness"
    ADAPTIVE = "adaptive"


@dataclass
class AggregationConfig:
    """Configuration for aggregation strategies"""
    # Basic settings
    method: AggregationMethod = AggregationMethod.FEDAVG
    weighting_scheme: WeightingScheme = WeightingScheme.DATA_SIZE
    
    # FedProx settings
    fedprox_mu: float = 0.01
    
    # FedNova settings
    fednova_tau_eff: float = 1.0
    
    # Robust aggregation settings
    trimmed_mean_beta: float = 0.1  # Fraction to trim
    krum_f: int = 2  # Number of Byzantine clients
    bulyan_f: int = 2  # Number of Byzantine clients
    
    # Adaptive settings
    enable_adaptive_weighting: bool = True
    performance_window: int = 10
    staleness_penalty: float = 0.9
    
    # Optimization settings
    server_learning_rate: float = 1.0
    server_momentum: float = 0.0
    server_optimizer: str = "sgd"  # sgd, adam, adagrad
    
    # Quality control
    min_client_weight: float = 0.01
    max_client_weight: float = 10.0
    gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    
    # Monitoring
    enable_monitoring: bool = True
    log_aggregation_stats: bool = True


@dataclass
class ClientUpdate:
    """Represents an update from a client"""
    client_id: str
    model_state: Dict[str, torch.Tensor]
    data_size: int
    loss: float
    accuracy: float
    gradient_norm: float
    local_epochs: int
    staleness: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of model aggregation"""
    aggregated_model: Dict[str, torch.Tensor]
    client_weights: Dict[str, float]
    aggregation_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class AggregationStats:
    """Track aggregation statistics"""
    
    def __init__(self):
        self.aggregation_times = []
        self.client_participation = defaultdict(int)
        self.weight_distributions = []
        self.quality_scores = []
        self.convergence_metrics = []
    
    def add_aggregation_stat(self, aggregation_time: float, client_weights: Dict[str, float],
                           quality_score: float, convergence_metric: float) -> None:
        """Add aggregation statistics"""
        self.aggregation_times.append(aggregation_time)
        self.weight_distributions.append(client_weights.copy())
        self.quality_scores.append(quality_score)
        self.convergence_metrics.append(convergence_metric)
        
        # Update participation
        for client_id in client_weights:
            self.client_participation[client_id] += 1
    
    def get_average_aggregation_time(self) -> float:
        """Get average aggregation time"""
        return np.mean(self.aggregation_times) if self.aggregation_times else 0.0
    
    def get_client_participation_rate(self, client_id: str) -> float:
        """Get participation rate for a client"""
        total_rounds = len(self.weight_distributions)
        return self.client_participation[client_id] / total_rounds if total_rounds > 0 else 0.0
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get weight distribution statistics"""
        if not self.weight_distributions:
            return {}
        
        all_weights = []
        for weight_dist in self.weight_distributions:
            all_weights.extend(weight_dist.values())
        
        return {
            'mean_weight': np.mean(all_weights),
            'std_weight': np.std(all_weights),
            'min_weight': np.min(all_weights),
            'max_weight': np.max(all_weights)
        }


class WeightCalculator:
    """Calculate client weights based on various schemes"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_history = defaultdict(list)
    
    def calculate_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Calculate weights for client updates"""
        if self.config.weighting_scheme == WeightingScheme.UNIFORM:
            return self._uniform_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.DATA_SIZE:
            return self._data_size_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.LOSS_BASED:
            return self._loss_based_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.ACCURACY_BASED:
            return self._accuracy_based_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.GRADIENT_NORM:
            return self._gradient_norm_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.STALENESS:
            return self._staleness_weights(client_updates)
        elif self.config.weighting_scheme == WeightingScheme.ADAPTIVE:
            return self._adaptive_weights(client_updates)
        else:
            return self._data_size_weights(client_updates)
    
    def _uniform_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Uniform weighting"""
        weight = 1.0 / len(client_updates)
        return {update.client_id: weight for update in client_updates}
    
    def _data_size_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Weight by data size"""
        total_data = sum(update.data_size for update in client_updates)
        weights = {}
        
        for update in client_updates:
            weight = update.data_size / total_data
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _loss_based_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Weight inversely by loss (lower loss = higher weight)"""
        # Invert losses and add small epsilon to avoid division by zero
        epsilon = 1e-8
        inv_losses = [1.0 / (update.loss + epsilon) for update in client_updates]
        total_inv_loss = sum(inv_losses)
        
        weights = {}
        for i, update in enumerate(client_updates):
            weight = inv_losses[i] / total_inv_loss
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _accuracy_based_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Weight by accuracy"""
        total_accuracy = sum(update.accuracy for update in client_updates)
        weights = {}
        
        for update in client_updates:
            weight = update.accuracy / total_accuracy if total_accuracy > 0 else 1.0 / len(client_updates)
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _gradient_norm_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Weight by gradient norm"""
        total_grad_norm = sum(update.gradient_norm for update in client_updates)
        weights = {}
        
        for update in client_updates:
            weight = update.gradient_norm / total_grad_norm if total_grad_norm > 0 else 1.0 / len(client_updates)
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _staleness_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Weight with staleness penalty"""
        weights = {}
        
        for update in client_updates:
            # Apply exponential decay based on staleness
            staleness_factor = self.config.staleness_penalty ** update.staleness
            base_weight = update.data_size  # Use data size as base
            weight = base_weight * staleness_factor
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _adaptive_weights(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Adaptive weighting based on historical performance"""
        weights = {}
        
        for update in client_updates:
            # Update performance history
            self.performance_history[update.client_id].append({
                'accuracy': update.accuracy,
                'loss': update.loss,
                'data_size': update.data_size,
                'gradient_norm': update.gradient_norm
            })
            
            # Keep only recent history
            if len(self.performance_history[update.client_id]) > self.config.performance_window:
                self.performance_history[update.client_id].pop(0)
            
            # Calculate adaptive weight
            weight = self._calculate_adaptive_weight(update)
            weights[update.client_id] = weight
        
        return self._normalize_weights(weights)
    
    def _calculate_adaptive_weight(self, update: ClientUpdate) -> float:
        """Calculate adaptive weight for a client"""
        history = self.performance_history[update.client_id]
        
        if len(history) < 2:
            # Not enough history, use data size
            return update.data_size
        
        # Calculate trends
        recent_accuracies = [h['accuracy'] for h in history[-3:]]
        recent_losses = [h['loss'] for h in history[-3:]]
        
        # Accuracy trend (positive is good)
        acc_trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0] if len(recent_accuracies) > 1 else 0
        
        # Loss trend (negative is good)
        loss_trend = -np.polyfit(range(len(recent_losses)), recent_losses, 1)[0] if len(recent_losses) > 1 else 0
        
        # Combine trends with current performance
        performance_score = (
            0.4 * update.accuracy +
            0.2 * (1.0 / (update.loss + 1e-8)) +
            0.2 * max(0, acc_trend) +
            0.2 * max(0, loss_trend)
        )
        
        # Combine with data size
        adaptive_weight = update.data_size * (1.0 + performance_score)
        
        return adaptive_weight
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize and clip weights"""
        # Clip weights
        for client_id in weights:
            weights[client_id] = max(self.config.min_client_weight, 
                                   min(self.config.max_client_weight, weights[client_id]))
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for client_id in weights:
                weights[client_id] /= total_weight
        else:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(weights)
            for client_id in weights:
                weights[client_id] = uniform_weight
        
        return weights


class FederatedAggregator:
    """Main federated aggregation class"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.weight_calculator = WeightCalculator(config)
        self.stats = AggregationStats()
        
        # Server optimizer state
        self.server_optimizer_state = {}
        self.global_model_state = None
        
        # SCAFFOLD control variates
        self.server_control = None
        self.client_controls = {}
    
    def aggregate(self, client_updates: List[ClientUpdate]) -> AggregationResult:
        """Aggregate client updates"""
        start_time = time.time()
        
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate client weights
        client_weights = self.weight_calculator.calculate_weights(client_updates)
        
        # Apply gradient clipping if enabled
        if self.config.gradient_clipping:
            client_updates = self._clip_gradients(client_updates)
        
        # Perform aggregation based on method
        if self.config.method == AggregationMethod.FEDAVG:
            aggregated_model = self._fedavg_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.FEDPROX:
            aggregated_model = self._fedprox_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.FEDNOVA:
            aggregated_model = self._fednova_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.FEDOPT:
            aggregated_model = self._fedopt_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.SCAFFOLD:
            aggregated_model = self._scaffold_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.WEIGHTED_AVERAGE:
            aggregated_model = self._weighted_average_aggregation(client_updates, client_weights)
        elif self.config.method == AggregationMethod.MEDIAN:
            aggregated_model = self._median_aggregation(client_updates)
        elif self.config.method == AggregationMethod.TRIMMED_MEAN:
            aggregated_model = self._trimmed_mean_aggregation(client_updates)
        elif self.config.method == AggregationMethod.KRUM:
            aggregated_model = self._krum_aggregation(client_updates)
        elif self.config.method == AggregationMethod.BULYAN:
            aggregated_model = self._bulyan_aggregation(client_updates)
        elif self.config.method == AggregationMethod.ADAPTIVE:
            aggregated_model = self._adaptive_aggregation(client_updates, client_weights)
        else:
            aggregated_model = self._fedavg_aggregation(client_updates, client_weights)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(client_updates, aggregated_model)
        
        # Calculate aggregation stats
        aggregation_stats = self._calculate_aggregation_stats(client_updates, client_weights)
        
        aggregation_time = time.time() - start_time
        
        # Update statistics
        quality_score = quality_metrics.get('consistency_score', 0.0)
        convergence_metric = quality_metrics.get('convergence_rate', 0.0)
        self.stats.add_aggregation_stat(aggregation_time, client_weights, quality_score, convergence_metric)
        
        # Store global model state
        self.global_model_state = aggregated_model.copy()
        
        if self.config.log_aggregation_stats:
            self.logger.info(f"Aggregated {len(client_updates)} client updates in {aggregation_time:.3f}s")
            self.logger.info(f"Quality score: {quality_score:.3f}, Convergence rate: {convergence_metric:.3f}")
        
        return AggregationResult(
            aggregated_model=aggregated_model,
            client_weights=client_weights,
            aggregation_stats=aggregation_stats,
            quality_metrics=quality_metrics
        )
    
    def _clip_gradients(self, client_updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Apply gradient clipping to client updates"""
        clipped_updates = []
        
        for update in client_updates:
            if self.global_model_state is not None:
                # Calculate gradients (difference from global model)
                gradients = {}
                total_norm = 0.0
                
                for name, param in update.model_state.items():
                    if name in self.global_model_state:
                        grad = param - self.global_model_state[name]
                        gradients[name] = grad
                        total_norm += grad.norm().item() ** 2
                
                total_norm = math.sqrt(total_norm)
                
                # Apply clipping if needed
                if total_norm > self.config.gradient_clip_norm:
                    clip_coef = self.config.gradient_clip_norm / total_norm
                    
                    clipped_model_state = {}
                    for name, param in update.model_state.items():
                        if name in gradients:
                            clipped_grad = gradients[name] * clip_coef
                            clipped_model_state[name] = self.global_model_state[name] + clipped_grad
                        else:
                            clipped_model_state[name] = param
                    
                    # Create clipped update
                    clipped_update = copy.deepcopy(update)
                    clipped_update.model_state = clipped_model_state
                    clipped_updates.append(clipped_update)
                else:
                    clipped_updates.append(update)
            else:
                clipped_updates.append(update)
        
        return clipped_updates
    
    def _fedavg_aggregation(self, client_updates: List[ClientUpdate], 
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation"""
        aggregated_model = {}
        
        # Get parameter names from first client
        param_names = list(client_updates[0].model_state.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update in client_updates:
                if param_name in update.model_state:
                    weight = client_weights[update.client_id]
                    weighted_param = update.model_state[param_name] * weight
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            aggregated_model[param_name] = weighted_sum
        
        return aggregated_model
    
    def _fedprox_aggregation(self, client_updates: List[ClientUpdate], 
                            client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term"""
        # FedProx is similar to FedAvg but with proximal regularization during local training
        # The aggregation step is the same as FedAvg
        return self._fedavg_aggregation(client_updates, client_weights)
    
    def _fednova_aggregation(self, client_updates: List[ClientUpdate], 
                            client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """FedNova aggregation with normalized averaging"""
        aggregated_model = {}
        param_names = list(client_updates[0].model_state.keys())
        
        # Calculate effective local steps
        total_tau = sum(update.local_epochs for update in client_updates)
        
        for param_name in param_names:
            weighted_sum = None
            
            for update in client_updates:
                if param_name in update.model_state:
                    # FedNova normalization
                    tau_i = update.local_epochs
                    weight = client_weights[update.client_id] * (tau_i / total_tau)
                    weighted_param = update.model_state[param_name] * weight
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            aggregated_model[param_name] = weighted_sum
        
        return aggregated_model
    
    def _fedopt_aggregation(self, client_updates: List[ClientUpdate], 
                           client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """FedOpt aggregation with server optimizer"""
        # First get FedAvg result
        fedavg_model = self._fedavg_aggregation(client_updates, client_weights)
        
        if self.global_model_state is None:
            return fedavg_model
        
        # Calculate pseudo-gradient
        pseudo_gradient = {}
        for param_name in fedavg_model:
            if param_name in self.global_model_state:
                pseudo_gradient[param_name] = self.global_model_state[param_name] - fedavg_model[param_name]
        
        # Apply server optimizer
        if self.config.server_optimizer == "sgd":
            return self._apply_sgd_server_optimizer(pseudo_gradient)
        elif self.config.server_optimizer == "adam":
            return self._apply_adam_server_optimizer(pseudo_gradient)
        elif self.config.server_optimizer == "adagrad":
            return self._apply_adagrad_server_optimizer(pseudo_gradient)
        else:
            return fedavg_model
    
    def _scaffold_aggregation(self, client_updates: List[ClientUpdate], 
                             client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates"""
        # Initialize server control if not exists
        if self.server_control is None and self.global_model_state is not None:
            self.server_control = {name: torch.zeros_like(param) 
                                 for name, param in self.global_model_state.items()}
        
        # Standard FedAvg aggregation
        aggregated_model = self._fedavg_aggregation(client_updates, client_weights)
        
        # Update server control variate
        if self.server_control is not None:
            for param_name in aggregated_model:
                if param_name in self.server_control:
                    # Update server control (simplified version)
                    delta = aggregated_model[param_name] - self.global_model_state.get(param_name, 
                                                                                      torch.zeros_like(aggregated_model[param_name]))
                    self.server_control[param_name] += delta * self.config.server_learning_rate
        
        return aggregated_model
    
    def _weighted_average_aggregation(self, client_updates: List[ClientUpdate], 
                                     client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Simple weighted average aggregation"""
        return self._fedavg_aggregation(client_updates, client_weights)
    
    def _median_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation"""
        aggregated_model = {}
        param_names = list(client_updates[0].model_state.keys())
        
        for param_name in param_names:
            # Collect all parameters for this layer
            param_tensors = []
            for update in client_updates:
                if param_name in update.model_state:
                    param_tensors.append(update.model_state[param_name])
            
            if param_tensors:
                # Stack tensors and compute median
                stacked = torch.stack(param_tensors, dim=0)
                median_param = torch.median(stacked, dim=0)[0]
                aggregated_model[param_name] = median_param
        
        return aggregated_model
    
    def _trimmed_mean_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation"""
        aggregated_model = {}
        param_names = list(client_updates[0].model_state.keys())
        
        # Calculate number of clients to trim
        num_clients = len(client_updates)
        num_trim = int(num_clients * self.config.trimmed_mean_beta)
        
        for param_name in param_names:
            # Collect all parameters for this layer
            param_tensors = []
            for update in client_updates:
                if param_name in update.model_state:
                    param_tensors.append(update.model_state[param_name])
            
            if param_tensors:
                # Stack tensors
                stacked = torch.stack(param_tensors, dim=0)
                
                # Sort and trim
                sorted_tensors, _ = torch.sort(stacked, dim=0)
                if num_trim > 0:
                    trimmed = sorted_tensors[num_trim:-num_trim] if num_trim < num_clients // 2 else sorted_tensors
                else:
                    trimmed = sorted_tensors
                
                # Compute mean
                mean_param = torch.mean(trimmed, dim=0)
                aggregated_model[param_name] = mean_param
        
        return aggregated_model
    
    def _krum_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Krum aggregation for Byzantine robustness"""
        num_clients = len(client_updates)
        f = self.config.krum_f  # Number of Byzantine clients
        
        if num_clients <= 2 * f:
            # Fallback to median if not enough clients
            return self._median_aggregation(client_updates)
        
        # Calculate pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = self._calculate_model_distance(
                    client_updates[i].model_state,
                    client_updates[j].model_state
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Calculate Krum scores
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Sum of distances to closest n-f-1 clients
            closest_distances, _ = torch.topk(distances[i], num_clients - f - 1, largest=False)
            scores[i] = torch.sum(closest_distances)
        
        # Select client with minimum score
        selected_idx = torch.argmin(scores).item()
        return client_updates[selected_idx].model_state
    
    def _bulyan_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Bulyan aggregation for Byzantine robustness"""
        num_clients = len(client_updates)
        f = self.config.bulyan_f
        
        if num_clients <= 4 * f:
            # Fallback to trimmed mean
            return self._trimmed_mean_aggregation(client_updates)
        
        # Select 2f+1 clients using Krum
        selected_updates = []
        remaining_updates = client_updates.copy()
        
        for _ in range(2 * f + 1):
            if not remaining_updates:
                break
            
            # Run Krum on remaining clients
            krum_result = self._krum_aggregation(remaining_updates)
            
            # Find which client was selected
            selected_client = None
            for i, update in enumerate(remaining_updates):
                if self._models_equal(update.model_state, krum_result):
                    selected_client = i
                    break
            
            if selected_client is not None:
                selected_updates.append(remaining_updates.pop(selected_client))
        
        # Apply trimmed mean to selected clients
        if selected_updates:
            return self._trimmed_mean_aggregation(selected_updates)
        else:
            return self._median_aggregation(client_updates)
    
    def _adaptive_aggregation(self, client_updates: List[ClientUpdate], 
                             client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Adaptive aggregation based on client performance"""
        # Start with FedAvg
        base_result = self._fedavg_aggregation(client_updates, client_weights)
        
        # Calculate variance in client updates
        variance_score = self._calculate_update_variance(client_updates)
        
        # If high variance, use robust aggregation
        if variance_score > 0.5:  # Threshold for switching to robust method
            self.logger.info("High variance detected, switching to trimmed mean")
            return self._trimmed_mean_aggregation(client_updates)
        else:
            return base_result
    
    def _apply_sgd_server_optimizer(self, pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply SGD server optimizer"""
        updated_model = {}
        
        for param_name, grad in pseudo_gradient.items():
            if param_name in self.global_model_state:
                # SGD update with momentum
                if param_name not in self.server_optimizer_state:
                    self.server_optimizer_state[param_name] = torch.zeros_like(grad)
                
                momentum = self.server_optimizer_state[param_name]
                momentum = self.config.server_momentum * momentum + grad
                self.server_optimizer_state[param_name] = momentum
                
                updated_model[param_name] = (self.global_model_state[param_name] - 
                                           self.config.server_learning_rate * momentum)
        
        return updated_model
    
    def _apply_adam_server_optimizer(self, pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Adam server optimizer"""
        updated_model = {}
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        for param_name, grad in pseudo_gradient.items():
            if param_name in self.global_model_state:
                # Initialize Adam state
                if param_name not in self.server_optimizer_state:
                    self.server_optimizer_state[param_name] = {
                        'm': torch.zeros_like(grad),
                        'v': torch.zeros_like(grad),
                        't': 0
                    }
                
                state = self.server_optimizer_state[param_name]
                state['t'] += 1
                
                # Update biased first and second moment estimates
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad.pow(2)
                
                # Bias correction
                m_hat = state['m'] / (1 - beta1 ** state['t'])
                v_hat = state['v'] / (1 - beta2 ** state['t'])
                
                # Update parameters
                updated_model[param_name] = (self.global_model_state[param_name] - 
                                           self.config.server_learning_rate * m_hat / (v_hat.sqrt() + eps))
        
        return updated_model
    
    def _apply_adagrad_server_optimizer(self, pseudo_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Adagrad server optimizer"""
        updated_model = {}
        eps = 1e-8
        
        for param_name, grad in pseudo_gradient.items():
            if param_name in self.global_model_state:
                # Initialize Adagrad state
                if param_name not in self.server_optimizer_state:
                    self.server_optimizer_state[param_name] = torch.zeros_like(grad)
                
                # Accumulate squared gradients
                self.server_optimizer_state[param_name] += grad.pow(2)
                
                # Update parameters
                adapted_lr = self.config.server_learning_rate / (self.server_optimizer_state[param_name].sqrt() + eps)
                updated_model[param_name] = self.global_model_state[param_name] - adapted_lr * grad
        
        return updated_model
    
    def _calculate_model_distance(self, model1: Dict[str, torch.Tensor], 
                                 model2: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 distance between two models"""
        total_distance = 0.0
        
        for param_name in model1:
            if param_name in model2:
                diff = model1[param_name] - model2[param_name]
                total_distance += torch.norm(diff).item() ** 2
        
        return math.sqrt(total_distance)
    
    def _models_equal(self, model1: Dict[str, torch.Tensor], 
                     model2: Dict[str, torch.Tensor], tolerance: float = 1e-6) -> bool:
        """Check if two models are approximately equal"""
        for param_name in model1:
            if param_name not in model2:
                return False
            
            diff = torch.norm(model1[param_name] - model2[param_name]).item()
            if diff > tolerance:
                return False
        
        return True
    
    def _calculate_update_variance(self, client_updates: List[ClientUpdate]) -> float:
        """Calculate variance in client updates"""
        if len(client_updates) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(client_updates)):
            for j in range(i + 1, len(client_updates)):
                dist = self._calculate_model_distance(
                    client_updates[i].model_state,
                    client_updates[j].model_state
                )
                distances.append(dist)
        
        # Return normalized variance
        if distances:
            return np.std(distances) / (np.mean(distances) + 1e-8)
        else:
            return 0.0
    
    def _calculate_quality_metrics(self, client_updates: List[ClientUpdate], 
                                  aggregated_model: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate quality metrics for aggregation"""
        metrics = {}
        
        # Consistency score (how similar client updates are)
        if len(client_updates) > 1:
            pairwise_distances = []
            for i in range(len(client_updates)):
                for j in range(i + 1, len(client_updates)):
                    dist = self._calculate_model_distance(
                        client_updates[i].model_state,
                        client_updates[j].model_state
                    )
                    pairwise_distances.append(dist)
            
            avg_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
            metrics['consistency_score'] = 1.0 / (1.0 + avg_distance)  # Higher is better
        else:
            metrics['consistency_score'] = 1.0
        
        # Convergence rate (based on gradient norms)
        gradient_norms = [update.gradient_norm for update in client_updates]
        avg_gradient_norm = np.mean(gradient_norms)
        metrics['convergence_rate'] = 1.0 / (1.0 + avg_gradient_norm)  # Higher is better
        
        # Participation diversity
        data_sizes = [update.data_size for update in client_updates]
        if len(set(data_sizes)) > 1:
            metrics['diversity_score'] = np.std(data_sizes) / (np.mean(data_sizes) + 1e-8)
        else:
            metrics['diversity_score'] = 0.0
        
        return metrics
    
    def _calculate_aggregation_stats(self, client_updates: List[ClientUpdate], 
                                   client_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate aggregation statistics"""
        stats = {
            'num_clients': len(client_updates),
            'total_data_size': sum(update.data_size for update in client_updates),
            'avg_loss': np.mean([update.loss for update in client_updates]),
            'avg_accuracy': np.mean([update.accuracy for update in client_updates]),
            'weight_distribution': client_weights.copy(),
            'staleness_stats': {
                'avg_staleness': np.mean([update.staleness for update in client_updates]),
                'max_staleness': max(update.staleness for update in client_updates),
                'min_staleness': min(update.staleness for update in client_updates)
            }
        }
        
        return stats
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            'average_aggregation_time': self.stats.get_average_aggregation_time(),
            'client_participation': dict(self.stats.client_participation),
            'weight_statistics': self.stats.get_weight_statistics(),
            'quality_scores': {
                'mean': np.mean(self.stats.quality_scores) if self.stats.quality_scores else 0.0,
                'std': np.std(self.stats.quality_scores) if self.stats.quality_scores else 0.0
            },
            'convergence_metrics': {
                'mean': np.mean(self.stats.convergence_metrics) if self.stats.convergence_metrics else 0.0,
                'std': np.std(self.stats.convergence_metrics) if self.stats.convergence_metrics else 0.0
            }
        }


# Example usage
if __name__ == "__main__":
    # Create aggregation configuration
    config = AggregationConfig(
        method=AggregationMethod.FEDAVG,
        weighting_scheme=WeightingScheme.ADAPTIVE,
        enable_adaptive_weighting=True,
        gradient_clipping=True
    )
    
    # Create aggregator
    aggregator = FederatedAggregator(config)
    
    print("Federated aggregator created successfully!")
    print(f"Configuration: {config}")
    
    # Test with dummy client updates
    dummy_updates = [
        ClientUpdate(
            client_id=f"client_{i}",
            model_state={
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10)
            },
            data_size=100 + i * 50,
            loss=0.5 + i * 0.1,
            accuracy=0.8 - i * 0.05,
            gradient_norm=1.0 + i * 0.2,
            local_epochs=5
        )
        for i in range(3)
    ]
    
    # Perform aggregation
    result = aggregator.aggregate(dummy_updates)
    print(f"Aggregation completed!")
    print(f"Client weights: {result.client_weights}")
    print(f"Quality metrics: {result.quality_metrics}")
    
    # Get statistics
    stats = aggregator.get_aggregation_stats()
    print(f"Aggregation statistics: {stats}")