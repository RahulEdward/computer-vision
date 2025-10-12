"""
Client Selection Strategies for Federated Learning
फेडरेटेड लर्निंग के लिए क्लाइंट चयन रणनीतियां

Implements various client selection strategies for federated learning
to optimize convergence, fairness, and resource utilization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import random
import math
from collections import defaultdict, deque
import heapq


class SelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PROBABILISTIC = "probabilistic"
    POWER_OF_CHOICE = "power_of_choice"
    CLUSTERED = "clustered"
    GRADIENT_DIVERSITY = "gradient_diversity"
    LOSS_BASED = "loss_based"
    RESOURCE_AWARE = "resource_aware"
    FAIRNESS_AWARE = "fairness_aware"
    ADAPTIVE = "adaptive"
    MULTI_OBJECTIVE = "multi_objective"


class FairnessMetric(Enum):
    """Fairness metrics for client selection"""
    PARTICIPATION_RATE = "participation_rate"
    CONTRIBUTION_BALANCE = "contribution_balance"
    PERFORMANCE_EQUITY = "performance_equity"
    RESOURCE_UTILIZATION = "resource_utilization"


@dataclass
class ClientProfile:
    """Profile of a federated learning client"""
    client_id: str
    
    # Resource information
    compute_capacity: float = 1.0  # Relative compute power
    memory_capacity: float = 1.0   # Available memory (GB)
    bandwidth: float = 1.0         # Network bandwidth (Mbps)
    battery_level: float = 1.0     # Battery level (0-1)
    
    # Data information
    data_size: int = 0
    data_quality: float = 1.0      # Data quality score (0-1)
    label_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Performance history
    accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    training_time_history: List[float] = field(default_factory=list)
    communication_time_history: List[float] = field(default_factory=list)
    
    # Participation history
    participation_count: int = 0
    last_participation_round: int = -1
    consecutive_failures: int = 0
    reliability_score: float = 1.0
    
    # Availability
    is_available: bool = True
    availability_schedule: Dict[int, bool] = field(default_factory=dict)  # hour -> available
    timezone_offset: int = 0
    
    # Preferences
    max_local_epochs: int = 5
    preferred_batch_size: int = 32
    privacy_level: float = 1.0     # Privacy requirement (0-1)
    
    # Clustering information
    cluster_id: Optional[str] = None
    embedding: Optional[torch.Tensor] = None
    
    # Metadata
    device_type: str = "unknown"   # mobile, desktop, server, etc.
    os_type: str = "unknown"
    location: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelectionConfig:
    """Configuration for client selection"""
    # Basic settings
    strategy: SelectionStrategy = SelectionStrategy.RANDOM
    num_clients_per_round: int = 10
    min_clients_per_round: int = 5
    max_clients_per_round: int = 50
    
    # Probabilistic selection
    enable_probabilistic: bool = True
    selection_probability_base: float = 0.1
    
    # Power of choice
    power_of_choice_candidates: int = 20
    
    # Resource awareness
    min_compute_capacity: float = 0.1
    min_memory_capacity: float = 0.5
    min_bandwidth: float = 1.0
    min_battery_level: float = 0.2
    
    # Fairness settings
    enable_fairness: bool = True
    fairness_metric: FairnessMetric = FairnessMetric.PARTICIPATION_RATE
    fairness_weight: float = 0.3
    max_staleness: int = 10  # Maximum rounds without participation
    
    # Clustering settings
    enable_clustering: bool = False
    num_clusters: int = 5
    cluster_balance_weight: float = 0.2
    
    # Adaptive settings
    enable_adaptive: bool = True
    performance_window: int = 10
    adaptation_rate: float = 0.1
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ["performance", "fairness", "efficiency"])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "performance": 0.4,
        "fairness": 0.3,
        "efficiency": 0.3
    })
    
    # Constraints
    max_consecutive_selections: int = 3
    min_rounds_between_selections: int = 1
    enable_diversity_constraint: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_selection_stats: bool = True


class SelectionStats:
    """Track client selection statistics"""
    
    def __init__(self):
        self.selection_history = []
        self.client_participation = defaultdict(int)
        self.client_performance = defaultdict(list)
        self.fairness_scores = []
        self.diversity_scores = []
        self.efficiency_scores = []
        self.selection_times = []
    
    def add_selection_round(self, selected_clients: List[str], round_num: int,
                          fairness_score: float, diversity_score: float,
                          efficiency_score: float, selection_time: float) -> None:
        """Add selection round statistics"""
        self.selection_history.append({
            'round': round_num,
            'selected_clients': selected_clients.copy(),
            'num_selected': len(selected_clients),
            'timestamp': time.time()
        })
        
        # Update participation counts
        for client_id in selected_clients:
            self.client_participation[client_id] += 1
        
        # Update scores
        self.fairness_scores.append(fairness_score)
        self.diversity_scores.append(diversity_score)
        self.efficiency_scores.append(efficiency_score)
        self.selection_times.append(selection_time)
    
    def add_client_performance(self, client_id: str, accuracy: float, loss: float) -> None:
        """Add client performance data"""
        self.client_performance[client_id].append({
            'accuracy': accuracy,
            'loss': loss,
            'timestamp': time.time()
        })
    
    def get_participation_rate(self, client_id: str) -> float:
        """Get participation rate for a client"""
        total_rounds = len(self.selection_history)
        return self.client_participation[client_id] / total_rounds if total_rounds > 0 else 0.0
    
    def get_average_fairness_score(self) -> float:
        """Get average fairness score"""
        return np.mean(self.fairness_scores) if self.fairness_scores else 0.0
    
    def get_average_diversity_score(self) -> float:
        """Get average diversity score"""
        return np.mean(self.diversity_scores) if self.diversity_scores else 0.0
    
    def get_average_efficiency_score(self) -> float:
        """Get average efficiency score"""
        return np.mean(self.efficiency_scores) if self.efficiency_scores else 0.0
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selection statistics"""
        return {
            'total_rounds': len(self.selection_history),
            'average_clients_per_round': np.mean([r['num_selected'] for r in self.selection_history]) if self.selection_history else 0,
            'participation_rates': {client_id: self.get_participation_rate(client_id) 
                                  for client_id in self.client_participation},
            'fairness_score': self.get_average_fairness_score(),
            'diversity_score': self.get_average_diversity_score(),
            'efficiency_score': self.get_average_efficiency_score(),
            'average_selection_time': np.mean(self.selection_times) if self.selection_times else 0.0
        }


class ClientClusterer:
    """Cluster clients based on their characteristics"""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cluster_centers = None
        self.client_clusters = {}
    
    def cluster_clients(self, client_profiles: Dict[str, ClientProfile]) -> Dict[str, str]:
        """Cluster clients and return cluster assignments"""
        if not self.config.enable_clustering:
            return {}
        
        # Extract features for clustering
        features = []
        client_ids = []
        
        for client_id, profile in client_profiles.items():
            feature_vector = self._extract_features(profile)
            features.append(feature_vector)
            client_ids.append(client_id)
        
        if not features:
            return {}
        
        # Convert to tensor
        features_tensor = torch.stack(features)
        
        # Perform k-means clustering
        cluster_assignments = self._kmeans_clustering(features_tensor, self.config.num_clusters)
        
        # Update client profiles with cluster information
        cluster_mapping = {}
        for i, client_id in enumerate(client_ids):
            cluster_id = f"cluster_{cluster_assignments[i].item()}"
            cluster_mapping[client_id] = cluster_id
            client_profiles[client_id].cluster_id = cluster_id
        
        self.client_clusters = cluster_mapping
        return cluster_mapping
    
    def _extract_features(self, profile: ClientProfile) -> torch.Tensor:
        """Extract feature vector from client profile"""
        features = [
            profile.compute_capacity,
            profile.memory_capacity,
            profile.bandwidth,
            profile.data_size / 1000.0,  # Normalize
            profile.data_quality,
            profile.reliability_score,
            len(profile.accuracy_history),
            np.mean(profile.accuracy_history) if profile.accuracy_history else 0.0,
            np.mean(profile.loss_history) if profile.loss_history else 1.0,
            profile.participation_count / 100.0,  # Normalize
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _kmeans_clustering(self, features: torch.Tensor, k: int, max_iters: int = 100) -> torch.Tensor:
        """Simple k-means clustering implementation"""
        n_samples, n_features = features.shape
        
        # Initialize centroids randomly
        centroids = features[torch.randperm(n_samples)[:k]]
        
        for _ in range(max_iters):
            # Assign points to closest centroid
            distances = torch.cdist(features, centroids)
            assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = features[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            
            centroids = new_centroids
        
        self.cluster_centers = centroids
        return assignments
    
    def get_cluster_balance(self, selected_clients: List[str]) -> float:
        """Calculate cluster balance score for selected clients"""
        if not self.client_clusters:
            return 1.0
        
        # Count clients per cluster
        cluster_counts = defaultdict(int)
        for client_id in selected_clients:
            if client_id in self.client_clusters:
                cluster_counts[self.client_clusters[client_id]] += 1
        
        if not cluster_counts:
            return 1.0
        
        # Calculate balance (lower variance is better)
        counts = list(cluster_counts.values())
        if len(counts) <= 1:
            return 1.0
        
        mean_count = np.mean(counts)
        variance = np.var(counts)
        balance_score = 1.0 / (1.0 + variance / (mean_count + 1e-8))
        
        return balance_score


class ClientSelector:
    """Main client selection class"""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stats = SelectionStats()
        self.clusterer = ClientClusterer(config)
        
        # Selection state
        self.round_number = 0
        self.last_selected = set()
        self.selection_history = deque(maxlen=config.performance_window)
        self.client_selection_counts = defaultdict(int)
        
        # Adaptive parameters
        self.adaptive_weights = config.objective_weights.copy()
        self.performance_trends = defaultdict(list)
    
    def select_clients(self, client_profiles: Dict[str, ClientProfile], 
                      round_num: int) -> List[str]:
        """Select clients for the current round"""
        start_time = time.time()
        self.round_number = round_num
        
        # Filter available clients
        available_clients = self._filter_available_clients(client_profiles)
        
        if len(available_clients) < self.config.min_clients_per_round:
            self.logger.warning(f"Only {len(available_clients)} clients available, "
                              f"minimum required: {self.config.min_clients_per_round}")
            return list(available_clients.keys())
        
        # Update clustering if enabled
        if self.config.enable_clustering:
            self.clusterer.cluster_clients(client_profiles)
        
        # Perform selection based on strategy
        if self.config.strategy == SelectionStrategy.RANDOM:
            selected_clients = self._random_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.ROUND_ROBIN:
            selected_clients = self._round_robin_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.PROBABILISTIC:
            selected_clients = self._probabilistic_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.POWER_OF_CHOICE:
            selected_clients = self._power_of_choice_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.CLUSTERED:
            selected_clients = self._clustered_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.GRADIENT_DIVERSITY:
            selected_clients = self._gradient_diversity_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.LOSS_BASED:
            selected_clients = self._loss_based_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.RESOURCE_AWARE:
            selected_clients = self._resource_aware_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.FAIRNESS_AWARE:
            selected_clients = self._fairness_aware_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.ADAPTIVE:
            selected_clients = self._adaptive_selection(available_clients)
        elif self.config.strategy == SelectionStrategy.MULTI_OBJECTIVE:
            selected_clients = self._multi_objective_selection(available_clients)
        else:
            selected_clients = self._random_selection(available_clients)
        
        # Apply constraints
        selected_clients = self._apply_constraints(selected_clients, available_clients)
        
        # Calculate metrics
        fairness_score = self._calculate_fairness_score(selected_clients, client_profiles)
        diversity_score = self._calculate_diversity_score(selected_clients, client_profiles)
        efficiency_score = self._calculate_efficiency_score(selected_clients, client_profiles)
        
        selection_time = time.time() - start_time
        
        # Update statistics
        self.stats.add_selection_round(
            selected_clients, round_num, fairness_score, 
            diversity_score, efficiency_score, selection_time
        )
        
        # Update selection history
        self.last_selected = set(selected_clients)
        self.selection_history.append({
            'round': round_num,
            'selected': selected_clients,
            'fairness': fairness_score,
            'diversity': diversity_score,
            'efficiency': efficiency_score
        })
        
        # Update selection counts
        for client_id in selected_clients:
            self.client_selection_counts[client_id] += 1
        
        # Adaptive weight update
        if self.config.enable_adaptive:
            self._update_adaptive_weights(fairness_score, diversity_score, efficiency_score)
        
        if self.config.log_selection_stats:
            self.logger.info(f"Selected {len(selected_clients)} clients for round {round_num}")
            self.logger.info(f"Fairness: {fairness_score:.3f}, Diversity: {diversity_score:.3f}, "
                           f"Efficiency: {efficiency_score:.3f}")
        
        return selected_clients
    
    def _filter_available_clients(self, client_profiles: Dict[str, ClientProfile]) -> Dict[str, ClientProfile]:
        """Filter clients based on availability and resource constraints"""
        available_clients = {}
        
        for client_id, profile in client_profiles.items():
            # Check basic availability
            if not profile.is_available:
                continue
            
            # Check resource constraints
            if (profile.compute_capacity < self.config.min_compute_capacity or
                profile.memory_capacity < self.config.min_memory_capacity or
                profile.bandwidth < self.config.min_bandwidth or
                profile.battery_level < self.config.min_battery_level):
                continue
            
            # Check staleness constraint
            staleness = self.round_number - profile.last_participation_round
            if staleness > self.config.max_staleness and profile.last_participation_round >= 0:
                # Force include stale clients for fairness
                available_clients[client_id] = profile
                continue
            
            # Check consecutive failures
            if profile.consecutive_failures > 3:
                continue
            
            available_clients[client_id] = profile
        
        return available_clients
    
    def _random_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Random client selection"""
        client_ids = list(available_clients.keys())
        num_select = min(self.config.num_clients_per_round, len(client_ids))
        return random.sample(client_ids, num_select)
    
    def _round_robin_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Round-robin client selection"""
        client_ids = list(available_clients.keys())
        
        # Sort by last participation round (ascending)
        client_ids.sort(key=lambda cid: available_clients[cid].last_participation_round)
        
        num_select = min(self.config.num_clients_per_round, len(client_ids))
        return client_ids[:num_select]
    
    def _probabilistic_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Probabilistic client selection based on data size and performance"""
        client_ids = list(available_clients.keys())
        
        # Calculate selection probabilities
        probabilities = []
        for client_id in client_ids:
            profile = available_clients[client_id]
            
            # Base probability
            prob = self.config.selection_probability_base
            
            # Adjust based on data size
            prob *= (profile.data_size / 1000.0 + 1.0)
            
            # Adjust based on recent performance
            if profile.accuracy_history:
                recent_acc = np.mean(profile.accuracy_history[-3:])
                prob *= (recent_acc + 0.1)
            
            # Adjust based on reliability
            prob *= profile.reliability_score
            
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(client_ids)] * len(client_ids)
        
        # Sample clients
        num_select = min(self.config.num_clients_per_round, len(client_ids))
        selected_indices = np.random.choice(
            len(client_ids), size=num_select, replace=False, p=probabilities
        )
        
        return [client_ids[i] for i in selected_indices]
    
    def _power_of_choice_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Power of choice selection"""
        client_ids = list(available_clients.keys())
        num_candidates = min(self.config.power_of_choice_candidates, len(client_ids))
        num_select = min(self.config.num_clients_per_round, len(client_ids))
        
        selected_clients = []
        
        for _ in range(num_select):
            # Sample candidates
            candidates = random.sample(client_ids, min(num_candidates, len(client_ids)))
            
            # Score candidates
            best_client = None
            best_score = -float('inf')
            
            for candidate in candidates:
                if candidate in selected_clients:
                    continue
                
                score = self._calculate_client_score(available_clients[candidate])
                if score > best_score:
                    best_score = score
                    best_client = candidate
            
            if best_client:
                selected_clients.append(best_client)
                client_ids.remove(best_client)
        
        return selected_clients
    
    def _clustered_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Clustered client selection for balanced representation"""
        if not self.clusterer.client_clusters:
            return self._random_selection(available_clients)
        
        # Group clients by cluster
        cluster_clients = defaultdict(list)
        for client_id in available_clients:
            cluster_id = self.clusterer.client_clusters.get(client_id, "unknown")
            cluster_clients[cluster_id].append(client_id)
        
        # Select clients from each cluster
        selected_clients = []
        num_select = min(self.config.num_clients_per_round, len(available_clients))
        clients_per_cluster = max(1, num_select // len(cluster_clients))
        
        for cluster_id, clients in cluster_clients.items():
            if len(selected_clients) >= num_select:
                break
            
            # Select best clients from this cluster
            cluster_profiles = {cid: available_clients[cid] for cid in clients}
            cluster_selected = self._select_best_clients(cluster_profiles, clients_per_cluster)
            selected_clients.extend(cluster_selected)
        
        # Fill remaining slots if needed
        remaining_slots = num_select - len(selected_clients)
        if remaining_slots > 0:
            remaining_clients = [cid for cid in available_clients 
                               if cid not in selected_clients]
            if remaining_clients:
                additional = random.sample(remaining_clients, 
                                         min(remaining_slots, len(remaining_clients)))
                selected_clients.extend(additional)
        
        return selected_clients[:num_select]
    
    def _gradient_diversity_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Select clients to maximize gradient diversity"""
        # This is a simplified version - in practice, you'd need actual gradient information
        client_ids = list(available_clients.keys())
        
        # Use data distribution as proxy for gradient diversity
        selected_clients = []
        num_select = min(self.config.num_clients_per_round, len(client_ids))
        
        # Start with random client
        if client_ids:
            selected_clients.append(random.choice(client_ids))
            client_ids.remove(selected_clients[0])
        
        # Greedily select most diverse clients
        for _ in range(num_select - 1):
            if not client_ids:
                break
            
            best_client = None
            best_diversity = -1
            
            for candidate in client_ids:
                diversity = self._calculate_diversity_with_selected(
                    candidate, selected_clients, available_clients
                )
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_client = candidate
            
            if best_client:
                selected_clients.append(best_client)
                client_ids.remove(best_client)
        
        return selected_clients
    
    def _loss_based_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Select clients based on loss values"""
        client_scores = []
        
        for client_id, profile in available_clients.items():
            if profile.loss_history:
                # Higher loss = higher priority (more room for improvement)
                recent_loss = np.mean(profile.loss_history[-3:])
                score = recent_loss
            else:
                score = 1.0  # Default for new clients
            
            client_scores.append((score, client_id))
        
        # Sort by score (descending)
        client_scores.sort(reverse=True)
        
        num_select = min(self.config.num_clients_per_round, len(client_scores))
        return [client_id for _, client_id in client_scores[:num_select]]
    
    def _resource_aware_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Select clients based on resource availability"""
        client_scores = []
        
        for client_id, profile in available_clients.items():
            # Calculate resource score
            resource_score = (
                0.3 * profile.compute_capacity +
                0.2 * profile.memory_capacity +
                0.2 * profile.bandwidth +
                0.2 * profile.battery_level +
                0.1 * profile.reliability_score
            )
            
            client_scores.append((resource_score, client_id))
        
        # Sort by resource score (descending)
        client_scores.sort(reverse=True)
        
        num_select = min(self.config.num_clients_per_round, len(client_scores))
        return [client_id for _, client_id in client_scores[:num_select]]
    
    def _fairness_aware_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Select clients with fairness considerations"""
        client_scores = []
        
        for client_id, profile in available_clients.items():
            # Calculate fairness score (lower participation = higher priority)
            participation_rate = self.stats.get_participation_rate(client_id)
            staleness = self.round_number - profile.last_participation_round
            
            fairness_score = (
                (1.0 - participation_rate) * 0.5 +
                min(staleness / self.config.max_staleness, 1.0) * 0.3 +
                profile.data_size / 1000.0 * 0.2
            )
            
            client_scores.append((fairness_score, client_id))
        
        # Sort by fairness score (descending)
        client_scores.sort(reverse=True)
        
        num_select = min(self.config.num_clients_per_round, len(client_scores))
        return [client_id for _, client_id in client_scores[:num_select]]
    
    def _adaptive_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Adaptive selection based on recent performance"""
        # Analyze recent selection performance
        if len(self.selection_history) < 3:
            return self._random_selection(available_clients)
        
        # Calculate performance trends
        recent_fairness = np.mean([h['fairness'] for h in self.selection_history[-3:]])
        recent_diversity = np.mean([h['diversity'] for h in self.selection_history[-3:]])
        recent_efficiency = np.mean([h['efficiency'] for h in self.selection_history[-3:]])
        
        # Adapt strategy based on performance
        if recent_fairness < 0.5:
            return self._fairness_aware_selection(available_clients)
        elif recent_diversity < 0.5:
            return self._gradient_diversity_selection(available_clients)
        elif recent_efficiency < 0.5:
            return self._resource_aware_selection(available_clients)
        else:
            return self._multi_objective_selection(available_clients)
    
    def _multi_objective_selection(self, available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Multi-objective client selection"""
        client_scores = []
        
        for client_id, profile in available_clients.items():
            # Performance score
            if profile.accuracy_history:
                performance_score = np.mean(profile.accuracy_history[-3:])
            else:
                performance_score = 0.5  # Default for new clients
            
            # Fairness score
            participation_rate = self.stats.get_participation_rate(client_id)
            fairness_score = 1.0 - participation_rate
            
            # Efficiency score
            efficiency_score = (
                0.4 * profile.compute_capacity +
                0.3 * profile.reliability_score +
                0.3 * profile.bandwidth
            )
            
            # Combined score
            combined_score = (
                self.adaptive_weights["performance"] * performance_score +
                self.adaptive_weights["fairness"] * fairness_score +
                self.adaptive_weights["efficiency"] * efficiency_score
            )
            
            client_scores.append((combined_score, client_id))
        
        # Sort by combined score (descending)
        client_scores.sort(reverse=True)
        
        num_select = min(self.config.num_clients_per_round, len(client_scores))
        return [client_id for _, client_id in client_scores[:num_select]]
    
    def _apply_constraints(self, selected_clients: List[str], 
                          available_clients: Dict[str, ClientProfile]) -> List[str]:
        """Apply selection constraints"""
        # Remove clients that violate consecutive selection constraint
        if self.config.max_consecutive_selections > 0:
            filtered_clients = []
            for client_id in selected_clients:
                consecutive_count = self._get_consecutive_selections(client_id)
                if consecutive_count < self.config.max_consecutive_selections:
                    filtered_clients.append(client_id)
            selected_clients = filtered_clients
        
        # Ensure minimum rounds between selections
        if self.config.min_rounds_between_selections > 0:
            filtered_clients = []
            for client_id in selected_clients:
                profile = available_clients[client_id]
                rounds_since_last = self.round_number - profile.last_participation_round
                if rounds_since_last >= self.config.min_rounds_between_selections:
                    filtered_clients.append(client_id)
            selected_clients = filtered_clients
        
        # Ensure minimum number of clients
        if len(selected_clients) < self.config.min_clients_per_round:
            # Add more clients to meet minimum
            remaining_clients = [cid for cid in available_clients 
                               if cid not in selected_clients]
            additional_needed = self.config.min_clients_per_round - len(selected_clients)
            additional_clients = random.sample(
                remaining_clients, 
                min(additional_needed, len(remaining_clients))
            )
            selected_clients.extend(additional_clients)
        
        # Ensure maximum number of clients
        if len(selected_clients) > self.config.max_clients_per_round:
            selected_clients = selected_clients[:self.config.max_clients_per_round]
        
        return selected_clients
    
    def _get_consecutive_selections(self, client_id: str) -> int:
        """Get number of consecutive selections for a client"""
        consecutive = 0
        for i in range(len(self.selection_history) - 1, -1, -1):
            if client_id in self.selection_history[i]['selected']:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _calculate_client_score(self, profile: ClientProfile) -> float:
        """Calculate overall score for a client"""
        # Performance component
        if profile.accuracy_history:
            performance = np.mean(profile.accuracy_history[-3:])
        else:
            performance = 0.5
        
        # Resource component
        resources = (profile.compute_capacity + profile.memory_capacity + 
                    profile.bandwidth + profile.battery_level) / 4.0
        
        # Reliability component
        reliability = profile.reliability_score
        
        # Data component
        data_score = min(profile.data_size / 1000.0, 1.0)
        
        # Combined score
        score = 0.3 * performance + 0.3 * resources + 0.2 * reliability + 0.2 * data_score
        return score
    
    def _select_best_clients(self, client_profiles: Dict[str, ClientProfile], 
                           num_select: int) -> List[str]:
        """Select best clients from a subset"""
        client_scores = [(self._calculate_client_score(profile), client_id) 
                        for client_id, profile in client_profiles.items()]
        client_scores.sort(reverse=True)
        return [client_id for _, client_id in client_scores[:num_select]]
    
    def _calculate_diversity_with_selected(self, candidate: str, selected: List[str], 
                                         available_clients: Dict[str, ClientProfile]) -> float:
        """Calculate diversity of candidate with already selected clients"""
        if not selected:
            return 1.0
        
        candidate_profile = available_clients[candidate]
        
        # Calculate average similarity with selected clients
        similarities = []
        for selected_client in selected:
            selected_profile = available_clients[selected_client]
            similarity = self._calculate_client_similarity(candidate_profile, selected_profile)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity  # Higher diversity = lower similarity
        
        return diversity
    
    def _calculate_client_similarity(self, profile1: ClientProfile, profile2: ClientProfile) -> float:
        """Calculate similarity between two client profiles"""
        # Resource similarity
        resource_diff = abs(profile1.compute_capacity - profile2.compute_capacity) + \
                       abs(profile1.memory_capacity - profile2.memory_capacity) + \
                       abs(profile1.bandwidth - profile2.bandwidth)
        resource_similarity = 1.0 / (1.0 + resource_diff)
        
        # Data size similarity
        data_diff = abs(profile1.data_size - profile2.data_size) / max(profile1.data_size, profile2.data_size, 1)
        data_similarity = 1.0 - min(data_diff, 1.0)
        
        # Performance similarity
        if profile1.accuracy_history and profile2.accuracy_history:
            acc1 = np.mean(profile1.accuracy_history[-3:])
            acc2 = np.mean(profile2.accuracy_history[-3:])
            perf_similarity = 1.0 - abs(acc1 - acc2)
        else:
            perf_similarity = 0.5
        
        # Combined similarity
        similarity = (resource_similarity + data_similarity + perf_similarity) / 3.0
        return similarity
    
    def _calculate_fairness_score(self, selected_clients: List[str], 
                                 client_profiles: Dict[str, ClientProfile]) -> float:
        """Calculate fairness score for selected clients"""
        if self.config.fairness_metric == FairnessMetric.PARTICIPATION_RATE:
            # Calculate variance in participation rates
            participation_rates = [self.stats.get_participation_rate(cid) for cid in selected_clients]
            if len(participation_rates) > 1:
                variance = np.var(participation_rates)
                fairness = 1.0 / (1.0 + variance)
            else:
                fairness = 1.0
        else:
            # Default fairness calculation
            fairness = 0.5
        
        return fairness
    
    def _calculate_diversity_score(self, selected_clients: List[str], 
                                  client_profiles: Dict[str, ClientProfile]) -> float:
        """Calculate diversity score for selected clients"""
        if len(selected_clients) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(selected_clients)):
            for j in range(i + 1, len(selected_clients)):
                profile1 = client_profiles[selected_clients[i]]
                profile2 = client_profiles[selected_clients[j]]
                similarity = self._calculate_client_similarity(profile1, profile2)
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        # Add cluster balance if clustering is enabled
        if self.config.enable_clustering:
            cluster_balance = self.clusterer.get_cluster_balance(selected_clients)
            diversity = 0.7 * diversity + 0.3 * cluster_balance
        
        return diversity
    
    def _calculate_efficiency_score(self, selected_clients: List[str], 
                                   client_profiles: Dict[str, ClientProfile]) -> float:
        """Calculate efficiency score for selected clients"""
        if not selected_clients:
            return 0.0
        
        # Calculate average resource capacity
        resource_scores = []
        for client_id in selected_clients:
            profile = client_profiles[client_id]
            resource_score = (
                0.3 * profile.compute_capacity +
                0.2 * profile.memory_capacity +
                0.2 * profile.bandwidth +
                0.2 * profile.battery_level +
                0.1 * profile.reliability_score
            )
            resource_scores.append(resource_score)
        
        efficiency = np.mean(resource_scores)
        return efficiency
    
    def _update_adaptive_weights(self, fairness_score: float, diversity_score: float, 
                                efficiency_score: float) -> None:
        """Update adaptive weights based on recent performance"""
        # Simple adaptive mechanism - increase weight for underperforming objectives
        target_score = 0.7
        
        if fairness_score < target_score:
            self.adaptive_weights["fairness"] += self.config.adaptation_rate
        if diversity_score < target_score:
            self.adaptive_weights["diversity"] += self.config.adaptation_rate
        if efficiency_score < target_score:
            self.adaptive_weights["efficiency"] += self.config.adaptation_rate
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight
    
    def update_client_performance(self, client_id: str, accuracy: float, loss: float) -> None:
        """Update client performance information"""
        self.stats.add_client_performance(client_id, accuracy, loss)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selection statistics"""
        base_stats = self.stats.get_selection_statistics()
        
        # Add additional statistics
        base_stats.update({
            'adaptive_weights': self.adaptive_weights.copy(),
            'client_selection_counts': dict(self.client_selection_counts),
            'cluster_info': {
                'enabled': self.config.enable_clustering,
                'num_clusters': len(set(self.clusterer.client_clusters.values())) if self.clusterer.client_clusters else 0,
                'cluster_assignments': self.clusterer.client_clusters.copy()
            }
        })
        
        return base_stats


# Example usage
if __name__ == "__main__":
    # Create selection configuration
    config = SelectionConfig(
        strategy=SelectionStrategy.MULTI_OBJECTIVE,
        num_clients_per_round=10,
        enable_fairness=True,
        enable_clustering=True,
        enable_adaptive=True
    )
    
    # Create client selector
    selector = ClientSelector(config)
    
    print("Client selector created successfully!")
    print(f"Configuration: {config}")
    
    # Create dummy client profiles
    client_profiles = {}
    for i in range(50):
        client_profiles[f"client_{i}"] = ClientProfile(
            client_id=f"client_{i}",
            compute_capacity=random.uniform(0.5, 2.0),
            memory_capacity=random.uniform(1.0, 8.0),
            bandwidth=random.uniform(1.0, 100.0),
            battery_level=random.uniform(0.3, 1.0),
            data_size=random.randint(100, 10000),
            data_quality=random.uniform(0.7, 1.0),
            accuracy_history=[random.uniform(0.6, 0.9) for _ in range(random.randint(0, 10))],
            loss_history=[random.uniform(0.1, 1.0) for _ in range(random.randint(0, 10))],
            reliability_score=random.uniform(0.7, 1.0)
        )
    
    # Perform client selection
    selected_clients = selector.select_clients(client_profiles, round_num=1)
    print(f"Selected clients: {selected_clients}")
    
    # Get statistics
    stats = selector.get_selection_statistics()
    print(f"Selection statistics: {stats}")