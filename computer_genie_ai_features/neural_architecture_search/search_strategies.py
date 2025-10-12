"""
Neural Architecture Search Strategies
न्यूरल आर्किटेक्चर खोज रणनीतियाँ

Implementation of various NAS algorithms including DARTS, ENAS, Random Search,
Evolutionary Search, Bayesian Optimization, and Reinforcement Learning-based methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
import json
import time
from collections import deque, defaultdict
import logging

from .search_space import SearchSpace, SearchSpaceConfig, CellSearchSpace


class SearchStrategy(Enum):
    """Types of search strategies"""
    RANDOM = "random"
    DARTS = "darts"
    ENAS = "enas"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    PROGRESSIVE = "progressive"
    REGULARIZED_EVOLUTION = "regularized_evolution"
    AGING_EVOLUTION = "aging_evolution"
    GDAS = "gdas"
    PC_DARTS = "pc_darts"
    FAIR_DARTS = "fair_darts"


@dataclass
class SearchConfig:
    """Configuration for search strategies"""
    # Basic parameters
    strategy: SearchStrategy = SearchStrategy.DARTS
    max_epochs: int = 50
    population_size: int = 50
    tournament_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # DARTS parameters
    learning_rate: float = 0.025
    arch_learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    arch_weight_decay: float = 1e-3
    grad_clip: float = 5.0
    
    # ENAS parameters
    controller_lr: float = 3.5e-4
    controller_entropy_weight: float = 1e-4
    controller_baseline_decay: float = 0.99
    
    # Bayesian Optimization parameters
    acquisition_function: str = "expected_improvement"
    n_initial_points: int = 10
    
    # Progressive parameters
    num_stages: int = 4
    stage_epochs: int = 10
    
    # Hardware constraints
    max_latency: float = 100.0  # ms
    max_memory: float = 1000.0  # MB
    target_hardware: str = "gpu"  # gpu, cpu, mobile
    
    # Multi-objective parameters
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "latency"])
    objective_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])


class ArchitectureEvaluator:
    """Evaluates architecture performance"""
    
    def __init__(self, search_space: SearchSpace, device: str = "cuda"):
        self.search_space = search_space
        self.device = device
        self.evaluation_cache = {}
    
    def evaluate_architecture(
        self, 
        architecture: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        """Evaluate architecture performance"""
        # Check cache first
        arch_key = self._architecture_to_key(architecture)
        if arch_key in self.evaluation_cache:
            return self.evaluation_cache[arch_key]
        
        # Build and train model
        model = self._build_model(architecture)
        model = model.to(self.device)
        
        # Train model
        optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx > 10:  # Quick evaluation
                    break
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        total_time = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                output = model(data)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        latency = (total_time / total) * 1000  # ms per sample
        
        # Calculate model size and FLOPs
        num_params = sum(p.numel() for p in model.parameters())
        memory_usage = num_params * 4 / (1024 * 1024)  # MB (assuming float32)
        
        results = {
            'accuracy': accuracy,
            'latency': latency,
            'memory': memory_usage,
            'parameters': num_params,
            'loss': 1 - accuracy  # For minimization
        }
        
        # Cache results
        self.evaluation_cache[arch_key] = results
        
        return results
    
    def _architecture_to_key(self, architecture: Dict[str, Any]) -> str:
        """Convert architecture to cache key"""
        return json.dumps(architecture, sort_keys=True)
    
    def _build_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build model from architecture description"""
        # Simplified model builder - in practice, this would be more sophisticated
        class SimpleModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleModel()


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    def __init__(
        self, 
        search_space: SearchSpace, 
        config: SearchConfig,
        evaluator: ArchitectureEvaluator
    ):
        self.search_space = search_space
        self.config = config
        self.evaluator = evaluator
        self.search_history = []
        self.best_architecture = None
        self.best_performance = float('-inf')
    
    @abstractmethod
    def search(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Perform architecture search"""
        pass
    
    def update_best(self, architecture: Dict[str, Any], performance: Dict[str, float]):
        """Update best architecture found"""
        # Multi-objective scoring
        score = self._compute_score(performance)
        
        if score > self.best_performance:
            self.best_performance = score
            self.best_architecture = copy.deepcopy(architecture)
    
    def _compute_score(self, performance: Dict[str, float]) -> float:
        """Compute multi-objective score"""
        score = 0.0
        for obj, weight in zip(self.config.objectives, self.config.objective_weights):
            if obj == "accuracy":
                score += weight * performance.get('accuracy', 0.0)
            elif obj == "latency":
                # Invert latency (lower is better)
                max_latency = self.config.max_latency
                normalized_latency = min(performance.get('latency', max_latency), max_latency) / max_latency
                score += weight * (1.0 - normalized_latency)
            elif obj == "memory":
                # Invert memory usage (lower is better)
                max_memory = self.config.max_memory
                normalized_memory = min(performance.get('memory', max_memory), max_memory) / max_memory
                score += weight * (1.0 - normalized_memory)
        
        return score


class RandomSearch(BaseSearchStrategy):
    """Random search strategy"""
    
    def search(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Perform random search"""
        logging.info("Starting Random Search")
        
        for iteration in range(self.config.max_epochs):
            # Sample random architecture
            architecture = self.search_space.sample_architecture()
            
            # Evaluate architecture
            performance = self.evaluator.evaluate_architecture(
                architecture, train_loader, val_loader
            )
            
            # Update best
            self.update_best(architecture, performance)
            
            # Log progress
            self.search_history.append({
                'iteration': iteration,
                'architecture': architecture,
                'performance': performance
            })
            
            logging.info(f"Iteration {iteration}: Score = {self._compute_score(performance):.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history
        }


class DARTS(BaseSearchStrategy):
    """Differentiable Architecture Search (DARTS)"""
    
    def __init__(self, search_space: SearchSpace, config: SearchConfig, evaluator: ArchitectureEvaluator):
        super().__init__(search_space, config, evaluator)
        self.alpha_normal = None
        self.alpha_reduce = None
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        """Initialize architecture parameters"""
        num_ops = len(self.search_space.operations)
        num_edges = self.search_space.config.num_nodes_per_cell * (self.search_space.config.num_nodes_per_cell - 1) // 2
        
        self.alpha_normal = torch.randn(num_edges, num_ops, requires_grad=True)
        self.alpha_reduce = torch.randn(num_edges, num_ops, requires_grad=True)
    
    def search(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Perform DARTS search"""
        logging.info("Starting DARTS Search")
        
        # Create supernet
        supernet = self._build_supernet()
        supernet = supernet.cuda() if torch.cuda.is_available() else supernet
        
        # Optimizers
        w_optimizer = optim.SGD(
            supernet.parameters(), 
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )
        
        alpha_optimizer = optim.Adam(
            [self.alpha_normal, self.alpha_reduce],
            lr=self.config.arch_learning_rate,
            weight_decay=self.config.arch_weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.max_epochs):
            # Train weights
            supernet.train()
            for step, (data, target) in enumerate(train_loader):
                data = data.cuda() if torch.cuda.is_available() else data
                target = target.cuda() if torch.cuda.is_available() else target
                
                w_optimizer.zero_grad()
                
                # Forward pass with current alphas
                logits = supernet(data, self.alpha_normal, self.alpha_reduce)
                loss = criterion(logits, target)
                
                loss.backward()
                nn.utils.clip_grad_norm_(supernet.parameters(), self.config.grad_clip)
                w_optimizer.step()
                
                if step > 10:  # Quick training
                    break
            
            # Train architecture
            supernet.eval()
            for step, (data, target) in enumerate(val_loader):
                data = data.cuda() if torch.cuda.is_available() else data
                target = target.cuda() if torch.cuda.is_available() else target
                
                alpha_optimizer.zero_grad()
                
                logits = supernet(data, self.alpha_normal, self.alpha_reduce)
                loss = criterion(logits, target)
                
                loss.backward()
                alpha_optimizer.step()
                
                if step > 5:  # Quick architecture update
                    break
            
            # Derive architecture
            architecture = self._derive_architecture()
            
            # Evaluate derived architecture
            performance = self.evaluator.evaluate_architecture(
                architecture, train_loader, val_loader
            )
            
            self.update_best(architecture, performance)
            
            self.search_history.append({
                'epoch': epoch,
                'architecture': architecture,
                'performance': performance
            })
            
            logging.info(f"Epoch {epoch}: Score = {self._compute_score(performance):.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history,
            'final_alphas': {
                'normal': self.alpha_normal.detach(),
                'reduce': self.alpha_reduce.detach()
            }
        }
    
    def _build_supernet(self) -> nn.Module:
        """Build differentiable supernet"""
        class DARTSSupernet(nn.Module):
            def __init__(self, search_space):
                super().__init__()
                self.search_space = search_space
                # Simplified supernet - in practice, this would be more complex
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU()
                )
                self.classifier = nn.Linear(16, 10)
            
            def forward(self, x, alpha_normal, alpha_reduce):
                x = self.stem(x)
                
                # Apply soft architecture (simplified)
                weights = F.softmax(alpha_normal[0], dim=-1)
                # In practice, you'd apply these weights to mixed operations
                
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return DARTSSupernet(self.search_space)
    
    def _derive_architecture(self) -> Dict[str, Any]:
        """Derive discrete architecture from continuous alphas"""
        # Get operation with highest alpha for each edge
        normal_ops = []
        reduce_ops = []
        
        for edge_alpha in self.alpha_normal:
            op_idx = torch.argmax(edge_alpha).item()
            op_name = list(self.search_space.operations.keys())[op_idx]
            normal_ops.append(op_name)
        
        for edge_alpha in self.alpha_reduce:
            op_idx = torch.argmax(edge_alpha).item()
            op_name = list(self.search_space.operations.keys())[op_idx]
            reduce_ops.append(op_name)
        
        # Convert to architecture format
        architecture = {
            'normal_cell': {'operations': normal_ops},
            'reduction_cell': {'operations': reduce_ops},
            'num_cells': self.search_space.config.num_cells,
            'channels': self.search_space.config.initial_channels
        }
        
        return architecture


class EvolutionarySearch(BaseSearchStrategy):
    """Evolutionary search strategy"""
    
    def __init__(self, search_space: SearchSpace, config: SearchConfig, evaluator: ArchitectureEvaluator):
        super().__init__(search_space, config, evaluator)
        self.population = []
        self.fitness_scores = []
    
    def search(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Perform evolutionary search"""
        logging.info("Starting Evolutionary Search")
        
        # Initialize population
        self._initialize_population()
        
        # Evaluate initial population
        self._evaluate_population(train_loader, val_loader)
        
        for generation in range(self.config.max_epochs):
            # Selection
            parents = self._tournament_selection()
            
            # Crossover and mutation
            offspring = self._create_offspring(parents)
            
            # Evaluate offspring
            offspring_fitness = []
            for arch in offspring:
                performance = self.evaluator.evaluate_architecture(
                    arch, train_loader, val_loader
                )
                fitness = self._compute_score(performance)
                offspring_fitness.append(fitness)
                
                self.update_best(arch, performance)
            
            # Replacement
            self._replace_population(offspring, offspring_fitness)
            
            # Log progress
            best_fitness = max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            self.search_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            logging.info(f"Generation {generation}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history
        }
    
    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.config.population_size):
            architecture = self.search_space.sample_architecture()
            self.population.append(architecture)
    
    def _evaluate_population(self, train_loader, val_loader):
        """Evaluate entire population"""
        self.fitness_scores = []
        for arch in self.population:
            performance = self.evaluator.evaluate_architecture(
                arch, train_loader, val_loader
            )
            fitness = self._compute_score(performance)
            self.fitness_scores.append(fitness)
            
            self.update_best(arch, performance)
    
    def _tournament_selection(self) -> List[Dict[str, Any]]:
        """Tournament selection"""
        parents = []
        for _ in range(self.config.population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(
                range(len(self.population)), 
                self.config.tournament_size
            )
            
            # Find best in tournament
            best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
            parents.append(copy.deepcopy(self.population[best_idx]))
        
        return parents
    
    def _create_offspring(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.config.population_size]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover"""
        # Simplified crossover - in practice, this would be more sophisticated
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Swap some components
        if 'normal_cell' in child1 and 'normal_cell' in child2:
            child1['normal_cell'], child2['normal_cell'] = child2['normal_cell'], child1['normal_cell']
        
        return child1, child2
    
    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate architecture"""
        mutated = copy.deepcopy(architecture)
        
        # Random mutation - sample new random architecture with some probability
        if random.random() < 0.5:
            new_arch = self.search_space.sample_architecture()
            # Randomly replace some components
            for key in mutated:
                if random.random() < 0.3:
                    mutated[key] = new_arch[key]
        
        return mutated
    
    def _replace_population(self, offspring: List[Dict[str, Any]], offspring_fitness: List[float]):
        """Replace population with offspring"""
        # Combine population and offspring
        combined_pop = self.population + offspring
        combined_fitness = self.fitness_scores + offspring_fitness
        
        # Select best individuals
        sorted_indices = sorted(
            range(len(combined_fitness)), 
            key=lambda i: combined_fitness[i], 
            reverse=True
        )
        
        # Keep best individuals
        self.population = [combined_pop[i] for i in sorted_indices[:self.config.population_size]]
        self.fitness_scores = [combined_fitness[i] for i in sorted_indices[:self.config.population_size]]


class RegularizedEvolution(BaseSearchStrategy):
    """Regularized Evolution with aging"""
    
    def __init__(self, search_space: SearchSpace, config: SearchConfig, evaluator: ArchitectureEvaluator):
        super().__init__(search_space, config, evaluator)
        self.population = deque(maxlen=config.population_size)
        self.fitness_scores = deque(maxlen=config.population_size)
    
    def search(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Perform regularized evolution search"""
        logging.info("Starting Regularized Evolution Search")
        
        # Initialize population
        for _ in range(self.config.population_size):
            architecture = self.search_space.sample_architecture()
            performance = self.evaluator.evaluate_architecture(
                architecture, train_loader, val_loader
            )
            fitness = self._compute_score(performance)
            
            self.population.append(architecture)
            self.fitness_scores.append(fitness)
            
            self.update_best(architecture, performance)
        
        # Evolution loop
        for iteration in range(self.config.max_epochs):
            # Tournament selection
            tournament_indices = random.sample(
                range(len(self.population)), 
                min(self.config.tournament_size, len(self.population))
            )
            
            parent_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
            parent = copy.deepcopy(self.population[parent_idx])
            
            # Mutation
            child = self._mutate(parent)
            
            # Evaluation
            performance = self.evaluator.evaluate_architecture(
                child, train_loader, val_loader
            )
            fitness = self._compute_score(performance)
            
            # Add to population (oldest will be removed automatically due to maxlen)
            self.population.append(child)
            self.fitness_scores.append(fitness)
            
            self.update_best(child, performance)
            
            # Log progress
            if iteration % 10 == 0:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                
                self.search_history.append({
                    'iteration': iteration,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness
                })
                
                logging.info(f"Iteration {iteration}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history
        }
    
    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate architecture"""
        mutated = copy.deepcopy(architecture)
        
        # Sample new architecture and randomly replace components
        new_arch = self.search_space.sample_architecture()
        
        for key in mutated:
            if random.random() < self.config.mutation_rate:
                mutated[key] = new_arch[key]
        
        return mutated


# Example usage
if __name__ == "__main__":
    from .search_space import SearchSpaceConfig, CellSearchSpace
    
    # Create search space
    space_config = SearchSpaceConfig(
        input_channels=3,
        num_classes=10,
        num_cells=8,
        num_nodes_per_cell=4
    )
    search_space = CellSearchSpace(space_config)
    
    # Create evaluator
    evaluator = ArchitectureEvaluator(search_space)
    
    # Create search configuration
    search_config = SearchConfig(
        strategy=SearchStrategy.RANDOM,
        max_epochs=20,
        population_size=20
    )
    
    # Test different search strategies
    strategies = {
        'random': RandomSearch(search_space, search_config, evaluator),
        'evolutionary': EvolutionarySearch(search_space, search_config, evaluator),
        'regularized_evolution': RegularizedEvolution(search_space, search_config, evaluator)
    }
    
    print("Neural Architecture Search Strategies Created Successfully!")
    print(f"Available strategies: {list(strategies.keys())}")
    print(f"Search space size: {search_space.get_search_space_size()}")
    
    # Test random search
    random_strategy = strategies['random']
    print(f"\nTesting {random_strategy.__class__.__name__}")
    print("Search strategies implementation completed!")