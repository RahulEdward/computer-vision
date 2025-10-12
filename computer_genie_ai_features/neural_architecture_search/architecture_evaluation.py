"""
Architecture Evaluation Module
आर्किटेक्चर मूल्यांकन मॉड्यूल

Comprehensive evaluation framework for neural architectures including accuracy assessment,
robustness testing, efficiency analysis, and comparative benchmarking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import json
import logging
from collections import defaultdict

# Optional dependencies for visualization and metrics
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    accuracy_score = None
    precision_recall_fscore_support = None
    confusion_matrix = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .search_space import SearchSpace
from .hardware_aware import HardwarePredictor, HardwareConstraints


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    TOP5_ACCURACY = "top5_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOSS = "loss"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ENERGY = "energy"
    FLOPS = "flops"
    PARAMETERS = "parameters"
    MODEL_SIZE = "model_size"
    ROBUSTNESS = "robustness"
    CALIBRATION = "calibration"


class EvaluationMode(Enum):
    """Evaluation modes"""
    QUICK = "quick"          # Fast evaluation with subset
    STANDARD = "standard"    # Standard evaluation
    COMPREHENSIVE = "comprehensive"  # Full evaluation with all metrics
    BENCHMARK = "benchmark"  # Benchmark comparison


@dataclass
class EvaluationConfig:
    """Configuration for architecture evaluation"""
    # Evaluation settings
    mode: EvaluationMode = EvaluationMode.STANDARD
    metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY,
        EvaluationMetric.LOSS,
        EvaluationMetric.LATENCY,
        EvaluationMetric.PARAMETERS
    ])
    
    # Dataset settings
    batch_size: int = 32
    num_workers: int = 4
    subset_ratio: float = 0.1  # For quick evaluation
    
    # Training settings
    epochs: int = 50
    learning_rate: float = 0.025
    weight_decay: float = 3e-4
    momentum: float = 0.9
    
    # Hardware evaluation
    hardware_constraints: Optional[HardwareConstraints] = None
    
    # Robustness testing
    test_robustness: bool = False
    noise_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    
    # Calibration testing
    test_calibration: bool = False
    
    # Visualization
    save_plots: bool = True
    plot_dir: str = "evaluation_plots"


@dataclass
class EvaluationResult:
    """Results from architecture evaluation"""
    architecture_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    hardware_metrics: Dict[str, float] = field(default_factory=dict)
    robustness_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_time: float = 0.0
    model_state: Optional[Dict[str, Any]] = None


class ArchitectureTrainer:
    """Trains architectures for evaluation"""
    
    def __init__(self, config: EvaluationConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_architecture(
        self, 
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        architecture_id: str
    ) -> Dict[str, List[float]]:
        """Train architecture and return training history"""
        model = model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train: {train_loss:.4f}/{train_acc:.4f} - "
                    f"Val: {val_loss:.4f}/{val_acc:.4f}"
                )
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return history
    
    def _train_epoch(
        self, 
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train single epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self, 
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate single epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy


class RobustnessEvaluator:
    """Evaluates model robustness to various perturbations"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def evaluate_noise_robustness(
        self, 
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        noise_levels: List[float]
    ) -> Dict[str, float]:
        """Evaluate robustness to Gaussian noise"""
        model.eval()
        robustness_metrics = {}
        
        # Clean accuracy
        clean_acc = self._evaluate_accuracy(model, test_loader)
        robustness_metrics['clean_accuracy'] = clean_acc
        
        # Noisy accuracy
        for noise_level in noise_levels:
            noisy_acc = self._evaluate_noisy_accuracy(model, test_loader, noise_level)
            robustness_metrics[f'noise_{noise_level}_accuracy'] = noisy_acc
            robustness_metrics[f'noise_{noise_level}_drop'] = clean_acc - noisy_acc
        
        # Average robustness
        avg_drop = np.mean([
            robustness_metrics[f'noise_{noise}_drop'] 
            for noise in noise_levels
        ])
        robustness_metrics['average_robustness_drop'] = avg_drop
        
        return robustness_metrics
    
    def _evaluate_accuracy(
        self, 
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate clean accuracy"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
    
    def _evaluate_noisy_accuracy(
        self, 
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        noise_level: float
    ) -> float:
        """Evaluate accuracy with Gaussian noise"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Add Gaussian noise
                noise = torch.randn_like(data) * noise_level
                noisy_data = data + noise
                noisy_data = torch.clamp(noisy_data, 0, 1)
                
                output = model(noisy_data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total


class CalibrationEvaluator:
    """Evaluates model calibration"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def evaluate_calibration(
        self, 
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """Evaluate model calibration using reliability diagrams"""
        model.eval()
        
        all_confidences = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Get probabilities and predictions
                probs = F.softmax(output, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                all_confidences.extend(confidences.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_confidences = np.array(all_confidences)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate calibration metrics
        ece = self._expected_calibration_error(
            all_confidences, all_predictions, all_targets, num_bins
        )
        
        mce = self._maximum_calibration_error(
            all_confidences, all_predictions, all_targets, num_bins
        )
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce
        }
    
    def _expected_calibration_error(
        self, 
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_bins: int
    ) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(
        self, 
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_bins: int
    ) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce


class ArchitectureEvaluator:
    """Main architecture evaluation class"""
    
    def __init__(self, config: EvaluationConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Initialize components
        self.trainer = ArchitectureTrainer(config, device)
        self.robustness_evaluator = RobustnessEvaluator(device)
        self.calibration_evaluator = CalibrationEvaluator(device)
        
        # Hardware predictor (if constraints provided)
        self.hardware_predictor = None
        if config.hardware_constraints:
            from .hardware_aware import OperationProfiler, HardwarePredictor
            profiler = OperationProfiler(config.hardware_constraints.hardware_type)
            self.hardware_predictor = HardwarePredictor(profiler)
        
        # Results storage
        self.evaluation_results = []
    
    def evaluate_architecture(
        self,
        model: nn.Module,
        architecture: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        architecture_id: str
    ) -> EvaluationResult:
        """Comprehensive architecture evaluation"""
        start_time = time.time()
        
        result = EvaluationResult(architecture_id=architecture_id)
        
        # 1. Train the model
        if self.config.mode != EvaluationMode.QUICK:
            training_history = self.trainer.train_architecture(
                model, train_loader, val_loader, architecture_id
            )
            result.training_history = training_history
        
        # 2. Evaluate basic metrics
        basic_metrics = self._evaluate_basic_metrics(model, test_loader)
        result.metrics.update(basic_metrics)
        
        # 3. Hardware evaluation
        if self.hardware_predictor:
            hw_metrics = self.hardware_predictor.predict_metrics(architecture)
            result.hardware_metrics = hw_metrics
            result.metrics.update({f"hw_{k}": v for k, v in hw_metrics.items()})
        
        # 4. Robustness evaluation
        if self.config.test_robustness:
            robustness_metrics = self.robustness_evaluator.evaluate_noise_robustness(
                model, test_loader, self.config.noise_levels
            )
            result.robustness_metrics = robustness_metrics
            result.metrics.update({f"robust_{k}": v for k, v in robustness_metrics.items()})
        
        # 5. Calibration evaluation
        if self.config.test_calibration:
            calibration_metrics = self.calibration_evaluator.evaluate_calibration(
                model, test_loader
            )
            result.calibration_metrics = calibration_metrics
            result.metrics.update({f"calib_{k}": v for k, v in calibration_metrics.items()})
        
        # 6. Model complexity metrics
        complexity_metrics = self._evaluate_complexity(model)
        result.metrics.update(complexity_metrics)
        
        # Record evaluation time
        result.evaluation_time = time.time() - start_time
        
        # Save model state if needed
        if self.config.mode == EvaluationMode.COMPREHENSIVE:
            result.model_state = model.state_dict()
        
        # Store result
        self.evaluation_results.append(result)
        
        return result
    
    def _evaluate_basic_metrics(
        self, 
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate basic accuracy and loss metrics"""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Top-1 accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, dim=1)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
                
                total += target.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': 100.0 * correct / total,
            'top5_accuracy': 100.0 * top5_correct / total,
            'loss': total_loss / len(test_loader)
        }
        
        # Detailed classification metrics
        if len(np.unique(all_targets)) > 2:  # Multi-class
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return metrics
    
    def _evaluate_complexity(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model complexity metrics"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size (MB)
        model_size = total_params * 4 / (1024 * 1024)  # Assuming fp32
        
        # Estimate FLOPs (simplified)
        # This is a rough estimation - for accurate FLOPs, use tools like thop or fvcore
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        flops = self._estimate_flops(model, sample_input)
        
        return {
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size,
            'estimated_flops': flops
        }
    
    def _estimate_flops(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Rough FLOP estimation"""
        # This is a simplified estimation
        # For production use, consider using specialized libraries
        
        def conv_flop_count(input_shape, output_shape, kernel_size, groups=1):
            batch_size, in_channels, input_height, input_width = input_shape
            batch_size, out_channels, output_height, output_width = output_shape
            kernel_height, kernel_width = kernel_size
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(kernel_height * kernel_width) * in_channels // groups
            
            active_elements_count = batch_size * output_height * output_width
            overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
            
            return overall_conv_flops
        
        # This is a very rough estimation
        # In practice, you'd want to use a proper FLOP counting library
        total_flops = 0
        
        # Rough estimation based on model parameters
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Rough FLOP estimation for conv layers
                total_flops += module.weight.numel() * 2  # Multiply-add operations
            elif isinstance(module, nn.Linear):
                total_flops += module.weight.numel() * 2
        
        return total_flops / 1e9  # Convert to GFLOPs
    
    def compare_architectures(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Compare multiple architecture evaluation results"""
        comparison_data = []
        
        for result in results:
            row = {'architecture_id': result.architecture_id}
            row.update(result.metrics)
            row['evaluation_time'] = result.evaluation_time
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_results(self, results: List[EvaluationResult], save_dir: str = "plots"):
        """Plot evaluation results"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Accuracy vs Efficiency plot
        if len(results) > 1:
            accuracies = [r.metrics.get('accuracy', 0) for r in results]
            latencies = [r.metrics.get('hw_latency', r.hardware_metrics.get('latency', 0)) for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(latencies, accuracies)
            plt.xlabel('Latency (ms)')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy vs Latency Trade-off')
            
            for i, result in enumerate(results):
                plt.annotate(result.architecture_id, (latencies[i], accuracies[i]))
            
            plt.savefig(f"{save_dir}/accuracy_vs_latency.png")
            plt.close()
        
        # Training curves for individual architectures
        for result in results:
            if result.training_history:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(result.training_history['train_loss'], label='Train')
                plt.plot(result.training_history['val_loss'], label='Validation')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{result.architecture_id} - Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(result.training_history['train_acc'], label='Train')
                plt.plot(result.training_history['val_acc'], label='Validation')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title(f'{result.architecture_id} - Accuracy')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{result.architecture_id}_training.png")
                plt.close()


# Example usage
if __name__ == "__main__":
    # Create evaluation config
    config = EvaluationConfig(
        mode=EvaluationMode.STANDARD,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.LOSS,
            EvaluationMetric.PARAMETERS,
            EvaluationMetric.LATENCY
        ],
        epochs=10,  # Reduced for example
        test_robustness=True,
        test_calibration=True
    )
    
    # Create evaluator
    evaluator = ArchitectureEvaluator(config)
    
    print("Architecture Evaluation Framework Created Successfully!")
    print(f"Evaluation mode: {config.mode.value}")
    print(f"Metrics: {[m.value for m in config.metrics]}")
    print(f"Robustness testing: {config.test_robustness}")
    print(f"Calibration testing: {config.test_calibration}")
    
    print("Architecture evaluation implementation completed!")