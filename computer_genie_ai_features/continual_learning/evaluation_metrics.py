"""
Evaluation Metrics for Continual Learning
निरंतर शिक्षा के लिए मूल्यांकन मेट्रिक्स

Comprehensive evaluation metrics for continual learning including
forgetting measures, transfer metrics, and stability analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict
import logging
from abc import ABC, abstractmethod


class MetricType(Enum):
    """Types of continual learning metrics"""
    ACCURACY = "accuracy"
    FORGETTING = "forgetting"
    TRANSFER = "transfer"
    STABILITY = "stability"
    PLASTICITY = "plasticity"
    INTERFERENCE = "interference"
    RETENTION = "retention"
    LEARNING_CURVE = "learning_curve"


class ForgettingMeasure(Enum):
    """Different measures of catastrophic forgetting"""
    BACKWARD_TRANSFER = "backward_transfer"
    FORGETTING_MEASURE = "forgetting_measure"
    RETENTION_RATE = "retention_rate"
    PERFORMANCE_DROP = "performance_drop"
    RELATIVE_FORGETTING = "relative_forgetting"


class TransferMeasure(Enum):
    """Different measures of knowledge transfer"""
    FORWARD_TRANSFER = "forward_transfer"
    LEARNING_ACCELERATION = "learning_acceleration"
    ZERO_SHOT_TRANSFER = "zero_shot_transfer"
    POSITIVE_TRANSFER = "positive_transfer"
    NEGATIVE_TRANSFER = "negative_transfer"


@dataclass
class TaskResult:
    """Results for a single task"""
    task_id: int
    task_name: str
    accuracy: float
    loss: float
    training_time: float
    num_samples: int
    num_epochs: int
    convergence_epoch: Optional[int] = None
    learning_curve: List[float] = field(default_factory=list)
    validation_curve: List[float] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """Configuration for continual learning evaluation"""
    # Metrics to compute
    compute_forgetting: bool = True
    compute_transfer: bool = True
    compute_stability: bool = True
    compute_plasticity: bool = True
    
    # Forgetting measures
    forgetting_measures: List[ForgettingMeasure] = field(
        default_factory=lambda: [
            ForgettingMeasure.BACKWARD_TRANSFER,
            ForgettingMeasure.FORGETTING_MEASURE,
            ForgettingMeasure.RETENTION_RATE
        ]
    )
    
    # Transfer measures
    transfer_measures: List[TransferMeasure] = field(
        default_factory=lambda: [
            TransferMeasure.FORWARD_TRANSFER,
            TransferMeasure.LEARNING_ACCELERATION
        ]
    )
    
    # Evaluation parameters
    num_runs: int = 1                # Number of evaluation runs
    confidence_level: float = 0.95   # Confidence level for statistics
    
    # Plotting parameters
    plot_results: bool = True
    save_plots: bool = False
    plot_dir: str = "plots"
    
    # Device
    device: str = "cuda"
    verbose: bool = True


class ContinualLearningEvaluator:
    """Comprehensive evaluator for continual learning systems"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.task_results: List[TaskResult] = []
        self.accuracy_matrix: List[List[float]] = []  # [task_i][task_j] = accuracy on task j after training task i
        self.baseline_accuracies: List[float] = []    # Single-task baseline accuracies
        
        # Computed metrics
        self.forgetting_metrics: Dict[str, float] = {}
        self.transfer_metrics: Dict[str, float] = {}
        self.stability_metrics: Dict[str, float] = {}
        self.plasticity_metrics: Dict[str, float] = {}
        
        # Statistics
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def add_task_result(self, result: TaskResult):
        """Add result for a completed task"""
        self.task_results.append(result)
        
        if self.config.verbose:
            self.logger.info(f"Added result for task {result.task_id}: "
                           f"accuracy={result.accuracy:.4f}, loss={result.loss:.4f}")
    
    def update_accuracy_matrix(self, task_accuracies: List[float]):
        """Update accuracy matrix with current task accuracies"""
        self.accuracy_matrix.append(task_accuracies.copy())
        
        if self.config.verbose:
            self.logger.info(f"Updated accuracy matrix. Current shape: "
                           f"{len(self.accuracy_matrix)} x {len(task_accuracies)}")
    
    def set_baseline_accuracies(self, baseline_accuracies: List[float]):
        """Set single-task baseline accuracies"""
        self.baseline_accuracies = baseline_accuracies.copy()
        
        if self.config.verbose:
            self.logger.info(f"Set baseline accuracies: {baseline_accuracies}")
    
    def compute_forgetting_metrics(self) -> Dict[str, float]:
        """Compute various forgetting metrics"""
        if len(self.accuracy_matrix) < 2:
            return {}
        
        metrics = {}
        num_tasks = len(self.accuracy_matrix[-1])
        
        # Backward Transfer (BWT)
        if ForgettingMeasure.BACKWARD_TRANSFER in self.config.forgetting_measures:
            bwt = self._compute_backward_transfer()
            metrics['backward_transfer'] = bwt
        
        # Forgetting Measure
        if ForgettingMeasure.FORGETTING_MEASURE in self.config.forgetting_measures:
            forgetting = self._compute_forgetting_measure()
            metrics['forgetting_measure'] = forgetting
        
        # Retention Rate
        if ForgettingMeasure.RETENTION_RATE in self.config.forgetting_measures:
            retention = self._compute_retention_rate()
            metrics['retention_rate'] = retention
        
        # Performance Drop
        if ForgettingMeasure.PERFORMANCE_DROP in self.config.forgetting_measures:
            perf_drop = self._compute_performance_drop()
            metrics['performance_drop'] = perf_drop
        
        # Relative Forgetting
        if ForgettingMeasure.RELATIVE_FORGETTING in self.config.forgetting_measures:
            rel_forgetting = self._compute_relative_forgetting()
            metrics['relative_forgetting'] = rel_forgetting
        
        self.forgetting_metrics = metrics
        return metrics
    
    def _compute_backward_transfer(self) -> float:
        """Compute Backward Transfer (BWT)"""
        if len(self.accuracy_matrix) < 2:
            return 0.0
        
        bwt_sum = 0.0
        count = 0
        
        for i in range(len(self.accuracy_matrix) - 1):
            for j in range(i):
                # Accuracy on task j after learning task i vs after learning task j
                acc_after_i = self.accuracy_matrix[i][j]
                acc_after_j = self.accuracy_matrix[j][j]
                bwt_sum += acc_after_i - acc_after_j
                count += 1
        
        return bwt_sum / count if count > 0 else 0.0
    
    def _compute_forgetting_measure(self) -> float:
        """Compute average forgetting across all tasks"""
        if len(self.accuracy_matrix) < 2:
            return 0.0
        
        forgetting_sum = 0.0
        count = 0
        
        final_accuracies = self.accuracy_matrix[-1]
        
        for j in range(len(final_accuracies) - 1):  # Exclude current task
            # Maximum accuracy achieved on task j
            max_acc = max(self.accuracy_matrix[k][j] for k in range(j, len(self.accuracy_matrix)))
            # Final accuracy on task j
            final_acc = final_accuracies[j]
            
            forgetting_sum += max_acc - final_acc
            count += 1
        
        return forgetting_sum / count if count > 0 else 0.0
    
    def _compute_retention_rate(self) -> float:
        """Compute retention rate (1 - forgetting)"""
        forgetting = self._compute_forgetting_measure()
        return 1.0 - forgetting
    
    def _compute_performance_drop(self) -> float:
        """Compute average performance drop from peak"""
        if len(self.accuracy_matrix) < 2:
            return 0.0
        
        drop_sum = 0.0
        count = 0
        
        final_accuracies = self.accuracy_matrix[-1]
        
        for j in range(len(final_accuracies) - 1):
            # Peak accuracy on task j
            peak_acc = max(self.accuracy_matrix[k][j] for k in range(j, len(self.accuracy_matrix)))
            # Final accuracy on task j
            final_acc = final_accuracies[j]
            
            if peak_acc > 0:
                drop = (peak_acc - final_acc) / peak_acc
                drop_sum += drop
                count += 1
        
        return drop_sum / count if count > 0 else 0.0
    
    def _compute_relative_forgetting(self) -> float:
        """Compute forgetting relative to baseline performance"""
        if not self.baseline_accuracies or len(self.accuracy_matrix) < 2:
            return 0.0
        
        forgetting_sum = 0.0
        count = 0
        
        final_accuracies = self.accuracy_matrix[-1]
        
        for j in range(min(len(final_accuracies) - 1, len(self.baseline_accuracies))):
            baseline_acc = self.baseline_accuracies[j]
            final_acc = final_accuracies[j]
            
            if baseline_acc > 0:
                rel_forgetting = (baseline_acc - final_acc) / baseline_acc
                forgetting_sum += rel_forgetting
                count += 1
        
        return forgetting_sum / count if count > 0 else 0.0
    
    def compute_transfer_metrics(self) -> Dict[str, float]:
        """Compute various transfer metrics"""
        if len(self.accuracy_matrix) < 2:
            return {}
        
        metrics = {}
        
        # Forward Transfer (FWT)
        if TransferMeasure.FORWARD_TRANSFER in self.config.transfer_measures:
            fwt = self._compute_forward_transfer()
            metrics['forward_transfer'] = fwt
        
        # Learning Acceleration
        if TransferMeasure.LEARNING_ACCELERATION in self.config.transfer_measures:
            acceleration = self._compute_learning_acceleration()
            metrics['learning_acceleration'] = acceleration
        
        # Zero-shot Transfer
        if TransferMeasure.ZERO_SHOT_TRANSFER in self.config.transfer_measures:
            zero_shot = self._compute_zero_shot_transfer()
            metrics['zero_shot_transfer'] = zero_shot
        
        self.transfer_metrics = metrics
        return metrics
    
    def _compute_forward_transfer(self) -> float:
        """Compute Forward Transfer (FWT)"""
        if not self.baseline_accuracies or len(self.accuracy_matrix) < 2:
            return 0.0
        
        fwt_sum = 0.0
        count = 0
        
        for i in range(1, len(self.accuracy_matrix)):
            for j in range(i, len(self.accuracy_matrix[i])):
                if j < len(self.baseline_accuracies):
                    # Accuracy on task j when first encountered vs baseline
                    acc_first = self.accuracy_matrix[i][j] if j < len(self.accuracy_matrix[i]) else 0.0
                    baseline_acc = self.baseline_accuracies[j]
                    
                    fwt_sum += acc_first - baseline_acc
                    count += 1
        
        return fwt_sum / count if count > 0 else 0.0
    
    def _compute_learning_acceleration(self) -> float:
        """Compute learning acceleration based on convergence speed"""
        if len(self.task_results) < 2:
            return 0.0
        
        acceleration_sum = 0.0
        count = 0
        
        for i, result in enumerate(self.task_results[1:], 1):  # Skip first task
            if result.convergence_epoch is not None:
                # Compare with first task convergence
                first_task_epochs = self.task_results[0].convergence_epoch or self.task_results[0].num_epochs
                current_epochs = result.convergence_epoch
                
                if first_task_epochs > 0:
                    acceleration = (first_task_epochs - current_epochs) / first_task_epochs
                    acceleration_sum += acceleration
                    count += 1
        
        return acceleration_sum / count if count > 0 else 0.0
    
    def _compute_zero_shot_transfer(self) -> float:
        """Compute zero-shot transfer performance"""
        # This would require initial performance on new tasks before any training
        # For now, return 0 as placeholder
        return 0.0
    
    def compute_stability_metrics(self) -> Dict[str, float]:
        """Compute stability metrics"""
        metrics = {}
        
        if len(self.accuracy_matrix) < 2:
            return metrics
        
        # Stability: variance in performance across tasks
        final_accuracies = self.accuracy_matrix[-1]
        if len(final_accuracies) > 1:
            stability = 1.0 - np.std(final_accuracies)
            metrics['stability'] = max(0.0, stability)  # Ensure non-negative
        
        # Performance consistency
        if len(self.task_results) > 1:
            accuracies = [result.accuracy for result in self.task_results]
            consistency = 1.0 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0.0
            metrics['consistency'] = max(0.0, consistency)
        
        self.stability_metrics = metrics
        return metrics
    
    def compute_plasticity_metrics(self) -> Dict[str, float]:
        """Compute plasticity metrics"""
        metrics = {}
        
        if len(self.task_results) < 2:
            return metrics
        
        # Learning speed: how quickly new tasks are learned
        learning_speeds = []
        for result in self.task_results:
            if result.convergence_epoch is not None and result.convergence_epoch > 0:
                speed = 1.0 / result.convergence_epoch
                learning_speeds.append(speed)
        
        if learning_speeds:
            metrics['avg_learning_speed'] = np.mean(learning_speeds)
            metrics['learning_speed_std'] = np.std(learning_speeds)
        
        # Adaptation capability: final accuracy on new tasks
        if len(self.task_results) > 1:
            new_task_accuracies = [result.accuracy for result in self.task_results[1:]]
            metrics['avg_new_task_accuracy'] = np.mean(new_task_accuracies)
            metrics['new_task_accuracy_std'] = np.std(new_task_accuracies)
        
        self.plasticity_metrics = metrics
        return metrics
    
    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute all evaluation metrics"""
        all_metrics = {}
        
        if self.config.compute_forgetting:
            all_metrics['forgetting'] = self.compute_forgetting_metrics()
        
        if self.config.compute_transfer:
            all_metrics['transfer'] = self.compute_transfer_metrics()
        
        if self.config.compute_stability:
            all_metrics['stability'] = self.compute_stability_metrics()
        
        if self.config.compute_plasticity:
            all_metrics['plasticity'] = self.compute_plasticity_metrics()
        
        # Overall metrics
        overall_metrics = {}
        
        # Average accuracy
        if self.accuracy_matrix:
            final_accuracies = self.accuracy_matrix[-1]
            overall_metrics['average_accuracy'] = np.mean(final_accuracies)
            overall_metrics['accuracy_std'] = np.std(final_accuracies)
        
        # Task completion rate
        if self.task_results:
            completion_rate = len(self.task_results) / len(self.task_results)  # Always 1.0 for completed tasks
            overall_metrics['task_completion_rate'] = completion_rate
        
        all_metrics['overall'] = overall_metrics
        
        # Store evaluation
        self.evaluation_history.append({
            'timestamp': torch.tensor(0.0),  # Placeholder
            'metrics': all_metrics,
            'num_tasks': len(self.task_results)
        })
        
        return all_metrics
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot evaluation results"""
        if not self.accuracy_matrix:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continual Learning Evaluation Results', fontsize=16)
        
        # 1. Accuracy Matrix Heatmap
        ax1 = axes[0, 0]
        accuracy_array = np.array(self.accuracy_matrix)
        sns.heatmap(accuracy_array, annot=True, fmt='.3f', cmap='Blues', ax=ax1)
        ax1.set_title('Accuracy Matrix')
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Training Stage')
        
        # 2. Learning Curves
        ax2 = axes[0, 1]
        for i, result in enumerate(self.task_results):
            if result.learning_curve:
                ax2.plot(result.learning_curve, label=f'Task {result.task_id}')
        ax2.set_title('Learning Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Forgetting Analysis
        ax3 = axes[1, 0]
        if len(self.accuracy_matrix) > 1:
            # Plot accuracy degradation for each task
            for j in range(len(self.accuracy_matrix[0])):
                task_accuracies = [self.accuracy_matrix[i][j] for i in range(j, len(self.accuracy_matrix))]
                ax3.plot(range(j, len(self.accuracy_matrix)), task_accuracies, 
                        marker='o', label=f'Task {j}')
        ax3.set_title('Forgetting Analysis')
        ax3.set_xlabel('Training Stage')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Metrics Summary
        ax4 = axes[1, 1]
        metrics_data = []
        metrics_labels = []
        
        # Collect metrics for plotting
        if self.forgetting_metrics:
            for key, value in self.forgetting_metrics.items():
                metrics_data.append(value)
                metrics_labels.append(f'Forgetting: {key}')
        
        if self.transfer_metrics:
            for key, value in self.transfer_metrics.items():
                metrics_data.append(value)
                metrics_labels.append(f'Transfer: {key}')
        
        if metrics_data:
            bars = ax4.bar(range(len(metrics_data)), metrics_data)
            ax4.set_title('Metrics Summary')
            ax4.set_xticks(range(len(metrics_labels)))
            ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
            ax4.set_ylabel('Metric Value')
            
            # Color bars based on metric type
            for i, bar in enumerate(bars):
                if 'Forgetting' in metrics_labels[i]:
                    bar.set_color('red' if metrics_data[i] > 0 else 'green')
                else:
                    bar.set_color('blue' if metrics_data[i] > 0 else 'orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.config.plot_results:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("CONTINUAL LEARNING EVALUATION REPORT")
        report.append("=" * 60)
        
        # Basic information
        report.append(f"\nNumber of tasks: {len(self.task_results)}")
        report.append(f"Evaluation runs: {self.config.num_runs}")
        
        # Task results summary
        if self.task_results:
            report.append("\nTASK RESULTS:")
            report.append("-" * 40)
            for result in self.task_results:
                report.append(f"Task {result.task_id} ({result.task_name}):")
                report.append(f"  Accuracy: {result.accuracy:.4f}")
                report.append(f"  Loss: {result.loss:.4f}")
                report.append(f"  Training time: {result.training_time:.2f}s")
                report.append(f"  Samples: {result.num_samples}")
                report.append(f"  Epochs: {result.num_epochs}")
                if result.convergence_epoch:
                    report.append(f"  Convergence: epoch {result.convergence_epoch}")
                report.append("")
        
        # Forgetting metrics
        if self.forgetting_metrics:
            report.append("FORGETTING METRICS:")
            report.append("-" * 40)
            for metric, value in self.forgetting_metrics.items():
                report.append(f"{metric}: {value:.4f}")
            report.append("")
        
        # Transfer metrics
        if self.transfer_metrics:
            report.append("TRANSFER METRICS:")
            report.append("-" * 40)
            for metric, value in self.transfer_metrics.items():
                report.append(f"{metric}: {value:.4f}")
            report.append("")
        
        # Stability metrics
        if self.stability_metrics:
            report.append("STABILITY METRICS:")
            report.append("-" * 40)
            for metric, value in self.stability_metrics.items():
                report.append(f"{metric}: {value:.4f}")
            report.append("")
        
        # Plasticity metrics
        if self.plasticity_metrics:
            report.append("PLASTICITY METRICS:")
            report.append("-" * 40)
            for metric, value in self.plasticity_metrics.items():
                report.append(f"{metric}: {value:.4f}")
            report.append("")
        
        # Overall assessment
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        
        if self.accuracy_matrix:
            final_accuracies = self.accuracy_matrix[-1]
            avg_accuracy = np.mean(final_accuracies)
            report.append(f"Average final accuracy: {avg_accuracy:.4f}")
            
            if self.baseline_accuracies:
                baseline_avg = np.mean(self.baseline_accuracies)
                performance_ratio = avg_accuracy / baseline_avg if baseline_avg > 0 else 0
                report.append(f"Performance vs baseline: {performance_ratio:.4f}")
        
        # Recommendations
        report.append("\nRECOMMENDations:")
        report.append("-" * 40)
        
        if self.forgetting_metrics.get('forgetting_measure', 0) > 0.1:
            report.append("• High forgetting detected. Consider stronger regularization.")
        
        if self.transfer_metrics.get('forward_transfer', 0) < 0:
            report.append("• Negative transfer detected. Review task similarity and model capacity.")
        
        if self.stability_metrics.get('stability', 1) < 0.8:
            report.append("• Low stability. Consider ensemble methods or better initialization.")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        stats = {
            'num_tasks': len(self.task_results),
            'num_evaluations': len(self.evaluation_history),
            'config': {
                'compute_forgetting': self.config.compute_forgetting,
                'compute_transfer': self.config.compute_transfer,
                'compute_stability': self.config.compute_stability,
                'compute_plasticity': self.config.compute_plasticity,
                'num_runs': self.config.num_runs
            }
        }
        
        # Task statistics
        if self.task_results:
            accuracies = [result.accuracy for result in self.task_results]
            losses = [result.loss for result in self.task_results]
            training_times = [result.training_time for result in self.task_results]
            
            stats.update({
                'task_stats': {
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'avg_loss': np.mean(losses),
                    'avg_training_time': np.mean(training_times),
                    'total_training_time': np.sum(training_times)
                }
            })
        
        # Metric statistics
        if self.forgetting_metrics:
            stats['forgetting_stats'] = self.forgetting_metrics.copy()
        
        if self.transfer_metrics:
            stats['transfer_stats'] = self.transfer_metrics.copy()
        
        if self.stability_metrics:
            stats['stability_stats'] = self.stability_metrics.copy()
        
        if self.plasticity_metrics:
            stats['plasticity_stats'] = self.plasticity_metrics.copy()
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create evaluation configuration
    config = EvaluationConfig(
        compute_forgetting=True,
        compute_transfer=True,
        compute_stability=True,
        compute_plasticity=True,
        plot_results=True
    )
    
    # Create evaluator
    evaluator = ContinualLearningEvaluator(config)
    
    # Simulate some task results
    for i in range(3):
        result = TaskResult(
            task_id=i,
            task_name=f"Task_{i}",
            accuracy=0.8 + np.random.normal(0, 0.1),
            loss=0.5 + np.random.normal(0, 0.1),
            training_time=100.0 + np.random.normal(0, 20),
            num_samples=1000,
            num_epochs=50,
            convergence_epoch=30 + np.random.randint(-10, 10)
        )
        evaluator.add_task_result(result)
        
        # Simulate accuracy matrix
        accuracies = [0.8 + np.random.normal(0, 0.05) for _ in range(i + 1)]
        evaluator.update_accuracy_matrix(accuracies)
    
    # Set baseline accuracies
    evaluator.set_baseline_accuracies([0.85, 0.82, 0.88])
    
    # Compute all metrics
    all_metrics = evaluator.compute_all_metrics()
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    print("Continual learning evaluator created successfully!")