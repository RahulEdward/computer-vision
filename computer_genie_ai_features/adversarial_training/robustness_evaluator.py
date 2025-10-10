"""
Robustness Evaluation Framework
मजबूती मूल्यांकन ढांचा

Comprehensive evaluation framework for assessing model robustness against
various adversarial attacks and measuring defense effectiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from .adversarial_attacks import (
    AdversarialAttack, FGSMAttack, PGDAttack, CWAttack, AutoAttack,
    AttackConfig, AttackType, NormType
)
from .defense_mechanisms import DefenseMechanism, DefenseConfig


class EvaluationMetric(Enum):
    """Evaluation metrics for robustness assessment"""
    CLEAN_ACCURACY = "clean_accuracy"
    ROBUST_ACCURACY = "robust_accuracy"
    ATTACK_SUCCESS_RATE = "attack_success_rate"
    PERTURBATION_DISTANCE = "perturbation_distance"
    CONFIDENCE_DEGRADATION = "confidence_degradation"
    PREDICTION_CONSISTENCY = "prediction_consistency"
    CERTIFIED_ACCURACY = "certified_accuracy"
    ADVERSARIAL_DISTANCE = "adversarial_distance"
    ROBUSTNESS_CURVE = "robustness_curve"
    DECISION_BOUNDARY_DISTANCE = "decision_boundary_distance"


class RobustnessLevel(Enum):
    """Robustness assessment levels"""
    VERY_WEAK = "very_weak"      # < 10% robust accuracy
    WEAK = "weak"                # 10-30% robust accuracy
    MODERATE = "moderate"        # 30-60% robust accuracy
    STRONG = "strong"            # 60-80% robust accuracy
    VERY_STRONG = "very_strong"  # > 80% robust accuracy


@dataclass
class EvaluationConfig:
    """Configuration for robustness evaluation"""
    # Attack configurations for evaluation
    attack_configs: List[AttackConfig] = field(default_factory=lambda: [
        AttackConfig(AttackType.FGSM, epsilon=8.0/255.0),
        AttackConfig(AttackType.PGD, epsilon=8.0/255.0, num_steps=20),
        AttackConfig(AttackType.CW, epsilon=8.0/255.0, num_steps=100)
    ])
    
    # Evaluation parameters
    batch_size: int = 32
    num_samples: Optional[int] = None  # None means use entire dataset
    
    # Metrics to compute
    metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.CLEAN_ACCURACY,
        EvaluationMetric.ROBUST_ACCURACY,
        EvaluationMetric.ATTACK_SUCCESS_RATE,
        EvaluationMetric.PERTURBATION_DISTANCE
    ])
    
    # Robustness curve parameters
    epsilon_range: List[float] = field(default_factory=lambda: [
        0.0, 1.0/255.0, 2.0/255.0, 4.0/255.0, 8.0/255.0, 16.0/255.0
    ])
    
    # Certification parameters
    certification_samples: int = 1000
    certification_alpha: float = 0.001
    
    # Output parameters
    save_results: bool = True
    results_dir: str = "./evaluation_results"
    plot_results: bool = True
    
    # Device and performance
    device: str = "cuda"
    verbose: bool = True


@dataclass
class EvaluationResult:
    """Results from robustness evaluation"""
    model_name: str
    attack_name: str
    attack_config: AttackConfig
    
    # Basic metrics
    clean_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    
    # Advanced metrics
    avg_perturbation_distance: float
    confidence_degradation: float
    prediction_consistency: float
    
    # Per-class results
    per_class_clean_acc: Dict[int, float] = field(default_factory=dict)
    per_class_robust_acc: Dict[int, float] = field(default_factory=dict)
    
    # Timing information
    evaluation_time: float = 0.0
    attack_time: float = 0.0
    
    # Additional statistics
    successful_attacks: int = 0
    total_samples: int = 0
    
    def get_robustness_level(self) -> RobustnessLevel:
        """Determine robustness level based on robust accuracy"""
        if self.robust_accuracy < 0.1:
            return RobustnessLevel.VERY_WEAK
        elif self.robust_accuracy < 0.3:
            return RobustnessLevel.WEAK
        elif self.robust_accuracy < 0.6:
            return RobustnessLevel.MODERATE
        elif self.robust_accuracy < 0.8:
            return RobustnessLevel.STRONG
        else:
            return RobustnessLevel.VERY_STRONG


class RobustnessEvaluator:
    """Main robustness evaluation class"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)
        
        # Initialize attack instances
        self.attacks = self._create_attacks()
        
        # Results storage
        self.evaluation_results = []
        self.robustness_curves = {}
    
    def _create_attacks(self) -> Dict[str, AdversarialAttack]:
        """Create attack instances from configurations"""
        attacks = {}
        
        for attack_config in self.config.attack_configs:
            attack_name = f"{attack_config.attack_type.value}_eps_{attack_config.epsilon:.3f}"
            
            if attack_config.attack_type == AttackType.FGSM:
                attacks[attack_name] = FGSMAttack(attack_config)
            elif attack_config.attack_type == AttackType.PGD:
                attacks[attack_name] = PGDAttack(attack_config)
            elif attack_config.attack_type == AttackType.CW:
                attacks[attack_name] = CWAttack(attack_config)
            else:
                self.logger.warning(f"Unknown attack type: {attack_config.attack_type}")
        
        return attacks
    
    def evaluate_model(self, model: nn.Module, test_loader, model_name: str = "model") -> List[EvaluationResult]:
        """Evaluate model robustness against all configured attacks"""
        self.logger.info(f"Starting robustness evaluation for {model_name}")
        
        model.eval()
        results = []
        
        for attack_name, attack in self.attacks.items():
            self.logger.info(f"Evaluating against {attack_name}")
            
            result = self._evaluate_single_attack(
                model, test_loader, attack, attack_name, model_name
            )
            results.append(result)
            
            if self.config.verbose:
                self.logger.info(
                    f"{attack_name}: Clean Acc: {result.clean_accuracy:.3f}, "
                    f"Robust Acc: {result.robust_accuracy:.3f}, "
                    f"ASR: {result.attack_success_rate:.3f}"
                )
        
        self.evaluation_results.extend(results)
        
        # Generate robustness curves if requested
        if EvaluationMetric.ROBUSTNESS_CURVE in self.config.metrics:
            self._generate_robustness_curves(model, test_loader, model_name)
        
        # Save results
        if self.config.save_results:
            self._save_results(results, model_name)
        
        return results
    
    def _evaluate_single_attack(self, model: nn.Module, test_loader, attack: AdversarialAttack,
                               attack_name: str, model_name: str) -> EvaluationResult:
        """Evaluate model against a single attack"""
        start_time = time.time()
        
        # Initialize counters
        clean_correct = 0
        robust_correct = 0
        total_samples = 0
        successful_attacks = 0
        
        # Per-class counters
        per_class_clean_correct = defaultdict(int)
        per_class_robust_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        # Metric accumulators
        total_perturbation_distance = 0.0
        total_confidence_degradation = 0.0
        total_prediction_consistency = 0.0
        
        attack_time = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Limit samples if specified
                if self.config.num_samples and total_samples >= self.config.num_samples:
                    break
                
                # Clean evaluation
                clean_outputs = model(inputs)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_correct += (clean_preds == targets).sum().item()
                
                # Generate adversarial examples
                attack_start = time.time()
                adversarial = attack.attack(model, inputs, targets)
                attack_time += time.time() - attack_start
                
                # Robust evaluation
                robust_outputs = model(adversarial)
                robust_preds = robust_outputs.argmax(dim=1)
                robust_correct += (robust_preds == targets).sum().item()
                
                # Count successful attacks
                attack_success = (clean_preds == targets) & (robust_preds != targets)
                successful_attacks += attack_success.sum().item()
                
                # Per-class statistics
                for i in range(batch_size):
                    target_class = targets[i].item()
                    per_class_total[target_class] += 1
                    
                    if clean_preds[i] == targets[i]:
                        per_class_clean_correct[target_class] += 1
                    
                    if robust_preds[i] == targets[i]:
                        per_class_robust_correct[target_class] += 1
                
                # Compute additional metrics
                if EvaluationMetric.PERTURBATION_DISTANCE in self.config.metrics:
                    perturbation = adversarial - inputs
                    perturbation_distance = torch.norm(perturbation.view(batch_size, -1), dim=1)
                    total_perturbation_distance += perturbation_distance.sum().item()
                
                if EvaluationMetric.CONFIDENCE_DEGRADATION in self.config.metrics:
                    clean_confidence = F.softmax(clean_outputs, dim=1).max(dim=1)[0]
                    robust_confidence = F.softmax(robust_outputs, dim=1).max(dim=1)[0]
                    confidence_degradation = clean_confidence - robust_confidence
                    total_confidence_degradation += confidence_degradation.sum().item()
                
                if EvaluationMetric.PREDICTION_CONSISTENCY in self.config.metrics:
                    consistency = (clean_preds == robust_preds).float()
                    total_prediction_consistency += consistency.sum().item()
                
                total_samples += batch_size
        
        # Compute final metrics
        clean_accuracy = clean_correct / total_samples
        robust_accuracy = robust_correct / total_samples
        attack_success_rate = successful_attacks / total_samples
        
        avg_perturbation_distance = total_perturbation_distance / total_samples
        confidence_degradation = total_confidence_degradation / total_samples
        prediction_consistency = total_prediction_consistency / total_samples
        
        # Per-class accuracies
        per_class_clean_acc = {
            cls: per_class_clean_correct[cls] / per_class_total[cls]
            for cls in per_class_total.keys()
        }
        per_class_robust_acc = {
            cls: per_class_robust_correct[cls] / per_class_total[cls]
            for cls in per_class_total.keys()
        }
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            model_name=model_name,
            attack_name=attack_name,
            attack_config=attack.config,
            clean_accuracy=clean_accuracy,
            robust_accuracy=robust_accuracy,
            attack_success_rate=attack_success_rate,
            avg_perturbation_distance=avg_perturbation_distance,
            confidence_degradation=confidence_degradation,
            prediction_consistency=prediction_consistency,
            per_class_clean_acc=per_class_clean_acc,
            per_class_robust_acc=per_class_robust_acc,
            evaluation_time=evaluation_time,
            attack_time=attack_time,
            successful_attacks=successful_attacks,
            total_samples=total_samples
        )
    
    def _generate_robustness_curves(self, model: nn.Module, test_loader, model_name: str) -> None:
        """Generate robustness curves across different epsilon values"""
        self.logger.info(f"Generating robustness curves for {model_name}")
        
        curves = {}
        
        for attack_type in [AttackType.FGSM, AttackType.PGD]:
            curve_data = {'epsilon': [], 'robust_accuracy': []}
            
            for epsilon in self.config.epsilon_range:
                # Create attack config for this epsilon
                attack_config = AttackConfig(
                    attack_type=attack_type,
                    epsilon=epsilon,
                    step_size=epsilon/4 if epsilon > 0 else 0,
                    num_steps=20 if attack_type == AttackType.PGD else 1
                )
                
                # Create attack
                if attack_type == AttackType.FGSM:
                    attack = FGSMAttack(attack_config)
                else:
                    attack = PGDAttack(attack_config)
                
                # Evaluate
                robust_acc = self._quick_robustness_evaluation(model, test_loader, attack)
                
                curve_data['epsilon'].append(epsilon)
                curve_data['robust_accuracy'].append(robust_acc)
            
            curves[attack_type.value] = curve_data
        
        self.robustness_curves[model_name] = curves
    
    def _quick_robustness_evaluation(self, model: nn.Module, test_loader, attack: AdversarialAttack) -> float:
        """Quick robustness evaluation for curve generation"""
        model.eval()
        robust_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Limit samples for speed
                if total_samples >= 1000:
                    break
                
                # Generate adversarial examples
                adversarial = attack.attack(model, inputs, targets)
                
                # Evaluate robustness
                outputs = model(adversarial)
                preds = outputs.argmax(dim=1)
                robust_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)
        
        return robust_correct / total_samples
    
    def compare_models(self, models: Dict[str, nn.Module], test_loader) -> Dict[str, Any]:
        """Compare robustness of multiple models"""
        self.logger.info("Starting model comparison")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            results = self.evaluate_model(model, test_loader, model_name)
            comparison_results[model_name] = results
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        
        # Plot comparison if requested
        if self.config.plot_results:
            self._plot_model_comparison(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary,
            'robustness_curves': self.robustness_curves
        }
    
    def _generate_comparison_summary(self, comparison_results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """Generate summary of model comparison"""
        summary = {
            'best_clean_accuracy': None,
            'best_robust_accuracy': None,
            'best_overall_robustness': None,
            'attack_rankings': {},
            'robustness_levels': {}
        }
        
        # Find best models for different metrics
        best_clean_acc = 0.0
        best_robust_acc = 0.0
        best_overall = 0.0
        
        for model_name, results in comparison_results.items():
            avg_clean_acc = np.mean([r.clean_accuracy for r in results])
            avg_robust_acc = np.mean([r.robust_accuracy for r in results])
            overall_score = 0.3 * avg_clean_acc + 0.7 * avg_robust_acc
            
            if avg_clean_acc > best_clean_acc:
                best_clean_acc = avg_clean_acc
                summary['best_clean_accuracy'] = (model_name, avg_clean_acc)
            
            if avg_robust_acc > best_robust_acc:
                best_robust_acc = avg_robust_acc
                summary['best_robust_accuracy'] = (model_name, avg_robust_acc)
            
            if overall_score > best_overall:
                best_overall = overall_score
                summary['best_overall_robustness'] = (model_name, overall_score)
            
            # Determine robustness level
            summary['robustness_levels'][model_name] = results[0].get_robustness_level().value
        
        # Generate attack-specific rankings
        for attack_name in self.attacks.keys():
            attack_results = []
            for model_name, results in comparison_results.items():
                attack_result = next((r for r in results if r.attack_name == attack_name), None)
                if attack_result:
                    attack_results.append((model_name, attack_result.robust_accuracy))
            
            # Sort by robust accuracy
            attack_results.sort(key=lambda x: x[1], reverse=True)
            summary['attack_rankings'][attack_name] = attack_results
        
        return summary
    
    def evaluate_defense(self, model: nn.Module, defense: DefenseMechanism, 
                        test_loader, defense_name: str = "defense") -> List[EvaluationResult]:
        """Evaluate defense mechanism effectiveness"""
        self.logger.info(f"Evaluating defense: {defense_name}")
        
        # Create defended model wrapper
        defended_model = DefendedModelWrapper(model, defense)
        
        # Evaluate defended model
        results = self.evaluate_model(defended_model, test_loader, f"{defense_name}_defended")
        
        return results
    
    def adaptive_evaluation(self, model: nn.Module, test_loader, 
                          model_name: str = "model") -> Dict[str, Any]:
        """Perform adaptive evaluation with stronger attacks"""
        self.logger.info(f"Starting adaptive evaluation for {model_name}")
        
        # Start with standard evaluation
        standard_results = self.evaluate_model(model, test_loader, model_name)
        
        # If model shows high robustness, use stronger attacks
        avg_robust_acc = np.mean([r.robust_accuracy for r in standard_results])
        
        if avg_robust_acc > 0.6:  # High robustness threshold
            self.logger.info("High robustness detected, using stronger attacks")
            
            # Create stronger attack configurations
            strong_attacks = [
                AttackConfig(AttackType.PGD, epsilon=16.0/255.0, num_steps=100),
                AttackConfig(AttackType.CW, epsilon=16.0/255.0, num_steps=1000, c=10.0)
            ]
            
            # Evaluate with stronger attacks
            original_configs = self.config.attack_configs
            self.config.attack_configs = strong_attacks
            self.attacks = self._create_attacks()
            
            strong_results = self.evaluate_model(model, test_loader, f"{model_name}_strong")
            
            # Restore original configuration
            self.config.attack_configs = original_configs
            self.attacks = self._create_attacks()
            
            return {
                'standard_results': standard_results,
                'strong_results': strong_results,
                'adaptive_assessment': self._assess_adaptive_robustness(standard_results, strong_results)
            }
        
        return {
            'standard_results': standard_results,
            'strong_results': [],
            'adaptive_assessment': 'Standard evaluation sufficient'
        }
    
    def _assess_adaptive_robustness(self, standard_results: List[EvaluationResult], 
                                   strong_results: List[EvaluationResult]) -> str:
        """Assess adaptive robustness based on standard vs strong attack results"""
        standard_avg = np.mean([r.robust_accuracy for r in standard_results])
        strong_avg = np.mean([r.robust_accuracy for r in strong_results])
        
        robustness_drop = standard_avg - strong_avg
        
        if robustness_drop < 0.1:
            return "Truly robust - maintains performance under strong attacks"
        elif robustness_drop < 0.3:
            return "Moderately robust - some degradation under strong attacks"
        else:
            return "Potentially gradient masking - significant degradation under strong attacks"
    
    def _save_results(self, results: List[EvaluationResult], model_name: str) -> None:
        """Save evaluation results to files"""
        # Convert results to dictionary format
        results_dict = []
        for result in results:
            result_dict = {
                'model_name': result.model_name,
                'attack_name': result.attack_name,
                'attack_config': {
                    'attack_type': result.attack_config.attack_type.value,
                    'epsilon': result.attack_config.epsilon,
                    'num_steps': result.attack_config.num_steps,
                    'step_size': result.attack_config.step_size
                },
                'clean_accuracy': result.clean_accuracy,
                'robust_accuracy': result.robust_accuracy,
                'attack_success_rate': result.attack_success_rate,
                'avg_perturbation_distance': result.avg_perturbation_distance,
                'confidence_degradation': result.confidence_degradation,
                'prediction_consistency': result.prediction_consistency,
                'robustness_level': result.get_robustness_level().value,
                'evaluation_time': result.evaluation_time,
                'attack_time': result.attack_time
            }
            results_dict.append(result_dict)
        
        # Save to JSON
        results_file = os.path.join(self.config.results_dir, f"{model_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save to CSV for easy analysis
        df = pd.DataFrame(results_dict)
        csv_file = os.path.join(self.config.results_dir, f"{model_name}_results.csv")
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {results_file} and {csv_file}")
    
    def _plot_model_comparison(self, comparison_results: Dict[str, List[EvaluationResult]]) -> None:
        """Plot model comparison results"""
        if not self.config.plot_results:
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Clean vs Robust Accuracy
        model_names = []
        clean_accs = []
        robust_accs = []
        
        for model_name, results in comparison_results.items():
            model_names.append(model_name)
            clean_accs.append(np.mean([r.clean_accuracy for r in results]))
            robust_accs.append(np.mean([r.robust_accuracy for r in results]))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, robust_accs, width, label='Robust Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Clean vs Robust Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Attack Success Rate by Attack Type
        attack_names = list(self.attacks.keys())
        attack_data = {attack: [] for attack in attack_names}
        
        for model_name, results in comparison_results.items():
            for result in results:
                if result.attack_name in attack_data:
                    attack_data[result.attack_name].append(result.attack_success_rate)
        
        attack_positions = np.arange(len(attack_names))
        for i, model_name in enumerate(model_names):
            model_asrs = []
            for attack_name in attack_names:
                model_results = comparison_results[model_name]
                attack_result = next((r for r in model_results if r.attack_name == attack_name), None)
                model_asrs.append(attack_result.attack_success_rate if attack_result else 0)
            
            axes[0, 1].plot(attack_positions, model_asrs, marker='o', label=model_name)
        
        axes[0, 1].set_xlabel('Attack Types')
        axes[0, 1].set_ylabel('Attack Success Rate')
        axes[0, 1].set_title('Attack Success Rate by Attack Type')
        axes[0, 1].set_xticks(attack_positions)
        axes[0, 1].set_xticklabels([name.split('_')[0] for name in attack_names], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Robustness Curves
        if self.robustness_curves:
            for model_name, curves in self.robustness_curves.items():
                for attack_type, curve_data in curves.items():
                    axes[1, 0].plot(curve_data['epsilon'], curve_data['robust_accuracy'], 
                                   marker='o', label=f"{model_name}_{attack_type}")
            
            axes[1, 0].set_xlabel('Epsilon')
            axes[1, 0].set_ylabel('Robust Accuracy')
            axes[1, 0].set_title('Robustness Curves')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Evaluation Time Comparison
        eval_times = []
        for model_name, results in comparison_results.items():
            avg_time = np.mean([r.evaluation_time for r in results])
            eval_times.append(avg_time)
        
        axes[1, 1].bar(model_names, eval_times, alpha=0.8)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Evaluation Time (seconds)')
        axes[1, 1].set_title('Evaluation Time Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.config.results_dir, "model_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved to {plot_file}")
    
    def generate_report(self, results: List[EvaluationResult], model_name: str) -> str:
        """Generate comprehensive evaluation report"""
        report = f"# Robustness Evaluation Report for {model_name}\n\n"
        
        # Summary statistics
        avg_clean_acc = np.mean([r.clean_accuracy for r in results])
        avg_robust_acc = np.mean([r.robust_accuracy for r in results])
        avg_asr = np.mean([r.attack_success_rate for r in results])
        
        report += f"## Summary\n"
        report += f"- Average Clean Accuracy: {avg_clean_acc:.3f}\n"
        report += f"- Average Robust Accuracy: {avg_robust_acc:.3f}\n"
        report += f"- Average Attack Success Rate: {avg_asr:.3f}\n"
        report += f"- Robustness Level: {results[0].get_robustness_level().value}\n\n"
        
        # Detailed results
        report += f"## Detailed Results\n\n"
        for result in results:
            report += f"### {result.attack_name}\n"
            report += f"- Clean Accuracy: {result.clean_accuracy:.3f}\n"
            report += f"- Robust Accuracy: {result.robust_accuracy:.3f}\n"
            report += f"- Attack Success Rate: {result.attack_success_rate:.3f}\n"
            report += f"- Avg Perturbation Distance: {result.avg_perturbation_distance:.6f}\n"
            report += f"- Confidence Degradation: {result.confidence_degradation:.3f}\n"
            report += f"- Prediction Consistency: {result.prediction_consistency:.3f}\n"
            report += f"- Evaluation Time: {result.evaluation_time:.2f}s\n\n"
        
        # Recommendations
        report += f"## Recommendations\n"
        if avg_robust_acc < 0.3:
            report += "- Model shows weak robustness. Consider adversarial training.\n"
            report += "- Implement input preprocessing defenses.\n"
            report += "- Consider ensemble methods for improved robustness.\n"
        elif avg_robust_acc < 0.6:
            report += "- Model shows moderate robustness. Fine-tune defense mechanisms.\n"
            report += "- Consider certified defense methods.\n"
        else:
            report += "- Model shows strong robustness. Verify against adaptive attacks.\n"
            report += "- Consider deployment with additional monitoring.\n"
        
        return report


class DefendedModelWrapper(nn.Module):
    """Wrapper to apply defense mechanisms during evaluation"""
    
    def __init__(self, model: nn.Module, defense: DefenseMechanism):
        super().__init__()
        self.model = model
        self.defense = defense
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply defense to inputs
        defended_x = self.defense.defend(x, self.model)
        
        # Forward through model
        return self.model(defended_x)


# Example usage
if __name__ == "__main__":
    # Create evaluation configuration
    config = EvaluationConfig(
        attack_configs=[
            AttackConfig(AttackType.FGSM, epsilon=8.0/255.0),
            AttackConfig(AttackType.PGD, epsilon=8.0/255.0, num_steps=20),
            AttackConfig(AttackType.CW, epsilon=8.0/255.0, num_steps=100)
        ],
        metrics=[
            EvaluationMetric.CLEAN_ACCURACY,
            EvaluationMetric.ROBUST_ACCURACY,
            EvaluationMetric.ATTACK_SUCCESS_RATE,
            EvaluationMetric.ROBUSTNESS_CURVE
        ],
        batch_size=32,
        save_results=True,
        plot_results=True
    )
    
    # Create evaluator
    evaluator = RobustnessEvaluator(config)
    
    print("Robustness evaluator created successfully!")
    print(f"Configuration: {config}")
    print(f"Number of attacks: {len(evaluator.attacks)}")