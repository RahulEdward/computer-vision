"""
Multi-Task Trainer for Computer Genie
मल्टी-टास्क ट्रेनर - Computer Genie के लिए

Implements comprehensive training pipeline for multi-task learning
with support for various optimization strategies and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .multi_task_model import MultiTaskModel, MultiTaskConfig, TaskType
from .task_balancer import TaskBalancer, BalancingConfig


@dataclass
class TrainingConfig:
    """Configuration for multi-task training"""
    # Basic training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    
    # Multi-task specific
    freeze_backbone_epochs: int = 5
    task_balancing_config: BalancingConfig = None
    
    # Evaluation
    eval_frequency: int = 5  # Evaluate every N epochs
    save_frequency: int = 10  # Save checkpoint every N epochs
    early_stopping_patience: int = 20
    
    # Logging
    log_frequency: int = 100  # Log every N steps
    save_dir: str = "./checkpoints"
    experiment_name: str = "multi_task_experiment"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.task_balancing_config is None:
            self.task_balancing_config = BalancingConfig()


class MultiTaskLoss(nn.Module):
    """Multi-task loss computation with task-specific losses"""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        # Task-specific loss functions
        self.element_detection_loss = nn.CrossEntropyLoss()
        self.bbox_regression_loss = nn.SmoothL1Loss()
        self.ocr_loss = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding
        self.intent_loss = nn.CrossEntropyLoss()
        self.action_loss = nn.CrossEntropyLoss()
        
        # Auxiliary losses
        self.confidence_loss = nn.BCELoss()
        self.regression_loss = nn.MSELoss()
    
    def compute_element_detection_loss(self, predictions: Dict[str, torch.Tensor],
                                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute element detection loss"""
        class_loss = self.element_detection_loss(
            predictions["class_logits"], targets["class_labels"]
        )
        
        bbox_loss = self.bbox_regression_loss(
            predictions["bboxes"], targets["bboxes"]
        )
        
        confidence_loss = self.confidence_loss(
            predictions["confidence"], targets["confidence"]
        )
        
        return class_loss + bbox_loss + 0.1 * confidence_loss
    
    def compute_ocr_loss(self, predictions: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute OCR loss"""
        # Character sequence loss
        char_logits = predictions["char_logits"]  # [B, seq_len, vocab_size]
        char_targets = targets["char_sequence"]   # [B, seq_len]
        
        # Reshape for loss computation
        char_logits_flat = char_logits.view(-1, char_logits.size(-1))
        char_targets_flat = char_targets.view(-1)
        
        char_loss = self.ocr_loss(char_logits_flat, char_targets_flat)
        
        # Text region loss
        region_loss = self.bbox_regression_loss(
            predictions["text_regions"], targets["text_regions"]
        )
        
        # Language classification loss
        lang_loss = self.element_detection_loss(
            predictions["language_logits"], targets["language_labels"]
        )
        
        return char_loss + 0.5 * region_loss + 0.2 * lang_loss
    
    def compute_intent_loss(self, predictions: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute intent classification loss"""
        intent_loss = self.intent_loss(
            predictions["intent_logits"], targets["intent_labels"]
        )
        
        # Urgency regression loss
        urgency_loss = self.regression_loss(
            predictions["urgency"], targets["urgency"]
        )
        
        return intent_loss + 0.3 * urgency_loss
    
    def compute_action_loss(self, predictions: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute action prediction loss"""
        action_loss = self.action_loss(
            predictions["action_logits"], targets["action_labels"]
        )
        
        # Action parameters loss
        params_loss = self.regression_loss(
            predictions["action_params"], targets["action_params"]
        )
        
        # Success probability loss
        success_loss = self.confidence_loss(
            predictions["success_probability"], targets["success_probability"]
        )
        
        return action_loss + 0.5 * params_loss + 0.2 * success_loss
    
    def forward(self, predictions: Dict[str, Any], 
                targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute all task losses"""
        losses = {}
        
        task_outputs = predictions["task_outputs"]
        
        # Element detection loss
        if "element_detection" in task_outputs and "element_detection" in targets:
            losses["element_detection"] = self.compute_element_detection_loss(
                task_outputs["element_detection"], targets["element_detection"]
            )
        
        # OCR loss
        if "ocr" in task_outputs and "ocr" in targets:
            losses["ocr"] = self.compute_ocr_loss(
                task_outputs["ocr"], targets["ocr"]
            )
        
        # Intent classification loss
        if "intent_classification" in task_outputs and "intent_classification" in targets:
            losses["intent_classification"] = self.compute_intent_loss(
                task_outputs["intent_classification"], targets["intent_classification"]
            )
        
        # Action prediction loss
        if "action_prediction" in task_outputs and "action_prediction" in targets:
            losses["action_prediction"] = self.compute_action_loss(
                task_outputs["action_prediction"], targets["action_prediction"]
            )
        
        return losses


class TaskSpecificMetrics:
    """Compute task-specific evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics = defaultdict(list)
    
    def update_element_detection_metrics(self, predictions: Dict[str, torch.Tensor],
                                       targets: Dict[str, torch.Tensor]) -> None:
        """Update element detection metrics"""
        # Classification accuracy
        class_preds = torch.argmax(predictions["class_logits"], dim=1)
        class_targets = targets["class_labels"]
        accuracy = (class_preds == class_targets).float().mean().item()
        self.metrics["element_detection_accuracy"].append(accuracy)
        
        # IoU for bounding boxes (simplified)
        bbox_preds = predictions["bboxes"]
        bbox_targets = targets["bboxes"]
        iou = self._compute_iou(bbox_preds, bbox_targets)
        self.metrics["element_detection_iou"].append(iou)
    
    def update_ocr_metrics(self, predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor]) -> None:
        """Update OCR metrics"""
        # Character-level accuracy
        char_preds = torch.argmax(predictions["char_logits"], dim=-1)
        char_targets = targets["char_sequence"]
        
        # Mask out padding tokens
        mask = char_targets != -1
        if mask.sum() > 0:
            accuracy = ((char_preds == char_targets) & mask).float().sum() / mask.float().sum()
            self.metrics["ocr_char_accuracy"].append(accuracy.item())
        
        # Language classification accuracy
        lang_preds = torch.argmax(predictions["language_logits"], dim=1)
        lang_targets = targets["language_labels"]
        lang_accuracy = (lang_preds == lang_targets).float().mean().item()
        self.metrics["ocr_language_accuracy"].append(lang_accuracy)
    
    def update_intent_metrics(self, predictions: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor]) -> None:
        """Update intent classification metrics"""
        intent_preds = torch.argmax(predictions["intent_logits"], dim=1)
        intent_targets = targets["intent_labels"]
        accuracy = (intent_preds == intent_targets).float().mean().item()
        self.metrics["intent_accuracy"].append(accuracy)
        
        # Urgency MAE
        urgency_mae = torch.abs(predictions["urgency"] - targets["urgency"]).mean().item()
        self.metrics["urgency_mae"].append(urgency_mae)
    
    def update_action_metrics(self, predictions: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor]) -> None:
        """Update action prediction metrics"""
        action_preds = torch.argmax(predictions["action_logits"], dim=1)
        action_targets = targets["action_labels"]
        accuracy = (action_preds == action_targets).float().mean().item()
        self.metrics["action_accuracy"].append(accuracy)
        
        # Success probability MAE
        success_mae = torch.abs(
            predictions["success_probability"] - targets["success_probability"]
        ).mean().item()
        self.metrics["success_probability_mae"].append(success_mae)
    
    def _compute_iou(self, pred_boxes: torch.Tensor, 
                     target_boxes: torch.Tensor) -> float:
        """Compute IoU between predicted and target bounding boxes"""
        # Simplified IoU computation
        # pred_boxes and target_boxes: [B, 4] (x, y, w, h)
        
        # Convert to (x1, y1, x2, y2) format
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]
        
        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2]
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3]
        
        # Compute intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Compute union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Compute IoU
        iou = inter_area / (union_area + 1e-8)
        return iou.mean().item()
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all metrics"""
        avg_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
        return avg_metrics


class MultiTaskTrainer:
    """Main trainer class for multi-task learning"""
    
    def __init__(self, model: MultiTaskModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.loss_fn = MultiTaskLoss(model.config)
        self.metrics = TaskSpecificMetrics()
        
        # Task balancer
        task_names = ["element_detection", "ocr", "intent_classification", "action_prediction"]
        self.task_balancer = TaskBalancer(config.task_balancing_config, task_names)
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.training_history = defaultdict(list)
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = TaskSpecificMetrics()
        
        # Freeze backbone if needed
        if self.current_epoch < self.config.freeze_backbone_epochs:
            self.model.freeze_backbone(True)
        else:
            self.model.freeze_backbone(False)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = batch["image"].to(self.device)
            targets = {key: {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in value.items()} 
                      for key, value in batch["targets"].items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs)
                task_losses = self.loss_fn(outputs, targets)
                
                # Balance losses
                total_loss = self.task_balancer.balance_losses(task_losses)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Update metrics
            self._update_metrics(outputs, targets, epoch_metrics)
            
            # Record losses
            for task_name, loss in task_losses.items():
                epoch_losses[task_name].append(loss.item())
            epoch_losses["total"].append(total_loss.item())
            
            # Update task balancer
            self.task_balancer.step()
            
            # Logging
            if self.global_step % self.config.log_frequency == 0:
                self._log_training_step(task_losses, total_loss)
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Compute epoch averages
        epoch_avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items()}
        epoch_avg_metrics = epoch_metrics.get_average_metrics()
        
        return {**epoch_avg_losses, **epoch_avg_metrics}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        val_losses = defaultdict(list)
        val_metrics = TaskSpecificMetrics()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                inputs = batch["image"].to(self.device)
                targets = {key: {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in value.items()} 
                          for key, value in batch["targets"].items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(inputs)
                    task_losses = self.loss_fn(outputs, targets)
                    total_loss = self.task_balancer.balance_losses(task_losses)
                
                # Update metrics
                self._update_metrics(outputs, targets, val_metrics)
                
                # Record losses
                for task_name, loss in task_losses.items():
                    val_losses[task_name].append(loss.item())
                val_losses["total"].append(total_loss.item())
        
        # Compute averages
        val_avg_losses = {name: np.mean(losses) for name, losses in val_losses.items()}
        val_avg_metrics = val_metrics.get_average_metrics()
        
        return {**val_avg_losses, **val_avg_metrics}
    
    def _update_metrics(self, outputs: Dict[str, Any], targets: Dict[str, Any],
                       metrics: TaskSpecificMetrics) -> None:
        """Update metrics for all tasks"""
        task_outputs = outputs["task_outputs"]
        
        if "element_detection" in task_outputs and "element_detection" in targets:
            metrics.update_element_detection_metrics(
                task_outputs["element_detection"], targets["element_detection"]
            )
        
        if "ocr" in task_outputs and "ocr" in targets:
            metrics.update_ocr_metrics(
                task_outputs["ocr"], targets["ocr"]
            )
        
        if "intent_classification" in task_outputs and "intent_classification" in targets:
            metrics.update_intent_metrics(
                task_outputs["intent_classification"], targets["intent_classification"]
            )
        
        if "action_prediction" in task_outputs and "action_prediction" in targets:
            metrics.update_action_metrics(
                task_outputs["action_prediction"], targets["action_prediction"]
            )
    
    def _log_training_step(self, task_losses: Dict[str, torch.Tensor], 
                          total_loss: torch.Tensor) -> None:
        """Log training step information"""
        log_msg = f"Step {self.global_step}: Total Loss = {total_loss.item():.4f}"
        
        for task_name, loss in task_losses.items():
            log_msg += f", {task_name} = {loss.item():.4f}"
        
        weights = self.task_balancer.get_current_weights()
        log_msg += f", Weights = {weights}"
        
        self.logger.info(log_msg)
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch(train_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Use validation metric for plateau scheduler
                if val_loader is not None:
                    val_results = self.evaluate(val_loader)
                    main_metric = val_results.get("element_detection_accuracy", 0.0)
                    self.scheduler.step(main_metric)
                else:
                    self.scheduler.step(train_results.get("element_detection_accuracy", 0.0))
            else:
                self.scheduler.step()
            
            # Evaluate
            val_results = {}
            if val_loader is not None and epoch % self.config.eval_frequency == 0:
                val_results = self.evaluate(val_loader)
            
            # Update task balancer
            self.task_balancer.epoch()
            
            # Record history
            for key, value in train_results.items():
                self.training_history[f"train_{key}"].append(value)
            
            for key, value in val_results.items():
                self.training_history[f"val_{key}"].append(value)
            
            # Log epoch results
            self._log_epoch_results(epoch, train_results, val_results)
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if val_results:
                main_metric = val_results.get("element_detection_accuracy", 0.0)
                if main_metric > self.best_metric:
                    self.best_metric = main_metric
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return dict(self.training_history)
    
    def _log_epoch_results(self, epoch: int, train_results: Dict[str, float],
                          val_results: Dict[str, float]) -> None:
        """Log epoch results"""
        log_msg = f"Epoch {epoch}: "
        log_msg += f"Train Loss = {train_results.get('total', 0.0):.4f}"
        
        if val_results:
            log_msg += f", Val Loss = {val_results.get('total', 0.0):.4f}"
            log_msg += f", Val Acc = {val_results.get('element_detection_accuracy', 0.0):.4f}"
        
        self.logger.info(log_msg)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "training_history": dict(self.training_history),
            "config": self.config
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.save_dir) / f"{self.config.experiment_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.save_dir) / f"{self.config.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.training_history = defaultdict(list, checkpoint["training_history"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history"""
        if not self.training_history:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Multi-Task Training History")
        
        # Plot losses
        axes[0, 0].set_title("Total Loss")
        if "train_total" in self.training_history:
            axes[0, 0].plot(self.training_history["train_total"], label="Train")
        if "val_total" in self.training_history:
            axes[0, 0].plot(self.training_history["val_total"], label="Validation")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        
        # Plot element detection accuracy
        axes[0, 1].set_title("Element Detection Accuracy")
        if "train_element_detection_accuracy" in self.training_history:
            axes[0, 1].plot(self.training_history["train_element_detection_accuracy"], label="Train")
        if "val_element_detection_accuracy" in self.training_history:
            axes[0, 1].plot(self.training_history["val_element_detection_accuracy"], label="Validation")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        
        # Plot OCR accuracy
        axes[1, 0].set_title("OCR Character Accuracy")
        if "train_ocr_char_accuracy" in self.training_history:
            axes[1, 0].plot(self.training_history["train_ocr_char_accuracy"], label="Train")
        if "val_ocr_char_accuracy" in self.training_history:
            axes[1, 0].plot(self.training_history["val_ocr_char_accuracy"], label="Validation")
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        
        # Plot intent accuracy
        axes[1, 1].set_title("Intent Classification Accuracy")
        if "train_intent_accuracy" in self.training_history:
            axes[1, 1].plot(self.training_history["train_intent_accuracy"], label="Train")
        if "val_intent_accuracy" in self.training_history:
            axes[1, 1].plot(self.training_history["val_intent_accuracy"], label="Validation")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create model configuration
    model_config = MultiTaskConfig(
        backbone_model="efficientnet-b4",
        num_element_classes=50,
        num_intent_classes=20,
        num_action_classes=15
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        experiment_name="multi_task_demo"
    )
    
    # Create model and trainer
    model = MultiTaskModel(model_config)
    trainer = MultiTaskTrainer(model, training_config)
    
    print("Multi-task trainer created successfully!")
    print(f"Model parameters: {model.get_model_size()}")
    print(f"Training device: {training_config.device}")