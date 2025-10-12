"""
Feature-based Knowledge Distillation
फीचर-आधारित ज्ञान आसवन

Advanced feature distillation methods including attention transfer, relation-based distillation,
and structured knowledge transfer from intermediate layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import copy
import matplotlib.pyplot as plt


class FeatureMatchingMethod(Enum):
    """Methods for matching features between teacher and student"""
    MSE = "mse"                          # Mean Squared Error
    L1 = "l1"                           # L1 Loss
    COSINE = "cosine"                   # Cosine Similarity
    CORRELATION = "correlation"          # Correlation-based
    ATTENTION = "attention"             # Attention-based matching
    GRAM_MATRIX = "gram_matrix"         # Gram matrix matching
    JACOBIAN = "jacobian"               # Jacobian matching


class AttentionTransferMethod(Enum):
    """Methods for attention transfer"""
    SPATIAL = "spatial"                 # Spatial attention maps
    CHANNEL = "channel"                 # Channel attention
    MIXED = "mixed"                     # Mixed attention
    SELF_ATTENTION = "self_attention"   # Self-attention transfer
    CROSS_ATTENTION = "cross_attention" # Cross-attention transfer


class RelationDistillationMethod(Enum):
    """Methods for relation-based distillation"""
    INSTANCE_RELATION = "instance"      # Instance-wise relations
    STRUCTURAL_RELATION = "structural"  # Structural relations
    SEMANTIC_RELATION = "semantic"      # Semantic relations
    GRAPH_RELATION = "graph"           # Graph-based relations


@dataclass
class FeatureDistillationConfig:
    """Configuration for feature distillation"""
    # Feature matching parameters
    feature_matching_method: FeatureMatchingMethod = FeatureMatchingMethod.MSE
    feature_loss_weight: float = 1.0
    
    # Attention transfer parameters
    attention_method: AttentionTransferMethod = AttentionTransferMethod.SPATIAL
    attention_loss_weight: float = 1000.0
    attention_normalize: bool = True
    
    # Relation distillation parameters
    relation_method: RelationDistillationMethod = RelationDistillationMethod.INSTANCE_RELATION
    relation_loss_weight: float = 1.0
    
    # Layer selection
    teacher_layers: List[str] = field(default_factory=list)
    student_layers: List[str] = field(default_factory=list)
    
    # Adaptation parameters
    use_adaptation: bool = True
    adaptation_method: str = "linear"    # linear, conv, attention
    
    # Training parameters
    temperature: float = 4.0
    alpha: float = 0.7                   # Feature loss weight
    beta: float = 0.3                    # Task loss weight
    
    # Device and logging
    device: str = "cuda"
    verbose: bool = True


class FeatureAdapter(nn.Module):
    """Adapter for matching feature dimensions between teacher and student"""
    
    def __init__(self, student_dim: int, teacher_dim: int, method: str = "linear"):
        super().__init__()
        self.method = method
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        if method == "linear":
            self.adapter = nn.Linear(student_dim, teacher_dim)
        elif method == "conv":
            # For convolutional features
            self.adapter = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)
        elif method == "attention":
            self.adapter = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.ReLU(),
                nn.Linear(teacher_dim, teacher_dim)
            )
        else:
            raise ValueError(f"Unknown adaptation method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class AttentionTransfer(nn.Module):
    """Attention transfer mechanism"""
    
    def __init__(self, method: AttentionTransferMethod):
        super().__init__()
        self.method = method
    
    def compute_attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Compute attention map from feature map"""
        if self.method == AttentionTransferMethod.SPATIAL:
            # Spatial attention: sum across channels
            attention = torch.sum(feature_map.pow(2), dim=1, keepdim=True)
            
        elif self.method == AttentionTransferMethod.CHANNEL:
            # Channel attention: global average pooling
            attention = F.adaptive_avg_pool2d(feature_map.pow(2), (1, 1))
            
        elif self.method == AttentionTransferMethod.MIXED:
            # Mixed attention: both spatial and channel
            spatial_att = torch.sum(feature_map.pow(2), dim=1, keepdim=True)
            channel_att = F.adaptive_avg_pool2d(feature_map.pow(2), (1, 1))
            attention = spatial_att + channel_att.expand_as(spatial_att)
            
        else:
            # Default to spatial attention
            attention = torch.sum(feature_map.pow(2), dim=1, keepdim=True)
        
        return attention
    
    def forward(self, student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute attention transfer loss"""
        student_attention = self.compute_attention_map(student_features)
        teacher_attention = self.compute_attention_map(teacher_features)
        
        # Normalize attention maps
        student_attention = F.normalize(student_attention.view(student_attention.size(0), -1), p=2, dim=1)
        teacher_attention = F.normalize(teacher_attention.view(teacher_attention.size(0), -1), p=2, dim=1)
        
        # Compute MSE loss
        loss = F.mse_loss(student_attention, teacher_attention)
        return loss


class RelationDistillation(nn.Module):
    """Relation-based knowledge distillation"""
    
    def __init__(self, method: RelationDistillationMethod):
        super().__init__()
        self.method = method
    
    def compute_instance_relations(self, features: torch.Tensor) -> torch.Tensor:
        """Compute instance-wise relations"""
        # Flatten features
        batch_size = features.size(0)
        features_flat = features.view(batch_size, -1)
        
        # Normalize features
        features_norm = F.normalize(features_flat, p=2, dim=1)
        
        # Compute pairwise similarities
        relations = torch.mm(features_norm, features_norm.t())
        return relations
    
    def compute_structural_relations(self, features: torch.Tensor) -> torch.Tensor:
        """Compute structural relations"""
        batch_size, channels = features.size(0), features.size(1)
        
        # Reshape for structural analysis
        if len(features.shape) == 4:  # Conv features
            features_reshaped = features.view(batch_size, channels, -1)
            # Compute channel-wise correlations
            relations = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        else:  # FC features
            features_reshaped = features.view(batch_size, -1)
            relations = torch.mm(features_reshaped, features_reshaped.t())
        
        return relations
    
    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute relation distillation loss"""
        if self.method == RelationDistillationMethod.INSTANCE_RELATION:
            student_relations = self.compute_instance_relations(student_features)
            teacher_relations = self.compute_instance_relations(teacher_features)
            
        elif self.method == RelationDistillationMethod.STRUCTURAL_RELATION:
            student_relations = self.compute_structural_relations(student_features)
            teacher_relations = self.compute_structural_relations(teacher_features)
            
        else:
            # Default to instance relations
            student_relations = self.compute_instance_relations(student_features)
            teacher_relations = self.compute_instance_relations(teacher_features)
        
        # Compute loss
        loss = F.mse_loss(student_relations, teacher_relations)
        return loss


class FeatureDistiller:
    """Feature-based knowledge distillation"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: FeatureDistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Move models to device
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        self.teacher_model.eval()
        
        # Feature hooks
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []
        
        # Feature adapters
        self.adapters = nn.ModuleDict()
        
        # Attention transfer
        self.attention_transfer = AttentionTransfer(config.attention_method)
        
        # Relation distillation
        self.relation_distillation = RelationDistillation(config.relation_method)
        
        # Initialize feature extraction
        self._setup_feature_hooks()
        
        # Initialize optimizer
        params = list(self.student_model.parameters())
        if self.adapters:
            params.extend(list(self.adapters.parameters()))
        
        self.optimizer = torch.optim.Adam(params, lr=0.001)
        
        # Training statistics
        self.training_history = {
            'total_loss': [],
            'task_loss': [],
            'feature_loss': [],
            'attention_loss': [],
            'relation_loss': [],
            'accuracy': []
        }
    
    def _setup_feature_hooks(self):
        """Setup hooks for feature extraction"""
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        # Register hooks for specified layers
        if self.config.teacher_layers and self.config.student_layers:
            for teacher_layer, student_layer in zip(self.config.teacher_layers, self.config.student_layers):
                # Get teacher layer
                teacher_module = self._get_layer_by_name(self.teacher_model, teacher_layer)
                if teacher_module is not None:
                    hook = teacher_module.register_forward_hook(get_teacher_hook(teacher_layer))
                    self.teacher_hooks.append(hook)
                
                # Get student layer
                student_module = self._get_layer_by_name(self.student_model, student_layer)
                if student_module is not None:
                    hook = student_module.register_forward_hook(get_student_hook(student_layer))
                    self.student_hooks.append(hook)
                
                # Setup adapter if needed
                if self.config.use_adaptation:
                    self._setup_adapter(teacher_layer, student_layer, teacher_module, student_module)
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from model"""
        try:
            parts = layer_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            self.logger.warning(f"Layer {layer_name} not found in model")
            return None
    
    def _setup_adapter(self, teacher_layer: str, student_layer: str,
                      teacher_module: nn.Module, student_module: nn.Module):
        """Setup feature adapter"""
        # Get feature dimensions (this is a simplified approach)
        # In practice, you might need to run a forward pass to get actual dimensions
        
        if hasattr(teacher_module, 'out_features'):
            teacher_dim = teacher_module.out_features
        elif hasattr(teacher_module, 'out_channels'):
            teacher_dim = teacher_module.out_channels
        else:
            teacher_dim = 512  # Default
        
        if hasattr(student_module, 'out_features'):
            student_dim = student_module.out_features
        elif hasattr(student_module, 'out_channels'):
            student_dim = student_module.out_channels
        else:
            student_dim = 256  # Default
        
        if teacher_dim != student_dim:
            adapter = FeatureAdapter(student_dim, teacher_dim, self.config.adaptation_method)
            adapter.to(self.device)
            self.adapters[f"{student_layer}_to_{teacher_layer}"] = adapter
    
    def compute_feature_loss(self) -> torch.Tensor:
        """Compute feature matching loss"""
        total_loss = 0.0
        num_layers = 0
        
        for teacher_layer, student_layer in zip(self.config.teacher_layers, self.config.student_layers):
            if teacher_layer in self.teacher_features and student_layer in self.student_features:
                teacher_feat = self.teacher_features[teacher_layer]
                student_feat = self.student_features[student_layer]
                
                # Apply adapter if available
                adapter_key = f"{student_layer}_to_{teacher_layer}"
                if adapter_key in self.adapters:
                    student_feat = self.adapters[adapter_key](student_feat)
                
                # Compute feature loss based on method
                if self.config.feature_matching_method == FeatureMatchingMethod.MSE:
                    loss = F.mse_loss(student_feat, teacher_feat)
                elif self.config.feature_matching_method == FeatureMatchingMethod.L1:
                    loss = F.l1_loss(student_feat, teacher_feat)
                elif self.config.feature_matching_method == FeatureMatchingMethod.COSINE:
                    student_flat = student_feat.view(student_feat.size(0), -1)
                    teacher_flat = teacher_feat.view(teacher_feat.size(0), -1)
                    loss = 1 - F.cosine_similarity(student_flat, teacher_flat).mean()
                else:
                    loss = F.mse_loss(student_feat, teacher_feat)
                
                total_loss += loss
                num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def compute_attention_loss(self) -> torch.Tensor:
        """Compute attention transfer loss"""
        total_loss = 0.0
        num_layers = 0
        
        for teacher_layer, student_layer in zip(self.config.teacher_layers, self.config.student_layers):
            if teacher_layer in self.teacher_features and student_layer in self.student_features:
                teacher_feat = self.teacher_features[teacher_layer]
                student_feat = self.student_features[student_layer]
                
                # Only compute attention loss for conv features
                if len(teacher_feat.shape) == 4 and len(student_feat.shape) == 4:
                    loss = self.attention_transfer(student_feat, teacher_feat)
                    total_loss += loss
                    num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def compute_relation_loss(self) -> torch.Tensor:
        """Compute relation distillation loss"""
        total_loss = 0.0
        num_layers = 0
        
        for teacher_layer, student_layer in zip(self.config.teacher_layers, self.config.student_layers):
            if teacher_layer in self.teacher_features and student_layer in self.student_features:
                teacher_feat = self.teacher_features[teacher_layer]
                student_feat = self.student_features[student_layer]
                
                loss = self.relation_distillation(student_feat, teacher_feat)
                total_loss += loss
                num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        self.student_model.train()
        self.teacher_model.eval()
        
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Clear previous features
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch_data)
        
        student_logits = self.student_model(batch_data)
        
        # Compute task loss
        task_loss = F.cross_entropy(student_logits, batch_labels)
        
        # Compute feature losses
        feature_loss = self.compute_feature_loss()
        attention_loss = self.compute_attention_loss()
        relation_loss = self.compute_relation_loss()
        
        # Total loss
        total_loss = (self.config.beta * task_loss +
                     self.config.alpha * self.config.feature_loss_weight * feature_loss +
                     self.config.attention_loss_weight * attention_loss +
                     self.config.relation_loss_weight * relation_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        if self.adapters:
            torch.nn.utils.clip_grad_norm_(self.adapters.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            _, predicted = torch.max(student_logits, 1)
            accuracy = (predicted == batch_labels).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'feature_loss': feature_loss.item(),
            'attention_loss': attention_loss.item(),
            'relation_loss': relation_loss.item(),
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = 100) -> Dict[str, List[float]]:
        """Train with feature distillation"""
        if self.config.verbose:
            self.logger.info(f"Starting feature distillation training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_data, batch_labels in train_loader:
                step_metrics = self.train_step(batch_data, batch_labels)
                
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
            
            # Average metrics
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Store history
            for key, value in avg_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Log progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Total Loss: {avg_metrics['total_loss']:.4f}, "
                    f"Feature Loss: {avg_metrics['feature_loss']:.4f}, "
                    f"Attention Loss: {avg_metrics['attention_loss']:.4f}, "
                    f"Accuracy: {avg_metrics['accuracy']:.4f}"
                )
        
        return self.training_history
    
    def cleanup(self):
        """Remove hooks and cleanup"""
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()
        
        self.teacher_hooks.clear()
        self.student_hooks.clear()
        self.teacher_features.clear()
        self.student_features.clear()


class GramMatrixDistillation(nn.Module):
    """Gram matrix-based feature distillation"""
    
    def __init__(self):
        super().__init__()
    
    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        batch_size, channels, height, width = features.size()
        features_reshaped = features.view(batch_size, channels, height * width)
        
        # Compute Gram matrix
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        
        # Normalize
        gram = gram / (channels * height * width)
        
        return gram
    
    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix distillation loss"""
        student_gram = self.gram_matrix(student_features)
        teacher_gram = self.gram_matrix(teacher_features)
        
        loss = F.mse_loss(student_gram, teacher_gram)
        return loss


class JacobianMatching(nn.Module):
    """Jacobian-based feature matching"""
    
    def __init__(self):
        super().__init__()
    
    def compute_jacobian(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix"""
        batch_size = features.size(0)
        jacobians = []
        
        for i in range(batch_size):
            feature = features[i:i+1]
            target = targets[i:i+1]
            
            # Compute gradients
            grad = torch.autograd.grad(
                outputs=feature.sum(),
                inputs=target,
                create_graph=True,
                retain_graph=True
            )[0]
            
            jacobians.append(grad.view(-1))
        
        return torch.stack(jacobians)
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matching loss"""
        inputs.requires_grad_(True)
        
        student_jacobian = self.compute_jacobian(student_features, inputs)
        teacher_jacobian = self.compute_jacobian(teacher_features, inputs)
        
        loss = F.mse_loss(student_jacobian, teacher_jacobian)
        return loss


# Example usage
if __name__ == "__main__":
    # Create models
    teacher = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    student = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Create configuration
    config = FeatureDistillationConfig(
        teacher_layers=['0', '2'],  # Conv layers
        student_layers=['0', '2'],  # Corresponding student layers
        feature_matching_method=FeatureMatchingMethod.MSE,
        attention_method=AttentionTransferMethod.SPATIAL,
        relation_method=RelationDistillationMethod.INSTANCE_RELATION
    )
    
    # Create feature distiller
    distiller = FeatureDistiller(teacher, student, config)
    
    print("Feature Distillation Components Created Successfully!")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters())}")
    print(f"Adapter parameters: {sum(p.numel() for p in distiller.adapters.parameters())}")
    
    # Test with dummy data
    dummy_input = torch.randn(4, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (4,))
    
    # Test forward pass
    with torch.no_grad():
        teacher_output = teacher(dummy_input)
        student_output = student(dummy_input)
    
    print(f"Teacher output shape: {teacher_output.shape}")
    print(f"Student output shape: {student_output.shape}")
    
    # Cleanup
    distiller.cleanup()