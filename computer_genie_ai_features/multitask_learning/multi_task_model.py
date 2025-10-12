"""
Multi-Task Model Architecture for Computer Genie
मल्टी-टास्क मॉडल आर्किटेक्चर - Computer Genie के लिए

Implements a unified model that can simultaneously perform:
- UI element detection and classification
- Optical Character Recognition (OCR)
- User intent understanding
- Action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from transformers import AutoModel, AutoTokenizer
import math


class TaskType(Enum):
    """Types of tasks supported by the multi-task model"""
    ELEMENT_DETECTION = "element_detection"
    OCR = "ocr"
    INTENT_CLASSIFICATION = "intent_classification"
    ACTION_PREDICTION = "action_prediction"
    LAYOUT_UNDERSTANDING = "layout_understanding"
    ACCESSIBILITY_ANALYSIS = "accessibility_analysis"


@dataclass
class TaskWeight:
    """Weights for different tasks in multi-task learning"""
    element_detection: float = 1.0
    ocr: float = 1.0
    intent_classification: float = 1.0
    action_prediction: float = 0.8
    layout_understanding: float = 0.6
    accessibility_analysis: float = 0.4
    
    def get_weight(self, task_type: TaskType) -> float:
        """Get weight for specific task type"""
        return getattr(self, task_type.value, 1.0)
    
    def normalize(self) -> 'TaskWeight':
        """Normalize weights to sum to number of tasks"""
        total = sum([self.element_detection, self.ocr, self.intent_classification,
                    self.action_prediction, self.layout_understanding, 
                    self.accessibility_analysis])
        num_tasks = 6
        factor = num_tasks / total
        
        return TaskWeight(
            element_detection=self.element_detection * factor,
            ocr=self.ocr * factor,
            intent_classification=self.intent_classification * factor,
            action_prediction=self.action_prediction * factor,
            layout_understanding=self.layout_understanding * factor,
            accessibility_analysis=self.accessibility_analysis * factor
        )


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model"""
    # Model architecture
    backbone_model: str = "efficientnet-b4"
    shared_feature_dim: int = 1280
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    # Task-specific configurations
    num_element_classes: int = 50  # UI element types
    max_text_length: int = 256     # For OCR
    num_intent_classes: int = 20   # User intents
    num_action_classes: int = 15   # Possible actions
    
    # Training parameters
    task_weights: TaskWeight = None
    use_attention: bool = True
    use_cross_task_learning: bool = True
    freeze_backbone_epochs: int = 5
    
    # Language support
    supported_languages: List[str] = None
    multilingual_ocr: bool = True
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = TaskWeight()
        if self.supported_languages is None:
            self.supported_languages = ["en", "hi", "es", "fr", "de", "zh", "ja", "ar"]


class SharedBackbone(nn.Module):
    """Shared feature extraction backbone for all tasks"""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained backbone
        if "efficientnet" in config.backbone_model:
            self.backbone = models.efficientnet_b4(pretrained=True)
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            backbone_out_dim = 1792
        elif "resnet" in config.backbone_model:
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()
            backbone_out_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone_model}")
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, config.shared_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.shared_feature_dim, config.shared_feature_dim)
        )
        
        # Cross-task attention if enabled
        if config.use_attention:
            self.cross_task_attention = nn.MultiheadAttention(
                embed_dim=config.shared_feature_dim,
                num_heads=8,
                dropout=config.dropout_rate
            )
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(config.shared_feature_dim, config.shared_feature_dim, 3, padding=1),
            nn.BatchNorm2d(config.shared_feature_dim),
            nn.ReLU(),
            nn.Conv2d(config.shared_feature_dim, config.shared_feature_dim, 3, padding=1),
            nn.BatchNorm2d(config.shared_feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract shared features from input"""
        batch_size = x.size(0)
        
        # Extract features using backbone
        features = self.backbone(x)  # [B, backbone_out_dim]
        
        # Project features
        projected_features = self.feature_projection(features)  # [B, shared_feature_dim]
        
        # Reshape for spatial operations if needed
        # Assuming features are flattened, we need to reshape for conv operations
        spatial_size = int(math.sqrt(projected_features.size(1) // self.config.shared_feature_dim))
        if spatial_size * spatial_size * self.config.shared_feature_dim == projected_features.size(1):
            spatial_features = projected_features.view(
                batch_size, self.config.shared_feature_dim, spatial_size, spatial_size
            )
        else:
            # Create spatial features through adaptive pooling
            spatial_features = projected_features.unsqueeze(-1).unsqueeze(-1)
            spatial_features = F.adaptive_avg_pool2d(spatial_features, (7, 7))
        
        # Enhance features
        enhanced_features = self.feature_enhancer(spatial_features)
        
        # Apply cross-task attention if enabled
        if self.config.use_attention:
            # Flatten for attention
            flat_features = enhanced_features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
            attended_features, attention_weights = self.cross_task_attention(
                flat_features, flat_features, flat_features
            )
            attended_features = attended_features.permute(1, 2, 0).view_as(enhanced_features)
        else:
            attended_features = enhanced_features
            attention_weights = None
        
        return {
            "raw_features": features,
            "projected_features": projected_features,
            "spatial_features": attended_features,
            "attention_weights": attention_weights
        }


class TaskHead(nn.Module):
    """Base class for task-specific heads"""
    
    def __init__(self, input_dim: int, config: MultiTaskConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Common layers
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")


class ElementDetectionHead(TaskHead):
    """Head for UI element detection and classification"""
    
    def __init__(self, input_dim: int, config: MultiTaskConfig):
        super().__init__(input_dim, config)
        
        # Detection layers
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4)  # x, y, w, h
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, config.num_element_classes)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for element detection"""
        adapted_features = self.feature_adapter(features)
        
        # Predict bounding boxes, classes, and confidence
        bboxes = self.bbox_regressor(adapted_features)
        class_logits = self.classifier(adapted_features)
        confidence = self.confidence_estimator(adapted_features)
        
        return {
            "bboxes": bboxes,
            "class_logits": class_logits,
            "confidence": confidence
        }


class OCRHead(TaskHead):
    """Head for Optical Character Recognition"""
    
    def __init__(self, input_dim: int, config: MultiTaskConfig):
        super().__init__(input_dim, config)
        
        # OCR-specific layers
        self.text_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # Character prediction (supporting multiple languages)
        vocab_size = 10000  # Extended vocabulary for multilingual support
        self.char_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, vocab_size)
        )
        
        # Text region detection
        self.text_region_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4)  # Text region coordinates
        )
        
        # Language detection
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, len(config.supported_languages))
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for OCR"""
        adapted_features = self.feature_adapter(features)
        
        # Prepare sequence for LSTM (simulate text sequence)
        batch_size = adapted_features.size(0)
        seq_length = self.config.max_text_length
        
        # Expand features to sequence
        sequence_features = adapted_features.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Encode text sequence
        lstm_output, _ = self.text_encoder(sequence_features)
        
        # Decode characters
        char_logits = self.char_decoder(lstm_output)
        
        # Detect text regions
        text_regions = self.text_region_detector(adapted_features)
        
        # Classify language
        language_logits = self.language_classifier(adapted_features)
        
        return {
            "char_logits": char_logits,
            "text_regions": text_regions,
            "language_logits": language_logits
        }


class IntentClassificationHead(TaskHead):
    """Head for user intent understanding"""
    
    def __init__(self, input_dim: int, config: MultiTaskConfig):
        super().__init__(input_dim, config)
        
        # Intent classification layers
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, config.num_intent_classes)
        )
        
        # Context understanding
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # Urgency estimation
        self.urgency_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for intent classification"""
        adapted_features = self.feature_adapter(features)
        
        # Classify intent
        intent_logits = self.intent_classifier(adapted_features)
        
        # Encode context
        context_features = self.context_encoder(adapted_features)
        
        # Estimate urgency
        urgency = self.urgency_estimator(adapted_features)
        
        return {
            "intent_logits": intent_logits,
            "context_features": context_features,
            "urgency": urgency
        }


class ActionPredictionHead(TaskHead):
    """Head for predicting next actions"""
    
    def __init__(self, input_dim: int, config: MultiTaskConfig):
        super().__init__(input_dim, config)
        
        # Action prediction layers
        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, config.num_action_classes)
        )
        
        # Action parameters (coordinates, text input, etc.)
        self.action_params = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 8)  # Various action parameters
        )
        
        # Success probability
        self.success_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for action prediction"""
        adapted_features = self.feature_adapter(features)
        
        # Predict action
        action_logits = self.action_predictor(adapted_features)
        
        # Predict action parameters
        action_params = self.action_params(adapted_features)
        
        # Predict success probability
        success_prob = self.success_predictor(adapted_features)
        
        return {
            "action_logits": action_logits,
            "action_params": action_params,
            "success_probability": success_prob
        }


class MultiTaskModel(nn.Module):
    """Main multi-task model combining all components"""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        # Shared backbone
        self.backbone = SharedBackbone(config)
        
        # Task-specific heads
        feature_dim = config.shared_feature_dim
        self.element_detection_head = ElementDetectionHead(feature_dim, config)
        self.ocr_head = OCRHead(feature_dim, config)
        self.intent_head = IntentClassificationHead(feature_dim, config)
        self.action_head = ActionPredictionHead(feature_dim, config)
        
        # Task coordination layer
        if config.use_cross_task_learning:
            self.task_coordinator = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout_rate
            )
    
    def forward(self, x: torch.Tensor, 
                active_tasks: Optional[List[TaskType]] = None) -> Dict[str, Any]:
        """Forward pass through multi-task model"""
        if active_tasks is None:
            active_tasks = list(TaskType)
        
        # Extract shared features
        backbone_output = self.backbone(x)
        shared_features = backbone_output["spatial_features"]
        
        # Initialize outputs
        outputs = {
            "backbone_features": backbone_output,
            "task_outputs": {}
        }
        
        # Process each active task
        task_features = []
        
        if TaskType.ELEMENT_DETECTION in active_tasks:
            detection_output = self.element_detection_head(shared_features)
            outputs["task_outputs"]["element_detection"] = detection_output
            task_features.append(detection_output["class_logits"])
        
        if TaskType.OCR in active_tasks:
            ocr_output = self.ocr_head(shared_features)
            outputs["task_outputs"]["ocr"] = ocr_output
            # Use language logits as task feature
            task_features.append(ocr_output["language_logits"])
        
        if TaskType.INTENT_CLASSIFICATION in active_tasks:
            intent_output = self.intent_head(shared_features)
            outputs["task_outputs"]["intent_classification"] = intent_output
            task_features.append(intent_output["intent_logits"])
        
        if TaskType.ACTION_PREDICTION in active_tasks:
            action_output = self.action_head(shared_features)
            outputs["task_outputs"]["action_prediction"] = action_output
            task_features.append(action_output["action_logits"])
        
        # Apply cross-task coordination if enabled
        if self.config.use_cross_task_learning and len(task_features) > 1:
            # Stack task features for attention
            stacked_features = torch.stack(task_features, dim=1)  # [B, num_tasks, feature_dim]
            
            # Apply cross-task attention
            coordinated_features, task_attention = self.task_coordinator(
                stacked_features, stacked_features, stacked_features
            )
            
            outputs["cross_task_attention"] = task_attention
            outputs["coordinated_features"] = coordinated_features
        
        return outputs
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_task_parameters(self, task_type: TaskType) -> List[nn.Parameter]:
        """Get parameters for specific task"""
        if task_type == TaskType.ELEMENT_DETECTION:
            return list(self.element_detection_head.parameters())
        elif task_type == TaskType.OCR:
            return list(self.ocr_head.parameters())
        elif task_type == TaskType.INTENT_CLASSIFICATION:
            return list(self.intent_head.parameters())
        elif task_type == TaskType.ACTION_PREDICTION:
            return list(self.action_head.parameters())
        else:
            return []
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        task_params = {}
        for task_type in TaskType:
            task_params[task_type.value] = sum(
                p.numel() for p in self.get_task_parameters(task_type)
            )
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": sum(p.numel() for p in self.backbone.parameters()),
            "task_parameters": task_params
        }


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = MultiTaskConfig(
        backbone_model="efficientnet-b4",
        num_element_classes=50,
        num_intent_classes=20,
        num_action_classes=15,
        use_attention=True,
        use_cross_task_learning=True
    )
    
    # Create model
    model = MultiTaskModel(config)
    
    # Example input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print("Model created successfully!")
    print(f"Model size: {model.get_model_size()}")
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Task outputs: {list(outputs['task_outputs'].keys())}")