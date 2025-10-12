#!/usr/bin/env python3
"""
Vision Transformer for Screen Understanding
==========================================

Advanced Vision Transformer model जो screen content को समझता है।

Features:
- Patch-based image processing
- Self-attention for spatial relationships
- Multi-scale feature extraction
- Element detection and classification
- OCR integration
- Layout understanding

Author: Computer Genie AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VisionConfig:
    """Vision Transformer configuration."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    num_classes: int = 1000


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch_size, num_channels, height, width)
        Returns:
            embeddings: (batch_size, num_patches, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
        Returns:
            context_layer: (batch_size, sequence_length, hidden_size)
            attention_probs: (batch_size, num_heads, sequence_length, sequence_length)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer, attention_probs


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_after = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
        Returns:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_probs: (batch_size, num_heads, sequence_length, sequence_length)
        """
        # Self-attention
        attention_output, attention_probs = self.attention(
            self.layernorm_before(hidden_states)
        )
        attention_output = self.dropout(attention_output)
        hidden_states = hidden_states + attention_output
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(self.layernorm_after(hidden_states)))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = hidden_states + layer_output
        
        return hidden_states, attention_probs


class VisionTransformer(nn.Module):
    """
    Vision Transformer for screen understanding.
    
    यह model screen images को analyze करके:
    - UI elements detect करता है
    - Text content extract करता है
    - Layout structure समझता है
    - Interactive elements identify करता है
    """
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches + 1, config.hidden_size)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        # Classification heads
        self.element_classifier = nn.Linear(config.hidden_size, 50)  # 50 UI element types
        self.text_detector = nn.Linear(config.hidden_size, 2)  # Text/No-text
        self.clickable_detector = nn.Linear(config.hidden_size, 2)  # Clickable/Not-clickable
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Vision Transformer.
        
        Args:
            pixel_values: (batch_size, channels, height, width)
            
        Returns:
            Dict containing:
            - element_logits: (batch_size, num_patches, 50)
            - text_logits: (batch_size, num_patches, 2)
            - clickable_logits: (batch_size, num_patches, 2)
            - attention_maps: List of attention matrices
            - features: (batch_size, num_patches + 1, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        
        # Convert image to patch embeddings
        embeddings = self.patch_embedding(pixel_values)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        
        # Pass through transformer layers
        hidden_states = embeddings
        attention_maps = []
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states)
            attention_maps.append(attention_probs)
            
        # Final layer norm
        hidden_states = self.layernorm(hidden_states)
        
        # Extract patch features (exclude CLS token)
        patch_features = hidden_states[:, 1:, :]
        
        # Classification heads
        element_logits = self.element_classifier(patch_features)
        text_logits = self.text_detector(patch_features)
        clickable_logits = self.clickable_detector(patch_features)
        
        return {
            'element_logits': element_logits,
            'text_logits': text_logits,
            'clickable_logits': clickable_logits,
            'attention_maps': attention_maps,
            'features': hidden_states,
            'patch_features': patch_features
        }
    
    def get_attention_maps(self, pixel_values: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            pixel_values: Input image
            layer_idx: Which layer's attention to return (-1 for last layer)
            
        Returns:
            attention_maps: (batch_size, num_heads, num_patches+1, num_patches+1)
        """
        outputs = self.forward(pixel_values)
        if layer_idx == -1:
            layer_idx = len(outputs['attention_maps']) - 1
        return outputs['attention_maps'][layer_idx]
    
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features for downstream tasks.
        
        Args:
            pixel_values: Input image
            
        Returns:
            features: (batch_size, num_patches, hidden_size)
        """
        outputs = self.forward(pixel_values)
        return outputs['patch_features']


class ScreenAnalyzer:
    """
    High-level interface for screen analysis using Vision Transformer.
    
    यह class Vision Transformer को use करके screen analysis करती है।
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize screen analyzer.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.config = VisionConfig()
        self.model = VisionTransformer(self.config)
        
        if model_path:
            self.load_model(model_path)
            
        self.model.eval()
        
        # Element type mapping
        self.element_types = [
            'button', 'textbox', 'label', 'image', 'link', 'dropdown',
            'checkbox', 'radio', 'slider', 'menu', 'tab', 'window',
            'dialog', 'toolbar', 'statusbar', 'scrollbar', 'table',
            'list', 'tree', 'icon', 'separator', 'panel', 'frame',
            'canvas', 'video', 'audio', 'progress', 'spinner', 'tooltip',
            'notification', 'popup', 'modal', 'sidebar', 'header',
            'footer', 'navigation', 'breadcrumb', 'pagination', 'card',
            'tile', 'badge', 'chip', 'avatar', 'calendar', 'chart',
            'graph', 'map', 'timeline', 'carousel', 'accordion', 'other'
        ]
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def analyze_screen(self, image: np.ndarray) -> Dict:
        """
        Analyze screen image and extract UI elements.
        
        Args:
            image: Screen image as numpy array (H, W, C)
            
        Returns:
            Dict containing analysis results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0))
            
        # Post-process results
        results = self.postprocess_outputs(outputs, image.shape)
        
        return results
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model input size
        from PIL import Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.config.image_size, self.config.image_size))
        
        # Convert to tensor and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return image_tensor
    
    def postprocess_outputs(self, outputs: Dict, original_shape: Tuple) -> Dict:
        """Post-process model outputs to extract meaningful results."""
        # Get predictions
        element_probs = F.softmax(outputs['element_logits'], dim=-1)
        text_probs = F.softmax(outputs['text_logits'], dim=-1)
        clickable_probs = F.softmax(outputs['clickable_logits'], dim=-1)
        
        # Convert patch predictions to spatial coordinates
        patch_size = self.config.patch_size
        patches_per_side = self.config.image_size // patch_size
        
        elements = []
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                patch_idx = i * patches_per_side + j
                
                # Get patch coordinates
                x1 = j * patch_size
                y1 = i * patch_size
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                
                # Scale to original image size
                scale_x = original_shape[1] / self.config.image_size
                scale_y = original_shape[0] / self.config.image_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Get predictions for this patch
                element_pred = element_probs[0, patch_idx].argmax().item()
                element_conf = element_probs[0, patch_idx].max().item()
                
                text_pred = text_probs[0, patch_idx, 1].item()
                clickable_pred = clickable_probs[0, patch_idx, 1].item()
                
                # Only include confident predictions
                if element_conf > 0.5:
                    elements.append({
                        'type': self.element_types[element_pred],
                        'confidence': element_conf,
                        'bbox': [x1, y1, x2, y2],
                        'has_text': text_pred > 0.5,
                        'text_confidence': text_pred,
                        'is_clickable': clickable_pred > 0.5,
                        'clickable_confidence': clickable_pred
                    })
        
        return {
            'elements': elements,
            'attention_maps': outputs['attention_maps'],
            'raw_outputs': outputs
        }


# Example usage
if __name__ == "__main__":
    # Create model
    config = VisionConfig()
    model = VisionTransformer(config)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print("Vision Transformer Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: List of {len(value)} attention maps")
    
    # Create screen analyzer
    analyzer = ScreenAnalyzer()
    
    # Analyze dummy screen
    dummy_screen = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = analyzer.analyze_screen(dummy_screen)
    
    print(f"\nScreen Analysis Results:")
    print(f"  Found {len(results['elements'])} UI elements")
    for element in results['elements'][:5]:  # Show first 5
        print(f"    {element['type']}: {element['confidence']:.3f} at {element['bbox']}")