"""
Attention Visualizer for Computer Genie
ध्यान दृश्यकर्ता - Computer Genie के लिए

Generates visual attention maps showing where AI models focus their attention
when making decisions about screen elements and user interactions.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AttentionConfig:
    """Configuration for attention visualization"""
    heatmap_alpha: float = 0.6
    colormap: str = 'jet'
    blur_kernel_size: int = 15
    min_attention_threshold: float = 0.1
    max_overlay_intensity: float = 0.8
    save_attention_maps: bool = True
    attention_map_resolution: Tuple[int, int] = (224, 224)
    multi_head_aggregation: str = 'mean'  # 'mean', 'max', 'weighted'


class AttentionMap:
    """Represents an attention map with metadata"""
    
    def __init__(self, attention_weights: np.ndarray, 
                 source_image: np.ndarray,
                 model_layer: str,
                 prediction_confidence: float,
                 element_boxes: Optional[List[Dict]] = None):
        self.attention_weights = attention_weights
        self.source_image = source_image
        self.model_layer = model_layer
        self.prediction_confidence = prediction_confidence
        self.element_boxes = element_boxes or []
        self.timestamp = np.datetime64('now')
    
    def get_top_attention_regions(self, top_k: int = 5) -> List[Dict]:
        """Get top-k regions with highest attention"""
        h, w = self.attention_weights.shape
        flat_attention = self.attention_weights.flatten()
        top_indices = np.argpartition(flat_attention, -top_k)[-top_k:]
        
        regions = []
        for idx in top_indices:
            y, x = divmod(idx, w)
            regions.append({
                'position': (x, y),
                'attention_score': float(flat_attention[idx]),
                'relative_position': (x/w, y/h)
            })
        
        return sorted(regions, key=lambda x: x['attention_score'], reverse=True)


class AttentionVisualizer:
    """Visualizes attention maps for explainable AI"""
    
    def __init__(self, config: AttentionConfig = None):
        self.config = config or AttentionConfig()
        self.attention_history = []
        
    def extract_attention_weights(self, model: nn.Module, 
                                input_tensor: torch.Tensor,
                                layer_names: List[str] = None) -> Dict[str, torch.Tensor]:
        """Extract attention weights from transformer layers"""
        attention_weights = {}
        hooks = []
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights'):
                    attention_weights[name] = module.attention_weights.detach()
                elif isinstance(output, tuple) and len(output) > 1:
                    # For standard attention modules
                    attention_weights[name] = output[1].detach()
            return hook
        
        # Register hooks for attention layers
        for name, module in model.named_modules():
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn']):
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(attention_hook(name))
                    hooks.append(hook)
        
        # Forward pass to collect attention weights
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attention_weights
    
    def create_attention_heatmap(self, attention_weights: torch.Tensor,
                               input_shape: Tuple[int, int]) -> np.ndarray:
        """Create heatmap from attention weights"""
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            # Aggregate multi-head attention
            if self.config.multi_head_aggregation == 'mean':
                attention = attention_weights.mean(dim=1)
            elif self.config.multi_head_aggregation == 'max':
                attention = attention_weights.max(dim=1)[0]
            else:  # weighted
                attention = attention_weights.mean(dim=1)
        else:
            attention = attention_weights
        
        # Take the first batch and average over sequence dimension
        if attention.dim() == 3:
            attention = attention[0].mean(dim=0)
        elif attention.dim() == 2:
            attention = attention[0]
        
        # Convert to numpy and reshape to spatial dimensions
        attention_np = attention.cpu().numpy()
        
        # Reshape to 2D if needed (assuming square patches)
        if attention_np.ndim == 1:
            size = int(np.sqrt(len(attention_np)))
            attention_np = attention_np.reshape(size, size)
        
        # Resize to input shape
        attention_resized = cv2.resize(attention_np, input_shape, 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        attention_resized = (attention_resized - attention_resized.min()) / \
                          (attention_resized.max() - attention_resized.min() + 1e-8)
        
        return attention_resized
    
    def overlay_attention_on_image(self, image: np.ndarray, 
                                 attention_map: np.ndarray) -> np.ndarray:
        """Overlay attention heatmap on original image"""
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur to attention map for smoother visualization
        blurred_attention = cv2.GaussianBlur(
            attention_map, 
            (self.config.blur_kernel_size, self.config.blur_kernel_size), 
            0
        )
        
        # Apply threshold to focus on high attention areas
        thresholded_attention = np.where(
            blurred_attention > self.config.min_attention_threshold,
            blurred_attention,
            0
        )
        
        # Create colormap
        colormap = cm.get_cmap(self.config.colormap)
        colored_attention = colormap(thresholded_attention)[:, :, :3]
        colored_attention = (colored_attention * 255).astype(np.uint8)
        
        # Blend with original image
        overlay = cv2.addWeighted(
            image, 
            1 - self.config.heatmap_alpha,
            colored_attention,
            self.config.heatmap_alpha,
            0
        )
        
        return overlay
    
    def generate_attention_explanation(self, attention_map: AttentionMap) -> Dict[str, Any]:
        """Generate textual explanation of attention patterns"""
        top_regions = attention_map.get_top_attention_regions()
        
        explanation = {
            'summary': f"Model focused primarily on {len(top_regions)} key regions",
            'confidence': attention_map.prediction_confidence,
            'layer': attention_map.model_layer,
            'top_regions': top_regions,
            'attention_distribution': {
                'mean': float(np.mean(attention_map.attention_weights)),
                'std': float(np.std(attention_map.attention_weights)),
                'max': float(np.max(attention_map.attention_weights)),
                'entropy': float(-np.sum(attention_map.attention_weights * 
                                       np.log(attention_map.attention_weights + 1e-8)))
            }
        }
        
        # Add element-specific explanations if available
        if attention_map.element_boxes:
            element_attention = self._analyze_element_attention(
                attention_map.attention_weights, 
                attention_map.element_boxes
            )
            explanation['element_analysis'] = element_attention
        
        return explanation
    
    def _analyze_element_attention(self, attention_weights: np.ndarray, 
                                 element_boxes: List[Dict]) -> List[Dict]:
        """Analyze attention for specific UI elements"""
        h, w = attention_weights.shape
        element_analysis = []
        
        for element in element_boxes:
            x1, y1, x2, y2 = element['bbox']
            # Convert to attention map coordinates
            x1_norm, y1_norm = int(x1 * w), int(y1 * h)
            x2_norm, y2_norm = int(x2 * w), int(y2 * h)
            
            # Extract attention for this element
            element_attention = attention_weights[y1_norm:y2_norm, x1_norm:x2_norm]
            
            if element_attention.size > 0:
                analysis = {
                    'element_type': element.get('type', 'unknown'),
                    'element_text': element.get('text', ''),
                    'bbox': element['bbox'],
                    'attention_score': float(np.mean(element_attention)),
                    'max_attention': float(np.max(element_attention)),
                    'attention_coverage': float(np.sum(element_attention > 0.1) / element_attention.size)
                }
                element_analysis.append(analysis)
        
        return sorted(element_analysis, key=lambda x: x['attention_score'], reverse=True)
    
    def visualize_multi_layer_attention(self, model: nn.Module,
                                      input_tensor: torch.Tensor,
                                      input_image: np.ndarray,
                                      layer_names: List[str] = None) -> Dict[str, AttentionMap]:
        """Visualize attention across multiple layers"""
        attention_weights = self.extract_attention_weights(model, input_tensor, layer_names)
        attention_maps = {}
        
        for layer_name, weights in attention_weights.items():
            # Create heatmap
            heatmap = self.create_attention_heatmap(weights, input_image.shape[:2])
            
            # Create attention map object
            attention_map = AttentionMap(
                attention_weights=heatmap,
                source_image=input_image,
                model_layer=layer_name,
                prediction_confidence=0.0  # Will be filled by calling code
            )
            
            attention_maps[layer_name] = attention_map
            
            # Store in history
            self.attention_history.append(attention_map)
        
        return attention_maps
    
    def save_attention_visualization(self, attention_map: AttentionMap, 
                                   save_path: str) -> None:
        """Save attention visualization to file"""
        overlay = self.overlay_attention_on_image(
            attention_map.source_image,
            attention_map.attention_weights
        )
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(attention_map.source_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention_map.attention_weights, cmap=self.config.colormap)
        axes[1].set_title(f'Attention Map - {attention_map.model_layer}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (Confidence: {attention_map.prediction_confidence:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attention patterns over time"""
        if not self.attention_history:
            return {}
        
        all_attentions = [am.attention_weights for am in self.attention_history]
        all_confidences = [am.prediction_confidence for am in self.attention_history]
        
        return {
            'total_visualizations': len(self.attention_history),
            'average_confidence': np.mean(all_confidences),
            'attention_consistency': self._calculate_attention_consistency(all_attentions),
            'layer_distribution': self._get_layer_distribution(),
            'recent_performance': {
                'last_10_avg_confidence': np.mean(all_confidences[-10:]) if len(all_confidences) >= 10 else np.mean(all_confidences)
            }
        }
    
    def _calculate_attention_consistency(self, attention_maps: List[np.ndarray]) -> float:
        """Calculate consistency of attention patterns"""
        if len(attention_maps) < 2:
            return 1.0
        
        correlations = []
        for i in range(len(attention_maps) - 1):
            corr = np.corrcoef(attention_maps[i].flatten(), 
                             attention_maps[i+1].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _get_layer_distribution(self) -> Dict[str, int]:
        """Get distribution of visualizations across layers"""
        layer_counts = {}
        for am in self.attention_history:
            layer_counts[am.model_layer] = layer_counts.get(am.model_layer, 0) + 1
        return layer_counts


# Example usage and testing
if __name__ == "__main__":
    # Create sample attention visualizer
    config = AttentionConfig(heatmap_alpha=0.7, colormap='viridis')
    visualizer = AttentionVisualizer(config)
    
    # Create dummy data for testing
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_attention = np.random.rand(224, 224)
    
    # Create attention map
    attention_map = AttentionMap(
        attention_weights=dummy_attention,
        source_image=dummy_image,
        model_layer="transformer.layer.11.attention",
        prediction_confidence=0.85
    )
    
    # Generate explanation
    explanation = visualizer.generate_attention_explanation(attention_map)
    print("Attention Explanation:", json.dumps(explanation, indent=2))
    
    # Create overlay
    overlay = visualizer.overlay_attention_on_image(dummy_image, dummy_attention)
    print(f"Created attention overlay with shape: {overlay.shape}")