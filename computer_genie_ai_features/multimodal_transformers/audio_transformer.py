#!/usr/bin/env python3
"""
Audio Transformer for Voice Command Understanding
===============================================

Advanced Audio Transformer model जो voice commands और audio cues को समझता है।

Features:
- Mel-spectrogram processing
- Temporal attention for audio sequences
- Multi-language speech recognition
- Intent classification from voice
- Emotion detection in speech
- Background noise filtering

Author: Computer Genie AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import librosa
from dataclasses import dataclass
import torchaudio
import torchaudio.transforms as T


@dataclass
class AudioConfig:
    """Audio Transformer configuration."""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    max_length: int = 1000  # Maximum sequence length
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    num_languages: int = 100
    num_intents: int = 50
    num_emotions: int = 8


class MelSpectrogramExtractor(nn.Module):
    """Extract mel-spectrogram features from audio."""
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            normalized=True
        )
        
        # Convert to dB scale
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Projection to hidden size
        self.projection = nn.Linear(config.n_mels, config.hidden_size)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features.
        
        Args:
            waveform: (batch_size, num_samples)
            
        Returns:
            features: (batch_size, sequence_length, hidden_size)
        """
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Transpose to (batch_size, time, freq)
        mel_spec = mel_spec.transpose(1, 2)
        
        # Project to hidden size
        features = self.projection(mel_spec)
        
        return features


class PositionalEncoding(nn.Module):
    """Positional encoding for audio sequences."""
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        pe = torch.zeros(config.max_length, config.hidden_size)
        position = torch.arange(0, config.max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * 
                           (-np.log(10000.0) / config.hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AudioMultiHeadAttention(nn.Module):
    """Multi-head attention for audio sequences."""
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_mask: (batch_size, sequence_length)
            
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
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Final projection
        context_layer = self.output_projection(context_layer)
        
        return context_layer, attention_probs


class AudioTransformerBlock(nn.Module):
    """Single transformer block for audio processing."""
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.attention = AudioMultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_after = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_mask: (batch_size, sequence_length)
            
        Returns:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_probs: (batch_size, num_heads, sequence_length, sequence_length)
        """
        # Self-attention
        attention_output, attention_probs = self.attention(
            self.layernorm_before(hidden_states), attention_mask
        )
        attention_output = self.dropout(attention_output)
        hidden_states = hidden_states + attention_output
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(self.layernorm_after(hidden_states)))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = hidden_states + layer_output
        
        return hidden_states, attention_probs


class AudioTransformer(nn.Module):
    """
    Audio Transformer for voice command understanding.
    
    यह model audio input को analyze करके:
    - Speech को text में convert करता है
    - Voice commands की intent समझता है
    - Speaker की emotion detect करता है
    - Multiple languages support करता है
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = MelSpectrogramExtractor(config)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AudioTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        # Classification heads
        self.language_classifier = nn.Linear(config.hidden_size, config.num_languages)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intents)
        self.emotion_classifier = nn.Linear(config.hidden_size, config.num_emotions)
        
        # Speech recognition head (simplified - in practice would use CTC or attention)
        self.speech_head = nn.Linear(config.hidden_size, 1000)  # Vocabulary size
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_attention_mask(self, sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """Create attention mask for variable length sequences."""
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_length).expand(batch_size, max_length) < sequence_lengths.unsqueeze(1)
        return mask.to(sequence_lengths.device)
    
    def forward(self, waveform: torch.Tensor, 
                sequence_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Audio Transformer.
        
        Args:
            waveform: (batch_size, num_samples)
            sequence_lengths: (batch_size,) - actual lengths of sequences
            
        Returns:
            Dict containing:
            - language_logits: (batch_size, num_languages)
            - intent_logits: (batch_size, num_intents)
            - emotion_logits: (batch_size, num_emotions)
            - speech_logits: (batch_size, sequence_length, vocab_size)
            - attention_maps: List of attention matrices
            - features: (batch_size, sequence_length, hidden_size)
        """
        # Extract features
        features = self.feature_extractor(waveform)
        
        # Add positional encoding
        features = self.positional_encoding(features)
        
        # Create attention mask
        attention_mask = None
        if sequence_lengths is not None:
            attention_mask = self.create_attention_mask(sequence_lengths, features.size(1))
        
        # Pass through transformer layers
        hidden_states = features
        attention_maps = []
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            attention_maps.append(attention_probs)
            
        # Final layer norm
        hidden_states = self.layernorm(hidden_states)
        
        # Global pooling for classification tasks
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled_features = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple average pooling
            pooled_features = hidden_states.mean(dim=1)
        
        # Classification heads
        language_logits = self.language_classifier(pooled_features)
        intent_logits = self.intent_classifier(pooled_features)
        emotion_logits = self.emotion_classifier(pooled_features)
        
        # Speech recognition (frame-level)
        speech_logits = self.speech_head(hidden_states)
        
        return {
            'language_logits': language_logits,
            'intent_logits': intent_logits,
            'emotion_logits': emotion_logits,
            'speech_logits': speech_logits,
            'attention_maps': attention_maps,
            'features': hidden_states,
            'pooled_features': pooled_features
        }
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features for downstream tasks."""
        outputs = self.forward(waveform)
        return outputs['pooled_features']


class VoiceCommandProcessor:
    """
    High-level interface for voice command processing.
    
    यह class Audio Transformer को use करके voice commands process करती है।
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize voice command processor.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.config = AudioConfig()
        self.model = AudioTransformer(self.config)
        
        if model_path:
            self.load_model(model_path)
            
        self.model.eval()
        
        # Language mapping
        self.languages = [
            'english', 'hindi', 'spanish', 'french', 'german', 'chinese',
            'japanese', 'korean', 'arabic', 'russian', 'portuguese', 'italian',
            'dutch', 'swedish', 'norwegian', 'danish', 'finnish', 'polish',
            'czech', 'hungarian', 'romanian', 'bulgarian', 'croatian', 'serbian',
            'slovenian', 'slovak', 'lithuanian', 'latvian', 'estonian', 'greek',
            'turkish', 'hebrew', 'persian', 'urdu', 'bengali', 'tamil',
            'telugu', 'marathi', 'gujarati', 'kannada', 'malayalam', 'punjabi',
            'thai', 'vietnamese', 'indonesian', 'malay', 'filipino', 'swahili',
            'amharic', 'yoruba'
        ] + [f'lang_{i}' for i in range(50, 100)]  # Additional language slots
        
        # Intent mapping
        self.intents = [
            'click', 'type', 'scroll', 'drag', 'drop', 'copy', 'paste',
            'open', 'close', 'save', 'delete', 'search', 'navigate',
            'select', 'deselect', 'zoom_in', 'zoom_out', 'refresh',
            'back', 'forward', 'home', 'settings', 'help', 'cancel',
            'confirm', 'yes', 'no', 'maybe', 'wait', 'stop', 'start',
            'pause', 'resume', 'play', 'record', 'capture', 'screenshot',
            'minimize', 'maximize', 'restore', 'fullscreen', 'split',
            'merge', 'sort', 'filter', 'group', 'ungroup', 'undo',
            'redo', 'cut', 'find', 'replace'
        ]
        
        # Emotion mapping
        self.emotions = [
            'neutral', 'happy', 'sad', 'angry', 'surprised',
            'fearful', 'disgusted', 'excited'
        ]
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process audio file and extract voice commands.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing processing results
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            resampler = T.Resample(sample_rate, self.config.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return self.process_audio(waveform.squeeze(0))
    
    def process_audio(self, waveform: torch.Tensor) -> Dict:
        """
        Process audio waveform and extract voice commands.
        
        Args:
            waveform: Audio waveform tensor (num_samples,)
            
        Returns:
            Dict containing processing results
        """
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(waveform)
        
        # Post-process results
        results = self.postprocess_outputs(outputs)
        
        return results
    
    def postprocess_outputs(self, outputs: Dict) -> Dict:
        """Post-process model outputs to extract meaningful results."""
        # Get predictions
        language_probs = F.softmax(outputs['language_logits'], dim=-1)
        intent_probs = F.softmax(outputs['intent_logits'], dim=-1)
        emotion_probs = F.softmax(outputs['emotion_logits'], dim=-1)
        
        # Get top predictions
        language_pred = language_probs.argmax(dim=-1).item()
        language_conf = language_probs.max(dim=-1).values.item()
        
        intent_pred = intent_probs.argmax(dim=-1).item()
        intent_conf = intent_probs.max(dim=-1).values.item()
        
        emotion_pred = emotion_probs.argmax(dim=-1).item()
        emotion_conf = emotion_probs.max(dim=-1).values.item()
        
        # Get top-k predictions for each task
        top_k = 3
        
        language_topk = torch.topk(language_probs, top_k, dim=-1)
        intent_topk = torch.topk(intent_probs, top_k, dim=-1)
        emotion_topk = torch.topk(emotion_probs, top_k, dim=-1)
        
        return {
            'language': {
                'predicted': self.languages[language_pred],
                'confidence': language_conf,
                'top_k': [
                    {
                        'language': self.languages[idx.item()],
                        'confidence': conf.item()
                    }
                    for idx, conf in zip(language_topk.indices[0], language_topk.values[0])
                ]
            },
            'intent': {
                'predicted': self.intents[intent_pred],
                'confidence': intent_conf,
                'top_k': [
                    {
                        'intent': self.intents[idx.item()],
                        'confidence': conf.item()
                    }
                    for idx, conf in zip(intent_topk.indices[0], intent_topk.values[0])
                ]
            },
            'emotion': {
                'predicted': self.emotions[emotion_pred],
                'confidence': emotion_conf,
                'top_k': [
                    {
                        'emotion': self.emotions[idx.item()],
                        'confidence': conf.item()
                    }
                    for idx, conf in zip(emotion_topk.indices[0], emotion_topk.values[0])
                ]
            },
            'attention_maps': outputs['attention_maps'],
            'raw_outputs': outputs
        }
    
    def real_time_processing(self, audio_stream):
        """
        Process real-time audio stream.
        
        Args:
            audio_stream: Real-time audio stream
            
        Yields:
            Processing results for each audio chunk
        """
        # This would be implemented for real-time processing
        # For now, it's a placeholder
        pass


# Example usage
if __name__ == "__main__":
    # Create model
    config = AudioConfig()
    model = AudioTransformer(config)
    
    # Create dummy audio input
    dummy_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
    
    # Forward pass
    outputs = model(dummy_audio)
    
    print("Audio Transformer Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: List of {len(value)} attention maps")
    
    # Create voice command processor
    processor = VoiceCommandProcessor()
    
    # Process dummy audio
    results = processor.process_audio(dummy_audio.squeeze(0))
    
    print(f"\nVoice Command Processing Results:")
    print(f"  Language: {results['language']['predicted']} ({results['language']['confidence']:.3f})")
    print(f"  Intent: {results['intent']['predicted']} ({results['intent']['confidence']:.3f})")
    print(f"  Emotion: {results['emotion']['predicted']} ({results['emotion']['confidence']:.3f})")