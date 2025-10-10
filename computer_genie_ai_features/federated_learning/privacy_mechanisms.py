"""
Privacy Mechanisms for Federated Learning
फेडरेटेड लर्निंग के लिए प्राइवेसी मैकेनिज्म

Implementation of differential privacy, secure aggregation, and other
privacy-preserving techniques for federated learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import hashlib
import hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import logging
from collections import defaultdict


class PrivacyMechanism(Enum):
    """Privacy mechanisms"""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    LOCAL_PRIVACY = "local_privacy"
    FEDERATED_AVERAGING = "federated_averaging"


class NoiseType(Enum):
    """Types of noise for differential privacy"""
    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"
    EXPONENTIAL = "exponential"


@dataclass
class PrivacyConfig:
    """Configuration for privacy mechanisms"""
    # Differential Privacy
    enable_dp: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Failure probability
    noise_type: NoiseType = NoiseType.GAUSSIAN
    clipping_norm: float = 1.0  # Gradient clipping
    
    # Secure Aggregation
    enable_secure_aggregation: bool = True
    min_clients_for_aggregation: int = 3
    dropout_tolerance: float = 0.3  # 30% dropout tolerance
    
    # Local Privacy
    enable_local_privacy: bool = True
    local_epsilon: float = 2.0
    randomization_probability: float = 0.1
    
    # Encryption
    enable_encryption: bool = False
    key_size: int = 2048
    
    # Advanced settings
    adaptive_clipping: bool = True
    privacy_accounting: bool = True
    composition_method: str = "rdp"  # "basic", "advanced", "rdp"


class PrivacyAccountant:
    """Track privacy budget consumption"""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.privacy_history = []
    
    def consume_privacy_budget(self, epsilon: float, delta: float, 
                             mechanism: str, round_id: int) -> bool:
        """Consume privacy budget and check if within limits"""
        new_epsilon = self.consumed_epsilon + epsilon
        new_delta = self.consumed_delta + delta
        
        if new_epsilon <= self.total_epsilon and new_delta <= self.total_delta:
            self.consumed_epsilon = new_epsilon
            self.consumed_delta = new_delta
            
            self.privacy_history.append({
                'round': round_id,
                'mechanism': mechanism,
                'epsilon': epsilon,
                'delta': delta,
                'total_epsilon': self.consumed_epsilon,
                'total_delta': self.consumed_delta
            })
            
            return True
        
        return False
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (
            self.total_epsilon - self.consumed_epsilon,
            self.total_delta - self.consumed_delta
        )
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        return remaining_epsilon <= 0 or remaining_delta <= 0


class DifferentialPrivacy:
    """Differential Privacy implementation"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.accountant = PrivacyAccountant(config.epsilon, config.delta)
        
        # Adaptive clipping state
        self.clipping_history = []
        self.current_clipping_norm = config.clipping_norm
    
    def add_noise_to_gradients(self, model: nn.Module, 
                             sensitivity: float = 1.0) -> None:
        """Add differential privacy noise to gradients"""
        if not self.config.enable_dp:
            return
        
        # Clip gradients first
        self._clip_gradients(model)
        
        # Calculate noise scale
        noise_scale = self._calculate_noise_scale(sensitivity)
        
        # Add noise to gradients
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = self._generate_noise(param.grad.shape, noise_scale)
                    param.grad.add_(noise)
    
    def add_noise_to_model(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters"""
        if not self.config.enable_dp:
            return model_state
        
        noisy_model = {}
        noise_scale = self._calculate_noise_scale()
        
        for name, param in model_state.items():
            # Clip parameter values
            clipped_param = torch.clamp(param, -self.current_clipping_norm, self.current_clipping_norm)
            
            # Add noise
            noise = self._generate_noise(param.shape, noise_scale)
            noisy_param = clipped_param + noise
            
            noisy_model[name] = noisy_param
        
        return noisy_model
    
    def _clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to bound sensitivity"""
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        clip_coef = self.current_clipping_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Update adaptive clipping
        if self.config.adaptive_clipping:
            self._update_adaptive_clipping(total_norm)
        
        return min(total_norm, self.current_clipping_norm)
    
    def _update_adaptive_clipping(self, gradient_norm: float) -> None:
        """Update clipping norm adaptively"""
        self.clipping_history.append(gradient_norm)
        
        # Keep only recent history
        if len(self.clipping_history) > 100:
            self.clipping_history = self.clipping_history[-100:]
        
        # Update clipping norm based on percentile
        if len(self.clipping_history) >= 10:
            target_percentile = 50  # Median
            target_norm = np.percentile(self.clipping_history, target_percentile)
            
            # Smooth update
            alpha = 0.1
            self.current_clipping_norm = (
                alpha * target_norm + (1 - alpha) * self.current_clipping_norm
            )
    
    def _calculate_noise_scale(self, sensitivity: float = 1.0) -> float:
        """Calculate noise scale for differential privacy"""
        if self.config.noise_type == NoiseType.GAUSSIAN:
            # For Gaussian mechanism: σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
            sigma = math.sqrt(2 * math.log(1.25 / self.config.delta)) * sensitivity / self.config.epsilon
            return sigma
        elif self.config.noise_type == NoiseType.LAPLACIAN:
            # For Laplacian mechanism: b = sensitivity / ε
            return sensitivity / self.config.epsilon
        else:
            return sensitivity / self.config.epsilon
    
    def _generate_noise(self, shape: torch.Size, scale: float) -> torch.Tensor:
        """Generate noise tensor"""
        if self.config.noise_type == NoiseType.GAUSSIAN:
            return torch.normal(0, scale, size=shape)
        elif self.config.noise_type == NoiseType.LAPLACIAN:
            # Laplacian noise using exponential distribution
            uniform = torch.rand(shape)
            sign = torch.sign(uniform - 0.5)
            laplacian = -scale * sign * torch.log(1 - 2 * torch.abs(uniform - 0.5))
            return laplacian
        else:
            return torch.normal(0, scale, size=shape)
    
    def get_privacy_cost(self, num_rounds: int) -> Tuple[float, float]:
        """Calculate privacy cost for given number of rounds"""
        if self.config.composition_method == "basic":
            # Basic composition
            total_epsilon = num_rounds * self.config.epsilon
            total_delta = num_rounds * self.config.delta
        elif self.config.composition_method == "advanced":
            # Advanced composition
            total_epsilon = math.sqrt(2 * num_rounds * math.log(1/self.config.delta)) * self.config.epsilon
            total_delta = num_rounds * self.config.delta
        else:
            # RDP composition (simplified)
            total_epsilon = math.sqrt(num_rounds) * self.config.epsilon
            total_delta = self.config.delta
        
        return total_epsilon, total_delta


class SecureAggregation:
    """Secure Aggregation implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_keys = {}
        self.shared_secrets = {}
        self.masked_models = {}
    
    def generate_client_keypair(self, client_id: str) -> Tuple[bytes, bytes]:
        """Generate public-private key pair for client"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self.client_keys[client_id] = {
            'private': private_key,
            'public': public_key,
            'private_pem': private_pem,
            'public_pem': public_pem
        }
        
        return public_pem, private_pem
    
    def setup_shared_secrets(self, client_ids: List[str]) -> Dict[str, Dict[str, bytes]]:
        """Setup shared secrets between clients"""
        shared_secrets = {}
        
        for i, client_i in enumerate(client_ids):
            shared_secrets[client_i] = {}
            
            for j, client_j in enumerate(client_ids):
                if i != j:
                    # Generate shared secret
                    secret = secrets.token_bytes(32)
                    shared_secrets[client_i][client_j] = secret
        
        return shared_secrets
    
    def create_secret_shares(self, secret: bytes, num_shares: int, 
                           threshold: int) -> List[Tuple[int, bytes]]:
        """Create secret shares using Shamir's Secret Sharing"""
        # Simplified implementation - in practice, use a proper library
        shares = []
        
        # Convert secret to integer
        secret_int = int.from_bytes(secret, byteorder='big')
        
        # Generate random coefficients
        coefficients = [secret_int]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(2**256))
        
        # Generate shares
        for i in range(1, num_shares + 1):
            share_value = 0
            for j, coeff in enumerate(coefficients):
                share_value += coeff * (i ** j)
            
            share_bytes = (share_value % (2**256)).to_bytes(32, byteorder='big')
            shares.append((i, share_bytes))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, bytes]], 
                          threshold: int) -> bytes:
        """Reconstruct secret from shares"""
        if len(shares) < threshold:
            raise ValueError("Not enough shares to reconstruct secret")
        
        # Convert shares to integers
        points = []
        for x, share_bytes in shares[:threshold]:
            y = int.from_bytes(share_bytes, byteorder='big')
            points.append((x, y))
        
        # Lagrange interpolation at x=0
        secret = 0
        for i, (xi, yi) in enumerate(points):
            li = 1
            for j, (xj, _) in enumerate(points):
                if i != j:
                    li *= (-xj) / (xi - xj)
            secret += yi * li
        
        secret_int = int(secret) % (2**256)
        return secret_int.to_bytes(32, byteorder='big')
    
    def mask_model_update(self, model_update: Dict[str, torch.Tensor],
                         client_id: str, other_clients: List[str]) -> Dict[str, torch.Tensor]:
        """Mask model update for secure aggregation"""
        masked_update = {}
        
        for param_name, param_tensor in model_update.items():
            # Convert to numpy for easier manipulation
            param_array = param_tensor.detach().cpu().numpy()
            
            # Generate mask based on shared secrets
            mask = np.zeros_like(param_array)
            
            for other_client in other_clients:
                if other_client != client_id:
                    # Generate deterministic mask from shared secret
                    if client_id in self.shared_secrets and other_client in self.shared_secrets[client_id]:
                        secret = self.shared_secrets[client_id][other_client]
                        
                        # Use secret to seed random number generator
                        np.random.seed(int.from_bytes(secret[:4], byteorder='big'))
                        client_mask = np.random.normal(0, 1, param_array.shape)
                        
                        # Add or subtract based on client order
                        if client_id < other_client:
                            mask += client_mask
                        else:
                            mask -= client_mask
            
            # Apply mask
            masked_param = param_array + mask
            masked_update[param_name] = torch.from_numpy(masked_param).to(param_tensor.device)
        
        return masked_update
    
    def aggregate_masked_updates(self, masked_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate masked model updates"""
        if not masked_updates:
            return {}
        
        # Initialize aggregated model
        aggregated_model = {}
        
        for param_name in masked_updates[0].keys():
            # Sum all masked updates
            param_sum = torch.zeros_like(masked_updates[0][param_name])
            
            for masked_update in masked_updates:
                param_sum += masked_update[param_name]
            
            # Average (masks cancel out in summation)
            aggregated_model[param_name] = param_sum / len(masked_updates)
        
        return aggregated_model


class LocalPrivacy:
    """Local Differential Privacy implementation"""
    
    def __init__(self, epsilon: float = 2.0):
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)
    
    def add_noise_to_gradients(self, model: nn.Module) -> None:
        """Add local differential privacy noise to gradients"""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Apply randomized response mechanism
                    self._apply_randomized_response(param.grad)
    
    def _apply_randomized_response(self, gradient: torch.Tensor) -> None:
        """Apply randomized response to gradient"""
        # Probability of keeping true value
        p = math.exp(self.epsilon) / (math.exp(self.epsilon) + 1)
        
        # Generate random mask
        mask = torch.rand_like(gradient) < p
        
        # Apply randomized response
        noise = torch.randn_like(gradient) * 0.1
        gradient.data = torch.where(mask, gradient.data, noise)
    
    def privatize_data_sample(self, data: torch.Tensor) -> torch.Tensor:
        """Apply local privacy to data sample"""
        # Simple noise addition for demonstration
        noise_scale = 2.0 / self.epsilon
        noise = torch.normal(0, noise_scale, size=data.shape)
        return data + noise


class HomomorphicEncryption:
    """Simplified Homomorphic Encryption for demonstration"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self._generate_keys()
    
    def _generate_keys(self) -> None:
        """Generate encryption keys"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> bytes:
        """Encrypt tensor (simplified)"""
        # Convert tensor to bytes
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        
        # Encrypt using public key
        encrypted = self.public_key.encrypt(
            tensor_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def decrypt_tensor(self, encrypted_data: bytes, 
                      original_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Decrypt tensor"""
        # Decrypt using private key
        decrypted_bytes = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Convert back to tensor
        array = np.frombuffer(decrypted_bytes, dtype=np.float32)
        tensor = torch.from_numpy(array).reshape(original_shape)
        
        return tensor.to(dtype)
    
    def add_encrypted_tensors(self, encrypted1: bytes, encrypted2: bytes) -> bytes:
        """Add two encrypted tensors (simplified - not truly homomorphic)"""
        # This is a placeholder - true homomorphic encryption is much more complex
        # In practice, you would use libraries like SEAL, HElib, or Palisade
        return encrypted1  # Simplified


class PrivacyPreservingAggregator:
    """Main aggregator with privacy mechanisms"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize privacy mechanisms
        self.dp_mechanism = DifferentialPrivacy(config) if config.enable_dp else None
        self.secure_aggregation = SecureAggregation() if config.enable_secure_aggregation else None
        self.local_privacy = LocalPrivacy(config.local_epsilon) if config.enable_local_privacy else None
        self.homomorphic_encryption = HomomorphicEncryption() if config.enable_encryption else None
    
    def aggregate_with_privacy(self, client_updates: List[Dict[str, torch.Tensor]],
                             client_weights: List[float],
                             round_id: int) -> Dict[str, torch.Tensor]:
        """Aggregate client updates with privacy preservation"""
        if not client_updates:
            return {}
        
        # Apply secure aggregation if enabled
        if self.secure_aggregation and len(client_updates) >= self.config.min_clients_for_aggregation:
            # For demonstration, we'll use a simplified version
            aggregated_model = self._secure_aggregate(client_updates, client_weights)
        else:
            # Standard weighted averaging
            aggregated_model = self._weighted_average(client_updates, client_weights)
        
        # Apply differential privacy to aggregated model
        if self.dp_mechanism:
            aggregated_model = self.dp_mechanism.add_noise_to_model(aggregated_model)
        
        return aggregated_model
    
    def _secure_aggregate(self, client_updates: List[Dict[str, torch.Tensor]],
                         client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation"""
        # Simplified secure aggregation
        return self._weighted_average(client_updates, client_weights)
    
    def _weighted_average(self, client_updates: List[Dict[str, torch.Tensor]],
                         client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Perform weighted average of client updates"""
        if not client_updates:
            return {}
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated model
        aggregated_model = {}
        
        for param_name in client_updates[0].keys():
            # Weighted sum
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            
            for update, weight in zip(client_updates, normalized_weights):
                weighted_sum += weight * update[param_name]
            
            aggregated_model[param_name] = weighted_sum
        
        return aggregated_model
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy-related metrics"""
        metrics = {
            "dp_enabled": self.config.enable_dp,
            "secure_aggregation_enabled": self.config.enable_secure_aggregation,
            "local_privacy_enabled": self.config.enable_local_privacy,
            "encryption_enabled": self.config.enable_encryption
        }
        
        if self.dp_mechanism:
            remaining_epsilon, remaining_delta = self.dp_mechanism.accountant.get_remaining_budget()
            metrics.update({
                "remaining_epsilon": remaining_epsilon,
                "remaining_delta": remaining_delta,
                "budget_exhausted": self.dp_mechanism.accountant.is_budget_exhausted()
            })
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Create privacy configuration
    config = PrivacyConfig(
        enable_dp=True,
        epsilon=1.0,
        delta=1e-5,
        enable_secure_aggregation=True,
        enable_local_privacy=True
    )
    
    # Create privacy-preserving aggregator
    aggregator = PrivacyPreservingAggregator(config)
    
    print("Privacy mechanisms initialized successfully!")
    print(f"Configuration: {config}")
    
    # Test differential privacy
    dp = DifferentialPrivacy(config)
    
    # Create dummy model state
    dummy_model = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(1, 10),
        'layer2.bias': torch.randn(1)
    }
    
    # Add noise
    noisy_model = dp.add_noise_to_model(dummy_model)
    
    print("Differential privacy noise added successfully!")
    
    # Test secure aggregation
    sa = SecureAggregation()
    client_ids = ['client1', 'client2', 'client3']
    
    # Generate keys for clients
    for client_id in client_ids:
        public_key, private_key = sa.generate_client_keypair(client_id)
        print(f"Generated keys for {client_id}")
    
    print("Secure aggregation setup completed!")