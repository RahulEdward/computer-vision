"""
Federated Learning Client for Computer Genie
फेडरेटेड लर्निंग क्लाइंट - Computer Genie के लिए

Client-side implementation for federated learning that handles
local training, privacy-preserving updates, and server communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
import threading
import asyncio
import websockets
import pickle
import hashlib
import uuid
from pathlib import Path
from collections import defaultdict, deque
import psutil
import platform

from .privacy_mechanisms import DifferentialPrivacy, LocalPrivacy, PrivacyConfig
from .communication_manager import CommunicationManager, CommunicationConfig


class ClientState(Enum):
    """Client states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    TRAINING = "training"
    UPLOADING = "uploading"
    WAITING = "waiting"
    ERROR = "error"


class TrainingMode(Enum):
    """Training modes"""
    STANDARD = "standard"
    PERSONALIZED = "personalized"
    MULTI_TASK = "multi_task"
    CONTINUAL = "continual"


@dataclass
class ClientConfig:
    """Configuration for federated client"""
    # Basic settings
    client_id: Optional[str] = None
    server_host: str = "localhost"
    server_port: int = 8765
    
    # Training settings
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    training_mode: TrainingMode = TrainingMode.STANDARD
    
    # Privacy settings
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    enable_local_privacy: bool = True
    
    # Communication settings
    communication_config: CommunicationConfig = field(default_factory=CommunicationConfig)
    heartbeat_interval: int = 30  # seconds
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5  # seconds
    
    # Resource settings
    max_memory_usage: float = 0.8  # 80% of available memory
    min_battery_level: float = 0.2  # 20% battery minimum
    max_cpu_usage: float = 0.8  # 80% CPU usage
    
    # Data settings
    data_path: str = "./client_data"
    cache_size: int = 1000
    enable_data_augmentation: bool = True
    
    # Personalization settings
    personalization_layers: List[str] = field(default_factory=list)
    adaptation_rate: float = 0.1
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    save_directory: str = "./client_checkpoints"


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.cpu_usage_history = deque(maxlen=10)
        self.memory_usage_history = deque(maxlen=10)
        self.battery_level = 1.0
        
    def update_metrics(self) -> None:
        """Update resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage_history.append(cpu_percent / 100.0)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_usage_history.append(memory_percent / 100.0)
        
        # Battery level (if available)
        try:
            battery = psutil.sensors_battery()
            if battery:
                self.battery_level = battery.percent / 100.0
        except:
            self.battery_level = 1.0  # Assume full battery if not available
    
    def get_avg_cpu_usage(self) -> float:
        """Get average CPU usage"""
        return np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0.0
    
    def get_avg_memory_usage(self) -> float:
        """Get average memory usage"""
        return np.mean(self.memory_usage_history) if self.memory_usage_history else 0.0
    
    def get_compute_capability(self) -> float:
        """Estimate compute capability"""
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Simple heuristic for compute capability
        base_capability = cpu_count / 8.0  # Normalize by 8 cores
        if cpu_freq:
            freq_factor = cpu_freq.current / 2000.0  # Normalize by 2GHz
            base_capability *= freq_factor
        
        return min(base_capability, 2.0)  # Cap at 2.0
    
    def can_train(self, config: ClientConfig) -> bool:
        """Check if client can train based on resources"""
        return (
            self.get_avg_cpu_usage() < config.max_cpu_usage and
            self.get_avg_memory_usage() < config.max_memory_usage and
            self.battery_level > config.min_battery_level
        )


class LocalDataManager:
    """Manage local training data"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.data_path = Path(config.data_path)
        self.cache = {}
        self.cache_size = config.cache_size
        
        # Create data directory
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """Load dataset for training"""
        # Check cache first
        if dataset_name in self.cache:
            return self.cache[dataset_name]
        
        # Load from disk
        dataset_path = self.data_path / f"{dataset_name}.pt"
        if dataset_path.exists():
            try:
                dataset = torch.load(dataset_path)
                
                # Add to cache if space available
                if len(self.cache) < self.cache_size:
                    self.cache[dataset_name] = dataset
                
                return dataset
            except Exception as e:
                logging.error(f"Error loading dataset {dataset_name}: {e}")
        
        return None
    
    def save_dataset(self, dataset: Dataset, dataset_name: str) -> None:
        """Save dataset to disk"""
        dataset_path = self.data_path / f"{dataset_name}.pt"
        try:
            torch.save(dataset, dataset_path)
            
            # Add to cache
            if len(self.cache) < self.cache_size:
                self.cache[dataset_name] = dataset
        except Exception as e:
            logging.error(f"Error saving dataset {dataset_name}: {e}")
    
    def get_data_size(self) -> int:
        """Get total size of local data"""
        total_size = 0
        for dataset in self.cache.values():
            total_size += len(dataset)
        
        # Also check disk
        for dataset_file in self.data_path.glob("*.pt"):
            try:
                dataset = torch.load(dataset_file)
                if hasattr(dataset, '__len__'):
                    total_size += len(dataset)
            except:
                pass
        
        return total_size


class PersonalizationManager:
    """Manage model personalization"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.personalized_layers = {}
        self.adaptation_history = []
    
    def extract_personalized_layers(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract personalized layers from model"""
        personalized_params = {}
        
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in self.config.personalization_layers):
                personalized_params[name] = param.clone().detach()
        
        return personalized_params
    
    def apply_personalized_layers(self, model: nn.Module, 
                                personalized_params: Dict[str, torch.Tensor]) -> None:
        """Apply personalized layers to model"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in personalized_params:
                    param.copy_(personalized_params[name])
    
    def adapt_model(self, model: nn.Module, local_data: DataLoader, 
                   num_steps: int = 10) -> None:
        """Adapt model to local data"""
        if not self.config.personalization_layers:
            return
        
        # Create optimizer for personalized layers only
        personalized_params = []
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in self.config.personalization_layers):
                personalized_params.append(param)
        
        if not personalized_params:
            return
        
        optimizer = optim.SGD(personalized_params, lr=self.config.adaptation_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for step, (data, target) in enumerate(local_data):
            if step >= num_steps:
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': time.time(),
            'steps': step + 1,
            'final_loss': loss.item()
        })


class FederatedClient:
    """Main federated learning client"""
    
    def __init__(self, model: nn.Module, config: ClientConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Client state
        self.client_id = config.client_id or str(uuid.uuid4())
        self.state = ClientState.DISCONNECTED
        self.current_round = 0
        
        # Connection
        self.websocket = None
        self.connection_lock = threading.Lock()
        self.is_running = False
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.data_manager = LocalDataManager(config)
        self.personalization_manager = PersonalizationManager(config)
        
        # Privacy mechanisms
        self.privacy_mechanism = DifferentialPrivacy(config.privacy_config)
        self.local_privacy = LocalPrivacy() if config.enable_local_privacy else None
        
        # Communication
        self.communication_manager = CommunicationManager(config.communication_config)
        
        # Training state
        self.local_model = None
        self.training_dataset = None
        self.training_metrics = {}
        
        # Performance tracking
        self.training_history = []
        self.communication_history = []
        
        # Create save directory
        Path(config.save_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Federated client initialized: {self.client_id}")
    
    async def start_client(self) -> None:
        """Start the federated client"""
        self.is_running = True
        self.logger.info(f"Starting federated client: {self.client_id}")
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self.resource_monitoring_loop())
        
        # Start main client loop
        client_task = asyncio.create_task(self.client_loop())
        
        # Run both concurrently
        await asyncio.gather(monitor_task, client_task)
    
    async def client_loop(self) -> None:
        """Main client loop"""
        reconnect_attempts = 0
        
        while self.is_running:
            try:
                # Connect to server
                await self.connect_to_server()
                reconnect_attempts = 0
                
                # Handle server communication
                await self.handle_server_communication()
            
            except Exception as e:
                self.logger.error(f"Client loop error: {e}")
                self.state = ClientState.ERROR
                
                # Reconnection logic
                reconnect_attempts += 1
                if reconnect_attempts <= self.config.max_reconnect_attempts:
                    self.logger.info(f"Reconnecting in {self.config.reconnect_delay}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(self.config.reconnect_delay)
                else:
                    self.logger.error("Max reconnection attempts reached")
                    break
    
    async def connect_to_server(self) -> None:
        """Connect to federated server"""
        self.state = ClientState.CONNECTING
        
        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        self.logger.info(f"Connecting to server: {uri}")
        
        self.websocket = await websockets.connect(uri)
        
        # Send registration
        registration_data = {
            "client_id": self.client_id,
            "compute_capability": self.resource_monitor.get_compute_capability(),
            "data_size": self.data_manager.get_data_size(),
            "bandwidth": 1.0,  # Could be measured
            "battery_level": self.resource_monitor.battery_level,
            "platform": platform.system(),
            "capabilities": {
                "privacy": self.config.enable_local_privacy,
                "personalization": bool(self.config.personalization_layers),
                "multi_task": self.config.training_mode == TrainingMode.MULTI_TASK
            }
        }
        
        await self.websocket.send(json.dumps(registration_data))
        
        # Wait for registration confirmation
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get("type") == "registration_success":
            self.state = ClientState.CONNECTED
            self.current_round = response_data.get("server_round", 0)
            self.logger.info(f"Successfully connected to server, round: {self.current_round}")
        else:
            raise Exception("Registration failed")
    
    async def handle_server_communication(self) -> None:
        """Handle communication with server"""
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        
        try:
            # Listen for server messages
            async for message in self.websocket:
                await self.handle_server_message(message)
        finally:
            heartbeat_task.cancel()
    
    async def handle_server_message(self, message: str) -> None:
        """Handle message from server"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "training_request":
                await self.handle_training_request(data)
            elif message_type == "server_update":
                await self.handle_server_update(data)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling server message: {e}")
    
    async def handle_training_request(self, data: Dict[str, Any]) -> None:
        """Handle training request from server"""
        round_id = data.get("round_id")
        global_model_bytes = data.get("global_model")
        training_config = data.get("training_config", {})
        
        self.logger.info(f"Received training request for round {round_id}")
        
        # Check if we can train
        if not self.resource_monitor.can_train(self.config):
            self.logger.warning("Cannot train due to resource constraints")
            return
        
        self.state = ClientState.TRAINING
        
        try:
            # Load global model
            if global_model_bytes:
                compressed_model = pickle.loads(bytes.fromhex(global_model_bytes))
                global_model_state = self.communication_manager.decompress_model(compressed_model)
                self.model.load_state_dict(global_model_state)
            
            # Perform local training
            training_result = await self.perform_local_training(training_config)
            
            # Send update to server
            await self.send_training_update(round_id, training_result)
            
            self.state = ClientState.WAITING
        
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.state = ClientState.ERROR
    
    async def perform_local_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local training"""
        start_time = time.time()
        
        # Get training parameters
        epochs = training_config.get("epochs", self.config.local_epochs)
        batch_size = training_config.get("batch_size", self.config.batch_size)
        learning_rate = training_config.get("learning_rate", self.config.learning_rate)
        
        # Load training data
        if not self.training_dataset:
            self.training_dataset = self.data_manager.load_dataset("training_data")
        
        if not self.training_dataset:
            raise Exception("No training data available")
        
        # Create data loader
        train_loader = DataLoader(
            self.training_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Setup training
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Apply personalization if enabled
        if self.config.training_mode == TrainingMode.PERSONALIZED:
            self.personalization_manager.adapt_model(self.model, train_loader)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply local differential privacy if enabled
                if self.local_privacy:
                    self.local_privacy.add_noise_to_gradients(self.model)
                
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            num_samples += epoch_samples
            
            avg_epoch_loss = epoch_loss / epoch_samples
            self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_samples
        
        # Get model update
        model_update = self.model.state_dict()
        
        # Apply differential privacy to model update if enabled
        if self.config.privacy_config.enable_dp:
            model_update = self.privacy_mechanism.add_noise_to_model(model_update)
        
        # Prepare training result
        training_result = {
            "model_update": model_update,
            "metrics": {
                "loss": avg_loss,
                "training_time": training_time,
                "num_samples": num_samples,
                "epochs": epochs
            }
        }
        
        # Record training history
        self.training_history.append({
            "timestamp": time.time(),
            "round": self.current_round,
            "loss": avg_loss,
            "training_time": training_time,
            "num_samples": num_samples
        })
        
        return training_result
    
    async def send_training_update(self, round_id: int, training_result: Dict[str, Any]) -> None:
        """Send training update to server"""
        self.state = ClientState.UPLOADING
        start_time = time.time()
        
        try:
            # Compress model update
            model_update = training_result["model_update"]
            compressed_update = self.communication_manager.compress_model(model_update)
            
            # Serialize and encode
            update_bytes = pickle.dumps(compressed_update).hex()
            
            # Prepare message
            update_message = {
                "type": "training_update",
                "round_id": round_id,
                "client_id": self.client_id,
                "model_update": update_bytes,
                "metrics": training_result["metrics"]
            }
            
            # Send to server
            await self.websocket.send(json.dumps(update_message))
            
            communication_time = time.time() - start_time
            
            # Record communication history
            self.communication_history.append({
                "timestamp": time.time(),
                "round": round_id,
                "communication_time": communication_time,
                "update_size": len(update_bytes)
            })
            
            self.logger.info(f"Training update sent for round {round_id}")
        
        except Exception as e:
            self.logger.error(f"Error sending training update: {e}")
            raise
    
    async def handle_server_update(self, data: Dict[str, Any]) -> None:
        """Handle server update message"""
        update_type = data.get("update_type")
        
        if update_type == "round_complete":
            round_id = data.get("round_id")
            self.current_round = round_id + 1
            self.logger.info(f"Round {round_id} completed, moving to round {self.current_round}")
        
        elif update_type == "global_model":
            # Update global model if provided
            global_model_bytes = data.get("global_model")
            if global_model_bytes:
                compressed_model = pickle.loads(bytes.fromhex(global_model_bytes))
                global_model_state = self.communication_manager.decompress_model(compressed_model)
                self.model.load_state_dict(global_model_state)
                self.logger.info("Global model updated")
    
    async def heartbeat_loop(self) -> None:
        """Send periodic heartbeats to server"""
        while self.is_running and self.websocket:
            try:
                heartbeat_data = {
                    "type": "heartbeat",
                    "client_id": self.client_id,
                    "timestamp": time.time(),
                    "state": self.state.value,
                    "battery_level": self.resource_monitor.battery_level,
                    "compute_capability": self.resource_monitor.get_compute_capability()
                }
                
                await self.websocket.send(json.dumps(heartbeat_data))
                await asyncio.sleep(self.config.heartbeat_interval)
            
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break
    
    async def resource_monitoring_loop(self) -> None:
        """Monitor system resources"""
        while self.is_running:
            try:
                self.resource_monitor.update_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def save_checkpoint(self) -> None:
        """Save client checkpoint"""
        checkpoint = {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "model_state_dict": self.model.state_dict(),
            "training_history": self.training_history,
            "communication_history": self.communication_history,
            "config": self.config,
            "timestamp": time.time()
        }
        
        checkpoint_path = Path(self.config.save_directory) / f"client_checkpoint_{self.client_id}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load client checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.current_round = checkpoint["current_round"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.communication_history = checkpoint["communication_history"]
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            "client_id": self.client_id,
            "state": self.state.value,
            "current_round": self.current_round,
            "data_size": self.data_manager.get_data_size(),
            "compute_capability": self.resource_monitor.get_compute_capability(),
            "battery_level": self.resource_monitor.battery_level,
            "cpu_usage": self.resource_monitor.get_avg_cpu_usage(),
            "memory_usage": self.resource_monitor.get_avg_memory_usage(),
            "training_rounds": len(self.training_history),
            "is_running": self.is_running
        }
        
        if self.training_history:
            recent_training = self.training_history[-10:]
            stats["recent_avg_loss"] = np.mean([t["loss"] for t in recent_training])
            stats["recent_avg_training_time"] = np.mean([t["training_time"] for t in recent_training])
        
        return stats
    
    def stop_client(self) -> None:
        """Stop the federated client"""
        self.is_running = False
        self.state = ClientState.DISCONNECTED
        
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        
        self.logger.info("Federated client stopped")


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create client configuration
    config = ClientConfig(
        server_host="localhost",
        server_port=8765,
        local_epochs=5,
        batch_size=32,
        training_mode=TrainingMode.STANDARD
    )
    
    # Create model and client
    model = SimpleModel()
    client = FederatedClient(model, config)
    
    print("Federated client created successfully!")
    print(f"Client ID: {client.client_id}")
    print(f"Client configuration: {config}")
    
    # Note: To actually run the client, you would use:
    # asyncio.run(client.start_client())