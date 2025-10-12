"""
Federated Learning Server for Computer Genie
फेडरेटेड लर्निंग सर्वर - Computer Genie के लिए

Central coordination server for federated learning that manages
client selection, model aggregation, and privacy-preserving updates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
import threading
from pathlib import Path
from collections import defaultdict, deque
import uuid
import asyncio
import websockets
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor

from .privacy_mechanisms import DifferentialPrivacy, SecureAggregation, PrivacyConfig
from .communication_manager import CommunicationManager, CommunicationConfig
from .aggregation_strategies import FedAvg, FedProx, AggregationResult
from .client_selection import ClientSelector, SelectionStrategy, ClientMetrics


class AggregationStrategy(Enum):
    """Aggregation strategies for federated learning"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"


class ClientSelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    RESOURCE_AWARE = "resource_aware"
    DATA_QUALITY = "data_quality"
    LOSS_BASED = "loss_based"
    DIVERSITY_BASED = "diversity_based"


@dataclass
class ServerConfig:
    """Configuration for federated server"""
    # Basic settings
    server_id: str = "federated_server"
    port: int = 8765
    max_clients: int = 100
    
    # Training settings
    num_rounds: int = 100
    clients_per_round: int = 10
    min_clients_per_round: int = 5
    max_wait_time: int = 300  # seconds
    
    # Aggregation settings
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    aggregation_weights: str = "uniform"  # "uniform", "data_size", "loss_based"
    
    # Client selection
    client_selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    client_selection_fraction: float = 0.1
    
    # Privacy settings
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    enable_secure_aggregation: bool = True
    
    # Communication settings
    communication_config: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # Model settings
    model_checkpoint_frequency: int = 10
    save_directory: str = "./federated_checkpoints"
    
    # Performance settings
    enable_async_aggregation: bool = True
    max_concurrent_clients: int = 50
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"


class ClientInfo:
    """Information about a connected client"""
    
    def __init__(self, client_id: str, websocket=None):
        self.client_id = client_id
        self.websocket = websocket
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.is_active = True
        
        # Client capabilities
        self.compute_capability = 1.0
        self.data_size = 0
        self.bandwidth = 1.0
        self.battery_level = 1.0
        
        # Training history
        self.training_history = []
        self.last_loss = float('inf')
        self.participation_count = 0
        
        # Performance metrics
        self.avg_training_time = 0.0
        self.avg_communication_time = 0.0
        self.success_rate = 1.0
    
    def update_metrics(self, training_time: float, communication_time: float, 
                      loss: float, success: bool) -> None:
        """Update client performance metrics"""
        self.last_seen = time.time()
        self.last_loss = loss
        self.participation_count += 1
        
        # Update averages
        alpha = 0.1  # Exponential moving average factor
        self.avg_training_time = alpha * training_time + (1 - alpha) * self.avg_training_time
        self.avg_communication_time = alpha * communication_time + (1 - alpha) * self.avg_communication_time
        
        # Update success rate
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        
        # Store in history
        self.training_history.append({
            'timestamp': time.time(),
            'loss': loss,
            'training_time': training_time,
            'communication_time': communication_time,
            'success': success
        })
        
        # Keep only recent history
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
    
    def get_client_metrics(self) -> ClientMetrics:
        """Get client metrics for selection"""
        return ClientMetrics(
            client_id=self.client_id,
            data_size=self.data_size,
            compute_capability=self.compute_capability,
            bandwidth=self.bandwidth,
            battery_level=self.battery_level,
            last_loss=self.last_loss,
            success_rate=self.success_rate,
            participation_count=self.participation_count
        )


class TrainingRound:
    """Represents a single training round"""
    
    def __init__(self, round_id: int, selected_clients: List[str]):
        self.round_id = round_id
        self.selected_clients = selected_clients
        self.start_time = time.time()
        self.end_time = None
        
        # Client responses
        self.client_updates = {}
        self.client_metrics = {}
        self.completed_clients = set()
        self.failed_clients = set()
        
        # Round results
        self.aggregated_model = None
        self.round_loss = None
        self.round_accuracy = None
        
    def add_client_update(self, client_id: str, model_update: Dict[str, torch.Tensor],
                         metrics: Dict[str, Any]) -> None:
        """Add client update to the round"""
        self.client_updates[client_id] = model_update
        self.client_metrics[client_id] = metrics
        self.completed_clients.add(client_id)
    
    def mark_client_failed(self, client_id: str) -> None:
        """Mark client as failed for this round"""
        self.failed_clients.add(client_id)
    
    def is_complete(self, min_clients: int) -> bool:
        """Check if round is complete"""
        return len(self.completed_clients) >= min_clients
    
    def get_completion_rate(self) -> float:
        """Get completion rate for this round"""
        total_clients = len(self.selected_clients)
        completed_clients = len(self.completed_clients)
        return completed_clients / total_clients if total_clients > 0 else 0.0
    
    def finalize(self) -> None:
        """Finalize the round"""
        self.end_time = time.time()


class FederatedServer:
    """Main federated learning server"""
    
    def __init__(self, model: nn.Module, config: ServerConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Server state
        self.server_id = config.server_id
        self.current_round = 0
        self.is_running = False
        self.start_time = None
        
        # Client management
        self.connected_clients = {}  # client_id -> ClientInfo
        self.client_lock = threading.Lock()
        
        # Training rounds
        self.training_rounds = {}  # round_id -> TrainingRound
        self.current_training_round = None
        
        # Components
        self.privacy_mechanism = DifferentialPrivacy(config.privacy_config)
        self.secure_aggregation = SecureAggregation() if config.enable_secure_aggregation else None
        self.communication_manager = CommunicationManager(config.communication_config)
        self.client_selector = ClientSelector(config.client_selection_strategy)
        
        # Aggregation strategy
        self.aggregator = self._create_aggregator()
        
        # Performance monitoring
        self.round_history = []
        self.performance_metrics = defaultdict(list)
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_clients)
        
        # Create save directory
        Path(config.save_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Federated server initialized: {self.server_id}")
    
    def _create_aggregator(self):
        """Create aggregation strategy"""
        if self.config.aggregation_strategy == AggregationStrategy.FEDAVG:
            return FedAvg()
        elif self.config.aggregation_strategy == AggregationStrategy.FEDPROX:
            return FedProx(mu=0.01)
        else:
            return FedAvg()  # Default fallback
    
    async def start_server(self) -> None:
        """Start the federated learning server"""
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info(f"Starting federated server on port {self.config.port}")
        
        # Start WebSocket server
        start_server = websockets.serve(
            self.handle_client_connection,
            "localhost",
            self.config.port
        )
        
        # Start training loop
        training_task = asyncio.create_task(self.training_loop())
        
        # Run both concurrently
        await asyncio.gather(start_server, training_task)
    
    async def handle_client_connection(self, websocket, path) -> None:
        """Handle new client connection"""
        client_id = None
        
        try:
            # Client registration
            registration_msg = await websocket.recv()
            registration_data = json.loads(registration_msg)
            
            client_id = registration_data.get("client_id")
            if not client_id:
                client_id = str(uuid.uuid4())
            
            # Create client info
            client_info = ClientInfo(client_id, websocket)
            client_info.compute_capability = registration_data.get("compute_capability", 1.0)
            client_info.data_size = registration_data.get("data_size", 0)
            client_info.bandwidth = registration_data.get("bandwidth", 1.0)
            client_info.battery_level = registration_data.get("battery_level", 1.0)
            
            # Register client
            with self.client_lock:
                self.connected_clients[client_id] = client_info
            
            self.logger.info(f"Client registered: {client_id}")
            
            # Send registration confirmation
            await websocket.send(json.dumps({
                "type": "registration_success",
                "client_id": client_id,
                "server_round": self.current_round
            }))
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(client_id, message)
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            if client_id:
                with self.client_lock:
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
    
    async def handle_client_message(self, client_id: str, message: str) -> None:
        """Handle message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "training_update":
                await self.handle_training_update(client_id, data)
            elif message_type == "heartbeat":
                await self.handle_heartbeat(client_id, data)
            elif message_type == "capability_update":
                await self.handle_capability_update(client_id, data)
            else:
                self.logger.warning(f"Unknown message type from {client_id}: {message_type}")
        
        except Exception as e:
            self.logger.error(f"Error processing message from {client_id}: {e}")
    
    async def handle_training_update(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle training update from client"""
        if not self.current_training_round:
            self.logger.warning(f"Received update from {client_id} but no active round")
            return
        
        if client_id not in self.current_training_round.selected_clients:
            self.logger.warning(f"Received update from non-selected client: {client_id}")
            return
        
        try:
            # Decode model update
            model_update_bytes = data.get("model_update")
            if model_update_bytes:
                model_update = pickle.loads(bytes.fromhex(model_update_bytes))
            else:
                self.logger.error(f"No model update received from {client_id}")
                return
            
            # Extract metrics
            metrics = data.get("metrics", {})
            training_time = metrics.get("training_time", 0.0)
            communication_time = metrics.get("communication_time", 0.0)
            loss = metrics.get("loss", float('inf'))
            
            # Add to current round
            self.current_training_round.add_client_update(client_id, model_update, metrics)
            
            # Update client info
            with self.client_lock:
                if client_id in self.connected_clients:
                    client_info = self.connected_clients[client_id]
                    client_info.update_metrics(training_time, communication_time, loss, True)
            
            self.logger.info(f"Received update from {client_id}, loss: {loss:.4f}")
            
            # Check if round is complete
            if self.current_training_round.is_complete(self.config.min_clients_per_round):
                await self.finalize_training_round()
        
        except Exception as e:
            self.logger.error(f"Error processing training update from {client_id}: {e}")
            self.current_training_round.mark_client_failed(client_id)
    
    async def handle_heartbeat(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle heartbeat from client"""
        with self.client_lock:
            if client_id in self.connected_clients:
                client_info = self.connected_clients[client_id]
                client_info.last_seen = time.time()
                
                # Update capabilities if provided
                if "battery_level" in data:
                    client_info.battery_level = data["battery_level"]
                if "compute_capability" in data:
                    client_info.compute_capability = data["compute_capability"]
    
    async def handle_capability_update(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle capability update from client"""
        with self.client_lock:
            if client_id in self.connected_clients:
                client_info = self.connected_clients[client_id]
                
                if "data_size" in data:
                    client_info.data_size = data["data_size"]
                if "compute_capability" in data:
                    client_info.compute_capability = data["compute_capability"]
                if "bandwidth" in data:
                    client_info.bandwidth = data["bandwidth"]
                if "battery_level" in data:
                    client_info.battery_level = data["battery_level"]
    
    async def training_loop(self) -> None:
        """Main training loop"""
        self.logger.info("Starting federated training loop")
        
        while self.is_running and self.current_round < self.config.num_rounds:
            try:
                # Start new training round
                await self.start_training_round()
                
                # Wait for round completion or timeout
                await self.wait_for_round_completion()
                
                # Move to next round
                self.current_round += 1
                
                # Save checkpoint periodically
                if self.current_round % self.config.model_checkpoint_frequency == 0:
                    self.save_checkpoint()
                
                # Brief pause between rounds
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        self.logger.info("Federated training completed")
    
    async def start_training_round(self) -> None:
        """Start a new training round"""
        self.logger.info(f"Starting training round {self.current_round}")
        
        # Select clients for this round
        selected_clients = self.select_clients()
        
        if len(selected_clients) < self.config.min_clients_per_round:
            self.logger.warning(f"Not enough clients available: {len(selected_clients)}")
            return
        
        # Create training round
        self.current_training_round = TrainingRound(self.current_round, selected_clients)
        self.training_rounds[self.current_round] = self.current_training_round
        
        # Send training requests to selected clients
        await self.send_training_requests(selected_clients)
    
    def select_clients(self) -> List[str]:
        """Select clients for training round"""
        with self.client_lock:
            available_clients = [
                client_id for client_id, client_info in self.connected_clients.items()
                if client_info.is_active and time.time() - client_info.last_seen < 60
            ]
        
        if not available_clients:
            return []
        
        # Get client metrics
        client_metrics = []
        with self.client_lock:
            for client_id in available_clients:
                client_info = self.connected_clients[client_id]
                client_metrics.append(client_info.get_client_metrics())
        
        # Select clients using strategy
        num_clients = min(
            self.config.clients_per_round,
            len(available_clients),
            int(len(available_clients) * self.config.client_selection_fraction)
        )
        
        selected_metrics = self.client_selector.select_clients(client_metrics, num_clients)
        selected_client_ids = [metrics.client_id for metrics in selected_metrics]
        
        self.logger.info(f"Selected {len(selected_client_ids)} clients for round {self.current_round}")
        return selected_client_ids
    
    async def send_training_requests(self, selected_clients: List[str]) -> None:
        """Send training requests to selected clients"""
        # Prepare global model
        global_model_state = self.model.state_dict()
        
        # Apply compression if enabled
        compressed_model = self.communication_manager.compress_model(global_model_state)
        model_bytes = pickle.dumps(compressed_model).hex()
        
        # Send to each selected client
        for client_id in selected_clients:
            try:
                with self.client_lock:
                    if client_id in self.connected_clients:
                        client_info = self.connected_clients[client_id]
                        websocket = client_info.websocket
                        
                        training_request = {
                            "type": "training_request",
                            "round_id": self.current_round,
                            "global_model": model_bytes,
                            "training_config": {
                                "epochs": 5,
                                "batch_size": 32,
                                "learning_rate": 0.01
                            }
                        }
                        
                        await websocket.send(json.dumps(training_request))
                        self.logger.debug(f"Sent training request to {client_id}")
            
            except Exception as e:
                self.logger.error(f"Failed to send training request to {client_id}: {e}")
                if self.current_training_round:
                    self.current_training_round.mark_client_failed(client_id)
    
    async def wait_for_round_completion(self) -> None:
        """Wait for training round to complete"""
        if not self.current_training_round:
            return
        
        start_time = time.time()
        
        while (time.time() - start_time < self.config.max_wait_time and
               not self.current_training_round.is_complete(self.config.min_clients_per_round)):
            await asyncio.sleep(1)
        
        # Finalize round if not already done
        if self.current_training_round and not self.current_training_round.end_time:
            await self.finalize_training_round()
    
    async def finalize_training_round(self) -> None:
        """Finalize the current training round"""
        if not self.current_training_round:
            return
        
        self.logger.info(f"Finalizing round {self.current_round}")
        
        # Aggregate model updates
        aggregation_result = await self.aggregate_model_updates()
        
        if aggregation_result and aggregation_result.aggregated_model:
            # Update global model
            self.model.load_state_dict(aggregation_result.aggregated_model)
            
            # Store round results
            self.current_training_round.aggregated_model = aggregation_result.aggregated_model
            self.current_training_round.round_loss = aggregation_result.avg_loss
            
            # Record performance metrics
            self.record_round_metrics(aggregation_result)
        
        # Finalize round
        self.current_training_round.finalize()
        
        completion_rate = self.current_training_round.get_completion_rate()
        self.logger.info(f"Round {self.current_round} completed with {completion_rate:.2%} client participation")
        
        self.current_training_round = None
    
    async def aggregate_model_updates(self) -> Optional[AggregationResult]:
        """Aggregate model updates from clients"""
        if not self.current_training_round or not self.current_training_round.client_updates:
            return None
        
        try:
            # Prepare client updates
            client_models = []
            client_weights = []
            client_losses = []
            
            for client_id, model_update in self.current_training_round.client_updates.items():
                client_models.append(model_update)
                
                # Get client weight based on strategy
                if self.config.aggregation_weights == "data_size":
                    with self.client_lock:
                        weight = self.connected_clients[client_id].data_size
                elif self.config.aggregation_weights == "loss_based":
                    metrics = self.current_training_round.client_metrics[client_id]
                    weight = 1.0 / (metrics.get("loss", 1.0) + 1e-8)
                else:
                    weight = 1.0  # Uniform weighting
                
                client_weights.append(weight)
                
                # Get client loss
                metrics = self.current_training_round.client_metrics[client_id]
                client_losses.append(metrics.get("loss", 0.0))
            
            # Apply differential privacy if enabled
            if self.config.privacy_config.enable_dp:
                client_models = [
                    self.privacy_mechanism.add_noise_to_model(model)
                    for model in client_models
                ]
            
            # Perform aggregation
            aggregation_result = self.aggregator.aggregate(
                client_models, client_weights, client_losses
            )
            
            return aggregation_result
        
        except Exception as e:
            self.logger.error(f"Error during model aggregation: {e}")
            return None
    
    def record_round_metrics(self, aggregation_result: AggregationResult) -> None:
        """Record metrics for the training round"""
        round_metrics = {
            "round": self.current_round,
            "timestamp": time.time(),
            "num_clients": len(self.current_training_round.completed_clients),
            "completion_rate": self.current_training_round.get_completion_rate(),
            "avg_loss": aggregation_result.avg_loss,
            "loss_std": aggregation_result.loss_std,
            "convergence_metric": aggregation_result.convergence_metric
        }
        
        self.round_history.append(round_metrics)
        
        # Update performance metrics
        self.performance_metrics["avg_loss"].append(aggregation_result.avg_loss)
        self.performance_metrics["completion_rate"].append(
            self.current_training_round.get_completion_rate()
        )
        self.performance_metrics["num_clients"].append(
            len(self.current_training_round.completed_clients)
        )
    
    def save_checkpoint(self) -> None:
        """Save server checkpoint"""
        checkpoint = {
            "round": self.current_round,
            "model_state_dict": self.model.state_dict(),
            "round_history": self.round_history,
            "performance_metrics": dict(self.performance_metrics),
            "config": self.config,
            "timestamp": time.time()
        }
        
        checkpoint_path = Path(self.config.save_directory) / f"server_checkpoint_round_{self.current_round}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load server checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.current_round = checkpoint["round"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.round_history = checkpoint["round_history"]
        self.performance_metrics = defaultdict(list, checkpoint["performance_metrics"])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        with self.client_lock:
            num_connected_clients = len(self.connected_clients)
            active_clients = sum(
                1 for client in self.connected_clients.values()
                if time.time() - client.last_seen < 60
            )
        
        stats = {
            "server_id": self.server_id,
            "current_round": self.current_round,
            "total_rounds": self.config.num_rounds,
            "connected_clients": num_connected_clients,
            "active_clients": active_clients,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "is_running": self.is_running
        }
        
        if self.round_history:
            recent_rounds = self.round_history[-10:]
            stats["recent_avg_loss"] = np.mean([r["avg_loss"] for r in recent_rounds])
            stats["recent_completion_rate"] = np.mean([r["completion_rate"] for r in recent_rounds])
        
        return stats
    
    def stop_server(self) -> None:
        """Stop the federated server"""
        self.is_running = False
        self.logger.info("Federated server stopped")


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
    
    # Create server configuration
    config = ServerConfig(
        num_rounds=50,
        clients_per_round=5,
        aggregation_strategy=AggregationStrategy.FEDAVG,
        client_selection_strategy=ClientSelectionStrategy.RANDOM
    )
    
    # Create model and server
    model = SimpleModel()
    server = FederatedServer(model, config)
    
    print("Federated server created successfully!")
    print(f"Server configuration: {config}")
    
    # Note: To actually run the server, you would use:
    # asyncio.run(server.start_server())