#!/usr/bin/env python3
"""
Reinforcement Learning Agent for User Correction Learning
========================================================

Advanced RL agent जो user corrections से सीखकर अपनी performance improve करता है।

Features:
- User feedback integration
- Online learning from corrections
- Action space exploration
- Policy optimization
- Experience replay
- Multi-task learning

Author: Computer Genie AI Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import random
from collections import deque, namedtuple
import json
import time


@dataclass
class RLConfig:
    """Reinforcement Learning configuration."""
    state_dim: int = 512
    action_dim: int = 100
    hidden_dim: int = 256
    num_layers: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    reward_scale: float = 1.0


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for action value estimation."""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        # Network layers
        layers = []
        input_dim = config.state_dim
        
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = config.hidden_dim
            
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DQN.
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            q_values: (batch_size, action_dim)
        """
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Policy network for action selection."""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            action_probs: (batch_size, action_dim)
            state_value: (batch_size, 1)
        """
        shared_features = self.shared_layers(state)
        action_probs = self.policy_head(shared_features)
        state_value = self.value_head(shared_features)
        
        return action_probs, state_value


class RLAgent:
    """
    Reinforcement Learning Agent for UI automation.
    
    यह agent user interactions से सीखकर बेहतर automation provide करता है।
    """
    
    def __init__(self, config: RLConfig, device: str = 'cpu'):
        """
        Initialize RL Agent.
        
        Args:
            config: RL configuration
            device: Device for computation
        """
        self.config = config
        self.device = device
        
        # Networks
        self.q_network = DQN(config).to(device)
        self.target_network = DQN(config).to(device)
        self.policy_network = PolicyNetwork(config).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.epsilon = config.epsilon
        self.step_count = 0
        self.episode_count = 0
        
        # Action space mapping
        self.action_space = self._create_action_space()
        
    def _create_action_space(self) -> Dict[int, str]:
        """Create mapping from action indices to action names."""
        actions = [
            'click', 'double_click', 'right_click', 'drag', 'drop',
            'type_text', 'press_key', 'scroll_up', 'scroll_down',
            'scroll_left', 'scroll_right', 'zoom_in', 'zoom_out',
            'copy', 'paste', 'cut', 'undo', 'redo', 'select_all',
            'find', 'replace', 'save', 'open', 'close', 'refresh',
            'back', 'forward', 'home', 'end', 'page_up', 'page_down',
            'tab', 'shift_tab', 'enter', 'escape', 'delete', 'backspace',
            'move_up', 'move_down', 'move_left', 'move_right',
            'resize_window', 'minimize', 'maximize', 'restore',
            'switch_tab', 'new_tab', 'close_tab', 'switch_window',
            'take_screenshot', 'wait', 'hover', 'focus', 'blur',
            'select_text', 'clear_text', 'append_text', 'prepend_text',
            'check_checkbox', 'uncheck_checkbox', 'select_dropdown',
            'upload_file', 'download_file', 'print_page', 'reload',
            'go_to_url', 'search', 'filter', 'sort', 'group',
            'expand', 'collapse', 'toggle', 'activate', 'deactivate',
            'play', 'pause', 'stop', 'record', 'mute', 'unmute',
            'volume_up', 'volume_down', 'fullscreen', 'exit_fullscreen',
            'rotate', 'flip', 'crop', 'resize', 'move', 'duplicate',
            'align_left', 'align_center', 'align_right', 'justify',
            'bold', 'italic', 'underline', 'strikethrough', 'highlight',
            'change_color', 'change_font', 'change_size', 'insert',
            'remove', 'update', 'validate', 'submit', 'cancel',
            'confirm', 'deny', 'skip', 'retry', 'ignore', 'help'
        ]
        
        return {i: action for i, action in enumerate(actions)}
    
    def get_state_representation(self, screen_features: np.ndarray, 
                               audio_features: np.ndarray,
                               context: Dict) -> np.ndarray:
        """
        Create state representation from multimodal inputs.
        
        Args:
            screen_features: Visual features from screen
            audio_features: Audio features from voice commands
            context: Additional context information
            
        Returns:
            state: Combined state representation
        """
        # Combine features
        state_components = []
        
        # Screen features
        if screen_features is not None:
            state_components.append(screen_features.flatten())
            
        # Audio features
        if audio_features is not None:
            state_components.append(audio_features.flatten())
            
        # Context features
        context_features = self._encode_context(context)
        state_components.append(context_features)
        
        # Concatenate and pad/truncate to fixed size
        state = np.concatenate(state_components)
        
        if len(state) > self.config.state_dim:
            state = state[:self.config.state_dim]
        elif len(state) < self.config.state_dim:
            padding = np.zeros(self.config.state_dim - len(state))
            state = np.concatenate([state, padding])
            
        return state.astype(np.float32)
    
    def _encode_context(self, context: Dict) -> np.ndarray:
        """Encode context information into numerical features."""
        features = []
        
        # Time features
        current_time = time.time()
        features.extend([
            np.sin(2 * np.pi * (current_time % 86400) / 86400),  # Time of day
            np.cos(2 * np.pi * (current_time % 86400) / 86400),
            np.sin(2 * np.pi * (current_time % 604800) / 604800),  # Day of week
            np.cos(2 * np.pi * (current_time % 604800) / 604800)
        ])
        
        # Application context
        app_name = context.get('application', 'unknown')
        app_hash = hash(app_name) % 1000 / 1000.0
        features.append(app_hash)
        
        # Task context
        task_type = context.get('task_type', 'unknown')
        task_hash = hash(task_type) % 1000 / 1000.0
        features.append(task_hash)
        
        # User preferences (simplified)
        user_id = context.get('user_id', 'default')
        user_hash = hash(user_id) % 1000 / 1000.0
        features.append(user_hash)
        
        # Pad to fixed size
        while len(features) < 32:
            features.append(0.0)
            
        return np.array(features[:32], dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.config.action_dim - 1)
        else:
            # Greedy action (exploitation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
            return action
    
    def select_action_policy(self, state: np.ndarray) -> int:
        """
        Select action using policy network.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy_network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dict containing training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
            
        # Sample batch
        experiences = self.replay_buffer.sample(self.config.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.config.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        q_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        return {
            'q_loss': q_loss.item(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
    
    def train_policy(self, states: List[np.ndarray], actions: List[int], 
                    rewards: List[float]) -> Dict[str, float]:
        """
        Train policy network using policy gradient.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            
        Returns:
            Dict containing training metrics
        """
        if len(states) == 0:
            return {}
            
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        
        # Compute discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards):
            running_reward = reward + self.config.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        reward_batch = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # Normalize rewards
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        
        # Forward pass
        action_probs, state_values = self.policy_network(state_batch)
        
        # Compute advantages
        advantages = reward_batch - state_values.squeeze()
        
        # Policy loss
        log_probs = torch.log(action_probs.gather(1, action_batch.unsqueeze(1)).squeeze())
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(state_values.squeeze(), reward_batch)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']


class UserFeedbackAgent(RLAgent):
    """
    RL Agent that learns from user feedback and corrections.
    
    यह agent user के feedback को समझकर अपनी actions improve करता है।
    """
    
    def __init__(self, config: RLConfig, device: str = 'cpu'):
        super().__init__(config, device)
        
        # User feedback history
        self.feedback_history = deque(maxlen=1000)
        self.correction_patterns = {}
        
    def process_user_feedback(self, state: np.ndarray, action: int, 
                            feedback: str, correction: Optional[int] = None) -> float:
        """
        Process user feedback and return reward.
        
        Args:
            state: State when action was taken
            action: Action that was taken
            feedback: User feedback ('good', 'bad', 'perfect', etc.)
            correction: Corrected action if provided
            
        Returns:
            reward: Computed reward based on feedback
        """
        # Map feedback to reward
        feedback_rewards = {
            'perfect': 1.0,
            'good': 0.7,
            'okay': 0.3,
            'bad': -0.5,
            'wrong': -1.0,
            'terrible': -1.5
        }
        
        base_reward = feedback_rewards.get(feedback.lower(), 0.0)
        
        # Store feedback
        feedback_entry = {
            'state': state,
            'action': action,
            'feedback': feedback,
            'correction': correction,
            'timestamp': time.time(),
            'reward': base_reward
        }
        self.feedback_history.append(feedback_entry)
        
        # Learn correction patterns
        if correction is not None:
            state_key = self._state_to_key(state)
            if state_key not in self.correction_patterns:
                self.correction_patterns[state_key] = {}
            
            if action not in self.correction_patterns[state_key]:
                self.correction_patterns[state_key][action] = []
            
            self.correction_patterns[state_key][action].append(correction)
        
        return base_reward * self.config.reward_scale
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key for pattern storage."""
        # Quantize state for pattern matching
        quantized = np.round(state * 10).astype(int)
        return str(quantized.tolist())
    
    def get_correction_suggestion(self, state: np.ndarray, action: int) -> Optional[int]:
        """
        Get correction suggestion based on past patterns.
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            suggested_action: Suggested correction or None
        """
        state_key = self._state_to_key(state)
        
        if state_key in self.correction_patterns:
            if action in self.correction_patterns[state_key]:
                corrections = self.correction_patterns[state_key][action]
                # Return most common correction
                return max(set(corrections), key=corrections.count)
        
        return None
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of user feedback."""
        if not self.feedback_history:
            return {}
        
        feedbacks = [entry['feedback'] for entry in self.feedback_history]
        rewards = [entry['reward'] for entry in self.feedback_history]
        
        return {
            'total_feedback': len(feedbacks),
            'average_reward': np.mean(rewards),
            'feedback_distribution': {
                feedback: feedbacks.count(feedback) 
                for feedback in set(feedbacks)
            },
            'recent_performance': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'improvement_trend': self._calculate_trend(rewards)
        }
    
    def _calculate_trend(self, rewards: List[float]) -> str:
        """Calculate performance trend."""
        if len(rewards) < 10:
            return 'insufficient_data'
        
        recent = np.mean(rewards[-10:])
        older = np.mean(rewards[-20:-10]) if len(rewards) >= 20 else np.mean(rewards[:-10])
        
        if recent > older + 0.1:
            return 'improving'
        elif recent < older - 0.1:
            return 'declining'
        else:
            return 'stable'


# Example usage
if __name__ == "__main__":
    # Create RL agent
    config = RLConfig()
    agent = UserFeedbackAgent(config)
    
    # Simulate training episode
    state = np.random.randn(config.state_dim)
    action = agent.select_action(state)
    
    print(f"Selected action: {action} ({agent.action_space[action]})")
    
    # Simulate user feedback
    reward = agent.process_user_feedback(state, action, 'good')
    print(f"User feedback reward: {reward}")
    
    # Store experience and train
    next_state = np.random.randn(config.state_dim)
    agent.store_experience(state, action, reward, next_state, False)
    
    # Train if enough experiences
    if len(agent.replay_buffer) >= config.batch_size:
        metrics = agent.train_step()
        print(f"Training metrics: {metrics}")
    
    # Get feedback summary
    summary = agent.get_feedback_summary()
    print(f"Feedback summary: {summary}")