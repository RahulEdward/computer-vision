#!/usr/bin/env python3
"""
Reinforcement Learning Module for Computer Genie
==============================================

यह module reinforcement learning agents provide करता है जो user corrections से सीखते हैं।

Features:
- User feedback से learning
- Action space optimization
- Reward modeling
- Policy gradient methods
- Q-learning for UI interactions
- Multi-agent coordination

Author: Computer Genie AI Team
"""

from .rl_agent import RLAgent, UserFeedbackAgent
from .reward_model import RewardModel, UserPreferenceModel
from .policy_network import PolicyNetwork, ActorCriticNetwork
from .environment import UIEnvironment, ActionSpace
from .trainer import RLTrainer, OnlineTrainer

__all__ = [
    'RLAgent',
    'UserFeedbackAgent', 
    'RewardModel',
    'UserPreferenceModel',
    'PolicyNetwork',
    'ActorCriticNetwork',
    'UIEnvironment',
    'ActionSpace',
    'RLTrainer',
    'OnlineTrainer'
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"