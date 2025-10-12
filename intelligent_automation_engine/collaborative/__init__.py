"""Collaborative Automation Package

This package provides multi-agent collaboration capabilities for intelligent automation,
enabling distributed automation with agent coordination, task delegation, and 
collaborative decision-making.

Key Components:
- AgentCoordinator: Manages and coordinates multiple automation agents
- TaskDelegator: Handles intelligent task delegation among agents
- CommunicationHub: Provides centralized messaging and coordination
- CollaborationEngine: Orchestrates collaborative workflows
- AgentRegistry: Manages agent registration and discovery

Data Structures:
- Agent: Represents an automation agent with capabilities
- AgentCapability: Defines specific agent capabilities
- TaskAssignment: Represents task delegation to agents
- Message: Communication message between agents
- Channel: Communication channel for agent messaging
- CollaborativeWorkflow: Multi-agent workflow definition
"""

from .agent_coordinator import (
    AgentCoordinator,
    AgentType,
    AgentStatus,
    CoordinationStrategy,
    AgentCapability,
    Agent,
    CoordinationResult,
    CoordinationContext
)

from .task_delegator import (
    TaskDelegator,
    TaskPriority,
    AssignmentStatus,
    DelegationStrategy,
    CapabilityMatch,
    TaskRequirement,
    TaskAssignment,
    DelegationResult,
    DelegationContext
)

from .communication_hub import (
    CommunicationHub,
    MessageType,
    MessagePriority,
    DeliveryMode,
    MessageStatus,
    ChannelType,
    Message,
    Channel,
    Subscription,
    CommunicationStats,
    MessageHandler
)

from .collaboration_engine import (
    CollaborationEngine,
    CollaborationMode,
    WorkflowStatus,
    DecisionType,
    ConflictType,
    CollaborativeTask,
    CollaborativeWorkflow,
    CollaborativeDecision,
    Conflict,
    CollaborationMetrics
)

from .agent_registry import (
    AgentRegistry,
    AgentState,
    CapabilityLevel,
    RegistrationStatus,
    HealthStatus,
    RegisteredAgent,
    AgentGroup,
    DiscoveryQuery,
    DiscoveryResult,
    AgentMetrics
)

__all__ = [
    # Core classes
    'AgentCoordinator',
    'TaskDelegator', 
    'CommunicationHub',
    'CollaborationEngine',
    'AgentRegistry',
    
    # Enums
    'AgentType',
    'AgentStatus',
    'CoordinationStrategy',
    'TaskPriority',
    'AssignmentStatus',
    'DelegationStrategy',
    'CapabilityMatch',
    'MessageType',
    'MessagePriority',
    'DeliveryMode',
    'MessageStatus',
    'ChannelType',
    'CollaborationMode',
    'WorkflowStatus',
    'DecisionType',
    'ConflictType',
    'AgentState',
    'CapabilityLevel',
    'RegistrationStatus',
    'HealthStatus',
    
    # Data structures
    'AgentCapability',
    'Agent',
    'CoordinationResult',
    'CoordinationContext',
    'TaskRequirement',
    'TaskAssignment',
    'DelegationResult',
    'DelegationContext',
    'Message',
    'Channel',
    'Subscription',
    'CommunicationStats',
    'MessageHandler',
    'CollaborativeTask',
    'CollaborativeWorkflow',
    'CollaborativeDecision',
    'Conflict',
    'CollaborationMetrics',
    'RegisteredAgent',
    'AgentGroup',
    'DiscoveryQuery',
    'DiscoveryResult',
    'AgentMetrics'
]