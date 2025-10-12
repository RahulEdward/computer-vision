"""
Agent Registry for Multi-Agent Collaboration

This module manages agent registration, discovery, capability tracking,
and lifecycle management for collaborative automation systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable
from datetime import datetime, timedelta
import uuid
import logging
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States of an agent in the system"""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class CapabilityLevel(Enum):
    """Levels of capability proficiency"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class RegistrationStatus(Enum):
    """Status of agent registration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class HealthStatus(Enum):
    """Health status of an agent"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class AgentCapability:
    """Represents a specific capability of an agent"""
    name: str = ""
    category: str = ""
    description: str = ""
    level: CapabilityLevel = CapabilityLevel.BASIC
    
    # Performance metrics
    success_rate: float = 0.0
    average_execution_time: timedelta = timedelta(0)
    reliability_score: float = 0.0
    quality_score: float = 0.0
    
    # Resource requirements
    cpu_requirement: float = 0.0
    memory_requirement: float = 0.0
    storage_requirement: float = 0.0
    network_requirement: float = 0.0
    
    # Constraints and limitations
    max_concurrent_tasks: int = 1
    supported_input_types: List[str] = field(default_factory=list)
    supported_output_types: List[str] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    certification_level: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Performance and health metrics for an agent"""
    agent_id: str = ""
    
    # Performance metrics
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_response_time: timedelta = timedelta(0)
    uptime_percentage: float = 0.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0
    network_utilization: float = 0.0
    
    # Quality metrics
    overall_quality_score: float = 0.0
    user_satisfaction_score: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Collaboration metrics
    collaboration_score: float = 0.0
    communication_efficiency: float = 0.0
    team_contribution: float = 0.0
    
    # Health indicators
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: datetime = field(default_factory=datetime.now)
    health_issues: List[str] = field(default_factory=list)
    
    # Timestamps
    metrics_updated_at: datetime = field(default_factory=datetime.now)
    collection_period: timedelta = timedelta(hours=1)


@dataclass
class RegisteredAgent:
    """Represents a registered agent in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Agent properties
    type: str = ""
    version: str = "1.0"
    vendor: str = ""
    contact_info: Dict[str, str] = field(default_factory=dict)
    
    # Capabilities
    capabilities: Dict[str, AgentCapability] = field(default_factory=dict)
    supported_protocols: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    
    # Network information
    endpoint: str = ""
    port: int = 0
    api_version: str = "v1"
    authentication_method: str = "token"
    
    # Status and state
    state: AgentState = AgentState.OFFLINE
    registration_status: RegistrationStatus = RegistrationStatus.PENDING
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Performance and metrics
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Configuration
    max_concurrent_tasks: int = 5
    priority_level: int = 5
    trust_level: float = 0.5
    
    # Metadata
    registered_at: datetime = field(default_factory=datetime.now)
    registered_by: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentGroup:
    """Represents a group of related agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Group members
    agent_ids: Set[str] = field(default_factory=set)
    group_type: str = "functional"  # functional, organizational, temporary
    
    # Group properties
    shared_capabilities: List[str] = field(default_factory=list)
    group_policies: Dict[str, Any] = field(default_factory=dict)
    coordination_strategy: str = "centralized"
    
    # Access control
    access_permissions: Dict[str, List[str]] = field(default_factory=dict)
    admin_agents: Set[str] = field(default_factory=set)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class DiscoveryQuery:
    """Query for discovering agents with specific capabilities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Query criteria
    required_capabilities: List[str] = field(default_factory=list)
    preferred_capabilities: List[str] = field(default_factory=list)
    capability_levels: Dict[str, CapabilityLevel] = field(default_factory=dict)
    
    # Constraints
    max_agents: int = 10
    min_trust_level: float = 0.5
    min_quality_score: float = 0.7
    max_response_time: timedelta = timedelta(seconds=30)
    
    # Filters
    agent_types: List[str] = field(default_factory=list)
    exclude_agents: Set[str] = field(default_factory=set)
    location_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Preferences
    sort_by: str = "quality_score"  # quality_score, response_time, trust_level
    sort_order: str = "desc"  # asc, desc
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


@dataclass
class DiscoveryResult:
    """Result of agent discovery query"""
    query_id: str = ""
    
    # Results
    matching_agents: List[RegisteredAgent] = field(default_factory=list)
    partial_matches: List[RegisteredAgent] = field(default_factory=list)
    
    # Match scores
    match_scores: Dict[str, float] = field(default_factory=dict)  # agent_id -> score
    capability_matches: Dict[str, List[str]] = field(default_factory=dict)
    
    # Query statistics
    total_agents_searched: int = 0
    search_duration: timedelta = timedelta(0)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=30))


class AgentRegistry:
    """Registry for managing agent registration, discovery, and lifecycle"""
    
    def __init__(self):
        # Agent storage
        self.agents: Dict[str, RegisteredAgent] = {}
        self.agent_groups: Dict[str, AgentGroup] = {}
        
        # Capability indexing
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self.type_index: Dict[str, Set[str]] = {}        # type -> agent_ids
        
        # Discovery and search
        self.discovery_cache: Dict[str, DiscoveryResult] = {}
        self.active_queries: Dict[str, DiscoveryQuery] = {}
        
        # Health monitoring
        self.health_monitors: Dict[str, Callable] = {}
        self.health_check_interval = timedelta(minutes=5)
        self.last_health_check = datetime.now()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'agent_registered': [],
            'agent_unregistered': [],
            'agent_state_changed': [],
            'capability_updated': [],
            'health_status_changed': []
        }
        
        # Configuration
        self.max_agents = 1000
        self.default_trust_level = 0.5
        self.registration_approval_required = False
        self.auto_cleanup_interval = timedelta(hours=24)
        
        # Statistics
        self.registration_stats = {
            'total_registrations': 0,
            'active_agents': 0,
            'failed_registrations': 0,
            'discovery_queries': 0
        }
        
        logger.info("Agent registry initialized")
    
    def register_agent(self, agent: RegisteredAgent, auto_approve: bool = True) -> bool:
        """Register a new agent in the system"""
        try:
            # Validate agent data
            if not self._validate_agent_data(agent):
                logger.error(f"Invalid agent data: {agent.name}")
                return False
            
            # Check for duplicate registration
            if agent.id in self.agents:
                logger.warning(f"Agent already registered: {agent.id}")
                return False
            
            # Check capacity limits
            if len(self.agents) >= self.max_agents:
                logger.error("Maximum agent capacity reached")
                return False
            
            # Set registration status
            if auto_approve or not self.registration_approval_required:
                agent.registration_status = RegistrationStatus.APPROVED
            else:
                agent.registration_status = RegistrationStatus.PENDING
            
            # Initialize metrics
            agent.metrics.agent_id = agent.id
            
            # Store agent
            self.agents[agent.id] = agent
            
            # Update indexes
            self._update_capability_index(agent)
            self._update_type_index(agent)
            
            # Update statistics
            self.registration_stats['total_registrations'] += 1
            if agent.registration_status == RegistrationStatus.APPROVED:
                self.registration_stats['active_agents'] += 1
            
            # Trigger event
            self._trigger_event('agent_registered', agent)
            
            logger.info(f"Agent registered: {agent.name} ({agent.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            self.registration_stats['failed_registrations'] += 1
            return False
    
    def unregister_agent(self, agent_id: str, reason: str = "") -> bool:
        """Unregister an agent from the system"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Update state
            agent.state = AgentState.TERMINATED
            agent.registration_status = RegistrationStatus.REVOKED
            
            # Remove from indexes
            self._remove_from_capability_index(agent)
            self._remove_from_type_index(agent)
            
            # Remove from groups
            self._remove_agent_from_groups(agent_id)
            
            # Remove from registry
            del self.agents[agent_id]
            
            # Update statistics
            self.registration_stats['active_agents'] -= 1
            
            # Trigger event
            self._trigger_event('agent_unregistered', agent)
            
            logger.info(f"Agent unregistered: {agent.name} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent: {e}")
            return False
    
    def update_agent_state(self, agent_id: str, new_state: AgentState) -> bool:
        """Update the state of an agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            old_state = agent.state
            agent.state = new_state
            agent.last_seen = datetime.now()
            
            # Trigger event
            self._trigger_event('agent_state_changed', {
                'agent': agent,
                'old_state': old_state,
                'new_state': new_state
            })
            
            logger.info(f"Agent state updated: {agent.name} {old_state.value} -> {new_state.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent state: {e}")
            return False
    
    def update_agent_capabilities(self, agent_id: str, 
                                 capabilities: Dict[str, AgentCapability]) -> bool:
        """Update the capabilities of an agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Remove old capabilities from index
            self._remove_from_capability_index(agent)
            
            # Update capabilities
            agent.capabilities = capabilities
            
            # Update capability index
            self._update_capability_index(agent)
            
            # Trigger event
            self._trigger_event('capability_updated', agent)
            
            logger.info(f"Agent capabilities updated: {agent.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent capabilities: {e}")
            return False
    
    def discover_agents(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Discover agents based on query criteria"""
        try:
            start_time = datetime.now()
            
            # Check cache first
            cache_key = self._generate_query_cache_key(query)
            cached_result = self.discovery_cache.get(cache_key)
            if cached_result and cached_result.expires_at > datetime.now():
                return cached_result
            
            # Store active query
            self.active_queries[query.id] = query
            
            # Find matching agents
            matching_agents = []
            partial_matches = []
            match_scores = {}
            capability_matches = {}
            
            for agent in self.agents.values():
                if not self._is_agent_available(agent):
                    continue
                
                # Calculate match score
                score, matches = self._calculate_match_score(agent, query)
                
                if score >= 0.8:  # High match threshold
                    matching_agents.append(agent)
                    match_scores[agent.id] = score
                    capability_matches[agent.id] = matches
                elif score >= 0.5:  # Partial match threshold
                    partial_matches.append(agent)
                    match_scores[agent.id] = score
                    capability_matches[agent.id] = matches
            
            # Sort results
            matching_agents.sort(key=lambda a: match_scores[a.id], reverse=True)
            partial_matches.sort(key=lambda a: match_scores[a.id], reverse=True)
            
            # Limit results
            if query.max_agents > 0:
                matching_agents = matching_agents[:query.max_agents]
            
            # Create result
            result = DiscoveryResult(
                query_id=query.id,
                matching_agents=matching_agents,
                partial_matches=partial_matches,
                match_scores=match_scores,
                capability_matches=capability_matches,
                total_agents_searched=len(self.agents),
                search_duration=datetime.now() - start_time
            )
            
            # Cache result
            self.discovery_cache[cache_key] = result
            
            # Clean up active query
            self.active_queries.pop(query.id, None)
            
            # Update statistics
            self.registration_stats['discovery_queries'] += 1
            
            logger.info(f"Agent discovery completed: {len(matching_agents)} matches found")
            return result
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return DiscoveryResult(query_id=query.id)
    
    def get_agent(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[RegisteredAgent]:
        """Get all agents with a specific capability"""
        agent_ids = self.capability_index.get(capability, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_agents_by_type(self, agent_type: str) -> List[RegisteredAgent]:
        """Get all agents of a specific type"""
        agent_ids = self.type_index.get(agent_type, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_available_agents(self) -> List[RegisteredAgent]:
        """Get all available agents"""
        return [agent for agent in self.agents.values() if self._is_agent_available(agent)]
    
    def create_agent_group(self, group: AgentGroup) -> bool:
        """Create a new agent group"""
        try:
            if group.id in self.agent_groups:
                logger.warning(f"Agent group already exists: {group.id}")
                return False
            
            # Validate agent IDs
            for agent_id in group.agent_ids:
                if agent_id not in self.agents:
                    logger.error(f"Invalid agent ID in group: {agent_id}")
                    return False
            
            self.agent_groups[group.id] = group
            
            logger.info(f"Agent group created: {group.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create agent group: {e}")
            return False
    
    def add_agent_to_group(self, group_id: str, agent_id: str) -> bool:
        """Add an agent to a group"""
        try:
            group = self.agent_groups.get(group_id)
            if not group:
                logger.error(f"Agent group not found: {group_id}")
                return False
            
            if agent_id not in self.agents:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            group.agent_ids.add(agent_id)
            group.last_modified = datetime.now()
            
            logger.info(f"Agent added to group: {agent_id} -> {group.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add agent to group: {e}")
            return False
    
    def remove_agent_from_group(self, group_id: str, agent_id: str) -> bool:
        """Remove an agent from a group"""
        try:
            group = self.agent_groups.get(group_id)
            if not group:
                logger.error(f"Agent group not found: {group_id}")
                return False
            
            group.agent_ids.discard(agent_id)
            group.last_modified = datetime.now()
            
            logger.info(f"Agent removed from group: {agent_id} <- {group.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove agent from group: {e}")
            return False
    
    def update_agent_metrics(self, agent_id: str, metrics: AgentMetrics) -> bool:
        """Update metrics for an agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            old_health = agent.metrics.health_status
            agent.metrics = metrics
            agent.metrics.metrics_updated_at = datetime.now()
            
            # Check for health status changes
            if old_health != metrics.health_status:
                self._trigger_event('health_status_changed', {
                    'agent': agent,
                    'old_health': old_health,
                    'new_health': metrics.health_status
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent metrics: {e}")
            return False
    
    def perform_health_check(self, agent_id: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Perform health check on agents"""
        health_results = {}
        
        try:
            agents_to_check = [self.agents[agent_id]] if agent_id else self.agents.values()
            
            for agent in agents_to_check:
                # Check if agent is responsive
                if datetime.now() - agent.last_seen > timedelta(minutes=10):
                    health_status = HealthStatus.CRITICAL
                elif agent.state == AgentState.ERROR:
                    health_status = HealthStatus.CRITICAL
                elif agent.metrics.error_rate > 0.1:
                    health_status = HealthStatus.WARNING
                else:
                    health_status = HealthStatus.HEALTHY
                
                # Update agent health
                old_health = agent.metrics.health_status
                agent.metrics.health_status = health_status
                agent.metrics.last_health_check = datetime.now()
                
                health_results[agent.id] = health_status
                
                # Trigger event if health changed
                if old_health != health_status:
                    self._trigger_event('health_status_changed', {
                        'agent': agent,
                        'old_health': old_health,
                        'new_health': health_status
                    })
            
            self.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_results
    
    def add_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Add an event handler"""
        try:
            if event_type not in self.event_handlers:
                logger.error(f"Unknown event type: {event_type}")
                return False
            
            self.event_handlers[event_type].append(handler)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add event handler: {e}")
            return False
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_agents = sum(1 for agent in self.agents.values() 
                           if agent.registration_status == RegistrationStatus.APPROVED)
        
        online_agents = sum(1 for agent in self.agents.values() 
                           if agent.state == AgentState.ONLINE)
        
        capabilities_count = len(self.capability_index)
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_agents,
            'online_agents': online_agents,
            'agent_groups': len(self.agent_groups),
            'capabilities_tracked': capabilities_count,
            'discovery_cache_size': len(self.discovery_cache),
            'last_health_check': self.last_health_check,
            **self.registration_stats
        }
    
    # Helper methods
    def _validate_agent_data(self, agent: RegisteredAgent) -> bool:
        """Validate agent registration data"""
        if not agent.name or not agent.type:
            return False
        
        if not agent.endpoint and agent.type != "local":
            return False
        
        return True
    
    def _update_capability_index(self, agent: RegisteredAgent):
        """Update capability index for an agent"""
        for capability_name in agent.capabilities.keys():
            if capability_name not in self.capability_index:
                self.capability_index[capability_name] = set()
            self.capability_index[capability_name].add(agent.id)
    
    def _remove_from_capability_index(self, agent: RegisteredAgent):
        """Remove agent from capability index"""
        for capability_name in agent.capabilities.keys():
            if capability_name in self.capability_index:
                self.capability_index[capability_name].discard(agent.id)
                if not self.capability_index[capability_name]:
                    del self.capability_index[capability_name]
    
    def _update_type_index(self, agent: RegisteredAgent):
        """Update type index for an agent"""
        if agent.type not in self.type_index:
            self.type_index[agent.type] = set()
        self.type_index[agent.type].add(agent.id)
    
    def _remove_from_type_index(self, agent: RegisteredAgent):
        """Remove agent from type index"""
        if agent.type in self.type_index:
            self.type_index[agent.type].discard(agent.id)
            if not self.type_index[agent.type]:
                del self.type_index[agent.type]
    
    def _remove_agent_from_groups(self, agent_id: str):
        """Remove agent from all groups"""
        for group in self.agent_groups.values():
            group.agent_ids.discard(agent_id)
    
    def _is_agent_available(self, agent: RegisteredAgent) -> bool:
        """Check if an agent is available for tasks"""
        return (agent.registration_status == RegistrationStatus.APPROVED and
                agent.state in [AgentState.ONLINE, AgentState.IDLE] and
                agent.metrics.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING])
    
    def _calculate_match_score(self, agent: RegisteredAgent, 
                              query: DiscoveryQuery) -> Tuple[float, List[str]]:
        """Calculate how well an agent matches a query"""
        score = 0.0
        matches = []
        
        # Check required capabilities
        required_score = 0.0
        for capability in query.required_capabilities:
            if capability in agent.capabilities:
                matches.append(capability)
                cap = agent.capabilities[capability]
                
                # Score based on capability level
                level_scores = {
                    CapabilityLevel.BASIC: 0.2,
                    CapabilityLevel.INTERMEDIATE: 0.4,
                    CapabilityLevel.ADVANCED: 0.6,
                    CapabilityLevel.EXPERT: 0.8,
                    CapabilityLevel.MASTER: 1.0
                }
                required_score += level_scores.get(cap.level, 0.2)
        
        if query.required_capabilities:
            required_score /= len(query.required_capabilities)
        else:
            required_score = 1.0
        
        # Check preferred capabilities
        preferred_score = 0.0
        if query.preferred_capabilities:
            for capability in query.preferred_capabilities:
                if capability in agent.capabilities:
                    preferred_score += 0.1
            preferred_score = min(preferred_score, 1.0)
        else:
            preferred_score = 1.0
        
        # Check trust level
        trust_score = 1.0 if agent.trust_level >= query.min_trust_level else 0.0
        
        # Check quality score
        quality_score = 1.0 if agent.metrics.overall_quality_score >= query.min_quality_score else 0.0
        
        # Calculate overall score
        score = (required_score * 0.5 + preferred_score * 0.2 + 
                trust_score * 0.15 + quality_score * 0.15)
        
        return score, matches
    
    def _generate_query_cache_key(self, query: DiscoveryQuery) -> str:
        """Generate cache key for a discovery query"""
        key_data = {
            'required_capabilities': sorted(query.required_capabilities),
            'preferred_capabilities': sorted(query.preferred_capabilities),
            'min_trust_level': query.min_trust_level,
            'min_quality_score': query.min_quality_score,
            'agent_types': sorted(query.agent_types)
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers"""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                handler(data)
        except Exception as e:
            logger.error(f"Event handler failed: {e}")