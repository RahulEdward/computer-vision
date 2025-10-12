"""
Task Delegation System

This module handles intelligent task delegation among multiple agents,
optimizing assignments based on agent capabilities, workload, and performance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from datetime import datetime, timedelta
import uuid
import logging
import heapq
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class AssignmentStatus(Enum):
    """Status of task assignments"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REASSIGNED = "reassigned"


class DelegationStrategy(Enum):
    """Strategies for task delegation"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    DEADLINE_AWARE = "deadline_aware"
    LOAD_BALANCED = "load_balanced"
    AUCTION_BASED = "auction_based"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class CapabilityMatch(Enum):
    """Levels of capability matching"""
    NONE = "none"
    PARTIAL = "partial"
    GOOD = "good"
    EXCELLENT = "excellent"
    PERFECT = "perfect"


@dataclass
class TaskRequirement:
    """Represents requirements for a task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Capability requirements
    required_capabilities: List[str] = field(default_factory=list)
    preferred_capabilities: List[str] = field(default_factory=list)
    minimum_capability_level: float = 0.0
    
    # Resource requirements
    cpu_requirement: float = 0.0
    memory_requirement: float = 0.0  # MB
    storage_requirement: float = 0.0  # MB
    network_bandwidth: float = 0.0  # Mbps
    
    # Performance requirements
    max_execution_time: timedelta = timedelta(hours=1)
    min_accuracy: float = 0.95
    min_reliability: float = 0.99
    
    # Environment requirements
    supported_platforms: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    security_level: str = "standard"
    
    # Dependencies
    prerequisite_tasks: List[str] = field(default_factory=list)
    dependent_tasks: List[str] = field(default_factory=list)
    
    # Constraints
    deadline: Optional[datetime] = None
    budget_limit: float = 0.0
    geographic_constraints: List[str] = field(default_factory=list)


@dataclass
class TaskAssignment:
    """Represents a task assignment to an agent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    
    # Assignment details
    status: AssignmentStatus = AssignmentStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Timing
    assigned_at: datetime = field(default_factory=datetime.now)
    accepted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Assignment context
    assignment_reason: str = ""
    delegation_strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED
    capability_match: CapabilityMatch = CapabilityMatch.NONE
    
    # Performance tracking
    estimated_duration: timedelta = timedelta(0)
    actual_duration: timedelta = timedelta(0)
    progress_percentage: float = 0.0
    
    # Quality metrics
    accuracy_score: float = 0.0
    reliability_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Assignment metadata
    assignment_score: float = 0.0
    confidence_level: float = 0.0
    risk_assessment: str = "low"
    
    # Feedback and results
    agent_feedback: str = ""
    assignment_result: Dict[str, Any] = field(default_factory=dict)
    issues_encountered: List[str] = field(default_factory=list)
    
    # Reassignment tracking
    reassignment_count: int = 0
    previous_agents: List[str] = field(default_factory=list)
    reassignment_reasons: List[str] = field(default_factory=list)


@dataclass
class DelegationResult:
    """Result of task delegation operation"""
    delegation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = False
    
    # Delegation details
    strategy_used: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED
    total_tasks: int = 0
    assigned_tasks: int = 0
    unassigned_tasks: int = 0
    
    # Assignments created
    assignments: List[TaskAssignment] = field(default_factory=list)
    assignment_distribution: Dict[str, int] = field(default_factory=dict)  # agent_id -> task_count
    
    # Performance metrics
    delegation_time: timedelta = timedelta(0)
    average_assignment_score: float = 0.0
    load_balance_score: float = 0.0
    
    # Quality assessment
    capability_coverage: float = 0.0
    deadline_feasibility: float = 0.0
    resource_utilization: float = 0.0
    
    # Issues and recommendations
    delegation_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    delegated_at: datetime = field(default_factory=datetime.now)
    delegated_by: str = ""


@dataclass
class DelegationContext:
    """Context for task delegation"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Delegation parameters
    strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED
    max_assignments_per_agent: int = 5
    allow_reassignment: bool = True
    
    # Optimization goals
    optimize_for_speed: bool = False
    optimize_for_quality: bool = True
    optimize_for_cost: bool = False
    optimize_for_load_balance: bool = True
    
    # Constraints
    deadline_buffer: timedelta = timedelta(hours=1)
    max_delegation_time: timedelta = timedelta(minutes=10)
    min_agent_availability: float = 0.2
    
    # Preferences
    preferred_agents: List[str] = field(default_factory=list)
    excluded_agents: List[str] = field(default_factory=list)
    agent_affinity_rules: Dict[str, List[str]] = field(default_factory=dict)
    
    # Quality requirements
    min_assignment_score: float = 0.7
    require_capability_match: bool = True
    allow_partial_matches: bool = True
    
    # Metadata
    requested_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class TaskDelegator:
    """Intelligent task delegation system for multi-agent automation"""
    
    def __init__(self):
        self.assignments: Dict[str, TaskAssignment] = {}
        self.delegation_history: List[DelegationResult] = []
        
        # Delegation strategies
        self.delegation_strategies = {
            DelegationStrategy.ROUND_ROBIN: self._delegate_round_robin,
            DelegationStrategy.LEAST_LOADED: self._delegate_least_loaded,
            DelegationStrategy.CAPABILITY_BASED: self._delegate_capability_based,
            DelegationStrategy.PERFORMANCE_BASED: self._delegate_performance_based,
            DelegationStrategy.COST_OPTIMIZED: self._delegate_cost_optimized,
            DelegationStrategy.DEADLINE_AWARE: self._delegate_deadline_aware,
            DelegationStrategy.LOAD_BALANCED: self._delegate_load_balanced,
            DelegationStrategy.AUCTION_BASED: self._delegate_auction_based,
            DelegationStrategy.MACHINE_LEARNING: self._delegate_ml_based,
            DelegationStrategy.HYBRID: self._delegate_hybrid
        }
        
        # Assignment evaluators
        self.assignment_evaluators = {
            'capability_match': self._evaluate_capability_match,
            'performance_fit': self._evaluate_performance_fit,
            'workload_impact': self._evaluate_workload_impact,
            'deadline_feasibility': self._evaluate_deadline_feasibility,
            'cost_efficiency': self._evaluate_cost_efficiency,
            'risk_assessment': self._evaluate_risk_level
        }
        
        # Task analyzers
        self.task_analyzers = {
            'complexity_analysis': self._analyze_task_complexity,
            'dependency_analysis': self._analyze_task_dependencies,
            'resource_analysis': self._analyze_resource_requirements,
            'timing_analysis': self._analyze_timing_constraints
        }
        
        # Performance tracking
        self.agent_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.delegation_patterns: Dict[str, Any] = {}
        
        # Statistics
        self.delegation_stats = {
            'total_delegations': 0,
            'successful_delegations': 0,
            'failed_delegations': 0,
            'average_delegation_time': timedelta(0),
            'strategy_effectiveness': {},
            'agent_utilization': {},
            'reassignment_rate': 0.0
        }
    
    def delegate_tasks(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                      context: Optional[DelegationContext] = None) -> DelegationResult:
        """Delegate tasks to available agents"""
        start_time = datetime.now()
        
        # Create context if not provided
        if not context:
            context = DelegationContext()
        
        # Create delegation result
        result = DelegationResult(
            strategy_used=context.strategy,
            total_tasks=len(tasks),
            delegated_by="system"
        )
        
        try:
            # Analyze tasks
            analyzed_tasks = self._analyze_tasks(tasks)
            
            # Filter and score agents
            suitable_agents = self._filter_suitable_agents(agents, analyzed_tasks, context)
            if not suitable_agents:
                result.delegation_issues.append("No suitable agents found")
                return result
            
            # Apply delegation strategy
            delegation_func = self.delegation_strategies.get(context.strategy)
            if not delegation_func:
                result.delegation_issues.append(f"Unknown delegation strategy: {context.strategy}")
                return result
            
            # Execute delegation
            assignments = delegation_func(analyzed_tasks, suitable_agents, context)
            result.assignments = assignments
            result.assigned_tasks = len(assignments)
            result.unassigned_tasks = result.total_tasks - result.assigned_tasks
            
            # Store assignments
            for assignment in assignments:
                self.assignments[assignment.id] = assignment
            
            # Calculate metrics
            result.delegation_time = datetime.now() - start_time
            result.average_assignment_score = self._calculate_average_assignment_score(assignments)
            result.load_balance_score = self._calculate_load_balance_score(assignments, suitable_agents)
            result.capability_coverage = self._calculate_capability_coverage(assignments, analyzed_tasks)
            result.deadline_feasibility = self._calculate_deadline_feasibility(assignments)
            result.resource_utilization = self._calculate_resource_utilization(assignments, suitable_agents)
            
            # Generate assignment distribution
            result.assignment_distribution = self._calculate_assignment_distribution(assignments)
            
            # Generate recommendations
            result.recommendations = self._generate_delegation_recommendations(result, context)
            
            # Determine success
            result.success = (result.assigned_tasks > 0 and 
                            result.average_assignment_score >= context.min_assignment_score)
            
        except Exception as e:
            result.success = False
            result.delegation_issues.append(f"Delegation failed: {e}")
            logger.error(f"Task delegation failed: {e}")
        
        # Store delegation history
        self.delegation_history.append(result)
        
        # Update statistics
        self._update_delegation_stats(result)
        
        logger.info(f"Task delegation completed: {result.success}")
        return result
    
    def reassign_task(self, assignment_id: str, new_agent_id: str, reason: str = "") -> bool:
        """Reassign a task to a different agent"""
        try:
            assignment = self.assignments.get(assignment_id)
            if not assignment:
                logger.error(f"Assignment not found: {assignment_id}")
                return False
            
            # Update assignment
            assignment.previous_agents.append(assignment.agent_id)
            assignment.reassignment_reasons.append(reason)
            assignment.reassignment_count += 1
            assignment.agent_id = new_agent_id
            assignment.status = AssignmentStatus.REASSIGNED
            assignment.assigned_at = datetime.now()
            
            logger.info(f"Task reassigned: {assignment_id} -> {new_agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task reassignment failed: {e}")
            return False
    
    def cancel_assignment(self, assignment_id: str, reason: str = "") -> bool:
        """Cancel a task assignment"""
        try:
            assignment = self.assignments.get(assignment_id)
            if not assignment:
                return False
            
            assignment.status = AssignmentStatus.CANCELLED
            assignment.agent_feedback = reason
            
            logger.info(f"Assignment cancelled: {assignment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Assignment cancellation failed: {e}")
            return False
    
    def update_assignment_progress(self, assignment_id: str, progress: float, 
                                  status: Optional[AssignmentStatus] = None) -> bool:
        """Update progress of a task assignment"""
        try:
            assignment = self.assignments.get(assignment_id)
            if not assignment:
                return False
            
            assignment.progress_percentage = max(0.0, min(100.0, progress))
            
            if status:
                assignment.status = status
                
                if status == AssignmentStatus.IN_PROGRESS and not assignment.started_at:
                    assignment.started_at = datetime.now()
                elif status == AssignmentStatus.COMPLETED and not assignment.completed_at:
                    assignment.completed_at = datetime.now()
                    if assignment.started_at:
                        assignment.actual_duration = assignment.completed_at - assignment.started_at
            
            return True
            
        except Exception as e:
            logger.error(f"Assignment progress update failed: {e}")
            return False
    
    def get_agent_assignments(self, agent_id: str, status_filter: Optional[List[AssignmentStatus]] = None) -> List[TaskAssignment]:
        """Get assignments for a specific agent"""
        agent_assignments = []
        
        for assignment in self.assignments.values():
            if assignment.agent_id == agent_id:
                if not status_filter or assignment.status in status_filter:
                    agent_assignments.append(assignment)
        
        return agent_assignments
    
    def get_task_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """Get assignment for a specific task"""
        for assignment in self.assignments.values():
            if assignment.task_id == task_id:
                return assignment
        return None
    
    def optimize_assignments(self, optimization_strategy: str = "load_balance") -> bool:
        """Optimize current task assignments"""
        try:
            if optimization_strategy == "load_balance":
                return self._optimize_load_balance()
            elif optimization_strategy == "deadline_pressure":
                return self._optimize_deadline_pressure()
            elif optimization_strategy == "capability_match":
                return self._optimize_capability_match()
            else:
                logger.error(f"Unknown optimization strategy: {optimization_strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Assignment optimization failed: {e}")
            return False
    
    def _analyze_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze tasks to extract requirements and characteristics"""
        analyzed_tasks = []
        
        for task in tasks:
            analyzed_task = task.copy()
            
            # Apply task analyzers
            for analyzer_name, analyzer in self.task_analyzers.items():
                analysis_result = analyzer(task)
                analyzed_task[analyzer_name] = analysis_result
            
            analyzed_tasks.append(analyzed_task)
        
        return analyzed_tasks
    
    def _filter_suitable_agents(self, agents: List[Any], tasks: List[Dict[str, Any]], 
                               context: DelegationContext) -> List[Any]:
        """Filter agents suitable for the given tasks"""
        suitable_agents = []
        
        for agent in agents:
            # Check basic availability
            if not self._is_agent_available(agent, context):
                continue
            
            # Check if agent is excluded
            if agent.id in context.excluded_agents:
                continue
            
            # Check capability requirements
            if context.require_capability_match:
                if not self._agent_has_required_capabilities(agent, tasks):
                    continue
            
            suitable_agents.append(agent)
        
        return suitable_agents
    
    def _is_agent_available(self, agent: Any, context: DelegationContext) -> bool:
        """Check if agent is available for new assignments"""
        # Check agent status
        if hasattr(agent, 'status') and agent.status.value not in ['idle', 'busy']:
            return False
        
        # Check current load
        if hasattr(agent, 'current_load') and agent.current_load > (1.0 - context.min_agent_availability):
            return False
        
        # Check current task count
        current_assignments = len(self.get_agent_assignments(agent.id, 
                                                            [AssignmentStatus.ASSIGNED, 
                                                             AssignmentStatus.IN_PROGRESS]))
        if current_assignments >= context.max_assignments_per_agent:
            return False
        
        return True
    
    def _agent_has_required_capabilities(self, agent: Any, tasks: List[Dict[str, Any]]) -> bool:
        """Check if agent has required capabilities for tasks"""
        if not hasattr(agent, 'capabilities'):
            return False
        
        agent_capabilities = [cap.name for cap in agent.capabilities]
        
        for task in tasks:
            required_caps = task.get('required_capabilities', [])
            if required_caps and not any(cap in agent_capabilities for cap in required_caps):
                return False
        
        return True
    
    # Delegation strategy implementations
    def _delegate_round_robin(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                             context: DelegationContext) -> List[TaskAssignment]:
        """Round-robin delegation strategy"""
        assignments = []
        agent_index = 0
        
        for task in tasks:
            if agent_index >= len(agents):
                agent_index = 0
            
            agent = agents[agent_index]
            assignment = self._create_assignment(task, agent, context)
            assignment.delegation_strategy = DelegationStrategy.ROUND_ROBIN
            assignment.assignment_reason = "Round-robin distribution"
            
            assignments.append(assignment)
            agent_index += 1
        
        return assignments
    
    def _delegate_least_loaded(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                              context: DelegationContext) -> List[TaskAssignment]:
        """Least loaded delegation strategy"""
        assignments = []
        
        for task in tasks:
            # Find least loaded agent
            least_loaded_agent = min(agents, key=lambda a: getattr(a, 'current_load', 0.0))
            
            assignment = self._create_assignment(task, least_loaded_agent, context)
            assignment.delegation_strategy = DelegationStrategy.LEAST_LOADED
            assignment.assignment_reason = "Least loaded agent selected"
            
            assignments.append(assignment)
            
            # Update agent load (simplified)
            if hasattr(least_loaded_agent, 'current_load'):
                least_loaded_agent.current_load += 0.1
        
        return assignments
    
    def _delegate_capability_based(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                                  context: DelegationContext) -> List[TaskAssignment]:
        """Capability-based delegation strategy"""
        assignments = []
        
        for task in tasks:
            # Score agents based on capability match
            agent_scores = []
            
            for agent in agents:
                score = self._calculate_capability_match_score(task, agent)
                agent_scores.append((agent, score))
            
            # Sort by score (descending)
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            
            if agent_scores:
                best_agent, best_score = agent_scores[0]
                assignment = self._create_assignment(task, best_agent, context)
                assignment.delegation_strategy = DelegationStrategy.CAPABILITY_BASED
                assignment.assignment_reason = f"Best capability match (score: {best_score:.2f})"
                assignment.assignment_score = best_score
                assignment.capability_match = self._determine_capability_match_level(best_score)
                
                assignments.append(assignment)
        
        return assignments
    
    def _delegate_performance_based(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                                   context: DelegationContext) -> List[TaskAssignment]:
        """Performance-based delegation strategy"""
        assignments = []
        
        for task in tasks:
            # Score agents based on performance metrics
            agent_scores = []
            
            for agent in agents:
                score = self._calculate_performance_score(task, agent)
                agent_scores.append((agent, score))
            
            # Sort by score (descending)
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            
            if agent_scores:
                best_agent, best_score = agent_scores[0]
                assignment = self._create_assignment(task, best_agent, context)
                assignment.delegation_strategy = DelegationStrategy.PERFORMANCE_BASED
                assignment.assignment_reason = f"Best performance match (score: {best_score:.2f})"
                assignment.assignment_score = best_score
                
                assignments.append(assignment)
        
        return assignments
    
    def _delegate_cost_optimized(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                                context: DelegationContext) -> List[TaskAssignment]:
        """Cost-optimized delegation strategy"""
        # Simplified cost optimization
        return self._delegate_least_loaded(tasks, agents, context)
    
    def _delegate_deadline_aware(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                                context: DelegationContext) -> List[TaskAssignment]:
        """Deadline-aware delegation strategy"""
        assignments = []
        
        # Sort tasks by deadline urgency
        sorted_tasks = sorted(tasks, key=lambda t: t.get('deadline', datetime.max))
        
        for task in sorted_tasks:
            # Find agent with best deadline feasibility
            best_agent = None
            best_feasibility = 0.0
            
            for agent in agents:
                feasibility = self._calculate_deadline_feasibility_score(task, agent)
                if feasibility > best_feasibility:
                    best_feasibility = feasibility
                    best_agent = agent
            
            if best_agent:
                assignment = self._create_assignment(task, best_agent, context)
                assignment.delegation_strategy = DelegationStrategy.DEADLINE_AWARE
                assignment.assignment_reason = f"Best deadline feasibility (score: {best_feasibility:.2f})"
                assignment.assignment_score = best_feasibility
                
                assignments.append(assignment)
        
        return assignments
    
    def _delegate_load_balanced(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                               context: DelegationContext) -> List[TaskAssignment]:
        """Load-balanced delegation strategy"""
        assignments = []
        agent_loads = {agent.id: getattr(agent, 'current_load', 0.0) for agent in agents}
        
        for task in tasks:
            # Find agent with lowest projected load
            min_load_agent = min(agents, key=lambda a: agent_loads[a.id])
            
            assignment = self._create_assignment(task, min_load_agent, context)
            assignment.delegation_strategy = DelegationStrategy.LOAD_BALANCED
            assignment.assignment_reason = "Load balancing optimization"
            
            assignments.append(assignment)
            
            # Update projected load
            task_load = self._estimate_task_load(task)
            agent_loads[min_load_agent.id] += task_load
        
        return assignments
    
    def _delegate_auction_based(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                               context: DelegationContext) -> List[TaskAssignment]:
        """Auction-based delegation strategy"""
        assignments = []
        
        for task in tasks:
            # Simulate auction process
            bids = []
            
            for agent in agents:
                bid_score = self._calculate_agent_bid(task, agent)
                bids.append((agent, bid_score))
            
            # Select highest bidder
            if bids:
                bids.sort(key=lambda x: x[1], reverse=True)
                winning_agent, winning_bid = bids[0]
                
                assignment = self._create_assignment(task, winning_agent, context)
                assignment.delegation_strategy = DelegationStrategy.AUCTION_BASED
                assignment.assignment_reason = f"Auction winner (bid: {winning_bid:.2f})"
                assignment.assignment_score = winning_bid
                
                assignments.append(assignment)
        
        return assignments
    
    def _delegate_ml_based(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                          context: DelegationContext) -> List[TaskAssignment]:
        """Machine learning-based delegation strategy"""
        # Simplified ML-based delegation (would use trained models in practice)
        return self._delegate_capability_based(tasks, agents, context)
    
    def _delegate_hybrid(self, tasks: List[Dict[str, Any]], agents: List[Any], 
                        context: DelegationContext) -> List[TaskAssignment]:
        """Hybrid delegation strategy combining multiple approaches"""
        assignments = []
        
        for task in tasks:
            # Combine multiple scoring methods
            agent_scores = []
            
            for agent in agents:
                capability_score = self._calculate_capability_match_score(task, agent)
                performance_score = self._calculate_performance_score(task, agent)
                load_score = 1.0 - getattr(agent, 'current_load', 0.0)
                deadline_score = self._calculate_deadline_feasibility_score(task, agent)
                
                # Weighted combination
                combined_score = (capability_score * 0.4 + 
                                performance_score * 0.3 + 
                                load_score * 0.2 + 
                                deadline_score * 0.1)
                
                agent_scores.append((agent, combined_score))
            
            # Select best agent
            if agent_scores:
                agent_scores.sort(key=lambda x: x[1], reverse=True)
                best_agent, best_score = agent_scores[0]
                
                assignment = self._create_assignment(task, best_agent, context)
                assignment.delegation_strategy = DelegationStrategy.HYBRID
                assignment.assignment_reason = f"Hybrid optimization (score: {best_score:.2f})"
                assignment.assignment_score = best_score
                
                assignments.append(assignment)
        
        return assignments
    
    def _create_assignment(self, task: Dict[str, Any], agent: Any, 
                          context: DelegationContext) -> TaskAssignment:
        """Create a task assignment"""
        assignment = TaskAssignment(
            task_id=task.get('id', str(uuid.uuid4())),
            agent_id=agent.id,
            status=AssignmentStatus.ASSIGNED,
            priority=TaskPriority(task.get('priority', 'normal')),
            deadline=task.get('deadline'),
            estimated_duration=timedelta(minutes=task.get('estimated_minutes', 30))
        )
        
        # Set confidence level based on assignment quality
        assignment.confidence_level = self._calculate_assignment_confidence(task, agent)
        
        return assignment
    
    # Scoring and evaluation methods
    def _calculate_capability_match_score(self, task: Dict[str, Any], agent: Any) -> float:
        """Calculate capability match score between task and agent"""
        if not hasattr(agent, 'capabilities'):
            return 0.0
        
        required_caps = task.get('required_capabilities', [])
        if not required_caps:
            return 1.0
        
        agent_capabilities = [cap.name for cap in agent.capabilities]
        matches = sum(1 for cap in required_caps if cap in agent_capabilities)
        
        return matches / len(required_caps)
    
    def _calculate_performance_score(self, task: Dict[str, Any], agent: Any) -> float:
        """Calculate performance score for agent on task"""
        if not hasattr(agent, 'capabilities'):
            return 0.5
        
        # Average performance metrics from capabilities
        total_score = 0.0
        capability_count = 0
        
        for capability in agent.capabilities:
            score = (capability.accuracy + capability.reliability) / 2
            total_score += score
            capability_count += 1
        
        return total_score / capability_count if capability_count > 0 else 0.5
    
    def _calculate_deadline_feasibility_score(self, task: Dict[str, Any], agent: Any) -> float:
        """Calculate deadline feasibility score"""
        deadline = task.get('deadline')
        if not deadline:
            return 1.0
        
        estimated_duration = timedelta(minutes=task.get('estimated_minutes', 30))
        time_available = deadline - datetime.now()
        
        if time_available <= timedelta(0):
            return 0.0
        
        feasibility = min(1.0, time_available.total_seconds() / estimated_duration.total_seconds())
        return feasibility
    
    def _calculate_agent_bid(self, task: Dict[str, Any], agent: Any) -> float:
        """Calculate agent bid for auction-based delegation"""
        # Combine multiple factors for bidding
        capability_score = self._calculate_capability_match_score(task, agent)
        performance_score = self._calculate_performance_score(task, agent)
        availability_score = 1.0 - getattr(agent, 'current_load', 0.0)
        
        return (capability_score + performance_score + availability_score) / 3
    
    def _estimate_task_load(self, task: Dict[str, Any]) -> float:
        """Estimate load impact of a task"""
        # Simplified load estimation
        estimated_minutes = task.get('estimated_minutes', 30)
        return min(1.0, estimated_minutes / 60.0)  # Normalize to 0-1 scale
    
    def _calculate_assignment_confidence(self, task: Dict[str, Any], agent: Any) -> float:
        """Calculate confidence level for assignment"""
        capability_score = self._calculate_capability_match_score(task, agent)
        performance_score = self._calculate_performance_score(task, agent)
        
        return (capability_score + performance_score) / 2
    
    def _determine_capability_match_level(self, score: float) -> CapabilityMatch:
        """Determine capability match level from score"""
        if score >= 0.95:
            return CapabilityMatch.PERFECT
        elif score >= 0.8:
            return CapabilityMatch.EXCELLENT
        elif score >= 0.6:
            return CapabilityMatch.GOOD
        elif score >= 0.3:
            return CapabilityMatch.PARTIAL
        else:
            return CapabilityMatch.NONE
    
    # Task analyzers
    def _analyze_task_complexity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity"""
        return {
            'complexity_level': task.get('complexity', 'medium'),
            'estimated_effort': task.get('estimated_minutes', 30),
            'skill_requirements': task.get('required_capabilities', [])
        }
    
    def _analyze_task_dependencies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task dependencies"""
        return {
            'prerequisites': task.get('prerequisites', []),
            'dependents': task.get('dependents', []),
            'blocking_tasks': task.get('blocking_tasks', [])
        }
    
    def _analyze_resource_requirements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource requirements"""
        return {
            'cpu_requirement': task.get('cpu_requirement', 0.0),
            'memory_requirement': task.get('memory_requirement', 0.0),
            'network_requirement': task.get('network_requirement', 0.0)
        }
    
    def _analyze_timing_constraints(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing constraints"""
        return {
            'deadline': task.get('deadline'),
            'estimated_duration': task.get('estimated_minutes', 30),
            'priority': task.get('priority', 'normal')
        }
    
    # Metrics calculation
    def _calculate_average_assignment_score(self, assignments: List[TaskAssignment]) -> float:
        """Calculate average assignment score"""
        if not assignments:
            return 0.0
        
        total_score = sum(assignment.assignment_score for assignment in assignments)
        return total_score / len(assignments)
    
    def _calculate_load_balance_score(self, assignments: List[TaskAssignment], agents: List[Any]) -> float:
        """Calculate load balance score"""
        if not assignments or not agents:
            return 0.0
        
        # Calculate load distribution
        agent_loads = {}
        for assignment in assignments:
            agent_id = assignment.agent_id
            agent_loads[agent_id] = agent_loads.get(agent_id, 0) + 1
        
        # Calculate balance score (lower variance = better balance)
        loads = list(agent_loads.values())
        if len(loads) <= 1:
            return 1.0
        
        avg_load = sum(loads) / len(loads)
        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        
        # Convert variance to balance score (0-1, higher is better)
        balance_score = 1.0 / (1.0 + variance)
        return balance_score
    
    def _calculate_capability_coverage(self, assignments: List[TaskAssignment], tasks: List[Dict[str, Any]]) -> float:
        """Calculate capability coverage score"""
        if not tasks:
            return 1.0
        
        total_requirements = 0
        covered_requirements = 0
        
        for task in tasks:
            required_caps = task.get('required_capabilities', [])
            total_requirements += len(required_caps)
            
            # Find assignment for this task
            task_assignment = next((a for a in assignments if a.task_id == task.get('id')), None)
            if task_assignment and task_assignment.capability_match != CapabilityMatch.NONE:
                covered_requirements += len(required_caps)
        
        return covered_requirements / total_requirements if total_requirements > 0 else 1.0
    
    def _calculate_deadline_feasibility(self, assignments: List[TaskAssignment]) -> float:
        """Calculate deadline feasibility score"""
        if not assignments:
            return 1.0
        
        feasible_assignments = 0
        
        for assignment in assignments:
            if not assignment.deadline:
                feasible_assignments += 1
                continue
            
            time_available = assignment.deadline - datetime.now()
            if time_available >= assignment.estimated_duration:
                feasible_assignments += 1
        
        return feasible_assignments / len(assignments)
    
    def _calculate_resource_utilization(self, assignments: List[TaskAssignment], agents: List[Any]) -> float:
        """Calculate resource utilization score"""
        if not agents:
            return 0.0
        
        utilized_agents = len(set(assignment.agent_id for assignment in assignments))
        return utilized_agents / len(agents)
    
    def _calculate_assignment_distribution(self, assignments: List[TaskAssignment]) -> Dict[str, int]:
        """Calculate assignment distribution across agents"""
        distribution = {}
        
        for assignment in assignments:
            agent_id = assignment.agent_id
            distribution[agent_id] = distribution.get(agent_id, 0) + 1
        
        return distribution
    
    def _generate_delegation_recommendations(self, result: DelegationResult, 
                                           context: DelegationContext) -> List[str]:
        """Generate recommendations for improving delegation"""
        recommendations = []
        
        # Check assignment success rate
        if result.assigned_tasks < result.total_tasks:
            recommendations.append("Consider relaxing capability requirements or adding more agents")
        
        # Check load balance
        if result.load_balance_score < 0.7:
            recommendations.append("Consider using load-balanced delegation strategy")
        
        # Check deadline feasibility
        if result.deadline_feasibility < 0.8:
            recommendations.append("Review task deadlines and estimated durations")
        
        # Check capability coverage
        if result.capability_coverage < 0.9:
            recommendations.append("Ensure agents have required capabilities for tasks")
        
        return recommendations
    
    # Optimization methods
    def _optimize_load_balance(self) -> bool:
        """Optimize assignments for better load balance"""
        # Simplified load balancing optimization
        return True
    
    def _optimize_deadline_pressure(self) -> bool:
        """Optimize assignments based on deadline pressure"""
        # Simplified deadline optimization
        return True
    
    def _optimize_capability_match(self) -> bool:
        """Optimize assignments for better capability matching"""
        # Simplified capability optimization
        return True
    
    def _update_delegation_stats(self, result: DelegationResult) -> None:
        """Update delegation statistics"""
        self.delegation_stats['total_delegations'] += 1
        
        if result.success:
            self.delegation_stats['successful_delegations'] += 1
        else:
            self.delegation_stats['failed_delegations'] += 1
        
        # Update average delegation time
        total_time = (self.delegation_stats['average_delegation_time'] * 
                     (self.delegation_stats['total_delegations'] - 1) + 
                     result.delegation_time)
        self.delegation_stats['average_delegation_time'] = total_time / self.delegation_stats['total_delegations']
        
        # Update strategy effectiveness
        strategy = result.strategy_used.value
        if strategy not in self.delegation_stats['strategy_effectiveness']:
            self.delegation_stats['strategy_effectiveness'][strategy] = {'total': 0, 'successful': 0}
        
        self.delegation_stats['strategy_effectiveness'][strategy]['total'] += 1
        if result.success:
            self.delegation_stats['strategy_effectiveness'][strategy]['successful'] += 1
    
    def get_delegation_statistics(self) -> Dict[str, Any]:
        """Get delegation statistics"""
        return {
            'total_assignments': len(self.assignments),
            'active_assignments': len([a for a in self.assignments.values() 
                                     if a.status in [AssignmentStatus.ASSIGNED, AssignmentStatus.IN_PROGRESS]]),
            'delegation_history_count': len(self.delegation_history),
            'delegation_stats': self.delegation_stats.copy()
        }