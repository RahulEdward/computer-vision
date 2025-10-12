"""
Goal-Oriented Planning Engine

This module implements the main planning engine that analyzes automation goals
and creates optimal execution paths to achieve them.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Callable, Any, Tuple
from datetime import datetime, timedelta
import uuid
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of automation goals"""
    NAVIGATION = "navigation"
    DATA_ENTRY = "data_entry"
    DATA_EXTRACTION = "data_extraction"
    WORKFLOW_AUTOMATION = "workflow_automation"
    TESTING = "testing"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    CUSTOM = "custom"


class GoalStatus(Enum):
    """Status of goal execution"""
    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """Priority levels for goals and tasks"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


@dataclass
class Goal:
    """Represents an automation goal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal_type: GoalType = GoalType.CUSTOM
    priority: Priority = Priority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    
    # Goal definition
    target_state: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Timing
    deadline: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Relationships
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubGoal:
    """Represents a sub-goal within a larger goal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_goal_id: str = ""
    name: str = ""
    description: str = ""
    priority: Priority = Priority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    
    # Sub-goal specifics
    required_actions: List[str] = field(default_factory=list)
    success_condition: str = ""
    failure_condition: str = ""
    
    # Execution
    execution_order: int = 0
    parallel_execution: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    # Timing
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Task:
    """Represents an individual task within a sub-goal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sub_goal_id: str = ""
    name: str = ""
    description: str = ""
    action_type: str = ""
    
    # Task parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    validation_rules: List[str] = field(default_factory=list)
    
    # Execution
    execution_order: int = 0
    is_critical: bool = False
    can_skip: bool = False
    retry_on_failure: bool = True
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Status tracking
    status: GoalStatus = GoalStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[timedelta] = None


@dataclass
class PlanningContext:
    """Context for planning session"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    application_context: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    
    # Planning preferences
    optimization_strategy: str = "balanced"  # speed, reliability, resource_efficient
    risk_tolerance: str = "medium"  # low, medium, high
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[timedelta] = None


@dataclass
class PlanningResult:
    """Result of planning process"""
    session_id: str = ""
    goal_id: str = ""
    success: bool = False
    
    # Planning output
    execution_plan: List[str] = field(default_factory=list)  # Ordered list of sub-goal IDs
    estimated_duration: timedelta = timedelta(0)
    confidence_score: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Alternative plans
    alternative_plans: List[Dict[str, Any]] = field(default_factory=list)
    fallback_plan: Optional[Dict[str, Any]] = None
    
    # Analysis
    complexity_score: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    potential_issues: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    planning_time: timedelta = timedelta(0)
    created_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class GoalPlanner:
    """Main goal-oriented planning engine"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.sub_goals: Dict[str, SubGoal] = {}
        self.tasks: Dict[str, Task] = {}
        self.planning_sessions: Dict[str, PlanningContext] = {}
        
        # Planning algorithms
        self.planning_strategies = {
            'forward_chaining': self._forward_chaining_plan,
            'backward_chaining': self._backward_chaining_plan,
            'hierarchical': self._hierarchical_plan,
            'hybrid': self._hybrid_plan
        }
        
        # Heuristics for path finding
        self.heuristics = {
            'shortest_path': self._shortest_path_heuristic,
            'least_risk': self._least_risk_heuristic,
            'resource_optimal': self._resource_optimal_heuristic,
            'time_optimal': self._time_optimal_heuristic
        }
        
        # Statistics
        self.planning_stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'average_planning_time': timedelta(0),
            'complexity_distribution': {}
        }
    
    def create_goal(self, name: str, description: str, goal_type: GoalType,
                   target_state: Dict[str, Any], success_criteria: List[str],
                   **kwargs) -> Goal:
        """Create a new automation goal"""
        goal = Goal(
            name=name,
            description=description,
            goal_type=goal_type,
            target_state=target_state,
            success_criteria=success_criteria,
            **kwargs
        )
        
        self.goals[goal.id] = goal
        logger.info(f"Created goal: {goal.name} ({goal.id})")
        return goal
    
    def plan_goal_achievement(self, goal_id: str, context: PlanningContext) -> PlanningResult:
        """Create an optimal plan to achieve the specified goal"""
        start_time = datetime.now()
        
        try:
            goal = self.goals.get(goal_id)
            if not goal:
                raise ValueError(f"Goal {goal_id} not found")
            
            # Store planning session
            self.planning_sessions[context.session_id] = context
            
            # Update goal status
            goal.status = GoalStatus.PLANNING
            
            # Decompose goal into sub-goals and tasks
            self._decompose_goal(goal, context)
            
            # Select planning strategy
            strategy = self._select_planning_strategy(goal, context)
            
            # Generate execution plan
            execution_plan = self.planning_strategies[strategy](goal, context)
            
            # Optimize the plan
            optimized_plan = self._optimize_plan(execution_plan, context)
            
            # Validate the plan
            validation_result = self._validate_plan(optimized_plan, goal, context)
            
            # Calculate metrics
            metrics = self._calculate_plan_metrics(optimized_plan, goal)
            
            # Create result
            planning_time = datetime.now() - start_time
            result = PlanningResult(
                session_id=context.session_id,
                goal_id=goal_id,
                success=validation_result['valid'],
                execution_plan=optimized_plan,
                estimated_duration=metrics['duration'],
                confidence_score=metrics['confidence'],
                risk_assessment=metrics['risks'],
                complexity_score=metrics['complexity'],
                resource_requirements=metrics['resources'],
                potential_issues=validation_result['issues'],
                optimization_suggestions=metrics['optimizations'],
                planning_time=planning_time
            )
            
            # Generate alternatives
            result.alternative_plans = self._generate_alternative_plans(goal, context, 3)
            result.fallback_plan = self._generate_fallback_plan(goal, context)
            
            # Update statistics
            self._update_planning_stats(result)
            
            # Update goal status
            goal.status = GoalStatus.READY if result.success else GoalStatus.FAILED
            
            logger.info(f"Planning completed for goal {goal.name}: {result.success}")
            return result
            
        except Exception as e:
            planning_time = datetime.now() - start_time
            error_result = PlanningResult(
                session_id=context.session_id,
                goal_id=goal_id,
                success=False,
                planning_time=planning_time,
                error=str(e)
            )
            
            if goal_id in self.goals:
                self.goals[goal_id].status = GoalStatus.FAILED
            
            logger.error(f"Planning failed for goal {goal_id}: {e}")
            return error_result
    
    def _decompose_goal(self, goal: Goal, context: PlanningContext) -> None:
        """Decompose a goal into sub-goals and tasks"""
        # Analyze goal complexity
        complexity = self._analyze_goal_complexity(goal)
        
        if complexity['requires_decomposition']:
            # Create sub-goals based on goal type and target state
            sub_goals = self._create_sub_goals(goal, context)
            
            for sub_goal in sub_goals:
                self.sub_goals[sub_goal.id] = sub_goal
                goal.sub_goals.append(sub_goal.id)
                
                # Create tasks for each sub-goal
                tasks = self._create_tasks_for_sub_goal(sub_goal, goal, context)
                for task in tasks:
                    self.tasks[task.id] = task
                    sub_goal.required_actions.append(task.id)
    
    def _select_planning_strategy(self, goal: Goal, context: PlanningContext) -> str:
        """Select the best planning strategy for the goal"""
        # Analyze goal characteristics
        has_clear_path = len(goal.sub_goals) > 0
        has_dependencies = any(self.sub_goals[sg_id].required_actions 
                             for sg_id in goal.sub_goals)
        complexity = len(goal.sub_goals) + sum(len(self.sub_goals[sg_id].required_actions) 
                                             for sg_id in goal.sub_goals)
        
        # Select strategy based on characteristics
        if complexity > 20:
            return 'hierarchical'
        elif has_dependencies and has_clear_path:
            return 'hybrid'
        elif has_clear_path:
            return 'forward_chaining'
        else:
            return 'backward_chaining'
    
    def _forward_chaining_plan(self, goal: Goal, context: PlanningContext) -> List[str]:
        """Create plan using forward chaining from current state to goal"""
        plan = []
        
        # Sort sub-goals by priority and dependencies
        sorted_sub_goals = self._sort_sub_goals_by_dependencies(goal.sub_goals)
        
        for sub_goal_id in sorted_sub_goals:
            sub_goal = self.sub_goals[sub_goal_id]
            
            # Check if sub-goal can be executed in parallel
            if sub_goal.parallel_execution and plan:
                # Add to parallel execution group
                if isinstance(plan[-1], list):
                    plan[-1].append(sub_goal_id)
                else:
                    plan[-1] = [plan[-1], sub_goal_id]
            else:
                plan.append(sub_goal_id)
        
        return plan
    
    def _backward_chaining_plan(self, goal: Goal, context: PlanningContext) -> List[str]:
        """Create plan using backward chaining from goal to current state"""
        plan = []
        
        # Start from goal state and work backwards
        required_states = [goal.target_state]
        
        while required_states:
            current_state = required_states.pop(0)
            
            # Find sub-goals that can achieve this state
            achieving_sub_goals = self._find_achieving_sub_goals(current_state, goal.sub_goals)
            
            for sub_goal_id in achieving_sub_goals:
                if sub_goal_id not in plan:
                    plan.insert(0, sub_goal_id)
                    
                    # Add prerequisites to required states
                    sub_goal = self.sub_goals[sub_goal_id]
                    prerequisites = self._get_sub_goal_prerequisites(sub_goal)
                    required_states.extend(prerequisites)
        
        return plan
    
    def _hierarchical_plan(self, goal: Goal, context: PlanningContext) -> List[str]:
        """Create hierarchical plan breaking down complex goals"""
        plan = []
        
        # Group sub-goals by hierarchy level
        hierarchy_levels = self._build_hierarchy_levels(goal.sub_goals)
        
        for level in sorted(hierarchy_levels.keys()):
            level_sub_goals = hierarchy_levels[level]
            
            # Sort within level by priority
            level_sub_goals.sort(key=lambda sg_id: self.sub_goals[sg_id].priority.value)
            
            # Add level to plan
            if len(level_sub_goals) == 1:
                plan.append(level_sub_goals[0])
            else:
                plan.append(level_sub_goals)  # Parallel execution
        
        return plan
    
    def _hybrid_plan(self, goal: Goal, context: PlanningContext) -> List[str]:
        """Create plan using hybrid approach combining multiple strategies"""
        # Use hierarchical for high-level structure
        high_level_plan = self._hierarchical_plan(goal, context)
        
        # Use forward chaining for detailed sequencing
        detailed_plan = []
        for item in high_level_plan:
            if isinstance(item, list):
                # Parallel group - optimize internal ordering
                optimized_group = self._optimize_parallel_group(item, context)
                detailed_plan.append(optimized_group)
            else:
                detailed_plan.append(item)
        
        return detailed_plan
    
    def _optimize_plan(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Optimize the execution plan based on context preferences"""
        optimization_strategy = context.optimization_strategy
        
        if optimization_strategy == "speed":
            return self._optimize_for_speed(plan, context)
        elif optimization_strategy == "reliability":
            return self._optimize_for_reliability(plan, context)
        elif optimization_strategy == "resource_efficient":
            return self._optimize_for_resources(plan, context)
        else:  # balanced
            return self._optimize_balanced(plan, context)
    
    def _validate_plan(self, plan: List[str], goal: Goal, context: PlanningContext) -> Dict[str, Any]:
        """Validate the execution plan for feasibility and correctness"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check plan completeness
        if not self._plan_covers_goal(plan, goal):
            validation_result['valid'] = False
            validation_result['issues'].append("Plan does not fully cover goal requirements")
        
        # Check dependencies
        dependency_issues = self._check_dependencies(plan)
        if dependency_issues:
            validation_result['valid'] = False
            validation_result['issues'].extend(dependency_issues)
        
        # Check resource constraints
        resource_issues = self._check_resource_constraints(plan, context)
        if resource_issues:
            validation_result['warnings'].extend(resource_issues)
        
        # Check timing constraints
        timing_issues = self._check_timing_constraints(plan, goal)
        if timing_issues:
            validation_result['warnings'].extend(timing_issues)
        
        return validation_result
    
    def _calculate_plan_metrics(self, plan: List[str], goal: Goal) -> Dict[str, Any]:
        """Calculate various metrics for the execution plan"""
        metrics = {
            'duration': timedelta(0),
            'confidence': 0.0,
            'complexity': 0.0,
            'risks': {},
            'resources': {},
            'optimizations': []
        }
        
        # Calculate duration
        total_duration = timedelta(0)
        for item in plan:
            if isinstance(item, list):
                # Parallel execution - take maximum duration
                parallel_durations = [self._get_sub_goal_duration(sg_id) for sg_id in item]
                total_duration += max(parallel_durations, default=timedelta(0))
            else:
                total_duration += self._get_sub_goal_duration(item)
        
        metrics['duration'] = total_duration
        
        # Calculate confidence based on sub-goal success rates
        confidence_scores = []
        for item in plan:
            if isinstance(item, list):
                parallel_confidence = min(self._get_sub_goal_confidence(sg_id) for sg_id in item)
                confidence_scores.append(parallel_confidence)
            else:
                confidence_scores.append(self._get_sub_goal_confidence(item))
        
        metrics['confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Calculate complexity
        total_tasks = sum(len(self.sub_goals[sg_id].required_actions) 
                         for sg_id in goal.sub_goals)
        metrics['complexity'] = min(total_tasks / 10.0, 1.0)  # Normalize to 0-1
        
        # Assess risks
        metrics['risks'] = self._assess_plan_risks(plan, goal)
        
        # Calculate resource requirements
        metrics['resources'] = self._calculate_resource_requirements(plan)
        
        # Generate optimization suggestions
        metrics['optimizations'] = self._generate_optimization_suggestions(plan, goal)
        
        return metrics
    
    def get_goal_status(self, goal_id: str) -> Optional[GoalStatus]:
        """Get the current status of a goal"""
        goal = self.goals.get(goal_id)
        return goal.status if goal else None
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning engine statistics"""
        return {
            'total_goals': len(self.goals),
            'active_goals': len([g for g in self.goals.values() 
                               if g.status in [GoalStatus.PLANNING, GoalStatus.EXECUTING]]),
            'completed_goals': len([g for g in self.goals.values() 
                                  if g.status == GoalStatus.COMPLETED]),
            'planning_stats': self.planning_stats.copy()
        }
    
    # Helper methods (simplified implementations)
    def _analyze_goal_complexity(self, goal: Goal) -> Dict[str, Any]:
        """Analyze goal complexity to determine decomposition needs"""
        return {'requires_decomposition': len(goal.success_criteria) > 1}
    
    def _create_sub_goals(self, goal: Goal, context: PlanningContext) -> List[SubGoal]:
        """Create sub-goals for a complex goal"""
        sub_goals = []
        for i, criterion in enumerate(goal.success_criteria):
            sub_goal = SubGoal(
                parent_goal_id=goal.id,
                name=f"SubGoal {i+1}",
                description=f"Achieve: {criterion}",
                success_condition=criterion,
                execution_order=i
            )
            sub_goals.append(sub_goal)
        return sub_goals
    
    def _create_tasks_for_sub_goal(self, sub_goal: SubGoal, goal: Goal, 
                                  context: PlanningContext) -> List[Task]:
        """Create tasks for a sub-goal"""
        # Simplified task creation
        task = Task(
            sub_goal_id=sub_goal.id,
            name=f"Execute {sub_goal.name}",
            description=sub_goal.description,
            action_type="automated_action"
        )
        return [task]
    
    def _sort_sub_goals_by_dependencies(self, sub_goal_ids: List[str]) -> List[str]:
        """Sort sub-goals considering their dependencies"""
        # Simplified topological sort
        return sorted(sub_goal_ids, key=lambda sg_id: self.sub_goals[sg_id].execution_order)
    
    def _find_achieving_sub_goals(self, state: Dict[str, Any], sub_goal_ids: List[str]) -> List[str]:
        """Find sub-goals that can achieve the given state"""
        # Simplified implementation
        return sub_goal_ids[:1] if sub_goal_ids else []
    
    def _get_sub_goal_prerequisites(self, sub_goal: SubGoal) -> List[Dict[str, Any]]:
        """Get prerequisites for a sub-goal"""
        return []  # Simplified
    
    def _build_hierarchy_levels(self, sub_goal_ids: List[str]) -> Dict[int, List[str]]:
        """Build hierarchy levels for sub-goals"""
        levels = {}
        for sg_id in sub_goal_ids:
            level = self.sub_goals[sg_id].execution_order
            if level not in levels:
                levels[level] = []
            levels[level].append(sg_id)
        return levels
    
    def _optimize_parallel_group(self, group: List[str], context: PlanningContext) -> List[str]:
        """Optimize execution order within a parallel group"""
        return group  # Simplified
    
    def _optimize_for_speed(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Optimize plan for execution speed"""
        return plan  # Simplified
    
    def _optimize_for_reliability(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Optimize plan for reliability"""
        return plan  # Simplified
    
    def _optimize_for_resources(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Optimize plan for resource efficiency"""
        return plan  # Simplified
    
    def _optimize_balanced(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Optimize plan with balanced approach"""
        return plan  # Simplified
    
    def _plan_covers_goal(self, plan: List[str], goal: Goal) -> bool:
        """Check if plan covers all goal requirements"""
        return len(plan) > 0  # Simplified
    
    def _check_dependencies(self, plan: List[str]) -> List[str]:
        """Check for dependency violations in plan"""
        return []  # Simplified
    
    def _check_resource_constraints(self, plan: List[str], context: PlanningContext) -> List[str]:
        """Check resource constraint violations"""
        return []  # Simplified
    
    def _check_timing_constraints(self, plan: List[str], goal: Goal) -> List[str]:
        """Check timing constraint violations"""
        return []  # Simplified
    
    def _get_sub_goal_duration(self, sub_goal_id: str) -> timedelta:
        """Get estimated duration for a sub-goal"""
        sub_goal = self.sub_goals.get(sub_goal_id)
        return sub_goal.estimated_duration or timedelta(minutes=5)
    
    def _get_sub_goal_confidence(self, sub_goal_id: str) -> float:
        """Get confidence score for a sub-goal"""
        return 0.8  # Simplified
    
    def _assess_plan_risks(self, plan: List[str], goal: Goal) -> Dict[str, Any]:
        """Assess risks in the execution plan"""
        return {'overall_risk': 'medium'}  # Simplified
    
    def _calculate_resource_requirements(self, plan: List[str]) -> Dict[str, Any]:
        """Calculate resource requirements for plan"""
        return {'cpu': 'medium', 'memory': 'low'}  # Simplified
    
    def _generate_optimization_suggestions(self, plan: List[str], goal: Goal) -> List[str]:
        """Generate optimization suggestions for the plan"""
        return ["Consider parallel execution for independent tasks"]  # Simplified
    
    def _generate_alternative_plans(self, goal: Goal, context: PlanningContext, count: int) -> List[Dict[str, Any]]:
        """Generate alternative execution plans"""
        return []  # Simplified
    
    def _generate_fallback_plan(self, goal: Goal, context: PlanningContext) -> Optional[Dict[str, Any]]:
        """Generate a fallback plan for error scenarios"""
        return None  # Simplified
    
    def _update_planning_stats(self, result: PlanningResult) -> None:
        """Update planning statistics"""
        self.planning_stats['total_plans'] += 1
        if result.success:
            self.planning_stats['successful_plans'] += 1
    
    def _shortest_path_heuristic(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Heuristic for shortest path planning"""
        return 1.0  # Simplified
    
    def _least_risk_heuristic(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Heuristic for least risk planning"""
        return 1.0  # Simplified
    
    def _resource_optimal_heuristic(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Heuristic for resource optimal planning"""
        return 1.0  # Simplified
    
    def _time_optimal_heuristic(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Heuristic for time optimal planning"""
        return 1.0  # Simplified