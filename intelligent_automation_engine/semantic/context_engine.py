"""
Context-Aware Engine

Provides context-aware automation decisions by understanding application states,
user intent, and environmental factors to make intelligent automation choices.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from .state_manager import ApplicationState, StateManager
from .transition_analyzer import TransitionAnalyzer, TransitionPattern


class ContextType(Enum):
    """Types of context information."""
    USER_INTENT = "user_intent"
    APPLICATION_STATE = "application_state"
    SYSTEM_STATE = "system_state"
    TEMPORAL = "temporal"
    ENVIRONMENTAL = "environmental"
    WORKFLOW = "workflow"
    SECURITY = "security"
    PERFORMANCE = "performance"


class DecisionType(Enum):
    """Types of automation decisions."""
    ACTION_SELECTION = "action_selection"
    TIMING_OPTIMIZATION = "timing_optimization"
    ERROR_RECOVERY = "error_recovery"
    WORKFLOW_ADAPTATION = "workflow_adaptation"
    RESOURCE_ALLOCATION = "resource_allocation"
    SECURITY_VALIDATION = "security_validation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class ContextInfo:
    """Information about the current context."""
    context_id: str
    context_type: ContextType
    timestamp: datetime
    
    # Context data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source: str = "unknown"
    confidence: float = 1.0
    expiry: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if context information has expired."""
        if self.expiry:
            return datetime.now() > self.expiry
        return False
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from context data."""
        return self.data.get(key, default)
    
    def set_value(self, key: str, value: Any):
        """Set a value in context data."""
        self.data[key] = value


@dataclass
class DecisionContext:
    """Context for making automation decisions."""
    decision_id: str
    decision_type: DecisionType
    timestamp: datetime
    
    # Current state
    current_state: Optional[ApplicationState] = None
    target_state: Optional[ApplicationState] = None
    
    # Available actions
    available_actions: List[str] = field(default_factory=list)
    
    # Context information
    contexts: List[ContextInfo] = field(default_factory=list)
    
    # Constraints
    time_constraints: Optional[Tuple[datetime, datetime]] = None
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    security_constraints: List[str] = field(default_factory=list)
    
    # Goals
    primary_goal: Optional[str] = None
    secondary_goals: List[str] = field(default_factory=list)
    
    def add_context(self, context: ContextInfo):
        """Add context information."""
        self.contexts.append(context)
    
    def get_contexts_by_type(self, context_type: ContextType) -> List[ContextInfo]:
        """Get contexts of a specific type."""
        return [ctx for ctx in self.contexts if ctx.context_type == context_type]
    
    def has_constraint(self, constraint_type: str) -> bool:
        """Check if a specific constraint exists."""
        return (constraint_type in self.resource_constraints or
                constraint_type in self.security_constraints)


@dataclass
class AutomationDecision:
    """Result of an automation decision."""
    decision_id: str
    decision_type: DecisionType
    timestamp: datetime
    
    # Decision result
    recommended_action: str
    confidence: float
    reasoning: str
    
    # Alternative options
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (action, confidence)
    
    # Context used
    context_factors: List[str] = field(default_factory=list)
    
    # Execution guidance
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "low"
    risk_factors: List[str] = field(default_factory=list)
    
    # Performance prediction
    estimated_duration: Optional[float] = None
    success_probability: float = 1.0
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get the confidence level enum."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ContextAwareEngine:
    """
    Context-aware engine that makes intelligent automation decisions based on
    comprehensive understanding of the current situation.
    """
    
    def __init__(self, state_manager: StateManager, 
                 transition_analyzer: TransitionAnalyzer):
        """Initialize the context-aware engine."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.transition_analyzer = transition_analyzer
        
        # Context storage
        self.active_contexts: Dict[str, ContextInfo] = {}
        self.context_history: List[ContextInfo] = []
        
        # Decision history
        self.decision_history: List[AutomationDecision] = []
        
        # Learning data
        self.decision_outcomes: Dict[str, Dict[str, Any]] = {}
        self.pattern_effectiveness: Dict[str, float] = {}
        
        # Configuration
        self.config = {
            'context_expiry_minutes': 30,
            'max_context_history': 1000,
            'max_decision_history': 500,
            'confidence_threshold': 0.6,
            'risk_tolerance': 'medium'
        }
        
        # Statistics
        self.stats = {
            'decisions_made': 0,
            'successful_decisions': 0,
            'context_updates': 0,
            'pattern_matches': 0
        }
    
    async def update_context(self, context: ContextInfo):
        """Update context information."""
        try:
            # Remove expired contexts
            await self._cleanup_expired_contexts()
            
            # Add new context
            self.active_contexts[context.context_id] = context
            self.context_history.append(context)
            
            # Limit history size
            if len(self.context_history) > self.config['max_context_history']:
                self.context_history = self.context_history[-self.config['max_context_history']:]
            
            self.stats['context_updates'] += 1
            self.logger.debug(f"Context updated: {context.context_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to update context: {e}")
    
    async def make_decision(self, decision_context: DecisionContext) -> AutomationDecision:
        """
        Make an intelligent automation decision based on current context.
        
        Args:
            decision_context: Context for the decision
            
        Returns:
            AutomationDecision: The recommended decision
        """
        try:
            self.logger.info(f"Making decision: {decision_context.decision_type.value}")
            
            # Gather relevant context
            relevant_contexts = await self._gather_relevant_context(decision_context)
            
            # Analyze current situation
            situation_analysis = await self._analyze_situation(decision_context, relevant_contexts)
            
            # Generate decision options
            options = await self._generate_decision_options(decision_context, situation_analysis)
            
            # Evaluate options
            best_option = await self._evaluate_options(options, decision_context, situation_analysis)
            
            # Create decision
            decision = await self._create_decision(best_option, decision_context, situation_analysis)
            
            # Store decision
            self.decision_history.append(decision)
            if len(self.decision_history) > self.config['max_decision_history']:
                self.decision_history = self.decision_history[-self.config['max_decision_history']:]
            
            self.stats['decisions_made'] += 1
            
            self.logger.info(f"Decision made: {decision.recommended_action} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            # Return a safe default decision
            return AutomationDecision(
                decision_id=f"error_{datetime.now().timestamp()}",
                decision_type=decision_context.decision_type,
                timestamp=datetime.now(),
                recommended_action="wait",
                confidence=0.1,
                reasoning=f"Error in decision making: {e}",
                risk_level="high"
            )
    
    async def _gather_relevant_context(self, decision_context: DecisionContext) -> List[ContextInfo]:
        """Gather context information relevant to the decision."""
        relevant_contexts = []
        
        # Add contexts from decision context
        relevant_contexts.extend(decision_context.contexts)
        
        # Add current application state context
        if decision_context.current_state:
            app_context = ContextInfo(
                context_id=f"app_state_{decision_context.current_state.state_id}",
                context_type=ContextType.APPLICATION_STATE,
                timestamp=datetime.now(),
                data={
                    'state': decision_context.current_state.to_dict(),
                    'application': decision_context.current_state.application_name,
                    'window_title': decision_context.current_state.window_title
                },
                source="state_manager"
            )
            relevant_contexts.append(app_context)
        
        # Add temporal context
        temporal_context = ContextInfo(
            context_id=f"temporal_{datetime.now().timestamp()}",
            context_type=ContextType.TEMPORAL,
            timestamp=datetime.now(),
            data={
                'current_time': datetime.now().isoformat(),
                'day_of_week': datetime.now().weekday(),
                'hour_of_day': datetime.now().hour
            },
            source="system"
        )
        relevant_contexts.append(temporal_context)
        
        # Add workflow context if available
        if decision_context.primary_goal:
            workflow_context = ContextInfo(
                context_id=f"workflow_{decision_context.decision_id}",
                context_type=ContextType.WORKFLOW,
                timestamp=datetime.now(),
                data={
                    'primary_goal': decision_context.primary_goal,
                    'secondary_goals': decision_context.secondary_goals,
                    'available_actions': decision_context.available_actions
                },
                source="workflow_manager"
            )
            relevant_contexts.append(workflow_context)
        
        return relevant_contexts
    
    async def _analyze_situation(self, decision_context: DecisionContext,
                                contexts: List[ContextInfo]) -> Dict[str, Any]:
        """Analyze the current situation."""
        analysis = {
            'complexity': 'low',
            'urgency': 'normal',
            'risk_factors': [],
            'opportunities': [],
            'constraints': [],
            'patterns_matched': [],
            'confidence_factors': []
        }
        
        # Analyze application state
        if decision_context.current_state:
            state = decision_context.current_state
            
            # Check for error states
            if state.state_type.value == 'error':
                analysis['urgency'] = 'high'
                analysis['risk_factors'].append('application_error')
            
            # Check for loading states
            if state.state_type.value == 'loading':
                analysis['constraints'].append('wait_for_load')
            
            # Check for modal dialogs
            if state.state_type.value == 'modal':
                analysis['constraints'].append('modal_dialog_active')
        
        # Analyze temporal context
        temporal_contexts = [ctx for ctx in contexts if ctx.context_type == ContextType.TEMPORAL]
        if temporal_contexts:
            temporal_data = temporal_contexts[0].data
            hour = temporal_data.get('hour_of_day', 12)
            
            # Business hours consideration
            if 9 <= hour <= 17:
                analysis['opportunities'].append('business_hours')
            else:
                analysis['constraints'].append('off_hours')
        
        # Analyze workflow context
        workflow_contexts = [ctx for ctx in contexts if ctx.context_type == ContextType.WORKFLOW]
        if workflow_contexts:
            workflow_data = workflow_contexts[0].data
            
            if len(workflow_data.get('available_actions', [])) > 5:
                analysis['complexity'] = 'high'
            elif len(workflow_data.get('available_actions', [])) > 2:
                analysis['complexity'] = 'medium'
        
        # Check for matching patterns
        if decision_context.current_state:
            patterns = self.transition_analyzer.get_patterns_for_state(
                decision_context.current_state.state_id
            )
            analysis['patterns_matched'] = [p.pattern_id for p in patterns]
            
            if patterns:
                analysis['confidence_factors'].append('historical_patterns')
                self.stats['pattern_matches'] += 1
        
        return analysis
    
    async def _generate_decision_options(self, decision_context: DecisionContext,
                                       situation_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate possible decision options."""
        options = []
        
        # Base options from available actions
        for action in decision_context.available_actions:
            option = {
                'action': action,
                'base_confidence': 0.5,
                'reasoning': f"Available action: {action}",
                'risk_level': 'medium',
                'estimated_duration': 1.0
            }
            options.append(option)
        
        # Add pattern-based options
        patterns_matched = situation_analysis.get('patterns_matched', [])
        for pattern_id in patterns_matched:
            pattern = self.transition_analyzer.get_pattern_by_id(pattern_id)
            if pattern and decision_context.current_state:
                next_state = pattern.get_next_state(decision_context.current_state.state_id)
                if next_state:
                    option = {
                        'action': f"follow_pattern_{pattern_id}",
                        'base_confidence': pattern.confidence,
                        'reasoning': f"Following historical pattern: {pattern.pattern_type.value}",
                        'risk_level': 'low',
                        'estimated_duration': pattern.average_duration,
                        'pattern_id': pattern_id
                    }
                    options.append(option)
        
        # Add safety options
        if situation_analysis.get('urgency') == 'high':
            options.append({
                'action': 'emergency_stop',
                'base_confidence': 0.9,
                'reasoning': 'High urgency situation detected',
                'risk_level': 'low',
                'estimated_duration': 0.1
            })
        
        # Add wait option for uncertain situations
        if situation_analysis.get('complexity') == 'high':
            options.append({
                'action': 'wait_and_observe',
                'base_confidence': 0.7,
                'reasoning': 'Complex situation requires observation',
                'risk_level': 'low',
                'estimated_duration': 2.0
            })
        
        return options
    
    async def _evaluate_options(self, options: List[Dict[str, Any]],
                              decision_context: DecisionContext,
                              situation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and select the best option."""
        if not options:
            return {
                'action': 'no_action',
                'base_confidence': 0.1,
                'reasoning': 'No viable options available',
                'risk_level': 'high',
                'estimated_duration': 0.0
            }
        
        # Score each option
        scored_options = []
        for option in options:
            score = await self._score_option(option, decision_context, situation_analysis)
            scored_options.append((score, option))
        
        # Sort by score (highest first)
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        return scored_options[0][1]  # Return best option
    
    async def _score_option(self, option: Dict[str, Any],
                           decision_context: DecisionContext,
                           situation_analysis: Dict[str, Any]) -> float:
        """Score an option based on various factors."""
        score = option['base_confidence']
        
        # Adjust for risk tolerance
        risk_level = option.get('risk_level', 'medium')
        risk_tolerance = self.config['risk_tolerance']
        
        if risk_tolerance == 'low' and risk_level == 'high':
            score *= 0.5
        elif risk_tolerance == 'high' and risk_level == 'low':
            score *= 1.2
        
        # Adjust for urgency
        urgency = situation_analysis.get('urgency', 'normal')
        if urgency == 'high':
            # Prefer faster actions
            duration = option.get('estimated_duration', 1.0)
            if duration < 1.0:
                score *= 1.3
            elif duration > 5.0:
                score *= 0.7
        
        # Adjust for constraints
        constraints = situation_analysis.get('constraints', [])
        if 'modal_dialog_active' in constraints:
            # Prefer actions that handle modals
            if 'modal' in option['action'].lower() or 'close' in option['action'].lower():
                score *= 1.4
        
        # Adjust for historical success
        action = option['action']
        if action in self.decision_outcomes:
            outcomes = self.decision_outcomes[action]
            success_rate = outcomes.get('success_rate', 0.5)
            score *= (0.5 + success_rate)
        
        # Adjust for pattern effectiveness
        pattern_id = option.get('pattern_id')
        if pattern_id and pattern_id in self.pattern_effectiveness:
            effectiveness = self.pattern_effectiveness[pattern_id]
            score *= (0.5 + effectiveness)
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    async def _create_decision(self, best_option: Dict[str, Any],
                             decision_context: DecisionContext,
                             situation_analysis: Dict[str, Any]) -> AutomationDecision:
        """Create the final automation decision."""
        # Calculate final confidence
        confidence = best_option['base_confidence']
        
        # Adjust confidence based on situation analysis
        confidence_factors = situation_analysis.get('confidence_factors', [])
        if 'historical_patterns' in confidence_factors:
            confidence *= 1.2
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Generate reasoning
        reasoning_parts = [best_option['reasoning']]
        
        if situation_analysis.get('patterns_matched'):
            reasoning_parts.append(f"Matched {len(situation_analysis['patterns_matched'])} historical patterns")
        
        if situation_analysis.get('risk_factors'):
            reasoning_parts.append(f"Risk factors: {', '.join(situation_analysis['risk_factors'])}")
        
        reasoning = "; ".join(reasoning_parts)
        
        # Create decision
        decision = AutomationDecision(
            decision_id=decision_context.decision_id,
            decision_type=decision_context.decision_type,
            timestamp=datetime.now(),
            recommended_action=best_option['action'],
            confidence=confidence,
            reasoning=reasoning,
            context_factors=[ctx.context_type.value for ctx in decision_context.contexts],
            execution_parameters={
                'estimated_duration': best_option.get('estimated_duration', 1.0),
                'pattern_id': best_option.get('pattern_id')
            },
            risk_level=best_option.get('risk_level', 'medium'),
            risk_factors=situation_analysis.get('risk_factors', []),
            estimated_duration=best_option.get('estimated_duration'),
            success_probability=confidence
        )
        
        return decision
    
    async def record_decision_outcome(self, decision_id: str, success: bool,
                                    actual_duration: Optional[float] = None,
                                    notes: Optional[str] = None):
        """Record the outcome of a decision for learning."""
        try:
            # Find the decision
            decision = None
            for d in self.decision_history:
                if d.decision_id == decision_id:
                    decision = d
                    break
            
            if not decision:
                self.logger.warning(f"Decision {decision_id} not found for outcome recording")
                return
            
            # Update decision outcomes
            action = decision.recommended_action
            if action not in self.decision_outcomes:
                self.decision_outcomes[action] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'success_rate': 0.0,
                    'average_duration': 0.0,
                    'durations': []
                }
            
            outcomes = self.decision_outcomes[action]
            outcomes['total_attempts'] += 1
            
            if success:
                outcomes['successful_attempts'] += 1
                self.stats['successful_decisions'] += 1
            
            outcomes['success_rate'] = outcomes['successful_attempts'] / outcomes['total_attempts']
            
            # Update duration tracking
            if actual_duration is not None:
                outcomes['durations'].append(actual_duration)
                if len(outcomes['durations']) > 100:  # Keep last 100 durations
                    outcomes['durations'] = outcomes['durations'][-100:]
                
                outcomes['average_duration'] = sum(outcomes['durations']) / len(outcomes['durations'])
            
            # Update pattern effectiveness
            pattern_id = decision.execution_parameters.get('pattern_id')
            if pattern_id:
                if pattern_id not in self.pattern_effectiveness:
                    self.pattern_effectiveness[pattern_id] = 0.5
                
                # Update with exponential moving average
                current_effectiveness = self.pattern_effectiveness[pattern_id]
                new_effectiveness = 1.0 if success else 0.0
                self.pattern_effectiveness[pattern_id] = (
                    0.8 * current_effectiveness + 0.2 * new_effectiveness
                )
            
            self.logger.debug(f"Recorded outcome for decision {decision_id}: success={success}")
            
        except Exception as e:
            self.logger.error(f"Failed to record decision outcome: {e}")
    
    async def _cleanup_expired_contexts(self):
        """Remove expired context information."""
        current_time = datetime.now()
        expired_contexts = []
        
        for context_id, context in self.active_contexts.items():
            if context.is_expired():
                expired_contexts.append(context_id)
        
        for context_id in expired_contexts:
            del self.active_contexts[context_id]
        
        if expired_contexts:
            self.logger.debug(f"Cleaned up {len(expired_contexts)} expired contexts")
    
    def get_active_contexts(self) -> List[ContextInfo]:
        """Get all active context information."""
        return list(self.active_contexts.values())
    
    def get_context_by_type(self, context_type: ContextType) -> List[ContextInfo]:
        """Get active contexts of a specific type."""
        return [ctx for ctx in self.active_contexts.values() 
                if ctx.context_type == context_type]
    
    def get_recent_decisions(self, limit: int = 10) -> List[AutomationDecision]:
        """Get recent decisions."""
        return self.decision_history[-limit:]
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision-making statistics."""
        total_decisions = self.stats['decisions_made']
        success_rate = (self.stats['successful_decisions'] / max(1, total_decisions))
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_contexts': len(self.active_contexts),
            'context_history_size': len(self.context_history),
            'decision_history_size': len(self.decision_history),
            'tracked_actions': len(self.decision_outcomes),
            'tracked_patterns': len(self.pattern_effectiveness)
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.active_contexts.clear()
        self.context_history.clear()
        self.decision_history.clear()
        self.decision_outcomes.clear()
        self.pattern_effectiveness.clear()