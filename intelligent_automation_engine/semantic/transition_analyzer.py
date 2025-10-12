"""
Transition Analyzer

Analyzes state transitions to understand application behavior patterns,
predict state changes, and optimize automation workflows.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, Counter

from .state_manager import ApplicationState, StateChange, StateType


class TransitionType(Enum):
    """Types of state transitions."""
    NAVIGATION = "navigation"
    MODAL_OPEN = "modal_open"
    MODAL_CLOSE = "modal_close"
    FORM_SUBMIT = "form_submit"
    DATA_LOAD = "data_load"
    ERROR_RECOVERY = "error_recovery"
    USER_ACTION = "user_action"
    SYSTEM_ACTION = "system_action"
    TIMEOUT = "timeout"
    REFRESH = "refresh"
    REDIRECT = "redirect"


class PatternType(Enum):
    """Types of transition patterns."""
    SEQUENTIAL = "sequential"
    CYCLICAL = "cyclical"
    BRANCHING = "branching"
    CONVERGENT = "convergent"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    WORKFLOW = "workflow"


@dataclass
class StateTransition:
    """Represents a transition between two states."""
    transition_id: str
    from_state: ApplicationState
    to_state: ApplicationState
    transition_type: TransitionType
    trigger_action: Optional[str]
    duration: float  # seconds
    timestamp: datetime
    
    # Transition properties
    success: bool = True
    error_message: Optional[str] = None
    confidence: float = 1.0
    
    # Context information
    user_initiated: bool = True
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.transition_id:
            self.transition_id = f"{self.from_state.state_id}->{self.to_state.state_id}"


@dataclass
class TransitionPattern:
    """Represents a pattern of state transitions."""
    pattern_id: str
    pattern_type: PatternType
    states: List[str]  # State IDs in order
    transitions: List[str]  # Transition IDs
    
    # Pattern metrics
    frequency: int = 0
    success_rate: float = 1.0
    average_duration: float = 0.0
    confidence: float = 1.0
    
    # Pattern properties
    is_deterministic: bool = True
    has_loops: bool = False
    max_iterations: Optional[int] = None
    
    # Conditions
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Optimization potential
    optimization_score: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    
    def get_pattern_length(self) -> int:
        """Get the length of the pattern."""
        return len(self.states)
    
    def contains_state(self, state_id: str) -> bool:
        """Check if pattern contains a specific state."""
        return state_id in self.states
    
    def get_next_state(self, current_state_id: str) -> Optional[str]:
        """Get the next state in the pattern."""
        try:
            current_index = self.states.index(current_state_id)
            if current_index < len(self.states) - 1:
                return self.states[current_index + 1]
        except ValueError:
            pass
        return None


@dataclass
class TransitionAnalysis:
    """Results of transition analysis."""
    application_name: str
    analysis_period: Tuple[datetime, datetime]
    
    # Basic statistics
    total_transitions: int
    unique_states: int
    unique_transitions: int
    
    # Patterns found
    patterns: List[TransitionPattern]
    
    # Performance metrics
    average_transition_time: float
    fastest_transition: float
    slowest_transition: float
    
    # Reliability metrics
    success_rate: float
    error_rate: float
    timeout_rate: float
    
    # Optimization insights
    optimization_opportunities: List[str]
    performance_bottlenecks: List[str]
    reliability_issues: List[str]
    
    # Predictions
    predicted_paths: Dict[str, List[str]]  # state_id -> likely next states
    transition_probabilities: Dict[str, float]  # transition_id -> probability


class TransitionAnalyzer:
    """
    Analyzes state transitions to understand application behavior and optimize automation.
    """
    
    def __init__(self, min_pattern_frequency: int = 3, 
                 pattern_confidence_threshold: float = 0.7):
        """Initialize the transition analyzer."""
        self.logger = logging.getLogger(__name__)
        self.min_pattern_frequency = min_pattern_frequency
        self.pattern_confidence_threshold = pattern_confidence_threshold
        
        # Analysis data
        self.transitions: List[StateTransition] = []
        self.patterns: List[TransitionPattern] = []
        
        # Caches
        self.transition_cache: Dict[str, List[StateTransition]] = {}
        self.pattern_cache: Dict[str, List[TransitionPattern]] = {}
        
        # Statistics
        self.stats = {
            'transitions_analyzed': 0,
            'patterns_discovered': 0,
            'predictions_made': 0,
            'prediction_accuracy': 0.0
        }
    
    async def analyze_transitions(self, state_changes: List[StateChange],
                                application_name: Optional[str] = None) -> TransitionAnalysis:
        """
        Analyze a list of state changes to identify patterns and insights.
        
        Args:
            state_changes: List of state changes to analyze
            application_name: Optional filter for specific application
            
        Returns:
            TransitionAnalysis: Complete analysis results
        """
        try:
            self.logger.info(f"Analyzing {len(state_changes)} state changes")
            
            # Filter by application if specified
            if application_name:
                state_changes = [sc for sc in state_changes 
                               if sc.new_state.application_name == application_name]
            
            # Convert state changes to transitions
            transitions = await self._convert_to_transitions(state_changes)
            self.transitions.extend(transitions)
            
            # Analyze patterns
            patterns = await self._discover_patterns(transitions)
            self.patterns.extend(patterns)
            
            # Calculate metrics
            analysis = await self._calculate_analysis_metrics(
                transitions, patterns, application_name or "all"
            )
            
            # Update statistics
            self.stats['transitions_analyzed'] += len(transitions)
            self.stats['patterns_discovered'] += len(patterns)
            
            self.logger.info(f"Analysis completed: {len(patterns)} patterns discovered")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Transition analysis failed: {e}")
            raise
    
    async def _convert_to_transitions(self, state_changes: List[StateChange]) -> List[StateTransition]:
        """Convert state changes to transitions."""
        transitions = []
        
        for i, change in enumerate(state_changes):
            if change.previous_state is None:
                continue
            
            # Calculate duration
            if i < len(state_changes) - 1:
                next_change = state_changes[i + 1]
                duration = (next_change.timestamp - change.timestamp).total_seconds()
            else:
                duration = 0.0
            
            # Determine transition type
            transition_type = self._classify_transition(change)
            
            transition = StateTransition(
                transition_id="",
                from_state=change.previous_state,
                to_state=change.new_state,
                transition_type=transition_type,
                trigger_action=change.trigger_action,
                duration=duration,
                timestamp=change.timestamp,
                user_initiated=change.trigger_action is not None
            )
            
            transitions.append(transition)
        
        return transitions
    
    def _classify_transition(self, state_change: StateChange) -> TransitionType:
        """Classify the type of transition based on state change."""
        from_state = state_change.previous_state
        to_state = state_change.new_state
        
        if not from_state:
            return TransitionType.SYSTEM_ACTION
        
        # Modal transitions
        if to_state.state_type == StateType.MODAL and from_state.state_type != StateType.MODAL:
            return TransitionType.MODAL_OPEN
        elif from_state.state_type == StateType.MODAL and to_state.state_type != StateType.MODAL:
            return TransitionType.MODAL_CLOSE
        
        # Error states
        if to_state.state_type == StateType.ERROR:
            return TransitionType.ERROR_RECOVERY
        
        # Loading states
        if to_state.state_type == StateType.LOADING:
            return TransitionType.DATA_LOAD
        
        # Form submissions
        if (from_state.state_type == StateType.FORM and 
            to_state.state_type != StateType.FORM and
            state_change.trigger_action and 'submit' in state_change.trigger_action.lower()):
            return TransitionType.FORM_SUBMIT
        
        # Navigation
        if from_state.window_title != to_state.window_title:
            return TransitionType.NAVIGATION
        
        # Default to user action
        if state_change.trigger_action:
            return TransitionType.USER_ACTION
        else:
            return TransitionType.SYSTEM_ACTION
    
    async def _discover_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Discover patterns in transitions."""
        patterns = []
        
        # Group transitions by application
        app_transitions = defaultdict(list)
        for transition in transitions:
            app_name = transition.from_state.application_name
            app_transitions[app_name].append(transition)
        
        # Analyze patterns for each application
        for app_name, app_trans in app_transitions.items():
            app_patterns = await self._find_app_patterns(app_trans)
            patterns.extend(app_patterns)
        
        return patterns
    
    async def _find_app_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Find patterns within an application's transitions."""
        patterns = []
        
        # Sequential patterns
        sequential_patterns = await self._find_sequential_patterns(transitions)
        patterns.extend(sequential_patterns)
        
        # Cyclical patterns
        cyclical_patterns = await self._find_cyclical_patterns(transitions)
        patterns.extend(cyclical_patterns)
        
        # Branching patterns
        branching_patterns = await self._find_branching_patterns(transitions)
        patterns.extend(branching_patterns)
        
        # Loop patterns
        loop_patterns = await self._find_loop_patterns(transitions)
        patterns.extend(loop_patterns)
        
        return patterns
    
    async def _find_sequential_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Find sequential patterns in transitions."""
        patterns = []
        sequence_counts = defaultdict(int)
        
        # Look for sequences of different lengths
        for seq_length in range(2, min(6, len(transitions) + 1)):
            for i in range(len(transitions) - seq_length + 1):
                sequence = transitions[i:i + seq_length]
                state_sequence = [t.from_state.state_id for t in sequence]
                state_sequence.append(sequence[-1].to_state.state_id)
                
                seq_key = tuple(state_sequence)
                sequence_counts[seq_key] += 1
        
        # Create patterns for frequent sequences
        for seq_states, frequency in sequence_counts.items():
            if frequency >= self.min_pattern_frequency:
                pattern = TransitionPattern(
                    pattern_id=f"seq_{hash(seq_states)}",
                    pattern_type=PatternType.SEQUENTIAL,
                    states=list(seq_states),
                    transitions=[],
                    frequency=frequency,
                    confidence=min(1.0, frequency / len(transitions))
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _find_cyclical_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Find cyclical patterns in transitions."""
        patterns = []
        state_visits = defaultdict(list)
        
        # Track when each state is visited
        for i, transition in enumerate(transitions):
            state_visits[transition.from_state.state_id].append(i)
            state_visits[transition.to_state.state_id].append(i)
        
        # Look for states that are revisited regularly
        for state_id, visit_indices in state_visits.items():
            if len(visit_indices) >= self.min_pattern_frequency:
                # Calculate intervals between visits
                intervals = [visit_indices[i+1] - visit_indices[i] 
                           for i in range(len(visit_indices) - 1)]
                
                if intervals:
                    avg_interval = statistics.mean(intervals)
                    interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
                    
                    # If intervals are relatively consistent, it's a cycle
                    if interval_std < avg_interval * 0.3:  # 30% variation threshold
                        pattern = TransitionPattern(
                            pattern_id=f"cycle_{state_id}",
                            pattern_type=PatternType.CYCLICAL,
                            states=[state_id],
                            transitions=[],
                            frequency=len(visit_indices),
                            confidence=1.0 - (interval_std / max(avg_interval, 1)),
                            has_loops=True
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _find_branching_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Find branching patterns in transitions."""
        patterns = []
        state_outcomes = defaultdict(lambda: defaultdict(int))
        
        # Track outcomes from each state
        for transition in transitions:
            from_state = transition.from_state.state_id
            to_state = transition.to_state.state_id
            state_outcomes[from_state][to_state] += 1
        
        # Find states with multiple outcomes
        for from_state, outcomes in state_outcomes.items():
            if len(outcomes) >= 2:  # At least 2 different outcomes
                total_transitions = sum(outcomes.values())
                
                if total_transitions >= self.min_pattern_frequency:
                    # Calculate entropy to measure branching
                    entropy = 0
                    for count in outcomes.values():
                        prob = count / total_transitions
                        if prob > 0:
                            entropy -= prob * (prob ** 0.5)  # Simplified entropy
                    
                    pattern = TransitionPattern(
                        pattern_id=f"branch_{from_state}",
                        pattern_type=PatternType.BRANCHING,
                        states=[from_state] + list(outcomes.keys()),
                        transitions=[],
                        frequency=total_transitions,
                        confidence=entropy,  # Higher entropy = more branching
                        is_deterministic=False
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _find_loop_patterns(self, transitions: List[StateTransition]) -> List[TransitionPattern]:
        """Find loop patterns in transitions."""
        patterns = []
        
        # Look for immediate loops (state -> state)
        for transition in transitions:
            if transition.from_state.state_id == transition.to_state.state_id:
                # This is a self-loop
                pattern = TransitionPattern(
                    pattern_id=f"loop_{transition.from_state.state_id}",
                    pattern_type=PatternType.LOOP,
                    states=[transition.from_state.state_id],
                    transitions=[transition.transition_id],
                    frequency=1,
                    confidence=1.0,
                    has_loops=True,
                    max_iterations=1
                )
                patterns.append(pattern)
        
        # Look for longer loops (A -> B -> A)
        state_sequences = []
        for i in range(len(transitions) - 1):
            state_sequences.append([
                transitions[i].from_state.state_id,
                transitions[i].to_state.state_id,
                transitions[i+1].to_state.state_id
            ])
        
        # Find sequences that return to start
        loop_counts = defaultdict(int)
        for seq in state_sequences:
            if len(seq) >= 3 and seq[0] == seq[-1]:
                loop_key = tuple(seq)
                loop_counts[loop_key] += 1
        
        for loop_states, frequency in loop_counts.items():
            if frequency >= self.min_pattern_frequency:
                pattern = TransitionPattern(
                    pattern_id=f"loop_{hash(loop_states)}",
                    pattern_type=PatternType.LOOP,
                    states=list(loop_states),
                    transitions=[],
                    frequency=frequency,
                    confidence=min(1.0, frequency / len(state_sequences)),
                    has_loops=True
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _calculate_analysis_metrics(self, transitions: List[StateTransition],
                                        patterns: List[TransitionPattern],
                                        application_name: str) -> TransitionAnalysis:
        """Calculate comprehensive analysis metrics."""
        if not transitions:
            return TransitionAnalysis(
                application_name=application_name,
                analysis_period=(datetime.now(), datetime.now()),
                total_transitions=0,
                unique_states=0,
                unique_transitions=0,
                patterns=[],
                average_transition_time=0.0,
                fastest_transition=0.0,
                slowest_transition=0.0,
                success_rate=1.0,
                error_rate=0.0,
                timeout_rate=0.0,
                optimization_opportunities=[],
                performance_bottlenecks=[],
                reliability_issues=[],
                predicted_paths={},
                transition_probabilities={}
            )
        
        # Basic statistics
        total_transitions = len(transitions)
        unique_states = len(set(t.from_state.state_id for t in transitions) |
                          set(t.to_state.state_id for t in transitions))
        unique_transitions = len(set(t.transition_id for t in transitions))
        
        # Time analysis
        durations = [t.duration for t in transitions if t.duration > 0]
        if durations:
            average_transition_time = statistics.mean(durations)
            fastest_transition = min(durations)
            slowest_transition = max(durations)
        else:
            average_transition_time = fastest_transition = slowest_transition = 0.0
        
        # Success analysis
        successful_transitions = sum(1 for t in transitions if t.success)
        success_rate = successful_transitions / total_transitions
        error_rate = sum(1 for t in transitions if not t.success) / total_transitions
        timeout_rate = sum(1 for t in transitions if t.duration > 30) / total_transitions
        
        # Analysis period
        timestamps = [t.timestamp for t in transitions]
        analysis_period = (min(timestamps), max(timestamps))
        
        # Generate insights
        optimization_opportunities = await self._identify_optimization_opportunities(transitions, patterns)
        performance_bottlenecks = await self._identify_performance_bottlenecks(transitions)
        reliability_issues = await self._identify_reliability_issues(transitions)
        
        # Generate predictions
        predicted_paths = await self._generate_predicted_paths(transitions)
        transition_probabilities = await self._calculate_transition_probabilities(transitions)
        
        return TransitionAnalysis(
            application_name=application_name,
            analysis_period=analysis_period,
            total_transitions=total_transitions,
            unique_states=unique_states,
            unique_transitions=unique_transitions,
            patterns=patterns,
            average_transition_time=average_transition_time,
            fastest_transition=fastest_transition,
            slowest_transition=slowest_transition,
            success_rate=success_rate,
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            optimization_opportunities=optimization_opportunities,
            performance_bottlenecks=performance_bottlenecks,
            reliability_issues=reliability_issues,
            predicted_paths=predicted_paths,
            transition_probabilities=transition_probabilities
        )
    
    async def _identify_optimization_opportunities(self, transitions: List[StateTransition],
                                                 patterns: List[TransitionPattern]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Look for slow transitions
        durations = [t.duration for t in transitions if t.duration > 0]
        if durations:
            avg_duration = statistics.mean(durations)
            slow_transitions = [t for t in transitions if t.duration > avg_duration * 2]
            
            if slow_transitions:
                opportunities.append(f"Optimize {len(slow_transitions)} slow transitions")
        
        # Look for repetitive patterns
        repetitive_patterns = [p for p in patterns if p.frequency > 10]
        if repetitive_patterns:
            opportunities.append(f"Automate {len(repetitive_patterns)} repetitive patterns")
        
        # Look for error-prone transitions
        error_transitions = [t for t in transitions if not t.success]
        if error_transitions:
            opportunities.append(f"Improve reliability of {len(error_transitions)} error-prone transitions")
        
        return opportunities
    
    async def _identify_performance_bottlenecks(self, transitions: List[StateTransition]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Find slowest transitions
        durations = [t.duration for t in transitions if t.duration > 0]
        if durations:
            avg_duration = statistics.mean(durations)
            threshold = avg_duration * 3  # 3x average
            
            slow_transitions = [t for t in transitions if t.duration > threshold]
            for transition in slow_transitions[:5]:  # Top 5
                bottlenecks.append(f"Slow transition: {transition.from_state.state_id} -> {transition.to_state.state_id} ({transition.duration:.1f}s)")
        
        return bottlenecks
    
    async def _identify_reliability_issues(self, transitions: List[StateTransition]) -> List[str]:
        """Identify reliability issues."""
        issues = []
        
        # Find transitions with high error rates
        transition_errors = defaultdict(list)
        for transition in transitions:
            key = f"{transition.from_state.state_id}->{transition.to_state.state_id}"
            transition_errors[key].append(transition.success)
        
        for transition_key, successes in transition_errors.items():
            if len(successes) >= 3:  # At least 3 attempts
                success_rate = sum(successes) / len(successes)
                if success_rate < 0.8:  # Less than 80% success
                    issues.append(f"Low success rate for {transition_key}: {success_rate:.1%}")
        
        return issues
    
    async def _generate_predicted_paths(self, transitions: List[StateTransition]) -> Dict[str, List[str]]:
        """Generate predicted paths from each state."""
        predictions = defaultdict(lambda: defaultdict(int))
        
        # Count transitions from each state
        for transition in transitions:
            from_state = transition.from_state.state_id
            to_state = transition.to_state.state_id
            predictions[from_state][to_state] += 1
        
        # Convert to sorted lists of likely next states
        predicted_paths = {}
        for from_state, outcomes in predictions.items():
            sorted_outcomes = sorted(outcomes.items(), key=lambda x: x[1], reverse=True)
            predicted_paths[from_state] = [state for state, _ in sorted_outcomes[:3]]  # Top 3
        
        return predicted_paths
    
    async def _calculate_transition_probabilities(self, transitions: List[StateTransition]) -> Dict[str, float]:
        """Calculate transition probabilities."""
        transition_counts = defaultdict(int)
        state_counts = defaultdict(int)
        
        # Count transitions and states
        for transition in transitions:
            from_state = transition.from_state.state_id
            transition_key = f"{from_state}->{transition.to_state.state_id}"
            
            transition_counts[transition_key] += 1
            state_counts[from_state] += 1
        
        # Calculate probabilities
        probabilities = {}
        for transition_key, count in transition_counts.items():
            from_state = transition_key.split('->')[0]
            probability = count / state_counts[from_state]
            probabilities[transition_key] = probability
        
        return probabilities
    
    def predict_next_states(self, current_state_id: str, 
                           top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict the most likely next states."""
        predictions = []
        
        # Look through recent transitions
        for transition in reversed(self.transitions[-100:]):  # Last 100 transitions
            if transition.from_state.state_id == current_state_id:
                predictions.append((transition.to_state.state_id, transition.confidence))
        
        # Count and sort predictions
        state_counts = defaultdict(float)
        for state_id, confidence in predictions:
            state_counts[state_id] += confidence
        
        # Sort by count and return top k
        sorted_predictions = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[TransitionPattern]:
        """Get a pattern by its ID."""
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None
    
    def get_patterns_for_state(self, state_id: str) -> List[TransitionPattern]:
        """Get all patterns that include a specific state."""
        return [pattern for pattern in self.patterns if pattern.contains_state(state_id)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear analysis caches."""
        self.transition_cache.clear()
        self.pattern_cache.clear()
    
    async def cleanup(self):
        """Cleanup resources."""
        self.transitions.clear()
        self.patterns.clear()
        self.clear_cache()