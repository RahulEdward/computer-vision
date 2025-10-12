"""
Loop Detector for Automation Pattern Analysis

This module detects repetitive patterns, loops, and cycles in automation sequences
to identify optimization opportunities and improve workflow efficiency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Tuple, Callable
from datetime import datetime, timedelta
import uuid
import logging
import hashlib
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)


class LoopType(Enum):
    """Types of loops that can be detected"""
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    REPEAT_UNTIL = "repeat_until"
    INFINITE_LOOP = "infinite_loop"
    NESTED_LOOP = "nested_loop"
    CONDITIONAL_LOOP = "conditional_loop"
    RECURSIVE_PATTERN = "recursive_pattern"
    CYCLIC_PATTERN = "cyclic_pattern"


class PatternType(Enum):
    """Types of patterns that can be detected"""
    SEQUENTIAL = "sequential"
    REPETITIVE = "repetitive"
    ALTERNATING = "alternating"
    BRANCHING = "branching"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    CONDITIONAL = "conditional"


class DetectionStrategy(Enum):
    """Strategies for pattern detection"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    STRUCTURAL_MATCH = "structural_match"
    TEMPORAL_MATCH = "temporal_match"
    STATISTICAL_MATCH = "statistical_match"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


@dataclass
class ActionStep:
    """Represents a single action step in automation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""
    target: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta = timedelta(0)
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    success: bool = True
    error_message: str = ""
    retry_count: int = 0
    
    def __hash__(self):
        """Generate hash for pattern matching"""
        key_data = f"{self.action_type}:{self.target}:{sorted(self.parameters.items())}"
        return hash(key_data)
    
    def __eq__(self, other):
        """Check equality for pattern matching"""
        if not isinstance(other, ActionStep):
            return False
        return (self.action_type == other.action_type and
                self.target == other.target and
                self.parameters == other.parameters)


@dataclass
class Pattern:
    """Represents a detected pattern in automation sequence"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: PatternType = PatternType.SEQUENTIAL
    
    # Pattern structure
    steps: List[ActionStep] = field(default_factory=list)
    length: int = 0
    frequency: int = 1
    
    # Pattern characteristics
    is_exact: bool = True
    similarity_score: float = 1.0
    confidence: float = 0.0
    
    # Occurrence information
    first_occurrence: int = 0
    last_occurrence: int = 0
    occurrences: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) positions
    
    # Performance metrics
    average_duration: timedelta = timedelta(0)
    success_rate: float = 1.0
    error_patterns: List[str] = field(default_factory=list)
    
    # Optimization potential
    optimization_score: float = 0.0
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    detection_strategy: DetectionStrategy = DetectionStrategy.EXACT_MATCH
    tags: List[str] = field(default_factory=list)


@dataclass
class Loop:
    """Represents a detected loop pattern"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: LoopType = LoopType.FOR_LOOP
    
    # Loop structure
    body_pattern: Pattern = field(default_factory=Pattern)
    condition: str = ""
    iteration_count: int = 0
    
    # Loop boundaries
    start_position: int = 0
    end_position: int = 0
    nested_loops: List['Loop'] = field(default_factory=list)
    
    # Loop characteristics
    is_infinite: bool = False
    has_break_condition: bool = False
    break_conditions: List[str] = field(default_factory=list)
    
    # Performance analysis
    total_iterations: int = 0
    average_iteration_time: timedelta = timedelta(0)
    efficiency_score: float = 0.0
    
    # Optimization opportunities
    can_parallelize: bool = False
    can_vectorize: bool = False
    can_cache: bool = False
    optimization_potential: float = 0.0
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Result of pattern/loop detection analysis"""
    sequence_id: str = ""
    
    # Detection results
    patterns: List[Pattern] = field(default_factory=list)
    loops: List[Loop] = field(default_factory=list)
    
    # Analysis statistics
    total_steps: int = 0
    unique_patterns: int = 0
    repetitive_patterns: int = 0
    optimization_opportunities: int = 0
    
    # Performance metrics
    detection_time: timedelta = timedelta(0)
    coverage_percentage: float = 0.0
    confidence_score: float = 0.0
    
    # Recommendations
    optimization_suggestions: List[str] = field(default_factory=list)
    performance_improvements: List[str] = field(default_factory=list)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    strategy_used: DetectionStrategy = DetectionStrategy.EXACT_MATCH
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopContext:
    """Context information for loop detection"""
    sequence_id: str = ""
    
    # Detection parameters
    min_pattern_length: int = 2
    max_pattern_length: int = 50
    min_frequency: int = 2
    similarity_threshold: float = 0.8
    
    # Analysis scope
    analyze_nested_loops: bool = True
    detect_infinite_loops: bool = True
    include_conditional_patterns: bool = True
    
    # Performance settings
    max_analysis_time: timedelta = timedelta(minutes=5)
    enable_caching: bool = True
    parallel_processing: bool = False
    
    # Filtering options
    ignore_single_actions: bool = True
    filter_by_success_rate: bool = True
    min_success_rate: float = 0.7
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


class LoopDetector:
    """Detector for identifying loops and patterns in automation sequences"""
    
    def __init__(self):
        # Detection strategies
        self.detection_strategies = {
            DetectionStrategy.EXACT_MATCH: self._detect_exact_patterns,
            DetectionStrategy.FUZZY_MATCH: self._detect_fuzzy_patterns,
            DetectionStrategy.SEMANTIC_MATCH: self._detect_semantic_patterns,
            DetectionStrategy.STRUCTURAL_MATCH: self._detect_structural_patterns,
            DetectionStrategy.TEMPORAL_MATCH: self._detect_temporal_patterns,
            DetectionStrategy.STATISTICAL_MATCH: self._detect_statistical_patterns,
            DetectionStrategy.MACHINE_LEARNING: self._detect_ml_patterns,
            DetectionStrategy.HYBRID: self._detect_hybrid_patterns
        }
        
        # Pattern cache
        self.pattern_cache: Dict[str, List[Pattern]] = {}
        self.loop_cache: Dict[str, List[Loop]] = {}
        
        # Analysis history
        self.detection_history: List[DetectionResult] = []
        self.performance_stats = {
            'total_detections': 0,
            'patterns_found': 0,
            'loops_found': 0,
            'average_detection_time': timedelta(0)
        }
        
        # Configuration
        self.default_context = LoopContext()
        self.enable_optimization_analysis = True
        self.cache_results = True
        
        logger.info("Loop detector initialized")
    
    def detect_patterns(self, sequence: List[ActionStep], 
                       context: Optional[LoopContext] = None,
                       strategy: DetectionStrategy = DetectionStrategy.HYBRID) -> DetectionResult:
        """Detect patterns in an automation sequence"""
        try:
            start_time = datetime.now()
            context = context or self.default_context
            
            # Check cache first
            if self.cache_results:
                cache_key = self._generate_cache_key(sequence, context, strategy)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
            
            # Initialize result
            result = DetectionResult(
                sequence_id=context.sequence_id,
                total_steps=len(sequence),
                strategy_used=strategy
            )
            
            # Detect patterns using specified strategy
            detection_func = self.detection_strategies.get(strategy)
            if not detection_func:
                logger.error(f"Unknown detection strategy: {strategy}")
                return result
            
            patterns = detection_func(sequence, context)
            result.patterns = patterns
            result.unique_patterns = len(set(p.id for p in patterns))
            result.repetitive_patterns = len([p for p in patterns if p.frequency > 1])
            
            # Detect loops from patterns
            loops = self._detect_loops_from_patterns(patterns, sequence, context)
            result.loops = loops
            
            # Calculate metrics
            result.coverage_percentage = self._calculate_coverage(patterns, len(sequence))
            result.confidence_score = self._calculate_confidence(patterns, loops)
            result.detection_time = datetime.now() - start_time
            
            # Generate optimization suggestions
            if self.enable_optimization_analysis:
                result.optimization_suggestions = self._generate_optimization_suggestions(patterns, loops)
                result.performance_improvements = self._generate_performance_improvements(patterns, loops)
                result.optimization_opportunities = len(result.optimization_suggestions)
            
            # Cache result
            if self.cache_results:
                self._cache_result(cache_key, result)
            
            # Update statistics
            self._update_statistics(result)
            
            # Store in history
            self.detection_history.append(result)
            
            logger.info(f"Pattern detection completed: {len(patterns)} patterns, {len(loops)} loops found")
            return result
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return DetectionResult(sequence_id=context.sequence_id if context else "")
    
    def detect_loops(self, sequence: List[ActionStep], 
                    context: Optional[LoopContext] = None) -> List[Loop]:
        """Detect loops specifically in an automation sequence"""
        try:
            context = context or self.default_context
            
            # First detect patterns
            result = self.detect_patterns(sequence, context, DetectionStrategy.HYBRID)
            
            # Extract loops from result
            return result.loops
            
        except Exception as e:
            logger.error(f"Loop detection failed: {e}")
            return []
    
    def analyze_sequence_efficiency(self, sequence: List[ActionStep]) -> Dict[str, Any]:
        """Analyze the efficiency of an automation sequence"""
        try:
            # Detect patterns and loops
            result = self.detect_patterns(sequence)
            
            # Calculate efficiency metrics
            total_steps = len(sequence)
            repetitive_steps = sum(p.frequency * p.length for p in result.patterns if p.frequency > 1)
            efficiency_ratio = 1.0 - (repetitive_steps / total_steps) if total_steps > 0 else 0.0
            
            # Identify optimization opportunities
            loop_optimization_potential = sum(l.optimization_potential for l in result.loops)
            pattern_optimization_potential = sum(p.optimization_score for p in result.patterns)
            
            # Calculate time savings potential
            total_duration = sum((step.duration for step in sequence), timedelta(0))
            potential_savings = self._calculate_time_savings(result.patterns, result.loops)
            
            return {
                'total_steps': total_steps,
                'unique_patterns': result.unique_patterns,
                'repetitive_patterns': result.repetitive_patterns,
                'loops_detected': len(result.loops),
                'efficiency_ratio': efficiency_ratio,
                'optimization_potential': loop_optimization_potential + pattern_optimization_potential,
                'total_duration': total_duration,
                'potential_time_savings': potential_savings,
                'optimization_suggestions': result.optimization_suggestions,
                'confidence_score': result.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Efficiency analysis failed: {e}")
            return {}
    
    def get_optimization_recommendations(self, sequence: List[ActionStep]) -> List[Dict[str, Any]]:
        """Get specific optimization recommendations for a sequence"""
        try:
            result = self.detect_patterns(sequence)
            recommendations = []
            
            # Loop optimization recommendations
            for loop in result.loops:
                if loop.optimization_potential > 0.5:
                    rec = {
                        'type': 'loop_optimization',
                        'loop_id': loop.id,
                        'loop_type': loop.type.value,
                        'potential': loop.optimization_potential,
                        'suggestions': []
                    }
                    
                    if loop.can_parallelize:
                        rec['suggestions'].append('Consider parallelizing loop iterations')
                    if loop.can_vectorize:
                        rec['suggestions'].append('Consider vectorizing operations')
                    if loop.can_cache:
                        rec['suggestions'].append('Consider caching repeated calculations')
                    
                    recommendations.append(rec)
            
            # Pattern optimization recommendations
            for pattern in result.patterns:
                if pattern.optimization_score > 0.5 and pattern.frequency > 2:
                    rec = {
                        'type': 'pattern_optimization',
                        'pattern_id': pattern.id,
                        'pattern_type': pattern.type.value,
                        'frequency': pattern.frequency,
                        'optimization_score': pattern.optimization_score,
                        'suggestions': pattern.optimization_suggestions
                    }
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    # Detection strategy implementations
    def _detect_exact_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect exact matching patterns"""
        patterns = []
        
        try:
            # Use sliding window approach
            for length in range(context.min_pattern_length, min(context.max_pattern_length + 1, len(sequence))):
                pattern_counts = defaultdict(list)
                
                # Find all subsequences of this length
                for i in range(len(sequence) - length + 1):
                    subseq = sequence[i:i + length]
                    pattern_hash = self._hash_sequence(subseq)
                    pattern_counts[pattern_hash].append((i, subseq))
                
                # Identify patterns that occur multiple times
                for pattern_hash, occurrences in pattern_counts.items():
                    if len(occurrences) >= context.min_frequency:
                        pattern = Pattern(
                            type=PatternType.REPETITIVE,
                            steps=occurrences[0][1],
                            length=length,
                            frequency=len(occurrences),
                            is_exact=True,
                            similarity_score=1.0,
                            confidence=1.0,
                            first_occurrence=occurrences[0][0],
                            last_occurrence=occurrences[-1][0],
                            occurrences=[(occ[0], occ[0] + length - 1) for occ in occurrences],
                            detection_strategy=DetectionStrategy.EXACT_MATCH
                        )
                        
                        # Calculate performance metrics
                        self._calculate_pattern_metrics(pattern, sequence)
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Exact pattern detection failed: {e}")
        
        return patterns
    
    def _detect_fuzzy_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect fuzzy matching patterns with similarity threshold"""
        patterns = []
        
        try:
            # Use approximate string matching techniques
            for length in range(context.min_pattern_length, min(context.max_pattern_length + 1, len(sequence))):
                for i in range(len(sequence) - length + 1):
                    candidate = sequence[i:i + length]
                    similar_patterns = []
                    
                    # Find similar patterns
                    for j in range(i + length, len(sequence) - length + 1):
                        other = sequence[j:j + length]
                        similarity = self._calculate_sequence_similarity(candidate, other)
                        
                        if similarity >= context.similarity_threshold:
                            similar_patterns.append((j, other, similarity))
                    
                    # Create pattern if enough similar occurrences found
                    if len(similar_patterns) >= context.min_frequency - 1:
                        pattern = Pattern(
                            type=PatternType.REPETITIVE,
                            steps=candidate,
                            length=length,
                            frequency=len(similar_patterns) + 1,
                            is_exact=False,
                            similarity_score=sum(sim for _, _, sim in similar_patterns) / len(similar_patterns),
                            confidence=0.8,
                            first_occurrence=i,
                            last_occurrence=similar_patterns[-1][0] if similar_patterns else i,
                            occurrences=[(i, i + length - 1)] + [(pos, pos + length - 1) for pos, _, _ in similar_patterns],
                            detection_strategy=DetectionStrategy.FUZZY_MATCH
                        )
                        
                        self._calculate_pattern_metrics(pattern, sequence)
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Fuzzy pattern detection failed: {e}")
        
        return patterns
    
    def _detect_semantic_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect semantically similar patterns"""
        patterns = []
        
        try:
            # Group actions by semantic similarity
            semantic_groups = self._group_by_semantic_similarity(sequence)
            
            # Find repetitive semantic patterns
            for group_key, actions in semantic_groups.items():
                if len(actions) >= context.min_frequency:
                    # Create semantic pattern
                    pattern = Pattern(
                        type=PatternType.REPETITIVE,
                        steps=[action for _, action in actions],
                        length=1,  # Semantic patterns are typically single actions
                        frequency=len(actions),
                        is_exact=False,
                        similarity_score=0.9,  # High semantic similarity
                        confidence=0.7,
                        first_occurrence=actions[0][0],
                        last_occurrence=actions[-1][0],
                        occurrences=[(pos, pos) for pos, _ in actions],
                        detection_strategy=DetectionStrategy.SEMANTIC_MATCH,
                        tags=['semantic']
                    )
                    
                    self._calculate_pattern_metrics(pattern, sequence)
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Semantic pattern detection failed: {e}")
        
        return patterns
    
    def _detect_structural_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect structural patterns based on action relationships"""
        patterns = []
        
        try:
            # Analyze structural relationships
            structure_graph = self._build_structure_graph(sequence)
            
            # Find repetitive structural patterns
            structural_patterns = self._find_structural_repetitions(structure_graph, context)
            
            for struct_pattern in structural_patterns:
                pattern = Pattern(
                    type=PatternType.STRUCTURAL,
                    steps=struct_pattern['steps'],
                    length=len(struct_pattern['steps']),
                    frequency=struct_pattern['frequency'],
                    is_exact=False,
                    similarity_score=struct_pattern['similarity'],
                    confidence=0.6,
                    occurrences=struct_pattern['occurrences'],
                    detection_strategy=DetectionStrategy.STRUCTURAL_MATCH,
                    tags=['structural']
                )
                
                self._calculate_pattern_metrics(pattern, sequence)
                patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Structural pattern detection failed: {e}")
        
        return patterns
    
    def _detect_temporal_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect temporal patterns based on timing"""
        patterns = []
        
        try:
            # Analyze timing patterns
            time_intervals = [step.timestamp for step in sequence]
            
            # Find regular time intervals
            regular_intervals = self._find_regular_intervals(time_intervals, context)
            
            for interval_pattern in regular_intervals:
                pattern = Pattern(
                    type=PatternType.TEMPORAL,
                    steps=interval_pattern['steps'],
                    length=len(interval_pattern['steps']),
                    frequency=interval_pattern['frequency'],
                    is_exact=False,
                    similarity_score=interval_pattern['regularity'],
                    confidence=0.5,
                    occurrences=interval_pattern['occurrences'],
                    detection_strategy=DetectionStrategy.TEMPORAL_MATCH,
                    tags=['temporal']
                )
                
                self._calculate_pattern_metrics(pattern, sequence)
                patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Temporal pattern detection failed: {e}")
        
        return patterns
    
    def _detect_statistical_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect patterns using statistical analysis"""
        patterns = []
        
        try:
            # Statistical analysis of action frequencies
            action_frequencies = self._calculate_action_frequencies(sequence)
            
            # Find statistically significant patterns
            significant_patterns = self._find_statistical_patterns(action_frequencies, sequence, context)
            
            for stat_pattern in significant_patterns:
                pattern = Pattern(
                    type=PatternType.REPETITIVE,
                    steps=stat_pattern['steps'],
                    length=len(stat_pattern['steps']),
                    frequency=stat_pattern['frequency'],
                    is_exact=False,
                    similarity_score=stat_pattern['significance'],
                    confidence=stat_pattern['confidence'],
                    occurrences=stat_pattern['occurrences'],
                    detection_strategy=DetectionStrategy.STATISTICAL_MATCH,
                    tags=['statistical']
                )
                
                self._calculate_pattern_metrics(pattern, sequence)
                patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Statistical pattern detection failed: {e}")
        
        return patterns
    
    def _detect_ml_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect patterns using machine learning techniques"""
        patterns = []
        
        try:
            # Simplified ML-based pattern detection
            # In a real implementation, this would use trained models
            
            # Feature extraction
            features = self._extract_sequence_features(sequence)
            
            # Clustering similar sequences
            clusters = self._cluster_sequences(features, context)
            
            # Convert clusters to patterns
            for cluster in clusters:
                if len(cluster['members']) >= context.min_frequency:
                    pattern = Pattern(
                        type=PatternType.REPETITIVE,
                        steps=cluster['representative'],
                        length=len(cluster['representative']),
                        frequency=len(cluster['members']),
                        is_exact=False,
                        similarity_score=cluster['cohesion'],
                        confidence=cluster['confidence'],
                        occurrences=cluster['occurrences'],
                        detection_strategy=DetectionStrategy.MACHINE_LEARNING,
                        tags=['ml', 'clustered']
                    )
                    
                    self._calculate_pattern_metrics(pattern, sequence)
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"ML pattern detection failed: {e}")
        
        return patterns
    
    def _detect_hybrid_patterns(self, sequence: List[ActionStep], context: LoopContext) -> List[Pattern]:
        """Detect patterns using hybrid approach combining multiple strategies"""
        all_patterns = []
        
        try:
            # Run multiple detection strategies
            strategies = [
                DetectionStrategy.EXACT_MATCH,
                DetectionStrategy.FUZZY_MATCH,
                DetectionStrategy.SEMANTIC_MATCH,
                DetectionStrategy.STRUCTURAL_MATCH
            ]
            
            for strategy in strategies:
                detection_func = self.detection_strategies.get(strategy)
                if detection_func and strategy != DetectionStrategy.HYBRID:
                    patterns = detection_func(sequence, context)
                    all_patterns.extend(patterns)
            
            # Merge and deduplicate patterns
            merged_patterns = self._merge_similar_patterns(all_patterns, context)
            
            # Rank patterns by confidence and relevance
            ranked_patterns = self._rank_patterns(merged_patterns)
            
            return ranked_patterns
            
        except Exception as e:
            logger.error(f"Hybrid pattern detection failed: {e}")
            return []
    
    def _detect_loops_from_patterns(self, patterns: List[Pattern], 
                                   sequence: List[ActionStep], 
                                   context: LoopContext) -> List[Loop]:
        """Detect loops from identified patterns"""
        loops = []
        
        try:
            for pattern in patterns:
                if pattern.frequency >= 2:  # Potential loop
                    # Analyze if pattern forms a loop
                    loop_analysis = self._analyze_loop_structure(pattern, sequence)
                    
                    if loop_analysis['is_loop']:
                        loop = Loop(
                            type=loop_analysis['loop_type'],
                            body_pattern=pattern,
                            condition=loop_analysis.get('condition', ''),
                            iteration_count=pattern.frequency,
                            start_position=pattern.first_occurrence,
                            end_position=pattern.last_occurrence + pattern.length - 1,
                            is_infinite=loop_analysis.get('is_infinite', False),
                            has_break_condition=loop_analysis.get('has_break_condition', False),
                            break_conditions=loop_analysis.get('break_conditions', []),
                            total_iterations=pattern.frequency,
                            confidence=pattern.confidence * 0.9,  # Slightly lower confidence for loops
                            tags=pattern.tags + ['derived_from_pattern']
                        )
                        
                        # Calculate loop-specific metrics
                        self._calculate_loop_metrics(loop, sequence)
                        loops.append(loop)
            
            # Detect nested loops
            if context.analyze_nested_loops:
                nested_loops = self._detect_nested_loops(loops)
                for parent_loop in nested_loops:
                    parent_loop.nested_loops = nested_loops[parent_loop]
            
        except Exception as e:
            logger.error(f"Loop detection from patterns failed: {e}")
        
        return loops
    
    # Helper methods
    def _hash_sequence(self, sequence: List[ActionStep]) -> str:
        """Generate hash for a sequence of action steps"""
        sequence_str = '|'.join(f"{step.action_type}:{step.target}" for step in sequence)
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def _calculate_sequence_similarity(self, seq1: List[ActionStep], seq2: List[ActionStep]) -> float:
        """Calculate similarity between two sequences"""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for s1, s2 in zip(seq1, seq2) if s1 == s2)
        return matches / len(seq1) if seq1 else 0.0
    
    def _calculate_pattern_metrics(self, pattern: Pattern, sequence: List[ActionStep]):
        """Calculate performance metrics for a pattern"""
        try:
            # Calculate average duration
            total_duration = timedelta(0)
            success_count = 0
            
            for start, end in pattern.occurrences:
                for i in range(start, min(end + 1, len(sequence))):
                    step = sequence[i]
                    total_duration += step.duration
                    if step.success:
                        success_count += 1
            
            total_steps = sum(end - start + 1 for start, end in pattern.occurrences)
            pattern.average_duration = total_duration / len(pattern.occurrences) if pattern.occurrences else timedelta(0)
            pattern.success_rate = success_count / total_steps if total_steps > 0 else 0.0
            
            # Calculate optimization score
            pattern.optimization_score = self._calculate_optimization_score(pattern)
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern metrics: {e}")
    
    def _calculate_loop_metrics(self, loop: Loop, sequence: List[ActionStep]):
        """Calculate performance metrics for a loop"""
        try:
            # Calculate iteration metrics
            if loop.iteration_count > 0:
                total_time = sum(step.duration for step in sequence[loop.start_position:loop.end_position + 1])
                loop.average_iteration_time = total_time / loop.iteration_count
            
            # Analyze optimization potential
            loop.can_parallelize = self._can_parallelize_loop(loop, sequence)
            loop.can_vectorize = self._can_vectorize_loop(loop, sequence)
            loop.can_cache = self._can_cache_loop(loop, sequence)
            
            # Calculate efficiency score
            loop.efficiency_score = self._calculate_loop_efficiency(loop)
            
            # Calculate optimization potential
            loop.optimization_potential = self._calculate_loop_optimization_potential(loop)
            
        except Exception as e:
            logger.error(f"Failed to calculate loop metrics: {e}")
    
    def _calculate_optimization_score(self, pattern: Pattern) -> float:
        """Calculate optimization potential score for a pattern"""
        score = 0.0
        
        # Higher frequency = higher optimization potential
        frequency_score = min(pattern.frequency / 10.0, 1.0)
        score += frequency_score * 0.4
        
        # Longer patterns = higher optimization potential
        length_score = min(pattern.length / 20.0, 1.0)
        score += length_score * 0.3
        
        # Lower success rate = higher optimization potential
        if pattern.success_rate < 1.0:
            score += (1.0 - pattern.success_rate) * 0.3
        
        return min(score, 1.0)
    
    def _generate_cache_key(self, sequence: List[ActionStep], 
                           context: LoopContext, 
                           strategy: DetectionStrategy) -> str:
        """Generate cache key for detection results"""
        sequence_hash = self._hash_sequence(sequence)
        context_hash = hashlib.md5(str(context.__dict__).encode()).hexdigest()
        return f"{sequence_hash}:{context_hash}:{strategy.value}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result"""
        # Simplified cache implementation
        return None
    
    def _cache_result(self, cache_key: str, result: DetectionResult):
        """Cache detection result"""
        # Simplified cache implementation
        pass
    
    def _calculate_coverage(self, patterns: List[Pattern], total_steps: int) -> float:
        """Calculate how much of the sequence is covered by patterns"""
        if total_steps == 0:
            return 0.0
        
        covered_positions = set()
        for pattern in patterns:
            for start, end in pattern.occurrences:
                covered_positions.update(range(start, end + 1))
        
        return len(covered_positions) / total_steps
    
    def _calculate_confidence(self, patterns: List[Pattern], loops: List[Loop]) -> float:
        """Calculate overall confidence score"""
        if not patterns and not loops:
            return 0.0
        
        total_confidence = sum(p.confidence for p in patterns) + sum(l.confidence for l in loops)
        total_items = len(patterns) + len(loops)
        
        return total_confidence / total_items
    
    def _generate_optimization_suggestions(self, patterns: List[Pattern], loops: List[Loop]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Pattern-based suggestions
        high_freq_patterns = [p for p in patterns if p.frequency > 5]
        if high_freq_patterns:
            suggestions.append("Consider extracting frequently repeated patterns into reusable functions")
        
        # Loop-based suggestions
        parallelizable_loops = [l for l in loops if l.can_parallelize]
        if parallelizable_loops:
            suggestions.append("Consider parallelizing independent loop iterations")
        
        cacheable_loops = [l for l in loops if l.can_cache]
        if cacheable_loops:
            suggestions.append("Consider caching results of repeated calculations in loops")
        
        return suggestions
    
    def _generate_performance_improvements(self, patterns: List[Pattern], loops: List[Loop]) -> List[str]:
        """Generate performance improvement suggestions"""
        improvements = []
        
        # Identify inefficient patterns
        inefficient_patterns = [p for p in patterns if p.success_rate < 0.8]
        if inefficient_patterns:
            improvements.append("Review and improve error-prone patterns")
        
        # Identify slow loops
        slow_loops = [l for l in loops if l.average_iteration_time > timedelta(seconds=1)]
        if slow_loops:
            improvements.append("Optimize slow-running loop iterations")
        
        return improvements
    
    def _update_statistics(self, result: DetectionResult):
        """Update performance statistics"""
        self.performance_stats['total_detections'] += 1
        self.performance_stats['patterns_found'] += len(result.patterns)
        self.performance_stats['loops_found'] += len(result.loops)
        
        # Update average detection time
        current_avg = self.performance_stats['average_detection_time']
        new_avg = (current_avg * (self.performance_stats['total_detections'] - 1) + result.detection_time) / self.performance_stats['total_detections']
        self.performance_stats['average_detection_time'] = new_avg
    
    # Simplified implementations for complex methods
    def _group_by_semantic_similarity(self, sequence: List[ActionStep]) -> Dict[str, List[Tuple[int, ActionStep]]]:
        """Group actions by semantic similarity"""
        groups = defaultdict(list)
        for i, step in enumerate(sequence):
            # Simplified semantic grouping by action type
            semantic_key = step.action_type
            groups[semantic_key].append((i, step))
        return groups
    
    def _build_structure_graph(self, sequence: List[ActionStep]) -> Dict[str, Any]:
        """Build structural relationship graph"""
        # Simplified structure analysis
        return {'nodes': sequence, 'edges': []}
    
    def _find_structural_repetitions(self, graph: Dict[str, Any], context: LoopContext) -> List[Dict[str, Any]]:
        """Find repetitive structural patterns"""
        # Simplified structural pattern finding
        return []
    
    def _find_regular_intervals(self, timestamps: List[datetime], context: LoopContext) -> List[Dict[str, Any]]:
        """Find regular time intervals"""
        # Simplified temporal pattern finding
        return []
    
    def _calculate_action_frequencies(self, sequence: List[ActionStep]) -> Dict[str, int]:
        """Calculate frequency of each action type"""
        frequencies = defaultdict(int)
        for step in sequence:
            frequencies[step.action_type] += 1
        return frequencies
    
    def _find_statistical_patterns(self, frequencies: Dict[str, int], 
                                  sequence: List[ActionStep], 
                                  context: LoopContext) -> List[Dict[str, Any]]:
        """Find statistically significant patterns"""
        # Simplified statistical analysis
        return []
    
    def _extract_sequence_features(self, sequence: List[ActionStep]) -> List[List[float]]:
        """Extract features for ML analysis"""
        # Simplified feature extraction
        return [[float(hash(step.action_type) % 1000)] for step in sequence]
    
    def _cluster_sequences(self, features: List[List[float]], context: LoopContext) -> List[Dict[str, Any]]:
        """Cluster similar sequences"""
        # Simplified clustering
        return []
    
    def _merge_similar_patterns(self, patterns: List[Pattern], context: LoopContext) -> List[Pattern]:
        """Merge similar patterns from different strategies"""
        # Simplified pattern merging
        return patterns
    
    def _rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Rank patterns by relevance and confidence"""
        return sorted(patterns, key=lambda p: (p.confidence, p.frequency), reverse=True)
    
    def _analyze_loop_structure(self, pattern: Pattern, sequence: List[ActionStep]) -> Dict[str, Any]:
        """Analyze if a pattern represents a loop structure"""
        # Simplified loop analysis
        return {
            'is_loop': pattern.frequency >= 2,
            'loop_type': LoopType.FOR_LOOP,
            'condition': '',
            'is_infinite': False,
            'has_break_condition': False,
            'break_conditions': []
        }
    
    def _detect_nested_loops(self, loops: List[Loop]) -> Dict[Loop, List[Loop]]:
        """Detect nested loop relationships"""
        # Simplified nested loop detection
        return {}
    
    def _can_parallelize_loop(self, loop: Loop, sequence: List[ActionStep]) -> bool:
        """Check if loop can be parallelized"""
        # Simplified parallelization analysis
        return loop.iteration_count > 2
    
    def _can_vectorize_loop(self, loop: Loop, sequence: List[ActionStep]) -> bool:
        """Check if loop can be vectorized"""
        # Simplified vectorization analysis
        return False
    
    def _can_cache_loop(self, loop: Loop, sequence: List[ActionStep]) -> bool:
        """Check if loop results can be cached"""
        # Simplified caching analysis
        return loop.iteration_count > 3
    
    def _calculate_loop_efficiency(self, loop: Loop) -> float:
        """Calculate loop efficiency score"""
        # Simplified efficiency calculation
        return 0.8 if loop.iteration_count > 1 else 0.5
    
    def _calculate_loop_optimization_potential(self, loop: Loop) -> float:
        """Calculate optimization potential for a loop"""
        potential = 0.0
        
        if loop.can_parallelize:
            potential += 0.4
        if loop.can_vectorize:
            potential += 0.3
        if loop.can_cache:
            potential += 0.3
        
        return min(potential, 1.0)
    
    def _calculate_time_savings(self, patterns: List[Pattern], loops: List[Loop]) -> timedelta:
        """Calculate potential time savings from optimization"""
        total_savings = timedelta(0)
        
        # Calculate savings from pattern optimization
        for pattern in patterns:
            if pattern.optimization_score > 0.5:
                potential_reduction = pattern.average_duration * 0.2  # 20% improvement
                total_savings += potential_reduction * pattern.frequency
        
        # Calculate savings from loop optimization
        for loop in loops:
            if loop.optimization_potential > 0.5:
                potential_reduction = loop.average_iteration_time * 0.3  # 30% improvement
                total_savings += potential_reduction * loop.total_iterations
        
        return total_savings
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        return {
            'total_detections': self.performance_stats['total_detections'],
            'patterns_found': self.performance_stats['patterns_found'],
            'loops_found': self.performance_stats['loops_found'],
            'average_detection_time': self.performance_stats['average_detection_time'],
            'cache_size': len(self.pattern_cache) + len(self.loop_cache),
            'history_size': len(self.detection_history)
        }