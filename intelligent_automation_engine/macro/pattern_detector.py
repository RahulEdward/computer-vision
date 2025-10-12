"""
Pattern Detection

Analyzes recorded actions to detect patterns, repetitions, and optimization opportunities.
This enables the system to generalize from single recordings and create more robust automations.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import asyncio

from .recording_session import RecordedAction, ActionType, ElementInfo


class PatternType(Enum):
    """Types of detected patterns."""
    REPETITION = "repetition"
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    TEMPLATE = "template"
    NAVIGATION = "navigation"
    DATA_ENTRY = "data_entry"


@dataclass
class DetectedPattern:
    """A detected pattern in recorded actions."""
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    actions: List[RecordedAction]
    start_index: int
    end_index: int
    
    # Pattern-specific data
    repetition_count: Optional[int] = None
    variation_tolerance: float = 0.1
    template_variables: Dict[str, Any] = None
    optimization_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.template_variables is None:
            self.template_variables = {}
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []


@dataclass
class PatternAnalysis:
    """Results of pattern analysis."""
    total_actions: int
    patterns_found: List[DetectedPattern]
    optimization_potential: float
    suggested_improvements: List[str]
    execution_time_estimate: float
    complexity_score: float


class PatternDetector:
    """
    Analyzes recorded actions to detect patterns and optimization opportunities.
    """
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.logger = logging.getLogger(__name__)
        
        # Detection settings
        self.settings = {
            'min_repetition_count': 2,
            'max_sequence_gap': 5.0,  # seconds
            'position_tolerance': 10.0,  # pixels
            'text_similarity_threshold': 0.8,
            'element_similarity_threshold': 0.9,
            'min_pattern_confidence': 0.7
        }
        
        # Pattern cache
        self._pattern_cache: Dict[str, List[DetectedPattern]] = {}
        
        # Analysis statistics
        self.stats = {
            'patterns_detected': 0,
            'repetitions_found': 0,
            'sequences_identified': 0,
            'optimization_opportunities': 0
        }
    
    async def analyze_actions(self, actions: List[RecordedAction]) -> PatternAnalysis:
        """
        Analyze a list of actions to detect patterns.
        
        Args:
            actions: List of recorded actions to analyze
            
        Returns:
            PatternAnalysis: Analysis results with detected patterns
        """
        try:
            self.logger.info(f"Analyzing {len(actions)} actions for patterns")
            
            # Create cache key
            cache_key = self._create_cache_key(actions)
            
            # Check cache
            if cache_key in self._pattern_cache:
                patterns = self._pattern_cache[cache_key]
            else:
                # Detect patterns
                patterns = await self._detect_all_patterns(actions)
                self._pattern_cache[cache_key] = patterns
            
            # Calculate analysis metrics
            analysis = PatternAnalysis(
                total_actions=len(actions),
                patterns_found=patterns,
                optimization_potential=self._calculate_optimization_potential(patterns),
                suggested_improvements=self._generate_improvement_suggestions(patterns),
                execution_time_estimate=self._estimate_execution_time(actions, patterns),
                complexity_score=self._calculate_complexity_score(actions, patterns)
            )
            
            self.logger.info(f"Pattern analysis complete: {len(patterns)} patterns found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze actions: {e}")
            return PatternAnalysis(
                total_actions=len(actions),
                patterns_found=[],
                optimization_potential=0.0,
                suggested_improvements=[],
                execution_time_estimate=0.0,
                complexity_score=1.0
            )
    
    async def _detect_all_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect all types of patterns in the actions."""
        patterns = []
        
        # Detect different pattern types
        patterns.extend(await self._detect_repetition_patterns(actions))
        patterns.extend(await self._detect_sequence_patterns(actions))
        patterns.extend(await self._detect_loop_patterns(actions))
        patterns.extend(await self._detect_navigation_patterns(actions))
        patterns.extend(await self._detect_data_entry_patterns(actions))
        patterns.extend(await self._detect_template_patterns(actions))
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.settings['min_pattern_confidence']]
        
        # Remove overlapping patterns (keep highest confidence)
        patterns = self._remove_overlapping_patterns(patterns)
        
        # Update statistics
        self.stats['patterns_detected'] = len(patterns)
        self.stats['repetitions_found'] = len([p for p in patterns if p.pattern_type == PatternType.REPETITION])
        self.stats['sequences_identified'] = len([p for p in patterns if p.pattern_type == PatternType.SEQUENCE])
        
        return patterns
    
    async def _detect_repetition_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect repetitive action patterns."""
        patterns = []
        
        try:
            # Look for repeated sequences of actions
            for window_size in range(2, min(10, len(actions) // 2)):
                for start_idx in range(len(actions) - window_size * 2):
                    sequence = actions[start_idx:start_idx + window_size]
                    
                    # Look for repetitions of this sequence
                    repetitions = self._find_sequence_repetitions(actions, sequence, start_idx + window_size)
                    
                    if len(repetitions) >= self.settings['min_repetition_count']:
                        confidence = self._calculate_repetition_confidence(sequence, repetitions)
                        
                        if confidence >= self.settings['min_pattern_confidence']:
                            pattern = DetectedPattern(
                                pattern_id=f"rep_{start_idx}_{window_size}",
                                pattern_type=PatternType.REPETITION,
                                confidence=confidence,
                                actions=sequence,
                                start_index=start_idx,
                                end_index=start_idx + window_size * (len(repetitions) + 1),
                                repetition_count=len(repetitions) + 1
                            )
                            
                            pattern.optimization_suggestions = [
                                f"Convert to loop with {pattern.repetition_count} iterations",
                                "Add variable elements detection",
                                "Optimize wait times between repetitions"
                            ]
                            
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to detect repetition patterns: {e}")
        
        return patterns
    
    async def _detect_sequence_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect logical sequence patterns."""
        patterns = []
        
        try:
            # Group actions by application/window
            context_groups = self._group_actions_by_context(actions)
            
            for context, context_actions in context_groups.items():
                if len(context_actions) < 3:
                    continue
                
                # Look for common sequences within the same context
                sequences = self._identify_logical_sequences(context_actions)
                
                for sequence in sequences:
                    if len(sequence) >= 3:
                        confidence = self._calculate_sequence_confidence(sequence)
                        
                        if confidence >= self.settings['min_pattern_confidence']:
                            start_idx = actions.index(sequence[0])
                            end_idx = actions.index(sequence[-1]) + 1
                            
                            pattern = DetectedPattern(
                                pattern_id=f"seq_{start_idx}_{len(sequence)}",
                                pattern_type=PatternType.SEQUENCE,
                                confidence=confidence,
                                actions=sequence,
                                start_index=start_idx,
                                end_index=end_idx
                            )
                            
                            pattern.optimization_suggestions = [
                                "Group related actions together",
                                "Add error handling between steps",
                                "Optimize element selection methods"
                            ]
                            
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to detect sequence patterns: {e}")
        
        return patterns
    
    async def _detect_loop_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect loop-like patterns."""
        patterns = []
        
        try:
            # Look for patterns that suggest loops
            for i in range(len(actions) - 4):
                # Check for actions that return to similar states
                current_action = actions[i]
                
                # Look ahead for similar actions
                for j in range(i + 2, min(i + 20, len(actions))):
                    if self._actions_are_similar(current_action, actions[j]):
                        # Found potential loop
                        loop_actions = actions[i:j+1]
                        
                        # Analyze if this looks like a loop
                        if self._is_loop_pattern(loop_actions):
                            confidence = self._calculate_loop_confidence(loop_actions)
                            
                            if confidence >= self.settings['min_pattern_confidence']:
                                pattern = DetectedPattern(
                                    pattern_id=f"loop_{i}_{j}",
                                    pattern_type=PatternType.LOOP,
                                    confidence=confidence,
                                    actions=loop_actions,
                                    start_index=i,
                                    end_index=j + 1
                                )
                                
                                pattern.optimization_suggestions = [
                                    "Convert to while/for loop",
                                    "Add loop termination conditions",
                                    "Optimize loop body for efficiency"
                                ]
                                
                                patterns.append(pattern)
                                break
            
        except Exception as e:
            self.logger.error(f"Failed to detect loop patterns: {e}")
        
        return patterns
    
    async def _detect_navigation_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect navigation patterns."""
        patterns = []
        
        try:
            # Look for navigation sequences
            nav_actions = [action for action in actions 
                          if action.action_type in [ActionType.CLICK, ActionType.KEY_PRESS]]
            
            # Group navigation actions
            nav_sequences = self._group_navigation_actions(nav_actions)
            
            for sequence in nav_sequences:
                if len(sequence) >= 2:
                    confidence = self._calculate_navigation_confidence(sequence)
                    
                    if confidence >= self.settings['min_pattern_confidence']:
                        start_idx = actions.index(sequence[0])
                        end_idx = actions.index(sequence[-1]) + 1
                        
                        pattern = DetectedPattern(
                            pattern_id=f"nav_{start_idx}_{len(sequence)}",
                            pattern_type=PatternType.NAVIGATION,
                            confidence=confidence,
                            actions=sequence,
                            start_index=start_idx,
                            end_index=end_idx
                        )
                        
                        pattern.optimization_suggestions = [
                            "Use direct navigation methods",
                            "Add navigation state verification",
                            "Implement breadcrumb tracking"
                        ]
                        
                        patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to detect navigation patterns: {e}")
        
        return patterns
    
    async def _detect_data_entry_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect data entry patterns."""
        patterns = []
        
        try:
            # Look for data entry sequences
            entry_actions = [action for action in actions 
                           if action.action_type in [ActionType.TYPE, ActionType.CLICK]]
            
            # Group data entry actions
            entry_sequences = self._group_data_entry_actions(entry_actions)
            
            for sequence in entry_sequences:
                if len(sequence) >= 2:
                    confidence = self._calculate_data_entry_confidence(sequence)
                    
                    if confidence >= self.settings['min_pattern_confidence']:
                        start_idx = actions.index(sequence[0])
                        end_idx = actions.index(sequence[-1]) + 1
                        
                        pattern = DetectedPattern(
                            pattern_id=f"data_{start_idx}_{len(sequence)}",
                            pattern_type=PatternType.DATA_ENTRY,
                            confidence=confidence,
                            actions=sequence,
                            start_index=start_idx,
                            end_index=end_idx
                        )
                        
                        # Detect template variables
                        pattern.template_variables = self._extract_template_variables(sequence)
                        
                        pattern.optimization_suggestions = [
                            "Use form filling automation",
                            "Add data validation",
                            "Implement bulk data entry"
                        ]
                        
                        patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to detect data entry patterns: {e}")
        
        return patterns
    
    async def _detect_template_patterns(self, actions: List[RecordedAction]) -> List[DetectedPattern]:
        """Detect template patterns with variable elements."""
        patterns = []
        
        try:
            # Look for actions with similar structure but different data
            for i in range(len(actions) - 1):
                for j in range(i + 1, len(actions)):
                    if self._actions_are_template_similar(actions[i], actions[j]):
                        # Found potential template pattern
                        template_actions = [actions[i], actions[j]]
                        
                        # Look for more similar actions
                        for k in range(j + 1, len(actions)):
                            if self._actions_are_template_similar(actions[i], actions[k]):
                                template_actions.append(actions[k])
                        
                        if len(template_actions) >= 2:
                            confidence = self._calculate_template_confidence(template_actions)
                            
                            if confidence >= self.settings['min_pattern_confidence']:
                                pattern = DetectedPattern(
                                    pattern_id=f"template_{i}_{len(template_actions)}",
                                    pattern_type=PatternType.TEMPLATE,
                                    confidence=confidence,
                                    actions=template_actions,
                                    start_index=i,
                                    end_index=max(actions.index(action) for action in template_actions) + 1
                                )
                                
                                # Extract template variables
                                pattern.template_variables = self._extract_template_variables(template_actions)
                                
                                pattern.optimization_suggestions = [
                                    "Create parameterized template",
                                    "Add variable substitution",
                                    "Implement data-driven execution"
                                ]
                                
                                patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to detect template patterns: {e}")
        
        return patterns
    
    def _find_sequence_repetitions(self, actions: List[RecordedAction], 
                                  sequence: List[RecordedAction], 
                                  start_search: int) -> List[List[RecordedAction]]:
        """Find repetitions of a sequence in the actions."""
        repetitions = []
        sequence_length = len(sequence)
        
        for i in range(start_search, len(actions) - sequence_length + 1):
            candidate = actions[i:i + sequence_length]
            
            if self._sequences_are_similar(sequence, candidate):
                repetitions.append(candidate)
                i += sequence_length  # Skip ahead to avoid overlaps
        
        return repetitions
    
    def _sequences_are_similar(self, seq1: List[RecordedAction], 
                              seq2: List[RecordedAction]) -> bool:
        """Check if two sequences are similar."""
        if len(seq1) != len(seq2):
            return False
        
        similarity_count = 0
        for a1, a2 in zip(seq1, seq2):
            if self._actions_are_similar(a1, a2):
                similarity_count += 1
        
        similarity_ratio = similarity_count / len(seq1)
        return similarity_ratio >= 0.8
    
    def _actions_are_similar(self, action1: RecordedAction, action2: RecordedAction) -> bool:
        """Check if two actions are similar."""
        # Same action type
        if action1.action_type != action2.action_type:
            return False
        
        # Position similarity (if applicable)
        if action1.x is not None and action2.x is not None:
            distance = math.sqrt((action1.x - action2.x)**2 + (action1.y - action2.y)**2)
            if distance > self.settings['position_tolerance']:
                return False
        
        # Element similarity (if applicable)
        if action1.target_element and action2.target_element:
            if not self._elements_are_similar(action1.target_element, action2.target_element):
                return False
        
        # Text similarity (if applicable)
        if action1.text_input and action2.text_input:
            similarity = self._calculate_text_similarity(action1.text_input, action2.text_input)
            if similarity < self.settings['text_similarity_threshold']:
                return False
        
        return True
    
    def _actions_are_template_similar(self, action1: RecordedAction, action2: RecordedAction) -> bool:
        """Check if two actions are similar enough to be part of a template."""
        # Same action type
        if action1.action_type != action2.action_type:
            return False
        
        # Element structure similarity (but data can be different)
        if action1.target_element and action2.target_element:
            return self._elements_are_structurally_similar(action1.target_element, action2.target_element)
        
        # Position similarity for non-element actions
        if action1.x is not None and action2.x is not None:
            distance = math.sqrt((action1.x - action2.x)**2 + (action1.y - action2.y)**2)
            return distance <= self.settings['position_tolerance'] * 2  # More tolerant for templates
        
        return True
    
    def _elements_are_similar(self, elem1: ElementInfo, elem2: ElementInfo) -> bool:
        """Check if two elements are similar."""
        # Same element type
        if elem1.element_type != elem2.element_type:
            return False
        
        # Check various identifiers
        if elem1.id and elem2.id and elem1.id == elem2.id:
            return True
        
        if elem1.class_name and elem2.class_name and elem1.class_name == elem2.class_name:
            return True
        
        if elem1.xpath and elem2.xpath and elem1.xpath == elem2.xpath:
            return True
        
        # Text similarity
        if elem1.text and elem2.text:
            similarity = self._calculate_text_similarity(elem1.text, elem2.text)
            return similarity >= self.settings['element_similarity_threshold']
        
        return False
    
    def _elements_are_structurally_similar(self, elem1: ElementInfo, elem2: ElementInfo) -> bool:
        """Check if two elements have similar structure (for templates)."""
        # Same element type
        if elem1.element_type != elem2.element_type:
            return False
        
        # Same tag name
        if elem1.tag_name != elem2.tag_name:
            return False
        
        # Similar class names (structure, not necessarily exact match)
        if elem1.class_name and elem2.class_name:
            # Check if class names have similar structure
            classes1 = set(elem1.class_name.split())
            classes2 = set(elem2.class_name.split())
            
            # At least 50% overlap in class names
            if classes1 and classes2:
                overlap = len(classes1.intersection(classes2)) / len(classes1.union(classes2))
                return overlap >= 0.5
        
        return True
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _group_actions_by_context(self, actions: List[RecordedAction]) -> Dict[str, List[RecordedAction]]:
        """Group actions by their context (application/window)."""
        groups = defaultdict(list)
        
        for action in actions:
            context_key = f"{action.screen_context.application_name}:{action.screen_context.window_title}"
            groups[context_key].append(action)
        
        return dict(groups)
    
    def _identify_logical_sequences(self, actions: List[RecordedAction]) -> List[List[RecordedAction]]:
        """Identify logical sequences within a context."""
        sequences = []
        current_sequence = []
        
        for i, action in enumerate(actions):
            if not current_sequence:
                current_sequence = [action]
                continue
            
            # Check if this action logically follows the previous one
            prev_action = current_sequence[-1]
            time_gap = action.timestamp - prev_action.timestamp
            
            if time_gap <= self.settings['max_sequence_gap']:
                current_sequence.append(action)
            else:
                # End current sequence and start new one
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                current_sequence = [action]
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _is_loop_pattern(self, actions: List[RecordedAction]) -> bool:
        """Check if a sequence of actions represents a loop pattern."""
        if len(actions) < 4:
            return False
        
        # Look for repeated elements or similar actions
        action_types = [action.action_type for action in actions]
        type_counts = Counter(action_types)
        
        # If there are repeated action types, it might be a loop
        repeated_types = sum(1 for count in type_counts.values() if count > 1)
        
        return repeated_types >= 2
    
    def _group_navigation_actions(self, actions: List[RecordedAction]) -> List[List[RecordedAction]]:
        """Group navigation-related actions."""
        sequences = []
        current_sequence = []
        
        for action in actions:
            # Check if this is a navigation action
            is_nav = (action.action_type == ActionType.CLICK and 
                     action.target_element and 
                     action.target_element.element_type in ['link', 'button', 'menu_item'])
            
            if is_nav:
                current_sequence.append(action)
            else:
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                current_sequence = []
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _group_data_entry_actions(self, actions: List[RecordedAction]) -> List[List[RecordedAction]]:
        """Group data entry related actions."""
        sequences = []
        current_sequence = []
        
        for action in actions:
            # Check if this is a data entry action
            is_data_entry = (action.action_type == ActionType.TYPE or
                           (action.action_type == ActionType.CLICK and
                            action.target_element and
                            action.target_element.element_type in ['text_input', 'dropdown', 'checkbox']))
            
            if is_data_entry:
                current_sequence.append(action)
            else:
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                current_sequence = []
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _extract_template_variables(self, actions: List[RecordedAction]) -> Dict[str, Any]:
        """Extract template variables from similar actions."""
        variables = {}
        
        if len(actions) < 2:
            return variables
        
        # Compare first two actions to find differences
        action1, action2 = actions[0], actions[1]
        
        # Text differences
        if action1.text_input and action2.text_input and action1.text_input != action2.text_input:
            variables['text_input'] = {
                'type': 'string',
                'examples': [action1.text_input, action2.text_input],
                'pattern': 'variable_text'
            }
        
        # Position differences
        if (action1.x is not None and action2.x is not None and 
            abs(action1.x - action2.x) > self.settings['position_tolerance']):
            variables['position'] = {
                'type': 'coordinates',
                'examples': [(action1.x, action1.y), (action2.x, action2.y)],
                'pattern': 'variable_position'
            }
        
        # Element text differences
        if (action1.target_element and action2.target_element and
            action1.target_element.text and action2.target_element.text and
            action1.target_element.text != action2.target_element.text):
            variables['element_text'] = {
                'type': 'string',
                'examples': [action1.target_element.text, action2.target_element.text],
                'pattern': 'variable_element_text'
            }
        
        return variables
    
    def _calculate_repetition_confidence(self, sequence: List[RecordedAction], 
                                       repetitions: List[List[RecordedAction]]) -> float:
        """Calculate confidence for repetition patterns."""
        base_confidence = 0.8
        
        # More repetitions = higher confidence
        repetition_bonus = min(0.15, len(repetitions) * 0.05)
        
        # Longer sequences = higher confidence
        length_bonus = min(0.05, len(sequence) * 0.01)
        
        return min(1.0, base_confidence + repetition_bonus + length_bonus)
    
    def _calculate_sequence_confidence(self, sequence: List[RecordedAction]) -> float:
        """Calculate confidence for sequence patterns."""
        base_confidence = 0.7
        
        # Longer sequences = higher confidence
        length_bonus = min(0.2, len(sequence) * 0.02)
        
        # Same context = higher confidence
        contexts = set(f"{a.screen_context.application_name}:{a.screen_context.window_title}" 
                      for a in sequence)
        context_bonus = 0.1 if len(contexts) == 1 else 0.0
        
        return min(1.0, base_confidence + length_bonus + context_bonus)
    
    def _calculate_loop_confidence(self, actions: List[RecordedAction]) -> float:
        """Calculate confidence for loop patterns."""
        base_confidence = 0.6
        
        # Check for loop indicators
        action_types = [action.action_type for action in actions]
        type_counts = Counter(action_types)
        
        # More repeated types = higher confidence
        repeated_ratio = sum(1 for count in type_counts.values() if count > 1) / len(type_counts)
        repetition_bonus = repeated_ratio * 0.3
        
        return min(1.0, base_confidence + repetition_bonus)
    
    def _calculate_navigation_confidence(self, sequence: List[RecordedAction]) -> float:
        """Calculate confidence for navigation patterns."""
        base_confidence = 0.75
        
        # Check for navigation indicators
        nav_elements = sum(1 for action in sequence 
                          if action.target_element and 
                          action.target_element.element_type in ['link', 'button', 'menu_item'])
        
        nav_ratio = nav_elements / len(sequence)
        nav_bonus = nav_ratio * 0.2
        
        return min(1.0, base_confidence + nav_bonus)
    
    def _calculate_data_entry_confidence(self, sequence: List[RecordedAction]) -> float:
        """Calculate confidence for data entry patterns."""
        base_confidence = 0.8
        
        # Check for data entry indicators
        entry_actions = sum(1 for action in sequence 
                           if action.action_type in [ActionType.TYPE, ActionType.CLICK])
        
        entry_ratio = entry_actions / len(sequence)
        entry_bonus = entry_ratio * 0.15
        
        return min(1.0, base_confidence + entry_bonus)
    
    def _calculate_template_confidence(self, actions: List[RecordedAction]) -> float:
        """Calculate confidence for template patterns."""
        base_confidence = 0.7
        
        # More similar actions = higher confidence
        similarity_bonus = min(0.2, len(actions) * 0.05)
        
        # Variable detection = higher confidence
        variables = self._extract_template_variables(actions)
        variable_bonus = min(0.1, len(variables) * 0.05)
        
        return min(1.0, base_confidence + similarity_bonus + variable_bonus)
    
    def _remove_overlapping_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Remove overlapping patterns, keeping the highest confidence ones."""
        # Sort by confidence (descending)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        filtered_patterns = []
        used_ranges = set()
        
        for pattern in patterns:
            # Check if this pattern overlaps with any already selected pattern
            pattern_range = set(range(pattern.start_index, pattern.end_index))
            
            if not pattern_range.intersection(used_ranges):
                filtered_patterns.append(pattern)
                used_ranges.update(pattern_range)
        
        return filtered_patterns
    
    def _calculate_optimization_potential(self, patterns: List[DetectedPattern]) -> float:
        """Calculate the optimization potential based on detected patterns."""
        if not patterns:
            return 0.0
        
        total_potential = 0.0
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.REPETITION:
                # High potential for repetitions
                total_potential += 0.8 * pattern.confidence
            elif pattern.pattern_type == PatternType.LOOP:
                # High potential for loops
                total_potential += 0.7 * pattern.confidence
            elif pattern.pattern_type == PatternType.TEMPLATE:
                # Medium potential for templates
                total_potential += 0.6 * pattern.confidence
            else:
                # Lower potential for other patterns
                total_potential += 0.4 * pattern.confidence
        
        return min(1.0, total_potential / len(patterns))
    
    def _generate_improvement_suggestions(self, patterns: List[DetectedPattern]) -> List[str]:
        """Generate improvement suggestions based on detected patterns."""
        suggestions = []
        
        repetition_count = len([p for p in patterns if p.pattern_type == PatternType.REPETITION])
        loop_count = len([p for p in patterns if p.pattern_type == PatternType.LOOP])
        template_count = len([p for p in patterns if p.pattern_type == PatternType.TEMPLATE])
        
        if repetition_count > 0:
            suggestions.append(f"Convert {repetition_count} repetitive sequences to loops")
        
        if loop_count > 0:
            suggestions.append(f"Optimize {loop_count} loop patterns with better termination conditions")
        
        if template_count > 0:
            suggestions.append(f"Create {template_count} parameterized templates for reusability")
        
        # General suggestions
        if len(patterns) > 5:
            suggestions.append("Consider breaking down complex workflow into smaller modules")
        
        suggestions.append("Add error handling and recovery mechanisms")
        suggestions.append("Implement smart waiting strategies")
        
        return suggestions
    
    def _estimate_execution_time(self, actions: List[RecordedAction], 
                                patterns: List[DetectedPattern]) -> float:
        """Estimate execution time for the optimized workflow."""
        if not actions:
            return 0.0
        
        # Base execution time from recorded actions
        total_time = actions[-1].timestamp - actions[0].timestamp
        
        # Apply optimization factors based on patterns
        optimization_factor = 1.0
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.REPETITION:
                # Repetitions can be optimized significantly
                optimization_factor *= 0.7
            elif pattern.pattern_type == PatternType.LOOP:
                # Loops can be optimized moderately
                optimization_factor *= 0.8
            elif pattern.pattern_type == PatternType.TEMPLATE:
                # Templates provide some optimization
                optimization_factor *= 0.9
        
        return total_time * optimization_factor
    
    def _calculate_complexity_score(self, actions: List[RecordedAction], 
                                   patterns: List[DetectedPattern]) -> float:
        """Calculate complexity score for the workflow."""
        if not actions:
            return 0.0
        
        # Base complexity from number of actions
        base_complexity = min(1.0, len(actions) / 100.0)
        
        # Complexity from different action types
        action_types = set(action.action_type for action in actions)
        type_complexity = len(action_types) / len(ActionType)
        
        # Complexity from context switches
        contexts = set(f"{a.screen_context.application_name}:{a.screen_context.window_title}" 
                      for a in actions)
        context_complexity = min(1.0, len(contexts) / 10.0)
        
        # Pattern complexity (more patterns = more complex)
        pattern_complexity = min(1.0, len(patterns) / 20.0)
        
        # Weighted average
        complexity = (base_complexity * 0.4 + 
                     type_complexity * 0.2 + 
                     context_complexity * 0.2 + 
                     pattern_complexity * 0.2)
        
        return complexity
    
    def _create_cache_key(self, actions: List[RecordedAction]) -> str:
        """Create a cache key for the actions."""
        # Create a hash based on action types and timestamps
        import hashlib
        
        key_data = []
        for action in actions:
            key_data.append(f"{action.action_type.value}:{action.timestamp}")
        
        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the pattern cache."""
        self._pattern_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern detection statistics."""
        return self.stats.copy()