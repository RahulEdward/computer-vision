"""
Action Generalizer

Converts specific recorded actions into generalized, reusable automation patterns.
This enables the system to create robust automations that work across different contexts.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .recording_session import RecordedAction, ActionType, ElementInfo
from .pattern_detector import DetectedPattern, PatternType


class GeneralizationType(Enum):
    """Types of generalization strategies."""
    ELEMENT_SELECTOR = "element_selector"
    POSITION_RELATIVE = "position_relative"
    TEXT_PATTERN = "text_pattern"
    TIMING_ADAPTIVE = "timing_adaptive"
    CONTEXT_AWARE = "context_aware"
    DATA_DRIVEN = "data_driven"
    CONDITIONAL = "conditional"
    PARAMETERIZED = "parameterized"


@dataclass
class GeneralizationRule:
    """A rule for generalizing actions."""
    rule_id: str
    rule_type: GeneralizationType
    confidence: float
    description: str
    
    # Rule parameters
    original_value: Any
    generalized_value: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    
    # Validation
    success_rate: float = 1.0
    test_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneralizedAction:
    """A generalized action that can be reused across contexts."""
    action_id: str
    original_action: RecordedAction
    action_type: ActionType
    
    # Generalized properties
    element_selector: Optional[str] = None
    position_strategy: Optional[str] = None
    text_pattern: Optional[str] = None
    timing_strategy: Optional[str] = None
    
    # Parameters and variables
    parameters: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    
    # Generalization metadata
    generalization_rules: List[GeneralizationRule] = field(default_factory=list)
    confidence: float = 1.0
    robustness_score: float = 1.0
    
    # Execution context
    required_context: Dict[str, Any] = field(default_factory=dict)
    fallback_strategies: List[str] = field(default_factory=list)


@dataclass
class GeneralizationResult:
    """Result of action generalization."""
    original_actions: List[RecordedAction]
    generalized_actions: List[GeneralizedAction]
    patterns_used: List[DetectedPattern]
    
    # Quality metrics
    generalization_confidence: float
    robustness_score: float
    reusability_score: float
    
    # Recommendations
    optimization_suggestions: List[str]
    validation_requirements: List[str]
    risk_factors: List[str]


class ActionGeneralizer:
    """
    Converts specific recorded actions into generalized, reusable automation patterns.
    """
    
    def __init__(self):
        """Initialize the action generalizer."""
        self.logger = logging.getLogger(__name__)
        
        # Generalization settings
        self.settings = {
            'min_confidence_threshold': 0.7,
            'max_position_variance': 50.0,  # pixels
            'text_similarity_threshold': 0.8,
            'timing_tolerance': 2.0,  # seconds
            'element_selector_preference': ['id', 'class', 'xpath', 'text'],
            'enable_smart_waiting': True,
            'enable_fallback_strategies': True
        }
        
        # Element selector patterns
        self.selector_patterns = {
            'id': r'#([a-zA-Z][\w-]*)',
            'class': r'\.([a-zA-Z][\w-]*)',
            'attribute': r'\[([a-zA-Z-]+)([~|^$*]?=)?["\']?([^"\']*)["\']?\]',
            'text': r':contains\(["\']([^"\']*)["\']?\)',
            'nth_child': r':nth-child\((\d+)\)',
            'tag': r'^([a-zA-Z]+)'
        }
        
        # Text patterns for generalization
        self.text_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'number': r'\b\d+\b',
            'currency': r'\$\d+(?:\.\d{2})?',
            'url': r'https?://[^\s]+',
            'word': r'\b\w+\b',
            'alphanumeric': r'\b[A-Za-z0-9]+\b'
        }
        
        # Statistics
        self.stats = {
            'actions_generalized': 0,
            'rules_created': 0,
            'patterns_applied': 0,
            'success_rate': 0.0
        }
    
    async def generalize_actions(self, actions: List[RecordedAction], 
                                patterns: Optional[List[DetectedPattern]] = None) -> GeneralizationResult:
        """
        Generalize a list of recorded actions.
        
        Args:
            actions: List of recorded actions to generalize
            patterns: Optional detected patterns to guide generalization
            
        Returns:
            GeneralizationResult: Generalization results and metrics
        """
        try:
            self.logger.info(f"Generalizing {len(actions)} actions")
            
            if patterns is None:
                patterns = []
            
            # Generalize individual actions
            generalized_actions = []
            
            for i, action in enumerate(actions):
                generalized = await self._generalize_single_action(action, i, actions, patterns)
                generalized_actions.append(generalized)
            
            # Apply pattern-based generalizations
            generalized_actions = await self._apply_pattern_generalizations(
                generalized_actions, patterns
            )
            
            # Calculate quality metrics
            result = GeneralizationResult(
                original_actions=actions,
                generalized_actions=generalized_actions,
                patterns_used=patterns,
                generalization_confidence=self._calculate_generalization_confidence(generalized_actions),
                robustness_score=self._calculate_robustness_score(generalized_actions),
                reusability_score=self._calculate_reusability_score(generalized_actions),
                optimization_suggestions=self._generate_optimization_suggestions(generalized_actions),
                validation_requirements=self._generate_validation_requirements(generalized_actions),
                risk_factors=self._identify_risk_factors(generalized_actions)
            )
            
            # Update statistics
            self.stats['actions_generalized'] += len(actions)
            self.stats['patterns_applied'] += len(patterns)
            
            self.logger.info(f"Generalization complete: {len(generalized_actions)} actions generalized")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generalize actions: {e}")
            return GeneralizationResult(
                original_actions=actions,
                generalized_actions=[],
                patterns_used=[],
                generalization_confidence=0.0,
                robustness_score=0.0,
                reusability_score=0.0,
                optimization_suggestions=[],
                validation_requirements=[],
                risk_factors=[]
            )
    
    async def _generalize_single_action(self, action: RecordedAction, 
                                       index: int, 
                                       all_actions: List[RecordedAction],
                                       patterns: List[DetectedPattern]) -> GeneralizedAction:
        """Generalize a single recorded action."""
        generalized = GeneralizedAction(
            action_id=f"action_{index}",
            original_action=action,
            action_type=action.action_type
        )
        
        # Apply different generalization strategies
        await self._generalize_element_selector(generalized)
        await self._generalize_position_strategy(generalized, all_actions)
        await self._generalize_text_pattern(generalized)
        await self._generalize_timing_strategy(generalized, all_actions, index)
        await self._add_context_awareness(generalized)
        await self._add_fallback_strategies(generalized)
        
        # Calculate confidence and robustness
        generalized.confidence = self._calculate_action_confidence(generalized)
        generalized.robustness_score = self._calculate_action_robustness(generalized)
        
        return generalized
    
    async def _generalize_element_selector(self, generalized: GeneralizedAction):
        """Generalize element selection strategy."""
        action = generalized.original_action
        
        if not action.target_element:
            return
        
        element = action.target_element
        selectors = []
        
        # Try different selector strategies in order of preference
        for selector_type in self.settings['element_selector_preference']:
            selector = self._create_element_selector(element, selector_type)
            if selector:
                selectors.append((selector_type, selector))
        
        if selectors:
            # Use the first (most preferred) selector
            selector_type, selector = selectors[0]
            generalized.element_selector = selector
            
            # Create generalization rule
            rule = GeneralizationRule(
                rule_id=f"element_selector_{selector_type}",
                rule_type=GeneralizationType.ELEMENT_SELECTOR,
                confidence=self._calculate_selector_confidence(selector_type, element),
                description=f"Use {selector_type} selector for element identification",
                original_value=element,
                generalized_value=selector,
                parameters={'selector_type': selector_type, 'fallback_selectors': selectors[1:]}
            )
            
            generalized.generalization_rules.append(rule)
            
            # Add fallback selectors
            if len(selectors) > 1:
                generalized.fallback_strategies.extend([s[1] for s in selectors[1:]])
    
    async def _generalize_position_strategy(self, generalized: GeneralizedAction, 
                                          all_actions: List[RecordedAction]):
        """Generalize position-based targeting."""
        action = generalized.original_action
        
        if action.x is None or action.y is None:
            return
        
        # Analyze position patterns
        position_strategy = self._determine_position_strategy(action, all_actions)
        
        if position_strategy:
            generalized.position_strategy = position_strategy['strategy']
            
            rule = GeneralizationRule(
                rule_id="position_strategy",
                rule_type=GeneralizationType.POSITION_RELATIVE,
                confidence=position_strategy['confidence'],
                description=position_strategy['description'],
                original_value=(action.x, action.y),
                generalized_value=position_strategy['strategy'],
                parameters=position_strategy.get('parameters', {})
            )
            
            generalized.generalization_rules.append(rule)
    
    async def _generalize_text_pattern(self, generalized: GeneralizedAction):
        """Generalize text input patterns."""
        action = generalized.original_action
        
        if not action.text_input:
            return
        
        # Detect text patterns
        pattern_info = self._detect_text_pattern(action.text_input)
        
        if pattern_info:
            generalized.text_pattern = pattern_info['pattern']
            generalized.parameters['text_type'] = pattern_info['type']
            
            rule = GeneralizationRule(
                rule_id="text_pattern",
                rule_type=GeneralizationType.TEXT_PATTERN,
                confidence=pattern_info['confidence'],
                description=f"Use {pattern_info['type']} pattern for text input",
                original_value=action.text_input,
                generalized_value=pattern_info['pattern'],
                parameters={'text_type': pattern_info['type']}
            )
            
            generalized.generalization_rules.append(rule)
        else:
            # Use parameterized text
            generalized.text_pattern = "{{text_input}}"
            generalized.variables['text_input'] = action.text_input
    
    async def _generalize_timing_strategy(self, generalized: GeneralizedAction, 
                                        all_actions: List[RecordedAction], 
                                        index: int):
        """Generalize timing and waiting strategies."""
        action = generalized.original_action
        
        # Analyze timing patterns
        timing_strategy = self._determine_timing_strategy(action, all_actions, index)
        
        if timing_strategy:
            generalized.timing_strategy = timing_strategy['strategy']
            
            rule = GeneralizationRule(
                rule_id="timing_strategy",
                rule_type=GeneralizationType.TIMING_ADAPTIVE,
                confidence=timing_strategy['confidence'],
                description=timing_strategy['description'],
                original_value=action.timestamp,
                generalized_value=timing_strategy['strategy'],
                parameters=timing_strategy.get('parameters', {})
            )
            
            generalized.generalization_rules.append(rule)
    
    async def _add_context_awareness(self, generalized: GeneralizedAction):
        """Add context awareness to the generalized action."""
        action = generalized.original_action
        
        # Required context
        generalized.required_context = {
            'application': action.screen_context.application_name,
            'window_title_pattern': self._generalize_window_title(action.screen_context.window_title),
            'screen_resolution': (action.screen_context.screen_width, action.screen_context.screen_height)
        }
        
        # Context-aware rule
        rule = GeneralizationRule(
            rule_id="context_awareness",
            rule_type=GeneralizationType.CONTEXT_AWARE,
            confidence=0.9,
            description="Ensure action executes in correct context",
            original_value=action.screen_context,
            generalized_value=generalized.required_context
        )
        
        generalized.generalization_rules.append(rule)
    
    async def _add_fallback_strategies(self, generalized: GeneralizedAction):
        """Add fallback strategies for robust execution."""
        if not self.settings['enable_fallback_strategies']:
            return
        
        action = generalized.original_action
        
        # Add common fallback strategies based on action type
        if action.action_type == ActionType.CLICK:
            generalized.fallback_strategies.extend([
                "retry_with_offset",
                "use_alternative_selector",
                "wait_and_retry",
                "scroll_into_view"
            ])
        elif action.action_type == ActionType.TYPE:
            generalized.fallback_strategies.extend([
                "clear_field_first",
                "use_clipboard",
                "type_with_delays",
                "focus_element_first"
            ])
        elif action.action_type == ActionType.KEY_PRESS:
            generalized.fallback_strategies.extend([
                "use_alternative_key_combination",
                "send_as_text",
                "retry_with_delay"
            ])
    
    async def _apply_pattern_generalizations(self, actions: List[GeneralizedAction], 
                                           patterns: List[DetectedPattern]) -> List[GeneralizedAction]:
        """Apply pattern-based generalizations."""
        for pattern in patterns:
            if pattern.pattern_type == PatternType.REPETITION:
                actions = await self._apply_repetition_generalization(actions, pattern)
            elif pattern.pattern_type == PatternType.TEMPLATE:
                actions = await self._apply_template_generalization(actions, pattern)
            elif pattern.pattern_type == PatternType.LOOP:
                actions = await self._apply_loop_generalization(actions, pattern)
        
        return actions
    
    async def _apply_repetition_generalization(self, actions: List[GeneralizedAction], 
                                             pattern: DetectedPattern) -> List[GeneralizedAction]:
        """Apply generalization for repetition patterns."""
        # Find actions in the repetition pattern
        pattern_actions = actions[pattern.start_index:pattern.end_index]
        
        for action in pattern_actions:
            # Add repetition-specific parameters
            action.parameters['is_repetition'] = True
            action.parameters['repetition_count'] = pattern.repetition_count
            
            # Add repetition rule
            rule = GeneralizationRule(
                rule_id="repetition_pattern",
                rule_type=GeneralizationType.PARAMETERIZED,
                confidence=pattern.confidence,
                description="Part of repetitive pattern",
                original_value=action.original_action,
                generalized_value="loop_iteration",
                parameters={'repetition_count': pattern.repetition_count}
            )
            
            action.generalization_rules.append(rule)
        
        return actions
    
    async def _apply_template_generalization(self, actions: List[GeneralizedAction], 
                                           pattern: DetectedPattern) -> List[GeneralizedAction]:
        """Apply generalization for template patterns."""
        # Extract template variables from pattern
        template_vars = pattern.template_variables
        
        for action in actions:
            if action.original_action in pattern.actions:
                # Apply template variables
                for var_name, var_info in template_vars.items():
                    action.variables[var_name] = var_info.get('pattern', f"{{{{{var_name}}}}}")
                
                # Add template rule
                rule = GeneralizationRule(
                    rule_id="template_pattern",
                    rule_type=GeneralizationType.DATA_DRIVEN,
                    confidence=pattern.confidence,
                    description="Part of template pattern with variables",
                    original_value=action.original_action,
                    generalized_value="template_instance",
                    parameters={'template_variables': template_vars}
                )
                
                action.generalization_rules.append(rule)
        
        return actions
    
    async def _apply_loop_generalization(self, actions: List[GeneralizedAction], 
                                       pattern: DetectedPattern) -> List[GeneralizedAction]:
        """Apply generalization for loop patterns."""
        pattern_actions = actions[pattern.start_index:pattern.end_index]
        
        for action in pattern_actions:
            # Add loop-specific parameters
            action.parameters['is_loop_body'] = True
            action.parameters['loop_condition'] = "{{loop_condition}}"
            
            # Add conditional rule
            rule = GeneralizationRule(
                rule_id="loop_pattern",
                rule_type=GeneralizationType.CONDITIONAL,
                confidence=pattern.confidence,
                description="Part of loop pattern",
                original_value=action.original_action,
                generalized_value="loop_body",
                parameters={'loop_type': 'while', 'condition': '{{loop_condition}}'}
            )
            
            action.generalization_rules.append(rule)
        
        return actions
    
    def _create_element_selector(self, element: ElementInfo, selector_type: str) -> Optional[str]:
        """Create an element selector of the specified type."""
        if selector_type == 'id' and element.id:
            return f"#{element.id}"
        
        elif selector_type == 'class' and element.class_name:
            # Use the first class name
            classes = element.class_name.split()
            if classes:
                return f".{classes[0]}"
        
        elif selector_type == 'xpath' and element.xpath:
            return element.xpath
        
        elif selector_type == 'text' and element.text:
            # Create a text-based selector
            return f"[text*='{element.text}']"
        
        elif selector_type == 'tag' and element.tag_name:
            return element.tag_name
        
        return None
    
    def _calculate_selector_confidence(self, selector_type: str, element: ElementInfo) -> float:
        """Calculate confidence for a selector type."""
        confidence_map = {
            'id': 0.95,
            'xpath': 0.85,
            'class': 0.75,
            'text': 0.65,
            'tag': 0.45
        }
        
        base_confidence = confidence_map.get(selector_type, 0.5)
        
        # Adjust based on element properties
        if selector_type == 'id' and element.id and len(element.id) > 3:
            base_confidence += 0.05
        
        if selector_type == 'class' and element.class_name:
            class_count = len(element.class_name.split())
            if class_count == 1:
                base_confidence += 0.1
            elif class_count > 3:
                base_confidence -= 0.1
        
        return min(1.0, base_confidence)
    
    def _determine_position_strategy(self, action: RecordedAction, 
                                   all_actions: List[RecordedAction]) -> Optional[Dict[str, Any]]:
        """Determine the best position strategy for an action."""
        # Analyze position patterns in nearby actions
        nearby_actions = self._get_nearby_actions(action, all_actions, 3)
        
        # Check for relative positioning
        if len(nearby_actions) > 1:
            # Look for relative patterns
            relative_info = self._analyze_relative_positioning(action, nearby_actions)
            if relative_info:
                return {
                    'strategy': 'relative',
                    'confidence': 0.8,
                    'description': 'Use relative positioning based on nearby elements',
                    'parameters': relative_info
                }
        
        # Check for screen region patterns
        region_info = self._analyze_screen_regions(action)
        if region_info:
            return {
                'strategy': 'region_based',
                'confidence': 0.7,
                'description': f'Use {region_info["region"]} screen region',
                'parameters': region_info
            }
        
        # Default to absolute positioning with tolerance
        return {
            'strategy': 'absolute_with_tolerance',
            'confidence': 0.6,
            'description': 'Use absolute positioning with tolerance for variations',
            'parameters': {
                'tolerance': self.settings['max_position_variance'],
                'x': action.x,
                'y': action.y
            }
        }
    
    def _detect_text_pattern(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect patterns in text input."""
        for pattern_name, pattern_regex in self.text_patterns.items():
            if re.match(pattern_regex, text):
                return {
                    'type': pattern_name,
                    'pattern': pattern_regex,
                    'confidence': 0.9
                }
        
        # Check for common patterns
        if text.isdigit():
            return {
                'type': 'number',
                'pattern': r'\d+',
                'confidence': 0.95
            }
        
        if text.isalpha():
            return {
                'type': 'text',
                'pattern': r'[A-Za-z]+',
                'confidence': 0.8
            }
        
        return None
    
    def _determine_timing_strategy(self, action: RecordedAction, 
                                 all_actions: List[RecordedAction], 
                                 index: int) -> Optional[Dict[str, Any]]:
        """Determine timing strategy for an action."""
        if not self.settings['enable_smart_waiting']:
            return None
        
        # Analyze timing patterns
        if index > 0:
            prev_action = all_actions[index - 1]
            time_gap = action.timestamp - prev_action.timestamp
            
            if time_gap > 2.0:
                # Long wait - might need smart waiting
                return {
                    'strategy': 'smart_wait',
                    'confidence': 0.8,
                    'description': 'Use smart waiting for element availability',
                    'parameters': {
                        'max_wait': min(30.0, time_gap * 2),
                        'poll_interval': 0.5
                    }
                }
            elif time_gap > 0.5:
                # Medium wait - fixed delay
                return {
                    'strategy': 'fixed_delay',
                    'confidence': 0.7,
                    'description': 'Use fixed delay between actions',
                    'parameters': {'delay': time_gap}
                }
        
        return None
    
    def _generalize_window_title(self, title: str) -> str:
        """Generalize window title to handle variations."""
        # Remove common variable parts
        patterns_to_remove = [
            r'\d+',  # Numbers
            r'\([^)]*\)',  # Content in parentheses
            r'\[[^\]]*\]',  # Content in brackets
            r' - \w+$',  # Trailing application name
        ]
        
        generalized = title
        for pattern in patterns_to_remove:
            generalized = re.sub(pattern, '*', generalized)
        
        return generalized
    
    def _get_nearby_actions(self, action: RecordedAction, 
                           all_actions: List[RecordedAction], 
                           window: int) -> List[RecordedAction]:
        """Get actions near the given action in time."""
        action_index = all_actions.index(action)
        start_idx = max(0, action_index - window)
        end_idx = min(len(all_actions), action_index + window + 1)
        
        return all_actions[start_idx:end_idx]
    
    def _analyze_relative_positioning(self, action: RecordedAction, 
                                    nearby_actions: List[RecordedAction]) -> Optional[Dict[str, Any]]:
        """Analyze relative positioning patterns."""
        if action.x is None or action.y is None:
            return None
        
        for other_action in nearby_actions:
            if (other_action != action and 
                other_action.x is not None and 
                other_action.y is not None):
                
                dx = action.x - other_action.x
                dy = action.y - other_action.y
                
                # Check for common relative patterns
                if abs(dx) < 10 and abs(dy) > 30:
                    # Vertical alignment
                    return {
                        'reference_action': other_action,
                        'offset_x': dx,
                        'offset_y': dy,
                        'pattern': 'vertical_alignment'
                    }
                elif abs(dy) < 10 and abs(dx) > 30:
                    # Horizontal alignment
                    return {
                        'reference_action': other_action,
                        'offset_x': dx,
                        'offset_y': dy,
                        'pattern': 'horizontal_alignment'
                    }
        
        return None
    
    def _analyze_screen_regions(self, action: RecordedAction) -> Optional[Dict[str, Any]]:
        """Analyze which screen region the action belongs to."""
        if action.x is None or action.y is None:
            return None
        
        screen_width = action.screen_context.screen_width
        screen_height = action.screen_context.screen_height
        
        # Define regions
        x_ratio = action.x / screen_width
        y_ratio = action.y / screen_height
        
        # Determine region
        if x_ratio < 0.33:
            x_region = 'left'
        elif x_ratio < 0.67:
            x_region = 'center'
        else:
            x_region = 'right'
        
        if y_ratio < 0.33:
            y_region = 'top'
        elif y_ratio < 0.67:
            y_region = 'middle'
        else:
            y_region = 'bottom'
        
        return {
            'region': f"{y_region}_{x_region}",
            'x_ratio': x_ratio,
            'y_ratio': y_ratio
        }
    
    def _calculate_action_confidence(self, action: GeneralizedAction) -> float:
        """Calculate confidence for a generalized action."""
        if not action.generalization_rules:
            return 0.5
        
        # Average confidence of all rules
        total_confidence = sum(rule.confidence for rule in action.generalization_rules)
        avg_confidence = total_confidence / len(action.generalization_rules)
        
        # Bonus for multiple strategies
        strategy_bonus = min(0.1, len(action.generalization_rules) * 0.02)
        
        # Bonus for fallback strategies
        fallback_bonus = min(0.1, len(action.fallback_strategies) * 0.02)
        
        return min(1.0, avg_confidence + strategy_bonus + fallback_bonus)
    
    def _calculate_action_robustness(self, action: GeneralizedAction) -> float:
        """Calculate robustness score for a generalized action."""
        robustness = 0.5  # Base score
        
        # Element selector robustness
        if action.element_selector:
            if '#' in action.element_selector:  # ID selector
                robustness += 0.3
            elif '.' in action.element_selector:  # Class selector
                robustness += 0.2
            elif '[' in action.element_selector:  # Attribute selector
                robustness += 0.15
        
        # Fallback strategies
        robustness += min(0.2, len(action.fallback_strategies) * 0.05)
        
        # Context awareness
        if action.required_context:
            robustness += 0.1
        
        return min(1.0, robustness)
    
    def _calculate_generalization_confidence(self, actions: List[GeneralizedAction]) -> float:
        """Calculate overall generalization confidence."""
        if not actions:
            return 0.0
        
        total_confidence = sum(action.confidence for action in actions)
        return total_confidence / len(actions)
    
    def _calculate_robustness_score(self, actions: List[GeneralizedAction]) -> float:
        """Calculate overall robustness score."""
        if not actions:
            return 0.0
        
        total_robustness = sum(action.robustness_score for action in actions)
        return total_robustness / len(actions)
    
    def _calculate_reusability_score(self, actions: List[GeneralizedAction]) -> float:
        """Calculate reusability score."""
        if not actions:
            return 0.0
        
        reusability_factors = []
        
        for action in actions:
            score = 0.5  # Base score
            
            # Parameterization increases reusability
            if action.variables:
                score += min(0.3, len(action.variables) * 0.1)
            
            # Pattern-based actions are more reusable
            pattern_rules = [r for r in action.generalization_rules 
                           if r.rule_type in [GeneralizationType.DATA_DRIVEN, 
                                            GeneralizationType.PARAMETERIZED]]
            if pattern_rules:
                score += 0.2
            
            reusability_factors.append(score)
        
        return sum(reusability_factors) / len(reusability_factors)
    
    def _generate_optimization_suggestions(self, actions: List[GeneralizedAction]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Analyze common patterns
        selector_types = []
        timing_strategies = []
        
        for action in actions:
            for rule in action.generalization_rules:
                if rule.rule_type == GeneralizationType.ELEMENT_SELECTOR:
                    selector_types.append(rule.parameters.get('selector_type'))
                elif rule.rule_type == GeneralizationType.TIMING_ADAPTIVE:
                    timing_strategies.append(rule.generalized_value)
        
        # Selector suggestions
        if 'xpath' in selector_types and selector_types.count('xpath') > len(selector_types) * 0.5:
            suggestions.append("Consider using more robust selectors instead of XPath")
        
        # Timing suggestions
        if 'smart_wait' in timing_strategies:
            suggestions.append("Implement smart waiting for better reliability")
        
        # General suggestions
        actions_with_fallbacks = sum(1 for a in actions if a.fallback_strategies)
        if actions_with_fallbacks < len(actions) * 0.5:
            suggestions.append("Add more fallback strategies for robustness")
        
        return suggestions
    
    def _generate_validation_requirements(self, actions: List[GeneralizedAction]) -> List[str]:
        """Generate validation requirements."""
        requirements = []
        
        # Context validation
        contexts = set()
        for action in actions:
            if action.required_context:
                contexts.add(action.required_context.get('application'))
        
        if len(contexts) > 1:
            requirements.append("Test across multiple applications")
        
        # Element validation
        element_actions = [a for a in actions if a.element_selector]
        if element_actions:
            requirements.append("Validate element selectors across different page states")
        
        # Timing validation
        timing_actions = [a for a in actions if a.timing_strategy]
        if timing_actions:
            requirements.append("Test timing strategies under different system loads")
        
        return requirements
    
    def _identify_risk_factors(self, actions: List[GeneralizedAction]) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # Low confidence actions
        low_confidence_actions = [a for a in actions if a.confidence < 0.7]
        if low_confidence_actions:
            risks.append(f"{len(low_confidence_actions)} actions have low confidence scores")
        
        # Position-dependent actions
        position_actions = [a for a in actions if a.position_strategy == 'absolute_with_tolerance']
        if position_actions:
            risks.append(f"{len(position_actions)} actions depend on absolute positioning")
        
        # Actions without fallbacks
        no_fallback_actions = [a for a in actions if not a.fallback_strategies]
        if no_fallback_actions:
            risks.append(f"{len(no_fallback_actions)} actions lack fallback strategies")
        
        return risks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generalization statistics."""
        return self.stats.copy()