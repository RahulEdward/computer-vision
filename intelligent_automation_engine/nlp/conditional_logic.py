"""
Conditional Logic Engine

Understands and processes conditional statements from natural language
including if-then-else logic, complex conditions, and nested logic.
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConditionType(Enum):
    """Types of conditions that can be evaluated."""
    ELEMENT_EXISTS = "element_exists"
    ELEMENT_VISIBLE = "element_visible"
    TEXT_CONTAINS = "text_contains"
    VALUE_EQUALS = "value_equals"
    VALUE_GREATER = "value_greater"
    VALUE_LESS = "value_less"
    TIME_BASED = "time_based"
    CUSTOM = "custom"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Condition:
    """Represents a single condition to be evaluated."""
    condition_type: ConditionType
    target: str
    operator: str
    value: Any
    confidence: float = 1.0


@dataclass
class ConditionalBlock:
    """Represents a conditional block with if-then-else logic."""
    conditions: List[Condition]
    logical_operators: List[LogicalOperator]
    then_actions: List[str]
    else_actions: Optional[List[str]] = None
    confidence: float = 1.0


class ConditionalLogicEngine:
    """
    Advanced engine for understanding and processing conditional logic
    from natural language descriptions.
    """
    
    def __init__(self):
        """Initialize the conditional logic engine."""
        self.logger = logging.getLogger(__name__)
        
        # Condition patterns for natural language understanding
        self.condition_patterns = {
            ConditionType.ELEMENT_EXISTS: [
                r"if (.+) exists",
                r"if (.+) is present",
                r"if (.+) is found",
                r"when (.+) appears"
            ],
            ConditionType.ELEMENT_VISIBLE: [
                r"if (.+) is visible",
                r"if (.+) is shown",
                r"if (.+) can be seen",
                r"when (.+) is displayed"
            ],
            ConditionType.TEXT_CONTAINS: [
                r"if (.+) contains (.+)",
                r"if (.+) includes (.+)",
                r"if (.+) has (.+)",
                r"when (.+) shows (.+)"
            ],
            ConditionType.VALUE_EQUALS: [
                r"if (.+) (?:is|equals?) (.+)",
                r"if (.+) = (.+)",
                r"when (.+) (?:is|equals?) (.+)"
            ],
            ConditionType.VALUE_GREATER: [
                r"if (.+) (?:is )?(?:greater than|>) (.+)",
                r"if (.+) (?:is )?(?:more than|above) (.+)",
                r"when (.+) (?:exceeds|is over) (.+)"
            ],
            ConditionType.VALUE_LESS: [
                r"if (.+) (?:is )?(?:less than|<) (.+)",
                r"if (.+) (?:is )?(?:below|under) (.+)",
                r"when (.+) (?:is )?(?:smaller than|fewer than) (.+)"
            ],
            ConditionType.TIME_BASED: [
                r"if (?:it is|the time is) (.+)",
                r"when (?:it is|the time is) (.+)",
                r"if (?:it's|the time's) (.+)",
                r"after (.+) (?:seconds?|minutes?|hours?)"
            ]
        }
        
        # Logical operator patterns
        self.logical_patterns = {
            LogicalOperator.AND: [r"\band\b", r"&", r"&&"],
            LogicalOperator.OR: [r"\bor\b", r"\|", r"\|\|"],
            LogicalOperator.NOT: [r"\bnot\b", r"!", r"~"]
        }
        
        # If-then-else patterns
        self.conditional_patterns = [
            r"if (.+?) then (.+?)(?:\s+else (.+?))?(?:\.|$)",
            r"when (.+?) (?:then )?do (.+?)(?:\s+otherwise (.+?))?(?:\.|$)",
            r"if (.+?) then (.+?)(?:\s+otherwise (.+?))?(?:\.|$)",
            r"(?:should|must) (.+?) if (.+?)(?:\s+else (.+?))?(?:\.|$)"
        ]
    
    async def is_conditional(self, sentence: str) -> bool:
        """
        Check if a sentence contains conditional logic.
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            bool: True if sentence contains conditional logic
        """
        # Check for explicit conditional keywords
        conditional_keywords = [
            r"\bif\b", r"\bwhen\b", r"\bunless\b", r"\bshould\b",
            r"\bthen\b", r"\belse\b", r"\botherwise\b"
        ]
        
        for keyword in conditional_keywords:
            if re.search(keyword, sentence, re.IGNORECASE):
                return True
        
        return False
    
    async def parse_conditional(self, sentence: str, step_index: int) -> List[Dict[str, Any]]:
        """
        Parse conditional logic from a sentence into workflow steps.
        
        Args:
            sentence: Sentence containing conditional logic
            step_index: Index of the step in the workflow
            
        Returns:
            List[Dict]: List of workflow steps representing the conditional logic
        """
        self.logger.info(f"Parsing conditional logic: {sentence}")
        
        try:
            # Extract conditional blocks
            conditional_blocks = await self._extract_conditional_blocks(sentence)
            
            # Convert to workflow steps
            workflow_steps = []
            for i, block in enumerate(conditional_blocks):
                step = await self._conditional_block_to_step(block, step_index + i)
                workflow_steps.append(step)
            
            return workflow_steps
            
        except Exception as e:
            self.logger.error(f"Failed to parse conditional logic: {e}")
            # Return a basic conditional step as fallback
            return [{
                'action_type': 'condition',
                'target': 'unknown',
                'parameters': {
                    'raw_text': sentence,
                    'type': 'conditional'
                },
                'description': sentence,
                'confidence': 0.3
            }]
    
    async def _extract_conditional_blocks(self, sentence: str) -> List[ConditionalBlock]:
        """Extract conditional blocks from a sentence."""
        blocks = []
        
        # Try to match against conditional patterns
        for pattern in self.conditional_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groups()
                
                # Extract condition part
                condition_text = groups[0] if len(groups) > 0 else ""
                then_text = groups[1] if len(groups) > 1 else ""
                else_text = groups[2] if len(groups) > 2 and groups[2] else None
                
                # Parse conditions
                conditions, operators = await self._parse_conditions(condition_text)
                
                # Parse actions
                then_actions = self._parse_actions(then_text)
                else_actions = self._parse_actions(else_text) if else_text else None
                
                # Create conditional block
                block = ConditionalBlock(
                    conditions=conditions,
                    logical_operators=operators,
                    then_actions=then_actions,
                    else_actions=else_actions,
                    confidence=0.8
                )
                
                blocks.append(block)
                break
        
        # If no pattern matched, try to extract simple conditions
        if not blocks:
            simple_conditions, operators = await self._parse_conditions(sentence)
            if simple_conditions:
                block = ConditionalBlock(
                    conditions=simple_conditions,
                    logical_operators=operators,
                    then_actions=["continue"],
                    confidence=0.6
                )
                blocks.append(block)
        
        return blocks
    
    async def _parse_conditions(self, condition_text: str) -> Tuple[List[Condition], List[LogicalOperator]]:
        """Parse conditions and logical operators from text."""
        conditions = []
        operators = []
        
        # Split by logical operators while preserving them
        parts = []
        current_part = ""
        
        for word in condition_text.split():
            is_operator = False
            operator_found = None
            
            # Check if word is a logical operator
            for op, patterns in self.logical_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, word, re.IGNORECASE):
                        is_operator = True
                        operator_found = op
                        break
                if is_operator:
                    break
            
            if is_operator:
                if current_part.strip():
                    parts.append(current_part.strip())
                    operators.append(operator_found)
                current_part = ""
            else:
                current_part += " " + word
        
        # Add the last part
        if current_part.strip():
            parts.append(current_part.strip())
        
        # Parse each condition part
        for part in parts:
            condition = await self._parse_single_condition(part)
            if condition:
                conditions.append(condition)
        
        return conditions, operators
    
    async def _parse_single_condition(self, condition_text: str) -> Optional[Condition]:
        """Parse a single condition from text."""
        # Try to match against condition patterns
        for condition_type, patterns in self.condition_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, condition_text, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if condition_type == ConditionType.ELEMENT_EXISTS:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="exists",
                            value=True,
                            confidence=0.9
                        )
                    
                    elif condition_type == ConditionType.ELEMENT_VISIBLE:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="visible",
                            value=True,
                            confidence=0.9
                        )
                    
                    elif condition_type == ConditionType.TEXT_CONTAINS:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="contains",
                            value=groups[1],
                            confidence=0.85
                        )
                    
                    elif condition_type == ConditionType.VALUE_EQUALS:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="equals",
                            value=self._parse_value(groups[1]),
                            confidence=0.9
                        )
                    
                    elif condition_type == ConditionType.VALUE_GREATER:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="greater_than",
                            value=self._parse_value(groups[1]),
                            confidence=0.85
                        )
                    
                    elif condition_type == ConditionType.VALUE_LESS:
                        return Condition(
                            condition_type=condition_type,
                            target=groups[0],
                            operator="less_than",
                            value=self._parse_value(groups[1]),
                            confidence=0.85
                        )
                    
                    elif condition_type == ConditionType.TIME_BASED:
                        return Condition(
                            condition_type=condition_type,
                            target="time",
                            operator="time_check",
                            value=groups[0],
                            confidence=0.8
                        )
        
        # If no pattern matched, create a custom condition
        return Condition(
            condition_type=ConditionType.CUSTOM,
            target="unknown",
            operator="custom",
            value=condition_text,
            confidence=0.5
        )
    
    def _parse_value(self, value_text: str) -> Any:
        """Parse a value from text into appropriate type."""
        value_text = value_text.strip()
        
        # Try to parse as number
        if value_text.isdigit():
            return int(value_text)
        
        try:
            return float(value_text)
        except ValueError:
            pass
        
        # Try to parse as boolean
        if value_text.lower() in ['true', 'yes', 'on']:
            return True
        elif value_text.lower() in ['false', 'no', 'off']:
            return False
        
        # Return as string
        return value_text
    
    def _parse_actions(self, action_text: str) -> List[str]:
        """Parse actions from text."""
        if not action_text:
            return []
        
        # Split actions by common separators
        actions = re.split(r'[,;]|\band\b', action_text)
        return [action.strip() for action in actions if action.strip()]
    
    async def _conditional_block_to_step(self, block: ConditionalBlock, step_index: int) -> Dict[str, Any]:
        """Convert a conditional block to a workflow step."""
        # Build condition parameters
        condition_params = {
            'conditions': [],
            'logical_operators': [op.value for op in block.logical_operators],
            'then_actions': block.then_actions,
            'else_actions': block.else_actions,
            'type': 'conditional'
        }
        
        # Add individual conditions
        for condition in block.conditions:
            condition_params['conditions'].append({
                'type': condition.condition_type.value,
                'target': condition.target,
                'operator': condition.operator,
                'value': condition.value,
                'confidence': condition.confidence
            })
        
        # Determine primary target
        primary_target = "conditional_logic"
        if block.conditions:
            primary_target = block.conditions[0].target
        
        return {
            'action_type': 'condition',
            'target': primary_target,
            'parameters': condition_params,
            'description': f"Conditional logic with {len(block.conditions)} condition(s)",
            'confidence': block.confidence
        }
    
    async def evaluate_condition(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against the current context.
        
        Args:
            condition: Condition to evaluate
            context: Current execution context
            
        Returns:
            bool: Result of condition evaluation
        """
        try:
            if condition.condition_type == ConditionType.ELEMENT_EXISTS:
                return context.get('elements', {}).get(condition.target, False)
            
            elif condition.condition_type == ConditionType.ELEMENT_VISIBLE:
                element_data = context.get('elements', {}).get(condition.target, {})
                return element_data.get('visible', False)
            
            elif condition.condition_type == ConditionType.TEXT_CONTAINS:
                element_text = context.get('elements', {}).get(condition.target, {}).get('text', '')
                return str(condition.value).lower() in element_text.lower()
            
            elif condition.condition_type == ConditionType.VALUE_EQUALS:
                actual_value = context.get('variables', {}).get(condition.target)
                return actual_value == condition.value
            
            elif condition.condition_type == ConditionType.VALUE_GREATER:
                actual_value = context.get('variables', {}).get(condition.target, 0)
                return float(actual_value) > float(condition.value)
            
            elif condition.condition_type == ConditionType.VALUE_LESS:
                actual_value = context.get('variables', {}).get(condition.target, 0)
                return float(actual_value) < float(condition.value)
            
            elif condition.condition_type == ConditionType.TIME_BASED:
                # Implement time-based condition evaluation
                current_time = context.get('current_time', '')
                return self._evaluate_time_condition(condition.value, current_time)
            
            else:
                # Custom condition - return True by default
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition: {e}")
            return False
    
    def _evaluate_time_condition(self, condition_value: str, current_time: str) -> bool:
        """Evaluate time-based conditions."""
        # Implement time comparison logic
        # This is a simplified implementation
        return True