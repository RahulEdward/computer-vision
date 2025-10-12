"""
Condition Parser

Parses natural language conditional statements into structured representations
that can be evaluated by the logic engine.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class ConditionType(Enum):
    """Types of conditions."""
    SIMPLE = "simple"
    COMPOUND = "compound"
    NESTED = "nested"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    COMPARISON = "comparison"
    EXISTENCE = "existence"
    PATTERN = "pattern"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"
    IMPLIES = "implies"


class ComparisonOperator(Enum):
    """Comparison operators."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"
    IN = "in"
    NOT_IN = "not_in"


@dataclass
class ConditionElement:
    """Represents an element in a condition."""
    element_type: str  # variable, value, function, expression
    value: Any
    data_type: str = "string"  # string, number, boolean, date, object
    
    # Context information
    source: Optional[str] = None  # Where this element comes from
    confidence: float = 1.0
    
    # Metadata
    original_text: Optional[str] = None
    position: Optional[Tuple[int, int]] = None  # Start and end position in text


@dataclass
class ParsedCondition:
    """Represents a parsed condition."""
    condition_id: str
    condition_type: ConditionType
    
    # Core condition components
    left_operand: Optional[ConditionElement] = None
    operator: Optional[Union[ComparisonOperator, LogicalOperator]] = None
    right_operand: Optional[ConditionElement] = None
    
    # For compound conditions
    sub_conditions: List['ParsedCondition'] = field(default_factory=list)
    logical_operator: Optional[LogicalOperator] = None
    
    # Negation
    is_negated: bool = False
    
    # Context and metadata
    original_text: str = ""
    confidence: float = 1.0
    variables: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    
    # Temporal information
    temporal_context: Optional[str] = None
    
    def is_simple(self) -> bool:
        """Check if this is a simple condition."""
        return self.condition_type == ConditionType.SIMPLE and not self.sub_conditions
    
    def is_compound(self) -> bool:
        """Check if this is a compound condition."""
        return len(self.sub_conditions) > 0
    
    def get_all_variables(self) -> List[str]:
        """Get all variables used in this condition."""
        variables = set(self.variables)
        
        for sub_condition in self.sub_conditions:
            variables.update(sub_condition.get_all_variables())
        
        return list(variables)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'condition_id': self.condition_id,
            'condition_type': self.condition_type.value,
            'left_operand': self.left_operand.__dict__ if self.left_operand else None,
            'operator': self.operator.value if self.operator else None,
            'right_operand': self.right_operand.__dict__ if self.right_operand else None,
            'sub_conditions': [sc.to_dict() for sc in self.sub_conditions],
            'logical_operator': self.logical_operator.value if self.logical_operator else None,
            'is_negated': self.is_negated,
            'original_text': self.original_text,
            'confidence': self.confidence,
            'variables': self.variables,
            'functions': self.functions,
            'temporal_context': self.temporal_context
        }


@dataclass
class ConditionContext:
    """Context for condition parsing."""
    context_id: str
    
    # Available variables and their types
    variables: Dict[str, str] = field(default_factory=dict)
    
    # Available functions
    functions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Domain-specific context
    domain: Optional[str] = None
    application: Optional[str] = None
    
    # Temporal context
    current_time: Optional[datetime] = None
    timezone: Optional[str] = None
    
    # User context
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    
    # Parsing configuration
    strict_mode: bool = False
    allow_fuzzy_matching: bool = True
    confidence_threshold: float = 0.7


class ConditionParser:
    """
    Parses natural language conditional statements into structured representations.
    """
    
    def __init__(self):
        """Initialize the condition parser."""
        self.logger = logging.getLogger(__name__)
        
        # Parsing patterns
        self.patterns = self._initialize_patterns()
        
        # Operator mappings
        self.comparison_mappings = self._initialize_comparison_mappings()
        self.logical_mappings = self._initialize_logical_mappings()
        
        # Data type patterns
        self.data_type_patterns = self._initialize_data_type_patterns()
        
        # Statistics
        self.stats = {
            'conditions_parsed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'average_confidence': 0.0
        }
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize parsing patterns."""
        return {
            'if_patterns': [
                r'if\s+(.+?)\s+then',
                r'when\s+(.+?)\s+then',
                r'whenever\s+(.+?)\s+do',
                r'in case\s+(.+?)\s+then',
                r'provided that\s+(.+?)\s+then'
            ],
            
            'then_patterns': [
                r'then\s+(.+?)(?:\s+else|\s+otherwise|$)',
                r'do\s+(.+?)(?:\s+else|\s+otherwise|$)',
                r'execute\s+(.+?)(?:\s+else|\s+otherwise|$)',
                r'perform\s+(.+?)(?:\s+else|\s+otherwise|$)'
            ],
            
            'else_patterns': [
                r'else\s+(.+?)$',
                r'otherwise\s+(.+?)$',
                r'if not\s+(.+?)$',
                r'alternatively\s+(.+?)$'
            ],
            
            'comparison_patterns': [
                r'(.+?)\s+(is|equals?|==)\s+(.+)',
                r'(.+?)\s+(is not|!=|<>)\s+(.+)',
                r'(.+?)\s+(greater than|>)\s+(.+)',
                r'(.+?)\s+(less than|<)\s+(.+)',
                r'(.+?)\s+(greater than or equal to|>=)\s+(.+)',
                r'(.+?)\s+(less than or equal to|<=)\s+(.+)',
                r'(.+?)\s+(contains|includes)\s+(.+)',
                r'(.+?)\s+(starts with|begins with)\s+(.+)',
                r'(.+?)\s+(ends with|finishes with)\s+(.+)',
                r'(.+?)\s+(matches|fits)\s+(.+)',
                r'(.+?)\s+(is in|in)\s+(.+)',
                r'(.+?)\s+(is not in|not in)\s+(.+)'
            ],
            
            'logical_patterns': [
                r'(.+?)\s+(and|&)\s+(.+)',
                r'(.+?)\s+(or|\|)\s+(.+)',
                r'not\s+(.+)',
                r'(.+?)\s+(xor)\s+(.+)',
                r'(.+?)\s+(implies|means)\s+(.+)'
            ],
            
            'existence_patterns': [
                r'(.+?)\s+(exists|is present)',
                r'(.+?)\s+(does not exist|is not present|is missing)',
                r'there is\s+(.+)',
                r'there is no\s+(.+)',
                r'(.+?)\s+is\s+(empty|blank)',
                r'(.+?)\s+is not\s+(empty|blank)'
            ],
            
            'temporal_patterns': [
                r'(.+?)\s+(before|after|during)\s+(.+)',
                r'(.+?)\s+(at|on|in)\s+(.+)',
                r'(.+?)\s+(within|for)\s+(.+)',
                r'(.+?)\s+(since|until)\s+(.+)'
            ]
        }
    
    def _initialize_comparison_mappings(self) -> Dict[str, ComparisonOperator]:
        """Initialize comparison operator mappings."""
        return {
            'is': ComparisonOperator.EQUALS,
            'equals': ComparisonOperator.EQUALS,
            'equal': ComparisonOperator.EQUALS,
            '==': ComparisonOperator.EQUALS,
            '=': ComparisonOperator.EQUALS,
            
            'is not': ComparisonOperator.NOT_EQUALS,
            'not equals': ComparisonOperator.NOT_EQUALS,
            '!=': ComparisonOperator.NOT_EQUALS,
            '<>': ComparisonOperator.NOT_EQUALS,
            
            'greater than': ComparisonOperator.GREATER_THAN,
            '>': ComparisonOperator.GREATER_THAN,
            'more than': ComparisonOperator.GREATER_THAN,
            'above': ComparisonOperator.GREATER_THAN,
            
            'less than': ComparisonOperator.LESS_THAN,
            '<': ComparisonOperator.LESS_THAN,
            'below': ComparisonOperator.LESS_THAN,
            'under': ComparisonOperator.LESS_THAN,
            
            'greater than or equal to': ComparisonOperator.GREATER_EQUAL,
            '>=': ComparisonOperator.GREATER_EQUAL,
            'at least': ComparisonOperator.GREATER_EQUAL,
            
            'less than or equal to': ComparisonOperator.LESS_EQUAL,
            '<=': ComparisonOperator.LESS_EQUAL,
            'at most': ComparisonOperator.LESS_EQUAL,
            
            'contains': ComparisonOperator.CONTAINS,
            'includes': ComparisonOperator.CONTAINS,
            'has': ComparisonOperator.CONTAINS,
            
            'starts with': ComparisonOperator.STARTS_WITH,
            'begins with': ComparisonOperator.STARTS_WITH,
            
            'ends with': ComparisonOperator.ENDS_WITH,
            'finishes with': ComparisonOperator.ENDS_WITH,
            
            'matches': ComparisonOperator.MATCHES,
            'fits': ComparisonOperator.MATCHES,
            
            'in': ComparisonOperator.IN,
            'is in': ComparisonOperator.IN,
            
            'not in': ComparisonOperator.NOT_IN,
            'is not in': ComparisonOperator.NOT_IN
        }
    
    def _initialize_logical_mappings(self) -> Dict[str, LogicalOperator]:
        """Initialize logical operator mappings."""
        return {
            'and': LogicalOperator.AND,
            '&': LogicalOperator.AND,
            '&&': LogicalOperator.AND,
            'also': LogicalOperator.AND,
            'plus': LogicalOperator.AND,
            
            'or': LogicalOperator.OR,
            '|': LogicalOperator.OR,
            '||': LogicalOperator.OR,
            'alternatively': LogicalOperator.OR,
            'either': LogicalOperator.OR,
            
            'not': LogicalOperator.NOT,
            '!': LogicalOperator.NOT,
            'negate': LogicalOperator.NOT,
            
            'xor': LogicalOperator.XOR,
            'exclusive or': LogicalOperator.XOR,
            
            'implies': LogicalOperator.IMPLIES,
            'means': LogicalOperator.IMPLIES,
            'suggests': LogicalOperator.IMPLIES
        }
    
    def _initialize_data_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize data type detection patterns."""
        return {
            'number': [
                r'^\d+$',  # Integer
                r'^\d+\.\d+$',  # Decimal
                r'^-?\d+$',  # Negative integer
                r'^-?\d+\.\d+$',  # Negative decimal
                r'^\d+%$',  # Percentage
                r'^\$\d+(\.\d+)?$'  # Currency
            ],
            
            'boolean': [
                r'^(true|false)$',
                r'^(yes|no)$',
                r'^(on|off)$',
                r'^(enabled|disabled)$',
                r'^(active|inactive)$'
            ],
            
            'date': [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
                r'^(today|tomorrow|yesterday)$',
                r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$'
            ],
            
            'time': [
                r'^\d{2}:\d{2}$',  # HH:MM
                r'^\d{2}:\d{2}:\d{2}$',  # HH:MM:SS
                r'^\d{1,2}:\d{2}\s?(am|pm)$',  # 12-hour format
                r'^(now|noon|midnight)$'
            ]
        }
    
    def parse_condition(self, text: str, context: Optional[ConditionContext] = None) -> ParsedCondition:
        """
        Parse a natural language condition into a structured representation.
        
        Args:
            text: Natural language condition text
            context: Parsing context
            
        Returns:
            ParsedCondition: Parsed condition structure
        """
        try:
            self.stats['conditions_parsed'] += 1
            
            # Clean and normalize text
            normalized_text = self._normalize_text(text)
            
            # Create condition ID
            condition_id = f"condition_{datetime.now().timestamp()}"
            
            # Detect condition type
            condition_type = self._detect_condition_type(normalized_text)
            
            # Parse based on type
            if condition_type == ConditionType.COMPOUND:
                parsed_condition = self._parse_compound_condition(
                    condition_id, normalized_text, context
                )
            elif condition_type == ConditionType.NESTED:
                parsed_condition = self._parse_nested_condition(
                    condition_id, normalized_text, context
                )
            else:
                parsed_condition = self._parse_simple_condition(
                    condition_id, normalized_text, context
                )
            
            # Set original text and type
            parsed_condition.original_text = text
            parsed_condition.condition_type = condition_type
            
            # Calculate confidence
            parsed_condition.confidence = self._calculate_confidence(parsed_condition, context)
            
            # Extract variables and functions
            parsed_condition.variables = self._extract_variables(parsed_condition, context)
            parsed_condition.functions = self._extract_functions(parsed_condition, context)
            
            self.stats['successful_parses'] += 1
            self.logger.debug(f"Successfully parsed condition: {text}")
            
            return parsed_condition
            
        except Exception as e:
            self.stats['failed_parses'] += 1
            self.logger.error(f"Failed to parse condition '{text}': {e}")
            
            # Return error condition
            return ParsedCondition(
                condition_id=f"error_{datetime.now().timestamp()}",
                condition_type=ConditionType.SIMPLE,
                original_text=text,
                confidence=0.0
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for parsing."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = text.replace(',', ' ')
        text = text.replace(';', ' ')
        
        # Handle contractions
        contractions = {
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "doesn't": "does not",
            "don't": "do not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "can't": "cannot"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _detect_condition_type(self, text: str) -> ConditionType:
        """Detect the type of condition."""
        # Check for logical operators (compound)
        logical_words = ['and', 'or', '&', '|', '&&', '||']
        if any(word in text for word in logical_words):
            return ConditionType.COMPOUND
        
        # Check for nested parentheses
        if '(' in text and ')' in text:
            return ConditionType.NESTED
        
        # Check for temporal keywords
        temporal_words = ['before', 'after', 'during', 'at', 'on', 'in', 'within', 'since', 'until']
        if any(word in text for word in temporal_words):
            return ConditionType.TEMPORAL
        
        # Check for existence patterns
        existence_words = ['exists', 'is present', 'is missing', 'there is', 'empty', 'blank']
        if any(word in text for word in existence_words):
            return ConditionType.EXISTENCE
        
        # Check for comparison operators
        comparison_words = ['is', 'equals', '==', '!=', '>', '<', '>=', '<=', 'contains', 'matches']
        if any(word in text for word in comparison_words):
            return ConditionType.COMPARISON
        
        # Default to simple
        return ConditionType.SIMPLE
    
    def _parse_simple_condition(self, condition_id: str, text: str,
                               context: Optional[ConditionContext]) -> ParsedCondition:
        """Parse a simple condition."""
        condition = ParsedCondition(
            condition_id=condition_id,
            condition_type=ConditionType.SIMPLE
        )
        
        # Try comparison patterns
        for pattern in self.patterns['comparison_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                left_text = match.group(1).strip()
                operator_text = match.group(2).strip()
                right_text = match.group(3).strip()
                
                # Create operands
                condition.left_operand = self._create_condition_element(left_text, context)
                condition.right_operand = self._create_condition_element(right_text, context)
                
                # Map operator
                condition.operator = self.comparison_mappings.get(
                    operator_text.lower(), ComparisonOperator.EQUALS
                )
                
                return condition
        
        # Try existence patterns
        for pattern in self.patterns['existence_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                element_text = match.group(1).strip()
                condition.left_operand = self._create_condition_element(element_text, context)
                
                # Determine if it's negated
                if any(neg in text for neg in ['not', 'no', 'missing']):
                    condition.is_negated = True
                
                return condition
        
        # Fallback: treat as simple boolean expression
        condition.left_operand = self._create_condition_element(text, context)
        return condition
    
    def _parse_compound_condition(self, condition_id: str, text: str,
                                context: Optional[ConditionContext]) -> ParsedCondition:
        """Parse a compound condition with logical operators."""
        condition = ParsedCondition(
            condition_id=condition_id,
            condition_type=ConditionType.COMPOUND
        )
        
        # Find logical operators
        for pattern in self.patterns['logical_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # Binary operator
                    left_text = match.group(1).strip()
                    operator_text = match.group(2).strip()
                    right_text = match.group(3).strip()
                    
                    # Parse sub-conditions
                    left_condition = self.parse_condition(left_text, context)
                    right_condition = self.parse_condition(right_text, context)
                    
                    condition.sub_conditions = [left_condition, right_condition]
                    condition.logical_operator = self.logical_mappings.get(
                        operator_text.lower(), LogicalOperator.AND
                    )
                    
                elif len(match.groups()) == 1:  # Unary operator (NOT)
                    operand_text = match.group(1).strip()
                    operand_condition = self.parse_condition(operand_text, context)
                    
                    condition.sub_conditions = [operand_condition]
                    condition.logical_operator = LogicalOperator.NOT
                    condition.is_negated = True
                
                return condition
        
        # Fallback: split on common logical words
        for logical_word in ['and', 'or']:
            if logical_word in text:
                parts = text.split(logical_word, 1)
                if len(parts) == 2:
                    left_condition = self.parse_condition(parts[0].strip(), context)
                    right_condition = self.parse_condition(parts[1].strip(), context)
                    
                    condition.sub_conditions = [left_condition, right_condition]
                    condition.logical_operator = self.logical_mappings[logical_word]
                    
                    return condition
        
        # If no logical operator found, treat as simple
        return self._parse_simple_condition(condition_id, text, context)
    
    def _parse_nested_condition(self, condition_id: str, text: str,
                              context: Optional[ConditionContext]) -> ParsedCondition:
        """Parse a nested condition with parentheses."""
        condition = ParsedCondition(
            condition_id=condition_id,
            condition_type=ConditionType.NESTED
        )
        
        # Extract parenthesized expressions
        parentheses_pattern = r'\(([^)]+)\)'
        matches = re.findall(parentheses_pattern, text)
        
        if matches:
            # Parse each parenthesized expression
            for match in matches:
                sub_condition = self.parse_condition(match, context)
                condition.sub_conditions.append(sub_condition)
            
            # Remove parentheses and parse the rest
            remaining_text = re.sub(parentheses_pattern, 'SUBCONDITION', text)
            
            # Find logical operators in remaining text
            for logical_word in ['and', 'or']:
                if logical_word in remaining_text:
                    condition.logical_operator = self.logical_mappings[logical_word]
                    break
        
        return condition
    
    def _create_condition_element(self, text: str, 
                                context: Optional[ConditionContext]) -> ConditionElement:
        """Create a condition element from text."""
        # Detect data type
        data_type = self._detect_data_type(text)
        
        # Convert value based on type
        value = self._convert_value(text, data_type)
        
        # Determine element type
        element_type = self._determine_element_type(text, context)
        
        return ConditionElement(
            element_type=element_type,
            value=value,
            data_type=data_type,
            original_text=text,
            confidence=0.8  # Default confidence
        )
    
    def _detect_data_type(self, text: str) -> str:
        """Detect the data type of a text value."""
        text = text.strip()
        
        # Check each data type pattern
        for data_type, patterns in self.data_type_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return data_type
        
        # Default to string
        return "string"
    
    def _convert_value(self, text: str, data_type: str) -> Any:
        """Convert text value to appropriate Python type."""
        text = text.strip()
        
        try:
            if data_type == "number":
                if '.' in text:
                    return float(text.replace('$', '').replace('%', ''))
                else:
                    return int(text.replace('$', '').replace('%', ''))
            
            elif data_type == "boolean":
                true_values = ['true', 'yes', 'on', 'enabled', 'active']
                return text.lower() in true_values
            
            elif data_type in ["date", "time"]:
                # Return as string for now, can be parsed later
                return text
            
            else:
                # Remove quotes if present
                if text.startswith('"') and text.endswith('"'):
                    return text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    return text[1:-1]
                else:
                    return text
                    
        except (ValueError, TypeError):
            return text  # Fallback to string
    
    def _determine_element_type(self, text: str, 
                              context: Optional[ConditionContext]) -> str:
        """Determine the type of condition element."""
        # Check if it's a variable (from context)
        if context and text in context.variables:
            return "variable"
        
        # Check if it's a function call
        if '(' in text and ')' in text:
            return "function"
        
        # Check if it's a complex expression
        if any(op in text for op in ['+', '-', '*', '/', '%']):
            return "expression"
        
        # Default to value
        return "value"
    
    def _calculate_confidence(self, condition: ParsedCondition,
                            context: Optional[ConditionContext]) -> float:
        """Calculate confidence score for parsed condition."""
        confidence = 1.0
        
        # Reduce confidence for missing operators
        if condition.is_simple() and not condition.operator:
            confidence -= 0.2
        
        # Reduce confidence for unknown variables
        if context:
            for var in condition.get_all_variables():
                if var not in context.variables:
                    confidence -= 0.1
        
        # Reduce confidence for complex nested conditions
        if condition.condition_type == ConditionType.NESTED:
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _extract_variables(self, condition: ParsedCondition,
                         context: Optional[ConditionContext]) -> List[str]:
        """Extract variable names from condition."""
        variables = []
        
        # Extract from operands
        if condition.left_operand and condition.left_operand.element_type == "variable":
            variables.append(condition.left_operand.value)
        
        if condition.right_operand and condition.right_operand.element_type == "variable":
            variables.append(condition.right_operand.value)
        
        # Extract from sub-conditions
        for sub_condition in condition.sub_conditions:
            variables.extend(sub_condition.get_all_variables())
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_functions(self, condition: ParsedCondition,
                         context: Optional[ConditionContext]) -> List[str]:
        """Extract function names from condition."""
        functions = []
        
        # Extract from operands
        if condition.left_operand and condition.left_operand.element_type == "function":
            func_name = condition.left_operand.value.split('(')[0]
            functions.append(func_name)
        
        if condition.right_operand and condition.right_operand.element_type == "function":
            func_name = condition.right_operand.value.split('(')[0]
            functions.append(func_name)
        
        # Extract from sub-conditions
        for sub_condition in condition.sub_conditions:
            functions.extend(sub_condition.functions)
        
        return list(set(functions))  # Remove duplicates
    
    def parse_if_then_else(self, text: str, 
                          context: Optional[ConditionContext] = None) -> Dict[str, ParsedCondition]:
        """
        Parse a complete if-then-else statement.
        
        Args:
            text: Natural language if-then-else statement
            context: Parsing context
            
        Returns:
            Dict containing 'if', 'then', and optionally 'else' conditions
        """
        try:
            result = {}
            
            # Extract IF condition
            for pattern in self.patterns['if_patterns']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if_text = match.group(1).strip()
                    result['if'] = self.parse_condition(if_text, context)
                    break
            
            # Extract THEN action
            for pattern in self.patterns['then_patterns']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    then_text = match.group(1).strip()
                    result['then'] = self.parse_condition(then_text, context)
                    break
            
            # Extract ELSE action (optional)
            for pattern in self.patterns['else_patterns']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    else_text = match.group(1).strip()
                    result['else'] = self.parse_condition(else_text, context)
                    break
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse if-then-else statement: {e}")
            return {}
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        total_parses = self.stats['conditions_parsed']
        success_rate = self.stats['successful_parses'] / max(1, total_parses)
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'patterns_loaded': len(self.patterns),
            'comparison_operators': len(self.comparison_mappings),
            'logical_operators': len(self.logical_mappings)
        }