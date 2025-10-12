"""Conditional Logic Understanding Module

This module provides natural language understanding of conditional logic statements
like "if X then Y else Z" for intelligent automation workflows.

Key Components:
- ConditionParser: Parses natural language conditional statements
- LogicEngine: Executes conditional logic and manages flow
- NaturalLanguageProcessor: Processes natural language for conditional patterns
- ConditionEvaluator: Evaluates conditions against runtime context

Features:
- Natural language parsing of if-then-else statements
- Support for complex nested conditions
- Logical operators (AND, OR, NOT)
- Comparison operators (equals, greater than, contains, etc.)
- Variable and function support
- Context-aware evaluation
- Performance optimization and caching
"""

from .condition_parser import (
    ConditionParser,
    ParsedCondition,
    ConditionElement,
    ConditionContext,
    ConditionType,
    LogicalOperator,
    ComparisonOperator
)

from .logic_engine import (
    LogicEngine,
    ConditionalStatement,
    ConditionalBranch,
    ExecutionPath,
    LogicResult,
    BranchType,
    ExecutionStatus
)

from .nlp_processor import (
    NaturalLanguageProcessor,
    ProcessedText,
    LanguageIntent,
    LanguageEntity,
    LanguagePattern,
    IntentType
)

from .condition_evaluator import (
    ConditionEvaluator,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus
)

__all__ = [
    # Core classes
    'ConditionParser',
    'LogicEngine', 
    'NaturalLanguageProcessor',
    'ConditionEvaluator',
    
    # Data structures
    'ParsedCondition',
    'ConditionElement',
    'ConditionContext',
    'ConditionalStatement',
    'ConditionalBranch',
    'ExecutionPath',
    'LogicResult',
    'ProcessedText',
    'LanguageIntent',
    'LanguageEntity',
    'EvaluationContext',
    'EvaluationResult',
    
    # Enums
    'ConditionType',
    'LogicalOperator',
    'ComparisonOperator',
    'BranchType',
    'ExecutionStatus',
    'LanguagePattern',
    'IntentType',
    'EvaluationStatus'
]