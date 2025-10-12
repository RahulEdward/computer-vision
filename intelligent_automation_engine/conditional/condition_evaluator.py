"""
Condition Evaluator

Evaluates parsed conditions against runtime context and data to determine
if conditions are met for conditional logic execution.
"""

import logging
import asyncio
import operator
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import json
import math

from .condition_parser import ParsedCondition, ConditionElement, ConditionType, ComparisonOperator, LogicalOperator


class EvaluationStatus(Enum):
    """Status of condition evaluation."""
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"


@dataclass
class EvaluationContext:
    """Context for condition evaluation."""
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    
    # Runtime context
    current_time: datetime = field(default_factory=datetime.now)
    user_context: Dict[str, Any] = field(default_factory=dict)
    application_state: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    strict_mode: bool = False
    case_sensitive: bool = False
    timeout_seconds: float = 5.0
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable value."""
        self.variables[name] = value
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists."""
        return name in self.variables
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a function."""
        return self.functions.get(name)
    
    def register_function(self, name: str, func: Callable):
        """Register a function."""
        self.functions[name] = func
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'variables': self.variables,
            'functions': list(self.functions.keys()),
            'current_time': self.current_time.isoformat(),
            'user_context': self.user_context,
            'application_state': self.application_state,
            'strict_mode': self.strict_mode,
            'case_sensitive': self.case_sensitive,
            'timeout_seconds': self.timeout_seconds
        }


@dataclass
class EvaluationResult:
    """Result of condition evaluation."""
    result: bool
    status: EvaluationStatus
    
    # Evaluation details
    condition_id: Optional[str] = None
    evaluated_expression: str = ""
    
    # Performance
    evaluation_time: float = 0.0
    
    # Context
    variables_used: List[str] = field(default_factory=list)
    functions_called: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Intermediate results
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_variable_used(self, variable: str):
        """Add a variable to the used list."""
        if variable not in self.variables_used:
            self.variables_used.append(variable)
    
    def add_function_called(self, function: str):
        """Add a function to the called list."""
        if function not in self.functions_called:
            self.functions_called.append(function)
    
    def set_error(self, message: str, details: Dict[str, Any] = None):
        """Set error information."""
        self.status = EvaluationStatus.ERROR
        self.error_message = message
        self.error_details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'result': self.result,
            'status': self.status.value,
            'condition_id': self.condition_id,
            'evaluated_expression': self.evaluated_expression,
            'evaluation_time': self.evaluation_time,
            'variables_used': self.variables_used,
            'functions_called': self.functions_called,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'intermediate_results': self.intermediate_results
        }


class ConditionEvaluator:
    """
    Evaluates parsed conditions against runtime context and data to determine
    if conditions are met for conditional logic execution.
    """
    
    def __init__(self):
        """Initialize the condition evaluator."""
        self.logger = logging.getLogger(__name__)
        
        # Operator mappings
        self.comparison_operators = {
            ComparisonOperator.EQUALS: operator.eq,
            ComparisonOperator.NOT_EQUALS: operator.ne,
            ComparisonOperator.GREATER_THAN: operator.gt,
            ComparisonOperator.GREATER_EQUAL: operator.ge,
            ComparisonOperator.LESS_THAN: operator.lt,
            ComparisonOperator.LESS_EQUAL: operator.le,
            ComparisonOperator.CONTAINS: lambda a, b: b in a if hasattr(a, '__contains__') else False,
            ComparisonOperator.NOT_CONTAINS: lambda a, b: b not in a if hasattr(a, '__contains__') else True,
            ComparisonOperator.STARTS_WITH: lambda a, b: str(a).startswith(str(b)),
            ComparisonOperator.ENDS_WITH: lambda a, b: str(a).endswith(str(b)),
            ComparisonOperator.MATCHES: lambda a, b: bool(re.search(str(b), str(a))),
            ComparisonOperator.IN: lambda a, b: a in b if hasattr(b, '__contains__') else False,
            ComparisonOperator.NOT_IN: lambda a, b: a not in b if hasattr(b, '__contains__') else True
        }
        
        self.logical_operators = {
            LogicalOperator.AND: lambda a, b: a and b,
            LogicalOperator.OR: lambda a, b: a or b,
            LogicalOperator.NOT: lambda a: not a
        }
        
        # Built-in functions
        self.builtin_functions = self._initialize_builtin_functions()
        
        # Configuration
        self.config = {
            'max_recursion_depth': 10,
            'enable_type_coercion': True,
            'enable_fuzzy_matching': False,
            'cache_results': True,
            'default_timeout': 5.0
        }
        
        # Cache for evaluation results
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        
        # Statistics
        self.stats = {
            'evaluations_performed': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'cached_evaluations': 0,
            'total_evaluation_time': 0.0,
            'average_evaluation_time': 0.0
        }
    
    def _initialize_builtin_functions(self) -> Dict[str, Callable]:
        """Initialize built-in functions for condition evaluation."""
        return {
            # Math functions
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'floor': math.floor,
            'ceil': math.ceil,
            'sqrt': math.sqrt,
            'pow': pow,
            
            # String functions
            'len': len,
            'lower': lambda s: str(s).lower(),
            'upper': lambda s: str(s).upper(),
            'strip': lambda s: str(s).strip(),
            'split': lambda s, sep=' ': str(s).split(sep),
            'join': lambda sep, items: sep.join(str(item) for item in items),
            'replace': lambda s, old, new: str(s).replace(old, new),
            
            # Type functions
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'type': type,
            'isinstance': isinstance,
            
            # Collection functions
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'any': any,
            'all': all,
            
            # Time functions
            'now': datetime.now,
            'today': datetime.today,
            'time': lambda: datetime.now().time(),
            'date': lambda: datetime.now().date(),
            
            # Utility functions
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'filter': filter,
            'map': map
        }
    
    async def evaluate_condition(self, condition: ParsedCondition, 
                               context: EvaluationContext) -> EvaluationResult:
        """
        Evaluate a parsed condition against the given context.
        
        Args:
            condition: Parsed condition to evaluate
            context: Evaluation context with variables and functions
            
        Returns:
            EvaluationResult: Result of the evaluation
        """
        import time
        start_time = time.time()
        
        # Create result object
        result = EvaluationResult(
            result=False,
            status=EvaluationStatus.PENDING,
            condition_id=condition.condition_id,
            evaluated_expression=condition.original_text
        )
        
        try:
            # Check cache if enabled
            if self.config['cache_results']:
                cache_key = self._generate_cache_key(condition, context)
                if cache_key in self.evaluation_cache:
                    cached_result = self.evaluation_cache[cache_key]
                    self.stats['cached_evaluations'] += 1
                    return cached_result
            
            # Evaluate with timeout
            timeout = context.timeout_seconds or self.config['default_timeout']
            
            evaluation_result = await asyncio.wait_for(
                self._evaluate_condition_internal(condition, context, result),
                timeout=timeout
            )
            
            result.result = evaluation_result
            result.status = EvaluationStatus.SUCCESS
            
            # Cache result if enabled
            if self.config['cache_results']:
                self.evaluation_cache[cache_key] = result
            
            # Update statistics
            self.stats['evaluations_performed'] += 1
            self.stats['successful_evaluations'] += 1
            
        except asyncio.TimeoutError:
            result.status = EvaluationStatus.TIMEOUT
            result.set_error(f"Evaluation timed out after {timeout} seconds")
            self.stats['failed_evaluations'] += 1
            
        except Exception as e:
            result.set_error(str(e), {'exception_type': type(e).__name__})
            self.stats['failed_evaluations'] += 1
            self.logger.error(f"Error evaluating condition: {e}")
        
        finally:
            # Record evaluation time
            result.evaluation_time = time.time() - start_time
            self.stats['total_evaluation_time'] += result.evaluation_time
            
            if self.stats['evaluations_performed'] > 0:
                self.stats['average_evaluation_time'] = (
                    self.stats['total_evaluation_time'] / self.stats['evaluations_performed']
                )
        
        return result
    
    async def _evaluate_condition_internal(self, condition: ParsedCondition,
                                         context: EvaluationContext,
                                         result: EvaluationResult) -> bool:
        """Internal condition evaluation logic."""
        if condition.condition_type == ConditionType.SIMPLE:
            return await self._evaluate_simple_condition(condition, context, result)
        
        elif condition.condition_type == ConditionType.COMPOUND:
            return await self._evaluate_compound_condition(condition, context, result)
        
        elif condition.condition_type == ConditionType.NESTED:
            return await self._evaluate_nested_condition(condition, context, result)
        
        elif condition.condition_type == ConditionType.FUNCTION_CALL:
            return await self._evaluate_function_condition(condition, context, result)
        
        elif condition.condition_type == ConditionType.EXPRESSION:
            return await self._evaluate_expression_condition(condition, context, result)
        
        else:
            raise ValueError(f"Unsupported condition type: {condition.condition_type}")
    
    async def _evaluate_simple_condition(self, condition: ParsedCondition,
                                        context: EvaluationContext,
                                        result: EvaluationResult) -> bool:
        """Evaluate a simple condition."""
        if not condition.elements or len(condition.elements) < 3:
            raise ValueError("Simple condition requires at least 3 elements (left, operator, right)")
        
        left_element = condition.elements[0]
        operator_element = condition.elements[1]
        right_element = condition.elements[2]
        
        # Evaluate left and right operands
        left_value = await self._evaluate_element(left_element, context, result)
        right_value = await self._evaluate_element(right_element, context, result)
        
        # Get comparison operator
        if operator_element.element_type != 'operator':
            raise ValueError(f"Expected operator, got: {operator_element.element_type}")
        
        operator_func = self.comparison_operators.get(operator_element.comparison_operator)
        if not operator_func:
            raise ValueError(f"Unsupported comparison operator: {operator_element.comparison_operator}")
        
        # Perform type coercion if enabled
        if self.config['enable_type_coercion']:
            left_value, right_value = self._coerce_types(left_value, right_value)
        
        # Evaluate comparison
        comparison_result = operator_func(left_value, right_value)
        
        # Store intermediate result
        result.intermediate_results['simple_comparison'] = {
            'left_value': left_value,
            'operator': operator_element.comparison_operator.value,
            'right_value': right_value,
            'result': comparison_result
        }
        
        return bool(comparison_result)
    
    async def _evaluate_compound_condition(self, condition: ParsedCondition,
                                         context: EvaluationContext,
                                         result: EvaluationResult) -> bool:
        """Evaluate a compound condition with logical operators."""
        if not condition.sub_conditions or len(condition.sub_conditions) < 2:
            raise ValueError("Compound condition requires at least 2 sub-conditions")
        
        # Find logical operators in elements
        logical_ops = [elem for elem in condition.elements 
                      if elem.element_type == 'operator' and elem.logical_operator]
        
        if not logical_ops:
            raise ValueError("Compound condition requires logical operators")
        
        # Evaluate sub-conditions
        sub_results = []
        for sub_condition in condition.sub_conditions:
            sub_result = await self._evaluate_condition_internal(sub_condition, context, result)
            sub_results.append(sub_result)
        
        # Apply logical operators
        final_result = sub_results[0]
        
        for i, logical_op in enumerate(logical_ops):
            if i + 1 < len(sub_results):
                operator_func = self.logical_operators.get(logical_op.logical_operator)
                if operator_func:
                    final_result = operator_func(final_result, sub_results[i + 1])
        
        # Store intermediate result
        result.intermediate_results['compound_evaluation'] = {
            'sub_results': sub_results,
            'logical_operators': [op.logical_operator.value for op in logical_ops],
            'final_result': final_result
        }
        
        return final_result
    
    async def _evaluate_nested_condition(self, condition: ParsedCondition,
                                        context: EvaluationContext,
                                        result: EvaluationResult) -> bool:
        """Evaluate a nested condition."""
        if not condition.sub_conditions:
            raise ValueError("Nested condition requires sub-conditions")
        
        # Evaluate all sub-conditions and combine with AND logic by default
        sub_results = []
        for sub_condition in condition.sub_conditions:
            sub_result = await self._evaluate_condition_internal(sub_condition, context, result)
            sub_results.append(sub_result)
        
        # Default to AND logic for nested conditions
        final_result = all(sub_results)
        
        # Store intermediate result
        result.intermediate_results['nested_evaluation'] = {
            'sub_results': sub_results,
            'final_result': final_result
        }
        
        return final_result
    
    async def _evaluate_function_condition(self, condition: ParsedCondition,
                                         context: EvaluationContext,
                                         result: EvaluationResult) -> bool:
        """Evaluate a function call condition."""
        # Find function call element
        function_elements = [elem for elem in condition.elements 
                           if elem.element_type == 'function']
        
        if not function_elements:
            raise ValueError("Function condition requires a function element")
        
        function_element = function_elements[0]
        function_name = function_element.value
        
        # Get function
        func = context.get_function(function_name) or self.builtin_functions.get(function_name)
        if not func:
            raise ValueError(f"Function not found: {function_name}")
        
        result.add_function_called(function_name)
        
        # Evaluate function arguments
        args = []
        for arg_element in function_element.function_args:
            arg_value = await self._evaluate_element(arg_element, context, result)
            args.append(arg_value)
        
        # Call function
        if asyncio.iscoroutinefunction(func):
            function_result = await func(*args)
        else:
            function_result = func(*args)
        
        # Store intermediate result
        result.intermediate_results['function_call'] = {
            'function_name': function_name,
            'arguments': args,
            'result': function_result
        }
        
        return bool(function_result)
    
    async def _evaluate_expression_condition(self, condition: ParsedCondition,
                                           context: EvaluationContext,
                                           result: EvaluationResult) -> bool:
        """Evaluate an expression condition."""
        # This is a simplified expression evaluator
        # In a production system, you might want to use a proper expression parser
        
        expression = condition.original_text
        
        # Replace variables in expression
        for var_name, var_value in context.variables.items():
            # Simple variable replacement (could be improved with proper parsing)
            expression = expression.replace(f"{{{var_name}}}", str(var_value))
            expression = expression.replace(f"${var_name}", str(var_value))
        
        try:
            # Evaluate expression (WARNING: This is unsafe for production use)
            # In production, use a safe expression evaluator
            expression_result = eval(expression, {"__builtins__": {}}, self.builtin_functions)
            
            # Store intermediate result
            result.intermediate_results['expression_evaluation'] = {
                'original_expression': condition.original_text,
                'evaluated_expression': expression,
                'result': expression_result
            }
            
            return bool(expression_result)
            
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")
    
    async def _evaluate_element(self, element: ConditionElement,
                              context: EvaluationContext,
                              result: EvaluationResult) -> Any:
        """Evaluate a condition element."""
        if element.element_type == 'variable':
            result.add_variable_used(element.value)
            
            if context.has_variable(element.value):
                return context.get_variable(element.value)
            elif context.strict_mode:
                raise ValueError(f"Variable not found: {element.value}")
            else:
                return element.value  # Return variable name if not found in non-strict mode
        
        elif element.element_type == 'literal':
            return element.normalized_value if element.normalized_value is not None else element.value
        
        elif element.element_type == 'function':
            # Handle function calls
            function_name = element.value
            func = context.get_function(function_name) or self.builtin_functions.get(function_name)
            
            if not func:
                raise ValueError(f"Function not found: {function_name}")
            
            result.add_function_called(function_name)
            
            # Evaluate function arguments
            args = []
            for arg_element in element.function_args:
                arg_value = await self._evaluate_element(arg_element, context, result)
                args.append(arg_value)
            
            # Call function
            if asyncio.iscoroutinefunction(func):
                return await func(*args)
            else:
                return func(*args)
        
        else:
            return element.value
    
    def _coerce_types(self, left: Any, right: Any) -> Tuple[Any, Any]:
        """Coerce types for comparison."""
        # If both are the same type, no coercion needed
        if type(left) == type(right):
            return left, right
        
        # Try to convert to numbers
        try:
            if isinstance(left, str) and left.replace('.', '').replace('-', '').isdigit():
                left = float(left) if '.' in left else int(left)
            if isinstance(right, str) and right.replace('.', '').replace('-', '').isdigit():
                right = float(right) if '.' in right else int(right)
        except (ValueError, AttributeError):
            pass
        
        # Convert to strings for string operations
        if isinstance(left, str) or isinstance(right, str):
            return str(left), str(right)
        
        return left, right
    
    def _generate_cache_key(self, condition: ParsedCondition, 
                          context: EvaluationContext) -> str:
        """Generate a cache key for condition and context."""
        # Create a hash of the condition and relevant context
        condition_str = f"{condition.condition_id}:{condition.original_text}"
        context_str = json.dumps(context.variables, sort_keys=True, default=str)
        
        import hashlib
        return hashlib.md5(f"{condition_str}:{context_str}".encode()).hexdigest()
    
    def register_function(self, name: str, func: Callable):
        """Register a custom function."""
        self.builtin_functions[name] = func
        self.logger.debug(f"Registered function: {name}")
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self.evaluation_cache.clear()
        self.logger.debug("Evaluation cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            **self.stats,
            'cache_size': len(self.evaluation_cache),
            'builtin_functions': len(self.builtin_functions),
            'comparison_operators': len(self.comparison_operators),
            'logical_operators': len(self.logical_operators)
        }