"""
Logic Engine

Executes conditional logic and manages the flow of automation based on
parsed conditions and their evaluation results.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid

from .condition_parser import ParsedCondition, ConditionContext


class BranchType(Enum):
    """Types of conditional branches."""
    IF = "if"
    THEN = "then"
    ELSE = "else"
    ELIF = "elif"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"


class ExecutionStatus(Enum):
    """Execution status of conditional statements."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class ConditionalBranch:
    """Represents a branch in conditional logic."""
    branch_id: str
    branch_type: BranchType
    condition: Optional[ParsedCondition] = None
    
    # Actions to execute
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Nested branches
    sub_branches: List['ConditionalBranch'] = field(default_factory=list)
    
    # Execution control
    is_active: bool = True
    priority: int = 0
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def add_action(self, action: Dict[str, Any]):
        """Add an action to this branch."""
        self.actions.append(action)
    
    def add_sub_branch(self, branch: 'ConditionalBranch'):
        """Add a sub-branch."""
        self.sub_branches.append(branch)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'branch_id': self.branch_id,
            'branch_type': self.branch_type.value,
            'condition': self.condition.to_dict() if self.condition else None,
            'actions': self.actions,
            'sub_branches': [sb.to_dict() for sb in self.sub_branches],
            'is_active': self.is_active,
            'priority': self.priority,
            'description': self.description,
            'tags': self.tags
        }


@dataclass
class ExecutionPath:
    """Represents a path of execution through conditional logic."""
    path_id: str
    branches_executed: List[str] = field(default_factory=list)
    actions_executed: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Results
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Context
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_branch(self, branch_id: str):
        """Add a branch to the execution path."""
        self.branches_executed.append(branch_id)
    
    def add_action(self, action: Dict[str, Any]):
        """Add an action to the execution path."""
        self.actions_executed.append(action)
    
    def add_result(self, result: Any):
        """Add a result to the execution path."""
        self.results.append(result)
    
    def add_error(self, error: str):
        """Add an error to the execution path."""
        self.errors.append(error)
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED and not self.errors


@dataclass
class ConditionalStatement:
    """Represents a complete conditional statement."""
    statement_id: str
    name: str
    description: str = ""
    
    # Branches
    branches: List[ConditionalBranch] = field(default_factory=list)
    
    # Execution configuration
    execution_mode: str = "sequential"  # sequential, parallel, optimized
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    
    # State
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_branch(self, branch: ConditionalBranch):
        """Add a branch to the statement."""
        self.branches.append(branch)
    
    def get_branch_by_type(self, branch_type: BranchType) -> Optional[ConditionalBranch]:
        """Get the first branch of a specific type."""
        for branch in self.branches:
            if branch.branch_type == branch_type:
                return branch
        return None
    
    def get_branches_by_type(self, branch_type: BranchType) -> List[ConditionalBranch]:
        """Get all branches of a specific type."""
        return [branch for branch in self.branches if branch.branch_type == branch_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'statement_id': self.statement_id,
            'name': self.name,
            'description': self.description,
            'branches': [branch.to_dict() for branch in self.branches],
            'execution_mode': self.execution_mode,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'variables': self.variables
        }


@dataclass
class LogicResult:
    """Result of logic engine execution."""
    result_id: str
    statement_id: str
    
    # Execution details
    execution_path: ExecutionPath
    final_result: Any = None
    
    # Performance metrics
    total_duration: float = 0.0
    branches_evaluated: int = 0
    actions_executed: int = 0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Context
    final_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'result_id': self.result_id,
            'statement_id': self.statement_id,
            'execution_path': {
                'path_id': self.execution_path.path_id,
                'branches_executed': self.execution_path.branches_executed,
                'actions_executed': len(self.execution_path.actions_executed),
                'status': self.execution_path.status.value,
                'duration': self.execution_path.get_duration(),
                'errors': self.execution_path.errors
            },
            'final_result': self.final_result,
            'total_duration': self.total_duration,
            'branches_evaluated': self.branches_evaluated,
            'actions_executed': self.actions_executed,
            'success': self.success,
            'error_message': self.error_message,
            'final_context': self.final_context
        }


class LogicEngine:
    """
    Executes conditional logic and manages the flow of automation based on
    parsed conditions and their evaluation results.
    """
    
    def __init__(self):
        """Initialize the logic engine."""
        self.logger = logging.getLogger(__name__)
        
        # Conditional statements
        self.statements: Dict[str, ConditionalStatement] = {}
        
        # Execution history
        self.execution_history: List[LogicResult] = []
        
        # Action handlers
        self.action_handlers: Dict[str, Callable] = {}
        
        # Condition evaluator (will be injected)
        self.condition_evaluator = None
        
        # Configuration
        self.config = {
            'max_execution_depth': 10,
            'default_timeout': 30.0,
            'parallel_execution': True,
            'optimize_execution': True,
            'max_history_size': 1000
        }
        
        # Statistics
        self.stats = {
            'statements_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def set_condition_evaluator(self, evaluator):
        """Set the condition evaluator."""
        self.condition_evaluator = evaluator
    
    def register_action_handler(self, action_type: str, handler: Callable):
        """Register an action handler."""
        self.action_handlers[action_type] = handler
        self.logger.debug(f"Registered action handler for: {action_type}")
    
    def create_statement(self, name: str, description: str = "") -> ConditionalStatement:
        """Create a new conditional statement."""
        statement_id = str(uuid.uuid4())
        statement = ConditionalStatement(
            statement_id=statement_id,
            name=name,
            description=description
        )
        
        self.statements[statement_id] = statement
        self.logger.debug(f"Created conditional statement: {name}")
        
        return statement
    
    def create_if_then_else_statement(self, name: str, 
                                    if_condition: ParsedCondition,
                                    then_actions: List[Dict[str, Any]],
                                    else_actions: Optional[List[Dict[str, Any]]] = None) -> ConditionalStatement:
        """Create a simple if-then-else statement."""
        statement = self.create_statement(name, "Simple if-then-else statement")
        
        # Create IF branch
        if_branch = ConditionalBranch(
            branch_id=str(uuid.uuid4()),
            branch_type=BranchType.IF,
            condition=if_condition
        )
        
        # Create THEN branch
        then_branch = ConditionalBranch(
            branch_id=str(uuid.uuid4()),
            branch_type=BranchType.THEN,
            actions=then_actions
        )
        
        # Create ELSE branch if provided
        if else_actions:
            else_branch = ConditionalBranch(
                branch_id=str(uuid.uuid4()),
                branch_type=BranchType.ELSE,
                actions=else_actions
            )
            statement.add_branch(else_branch)
        
        statement.add_branch(if_branch)
        statement.add_branch(then_branch)
        
        return statement
    
    def create_switch_statement(self, name: str, 
                              switch_condition: ParsedCondition,
                              cases: Dict[Any, List[Dict[str, Any]]],
                              default_actions: Optional[List[Dict[str, Any]]] = None) -> ConditionalStatement:
        """Create a switch statement."""
        statement = self.create_statement(name, "Switch statement")
        
        # Create SWITCH branch
        switch_branch = ConditionalBranch(
            branch_id=str(uuid.uuid4()),
            branch_type=BranchType.SWITCH,
            condition=switch_condition
        )
        statement.add_branch(switch_branch)
        
        # Create CASE branches
        for case_value, case_actions in cases.items():
            case_branch = ConditionalBranch(
                branch_id=str(uuid.uuid4()),
                branch_type=BranchType.CASE,
                actions=case_actions,
                description=f"Case: {case_value}"
            )
            case_branch.tags.append(f"case_value:{case_value}")
            statement.add_branch(case_branch)
        
        # Create DEFAULT branch if provided
        if default_actions:
            default_branch = ConditionalBranch(
                branch_id=str(uuid.uuid4()),
                branch_type=BranchType.DEFAULT,
                actions=default_actions
            )
            statement.add_branch(default_branch)
        
        return statement
    
    async def execute_statement(self, statement_id: str, 
                              context: Dict[str, Any] = None) -> LogicResult:
        """
        Execute a conditional statement.
        
        Args:
            statement_id: ID of the statement to execute
            context: Execution context
            
        Returns:
            LogicResult: Execution result
        """
        start_time = datetime.now()
        
        try:
            # Get statement
            if statement_id not in self.statements:
                raise ValueError(f"Statement not found: {statement_id}")
            
            statement = self.statements[statement_id]
            
            if not statement.is_active:
                raise ValueError(f"Statement is not active: {statement_id}")
            
            # Create execution path
            execution_path = ExecutionPath(
                path_id=str(uuid.uuid4()),
                start_time=start_time,
                status=ExecutionStatus.EXECUTING,
                execution_context=context or {}
            )
            
            self.logger.info(f"Executing conditional statement: {statement.name}")
            
            # Execute based on statement structure
            final_result = await self._execute_statement_logic(statement, execution_path, context or {})
            
            # Complete execution
            execution_path.end_time = datetime.now()
            execution_path.status = ExecutionStatus.COMPLETED
            
            # Create result
            result = LogicResult(
                result_id=str(uuid.uuid4()),
                statement_id=statement_id,
                execution_path=execution_path,
                final_result=final_result,
                total_duration=execution_path.get_duration() or 0.0,
                branches_evaluated=len(execution_path.branches_executed),
                actions_executed=len(execution_path.actions_executed),
                success=True,
                final_context=execution_path.execution_context
            )
            
            # Update statistics
            self.stats['statements_executed'] += 1
            self.stats['successful_executions'] += 1
            self.stats['total_execution_time'] += result.total_duration
            self.stats['average_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['statements_executed']
            )
            
            # Store in history
            self.execution_history.append(result)
            if len(self.execution_history) > self.config['max_history_size']:
                self.execution_history = self.execution_history[-self.config['max_history_size']:]
            
            self.logger.info(f"Statement execution completed successfully: {statement.name}")
            return result
            
        except Exception as e:
            # Handle execution failure
            execution_path.end_time = datetime.now()
            execution_path.status = ExecutionStatus.FAILED
            execution_path.add_error(str(e))
            
            result = LogicResult(
                result_id=str(uuid.uuid4()),
                statement_id=statement_id,
                execution_path=execution_path,
                total_duration=execution_path.get_duration() or 0.0,
                success=False,
                error_message=str(e)
            )
            
            self.stats['statements_executed'] += 1
            self.stats['failed_executions'] += 1
            
            self.logger.error(f"Statement execution failed: {e}")
            return result
    
    async def _execute_statement_logic(self, statement: ConditionalStatement,
                                     execution_path: ExecutionPath,
                                     context: Dict[str, Any]) -> Any:
        """Execute the logic of a conditional statement."""
        # Detect statement pattern
        if_branches = statement.get_branches_by_type(BranchType.IF)
        then_branches = statement.get_branches_by_type(BranchType.THEN)
        else_branches = statement.get_branches_by_type(BranchType.ELSE)
        switch_branches = statement.get_branches_by_type(BranchType.SWITCH)
        
        if if_branches and then_branches:
            # If-then-else pattern
            return await self._execute_if_then_else(
                statement, execution_path, context, if_branches[0], then_branches[0],
                else_branches[0] if else_branches else None
            )
        
        elif switch_branches:
            # Switch pattern
            case_branches = statement.get_branches_by_type(BranchType.CASE)
            default_branches = statement.get_branches_by_type(BranchType.DEFAULT)
            
            return await self._execute_switch(
                statement, execution_path, context, switch_branches[0], case_branches,
                default_branches[0] if default_branches else None
            )
        
        else:
            # Sequential execution of all branches
            return await self._execute_sequential(statement, execution_path, context)
    
    async def _execute_if_then_else(self, statement: ConditionalStatement,
                                  execution_path: ExecutionPath,
                                  context: Dict[str, Any],
                                  if_branch: ConditionalBranch,
                                  then_branch: ConditionalBranch,
                                  else_branch: Optional[ConditionalBranch]) -> Any:
        """Execute if-then-else logic."""
        execution_path.add_branch(if_branch.branch_id)
        
        # Evaluate IF condition
        if not self.condition_evaluator:
            raise RuntimeError("Condition evaluator not set")
        
        condition_result = await self.condition_evaluator.evaluate_condition(
            if_branch.condition, context
        )
        
        if condition_result.result:
            # Execute THEN branch
            execution_path.add_branch(then_branch.branch_id)
            return await self._execute_branch_actions(then_branch, execution_path, context)
        
        elif else_branch:
            # Execute ELSE branch
            execution_path.add_branch(else_branch.branch_id)
            return await self._execute_branch_actions(else_branch, execution_path, context)
        
        return None
    
    async def _execute_switch(self, statement: ConditionalStatement,
                            execution_path: ExecutionPath,
                            context: Dict[str, Any],
                            switch_branch: ConditionalBranch,
                            case_branches: List[ConditionalBranch],
                            default_branch: Optional[ConditionalBranch]) -> Any:
        """Execute switch logic."""
        execution_path.add_branch(switch_branch.branch_id)
        
        # Evaluate switch condition
        if not self.condition_evaluator:
            raise RuntimeError("Condition evaluator not set")
        
        switch_result = await self.condition_evaluator.evaluate_condition(
            switch_branch.condition, context
        )
        
        switch_value = switch_result.result
        
        # Find matching case
        for case_branch in case_branches:
            case_value_tag = next(
                (tag for tag in case_branch.tags if tag.startswith("case_value:")), 
                None
            )
            
            if case_value_tag:
                case_value = case_value_tag.split(":", 1)[1]
                
                # Convert types for comparison
                if str(switch_value) == str(case_value):
                    execution_path.add_branch(case_branch.branch_id)
                    return await self._execute_branch_actions(case_branch, execution_path, context)
        
        # No case matched, execute default if available
        if default_branch:
            execution_path.add_branch(default_branch.branch_id)
            return await self._execute_branch_actions(default_branch, execution_path, context)
        
        return None
    
    async def _execute_sequential(self, statement: ConditionalStatement,
                                execution_path: ExecutionPath,
                                context: Dict[str, Any]) -> Any:
        """Execute all branches sequentially."""
        results = []
        
        for branch in statement.branches:
            if branch.is_active:
                execution_path.add_branch(branch.branch_id)
                
                # Evaluate condition if present
                if branch.condition:
                    if not self.condition_evaluator:
                        raise RuntimeError("Condition evaluator not set")
                    
                    condition_result = await self.condition_evaluator.evaluate_condition(
                        branch.condition, context
                    )
                    
                    if not condition_result.result:
                        continue  # Skip this branch
                
                # Execute branch actions
                result = await self._execute_branch_actions(branch, execution_path, context)
                results.append(result)
        
        return results
    
    async def _execute_branch_actions(self, branch: ConditionalBranch,
                                    execution_path: ExecutionPath,
                                    context: Dict[str, Any]) -> Any:
        """Execute actions in a branch."""
        results = []
        
        for action in branch.actions:
            execution_path.add_action(action)
            
            try:
                # Execute action
                result = await self._execute_action(action, context)
                results.append(result)
                execution_path.add_result(result)
                
            except Exception as e:
                error_msg = f"Action execution failed: {e}"
                execution_path.add_error(error_msg)
                self.logger.error(error_msg)
                
                # Continue with next action or stop based on configuration
                if not self.config.get('continue_on_error', True):
                    raise
        
        # Execute sub-branches
        for sub_branch in branch.sub_branches:
            if sub_branch.is_active:
                sub_result = await self._execute_branch_actions(sub_branch, execution_path, context)
                results.append(sub_result)
        
        return results if len(results) > 1 else (results[0] if results else None)
    
    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a single action."""
        action_type = action.get('type', 'unknown')
        
        # Get action handler
        if action_type in self.action_handlers:
            handler = self.action_handlers[action_type]
            
            # Execute with timeout if specified
            timeout = action.get('timeout', self.config['default_timeout'])
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(handler(action, context), timeout=timeout)
                else:
                    result = handler(action, context)
                
                return result
                
            except asyncio.TimeoutError:
                raise RuntimeError(f"Action timed out after {timeout} seconds")
        
        else:
            # Default action execution
            self.logger.warning(f"No handler for action type: {action_type}")
            return f"Executed {action_type}: {action.get('description', 'No description')}"
    
    def get_statement(self, statement_id: str) -> Optional[ConditionalStatement]:
        """Get a conditional statement by ID."""
        return self.statements.get(statement_id)
    
    def list_statements(self) -> List[ConditionalStatement]:
        """List all conditional statements."""
        return list(self.statements.values())
    
    def delete_statement(self, statement_id: str) -> bool:
        """Delete a conditional statement."""
        if statement_id in self.statements:
            del self.statements[statement_id]
            self.logger.debug(f"Deleted statement: {statement_id}")
            return True
        return False
    
    def get_execution_history(self, limit: int = 10) -> List[LogicResult]:
        """Get recent execution history."""
        return self.execution_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.stats,
            'active_statements': len([s for s in self.statements.values() if s.is_active]),
            'total_statements': len(self.statements),
            'registered_handlers': len(self.action_handlers),
            'execution_history_size': len(self.execution_history)
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.statements.clear()
        self.execution_history.clear()
        self.action_handlers.clear()
        self.logger.info("Logic engine cleaned up")