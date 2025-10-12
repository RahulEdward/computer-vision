"""
Semantic Validator

Validates automation workflows and actions against semantic understanding
of application states and business logic to ensure safe and correct execution.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

from .state_manager import ApplicationState, StateManager
from .context_engine import ContextAwareEngine, DecisionContext, ContextInfo, ContextType


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    NORMAL = "normal"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationResult(Enum):
    """Validation results."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    BLOCKED = "blocked"


class ValidationCategory(Enum):
    """Categories of validation."""
    SEMANTIC = "semantic"
    SAFETY = "safety"
    BUSINESS_LOGIC = "business_logic"
    DATA_INTEGRITY = "data_integrity"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"


class RiskLevel(Enum):
    """Risk levels for actions."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    rule_id: str
    name: str
    description: str
    category: ValidationCategory
    
    # Rule definition
    condition: str  # Condition to check
    action_pattern: Optional[str] = None  # Pattern of actions this applies to
    state_pattern: Optional[str] = None  # Pattern of states this applies to
    
    # Validation behavior
    validation_level: ValidationLevel = ValidationLevel.NORMAL
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Rule metadata
    enabled: bool = True
    priority: int = 100  # Lower number = higher priority
    
    # Error handling
    error_message: str = "Validation failed"
    suggested_action: Optional[str] = None
    
    # Context
    applicable_applications: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    
    def applies_to_application(self, application_name: str) -> bool:
        """Check if rule applies to an application."""
        if not self.applicable_applications:
            return True  # Applies to all if not specified
        return application_name in self.applicable_applications
    
    def applies_to_context(self, context: str) -> bool:
        """Check if rule applies to a context."""
        if not self.applicable_contexts:
            return True  # Applies to all if not specified
        return context in self.applicable_contexts


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    issue_id: str
    rule_id: str
    category: ValidationCategory
    result: ValidationResult
    risk_level: RiskLevel
    
    # Issue details
    message: str
    description: str
    
    # Context
    action: Optional[str] = None
    state: Optional[str] = None
    element: Optional[str] = None
    
    # Resolution
    suggested_fix: Optional[str] = None
    can_auto_fix: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0


@dataclass
class ValidationContext:
    """Context for validation."""
    context_id: str
    application_name: str
    current_state: Optional[ApplicationState] = None
    target_state: Optional[ApplicationState] = None
    
    # Action being validated
    action: Optional[str] = None
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow context
    workflow_step: Optional[int] = None
    previous_actions: List[str] = field(default_factory=list)
    
    # User context
    user_intent: Optional[str] = None
    user_permissions: List[str] = field(default_factory=list)
    
    # Environment context
    environment: str = "production"
    time_constraints: Optional[Tuple[datetime, datetime]] = None
    
    # Data context
    sensitive_data_present: bool = False
    data_modifications: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""
    report_id: str
    context: ValidationContext
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Results
    overall_result: ValidationResult = ValidationResult.VALID
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Statistics
    rules_checked: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    blocked_count: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    auto_fixes_available: List[str] = field(default_factory=list)
    
    # Performance
    validation_duration: float = 0.0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        
        if issue.result == ValidationResult.WARNING:
            self.warnings_count += 1
        elif issue.result == ValidationResult.ERROR:
            self.errors_count += 1
        elif issue.result == ValidationResult.BLOCKED:
            self.blocked_count += 1
        
        # Update overall result
        if issue.result == ValidationResult.BLOCKED:
            self.overall_result = ValidationResult.BLOCKED
        elif issue.result == ValidationResult.ERROR and self.overall_result != ValidationResult.BLOCKED:
            self.overall_result = ValidationResult.ERROR
        elif issue.result == ValidationResult.WARNING and self.overall_result == ValidationResult.VALID:
            self.overall_result = ValidationResult.WARNING
    
    def has_blocking_issues(self) -> bool:
        """Check if there are blocking issues."""
        return self.overall_result == ValidationResult.BLOCKED
    
    def has_errors(self) -> bool:
        """Check if there are errors."""
        return self.overall_result in [ValidationResult.ERROR, ValidationResult.BLOCKED]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_high_risk_issues(self) -> List[ValidationIssue]:
        """Get high-risk issues."""
        return [issue for issue in self.issues 
                if issue.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]


class SemanticValidator:
    """
    Validates automation workflows and actions against semantic understanding
    of application states and business logic.
    """
    
    def __init__(self, state_manager: StateManager, 
                 context_engine: ContextAwareEngine):
        """Initialize the semantic validator."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.context_engine = context_engine
        
        # Validation rules
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_categories: Dict[ValidationCategory, List[str]] = {}
        
        # Validation history
        self.validation_history: List[ValidationReport] = []
        
        # Configuration
        self.config = {
            'default_validation_level': ValidationLevel.NORMAL,
            'max_history_size': 1000,
            'auto_fix_enabled': True,
            'parallel_validation': True,
            'timeout_seconds': 30
        }
        
        # Statistics
        self.stats = {
            'validations_performed': 0,
            'issues_found': 0,
            'auto_fixes_applied': 0,
            'blocked_actions': 0,
            'rules_triggered': 0
        }
        
        # Initialize built-in rules
        self._initialize_builtin_rules()
    
    def _initialize_builtin_rules(self):
        """Initialize built-in validation rules."""
        
        # Safety rules
        self.add_rule(ValidationRule(
            rule_id="safety_destructive_action",
            name="Destructive Action Safety",
            description="Prevent destructive actions without confirmation",
            category=ValidationCategory.SAFETY,
            condition="action in ['delete', 'remove', 'clear', 'reset', 'format']",
            validation_level=ValidationLevel.STRICT,
            risk_level=RiskLevel.HIGH,
            error_message="Destructive action requires explicit confirmation",
            suggested_action="Add confirmation step before destructive action"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="safety_system_critical",
            name="System Critical Area",
            description="Prevent actions in system-critical areas",
            category=ValidationCategory.SAFETY,
            condition="'system' in current_state.window_title.lower() or 'admin' in current_state.window_title.lower()",
            validation_level=ValidationLevel.PARANOID,
            risk_level=RiskLevel.CRITICAL,
            error_message="Action in system-critical area is blocked",
            suggested_action="Verify system area access is intentional"
        ))
        
        # Data integrity rules
        self.add_rule(ValidationRule(
            rule_id="data_required_field",
            name="Required Field Validation",
            description="Ensure required fields are not left empty",
            category=ValidationCategory.DATA_INTEGRITY,
            condition="action == 'submit' and has_empty_required_fields()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.MEDIUM,
            error_message="Required fields must be filled before submission",
            suggested_action="Fill all required fields before submitting"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="data_format_validation",
            name="Data Format Validation",
            description="Validate data format before entry",
            category=ValidationCategory.DATA_INTEGRITY,
            condition="action == 'type' and not validate_data_format()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.LOW,
            error_message="Data format is invalid",
            suggested_action="Correct data format before entry"
        ))
        
        # Business logic rules
        self.add_rule(ValidationRule(
            rule_id="business_workflow_order",
            name="Workflow Step Order",
            description="Ensure workflow steps are executed in correct order",
            category=ValidationCategory.BUSINESS_LOGIC,
            condition="not is_valid_workflow_step()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.MEDIUM,
            error_message="Workflow step is out of order",
            suggested_action="Complete prerequisite steps first"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="business_state_transition",
            name="Valid State Transition",
            description="Ensure state transitions are valid",
            category=ValidationCategory.BUSINESS_LOGIC,
            condition="not is_valid_state_transition()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.MEDIUM,
            error_message="Invalid state transition attempted",
            suggested_action="Follow valid state transition path"
        ))
        
        # Security rules
        self.add_rule(ValidationRule(
            rule_id="security_sensitive_data",
            name="Sensitive Data Protection",
            description="Protect sensitive data from exposure",
            category=ValidationCategory.SECURITY,
            condition="sensitive_data_present and action in ['copy', 'screenshot', 'log']",
            validation_level=ValidationLevel.STRICT,
            risk_level=RiskLevel.HIGH,
            error_message="Action may expose sensitive data",
            suggested_action="Mask or exclude sensitive data"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="security_permission_check",
            name="Permission Validation",
            description="Verify user has required permissions",
            category=ValidationCategory.SECURITY,
            condition="not has_required_permissions()",
            validation_level=ValidationLevel.STRICT,
            risk_level=RiskLevel.HIGH,
            error_message="Insufficient permissions for action",
            suggested_action="Obtain required permissions"
        ))
        
        # Performance rules
        self.add_rule(ValidationRule(
            rule_id="performance_resource_intensive",
            name="Resource Intensive Action",
            description="Warn about resource-intensive actions",
            category=ValidationCategory.PERFORMANCE,
            condition="is_resource_intensive_action()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.LOW,
            error_message="Action may consume significant resources",
            suggested_action="Consider scheduling during off-peak hours"
        ))
        
        # Accessibility rules
        self.add_rule(ValidationRule(
            rule_id="accessibility_element_access",
            name="Element Accessibility",
            description="Ensure elements are accessible for automation",
            category=ValidationCategory.ACCESSIBILITY,
            condition="not is_element_accessible()",
            validation_level=ValidationLevel.NORMAL,
            risk_level=RiskLevel.MEDIUM,
            error_message="Element may not be accessible for automation",
            suggested_action="Use alternative element selection method"
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules[rule.rule_id] = rule
        
        # Update category index
        if rule.category not in self.rule_categories:
            self.rule_categories[rule.category] = []
        self.rule_categories[rule.category].append(rule.rule_id)
        
        self.logger.debug(f"Added validation rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str):
        """Remove a validation rule."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            del self.rules[rule_id]
            
            # Update category index
            if rule.category in self.rule_categories:
                if rule_id in self.rule_categories[rule.category]:
                    self.rule_categories[rule.category].remove(rule_id)
            
            self.logger.debug(f"Removed validation rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a validation rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable a validation rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    async def validate_action(self, context: ValidationContext,
                            validation_level: Optional[ValidationLevel] = None) -> ValidationReport:
        """
        Validate an action against semantic rules.
        
        Args:
            context: Validation context
            validation_level: Override validation level
            
        Returns:
            ValidationReport: Validation results
        """
        start_time = datetime.now()
        
        try:
            # Create validation report
            report = ValidationReport(
                report_id=f"validation_{datetime.now().timestamp()}",
                context=context
            )
            
            # Determine validation level
            level = validation_level or self.config['default_validation_level']
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(context, level)
            
            self.logger.info(f"Validating action '{context.action}' with {len(applicable_rules)} rules")
            
            # Validate against each rule
            if self.config['parallel_validation']:
                # Parallel validation for better performance
                validation_tasks = [
                    self._validate_against_rule(rule, context)
                    for rule in applicable_rules
                ]
                
                validation_results = await asyncio.gather(
                    *validation_tasks, 
                    return_exceptions=True
                )
                
                for result in validation_results:
                    if isinstance(result, ValidationIssue):
                        report.add_issue(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Validation error: {result}")
            else:
                # Sequential validation
                for rule in applicable_rules:
                    try:
                        issue = await self._validate_against_rule(rule, context)
                        if issue:
                            report.add_issue(issue)
                    except Exception as e:
                        self.logger.error(f"Rule validation failed for {rule.rule_id}: {e}")
            
            report.rules_checked = len(applicable_rules)
            
            # Generate recommendations
            await self._generate_recommendations(report)
            
            # Check for auto-fixes
            if self.config['auto_fix_enabled']:
                await self._identify_auto_fixes(report)
            
            # Calculate validation duration
            end_time = datetime.now()
            report.validation_duration = (end_time - start_time).total_seconds()
            
            # Store in history
            self.validation_history.append(report)
            if len(self.validation_history) > self.config['max_history_size']:
                self.validation_history = self.validation_history[-self.config['max_history_size']:]
            
            # Update statistics
            self.stats['validations_performed'] += 1
            self.stats['issues_found'] += len(report.issues)
            if report.has_blocking_issues():
                self.stats['blocked_actions'] += 1
            
            self.logger.info(f"Validation completed: {report.overall_result.value} "
                           f"({len(report.issues)} issues)")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            # Return error report
            error_report = ValidationReport(
                report_id=f"error_{datetime.now().timestamp()}",
                context=context,
                overall_result=ValidationResult.ERROR
            )
            error_report.add_issue(ValidationIssue(
                issue_id=f"error_{datetime.now().timestamp()}",
                rule_id="system_error",
                category=ValidationCategory.SEMANTIC,
                result=ValidationResult.ERROR,
                risk_level=RiskLevel.HIGH,
                message=f"Validation system error: {e}",
                description="An error occurred during validation"
            ))
            return error_report
    
    def _get_applicable_rules(self, context: ValidationContext,
                            level: ValidationLevel) -> List[ValidationRule]:
        """Get rules applicable to the context and validation level."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check validation level
            if self._is_rule_applicable_for_level(rule, level):
                # Check application
                if rule.applies_to_application(context.application_name):
                    applicable_rules.append(rule)
        
        # Sort by priority (lower number = higher priority)
        applicable_rules.sort(key=lambda r: r.priority)
        
        return applicable_rules
    
    def _is_rule_applicable_for_level(self, rule: ValidationRule,
                                    level: ValidationLevel) -> bool:
        """Check if rule is applicable for validation level."""
        level_hierarchy = {
            ValidationLevel.PERMISSIVE: 1,
            ValidationLevel.NORMAL: 2,
            ValidationLevel.STRICT: 3,
            ValidationLevel.PARANOID: 4
        }
        
        return level_hierarchy[level] >= level_hierarchy[rule.validation_level]
    
    async def _validate_against_rule(self, rule: ValidationRule,
                                   context: ValidationContext) -> Optional[ValidationIssue]:
        """Validate context against a specific rule."""
        try:
            # Create evaluation context
            eval_context = self._create_evaluation_context(context)
            
            # Evaluate rule condition
            if self._evaluate_condition(rule.condition, eval_context):
                # Rule triggered - create issue
                issue = ValidationIssue(
                    issue_id=f"issue_{rule.rule_id}_{datetime.now().timestamp()}",
                    rule_id=rule.rule_id,
                    category=rule.category,
                    result=self._determine_result_from_risk(rule.risk_level),
                    risk_level=rule.risk_level,
                    message=rule.error_message,
                    description=rule.description,
                    action=context.action,
                    state=context.current_state.state_id if context.current_state else None,
                    suggested_fix=rule.suggested_action
                )
                
                self.stats['rules_triggered'] += 1
                return issue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Rule evaluation failed for {rule.rule_id}: {e}")
            return None
    
    def _create_evaluation_context(self, context: ValidationContext) -> Dict[str, Any]:
        """Create context for rule evaluation."""
        eval_context = {
            'action': context.action,
            'action_parameters': context.action_parameters,
            'current_state': context.current_state,
            'target_state': context.target_state,
            'application_name': context.application_name,
            'workflow_step': context.workflow_step,
            'previous_actions': context.previous_actions,
            'user_intent': context.user_intent,
            'user_permissions': context.user_permissions,
            'environment': context.environment,
            'sensitive_data_present': context.sensitive_data_present,
            'data_modifications': context.data_modifications,
            
            # Helper functions
            'has_empty_required_fields': lambda: self._has_empty_required_fields(context),
            'validate_data_format': lambda: self._validate_data_format(context),
            'is_valid_workflow_step': lambda: self._is_valid_workflow_step(context),
            'is_valid_state_transition': lambda: self._is_valid_state_transition(context),
            'has_required_permissions': lambda: self._has_required_permissions(context),
            'is_resource_intensive_action': lambda: self._is_resource_intensive_action(context),
            'is_element_accessible': lambda: self._is_element_accessible(context)
        }
        
        return eval_context
    
    def _evaluate_condition(self, condition: str, eval_context: Dict[str, Any]) -> bool:
        """Evaluate a rule condition."""
        try:
            # Simple expression evaluation
            # In production, use a more secure expression evaluator
            return eval(condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            self.logger.error(f"Condition evaluation failed: {condition} - {e}")
            return False
    
    def _determine_result_from_risk(self, risk_level: RiskLevel) -> ValidationResult:
        """Determine validation result from risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return ValidationResult.BLOCKED
        elif risk_level == RiskLevel.HIGH:
            return ValidationResult.ERROR
        elif risk_level == RiskLevel.MEDIUM:
            return ValidationResult.WARNING
        else:
            return ValidationResult.WARNING
    
    # Helper functions for rule evaluation
    
    def _has_empty_required_fields(self, context: ValidationContext) -> bool:
        """Check if there are empty required fields."""
        if not context.current_state:
            return False
        
        # Look for required fields that are empty
        for element in context.current_state.elements:
            if (element.properties.get('required', False) and
                not element.properties.get('value', '').strip()):
                return True
        
        return False
    
    def _validate_data_format(self, context: ValidationContext) -> bool:
        """Validate data format."""
        if context.action != 'type' or 'text' not in context.action_parameters:
            return True
        
        text = context.action_parameters['text']
        element_type = context.action_parameters.get('element_type', '')
        
        # Email validation
        if 'email' in element_type.lower():
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, text))
        
        # Phone validation
        if 'phone' in element_type.lower():
            phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
            return bool(re.match(phone_pattern, text))
        
        # Number validation
        if 'number' in element_type.lower():
            try:
                float(text)
                return True
            except ValueError:
                return False
        
        return True  # Default to valid
    
    def _is_valid_workflow_step(self, context: ValidationContext) -> bool:
        """Check if workflow step is valid."""
        if not context.workflow_step or not context.previous_actions:
            return True  # No workflow context
        
        # Simple workflow validation - check if previous step was completed
        # In practice, this would check against a workflow definition
        required_previous_actions = {
            2: ['login'],
            3: ['login', 'navigate'],
            4: ['login', 'navigate', 'fill_form']
        }
        
        step = context.workflow_step
        if step in required_previous_actions:
            required = required_previous_actions[step]
            return all(action in context.previous_actions for action in required)
        
        return True
    
    def _is_valid_state_transition(self, context: ValidationContext) -> bool:
        """Check if state transition is valid."""
        if not context.current_state or not context.target_state:
            return True
        
        current_type = context.current_state.state_type
        target_type = context.target_state.state_type
        
        # Define valid transitions
        valid_transitions = {
            'idle': ['loading', 'active', 'error'],
            'loading': ['active', 'error', 'idle'],
            'active': ['loading', 'idle', 'modal', 'error'],
            'modal': ['active', 'error'],
            'error': ['idle', 'loading']
        }
        
        current_type_str = current_type.value if hasattr(current_type, 'value') else str(current_type)
        target_type_str = target_type.value if hasattr(target_type, 'value') else str(target_type)
        
        return target_type_str in valid_transitions.get(current_type_str, [])
    
    def _has_required_permissions(self, context: ValidationContext) -> bool:
        """Check if user has required permissions."""
        action = context.action
        
        # Define required permissions for actions
        permission_requirements = {
            'delete': ['delete_permission'],
            'admin': ['admin_permission'],
            'modify_system': ['system_admin'],
            'access_sensitive': ['sensitive_data_access']
        }
        
        for action_pattern, required_perms in permission_requirements.items():
            if action_pattern in action.lower():
                return all(perm in context.user_permissions for perm in required_perms)
        
        return True  # Default to allowed
    
    def _is_resource_intensive_action(self, context: ValidationContext) -> bool:
        """Check if action is resource intensive."""
        intensive_actions = [
            'bulk_import', 'mass_update', 'large_download',
            'report_generation', 'data_export', 'backup'
        ]
        
        return any(action in context.action.lower() for action in intensive_actions)
    
    def _is_element_accessible(self, context: ValidationContext) -> bool:
        """Check if element is accessible for automation."""
        if not context.current_state:
            return True
        
        # Check if target element exists and is accessible
        target_element = context.action_parameters.get('element_id')
        if not target_element:
            return True
        
        # Find element in current state
        for element in context.current_state.elements:
            if element.element_id == target_element:
                return (element.is_visible and 
                       element.is_enabled and
                       (element.automation_id or element.name))
        
        return False  # Element not found or not accessible
    
    async def _generate_recommendations(self, report: ValidationReport):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Category-based recommendations
        categories = set(issue.category for issue in report.issues)
        
        if ValidationCategory.SAFETY in categories:
            recommendations.append("Review safety implications before proceeding")
        
        if ValidationCategory.SECURITY in categories:
            recommendations.append("Verify security requirements are met")
        
        if ValidationCategory.DATA_INTEGRITY in categories:
            recommendations.append("Validate all data before submission")
        
        if ValidationCategory.BUSINESS_LOGIC in categories:
            recommendations.append("Ensure business rules are followed")
        
        # Risk-based recommendations
        high_risk_issues = report.get_high_risk_issues()
        if high_risk_issues:
            recommendations.append(f"Address {len(high_risk_issues)} high-risk issues before proceeding")
        
        # Performance recommendations
        if ValidationCategory.PERFORMANCE in categories:
            recommendations.append("Consider performance impact of this action")
        
        report.recommendations = recommendations
    
    async def _identify_auto_fixes(self, report: ValidationReport):
        """Identify potential auto-fixes for issues."""
        auto_fixes = []
        
        for issue in report.issues:
            if issue.can_auto_fix:
                auto_fixes.append(f"Auto-fix available for: {issue.message}")
        
        # Common auto-fixes
        data_issues = report.get_issues_by_category(ValidationCategory.DATA_INTEGRITY)
        if data_issues:
            auto_fixes.append("Auto-validate and format data entries")
        
        accessibility_issues = report.get_issues_by_category(ValidationCategory.ACCESSIBILITY)
        if accessibility_issues:
            auto_fixes.append("Auto-select alternative element selectors")
        
        report.auto_fixes_available = auto_fixes
    
    async def apply_auto_fixes(self, report: ValidationReport) -> ValidationReport:
        """Apply automatic fixes to validation issues."""
        try:
            fixed_issues = []
            
            for issue in report.issues:
                if issue.can_auto_fix:
                    # Apply fix based on issue type
                    if await self._apply_auto_fix(issue, report.context):
                        fixed_issues.append(issue)
                        self.stats['auto_fixes_applied'] += 1
            
            # Remove fixed issues
            for fixed_issue in fixed_issues:
                report.issues.remove(fixed_issue)
            
            # Recalculate overall result
            if not report.issues:
                report.overall_result = ValidationResult.VALID
            else:
                # Recalculate based on remaining issues
                if any(issue.result == ValidationResult.BLOCKED for issue in report.issues):
                    report.overall_result = ValidationResult.BLOCKED
                elif any(issue.result == ValidationResult.ERROR for issue in report.issues):
                    report.overall_result = ValidationResult.ERROR
                elif any(issue.result == ValidationResult.WARNING for issue in report.issues):
                    report.overall_result = ValidationResult.WARNING
                else:
                    report.overall_result = ValidationResult.VALID
            
            self.logger.info(f"Applied {len(fixed_issues)} auto-fixes")
            return report
            
        except Exception as e:
            self.logger.error(f"Auto-fix application failed: {e}")
            return report
    
    async def _apply_auto_fix(self, issue: ValidationIssue, 
                            context: ValidationContext) -> bool:
        """Apply a specific auto-fix."""
        try:
            # Implement specific auto-fixes based on issue type
            if issue.category == ValidationCategory.DATA_INTEGRITY:
                return await self._fix_data_integrity_issue(issue, context)
            elif issue.category == ValidationCategory.ACCESSIBILITY:
                return await self._fix_accessibility_issue(issue, context)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Auto-fix failed for issue {issue.issue_id}: {e}")
            return False
    
    async def _fix_data_integrity_issue(self, issue: ValidationIssue,
                                      context: ValidationContext) -> bool:
        """Fix data integrity issues."""
        # Example: Auto-format data
        if 'format' in issue.message.lower():
            # Apply data formatting
            return True
        
        return False
    
    async def _fix_accessibility_issue(self, issue: ValidationIssue,
                                     context: ValidationContext) -> bool:
        """Fix accessibility issues."""
        # Example: Find alternative element selector
        if 'accessible' in issue.message.lower():
            # Find alternative selector
            return True
        
        return False
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = self.stats['validations_performed']
        
        return {
            **self.stats,
            'success_rate': (total_validations - self.stats['blocked_actions']) / max(1, total_validations),
            'active_rules': len([r for r in self.rules.values() if r.enabled]),
            'total_rules': len(self.rules),
            'validation_history_size': len(self.validation_history)
        }
    
    def get_recent_validations(self, limit: int = 10) -> List[ValidationReport]:
        """Get recent validation reports."""
        return self.validation_history[-limit:]
    
    def get_rules_by_category(self, category: ValidationCategory) -> List[ValidationRule]:
        """Get rules by category."""
        rule_ids = self.rule_categories.get(category, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    async def cleanup(self):
        """Cleanup resources."""
        self.validation_history.clear()
        self.logger.info("Semantic validator cleaned up")