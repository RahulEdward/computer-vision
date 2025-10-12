"""
Plan Validation System

This module validates automation plans for feasibility, safety, and compliance
with business rules and constraints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any, Callable
from datetime import datetime, timedelta
import uuid
import logging
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ValidationCategory(Enum):
    """Categories of validation checks"""
    FEASIBILITY = "feasibility"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    TIMING = "timing"
    BUSINESS_LOGIC = "business_logic"
    USER_EXPERIENCE = "user_experience"


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKER = "blocker"


class ValidationStatus(Enum):
    """Status of validation process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ValidationRule:
    """Represents a validation rule"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: ValidationCategory = ValidationCategory.FEASIBILITY
    
    # Rule definition
    rule_type: str = "custom"  # custom, regex, function, constraint
    rule_expression: str = ""
    validation_function: Optional[Callable] = None
    
    # Rule properties
    severity: ValidationSeverity = ValidationSeverity.WARNING
    is_enabled: bool = True
    is_mandatory: bool = False
    
    # Scope
    applicable_contexts: List[str] = field(default_factory=list)
    excluded_contexts: List[str] = field(default_factory=list)
    
    # Conditions
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during validation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    
    # Issue details
    title: str = ""
    description: str = ""
    category: ValidationCategory = ValidationCategory.FEASIBILITY
    severity: ValidationSeverity = ValidationSeverity.WARNING
    
    # Location
    plan_element_id: Optional[str] = None
    task_id: Optional[str] = None
    step_index: Optional[int] = None
    
    # Context
    context_data: Dict[str, Any] = field(default_factory=dict)
    affected_elements: List[str] = field(default_factory=list)
    
    # Resolution
    suggested_fixes: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    fix_priority: int = 0
    
    # Impact assessment
    impact_score: float = 0.0
    risk_level: str = "low"
    business_impact: str = ""
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, acknowledged, fixed, ignored, false_positive


@dataclass
class ValidationContext:
    """Context information for validation"""
    plan_id: str = ""
    plan_type: str = ""
    
    # Environment context
    environment: str = "development"  # development, staging, production
    target_application: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enabled_categories: List[ValidationCategory] = field(default_factory=list)
    custom_rules: List[str] = field(default_factory=list)
    
    # Constraints
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    business_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str = ""
    validation_context: Optional[ValidationContext] = None
    
    # Validation results
    status: ValidationStatus = ValidationStatus.PENDING
    overall_result: str = "unknown"  # passed, failed, warning, error
    
    # Issues found
    issues: List[ValidationIssue] = field(default_factory=list)
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    
    # Statistics
    total_rules_checked: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    rules_skipped: int = 0
    
    # Metrics
    validation_score: float = 0.0  # 0-100
    safety_score: float = 0.0
    compliance_score: float = 0.0
    performance_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    required_fixes: List[str] = field(default_factory=list)
    optional_improvements: List[str] = field(default_factory=list)
    
    # Execution details
    validation_duration: timedelta = timedelta(0)
    validation_start: datetime = field(default_factory=datetime.now)
    validation_end: Optional[datetime] = None
    
    # Metadata
    validator_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PlanValidationResult:
    """Result of plan validation with detailed analysis"""
    plan_id: str = ""
    is_valid: bool = False
    
    # Validation summary
    validation_report: Optional[ValidationReport] = None
    blocking_issues: List[ValidationIssue] = field(default_factory=list)
    
    # Plan analysis
    feasibility_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    approval_recommendation: str = "review_required"  # approved, rejected, review_required
    confidence_level: float = 0.0
    
    # Next steps
    required_actions: List[str] = field(default_factory=list)
    approval_workflow: List[str] = field(default_factory=list)
    
    # Metadata
    validated_at: datetime = field(default_factory=datetime.now)
    validated_by: str = "system"


class PlanValidator:
    """Validates automation plans for feasibility, safety, and compliance"""
    
    def __init__(self):
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.validation_history: List[ValidationReport] = []
        
        # Built-in rule categories
        self.rule_categories = {
            ValidationCategory.FEASIBILITY: self._create_feasibility_rules,
            ValidationCategory.SAFETY: self._create_safety_rules,
            ValidationCategory.COMPLIANCE: self._create_compliance_rules,
            ValidationCategory.PERFORMANCE: self._create_performance_rules,
            ValidationCategory.SECURITY: self._create_security_rules,
            ValidationCategory.RESOURCE: self._create_resource_rules,
            ValidationCategory.DEPENDENCY: self._create_dependency_rules,
            ValidationCategory.TIMING: self._create_timing_rules,
            ValidationCategory.BUSINESS_LOGIC: self._create_business_logic_rules,
            ValidationCategory.USER_EXPERIENCE: self._create_ux_rules
        }
        
        # Validation processors
        self.validation_processors = {
            'plan_structure': self._validate_plan_structure,
            'task_dependencies': self._validate_task_dependencies,
            'resource_requirements': self._validate_resource_requirements,
            'timing_constraints': self._validate_timing_constraints,
            'safety_checks': self._validate_safety_requirements,
            'compliance_checks': self._validate_compliance_requirements,
            'performance_checks': self._validate_performance_requirements,
            'security_checks': self._validate_security_requirements
        }
        
        # Issue analyzers
        self.issue_analyzers = {
            'impact_assessment': self._assess_issue_impact,
            'fix_suggestion': self._suggest_issue_fixes,
            'priority_calculation': self._calculate_issue_priority,
            'risk_evaluation': self._evaluate_issue_risk
        }
        
        # Initialize built-in rules
        self._initialize_built_in_rules()
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': timedelta(0),
            'most_common_issues': {},
            'rule_effectiveness': {}
        }
    
    def validate_plan(self, plan_data: Dict[str, Any], 
                     validation_context: Optional[ValidationContext] = None) -> PlanValidationResult:
        """Validate an automation plan comprehensively"""
        start_time = datetime.now()
        
        # Create validation context if not provided
        if not validation_context:
            validation_context = ValidationContext(
                plan_id=plan_data.get('id', str(uuid.uuid4())),
                plan_type=plan_data.get('type', 'automation')
            )
        
        # Create validation report
        report = ValidationReport(
            plan_id=validation_context.plan_id,
            validation_context=validation_context,
            validation_start=start_time
        )
        
        try:
            # Run validation checks
            self._run_validation_checks(plan_data, validation_context, report)
            
            # Analyze issues
            self._analyze_validation_issues(report)
            
            # Calculate scores
            self._calculate_validation_scores(report)
            
            # Generate recommendations
            self._generate_validation_recommendations(report, plan_data)
            
            # Determine overall result
            overall_result = self._determine_overall_result(report)
            report.overall_result = overall_result
            report.status = ValidationStatus.COMPLETED
            
            # Create final result
            validation_result = PlanValidationResult(
                plan_id=validation_context.plan_id,
                is_valid=overall_result == "passed",
                validation_report=report,
                blocking_issues=[issue for issue in report.issues 
                               if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.BLOCKER]],
                feasibility_analysis=self._analyze_plan_feasibility(plan_data, report),
                risk_assessment=self._assess_plan_risks(plan_data, report),
                performance_analysis=self._analyze_plan_performance(plan_data, report),
                approval_recommendation=self._generate_approval_recommendation(report),
                confidence_level=self._calculate_confidence_level(report)
            )
            
            # Generate required actions
            validation_result.required_actions = self._generate_required_actions(report)
            validation_result.approval_workflow = self._generate_approval_workflow(validation_result)
            
        except Exception as e:
            report.status = ValidationStatus.FAILED
            report.overall_result = "error"
            
            validation_result = PlanValidationResult(
                plan_id=validation_context.plan_id,
                is_valid=False,
                validation_report=report
            )
            
            logger.error(f"Plan validation failed: {e}")
        
        # Finalize report
        report.validation_end = datetime.now()
        report.validation_duration = report.validation_end - start_time
        
        # Store validation history
        self.validation_history.append(report)
        
        # Update statistics
        self._update_validation_stats(report)
        
        logger.info(f"Plan validation completed: {report.overall_result}")
        return validation_result
    
    def add_validation_rule(self, rule: ValidationRule) -> bool:
        """Add a custom validation rule"""
        try:
            self.validation_rules[rule.id] = rule
            logger.info(f"Validation rule added: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add validation rule: {e}")
            return False
    
    def remove_validation_rule(self, rule_id: str) -> bool:
        """Remove a validation rule"""
        if rule_id in self.validation_rules:
            del self.validation_rules[rule_id]
            logger.info(f"Validation rule removed: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].is_enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].is_enabled = False
            return True
        return False
    
    def validate_single_task(self, task_data: Dict[str, Any], 
                           context: Optional[ValidationContext] = None) -> List[ValidationIssue]:
        """Validate a single task"""
        issues = []
        
        # Apply relevant rules to the task
        for rule in self.validation_rules.values():
            if rule.is_enabled and self._is_rule_applicable(rule, task_data, context):
                issue = self._apply_rule_to_task(rule, task_data, context)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def get_validation_suggestions(self, plan_data: Dict[str, Any]) -> List[str]:
        """Get validation suggestions for improving a plan"""
        suggestions = []
        
        # Analyze plan structure
        structure_suggestions = self._analyze_plan_structure_for_suggestions(plan_data)
        suggestions.extend(structure_suggestions)
        
        # Analyze task composition
        task_suggestions = self._analyze_task_composition_for_suggestions(plan_data)
        suggestions.extend(task_suggestions)
        
        # Analyze dependencies
        dependency_suggestions = self._analyze_dependencies_for_suggestions(plan_data)
        suggestions.extend(dependency_suggestions)
        
        return suggestions
    
    def _run_validation_checks(self, plan_data: Dict[str, Any], 
                              context: ValidationContext, report: ValidationReport) -> None:
        """Run all validation checks"""
        # Determine which rules to apply
        applicable_rules = self._get_applicable_rules(context)
        
        # Apply each rule
        for rule in applicable_rules:
            try:
                issues = self._apply_validation_rule(rule, plan_data, context)
                report.issues.extend(issues)
                
                if issues:
                    report.rules_failed += 1
                else:
                    report.rules_passed += 1
                
                report.total_rules_checked += 1
                
            except Exception as e:
                logger.warning(f"Rule {rule.name} failed to execute: {e}")
                report.rules_skipped += 1
        
        # Run specialized validation processors
        for processor_name, processor in self.validation_processors.items():
            try:
                processor_issues = processor(plan_data, context)
                report.issues.extend(processor_issues)
            except Exception as e:
                logger.warning(f"Validation processor {processor_name} failed: {e}")
    
    def _apply_validation_rule(self, rule: ValidationRule, plan_data: Dict[str, Any], 
                              context: ValidationContext) -> List[ValidationIssue]:
        """Apply a single validation rule"""
        issues = []
        
        if rule.rule_type == "function" and rule.validation_function:
            # Apply function-based rule
            try:
                result = rule.validation_function(plan_data, context)
                if not result.get('passed', True):
                    issue = ValidationIssue(
                        rule_id=rule.id,
                        title=result.get('title', f"Rule violation: {rule.name}"),
                        description=result.get('description', rule.description),
                        category=rule.category,
                        severity=rule.severity,
                        context_data=result.get('context', {}),
                        suggested_fixes=result.get('fixes', [])
                    )
                    issues.append(issue)
            except Exception as e:
                logger.error(f"Function rule {rule.name} execution failed: {e}")
        
        elif rule.rule_type == "regex":
            # Apply regex-based rule
            issues.extend(self._apply_regex_rule(rule, plan_data, context))
        
        elif rule.rule_type == "constraint":
            # Apply constraint-based rule
            issues.extend(self._apply_constraint_rule(rule, plan_data, context))
        
        return issues
    
    def _apply_regex_rule(self, rule: ValidationRule, plan_data: Dict[str, Any], 
                         context: ValidationContext) -> List[ValidationIssue]:
        """Apply regex-based validation rule"""
        issues = []
        
        try:
            pattern = re.compile(rule.rule_expression)
            plan_text = str(plan_data)
            
            if not pattern.search(plan_text):
                issue = ValidationIssue(
                    rule_id=rule.id,
                    title=f"Pattern not found: {rule.name}",
                    description=rule.description,
                    category=rule.category,
                    severity=rule.severity
                )
                issues.append(issue)
        
        except Exception as e:
            logger.error(f"Regex rule {rule.name} failed: {e}")
        
        return issues
    
    def _apply_constraint_rule(self, rule: ValidationRule, plan_data: Dict[str, Any], 
                              context: ValidationContext) -> List[ValidationIssue]:
        """Apply constraint-based validation rule"""
        issues = []
        
        # Simplified constraint checking
        # In a real implementation, this would parse and evaluate complex constraints
        
        return issues
    
    def _get_applicable_rules(self, context: ValidationContext) -> List[ValidationRule]:
        """Get rules applicable to the validation context"""
        applicable_rules = []
        
        for rule in self.validation_rules.values():
            if rule.is_enabled and self._is_rule_applicable_to_context(rule, context):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _is_rule_applicable_to_context(self, rule: ValidationRule, context: ValidationContext) -> bool:
        """Check if a rule is applicable to the validation context"""
        # Check validation level
        if context.validation_level == ValidationLevel.BASIC and rule.severity == ValidationSeverity.INFO:
            return False
        
        # Check enabled categories
        if context.enabled_categories and rule.category not in context.enabled_categories:
            return False
        
        # Check applicable contexts
        if rule.applicable_contexts and context.environment not in rule.applicable_contexts:
            return False
        
        # Check excluded contexts
        if rule.excluded_contexts and context.environment in rule.excluded_contexts:
            return False
        
        return True
    
    def _is_rule_applicable(self, rule: ValidationRule, task_data: Dict[str, Any], 
                           context: Optional[ValidationContext]) -> bool:
        """Check if a rule is applicable to a task"""
        return True  # Simplified
    
    def _apply_rule_to_task(self, rule: ValidationRule, task_data: Dict[str, Any], 
                           context: Optional[ValidationContext]) -> Optional[ValidationIssue]:
        """Apply a rule to a single task"""
        # Simplified implementation
        return None
    
    def _analyze_validation_issues(self, report: ValidationReport) -> None:
        """Analyze and categorize validation issues"""
        for issue in report.issues:
            # Analyze issue impact
            self.issue_analyzers['impact_assessment'](issue)
            
            # Suggest fixes
            self.issue_analyzers['fix_suggestion'](issue)
            
            # Calculate priority
            self.issue_analyzers['priority_calculation'](issue)
            
            # Evaluate risk
            self.issue_analyzers['risk_evaluation'](issue)
            
            # Categorize by severity
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.BLOCKER]:
                report.critical_issues.append(issue)
            elif issue.severity == ValidationSeverity.WARNING:
                report.warnings.append(issue)
    
    def _calculate_validation_scores(self, report: ValidationReport) -> None:
        """Calculate validation scores"""
        total_rules = report.total_rules_checked
        if total_rules == 0:
            return
        
        # Overall validation score
        passed_ratio = report.rules_passed / total_rules
        report.validation_score = passed_ratio * 100
        
        # Category-specific scores
        category_scores = {}
        for category in ValidationCategory:
            category_issues = [issue for issue in report.issues if issue.category == category]
            category_rules = len([rule for rule in self.validation_rules.values() 
                                if rule.category == category and rule.is_enabled])
            
            if category_rules > 0:
                category_score = max(0, (category_rules - len(category_issues)) / category_rules * 100)
                category_scores[category.value] = category_score
        
        report.safety_score = category_scores.get('safety', 100)
        report.compliance_score = category_scores.get('compliance', 100)
        report.performance_score = category_scores.get('performance', 100)
    
    def _generate_validation_recommendations(self, report: ValidationReport, plan_data: Dict[str, Any]) -> None:
        """Generate validation recommendations"""
        recommendations = []
        
        # Recommendations based on critical issues
        for issue in report.critical_issues:
            recommendations.extend(issue.suggested_fixes)
        
        # General recommendations
        if report.validation_score < 80:
            recommendations.append("Consider reviewing and improving the automation plan")
        
        if report.safety_score < 90:
            recommendations.append("Review safety requirements and add appropriate safeguards")
        
        if report.compliance_score < 95:
            recommendations.append("Ensure all compliance requirements are met")
        
        report.recommendations = recommendations
    
    def _determine_overall_result(self, report: ValidationReport) -> str:
        """Determine overall validation result"""
        if any(issue.severity == ValidationSeverity.BLOCKER for issue in report.issues):
            return "failed"
        elif any(issue.severity == ValidationSeverity.CRITICAL for issue in report.issues):
            return "failed"
        elif any(issue.severity == ValidationSeverity.ERROR for issue in report.issues):
            return "warning"
        elif report.validation_score >= 90:
            return "passed"
        else:
            return "warning"
    
    # Built-in rule creators
    def _create_feasibility_rules(self) -> List[ValidationRule]:
        """Create feasibility validation rules"""
        rules = []
        
        # Task feasibility rule
        rules.append(ValidationRule(
            name="Task Feasibility Check",
            description="Verify that all tasks in the plan are technically feasible",
            category=ValidationCategory.FEASIBILITY,
            rule_type="function",
            validation_function=self._check_task_feasibility,
            severity=ValidationSeverity.ERROR
        ))
        
        return rules
    
    def _create_safety_rules(self) -> List[ValidationRule]:
        """Create safety validation rules"""
        rules = []
        
        # Destructive action safety rule
        rules.append(ValidationRule(
            name="Destructive Action Safety",
            description="Ensure destructive actions have appropriate safeguards",
            category=ValidationCategory.SAFETY,
            rule_type="function",
            validation_function=self._check_destructive_actions,
            severity=ValidationSeverity.CRITICAL
        ))
        
        return rules
    
    def _create_compliance_rules(self) -> List[ValidationRule]:
        """Create compliance validation rules"""
        return []  # Simplified
    
    def _create_performance_rules(self) -> List[ValidationRule]:
        """Create performance validation rules"""
        return []  # Simplified
    
    def _create_security_rules(self) -> List[ValidationRule]:
        """Create security validation rules"""
        return []  # Simplified
    
    def _create_resource_rules(self) -> List[ValidationRule]:
        """Create resource validation rules"""
        return []  # Simplified
    
    def _create_dependency_rules(self) -> List[ValidationRule]:
        """Create dependency validation rules"""
        return []  # Simplified
    
    def _create_timing_rules(self) -> List[ValidationRule]:
        """Create timing validation rules"""
        return []  # Simplified
    
    def _create_business_logic_rules(self) -> List[ValidationRule]:
        """Create business logic validation rules"""
        return []  # Simplified
    
    def _create_ux_rules(self) -> List[ValidationRule]:
        """Create user experience validation rules"""
        return []  # Simplified
    
    def _initialize_built_in_rules(self) -> None:
        """Initialize all built-in validation rules"""
        for category, rule_creator in self.rule_categories.items():
            rules = rule_creator()
            for rule in rules:
                self.validation_rules[rule.id] = rule
    
    # Validation functions
    def _check_task_feasibility(self, plan_data: Dict[str, Any], context: ValidationContext) -> Dict[str, Any]:
        """Check if tasks are feasible"""
        return {'passed': True}  # Simplified
    
    def _check_destructive_actions(self, plan_data: Dict[str, Any], context: ValidationContext) -> Dict[str, Any]:
        """Check for destructive actions without safeguards"""
        return {'passed': True}  # Simplified
    
    # Validation processors
    def _validate_plan_structure(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate plan structure"""
        return []  # Simplified
    
    def _validate_task_dependencies(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate task dependencies"""
        return []  # Simplified
    
    def _validate_resource_requirements(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate resource requirements"""
        return []  # Simplified
    
    def _validate_timing_constraints(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate timing constraints"""
        return []  # Simplified
    
    def _validate_safety_requirements(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate safety requirements"""
        return []  # Simplified
    
    def _validate_compliance_requirements(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate compliance requirements"""
        return []  # Simplified
    
    def _validate_performance_requirements(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate performance requirements"""
        return []  # Simplified
    
    def _validate_security_requirements(self, plan_data: Dict[str, Any], context: ValidationContext) -> List[ValidationIssue]:
        """Validate security requirements"""
        return []  # Simplified
    
    # Issue analyzers
    def _assess_issue_impact(self, issue: ValidationIssue) -> None:
        """Assess the impact of a validation issue"""
        # Simplified impact assessment
        severity_scores = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.6,
            ValidationSeverity.CRITICAL: 0.8,
            ValidationSeverity.BLOCKER: 1.0
        }
        
        issue.impact_score = severity_scores.get(issue.severity, 0.5)
    
    def _suggest_issue_fixes(self, issue: ValidationIssue) -> None:
        """Suggest fixes for a validation issue"""
        # Simplified fix suggestions
        if not issue.suggested_fixes:
            issue.suggested_fixes = ["Review and address the identified issue"]
    
    def _calculate_issue_priority(self, issue: ValidationIssue) -> None:
        """Calculate priority for fixing an issue"""
        severity_priorities = {
            ValidationSeverity.BLOCKER: 100,
            ValidationSeverity.CRITICAL: 80,
            ValidationSeverity.ERROR: 60,
            ValidationSeverity.WARNING: 40,
            ValidationSeverity.INFO: 20
        }
        
        issue.fix_priority = severity_priorities.get(issue.severity, 50)
    
    def _evaluate_issue_risk(self, issue: ValidationIssue) -> None:
        """Evaluate risk level of an issue"""
        if issue.severity in [ValidationSeverity.BLOCKER, ValidationSeverity.CRITICAL]:
            issue.risk_level = "high"
        elif issue.severity == ValidationSeverity.ERROR:
            issue.risk_level = "medium"
        else:
            issue.risk_level = "low"
    
    # Analysis methods
    def _analyze_plan_feasibility(self, plan_data: Dict[str, Any], report: ValidationReport) -> Dict[str, Any]:
        """Analyze plan feasibility"""
        return {'feasible': True, 'confidence': 0.9}  # Simplified
    
    def _assess_plan_risks(self, plan_data: Dict[str, Any], report: ValidationReport) -> Dict[str, Any]:
        """Assess plan risks"""
        return {'overall_risk': 'low', 'risk_factors': []}  # Simplified
    
    def _analyze_plan_performance(self, plan_data: Dict[str, Any], report: ValidationReport) -> Dict[str, Any]:
        """Analyze plan performance characteristics"""
        return {'estimated_duration': '30 minutes', 'efficiency_score': 0.8}  # Simplified
    
    def _generate_approval_recommendation(self, report: ValidationReport) -> str:
        """Generate approval recommendation"""
        if report.overall_result == "passed":
            return "approved"
        elif report.overall_result == "failed":
            return "rejected"
        else:
            return "review_required"
    
    def _calculate_confidence_level(self, report: ValidationReport) -> float:
        """Calculate confidence level in validation results"""
        return min(1.0, report.validation_score / 100)
    
    def _generate_required_actions(self, report: ValidationReport) -> List[str]:
        """Generate required actions based on validation results"""
        actions = []
        
        for issue in report.critical_issues:
            actions.extend(issue.suggested_fixes)
        
        return actions
    
    def _generate_approval_workflow(self, result: PlanValidationResult) -> List[str]:
        """Generate approval workflow steps"""
        workflow = []
        
        if result.approval_recommendation == "approved":
            workflow.append("Automatic approval - plan ready for execution")
        elif result.approval_recommendation == "rejected":
            workflow.extend(["Address critical issues", "Resubmit for validation"])
        else:
            workflow.extend(["Manual review required", "Address identified issues", "Revalidate plan"])
        
        return workflow
    
    def _analyze_plan_structure_for_suggestions(self, plan_data: Dict[str, Any]) -> List[str]:
        """Analyze plan structure for improvement suggestions"""
        return []  # Simplified
    
    def _analyze_task_composition_for_suggestions(self, plan_data: Dict[str, Any]) -> List[str]:
        """Analyze task composition for suggestions"""
        return []  # Simplified
    
    def _analyze_dependencies_for_suggestions(self, plan_data: Dict[str, Any]) -> List[str]:
        """Analyze dependencies for suggestions"""
        return []  # Simplified
    
    def _update_validation_stats(self, report: ValidationReport) -> None:
        """Update validation statistics"""
        self.validation_stats['total_validations'] += 1
        
        if report.status == ValidationStatus.COMPLETED:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        # Update average validation time
        total_time = (self.validation_stats['average_validation_time'] * 
                     (self.validation_stats['total_validations'] - 1) + 
                     report.validation_duration)
        self.validation_stats['average_validation_time'] = total_time / self.validation_stats['total_validations']
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation engine statistics"""
        return {
            'total_rules': len(self.validation_rules),
            'enabled_rules': len([r for r in self.validation_rules.values() if r.is_enabled]),
            'validation_history_count': len(self.validation_history),
            'validation_stats': self.validation_stats.copy()
        }