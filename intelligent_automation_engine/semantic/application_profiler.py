"""
Application Profiler

Profiles applications to understand their behavior, UI patterns, and automation
opportunities for better automation planning and execution.
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from .state_manager import ApplicationState, ElementInfo


class ProfileType(Enum):
    """Types of application profiles."""
    UI_STRUCTURE = "ui_structure"
    BEHAVIOR_PATTERN = "behavior_pattern"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    AUTOMATION_READINESS = "automation_readiness"
    SECURITY = "security"


class ElementPattern(Enum):
    """Common UI element patterns."""
    BUTTON = "button"
    INPUT_FIELD = "input_field"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    TABLE = "table"
    LIST = "list"
    MENU = "menu"
    DIALOG = "dialog"
    TAB = "tab"
    TREE = "tree"
    TOOLBAR = "toolbar"
    STATUS_BAR = "status_bar"
    NAVIGATION = "navigation"


class AutomationComplexity(Enum):
    """Automation complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class UIElement:
    """Represents a UI element in the application."""
    element_id: str
    element_type: str
    pattern: ElementPattern
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Position and size
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    
    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Automation info
    automation_id: Optional[str] = None
    accessibility_name: Optional[str] = None
    selectors: Dict[str, str] = field(default_factory=dict)  # xpath, css, etc.
    
    # Interaction info
    is_interactive: bool = False
    supported_actions: List[str] = field(default_factory=list)
    
    # Stability
    stability_score: float = 1.0
    change_frequency: float = 0.0
    
    def get_selector(self, selector_type: str = "xpath") -> Optional[str]:
        """Get a selector for this element."""
        return self.selectors.get(selector_type)
    
    def add_child(self, child_id: str):
        """Add a child element."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'element_id': self.element_id,
            'element_type': self.element_type,
            'pattern': self.pattern.value,
            'properties': self.properties,
            'bounds': self.bounds,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'automation_id': self.automation_id,
            'accessibility_name': self.accessibility_name,
            'selectors': self.selectors,
            'is_interactive': self.is_interactive,
            'supported_actions': self.supported_actions,
            'stability_score': self.stability_score,
            'change_frequency': self.change_frequency
        }


@dataclass
class UIStructure:
    """Represents the UI structure of an application."""
    structure_id: str
    application_name: str
    window_title: str
    timestamp: datetime
    
    # Elements
    elements: Dict[str, UIElement] = field(default_factory=dict)
    root_elements: List[str] = field(default_factory=list)
    
    # Patterns
    detected_patterns: Dict[ElementPattern, List[str]] = field(default_factory=dict)
    
    # Metadata
    total_elements: int = 0
    interactive_elements: int = 0
    automation_ready_elements: int = 0
    
    def add_element(self, element: UIElement):
        """Add an element to the structure."""
        self.elements[element.element_id] = element
        
        if not element.parent_id:
            self.root_elements.append(element.element_id)
        
        # Update pattern detection
        pattern = element.pattern
        if pattern not in self.detected_patterns:
            self.detected_patterns[pattern] = []
        self.detected_patterns[pattern].append(element.element_id)
        
        # Update counts
        self.total_elements = len(self.elements)
        self.interactive_elements = sum(1 for e in self.elements.values() if e.is_interactive)
        self.automation_ready_elements = sum(1 for e in self.elements.values() 
                                           if e.automation_id or e.selectors)
    
    def get_elements_by_pattern(self, pattern: ElementPattern) -> List[UIElement]:
        """Get elements matching a specific pattern."""
        element_ids = self.detected_patterns.get(pattern, [])
        return [self.elements[eid] for eid in element_ids if eid in self.elements]
    
    def get_automation_readiness_score(self) -> float:
        """Calculate automation readiness score."""
        if self.total_elements == 0:
            return 0.0
        
        return self.automation_ready_elements / self.total_elements


@dataclass
class BehaviorPattern:
    """Represents a behavior pattern in the application."""
    pattern_id: str
    pattern_type: str
    description: str
    
    # Occurrence data
    occurrences: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    # Pattern data
    trigger_conditions: List[str] = field(default_factory=list)
    typical_sequence: List[str] = field(default_factory=list)
    variations: List[List[str]] = field(default_factory=list)
    
    # Timing
    average_duration: float = 0.0
    duration_variance: float = 0.0
    
    # Reliability
    success_rate: float = 1.0
    failure_modes: List[str] = field(default_factory=list)
    
    # Automation potential
    automation_complexity: AutomationComplexity = AutomationComplexity.MODERATE
    automation_confidence: float = 0.5


@dataclass
class PerformanceProfile:
    """Performance characteristics of the application."""
    profile_id: str
    application_name: str
    
    # Response times
    average_response_time: float = 0.0
    response_time_variance: float = 0.0
    
    # Resource usage
    memory_usage: Dict[str, float] = field(default_factory=dict)
    cpu_usage: Dict[str, float] = field(default_factory=dict)
    
    # UI responsiveness
    ui_lag_frequency: float = 0.0
    freeze_incidents: int = 0
    
    # Load times
    startup_time: float = 0.0
    page_load_times: Dict[str, float] = field(default_factory=dict)
    
    # Bottlenecks
    identified_bottlenecks: List[str] = field(default_factory=list)
    performance_recommendations: List[str] = field(default_factory=list)


@dataclass
class ApplicationProfile:
    """Complete profile of an application."""
    profile_id: str
    application_name: str
    version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Profile components
    ui_structures: Dict[str, UIStructure] = field(default_factory=dict)
    behavior_patterns: Dict[str, BehaviorPattern] = field(default_factory=dict)
    performance_profile: Optional[PerformanceProfile] = None
    
    # Automation assessment
    automation_readiness: float = 0.0
    automation_complexity: AutomationComplexity = AutomationComplexity.MODERATE
    automation_recommendations: List[str] = field(default_factory=list)
    
    # Security considerations
    security_constraints: List[str] = field(default_factory=list)
    sensitive_elements: List[str] = field(default_factory=list)
    
    # Statistics
    total_interactions: int = 0
    successful_automations: int = 0
    failed_automations: int = 0
    
    def get_success_rate(self) -> float:
        """Get automation success rate."""
        total = self.successful_automations + self.failed_automations
        if total == 0:
            return 0.0
        return self.successful_automations / total
    
    def update_timestamp(self):
        """Update the last updated timestamp."""
        self.updated_at = datetime.now()


class ApplicationProfiler:
    """
    Profiles applications to understand their behavior and automation potential.
    """
    
    def __init__(self):
        """Initialize the application profiler."""
        self.logger = logging.getLogger(__name__)
        
        # Profile storage
        self.profiles: Dict[str, ApplicationProfile] = {}
        
        # Active profiling sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Pattern recognition
        self.pattern_templates: Dict[str, Dict[str, Any]] = {}
        self.element_classifiers: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            'min_pattern_occurrences': 3,
            'max_profile_age_days': 30,
            'ui_stability_threshold': 0.8,
            'performance_sample_size': 100,
            'automation_confidence_threshold': 0.7
        }
        
        # Statistics
        self.stats = {
            'profiles_created': 0,
            'patterns_detected': 0,
            'ui_structures_analyzed': 0,
            'performance_samples': 0
        }
        
        self._initialize_pattern_templates()
    
    def _initialize_pattern_templates(self):
        """Initialize common pattern templates."""
        self.pattern_templates = {
            'login_flow': {
                'sequence': ['username_input', 'password_input', 'login_button'],
                'variations': [
                    ['email_input', 'password_input', 'signin_button'],
                    ['username_input', 'password_input', 'captcha', 'login_button']
                ],
                'complexity': AutomationComplexity.SIMPLE
            },
            'form_submission': {
                'sequence': ['form_fields', 'validation', 'submit_button'],
                'variations': [
                    ['form_fields', 'submit_button'],
                    ['form_fields', 'preview', 'confirm', 'submit_button']
                ],
                'complexity': AutomationComplexity.MODERATE
            },
            'data_entry': {
                'sequence': ['select_field', 'enter_data', 'move_to_next'],
                'variations': [
                    ['click_field', 'clear_field', 'enter_data', 'tab_to_next'],
                    ['select_field', 'enter_data', 'validate', 'move_to_next']
                ],
                'complexity': AutomationComplexity.SIMPLE
            },
            'navigation': {
                'sequence': ['menu_click', 'submenu_select', 'page_load'],
                'variations': [
                    ['breadcrumb_click', 'page_load'],
                    ['tab_click', 'content_load'],
                    ['link_click', 'page_navigation']
                ],
                'complexity': AutomationComplexity.SIMPLE
            }
        }
    
    async def start_profiling_session(self, application_name: str, 
                                    session_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a profiling session for an application.
        
        Args:
            application_name: Name of the application to profile
            session_config: Optional configuration for the session
            
        Returns:
            str: Session ID
        """
        try:
            session_id = f"profile_{application_name}_{datetime.now().timestamp()}"
            
            config = session_config or {}
            session = {
                'session_id': session_id,
                'application_name': application_name,
                'start_time': datetime.now(),
                'config': config,
                'collected_data': {
                    'ui_snapshots': [],
                    'interactions': [],
                    'performance_samples': [],
                    'behavior_observations': []
                },
                'statistics': {
                    'snapshots_taken': 0,
                    'interactions_recorded': 0,
                    'patterns_detected': 0
                }
            }
            
            self.active_sessions[session_id] = session
            
            self.logger.info(f"Started profiling session for {application_name}: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start profiling session: {e}")
            raise
    
    async def capture_ui_structure(self, session_id: str, 
                                 application_state: ApplicationState) -> str:
        """
        Capture the UI structure of the application.
        
        Args:
            session_id: Active profiling session ID
            application_state: Current application state
            
        Returns:
            str: Structure ID
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Create UI structure
            structure_id = f"ui_{application_state.application_name}_{datetime.now().timestamp()}"
            structure = UIStructure(
                structure_id=structure_id,
                application_name=application_state.application_name,
                window_title=application_state.window_title,
                timestamp=datetime.now()
            )
            
            # Analyze elements from application state
            await self._analyze_ui_elements(structure, application_state)
            
            # Store in session
            session['collected_data']['ui_snapshots'].append(structure)
            session['statistics']['snapshots_taken'] += 1
            
            self.stats['ui_structures_analyzed'] += 1
            
            self.logger.debug(f"Captured UI structure: {structure_id}")
            return structure_id
            
        except Exception as e:
            self.logger.error(f"Failed to capture UI structure: {e}")
            raise
    
    async def _analyze_ui_elements(self, structure: UIStructure, 
                                 application_state: ApplicationState):
        """Analyze UI elements from application state."""
        try:
            # Process elements from application state
            for element_info in application_state.elements:
                # Create UI element
                element = UIElement(
                    element_id=element_info.element_id,
                    element_type=element_info.element_type,
                    pattern=self._classify_element_pattern(element_info),
                    properties=element_info.properties.copy(),
                    automation_id=element_info.automation_id,
                    accessibility_name=element_info.name,
                    is_interactive=element_info.is_enabled and element_info.is_visible
                )
                
                # Generate selectors
                element.selectors = self._generate_selectors(element_info)
                
                # Determine supported actions
                element.supported_actions = self._determine_supported_actions(element_info)
                
                # Add to structure
                structure.add_element(element)
            
            # Build hierarchy
            await self._build_element_hierarchy(structure, application_state)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze UI elements: {e}")
    
    def _classify_element_pattern(self, element_info: ElementInfo) -> ElementPattern:
        """Classify an element into a pattern."""
        element_type = element_info.element_type.lower()
        name = (element_info.name or "").lower()
        
        # Button patterns
        if 'button' in element_type or 'btn' in name:
            return ElementPattern.BUTTON
        
        # Input patterns
        if 'edit' in element_type or 'input' in element_type or 'textbox' in element_type:
            return ElementPattern.INPUT_FIELD
        
        # Dropdown patterns
        if 'combobox' in element_type or 'dropdown' in element_type:
            return ElementPattern.DROPDOWN
        
        # Checkbox patterns
        if 'checkbox' in element_type:
            return ElementPattern.CHECKBOX
        
        # Radio button patterns
        if 'radiobutton' in element_type:
            return ElementPattern.RADIO_BUTTON
        
        # Table patterns
        if 'table' in element_type or 'datagrid' in element_type:
            return ElementPattern.TABLE
        
        # List patterns
        if 'list' in element_type or 'listbox' in element_type:
            return ElementPattern.LIST
        
        # Menu patterns
        if 'menu' in element_type:
            return ElementPattern.MENU
        
        # Dialog patterns
        if 'dialog' in element_type or 'window' in element_type:
            return ElementPattern.DIALOG
        
        # Tab patterns
        if 'tab' in element_type:
            return ElementPattern.TAB
        
        # Tree patterns
        if 'tree' in element_type:
            return ElementPattern.TREE
        
        # Toolbar patterns
        if 'toolbar' in element_type:
            return ElementPattern.TOOLBAR
        
        # Status bar patterns
        if 'statusbar' in element_type:
            return ElementPattern.STATUS_BAR
        
        # Default to button for interactive elements
        if element_info.is_enabled:
            return ElementPattern.BUTTON
        
        return ElementPattern.BUTTON  # Default
    
    def _generate_selectors(self, element_info: ElementInfo) -> Dict[str, str]:
        """Generate selectors for an element."""
        selectors = {}
        
        # Automation ID selector
        if element_info.automation_id:
            selectors['automation_id'] = f"[AutomationId='{element_info.automation_id}']"
        
        # Name selector
        if element_info.name:
            selectors['name'] = f"[Name='{element_info.name}']"
        
        # Class name selector
        if element_info.class_name:
            selectors['class'] = f"[ClassName='{element_info.class_name}']"
        
        # Generate XPath
        xpath_parts = []
        if element_info.element_type:
            xpath_parts.append(f"//{element_info.element_type}")
        
        if element_info.automation_id:
            xpath_parts.append(f"[@AutomationId='{element_info.automation_id}']")
        elif element_info.name:
            xpath_parts.append(f"[@Name='{element_info.name}']")
        
        if xpath_parts:
            selectors['xpath'] = "".join(xpath_parts)
        
        return selectors
    
    def _determine_supported_actions(self, element_info: ElementInfo) -> List[str]:
        """Determine supported actions for an element."""
        actions = []
        
        if element_info.is_enabled:
            actions.append('click')
            
            element_type = element_info.element_type.lower()
            
            if 'edit' in element_type or 'textbox' in element_type:
                actions.extend(['type', 'clear', 'select_all'])
            
            if 'combobox' in element_type or 'dropdown' in element_type:
                actions.extend(['select', 'expand'])
            
            if 'checkbox' in element_type:
                actions.extend(['check', 'uncheck', 'toggle'])
            
            if 'radiobutton' in element_type:
                actions.append('select')
            
            if 'scrollbar' in element_type:
                actions.extend(['scroll_up', 'scroll_down'])
        
        if element_info.is_visible:
            actions.extend(['hover', 'right_click'])
        
        return actions
    
    async def _build_element_hierarchy(self, structure: UIStructure, 
                                     application_state: ApplicationState):
        """Build the element hierarchy."""
        # This would typically use the parent-child relationships from the UI framework
        # For now, we'll use a simple heuristic based on bounds
        
        elements = list(structure.elements.values())
        
        for element in elements:
            if element.bounds:
                # Find potential parents (elements that contain this one)
                for potential_parent in elements:
                    if (potential_parent.element_id != element.element_id and
                        potential_parent.bounds and
                        self._is_contained_in(element.bounds, potential_parent.bounds)):
                        
                        element.parent_id = potential_parent.element_id
                        potential_parent.add_child(element.element_id)
                        break
    
    def _is_contained_in(self, child_bounds: Tuple[int, int, int, int],
                        parent_bounds: Tuple[int, int, int, int]) -> bool:
        """Check if child bounds are contained within parent bounds."""
        cx, cy, cw, ch = child_bounds
        px, py, pw, ph = parent_bounds
        
        return (cx >= px and cy >= py and 
                cx + cw <= px + pw and cy + ch <= py + ph)
    
    async def record_interaction(self, session_id: str, interaction_data: Dict[str, Any]):
        """Record a user interaction during profiling."""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Add timestamp
            interaction_data['timestamp'] = datetime.now()
            
            # Store interaction
            session['collected_data']['interactions'].append(interaction_data)
            session['statistics']['interactions_recorded'] += 1
            
            # Analyze for patterns
            await self._analyze_interaction_patterns(session)
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
    
    async def _analyze_interaction_patterns(self, session: Dict[str, Any]):
        """Analyze interactions for behavior patterns."""
        try:
            interactions = session['collected_data']['interactions']
            
            if len(interactions) < 3:
                return  # Need minimum interactions for pattern detection
            
            # Look for sequential patterns
            recent_interactions = interactions[-10:]  # Last 10 interactions
            
            # Extract action sequence
            action_sequence = [i.get('action', 'unknown') for i in recent_interactions]
            
            # Check against known patterns
            for pattern_name, pattern_template in self.pattern_templates.items():
                if self._matches_pattern_template(action_sequence, pattern_template):
                    # Record pattern detection
                    pattern_observation = {
                        'pattern_name': pattern_name,
                        'detected_at': datetime.now(),
                        'sequence': action_sequence,
                        'confidence': self._calculate_pattern_confidence(action_sequence, pattern_template)
                    }
                    
                    session['collected_data']['behavior_observations'].append(pattern_observation)
                    session['statistics']['patterns_detected'] += 1
                    self.stats['patterns_detected'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to analyze interaction patterns: {e}")
    
    def _matches_pattern_template(self, sequence: List[str], 
                                template: Dict[str, Any]) -> bool:
        """Check if a sequence matches a pattern template."""
        template_sequence = template['sequence']
        variations = template.get('variations', [])
        
        # Check main sequence
        if self._sequence_similarity(sequence, template_sequence) > 0.7:
            return True
        
        # Check variations
        for variation in variations:
            if self._sequence_similarity(sequence, variation) > 0.7:
                return True
        
        return False
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple similarity based on common elements
        common = set(seq1) & set(seq2)
        total = set(seq1) | set(seq2)
        
        if not total:
            return 0.0
        
        return len(common) / len(total)
    
    def _calculate_pattern_confidence(self, sequence: List[str], 
                                    template: Dict[str, Any]) -> float:
        """Calculate confidence in pattern match."""
        template_sequence = template['sequence']
        similarity = self._sequence_similarity(sequence, template_sequence)
        
        # Adjust based on sequence length match
        length_factor = min(len(sequence), len(template_sequence)) / max(len(sequence), len(template_sequence))
        
        return similarity * length_factor
    
    async def end_profiling_session(self, session_id: str) -> ApplicationProfile:
        """
        End a profiling session and generate the application profile.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            ApplicationProfile: Generated profile
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            application_name = session['application_name']
            
            # Create or update application profile
            profile_id = f"profile_{application_name}"
            
            if profile_id in self.profiles:
                profile = self.profiles[profile_id]
                profile.update_timestamp()
            else:
                profile = ApplicationProfile(
                    profile_id=profile_id,
                    application_name=application_name
                )
                self.profiles[profile_id] = profile
                self.stats['profiles_created'] += 1
            
            # Process collected data
            await self._process_ui_structures(profile, session['collected_data']['ui_snapshots'])
            await self._process_behavior_patterns(profile, session['collected_data']['behavior_observations'])
            await self._process_performance_data(profile, session['collected_data']['performance_samples'])
            
            # Calculate automation assessment
            await self._assess_automation_potential(profile)
            
            # Clean up session
            del self.active_sessions[session_id]
            
            self.logger.info(f"Completed profiling session for {application_name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to end profiling session: {e}")
            raise
    
    async def _process_ui_structures(self, profile: ApplicationProfile, 
                                   ui_snapshots: List[UIStructure]):
        """Process UI structure data."""
        for structure in ui_snapshots:
            profile.ui_structures[structure.structure_id] = structure
    
    async def _process_behavior_patterns(self, profile: ApplicationProfile,
                                       behavior_observations: List[Dict[str, Any]]):
        """Process behavior pattern data."""
        pattern_counts = Counter()
        pattern_data = defaultdict(list)
        
        for observation in behavior_observations:
            pattern_name = observation['pattern_name']
            pattern_counts[pattern_name] += 1
            pattern_data[pattern_name].append(observation)
        
        # Create behavior patterns
        for pattern_name, count in pattern_counts.items():
            if count >= self.config['min_pattern_occurrences']:
                observations = pattern_data[pattern_name]
                
                pattern = BehaviorPattern(
                    pattern_id=f"{pattern_name}_{profile.application_name}",
                    pattern_type=pattern_name,
                    description=f"Detected {pattern_name} pattern",
                    occurrences=count,
                    first_seen=min(obs['detected_at'] for obs in observations),
                    last_seen=max(obs['detected_at'] for obs in observations)
                )
                
                # Calculate average confidence
                confidences = [obs['confidence'] for obs in observations]
                pattern.automation_confidence = sum(confidences) / len(confidences)
                
                # Set complexity from template
                if pattern_name in self.pattern_templates:
                    pattern.automation_complexity = self.pattern_templates[pattern_name].get(
                        'complexity', AutomationComplexity.MODERATE
                    )
                
                profile.behavior_patterns[pattern.pattern_id] = pattern
    
    async def _process_performance_data(self, profile: ApplicationProfile,
                                      performance_samples: List[Dict[str, Any]]):
        """Process performance data."""
        if not performance_samples:
            return
        
        # Create performance profile
        perf_profile = PerformanceProfile(
            profile_id=f"perf_{profile.application_name}",
            application_name=profile.application_name
        )
        
        # Process response times
        response_times = [s.get('response_time', 0) for s in performance_samples if 'response_time' in s]
        if response_times:
            perf_profile.average_response_time = sum(response_times) / len(response_times)
            mean = perf_profile.average_response_time
            perf_profile.response_time_variance = sum((t - mean) ** 2 for t in response_times) / len(response_times)
        
        profile.performance_profile = perf_profile
    
    async def _assess_automation_potential(self, profile: ApplicationProfile):
        """Assess the automation potential of the application."""
        try:
            scores = []
            
            # UI structure assessment
            if profile.ui_structures:
                ui_scores = [struct.get_automation_readiness_score() 
                           for struct in profile.ui_structures.values()]
                if ui_scores:
                    scores.append(sum(ui_scores) / len(ui_scores))
            
            # Behavior pattern assessment
            if profile.behavior_patterns:
                pattern_scores = [pattern.automation_confidence 
                                for pattern in profile.behavior_patterns.values()]
                if pattern_scores:
                    scores.append(sum(pattern_scores) / len(pattern_scores))
            
            # Calculate overall readiness
            if scores:
                profile.automation_readiness = sum(scores) / len(scores)
            else:
                profile.automation_readiness = 0.5  # Default
            
            # Determine complexity
            if profile.automation_readiness > 0.8:
                profile.automation_complexity = AutomationComplexity.SIMPLE
            elif profile.automation_readiness > 0.6:
                profile.automation_complexity = AutomationComplexity.MODERATE
            elif profile.automation_readiness > 0.4:
                profile.automation_complexity = AutomationComplexity.COMPLEX
            else:
                profile.automation_complexity = AutomationComplexity.VERY_COMPLEX
            
            # Generate recommendations
            await self._generate_automation_recommendations(profile)
            
        except Exception as e:
            self.logger.error(f"Failed to assess automation potential: {e}")
    
    async def _generate_automation_recommendations(self, profile: ApplicationProfile):
        """Generate automation recommendations."""
        recommendations = []
        
        # UI structure recommendations
        if profile.ui_structures:
            total_elements = sum(struct.total_elements for struct in profile.ui_structures.values())
            automation_ready = sum(struct.automation_ready_elements for struct in profile.ui_structures.values())
            
            if total_elements > 0:
                readiness_ratio = automation_ready / total_elements
                
                if readiness_ratio < 0.5:
                    recommendations.append("Improve element identification by adding automation IDs")
                
                if readiness_ratio < 0.3:
                    recommendations.append("Consider using image-based automation for elements without IDs")
        
        # Pattern-based recommendations
        if profile.behavior_patterns:
            simple_patterns = [p for p in profile.behavior_patterns.values() 
                             if p.automation_complexity == AutomationComplexity.SIMPLE]
            
            if simple_patterns:
                recommendations.append(f"Start automation with {len(simple_patterns)} simple patterns identified")
        
        # Performance recommendations
        if profile.performance_profile:
            if profile.performance_profile.average_response_time > 2.0:
                recommendations.append("Consider adding wait strategies for slow response times")
        
        # General recommendations
        if profile.automation_readiness < 0.5:
            recommendations.append("Application may require significant automation framework setup")
        
        if not recommendations:
            recommendations.append("Application appears ready for automation")
        
        profile.automation_recommendations = recommendations
    
    def get_profile(self, application_name: str) -> Optional[ApplicationProfile]:
        """Get profile for an application."""
        profile_id = f"profile_{application_name}"
        return self.profiles.get(profile_id)
    
    def get_all_profiles(self) -> List[ApplicationProfile]:
        """Get all application profiles."""
        return list(self.profiles.values())
    
    def get_profiling_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        return {
            **self.stats,
            'active_sessions': len(self.active_sessions),
            'total_profiles': len(self.profiles),
            'pattern_templates': len(self.pattern_templates)
        }
    
    async def cleanup_old_profiles(self):
        """Clean up old profiles."""
        cutoff_date = datetime.now() - timedelta(days=self.config['max_profile_age_days'])
        
        old_profiles = [
            profile_id for profile_id, profile in self.profiles.items()
            if profile.updated_at < cutoff_date
        ]
        
        for profile_id in old_profiles:
            del self.profiles[profile_id]
        
        if old_profiles:
            self.logger.info(f"Cleaned up {len(old_profiles)} old profiles")
    
    async def export_profile(self, application_name: str, 
                           export_format: str = "json") -> Optional[str]:
        """Export an application profile."""
        profile = self.get_profile(application_name)
        if not profile:
            return None
        
        if export_format.lower() == "json":
            # Convert profile to JSON-serializable format
            profile_data = {
                'profile_id': profile.profile_id,
                'application_name': profile.application_name,
                'version': profile.version,
                'created_at': profile.created_at.isoformat(),
                'updated_at': profile.updated_at.isoformat(),
                'automation_readiness': profile.automation_readiness,
                'automation_complexity': profile.automation_complexity.value,
                'automation_recommendations': profile.automation_recommendations,
                'security_constraints': profile.security_constraints,
                'statistics': {
                    'total_interactions': profile.total_interactions,
                    'successful_automations': profile.successful_automations,
                    'failed_automations': profile.failed_automations,
                    'success_rate': profile.get_success_rate()
                },
                'ui_structures': {
                    struct_id: {
                        'structure_id': struct.structure_id,
                        'window_title': struct.window_title,
                        'timestamp': struct.timestamp.isoformat(),
                        'total_elements': struct.total_elements,
                        'interactive_elements': struct.interactive_elements,
                        'automation_ready_elements': struct.automation_ready_elements,
                        'automation_readiness_score': struct.get_automation_readiness_score(),
                        'detected_patterns': {
                            pattern.value: len(elements) 
                            for pattern, elements in struct.detected_patterns.items()
                        }
                    }
                    for struct_id, struct in profile.ui_structures.items()
                },
                'behavior_patterns': {
                    pattern_id: {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'description': pattern.description,
                        'occurrences': pattern.occurrences,
                        'automation_complexity': pattern.automation_complexity.value,
                        'automation_confidence': pattern.automation_confidence,
                        'success_rate': pattern.success_rate
                    }
                    for pattern_id, pattern in profile.behavior_patterns.items()
                }
            }
            
            return json.dumps(profile_data, indent=2)
        
        return None