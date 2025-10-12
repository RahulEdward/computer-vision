"""
State Manager

Manages application states, tracks state changes, and provides state-aware
automation capabilities for intelligent workflow execution.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import weakref


class StateType(Enum):
    """Types of application states."""
    WINDOW = "window"
    DIALOG = "dialog"
    MENU = "menu"
    FORM = "form"
    LIST = "list"
    LOADING = "loading"
    ERROR = "error"
    SUCCESS = "success"
    IDLE = "idle"
    BUSY = "busy"
    MODAL = "modal"
    TOOLTIP = "tooltip"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class StateChangeType(Enum):
    """Types of state changes."""
    CREATED = "created"
    UPDATED = "updated"
    DESTROYED = "destroyed"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    RESTORED = "restored"
    MOVED = "moved"
    RESIZED = "resized"


@dataclass
class ElementInfo:
    """Information about a UI element."""
    element_id: str
    element_type: str
    text: Optional[str] = None
    value: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (0, 0)
    visible: bool = True
    enabled: bool = True
    focused: bool = False


@dataclass
class ApplicationState:
    """Represents the state of an application or UI component."""
    state_id: str
    application_name: str
    window_title: str
    state_type: StateType
    timestamp: datetime
    
    # UI Elements
    elements: Dict[str, ElementInfo] = field(default_factory=dict)
    
    # State properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Window information
    window_handle: Optional[str] = None
    window_position: Tuple[int, int] = (0, 0)
    window_size: Tuple[int, int] = (0, 0)
    is_active: bool = False
    is_visible: bool = True
    
    # Context information
    url: Optional[str] = None  # For web applications
    page_title: Optional[str] = None
    breadcrumbs: List[str] = field(default_factory=list)
    
    # Metadata
    screenshot_hash: Optional[str] = None
    dom_hash: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.state_id:
            self.state_id = self._generate_state_id()
    
    def _generate_state_id(self) -> str:
        """Generate a unique state ID based on state characteristics."""
        state_data = f"{self.application_name}:{self.window_title}:{self.state_type.value}:{len(self.elements)}"
        return hashlib.md5(state_data.encode()).hexdigest()[:12]
    
    def get_element(self, element_id: str) -> Optional[ElementInfo]:
        """Get an element by ID."""
        return self.elements.get(element_id)
    
    def add_element(self, element: ElementInfo):
        """Add an element to the state."""
        self.elements[element.element_id] = element
    
    def remove_element(self, element_id: str):
        """Remove an element from the state."""
        self.elements.pop(element_id, None)
    
    def find_elements_by_type(self, element_type: str) -> List[ElementInfo]:
        """Find all elements of a specific type."""
        return [elem for elem in self.elements.values() if elem.element_type == element_type]
    
    def find_elements_by_text(self, text: str, partial: bool = True) -> List[ElementInfo]:
        """Find elements containing specific text."""
        results = []
        for elem in self.elements.values():
            if elem.text:
                if partial and text.lower() in elem.text.lower():
                    results.append(elem)
                elif not partial and text.lower() == elem.text.lower():
                    results.append(elem)
        return results
    
    def get_visible_elements(self) -> List[ElementInfo]:
        """Get all visible elements."""
        return [elem for elem in self.elements.values() if elem.visible]
    
    def get_enabled_elements(self) -> List[ElementInfo]:
        """Get all enabled elements."""
        return [elem for elem in self.elements.values() if elem.enabled]
    
    def get_focused_element(self) -> Optional[ElementInfo]:
        """Get the currently focused element."""
        for elem in self.elements.values():
            if elem.focused:
                return elem
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            'state_id': self.state_id,
            'application_name': self.application_name,
            'window_title': self.window_title,
            'state_type': self.state_type.value,
            'timestamp': self.timestamp.isoformat(),
            'elements': {eid: {
                'element_id': elem.element_id,
                'element_type': elem.element_type,
                'text': elem.text,
                'value': elem.value,
                'attributes': elem.attributes,
                'position': elem.position,
                'size': elem.size,
                'visible': elem.visible,
                'enabled': elem.enabled,
                'focused': elem.focused
            } for eid, elem in self.elements.items()},
            'properties': self.properties,
            'window_handle': self.window_handle,
            'window_position': self.window_position,
            'window_size': self.window_size,
            'is_active': self.is_active,
            'is_visible': self.is_visible,
            'url': self.url,
            'page_title': self.page_title,
            'breadcrumbs': self.breadcrumbs,
            'screenshot_hash': self.screenshot_hash,
            'dom_hash': self.dom_hash,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationState':
        """Create state from dictionary representation."""
        elements = {}
        for eid, elem_data in data.get('elements', {}).items():
            elements[eid] = ElementInfo(**elem_data)
        
        return cls(
            state_id=data['state_id'],
            application_name=data['application_name'],
            window_title=data['window_title'],
            state_type=StateType(data['state_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            elements=elements,
            properties=data.get('properties', {}),
            window_handle=data.get('window_handle'),
            window_position=tuple(data.get('window_position', (0, 0))),
            window_size=tuple(data.get('window_size', (0, 0))),
            is_active=data.get('is_active', False),
            is_visible=data.get('is_visible', True),
            url=data.get('url'),
            page_title=data.get('page_title'),
            breadcrumbs=data.get('breadcrumbs', []),
            screenshot_hash=data.get('screenshot_hash'),
            dom_hash=data.get('dom_hash'),
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class StateChange:
    """Represents a change in application state."""
    change_id: str
    previous_state: Optional[ApplicationState]
    new_state: ApplicationState
    change_type: StateChangeType
    timestamp: datetime
    trigger_action: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.change_id:
            self.change_id = self._generate_change_id()
    
    def _generate_change_id(self) -> str:
        """Generate a unique change ID."""
        change_data = f"{self.new_state.state_id}:{self.change_type.value}:{self.timestamp.isoformat()}"
        return hashlib.md5(change_data.encode()).hexdigest()[:12]


class StateManager:
    """
    Manages application states and tracks state changes for intelligent automation.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize the state manager."""
        self.logger = logging.getLogger(__name__)
        self.max_history_size = max_history_size
        
        # State storage
        self.current_states: Dict[str, ApplicationState] = {}  # app_name -> current state
        self.state_history: List[ApplicationState] = []
        self.state_changes: List[StateChange] = []
        
        # State tracking
        self.state_observers: List[Callable[[StateChange], None]] = []
        self.state_cache: Dict[str, ApplicationState] = {}
        
        # Application tracking
        self.tracked_applications: Set[str] = set()
        self.application_windows: Dict[str, List[str]] = {}  # app_name -> window_handles
        
        # Performance metrics
        self.metrics = {
            'states_tracked': 0,
            'state_changes_detected': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the periodic cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self._cleanup_old_states()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def track_application(self, application_name: str):
        """Start tracking an application."""
        self.tracked_applications.add(application_name)
        if application_name not in self.application_windows:
            self.application_windows[application_name] = []
        
        self.logger.info(f"Started tracking application: {application_name}")
    
    async def untrack_application(self, application_name: str):
        """Stop tracking an application."""
        self.tracked_applications.discard(application_name)
        
        # Remove current state
        self.current_states.pop(application_name, None)
        
        # Remove from windows tracking
        self.application_windows.pop(application_name, None)
        
        self.logger.info(f"Stopped tracking application: {application_name}")
    
    async def update_state(self, new_state: ApplicationState, 
                          trigger_action: Optional[str] = None) -> StateChange:
        """
        Update the state of an application.
        
        Args:
            new_state: The new application state
            trigger_action: Optional action that triggered the state change
            
        Returns:
            StateChange: Information about the state change
        """
        app_name = new_state.application_name
        previous_state = self.current_states.get(app_name)
        
        # Determine change type
        if previous_state is None:
            change_type = StateChangeType.CREATED
        elif self._states_equal(previous_state, new_state):
            change_type = StateChangeType.UPDATED
        else:
            change_type = StateChangeType.UPDATED
        
        # Create state change record
        state_change = StateChange(
            change_id="",
            previous_state=previous_state,
            new_state=new_state,
            change_type=change_type,
            timestamp=datetime.now(),
            trigger_action=trigger_action
        )
        
        # Update current state
        self.current_states[app_name] = new_state
        
        # Add to history
        self.state_history.append(new_state)
        self.state_changes.append(state_change)
        
        # Update metrics
        self.metrics['states_tracked'] += 1
        self.metrics['state_changes_detected'] += 1
        
        # Notify observers
        await self._notify_observers(state_change)
        
        # Cache the state
        self.state_cache[new_state.state_id] = new_state
        
        self.logger.debug(f"State updated for {app_name}: {change_type.value}")
        return state_change
    
    def get_current_state(self, application_name: str) -> Optional[ApplicationState]:
        """Get the current state of an application."""
        return self.current_states.get(application_name)
    
    def get_state_by_id(self, state_id: str) -> Optional[ApplicationState]:
        """Get a state by its ID."""
        # Check cache first
        if state_id in self.state_cache:
            self.metrics['cache_hits'] += 1
            return self.state_cache[state_id]
        
        # Search in history
        for state in self.state_history:
            if state.state_id == state_id:
                self.state_cache[state_id] = state
                self.metrics['cache_misses'] += 1
                return state
        
        self.metrics['cache_misses'] += 1
        return None
    
    def get_state_history(self, application_name: str, 
                         limit: Optional[int] = None) -> List[ApplicationState]:
        """Get the state history for an application."""
        history = [state for state in self.state_history 
                  if state.application_name == application_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_recent_changes(self, application_name: Optional[str] = None,
                          since: Optional[datetime] = None,
                          limit: Optional[int] = None) -> List[StateChange]:
        """Get recent state changes."""
        changes = self.state_changes
        
        # Filter by application
        if application_name:
            changes = [change for change in changes 
                      if change.new_state.application_name == application_name]
        
        # Filter by time
        if since:
            changes = [change for change in changes if change.timestamp >= since]
        
        # Apply limit
        if limit:
            changes = changes[-limit:]
        
        return changes
    
    def find_states_by_type(self, state_type: StateType,
                           application_name: Optional[str] = None) -> List[ApplicationState]:
        """Find states by type."""
        states = self.state_history
        
        if application_name:
            states = [state for state in states 
                     if state.application_name == application_name]
        
        return [state for state in states if state.state_type == state_type]
    
    def find_states_by_window_title(self, window_title: str,
                                   partial: bool = True) -> List[ApplicationState]:
        """Find states by window title."""
        results = []
        for state in self.state_history:
            if partial and window_title.lower() in state.window_title.lower():
                results.append(state)
            elif not partial and window_title.lower() == state.window_title.lower():
                results.append(state)
        return results
    
    def find_states_with_element(self, element_type: Optional[str] = None,
                                element_text: Optional[str] = None) -> List[ApplicationState]:
        """Find states containing specific elements."""
        results = []
        for state in self.state_history:
            for element in state.elements.values():
                match = True
                
                if element_type and element.element_type != element_type:
                    match = False
                
                if element_text and element.text:
                    if element_text.lower() not in element.text.lower():
                        match = False
                
                if match:
                    results.append(state)
                    break
        
        return results
    
    def get_state_transitions(self, from_state_id: str,
                             to_state_id: Optional[str] = None) -> List[StateChange]:
        """Get state transitions from a specific state."""
        transitions = []
        
        for change in self.state_changes:
            if change.previous_state and change.previous_state.state_id == from_state_id:
                if to_state_id is None or change.new_state.state_id == to_state_id:
                    transitions.append(change)
        
        return transitions
    
    def predict_next_state(self, current_state: ApplicationState,
                          action: str) -> Optional[ApplicationState]:
        """Predict the next state based on current state and action."""
        # Find similar historical transitions
        similar_transitions = []
        
        for change in self.state_changes:
            if (change.previous_state and 
                change.trigger_action == action and
                self._states_similar(change.previous_state, current_state)):
                similar_transitions.append(change)
        
        if similar_transitions:
            # Return the most recent similar transition's result
            most_recent = max(similar_transitions, key=lambda x: x.timestamp)
            return most_recent.new_state
        
        return None
    
    def add_state_observer(self, observer: Callable[[StateChange], None]):
        """Add a state change observer."""
        self.state_observers.append(observer)
    
    def remove_state_observer(self, observer: Callable[[StateChange], None]):
        """Remove a state change observer."""
        if observer in self.state_observers:
            self.state_observers.remove(observer)
    
    async def _notify_observers(self, state_change: StateChange):
        """Notify all observers of a state change."""
        for observer in self.state_observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(state_change)
                else:
                    observer(state_change)
            except Exception as e:
                self.logger.error(f"Observer notification failed: {e}")
    
    def _states_equal(self, state1: ApplicationState, state2: ApplicationState) -> bool:
        """Check if two states are equal."""
        return (state1.state_id == state2.state_id and
                state1.window_title == state2.window_title and
                state1.state_type == state2.state_type and
                len(state1.elements) == len(state2.elements))
    
    def _states_similar(self, state1: ApplicationState, state2: ApplicationState,
                       similarity_threshold: float = 0.8) -> bool:
        """Check if two states are similar."""
        if state1.application_name != state2.application_name:
            return False
        
        if state1.state_type != state2.state_type:
            return False
        
        # Compare elements
        common_elements = 0
        total_elements = max(len(state1.elements), len(state2.elements))
        
        if total_elements == 0:
            return True
        
        for elem_id, elem1 in state1.elements.items():
            elem2 = state2.elements.get(elem_id)
            if elem2 and elem1.element_type == elem2.element_type:
                common_elements += 1
        
        similarity = common_elements / total_elements
        return similarity >= similarity_threshold
    
    async def _cleanup_old_states(self):
        """Clean up old states to prevent memory issues."""
        if len(self.state_history) > self.max_history_size:
            # Remove oldest states
            excess_count = len(self.state_history) - self.max_history_size
            removed_states = self.state_history[:excess_count]
            self.state_history = self.state_history[excess_count:]
            
            # Clean up cache
            for state in removed_states:
                self.state_cache.pop(state.state_id, None)
            
            self.logger.debug(f"Cleaned up {excess_count} old states")
        
        # Clean up old state changes
        if len(self.state_changes) > self.max_history_size:
            excess_count = len(self.state_changes) - self.max_history_size
            self.state_changes = self.state_changes[excess_count:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics."""
        return {
            **self.metrics,
            'tracked_applications': len(self.tracked_applications),
            'current_states': len(self.current_states),
            'state_history_size': len(self.state_history),
            'state_changes_size': len(self.state_changes),
            'cache_size': len(self.state_cache),
            'cache_hit_rate': (self.metrics['cache_hits'] / 
                             max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']))
        }
    
    def export_states(self, application_name: Optional[str] = None) -> Dict[str, Any]:
        """Export states to a dictionary format."""
        states_to_export = self.state_history
        
        if application_name:
            states_to_export = [state for state in states_to_export 
                              if state.application_name == application_name]
        
        return {
            'states': [state.to_dict() for state in states_to_export],
            'exported_at': datetime.now().isoformat(),
            'total_states': len(states_to_export)
        }
    
    def import_states(self, data: Dict[str, Any]):
        """Import states from a dictionary format."""
        for state_data in data.get('states', []):
            state = ApplicationState.from_dict(state_data)
            self.state_history.append(state)
            self.state_cache[state.state_id] = state
        
        self.logger.info(f"Imported {len(data.get('states', []))} states")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all data
        self.current_states.clear()
        self.state_history.clear()
        self.state_changes.clear()
        self.state_cache.clear()
        self.state_observers.clear()
        self.tracked_applications.clear()
        self.application_windows.clear()