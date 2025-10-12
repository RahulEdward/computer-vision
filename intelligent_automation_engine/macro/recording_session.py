"""
Recording Session Management

Handles the recording of user actions and manages recording sessions
with context awareness and intelligent action capture.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio


class ActionType(Enum):
    """Types of recorded actions."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    KEY_PRESS = "key_press"
    KEY_COMBINATION = "key_combination"
    MOUSE_MOVE = "mouse_move"
    SCROLL = "scroll"
    DRAG_DROP = "drag_drop"
    WINDOW_ACTION = "window_action"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    ELEMENT_INTERACTION = "element_interaction"


class ElementType(Enum):
    """Types of UI elements."""
    BUTTON = "button"
    TEXT_INPUT = "text_input"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    LINK = "link"
    IMAGE = "image"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    DIALOG = "dialog"
    WINDOW = "window"
    UNKNOWN = "unknown"


@dataclass
class ElementInfo:
    """Information about a UI element."""
    element_type: ElementType
    text: Optional[str] = None
    id: Optional[str] = None
    class_name: Optional[str] = None
    tag_name: Optional[str] = None
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    attributes: Dict[str, str] = None
    bounding_box: Dict[str, float] = None
    parent_info: Optional['ElementInfo'] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.bounding_box is None:
            self.bounding_box = {}


@dataclass
class ScreenContext:
    """Context information about the screen state."""
    window_title: str
    application_name: str
    url: Optional[str] = None
    screen_resolution: Tuple[int, int] = None
    active_window_bounds: Dict[str, float] = None
    timestamp: float = None
    screenshot_path: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.active_window_bounds is None:
            self.active_window_bounds = {}


@dataclass
class RecordedAction:
    """A single recorded user action."""
    action_id: str
    action_type: ActionType
    timestamp: float
    screen_context: ScreenContext
    
    # Position information
    x: Optional[float] = None
    y: Optional[float] = None
    
    # Element information
    target_element: Optional[ElementInfo] = None
    
    # Action-specific data
    text_input: Optional[str] = None
    key_code: Optional[str] = None
    key_modifiers: List[str] = None
    scroll_direction: Optional[str] = None
    scroll_amount: Optional[int] = None
    
    # Drag and drop
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    
    # Wait information
    wait_duration: Optional[float] = None
    wait_condition: Optional[str] = None
    
    # Additional metadata
    confidence: float = 1.0
    is_user_initiated: bool = True
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.key_modifiers is None:
            self.key_modifiers = []


class RecordingSession:
    """
    Manages a recording session, capturing user actions and maintaining context.
    """
    
    def __init__(self, session_name: str = None):
        """Initialize a recording session."""
        self.logger = logging.getLogger(__name__)
        
        # Session metadata
        self.session_id = self._generate_session_id()
        self.session_name = session_name or f"Recording_{self.session_id}"
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        
        # Recorded actions
        self.actions: List[RecordedAction] = []
        self.action_counter = 0
        
        # Context tracking
        self.current_context: Optional[ScreenContext] = None
        self.context_history: List[ScreenContext] = []
        
        # Recording settings
        self.settings = {
            'capture_screenshots': True,
            'capture_element_info': True,
            'capture_mouse_moves': False,
            'min_action_interval': 0.1,  # Minimum time between actions
            'auto_detect_waits': True,
            'smart_element_detection': True
        }
        
        # Performance tracking
        self.stats = {
            'total_actions': 0,
            'unique_elements': 0,
            'context_switches': 0,
            'recording_duration': 0
        }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def start_recording(self) -> bool:
        """
        Start the recording session.
        
        Returns:
            bool: True if recording started successfully
        """
        try:
            if self.is_recording:
                self.logger.warning("Recording session already active")
                return False
            
            self.logger.info(f"Starting recording session: {self.session_name}")
            
            # Initialize recording state
            self.is_recording = True
            self.is_paused = False
            self.start_time = time.time()
            
            # Capture initial context
            await self._capture_initial_context()
            
            self.logger.info("Recording session started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return False
    
    async def stop_recording(self) -> bool:
        """
        Stop the recording session.
        
        Returns:
            bool: True if recording stopped successfully
        """
        try:
            if not self.is_recording:
                self.logger.warning("No active recording session")
                return False
            
            self.logger.info("Stopping recording session")
            
            # Update session state
            self.is_recording = False
            self.is_paused = False
            self.end_time = time.time()
            
            # Update statistics
            self.stats['recording_duration'] = self.end_time - self.start_time
            self.stats['total_actions'] = len(self.actions)
            
            self.logger.info(f"Recording session stopped. Captured {len(self.actions)} actions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            return False
    
    async def pause_recording(self):
        """Pause the recording session."""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.logger.info("Recording session paused")
    
    async def resume_recording(self):
        """Resume the recording session."""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            self.logger.info("Recording session resumed")
    
    async def record_action(self, action: RecordedAction) -> bool:
        """
        Record a user action.
        
        Args:
            action: The action to record
            
        Returns:
            bool: True if action was recorded successfully
        """
        try:
            if not self.is_recording or self.is_paused:
                return False
            
            # Check minimum interval
            if self.actions and action.timestamp - self.actions[-1].timestamp < self.settings['min_action_interval']:
                return False
            
            # Update action ID
            self.action_counter += 1
            action.action_id = f"{self.session_id}_{self.action_counter:04d}"
            
            # Detect context changes
            await self._update_context(action.screen_context)
            
            # Add action to session
            self.actions.append(action)
            
            # Auto-detect waits if enabled
            if self.settings['auto_detect_waits']:
                await self._detect_wait_actions()
            
            self.logger.debug(f"Recorded action: {action.action_type.value} at ({action.x}, {action.y})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record action: {e}")
            return False
    
    async def record_click(self, x: float, y: float, button: str = "left", 
                          element_info: Optional[ElementInfo] = None) -> bool:
        """Record a click action."""
        action_type = ActionType.CLICK
        if button == "right":
            action_type = ActionType.RIGHT_CLICK
        
        action = RecordedAction(
            action_id="",
            action_type=action_type,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            x=x,
            y=y,
            target_element=element_info
        )
        
        return await self.record_action(action)
    
    async def record_double_click(self, x: float, y: float, 
                                 element_info: Optional[ElementInfo] = None) -> bool:
        """Record a double-click action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.DOUBLE_CLICK,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            x=x,
            y=y,
            target_element=element_info
        )
        
        return await self.record_action(action)
    
    async def record_type(self, text: str, element_info: Optional[ElementInfo] = None) -> bool:
        """Record a typing action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.TYPE,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            text_input=text,
            target_element=element_info
        )
        
        return await self.record_action(action)
    
    async def record_key_press(self, key_code: str, modifiers: List[str] = None) -> bool:
        """Record a key press action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.KEY_PRESS,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            key_code=key_code,
            key_modifiers=modifiers or []
        )
        
        return await self.record_action(action)
    
    async def record_scroll(self, x: float, y: float, direction: str, amount: int) -> bool:
        """Record a scroll action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.SCROLL,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            x=x,
            y=y,
            scroll_direction=direction,
            scroll_amount=amount
        )
        
        return await self.record_action(action)
    
    async def record_drag_drop(self, start_x: float, start_y: float, 
                              end_x: float, end_y: float,
                              source_element: Optional[ElementInfo] = None,
                              target_element: Optional[ElementInfo] = None) -> bool:
        """Record a drag and drop action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.DRAG_DROP,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            target_element=target_element
        )
        
        return await self.record_action(action)
    
    async def record_wait(self, duration: float, condition: str = None) -> bool:
        """Record a wait action."""
        action = RecordedAction(
            action_id="",
            action_type=ActionType.WAIT,
            timestamp=time.time(),
            screen_context=await self._get_current_context(),
            wait_duration=duration,
            wait_condition=condition
        )
        
        return await self.record_action(action)
    
    async def _capture_initial_context(self):
        """Capture the initial screen context."""
        try:
            context = await self._get_current_context()
            self.current_context = context
            self.context_history.append(context)
            
        except Exception as e:
            self.logger.error(f"Failed to capture initial context: {e}")
    
    async def _get_current_context(self) -> ScreenContext:
        """Get the current screen context."""
        try:
            # This would integrate with system APIs to get window information
            # For now, return a basic context
            context = ScreenContext(
                window_title="Unknown Window",
                application_name="Unknown Application",
                timestamp=time.time()
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get current context: {e}")
            return ScreenContext(
                window_title="Error",
                application_name="Error",
                timestamp=time.time()
            )
    
    async def _update_context(self, new_context: ScreenContext):
        """Update the current context and detect changes."""
        try:
            if (self.current_context and 
                (self.current_context.window_title != new_context.window_title or
                 self.current_context.application_name != new_context.application_name)):
                
                # Context change detected
                self.stats['context_switches'] += 1
                self.context_history.append(new_context)
                
                self.logger.debug(f"Context change: {new_context.application_name} - {new_context.window_title}")
            
            self.current_context = new_context
            
        except Exception as e:
            self.logger.error(f"Failed to update context: {e}")
    
    async def _detect_wait_actions(self):
        """Automatically detect wait actions between recorded actions."""
        try:
            if len(self.actions) < 2:
                return
            
            last_action = self.actions[-1]
            previous_action = self.actions[-2]
            
            # Calculate time gap
            time_gap = last_action.timestamp - previous_action.timestamp
            
            # If gap is significant, insert a wait action
            if time_gap > 2.0:  # 2 seconds threshold
                wait_action = RecordedAction(
                    action_id=f"{self.session_id}_{self.action_counter:04d}_wait",
                    action_type=ActionType.WAIT,
                    timestamp=previous_action.timestamp + 0.1,
                    screen_context=previous_action.screen_context,
                    wait_duration=time_gap - 0.1,
                    wait_condition="time_delay",
                    is_user_initiated=False
                )
                
                # Insert wait action before the last action
                self.actions.insert(-1, wait_action)
                
                self.logger.debug(f"Auto-detected wait: {time_gap:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to detect wait actions: {e}")
    
    def get_actions_by_type(self, action_type: ActionType) -> List[RecordedAction]:
        """Get all actions of a specific type."""
        return [action for action in self.actions if action.action_type == action_type]
    
    def get_actions_in_timeframe(self, start_time: float, end_time: float) -> List[RecordedAction]:
        """Get actions within a specific timeframe."""
        return [action for action in self.actions 
                if start_time <= action.timestamp <= end_time]
    
    def get_actions_by_context(self, application_name: str = None, 
                              window_title: str = None) -> List[RecordedAction]:
        """Get actions filtered by context."""
        filtered_actions = []
        
        for action in self.actions:
            context = action.screen_context
            
            if application_name and context.application_name != application_name:
                continue
            
            if window_title and context.window_title != window_title:
                continue
            
            filtered_actions.append(action)
        
        return filtered_actions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'is_recording': self.is_recording,
            'is_paused': self.is_paused,
            'settings': self.settings,
            'stats': self.stats,
            'actions': [asdict(action) for action in self.actions],
            'context_history': [asdict(context) for context in self.context_history]
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load session from dictionary."""
        self.session_id = data.get('session_id', self.session_id)
        self.session_name = data.get('session_name', self.session_name)
        self.start_time = data.get('start_time', self.start_time)
        self.end_time = data.get('end_time')
        self.is_recording = data.get('is_recording', False)
        self.is_paused = data.get('is_paused', False)
        self.settings.update(data.get('settings', {}))
        self.stats.update(data.get('stats', {}))
        
        # Load actions
        self.actions = []
        for action_data in data.get('actions', []):
            # Convert dictionaries back to dataclasses
            screen_context = ScreenContext(**action_data['screen_context'])
            
            target_element = None
            if action_data.get('target_element'):
                target_element = ElementInfo(**action_data['target_element'])
            
            action = RecordedAction(
                **{k: v for k, v in action_data.items() 
                   if k not in ['screen_context', 'target_element']},
                screen_context=screen_context,
                target_element=target_element
            )
            action.action_type = ActionType(action.action_type)
            
            self.actions.append(action)
        
        # Load context history
        self.context_history = []
        for context_data in data.get('context_history', []):
            context = ScreenContext(**context_data)
            self.context_history.append(context)
    
    async def save_to_file(self, file_path: Path) -> bool:
        """Save session to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Session saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    async def load_from_file(self, file_path: Path) -> bool:
        """Load session from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.from_dict(data)
            
            self.logger.info(f"Session loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False