"""
Smart Macro Recorder

Main interface for intelligent macro recording that combines recording, pattern detection,
and action generalization to create robust, reusable automation workflows.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .recording_session import RecordingSession, RecordedAction, ActionType
from .pattern_detector import PatternDetector, PatternAnalysis
from .action_generalizer import ActionGeneralizer, GeneralizationResult


class RecorderState(Enum):
    """States of the smart macro recorder."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    ANALYZING = "analyzing"
    GENERALIZING = "generalizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RecordingConfig:
    """Configuration for smart macro recording."""
    # Recording settings
    capture_mouse_moves: bool = False
    capture_scrolls: bool = True
    capture_key_combinations: bool = True
    min_action_interval: float = 0.1  # seconds
    
    # Pattern detection settings
    enable_pattern_detection: bool = True
    pattern_confidence_threshold: float = 0.7
    
    # Generalization settings
    enable_generalization: bool = True
    generalization_confidence_threshold: float = 0.7
    
    # Output settings
    auto_save: bool = True
    save_directory: Optional[str] = None
    export_formats: List[str] = field(default_factory=lambda: ['json', 'python'])


@dataclass
class SmartRecordingResult:
    """Result of smart macro recording."""
    session_id: str
    original_actions: List[RecordedAction]
    pattern_analysis: Optional[PatternAnalysis]
    generalization_result: Optional[GeneralizationResult]
    
    # Metrics
    recording_duration: float
    total_actions: int
    patterns_detected: int
    optimization_potential: float
    
    # Generated outputs
    generated_code: Dict[str, str] = field(default_factory=dict)
    workflow_description: str = ""
    
    # Quality assessment
    automation_confidence: float = 0.0
    robustness_score: float = 0.0
    reusability_score: float = 0.0


class SmartMacroRecorder:
    """
    Intelligent macro recorder that captures user actions, detects patterns,
    and generates robust, reusable automation workflows.
    """
    
    def __init__(self, config: Optional[RecordingConfig] = None):
        """Initialize the smart macro recorder."""
        self.logger = logging.getLogger(__name__)
        self.config = config or RecordingConfig()
        
        # Core components
        self.recording_session = RecordingSession()
        self.pattern_detector = PatternDetector()
        self.action_generalizer = ActionGeneralizer()
        
        # State management
        self.state = RecorderState.IDLE
        self.current_session_id: Optional[str] = None
        self.recording_start_time: Optional[float] = None
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'recording_started': [],
            'recording_stopped': [],
            'action_recorded': [],
            'pattern_detected': [],
            'analysis_completed': [],
            'error_occurred': []
        }
        
        # Results storage
        self.results: Dict[str, SmartRecordingResult] = {}
        
        # Statistics
        self.stats = {
            'total_recordings': 0,
            'total_actions_recorded': 0,
            'patterns_detected': 0,
            'successful_generalizations': 0
        }
    
    async def start_recording(self, session_name: Optional[str] = None) -> str:
        """
        Start a new smart recording session.
        
        Args:
            session_name: Optional name for the recording session
            
        Returns:
            str: Session ID for the recording
        """
        try:
            if self.state != RecorderState.IDLE:
                raise ValueError(f"Cannot start recording in state: {self.state}")
            
            self.logger.info("Starting smart macro recording")
            self.state = RecorderState.RECORDING
            
            # Start recording session
            session_id = await self.recording_session.start_recording(session_name)
            self.current_session_id = session_id
            self.recording_start_time = asyncio.get_event_loop().time()
            
            # Configure recording based on settings
            await self._configure_recording_session()
            
            # Notify callbacks
            await self._notify_callbacks('recording_started', {'session_id': session_id})
            
            self.logger.info(f"Smart recording started with session ID: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.state = RecorderState.ERROR
            await self._notify_callbacks('error_occurred', {'error': str(e)})
            raise
    
    async def stop_recording(self) -> SmartRecordingResult:
        """
        Stop the current recording session and process the results.
        
        Returns:
            SmartRecordingResult: Complete analysis and generalization results
        """
        try:
            if self.state != RecorderState.RECORDING:
                raise ValueError(f"Cannot stop recording in state: {self.state}")
            
            self.logger.info("Stopping smart macro recording")
            
            # Stop recording session
            await self.recording_session.stop_recording()
            recording_duration = asyncio.get_event_loop().time() - self.recording_start_time
            
            # Get recorded actions
            actions = self.recording_session.get_actions()
            
            # Notify callbacks
            await self._notify_callbacks('recording_stopped', {
                'session_id': self.current_session_id,
                'action_count': len(actions),
                'duration': recording_duration
            })
            
            # Process the recording
            result = await self._process_recording(actions, recording_duration)
            
            # Store result
            self.results[self.current_session_id] = result
            
            # Auto-save if enabled
            if self.config.auto_save:
                await self._save_result(result)
            
            # Update statistics
            self.stats['total_recordings'] += 1
            self.stats['total_actions_recorded'] += len(actions)
            
            # Reset state
            self.state = RecorderState.COMPLETED
            self.current_session_id = None
            self.recording_start_time = None
            
            self.logger.info(f"Smart recording completed: {len(actions)} actions processed")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            self.state = RecorderState.ERROR
            await self._notify_callbacks('error_occurred', {'error': str(e)})
            raise
    
    async def pause_recording(self):
        """Pause the current recording session."""
        if self.state == RecorderState.RECORDING:
            await self.recording_session.pause_recording()
            self.state = RecorderState.PAUSED
            self.logger.info("Recording paused")
    
    async def resume_recording(self):
        """Resume the paused recording session."""
        if self.state == RecorderState.PAUSED:
            await self.recording_session.resume_recording()
            self.state = RecorderState.RECORDING
            self.logger.info("Recording resumed")
    
    async def add_manual_action(self, action: RecordedAction):
        """
        Manually add an action to the current recording.
        
        Args:
            action: The action to add
        """
        if self.state in [RecorderState.RECORDING, RecorderState.PAUSED]:
            await self.recording_session.add_action(action)
            await self._notify_callbacks('action_recorded', {'action': action})
    
    async def analyze_existing_recording(self, actions: List[RecordedAction]) -> SmartRecordingResult:
        """
        Analyze an existing list of recorded actions.
        
        Args:
            actions: List of recorded actions to analyze
            
        Returns:
            SmartRecordingResult: Analysis and generalization results
        """
        try:
            self.logger.info(f"Analyzing existing recording with {len(actions)} actions")
            
            # Calculate duration from actions
            if actions:
                duration = actions[-1].timestamp - actions[0].timestamp
            else:
                duration = 0.0
            
            # Process the actions
            result = await self._process_recording(actions, duration)
            
            self.logger.info("Analysis of existing recording completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze existing recording: {e}")
            raise
    
    async def _process_recording(self, actions: List[RecordedAction], 
                                duration: float) -> SmartRecordingResult:
        """Process recorded actions through pattern detection and generalization."""
        session_id = self.current_session_id or "analysis_session"
        
        # Initialize result
        result = SmartRecordingResult(
            session_id=session_id,
            original_actions=actions,
            pattern_analysis=None,
            generalization_result=None,
            recording_duration=duration,
            total_actions=len(actions),
            patterns_detected=0,
            optimization_potential=0.0
        )
        
        if not actions:
            return result
        
        # Pattern detection
        if self.config.enable_pattern_detection:
            result.pattern_analysis = await self._detect_patterns(actions)
            result.patterns_detected = len(result.pattern_analysis.patterns_found)
            result.optimization_potential = result.pattern_analysis.optimization_potential
            
            # Update statistics
            self.stats['patterns_detected'] += result.patterns_detected
        
        # Action generalization
        if self.config.enable_generalization:
            patterns = result.pattern_analysis.patterns_found if result.pattern_analysis else []
            result.generalization_result = await self._generalize_actions(actions, patterns)
            
            if result.generalization_result:
                result.automation_confidence = result.generalization_result.generalization_confidence
                result.robustness_score = result.generalization_result.robustness_score
                result.reusability_score = result.generalization_result.reusability_score
                
                # Update statistics
                if result.automation_confidence >= self.config.generalization_confidence_threshold:
                    self.stats['successful_generalizations'] += 1
        
        # Generate code and descriptions
        await self._generate_outputs(result)
        
        return result
    
    async def _detect_patterns(self, actions: List[RecordedAction]) -> PatternAnalysis:
        """Detect patterns in recorded actions."""
        try:
            self.state = RecorderState.ANALYZING
            self.logger.info("Detecting patterns in recorded actions")
            
            analysis = await self.pattern_detector.analyze_actions(actions)
            
            # Filter patterns by confidence threshold
            filtered_patterns = [
                p for p in analysis.patterns_found 
                if p.confidence >= self.config.pattern_confidence_threshold
            ]
            analysis.patterns_found = filtered_patterns
            
            # Notify callbacks
            await self._notify_callbacks('pattern_detected', {
                'patterns': filtered_patterns,
                'analysis': analysis
            })
            
            self.logger.info(f"Pattern detection completed: {len(filtered_patterns)} patterns found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            raise
    
    async def _generalize_actions(self, actions: List[RecordedAction], 
                                 patterns: List) -> GeneralizationResult:
        """Generalize recorded actions."""
        try:
            self.state = RecorderState.GENERALIZING
            self.logger.info("Generalizing recorded actions")
            
            result = await self.action_generalizer.generalize_actions(actions, patterns)
            
            # Filter by confidence threshold
            if result.generalization_confidence < self.config.generalization_confidence_threshold:
                self.logger.warning(f"Generalization confidence ({result.generalization_confidence:.2f}) "
                                  f"below threshold ({self.config.generalization_confidence_threshold})")
            
            self.logger.info("Action generalization completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Action generalization failed: {e}")
            raise
    
    async def _generate_outputs(self, result: SmartRecordingResult):
        """Generate code and descriptions from the analysis results."""
        try:
            # Generate workflow description
            result.workflow_description = self._generate_workflow_description(result)
            
            # Generate code in different formats
            if 'python' in self.config.export_formats:
                result.generated_code['python'] = self._generate_python_code(result)
            
            if 'json' in self.config.export_formats:
                result.generated_code['json'] = self._generate_json_workflow(result)
            
            if 'yaml' in self.config.export_formats:
                result.generated_code['yaml'] = self._generate_yaml_workflow(result)
            
        except Exception as e:
            self.logger.error(f"Failed to generate outputs: {e}")
    
    def _generate_workflow_description(self, result: SmartRecordingResult) -> str:
        """Generate a human-readable description of the workflow."""
        description_parts = []
        
        # Basic info
        description_parts.append(f"Recorded workflow with {result.total_actions} actions")
        description_parts.append(f"Duration: {result.recording_duration:.1f} seconds")
        
        # Pattern information
        if result.pattern_analysis and result.patterns_detected > 0:
            description_parts.append(f"Detected {result.patterns_detected} patterns:")
            
            for pattern in result.pattern_analysis.patterns_found:
                pattern_desc = f"- {pattern.pattern_type.value.title()}: {pattern.description}"
                if pattern.repetition_count:
                    pattern_desc += f" (repeated {pattern.repetition_count} times)"
                description_parts.append(pattern_desc)
        
        # Quality metrics
        if result.generalization_result:
            description_parts.append(f"Automation confidence: {result.automation_confidence:.1%}")
            description_parts.append(f"Robustness score: {result.robustness_score:.1%}")
            description_parts.append(f"Reusability score: {result.reusability_score:.1%}")
        
        # Optimization potential
        if result.optimization_potential > 0:
            description_parts.append(f"Optimization potential: {result.optimization_potential:.1%}")
        
        return "\n".join(description_parts)
    
    def _generate_python_code(self, result: SmartRecordingResult) -> str:
        """Generate Python code for the workflow."""
        code_lines = [
            "# Generated automation workflow",
            "import asyncio",
            "from intelligent_automation_engine import AutomationEngine",
            "",
            "async def recorded_workflow():",
            '    """Generated from smart macro recording."""',
            "    engine = AutomationEngine()",
            ""
        ]
        
        if result.generalization_result:
            # Use generalized actions
            for i, action in enumerate(result.generalization_result.generalized_actions):
                code_lines.extend(self._generate_action_code(action, i))
        else:
            # Use original actions
            for i, action in enumerate(result.original_actions):
                code_lines.extend(self._generate_simple_action_code(action, i))
        
        code_lines.extend([
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(recorded_workflow())"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_action_code(self, action, index: int) -> List[str]:
        """Generate Python code for a generalized action."""
        lines = [f"    # Action {index + 1}: {action.action_type.value}"]
        
        if action.action_type == ActionType.CLICK:
            if action.element_selector:
                lines.append(f'    await engine.click_element("{action.element_selector}")')
            else:
                lines.append(f"    await engine.click({action.original_action.x}, {action.original_action.y})")
        
        elif action.action_type == ActionType.TYPE:
            text = action.variables.get('text_input', action.original_action.text_input)
            lines.append(f'    await engine.type_text("{text}")')
        
        elif action.action_type == ActionType.KEY_PRESS:
            lines.append(f'    await engine.press_key("{action.original_action.key}")')
        
        elif action.action_type == ActionType.SCROLL:
            lines.append(f"    await engine.scroll({action.original_action.scroll_x}, {action.original_action.scroll_y})")
        
        # Add timing if specified
        if action.timing_strategy:
            if 'smart_wait' in action.timing_strategy:
                lines.append("    await engine.smart_wait()")
            elif 'fixed_delay' in action.timing_strategy:
                delay = action.parameters.get('delay', 1.0)
                lines.append(f"    await asyncio.sleep({delay})")
        
        lines.append("")
        return lines
    
    def _generate_simple_action_code(self, action: RecordedAction, index: int) -> List[str]:
        """Generate simple Python code for an original action."""
        lines = [f"    # Action {index + 1}: {action.action_type.value}"]
        
        if action.action_type == ActionType.CLICK:
            lines.append(f"    await engine.click({action.x}, {action.y})")
        elif action.action_type == ActionType.TYPE:
            lines.append(f'    await engine.type_text("{action.text_input}")')
        elif action.action_type == ActionType.KEY_PRESS:
            lines.append(f'    await engine.press_key("{action.key}")')
        elif action.action_type == ActionType.SCROLL:
            lines.append(f"    await engine.scroll({action.scroll_x}, {action.scroll_y})")
        
        lines.append("")
        return lines
    
    def _generate_json_workflow(self, result: SmartRecordingResult) -> str:
        """Generate JSON representation of the workflow."""
        workflow_data = {
            'metadata': {
                'session_id': result.session_id,
                'total_actions': result.total_actions,
                'recording_duration': result.recording_duration,
                'patterns_detected': result.patterns_detected,
                'automation_confidence': result.automation_confidence,
                'generated_at': asyncio.get_event_loop().time()
            },
            'actions': []
        }
        
        if result.generalization_result:
            for action in result.generalization_result.generalized_actions:
                action_data = {
                    'action_id': action.action_id,
                    'action_type': action.action_type.value,
                    'element_selector': action.element_selector,
                    'position_strategy': action.position_strategy,
                    'text_pattern': action.text_pattern,
                    'timing_strategy': action.timing_strategy,
                    'parameters': action.parameters,
                    'variables': action.variables,
                    'confidence': action.confidence,
                    'fallback_strategies': action.fallback_strategies
                }
                workflow_data['actions'].append(action_data)
        else:
            for i, action in enumerate(result.original_actions):
                action_data = {
                    'action_id': f'action_{i}',
                    'action_type': action.action_type.value,
                    'x': action.x,
                    'y': action.y,
                    'text_input': action.text_input,
                    'key': action.key,
                    'timestamp': action.timestamp
                }
                workflow_data['actions'].append(action_data)
        
        return json.dumps(workflow_data, indent=2)
    
    def _generate_yaml_workflow(self, result: SmartRecordingResult) -> str:
        """Generate YAML representation of the workflow."""
        # Simple YAML generation (would use PyYAML in production)
        lines = [
            f"session_id: {result.session_id}",
            f"total_actions: {result.total_actions}",
            f"recording_duration: {result.recording_duration}",
            f"automation_confidence: {result.automation_confidence}",
            "",
            "actions:"
        ]
        
        if result.generalization_result:
            for action in result.generalization_result.generalized_actions:
                lines.extend([
                    f"  - action_id: {action.action_id}",
                    f"    action_type: {action.action_type.value}",
                    f"    element_selector: {action.element_selector}",
                    f"    confidence: {action.confidence}",
                    ""
                ])
        
        return "\n".join(lines)
    
    async def _configure_recording_session(self):
        """Configure the recording session based on settings."""
        # Configure what to capture
        self.recording_session.capture_mouse_moves = self.config.capture_mouse_moves
        self.recording_session.capture_scrolls = self.config.capture_scrolls
        self.recording_session.capture_key_combinations = self.config.capture_key_combinations
        self.recording_session.min_action_interval = self.config.min_action_interval
    
    async def _save_result(self, result: SmartRecordingResult):
        """Save the recording result to disk."""
        try:
            if not self.config.save_directory:
                return
            
            save_dir = Path(self.config.save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save in requested formats
            for format_type, code in result.generated_code.items():
                filename = f"{result.session_id}.{format_type}"
                filepath = save_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                self.logger.info(f"Saved {format_type} workflow to {filepath}")
            
            # Save description
            desc_file = save_dir / f"{result.session_id}_description.txt"
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(result.workflow_description)
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks of events."""
        try:
            for callback in self.callbacks.get(event_type, []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            self.logger.error(f"Callback notification failed for {event_type}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for recording events.
        
        Args:
            event_type: Type of event ('recording_started', 'recording_stopped', etc.)
            callback: Callback function to register
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def get_current_state(self) -> RecorderState:
        """Get the current state of the recorder."""
        return self.state
    
    def get_recording_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current recording session."""
        if self.current_session_id and self.recording_start_time:
            current_time = asyncio.get_event_loop().time()
            return {
                'session_id': self.current_session_id,
                'state': self.state.value,
                'duration': current_time - self.recording_start_time,
                'action_count': len(self.recording_session.get_actions())
            }
        return None
    
    def get_result(self, session_id: str) -> Optional[SmartRecordingResult]:
        """Get the result for a specific session."""
        return self.results.get(session_id)
    
    def get_all_results(self) -> Dict[str, SmartRecordingResult]:
        """Get all recording results."""
        return self.results.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return self.stats.copy()
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.state == RecorderState.RECORDING:
            await self.stop_recording()
        
        # Clear caches
        self.pattern_detector.clear_cache()
        self.results.clear()