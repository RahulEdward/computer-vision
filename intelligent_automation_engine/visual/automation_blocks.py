"""
Automation Blocks

Defines the visual automation blocks that users can drag and drop
to create workflows in the visual programming interface.
"""

import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class BlockType(Enum):
    """Types of automation blocks."""
    # Basic Actions
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    DRAG_DROP = "drag_drop"
    
    # Data Operations
    COPY = "copy"
    PASTE = "paste"
    EXTRACT = "extract"
    SAVE = "save"
    LOAD = "load"
    
    # Control Flow
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    
    # Variables
    SET_VARIABLE = "set_variable"
    GET_VARIABLE = "get_variable"
    
    # Functions
    FUNCTION = "function"
    CALL_FUNCTION = "call_function"
    
    # Advanced
    AI_DECISION = "ai_decision"
    SMART_WAIT = "smart_wait"
    ADAPTIVE_CLICK = "adaptive_click"
    
    # System
    START = "start"
    END = "end"
    ERROR_HANDLER = "error_handler"


class BlockCategory(Enum):
    """Categories for organizing blocks."""
    BASIC = "basic"
    ADVANCED = "advanced"
    CONTROL_FLOW = "control_flow"
    DATA = "data"
    SYSTEM = "system"
    AI = "ai"


@dataclass
class BlockPort:
    """Represents an input or output port on a block."""
    port_id: str
    name: str
    data_type: str
    is_input: bool
    is_required: bool = True
    default_value: Any = None
    description: str = ""


@dataclass
class BlockConnection:
    """Represents a connection between two block ports."""
    connection_id: str
    source_block_id: str
    source_port_id: str
    target_block_id: str
    target_port_id: str
    data_type: str


@dataclass
class BlockPosition:
    """Position and size of a block on the canvas."""
    x: float
    y: float
    width: float = 200.0
    height: float = 100.0


@dataclass
class AutomationBlock:
    """
    Represents a visual automation block that can be connected
    to other blocks to create workflows.
    """
    block_id: str
    block_type: BlockType
    name: str
    description: str
    category: BlockCategory
    position: BlockPosition
    
    # Ports for connections
    input_ports: List[BlockPort] = field(default_factory=list)
    output_ports: List[BlockPort] = field(default_factory=list)
    
    # Block configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Visual properties
    color: str = "#4A90E2"
    icon: str = "⚙️"
    
    # Execution state
    is_enabled: bool = True
    is_selected: bool = False
    execution_status: str = "ready"  # ready, running, completed, error
    
    def __post_init__(self):
        """Initialize block after creation."""
        if not self.input_ports and not self.output_ports:
            self._setup_default_ports()
    
    def _setup_default_ports(self):
        """Setup default ports based on block type."""
        # Most blocks have a flow input and output
        if self.block_type not in [BlockType.START]:
            self.input_ports.append(BlockPort(
                port_id=f"{self.block_id}_flow_in",
                name="Flow In",
                data_type="flow",
                is_input=True,
                description="Execution flow input"
            ))
        
        if self.block_type not in [BlockType.END]:
            self.output_ports.append(BlockPort(
                port_id=f"{self.block_id}_flow_out",
                name="Flow Out",
                data_type="flow",
                is_input=False,
                description="Execution flow output"
            ))
        
        # Add specific ports based on block type
        self._add_type_specific_ports()
    
    def _add_type_specific_ports(self):
        """Add ports specific to the block type."""
        if self.block_type == BlockType.CLICK:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_target",
                    name="Target",
                    data_type="string",
                    is_input=True,
                    description="Element to click"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_selector_type",
                    name="Selector Type",
                    data_type="string",
                    is_input=True,
                    default_value="text",
                    description="Type of selector (text, id, class, xpath)"
                )
            ])
        
        elif self.block_type == BlockType.TYPE:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_target",
                    name="Target",
                    data_type="string",
                    is_input=True,
                    description="Element to type into"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_text",
                    name="Text",
                    data_type="string",
                    is_input=True,
                    description="Text to type"
                )
            ])
        
        elif self.block_type == BlockType.WAIT:
            self.input_ports.append(BlockPort(
                port_id=f"{self.block_id}_duration",
                name="Duration",
                data_type="number",
                is_input=True,
                default_value=1.0,
                description="Wait duration in seconds"
            ))
        
        elif self.block_type == BlockType.NAVIGATE:
            self.input_ports.append(BlockPort(
                port_id=f"{self.block_id}_url",
                name="URL",
                data_type="string",
                is_input=True,
                description="URL to navigate to"
            ))
        
        elif self.block_type == BlockType.CONDITION:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_condition",
                    name="Condition",
                    data_type="string",
                    is_input=True,
                    description="Condition to evaluate"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_true_flow",
                    name="True Flow",
                    data_type="flow",
                    is_input=True,
                    description="Flow when condition is true"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_false_flow",
                    name="False Flow",
                    data_type="flow",
                    is_input=True,
                    description="Flow when condition is false"
                )
            ])
            
            self.output_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_true_out",
                    name="True",
                    data_type="flow",
                    is_input=False,
                    description="Output when condition is true"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_false_out",
                    name="False",
                    data_type="flow",
                    is_input=False,
                    description="Output when condition is false"
                )
            ])
        
        elif self.block_type == BlockType.LOOP:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_count",
                    name="Count",
                    data_type="number",
                    is_input=True,
                    default_value=1,
                    description="Number of iterations"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_loop_body",
                    name="Loop Body",
                    data_type="flow",
                    is_input=True,
                    description="Flow to repeat"
                )
            ])
            
            self.output_ports.append(BlockPort(
                port_id=f"{self.block_id}_loop_out",
                name="Loop Out",
                data_type="flow",
                is_input=False,
                description="Output after loop completion"
            ))
        
        elif self.block_type == BlockType.SET_VARIABLE:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_var_name",
                    name="Variable Name",
                    data_type="string",
                    is_input=True,
                    description="Name of the variable"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_var_value",
                    name="Value",
                    data_type="any",
                    is_input=True,
                    description="Value to assign"
                )
            ])
        
        elif self.block_type == BlockType.GET_VARIABLE:
            self.input_ports.append(BlockPort(
                port_id=f"{self.block_id}_var_name",
                name="Variable Name",
                data_type="string",
                is_input=True,
                description="Name of the variable"
            ))
            
            self.output_ports.append(BlockPort(
                port_id=f"{self.block_id}_var_value",
                name="Value",
                data_type="any",
                is_input=False,
                description="Variable value"
            ))
        
        elif self.block_type == BlockType.EXTRACT:
            self.input_ports.extend([
                BlockPort(
                    port_id=f"{self.block_id}_target",
                    name="Target",
                    data_type="string",
                    is_input=True,
                    description="Element to extract from"
                ),
                BlockPort(
                    port_id=f"{self.block_id}_attribute",
                    name="Attribute",
                    data_type="string",
                    is_input=True,
                    default_value="text",
                    description="Attribute to extract"
                )
            ])
            
            self.output_ports.append(BlockPort(
                port_id=f"{self.block_id}_extracted_value",
                name="Extracted Value",
                data_type="string",
                is_input=False,
                description="Extracted value"
            ))
    
    def get_port_by_id(self, port_id: str) -> Optional[BlockPort]:
        """Get a port by its ID."""
        for port in self.input_ports + self.output_ports:
            if port.port_id == port_id:
                return port
        return None
    
    def get_input_port(self, name: str) -> Optional[BlockPort]:
        """Get an input port by name."""
        for port in self.input_ports:
            if port.name == name:
                return port
        return None
    
    def get_output_port(self, name: str) -> Optional[BlockPort]:
        """Get an output port by name."""
        for port in self.output_ports:
            if port.name == name:
                return port
        return None
    
    def set_parameter(self, name: str, value: Any):
        """Set a block parameter."""
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a block parameter."""
        return self.parameters.get(name, default)
    
    def validate_connections(self, connections: List[BlockConnection]) -> List[str]:
        """Validate connections to this block."""
        errors = []
        
        # Check required input ports
        connected_inputs = set()
        for conn in connections:
            if conn.target_block_id == self.block_id:
                connected_inputs.add(conn.target_port_id)
        
        for port in self.input_ports:
            if port.is_required and port.port_id not in connected_inputs:
                if port.default_value is None:
                    errors.append(f"Required input port '{port.name}' is not connected")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary representation."""
        return {
            'block_id': self.block_id,
            'block_type': self.block_type.value,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'position': {
                'x': self.position.x,
                'y': self.position.y,
                'width': self.position.width,
                'height': self.position.height
            },
            'input_ports': [
                {
                    'port_id': port.port_id,
                    'name': port.name,
                    'data_type': port.data_type,
                    'is_required': port.is_required,
                    'default_value': port.default_value,
                    'description': port.description
                }
                for port in self.input_ports
            ],
            'output_ports': [
                {
                    'port_id': port.port_id,
                    'name': port.name,
                    'data_type': port.data_type,
                    'description': port.description
                }
                for port in self.output_ports
            ],
            'parameters': self.parameters,
            'properties': self.properties,
            'color': self.color,
            'icon': self.icon,
            'is_enabled': self.is_enabled,
            'execution_status': self.execution_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomationBlock':
        """Create block from dictionary representation."""
        position = BlockPosition(
            x=data['position']['x'],
            y=data['position']['y'],
            width=data['position'].get('width', 200.0),
            height=data['position'].get('height', 100.0)
        )
        
        block = cls(
            block_id=data['block_id'],
            block_type=BlockType(data['block_type']),
            name=data['name'],
            description=data['description'],
            category=BlockCategory(data['category']),
            position=position,
            parameters=data.get('parameters', {}),
            properties=data.get('properties', {}),
            color=data.get('color', '#4A90E2'),
            icon=data.get('icon', '⚙️'),
            is_enabled=data.get('is_enabled', True)
        )
        
        # Restore ports
        block.input_ports = [
            BlockPort(
                port_id=port_data['port_id'],
                name=port_data['name'],
                data_type=port_data['data_type'],
                is_input=True,
                is_required=port_data.get('is_required', True),
                default_value=port_data.get('default_value'),
                description=port_data.get('description', '')
            )
            for port_data in data.get('input_ports', [])
        ]
        
        block.output_ports = [
            BlockPort(
                port_id=port_data['port_id'],
                name=port_data['name'],
                data_type=port_data['data_type'],
                is_input=False,
                description=port_data.get('description', '')
            )
            for port_data in data.get('output_ports', [])
        ]
        
        return block


class BlockFactory:
    """Factory for creating automation blocks."""
    
    @staticmethod
    def create_block(block_type: BlockType, position: BlockPosition, **kwargs) -> AutomationBlock:
        """Create a new automation block."""
        block_id = kwargs.get('block_id', str(uuid.uuid4()))
        
        # Block definitions
        block_definitions = {
            BlockType.START: {
                'name': 'Start',
                'description': 'Starting point of the workflow',
                'category': BlockCategory.SYSTEM,
                'color': '#28A745',
                'icon': '▶️'
            },
            BlockType.END: {
                'name': 'End',
                'description': 'End point of the workflow',
                'category': BlockCategory.SYSTEM,
                'color': '#DC3545',
                'icon': '⏹️'
            },
            BlockType.CLICK: {
                'name': 'Click',
                'description': 'Click on an element',
                'category': BlockCategory.BASIC,
                'color': '#007BFF',
                'icon': '👆'
            },
            BlockType.TYPE: {
                'name': 'Type',
                'description': 'Type text into an element',
                'category': BlockCategory.BASIC,
                'color': '#6F42C1',
                'icon': '⌨️'
            },
            BlockType.WAIT: {
                'name': 'Wait',
                'description': 'Wait for a specified duration',
                'category': BlockCategory.BASIC,
                'color': '#FFC107',
                'icon': '⏱️'
            },
            BlockType.NAVIGATE: {
                'name': 'Navigate',
                'description': 'Navigate to a URL',
                'category': BlockCategory.BASIC,
                'color': '#17A2B8',
                'icon': '🌐'
            },
            BlockType.SCROLL: {
                'name': 'Scroll',
                'description': 'Scroll the page',
                'category': BlockCategory.BASIC,
                'color': '#20C997',
                'icon': '📜'
            },
            BlockType.CONDITION: {
                'name': 'Condition',
                'description': 'Conditional branching',
                'category': BlockCategory.CONTROL_FLOW,
                'color': '#FD7E14',
                'icon': '❓'
            },
            BlockType.LOOP: {
                'name': 'Loop',
                'description': 'Repeat actions',
                'category': BlockCategory.CONTROL_FLOW,
                'color': '#E83E8C',
                'icon': '🔄'
            },
            BlockType.SET_VARIABLE: {
                'name': 'Set Variable',
                'description': 'Set a variable value',
                'category': BlockCategory.DATA,
                'color': '#6C757D',
                'icon': '📝'
            },
            BlockType.GET_VARIABLE: {
                'name': 'Get Variable',
                'description': 'Get a variable value',
                'category': BlockCategory.DATA,
                'color': '#6C757D',
                'icon': '📖'
            },
            BlockType.EXTRACT: {
                'name': 'Extract',
                'description': 'Extract data from an element',
                'category': BlockCategory.DATA,
                'color': '#495057',
                'icon': '📤'
            },
            BlockType.AI_DECISION: {
                'name': 'AI Decision',
                'description': 'AI-powered decision making',
                'category': BlockCategory.AI,
                'color': '#FF6B6B',
                'icon': '🤖'
            },
            BlockType.SMART_WAIT: {
                'name': 'Smart Wait',
                'description': 'Intelligent waiting for conditions',
                'category': BlockCategory.AI,
                'color': '#4ECDC4',
                'icon': '🧠'
            }
        }
        
        definition = block_definitions.get(block_type, {
            'name': block_type.value.title(),
            'description': f'{block_type.value} action',
            'category': BlockCategory.BASIC,
            'color': '#6C757D',
            'icon': '⚙️'
        })
        
        # Override with provided kwargs
        definition.update(kwargs)
        
        return AutomationBlock(
            block_id=block_id,
            block_type=block_type,
            position=position,
            **definition
        )