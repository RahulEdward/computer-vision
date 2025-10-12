"""
Visual Interface Builder

Creates and manages the drag-and-drop visual programming interface
for building automation workflows.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from .automation_blocks import AutomationBlock, BlockFactory, BlockType, BlockCategory, BlockPosition
from .flow_manager import FlowManager
from .canvas_renderer import CanvasRenderer


@dataclass
class InterfaceConfig:
    """Configuration for the visual interface."""
    canvas_width: int = 1200
    canvas_height: int = 800
    grid_size: int = 20
    snap_to_grid: bool = True
    auto_save: bool = True
    theme: str = "light"  # light, dark
    show_minimap: bool = True
    show_properties_panel: bool = True
    show_block_palette: bool = True


@dataclass
class BlockPalette:
    """Configuration for the block palette."""
    categories: Dict[BlockCategory, List[BlockType]]
    collapsed_categories: List[BlockCategory]
    search_filter: str = ""


class VisualInterfaceBuilder:
    """
    Main class for the visual programming interface that allows users
    to create automation workflows using drag-and-drop blocks.
    """
    
    def __init__(self, config: Optional[InterfaceConfig] = None):
        """Initialize the visual interface builder."""
        self.logger = logging.getLogger(__name__)
        self.config = config or InterfaceConfig()
        
        # Core components
        self.flow_manager = FlowManager()
        self.canvas_renderer = CanvasRenderer(self.config)
        
        # Interface state
        self.selected_blocks: List[str] = []
        self.clipboard: List[Dict[str, Any]] = []
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []
        
        # Block palette
        self.block_palette = self._create_default_palette()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'block_added': [],
            'block_removed': [],
            'block_selected': [],
            'connection_created': [],
            'connection_removed': [],
            'workflow_changed': [],
            'validation_changed': []
        }
        
        # Current workflow
        self.current_workflow_path: Optional[Path] = None
        self.is_modified = False
    
    def _create_default_palette(self) -> BlockPalette:
        """Create the default block palette."""
        categories = {
            BlockCategory.SYSTEM: [
                BlockType.START,
                BlockType.END,
                BlockType.ERROR_HANDLER
            ],
            BlockCategory.BASIC: [
                BlockType.CLICK,
                BlockType.TYPE,
                BlockType.WAIT,
                BlockType.NAVIGATE,
                BlockType.SCROLL,
                BlockType.DRAG_DROP
            ],
            BlockCategory.CONTROL_FLOW: [
                BlockType.CONDITION,
                BlockType.LOOP,
                BlockType.PARALLEL,
                BlockType.SEQUENCE
            ],
            BlockCategory.DATA: [
                BlockType.SET_VARIABLE,
                BlockType.GET_VARIABLE,
                BlockType.EXTRACT,
                BlockType.COPY,
                BlockType.PASTE,
                BlockType.SAVE,
                BlockType.LOAD
            ],
            BlockCategory.AI: [
                BlockType.AI_DECISION,
                BlockType.SMART_WAIT,
                BlockType.ADAPTIVE_CLICK
            ]
        }
        
        return BlockPalette(
            categories=categories,
            collapsed_categories=[]
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the visual interface.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            self.logger.info("Initializing visual interface builder")
            
            # Initialize canvas renderer
            await self.canvas_renderer.initialize()
            
            # Create default workflow with start and end blocks
            await self._create_default_workflow()
            
            self.logger.info("Visual interface builder initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visual interface: {e}")
            return False
    
    async def _create_default_workflow(self):
        """Create a default workflow with start and end blocks."""
        # Add start block
        start_position = BlockPosition(x=100, y=200)
        start_block = BlockFactory.create_block(BlockType.START, start_position)
        self.flow_manager.add_block(start_block)
        
        # Add end block
        end_position = BlockPosition(x=500, y=200)
        end_block = BlockFactory.create_block(BlockType.END, end_position)
        self.flow_manager.add_block(end_block)
    
    async def add_block_from_palette(self, block_type: BlockType, position: BlockPosition) -> Optional[str]:
        """
        Add a block from the palette to the canvas.
        
        Args:
            block_type: Type of block to add
            position: Position on the canvas
            
        Returns:
            str: Block ID if successful, None otherwise
        """
        try:
            # Snap to grid if enabled
            if self.config.snap_to_grid:
                position.x = round(position.x / self.config.grid_size) * self.config.grid_size
                position.y = round(position.y / self.config.grid_size) * self.config.grid_size
            
            # Create block
            block = BlockFactory.create_block(block_type, position)
            
            # Add to flow manager
            if self.flow_manager.add_block(block):
                # Save state for undo
                self._save_state()
                
                # Trigger events
                await self._trigger_event('block_added', block.block_id)
                await self._trigger_event('workflow_changed')
                
                self.is_modified = True
                
                self.logger.info(f"Added block {block.block_id} at ({position.x}, {position.y})")
                return block.block_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            return None
    
    async def remove_block(self, block_id: str) -> bool:
        """
        Remove a block from the canvas.
        
        Args:
            block_id: ID of block to remove
            
        Returns:
            bool: True if successful
        """
        try:
            # Remove from selection
            if block_id in self.selected_blocks:
                self.selected_blocks.remove(block_id)
            
            # Remove from flow manager
            if self.flow_manager.remove_block(block_id):
                # Save state for undo
                self._save_state()
                
                # Trigger events
                await self._trigger_event('block_removed', block_id)
                await self._trigger_event('workflow_changed')
                
                self.is_modified = True
                
                self.logger.info(f"Removed block {block_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove block: {e}")
            return False
    
    async def move_block(self, block_id: str, new_position: BlockPosition) -> bool:
        """
        Move a block to a new position.
        
        Args:
            block_id: ID of block to move
            new_position: New position
            
        Returns:
            bool: True if successful
        """
        try:
            block = self.flow_manager.blocks.get(block_id)
            if not block:
                return False
            
            # Snap to grid if enabled
            if self.config.snap_to_grid:
                new_position.x = round(new_position.x / self.config.grid_size) * self.config.grid_size
                new_position.y = round(new_position.y / self.config.grid_size) * self.config.grid_size
            
            # Update position
            block.position = new_position
            
            # Trigger events
            await self._trigger_event('workflow_changed')
            
            self.is_modified = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move block: {e}")
            return False
    
    async def create_connection(self, source_block_id: str, source_port_id: str,
                              target_block_id: str, target_port_id: str) -> bool:
        """
        Create a connection between two blocks.
        
        Args:
            source_block_id: Source block ID
            source_port_id: Source port ID
            target_block_id: Target block ID
            target_port_id: Target port ID
            
        Returns:
            bool: True if successful
        """
        try:
            connection = self.flow_manager.create_connection(
                source_block_id, source_port_id,
                target_block_id, target_port_id
            )
            
            if connection:
                # Save state for undo
                self._save_state()
                
                # Trigger events
                await self._trigger_event('connection_created', connection.connection_id)
                await self._trigger_event('workflow_changed')
                
                self.is_modified = True
                
                self.logger.info(f"Created connection {connection.connection_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return False
    
    async def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            bool: True if successful
        """
        try:
            if self.flow_manager.remove_connection(connection_id):
                # Save state for undo
                self._save_state()
                
                # Trigger events
                await self._trigger_event('connection_removed', connection_id)
                await self._trigger_event('workflow_changed')
                
                self.is_modified = True
                
                self.logger.info(f"Removed connection {connection_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove connection: {e}")
            return False
    
    async def select_block(self, block_id: str, multi_select: bool = False) -> bool:
        """
        Select a block.
        
        Args:
            block_id: Block ID to select
            multi_select: Whether to add to existing selection
            
        Returns:
            bool: True if successful
        """
        try:
            if not multi_select:
                # Clear existing selection
                for selected_id in self.selected_blocks:
                    block = self.flow_manager.blocks.get(selected_id)
                    if block:
                        block.is_selected = False
                
                self.selected_blocks.clear()
            
            # Add to selection
            if block_id not in self.selected_blocks:
                block = self.flow_manager.blocks.get(block_id)
                if block:
                    block.is_selected = True
                    self.selected_blocks.append(block_id)
                    
                    # Trigger event
                    await self._trigger_event('block_selected', block_id)
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to select block: {e}")
            return False
    
    async def clear_selection(self):
        """Clear the current selection."""
        for block_id in self.selected_blocks:
            block = self.flow_manager.blocks.get(block_id)
            if block:
                block.is_selected = False
        
        self.selected_blocks.clear()
    
    async def copy_selected_blocks(self):
        """Copy selected blocks to clipboard."""
        try:
            self.clipboard.clear()
            
            for block_id in self.selected_blocks:
                block = self.flow_manager.blocks.get(block_id)
                if block:
                    self.clipboard.append(block.to_dict())
            
            self.logger.info(f"Copied {len(self.clipboard)} blocks to clipboard")
            
        except Exception as e:
            self.logger.error(f"Failed to copy blocks: {e}")
    
    async def paste_blocks(self, offset_x: float = 50, offset_y: float = 50) -> List[str]:
        """
        Paste blocks from clipboard.
        
        Args:
            offset_x: X offset for pasted blocks
            offset_y: Y offset for pasted blocks
            
        Returns:
            List[str]: IDs of pasted blocks
        """
        try:
            pasted_block_ids = []
            
            for block_data in self.clipboard:
                # Create new block with offset position
                position = BlockPosition(
                    x=block_data['position']['x'] + offset_x,
                    y=block_data['position']['y'] + offset_y,
                    width=block_data['position']['width'],
                    height=block_data['position']['height']
                )
                
                block = BlockFactory.create_block(
                    BlockType(block_data['block_type']),
                    position
                )
                
                # Copy parameters
                block.parameters = block_data.get('parameters', {}).copy()
                block.properties = block_data.get('properties', {}).copy()
                
                # Add to flow manager
                if self.flow_manager.add_block(block):
                    pasted_block_ids.append(block.block_id)
            
            if pasted_block_ids:
                # Save state for undo
                self._save_state()
                
                # Trigger events
                await self._trigger_event('workflow_changed')
                
                self.is_modified = True
            
            self.logger.info(f"Pasted {len(pasted_block_ids)} blocks")
            return pasted_block_ids
            
        except Exception as e:
            self.logger.error(f"Failed to paste blocks: {e}")
            return []
    
    async def delete_selected_blocks(self) -> bool:
        """
        Delete selected blocks.
        
        Returns:
            bool: True if successful
        """
        try:
            blocks_to_delete = self.selected_blocks.copy()
            
            for block_id in blocks_to_delete:
                await self.remove_block(block_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete selected blocks: {e}")
            return False
    
    async def validate_workflow(self) -> Dict[str, Any]:
        """
        Validate the current workflow.
        
        Returns:
            Dict: Validation result
        """
        try:
            validation_result = self.flow_manager.validate_flow()
            
            # Trigger validation event
            await self._trigger_event('validation_changed', validation_result)
            
            return {
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate workflow: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': []
            }
    
    async def save_workflow(self, file_path: Optional[Path] = None) -> bool:
        """
        Save the current workflow.
        
        Args:
            file_path: Path to save to (uses current path if None)
            
        Returns:
            bool: True if successful
        """
        try:
            if file_path:
                self.current_workflow_path = file_path
            elif not self.current_workflow_path:
                self.logger.error("No file path specified for save")
                return False
            
            # Create workflow data
            workflow_data = {
                'version': '1.0',
                'metadata': {
                    'name': self.current_workflow_path.stem,
                    'description': '',
                    'created_at': '',
                    'modified_at': ''
                },
                'config': {
                    'canvas_width': self.config.canvas_width,
                    'canvas_height': self.config.canvas_height,
                    'grid_size': self.config.grid_size,
                    'snap_to_grid': self.config.snap_to_grid
                },
                'flow': self.flow_manager.to_dict()
            }
            
            # Save to file
            with open(self.current_workflow_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2)
            
            self.is_modified = False
            
            self.logger.info(f"Saved workflow to {self.current_workflow_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow: {e}")
            return False
    
    async def load_workflow(self, file_path: Path) -> bool:
        """
        Load a workflow from file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            bool: True if successful
        """
        try:
            # Load workflow data
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            # Update config if present
            if 'config' in workflow_data:
                config_data = workflow_data['config']
                self.config.canvas_width = config_data.get('canvas_width', self.config.canvas_width)
                self.config.canvas_height = config_data.get('canvas_height', self.config.canvas_height)
                self.config.grid_size = config_data.get('grid_size', self.config.grid_size)
                self.config.snap_to_grid = config_data.get('snap_to_grid', self.config.snap_to_grid)
            
            # Load flow
            if 'flow' in workflow_data:
                self.flow_manager.from_dict(workflow_data['flow'])
            
            self.current_workflow_path = file_path
            self.is_modified = False
            
            # Clear undo/redo stacks
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            # Trigger events
            await self._trigger_event('workflow_changed')
            
            self.logger.info(f"Loaded workflow from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow: {e}")
            return False
    
    async def export_workflow(self, format_type: str = "json") -> Optional[Dict[str, Any]]:
        """
        Export workflow in specified format.
        
        Args:
            format_type: Export format (json, python, etc.)
            
        Returns:
            Dict: Exported workflow data
        """
        try:
            if format_type == "json":
                return self.flow_manager.to_dict()
            elif format_type == "python":
                return await self._export_to_python()
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to export workflow: {e}")
            return None
    
    async def _export_to_python(self) -> Dict[str, Any]:
        """Export workflow as Python code."""
        # This would generate Python code from the visual workflow
        # Implementation would convert blocks and connections to Python automation code
        execution_order = self.flow_manager.get_execution_order()
        
        python_code = "# Generated automation workflow\n\n"
        python_code += "import asyncio\n"
        python_code += "from intelligent_automation_engine import AutomationEngine\n\n"
        python_code += "async def main():\n"
        python_code += "    engine = AutomationEngine()\n"
        python_code += "    await engine.initialize()\n\n"
        
        for block_id in execution_order:
            block = self.flow_manager.blocks.get(block_id)
            if block:
                python_code += f"    # {block.name}\n"
                python_code += f"    # {block.description}\n"
                # Add block-specific code generation here
                python_code += "\n"
        
        python_code += "if __name__ == '__main__':\n"
        python_code += "    asyncio.run(main())\n"
        
        return {
            'code': python_code,
            'execution_order': execution_order
        }
    
    def _save_state(self):
        """Save current state for undo functionality."""
        state = self.flow_manager.to_dict()
        self.undo_stack.append(state)
        
        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        
        # Clear redo stack
        self.redo_stack.clear()
    
    async def undo(self) -> bool:
        """Undo the last action."""
        try:
            if not self.undo_stack:
                return False
            
            # Save current state to redo stack
            current_state = self.flow_manager.to_dict()
            self.redo_stack.append(current_state)
            
            # Restore previous state
            previous_state = self.undo_stack.pop()
            self.flow_manager.from_dict(previous_state)
            
            # Trigger events
            await self._trigger_event('workflow_changed')
            
            self.is_modified = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to undo: {e}")
            return False
    
    async def redo(self) -> bool:
        """Redo the last undone action."""
        try:
            if not self.redo_stack:
                return False
            
            # Save current state to undo stack
            current_state = self.flow_manager.to_dict()
            self.undo_stack.append(current_state)
            
            # Restore next state
            next_state = self.redo_stack.pop()
            self.flow_manager.from_dict(next_state)
            
            # Trigger events
            await self._trigger_event('workflow_changed')
            
            self.is_modified = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to redo: {e}")
            return False
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove an event handler."""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _trigger_event(self, event_type: str, *args):
        """Trigger an event."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*args)
                    else:
                        handler(*args)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about the current workflow."""
        return {
            'total_blocks': len(self.flow_manager.blocks),
            'total_connections': len(self.flow_manager.connections),
            'block_types': {
                block_type.value: sum(1 for block in self.flow_manager.blocks.values() 
                                    if block.block_type == block_type)
                for block_type in BlockType
            },
            'execution_order_length': len(self.flow_manager.get_execution_order()),
            'parallel_groups': len(self.flow_manager.get_parallel_groups()),
            'is_valid': self.flow_manager.validate_flow().is_valid
        }