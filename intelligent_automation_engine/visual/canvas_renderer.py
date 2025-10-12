"""
Canvas Renderer

Handles the rendering of the visual programming interface canvas,
including blocks, connections, grid, and interactive elements.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .automation_blocks import AutomationBlock, BlockType, BlockCategory, BlockPosition
from .flow_manager import FlowManager


class RenderMode(Enum):
    """Rendering modes for the canvas."""
    NORMAL = "normal"
    PREVIEW = "preview"
    EXECUTION = "execution"
    DEBUG = "debug"


@dataclass
class CanvasTheme:
    """Theme configuration for the canvas."""
    # Background colors
    background_color: str = "#f8f9fa"
    grid_color: str = "#e9ecef"
    grid_major_color: str = "#dee2e6"
    
    # Block colors
    block_default_color: str = "#ffffff"
    block_border_color: str = "#6c757d"
    block_selected_color: str = "#007bff"
    block_error_color: str = "#dc3545"
    block_success_color: str = "#28a745"
    
    # Connection colors
    connection_color: str = "#6c757d"
    connection_selected_color: str = "#007bff"
    connection_error_color: str = "#dc3545"
    
    # Text colors
    text_primary: str = "#212529"
    text_secondary: str = "#6c757d"
    text_muted: str = "#adb5bd"
    
    # UI colors
    selection_color: str = "#007bff"
    hover_color: str = "#e9ecef"
    
    # Block category colors
    category_colors: Dict[BlockCategory, str] = None
    
    def __post_init__(self):
        if self.category_colors is None:
            self.category_colors = {
                BlockCategory.SYSTEM: "#6f42c1",
                BlockCategory.BASIC: "#007bff",
                BlockCategory.CONTROL_FLOW: "#fd7e14",
                BlockCategory.DATA: "#20c997",
                BlockCategory.AI: "#e83e8c"
            }


@dataclass
class ViewportState:
    """State of the canvas viewport."""
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    width: int = 1200
    height: int = 800


@dataclass
class RenderContext:
    """Context for rendering operations."""
    viewport: ViewportState
    theme: CanvasTheme
    mode: RenderMode
    show_grid: bool = True
    show_minimap: bool = True
    show_debug_info: bool = False


class CanvasRenderer:
    """
    Handles rendering of the visual programming canvas including blocks,
    connections, grid, and interactive elements.
    """
    
    def __init__(self, config):
        """Initialize the canvas renderer."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Rendering state
        self.viewport = ViewportState(
            width=config.canvas_width,
            height=config.canvas_height
        )
        
        # Theme
        self.theme = CanvasTheme()
        if config.theme == "dark":
            self._apply_dark_theme()
        
        # Render context
        self.context = RenderContext(
            viewport=self.viewport,
            theme=self.theme,
            mode=RenderMode.NORMAL,
            show_grid=True,
            show_minimap=config.show_minimap
        )
        
        # Cached render data
        self._cached_grid = None
        self._cached_blocks = {}
        self._cached_connections = {}
        
        # Performance metrics
        self.render_stats = {
            'frame_count': 0,
            'last_render_time': 0,
            'average_render_time': 0
        }
    
    def _apply_dark_theme(self):
        """Apply dark theme colors."""
        self.theme.background_color = "#1a1a1a"
        self.theme.grid_color = "#333333"
        self.theme.grid_major_color = "#444444"
        self.theme.block_default_color = "#2d2d2d"
        self.theme.block_border_color = "#555555"
        self.theme.text_primary = "#ffffff"
        self.theme.text_secondary = "#cccccc"
        self.theme.text_muted = "#888888"
        self.theme.hover_color = "#404040"
    
    async def initialize(self) -> bool:
        """
        Initialize the canvas renderer.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            self.logger.info("Initializing canvas renderer")
            
            # Pre-render grid
            self._render_grid_cache()
            
            self.logger.info("Canvas renderer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize canvas renderer: {e}")
            return False
    
    def _render_grid_cache(self):
        """Pre-render grid for better performance."""
        try:
            grid_data = {
                'minor_lines': [],
                'major_lines': [],
                'grid_size': self.config.grid_size
            }
            
            # Calculate grid lines
            viewport_left = self.viewport.x
            viewport_right = self.viewport.x + self.viewport.width / self.viewport.zoom
            viewport_top = self.viewport.y
            viewport_bottom = self.viewport.y + self.viewport.height / self.viewport.zoom
            
            # Minor grid lines
            start_x = math.floor(viewport_left / self.config.grid_size) * self.config.grid_size
            end_x = math.ceil(viewport_right / self.config.grid_size) * self.config.grid_size
            
            for x in range(int(start_x), int(end_x) + self.config.grid_size, self.config.grid_size):
                grid_data['minor_lines'].append({
                    'type': 'vertical',
                    'x': x,
                    'y1': viewport_top,
                    'y2': viewport_bottom
                })
            
            start_y = math.floor(viewport_top / self.config.grid_size) * self.config.grid_size
            end_y = math.ceil(viewport_bottom / self.config.grid_size) * self.config.grid_size
            
            for y in range(int(start_y), int(end_y) + self.config.grid_size, self.config.grid_size):
                grid_data['minor_lines'].append({
                    'type': 'horizontal',
                    'x1': viewport_left,
                    'x2': viewport_right,
                    'y': y
                })
            
            # Major grid lines (every 5th line)
            major_grid_size = self.config.grid_size * 5
            
            start_x = math.floor(viewport_left / major_grid_size) * major_grid_size
            end_x = math.ceil(viewport_right / major_grid_size) * major_grid_size
            
            for x in range(int(start_x), int(end_x) + major_grid_size, major_grid_size):
                grid_data['major_lines'].append({
                    'type': 'vertical',
                    'x': x,
                    'y1': viewport_top,
                    'y2': viewport_bottom
                })
            
            start_y = math.floor(viewport_top / major_grid_size) * major_grid_size
            end_y = math.ceil(viewport_bottom / major_grid_size) * major_grid_size
            
            for y in range(int(start_y), int(end_y) + major_grid_size, major_grid_size):
                grid_data['major_lines'].append({
                    'type': 'horizontal',
                    'x1': viewport_left,
                    'x2': viewport_right,
                    'y': y
                })
            
            self._cached_grid = grid_data
            
        except Exception as e:
            self.logger.error(f"Failed to render grid cache: {e}")
    
    def render_canvas(self, flow_manager: FlowManager) -> Dict[str, Any]:
        """
        Render the complete canvas.
        
        Args:
            flow_manager: Flow manager containing blocks and connections
            
        Returns:
            Dict: Rendered canvas data
        """
        try:
            render_start_time = self._get_current_time()
            
            # Create render data
            render_data = {
                'viewport': {
                    'x': self.viewport.x,
                    'y': self.viewport.y,
                    'zoom': self.viewport.zoom,
                    'width': self.viewport.width,
                    'height': self.viewport.height
                },
                'theme': self._get_theme_data(),
                'grid': self._render_grid() if self.context.show_grid else None,
                'blocks': self._render_blocks(flow_manager.blocks),
                'connections': self._render_connections(flow_manager.connections),
                'selection': self._render_selection(flow_manager),
                'minimap': self._render_minimap(flow_manager) if self.context.show_minimap else None,
                'debug_info': self._render_debug_info(flow_manager) if self.context.show_debug_info else None
            }
            
            # Update performance stats
            render_time = self._get_current_time() - render_start_time
            self._update_render_stats(render_time)
            
            return render_data
            
        except Exception as e:
            self.logger.error(f"Failed to render canvas: {e}")
            return {}
    
    def _get_theme_data(self) -> Dict[str, Any]:
        """Get theme data for rendering."""
        return {
            'background_color': self.theme.background_color,
            'grid_color': self.theme.grid_color,
            'grid_major_color': self.theme.grid_major_color,
            'block_default_color': self.theme.block_default_color,
            'block_border_color': self.theme.block_border_color,
            'block_selected_color': self.theme.block_selected_color,
            'connection_color': self.theme.connection_color,
            'connection_selected_color': self.theme.connection_selected_color,
            'text_primary': self.theme.text_primary,
            'text_secondary': self.theme.text_secondary,
            'selection_color': self.theme.selection_color,
            'category_colors': {cat.value: color for cat, color in self.theme.category_colors.items()}
        }
    
    def _render_grid(self) -> Dict[str, Any]:
        """Render the grid."""
        if not self._cached_grid:
            self._render_grid_cache()
        
        return self._cached_grid
    
    def _render_blocks(self, blocks: Dict[str, AutomationBlock]) -> List[Dict[str, Any]]:
        """
        Render all blocks.
        
        Args:
            blocks: Dictionary of blocks to render
            
        Returns:
            List: Rendered block data
        """
        rendered_blocks = []
        
        for block in blocks.values():
            # Check if block is in viewport
            if not self._is_block_in_viewport(block):
                continue
            
            # Get cached render data or create new
            block_key = f"{block.block_id}_{block.last_modified}"
            if block_key not in self._cached_blocks:
                self._cached_blocks[block_key] = self._render_block(block)
            
            rendered_blocks.append(self._cached_blocks[block_key])
        
        return rendered_blocks
    
    def _render_block(self, block: AutomationBlock) -> Dict[str, Any]:
        """
        Render a single block.
        
        Args:
            block: Block to render
            
        Returns:
            Dict: Rendered block data
        """
        # Get category color
        category_color = self.theme.category_colors.get(
            block.category, 
            self.theme.block_default_color
        )
        
        # Determine block state color
        if block.is_selected:
            border_color = self.theme.block_selected_color
        elif block.has_error:
            border_color = self.theme.block_error_color
        elif block.execution_state == "completed":
            border_color = self.theme.block_success_color
        else:
            border_color = self.theme.block_border_color
        
        return {
            'id': block.block_id,
            'type': block.block_type.value,
            'category': block.category.value,
            'name': block.name,
            'description': block.description,
            'position': {
                'x': block.position.x,
                'y': block.position.y,
                'width': block.position.width,
                'height': block.position.height
            },
            'style': {
                'background_color': category_color,
                'border_color': border_color,
                'text_color': self.theme.text_primary,
                'opacity': 0.5 if block.is_disabled else 1.0
            },
            'ports': {
                'input': [self._render_port(port) for port in block.input_ports],
                'output': [self._render_port(port) for port in block.output_ports]
            },
            'state': {
                'is_selected': block.is_selected,
                'is_disabled': block.is_disabled,
                'has_error': block.has_error,
                'execution_state': block.execution_state
            },
            'parameters': block.parameters,
            'properties': block.properties
        }
    
    def _render_port(self, port) -> Dict[str, Any]:
        """Render a block port."""
        return {
            'id': port.port_id,
            'name': port.name,
            'type': port.port_type.value,
            'data_type': port.data_type.value,
            'position': {
                'x': port.position.x,
                'y': port.position.y
            },
            'is_connected': port.is_connected,
            'is_required': port.is_required
        }
    
    def _render_connections(self, connections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Render all connections.
        
        Args:
            connections: Dictionary of connections to render
            
        Returns:
            List: Rendered connection data
        """
        rendered_connections = []
        
        for connection in connections.values():
            # Get cached render data or create new
            connection_key = f"{connection.connection_id}_{connection.last_modified}"
            if connection_key not in self._cached_connections:
                self._cached_connections[connection_key] = self._render_connection(connection)
            
            rendered_connections.append(self._cached_connections[connection_key])
        
        return rendered_connections
    
    def _render_connection(self, connection) -> Dict[str, Any]:
        """
        Render a single connection.
        
        Args:
            connection: Connection to render
            
        Returns:
            Dict: Rendered connection data
        """
        # Calculate connection path
        path = self._calculate_connection_path(connection)
        
        # Determine connection color
        if connection.is_selected:
            color = self.theme.connection_selected_color
        elif connection.has_error:
            color = self.theme.connection_error_color
        else:
            color = self.theme.connection_color
        
        return {
            'id': connection.connection_id,
            'source': {
                'block_id': connection.source_block_id,
                'port_id': connection.source_port_id
            },
            'target': {
                'block_id': connection.target_block_id,
                'port_id': connection.target_port_id
            },
            'path': path,
            'style': {
                'color': color,
                'width': 2,
                'style': 'solid'
            },
            'state': {
                'is_selected': connection.is_selected,
                'has_error': connection.has_error,
                'data_flow_active': connection.data_flow_active
            }
        }
    
    def _calculate_connection_path(self, connection) -> List[Dict[str, float]]:
        """Calculate the path for a connection."""
        # This would calculate the bezier curve or line path for the connection
        # For now, return a simple straight line
        return [
            {'x': connection.source_position.x, 'y': connection.source_position.y},
            {'x': connection.target_position.x, 'y': connection.target_position.y}
        ]
    
    def _render_selection(self, flow_manager: FlowManager) -> Optional[Dict[str, Any]]:
        """Render selection rectangle."""
        selected_blocks = [block for block in flow_manager.blocks.values() if block.is_selected]
        
        if not selected_blocks:
            return None
        
        # Calculate bounding box
        min_x = min(block.position.x for block in selected_blocks)
        min_y = min(block.position.y for block in selected_blocks)
        max_x = max(block.position.x + block.position.width for block in selected_blocks)
        max_y = max(block.position.y + block.position.height for block in selected_blocks)
        
        return {
            'x': min_x - 5,
            'y': min_y - 5,
            'width': max_x - min_x + 10,
            'height': max_y - min_y + 10,
            'color': self.theme.selection_color
        }
    
    def _render_minimap(self, flow_manager: FlowManager) -> Dict[str, Any]:
        """Render minimap."""
        # Calculate minimap bounds
        all_blocks = list(flow_manager.blocks.values())
        if not all_blocks:
            return {
                'bounds': {'x': 0, 'y': 0, 'width': 200, 'height': 150},
                'viewport': {'x': 0, 'y': 0, 'width': 200, 'height': 150},
                'blocks': []
            }
        
        min_x = min(block.position.x for block in all_blocks)
        min_y = min(block.position.y for block in all_blocks)
        max_x = max(block.position.x + block.position.width for block in all_blocks)
        max_y = max(block.position.y + block.position.height for block in all_blocks)
        
        # Add padding
        padding = 50
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Calculate scale
        minimap_width = 200
        minimap_height = 150
        scale_x = minimap_width / (max_x - min_x)
        scale_y = minimap_height / (max_y - min_y)
        scale = min(scale_x, scale_y)
        
        # Render minimap blocks
        minimap_blocks = []
        for block in all_blocks:
            minimap_blocks.append({
                'x': (block.position.x - min_x) * scale,
                'y': (block.position.y - min_y) * scale,
                'width': block.position.width * scale,
                'height': block.position.height * scale,
                'color': self.theme.category_colors.get(block.category, self.theme.block_default_color)
            })
        
        # Calculate viewport rectangle in minimap
        viewport_rect = {
            'x': (self.viewport.x - min_x) * scale,
            'y': (self.viewport.y - min_y) * scale,
            'width': (self.viewport.width / self.viewport.zoom) * scale,
            'height': (self.viewport.height / self.viewport.zoom) * scale
        }
        
        return {
            'bounds': {
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            'viewport': viewport_rect,
            'blocks': minimap_blocks,
            'scale': scale
        }
    
    def _render_debug_info(self, flow_manager: FlowManager) -> Dict[str, Any]:
        """Render debug information."""
        return {
            'viewport': {
                'x': self.viewport.x,
                'y': self.viewport.y,
                'zoom': self.viewport.zoom
            },
            'performance': self.render_stats,
            'flow_stats': {
                'blocks': len(flow_manager.blocks),
                'connections': len(flow_manager.connections),
                'execution_order': len(flow_manager.get_execution_order())
            },
            'cache_stats': {
                'cached_blocks': len(self._cached_blocks),
                'cached_connections': len(self._cached_connections)
            }
        }
    
    def _is_block_in_viewport(self, block: AutomationBlock) -> bool:
        """Check if a block is visible in the current viewport."""
        viewport_left = self.viewport.x
        viewport_right = self.viewport.x + self.viewport.width / self.viewport.zoom
        viewport_top = self.viewport.y
        viewport_bottom = self.viewport.y + self.viewport.height / self.viewport.zoom
        
        block_left = block.position.x
        block_right = block.position.x + block.position.width
        block_top = block.position.y
        block_bottom = block.position.y + block.position.height
        
        return not (block_right < viewport_left or 
                   block_left > viewport_right or
                   block_bottom < viewport_top or
                   block_top > viewport_bottom)
    
    def set_viewport(self, x: float, y: float, zoom: float):
        """Set the viewport position and zoom."""
        self.viewport.x = x
        self.viewport.y = y
        self.viewport.zoom = max(0.1, min(5.0, zoom))  # Clamp zoom
        
        # Invalidate grid cache
        self._cached_grid = None
    
    def pan_viewport(self, delta_x: float, delta_y: float):
        """Pan the viewport by the given delta."""
        self.viewport.x += delta_x / self.viewport.zoom
        self.viewport.y += delta_y / self.viewport.zoom
        
        # Invalidate grid cache
        self._cached_grid = None
    
    def zoom_viewport(self, zoom_delta: float, center_x: float, center_y: float):
        """Zoom the viewport around a center point."""
        old_zoom = self.viewport.zoom
        new_zoom = max(0.1, min(5.0, old_zoom * zoom_delta))
        
        if new_zoom != old_zoom:
            # Adjust position to zoom around center point
            world_x = self.viewport.x + center_x / old_zoom
            world_y = self.viewport.y + center_y / old_zoom
            
            self.viewport.zoom = new_zoom
            self.viewport.x = world_x - center_x / new_zoom
            self.viewport.y = world_y - center_y / new_zoom
            
            # Invalidate grid cache
            self._cached_grid = None
    
    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = self.viewport.x + screen_x / self.viewport.zoom
        world_y = self.viewport.y + screen_y / self.viewport.zoom
        return world_x, world_y
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates to screen coordinates."""
        screen_x = (world_x - self.viewport.x) * self.viewport.zoom
        screen_y = (world_y - self.viewport.y) * self.viewport.zoom
        return screen_x, screen_y
    
    def clear_cache(self):
        """Clear all render caches."""
        self._cached_grid = None
        self._cached_blocks.clear()
        self._cached_connections.clear()
    
    def _get_current_time(self) -> float:
        """Get current time in milliseconds."""
        import time
        return time.time() * 1000
    
    def _update_render_stats(self, render_time: float):
        """Update rendering performance statistics."""
        self.render_stats['frame_count'] += 1
        self.render_stats['last_render_time'] = render_time
        
        # Calculate rolling average
        alpha = 0.1  # Smoothing factor
        if self.render_stats['average_render_time'] == 0:
            self.render_stats['average_render_time'] = render_time
        else:
            self.render_stats['average_render_time'] = (
                alpha * render_time + 
                (1 - alpha) * self.render_stats['average_render_time']
            )