"""
Flow Manager

Manages the flow connections between automation blocks and handles
the execution order and data flow in visual workflows.
"""

import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .automation_blocks import AutomationBlock, BlockConnection, BlockType


@dataclass
class FlowValidationResult:
    """Result of flow validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExecutionPath:
    """Represents an execution path through the workflow."""
    path_id: str
    blocks: List[str]  # Block IDs in execution order
    is_parallel: bool = False
    parent_path_id: Optional[str] = None


class FlowManager:
    """
    Manages the flow and connections between automation blocks
    in a visual workflow.
    """
    
    def __init__(self):
        """Initialize the flow manager."""
        self.logger = logging.getLogger(__name__)
        
        # Storage for blocks and connections
        self.blocks: Dict[str, AutomationBlock] = {}
        self.connections: Dict[str, BlockConnection] = {}
        
        # Flow analysis cache
        self._execution_order_cache: Optional[List[str]] = None
        self._parallel_groups_cache: Optional[List[List[str]]] = None
        self._validation_cache: Optional[FlowValidationResult] = None
        
        # Flow state
        self.is_modified = False
    
    def add_block(self, block: AutomationBlock) -> bool:
        """
        Add a block to the flow.
        
        Args:
            block: Block to add
            
        Returns:
            bool: True if block was added successfully
        """
        try:
            if block.block_id in self.blocks:
                self.logger.warning(f"Block {block.block_id} already exists")
                return False
            
            self.blocks[block.block_id] = block
            self._invalidate_cache()
            
            self.logger.info(f"Added block {block.block_id} ({block.block_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            return False
    
    def remove_block(self, block_id: str) -> bool:
        """
        Remove a block from the flow.
        
        Args:
            block_id: ID of block to remove
            
        Returns:
            bool: True if block was removed successfully
        """
        try:
            if block_id not in self.blocks:
                self.logger.warning(f"Block {block_id} not found")
                return False
            
            # Remove all connections to/from this block
            connections_to_remove = []
            for conn_id, connection in self.connections.items():
                if (connection.source_block_id == block_id or 
                    connection.target_block_id == block_id):
                    connections_to_remove.append(conn_id)
            
            for conn_id in connections_to_remove:
                self.remove_connection(conn_id)
            
            # Remove the block
            del self.blocks[block_id]
            self._invalidate_cache()
            
            self.logger.info(f"Removed block {block_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove block: {e}")
            return False
    
    def add_connection(self, connection: BlockConnection) -> bool:
        """
        Add a connection between blocks.
        
        Args:
            connection: Connection to add
            
        Returns:
            bool: True if connection was added successfully
        """
        try:
            # Validate connection
            if not self._validate_connection(connection):
                return False
            
            # Check for existing connection to the same input port
            existing_conn = self._find_connection_to_port(
                connection.target_block_id, 
                connection.target_port_id
            )
            
            if existing_conn:
                self.logger.warning(
                    f"Removing existing connection to port {connection.target_port_id}"
                )
                self.remove_connection(existing_conn.connection_id)
            
            self.connections[connection.connection_id] = connection
            self._invalidate_cache()
            
            self.logger.info(
                f"Added connection {connection.connection_id} "
                f"from {connection.source_block_id} to {connection.target_block_id}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add connection: {e}")
            return False
    
    def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a connection.
        
        Args:
            connection_id: ID of connection to remove
            
        Returns:
            bool: True if connection was removed successfully
        """
        try:
            if connection_id not in self.connections:
                self.logger.warning(f"Connection {connection_id} not found")
                return False
            
            del self.connections[connection_id]
            self._invalidate_cache()
            
            self.logger.info(f"Removed connection {connection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove connection: {e}")
            return False
    
    def create_connection(self, source_block_id: str, source_port_id: str,
                         target_block_id: str, target_port_id: str) -> Optional[BlockConnection]:
        """
        Create a new connection between blocks.
        
        Args:
            source_block_id: Source block ID
            source_port_id: Source port ID
            target_block_id: Target block ID
            target_port_id: Target port ID
            
        Returns:
            BlockConnection: Created connection or None if failed
        """
        try:
            # Get blocks and ports
            source_block = self.blocks.get(source_block_id)
            target_block = self.blocks.get(target_block_id)
            
            if not source_block or not target_block:
                self.logger.error("Source or target block not found")
                return None
            
            source_port = source_block.get_port_by_id(source_port_id)
            target_port = target_block.get_port_by_id(target_port_id)
            
            if not source_port or not target_port:
                self.logger.error("Source or target port not found")
                return None
            
            # Validate port compatibility
            if source_port.is_input or not target_port.is_input:
                self.logger.error("Invalid port types for connection")
                return None
            
            # Create connection
            connection = BlockConnection(
                connection_id=str(uuid.uuid4()),
                source_block_id=source_block_id,
                source_port_id=source_port_id,
                target_block_id=target_block_id,
                target_port_id=target_port_id,
                data_type=source_port.data_type
            )
            
            if self.add_connection(connection):
                return connection
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def get_execution_order(self) -> List[str]:
        """
        Get the execution order of blocks based on connections.
        
        Returns:
            List[str]: Block IDs in execution order
        """
        if self._execution_order_cache is not None:
            return self._execution_order_cache
        
        try:
            # Find start blocks
            start_blocks = [
                block_id for block_id, block in self.blocks.items()
                if block.block_type == BlockType.START
            ]
            
            if not start_blocks:
                # If no start block, find blocks with no incoming flow connections
                start_blocks = self._find_root_blocks()
            
            # Perform topological sort
            execution_order = []
            visited = set()
            
            for start_block in start_blocks:
                self._topological_sort(start_block, visited, execution_order)
            
            # Add any remaining unvisited blocks
            for block_id in self.blocks:
                if block_id not in visited:
                    execution_order.append(block_id)
            
            self._execution_order_cache = execution_order
            return execution_order
            
        except Exception as e:
            self.logger.error(f"Failed to get execution order: {e}")
            return list(self.blocks.keys())
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Identify groups of blocks that can be executed in parallel.
        
        Returns:
            List[List[str]]: Groups of block IDs that can run in parallel
        """
        if self._parallel_groups_cache is not None:
            return self._parallel_groups_cache
        
        try:
            execution_order = self.get_execution_order()
            parallel_groups = []
            
            # Build dependency graph
            dependencies = defaultdict(set)
            dependents = defaultdict(set)
            
            for connection in self.connections.values():
                if connection.data_type == "flow":
                    dependencies[connection.target_block_id].add(connection.source_block_id)
                    dependents[connection.source_block_id].add(connection.target_block_id)
            
            # Find parallel groups
            processed = set()
            
            for block_id in execution_order:
                if block_id in processed:
                    continue
                
                # Find all blocks that can run in parallel with this one
                parallel_group = [block_id]
                
                # Check remaining blocks
                for other_block_id in execution_order:
                    if (other_block_id != block_id and 
                        other_block_id not in processed and
                        self._can_run_in_parallel(block_id, other_block_id, dependencies, dependents)):
                        parallel_group.append(other_block_id)
                
                parallel_groups.append(parallel_group)
                processed.update(parallel_group)
            
            self._parallel_groups_cache = parallel_groups
            return parallel_groups
            
        except Exception as e:
            self.logger.error(f"Failed to get parallel groups: {e}")
            return [[block_id] for block_id in self.blocks.keys()]
    
    def validate_flow(self) -> FlowValidationResult:
        """
        Validate the current flow for errors and issues.
        
        Returns:
            FlowValidationResult: Validation result
        """
        if self._validation_cache is not None:
            return self._validation_cache
        
        errors = []
        warnings = []
        
        try:
            # Check for start and end blocks
            start_blocks = [b for b in self.blocks.values() if b.block_type == BlockType.START]
            end_blocks = [b for b in self.blocks.values() if b.block_type == BlockType.END]
            
            if not start_blocks:
                warnings.append("No start block found")
            elif len(start_blocks) > 1:
                warnings.append("Multiple start blocks found")
            
            if not end_blocks:
                warnings.append("No end block found")
            
            # Check for cycles
            if self._has_cycles():
                errors.append("Workflow contains cycles")
            
            # Check for disconnected blocks
            disconnected = self._find_disconnected_blocks()
            if disconnected:
                warnings.append(f"Disconnected blocks found: {', '.join(disconnected)}")
            
            # Validate individual blocks
            for block in self.blocks.values():
                block_connections = [
                    conn for conn in self.connections.values()
                    if conn.target_block_id == block.block_id
                ]
                
                block_errors = block.validate_connections(block_connections)
                errors.extend([f"Block {block.name}: {error}" for error in block_errors])
            
            # Check for unreachable blocks
            execution_order = self.get_execution_order()
            unreachable = set(self.blocks.keys()) - set(execution_order)
            if unreachable:
                warnings.append(f"Unreachable blocks: {', '.join(unreachable)}")
            
            result = FlowValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            
            self._validation_cache = result
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to validate flow: {e}")
            return FlowValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {e}"]
            )
    
    def get_block_dependencies(self, block_id: str) -> List[str]:
        """
        Get the dependencies of a block.
        
        Args:
            block_id: Block ID
            
        Returns:
            List[str]: List of block IDs this block depends on
        """
        dependencies = []
        
        for connection in self.connections.values():
            if (connection.target_block_id == block_id and 
                connection.data_type == "flow"):
                dependencies.append(connection.source_block_id)
        
        return dependencies
    
    def get_block_dependents(self, block_id: str) -> List[str]:
        """
        Get the dependents of a block.
        
        Args:
            block_id: Block ID
            
        Returns:
            List[str]: List of block IDs that depend on this block
        """
        dependents = []
        
        for connection in self.connections.values():
            if (connection.source_block_id == block_id and 
                connection.data_type == "flow"):
                dependents.append(connection.target_block_id)
        
        return dependents
    
    def _validate_connection(self, connection: BlockConnection) -> bool:
        """Validate a connection."""
        # Check if blocks exist
        if (connection.source_block_id not in self.blocks or
            connection.target_block_id not in self.blocks):
            self.logger.error("Source or target block not found")
            return False
        
        # Check if ports exist
        source_block = self.blocks[connection.source_block_id]
        target_block = self.blocks[connection.target_block_id]
        
        source_port = source_block.get_port_by_id(connection.source_port_id)
        target_port = target_block.get_port_by_id(connection.target_port_id)
        
        if not source_port or not target_port:
            self.logger.error("Source or target port not found")
            return False
        
        # Check port types
        if source_port.is_input or not target_port.is_input:
            self.logger.error("Invalid port types for connection")
            return False
        
        # Check data type compatibility
        if (source_port.data_type != target_port.data_type and
            source_port.data_type != "any" and target_port.data_type != "any"):
            self.logger.error("Incompatible data types")
            return False
        
        # Check for self-connection
        if connection.source_block_id == connection.target_block_id:
            self.logger.error("Self-connections not allowed")
            return False
        
        return True
    
    def _find_connection_to_port(self, block_id: str, port_id: str) -> Optional[BlockConnection]:
        """Find existing connection to a specific port."""
        for connection in self.connections.values():
            if (connection.target_block_id == block_id and
                connection.target_port_id == port_id):
                return connection
        return None
    
    def _find_root_blocks(self) -> List[str]:
        """Find blocks with no incoming flow connections."""
        root_blocks = []
        
        for block_id in self.blocks:
            has_incoming_flow = False
            
            for connection in self.connections.values():
                if (connection.target_block_id == block_id and
                    connection.data_type == "flow"):
                    has_incoming_flow = True
                    break
            
            if not has_incoming_flow:
                root_blocks.append(block_id)
        
        return root_blocks
    
    def _topological_sort(self, block_id: str, visited: Set[str], result: List[str]):
        """Perform topological sort starting from a block."""
        if block_id in visited:
            return
        
        visited.add(block_id)
        
        # Visit all dependent blocks first
        for connection in self.connections.values():
            if (connection.source_block_id == block_id and
                connection.data_type == "flow"):
                self._topological_sort(connection.target_block_id, visited, result)
        
        result.append(block_id)
    
    def _has_cycles(self) -> bool:
        """Check if the flow has cycles."""
        # Use DFS to detect cycles
        white = set(self.blocks.keys())  # Unvisited
        gray = set()   # Currently being processed
        black = set()  # Completely processed
        
        def dfs(block_id: str) -> bool:
            if block_id in gray:
                return True  # Back edge found - cycle detected
            
            if block_id in black:
                return False  # Already processed
            
            white.remove(block_id)
            gray.add(block_id)
            
            # Visit all dependent blocks
            for connection in self.connections.values():
                if (connection.source_block_id == block_id and
                    connection.data_type == "flow"):
                    if dfs(connection.target_block_id):
                        return True
            
            gray.remove(block_id)
            black.add(block_id)
            return False
        
        while white:
            if dfs(next(iter(white))):
                return True
        
        return False
    
    def _find_disconnected_blocks(self) -> List[str]:
        """Find blocks that are not connected to the main flow."""
        if not self.blocks:
            return []
        
        # Find all blocks reachable from start blocks
        start_blocks = [
            block_id for block_id, block in self.blocks.items()
            if block.block_type == BlockType.START
        ]
        
        if not start_blocks:
            start_blocks = self._find_root_blocks()
        
        if not start_blocks:
            # If no clear start, pick the first block
            start_blocks = [next(iter(self.blocks.keys()))]
        
        reachable = set()
        
        def dfs(block_id: str):
            if block_id in reachable:
                return
            
            reachable.add(block_id)
            
            # Follow all outgoing connections
            for connection in self.connections.values():
                if connection.source_block_id == block_id:
                    dfs(connection.target_block_id)
            
            # Follow all incoming connections (for bidirectional reachability)
            for connection in self.connections.values():
                if connection.target_block_id == block_id:
                    dfs(connection.source_block_id)
        
        for start_block in start_blocks:
            dfs(start_block)
        
        return [block_id for block_id in self.blocks if block_id not in reachable]
    
    def _can_run_in_parallel(self, block1_id: str, block2_id: str,
                           dependencies: Dict[str, Set[str]],
                           dependents: Dict[str, Set[str]]) -> bool:
        """Check if two blocks can run in parallel."""
        # Blocks can run in parallel if they don't depend on each other
        return (block1_id not in dependencies[block2_id] and
                block2_id not in dependencies[block1_id] and
                block1_id not in dependents[block2_id] and
                block2_id not in dependents[block1_id])
    
    def _invalidate_cache(self):
        """Invalidate all cached results."""
        self._execution_order_cache = None
        self._parallel_groups_cache = None
        self._validation_cache = None
        self.is_modified = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flow to dictionary representation."""
        return {
            'blocks': {
                block_id: block.to_dict()
                for block_id, block in self.blocks.items()
            },
            'connections': {
                conn_id: {
                    'connection_id': conn.connection_id,
                    'source_block_id': conn.source_block_id,
                    'source_port_id': conn.source_port_id,
                    'target_block_id': conn.target_block_id,
                    'target_port_id': conn.target_port_id,
                    'data_type': conn.data_type
                }
                for conn_id, conn in self.connections.items()
            }
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load flow from dictionary representation."""
        # Clear existing data
        self.blocks.clear()
        self.connections.clear()
        
        # Load blocks
        for block_data in data.get('blocks', {}).values():
            block = AutomationBlock.from_dict(block_data)
            self.blocks[block.block_id] = block
        
        # Load connections
        for conn_data in data.get('connections', {}).values():
            connection = BlockConnection(
                connection_id=conn_data['connection_id'],
                source_block_id=conn_data['source_block_id'],
                source_port_id=conn_data['source_port_id'],
                target_block_id=conn_data['target_block_id'],
                target_port_id=conn_data['target_port_id'],
                data_type=conn_data['data_type']
            )
            self.connections[connection.connection_id] = connection
        
        self._invalidate_cache()