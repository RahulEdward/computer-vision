import React, { useCallback, useRef, useMemo } from 'react';
import { Node, Edge, XYPosition, Rect } from 'reactflow';

export interface SnapGrid {
  x: number;
  y: number;
}

export interface SnapPoint {
  x: number;
  y: number;
  type: 'grid' | 'node' | 'edge' | 'guide';
  nodeId?: string;
  edgeId?: string;
}

export interface CollisionBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DragState {
  isDragging: boolean;
  draggedNodes: string[];
  startPosition: XYPosition;
  currentPosition: XYPosition;
  offset: XYPosition;
  snapPoints: SnapPoint[];
  collisions: string[];
  previewPosition?: XYPosition;
}

export interface DragDropConfig {
  snapToGrid: boolean;
  snapToNodes: boolean;
  snapToEdges: boolean;
  snapThreshold: number;
  gridSize: SnapGrid;
  collisionDetection: boolean;
  preventOverlap: boolean;
  magneticSnapping: boolean;
  showSnapGuides: boolean;
  showCollisionHighlight: boolean;
  allowGroupDrag: boolean;
  constrainToParent: boolean;
}

interface DragDropSystemProps {
  nodes: Node[];
  edges: Edge[];
  config: DragDropConfig;
  onNodesChange: (nodes: Node[]) => void;
  onDragStart: (nodeIds: string[], position: XYPosition) => void;
  onDragMove: (nodeIds: string[], position: XYPosition, snapPoints: SnapPoint[]) => void;
  onDragEnd: (nodeIds: string[], position: XYPosition) => void;
  children: React.ReactNode;
}

export const DragDropSystem: React.FC<DragDropSystemProps> = ({
  nodes,
  edges,
  config,
  onNodesChange,
  onDragStart,
  onDragMove,
  onDragEnd,
  children
}) => {
  const dragStateRef = useRef<DragState>({
    isDragging: false,
    draggedNodes: [],
    startPosition: { x: 0, y: 0 },
    currentPosition: { x: 0, y: 0 },
    offset: { x: 0, y: 0 },
    snapPoints: [],
    collisions: []
  });

  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate node bounds
  const getNodeBounds = useCallback((node: Node): CollisionBounds => {
    const width = node.width || 200;
    const height = node.height || 100;
    return {
      x: node.position.x,
      y: node.position.y,
      width,
      height
    };
  }, []);

  // Check collision between two bounds
  const checkCollision = useCallback((bounds1: CollisionBounds, bounds2: CollisionBounds): boolean => {
    return !(
      bounds1.x + bounds1.width <= bounds2.x ||
      bounds2.x + bounds2.width <= bounds1.x ||
      bounds1.y + bounds1.height <= bounds2.y ||
      bounds2.y + bounds2.height <= bounds1.y
    );
  }, []);

  // Generate snap points
  const generateSnapPoints = useCallback((draggedNodeIds: string[], position: XYPosition): SnapPoint[] => {
    const snapPoints: SnapPoint[] = [];

    // Grid snap points
    if (config.snapToGrid) {
      const gridX = Math.round(position.x / config.gridSize.x) * config.gridSize.x;
      const gridY = Math.round(position.y / config.gridSize.y) * config.gridSize.y;
      snapPoints.push({
        x: gridX,
        y: gridY,
        type: 'grid'
      });
    }

    // Node snap points
    if (config.snapToNodes) {
      nodes.forEach(node => {
        if (draggedNodeIds.includes(node.id)) return;

        const bounds = getNodeBounds(node);
        
        // Snap to node edges
        snapPoints.push(
          { x: bounds.x, y: position.y, type: 'node', nodeId: node.id },
          { x: bounds.x + bounds.width, y: position.y, type: 'node', nodeId: node.id },
          { x: position.x, y: bounds.y, type: 'node', nodeId: node.id },
          { x: position.x, y: bounds.y + bounds.height, type: 'node', nodeId: node.id }
        );

        // Snap to node center
        snapPoints.push({
          x: bounds.x + bounds.width / 2,
          y: bounds.y + bounds.height / 2,
          type: 'node',
          nodeId: node.id
        });
      });
    }

    // Edge snap points
    if (config.snapToEdges) {
      edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode) {
          const sourceBounds = getNodeBounds(sourceNode);
          const targetBounds = getNodeBounds(targetNode);
          
          // Calculate edge midpoint
          const midX = (sourceBounds.x + sourceBounds.width / 2 + targetBounds.x + targetBounds.width / 2) / 2;
          const midY = (sourceBounds.y + sourceBounds.height / 2 + targetBounds.y + targetBounds.height / 2) / 2;
          
          snapPoints.push({
            x: midX,
            y: midY,
            type: 'edge',
            edgeId: edge.id
          });
        }
      });
    }

    return snapPoints;
  }, [nodes, edges, config, getNodeBounds]);

  // Find best snap point
  const findBestSnapPoint = useCallback((position: XYPosition, snapPoints: SnapPoint[]): SnapPoint | null => {
    let bestSnap: SnapPoint | null = null;
    let minDistance = config.snapThreshold;

    snapPoints.forEach(snap => {
      const distance = Math.sqrt(
        Math.pow(position.x - snap.x, 2) + Math.pow(position.y - snap.y, 2)
      );

      if (distance < minDistance) {
        minDistance = distance;
        bestSnap = snap;
      }
    });

    return bestSnap;
  }, [config.snapThreshold]);

  // Detect collisions
  const detectCollisions = useCallback((draggedNodeIds: string[], position: XYPosition): string[] => {
    if (!config.collisionDetection) return [];

    const collisions: string[] = [];
    
    draggedNodeIds.forEach(draggedId => {
      const draggedNode = nodes.find(n => n.id === draggedId);
      if (!draggedNode) return;

      const draggedBounds = {
        ...getNodeBounds(draggedNode),
        x: position.x,
        y: position.y
      };

      nodes.forEach(node => {
        if (draggedNodeIds.includes(node.id)) return;

        const nodeBounds = getNodeBounds(node);
        if (checkCollision(draggedBounds, nodeBounds)) {
          collisions.push(node.id);
        }
      });
    });

    return collisions;
  }, [nodes, config.collisionDetection, getNodeBounds, checkCollision]);

  // Handle drag start
  const handleDragStart = useCallback((nodeIds: string[], event: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const position = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };

    const primaryNode = nodes.find(n => nodeIds.includes(n.id));
    const offset = primaryNode ? {
      x: position.x - primaryNode.position.x,
      y: position.y - primaryNode.position.y
    } : { x: 0, y: 0 };

    dragStateRef.current = {
      isDragging: true,
      draggedNodes: nodeIds,
      startPosition: position,
      currentPosition: position,
      offset,
      snapPoints: [],
      collisions: []
    };

    onDragStart(nodeIds, position);
  }, [nodes, onDragStart]);

  // Handle drag move
  const handleDragMove = useCallback((event: React.MouseEvent) => {
    if (!dragStateRef.current.isDragging) return;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const currentPosition = {
      x: event.clientX - rect.left - dragStateRef.current.offset.x,
      y: event.clientY - rect.top - dragStateRef.current.offset.y
    };

    // Generate snap points
    const snapPoints = generateSnapPoints(dragStateRef.current.draggedNodes, currentPosition);
    
    // Find best snap
    const bestSnap = findBestSnapPoint(currentPosition, snapPoints);
    const snappedPosition = bestSnap ? { x: bestSnap.x, y: bestSnap.y } : currentPosition;

    // Detect collisions
    const collisions = detectCollisions(dragStateRef.current.draggedNodes, snappedPosition);

    // Prevent overlap if configured
    let finalPosition = snappedPosition;
    if (config.preventOverlap && collisions.length > 0) {
      finalPosition = dragStateRef.current.currentPosition; // Keep previous position
    }

    dragStateRef.current.currentPosition = finalPosition;
    dragStateRef.current.snapPoints = snapPoints;
    dragStateRef.current.collisions = collisions;
    dragStateRef.current.previewPosition = bestSnap ? snappedPosition : undefined;

    onDragMove(dragStateRef.current.draggedNodes, finalPosition, snapPoints);
  }, [generateSnapPoints, findBestSnapPoint, detectCollisions, config.preventOverlap, onDragMove]);

  // Handle drag end
  const handleDragEnd = useCallback(() => {
    if (!dragStateRef.current.isDragging) return;

    const { draggedNodes, currentPosition } = dragStateRef.current;

    // Update node positions
    const updatedNodes = nodes.map(node => {
      if (draggedNodes.includes(node.id)) {
        return {
          ...node,
          position: currentPosition
        };
      }
      return node;
    });

    onNodesChange(updatedNodes);
    onDragEnd(draggedNodes, currentPosition);

    // Reset drag state
    dragStateRef.current = {
      isDragging: false,
      draggedNodes: [],
      startPosition: { x: 0, y: 0 },
      currentPosition: { x: 0, y: 0 },
      offset: { x: 0, y: 0 },
      snapPoints: [],
      collisions: []
    };
  }, [nodes, onNodesChange, onDragEnd]);

  // Render snap guides
  const renderSnapGuides = useCallback(() => {
    if (!config.showSnapGuides || !dragStateRef.current.isDragging) return null;

    return (
      <div className="absolute inset-0 pointer-events-none z-10">
        {dragStateRef.current.snapPoints.map((snap, index) => (
          <div
            key={index}
            className={`absolute w-1 h-1 rounded-full ${
              snap.type === 'grid' ? 'bg-blue-400' :
              snap.type === 'node' ? 'bg-green-400' :
              snap.type === 'edge' ? 'bg-yellow-400' :
              'bg-purple-400'
            }`}
            style={{
              left: snap.x - 2,
              top: snap.y - 2,
              boxShadow: '0 0 4px rgba(0,0,0,0.5)'
            }}
          />
        ))}
        
        {dragStateRef.current.previewPosition && (
          <div
            className="absolute border-2 border-dashed border-blue-400 rounded opacity-50"
            style={{
              left: dragStateRef.current.previewPosition.x,
              top: dragStateRef.current.previewPosition.y,
              width: 200, // Default node width
              height: 100 // Default node height
            }}
          />
        )}
      </div>
    );
  }, [config.showSnapGuides]);

  // Render collision highlights
  const renderCollisionHighlights = useCallback(() => {
    if (!config.showCollisionHighlight || !dragStateRef.current.isDragging) return null;

    return (
      <div className="absolute inset-0 pointer-events-none z-10">
        {dragStateRef.current.collisions.map(nodeId => {
          const node = nodes.find(n => n.id === nodeId);
          if (!node) return null;

          const bounds = getNodeBounds(node);
          return (
            <div
              key={nodeId}
              className="absolute border-2 border-red-500 rounded bg-red-500 bg-opacity-20"
              style={{
                left: bounds.x,
                top: bounds.y,
                width: bounds.width,
                height: bounds.height
              }}
            />
          );
        })}
      </div>
    );
  }, [config.showCollisionHighlight, nodes, getNodeBounds]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full"
      onMouseMove={handleDragMove}
      onMouseUp={handleDragEnd}
      onMouseLeave={handleDragEnd}
    >
      {children}
      {renderSnapGuides()}
      {renderCollisionHighlights()}
    </div>
  );
};

// Hook for drag and drop functionality
export const useDragDrop = (config: DragDropConfig) => {
  const [dragState, setDragState] = React.useState<DragState>({
    isDragging: false,
    draggedNodes: [],
    startPosition: { x: 0, y: 0 },
    currentPosition: { x: 0, y: 0 },
    offset: { x: 0, y: 0 },
    snapPoints: [],
    collisions: []
  });

  const startDrag = useCallback((nodeIds: string[], position: XYPosition, offset: XYPosition = { x: 0, y: 0 }) => {
    setDragState({
      isDragging: true,
      draggedNodes: nodeIds,
      startPosition: position,
      currentPosition: position,
      offset,
      snapPoints: [],
      collisions: []
    });
  }, []);

  const updateDrag = useCallback((position: XYPosition, snapPoints: SnapPoint[] = [], collisions: string[] = []) => {
    setDragState(prev => ({
      ...prev,
      currentPosition: position,
      snapPoints,
      collisions
    }));
  }, []);

  const endDrag = useCallback(() => {
    setDragState({
      isDragging: false,
      draggedNodes: [],
      startPosition: { x: 0, y: 0 },
      currentPosition: { x: 0, y: 0 },
      offset: { x: 0, y: 0 },
      snapPoints: [],
      collisions: []
    });
  }, []);

  return {
    dragState,
    startDrag,
    updateDrag,
    endDrag
  };
};

// Default configuration
export const defaultDragDropConfig: DragDropConfig = {
  snapToGrid: true,
  snapToNodes: true,
  snapToEdges: false,
  snapThreshold: 20,
  gridSize: { x: 20, y: 20 },
  collisionDetection: true,
  preventOverlap: false,
  magneticSnapping: true,
  showSnapGuides: true,
  showCollisionHighlight: true,
  allowGroupDrag: true,
  constrainToParent: false
};