'use client';

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  NodeTypes,
  EdgeTypes,
  ReactFlowInstance,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  SelectionMode,
  ConnectionMode,
  Viewport,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Import custom components
import { EnterpriseNode } from './nodes/EnterpriseNode';
import { GroupNode } from './nodes/GroupNode';
import { ChildNode } from './nodes/ChildNode';
import { CustomEdge } from './edges/CustomEdge';
import { NodePalette } from './panels/NodePalette';
import { PropertiesPanel } from './panels/PropertiesPanel';
import { PerformanceMonitor } from './panels/PerformanceMonitor';
import { CollaborationPanel } from './panels/CollaborationPanel';
import { WorkflowToolbar } from './panels/WorkflowToolbar';
import { ContextMenu, createNodeContextMenu, createCanvasContextMenu, ContextMenuPosition } from './panels/ContextMenu';
import { MiniMap, defaultMiniMapSettings, MiniMapSettings } from './panels/MiniMap';
import { DragDropSystem, defaultDragDropConfig, DragDropConfig } from './utils/DragDropSystem';
import { PerformanceOptimizer, defaultOptimizationConfig, OptimizationConfig, PerformanceMetrics } from './utils/PerformanceOptimizer';
import { ComplexWorkflowTest } from './test/ComplexWorkflowTest';
import { WorkflowValidator, ValidationResult } from './validation/WorkflowValidator';

// Types and interfaces
export interface HierarchicalNode extends Node {
  data: {
    label: string;
    type: string;
    parentId?: string;
    childIds?: string[];
    isCollapsed?: boolean;
    level: number;
    groupId?: string;
    metadata?: Record<string, any>;
    performance?: {
      executionTime?: number;
      memoryUsage?: number;
      status: 'idle' | 'running' | 'completed' | 'error';
    };
  };
}

export interface WorkflowGroup {
  id: string;
  name: string;
  description?: string;
  nodeIds: string[];
  color: string;
  isCollapsed: boolean;
  position: { x: number; y: number };
  size: { width: number; height: number };
}

export interface CanvasState {
  nodes: HierarchicalNode[];
  edges: Edge[];
  groups: WorkflowGroup[];
  selectedNodes: string[];
  selectedEdges: string[];
  clipboard: {
    nodes: HierarchicalNode[];
    edges: Edge[];
  };
  history: {
    past: Array<{ nodes: HierarchicalNode[]; edges: Edge[] }>;
    future: Array<{ nodes: HierarchicalNode[]; edges: Edge[] }>;
  };
  performance: {
    nodeCount: number;
    edgeCount: number;
    renderTime: number;
    memoryUsage: number;
  };
}

const nodeTypes: NodeTypes = {
  enterprise: EnterpriseNode,
  group: GroupNode,
  child: ChildNode,
};

const edgeTypes: EdgeTypes = {
  custom: CustomEdge,
};

const initialNodes: HierarchicalNode[] = [
  {
    id: 'start-1',
    type: 'enterprise',
    position: { x: 100, y: 100 },
    data: {
      label: 'Start Trigger',
      type: 'trigger',
      level: 0,
      performance: { status: 'idle' }
    },
  },
  {
    id: 'group-1',
    type: 'group',
    position: { x: 300, y: 50 },
    data: {
      label: 'Data Processing Group',
      type: 'group',
      level: 0,
      childIds: ['process-1', 'transform-1'],
      isCollapsed: false,
      performance: { status: 'idle' }
    },
    style: {
      width: 400,
      height: 300,
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      border: '2px dashed #3b82f6',
    },
  },
  {
    id: 'process-1',
    type: 'child',
    position: { x: 320, y: 100 },
    data: {
      label: 'Process Data',
      type: 'action',
      parentId: 'group-1',
      level: 1,
      performance: { status: 'idle' }
    },
    parentNode: 'group-1',
    extent: 'parent',
  },
  {
    id: 'transform-1',
    type: 'child',
    position: { x: 320, y: 200 },
    data: {
      label: 'Transform Data',
      type: 'transform',
      parentId: 'group-1',
      level: 1,
      performance: { status: 'idle' }
    },
    parentNode: 'group-1',
    extent: 'parent',
  },
];

const initialEdges: Edge[] = [
  {
    id: 'e1-2',
    source: 'start-1',
    target: 'process-1',
    type: 'custom',
    animated: true,
  },
  {
    id: 'e2-3',
    source: 'process-1',
    target: 'transform-1',
    type: 'custom',
  },
];

// Unique ID generator to prevent duplicate keys
let nodeCounter = 0;
const generateUniqueId = (prefix: string): string => {
  nodeCounter++;
  return `${prefix}-${Date.now()}-${nodeCounter}`;
};

export const EnterpriseWorkflowCanvas: React.FC = () => {
  // Handle ResizeObserver errors
  useEffect(() => {
    const handleResizeObserverError = (e: ErrorEvent) => {
      if (e.message === 'ResizeObserver loop completed with undelivered notifications.') {
        e.stopImmediatePropagation();
      }
    };
    
    window.addEventListener('error', handleResizeObserverError);
    return () => window.removeEventListener('error', handleResizeObserverError);
  }, []);

  // State management
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, zoom: 1 });
  
  // UI state
  const [selectedNode, setSelectedNode] = useState<HierarchicalNode | null>(null);
  const [showPropertiesPanel, setShowPropertiesPanel] = useState(false);
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(true);
  const [showCollaboration, setShowCollaboration] = useState(false);
  const [showNodePalette, setShowNodePalette] = useState(true);
  const [showTestSuite, setShowTestSuite] = useState(true);
  const [showValidator, setShowValidator] = useState(false);
  const [contextMenu, setContextMenu] = useState<ContextMenuPosition | null>(null);

  // Configuration states
  const [dragDropConfig, setDragDropConfig] = useState<DragDropConfig>(defaultDragDropConfig);
  const [optimizationConfig, setOptimizationConfig] = useState<OptimizationConfig>(defaultOptimizationConfig);
  const [miniMapSettings, setMiniMapSettings] = useState<MiniMapSettings>(defaultMiniMapSettings);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    memoryUsage: 0,
    fps: 0,
    lastUpdate: Date.now()
  });

  // Performance tracking
  const [canvasState, setCanvasState] = useState<CanvasState>({
    nodes: initialNodes,
    edges: initialEdges,
    groups: [],
    selectedNodes: [],
    selectedEdges: [],
    clipboard: { nodes: [], edges: [] },
    history: { past: [], future: [] },
    performance: {
      nodeCount: initialNodes.length,
      edgeCount: initialEdges.length,
      renderTime: 0,
      memoryUsage: 0,
    },
  });

  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  const performanceRef = useRef<{
    startTime: number;
    frameCount: number;
  }>({ startTime: Date.now(), frameCount: 0 });

  // Performance monitoring
  useEffect(() => {
    const updatePerformance = () => {
      performanceRef.current.frameCount++;
      const now = Date.now();
      const elapsed = now - performanceRef.current.startTime;
      
      if (elapsed >= 1000) {
        const fps = (performanceRef.current.frameCount * 1000) / elapsed;
        setCanvasState(prev => ({
          ...prev,
          performance: {
            ...prev.performance,
            renderTime: 1000 / fps,
            nodeCount: nodes.length,
            edgeCount: edges.length,
          },
        }));
        
        performanceRef.current.startTime = now;
        performanceRef.current.frameCount = 0;
      }
    };

    const interval = setInterval(updatePerformance, 100);
    return () => clearInterval(interval);
  }, [nodes.length, edges.length]);

  // Connection handling with validation
  const onConnect: OnConnect = useCallback(
    (params: Connection) => {
      // Validate connection
      if (!params.source || !params.target) return;
      
      // Prevent self-connections
      if (params.source === params.target) return;
      
      // Check for circular dependencies
      const wouldCreateCycle = checkForCycle(params.source, params.target, edges);
      if (wouldCreateCycle) {
        console.warn('Connection would create a cycle');
        return;
      }

      const newEdge = {
        ...params,
        id: `e${params.source}-${params.target}`,
        type: 'custom',
        animated: true,
      };

      setEdges((eds) => addEdge(newEdge, eds));
    },
    [edges, setEdges]
  );

  // Cycle detection
  const checkForCycle = (source: string, target: string, currentEdges: Edge[]): boolean => {
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (nodeId: string): boolean => {
      if (recursionStack.has(nodeId)) return true;
      if (visited.has(nodeId)) return false;

      visited.add(nodeId);
      recursionStack.add(nodeId);

      const outgoingEdges = currentEdges.filter(edge => edge.source === nodeId);
      for (const edge of outgoingEdges) {
        if (edge.target === source && nodeId === target) return true;
        if (hasCycle(edge.target)) return true;
      }

      recursionStack.delete(nodeId);
      return false;
    };

    return hasCycle(target);
  };

  // Performance optimization callbacks
  const handleOptimizedNodesChange = useCallback((optimizedNodes: Node[], metrics: PerformanceMetrics) => {
    setPerformanceMetrics(metrics);
  }, []);

  const handleOptimizedEdgesChange = useCallback((optimizedEdges: Edge[], metrics: PerformanceMetrics) => {
    setPerformanceMetrics(prev => ({ ...prev, ...metrics }));
  }, []);

  // Drag and drop callbacks
  const handleDragStart = useCallback((nodeIds: string[], position: { x: number; y: number }) => {
    console.log('Drag started:', nodeIds, position);
  }, []);

  const handleDragMove = useCallback((nodeIds: string[], position: { x: number; y: number }, snapPoints: any[]) => {
    console.log('Drag move:', nodeIds, position, snapPoints);
  }, []);

  const handleDragEnd = useCallback((nodeIds: string[], position: { x: number; y: number }) => {
    console.log('Drag ended:', nodeIds, position);
  }, []);

  // Viewport change handler
  const handleViewportChange = useCallback((newViewport: Partial<Viewport>) => {
    setViewport(prev => ({ ...prev, ...newViewport }));
  }, []);

  // Node selection handling
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node as HierarchicalNode);
    setShowPropertiesPanel(true);
  }, []);

  // Context menu handling
  const onNodeContextMenu = useCallback((event: React.MouseEvent, node: Node) => {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY });
  }, []);

  const onPaneContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY });
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Test workflow loading
  const loadTestWorkflow = useCallback((testNodes: Node[], testEdges: Edge[]) => {
    console.log(`Loading test workflow with ${testNodes.length} nodes and ${testEdges.length} edges`);
    
    // Update the canvas state
    setNodes(testNodes);
    setEdges(testEdges);
    
    // Update canvas state for tracking
    setCanvasState(prev => ({
      ...prev,
      nodes: testNodes,
      edges: testEdges,
      performance: {
        ...prev.performance,
        nodeCount: testNodes.length,
        edgeCount: testEdges.length,
      }
    }));

    // Fit view after loading
    setTimeout(() => {
      if (reactFlowInstance) {
        reactFlowInstance.fitView({ padding: 0.1 });
      }
    }, 100);
  }, [setNodes, setEdges, reactFlowInstance]);

  // Node operations
  const addNode = useCallback((type: string, position?: { x: number; y: number }) => {
    const newNode: HierarchicalNode = {
      id: generateUniqueId(type),
      type: 'enterprise',
      position: position || { x: Math.random() * 400, y: Math.random() * 400 },
      data: {
        label: `New ${type}`,
        type,
        level: 0,
        performance: { status: 'idle' }
      },
    };

    setNodes((nds) => [...nds, newNode]);
  }, [setNodes]);

  const deleteNode = useCallback((nodeId: string) => {
    setNodes((nds) => nds.filter(node => node.id !== nodeId));
    setEdges((eds) => eds.filter(edge => edge.source !== nodeId && edge.target !== nodeId));
  }, [setNodes, setEdges]);

  const duplicateNode = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const newNode: HierarchicalNode = {
      ...node,
      id: generateUniqueId(`${node.data.type}-copy`),
      position: {
        x: node.position.x + 50,
        y: node.position.y + 50,
      },
      data: {
        ...node.data,
        label: `${node.data.label} (Copy)`,
      },
    };

    setNodes((nds) => [...nds, newNode]);
  }, [nodes, setNodes]);

  // Group operations
  const createGroup = useCallback((selectedNodeIds: string[]) => {
    if (selectedNodeIds.length < 2) return;

    const selectedNodes = nodes.filter(node => selectedNodeIds.includes(node.id));
    const bounds = calculateBounds(selectedNodes);
    
    const groupId = generateUniqueId('group');
    const groupNode: HierarchicalNode = {
      id: groupId,
      type: 'group',
      position: { x: bounds.x - 20, y: bounds.y - 20 },
      data: {
        label: 'New Group',
        type: 'group',
        level: 0,
        childIds: selectedNodeIds,
        isCollapsed: false,
        performance: { status: 'idle' }
      },
      style: {
        width: bounds.width + 40,
        height: bounds.height + 40,
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        border: '2px dashed #3b82f6',
      },
    };

    // Update child nodes
    const updatedNodes = nodes.map(node => {
      if (selectedNodeIds.includes(node.id)) {
        return {
          ...node,
          parentNode: groupId,
          extent: 'parent' as const,
          data: {
            ...node.data,
            parentId: groupId,
            level: 1,
          },
        };
      }
      return node;
    });

    setNodes([...updatedNodes, groupNode]);
  }, [nodes, setNodes]);

  const calculateBounds = (nodes: HierarchicalNode[]) => {
    const xs = nodes.map(node => node.position.x);
    const ys = nodes.map(node => node.position.y);
    
    return {
      x: Math.min(...xs),
      y: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs) + 200,
      height: Math.max(...ys) - Math.min(...ys) + 100,
    };
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 'z':
            event.preventDefault();
            // Undo functionality
            break;
          case 'y':
            event.preventDefault();
            // Redo functionality
            break;
          case 'c':
            event.preventDefault();
            // Copy functionality
            break;
          case 'v':
            event.preventDefault();
            // Paste functionality
            break;
          case 'a':
            event.preventDefault();
            // Select all functionality
            break;
        }
      }
      
      if (event.key === 'Delete') {
        // Delete selected nodes
        canvasState.selectedNodes.forEach(nodeId => deleteNode(nodeId));
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [canvasState.selectedNodes, deleteNode]);

  // Memoized components for performance
  const memoizedBackground = useMemo(() => (
    <Background 
      color="#1f2937" 
      gap={20} 
      size={1}
      variant={BackgroundVariant.Dots} 
    />
  ), []);

  const memoizedControls = useMemo(() => (
    <Controls 
      showZoom={true}
      showFitView={true}
      showInteractive={true}
      position="bottom-left"
    />
  ), []);

  // MiniMap temporarily disabled due to prop requirements
  const memoizedMiniMap = null;

  return (
    <div className="h-screen w-full bg-gray-900 relative">
      <ReactFlowProvider>
        <div className="relative h-full w-full">
          {/* Performance Optimizer temporarily disabled */}

          {/* Drag Drop System temporarily disabled */}

          <ReactFlow
            ref={canvasRef}
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onNodeClick={onNodeClick}
            onNodeContextMenu={onNodeContextMenu}
            onPaneContextMenu={onPaneContextMenu}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            minZoom={0.1}
            maxZoom={2}
            attributionPosition="bottom-left"
            className="bg-gray-900"
            fitView
          >
            <Background color="#374151" gap={20} />
            <Controls />
          </ReactFlow>

          {/* MiniMap temporarily disabled */}
        </div>

        {/* Toolbar temporarily disabled */}

        {/* Performance Monitor temporarily disabled */}

        {/* Node Palette */}
        {showNodePalette && (
          <NodePalette onAddNode={addNode} />
        )}

        {/* Properties Panel temporarily disabled */}

        {/* Collaboration Panel temporarily disabled */}

        {/* Context Menu temporarily disabled */}

        {/* Test Suite */}
        {showTestSuite && (
          <ComplexWorkflowTest onLoadWorkflow={loadTestWorkflow} />
        )}

        {/* Validation Toggle Button */}
        <button
          onClick={() => setShowValidator(!showValidator)}
          className={`fixed top-4 right-4 z-50 px-4 py-2 rounded-lg shadow-lg transition-all duration-200 ${
            showValidator 
              ? 'bg-blue-600 text-white' 
              : 'bg-white text-gray-700 hover:bg-gray-50'
          }`}
          title="Toggle Workflow Validation"
        >
          {showValidator ? '✓ Validation On' : '⚠ Validate Workflow'}
        </button>

        {/* Workflow Validator temporarily disabled */}
      </ReactFlowProvider>
    </div>
  );
};