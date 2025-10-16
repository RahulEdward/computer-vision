'use client';

import { useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { ArrowLeftIcon, TrashIcon } from '@heroicons/react/24/outline';

// Initial nodes with connections
const initialNodes: Node[] = [
  {
    id: '1',
    type: 'input',
    data: { label: '▶ Start' },
    position: { x: 250, y: 50 },
    style: { background: '#10b981', color: 'white', border: '2px solid #059669', padding: '10px', borderRadius: '8px' },
  },
  {
    id: '2',
    data: { label: 'Process Data' },
    position: { x: 250, y: 150 },
    style: { background: '#3b82f6', color: 'white', border: '1px solid #2563eb', padding: '10px', borderRadius: '8px' },
  },
  {
    id: '3',
    data: { label: 'API Call' },
    position: { x: 250, y: 250 },
    style: { background: '#8b5cf6', color: 'white', border: '1px solid #7c3aed', padding: '10px', borderRadius: '8px' },
  },
  {
    id: '4',
    type: 'output',
    data: { label: '■ End' },
    position: { x: 250, y: 350 },
    style: { background: '#ef4444', color: 'white', border: '2px solid #dc2626', padding: '10px', borderRadius: '8px' },
  },
];

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', animated: true, style: { stroke: '#10b981' } },
  { id: 'e2-3', source: '2', target: '3', animated: true, style: { stroke: '#3b82f6' } },
  { id: 'e3-4', source: '3', target: '4', animated: true, style: { stroke: '#8b5cf6' } },
];

function FlowCanvas() {
  const router = useRouter();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { fitView } = useReactFlow();

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)),
    [setEdges]
  );

  const handleFitView = () => {
    fitView({ padding: 0.2, duration: 400 });
  };

  const handleDelete = useCallback(() => {
    setNodes((nds) => nds.filter((node) => !node.selected));
    setEdges((eds) => eds.filter((edge) => !edge.selected));
  }, [setNodes, setEdges]);

  // Keyboard shortcut for delete
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        handleDelete();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleDelete]);

  const hasSelection = nodes.some((n) => n.selected) || edges.some((e) => e.selected);

  return (
    <div className="h-screen w-full flex flex-col bg-gray-50">
      {/* Simple Toolbar */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push('/dashboard')}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg transition-all"
          >
            <ArrowLeftIcon className="h-4 w-4" />
            <span>Dashboard</span>
          </button>
          
          <button
            onClick={handleFitView}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-all"
          >
            Fit View
          </button>

          {hasSelection && (
            <button
              onClick={handleDelete}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-all"
            >
              <TrashIcon className="h-4 w-4" />
              <span>Delete</span>
            </button>
          )}
        </div>

        <div className="text-sm text-gray-600">
          <span className="font-medium">{nodes.length}</span> nodes • 
          <span className="font-medium ml-2">{edges.length}</span> connections
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodesDraggable={true}
          nodesConnectable={true}
          elementsSelectable={true}
          selectNodesOnDrag={false}
          fitView
          attributionPosition="bottom-left"
        >
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>

      {/* Help Text */}
      <div className="bg-blue-50 border-t border-blue-200 px-4 py-2 text-sm text-blue-800">
        💡 <strong>Tip:</strong> Drag from one node's edge to another to connect them. Click to select, Delete key to remove.
      </div>
    </div>
  );
}

export default function SimpleWorkflowBuilder() {
  return (
    <ReactFlowProvider>
      <FlowCanvas />
    </ReactFlowProvider>
  );
}
