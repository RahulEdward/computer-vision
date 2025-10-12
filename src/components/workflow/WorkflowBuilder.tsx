'use client';

import React, { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  Panel,
  NodeTypes,
  EdgeTypes
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/lib/store';
import { useCollaboration } from '@/lib/webrtc-collaboration';

// Custom Node Components
const TriggerNode = ({ data, selected }: any) => (
  <motion.div
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    className={`px-4 py-3 shadow-lg rounded-lg bg-gradient-to-r from-green-400 to-green-600 text-white min-w-[150px] ${
      selected ? 'ring-2 ring-blue-400' : ''
    }`}
  >
    <div className="flex items-center space-x-2">
      <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
      <div className="font-semibold">{data.label}</div>
    </div>
    <div className="text-xs mt-1 opacity-90">{data.description}</div>
    <div className="flex space-x-1 mt-2">
      <div className="w-2 h-2 bg-white rounded-full" />
      <div className="w-2 h-2 bg-white rounded-full" />
      <div className="w-2 h-2 bg-white rounded-full" />
    </div>
  </motion.div>
);

const ActionNode = ({ data, selected }: any) => (
  <motion.div
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    className={`px-4 py-3 shadow-lg rounded-lg bg-gradient-to-r from-blue-400 to-blue-600 text-white min-w-[150px] ${
      selected ? 'ring-2 ring-blue-400' : ''
    }`}
  >
    <div className="flex items-center space-x-2">
      <div className="w-3 h-3 bg-white rounded-full" />
      <div className="font-semibold">{data.label}</div>
    </div>
    <div className="text-xs mt-1 opacity-90">{data.description}</div>
    {data.status && (
      <div className={`text-xs mt-2 px-2 py-1 rounded ${
        data.status === 'running' ? 'bg-yellow-500' :
        data.status === 'success' ? 'bg-green-500' :
        data.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
      }`}>
        {data.status}
      </div>
    )}
  </motion.div>
);

const ConditionNode = ({ data, selected }: any) => (
  <motion.div
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    className={`px-4 py-3 shadow-lg rounded-lg bg-gradient-to-r from-purple-400 to-purple-600 text-white min-w-[150px] ${
      selected ? 'ring-2 ring-blue-400' : ''
    }`}
  >
    <div className="flex items-center space-x-2">
      <div className="w-3 h-3 bg-white rounded-full" />
      <div className="font-semibold">{data.label}</div>
    </div>
    <div className="text-xs mt-1 opacity-90">{data.condition}</div>
    <div className="flex justify-between mt-2">
      <span className="text-xs bg-green-500 px-2 py-1 rounded">True</span>
      <span className="text-xs bg-red-500 px-2 py-1 rounded">False</span>
    </div>
  </motion.div>
);

const nodeTypes: NodeTypes = {
  trigger: TriggerNode,
  action: ActionNode,
  condition: ConditionNode,
};

// Custom Edge Component
const AnimatedEdge = ({ id, sourceX, sourceY, targetX, targetY, style = {} }: any) => {
  const edgePath = `M${sourceX},${sourceY} C${sourceX + 50},${sourceY} ${targetX - 50},${targetY} ${targetX},${targetY}`;
  
  return (
    <g>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={edgePath}
        strokeWidth={2}
        stroke="#64748b"
        fill="none"
      />
      <circle r="3" fill="#3b82f6" className="animate-pulse">
        <animateMotion dur="2s" repeatCount="indefinite">
          <mpath href={`#${id}`} />
        </animateMotion>
      </circle>
    </g>
  );
};

const edgeTypes: EdgeTypes = {
  animated: AnimatedEdge,
};

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'trigger',
    position: { x: 100, y: 100 },
    data: { 
      label: 'HTTP Webhook',
      description: 'Receives HTTP requests'
    },
  },
  {
    id: '2',
    type: 'condition',
    position: { x: 300, y: 200 },
    data: { 
      label: 'Check Status',
      condition: 'status === "active"'
    },
  },
  {
    id: '3',
    type: 'action',
    position: { x: 500, y: 100 },
    data: { 
      label: 'Send Email',
      description: 'Send notification email',
      status: 'ready'
    },
  },
  {
    id: '4',
    type: 'action',
    position: { x: 500, y: 300 },
    data: { 
      label: 'Log Error',
      description: 'Log to error system',
      status: 'ready'
    },
  },
];

const initialEdges: Edge[] = [
  { 
    id: 'e1-2', 
    source: '1', 
    target: '2', 
    type: 'animated',
    animated: true 
  },
  { 
    id: 'e2-3', 
    source: '2', 
    target: '3', 
    type: 'animated',
    sourceHandle: 'true',
    animated: true 
  },
  { 
    id: 'e2-4', 
    source: '2', 
    target: '4', 
    type: 'animated',
    sourceHandle: 'false',
    animated: true 
  },
];

export default function WorkflowBuilder() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const { recordInteraction } = useDashboardStore();

  const onConnect = useCallback(
    (params: Connection) => {
      recordInteraction();
      setEdges((eds) => addEdge({ ...params, type: 'animated', animated: true }, eds));
    },
    [setEdges, recordInteraction]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    recordInteraction();
    setSelectedNode(node);
  }, [recordInteraction]);

  const addNode = (type: string) => {
    recordInteraction();
    const newNode: Node = {
      id: `${nodes.length + 1}`,
      type,
      position: { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
      data: {
        label: `New ${type}`,
        description: `Description for ${type}`,
        ...(type === 'condition' && { condition: 'condition === true' }),
        ...(type === 'action' && { status: 'ready' })
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const runWorkflow = async () => {
    setIsRunning(true);
    recordInteraction();
    
    // Simulate workflow execution
    for (const node of nodes) {
      if (node.type === 'action') {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === node.id
              ? { ...n, data: { ...n.data, status: 'running' } }
              : n
          )
        );
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setNodes((nds) =>
          nds.map((n) =>
            n.id === node.id
              ? { ...n, data: { ...n.data, status: 'success' } }
              : n
          )
        );
      }
    }
    
    setIsRunning(false);
  };

  const saveWorkflow = () => {
    recordInteraction();
    const workflow = { nodes, edges };
    localStorage.setItem('workflow', JSON.stringify(workflow));
    // Show success toast
  };

  const loadWorkflow = () => {
    recordInteraction();
    const saved = localStorage.getItem('workflow');
    if (saved) {
      const workflow = JSON.parse(saved);
      setNodes(workflow.nodes);
      setEdges(workflow.edges);
    }
  };

  return (
    <div className="h-full bg-slate-50 dark:bg-slate-900 rounded-lg overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        className="bg-slate-50 dark:bg-slate-900"
      >
        <Controls className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700" />
        <MiniMap 
          className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
          nodeColor={(node) => {
            switch (node.type) {
              case 'trigger': return '#10b981';
              case 'action': return '#3b82f6';
              case 'condition': return '#8b5cf6';
              default: return '#64748b';
            }
          }}
        />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        
        {/* Toolbar */}
        <Panel position="top-left" className="bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 p-4">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => addNode('trigger')}
              className="px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm"
            >
              + Trigger
            </button>
            <button
              onClick={() => addNode('action')}
              className="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
            >
              + Action
            </button>
            <button
              onClick={() => addNode('condition')}
              className="px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors text-sm"
            >
              + Condition
            </button>
            <div className="w-px bg-slate-300 dark:bg-slate-600 mx-2" />
            <button
              onClick={runWorkflow}
              disabled={isRunning}
              className="px-3 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors text-sm disabled:opacity-50"
            >
              {isRunning ? '⏳ Running...' : '▶️ Run'}
            </button>
            <button
              onClick={saveWorkflow}
              className="px-3 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
            >
              💾 Save
            </button>
            <button
              onClick={loadWorkflow}
              className="px-3 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
            >
              📁 Load
            </button>
          </div>
        </Panel>

        {/* Node Properties Panel */}
        {selectedNode && (
          <Panel position="top-right" className="bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 p-4 w-80">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-lg">Node Properties</h3>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-slate-400 hover:text-slate-600"
                >
                  ✕
                </button>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Label</label>
                <input
                  type="text"
                  value={selectedNode.data.label}
                  onChange={(e) => {
                    setNodes((nds) =>
                      nds.map((n) =>
                        n.id === selectedNode.id
                          ? { ...n, data: { ...n.data, label: e.target.value } }
                          : n
                      )
                    );
                    setSelectedNode({ ...selectedNode, data: { ...selectedNode.data, label: e.target.value } });
                  }}
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Description</label>
                <textarea
                  value={selectedNode.data.description}
                  onChange={(e) => {
                    setNodes((nds) =>
                      nds.map((n) =>
                        n.id === selectedNode.id
                          ? { ...n, data: { ...n.data, description: e.target.value } }
                          : n
                      )
                    );
                    setSelectedNode({ ...selectedNode, data: { ...selectedNode.data, description: e.target.value } });
                  }}
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 h-20 resize-none"
                />
              </div>

              {selectedNode.type === 'condition' && (
                <div>
                  <label className="block text-sm font-medium mb-1">Condition</label>
                  <input
                    type="text"
                    value={selectedNode.data.condition}
                    onChange={(e) => {
                      setNodes((nds) =>
                        nds.map((n) =>
                          n.id === selectedNode.id
                            ? { ...n, data: { ...n.data, condition: e.target.value } }
                            : n
                        )
                      );
                      setSelectedNode({ ...selectedNode, data: { ...selectedNode.data, condition: e.target.value } });
                    }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700"
                  />
                </div>
              )}

              <button
                onClick={() => {
                  setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
                  setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
                  setSelectedNode(null);
                }}
                className="w-full px-3 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                Delete Node
              </button>
            </div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}