'use client';

import { useCallback, useState } from 'react';
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
import { 
  ArrowLeftIcon, 
  TrashIcon, 
  PlayIcon, 
  PlusIcon,
  PencilIcon,
  XMarkIcon,
  CheckIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

// Node types available
const nodeTypes = [
  { type: 'trigger', label: '⚡ Trigger', color: '#10b981', icon: '⚡' },
  { type: 'action', label: '⚙️ Action', color: '#3b82f6', icon: '⚙️' },
  { type: 'condition', label: '❓ Condition', color: '#f59e0b', icon: '❓' },
  { type: 'api', label: '🌐 API Call', color: '#8b5cf6', icon: '🌐' },
  { type: 'email', label: '📧 Email', color: '#ec4899', icon: '📧' },
  { type: 'database', label: '💾 Database', color: '#06b6d4', icon: '💾' },
  { type: 'transform', label: '🔄 Transform', color: '#84cc16', icon: '🔄' },
  { type: 'delay', label: '⏱️ Delay', color: '#f97316', icon: '⏱️' },
];

// Initial demo nodes
const initialNodes: Node[] = [
  {
    id: '1',
    type: 'input',
    data: { label: '⚡ Start Trigger', nodeType: 'trigger' },
    position: { x: 250, y: 50 },
    style: { background: '#10b981', color: 'white', border: '2px solid #059669', padding: '12px 20px', borderRadius: '8px', fontWeight: '500' },
  },
  {
    id: '2',
    data: { label: '⚙️ Process Data', nodeType: 'action' },
    position: { x: 250, y: 150 },
    style: { background: '#3b82f6', color: 'white', border: '1px solid #2563eb', padding: '12px 20px', borderRadius: '8px', fontWeight: '500' },
  },
  {
    id: '3',
    data: { label: '🌐 API Call', nodeType: 'api' },
    position: { x: 250, y: 250 },
    style: { background: '#8b5cf6', color: 'white', border: '1px solid #7c3aed', padding: '12px 20px', borderRadius: '8px', fontWeight: '500' },
  },
  {
    id: '4',
    type: 'output',
    data: { label: '✅ Complete', nodeType: 'output' },
    position: { x: 250, y: 350 },
    style: { background: '#ef4444', color: 'white', border: '2px solid #dc2626', padding: '12px 20px', borderRadius: '8px', fontWeight: '500' },
  },
];

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', animated: true, style: { stroke: '#10b981', strokeWidth: 2 } },
  { id: 'e2-3', source: '2', target: '3', animated: true, style: { stroke: '#3b82f6', strokeWidth: 2 } },
  { id: 'e3-4', source: '3', target: '4', animated: true, style: { stroke: '#8b5cf6', strokeWidth: 2 } },
];

function WorkflowBuilderContent() {
  const router = useRouter();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [editLabel, setEditLabel] = useState('');
  const [nodeConfig, setNodeConfig] = useState<any>({});
  const [isRunning, setIsRunning] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const { project } = useReactFlow();

  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge = {
        ...params,
        animated: true,
        style: { stroke: '#8b5cf6', strokeWidth: 2 },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const addNode = (nodeType: typeof nodeTypes[0]) => {
    const newId = (nodes.length + 1).toString();
    const newNode: Node = {
      id: newId,
      data: { label: nodeType.label, nodeType: nodeType.type },
      position: { x: 250, y: nodes.length * 120 + 50 },
      style: { 
        background: nodeType.color, 
        color: 'white', 
        border: '1px solid rgba(255,255,255,0.3)', 
        padding: '12px 20px', 
        borderRadius: '8px',
        fontWeight: '500'
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setShowAddMenu(false);
  };

  const deleteNode = () => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
      setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
      setSelectedNode(null);
    }
  };

  const editNode = () => {
    if (selectedNode) {
      setEditLabel(selectedNode.data.label);
      setShowEditModal(true);
    }
  };

  const configureNode = () => {
    if (selectedNode) {
      setNodeConfig(selectedNode.data.config || {});
      setShowConfigModal(true);
    }
  };

  const saveConfig = () => {
    if (selectedNode) {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === selectedNode.id
            ? { ...node, data: { ...node.data, config: nodeConfig } }
            : node
        )
      );
      setShowConfigModal(false);
      setSelectedNode(null);
    }
  };

  const saveEdit = () => {
    if (selectedNode && editLabel.trim()) {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === selectedNode.id
            ? { ...node, data: { ...node.data, label: editLabel } }
            : node
        )
      );
      setShowEditModal(false);
      setSelectedNode(null);
    }
  };

  const runWorkflow = async () => {
    setIsRunning(true);
    setExecutionLog(['🚀 Starting workflow execution...']);
    
    // Simulate workflow execution
    for (let i = 0; i < nodes.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setExecutionLog(prev => [...prev, `✅ Executed: ${nodes[i].data.label}`]);
      
      // Highlight current node
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodes[i].id
            ? { ...node, style: { ...node.style, border: '3px solid #fbbf24', boxShadow: '0 0 20px #fbbf24' } }
            : node
        )
      );
      
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Reset highlight
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodes[i].id
            ? { ...node, style: { ...node.style, border: '1px solid rgba(255,255,255,0.3)', boxShadow: 'none' } }
            : node
        )
      );
    }
    
    setExecutionLog(prev => [...prev, '🎉 Workflow completed successfully!']);
    setIsRunning(false);
  };

  const clearWorkflow = () => {
    if (confirm('Are you sure you want to clear the entire workflow?')) {
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);
      setExecutionLog([]);
    }
  };

  const saveWorkflow = () => {
    const workflow = {
      nodes,
      edges,
      name: 'My Workflow',
      createdAt: new Date().toISOString(),
    };
    console.log('Saving workflow:', workflow);
    alert('Workflow saved successfully! (Check console for details)');
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/40 backdrop-blur-xl border-b border-white/10 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => router.push('/dashboard')}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            >
              <ArrowLeftIcon className="h-5 w-5" />
            </button>
            <h1 className="text-xl font-bold text-white">Workflow Builder</h1>
            <span className="text-sm text-gray-400">
              {nodes.length} nodes • {edges.length} connections
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowAddMenu(!showAddMenu)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              <PlusIcon className="h-5 w-5" />
              <span>Add Node</span>
            </button>
            
            {selectedNode && (
              <>
                <button
                  onClick={editNode}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <PencilIcon className="h-5 w-5" />
                  <span>Edit Name</span>
                </button>
                <button
                  onClick={configureNode}
                  className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  <Cog6ToothIcon className="h-5 w-5" />
                  <span>Configure</span>
                </button>
                <button
                  onClick={deleteNode}
                  className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  <TrashIcon className="h-5 w-5" />
                  <span>Delete</span>
                </button>
              </>
            )}
            
            <button
              onClick={runWorkflow}
              disabled={isRunning || nodes.length === 0}
              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PlayIcon className="h-5 w-5" />
              <span>{isRunning ? 'Running...' : 'Run'}</span>
            </button>
            
            <button
              onClick={saveWorkflow}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Save
            </button>
            
            <button
              onClick={clearWorkflow}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Add Node Menu */}
      {showAddMenu && (
        <div className="absolute top-20 right-4 z-50 bg-black/90 backdrop-blur-xl border border-white/20 rounded-xl p-4 shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-semibold">Add Node</h3>
            <button
              onClick={() => setShowAddMenu(false)}
              className="text-gray-400 hover:text-white"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {nodeTypes.map((nodeType) => (
              <button
                key={nodeType.type}
                onClick={() => addNode(nodeType)}
                className="flex items-center space-x-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors text-sm"
                style={{ borderLeft: `3px solid ${nodeType.color}` }}
              >
                <span>{nodeType.icon}</span>
                <span>{nodeType.label.replace(/[⚡⚙️❓🌐📧💾🔄⏱️]/g, '').trim()}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {showEditModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#2a2435] border border-white/10 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-white mb-4">Edit Node Name</h3>
            <input
              type="text"
              value={editLabel}
              onChange={(e) => setEditLabel(e.target.value)}
              className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 mb-4"
              placeholder="Node label"
            />
            <div className="flex space-x-3">
              <button
                onClick={() => setShowEditModal(false)}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveEdit}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Configure Modal */}
      {showConfigModal && selectedNode && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 overflow-y-auto">
          <div className="bg-[#2a2435] border border-white/10 rounded-xl p-6 max-w-2xl w-full mx-4 my-8">
            <h3 className="text-xl font-bold text-white mb-4">
              Configure: {selectedNode.data.label}
            </h3>
            
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
              {/* Condition Node Configuration */}
              {selectedNode.data.nodeType === 'condition' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Condition Type</label>
                    <select
                      value={nodeConfig.conditionType || 'equals'}
                      onChange={(e) => setNodeConfig({...nodeConfig, conditionType: e.target.value})}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="equals">Equals (==)</option>
                      <option value="notEquals">Not Equals (!=)</option>
                      <option value="greaterThan">Greater Than (&gt;)</option>
                      <option value="lessThan">Less Than (&lt;)</option>
                      <option value="contains">Contains</option>
                      <option value="startsWith">Starts With</option>
                      <option value="endsWith">Ends With</option>
                      <option value="isEmpty">Is Empty</option>
                      <option value="isNotEmpty">Is Not Empty</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Field to Check</label>
                    <input
                      type="text"
                      value={nodeConfig.field || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, field: e.target.value})}
                      placeholder="e.g., email, status, amount"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Value to Compare</label>
                    <input
                      type="text"
                      value={nodeConfig.value || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, value: e.target.value})}
                      placeholder="e.g., VIP, completed, 100"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </>
              )}

              {/* API Node Configuration */}
              {selectedNode.data.nodeType === 'api' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">HTTP Method</label>
                    <select
                      value={nodeConfig.method || 'GET'}
                      onChange={(e) => setNodeConfig({...nodeConfig, method: e.target.value})}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="GET">GET</option>
                      <option value="POST">POST</option>
                      <option value="PUT">PUT</option>
                      <option value="PATCH">PATCH</option>
                      <option value="DELETE">DELETE</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">API URL</label>
                    <input
                      type="text"
                      value={nodeConfig.url || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, url: e.target.value})}
                      placeholder="https://api.example.com/endpoint"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Headers (JSON)</label>
                    <textarea
                      value={nodeConfig.headers || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, headers: e.target.value})}
                      placeholder='{"Authorization": "Bearer token", "Content-Type": "application/json"}'
                      rows={3}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Body (JSON)</label>
                    <textarea
                      value={nodeConfig.body || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, body: e.target.value})}
                      placeholder='{"key": "value"}'
                      rows={4}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                    />
                  </div>
                </>
              )}

              {/* Email Node Configuration */}
              {selectedNode.data.nodeType === 'email' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">To Email</label>
                    <input
                      type="email"
                      value={nodeConfig.to || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, to: e.target.value})}
                      placeholder="recipient@example.com"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Subject</label>
                    <input
                      type="text"
                      value={nodeConfig.subject || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, subject: e.target.value})}
                      placeholder="Email subject"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Message</label>
                    <textarea
                      value={nodeConfig.message || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, message: e.target.value})}
                      placeholder="Email message body"
                      rows={6}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </>
              )}

              {/* Database Node Configuration */}
              {selectedNode.data.nodeType === 'database' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Operation</label>
                    <select
                      value={nodeConfig.operation || 'insert'}
                      onChange={(e) => setNodeConfig({...nodeConfig, operation: e.target.value})}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="insert">Insert</option>
                      <option value="update">Update</option>
                      <option value="delete">Delete</option>
                      <option value="select">Select</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Table Name</label>
                    <input
                      type="text"
                      value={nodeConfig.table || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, table: e.target.value})}
                      placeholder="users, orders, products"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Data (JSON)</label>
                    <textarea
                      value={nodeConfig.data || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, data: e.target.value})}
                      placeholder='{"name": "John", "email": "john@example.com"}'
                      rows={4}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                    />
                  </div>
                </>
              )}

              {/* Transform Node Configuration */}
              {selectedNode.data.nodeType === 'transform' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Transform Type</label>
                    <select
                      value={nodeConfig.transformType || 'map'}
                      onChange={(e) => setNodeConfig({...nodeConfig, transformType: e.target.value})}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="map">Map Fields</option>
                      <option value="filter">Filter Data</option>
                      <option value="format">Format Data</option>
                      <option value="merge">Merge Data</option>
                      <option value="split">Split Data</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Transformation Rules (JSON)</label>
                    <textarea
                      value={nodeConfig.rules || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, rules: e.target.value})}
                      placeholder='{"oldField": "newField", "email": "userEmail"}'
                      rows={6}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                    />
                  </div>
                </>
              )}

              {/* Delay Node Configuration */}
              {selectedNode.data.nodeType === 'delay' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Delay Duration</label>
                    <input
                      type="number"
                      value={nodeConfig.duration || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, duration: e.target.value})}
                      placeholder="5"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Time Unit</label>
                    <select
                      value={nodeConfig.unit || 'seconds'}
                      onChange={(e) => setNodeConfig({...nodeConfig, unit: e.target.value})}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="seconds">Seconds</option>
                      <option value="minutes">Minutes</option>
                      <option value="hours">Hours</option>
                      <option value="days">Days</option>
                    </select>
                  </div>
                </>
              )}

              {/* Trigger/Action Node Configuration */}
              {(selectedNode.data.nodeType === 'trigger' || selectedNode.data.nodeType === 'action') && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
                    <textarea
                      value={nodeConfig.description || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, description: e.target.value})}
                      placeholder="Describe what this node does..."
                      rows={3}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Custom Parameters (JSON)</label>
                    <textarea
                      value={nodeConfig.parameters || ''}
                      onChange={(e) => setNodeConfig({...nodeConfig, parameters: e.target.value})}
                      placeholder='{"key": "value"}'
                      rows={4}
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                    />
                  </div>
                </>
              )}
            </div>

            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setShowConfigModal(false)}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveConfig}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Canvas */}
        <div className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            fitView
            className="bg-slate-950"
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#4b5563" />
            <Controls className="bg-black/40 border border-white/10" />
            <MiniMap 
              className="bg-black/40 border border-white/10" 
              nodeColor={(node) => node.style?.background as string || '#3b82f6'}
            />
          </ReactFlow>
        </div>

        {/* Execution Log Sidebar */}
        {executionLog.length > 0 && (
          <div className="w-80 bg-black/40 backdrop-blur-xl border-l border-white/10 p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold">Execution Log</h3>
              <button
                onClick={() => setExecutionLog([])}
                className="text-gray-400 hover:text-white"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-2">
              {executionLog.map((log, index) => (
                <div
                  key={index}
                  className="text-sm text-gray-300 bg-white/5 rounded px-3 py-2 font-mono"
                >
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="bg-black/40 backdrop-blur-xl border-t border-white/10 p-3">
        <div className="flex items-center justify-center space-x-6 text-sm text-gray-400">
          <span>💡 Click nodes to select</span>
          <span>🔗 Drag from node to node to connect</span>
          <span>➕ Add nodes from top menu</span>
          <span>▶️ Run to execute workflow</span>
        </div>
      </div>
    </div>
  );
}

export default function FullWorkflowBuilder() {
  return (
    <ReactFlowProvider>
      <WorkflowBuilderContent />
    </ReactFlowProvider>
  );
}
