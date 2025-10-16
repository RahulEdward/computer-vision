import React, { useState, useCallback } from 'react';
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
  ConnectionMode,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './n8n-nodes.css';
import { motion } from 'framer-motion';
import {
  PlayIcon,
  StopIcon,
  PlusIcon,
  TrashIcon,
  CogIcon,
  ClockIcon as TimerIcon,
  CircleStackIcon,
  CodeBracketIcon,
  EnvelopeIcon,
  GlobeAltIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  DocumentTextIcon,
  DocumentIcon,
  CloudIcon,
} from '@heroicons/react/24/outline';
import { workflowEngine, WorkflowExecution, WorkflowNode as EngineWorkflowNode, WorkflowEdge as EngineWorkflowEdge } from '../../services/workflowEngine';
import { nodeRegistry } from '../../nodes/NodeRegistry';
import { INodeTypeDescription, NODE_CATEGORIES } from '../../types/n8nTypes';
import { ExecutionContext } from '../../services/ExecutionContext';
import NodePropertyEditor from './NodePropertyEditor';
import { NodeValidator, IValidationResult } from '../../services/NodeValidator';
import { ValidationPanel, ValidationStatus } from './ValidationPanel';
import { WorkflowErrorBoundary } from './ErrorBoundary';
import WorkflowGuide from './WorkflowGuide';
import WorkflowToolbar from './WorkflowToolbar';
import { useReactFlow } from 'reactflow';

const initialNodes: Node[] = [
  {
    id: 'start',
    type: 'default',
    data: { label: '▶ Start', type: 'trigger' },
    position: { x: 100, y: 200 },
  },
  {
    id: '1',
    type: 'default',
    data: { label: 'Process Data', type: 'action' },
    position: { x: 300, y: 200 },
  },
  {
    id: '2',
    type: 'default',
    data: { label: 'API Call', type: 'action' },
    position: { x: 500, y: 200 },
  },
  {
    id: 'end',
    type: 'default',
    data: { label: '■ End', type: 'output' },
    position: { x: 700, y: 200 },
  },
];

const initialEdges: Edge[] = [
  { id: 'e-start-1', source: 'start', target: '1', animated: true },
  { id: 'e-1-2', source: '1', target: '2', animated: true },
  { id: 'e-2-end', source: '2', target: 'end', animated: true },
];

// Get node types from registry
const getNodeTypesFromRegistry = () => {
  const nodeDescriptions = nodeRegistry.getAllNodeDescriptions();
  return nodeDescriptions.map((desc: INodeTypeDescription) => ({
    type: desc.name,
    label: desc.displayName,
    icon: getIconForNodeType(desc.name),
    color: getColorForCategory(desc.group[0]),
    category: desc.group[0],
    description: desc.description,
    nodeDescription: desc,
  }));
};

const getIconForNodeType = (nodeType: string) => {
  const iconMap: Record<string, any> = {
    timer: TimerIcon,
    httpRequest: CloudIcon,
    transform: CodeBracketIcon,
    condition: DocumentTextIcon,
    database: CircleStackIcon,
    file: DocumentIcon,
    email: EnvelopeIcon,
    code: CodeBracketIcon,
  };
  return iconMap[nodeType] || CodeBracketIcon;
};

const getColorForCategory = (category: string) => {
  const colorMap: Record<string, string> = {
    [NODE_CATEGORIES.TRIGGER]: 'from-green-500 to-green-600',
    [NODE_CATEGORIES.ACTION]: 'from-blue-500 to-blue-600',
    [NODE_CATEGORIES.TRANSFORM]: 'from-purple-500 to-purple-600',
    [NODE_CATEGORIES.CORE]: 'from-gray-500 to-gray-600',
    [NODE_CATEGORIES.INTEGRATION]: 'from-indigo-500 to-indigo-600',
    [NODE_CATEGORIES.UTILITY]: 'from-yellow-500 to-yellow-600',
  };
  return colorMap[category] || 'from-gray-500 to-gray-600';
};

const WorkflowBuilder = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedNodeType, setSelectedNodeType] = useState('default');
  const [currentExecution, setCurrentExecution] = useState<WorkflowExecution | null>(null);
  const [executionHistory, setExecutionHistory] = useState<WorkflowExecution[]>([]);
  const [showExecutionPanel, setShowExecutionPanel] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showPropertyEditor, setShowPropertyEditor] = useState(false);
  const [validationResults, setValidationResults] = useState<Record<string, IValidationResult>>({});
  const [showValidationPanel, setShowValidationPanel] = useState(false);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Fit view to screen
  const handleFitView = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.fitView({ padding: 0.2, duration: 400 });
    }
  }, [reactFlowInstance]);

  // Delete selected nodes
  const handleDeleteSelected = useCallback(() => {
    const selectedNodes = nodes.filter(node => node.selected);
    if (selectedNodes.length > 0) {
      const selectedNodeIds = selectedNodes.map(n => n.id);
      setNodes(nodes.filter(node => !node.selected));
      setEdges(edges.filter(edge => 
        !selectedNodeIds.includes(edge.source) && !selectedNodeIds.includes(edge.target)
      ));
    }
  }, [nodes, edges, setNodes, setEdges]);

  // Handle keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        handleDeleteSelected();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleDeleteSelected]);

  const addNode = (type: string) => {
    // Get node description from registry
    const nodeInstance = nodeRegistry.getNode(type);
    if (!nodeInstance) {
      console.error(`Node type ${type} not found in registry`);
      return;
    }

    const nodeDescription = nodeInstance.description;
    
    // Create default parameter values from node properties
    const defaultParameters: Record<string, any> = {};
    nodeDescription.properties.forEach(prop => {
      if (prop.default !== undefined) {
        defaultParameters[prop.name] = prop.default;
      }
    });
    
    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      type: 'default',
      position: { x: Math.random() * 400, y: Math.random() * 400 },
      data: { 
        label: nodeDescription.displayName,
        type: type,
        nodeType: type,
        description: nodeDescription.description,
        parameters: defaultParameters,
        nodeDescription: nodeDescription,
      },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const handleNodeClick = (event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setShowPropertyEditor(true);
  };

  const handlePropertyChange = (nodeId: string, newValues: Record<string, any>) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                parameters: newValues,
              },
            }
          : node
      )
    );
  };

  const executeWorkflow = async () => {
    if (isRunning) return;

    setIsRunning(true);
    setShowExecutionPanel(true);

    try {
      // Convert ReactFlow nodes/edges to engine format
      const engineNodes: EngineWorkflowNode[] = nodes.map(node => ({
        id: node.id,
        type: node.data.nodeType || 'default',
        data: {
          label: node.data.label,
          config: node.data.config,
          code: node.data.code,
        },
        position: node.position,
      }));

      const engineEdges: EngineWorkflowEdge[] = edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
      }));

      const execution = await workflowEngine.executeWorkflow(engineNodes, engineEdges);
      
      setCurrentExecution(execution);
      setExecutionHistory(prev => [execution, ...prev.slice(0, 9)]);
      
    } catch (error) {
      console.error('Workflow execution failed:', error);
    } finally {
      setIsRunning(false);
    }
  };



  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Workflow Guide */}
      <WorkflowGuide />
      
      {/* Workflow Toolbar */}
      <WorkflowToolbar 
        onFitView={handleFitView}
        onDeleteSelected={handleDeleteSelected}
        hasSelectedNodes={nodes.some(node => node.selected)}
      />
      
      {/* Main Content Area */}
      <div className="flex-1 flex">
        {/* Left Sidebar - Node Palette */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Add Nodes</h3>
            <div className="relative">
              <input
                type="text"
                placeholder="Search nodes..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
              <div className="absolute left-3 top-2.5">
                <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-4">
              {/* Dynamic Node Categories */}
              {nodeRegistry.getAvailableCategories().map((category) => {
                const categoryNodes = nodeRegistry.getNodesByCategory(category);
                return (
                  <div key={category}>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {categoryNodes.map((nodeType, index) => {
                        const IconComponent = getIconForNodeType(nodeType.name);
                        const colorClass = `bg-gradient-to-r ${getColorForCategory(category)}`;
                        const uniqueKey = `${category}-${nodeType.name}-${index}`;
                        
                        return (
                          <motion.div
                            key={uniqueKey}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => addNode(nodeType.name)}
                            className="p-3 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg cursor-pointer transition-all duration-200 group"
                          >
                            <div className="flex flex-col items-center text-center space-y-1">
                              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colorClass} group-hover:scale-110 transition-transform duration-200`}>
                                <IconComponent className="h-4 w-4 text-white" />
                              </div>
                              <span className="text-xs font-medium text-gray-700">{nodeType.displayName}</span>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Canvas Area */}
        <div className="flex-1 flex flex-col bg-gray-100">
          {/* Canvas Header */}
          <div className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <span>Zoom:</span>
                <span className="font-medium">100%</span>
              </div>
              <div className="h-4 w-px bg-gray-300"></div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <span>Nodes:</span>
                <span className="font-medium">{nodes.length}</span>
              </div>
            </div>
          </div>

          {/* Main Canvas */}
          <div className="flex-1 relative">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={handleNodeClick}
              onInit={setReactFlowInstance}
              connectionMode={ConnectionMode.Loose}
              fitView
              deleteKeyCode="Delete"
              className="bg-gray-50"
              nodeTypes={{
                default: ({ data, selected }) => {
                  const getNodeIcon = (type: string) => {
                    switch (type) {
                      case 'trigger':
                        return <PlayIcon className="h-4 w-4 text-green-600" />;
                      case 'action':
                        return <CogIcon className="h-4 w-4 text-blue-600" />;
                      case 'condition':
                        return <CodeBracketIcon className="h-4 w-4 text-orange-600" />;
                      case 'data':
                        return <CircleStackIcon className="h-4 w-4 text-purple-600" />;
                      default:
                        return <CodeBracketIcon className="h-4 w-4 text-gray-600" />;
                    }
                  };

                  const getNodeColor = (type: string) => {
                    switch (type) {
                      case 'trigger':
                        return selected ? 'border-green-500 shadow-lg shadow-green-100' : 'border-gray-200 hover:border-green-300';
                      case 'action':
                        return selected ? 'border-blue-500 shadow-lg shadow-blue-100' : 'border-gray-200 hover:border-blue-300';
                      case 'condition':
                        return selected ? 'border-orange-500 shadow-lg shadow-orange-100' : 'border-gray-200 hover:border-orange-300';
                      case 'data':
                        return selected ? 'border-purple-500 shadow-lg shadow-purple-100' : 'border-gray-200 hover:border-purple-300';
                      default:
                        return selected ? 'border-gray-500 shadow-lg' : 'border-gray-200 hover:border-gray-300';
                    }
                  };

                  const getBgColor = (type: string) => {
                    switch (type) {
                      case 'trigger':
                        return 'bg-green-50';
                      case 'action':
                        return 'bg-blue-50';
                      case 'condition':
                        return 'bg-orange-50';
                      case 'data':
                        return 'bg-purple-50';
                      default:
                        return 'bg-gray-50';
                    }
                  };

                  return (
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`px-4 py-3 bg-white border-2 rounded-xl shadow-sm transition-all duration-200 min-w-[180px] cursor-pointer ${getNodeColor(data.type || 'default')}`}
                    >
                      <div className="flex items-center space-x-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${getBgColor(data.type || 'default')}`}>
                          {getNodeIcon(data.type || 'default')}
                        </div>
                        <div className="flex-1">
                          <div className="font-semibold text-gray-900 text-sm">{data.label}</div>
                          <div className="text-xs text-gray-500 flex items-center space-x-1">
                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                            <span>Ready</span>
                          </div>
                        </div>
                        {selected && (
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                        )}
                      </div>
                      
                      {/* Connection Points */}
                      <div className="absolute -left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-white border-2 border-gray-300 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
                      <div className="absolute -right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-white border-2 border-gray-300 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    </motion.div>
                  );
                },
              }}
            >
              <Background 
                color="#e5e7eb" 
                gap={20} 
                size={1}
                variant={BackgroundVariant.Dots}
              />
              <Controls 
                className="bg-white border border-gray-200 shadow-sm rounded-lg"
                showInteractive={false}
              />
              <MiniMap
                className="bg-white border border-gray-200 shadow-sm rounded-lg"
                nodeColor="#8b5cf6"
                maskColor="rgba(0, 0, 0, 0.1)"
                nodeStrokeWidth={2}
              />
            </ReactFlow>
            
            {/* Empty State */}
            {nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                    <CodeBracketIcon className="h-8 w-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Start building your workflow</h3>
                  <p className="text-gray-500 mb-4">Drag nodes from the sidebar to get started</p>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => addNode('trigger')}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-all duration-200"
                  >
                    Add your first node
                  </motion.button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar - Properties Panel */}
        <div className="w-80 bg-white border-l border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Workflow Settings</h3>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Workflow Details */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Details</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Workflow Name
                  </label>
                  <input
                    type="text"
                    defaultValue="My Automation Workflow"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Describe what this workflow does..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Execution Settings */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Execution</h4>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Trigger Type
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    <option value="manual">Manual</option>
                    <option value="schedule">Scheduled</option>
                    <option value="webhook">Webhook</option>
                    <option value="file">File Change</option>
                  </select>
                </div>
                <div>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded border-gray-300 text-purple-600 focus:ring-purple-500" />
                    <span className="text-xs text-gray-700">Save execution data</span>
                  </label>
                </div>
                <div>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded border-gray-300 text-purple-600 focus:ring-purple-500" />
                    <span className="text-xs text-gray-700">Enable error notifications</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Statistics</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Total Runs:</span>
                  <span className="font-medium text-gray-900">1,247</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Success Rate:</span>
                  <span className="font-medium text-green-600">98.5%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Avg. Duration:</span>
                  <span className="font-medium text-gray-900">2.3s</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Last Run:</span>
                  <span className="font-medium text-gray-900">5 min ago</span>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="pt-4 border-t border-gray-200">
              <div className="space-y-2">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-all duration-200"
                >
                  Save Workflow
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-all duration-200"
                >
                  Export
                </motion.button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Node Property Editor */}
      {showPropertyEditor && selectedNode && selectedNode.data.nodeDescription && (
        <NodePropertyEditor
          nodeId={selectedNode.id}
          properties={selectedNode.data.nodeDescription.properties}
          values={selectedNode.data.parameters || {}}
          onChange={handlePropertyChange}
          onClose={() => {
            setShowPropertyEditor(false);
            setSelectedNode(null);
          }}
        />
      )}

    </div>
  );
};

export default WorkflowBuilder;