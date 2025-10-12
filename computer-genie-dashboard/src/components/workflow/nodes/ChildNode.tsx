import React, { memo, useState, useCallback, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw,
  Settings,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  BarChart3,
  Eye,
  EyeOff,
  Link,
  Unlink,
  ArrowUp,
  Copy,
  Trash2,
  Edit3,
  Zap,
  Database,
  Globe,
  Code,
  Filter,
  Send,
  FileText,
  Calendar,
  User
} from 'lucide-react';

export interface ChildNodeData {
  label: string;
  type: 'action' | 'trigger' | 'transform' | 'condition' | 'integration' | 'utility';
  parentId?: string;
  level: number;
  status: 'idle' | 'running' | 'completed' | 'error' | 'paused';
  description?: string;
  category?: string;
  parameters?: Record<string, any>;
  metadata?: {
    createdAt?: number;
    updatedAt?: number;
    createdBy?: string;
    version?: string;
    tags?: string[];
  };
  performance?: {
    executionTime?: number;
    lastExecuted?: number;
    executionCount?: number;
    successRate?: number;
    averageExecutionTime?: number;
  };
  config?: {
    timeout?: number;
    retryCount?: number;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    enabled?: boolean;
    breakpoint?: boolean;
  };
  validation?: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    requiredInputs?: string[];
    providedOutputs?: string[];
  };
  connections?: {
    inputConnections: number;
    outputConnections: number;
    maxInputs?: number;
    maxOutputs?: number;
  };
}

const getNodeIcon = (type: string, category?: string) => {
  switch (type) {
    case 'trigger':
      return <Zap className="w-4 h-4" />;
    case 'action':
      switch (category) {
        case 'http': return <Globe className="w-4 h-4" />;
        case 'database': return <Database className="w-4 h-4" />;
        case 'file': return <FileText className="w-4 h-4" />;
        case 'email': return <Send className="w-4 h-4" />;
        default: return <Play className="w-4 h-4" />;
      }
    case 'transform':
      return <Code className="w-4 h-4" />;
    case 'condition':
      return <Filter className="w-4 h-4" />;
    case 'integration':
      return <Link className="w-4 h-4" />;
    case 'utility':
      return <Settings className="w-4 h-4" />;
    default:
      return <Activity className="w-4 h-4" />;
  }
};

const getNodeColor = (type: string, status: string) => {
  const baseColors = {
    trigger: { bg: 'bg-yellow-900/20', border: 'border-yellow-500', text: 'text-yellow-400' },
    action: { bg: 'bg-blue-900/20', border: 'border-blue-500', text: 'text-blue-400' },
    transform: { bg: 'bg-purple-900/20', border: 'border-purple-500', text: 'text-purple-400' },
    condition: { bg: 'bg-orange-900/20', border: 'border-orange-500', text: 'text-orange-400' },
    integration: { bg: 'bg-green-900/20', border: 'border-green-500', text: 'text-green-400' },
    utility: { bg: 'bg-gray-900/20', border: 'border-gray-500', text: 'text-gray-400' },
  };

  const statusOverrides = {
    running: { border: 'border-blue-400 animate-pulse', bg: 'bg-blue-900/30' },
    completed: { border: 'border-green-400', bg: 'bg-green-900/30' },
    error: { border: 'border-red-400', bg: 'bg-red-900/30' },
    paused: { border: 'border-yellow-400', bg: 'bg-yellow-900/30' },
  };

  const base = baseColors[type as keyof typeof baseColors] || baseColors.action;
  const statusOverride = statusOverrides[status as keyof typeof statusOverrides];

  return {
    ...base,
    ...statusOverride,
  };
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'running':
      return <Activity className="w-3 h-3 text-blue-400 animate-pulse" />;
    case 'completed':
      return <CheckCircle className="w-3 h-3 text-green-400" />;
    case 'error':
      return <AlertTriangle className="w-3 h-3 text-red-400" />;
    case 'paused':
      return <Pause className="w-3 h-3 text-yellow-400" />;
    default:
      return <Clock className="w-3 h-3 text-gray-400" />;
  }
};

const formatExecutionTime = (time?: number) => {
  if (!time) return 'N/A';
  if (time < 1000) return `${time}ms`;
  return `${(time / 1000).toFixed(2)}s`;
};

const getPriorityColor = (priority?: string) => {
  switch (priority) {
    case 'critical': return 'text-red-400 bg-red-900/20';
    case 'high': return 'text-orange-400 bg-orange-900/20';
    case 'medium': return 'text-yellow-400 bg-yellow-900/20';
    case 'low': return 'text-green-400 bg-green-900/20';
    default: return 'text-gray-400 bg-gray-900/20';
  }
};

export const ChildNode = memo<NodeProps<ChildNodeData>>(({ data, selected, id }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editLabel, setEditLabel] = useState(data.label);

  const nodeColors = useMemo(() => getNodeColor(data.type, data.status), [data.type, data.status]);

  const handleToggleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded(!isExpanded);
  }, [isExpanded]);

  const handleEditLabel = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(true);
  }, []);

  const handleSaveLabel = useCallback(() => {
    setIsEditing(false);
    // This would be handled by the parent component
    console.log('Save label:', editLabel);
  }, [editLabel]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveLabel();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setEditLabel(data.label);
    }
  }, [handleSaveLabel, data.label]);

  const handleExecute = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    console.log('Execute node:', id);
  }, [id]);

  const handlePause = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    console.log('Pause node:', id);
  }, [id]);

  const handleStop = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    console.log('Stop node:', id);
  }, [id]);

  const handleReset = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    console.log('Reset node:', id);
  }, [id]);

  const canExecute = data.status === 'idle' || data.status === 'completed' || data.status === 'error';
  const canPause = data.status === 'running';
  const canStop = data.status === 'running' || data.status === 'paused';

  return (
    <div 
      className={`
        relative border-2 rounded-lg transition-all duration-200 min-w-[200px]
        ${selected ? 'border-blue-400 shadow-lg shadow-blue-400/20' : nodeColors.border}
        ${nodeColors.bg}
        hover:shadow-xl
        ${!data.config?.enabled ? 'opacity-50' : ''}
      `}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
        style={{ left: -6 }}
      />

      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-600 p-2 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className={nodeColors.text}>
              {getNodeIcon(data.type, data.category)}
            </div>
            
            {isEditing ? (
              <input
                type="text"
                value={editLabel}
                onChange={(e) => setEditLabel(e.target.value)}
                onBlur={handleSaveLabel}
                onKeyDown={handleKeyPress}
                className="bg-gray-700 text-white px-1 py-0.5 rounded text-xs border border-gray-600 focus:border-blue-400 outline-none"
                autoFocus
              />
            ) : (
              <span className="text-white font-medium text-xs">
                {data.label}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-1">
            {getStatusIcon(data.status)}
            
            {data.config?.breakpoint && (
              <div className="w-2 h-2 bg-red-500 rounded-full" title="Breakpoint" />
            )}
            
            {data.config?.priority && (
              <span className={`px-1 py-0.5 rounded text-xs ${getPriorityColor(data.config.priority)}`}>
                {data.config.priority.charAt(0).toUpperCase()}
              </span>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center space-x-1">
            {canExecute && (
              <button
                onClick={handleExecute}
                className="text-green-400 hover:text-green-300 transition-colors"
                title="Execute"
              >
                <Play className="w-3 h-3" />
              </button>
            )}
            
            {canPause && (
              <button
                onClick={handlePause}
                className="text-yellow-400 hover:text-yellow-300 transition-colors"
                title="Pause"
              >
                <Pause className="w-3 h-3" />
              </button>
            )}
            
            {canStop && (
              <button
                onClick={handleStop}
                className="text-red-400 hover:text-red-300 transition-colors"
                title="Stop"
              >
                <Square className="w-3 h-3" />
              </button>
            )}
            
            <button
              onClick={handleReset}
              className="text-gray-400 hover:text-gray-300 transition-colors"
              title="Reset"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
          
          <div className="flex items-center space-x-1">
            <button
              onClick={handleEditLabel}
              className="text-gray-400 hover:text-white transition-colors"
              title="Edit"
            >
              <Edit3 className="w-3 h-3" />
            </button>
            
            <button
              onClick={handleToggleExpand}
              className="text-gray-400 hover:text-white transition-colors"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-2 space-y-2">
        {/* Description */}
        {data.description && (
          <div className="text-gray-300 text-xs">
            {data.description}
          </div>
        )}

        {/* Connection Info */}
        {data.connections && (
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">In: {data.connections.inputConnections}</span>
              {data.connections.maxInputs && (
                <span className="text-gray-500">/{data.connections.maxInputs}</span>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">Out: {data.connections.outputConnections}</span>
              {data.connections.maxOutputs && (
                <span className="text-gray-500">/{data.connections.maxOutputs}</span>
              )}
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        {isExpanded && data.performance && (
          <div className="bg-gray-900/50 rounded p-2 space-y-1">
            <div className="text-xs text-gray-300 font-medium">Performance</div>
            <div className="grid grid-cols-2 gap-1 text-xs">
              {data.performance.executionTime && (
                <div>
                  <span className="text-gray-400">Last:</span>
                  <span className="text-white ml-1">
                    {formatExecutionTime(data.performance.executionTime)}
                  </span>
                </div>
              )}
              {data.performance.averageExecutionTime && (
                <div>
                  <span className="text-gray-400">Avg:</span>
                  <span className="text-white ml-1">
                    {formatExecutionTime(data.performance.averageExecutionTime)}
                  </span>
                </div>
              )}
              {data.performance.executionCount && (
                <div>
                  <span className="text-gray-400">Runs:</span>
                  <span className="text-white ml-1">{data.performance.executionCount}</span>
                </div>
              )}
              {data.performance.successRate !== undefined && (
                <div>
                  <span className="text-gray-400">Success:</span>
                  <span className="text-white ml-1">
                    {(data.performance.successRate * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Configuration */}
        {isExpanded && data.config && (
          <div className="bg-gray-900/50 rounded p-2 space-y-1">
            <div className="text-xs text-gray-300 font-medium">Configuration</div>
            <div className="space-y-1 text-xs">
              {data.config.timeout && (
                <div>
                  <span className="text-gray-400">Timeout:</span>
                  <span className="text-white ml-1">{data.config.timeout}ms</span>
                </div>
              )}
              {data.config.retryCount !== undefined && (
                <div>
                  <span className="text-gray-400">Retries:</span>
                  <span className="text-white ml-1">{data.config.retryCount}</span>
                </div>
              )}
              <div>
                <span className="text-gray-400">Enabled:</span>
                <span className="text-white ml-1">
                  {data.config.enabled !== false ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Validation Issues */}
        {data.validation && !data.validation.isValid && (
          <div className="bg-red-900/20 border border-red-700 rounded p-2">
            <div className="flex items-center space-x-1 text-red-400 text-xs font-medium">
              <AlertTriangle className="w-3 h-3" />
              <span>Issues</span>
            </div>
            {data.validation.errors.length > 0 && (
              <div className="mt-1 space-y-1">
                {data.validation.errors.slice(0, 2).map((error, index) => (
                  <div key={index} className="text-xs text-red-300">
                    • {error}
                  </div>
                ))}
                {data.validation.errors.length > 2 && (
                  <div className="text-xs text-red-400">
                    +{data.validation.errors.length - 2} more
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Metadata */}
        {isExpanded && data.metadata && (
          <div className="bg-gray-900/50 rounded p-2 space-y-1">
            <div className="text-xs text-gray-300 font-medium">Metadata</div>
            <div className="space-y-1 text-xs">
              {data.metadata.version && (
                <div>
                  <span className="text-gray-400">Version:</span>
                  <span className="text-white ml-1">{data.metadata.version}</span>
                </div>
              )}
              {data.metadata.createdBy && (
                <div>
                  <span className="text-gray-400">Author:</span>
                  <span className="text-white ml-1">{data.metadata.createdBy}</span>
                </div>
              )}
              {data.metadata.tags && data.metadata.tags.length > 0 && (
                <div>
                  <span className="text-gray-400">Tags:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {data.metadata.tags.slice(0, 3).map((tag, index) => (
                      <span
                        key={index}
                        className="px-1 py-0.5 bg-blue-900/30 text-blue-300 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                    {data.metadata.tags.length > 3 && (
                      <span className="text-xs text-gray-400">
                        +{data.metadata.tags.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
        style={{ right: -6 }}
      />

      {/* Parent Connection Indicator */}
      {data.parentId && (
        <div className="absolute -top-1 -left-1 w-3 h-3 bg-purple-500 rounded-full flex items-center justify-center">
          <ArrowUp className="w-2 h-2 text-white" />
        </div>
      )}

      {/* Level Indicator */}
      {data.level > 0 && (
        <div className="absolute -top-2 -right-2 w-4 h-4 bg-indigo-500 rounded-full flex items-center justify-center">
          <span className="text-white text-xs font-bold">{data.level}</span>
        </div>
      )}
    </div>
  );
});

ChildNode.displayName = 'ChildNode';