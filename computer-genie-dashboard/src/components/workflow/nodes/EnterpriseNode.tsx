import React, { memo, useState, useCallback, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity,
  Zap,
  Database,
  Globe,
  Code,
  Filter,
  GitBranch,
  Mail,
  FileText,
  Cpu,
  BarChart3,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2
} from 'lucide-react';

export interface EnterpriseNodeData {
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
    lastRun?: number;
    successRate?: number;
  };
  config?: {
    retries?: number;
    timeout?: number;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    tags?: string[];
  };
  validation?: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  };
}

const getNodeIcon = (type: string) => {
  const iconMap: Record<string, React.ComponentType<any>> = {
    trigger: Play,
    action: Zap,
    condition: GitBranch,
    transform: Filter,
    database: Database,
    api: Globe,
    code: Code,
    email: Mail,
    file: FileText,
    timer: Clock,
    webhook: Activity,
    http: Globe,
    process: Cpu,
    analytics: BarChart3,
  };
  
  return iconMap[type] || Activity;
};

const getNodeColor = (type: string, status: string) => {
  const baseColors: Record<string, string> = {
    trigger: 'from-green-500 to-green-600',
    action: 'from-blue-500 to-blue-600',
    condition: 'from-yellow-500 to-yellow-600',
    transform: 'from-purple-500 to-purple-600',
    database: 'from-indigo-500 to-indigo-600',
    api: 'from-cyan-500 to-cyan-600',
    code: 'from-gray-500 to-gray-600',
    email: 'from-red-500 to-red-600',
    file: 'from-orange-500 to-orange-600',
    timer: 'from-pink-500 to-pink-600',
    webhook: 'from-teal-500 to-teal-600',
  };

  const statusOverrides: Record<string, string> = {
    running: 'from-blue-400 to-blue-500 animate-pulse',
    completed: 'from-green-400 to-green-500',
    error: 'from-red-400 to-red-500',
  };

  return statusOverrides[status] || baseColors[type] || 'from-gray-500 to-gray-600';
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'running':
      return <Play className="w-3 h-3 text-blue-400 animate-spin" />;
    case 'completed':
      return <CheckCircle className="w-3 h-3 text-green-400" />;
    case 'error':
      return <AlertTriangle className="w-3 h-3 text-red-400" />;
    default:
      return <Pause className="w-3 h-3 text-gray-400" />;
  }
};

export const EnterpriseNode = memo<NodeProps<EnterpriseNodeData>>(({ data, selected, id }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const IconComponent = useMemo(() => getNodeIcon(data.type), [data.type]);
  const nodeColor = useMemo(() => getNodeColor(data.type, data.performance?.status || 'idle'), [data.type, data.performance?.status]);
  const statusIcon = useMemo(() => getStatusIcon(data.performance?.status || 'idle'), [data.performance?.status]);

  const handleToggleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded(!isExpanded);
  }, [isExpanded]);

  const handleToggleDetails = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setShowDetails(!showDetails);
  }, [showDetails]);

  const formatExecutionTime = (time?: number) => {
    if (!time) return 'N/A';
    if (time < 1000) return `${time}ms`;
    return `${(time / 1000).toFixed(2)}s`;
  };

  const formatMemoryUsage = (memory?: number) => {
    if (!memory) return 'N/A';
    if (memory < 1024) return `${memory}B`;
    if (memory < 1024 * 1024) return `${(memory / 1024).toFixed(1)}KB`;
    return `${(memory / (1024 * 1024)).toFixed(1)}MB`;
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

  return (
    <div className={`
      relative bg-gray-800 border-2 rounded-lg shadow-lg transition-all duration-200
      ${selected ? 'border-blue-400 shadow-blue-400/20' : 'border-gray-600'}
      ${data.performance?.status === 'running' ? 'animate-pulse' : ''}
      hover:shadow-xl hover:border-gray-500
      min-w-[200px] max-w-[300px]
    `}>
      {/* Input Handles */}
      {data.type !== 'trigger' && (
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
          style={{ left: -6 }}
        />
      )}

      {/* Header */}
      <div className={`
        bg-gradient-to-r ${nodeColor} p-3 rounded-t-lg flex items-center justify-between
      `}>
        <div className="flex items-center space-x-2">
          <IconComponent className="w-4 h-4 text-white" />
          <span className="text-white font-medium text-sm truncate">
            {data.label}
          </span>
        </div>
        
        <div className="flex items-center space-x-1">
          {statusIcon}
          <button
            onClick={handleToggleDetails}
            className="text-white/70 hover:text-white transition-colors"
          >
            {showDetails ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
          </button>
          <button
            onClick={handleToggleExpand}
            className="text-white/70 hover:text-white transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-3 space-y-2">
        {/* Basic Info */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400 capitalize">{data.type}</span>
          {data.config?.priority && (
            <span className={`px-2 py-1 rounded text-xs ${getPriorityColor(data.config.priority)}`}>
              {data.config.priority}
            </span>
          )}
        </div>

        {/* Performance Metrics */}
        {showDetails && data.performance && (
          <div className="bg-gray-900/50 rounded p-2 space-y-1">
            <div className="text-xs text-gray-300 font-medium">Performance</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-400">Time:</span>
                <span className="text-white ml-1">
                  {formatExecutionTime(data.performance.executionTime)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Memory:</span>
                <span className="text-white ml-1">
                  {formatMemoryUsage(data.performance.memoryUsage)}
                </span>
              </div>
              {data.performance.successRate !== undefined && (
                <div className="col-span-2">
                  <span className="text-gray-400">Success Rate:</span>
                  <span className="text-white ml-1">
                    {(data.performance.successRate * 100).toFixed(1)}%
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
              {data.config.retries !== undefined && (
                <div>
                  <span className="text-gray-400">Retries:</span>
                  <span className="text-white ml-1">{data.config.retries}</span>
                </div>
              )}
              {data.config.timeout !== undefined && (
                <div>
                  <span className="text-gray-400">Timeout:</span>
                  <span className="text-white ml-1">{data.config.timeout}ms</span>
                </div>
              )}
              {data.config.tags && data.config.tags.length > 0 && (
                <div>
                  <span className="text-gray-400">Tags:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {data.config.tags.map((tag, index) => (
                      <span
                        key={index}
                        className="px-1 py-0.5 bg-blue-900/30 text-blue-300 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Validation Errors */}
        {data.validation && !data.validation.isValid && (
          <div className="bg-red-900/20 border border-red-700 rounded p-2">
            <div className="flex items-center space-x-1 text-red-400 text-xs font-medium">
              <AlertTriangle className="w-3 h-3" />
              <span>Validation Issues</span>
            </div>
            {data.validation.errors.length > 0 && (
              <div className="mt-1 space-y-1">
                {data.validation.errors.map((error, index) => (
                  <div key={index} className="text-xs text-red-300">
                    • {error}
                  </div>
                ))}
              </div>
            )}
            {data.validation.warnings.length > 0 && (
              <div className="mt-1 space-y-1">
                {data.validation.warnings.map((warning, index) => (
                  <div key={index} className="text-xs text-yellow-300">
                    ⚠ {warning}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Child Nodes Indicator */}
        {data.childIds && data.childIds.length > 0 && (
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>Child Nodes: {data.childIds.length}</span>
            {data.isCollapsed && (
              <span className="text-blue-400">Collapsed</span>
            )}
          </div>
        )}
      </div>

      {/* Output Handles */}
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
        style={{ right: -6 }}
      />

      {/* Level Indicator */}
      {data.level > 0 && (
        <div className="absolute -top-2 -left-2 w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
          <span className="text-white text-xs font-bold">{data.level}</span>
        </div>
      )}

      {/* Status Indicator */}
      <div className={`
        absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-gray-800
        ${data.performance?.status === 'running' ? 'bg-blue-400 animate-pulse' : ''}
        ${data.performance?.status === 'completed' ? 'bg-green-400' : ''}
        ${data.performance?.status === 'error' ? 'bg-red-400' : ''}
        ${data.performance?.status === 'idle' ? 'bg-gray-400' : ''}
      `} />
    </div>
  );
});

EnterpriseNode.displayName = 'EnterpriseNode';