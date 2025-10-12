import React, { memo, useState, useCallback, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  ChevronDown, 
  ChevronRight, 
  Folder, 
  FolderOpen,
  Settings,
  Users,
  Activity,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  Clock,
  Maximize2,
  Minimize2,
  Copy,
  Trash2,
  Edit3
} from 'lucide-react';

export interface GroupNodeData {
  label: string;
  type: 'group';
  childIds?: string[];
  isCollapsed?: boolean;
  level: number;
  description?: string;
  color?: string;
  metadata?: {
    createdAt?: number;
    updatedAt?: number;
    createdBy?: string;
    tags?: string[];
  };
  performance?: {
    childrenStatus: Record<string, 'idle' | 'running' | 'completed' | 'error'>;
    totalExecutionTime?: number;
    averageExecutionTime?: number;
    successRate?: number;
  };
  config?: {
    allowParallelExecution?: boolean;
    maxConcurrentChildren?: number;
    errorHandling?: 'stop' | 'continue' | 'retry';
    priority?: 'low' | 'medium' | 'high' | 'critical';
  };
  validation?: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  };
}

const getGroupStatusColor = (performance?: GroupNodeData['performance']) => {
  if (!performance?.childrenStatus) return 'border-gray-600';
  
  const statuses = Object.values(performance.childrenStatus);
  const hasError = statuses.includes('error');
  const hasRunning = statuses.includes('running');
  const allCompleted = statuses.length > 0 && statuses.every(s => s === 'completed');
  
  if (hasError) return 'border-red-500';
  if (hasRunning) return 'border-blue-500 animate-pulse';
  if (allCompleted) return 'border-green-500';
  return 'border-gray-600';
};

const getGroupBackgroundColor = (color?: string, isCollapsed?: boolean) => {
  const opacity = isCollapsed ? '0.05' : '0.1';
  
  switch (color) {
    case 'blue': return `rgba(59, 130, 246, ${opacity})`;
    case 'green': return `rgba(34, 197, 94, ${opacity})`;
    case 'purple': return `rgba(147, 51, 234, ${opacity})`;
    case 'red': return `rgba(239, 68, 68, ${opacity})`;
    case 'yellow': return `rgba(245, 158, 11, ${opacity})`;
    case 'indigo': return `rgba(99, 102, 241, ${opacity})`;
    case 'pink': return `rgba(236, 72, 153, ${opacity})`;
    case 'teal': return `rgba(20, 184, 166, ${opacity})`;
    default: return `rgba(59, 130, 246, ${opacity})`;
  }
};

const getGroupBorderColor = (color?: string) => {
  switch (color) {
    case 'blue': return '#3b82f6';
    case 'green': return '#22c55e';
    case 'purple': return '#9333ea';
    case 'red': return '#ef4444';
    case 'yellow': return '#f59e0b';
    case 'indigo': return '#6366f1';
    case 'pink': return '#ec4899';
    case 'teal': return '#14b8a6';
    default: return '#3b82f6';
  }
};

export const GroupNode = memo<NodeProps<GroupNodeData>>(({ data, selected, id }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editLabel, setEditLabel] = useState(data.label);

  const statusColor = useMemo(() => getGroupStatusColor(data.performance), [data.performance]);
  const backgroundColor = useMemo(() => getGroupBackgroundColor(data.color, data.isCollapsed), [data.color, data.isCollapsed]);
  const borderColor = useMemo(() => getGroupBorderColor(data.color), [data.color]);

  const handleToggleCollapse = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // This would be handled by the parent component
    console.log('Toggle collapse for group:', id);
  }, [id]);

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

  const getChildrenStatusSummary = () => {
    if (!data.performance?.childrenStatus) return null;
    
    const statuses = Object.values(data.performance.childrenStatus);
    const counts = {
      idle: statuses.filter(s => s === 'idle').length,
      running: statuses.filter(s => s === 'running').length,
      completed: statuses.filter(s => s === 'completed').length,
      error: statuses.filter(s => s === 'error').length,
    };
    
    return counts;
  };

  const childrenSummary = getChildrenStatusSummary();

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

  return (
    <div 
      className={`
        relative border-2 border-dashed rounded-lg transition-all duration-200
        ${selected ? 'border-blue-400 shadow-lg shadow-blue-400/20' : statusColor}
        ${data.isCollapsed ? 'min-w-[250px] min-h-[100px]' : 'min-w-[400px] min-h-[300px]'}
        hover:shadow-xl
      `}
      style={{ 
        backgroundColor,
        borderColor: selected ? '#60a5fa' : borderColor,
      }}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="w-4 h-4 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
        style={{ left: -8, top: 20 }}
      />

      {/* Header */}
      <div className={`
        bg-gray-800 border-b border-gray-600 p-3 rounded-t-lg flex items-center justify-between
        ${data.isCollapsed ? 'rounded-b-lg' : ''}
      `}>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleToggleCollapse}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {data.isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
          
          {data.isCollapsed ? (
            <Folder className="w-4 h-4 text-blue-400" />
          ) : (
            <FolderOpen className="w-4 h-4 text-blue-400" />
          )}
          
          {isEditing ? (
            <input
              type="text"
              value={editLabel}
              onChange={(e) => setEditLabel(e.target.value)}
              onBlur={handleSaveLabel}
              onKeyDown={handleKeyPress}
              className="bg-gray-700 text-white px-2 py-1 rounded text-sm border border-gray-600 focus:border-blue-400 outline-none"
              autoFocus
            />
          ) : (
            <span className="text-white font-medium text-sm">
              {data.label}
            </span>
          )}
          
          {data.childIds && (
            <span className="text-gray-400 text-xs">
              ({data.childIds.length} nodes)
            </span>
          )}
        </div>
        
        <div className="flex items-center space-x-1">
          {data.config?.priority && (
            <span className={`px-2 py-1 rounded text-xs ${getPriorityColor(data.config.priority)}`}>
              {data.config.priority}
            </span>
          )}
          
          <button
            onClick={handleEditLabel}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <Edit3 className="w-3 h-3" />
          </button>
          
          <button
            onClick={handleToggleExpand}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Content - Only show when not collapsed */}
      {!data.isCollapsed && (
        <div className="p-3 space-y-3">
          {/* Description */}
          {data.description && (
            <div className="text-gray-300 text-sm">
              {data.description}
            </div>
          )}

          {/* Children Status Summary */}
          {childrenSummary && (
            <div className="bg-gray-900/50 rounded p-2">
              <div className="text-xs text-gray-300 font-medium mb-2">Children Status</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {childrenSummary.running > 0 && (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    <span className="text-blue-400">Running: {childrenSummary.running}</span>
                  </div>
                )}
                {childrenSummary.completed > 0 && (
                  <div className="flex items-center space-x-1">
                    <CheckCircle className="w-3 h-3 text-green-400" />
                    <span className="text-green-400">Completed: {childrenSummary.completed}</span>
                  </div>
                )}
                {childrenSummary.error > 0 && (
                  <div className="flex items-center space-x-1">
                    <AlertTriangle className="w-3 h-3 text-red-400" />
                    <span className="text-red-400">Error: {childrenSummary.error}</span>
                  </div>
                )}
                {childrenSummary.idle > 0 && (
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3 text-gray-400" />
                    <span className="text-gray-400">Idle: {childrenSummary.idle}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {isExpanded && data.performance && (
            <div className="bg-gray-900/50 rounded p-2 space-y-2">
              <div className="text-xs text-gray-300 font-medium">Performance</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {data.performance.totalExecutionTime && (
                  <div>
                    <span className="text-gray-400">Total Time:</span>
                    <span className="text-white ml-1">
                      {formatExecutionTime(data.performance.totalExecutionTime)}
                    </span>
                  </div>
                )}
                {data.performance.averageExecutionTime && (
                  <div>
                    <span className="text-gray-400">Avg Time:</span>
                    <span className="text-white ml-1">
                      {formatExecutionTime(data.performance.averageExecutionTime)}
                    </span>
                  </div>
                )}
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
                {data.config.allowParallelExecution !== undefined && (
                  <div>
                    <span className="text-gray-400">Parallel Execution:</span>
                    <span className="text-white ml-1">
                      {data.config.allowParallelExecution ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                )}
                {data.config.maxConcurrentChildren && (
                  <div>
                    <span className="text-gray-400">Max Concurrent:</span>
                    <span className="text-white ml-1">{data.config.maxConcurrentChildren}</span>
                  </div>
                )}
                {data.config.errorHandling && (
                  <div>
                    <span className="text-gray-400">Error Handling:</span>
                    <span className="text-white ml-1 capitalize">{data.config.errorHandling}</span>
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

          {/* Metadata */}
          {isExpanded && data.metadata && (
            <div className="bg-gray-900/50 rounded p-2 space-y-1">
              <div className="text-xs text-gray-300 font-medium">Metadata</div>
              <div className="space-y-1 text-xs">
                {data.metadata.createdBy && (
                  <div>
                    <span className="text-gray-400">Created by:</span>
                    <span className="text-white ml-1">{data.metadata.createdBy}</span>
                  </div>
                )}
                {data.metadata.createdAt && (
                  <div>
                    <span className="text-gray-400">Created:</span>
                    <span className="text-white ml-1">
                      {new Date(data.metadata.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                )}
                {data.metadata.tags && data.metadata.tags.length > 0 && (
                  <div>
                    <span className="text-gray-400">Tags:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {data.metadata.tags.map((tag, index) => (
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
        </div>
      )}

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="w-4 h-4 bg-gray-600 border-2 border-gray-400 hover:bg-blue-500 transition-colors"
        style={{ right: -8, top: 20 }}
      />

      {/* Level Indicator */}
      {data.level > 0 && (
        <div className="absolute -top-2 -left-2 w-5 h-5 bg-purple-500 rounded-full flex items-center justify-center">
          <span className="text-white text-xs font-bold">{data.level}</span>
        </div>
      )}
    </div>
  );
});

GroupNode.displayName = 'GroupNode';