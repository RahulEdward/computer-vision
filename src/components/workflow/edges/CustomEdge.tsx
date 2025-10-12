import React, { memo } from 'react';
import {
  EdgeProps,
  getBezierPath,
  EdgeLabelRenderer,
  BaseEdge,
  EdgeMarker,
  MarkerType,
} from 'reactflow';
import { 
  Zap, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity,
  X,
  Settings
} from 'lucide-react';

export interface CustomEdgeData {
  label?: string;
  status?: 'idle' | 'active' | 'success' | 'error' | 'warning';
  dataType?: 'json' | 'text' | 'binary' | 'stream' | 'file';
  bandwidth?: number;
  latency?: number;
  throughput?: number;
  errorRate?: number;
  isConditional?: boolean;
  condition?: string;
  animated?: boolean;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  metadata?: {
    createdAt?: number;
    lastDataFlow?: number;
    totalDataTransferred?: number;
  };
}

const getEdgeColor = (status?: string, priority?: string) => {
  if (status) {
    switch (status) {
      case 'active': return '#3b82f6'; // blue
      case 'success': return '#22c55e'; // green
      case 'error': return '#ef4444'; // red
      case 'warning': return '#f59e0b'; // yellow
      default: return '#6b7280'; // gray
    }
  }
  
  if (priority) {
    switch (priority) {
      case 'critical': return '#dc2626'; // red
      case 'high': return '#ea580c'; // orange
      case 'medium': return '#ca8a04'; // yellow
      case 'low': return '#16a34a'; // green
      default: return '#6b7280'; // gray
    }
  }
  
  return '#6b7280'; // default gray
};

const getDataTypeIcon = (dataType?: string) => {
  switch (dataType) {
    case 'json': return '{}';
    case 'text': return 'T';
    case 'binary': return '01';
    case 'stream': return '~';
    case 'file': return '📄';
    default: return '•';
  }
};

const getStatusIcon = (status?: string) => {
  switch (status) {
    case 'active':
      return <Activity className="w-3 h-3 text-blue-400" />;
    case 'success':
      return <CheckCircle className="w-3 h-3 text-green-400" />;
    case 'error':
      return <AlertTriangle className="w-3 h-3 text-red-400" />;
    case 'warning':
      return <AlertTriangle className="w-3 h-3 text-yellow-400" />;
    default:
      return <Clock className="w-3 h-3 text-gray-400" />;
  }
};

const formatBytes = (bytes?: number) => {
  if (!bytes) return 'N/A';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
};

const formatLatency = (latency?: number) => {
  if (!latency) return 'N/A';
  if (latency < 1000) return `${latency}ms`;
  return `${(latency / 1000).toFixed(2)}s`;
};

export const CustomEdge = memo<EdgeProps<CustomEdgeData>>(({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
  selected,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const edgeColor = getEdgeColor(data?.status, data?.priority);
  const isAnimated = data?.animated || data?.status === 'active';
  const strokeWidth = selected ? 3 : data?.priority === 'critical' ? 2.5 : 2;

  const edgeStyle = {
    ...style,
    stroke: edgeColor,
    strokeWidth,
    strokeDasharray: data?.isConditional ? '5,5' : undefined,
  };

  const markerEndId = `marker-${id}`;

  return (
    <>
      {/* Custom marker definition */}
      <defs>
        <marker
          id={markerEndId}
          markerWidth="12"
          markerHeight="12"
          refX="6"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path
            d="M0,0 L0,6 L9,3 z"
            fill={edgeColor}
            stroke={edgeColor}
            strokeWidth="1"
          />
        </marker>
      </defs>

      {/* Main edge path */}
      <BaseEdge
        path={edgePath}
        style={edgeStyle}
        markerEnd={`url(#${markerEndId})`}
      />

      {/* Animated flow indicator */}
      {isAnimated && (
        <BaseEdge
          path={edgePath}
          style={{
            ...edgeStyle,
            strokeDasharray: '8,8',
            strokeDashoffset: '16',
            animation: 'dash 1s linear infinite',
            opacity: 0.6,
          }}
        />
      )}

      {/* Edge label */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 10,
            pointerEvents: 'all',
          }}
          className="nodrag nopan"
        >
          {(data?.label || data?.dataType || data?.status) && (
            <div className={`
              bg-gray-800 border border-gray-600 rounded-lg px-2 py-1 shadow-lg
              ${selected ? 'border-blue-400 shadow-blue-400/20' : ''}
              transition-all duration-200
            `}>
              {/* Main label */}
              {data?.label && (
                <div className="text-white text-xs font-medium mb-1">
                  {data.label}
                </div>
              )}

              {/* Status and data type */}
              <div className="flex items-center space-x-2">
                {data?.status && (
                  <div className="flex items-center space-x-1">
                    {getStatusIcon(data.status)}
                    <span className="text-xs text-gray-300 capitalize">
                      {data.status}
                    </span>
                  </div>
                )}

                {data?.dataType && (
                  <div className="flex items-center space-x-1">
                    <span className="text-xs text-blue-400 font-mono">
                      {getDataTypeIcon(data.dataType)}
                    </span>
                    <span className="text-xs text-gray-300 uppercase">
                      {data.dataType}
                    </span>
                  </div>
                )}
              </div>

              {/* Performance metrics */}
              {(data?.bandwidth || data?.latency || data?.throughput) && (
                <div className="mt-1 pt-1 border-t border-gray-700 space-y-1">
                  {data.bandwidth && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Bandwidth:</span>
                      <span className="text-white">{formatBytes(data.bandwidth)}/s</span>
                    </div>
                  )}
                  {data.latency && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Latency:</span>
                      <span className="text-white">{formatLatency(data.latency)}</span>
                    </div>
                  )}
                  {data.throughput && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Throughput:</span>
                      <span className="text-white">{data.throughput}/s</span>
                    </div>
                  )}
                  {data.errorRate !== undefined && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Error Rate:</span>
                      <span className={`${data.errorRate > 0.1 ? 'text-red-400' : 'text-green-400'}`}>
                        {(data.errorRate * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Conditional logic */}
              {data?.isConditional && data?.condition && (
                <div className="mt-1 pt-1 border-t border-gray-700">
                  <div className="text-xs text-yellow-400">
                    <span className="font-medium">If:</span> {data.condition}
                  </div>
                </div>
              )}

              {/* Priority indicator */}
              {data?.priority && data.priority !== 'medium' && (
                <div className="mt-1 pt-1 border-t border-gray-700">
                  <div className={`text-xs font-medium ${
                    data.priority === 'critical' ? 'text-red-400' :
                    data.priority === 'high' ? 'text-orange-400' :
                    data.priority === 'low' ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    Priority: {data.priority.toUpperCase()}
                  </div>
                </div>
              )}

              {/* Metadata */}
              {selected && data?.metadata && (
                <div className="mt-1 pt-1 border-t border-gray-700 space-y-1">
                  {data.metadata.totalDataTransferred && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Total Data:</span>
                      <span className="text-white">
                        {formatBytes(data.metadata.totalDataTransferred)}
                      </span>
                    </div>
                  )}
                  {data.metadata.lastDataFlow && (
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Last Flow:</span>
                      <span className="text-white">
                        {new Date(data.metadata.lastDataFlow).toLocaleTimeString()}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </EdgeLabelRenderer>

      {/* Add CSS for animation */}
      <style jsx>{`
        @keyframes dash {
          to {
            stroke-dashoffset: 0;
          }
        }
      `}</style>
    </>
  );
});

CustomEdge.displayName = 'CustomEdge';