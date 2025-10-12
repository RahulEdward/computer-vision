import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  BarChart3,
  Cpu,
  Memory,
  Timer,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Database,
  Network,
  HardDrive,
  Gauge,
  LineChart,
  PieChart,
  Monitor,
  Server,
  Wifi,
  RefreshCw,
  Pause,
  Play,
  Square,
  Download,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Settings,
  X,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';

export interface PerformanceMetrics {
  timestamp: number;
  cpu: {
    usage: number;
    cores: number;
    temperature?: number;
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    latency: number;
    throughput: number;
  };
  storage: {
    used: number;
    total: number;
    readSpeed: number;
    writeSpeed: number;
  };
  workflow: {
    nodesExecuted: number;
    totalNodes: number;
    executionTime: number;
    successRate: number;
    errorCount: number;
    warningCount: number;
    throughput: number;
  };
  system: {
    uptime: number;
    loadAverage: number[];
    processes: number;
  };
}

export interface PerformanceAlert {
  id: string;
  type: 'error' | 'warning' | 'info';
  metric: string;
  message: string;
  timestamp: number;
  threshold?: number;
  currentValue?: number;
}

interface PerformanceMonitorProps {
  metrics: PerformanceMetrics[];
  alerts: PerformanceAlert[];
  isMonitoring: boolean;
  refreshInterval: number;
  onToggleMonitoring: () => void;
  onRefreshIntervalChange: (interval: number) => void;
  onExportMetrics: () => void;
  onClearAlerts: () => void;
  className?: string;
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  metrics,
  alerts,
  isMonitoring,
  refreshInterval,
  onToggleMonitoring,
  onRefreshIntervalChange,
  onExportMetrics,
  onClearAlerts,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string>('overview');
  const [showAlerts, setShowAlerts] = useState(true);
  const [timeRange, setTimeRange] = useState<'1m' | '5m' | '15m' | '1h'>('5m');

  const latestMetrics = metrics[metrics.length - 1];
  
  const filteredMetrics = useMemo(() => {
    const now = Date.now();
    const ranges = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000
    };
    
    const cutoff = now - ranges[timeRange];
    return metrics.filter(m => m.timestamp >= cutoff);
  }, [metrics, timeRange]);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
    return `${(ms / 3600000).toFixed(1)}h`;
  };

  const formatUptime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  const getMetricTrend = (metricPath: string): 'up' | 'down' | 'stable' => {
    if (filteredMetrics.length < 2) return 'stable';
    
    const recent = filteredMetrics.slice(-5);
    const getValue = (metrics: PerformanceMetrics, path: string): number => {
      const keys = path.split('.');
      let value: any = metrics;
      for (const key of keys) {
        value = value?.[key];
      }
      return typeof value === 'number' ? value : 0;
    };

    const values = recent.map(m => getValue(m, metricPath));
    const first = values[0];
    const last = values[values.length - 1];
    
    const threshold = 0.05; // 5% change threshold
    const change = Math.abs(last - first) / first;
    
    if (change < threshold) return 'stable';
    return last > first ? 'up' : 'down';
  };

  const getStatusColor = (percentage: number, inverted = false): string => {
    if (inverted) {
      if (percentage >= 90) return 'text-red-400';
      if (percentage >= 70) return 'text-yellow-400';
      return 'text-green-400';
    } else {
      if (percentage >= 90) return 'text-green-400';
      if (percentage >= 70) return 'text-yellow-400';
      return 'text-red-400';
    }
  };

  const renderMetricCard = (
    title: string,
    value: string,
    percentage?: number,
    trend?: 'up' | 'down' | 'stable',
    icon?: React.ReactNode,
    inverted = false
  ) => (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {icon && <div className="text-gray-400">{icon}</div>}
          <span className="text-sm font-medium text-gray-300">{title}</span>
        </div>
        {trend && (
          <div className={`flex items-center space-x-1 ${
            trend === 'up' ? 'text-green-400' :
            trend === 'down' ? 'text-red-400' :
            'text-gray-400'
          }`}>
            {trend === 'up' && <TrendingUp className="w-3 h-3" />}
            {trend === 'down' && <TrendingDown className="w-3 h-3" />}
            {trend === 'stable' && <div className="w-3 h-0.5 bg-current" />}
          </div>
        )}
      </div>
      
      <div className="text-xl font-bold text-white mb-1">{value}</div>
      
      {percentage !== undefined && (
        <div className="flex items-center space-x-2">
          <div className="flex-1 bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                inverted
                  ? percentage >= 90 ? 'bg-red-500' :
                    percentage >= 70 ? 'bg-yellow-500' : 'bg-green-500'
                  : percentage >= 90 ? 'bg-green-500' :
                    percentage >= 70 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(percentage, 100)}%` }}
            />
          </div>
          <span className={`text-xs font-medium ${getStatusColor(percentage, inverted)}`}>
            {percentage.toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );

  const renderAlerts = () => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">Active Alerts</h4>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">{alerts.length} alerts</span>
          {alerts.length > 0 && (
            <button
              onClick={onClearAlerts}
              className="text-xs text-red-400 hover:text-red-300"
            >
              Clear All
            </button>
          )}
        </div>
      </div>
      
      {alerts.length === 0 ? (
        <div className="text-center py-4 text-gray-500 text-sm">
          <CheckCircle className="w-6 h-6 mx-auto mb-2 text-green-400" />
          No active alerts
        </div>
      ) : (
        <div className="space-y-2 max-h-32 overflow-y-auto">
          {alerts.map(alert => (
            <div
              key={alert.id}
              className={`p-2 rounded border-l-4 ${
                alert.type === 'error' ? 'bg-red-900/20 border-red-500' :
                alert.type === 'warning' ? 'bg-yellow-900/20 border-yellow-500' :
                'bg-blue-900/20 border-blue-500'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-1">
                    {alert.type === 'error' && <AlertTriangle className="w-3 h-3 text-red-400" />}
                    {alert.type === 'warning' && <AlertTriangle className="w-3 h-3 text-yellow-400" />}
                    {alert.type === 'info' && <Info className="w-3 h-3 text-blue-400" />}
                    <span className="text-xs font-medium text-gray-300">{alert.metric}</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">{alert.message}</p>
                  {alert.threshold && alert.currentValue && (
                    <div className="text-xs text-gray-500 mt-1">
                      Current: {alert.currentValue} | Threshold: {alert.threshold}
                    </div>
                  )}
                </div>
                <span className="text-xs text-gray-500">
                  {formatDuration(Date.now() - alert.timestamp)} ago
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  if (!isExpanded) {
    return (
      <div className={`bg-gray-900 border border-gray-700 rounded-lg ${className}`}>
        <div className="p-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-sm font-medium text-white">Performance</span>
            </div>
            
            {latestMetrics && (
              <div className="flex items-center space-x-4 text-xs text-gray-400">
                <div className="flex items-center space-x-1">
                  <Cpu className="w-3 h-3" />
                  <span>{latestMetrics.cpu.usage.toFixed(1)}%</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Memory className="w-3 h-3" />
                  <span>{latestMetrics.memory.percentage.toFixed(1)}%</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Timer className="w-3 h-3" />
                  <span>{formatDuration(latestMetrics.workflow.executionTime)}</span>
                </div>
              </div>
            )}
            
            {alerts.length > 0 && (
              <div className="flex items-center space-x-1 text-red-400">
                <AlertTriangle className="w-3 h-3" />
                <span className="text-xs">{alerts.length}</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-green-400' : 'bg-gray-600'}`} />
            <button
              onClick={() => setIsExpanded(true)}
              className="text-gray-400 hover:text-white"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-green-400" />
            <h3 className="text-lg font-semibold text-white">Performance Monitor</h3>
            <div className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-green-400' : 'bg-gray-600'}`} />
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white text-sm focus:outline-none focus:border-blue-400"
            >
              <option value="1m">1 minute</option>
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="1h">1 hour</option>
            </select>
            
            <button
              onClick={onToggleMonitoring}
              className={`p-2 rounded transition-colors ${
                isMonitoring
                  ? 'text-red-400 hover:text-red-300 hover:bg-gray-800'
                  : 'text-green-400 hover:text-green-300 hover:bg-gray-800'
              }`}
              title={isMonitoring ? 'Stop monitoring' : 'Start monitoring'}
            >
              {isMonitoring ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            
            <button
              onClick={onExportMetrics}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Export metrics"
            >
              <Download className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setIsExpanded(false)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Minimize"
            >
              <Minimize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <div className="p-4">
        {latestMetrics ? (
          <div className="space-y-6">
            {/* Overview Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {renderMetricCard(
                'CPU Usage',
                `${latestMetrics.cpu.usage.toFixed(1)}%`,
                latestMetrics.cpu.usage,
                getMetricTrend('cpu.usage'),
                <Cpu className="w-4 h-4" />,
                true
              )}
              
              {renderMetricCard(
                'Memory',
                formatBytes(latestMetrics.memory.used),
                latestMetrics.memory.percentage,
                getMetricTrend('memory.percentage'),
                <Memory className="w-4 h-4" />,
                true
              )}
              
              {renderMetricCard(
                'Execution Time',
                formatDuration(latestMetrics.workflow.executionTime),
                undefined,
                getMetricTrend('workflow.executionTime'),
                <Timer className="w-4 h-4" />
              )}
              
              {renderMetricCard(
                'Success Rate',
                `${(latestMetrics.workflow.successRate * 100).toFixed(1)}%`,
                latestMetrics.workflow.successRate * 100,
                getMetricTrend('workflow.successRate'),
                <CheckCircle className="w-4 h-4" />
              )}
            </div>

            {/* Detailed Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* System Metrics */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
                  <Server className="w-4 h-4" />
                  <span>System Metrics</span>
                </h4>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Uptime</span>
                    <span className="text-white">{formatUptime(latestMetrics.system.uptime)}</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Load Average</span>
                    <span className="text-white">
                      {latestMetrics.system.loadAverage.map(l => l.toFixed(2)).join(', ')}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Processes</span>
                    <span className="text-white">{latestMetrics.system.processes}</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">CPU Cores</span>
                    <span className="text-white">{latestMetrics.cpu.cores}</span>
                  </div>
                  
                  {latestMetrics.cpu.temperature && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">CPU Temperature</span>
                      <span className={`${
                        latestMetrics.cpu.temperature > 80 ? 'text-red-400' :
                        latestMetrics.cpu.temperature > 60 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {latestMetrics.cpu.temperature}°C
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Workflow Metrics */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
                  <Zap className="w-4 h-4" />
                  <span>Workflow Metrics</span>
                </h4>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Nodes Executed</span>
                    <span className="text-white">
                      {latestMetrics.workflow.nodesExecuted} / {latestMetrics.workflow.totalNodes}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Throughput</span>
                    <span className="text-white">{latestMetrics.workflow.throughput.toFixed(1)} nodes/s</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Errors</span>
                    <span className={latestMetrics.workflow.errorCount > 0 ? 'text-red-400' : 'text-green-400'}>
                      {latestMetrics.workflow.errorCount}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Warnings</span>
                    <span className={latestMetrics.workflow.warningCount > 0 ? 'text-yellow-400' : 'text-green-400'}>
                      {latestMetrics.workflow.warningCount}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Network Latency</span>
                    <span className="text-white">{latestMetrics.network.latency}ms</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Storage I/O</span>
                    <span className="text-white">
                      R: {formatBytes(latestMetrics.storage.readSpeed)}/s
                      W: {formatBytes(latestMetrics.storage.writeSpeed)}/s
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Alerts */}
            {showAlerts && (
              <div>
                {renderAlerts()}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Activity className="w-8 h-8 mx-auto mb-2" />
            <p>No performance data available</p>
            <p className="text-sm">Start monitoring to see metrics</p>
          </div>
        )}
      </div>
    </div>
  );
};