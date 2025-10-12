import React, { useState, useCallback } from 'react';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  Save,
  FolderOpen,
  Download,
  Upload,
  Copy,
  Scissors,
  Clipboard,
  Undo,
  Redo,
  ZoomIn,
  ZoomOut,
  Maximize,
  Minimize,
  Grid,
  Eye,
  EyeOff,
  Settings,
  HelpCircle,
  Search,
  Filter,
  SortAsc,
  SortDesc,
  Layout,
  Layers,
  GitBranch,
  Share2,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  BarChart3,
  Users,
  MessageSquare,
  Bell,
  Bookmark,
  Star,
  Tag,
  FileText,
  Code,
  Database,
  Globe,
  Zap,
  Shield,
  Cpu,
  Memory,
  Timer,
  RefreshCw,
  Plus,
  Minus,
  X,
  ChevronDown,
  MoreHorizontal
} from 'lucide-react';

export interface WorkflowState {
  isRunning: boolean;
  isPaused: boolean;
  isModified: boolean;
  canUndo: boolean;
  canRedo: boolean;
  zoomLevel: number;
  viewMode: 'design' | 'debug' | 'monitor';
  showGrid: boolean;
  showMinimap: boolean;
  isLocked: boolean;
  collaborators: number;
  notifications: number;
}

export interface WorkflowStats {
  totalNodes: number;
  totalEdges: number;
  executionTime?: number;
  lastExecuted?: number;
  successRate?: number;
  errorCount: number;
  warningCount: number;
}

interface WorkflowToolbarProps {
  workflowState: WorkflowState;
  workflowStats: WorkflowStats;
  onExecute: () => void;
  onPause: () => void;
  onStop: () => void;
  onReset: () => void;
  onSave: () => void;
  onLoad: () => void;
  onExport: () => void;
  onImport: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomFit: () => void;
  onZoomReset: () => void;
  onToggleGrid: () => void;
  onToggleMinimap: () => void;
  onToggleLock: () => void;
  onViewModeChange: (mode: 'design' | 'debug' | 'monitor') => void;
  onSearch: (query: string) => void;
  onSettings: () => void;
  onHelp: () => void;
  onShare: () => void;
  onCollaboration: () => void;
  className?: string;
}

export const WorkflowToolbar: React.FC<WorkflowToolbarProps> = ({
  workflowState,
  workflowStats,
  onExecute,
  onPause,
  onStop,
  onReset,
  onSave,
  onLoad,
  onExport,
  onImport,
  onUndo,
  onRedo,
  onZoomIn,
  onZoomOut,
  onZoomFit,
  onZoomReset,
  onToggleGrid,
  onToggleMinimap,
  onToggleLock,
  onViewModeChange,
  onSearch,
  onSettings,
  onHelp,
  onShare,
  onCollaboration,
  className = ''
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showMoreActions, setShowMoreActions] = useState(false);
  const [showViewOptions, setShowViewOptions] = useState(false);

  const handleSearchSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    onSearch(searchQuery);
  }, [searchQuery, onSearch]);

  const formatExecutionTime = (ms?: number) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const formatLastExecuted = (timestamp?: number) => {
    if (!timestamp) return 'Never';
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  return (
    <div className={`bg-gray-900 border-b border-gray-700 px-4 py-2 flex items-center justify-between ${className}`}>
      {/* Left Section - Execution Controls */}
      <div className="flex items-center space-x-2">
        {/* Execution Controls */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          {workflowState.isRunning ? (
            <>
              <button
                onClick={onPause}
                className="flex items-center space-x-1 px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors"
                title="Pause execution"
              >
                <Pause className="w-4 h-4" />
                <span className="text-sm font-medium">Pause</span>
              </button>
              <button
                onClick={onStop}
                className="flex items-center space-x-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                title="Stop execution"
              >
                <Square className="w-4 h-4" />
                <span className="text-sm font-medium">Stop</span>
              </button>
            </>
          ) : (
            <button
              onClick={onExecute}
              className="flex items-center space-x-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              title="Execute workflow"
            >
              <Play className="w-4 h-4" />
              <span className="text-sm font-medium">Execute</span>
            </button>
          )}
          
          <button
            onClick={onReset}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Reset workflow"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* File Operations */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          <button
            onClick={onSave}
            className={`p-1.5 rounded transition-colors ${
              workflowState.isModified
                ? 'text-blue-400 hover:text-blue-300 hover:bg-gray-800'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
            title="Save workflow"
          >
            <Save className="w-4 h-4" />
          </button>
          
          <button
            onClick={onLoad}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Load workflow"
          >
            <FolderOpen className="w-4 h-4" />
          </button>

          <div className="relative">
            <button
              onClick={() => setShowMoreActions(!showMoreActions)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="More actions"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>

            {showMoreActions && (
              <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50 min-w-[160px]">
                <button
                  onClick={() => {
                    onExport();
                    setShowMoreActions(false);
                  }}
                  className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 first:rounded-t-lg"
                >
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
                <button
                  onClick={() => {
                    onImport();
                    setShowMoreActions(false);
                  }}
                  className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700"
                >
                  <Upload className="w-4 h-4" />
                  <span>Import</span>
                </button>
                <button
                  onClick={() => {
                    onShare();
                    setShowMoreActions(false);
                  }}
                  className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 last:rounded-b-lg"
                >
                  <Share2 className="w-4 h-4" />
                  <span>Share</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Edit Operations */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          <button
            onClick={onUndo}
            disabled={!workflowState.canUndo}
            className={`p-1.5 rounded transition-colors ${
              workflowState.canUndo
                ? 'text-gray-400 hover:text-white hover:bg-gray-800'
                : 'text-gray-600 cursor-not-allowed'
            }`}
            title="Undo"
          >
            <Undo className="w-4 h-4" />
          </button>
          
          <button
            onClick={onRedo}
            disabled={!workflowState.canRedo}
            className={`p-1.5 rounded transition-colors ${
              workflowState.canRedo
                ? 'text-gray-400 hover:text-white hover:bg-gray-800'
                : 'text-gray-600 cursor-not-allowed'
            }`}
            title="Redo"
          >
            <Redo className="w-4 h-4" />
          </button>
        </div>

        {/* View Mode */}
        <div className="flex items-center space-x-1">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {(['design', 'debug', 'monitor'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => onViewModeChange(mode)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                  workflowState.viewMode === mode
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Center Section - Search */}
      <div className="flex-1 max-w-md mx-4">
        <form onSubmit={handleSearchSubmit} className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search nodes, connections..."
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-400 transition-colors"
          />
        </form>
      </div>

      {/* Right Section - View Controls & Status */}
      <div className="flex items-center space-x-2">
        {/* Workflow Stats */}
        <div className="flex items-center space-x-4 text-xs text-gray-400 border-r border-gray-700 pr-3">
          <div className="flex items-center space-x-1">
            <Layers className="w-3 h-3" />
            <span>{workflowStats.totalNodes} nodes</span>
          </div>
          
          <div className="flex items-center space-x-1">
            <GitBranch className="w-3 h-3" />
            <span>{workflowStats.totalEdges} edges</span>
          </div>

          {workflowStats.executionTime && (
            <div className="flex items-center space-x-1">
              <Timer className="w-3 h-3" />
              <span>{formatExecutionTime(workflowStats.executionTime)}</span>
            </div>
          )}

          <div className="flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>{formatLastExecuted(workflowStats.lastExecuted)}</span>
          </div>

          {workflowStats.errorCount > 0 && (
            <div className="flex items-center space-x-1 text-red-400">
              <AlertTriangle className="w-3 h-3" />
              <span>{workflowStats.errorCount}</span>
            </div>
          )}

          {workflowStats.successRate !== undefined && (
            <div className="flex items-center space-x-1 text-green-400">
              <CheckCircle className="w-3 h-3" />
              <span>{Math.round(workflowStats.successRate * 100)}%</span>
            </div>
          )}
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          <button
            onClick={onZoomOut}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          
          <span className="text-xs text-gray-400 min-w-[3rem] text-center">
            {Math.round(workflowState.zoomLevel * 100)}%
          </span>
          
          <button
            onClick={onZoomIn}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <button
            onClick={onZoomFit}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Fit to screen"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>

        {/* View Options */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          <div className="relative">
            <button
              onClick={() => setShowViewOptions(!showViewOptions)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="View options"
            >
              <Layout className="w-4 h-4" />
            </button>

            {showViewOptions && (
              <div className="absolute top-full right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50 min-w-[160px]">
                <button
                  onClick={() => {
                    onToggleGrid();
                    setShowViewOptions(false);
                  }}
                  className="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 first:rounded-t-lg"
                >
                  <div className="flex items-center space-x-2">
                    <Grid className="w-4 h-4" />
                    <span>Grid</span>
                  </div>
                  {workflowState.showGrid && <CheckCircle className="w-4 h-4 text-green-400" />}
                </button>
                
                <button
                  onClick={() => {
                    onToggleMinimap();
                    setShowViewOptions(false);
                  }}
                  className="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 last:rounded-b-lg"
                >
                  <div className="flex items-center space-x-2">
                    <Minimize className="w-4 h-4" />
                    <span>Minimap</span>
                  </div>
                  {workflowState.showMinimap && <CheckCircle className="w-4 h-4 text-green-400" />}
                </button>
              </div>
            )}
          </div>

          <button
            onClick={onToggleLock}
            className={`p-1.5 rounded transition-colors ${
              workflowState.isLocked
                ? 'text-red-400 hover:text-red-300 hover:bg-gray-800'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
            title={workflowState.isLocked ? 'Unlock workflow' : 'Lock workflow'}
          >
            {workflowState.isLocked ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
          </button>
        </div>

        {/* Collaboration & Notifications */}
        <div className="flex items-center space-x-1 border-r border-gray-700 pr-3">
          <button
            onClick={onCollaboration}
            className="relative p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Collaboration"
          >
            <Users className="w-4 h-4" />
            {workflowState.collaborators > 0 && (
              <span className="absolute -top-1 -right-1 bg-green-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {workflowState.collaborators}
              </span>
            )}
          </button>

          <button
            className="relative p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Notifications"
          >
            <Bell className="w-4 h-4" />
            {workflowState.notifications > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {workflowState.notifications}
              </span>
            )}
          </button>
        </div>

        {/* Settings & Help */}
        <div className="flex items-center space-x-1">
          <button
            onClick={onSettings}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
          
          <button
            onClick={onHelp}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            title="Help"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};