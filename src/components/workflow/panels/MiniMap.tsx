import React, { useCallback, useMemo } from 'react';
import { MiniMap as ReactFlowMiniMap } from 'reactflow';
import {
  Maximize2,
  Minimize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  Eye,
  EyeOff,
  Settings,
  Grid,
  Layers,
  Filter,
  Search,
  MapPin,
  Navigation,
  Compass,
  Target,
  Focus,
  Crosshair
} from 'lucide-react';

export interface MiniMapSettings {
  showNodes: boolean;
  showEdges: boolean;
  showLabels: boolean;
  showGrid: boolean;
  showViewport: boolean;
  nodeColor: string;
  edgeColor: string;
  backgroundColor: string;
  viewportColor: string;
  maskColor: string;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  width: number;
  height: number;
  zoomable: boolean;
  pannable: boolean;
  interactive: boolean;
}

export interface MiniMapNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: any;
  selected?: boolean;
  dragging?: boolean;
}

export interface MiniMapViewport {
  x: number;
  y: number;
  zoom: number;
  width: number;
  height: number;
}

interface MiniMapProps {
  nodes: MiniMapNode[];
  viewport: MiniMapViewport;
  settings: MiniMapSettings;
  onViewportChange: (viewport: Partial<MiniMapViewport>) => void;
  onNodeClick: (nodeId: string) => void;
  onNodeDoubleClick: (nodeId: string) => void;
  onSettingsChange: (settings: Partial<MiniMapSettings>) => void;
  className?: string;
}

export const MiniMap: React.FC<MiniMapProps> = ({
  nodes,
  viewport,
  settings,
  onViewportChange,
  onNodeClick,
  onNodeDoubleClick,
  onSettingsChange,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);
  const [isVisible, setIsVisible] = React.useState(true);

  // Calculate minimap bounds
  const bounds = useMemo(() => {
    if (nodes.length === 0) {
      return { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
    }

    const positions = nodes.map(node => node.position);
    const minX = Math.min(...positions.map(p => p.x)) - 100;
    const minY = Math.min(...positions.map(p => p.y)) - 100;
    const maxX = Math.max(...positions.map(p => p.x)) + 100;
    const maxY = Math.max(...positions.map(p => p.y)) + 100;

    return { minX, minY, maxX, maxY };
  }, [nodes]);

  // Get node color based on type and state
  const getNodeColor = useCallback((node: MiniMapNode) => {
    if (node.selected) return '#3b82f6'; // Blue for selected
    if (node.dragging) return '#f59e0b'; // Amber for dragging

    switch (node.type) {
      case 'trigger':
        return '#10b981'; // Green
      case 'action':
        return '#6366f1'; // Indigo
      case 'condition':
        return '#f59e0b'; // Amber
      case 'group':
        return '#8b5cf6'; // Purple
      case 'child':
        return '#06b6d4'; // Cyan
      default:
        return settings.nodeColor;
    }
  }, [settings.nodeColor]);

  // Handle viewport navigation
  const handleViewportClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert minimap coordinates to flow coordinates
    const flowX = bounds.minX + (x / rect.width) * (bounds.maxX - bounds.minX);
    const flowY = bounds.minY + (y / rect.height) * (bounds.maxY - bounds.minY);

    onViewportChange({
      x: -flowX + viewport.width / 2,
      y: -flowY + viewport.height / 2
    });
  }, [bounds, viewport.width, viewport.height, onViewportChange]);

  // Quick navigation actions
  const handleZoomIn = useCallback(() => {
    onViewportChange({ zoom: Math.min(viewport.zoom * 1.2, 4) });
  }, [viewport.zoom, onViewportChange]);

  const handleZoomOut = useCallback(() => {
    onViewportChange({ zoom: Math.max(viewport.zoom / 1.2, 0.1) });
  }, [viewport.zoom, onViewportChange]);

  const handleResetView = useCallback(() => {
    onViewportChange({ x: 0, y: 0, zoom: 1 });
  }, [onViewportChange]);

  const handleFitView = useCallback(() => {
    if (nodes.length === 0) return;

    const padding = 50;
    const width = bounds.maxX - bounds.minX + padding * 2;
    const height = bounds.maxY - bounds.minY + padding * 2;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    const scaleX = viewport.width / width;
    const scaleY = viewport.height / height;
    const scale = Math.min(scaleX, scaleY, 1);

    onViewportChange({
      x: viewport.width / 2 - centerX * scale,
      y: viewport.height / 2 - centerY * scale,
      zoom: scale
    });
  }, [nodes.length, bounds, viewport.width, viewport.height, onViewportChange]);

  // Settings panel
  const renderSettings = () => (
    <div className="absolute top-0 right-0 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-lg p-4 z-10">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-white">MiniMap Settings</h3>
        <button
          onClick={() => setShowSettings(false)}
          className="text-gray-400 hover:text-white"
        >
          ×
        </button>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-300">Show Nodes</label>
          <input
            type="checkbox"
            checked={settings.showNodes}
            onChange={(e) => onSettingsChange({ showNodes: e.target.checked })}
            className="rounded"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-300">Show Edges</label>
          <input
            type="checkbox"
            checked={settings.showEdges}
            onChange={(e) => onSettingsChange({ showEdges: e.target.checked })}
            className="rounded"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-300">Show Labels</label>
          <input
            type="checkbox"
            checked={settings.showLabels}
            onChange={(e) => onSettingsChange({ showLabels: e.target.checked })}
            className="rounded"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-300">Show Grid</label>
          <input
            type="checkbox"
            checked={settings.showGrid}
            onChange={(e) => onSettingsChange({ showGrid: e.target.checked })}
            className="rounded"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-300">Interactive</label>
          <input
            type="checkbox"
            checked={settings.interactive}
            onChange={(e) => onSettingsChange({ interactive: e.target.checked })}
            className="rounded"
          />
        </div>

        <div>
          <label className="text-xs text-gray-300 block mb-1">Position</label>
          <select
            value={settings.position}
            onChange={(e) => onSettingsChange({ position: e.target.value as any })}
            className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
          >
            <option value="top-left">Top Left</option>
            <option value="top-right">Top Right</option>
            <option value="bottom-left">Bottom Left</option>
            <option value="bottom-right">Bottom Right</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-300 block mb-1">Width</label>
          <input
            type="range"
            min="150"
            max="400"
            value={settings.width}
            onChange={(e) => onSettingsChange({ width: parseInt(e.target.value) })}
            className="w-full"
          />
          <span className="text-xs text-gray-400">{settings.width}px</span>
        </div>

        <div>
          <label className="text-xs text-gray-300 block mb-1">Height</label>
          <input
            type="range"
            min="100"
            max="300"
            value={settings.height}
            onChange={(e) => onSettingsChange({ height: parseInt(e.target.value) })}
            className="w-full"
          />
          <span className="text-xs text-gray-400">{settings.height}px</span>
        </div>
      </div>
    </div>
  );

  if (!isVisible) {
    return (
      <div className={`fixed ${settings.position.replace('-', ' ')} z-40 ${className}`}>
        <button
          onClick={() => setIsVisible(true)}
          className="bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
          title="Show MiniMap"
        >
          <MapPin className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className={`fixed ${settings.position.replace('-', ' ')} z-40 ${className}`}>
      <div className="relative">
        {/* Main MiniMap Container */}
        <div
          className={`bg-gray-900 border border-gray-700 rounded-lg overflow-hidden transition-all duration-200 ${
            isExpanded ? 'shadow-2xl' : 'shadow-lg'
          }`}
          style={{
            width: isExpanded ? settings.width * 1.2 : settings.width,
            height: isExpanded ? settings.height * 1.2 : settings.height
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-2 bg-gray-800 border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <Navigation className="w-3 h-3 text-gray-400" />
              <span className="text-xs font-medium text-gray-300">Overview</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <button
                onClick={handleZoomOut}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Zoom Out"
              >
                <ZoomOut className="w-3 h-3" />
              </button>
              
              <button
                onClick={handleZoomIn}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Zoom In"
              >
                <ZoomIn className="w-3 h-3" />
              </button>
              
              <button
                onClick={handleFitView}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Fit View"
              >
                <Target className="w-3 h-3" />
              </button>
              
              <button
                onClick={handleResetView}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Reset View"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
              
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Settings"
              >
                <Settings className="w-3 h-3" />
              </button>
              
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title={isExpanded ? "Collapse" : "Expand"}
              >
                {isExpanded ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
              </button>
              
              <button
                onClick={() => setIsVisible(false)}
                className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Hide MiniMap"
              >
                <EyeOff className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* MiniMap Content */}
          <div className="relative w-full h-full">
            <ReactFlowMiniMap
              nodeColor={getNodeColor}
              nodeStrokeColor="#374151"
              nodeStrokeWidth={1}
              nodeBorderRadius={4}
              maskColor={settings.maskColor}
              style={{
                backgroundColor: settings.backgroundColor,
                width: '100%',
                height: '100%'
              }}
              pannable={settings.pannable}
              zoomable={settings.zoomable}
              onClick={settings.interactive ? handleViewportClick : undefined}
            />

            {/* Custom overlay for additional features */}
            <div className="absolute inset-0 pointer-events-none">
              {/* Grid overlay */}
              {settings.showGrid && (
                <div
                  className="absolute inset-0 opacity-20"
                  style={{
                    backgroundImage: `
                      linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
                    `,
                    backgroundSize: '20px 20px'
                  }}
                />
              )}

              {/* Node labels */}
              {settings.showLabels && nodes.map(node => (
                <div
                  key={`label-${node.id}`}
                  className="absolute text-xs text-white font-medium pointer-events-none"
                  style={{
                    left: `${((node.position.x - bounds.minX) / (bounds.maxX - bounds.minX)) * 100}%`,
                    top: `${((node.position.y - bounds.minY) / (bounds.maxY - bounds.minY)) * 100}%`,
                    transform: 'translate(-50%, -50%)',
                    textShadow: '0 0 4px rgba(0,0,0,0.8)'
                  }}
                >
                  {node.data?.label || node.id}
                </div>
              ))}
            </div>
          </div>

          {/* Status Bar */}
          <div className="flex items-center justify-between p-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-400">
            <span>{nodes.length} nodes</span>
            <span>{Math.round(viewport.zoom * 100)}%</span>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && renderSettings()}
      </div>
    </div>
  );
};

// Default settings
export const defaultMiniMapSettings: MiniMapSettings = {
  showNodes: true,
  showEdges: true,
  showLabels: false,
  showGrid: false,
  showViewport: true,
  nodeColor: '#6b7280',
  edgeColor: '#4b5563',
  backgroundColor: '#111827',
  viewportColor: '#3b82f6',
  maskColor: 'rgba(0, 0, 0, 0.6)',
  position: 'bottom-right',
  width: 200,
  height: 150,
  zoomable: true,
  pannable: true,
  interactive: true
};