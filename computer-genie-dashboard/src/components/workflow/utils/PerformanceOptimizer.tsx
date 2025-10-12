import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import { Node, Edge, Viewport } from 'reactflow';

export interface PerformanceMetrics {
  renderTime: number;
  nodeCount: number;
  edgeCount: number;
  visibleNodes: number;
  visibleEdges: number;
  memoryUsage: number;
  fps: number;
  lastUpdate: number;
}

export interface ViewportBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OptimizationConfig {
  enableVirtualization: boolean;
  enableLevelOfDetail: boolean;
  enableCulling: boolean;
  enableBatching: boolean;
  maxVisibleNodes: number;
  maxVisibleEdges: number;
  cullingMargin: number;
  lodThresholds: {
    high: number;
    medium: number;
    low: number;
  };
  batchSize: number;
  debounceMs: number;
  enableMetrics: boolean;
}

export interface NodeLOD {
  id: string;
  level: 'high' | 'medium' | 'low' | 'hidden';
  simplified: boolean;
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface EdgeLOD {
  id: string;
  level: 'high' | 'medium' | 'low' | 'hidden';
  simplified: boolean;
  sourceVisible: boolean;
  targetVisible: boolean;
}

interface PerformanceOptimizerProps {
  nodes: Node[];
  edges: Edge[];
  viewport: Viewport;
  config: OptimizationConfig;
  onOptimizedNodesChange: (nodes: Node[], metrics: PerformanceMetrics) => void;
  onOptimizedEdgesChange: (edges: Edge[], metrics: PerformanceMetrics) => void;
  children: React.ReactNode;
}

export const PerformanceOptimizer: React.FC<PerformanceOptimizerProps> = ({
  nodes,
  edges,
  viewport,
  config,
  onOptimizedNodesChange,
  onOptimizedEdgesChange,
  children
}) => {
  const metricsRef = useRef<PerformanceMetrics>({
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    memoryUsage: 0,
    fps: 0,
    lastUpdate: Date.now()
  });

  const frameTimeRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef<number>(performance.now());

  // Calculate viewport bounds
  const viewportBounds = useMemo((): ViewportBounds => {
    const margin = config.cullingMargin;
    return {
      x: -viewport.x / viewport.zoom - margin,
      y: -viewport.y / viewport.zoom - margin,
      width: window.innerWidth / viewport.zoom + margin * 2,
      height: window.innerHeight / viewport.zoom + margin * 2
    };
  }, [viewport, config.cullingMargin]);

  // Check if node is in viewport
  const isNodeInViewport = useCallback((node: Node, bounds: ViewportBounds): boolean => {
    const nodeWidth = node.width || 200;
    const nodeHeight = node.height || 100;
    
    return !(
      node.position.x + nodeWidth < bounds.x ||
      node.position.x > bounds.x + bounds.width ||
      node.position.y + nodeHeight < bounds.y ||
      node.position.y > bounds.y + bounds.height
    );
  }, []);

  // Determine LOD level based on zoom
  const getLODLevel = useCallback((zoom: number): 'high' | 'medium' | 'low' => {
    if (zoom >= config.lodThresholds.high) return 'high';
    if (zoom >= config.lodThresholds.medium) return 'medium';
    return 'low';
  }, [config.lodThresholds]);

  // Create simplified node for LOD
  const createSimplifiedNode = useCallback((node: Node, level: 'medium' | 'low'): Node => {
    const simplified = { ...node };
    
    if (level === 'low') {
      // Minimal representation
      simplified.data = {
        ...node.data,
        simplified: true,
        showDetails: false,
        showHandles: false
      };
    } else if (level === 'medium') {
      // Reduced detail
      simplified.data = {
        ...node.data,
        simplified: true,
        showDetails: false,
        showHandles: true
      };
    }
    
    return simplified;
  }, []);

  // Create simplified edge for LOD
  const createSimplifiedEdge = useCallback((edge: Edge, level: 'medium' | 'low'): Edge => {
    const simplified = { ...edge };
    
    if (level === 'low') {
      simplified.style = {
        ...edge.style,
        strokeWidth: 1,
        stroke: '#6b7280'
      };
      simplified.animated = false;
      simplified.label = undefined;
    } else if (level === 'medium') {
      simplified.style = {
        ...edge.style,
        strokeWidth: 2
      };
      simplified.animated = false;
    }
    
    return simplified;
  }, []);

  // Optimize nodes based on viewport and LOD
  const optimizeNodes = useMemo((): { nodes: Node[]; nodeLODs: NodeLOD[] } => {
    const startTime = performance.now();
    const lodLevel = getLODLevel(viewport.zoom);
    const optimizedNodes: Node[] = [];
    const nodeLODs: NodeLOD[] = [];

    let visibleCount = 0;

    for (const node of nodes) {
      const nodeWidth = node.width || 200;
      const nodeHeight = node.height || 100;
      const bounds = {
        x: node.position.x,
        y: node.position.y,
        width: nodeWidth,
        height: nodeHeight
      };

      const isVisible = config.enableCulling ? isNodeInViewport(node, viewportBounds) : true;
      
      let finalLODLevel: 'high' | 'medium' | 'low' | 'hidden' = lodLevel;
      let optimizedNode = node;

      if (!isVisible) {
        finalLODLevel = 'hidden';
      } else {
        visibleCount++;
        
        // Apply max visible nodes limit
        if (config.maxVisibleNodes > 0 && visibleCount > config.maxVisibleNodes) {
          finalLODLevel = 'hidden';
        } else {
          // Apply LOD simplification
          if (config.enableLevelOfDetail && (lodLevel === 'medium' || lodLevel === 'low')) {
            optimizedNode = createSimplifiedNode(node, lodLevel);
          }
          
          optimizedNodes.push(optimizedNode);
        }
      }

      nodeLODs.push({
        id: node.id,
        level: finalLODLevel,
        simplified: finalLODLevel !== 'high',
        bounds
      });
    }

    const renderTime = performance.now() - startTime;
    
    metricsRef.current = {
      ...metricsRef.current,
      renderTime,
      nodeCount: nodes.length,
      visibleNodes: optimizedNodes.length,
      lastUpdate: Date.now()
    };

    return { nodes: optimizedNodes, nodeLODs };
  }, [
    nodes,
    viewport,
    viewportBounds,
    config,
    getLODLevel,
    isNodeInViewport,
    createSimplifiedNode
  ]);

  // Optimize edges based on visible nodes and LOD
  const optimizeEdges = useMemo((): { edges: Edge[]; edgeLODs: EdgeLOD[] } => {
    const startTime = performance.now();
    const lodLevel = getLODLevel(viewport.zoom);
    const optimizedEdges: Edge[] = [];
    const edgeLODs: EdgeLOD[] = [];
    const visibleNodeIds = new Set(optimizeNodes.nodes.map(n => n.id));

    let visibleCount = 0;

    for (const edge of edges) {
      const sourceVisible = visibleNodeIds.has(edge.source);
      const targetVisible = visibleNodeIds.has(edge.target);
      const isVisible = sourceVisible && targetVisible;

      let finalLODLevel: 'high' | 'medium' | 'low' | 'hidden' = lodLevel;
      let optimizedEdge = edge;

      if (!isVisible) {
        finalLODLevel = 'hidden';
      } else {
        visibleCount++;
        
        // Apply max visible edges limit
        if (config.maxVisibleEdges > 0 && visibleCount > config.maxVisibleEdges) {
          finalLODLevel = 'hidden';
        } else {
          // Apply LOD simplification
          if (config.enableLevelOfDetail && (lodLevel === 'medium' || lodLevel === 'low')) {
            optimizedEdge = createSimplifiedEdge(edge, lodLevel);
          }
          
          optimizedEdges.push(optimizedEdge);
        }
      }

      edgeLODs.push({
        id: edge.id,
        level: finalLODLevel,
        simplified: finalLODLevel !== 'high',
        sourceVisible,
        targetVisible
      });
    }

    const renderTime = performance.now() - startTime;
    
    metricsRef.current = {
      ...metricsRef.current,
      edgeCount: edges.length,
      visibleEdges: optimizedEdges.length,
      renderTime: metricsRef.current.renderTime + renderTime
    };

    return { edges: optimizedEdges, edgeLODs };
  }, [
    edges,
    viewport,
    optimizeNodes.nodes,
    config,
    getLODLevel,
    createSimplifiedEdge
  ]);

  // Calculate FPS
  useEffect(() => {
    const updateFPS = () => {
      const now = performance.now();
      const deltaTime = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;

      frameTimeRef.current.push(deltaTime);
      if (frameTimeRef.current.length > 60) {
        frameTimeRef.current.shift();
      }

      const avgFrameTime = frameTimeRef.current.reduce((a, b) => a + b, 0) / frameTimeRef.current.length;
      const fps = 1000 / avgFrameTime;

      metricsRef.current.fps = Math.round(fps);
    };

    const intervalId = setInterval(updateFPS, 100);
    return () => clearInterval(intervalId);
  }, []);

  // Memory usage estimation
  useEffect(() => {
    const estimateMemoryUsage = () => {
      const nodeMemory = optimizeNodes.nodes.length * 1024; // ~1KB per node
      const edgeMemory = optimizeEdges.edges.length * 512; // ~0.5KB per edge
      metricsRef.current.memoryUsage = nodeMemory + edgeMemory;
    };

    estimateMemoryUsage();
  }, [optimizeNodes.nodes.length, optimizeEdges.edges.length]);

  // Debounced updates
  const debouncedUpdateRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (debouncedUpdateRef.current) {
      clearTimeout(debouncedUpdateRef.current);
    }

    debouncedUpdateRef.current = setTimeout(() => {
      onOptimizedNodesChange(optimizeNodes.nodes, metricsRef.current);
      onOptimizedEdgesChange(optimizeEdges.edges, metricsRef.current);
    }, config.debounceMs);

    return () => {
      if (debouncedUpdateRef.current) {
        clearTimeout(debouncedUpdateRef.current);
      }
    };
  }, [optimizeNodes.nodes, optimizeEdges.edges, config.debounceMs, onOptimizedNodesChange, onOptimizedEdgesChange]);

  return <>{children}</>;
};

// Hook for performance optimization
export const usePerformanceOptimization = (config: OptimizationConfig) => {
  const [metrics, setMetrics] = React.useState<PerformanceMetrics>({
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    memoryUsage: 0,
    fps: 0,
    lastUpdate: Date.now()
  });

  const updateMetrics = useCallback((newMetrics: PerformanceMetrics) => {
    setMetrics(newMetrics);
  }, []);

  const getOptimizationRecommendations = useCallback((metrics: PerformanceMetrics): string[] => {
    const recommendations: string[] = [];

    if (metrics.fps < 30) {
      recommendations.push('Consider enabling virtualization to improve FPS');
    }

    if (metrics.visibleNodes > 1000) {
      recommendations.push('Too many visible nodes, enable culling or reduce max visible nodes');
    }

    if (metrics.visibleEdges > 2000) {
      recommendations.push('Too many visible edges, enable edge culling');
    }

    if (metrics.memoryUsage > 50 * 1024 * 1024) { // 50MB
      recommendations.push('High memory usage, consider enabling LOD');
    }

    if (metrics.renderTime > 16) { // 60fps = 16ms per frame
      recommendations.push('Render time too high, enable batching or LOD');
    }

    return recommendations;
  }, []);

  return {
    metrics,
    updateMetrics,
    getOptimizationRecommendations
  };
};

// Performance monitoring component
export const PerformanceMonitor: React.FC<{
  metrics: PerformanceMetrics;
  config: OptimizationConfig;
  onConfigChange: (config: Partial<OptimizationConfig>) => void;
}> = ({ metrics, config, onConfigChange }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getPerformanceStatus = (): 'good' | 'warning' | 'critical' => {
    if (metrics.fps < 20 || metrics.renderTime > 32) return 'critical';
    if (metrics.fps < 40 || metrics.renderTime > 16) return 'warning';
    return 'good';
  };

  const status = getPerformanceStatus();

  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50">
      <div
        className="flex items-center justify-between p-3 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            status === 'good' ? 'bg-green-400' :
            status === 'warning' ? 'bg-yellow-400' :
            'bg-red-400'
          }`} />
          <span className="text-sm font-medium text-white">Performance</span>
        </div>
        <div className="text-xs text-gray-400">
          {metrics.fps} FPS
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-gray-700 p-3 space-y-2">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-400">Nodes:</span>
              <span className="text-white ml-1">{metrics.visibleNodes}/{metrics.nodeCount}</span>
            </div>
            <div>
              <span className="text-gray-400">Edges:</span>
              <span className="text-white ml-1">{metrics.visibleEdges}/{metrics.edgeCount}</span>
            </div>
            <div>
              <span className="text-gray-400">Render:</span>
              <span className="text-white ml-1">{metrics.renderTime.toFixed(1)}ms</span>
            </div>
            <div>
              <span className="text-gray-400">Memory:</span>
              <span className="text-white ml-1">{formatBytes(metrics.memoryUsage)}</span>
            </div>
          </div>

          <div className="space-y-1">
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={config.enableVirtualization}
                onChange={(e) => onConfigChange({ enableVirtualization: e.target.checked })}
                className="rounded"
              />
              <span className="text-gray-300">Virtualization</span>
            </label>
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={config.enableLevelOfDetail}
                onChange={(e) => onConfigChange({ enableLevelOfDetail: e.target.checked })}
                className="rounded"
              />
              <span className="text-gray-300">Level of Detail</span>
            </label>
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={config.enableCulling}
                onChange={(e) => onConfigChange({ enableCulling: e.target.checked })}
                className="rounded"
              />
              <span className="text-gray-300">Culling</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

// Default configuration
export const defaultOptimizationConfig: OptimizationConfig = {
  enableVirtualization: true,
  enableLevelOfDetail: true,
  enableCulling: true,
  enableBatching: true,
  maxVisibleNodes: 500,
  maxVisibleEdges: 1000,
  cullingMargin: 100,
  lodThresholds: {
    high: 1.0,
    medium: 0.5,
    low: 0.25
  },
  batchSize: 50,
  debounceMs: 100,
  enableMetrics: true
};