import { useState, useEffect, useCallback, useMemo } from 'react';
import { websocketService, SystemMetrics } from '../services/websocket';
import { performanceMonitor, PerformanceMetrics, PerformanceAlert } from '../services/performanceMonitor';

interface SystemMetricsState {
  currentMetrics: SystemMetrics | null;
  historicalMetrics: SystemMetrics[];
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  error: string | null;
  performanceMetrics: PerformanceMetrics | null;
  performanceHistory: PerformanceMetrics[];
  alerts: PerformanceAlert[];
  isMonitoring: boolean;
}

const HISTORY_LIMIT = 100;

export const useSystemMetrics = () => {
  const [state, setState] = useState<SystemMetricsState>({
    currentMetrics: null,
    historicalMetrics: [],
    connectionStatus: 'disconnected',
    error: null,
    performanceMetrics: null,
    performanceHistory: [],
    alerts: [],
    isMonitoring: false,
  });

  const updateMetrics = useCallback((metrics: SystemMetrics) => {
    setState(prev => ({
      ...prev,
      currentMetrics: metrics,
      historicalMetrics: [...prev.historicalMetrics.slice(-(HISTORY_LIMIT - 1)), metrics],
      connectionStatus: 'connected',
      error: null,
    }));
  }, []);

  const handleConnectionError = useCallback((error: string) => {
    setState(prev => ({
      ...prev,
      connectionStatus: 'error',
      error,
    }));
  }, []);

  useEffect(() => {
    // Subscribe to system metrics
    const unsubscribe = websocketService.subscribe<SystemMetrics>('system_metrics', updateMetrics);

    // Check connection status
    const checkConnection = () => {
      if (!websocketService.isConnected()) {
        handleConnectionError('WebSocket disconnected');
      }
    };

    const connectionInterval = setInterval(checkConnection, 5000);

    // Try to connect if not already connected
    if (!websocketService.isConnected()) {
      websocketService.connect().catch((error) => {
        handleConnectionError(`Connection failed: ${error.message}`);
      });
    }

    return () => {
      unsubscribe();
      clearInterval(connectionInterval);
    };
  }, [updateMetrics, handleConnectionError]);

  // Generate mock data if WebSocket is not available (for development)
  useEffect(() => {
    if (!websocketService.isConnected()) {
      const mockInterval = setInterval(() => {
        const mockMetrics: SystemMetrics = {
          cpu: Math.random() * 100,
          memory: Math.random() * 100,
          disk: Math.random() * 100,
          network: {
            upload: Math.random() * 1000,
            download: Math.random() * 5000,
          },
          timestamp: Date.now(),
        };
        updateMetrics(mockMetrics);
      }, 2000);

      return () => clearInterval(mockInterval);
    }
  }, [updateMetrics, state.isConnected]);

  // Calculate average metrics from recent history
  const averageMetrics = useMemo(() => {
    if (state.historicalMetrics.length === 0) return null;
    
    const recent = state.historicalMetrics.slice(-10); // Last 10 measurements
    const sum = recent.reduce((acc, metrics) => ({
      cpu: acc.cpu + metrics.cpu,
      memory: acc.memory + metrics.memory,
      disk: acc.disk + metrics.disk,
      network: {
        download: acc.network.download + metrics.network.download,
        upload: acc.network.upload + metrics.network.upload,
        latency: acc.network.latency + metrics.network.latency,
      },
    }), {
      cpu: 0,
      memory: 0,
      disk: 0,
      network: { download: 0, upload: 0, latency: 0 },
    });

    return {
      cpu: Math.round((sum.cpu / recent.length) * 100) / 100,
      memory: Math.round((sum.memory / recent.length) * 100) / 100,
      disk: Math.round((sum.disk / recent.length) * 100) / 100,
      network: {
        download: Math.round((sum.network.download / recent.length) * 100) / 100,
        upload: Math.round((sum.network.upload / recent.length) * 100) / 100,
        latency: Math.round((sum.network.latency / recent.length) * 100) / 100,
      },
    };
  }, [state.historicalMetrics]);

  // Calculate trends (positive = increasing, negative = decreasing)
  const trends = useMemo(() => {
    if (state.historicalMetrics.length < 2) return null;
    
    const recent = state.historicalMetrics.slice(-5); // Last 5 measurements
    const older = state.historicalMetrics.slice(-10, -5); // Previous 5 measurements
    
    if (older.length === 0) return null;

    const recentAvg = recent.reduce((acc, m) => ({
      cpu: acc.cpu + m.cpu,
      memory: acc.memory + m.memory,
      disk: acc.disk + m.disk,
    }), { cpu: 0, memory: 0, disk: 0 });

    const olderAvg = older.reduce((acc, m) => ({
      cpu: acc.cpu + m.cpu,
      memory: acc.memory + m.memory,
      disk: acc.disk + m.disk,
    }), { cpu: 0, memory: 0, disk: 0 });

    return {
      cpu: Math.round(((recentAvg.cpu / recent.length) - (olderAvg.cpu / older.length)) * 100) / 100,
      memory: Math.round(((recentAvg.memory / recent.length) - (olderAvg.memory / older.length)) * 100) / 100,
      disk: Math.round(((recentAvg.disk / recent.length) - (olderAvg.disk / older.length)) * 100) / 100,
    };
  }, [state.historicalMetrics]);

  // Utility functions
  const clearAlerts = useCallback(() => {
    performanceMonitor.clearAlerts();
    setState(prev => ({ ...prev, alerts: [] }));
  }, []);

  const exportMetrics = useCallback((format: 'json' | 'csv' = 'json') => {
    return performanceMonitor.exportMetrics(format);
  }, []);

  const updateThresholds = useCallback((thresholds: any) => {
    performanceMonitor.updateThresholds(thresholds);
  }, []);

  return {
    // Legacy compatibility
    currentMetrics: state.currentMetrics,
    historicalMetrics: state.historicalMetrics,
    connectionStatus: state.connectionStatus,
    error: state.error,
    averageMetrics,
    trends,
    
    // Enhanced performance monitoring
    performanceMetrics: state.performanceMetrics,
    performanceHistory: state.performanceHistory,
    alerts: state.alerts,
    isMonitoring: state.isMonitoring,
    
    // Utility functions
    clearAlerts,
    exportMetrics,
    updateThresholds,
    getThresholds: () => performanceMonitor.getThresholds(),
  };
};