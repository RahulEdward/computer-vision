'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  XMarkIcon,
  BellIcon,
  CpuChipIcon,
  CircleStackIcon,
  ServerIcon,
  WifiIcon,
  FireIcon,
  ClockIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import { PerformanceAlert, PerformanceThresholds } from '../../services/performanceMonitor';

interface PerformanceAlertsProps {
  alerts: PerformanceAlert[];
  onClearAlerts: () => void;
  onUpdateThresholds: (thresholds: Partial<PerformanceThresholds>) => void;
  currentThresholds: PerformanceThresholds;
}

const PerformanceAlerts: React.FC<PerformanceAlertsProps> = ({
  alerts,
  onClearAlerts,
  onUpdateThresholds,
  currentThresholds,
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [thresholds, setThresholds] = useState(currentThresholds);

  const getSeverityColor = (severity: PerformanceAlert['severity']) => {
    switch (severity) {
      case 'critical':
        return 'text-red-300 bg-gradient-to-r from-red-950/40 to-red-900/20 border border-red-800/30 shadow-lg shadow-red-900/20';
      case 'high':
        return 'text-amber-300 bg-gradient-to-r from-amber-950/40 to-amber-900/20 border border-amber-800/30 shadow-lg shadow-amber-900/20';
      case 'medium':
        return 'text-yellow-300 bg-gradient-to-r from-yellow-950/40 to-yellow-900/20 border border-yellow-800/30 shadow-lg shadow-yellow-900/20';
      case 'low':
        return 'text-blue-300 bg-gradient-to-r from-blue-950/40 to-blue-900/20 border border-blue-800/30 shadow-lg shadow-blue-900/20';
      default:
        return 'text-slate-300 bg-gradient-to-r from-slate-800/40 to-slate-700/20 border border-slate-600/30 shadow-lg shadow-slate-900/20';
    }
  };

  const getTypeIcon = (type: PerformanceAlert['type']) => {
    switch (type) {
      case 'cpu':
        return <CpuChipIcon className="w-4 h-4" />;
      case 'memory':
        return <CircleStackIcon className="w-4 h-4" />;
      case 'disk':
        return <ServerIcon className="w-4 h-4" />;
      case 'network':
        return <WifiIcon className="w-4 h-4" />;
      case 'temperature':
        return <FireIcon className="w-4 h-4" />;
      default:
        return <ExclamationTriangleIcon className="w-4 h-4" />;
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) {
      return 'Just now';
    } else if (diff < 3600000) {
      return `${Math.floor(diff / 60000)}m ago`;
    } else if (diff < 86400000) {
      return `${Math.floor(diff / 3600000)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const handleSaveThresholds = () => {
    onUpdateThresholds(thresholds);
    setShowSettings(false);
  };

  const recentAlerts = alerts.slice(-10).reverse(); // Show last 10 alerts, most recent first
  const criticalAlerts = alerts.filter(alert => alert.severity === 'critical').length;
  const highAlerts = alerts.filter(alert => alert.severity === 'high').length;

  return (
    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 shadow-2xl shadow-slate-900/50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-br from-amber-500/20 to-amber-600/10 rounded-lg border border-amber-500/20">
            <BellIcon className="w-5 h-5 text-amber-400" />
          </div>
          <h3 className="text-xl font-bold text-slate-100 tracking-tight">Performance Alerts</h3>
          {(criticalAlerts > 0 || highAlerts > 0) && (
            <div className="flex items-center space-x-2">
              {criticalAlerts > 0 && (
                <span className="px-3 py-1 bg-gradient-to-r from-red-600 to-red-700 text-white text-xs font-medium rounded-full shadow-lg shadow-red-900/30">
                  {criticalAlerts} Critical
                </span>
              )}
              {highAlerts > 0 && (
                <span className="px-3 py-1 bg-gradient-to-r from-amber-600 to-amber-700 text-white text-xs font-medium rounded-full shadow-lg shadow-amber-900/30">
                  {highAlerts} High
                </span>
              )}
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 bg-gradient-to-br from-slate-700/50 to-slate-800/30 hover:from-slate-600/50 hover:to-slate-700/30 text-slate-400 hover:text-slate-200 transition-all duration-200 rounded-lg border border-slate-600/30 shadow-lg"
            title="Alert Settings"
          >
            <AdjustmentsHorizontalIcon className="w-4 h-4" />
          </button>
          {alerts.length > 0 && (
            <button
              onClick={onClearAlerts}
              className="px-4 py-2 bg-gradient-to-r from-slate-700/80 to-slate-800/60 hover:from-slate-600/80 hover:to-slate-700/60 text-slate-300 hover:text-white text-sm font-medium rounded-lg transition-all duration-200 border border-slate-600/30 shadow-lg"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {recentAlerts.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          <div className="p-4 bg-gradient-to-br from-slate-800/30 to-slate-700/20 rounded-xl border border-slate-700/30 inline-block mb-4">
            <BellIcon className="w-12 h-12 mx-auto opacity-60" />
          </div>
          <p className="text-lg font-medium text-slate-300">No alerts</p>
          <p className="text-sm text-slate-500 mt-1">System is running normally</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-80 overflow-y-auto scrollbar-thin scrollbar-track-slate-800 scrollbar-thumb-slate-600">
          <AnimatePresence>
            {recentAlerts.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className={`p-4 rounded-xl backdrop-blur-sm ${getSeverityColor(alert.severity)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 mt-1 p-2 bg-black/20 rounded-lg">
                      {getTypeIcon(alert.type)}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-semibold leading-relaxed">{alert.message}</p>
                      <div className="flex items-center space-x-6 mt-2 text-xs opacity-80">
                        <div className="flex items-center space-x-1.5">
                          <ClockIcon className="w-3 h-3" />
                          <span className="font-medium">{formatTimestamp(alert.timestamp)}</span>
                        </div>
                        <span className="font-medium">
                          Value: <span className="font-bold">{alert.value.toFixed(1)}</span>
                          {alert.type === 'temperature' ? '°C' : 
                           alert.type === 'network' ? 'ms' : '%'}
                        </span>
                        <span className="font-medium">
                          Threshold: <span className="font-bold">{alert.threshold}</span>
                          {alert.type === 'temperature' ? '°C' : 
                           alert.type === 'network' ? 'ms' : '%'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="bg-gradient-to-br from-slate-900/95 to-slate-800/90 backdrop-blur-xl rounded-2xl p-8 max-w-lg w-full mx-4 border border-slate-700/50 shadow-2xl shadow-slate-900/50"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-slate-100 tracking-tight">Alert Thresholds</h3>
                <button
                  onClick={() => setShowSettings(false)}
                  className="p-2 bg-gradient-to-br from-slate-700/50 to-slate-800/30 hover:from-slate-600/50 hover:to-slate-700/30 text-slate-400 hover:text-slate-200 transition-all duration-200 rounded-lg border border-slate-600/30"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-6">
                {/* CPU Thresholds */}
                <div>
                  <label className="block text-sm font-semibold text-slate-200 mb-3">
                    CPU Usage (%)
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Warning</label>
                      <input
                        type="number"
                        value={thresholds.cpu.warning}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          cpu: { ...prev.cpu, warning: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Critical</label>
                      <input
                        type="number"
                        value={thresholds.cpu.critical}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          cpu: { ...prev.cpu, critical: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                  </div>
                </div>

                {/* Memory Thresholds */}
                <div>
                  <label className="block text-sm font-semibold text-slate-200 mb-3">
                    Memory Usage (%)
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Warning</label>
                      <input
                        type="number"
                        value={thresholds.memory.warning}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          memory: { ...prev.memory, warning: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Critical</label>
                      <input
                        type="number"
                        value={thresholds.memory.critical}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          memory: { ...prev.memory, critical: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                  </div>
                </div>

                {/* Disk Thresholds */}
                <div>
                  <label className="block text-sm font-semibold text-slate-200 mb-3">
                    Disk Usage (%)
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Warning</label>
                      <input
                        type="number"
                        value={thresholds.disk.warning}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          disk: { ...prev.disk, warning: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Critical</label>
                      <input
                        type="number"
                        value={thresholds.disk.critical}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          disk: { ...prev.disk, critical: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="100"
                      />
                    </div>
                  </div>
                </div>

                {/* Temperature Thresholds */}
                <div>
                  <label className="block text-sm font-semibold text-slate-200 mb-3">
                    Temperature (°C)
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Warning</label>
                      <input
                        type="number"
                        value={thresholds.temperature.warning}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          temperature: { ...prev.temperature, warning: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="150"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Critical</label>
                      <input
                        type="number"
                        value={thresholds.temperature.critical}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          temperature: { ...prev.temperature, critical: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="150"
                      />
                    </div>
                  </div>
                </div>

                {/* Network Latency Thresholds */}
                <div>
                  <label className="block text-sm font-semibold text-slate-200 mb-3">
                    Network Latency (ms)
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Warning</label>
                      <input
                        type="number"
                        value={thresholds.networkLatency.warning}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          networkLatency: { ...prev.networkLatency, warning: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="5000"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-2">Critical</label>
                      <input
                        type="number"
                        value={thresholds.networkLatency.critical}
                        onChange={(e) => setThresholds(prev => ({
                          ...prev,
                          networkLatency: { ...prev.networkLatency, critical: Number(e.target.value) }
                        }))}
                        className="w-full px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-700/60 border border-slate-600/50 rounded-lg text-slate-100 text-sm focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
                        min="0"
                        max="5000"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-end space-x-4 mt-8 pt-6 border-t border-slate-700/50">
                <button
                  onClick={() => setShowSettings(false)}
                  className="px-6 py-3 bg-gradient-to-r from-slate-700/80 to-slate-800/60 hover:from-slate-600/80 hover:to-slate-700/60 text-slate-300 hover:text-white font-medium rounded-lg transition-all duration-200 border border-slate-600/30"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveThresholds}
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-semibold rounded-lg transition-all duration-200 shadow-lg shadow-blue-900/30"
                >
                  Save Changes
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PerformanceAlerts;