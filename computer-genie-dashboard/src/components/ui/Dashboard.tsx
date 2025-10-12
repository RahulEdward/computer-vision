'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  CpuChipIcon,
  ServerIcon,
  ClockIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  DocumentTextIcon,
  UserGroupIcon,
  BellIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  CircleStackIcon,
  CloudIcon,
  BoltIcon,
  WifiIcon,
  SignalIcon,
  SignalSlashIcon,
  FolderIcon,
  CogIcon,
  CodeBracketIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline';
import { useSystemMetrics } from '../../hooks/useSystemMetrics';
import FileManager from './FileManager';
import PerformanceAlerts from './PerformanceAlerts';
import { SavedWorkflow, SavedScript } from '../../services/fileSystem';

interface MetricCard {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down' | 'stable';
  icon: React.ComponentType<any>;
}

const Dashboard = () => {
  const {
    currentMetrics,
    connectionStatus,
    error,
    averageMetrics,
    trends,
    alerts,
    clearAlerts,
    updateThresholds,
    getThresholds
  } = useSystemMetrics();
  
  const [activeTab, setActiveTab] = useState('dashboard');
  const [activities, setActivities] = useState([
    { id: 1, action: 'System monitoring started', time: 'Just now', status: 'info' },
  ]);

  const [showFileManager, setShowFileManager] = useState(false);

  const handleWorkflowSelect = (workflow: SavedWorkflow) => {
    console.log('Selected workflow:', workflow);
    // TODO: Load workflow into workflow builder
    setShowFileManager(false);
  };

  const handleScriptSelect = (script: SavedScript) => {
    console.log('Selected script:', script);
    // TODO: Load script into code editor
    setShowFileManager(false);
  };

  const handleCreateNew = (type: 'workflow' | 'script') => {
    console.log('Create new:', type);
    // TODO: Navigate to appropriate editor
    setShowFileManager(false);
  };

  const metricCards: MetricCard[] = currentMetrics ? [
    { 
      title: 'CPU Usage', 
      value: `${Math.round(currentMetrics.cpu)}%`, 
      change: trends?.cpu ? `${trends.cpu > 0 ? '+' : ''}${trends.cpu.toFixed(1)}%` : '0%',
      trend: (trends?.cpu ?? 0) > 0 ? 'up' : (trends?.cpu ?? 0) < 0 ? 'down' : 'stable',
      icon: CpuChipIcon 
    },
    { 
      title: 'Memory Usage', 
      value: `${Math.round(currentMetrics.memory)}%`, 
      change: trends?.memory ? `${trends.memory > 0 ? '+' : ''}${trends.memory.toFixed(1)}%` : '0%',
      trend: (trends?.memory ?? 0) > 0 ? 'up' : (trends?.memory ?? 0) < 0 ? 'down' : 'stable',
      icon: ServerIcon 
    },
    { 
      title: 'Disk Usage', 
      value: `${Math.round(currentMetrics.disk)}%`, 
      change: trends?.disk ? `${trends.disk > 0 ? '+' : ''}${trends.disk.toFixed(1)}%` : '0%',
      trend: (trends?.disk ?? 0) > 0 ? 'up' : (trends?.disk ?? 0) < 0 ? 'down' : 'stable',
      icon: ChartBarIcon 
    },
    { 
      title: 'Network', 
      value: `${((currentMetrics.network.download + currentMetrics.network.upload) / 1000).toFixed(1)}MB/s`, 
      change: '0MB/s',
      trend: 'stable',
      icon: WifiIcon 
    },
  ] : [
    { title: 'CPU Usage', value: '--', change: '--', trend: 'stable', icon: CpuChipIcon },
    { title: 'Memory Usage', value: '--', change: '--', trend: 'stable', icon: ServerIcon },
    { title: 'Disk Usage', value: '--', change: '--', trend: 'stable', icon: ChartBarIcon },
    { title: 'Network', value: '--', change: '--', trend: 'stable', icon: WifiIcon },
  ];

  // Update activities based on metrics
  useEffect(() => {
    if (currentMetrics) {
      const newActivities: Array<{ id: number; action: string; time: string; status: string }> = [];
      
      if (currentMetrics.cpu > 80) {
        newActivities.push({
          id: Date.now() + 1,
          action: `High CPU usage detected: ${currentMetrics.cpu.toFixed(1)}%`,
          time: 'Just now',
          status: 'warning'
        });
      }
      
      if (currentMetrics.memory > 85) {
        newActivities.push({
          id: Date.now() + 2,
          action: `High memory usage: ${currentMetrics.memory.toFixed(1)}%`,
          time: 'Just now',
          status: 'error'
        });
      }
      
      const totalNetwork = currentMetrics.network.upload + currentMetrics.network.download;
      if (totalNetwork > 50000) { // 50MB/s
        newActivities.push({
          id: Date.now() + 3,
          action: `Network activity spike: ${(totalNetwork / 1000).toFixed(1)} MB/s`,
          time: 'Just now',
          status: 'info'
        });
      }
      
      if (newActivities.length > 0) {
        setActivities(prev => [...newActivities, ...prev.slice(0, 7)]);
      }
    }
  }, [currentMetrics]);

  const isConnected = connectionStatus === 'connected';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-800/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                  <CogIcon className="w-5 h-5 text-white" />
                </div>
                <h1 className="text-xl font-semibold text-slate-100">Computer Genie</h1>
              </div>
              <div className="hidden md:flex items-center space-x-1 ml-8">
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    activeTab === 'dashboard'
                      ? 'bg-slate-800 text-slate-100 shadow-lg'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('workflows')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    activeTab === 'workflows'
                      ? 'bg-slate-800 text-slate-100 shadow-lg'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Workflows
                </button>
                <button
                  onClick={() => setActiveTab('scripts')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    activeTab === 'scripts'
                      ? 'bg-slate-800 text-slate-100 shadow-lg'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Scripts
                </button>
                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    activeTab === 'analytics'
                      ? 'bg-slate-800 text-slate-100 shadow-lg'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Analytics
                </button>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowFileManager(true)}
                className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 rounded-lg transition-all duration-200"
                title="File Manager"
              >
                <FolderIcon className="w-5 h-5" />
              </button>
              <div className="flex items-center space-x-2 px-3 py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-emerald-400' :
                  connectionStatus === 'connecting' ? 'bg-amber-400' :
                  'bg-red-400'
                }`} />
                <span className="text-xs font-medium text-slate-300 capitalize">
                  {connectionStatus}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            {/* Performance Alerts */}
            <PerformanceAlerts
              alerts={alerts}
              onClearAlerts={clearAlerts}
              onUpdateThresholds={updateThresholds}
              currentThresholds={getThresholds()}
            />

            {/* System Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6 hover:border-slate-700/50 transition-all duration-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">CPU Usage</p>
                    <p className="text-2xl font-bold text-slate-100 mt-1">
                      {currentMetrics?.cpu.toFixed(1) || '0.0'}%
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 rounded-xl flex items-center justify-center">
                    <CpuChipIcon className="w-6 h-6 text-blue-400" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${currentMetrics?.cpu || 0}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6 hover:border-slate-700/50 transition-all duration-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">Memory</p>
                    <p className="text-2xl font-bold text-slate-100 mt-1">
                      {currentMetrics?.memory.toFixed(1) || '0.0'}%
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-violet-600/20 rounded-xl flex items-center justify-center">
                    <CircleStackIcon className="w-6 h-6 text-purple-400" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-purple-500 to-violet-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${currentMetrics?.memory || 0}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6 hover:border-slate-700/50 transition-all duration-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">Disk Usage</p>
                    <p className="text-2xl font-bold text-slate-100 mt-1">
                      {currentMetrics?.disk.toFixed(1) || '0.0'}%
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500/20 to-teal-600/20 rounded-xl flex items-center justify-center">
                    <ServerIcon className="w-6 h-6 text-emerald-400" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-emerald-500 to-teal-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${currentMetrics?.disk || 0}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6 hover:border-slate-700/50 transition-all duration-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">Network</p>
                    <p className="text-2xl font-bold text-slate-100 mt-1">
                      {currentMetrics ? Math.round((currentMetrics.network.upload + currentMetrics.network.download) / 1024) : '0'}KB/s
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-xl flex items-center justify-center">
                    <GlobeAltIcon className="w-6 h-6 text-cyan-400" />
                  </div>
                </div>
                <div className="mt-4 flex items-center space-x-4 text-xs">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full" />
                    <span className="text-slate-400">↓ {currentMetrics?.network.download.toFixed(1) || '0.0'} MB/s</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                    <span className="text-slate-400">↑ {currentMetrics?.network.upload.toFixed(1) || '0.0'} MB/s</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts and Analytics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-slate-100 mb-4">System Performance</h3>
                <div className="h-64 flex items-center justify-center text-slate-400">
                  <div className="text-center">
                    <ChartBarIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Performance charts will be displayed here</p>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-slate-100 mb-4">Recent Activity</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 p-3 bg-slate-800/30 rounded-lg border border-slate-700/30">
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-slate-200">Workflow executed successfully</p>
                      <p className="text-xs text-slate-400">2 minutes ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-slate-800/30 rounded-lg border border-slate-700/30">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-slate-200">System health check completed</p>
                      <p className="text-xs text-slate-400">5 minutes ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-slate-800/30 rounded-lg border border-slate-700/30">
                    <div className="w-2 h-2 bg-purple-400 rounded-full" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-slate-200">New script uploaded</p>
                      <p className="text-xs text-slate-400">10 minutes ago</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-slate-100 mb-6">Quick Actions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <button
                  onClick={() => setActiveTab('workflows')}
                  className="p-4 bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50 hover:border-slate-600/50 rounded-xl transition-all duration-200 group"
                >
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-indigo-600/20 rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200">
                    <CogIcon className="w-5 h-5 text-blue-400" />
                  </div>
                  <h4 className="font-medium text-slate-200 mb-1">Create Workflow</h4>
                  <p className="text-xs text-slate-400">Build automation workflows</p>
                </button>

                <button
                  onClick={() => setActiveTab('scripts')}
                  className="p-4 bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50 hover:border-slate-600/50 rounded-xl transition-all duration-200 group"
                >
                  <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-violet-600/20 rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200">
                    <CodeBracketIcon className="w-5 h-5 text-purple-400" />
                  </div>
                  <h4 className="font-medium text-slate-200 mb-1">Write Script</h4>
                  <p className="text-xs text-slate-400">Create custom scripts</p>
                </button>

                <button
                  onClick={() => setShowFileManager(true)}
                  className="p-4 bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50 hover:border-slate-600/50 rounded-xl transition-all duration-200 group"
                >
                  <div className="w-10 h-10 bg-gradient-to-br from-emerald-500/20 to-teal-600/20 rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200">
                    <FolderIcon className="w-5 h-5 text-emerald-400" />
                  </div>
                  <h4 className="font-medium text-slate-200 mb-1">Manage Files</h4>
                  <p className="text-xs text-slate-400">Organize your projects</p>
                </button>

                <button
                  onClick={() => setActiveTab('analytics')}
                  className="p-4 bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50 hover:border-slate-600/50 rounded-xl transition-all duration-200 group"
                >
                  <div className="w-10 h-10 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200">
                    <ChartBarIcon className="w-5 h-5 text-cyan-400" />
                  </div>
                  <h4 className="font-medium text-slate-200 mb-1">View Analytics</h4>
                  <p className="text-xs text-slate-400">Performance insights</p>
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* File Manager Modal */}
      {showFileManager && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-lg w-full max-w-4xl h-3/4 max-h-screen overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="text-xl font-bold text-white">File Manager</h2>
              <button
                onClick={() => setShowFileManager(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <FileManager
              onWorkflowSelect={handleWorkflowSelect}
              onScriptSelect={handleScriptSelect}
              onCreateNew={handleCreateNew}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;