'use client';

import React, { useEffect, useState, Suspense } from 'react';
import { useDashboardStore } from '@/lib/store';
import { useCollaboration } from '@/lib/webrtc-collaboration';
import { useVoiceControl } from '@/lib/voice-control';
import { usePerformanceMonitor } from '@/lib/performance-monitor';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';

// Lazy load heavy components
const WorkflowBuilder = React.lazy(() => import('../workflow/WorkflowBuilder'));
const MonacoEditor = React.lazy(() => import('../editor/MonacoEditor'));
const ThreeVisualization = React.lazy(() => import('../3d/ThreeVisualization'));
const ARPreview = React.lazy(() => import('../ar/ARPreview'));
const SemanticSearch = React.lazy(() => import('../search/SemanticSearch'));
const CollaborationPanel = React.lazy(() => import('../collaboration/CollaborationPanel'));

interface MainDashboardProps {
  roomId?: string;
}

export default function MainDashboard({ roomId = 'default-room' }: MainDashboardProps) {
  const {
    sidebarOpen,
    theme,
    layout,
    voiceEnabled,
    listening,
    arEnabled,
    threeDView,
    toggleSidebar,
    setTheme,
    recordInteraction
  } = useDashboardStore();

  const collaboration = useCollaboration(roomId);
  const { startListening, stopListening, speak, isSupported: voiceSupported } = useVoiceControl();
  const { metrics, score, measureInteraction } = usePerformanceMonitor();

  const [activeTab, setActiveTab] = useState('workflow');
  const [showPerformancePanel, setShowPerformancePanel] = useState(false);

  // Record interactions for performance monitoring
  const handleInteraction = (action: () => void) => {
    measureInteraction(() => {
      recordInteraction();
      action();
    });
  };

  // Voice control toggle
  const toggleVoiceListening = () => {
    if (listening) {
      stopListening();
    } else {
      startListening();
    }
  };

  // Theme is now handled by ThemeProvider in layout.tsx

  // Adaptive UI based on user behavior
  useEffect(() => {
    const handleUserActivity = () => {
      recordInteraction();
    };

    window.addEventListener('click', handleUserActivity);
    window.addEventListener('keydown', handleUserActivity);
    window.addEventListener('scroll', handleUserActivity);

    return () => {
      window.removeEventListener('click', handleUserActivity);
      window.removeEventListener('keydown', handleUserActivity);
      window.removeEventListener('scroll', handleUserActivity);
    };
  }, [recordInteraction]);

  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 ${theme}`}>
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 sticky top-0 z-50">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => handleInteraction(toggleSidebar)}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Computer Genie Dashboard
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            {/* Performance Score */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${score > 80 ? 'bg-green-500' : score > 60 ? 'bg-yellow-500' : 'bg-red-500'}`} />
              <span className="text-sm font-medium">{score}%</span>
              <button
                onClick={() => setShowPerformancePanel(!showPerformancePanel)}
                className="text-xs text-slate-500 hover:text-slate-700"
              >
                Performance
              </button>
            </div>

            {/* Theme Toggle */}
            <button
              onClick={() => {
                const newTheme = theme === 'dark' ? 'light' : 'dark';
                console.log('🎨 Theme Toggle Clicked!');
                console.log('Current theme:', theme);
                console.log('New theme:', newTheme);
                setTheme(newTheme);
                recordInteraction();
              }}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors border-2 border-slate-300 dark:border-slate-600"
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {theme === 'dark' ? (
                <svg className="w-6 h-6 text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.166a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.061l1.591-1.59zM21.75 12a.75.75 0 01-.75.75h-2.25a.75.75 0 010-1.5H21a.75.75 0 01.75.75zM17.834 18.894a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 10-1.061 1.06l1.59 1.591zM12 18a.75.75 0 01.75.75V21a.75.75 0 01-1.5 0v-2.25A.75.75 0 0112 18zM7.758 17.303a.75.75 0 00-1.061-1.06l-1.591 1.59a.75.75 0 001.06 1.061l1.591-1.59zM6 12a.75.75 0 01-.75.75H3a.75.75 0 010-1.5h2.25A.75.75 0 016 12zM6.697 7.757a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 00-1.061 1.06l1.59 1.591z" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-slate-700" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M9.528 1.718a.75.75 0 01.162.819A8.97 8.97 0 009 6a9 9 0 009 9 8.97 8.97 0 003.463-.69.75.75 0 01.981.98 10.503 10.503 0 01-9.694 6.46c-5.799 0-10.5-4.701-10.5-10.5 0-4.368 2.667-8.112 6.46-9.694a.75.75 0 01.818.162z" clipRule="evenodd" />
                </svg>
              )}
            </button>

            {/* Voice Control */}
            {voiceSupported && (
              <button
                onClick={toggleVoiceListening}
                className={`p-2 rounded-lg transition-colors ${
                  listening 
                    ? 'bg-red-500 text-white animate-pulse' 
                    : 'hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
                title={listening ? 'Stop Listening' : 'Start Voice Control'}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </button>
            )}

            {/* 3D Toggle */}
            <button
              onClick={() => handleInteraction(() => useDashboardStore.getState().toggle3D())}
              className={`p-2 rounded-lg transition-colors ${
                threeDView 
                  ? 'bg-blue-500 text-white' 
                  : 'hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </button>

            {/* AR Toggle */}
            <button
              onClick={() => handleInteraction(() => useDashboardStore.getState().toggleAR())}
              className={`p-2 rounded-lg transition-colors ${
                arEnabled 
                  ? 'bg-purple-500 text-white' 
                  : 'hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.aside
              initial={{ x: -300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -300, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="w-80 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm border-r border-slate-200 dark:border-slate-700 overflow-y-auto"
            >
              <div className="p-6">
                <Suspense fallback={<div className="animate-pulse bg-slate-200 h-32 rounded-lg" />}>
                  <SemanticSearch />
                </Suspense>
                
                <div className="mt-6">
                  <Suspense fallback={<div className="animate-pulse bg-slate-200 h-48 rounded-lg" />}>
                    <CollaborationPanel />
                  </Suspense>
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <main className="flex-1 overflow-hidden">
          {/* Tab Navigation */}
          <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'workflow', label: 'Workflow Builder', icon: '🔧' },
                { id: 'editor', label: 'Code Editor', icon: '💻' },
                { id: 'visualization', label: '3D View', icon: '🎯' },
                { id: 'ar', label: 'AR Preview', icon: '🥽' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => handleInteraction(() => setActiveTab(tab.id))}
                  className={`flex items-center space-x-2 py-4 px-2 border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span className="font-medium">{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="h-full p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
                className="h-full"
              >
                {activeTab === 'workflow' && (
                  <Suspense fallback={<div className="animate-pulse bg-slate-200 h-full rounded-lg" />}>
                    <WorkflowBuilder />
                  </Suspense>
                )}

                {activeTab === 'editor' && (
                  <Suspense fallback={<div className="animate-pulse bg-slate-200 h-full rounded-lg" />}>
                    <MonacoEditor />
                  </Suspense>
                )}

                {activeTab === 'visualization' && threeDView && (
                  <div className="h-full bg-slate-900 rounded-lg overflow-hidden">
                    <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                      <ambientLight intensity={0.5} />
                      <pointLight position={[10, 10, 10]} />
                      <Suspense fallback={null}>
                        <ThreeVisualization />
                        <Environment preset="city" />
                      </Suspense>
                      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                    </Canvas>
                  </div>
                )}

                {activeTab === 'ar' && arEnabled && (
                  <Suspense fallback={<div className="animate-pulse bg-slate-200 h-full rounded-lg" />}>
                    <ARPreview />
                  </Suspense>
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>

      {/* Performance Panel */}
      <AnimatePresence>
        {showPerformancePanel && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="fixed right-0 top-20 bottom-0 w-80 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border-l border-slate-200 dark:border-slate-700 p-6 overflow-y-auto z-40"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold">Performance Monitor</h3>
              <button
                onClick={() => setShowPerformancePanel(false)}
                className="p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg">
                <div className="text-sm text-slate-600 dark:text-slate-400">Overall Score</div>
                <div className={`text-2xl font-bold ${score > 80 ? 'text-green-600' : score > 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {score}%
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm">Interaction Latency</span>
                  <span className="text-sm font-mono">{metrics.interactionLatency.toFixed(1)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">FPS</span>
                  <span className="text-sm font-mono">{metrics.fps.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Memory Usage</span>
                  <span className="text-sm font-mono">{(metrics.memoryUsage * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Network Latency</span>
                  <span className="text-sm font-mono">{metrics.networkLatency.toFixed(1)}ms</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Status Indicator */}
      {listening && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-full shadow-lg animate-pulse">
          🎤 Listening...
        </div>
      )}
    </div>
  );
}