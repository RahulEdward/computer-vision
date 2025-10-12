'use client';

import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/lib/store';

export default function ARPreview() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isARActive, setIsARActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [arObjects, setArObjects] = useState<Array<{
    id: string;
    type: 'workflow' | 'node' | 'data';
    position: { x: number; y: number; z: number };
    label: string;
    color: string;
  }>>([]);

  const { recordInteraction } = useDashboardStore();

  useEffect(() => {
    // Get available cameras
    navigator.mediaDevices.enumerateDevices()
      .then(deviceList => {
        const cameras = deviceList.filter(device => device.kind === 'videoinput');
        setDevices(cameras);
        if (cameras.length > 0) {
          setSelectedDevice(cameras[0].deviceId);
        }
      })
      .catch(err => {
        console.error('Error enumerating devices:', err);
        setError('Could not access camera devices');
      });

    // Initialize AR objects
    setArObjects([
      {
        id: '1',
        type: 'workflow',
        position: { x: 0.2, y: 0.3, z: 0 },
        label: 'User Registration Flow',
        color: '#3b82f6'
      },
      {
        id: '2',
        type: 'node',
        position: { x: 0.7, y: 0.4, z: 0 },
        label: 'Email Sender',
        color: '#10b981'
      },
      {
        id: '3',
        type: 'data',
        position: { x: 0.5, y: 0.6, z: 0 },
        label: 'User Data',
        color: '#8b5cf6'
      }
    ]);
  }, []);

  const startAR = async () => {
    try {
      recordInteraction();
      setError(null);

      const constraints = {
        video: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: devices.length > 1 ? 'environment' : 'user'
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsARActive(true);
        
        // Start AR rendering loop
        requestAnimationFrame(renderAR);
      }
    } catch (err) {
      console.error('Error starting AR:', err);
      setError('Could not start camera. Please check permissions.');
    }
  };

  const stopAR = () => {
    recordInteraction();
    
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    setIsARActive(false);
  };

  const renderAR = () => {
    if (!isARActive || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw AR objects
    arObjects.forEach(obj => {
      const x = obj.position.x * canvas.width;
      const y = obj.position.y * canvas.height;
      
      // Draw object
      ctx.save();
      
      // Object shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      
      // Object shape based on type
      ctx.fillStyle = obj.color;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      
      switch (obj.type) {
        case 'workflow':
          // Draw rounded rectangle
          const width = 120;
          const height = 60;
          ctx.beginPath();
          ctx.roundRect(x - width/2, y - height/2, width, height, 10);
          ctx.fill();
          ctx.stroke();
          break;
          
        case 'node':
          // Draw circle
          ctx.beginPath();
          ctx.arc(x, y, 30, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          break;
          
        case 'data':
          // Draw diamond
          ctx.beginPath();
          ctx.moveTo(x, y - 25);
          ctx.lineTo(x + 25, y);
          ctx.lineTo(x, y + 25);
          ctx.lineTo(x - 25, y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          break;
      }
      
      ctx.restore();
      
      // Draw label
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 3;
      ctx.font = '14px Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Text background
      const textMetrics = ctx.measureText(obj.label);
      const textWidth = textMetrics.width + 10;
      const textHeight = 20;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x - textWidth/2, y + 40 - textHeight/2, textWidth, textHeight);
      
      // Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(obj.label, x, y + 40);
      
      // Connection lines (simple example)
      if (obj.id === '1') {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x + 60, y);
        
        const targetObj = arObjects.find(o => o.id === '2');
        if (targetObj) {
          const targetX = targetObj.position.x * canvas.width;
          const targetY = targetObj.position.y * canvas.height;
          ctx.lineTo(targetX - 30, targetY);
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Continue rendering
    if (isARActive) {
      requestAnimationFrame(renderAR);
    }
  };

  const addARObject = (type: 'workflow' | 'node' | 'data') => {
    recordInteraction();
    
    const newObject = {
      id: Date.now().toString(),
      type,
      position: {
        x: Math.random() * 0.6 + 0.2,
        y: Math.random() * 0.6 + 0.2,
        z: 0
      },
      label: `New ${type}`,
      color: type === 'workflow' ? '#3b82f6' : type === 'node' ? '#10b981' : '#8b5cf6'
    };
    
    setArObjects(prev => [...prev, newObject]);
  };

  const clearARObjects = () => {
    recordInteraction();
    setArObjects([]);
  };

  // Check if AR is supported
  const isARSupported = 'mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices;

  if (!isARSupported) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-100 dark:bg-slate-800 rounded-lg">
        <div className="text-center p-8">
          <svg className="w-16 h-16 mx-auto mb-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
            AR Not Supported
          </h3>
          <p className="text-slate-600 dark:text-slate-400">
            Your browser or device doesn't support AR features.
            Please use a modern mobile browser or enable camera permissions.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-slate-900 rounded-lg overflow-hidden relative">
      {/* AR Viewport */}
      <div className="relative w-full h-full">
        {!isARActive ? (
          // AR Setup Screen
          <div className="h-full flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
            <div className="text-center p-8 max-w-md">
              <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              
              <h2 className="text-2xl font-bold text-white mb-4">
                AR Workflow Preview
              </h2>
              
              <p className="text-slate-300 mb-6">
                Visualize your automation workflows in augmented reality. 
                See how data flows through your processes in real-time.
              </p>

              {devices.length > 1 && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Select Camera
                  </label>
                  <select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 text-white rounded-lg border border-slate-600 focus:ring-2 focus:ring-purple-500"
                  >
                    {devices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <button
                onClick={startAR}
                className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all transform hover:scale-105 font-medium"
              >
                Start AR Experience
              </button>

              {error && (
                <div className="mt-4 p-3 bg-red-900/50 border border-red-500 rounded-lg text-red-200 text-sm">
                  {error}
                </div>
              )}
            </div>
          </div>
        ) : (
          // AR Active View
          <>
            {/* Hidden video element */}
            <video
              ref={videoRef}
              className="hidden"
              autoPlay
              playsInline
              muted
            />
            
            {/* AR Canvas */}
            <canvas
              ref={canvasRef}
              className="w-full h-full object-cover"
            />
            
            {/* AR Controls Overlay */}
            <div className="absolute top-4 left-4 right-4 flex justify-between items-start">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg p-3">
                <h3 className="text-white font-semibold mb-2">AR Controls</h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => addARObject('workflow')}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
                  >
                    + Workflow
                  </button>
                  <button
                    onClick={() => addARObject('node')}
                    className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 transition-colors"
                  >
                    + Node
                  </button>
                  <button
                    onClick={() => addARObject('data')}
                    className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 transition-colors"
                  >
                    + Data
                  </button>
                </div>
              </div>
              
              <div className="flex space-x-2">
                <button
                  onClick={clearARObjects}
                  className="px-3 py-2 bg-red-600/80 text-white rounded-lg hover:bg-red-700/80 transition-colors"
                >
                  Clear All
                </button>
                <button
                  onClick={stopAR}
                  className="px-3 py-2 bg-slate-600/80 text-white rounded-lg hover:bg-slate-700/80 transition-colors"
                >
                  Stop AR
                </button>
              </div>
            </div>

            {/* AR Info Panel */}
            <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-4 text-white">
              <h4 className="font-semibold mb-2">AR Objects ({arObjects.length})</h4>
              <div className="space-y-1 text-sm">
                {arObjects.map((obj) => (
                  <div key={obj.id} className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: obj.color }}
                    />
                    <span>{obj.label}</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-3 pt-3 border-t border-white/20 text-xs text-slate-300">
                <p>• Tap to add objects to the AR scene</p>
                <p>• Objects represent workflow components</p>
                <p>• Lines show data flow connections</p>
              </div>
            </div>

            {/* Performance Indicator */}
            <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg p-2">
              <div className="flex items-center space-x-2 text-white text-sm">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span>AR Active</span>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Feature Info */}
      {!isARActive && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-4 left-4 right-4 bg-slate-800/90 backdrop-blur-sm rounded-lg p-4"
        >
          <h4 className="text-white font-semibold mb-2">AR Features</h4>
          <div className="grid grid-cols-2 gap-3 text-sm text-slate-300">
            <div className="flex items-center space-x-2">
              <span className="text-blue-400">🔧</span>
              <span>Workflow Visualization</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-green-400">⚙️</span>
              <span>Node Interactions</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-purple-400">📊</span>
              <span>Data Flow Tracking</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-yellow-400">🎯</span>
              <span>Real-time Updates</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}