'use client';

import React, { useRef, useMemo, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Text, Box, Sphere, Line, Html } from '@react-three/drei';
import { Vector3, Color } from 'three';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/lib/store';

// Node component for 3D workflow visualization
function WorkflowNode({ 
  position, 
  type, 
  label, 
  status = 'idle',
  onClick,
  isSelected = false 
}: {
  position: [number, number, number];
  type: 'trigger' | 'action' | 'condition';
  label: string;
  status?: 'idle' | 'running' | 'success' | 'error';
  onClick?: () => void;
  isSelected?: boolean;
}) {
  const meshRef = useRef<any>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(() => {
    switch (type) {
      case 'trigger': return '#10b981';
      case 'action': return '#3b82f6';
      case 'condition': return '#8b5cf6';
      default: return '#64748b';
    }
  }, [type]);

  const statusColor = useMemo(() => {
    switch (status) {
      case 'running': return '#f59e0b';
      case 'success': return '#10b981';
      case 'error': return '#ef4444';
      default: return color;
    }
  }, [status, color]);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.1;
      
      // Rotation based on status
      if (status === 'running') {
        meshRef.current.rotation.y += 0.02;
      }
      
      // Scale animation on hover or selection
      const targetScale = hovered || isSelected ? 1.2 : 1;
      meshRef.current.scale.lerp({ x: targetScale, y: targetScale, z: targetScale }, 0.1);
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        {type === 'condition' ? (
          <octahedronGeometry args={[0.8]} />
        ) : (
          <boxGeometry args={[1.5, 1, 1.5]} />
        )}
        <meshStandardMaterial 
          color={statusColor} 
          emissive={statusColor}
          emissiveIntensity={status === 'running' ? 0.3 : 0.1}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      
      {/* Node label */}
      <Html position={[0, -1.5, 0]} center>
        <div className="bg-black/80 text-white px-2 py-1 rounded text-xs whitespace-nowrap">
          {label}
        </div>
      </Html>
      
      {/* Status indicator */}
      {status === 'running' && (
        <Sphere position={[0, 1.5, 0]} args={[0.2]}>
          <meshBasicMaterial color="#f59e0b" />
        </Sphere>
      )}
      
      {/* Selection indicator */}
      {isSelected && (
        <mesh position={[0, 0, 0]}>
          <ringGeometry args={[1.8, 2.2, 32]} />
          <meshBasicMaterial color="#3b82f6" transparent opacity={0.5} />
        </mesh>
      )}
    </group>
  );
}

// Connection line between nodes
function NodeConnection({ 
  start, 
  end, 
  animated = false,
  status = 'idle' 
}: {
  start: [number, number, number];
  end: [number, number, number];
  animated?: boolean;
  status?: 'idle' | 'active' | 'success' | 'error';
}) {
  const lineRef = useRef<any>(null);
  const [progress, setProgress] = useState(0);

  const points = useMemo(() => {
    const startVec = new Vector3(...start);
    const endVec = new Vector3(...end);
    const midPoint = startVec.clone().lerp(endVec, 0.5);
    midPoint.y += 1; // Arc effect
    
    const curve = [];
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      const point = new Vector3();
      point.lerpVectors(startVec, midPoint, t * 2);
      if (t > 0.5) {
        point.lerpVectors(midPoint, endVec, (t - 0.5) * 2);
      }
      curve.push(point);
    }
    return curve;
  }, [start, end]);

  const color = useMemo(() => {
    switch (status) {
      case 'active': return '#f59e0b';
      case 'success': return '#10b981';
      case 'error': return '#ef4444';
      default: return '#64748b';
    }
  }, [status]);

  useFrame(() => {
    if (animated && status === 'active') {
      setProgress((prev) => (prev + 0.02) % 1);
    }
  });

  return (
    <group>
      <Line
        ref={lineRef}
        points={points}
        color={color}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      
      {/* Animated particle */}
      {animated && status === 'active' && (
        <Sphere position={points[Math.floor(progress * (points.length - 1))]} args={[0.1]}>
          <meshBasicMaterial color="#f59e0b" />
        </Sphere>
      )}
    </group>
  );
}

// Data flow visualization
function DataFlowVisualization() {
  const { camera } = useThree();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState<'idle' | 'running' | 'completed'>('idle');

  // Sample workflow data
  const nodes = [
    { id: '1', type: 'trigger' as const, label: 'HTTP Webhook', position: [-4, 0, 0] as [number, number, number], status: 'idle' as const },
    { id: '2', type: 'condition' as const, label: 'Validate Data', position: [-1, 2, 0] as [number, number, number], status: 'idle' as const },
    { id: '3', type: 'action' as const, label: 'Process Data', position: [2, 1, 0] as [number, number, number], status: 'idle' as const },
    { id: '4', type: 'action' as const, label: 'Send Email', position: [2, -1, 0] as [number, number, number], status: 'idle' as const },
    { id: '5', type: 'action' as const, label: 'Log Result', position: [5, 0, 0] as [number, number, number], status: 'idle' as const },
  ];

  const connections = [
    { from: '1', to: '2', status: 'idle' as const },
    { from: '2', to: '3', status: 'idle' as const },
    { from: '2', to: '4', status: 'idle' as const },
    { from: '3', to: '5', status: 'idle' as const },
    { from: '4', to: '5', status: 'idle' as const },
  ];

  const runWorkflow = () => {
    setWorkflowStatus('running');
    // Simulate workflow execution
    setTimeout(() => setWorkflowStatus('completed'), 5000);
  };

  // Auto-rotate camera
  useFrame((state) => {
    if (workflowStatus === 'idle') {
      camera.position.x = Math.sin(state.clock.elapsedTime * 0.1) * 10;
      camera.position.z = Math.cos(state.clock.elapsedTime * 0.1) * 10;
      camera.lookAt(0, 0, 0);
    }
  });

  return (
    <group>
      {/* Environment */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#3b82f6" />

      {/* Grid floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.3} wireframe />
      </mesh>

      {/* Workflow nodes */}
      {nodes.map((node) => (
        <WorkflowNode
          key={node.id}
          position={node.position}
          type={node.type}
          label={node.label}
          status={workflowStatus === 'running' ? 'running' : node.status}
          onClick={() => setSelectedNode(node.id)}
          isSelected={selectedNode === node.id}
        />
      ))}

      {/* Node connections */}
      {connections.map((connection, index) => {
        const startNode = nodes.find(n => n.id === connection.from);
        const endNode = nodes.find(n => n.id === connection.to);
        
        if (!startNode || !endNode) return null;
        
        return (
          <NodeConnection
            key={`${connection.from}-${connection.to}`}
            start={startNode.position}
            end={endNode.position}
            animated={workflowStatus === 'running'}
            status={workflowStatus === 'running' ? 'active' : connection.status}
          />
        );
      })}

      {/* Floating UI */}
      <Html position={[0, 4, 0]} center>
        <div className="bg-black/80 text-white p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">3D Workflow Visualization</h3>
          <div className="flex space-x-2">
            <button
              onClick={runWorkflow}
              disabled={workflowStatus === 'running'}
              className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {workflowStatus === 'running' ? 'Running...' : 'Run Workflow'}
            </button>
            <button
              onClick={() => setSelectedNode(null)}
              className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Clear Selection
            </button>
          </div>
          {selectedNode && (
            <div className="mt-2 text-sm">
              Selected: {nodes.find(n => n.id === selectedNode)?.label}
            </div>
          )}
        </div>
      </Html>

      {/* Performance metrics visualization */}
      <group position={[8, 2, 0]}>
        <Text
          position={[0, 1, 0]}
          fontSize={0.5}
          color="#3b82f6"
          anchorX="center"
          anchorY="middle"
        >
          Performance
        </Text>
        
        {/* CPU usage bar */}
        <Box position={[0, 0, 0]} args={[2, 0.2, 0.2]}>
          <meshBasicMaterial color="#1e293b" />
        </Box>
        <Box position={[-0.5, 0, 0.1]} args={[1, 0.15, 0.15]}>
          <meshBasicMaterial color="#10b981" />
        </Box>
        
        {/* Memory usage bar */}
        <Box position={[0, -0.5, 0]} args={[2, 0.2, 0.2]}>
          <meshBasicMaterial color="#1e293b" />
        </Box>
        <Box position={[-0.3, -0.5, 0.1]} args={[1.4, 0.15, 0.15]}>
          <meshBasicMaterial color="#f59e0b" />
        </Box>
        
        {/* Network latency indicator */}
        <Sphere position={[0, -1, 0]} args={[0.2]}>
          <meshBasicMaterial color="#3b82f6" />
        </Sphere>
      </group>

      {/* Data particles */}
      {Array.from({ length: 20 }).map((_, i) => (
        <Sphere
          key={i}
          position={[
            (Math.random() - 0.5) * 15,
            (Math.random() - 0.5) * 10,
            (Math.random() - 0.5) * 15
          ]}
          args={[0.05]}
        >
          <meshBasicMaterial 
            color={new Color().setHSL(Math.random(), 0.7, 0.5)} 
            transparent 
            opacity={0.6} 
          />
        </Sphere>
      ))}
    </group>
  );
}

export default function ThreeVisualization() {
  return <DataFlowVisualization />;
}