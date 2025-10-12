import React, { useState, useCallback, useEffect } from 'react';
import { Node, Edge } from 'reactflow';
import { HierarchicalNode } from '../types/WorkflowTypes';

interface ComplexWorkflowTestProps {
  onLoadWorkflow: (nodes: Node[], edges: Edge[]) => void;
}

export const ComplexWorkflowTest: React.FC<ComplexWorkflowTestProps> = ({ onLoadWorkflow }) => {
  const [testScenarios] = useState([
    {
      name: 'Small Workflow (10 nodes)',
      nodeCount: 10,
      description: 'Basic workflow for testing core functionality'
    },
    {
      name: 'Medium Workflow (50 nodes)',
      nodeCount: 50,
      description: 'Medium complexity workflow with groups and hierarchies'
    },
    {
      name: 'Large Workflow (200 nodes)',
      nodeCount: 200,
      description: 'Large-scale workflow for performance testing'
    },
    {
      name: 'Enterprise Workflow (500 nodes)',
      nodeCount: 500,
      description: 'Enterprise-scale workflow with complex relationships'
    }
  ]);

  const generateComplexWorkflow = useCallback((nodeCount: number) => {
    const nodes: HierarchicalNode[] = [];
    const edges: Edge[] = [];
    
    // Generate nodes with different types and hierarchical relationships
    for (let i = 0; i < nodeCount; i++) {
      const nodeType = i % 10 === 0 ? 'group' : i % 3 === 0 ? 'enterprise' : 'child';
      const parentId = nodeType === 'child' && i > 0 ? `node-${Math.floor(i / 5) * 5}` : undefined;
      
      const node: HierarchicalNode = {
        id: `node-${i}`,
        type: nodeType,
        position: {
          x: (i % 10) * 200 + Math.random() * 50,
          y: Math.floor(i / 10) * 150 + Math.random() * 50
        },
        data: {
          label: `${nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} Node ${i}`,
          type: nodeType,
          parentId,
          childIds: nodeType === 'group' ? [] : undefined,
          level: parentId ? 1 : 0,
          metadata: {
            created: new Date().toISOString(),
            description: `Generated test node ${i} of type ${nodeType}`,
            status: ['ready', 'running', 'completed', 'error'][Math.floor(Math.random() * 4)],
            progress: Math.random() * 100,
            testData: true
          },
          performance: {
            status: 'idle' as const
          }
        },
        style: {
          width: nodeType === 'group' ? 300 : 200,
          height: nodeType === 'group' ? 200 : 100,
        }
      };

      nodes.push(node);

      // Generate edges
      if (i > 0 && Math.random() > 0.3) {
        const sourceIndex = Math.floor(Math.random() * i);
        edges.push({
          id: `edge-${sourceIndex}-${i}`,
          source: `node-${sourceIndex}`,
          target: `node-${i}`,
          type: 'custom',
          animated: Math.random() > 0.7,
          style: {
            stroke: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'][Math.floor(Math.random() * 4)],
            strokeWidth: 2
          },
          data: {
            label: `Connection ${sourceIndex} → ${i}`,
            weight: Math.random() * 10,
            type: ['data', 'control', 'trigger'][Math.floor(Math.random() * 3)]
          }
        });
      }
    }

    // Update group nodes with their children
    nodes.forEach(node => {
      if (node.type === 'group') {
        node.childIds = nodes
          .filter(n => n.parentId === node.id)
          .map(n => n.id);
      }
    });

    return { nodes, edges };
  }, []);

  const loadTestWorkflow = useCallback((nodeCount: number) => {
    const { nodes, edges } = generateComplexWorkflow(nodeCount);
    onLoadWorkflow(nodes, edges);
  }, [generateComplexWorkflow, onLoadWorkflow]);

  const runPerformanceTest = useCallback(async (nodeCount: number) => {
    console.log(`Starting performance test with ${nodeCount} nodes...`);
    
    const startTime = performance.now();
    const { nodes, edges } = generateComplexWorkflow(nodeCount);
    const generationTime = performance.now() - startTime;
    
    console.log(`Generated ${nodes.length} nodes and ${edges.length} edges in ${generationTime.toFixed(2)}ms`);
    
    const loadStartTime = performance.now();
    onLoadWorkflow(nodes, edges);
    const loadTime = performance.now() - loadStartTime;
    
    console.log(`Loaded workflow in ${loadTime.toFixed(2)}ms`);
    console.log(`Total test time: ${(generationTime + loadTime).toFixed(2)}ms`);
    
    // Memory usage estimation
    const memoryEstimate = (nodes.length * 1000 + edges.length * 500) / 1024; // KB
    console.log(`Estimated memory usage: ${memoryEstimate.toFixed(2)} KB`);
  }, [generateComplexWorkflow, onLoadWorkflow]);

  return (
    <div className="fixed top-4 right-4 bg-gray-800 border border-gray-700 rounded-lg p-4 z-50 max-w-sm">
      <h3 className="text-white font-semibold mb-3">Workflow Test Suite</h3>
      
      <div className="space-y-2">
        {testScenarios.map((scenario, index) => (
          <div key={index} className="bg-gray-700 rounded p-3">
            <h4 className="text-white text-sm font-medium">{scenario.name}</h4>
            <p className="text-gray-300 text-xs mb-2">{scenario.description}</p>
            
            <div className="flex gap-2">
              <button
                onClick={() => loadTestWorkflow(scenario.nodeCount)}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
              >
                Load
              </button>
              <button
                onClick={() => runPerformanceTest(scenario.nodeCount)}
                className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors"
              >
                Test Performance
              </button>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-4 pt-3 border-t border-gray-600">
        <button
          onClick={() => {
            console.clear();
            console.log('Starting comprehensive test suite...');
            testScenarios.forEach((scenario, index) => {
              setTimeout(() => runPerformanceTest(scenario.nodeCount), index * 2000);
            });
          }}
          className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded transition-colors"
        >
          Run All Tests
        </button>
      </div>
      
      <div className="mt-3 text-xs text-gray-400">
        <p>• Check browser console for detailed performance metrics</p>
        <p>• Monitor performance panel for real-time stats</p>
        <p>• Test drag-drop, zoom, and selection features</p>
      </div>
    </div>
  );
};

export default ComplexWorkflowTest;