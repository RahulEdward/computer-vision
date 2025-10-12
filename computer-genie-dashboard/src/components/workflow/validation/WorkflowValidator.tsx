import React, { useState, useCallback, useEffect } from 'react';
import { Node, Edge } from 'reactflow';
import { AlertTriangle, CheckCircle, XCircle, Info, RefreshCw } from 'lucide-react';

export interface ValidationRule {
  id: string;
  name: string;
  description: string;
  severity: 'error' | 'warning' | 'info';
  validate: (nodes: Node[], edges: Edge[]) => ValidationResult[];
}

export interface ValidationResult {
  ruleId: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  nodeId?: string;
  edgeId?: string;
  suggestion?: string;
}

export interface WorkflowValidatorProps {
  nodes: Node[];
  edges: Edge[];
  onValidationChange?: (results: ValidationResult[]) => void;
  autoValidate?: boolean;
}

// Built-in validation rules
const defaultValidationRules: ValidationRule[] = [
  {
    id: 'no-orphaned-nodes',
    name: 'No Orphaned Nodes',
    description: 'Checks for nodes with no connections',
    severity: 'warning',
    validate: (nodes, edges) => {
      const connectedNodeIds = new Set([
        ...edges.map(e => e.source),
        ...edges.map(e => e.target)
      ]);
      
      return nodes
        .filter(node => !connectedNodeIds.has(node.id) && node.type !== 'group')
        .map(node => ({
          ruleId: 'no-orphaned-nodes',
          severity: 'warning' as const,
          message: `Node "${node.data?.label || node.id}" has no connections`,
          nodeId: node.id,
          suggestion: 'Connect this node to other nodes or remove it if not needed'
        }));
    }
  },
  {
    id: 'no-circular-dependencies',
    name: 'No Circular Dependencies',
    description: 'Detects circular dependencies in the workflow',
    severity: 'error',
    validate: (nodes, edges) => {
      const results: ValidationResult[] = [];
      const visited = new Set<string>();
      const recursionStack = new Set<string>();
      
      const hasCycle = (nodeId: string, path: string[] = []): boolean => {
        if (recursionStack.has(nodeId)) {
          results.push({
            ruleId: 'no-circular-dependencies',
            severity: 'error',
            message: `Circular dependency detected: ${[...path, nodeId].join(' → ')}`,
            suggestion: 'Remove one of the connections to break the cycle'
          });
          return true;
        }
        
        if (visited.has(nodeId)) return false;
        
        visited.add(nodeId);
        recursionStack.add(nodeId);
        
        const outgoingEdges = edges.filter(e => e.source === nodeId);
        for (const edge of outgoingEdges) {
          if (hasCycle(edge.target, [...path, nodeId])) {
            return true;
          }
        }
        
        recursionStack.delete(nodeId);
        return false;
      };
      
      nodes.forEach(node => {
        if (!visited.has(node.id)) {
          hasCycle(node.id);
        }
      });
      
      return results;
    }
  },
  {
    id: 'valid-node-positions',
    name: 'Valid Node Positions',
    description: 'Ensures all nodes have valid positions',
    severity: 'error',
    validate: (nodes) => {
      return nodes
        .filter(node => 
          !node.position || 
          typeof node.position.x !== 'number' || 
          typeof node.position.y !== 'number' ||
          isNaN(node.position.x) ||
          isNaN(node.position.y)
        )
        .map(node => ({
          ruleId: 'valid-node-positions',
          severity: 'error' as const,
          message: `Node "${node.data?.label || node.id}" has invalid position`,
          nodeId: node.id,
          suggestion: 'Set a valid position with numeric x and y coordinates'
        }));
    }
  },
  {
    id: 'no-duplicate-ids',
    name: 'No Duplicate IDs',
    description: 'Checks for duplicate node or edge IDs',
    severity: 'error',
    validate: (nodes, edges) => {
      const results: ValidationResult[] = [];
      const nodeIds = new Set<string>();
      const edgeIds = new Set<string>();
      
      // Check duplicate node IDs
      nodes.forEach(node => {
        if (nodeIds.has(node.id)) {
          results.push({
            ruleId: 'no-duplicate-ids',
            severity: 'error',
            message: `Duplicate node ID: "${node.id}"`,
            nodeId: node.id,
            suggestion: 'Ensure all node IDs are unique'
          });
        }
        nodeIds.add(node.id);
      });
      
      // Check duplicate edge IDs
      edges.forEach(edge => {
        if (edgeIds.has(edge.id)) {
          results.push({
            ruleId: 'no-duplicate-ids',
            severity: 'error',
            message: `Duplicate edge ID: "${edge.id}"`,
            edgeId: edge.id,
            suggestion: 'Ensure all edge IDs are unique'
          });
        }
        edgeIds.add(edge.id);
      });
      
      return results;
    }
  },
  {
    id: 'valid-edge-connections',
    name: 'Valid Edge Connections',
    description: 'Ensures all edges connect to existing nodes',
    severity: 'error',
    validate: (nodes, edges) => {
      const nodeIds = new Set(nodes.map(n => n.id));
      
      return edges
        .filter(edge => !nodeIds.has(edge.source) || !nodeIds.has(edge.target))
        .map(edge => ({
          ruleId: 'valid-edge-connections',
          severity: 'error' as const,
          message: `Edge "${edge.id}" connects to non-existent node(s)`,
          edgeId: edge.id,
          suggestion: 'Remove invalid edges or add missing nodes'
        }));
    }
  },
  {
    id: 'performance-check',
    name: 'Performance Check',
    description: 'Warns about potential performance issues',
    severity: 'warning',
    validate: (nodes, edges) => {
      const results: ValidationResult[] = [];
      
      if (nodes.length > 100) {
        results.push({
          ruleId: 'performance-check',
          severity: 'warning',
          message: `Large workflow detected: ${nodes.length} nodes`,
          suggestion: 'Consider enabling performance optimizations or breaking into smaller workflows'
        });
      }
      
      if (edges.length > 200) {
        results.push({
          ruleId: 'performance-check',
          severity: 'warning',
          message: `Many connections detected: ${edges.length} edges`,
          suggestion: 'Consider simplifying connections or using groups'
        });
      }
      
      return results;
    }
  }
];

export const WorkflowValidator: React.FC<WorkflowValidatorProps> = ({
  nodes,
  edges,
  onValidationChange,
  autoValidate = true
}) => {
  const [validationResults, setValidationResults] = useState<ValidationResult[]>([]);
  const [isValidating, setIsValidating] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const runValidation = useCallback(async () => {
    setIsValidating(true);
    
    try {
      const allResults: ValidationResult[] = [];
      
      for (const rule of defaultValidationRules) {
        try {
          const ruleResults = rule.validate(nodes, edges);
          allResults.push(...ruleResults);
        } catch (error) {
          console.error(`Validation rule "${rule.id}" failed:`, error);
          allResults.push({
            ruleId: rule.id,
            severity: 'error',
            message: `Validation rule "${rule.name}" encountered an error`,
            suggestion: 'Check the validation rule implementation'
          });
        }
      }
      
      setValidationResults(allResults);
      onValidationChange?.(allResults);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setIsValidating(false);
    }
  }, [nodes, edges, onValidationChange]);

  useEffect(() => {
    if (autoValidate) {
      const timeoutId = setTimeout(runValidation, 500); // Debounce validation
      return () => clearTimeout(timeoutId);
    }
  }, [autoValidate, runValidation]);

  const errorCount = validationResults.filter(r => r.severity === 'error').length;
  const warningCount = validationResults.filter(r => r.severity === 'warning').length;
  const infoCount = validationResults.filter(r => r.severity === 'info').length;

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'info': return <Info className="w-4 h-4 text-blue-500" />;
      default: return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'info': return 'border-blue-500 bg-blue-50';
      default: return 'border-green-500 bg-green-50';
    }
  };

  return (
    <div className="fixed bottom-4 left-4 bg-gray-800 border border-gray-700 rounded-lg p-4 z-50 max-w-md">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold">Workflow Validation</h3>
        <button
          onClick={runValidation}
          disabled={isValidating}
          className="p-1 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isValidating ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Summary */}
      <div className="flex items-center gap-4 mb-3 text-sm">
        {errorCount > 0 && (
          <div className="flex items-center gap-1 text-red-400">
            <XCircle className="w-4 h-4" />
            <span>{errorCount} errors</span>
          </div>
        )}
        {warningCount > 0 && (
          <div className="flex items-center gap-1 text-yellow-400">
            <AlertTriangle className="w-4 h-4" />
            <span>{warningCount} warnings</span>
          </div>
        )}
        {infoCount > 0 && (
          <div className="flex items-center gap-1 text-blue-400">
            <Info className="w-4 h-4" />
            <span>{infoCount} info</span>
          </div>
        )}
        {validationResults.length === 0 && (
          <div className="flex items-center gap-1 text-green-400">
            <CheckCircle className="w-4 h-4" />
            <span>All good!</span>
          </div>
        )}
      </div>

      {/* Toggle Details */}
      {validationResults.length > 0 && (
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-blue-400 hover:text-blue-300 text-sm mb-2 transition-colors"
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      )}

      {/* Detailed Results */}
      {showDetails && validationResults.length > 0 && (
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {validationResults.map((result, index) => (
            <div
              key={index}
              className={`p-2 rounded border-l-4 ${getSeverityColor(result.severity)}`}
            >
              <div className="flex items-start gap-2">
                {getSeverityIcon(result.severity)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">
                    {result.message}
                  </p>
                  {result.suggestion && (
                    <p className="text-xs text-gray-600 mt-1">
                      💡 {result.suggestion}
                    </p>
                  )}
                  {(result.nodeId || result.edgeId) && (
                    <p className="text-xs text-gray-500 mt-1">
                      {result.nodeId && `Node: ${result.nodeId}`}
                      {result.edgeId && `Edge: ${result.edgeId}`}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Validation Status */}
      <div className="mt-3 pt-2 border-t border-gray-600 text-xs text-gray-400">
        <p>
          Validated {nodes.length} nodes, {edges.length} edges
          {isValidating && ' • Validating...'}
        </p>
      </div>
    </div>
  );
};

export default WorkflowValidator;