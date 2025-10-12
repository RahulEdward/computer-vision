import { Node, Edge } from 'reactflow';
import { NodeInputPort, NodeOutputPort } from '../nodes/base/BaseNode';

export interface ConnectionValidationResult {
  valid: boolean;
  error?: string;
  warnings?: string[];
}

export interface DataFlowValidation {
  nodeId: string;
  valid: boolean;
  errors: string[];
  warnings: string[];
  inputValidation: Record<string, ConnectionValidationResult>;
  outputValidation: Record<string, ConnectionValidationResult>;
}

export interface ConnectionRule {
  id: string;
  name: string;
  description: string;
  validate: (source: Node, target: Node, sourcePort: string, targetPort: string) => ConnectionValidationResult;
}

export interface DataTypeDefinition {
  name: string;
  description: string;
  schema?: any;
  validator?: (data: any) => boolean;
}

export class NodeConnectionSystem {
  private connectionRules: Map<string, ConnectionRule> = new Map();
  private dataTypes: Map<string, DataTypeDefinition> = new Map();
  private nodePortCache: Map<string, { inputs: NodeInputPort[], outputs: NodeOutputPort[] }> = new Map();

  constructor() {
    this.initializeDefaultRules();
    this.initializeDefaultDataTypes();
  }

  /**
   * Initialize default connection validation rules
   */
  private initializeDefaultRules(): void {
    // Rule: No self-connections
    this.addConnectionRule({
      id: 'no-self-connection',
      name: 'No Self Connection',
      description: 'Nodes cannot connect to themselves',
      validate: (source, target) => ({
        valid: source.id !== target.id,
        error: source.id === target.id ? 'Nodes cannot connect to themselves' : undefined
      })
    });

    // Rule: No circular dependencies
    this.addConnectionRule({
      id: 'no-circular-dependency',
      name: 'No Circular Dependencies',
      description: 'Connections cannot create circular dependencies',
      validate: (source, target) => {
        // This would need a more complex implementation to detect cycles
        return { valid: true };
      }
    });

    // Rule: Trigger nodes cannot have inputs
    this.addConnectionRule({
      id: 'trigger-no-inputs',
      name: 'Trigger Nodes No Inputs',
      description: 'Trigger nodes cannot have input connections',
      validate: (source, target) => {
        const isTriggerTarget = target.type?.includes('trigger') || target.data?.nodeType?.includes('trigger');
        return {
          valid: !isTriggerTarget,
          error: isTriggerTarget ? 'Trigger nodes cannot have input connections' : undefined
        };
      }
    });

    // Rule: Maximum connections per port
    this.addConnectionRule({
      id: 'max-connections',
      name: 'Maximum Connections',
      description: 'Ports cannot exceed their maximum connection limit',
      validate: (source, target, sourcePort, targetPort) => {
        // This would check against actual port configurations
        return { valid: true };
      }
    });
  }

  /**
   * Initialize default data types
   */
  private initializeDefaultDataTypes(): void {
    this.addDataType({
      name: 'any',
      description: 'Any data type',
      validator: () => true
    });

    this.addDataType({
      name: 'object',
      description: 'JSON object',
      validator: (data) => typeof data === 'object' && data !== null && !Array.isArray(data)
    });

    this.addDataType({
      name: 'array',
      description: 'Array of items',
      validator: (data) => Array.isArray(data)
    });

    this.addDataType({
      name: 'string',
      description: 'String value',
      validator: (data) => typeof data === 'string'
    });

    this.addDataType({
      name: 'number',
      description: 'Numeric value',
      validator: (data) => typeof data === 'number' && !isNaN(data)
    });

    this.addDataType({
      name: 'boolean',
      description: 'Boolean value',
      validator: (data) => typeof data === 'boolean'
    });
  }

  /**
   * Add a new connection rule
   */
  public addConnectionRule(rule: ConnectionRule): void {
    this.connectionRules.set(rule.id, rule);
  }

  /**
   * Add a new data type
   */
  public addDataType(dataType: DataTypeDefinition): void {
    this.dataTypes.set(dataType.name, dataType);
  }

  /**
   * Validate a potential connection between two nodes
   */
  public validateConnection(
    source: Node,
    target: Node,
    sourcePort: string = 'main',
    targetPort: string = 'main',
    edges: Edge[] = []
  ): ConnectionValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Run all connection rules
    for (const rule of this.connectionRules.values()) {
      const result = rule.validate(source, target, sourcePort, targetPort);
      if (!result.valid && result.error) {
        errors.push(result.error);
      }
      if (result.warnings) {
        warnings.push(...result.warnings);
      }
    }

    // Check port compatibility
    const portValidation = this.validatePortCompatibility(source, target, sourcePort, targetPort);
    if (!portValidation.valid && portValidation.error) {
      errors.push(portValidation.error);
    }

    // Check for duplicate connections
    const duplicateCheck = this.checkDuplicateConnection(source, target, sourcePort, targetPort, edges);
    if (!duplicateCheck.valid && duplicateCheck.error) {
      errors.push(duplicateCheck.error);
    }

    return {
      valid: errors.length === 0,
      error: errors.length > 0 ? errors.join('; ') : undefined,
      warnings: warnings.length > 0 ? warnings : undefined
    };
  }

  /**
   * Validate port compatibility between source and target
   */
  private validatePortCompatibility(
    source: Node,
    target: Node,
    sourcePort: string,
    targetPort: string
  ): ConnectionValidationResult {
    const sourceOutputs = this.getNodeOutputPorts(source);
    const targetInputs = this.getNodeInputPorts(target);

    const sourcePortConfig = sourceOutputs.find(p => p.name === sourcePort);
    const targetPortConfig = targetInputs.find(p => p.name === targetPort);

    if (!sourcePortConfig) {
      return {
        valid: false,
        error: `Source node does not have output port '${sourcePort}'`
      };
    }

    if (!targetPortConfig) {
      return {
        valid: false,
        error: `Target node does not have input port '${targetPort}'`
      };
    }

    // Check port types compatibility
    if (sourcePortConfig.type !== targetPortConfig.type) {
      return {
        valid: false,
        error: `Port types incompatible: ${sourcePortConfig.type} -> ${targetPortConfig.type}`
      };
    }

    return { valid: true };
  }

  /**
   * Check for duplicate connections
   */
  private checkDuplicateConnection(
    source: Node,
    target: Node,
    sourcePort: string,
    targetPort: string,
    edges: Edge[]
  ): ConnectionValidationResult {
    const existingConnection = edges.find(edge =>
      edge.source === source.id &&
      edge.target === target.id &&
      edge.sourceHandle === sourcePort &&
      edge.targetHandle === targetPort
    );

    if (existingConnection) {
      return {
        valid: false,
        error: 'Connection already exists between these ports'
      };
    }

    return { valid: true };
  }

  /**
   * Validate the entire workflow's data flow
   */
  public validateWorkflowDataFlow(nodes: Node[], edges: Edge[]): DataFlowValidation[] {
    const validations: DataFlowValidation[] = [];

    for (const node of nodes) {
      const validation = this.validateNodeDataFlow(node, nodes, edges);
      validations.push(validation);
    }

    return validations;
  }

  /**
   * Validate data flow for a specific node
   */
  private validateNodeDataFlow(node: Node, allNodes: Node[], edges: Edge[]): DataFlowValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const inputValidation: Record<string, ConnectionValidationResult> = {};
    const outputValidation: Record<string, ConnectionValidationResult> = {};

    // Get node's input and output ports
    const inputPorts = this.getNodeInputPorts(node);
    const outputPorts = this.getNodeOutputPorts(node);

    // Validate inputs
    for (const inputPort of inputPorts) {
      const incomingEdges = edges.filter(edge => 
        edge.target === node.id && edge.targetHandle === inputPort.name
      );

      if (inputPort.required && incomingEdges.length === 0) {
        const error = `Required input port '${inputPort.displayName}' has no connections`;
        errors.push(error);
        inputValidation[inputPort.name] = { valid: false, error };
      } else if (inputPort.maxConnections && incomingEdges.length > inputPort.maxConnections) {
        const error = `Input port '${inputPort.displayName}' exceeds maximum connections (${inputPort.maxConnections})`;
        errors.push(error);
        inputValidation[inputPort.name] = { valid: false, error };
      } else {
        inputValidation[inputPort.name] = { valid: true };
      }
    }

    // Validate outputs
    for (const outputPort of outputPorts) {
      const outgoingEdges = edges.filter(edge => 
        edge.source === node.id && edge.sourceHandle === outputPort.name
      );

      if (outputPort.maxConnections && outputPort.maxConnections > 0 && outgoingEdges.length > outputPort.maxConnections) {
        const error = `Output port '${outputPort.displayName}' exceeds maximum connections (${outputPort.maxConnections})`;
        errors.push(error);
        outputValidation[outputPort.name] = { valid: false, error };
      } else {
        outputValidation[outputPort.name] = { valid: true };
      }
    }

    // Check for orphaned nodes (except triggers)
    const isTriggerNode = node.type?.includes('trigger') || node.data?.nodeType?.includes('trigger');
    if (!isTriggerNode) {
      const hasInputConnections = edges.some(edge => edge.target === node.id);
      if (!hasInputConnections) {
        warnings.push('Node has no input connections and is not a trigger');
      }
    }

    const hasOutputConnections = edges.some(edge => edge.source === node.id);
    if (!hasOutputConnections) {
      warnings.push('Node has no output connections');
    }

    return {
      nodeId: node.id,
      valid: errors.length === 0,
      errors,
      warnings,
      inputValidation,
      outputValidation
    };
  }

  /**
   * Get input ports for a node
   */
  private getNodeInputPorts(node: Node): NodeInputPort[] {
    // Try to get from cache first
    const cached = this.nodePortCache.get(node.id);
    if (cached) {
      return cached.inputs;
    }

    // Default input ports based on node type
    const defaultInputs: NodeInputPort[] = [
      {
        name: 'main',
        type: 'main',
        displayName: 'Input',
        required: true,
        maxConnections: 1
      }
    ];

    // Trigger nodes have no inputs
    if (node.type?.includes('trigger') || node.data?.nodeType?.includes('trigger')) {
      return [];
    }

    return defaultInputs;
  }

  /**
   * Get output ports for a node
   */
  private getNodeOutputPorts(node: Node): NodeOutputPort[] {
    // Try to get from cache first
    const cached = this.nodePortCache.get(node.id);
    if (cached) {
      return cached.outputs;
    }

    // Default output ports based on node type
    const defaultOutputs: NodeOutputPort[] = [
      {
        name: 'main',
        type: 'main',
        displayName: 'Output',
        maxConnections: -1
      }
    ];

    // Condition nodes have two outputs
    if (node.type === 'condition' || node.data?.nodeType === 'condition') {
      return [
        {
          name: 'true',
          type: 'main',
          displayName: 'True',
          maxConnections: -1
        },
        {
          name: 'false',
          type: 'main',
          displayName: 'False',
          maxConnections: -1
        }
      ];
    }

    return defaultOutputs;
  }

  /**
   * Cache node port configuration
   */
  public cacheNodePorts(nodeId: string, inputs: NodeInputPort[], outputs: NodeOutputPort[]): void {
    this.nodePortCache.set(nodeId, { inputs, outputs });
  }

  /**
   * Get all available data types
   */
  public getDataTypes(): DataTypeDefinition[] {
    return Array.from(this.dataTypes.values());
  }

  /**
   * Get all connection rules
   */
  public getConnectionRules(): ConnectionRule[] {
    return Array.from(this.connectionRules.values());
  }

  /**
   * Validate data against a specific type
   */
  public validateDataType(data: any, typeName: string): boolean {
    const dataType = this.dataTypes.get(typeName);
    if (!dataType || !dataType.validator) {
      return true; // If no validator, assume valid
    }
    
    return dataType.validator(data);
  }

  /**
   * Find the shortest path between two nodes
   */
  public findPath(sourceId: string, targetId: string, edges: Edge[]): string[] | null {
    const graph = new Map<string, string[]>();
    
    // Build adjacency list
    for (const edge of edges) {
      if (!graph.has(edge.source)) {
        graph.set(edge.source, []);
      }
      graph.get(edge.source)!.push(edge.target);
    }

    // BFS to find shortest path
    const queue: { nodeId: string; path: string[] }[] = [{ nodeId: sourceId, path: [sourceId] }];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const { nodeId, path } = queue.shift()!;
      
      if (nodeId === targetId) {
        return path;
      }

      if (visited.has(nodeId)) {
        continue;
      }
      visited.add(nodeId);

      const neighbors = graph.get(nodeId) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          queue.push({ nodeId: neighbor, path: [...path, neighbor] });
        }
      }
    }

    return null; // No path found
  }

  /**
   * Detect circular dependencies in the workflow
   */
  public detectCircularDependencies(edges: Edge[]): string[][] {
    const graph = new Map<string, string[]>();
    const cycles: string[][] = [];
    
    // Build adjacency list
    for (const edge of edges) {
      if (!graph.has(edge.source)) {
        graph.set(edge.source, []);
      }
      graph.get(edge.source)!.push(edge.target);
    }

    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const dfs = (nodeId: string, path: string[]): void => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      const neighbors = graph.get(nodeId) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          dfs(neighbor, [...path, neighbor]);
        } else if (recursionStack.has(neighbor)) {
          // Found a cycle
          const cycleStart = path.indexOf(neighbor);
          if (cycleStart !== -1) {
            cycles.push(path.slice(cycleStart));
          }
        }
      }

      recursionStack.delete(nodeId);
    };

    for (const nodeId of graph.keys()) {
      if (!visited.has(nodeId)) {
        dfs(nodeId, [nodeId]);
      }
    }

    return cycles;
  }
}