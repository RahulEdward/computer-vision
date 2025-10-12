import { BaseNode, NodeExecutionContext, NodeExecutionResult } from '../nodes/base/BaseNode';
import { NodeConnectionSystem } from './NodeConnectionSystem';
import { NodeRegistry } from '../nodes/NodeRegistry';
import { WorkflowNode, WorkflowEdge, WorkflowExecution, ExecutionLog } from './workflowEngine';

export interface EnhancedExecutionContext {
  executionId: string;
  workflowId: string;
  variables: Record<string, any>;
  nodeResults: Map<string, NodeExecutionResult>;
  connections: Map<string, any[]>;
  startTime: number;
  currentNode?: string;
  cancelled: boolean;
  errorHandling: {
    continueOnError: boolean;
    maxRetries: number;
    retryDelay: number;
  };
}

export interface WorkflowExecutionPlan {
  executionOrder: string[];
  dependencies: Map<string, string[]>;
  parallelGroups: string[][];
  criticalPath: string[];
}

export interface NodeExecutionMetrics {
  nodeId: string;
  nodeType: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  inputSize: number;
  outputSize: number;
  memoryUsage?: number;
  retryCount: number;
  errorMessage?: string;
}

export class EnhancedWorkflowEngine {
  private nodeRegistry: NodeRegistry;
  private connectionSystem: NodeConnectionSystem;
  private activeExecutions: Map<string, EnhancedExecutionContext> = new Map();
  private executionMetrics: Map<string, NodeExecutionMetrics[]> = new Map();
  private executionPlans: Map<string, WorkflowExecutionPlan> = new Map();

  constructor() {
    this.nodeRegistry = NodeRegistry.getInstance();
    this.connectionSystem = new NodeConnectionSystem();
  }

  /**
   * Create an execution plan for the workflow
   */
  private createExecutionPlan(nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowExecutionPlan {
    const dependencies = new Map<string, string[]>();
    const incomingEdges = new Map<string, WorkflowEdge[]>();
    
    // Initialize dependencies
    for (const node of nodes) {
      dependencies.set(node.id, []);
      incomingEdges.set(node.id, []);
    }

    // Build dependency graph
    for (const edge of edges) {
      const targetDeps = dependencies.get(edge.target) || [];
      targetDeps.push(edge.source);
      dependencies.set(edge.target, targetDeps);
      
      const targetEdges = incomingEdges.get(edge.target) || [];
      targetEdges.push(edge);
      incomingEdges.set(edge.target, targetEdges);
    }

    // Topological sort for execution order
    const executionOrder = this.topologicalSort(nodes, dependencies);
    
    // Identify parallel execution groups
    const parallelGroups = this.identifyParallelGroups(nodes, dependencies);
    
    // Calculate critical path
    const criticalPath = this.calculateCriticalPath(nodes, dependencies);

    return {
      executionOrder,
      dependencies,
      parallelGroups,
      criticalPath
    };
  }

  private topologicalSort(nodes: WorkflowNode[], dependencies: Map<string, string[]>): string[] {
    const visited = new Set<string>();
    const visiting = new Set<string>();
    const result: string[] = [];

    const visit = (nodeId: string) => {
      if (visiting.has(nodeId)) {
        throw new Error(`Circular dependency detected involving node: ${nodeId}`);
      }
      if (visited.has(nodeId)) {
        return;
      }

      visiting.add(nodeId);
      const deps = dependencies.get(nodeId) || [];
      for (const dep of deps) {
        visit(dep);
      }
      visiting.delete(nodeId);
      visited.add(nodeId);
      result.push(nodeId);
    };

    for (const node of nodes) {
      if (!visited.has(node.id)) {
        visit(node.id);
      }
    }

    return result.reverse();
  }

  private identifyParallelGroups(nodes: WorkflowNode[], dependencies: Map<string, string[]>): string[][] {
    const groups: string[][] = [];
    const processed = new Set<string>();
    
    for (const node of nodes) {
      if (processed.has(node.id)) continue;
      
      const deps = dependencies.get(node.id) || [];
      if (deps.length === 0) {
        // Find all nodes with no dependencies that can run in parallel
        const parallelNodes = nodes
          .filter(n => !processed.has(n.id) && (dependencies.get(n.id) || []).length === 0)
          .map(n => n.id);
        
        if (parallelNodes.length > 1) {
          groups.push(parallelNodes);
          parallelNodes.forEach(id => processed.add(id));
        }
      }
    }
    
    return groups;
  }

  private calculateCriticalPath(nodes: WorkflowNode[], dependencies: Map<string, string[]>): string[] {
    // Simplified critical path calculation
    // In a real implementation, this would consider node execution times
    const longestPath: string[] = [];
    const visited = new Set<string>();

    const findLongestPath = (nodeId: string, currentPath: string[]): string[] => {
      if (visited.has(nodeId)) return currentPath;
      
      visited.add(nodeId);
      const deps = dependencies.get(nodeId) || [];
      
      if (deps.length === 0) {
        return [...currentPath, nodeId];
      }

      let longest = currentPath;
      for (const dep of deps) {
        const path = findLongestPath(dep, [...currentPath, nodeId]);
        if (path.length > longest.length) {
          longest = path;
        }
      }
      
      return longest;
    };

    for (const node of nodes) {
      const path = findLongestPath(node.id, []);
      if (path.length > longestPath.length) {
        longestPath.splice(0, longestPath.length, ...path);
      }
    }

    return longestPath;
  }

  /**
   * Execute a workflow with enhanced error handling and monitoring
   */
  async executeWorkflow(
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
    initialData: Record<string, any> = {},
    options: {
      continueOnError?: boolean;
      maxRetries?: number;
      retryDelay?: number;
      enableParallelExecution?: boolean;
    } = {}
  ): Promise<WorkflowExecution> {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const workflowId = `workflow_${Date.now()}`;

    // Validate workflow before execution
    const validationResult = this.connectionSystem.validateWorkflowDataFlow(nodes, edges);
    if (!validationResult.isValid) {
      throw new Error(`Workflow validation failed: ${validationResult.errors.join(', ')}`);
    }

    // Create execution plan
    const executionPlan = this.createExecutionPlan(nodes, edges);
    this.executionPlans.set(executionId, executionPlan);

    // Initialize execution context
    const context: EnhancedExecutionContext = {
      executionId,
      workflowId,
      variables: { ...initialData },
      nodeResults: new Map(),
      connections: new Map(),
      startTime: Date.now(),
      cancelled: false,
      errorHandling: {
        continueOnError: options.continueOnError || false,
        maxRetries: options.maxRetries || 3,
        retryDelay: options.retryDelay || 1000
      }
    };

    this.activeExecutions.set(executionId, context);
    this.executionMetrics.set(executionId, []);

    const execution: WorkflowExecution = {
      id: executionId,
      workflowId,
      status: 'running',
      startTime: Date.now(),
      results: {},
      logs: [],
      retryCount: 0,
      maxRetries: context.errorHandling.maxRetries,
      recoveryActions: [],
      failedNodes: [],
      checkpoints: []
    };

    try {
      // Execute nodes according to the execution plan
      if (options.enableParallelExecution && executionPlan.parallelGroups.length > 0) {
        await this.executeWithParallelism(nodes, edges, context, execution);
      } else {
        await this.executeSequentially(nodes, edges, context, execution);
      }

      execution.status = 'completed';
      execution.endTime = Date.now();
      
    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.error = error instanceof Error ? error.message : String(error);
      
      this.logExecution(execution, 'error', `Workflow execution failed: ${execution.error}`);
    } finally {
      this.activeExecutions.delete(executionId);
    }

    return execution;
  }

  private async executeSequentially(
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
    context: EnhancedExecutionContext,
    execution: WorkflowExecution
  ): Promise<void> {
    const executionPlan = this.executionPlans.get(context.executionId)!;
    
    for (const nodeId of executionPlan.executionOrder) {
      if (context.cancelled) {
        break;
      }

      const node = nodes.find(n => n.id === nodeId);
      if (!node) {
        throw new Error(`Node not found: ${nodeId}`);
      }

      await this.executeNode(node, edges, context, execution);
    }
  }

  private async executeWithParallelism(
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
    context: EnhancedExecutionContext,
    execution: WorkflowExecution
  ): Promise<void> {
    const executionPlan = this.executionPlans.get(context.executionId)!;
    const completed = new Set<string>();
    
    // Execute parallel groups first
    for (const group of executionPlan.parallelGroups) {
      const promises = group.map(async (nodeId) => {
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
          await this.executeNode(node, edges, context, execution);
          completed.add(nodeId);
        }
      });
      
      await Promise.all(promises);
    }

    // Execute remaining nodes sequentially
    for (const nodeId of executionPlan.executionOrder) {
      if (completed.has(nodeId) || context.cancelled) {
        continue;
      }

      const node = nodes.find(n => n.id === nodeId);
      if (node) {
        await this.executeNode(node, edges, context, execution);
      }
    }
  }

  private async executeNode(
    node: WorkflowNode,
    edges: WorkflowEdge[],
    context: EnhancedExecutionContext,
    execution: WorkflowExecution
  ): Promise<void> {
    const startTime = Date.now();
    context.currentNode = node.id;

    // Initialize metrics
    const metrics: NodeExecutionMetrics = {
      nodeId: node.id,
      nodeType: node.type,
      startTime,
      status: 'running',
      inputSize: 0,
      outputSize: 0,
      retryCount: 0
    };

    this.executionMetrics.get(context.executionId)!.push(metrics);

    try {
      // Get node instance from registry
      const nodeInstance = this.nodeRegistry.getNode(node.type);
      if (!nodeInstance || !(nodeInstance instanceof BaseNode)) {
        throw new Error(`Node type not found or not a BaseNode: ${node.type}`);
      }

      // Prepare input data from connected nodes
      const inputData = this.prepareNodeInputData(node, edges, context);
      metrics.inputSize = JSON.stringify(inputData).length;

      // Create node execution context
      const nodeContext: NodeExecutionContext = {
        nodeId: node.id,
        executionId: context.executionId,
        inputData,
        parameters: node.data || {},
        credentials: {},
        variables: context.variables
      };

      // Execute node with retry logic
      let result: NodeExecutionResult;
      let retryCount = 0;
      
      while (retryCount <= context.errorHandling.maxRetries) {
        try {
          result = await nodeInstance.execute(nodeContext);
          break;
        } catch (error) {
          retryCount++;
          metrics.retryCount = retryCount;
          
          if (retryCount > context.errorHandling.maxRetries) {
            throw error;
          }
          
          this.logExecution(execution, 'warning', 
            `Node ${node.id} failed, retrying (${retryCount}/${context.errorHandling.maxRetries}): ${error}`);
          
          await new Promise(resolve => setTimeout(resolve, context.errorHandling.retryDelay));
        }
      }

      // Store results
      context.nodeResults.set(node.id, result!);
      execution.results[node.id] = result!.outputData;
      metrics.outputSize = JSON.stringify(result!.outputData).length;
      metrics.status = 'completed';
      metrics.endTime = Date.now();
      metrics.duration = metrics.endTime - metrics.startTime;

      this.logExecution(execution, 'info', `Node ${node.id} executed successfully`);

    } catch (error) {
      metrics.status = 'failed';
      metrics.endTime = Date.now();
      metrics.duration = metrics.endTime - metrics.startTime;
      metrics.errorMessage = error instanceof Error ? error.message : String(error);

      execution.failedNodes.push(node.id);
      
      if (!context.errorHandling.continueOnError) {
        throw error;
      }
      
      this.logExecution(execution, 'error', `Node ${node.id} failed: ${metrics.errorMessage}`);
    }
  }

  private prepareNodeInputData(
    node: WorkflowNode,
    edges: WorkflowEdge[],
    context: EnhancedExecutionContext
  ): any[] {
    const inputEdges = edges.filter(edge => edge.target === node.id);
    const inputData: any[] = [];

    for (const edge of inputEdges) {
      const sourceResult = context.nodeResults.get(edge.source);
      if (sourceResult) {
        // Handle specific output port if specified
        if (edge.sourceHandle && sourceResult.outputData[edge.sourceHandle]) {
          inputData.push(sourceResult.outputData[edge.sourceHandle]);
        } else {
          inputData.push(sourceResult.outputData);
        }
      }
    }

    return inputData.length > 0 ? inputData : [{}];
  }

  private logExecution(execution: WorkflowExecution, level: 'info' | 'warning' | 'error', message: string): void {
    const log: ExecutionLog = {
      timestamp: Date.now(),
      level,
      message,
      nodeId: execution.id
    };
    
    execution.logs.push(log);
  }

  /**
   * Cancel a running workflow execution
   */
  cancelExecution(executionId: string): boolean {
    const context = this.activeExecutions.get(executionId);
    if (context) {
      context.cancelled = true;
      return true;
    }
    return false;
  }

  /**
   * Get execution metrics for a workflow
   */
  getExecutionMetrics(executionId: string): NodeExecutionMetrics[] {
    return this.executionMetrics.get(executionId) || [];
  }

  /**
   * Get execution plan for a workflow
   */
  getExecutionPlan(executionId: string): WorkflowExecutionPlan | null {
    return this.executionPlans.get(executionId) || null;
  }

  /**
   * Get active executions
   */
  getActiveExecutions(): string[] {
    return Array.from(this.activeExecutions.keys());
  }
}