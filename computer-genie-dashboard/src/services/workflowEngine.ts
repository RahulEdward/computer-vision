export interface WorkflowNode {
  id: string;
  type: string;
  data: {
    label: string;
    config?: Record<string, any>;
    code?: string;
  };
  position: { x: number; y: number };
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  startTime: number;
  endTime?: number;
  results: Record<string, any>;
  logs: ExecutionLog[];
  error?: string;
}

export interface ExecutionLog {
  timestamp: number;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  nodeId?: string;
  data?: any;
}

export interface ExecutionContext {
  variables: Record<string, any>;
  results: Record<string, any>;
  logs: ExecutionLog[];
  cancelled: boolean;
}

export class WorkflowEngine {
  private executions = new Map<string, WorkflowExecution>();
  private nodeExecutors = new Map<string, NodeExecutor>();

  constructor() {
    this.registerDefaultExecutors();
  }

  private registerDefaultExecutors() {
    // HTTP Request Node
    this.registerNodeExecutor('http', new HttpRequestExecutor());
    
    // Data Transform Node
    this.registerNodeExecutor('transform', new DataTransformExecutor());
    
    // Condition Node
    this.registerNodeExecutor('condition', new ConditionExecutor());
    
    // Timer/Delay Node
    this.registerNodeExecutor('timer', new TimerExecutor());
    
    // File Operation Node
    this.registerNodeExecutor('file', new FileOperationExecutor());
    
    // Code Execution Node
    this.registerNodeExecutor('code', new CodeExecutionExecutor());
    
    // Email Node
    this.registerNodeExecutor('email', new EmailExecutor());
    
    // Database Node
    this.registerNodeExecutor('database', new DatabaseExecutor());
  }

  registerNodeExecutor(type: string, executor: NodeExecutor) {
    this.nodeExecutors.set(type, executor);
  }

  async executeWorkflow(
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
    initialData: Record<string, any> = {}
  ): Promise<WorkflowExecution> {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const workflowId = `workflow_${Date.now()}`;

    const execution: WorkflowExecution = {
      id: executionId,
      workflowId,
      status: 'running',
      startTime: Date.now(),
      results: {},
      logs: [],
    };

    this.executions.set(executionId, execution);

    try {
      const context: ExecutionContext = {
        variables: { ...initialData },
        results: {},
        logs: [],
        cancelled: false,
      };

      this.log(context, 'info', `Starting workflow execution: ${executionId}`);

      // Build execution graph
      const graph = this.buildExecutionGraph(nodes, edges);
      
      // Execute nodes in topological order
      await this.executeGraph(graph, context);

      execution.status = context.cancelled ? 'cancelled' : 'completed';
      execution.endTime = Date.now();
      execution.results = context.results;
      execution.logs = context.logs;

      this.log(context, 'info', `Workflow execution completed: ${executionId}`);

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.error = error instanceof Error ? error.message : 'Unknown error';
      execution.logs = execution.logs || [];
      
      this.log(execution as any, 'error', `Workflow execution failed: ${execution.error}`);
    }

    return execution;
  }

  private buildExecutionGraph(nodes: WorkflowNode[], edges: WorkflowEdge[]) {
    const graph = new Map<string, { node: WorkflowNode; dependencies: string[]; dependents: string[] }>();
    
    // Initialize nodes
    nodes.forEach(node => {
      graph.set(node.id, {
        node,
        dependencies: [],
        dependents: [],
      });
    });

    // Build dependencies
    edges.forEach(edge => {
      const source = graph.get(edge.source);
      const target = graph.get(edge.target);
      
      if (source && target) {
        target.dependencies.push(edge.source);
        source.dependents.push(edge.target);
      }
    });

    return graph;
  }

  private async executeGraph(
    graph: Map<string, { node: WorkflowNode; dependencies: string[]; dependents: string[] }>,
    context: ExecutionContext
  ) {
    const executed = new Set<string>();
    const executing = new Set<string>();

    const executeNode = async (nodeId: string): Promise<void> => {
      if (executed.has(nodeId) || executing.has(nodeId) || context.cancelled) {
        return;
      }

      const graphNode = graph.get(nodeId);
      if (!graphNode) return;

      // Wait for dependencies
      for (const depId of graphNode.dependencies) {
        if (!executed.has(depId)) {
          await executeNode(depId);
        }
      }

      if (context.cancelled) return;

      executing.add(nodeId);
      
      try {
        this.log(context, 'info', `Executing node: ${graphNode.node.data.label}`, nodeId);
        
        const executor = this.nodeExecutors.get(graphNode.node.type);
        if (!executor) {
          throw new Error(`No executor found for node type: ${graphNode.node.type}`);
        }

        const result = await executor.execute(graphNode.node, context);
        context.results[nodeId] = result;
        
        this.log(context, 'info', `Node completed: ${graphNode.node.data.label}`, nodeId, result);
        
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        this.log(context, 'error', `Node failed: ${errorMessage}`, nodeId);
        throw error;
      } finally {
        executing.delete(nodeId);
        executed.add(nodeId);
      }
    };

    // Find root nodes (nodes with no dependencies)
    const rootNodes = Array.from(graph.keys()).filter(
      nodeId => graph.get(nodeId)!.dependencies.length === 0
    );

    // Execute all root nodes in parallel
    await Promise.all(rootNodes.map(executeNode));
  }

  private log(context: ExecutionContext, level: ExecutionLog['level'], message: string, nodeId?: string, data?: any) {
    const log: ExecutionLog = {
      timestamp: Date.now(),
      level,
      message,
      nodeId,
      data,
    };
    context.logs.push(log);
  }

  getExecution(executionId: string): WorkflowExecution | undefined {
    return this.executions.get(executionId);
  }

  cancelExecution(executionId: string): boolean {
    const execution = this.executions.get(executionId);
    if (execution && execution.status === 'running') {
      execution.status = 'cancelled';
      return true;
    }
    return false;
  }

  getExecutions(): WorkflowExecution[] {
    return Array.from(this.executions.values());
  }
}

// Base class for node executors
export abstract class NodeExecutor {
  abstract execute(node: WorkflowNode, context: ExecutionContext): Promise<any>;
}

// HTTP Request Executor
class HttpRequestExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { url, method = 'GET', headers = {}, body } = node.data.config || {};
    
    if (!url) {
      throw new Error('URL is required for HTTP request');
    }

    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return { status: response.status, data };
  }
}

// Data Transform Executor
class DataTransformExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { transformCode, inputKey } = node.data.config || {};
    
    if (!transformCode) {
      throw new Error('Transform code is required');
    }

    const inputData = inputKey ? context.results[inputKey] : context.variables;
    
    // Create a safe execution environment
    const func = new Function('data', 'context', `
      return (function() {
        ${transformCode}
      })();
    `);

    return func(inputData, context);
  }
}

// Condition Executor
class ConditionExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { condition, trueValue, falseValue } = node.data.config || {};
    
    if (!condition) {
      throw new Error('Condition is required');
    }

    // Create a safe execution environment for condition evaluation
    const func = new Function('context', `
      return (${condition});
    `);

    const result = func(context);
    return result ? trueValue : falseValue;
  }
}

// Timer Executor
class TimerExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { delay = 1000 } = node.data.config || {};
    
    await new Promise(resolve => setTimeout(resolve, delay));
    return { delayed: delay };
  }
}

// File Operation Executor
class FileOperationExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { operation, content, filename } = node.data.config || {};
    
    if (operation === 'create') {
      // In a real implementation, this would write to the file system
      // For now, we'll simulate file creation
      return { 
        operation: 'create',
        filename,
        size: content?.length || 0,
        created: true 
      };
    } else if (operation === 'read') {
      // Simulate file reading
      return { 
        operation: 'read',
        filename,
        content: `Simulated content of ${filename}`,
        size: 100 
      };
    }
    
    throw new Error(`Unsupported file operation: ${operation}`);
  }
}

// Code Execution Executor
class CodeExecutionExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { code } = node.data;
    
    if (!code) {
      throw new Error('Code is required for execution');
    }

    // Create a safe execution environment
    const logs: string[] = [];
    const originalConsole = console.log;
    
    console.log = (...args) => {
      logs.push(args.join(' '));
    };

    try {
      const func = new Function('context', `
        ${code}
      `);
      
      const result = func(context);
      console.log = originalConsole;
      
      return { result, logs };
    } catch (error) {
      console.log = originalConsole;
      throw error;
    }
  }
}

// Email Executor
class EmailExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { to, subject, body } = node.data.config || {};
    
    if (!to || !subject) {
      throw new Error('To and subject are required for email');
    }

    // Simulate email sending
    return {
      sent: true,
      to,
      subject,
      timestamp: Date.now(),
      messageId: `msg_${Math.random().toString(36).substr(2, 9)}`,
    };
  }
}

// Database Executor
class DatabaseExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { operation, query, data } = node.data.config || {};
    
    // Simulate database operations
    if (operation === 'select') {
      return {
        operation: 'select',
        query,
        results: [
          { id: 1, name: 'Sample Record 1', created: Date.now() },
          { id: 2, name: 'Sample Record 2', created: Date.now() },
        ],
        count: 2,
      };
    } else if (operation === 'insert') {
      return {
        operation: 'insert',
        insertedId: Math.floor(Math.random() * 1000),
        data,
        success: true,
      };
    }
    
    throw new Error(`Unsupported database operation: ${operation}`);
  }
}

// Export singleton instance
export const workflowEngine = new WorkflowEngine();