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
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'retrying' | 'recovering';
  startTime: number;
  endTime?: number;
  results: Record<string, any>;
  logs: ExecutionLog[];
  error?: string;
  retryCount?: number;
  maxRetries?: number;
  recoveryActions?: RecoveryAction[];
  failedNodes?: string[];
  checkpoints?: ExecutionCheckpoint[];
}

export interface ExecutionLog {
  timestamp: number;
  level: 'info' | 'warn' | 'error' | 'debug' | 'success';
  message: string;
  nodeId?: string;
  data?: any;
  errorCode?: string;
  stackTrace?: string;
}

export interface ExecutionContext {
  variables: Record<string, any>;
  results: Record<string, any>;
  logs: ExecutionLog[];
  cancelled: boolean;
  retryCount: number;
  maxRetries: number;
  checkpoints: ExecutionCheckpoint[];
  errorHandling: ErrorHandlingConfig;
}

export interface ExecutionCheckpoint {
  id: string;
  timestamp: number;
  nodeId: string;
  state: Record<string, any>;
  results: Record<string, any>;
}

export interface ErrorHandlingConfig {
  enableRetries: boolean;
  maxRetries: number;
  retryDelay: number;
  enableCheckpoints: boolean;
  enableRecovery: boolean;
  failFast: boolean;
  continueOnError: boolean;
}

export interface RecoveryAction {
  type: 'retry' | 'skip' | 'rollback' | 'alternative' | 'manual';
  nodeId: string;
  description: string;
  automatic: boolean;
  executed: boolean;
  timestamp?: number;
}

export interface NodeExecutionResult {
  success: boolean;
  data?: any;
  error?: Error;
  retryable?: boolean;
  checkpointData?: Record<string, any>;
}

export class WorkflowEngine {
  private executions = new Map<string, WorkflowExecution>();
  private nodeExecutors = new Map<string, NodeExecutor>();
  private errorHandlers = new Map<string, ErrorHandler>();
  private recoveryStrategies = new Map<string, RecoveryStrategy>();

  constructor() {
    this.nodeExecutors = new Map();
    this.errorHandlers = new Map();
    this.recoveryStrategies = new Map();
    this.executions = new Map();
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

  private registerDefaultErrorHandlers() {
    this.errorHandlers.set('default', new DefaultErrorHandler());
    this.errorHandlers.set('network', new NetworkErrorHandler());
    this.errorHandlers.set('validation', new ValidationErrorHandler());
    this.errorHandlers.set('timeout', new TimeoutErrorHandler());
    this.errorHandlers.set('resource', new ResourceErrorHandler());
    this.errorHandlers.set('authentication', new AuthenticationErrorHandler());
  }

  private registerDefaultRecoveryStrategies() {
    this.recoveryStrategies.set('default', new DefaultRecoveryStrategy());
    this.recoveryStrategies.set('retry', new RetryRecoveryStrategy());
    this.recoveryStrategies.set('fallback', new FallbackRecoveryStrategy());
    this.recoveryStrategies.set('circuit_breaker', new CircuitBreakerRecoveryStrategy());
    this.recoveryStrategies.set('graceful_degradation', new GracefulDegradationStrategy());
  }

  registerNodeExecutor(type: string, executor: NodeExecutor) {
    this.nodeExecutors.set(type, executor);
  }

  registerErrorHandler(type: string, handler: ErrorHandler) {
    this.errorHandlers.set(type, handler);
  }

  registerRecoveryStrategy(type: string, strategy: RecoveryStrategy) {
    this.recoveryStrategies.set(type, strategy);
  }

  async executeWorkflow(
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
    initialData: Record<string, any> = {},
    errorHandlingConfig: Partial<ErrorHandlingConfig> = {}
  ): Promise<WorkflowExecution> {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const workflowId = `workflow_${Date.now()}`;

    const defaultErrorConfig: ErrorHandlingConfig = {
      enableRetries: true,
      maxRetries: 3,
      retryDelay: 1000,
      enableCheckpoints: true,
      enableRecovery: true,
      failFast: false,
      continueOnError: false,
      ...errorHandlingConfig
    };

    const execution: WorkflowExecution = {
      id: executionId,
      workflowId,
      status: 'running',
      startTime: Date.now(),
      results: {},
      logs: [],
      retryCount: 0,
      maxRetries: defaultErrorConfig.maxRetries,
      recoveryActions: [],
      failedNodes: [],
      checkpoints: [],
    };

    this.executions.set(executionId, execution);

    try {
      const context: ExecutionContext = {
        variables: { ...initialData },
        results: {},
        logs: [],
        cancelled: false,
        retryCount: 0,
        maxRetries: defaultErrorConfig.maxRetries,
        checkpoints: [],
        errorHandling: defaultErrorConfig,
      };

      this.log(context, 'info', `Starting workflow execution: ${executionId}`, undefined, { config: defaultErrorConfig });

      const graph = new Map<string, WorkflowNode[]>();
      const nodeMap = new Map(nodes.map(node => [node.id, node]));
      const reverseEdges = new Map<string, string[]>();

      for (const edge of edges) {
        if (!graph.has(edge.source)) {
          graph.set(edge.source, []);
        }
        graph.get(edge.source)!.push(nodeMap.get(edge.target)!);

        if (!reverseEdges.has(edge.target)) {
          reverseEdges.set(edge.target, []);
        }
        reverseEdges.get(edge.target)!.push(edge.source);
      }

      const startNodes = nodes.filter(node => !reverseEdges.has(node.id));
      graph.set('', startNodes);

      await this.executeGraphWithErrorHandling(graph, context, execution);

      execution.status = context.cancelled ? 'cancelled' : 'completed';
      execution.endTime = Date.now();
      execution.results = context.results;
      execution.logs = context.logs;
      execution.checkpoints = context.checkpoints;

      this.log(context, 'success', `Workflow execution completed successfully: ${executionId}`, undefined, {
        duration: execution.endTime - execution.startTime,
        nodesExecuted: Object.keys(context.results).length,
        checkpoints: context.checkpoints.length
      });

    } catch (error) {
      await this.handleWorkflowError(execution, error as Error, nodes);
    }

    return execution;
  }

  private async executeGraphWithErrorHandling(
    graph: Map<string, WorkflowNode[]>,
    context: ExecutionContext,
    execution: WorkflowExecution
  ): Promise<void> {
    const executed = new Set<string>();
    const failed = new Set<string>();
    const queue = [...(graph.get('') || [])];

    while (queue.length > 0 && !context.cancelled) {
      const node = queue.shift()!;
      
      if (executed.has(node.id) || failed.has(node.id)) continue;

      try {
        // Create checkpoint before execution if enabled
        if (context.errorHandling.enableCheckpoints) {
          await this.createCheckpoint(node.id, context);
        }

        // Execute node with error handling
        const result = await this.executeNodeWithErrorHandling(node, context, execution);
        
        if (result.success) {
          executed.add(node.id);
          context.results[node.id] = result.data;
          
          this.log(context, 'success', `Node executed successfully: ${node.data.label}`, node.id, result.data);
          
          // Add dependent nodes to queue
          const dependents = graph.get(node.id) || [];
          queue.push(...dependents);
        } else {
          failed.add(node.id);
          execution.failedNodes = execution.failedNodes || [];
          execution.failedNodes.push(node.id);
          
          if (context.errorHandling.failFast) {
            throw result.error || new Error(`Node ${node.id} failed`);
          }
          
          if (!context.errorHandling.continueOnError) {
            // Try recovery strategies
            const recovered = await this.attemptNodeRecovery(node, result.error!, context, execution);
            if (!recovered) {
              throw result.error || new Error(`Node ${node.id} failed and recovery unsuccessful`);
            }
          }
        }
      } catch (error) {
        failed.add(node.id);
        execution.failedNodes = execution.failedNodes || [];
        execution.failedNodes.push(node.id);
        
        this.log(context, 'error', `Node execution failed: ${node.data.label}`, node.id, { 
          error: error instanceof Error ? error.message : 'Unknown error',
          stackTrace: error instanceof Error ? error.stack : undefined
        });

        if (context.errorHandling.failFast) {
          throw error;
        }
      }
    }
  }

  private async executeNodeWithErrorHandling(
    node: WorkflowNode,
    context: ExecutionContext,
    execution: WorkflowExecution
  ): Promise<NodeExecutionResult> {
    const executor = this.nodeExecutors.get(node.type);
    if (!executor) {
      throw new Error(`No executor found for node type: ${node.type}`);
    }

    let lastError: Error | undefined;
    let retryCount = 0;
    const maxRetries = context.errorHandling.enableRetries ? context.errorHandling.maxRetries : 0;

    while (retryCount <= maxRetries) {
      try {
        this.log(context, 'info', `Executing node: ${node.data.label}${retryCount > 0 ? ` (retry ${retryCount})` : ''}`, node.id);
        
        const result = await executor.execute(node, context);
        
        return {
          success: true,
          data: result,
          checkpointData: { nodeId: node.id, result, timestamp: Date.now() }
        };
      } catch (error) {
        lastError = error as Error;
        retryCount++;
        
        this.log(context, 'warn', `Node execution failed (attempt ${retryCount}): ${lastError.message}`, node.id, {
          error: lastError.message,
          retryCount,
          maxRetries
        });

        if (retryCount <= maxRetries) {
          // Apply error-specific handling
          const errorHandler = this.getErrorHandler(lastError);
          const shouldRetry = await errorHandler.shouldRetry(lastError, retryCount, maxRetries);
          
          if (shouldRetry) {
            const delay = this.calculateRetryDelay(retryCount, context.errorHandling.retryDelay);
            await this.delay(delay);
            continue;
          }
        }
        
        break;
      }
    }

    return {
      success: false,
      error: lastError,
      retryable: this.isRetryableError(lastError!)
    };
  }

  private async handleWorkflowError(execution: WorkflowExecution, error: Error, nodes: WorkflowNode[]): Promise<void> {
    execution.status = 'failed';
    execution.endTime = Date.now();
    execution.error = error.message;
    
    // Log detailed error information
    const errorLog: ExecutionLog = {
      timestamp: Date.now(),
      level: 'error',
      message: `Workflow execution failed: ${error.message}`,
      data: {
        error: error.message,
        stackTrace: error.stack,
        failedNodes: execution.failedNodes,
        executionTime: execution.endTime - execution.startTime
      },
      errorCode: this.getErrorCode(error),
      stackTrace: error.stack
    };
    
    execution.logs = execution.logs || [];
    execution.logs.push(errorLog);

    // Generate recovery actions
    execution.recoveryActions = await this.generateRecoveryActions(error, execution, nodes);
  }

  private async attemptNodeRecovery(
    node: WorkflowNode,
    error: Error,
    context: ExecutionContext,
    execution: WorkflowExecution
  ): Promise<boolean> {
    const recoveryStrategies = this.getRecoveryStrategies(error, node);
    
    for (const strategy of recoveryStrategies) {
      try {
        this.log(context, 'info', `Attempting recovery strategy: ${strategy.name}`, node.id);
        
        const recovered = await strategy.recover(node, error, context);
        if (recovered) {
          this.log(context, 'success', `Recovery successful with strategy: ${strategy.name}`, node.id);
          
          const recoveryAction: RecoveryAction = {
            type: 'retry',
            nodeId: node.id,
            description: `Recovered using ${strategy.name}`,
            automatic: true,
            executed: true,
            timestamp: Date.now()
          };
          
          execution.recoveryActions = execution.recoveryActions || [];
          execution.recoveryActions.push(recoveryAction);
          
          return true;
        }
      } catch (recoveryError) {
        this.log(context, 'warn', `Recovery strategy failed: ${strategy.name}`, node.id, {
          error: recoveryError instanceof Error ? recoveryError.message : 'Unknown error'
        });
      }
    }
    
    return false;
  }

  private async generateRecoveryActions(
    error: Error,
    execution: WorkflowExecution,
    nodes: WorkflowNode[]
  ): Promise<RecoveryAction[]> {
    const actions: RecoveryAction[] = [];
    
    // Retry action for retryable errors
    if (this.isRetryableError(error) && (execution.retryCount || 0) < (execution.maxRetries || 3)) {
      actions.push({
        type: 'retry',
        nodeId: 'workflow',
        description: 'Retry entire workflow execution',
        automatic: false,
        executed: false
      });
    }
    
    // Rollback to last checkpoint
    if (execution.checkpoints && execution.checkpoints.length > 0) {
      actions.push({
        type: 'rollback',
        nodeId: 'workflow',
        description: 'Rollback to last successful checkpoint',
        automatic: false,
        executed: false
      });
    }
    
    // Skip failed nodes
    if (execution.failedNodes && execution.failedNodes.length > 0) {
      actions.push({
        type: 'skip',
        nodeId: 'workflow',
        description: 'Skip failed nodes and continue execution',
        automatic: false,
        executed: false
      });
    }
    
    // Manual intervention
    actions.push({
      type: 'manual',
      nodeId: 'workflow',
      description: 'Manual intervention required',
      automatic: false,
      executed: false
    });
    
    return actions;
  }

  private async createCheckpoint(nodeId: string, context: ExecutionContext): Promise<void> {
    const checkpoint: ExecutionCheckpoint = {
      id: `checkpoint_${Date.now()}_${nodeId}`,
      timestamp: Date.now(),
      nodeId,
      state: { ...context.variables },
      results: { ...context.results }
    };
    
    context.checkpoints.push(checkpoint);
    this.log(context, 'debug', `Checkpoint created for node: ${nodeId}`, nodeId, { checkpointId: checkpoint.id });
  }

  private getErrorHandler(error: Error): ErrorHandler {
    const errorMessage = error.message.toLowerCase();
    
    if (errorMessage.includes('network') || errorMessage.includes('fetch') || errorMessage.includes('timeout')) {
      return this.errorHandlers.get('network') || new DefaultErrorHandler();
    }
    if (errorMessage.includes('validation') || errorMessage.includes('invalid')) {
      return this.errorHandlers.get('validation') || new DefaultErrorHandler();
    }
    if (errorMessage.includes('timeout')) {
      return this.errorHandlers.get('timeout') || new DefaultErrorHandler();
    }
    if (errorMessage.includes('auth') || errorMessage.includes('unauthorized')) {
      return this.errorHandlers.get('authentication') || new DefaultErrorHandler();
    }
    
    return new DefaultErrorHandler();
  }

  private getRecoveryStrategies(error: Error, node: WorkflowNode): RecoveryStrategy[] {
    const strategies: RecoveryStrategy[] = [];
    
    if (this.isRetryableError(error)) {
      strategies.push(this.recoveryStrategies.get('retry') || new DefaultRecoveryStrategy());
    }
    
    if (node.type === 'http' || node.type === 'api') {
      strategies.push(this.recoveryStrategies.get('circuit_breaker') || new DefaultRecoveryStrategy());
    }
    
    strategies.push(this.recoveryStrategies.get('graceful_degradation') || new DefaultRecoveryStrategy());
    
    return strategies;
  }

  private isRetryableError(error: Error): boolean {
    const retryablePatterns = [
      /network/i,
      /timeout/i,
      /connection/i,
      /temporary/i,
      /rate limit/i,
      /503/,
      /502/,
      /504/
    ];
    
    return retryablePatterns.some(pattern => pattern.test(error.message));
  }

  private calculateRetryDelay(retryCount: number, baseDelay: number): number {
    // Exponential backoff with jitter
    const exponentialDelay = baseDelay * Math.pow(2, retryCount - 1);
    const jitter = Math.random() * 0.1 * exponentialDelay;
    return Math.min(exponentialDelay + jitter, 30000); // Max 30 seconds
  }

  private getErrorCode(error: Error): string {
    if (error.message.includes('network')) return 'NETWORK_ERROR';
    if (error.message.includes('timeout')) return 'TIMEOUT_ERROR';
    if (error.message.includes('validation')) return 'VALIDATION_ERROR';
    if (error.message.includes('auth')) return 'AUTH_ERROR';
    return 'UNKNOWN_ERROR';
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
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

// Export singleton instance
export const workflowEngine = new WorkflowEngine();

import {
  ErrorHandler,
  DefaultErrorHandler,
  NetworkErrorHandler,
  ValidationErrorHandler,
  TimeoutErrorHandler,
  ResourceErrorHandler,
  AuthenticationErrorHandler,
  RecoveryStrategy,
  DefaultRecoveryStrategy,
  RetryRecoveryStrategy,
  FallbackRecoveryStrategy,
  CircuitBreakerRecoveryStrategy,
  GracefulDegradationStrategy,
} from './workflowErrorHandlers';

import {
  NodeExecutor,
  HttpRequestExecutor,
  DataTransformExecutor,
  ConditionExecutor,
  TimerExecutor,
  FileOperationExecutor,
  CodeExecutionExecutor,
  EmailExecutor,
  DatabaseExecutor,
} from './nodeExecutors';