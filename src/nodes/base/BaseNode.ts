import { INodeType, INodeTypeDescription, INodeExecutionData, IExecuteFunctions } from '../../types/n8nTypes';

export interface NodeInputPort {
  name: string;
  type: 'main' | 'aux';
  displayName: string;
  required: boolean;
  maxConnections?: number;
}

export interface NodeOutputPort {
  name: string;
  type: 'main' | 'aux';
  displayName: string;
  maxConnections?: number;
}

export interface NodeExecutionContext {
  nodeId: string;
  executionId: string;
  inputData: INodeExecutionData[][];
  parameters: Record<string, any>;
  credentials?: Record<string, any>;
  workflow: {
    id: string;
    name: string;
  };
  mode: 'manual' | 'trigger' | 'webhook' | 'retry';
  timezone: string;
  retryCount: number;
  maxRetries: number;
}

export interface NodeExecutionResult {
  outputData: INodeExecutionData[][];
  error?: {
    message: string;
    description?: string;
    cause?: Error;
    timestamp: number;
    context?: Record<string, any>;
    httpCode?: number;
    retryable: boolean;
  };
  executionTime: number;
  itemsProcessed: number;
  metadata?: Record<string, any>;
}

export abstract class BaseNode implements INodeType {
  abstract description: INodeTypeDescription;
  
  protected inputPorts: NodeInputPort[] = [
    {
      name: 'main',
      type: 'main',
      displayName: 'Input',
      required: true,
      maxConnections: 1
    }
  ];
  
  protected outputPorts: NodeOutputPort[] = [
    {
      name: 'main',
      type: 'main',
      displayName: 'Output',
      maxConnections: -1 // unlimited
    }
  ];

  /**
   * Main execution method that handles the node logic
   */
  async execute(context: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const startTime = Date.now();
    const nodeContext = this.buildExecutionContext(context);
    
    try {
      // Pre-execution validation
      await this.validateInputs(nodeContext);
      
      // Execute the main node logic
      const result = await this.executeNode(nodeContext);
      
      // Post-execution validation
      await this.validateOutputs(result);
      
      // Log successful execution
      this.logExecution(nodeContext, result, startTime);
      
      return result.outputData;
    } catch (error) {
      // Handle and log errors
      const nodeError = this.handleExecutionError(error, nodeContext, startTime);
      throw nodeError;
    }
  }

  /**
   * Abstract method that each node must implement
   */
  protected abstract executeNode(context: NodeExecutionContext): Promise<NodeExecutionResult>;

  /**
   * Build execution context from n8n context
   */
  protected buildExecutionContext(context: IExecuteFunctions): NodeExecutionContext {
    const inputData = context.getInputData();
    const nodeId = context.getNodeParameter('nodeId', 0) as string || 'unknown';
    
    return {
      nodeId,
      executionId: `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      inputData: [inputData],
      parameters: this.extractParameters(context),
      workflow: {
        id: context.getNodeParameter('workflowId', 0) as string || 'unknown',
        name: context.getNodeParameter('workflowName', 0) as string || 'Unnamed Workflow'
      },
      mode: 'manual',
      timezone: 'UTC',
      retryCount: 0,
      maxRetries: 3
    };
  }

  /**
   * Extract all parameters for the node
   */
  protected extractParameters(context: IExecuteFunctions): Record<string, any> {
    const parameters: Record<string, any> = {};
    
    for (const property of this.description.properties) {
      try {
        parameters[property.name] = context.getNodeParameter(property.name, 0);
      } catch (error) {
        // Use default value if parameter is not set
        parameters[property.name] = property.default;
      }
    }
    
    return parameters;
  }

  /**
   * Validate input data before execution
   */
  protected async validateInputs(context: NodeExecutionContext): Promise<void> {
    // Check if required inputs are present
    for (const port of this.inputPorts) {
      if (port.required && (!context.inputData || context.inputData.length === 0)) {
        throw new Error(`Required input '${port.displayName}' is missing`);
      }
    }

    // Validate input data structure
    if (context.inputData && context.inputData.length > 0) {
      for (const inputSet of context.inputData) {
        for (const item of inputSet) {
          if (!item.json || typeof item.json !== 'object') {
            throw new Error('Invalid input data structure: json property must be an object');
          }
        }
      }
    }
  }

  /**
   * Validate output data after execution
   */
  protected async validateOutputs(result: NodeExecutionResult): Promise<void> {
    if (!result.outputData || !Array.isArray(result.outputData)) {
      throw new Error('Invalid output data: must be an array of arrays');
    }

    for (const outputSet of result.outputData) {
      if (!Array.isArray(outputSet)) {
        throw new Error('Invalid output data: each output must be an array');
      }
      
      for (const item of outputSet) {
        if (!item.json || typeof item.json !== 'object') {
          throw new Error('Invalid output item: json property must be an object');
        }
      }
    }
  }

  /**
   * Handle execution errors with proper context
   */
  protected handleExecutionError(error: any, context: NodeExecutionContext, startTime: number): Error {
    const executionTime = Date.now() - startTime;
    
    const nodeError = new Error(`Node execution failed: ${error.message}`);
    (nodeError as any).nodeId = context.nodeId;
    (nodeError as any).executionId = context.executionId;
    (nodeError as any).executionTime = executionTime;
    (nodeError as any).retryable = this.isRetryableError(error);
    (nodeError as any).originalError = error;
    
    // Log the error
    console.error(`[${context.nodeId}] Execution failed:`, {
      error: error.message,
      executionTime,
      retryCount: context.retryCount,
      stack: error.stack
    });
    
    return nodeError;
  }

  /**
   * Determine if an error is retryable
   */
  protected isRetryableError(error: any): boolean {
    // Network errors, timeouts, and temporary service unavailable errors are retryable
    const retryablePatterns = [
      /network/i,
      /timeout/i,
      /503/,
      /502/,
      /504/,
      /ECONNRESET/,
      /ENOTFOUND/,
      /ETIMEDOUT/
    ];
    
    const errorMessage = error.message || error.toString();
    return retryablePatterns.some(pattern => pattern.test(errorMessage));
  }

  /**
   * Log successful execution
   */
  protected logExecution(context: NodeExecutionContext, result: NodeExecutionResult, startTime: number): void {
    const executionTime = Date.now() - startTime;
    
    console.log(`[${context.nodeId}] Execution completed:`, {
      executionTime,
      itemsProcessed: result.itemsProcessed,
      outputItems: result.outputData.reduce((sum, set) => sum + set.length, 0)
    });
  }

  /**
   * Helper method to create standardized output data
   */
  protected createOutputData(data: any[]): INodeExecutionData[][] {
    const outputData: INodeExecutionData[] = data.map((item, index) => ({
      json: item,
      pairedItem: {
        item: index,
        input: 0
      }
    }));
    
    return [outputData];
  }

  /**
   * Helper method to get input items as simple objects
   */
  protected getInputItems(context: NodeExecutionContext): any[] {
    if (!context.inputData || context.inputData.length === 0) {
      return [];
    }
    
    return context.inputData[0].map(item => item.json);
  }

  /**
   * Get node input ports configuration
   */
  public getInputPorts(): NodeInputPort[] {
    return this.inputPorts;
  }

  /**
   * Get node output ports configuration
   */
  public getOutputPorts(): NodeOutputPort[] {
    return this.outputPorts;
  }
}