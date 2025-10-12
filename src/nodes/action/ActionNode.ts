import { BaseNode, NodeExecutionContext, NodeExecutionResult } from '../base/BaseNode';
import { INodeTypeDescription, NODE_CATEGORIES } from '../../types/n8nTypes';

export abstract class ActionNode extends BaseNode {
  constructor() {
    super();
    // Action nodes have standard input/output configuration
    this.inputPorts = [
      {
        name: 'main',
        type: 'main',
        displayName: 'Input',
        required: true,
        maxConnections: 1
      }
    ];
    this.outputPorts = [
      {
        name: 'main',
        type: 'main',
        displayName: 'Output',
        maxConnections: -1
      }
    ];
  }

  /**
   * Process each input item through the action
   */
  protected async executeNode(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    const inputItems = this.getInputItems(context);
    const processedItems: any[] = [];

    for (let i = 0; i < inputItems.length; i++) {
      try {
        const result = await this.processItem(inputItems[i], i, context);
        if (result !== null && result !== undefined) {
          processedItems.push(result);
        }
      } catch (error) {
        // Handle item-level errors
        const errorItem = {
          error: {
            message: error.message,
            itemIndex: i,
            timestamp: new Date().toISOString()
          },
          originalItem: inputItems[i]
        };
        processedItems.push(errorItem);
      }
    }

    return {
      outputData: this.createOutputData(processedItems),
      executionTime: Date.now() - startTime,
      itemsProcessed: inputItems.length,
      metadata: {
        successfulItems: processedItems.filter(item => !item.error).length,
        failedItems: processedItems.filter(item => item.error).length
      }
    };
  }

  /**
   * Process a single item - to be implemented by specific action nodes
   */
  protected abstract processItem(item: any, index: number, context: NodeExecutionContext): Promise<any>;
}

/**
 * HTTP Request Node - make HTTP requests to external APIs
 */
export class HttpRequestNode extends ActionNode {
  description: INodeTypeDescription = {
    displayName: 'HTTP Request',
    name: 'httpRequest',
    icon: 'fa:globe',
    group: [NODE_CATEGORIES.ACTION],
    version: 1,
    description: 'Make HTTP requests to external APIs',
    defaults: {
      name: 'HTTP Request'
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Method',
        name: 'method',
        type: 'options',
        options: [
          { name: 'GET', value: 'GET' },
          { name: 'POST', value: 'POST' },
          { name: 'PUT', value: 'PUT' },
          { name: 'DELETE', value: 'DELETE' },
          { name: 'PATCH', value: 'PATCH' }
        ],
        default: 'GET',
        description: 'HTTP method to use'
      },
      {
        displayName: 'URL',
        name: 'url',
        type: 'string',
        default: '',
        placeholder: 'https://api.example.com/data',
        description: 'URL to make the request to',
        required: true
      },
      {
        displayName: 'Headers',
        name: 'headers',
        type: 'json',
        default: '{}',
        description: 'Headers to send with the request'
      },
      {
        displayName: 'Body',
        name: 'body',
        type: 'json',
        default: '{}',
        description: 'Request body (for POST, PUT, PATCH)',
        displayOptions: {
          show: {
            method: ['POST', 'PUT', 'PATCH']
          }
        }
      },
      {
        displayName: 'Timeout',
        name: 'timeout',
        type: 'number',
        default: 30000,
        description: 'Request timeout in milliseconds'
      }
    ]
  };

  protected async processItem(item: any, index: number, context: NodeExecutionContext): Promise<any> {
    const method = context.parameters.method;
    const url = this.resolveExpression(context.parameters.url, item);
    const headers = this.parseJsonParameter(context.parameters.headers, item);
    const body = method !== 'GET' ? this.parseJsonParameter(context.parameters.body, item) : undefined;
    const timeout = context.parameters.timeout;

    try {
      const response = await this.makeHttpRequest({
        method,
        url,
        headers,
        body,
        timeout
      });

      return {
        statusCode: response.status,
        headers: response.headers,
        body: response.data,
        url,
        method,
        timestamp: new Date().toISOString(),
        originalItem: item
      };
    } catch (error) {
      throw new Error(`HTTP request failed: ${error.message}`);
    }
  }

  private async makeHttpRequest(options: any): Promise<any> {
    // Simulate HTTP request - in real implementation, use axios or fetch
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (options.url.includes('error')) {
          reject(new Error('Simulated HTTP error'));
        } else {
          resolve({
            status: 200,
            headers: { 'content-type': 'application/json' },
            data: { success: true, url: options.url, method: options.method }
          });
        }
      }, 100);
    });
  }

  private resolveExpression(expression: string, item: any): string {
    // Simple expression resolution - replace {{field}} with item values
    return expression.replace(/\{\{([^}]+)\}\}/g, (match, field) => {
      return item[field] || match;
    });
  }

  private parseJsonParameter(param: any, item: any): any {
    if (typeof param === 'string') {
      try {
        const resolved = this.resolveExpression(param, item);
        return JSON.parse(resolved);
      } catch {
        return param;
      }
    }
    return param;
  }
}

/**
 * Data Transform Node - transform and manipulate data
 */
export class DataTransformNode extends ActionNode {
  description: INodeTypeDescription = {
    displayName: 'Data Transform',
    name: 'dataTransform',
    icon: 'fa:exchange-alt',
    group: [NODE_CATEGORIES.TRANSFORM],
    version: 1,
    description: 'Transform and manipulate data',
    defaults: {
      name: 'Data Transform'
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          { name: 'Map Fields', value: 'map' },
          { name: 'Filter Items', value: 'filter' },
          { name: 'Sort Items', value: 'sort' },
          { name: 'Group Items', value: 'group' },
          { name: 'Aggregate', value: 'aggregate' }
        ],
        default: 'map',
        description: 'Type of transformation to perform'
      },
      {
        displayName: 'Field Mapping',
        name: 'fieldMapping',
        type: 'json',
        default: '{}',
        description: 'Map input fields to output fields',
        displayOptions: {
          show: {
            operation: ['map']
          }
        }
      },
      {
        displayName: 'Filter Expression',
        name: 'filterExpression',
        type: 'string',
        default: '',
        placeholder: 'item.value > 10',
        description: 'JavaScript expression to filter items',
        displayOptions: {
          show: {
            operation: ['filter']
          }
        }
      },
      {
        displayName: 'Sort Field',
        name: 'sortField',
        type: 'string',
        default: '',
        description: 'Field to sort by',
        displayOptions: {
          show: {
            operation: ['sort']
          }
        }
      },
      {
        displayName: 'Sort Order',
        name: 'sortOrder',
        type: 'options',
        options: [
          { name: 'Ascending', value: 'asc' },
          { name: 'Descending', value: 'desc' }
        ],
        default: 'asc',
        description: 'Sort order',
        displayOptions: {
          show: {
            operation: ['sort']
          }
        }
      }
    ]
  };

  protected async processItem(item: any, index: number, context: NodeExecutionContext): Promise<any> {
    const operation = context.parameters.operation;

    switch (operation) {
      case 'map':
        return this.mapFields(item, context.parameters.fieldMapping);
      case 'filter':
        return this.filterItem(item, context.parameters.filterExpression) ? item : null;
      case 'sort':
        // Sorting is handled at the collection level, return item as-is
        return item;
      default:
        return item;
    }
  }

  protected async executeNode(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const operation = context.parameters.operation;
    
    if (operation === 'sort') {
      return this.sortItems(context);
    } else if (operation === 'group') {
      return this.groupItems(context);
    } else if (operation === 'aggregate') {
      return this.aggregateItems(context);
    }
    
    // For other operations, use the default item-by-item processing
    return super.executeNode(context);
  }

  private mapFields(item: any, fieldMapping: any): any {
    const mapping = typeof fieldMapping === 'string' ? JSON.parse(fieldMapping) : fieldMapping;
    const result: any = {};

    for (const [outputField, inputField] of Object.entries(mapping)) {
      result[outputField] = this.getNestedValue(item, inputField as string);
    }

    return result;
  }

  private filterItem(item: any, expression: string): boolean {
    try {
      // Simple expression evaluation - in production, use a safe evaluator
      const func = new Function('item', `return ${expression}`);
      return func(item);
    } catch {
      return true; // If expression fails, include the item
    }
  }

  private async sortItems(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    const inputItems = this.getInputItems(context);
    const sortField = context.parameters.sortField;
    const sortOrder = context.parameters.sortOrder;

    const sortedItems = [...inputItems].sort((a, b) => {
      const aValue = this.getNestedValue(a, sortField);
      const bValue = this.getNestedValue(b, sortField);
      
      if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

    return {
      outputData: this.createOutputData(sortedItems),
      executionTime: Date.now() - startTime,
      itemsProcessed: inputItems.length
    };
  }

  private async groupItems(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    const inputItems = this.getInputItems(context);
    const groupField = context.parameters.groupField;

    const groups: Record<string, any[]> = {};
    
    for (const item of inputItems) {
      const groupKey = this.getNestedValue(item, groupField);
      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(item);
    }

    const groupedItems = Object.entries(groups).map(([key, items]) => ({
      groupKey: key,
      items,
      count: items.length
    }));

    return {
      outputData: this.createOutputData(groupedItems),
      executionTime: Date.now() - startTime,
      itemsProcessed: inputItems.length
    };
  }

  private async aggregateItems(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    const inputItems = this.getInputItems(context);
    
    const aggregation = {
      count: inputItems.length,
      sum: inputItems.reduce((sum, item) => sum + (Number(item.value) || 0), 0),
      average: 0,
      min: Math.min(...inputItems.map(item => Number(item.value) || 0)),
      max: Math.max(...inputItems.map(item => Number(item.value) || 0))
    };
    
    aggregation.average = aggregation.count > 0 ? aggregation.sum / aggregation.count : 0;

    return {
      outputData: this.createOutputData([aggregation]),
      executionTime: Date.now() - startTime,
      itemsProcessed: inputItems.length
    };
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }
}

/**
 * Condition Node - conditional logic and branching
 */
export class ConditionNode extends ActionNode {
  description: INodeTypeDescription = {
    displayName: 'Condition',
    name: 'condition',
    icon: 'fa:code-branch',
    group: [NODE_CATEGORIES.CORE],
    version: 1,
    description: 'Route data based on conditions',
    defaults: {
      name: 'Condition'
    },
    inputs: ['main'],
    outputs: ['true', 'false'],
    properties: [
      {
        displayName: 'Condition',
        name: 'condition',
        type: 'string',
        default: '',
        placeholder: 'item.value > 10',
        description: 'JavaScript expression for the condition',
        required: true
      }
    ]
  };

  constructor() {
    super();
    // Condition nodes have two outputs
    this.outputPorts = [
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

  protected async executeNode(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    const inputItems = this.getInputItems(context);
    const condition = context.parameters.condition;
    
    const trueItems: any[] = [];
    const falseItems: any[] = [];

    for (const item of inputItems) {
      try {
        const result = this.evaluateCondition(condition, item);
        if (result) {
          trueItems.push(item);
        } else {
          falseItems.push(item);
        }
      } catch (error) {
        // If condition evaluation fails, route to false
        falseItems.push({
          ...item,
          conditionError: error.message
        });
      }
    }

    return {
      outputData: [
        trueItems.map(item => ({ json: item, pairedItem: { item: 0, input: 0 } })),
        falseItems.map(item => ({ json: item, pairedItem: { item: 0, input: 0 } }))
      ],
      executionTime: Date.now() - startTime,
      itemsProcessed: inputItems.length,
      metadata: {
        trueItems: trueItems.length,
        falseItems: falseItems.length
      }
    };
  }

  protected async processItem(item: any, index: number, context: NodeExecutionContext): Promise<any> {
    // Not used for condition nodes as we override executeNode
    return item;
  }

  private evaluateCondition(condition: string, item: any): boolean {
    try {
      const func = new Function('item', `return ${condition}`);
      return Boolean(func(item));
    } catch (error) {
      throw new Error(`Condition evaluation failed: ${error.message}`);
    }
  }
}