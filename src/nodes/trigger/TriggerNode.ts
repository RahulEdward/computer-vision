import { BaseNode, NodeExecutionContext, NodeExecutionResult } from '../base/BaseNode';
import { INodeTypeDescription, NODE_CATEGORIES } from '../../types/n8nTypes';

export interface TriggerConfig {
  mode: 'manual' | 'webhook' | 'schedule' | 'event';
  schedule?: {
    interval: number;
    unit: 'seconds' | 'minutes' | 'hours' | 'days';
    timezone?: string;
  };
  webhook?: {
    path: string;
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
    authentication?: 'none' | 'basic' | 'header';
  };
  event?: {
    source: string;
    eventType: string;
    filters?: Record<string, any>;
  };
}

export abstract class TriggerNode extends BaseNode {
  protected triggerConfig: TriggerConfig = {
    mode: 'manual'
  };

  constructor() {
    super();
    // Trigger nodes don't have input ports
    this.inputPorts = [];
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
   * Trigger nodes execute differently - they generate data rather than process it
   */
  protected async executeNode(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    try {
      const triggerData = await this.trigger(context);
      
      return {
        outputData: this.createOutputData(triggerData),
        executionTime: Date.now() - startTime,
        itemsProcessed: triggerData.length,
        metadata: {
          triggerMode: this.triggerConfig.mode,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      throw new Error(`Trigger execution failed: ${error.message}`);
    }
  }

  /**
   * Abstract method for trigger logic
   */
  protected abstract trigger(context: NodeExecutionContext): Promise<any[]>;

  /**
   * Setup webhook endpoint for webhook triggers
   */
  public async setupWebhook(config: TriggerConfig['webhook']): Promise<string> {
    if (!config) {
      throw new Error('Webhook configuration is required');
    }

    // In a real implementation, this would register the webhook with the server
    const webhookUrl = `/webhook/${config.path}`;
    
    console.log(`Webhook registered: ${config.method} ${webhookUrl}`);
    
    return webhookUrl;
  }

  /**
   * Setup scheduled execution
   */
  public async setupSchedule(config: TriggerConfig['schedule']): Promise<void> {
    if (!config) {
      throw new Error('Schedule configuration is required');
    }

    // In a real implementation, this would register with a scheduler
    console.log(`Schedule registered: every ${config.interval} ${config.unit}`);
  }

  /**
   * Handle incoming webhook data
   */
  public async handleWebhook(request: any): Promise<any[]> {
    const triggerData = {
      method: request.method,
      headers: request.headers,
      body: request.body,
      query: request.query,
      timestamp: new Date().toISOString(),
      webhookId: `webhook_${Date.now()}`
    };

    return [triggerData];
  }

  /**
   * Validate trigger configuration
   */
  protected async validateTriggerConfig(): Promise<void> {
    switch (this.triggerConfig.mode) {
      case 'webhook':
        if (!this.triggerConfig.webhook) {
          throw new Error('Webhook configuration is required for webhook triggers');
        }
        break;
      case 'schedule':
        if (!this.triggerConfig.schedule) {
          throw new Error('Schedule configuration is required for scheduled triggers');
        }
        break;
      case 'event':
        if (!this.triggerConfig.event) {
          throw new Error('Event configuration is required for event triggers');
        }
        break;
    }
  }
}

/**
 * Manual Trigger Node - starts workflow manually
 */
export class ManualTriggerNode extends TriggerNode {
  description: INodeTypeDescription = {
    displayName: 'Manual Trigger',
    name: 'manualTrigger',
    icon: 'fa:hand-paper',
    group: [NODE_CATEGORIES.TRIGGER],
    version: 1,
    description: 'Manually trigger workflow execution',
    defaults: {
      name: 'Manual Trigger'
    },
    inputs: [],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Trigger Data',
        name: 'triggerData',
        type: 'json',
        default: '{}',
        description: 'Data to pass when manually triggering the workflow'
      }
    ]
  };

  protected async trigger(context: NodeExecutionContext): Promise<any[]> {
    const triggerData = context.parameters.triggerData || {};
    
    return [{
      ...triggerData,
      timestamp: new Date().toISOString(),
      triggeredBy: 'manual',
      executionId: context.executionId
    }];
  }
}

/**
 * Webhook Trigger Node - receives HTTP requests
 */
export class WebhookTriggerNode extends TriggerNode {
  description: INodeTypeDescription = {
    displayName: 'Webhook',
    name: 'webhook',
    icon: 'fa:satellite-dish',
    group: [NODE_CATEGORIES.TRIGGER],
    version: 1,
    description: 'Receive HTTP requests to trigger workflows',
    defaults: {
      name: 'Webhook'
    },
    inputs: [],
    outputs: ['main'],
    properties: [
      {
        displayName: 'HTTP Method',
        name: 'httpMethod',
        type: 'options',
        options: [
          { name: 'GET', value: 'GET' },
          { name: 'POST', value: 'POST' },
          { name: 'PUT', value: 'PUT' },
          { name: 'DELETE', value: 'DELETE' },
          { name: 'PATCH', value: 'PATCH' }
        ],
        default: 'POST',
        description: 'HTTP method to listen for'
      },
      {
        displayName: 'Path',
        name: 'path',
        type: 'string',
        default: '',
        placeholder: 'webhook-path',
        description: 'Path for the webhook URL'
      },
      {
        displayName: 'Authentication',
        name: 'authentication',
        type: 'options',
        options: [
          { name: 'None', value: 'none' },
          { name: 'Basic Auth', value: 'basic' },
          { name: 'Header Auth', value: 'header' }
        ],
        default: 'none',
        description: 'Authentication method for the webhook'
      }
    ]
  };

  constructor() {
    super();
    this.triggerConfig.mode = 'webhook';
  }

  protected async trigger(context: NodeExecutionContext): Promise<any[]> {
    // This would be called when a webhook request is received
    const webhookConfig = {
      path: context.parameters.path,
      method: context.parameters.httpMethod,
      authentication: context.parameters.authentication
    };

    this.triggerConfig.webhook = webhookConfig;
    
    // Setup webhook endpoint
    await this.setupWebhook(webhookConfig);
    
    // Return empty array - actual data comes from webhook requests
    return [];
  }
}

/**
 * Schedule Trigger Node - executes on a schedule
 */
export class ScheduleTriggerNode extends TriggerNode {
  description: INodeTypeDescription = {
    displayName: 'Schedule Trigger',
    name: 'scheduleTrigger',
    icon: 'fa:clock',
    group: [NODE_CATEGORIES.TRIGGER],
    version: 1,
    description: 'Trigger workflow execution on a schedule',
    defaults: {
      name: 'Schedule Trigger'
    },
    inputs: [],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Interval',
        name: 'interval',
        type: 'number',
        default: 1,
        description: 'Interval between executions'
      },
      {
        displayName: 'Unit',
        name: 'unit',
        type: 'options',
        options: [
          { name: 'Seconds', value: 'seconds' },
          { name: 'Minutes', value: 'minutes' },
          { name: 'Hours', value: 'hours' },
          { name: 'Days', value: 'days' }
        ],
        default: 'minutes',
        description: 'Time unit for the interval'
      },
      {
        displayName: 'Timezone',
        name: 'timezone',
        type: 'string',
        default: 'UTC',
        description: 'Timezone for schedule execution'
      }
    ]
  };

  constructor() {
    super();
    this.triggerConfig.mode = 'schedule';
  }

  protected async trigger(context: NodeExecutionContext): Promise<any[]> {
    const scheduleConfig = {
      interval: context.parameters.interval,
      unit: context.parameters.unit,
      timezone: context.parameters.timezone
    };

    this.triggerConfig.schedule = scheduleConfig;
    
    // Setup schedule
    await this.setupSchedule(scheduleConfig);
    
    return [{
      timestamp: new Date().toISOString(),
      scheduledExecution: true,
      interval: scheduleConfig.interval,
      unit: scheduleConfig.unit,
      timezone: scheduleConfig.timezone
    }];
  }
}