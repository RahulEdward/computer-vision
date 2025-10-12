import { BaseNode } from './BaseNode';
import { INodeTypeDescription, IExecuteFunctions, INodeExecutionData, NODE_CATEGORIES, CONNECTION_TYPES } from '../types/n8nTypes';

export class TimerNode extends BaseNode {
  description: INodeTypeDescription = {
    displayName: 'Timer',
    name: 'timer',
    icon: 'timer',
    group: [NODE_CATEGORIES.TRIGGER],
    version: 1,
    description: 'Triggers workflow execution at specified intervals',
    defaults: {
      name: 'Timer',
    },
    inputs: [],
    outputs: [CONNECTION_TYPES.MAIN],
    properties: [
      {
        displayName: 'Trigger Interval',
        name: 'interval',
        type: 'options',
        options: [
          {
            name: 'Every Minute',
            value: 60000,
            description: 'Trigger every minute',
          },
          {
            name: 'Every 5 Minutes',
            value: 300000,
            description: 'Trigger every 5 minutes',
          },
          {
            name: 'Every Hour',
            value: 3600000,
            description: 'Trigger every hour',
          },
          {
            name: 'Every Day',
            value: 86400000,
            description: 'Trigger every day',
          },
          {
            name: 'Custom',
            value: 'custom',
            description: 'Set custom interval',
          },
        ],
        default: 60000,
        description: 'How often to trigger the workflow',
      },
      {
        displayName: 'Custom Interval (ms)',
        name: 'customInterval',
        type: 'number',
        default: 60000,
        description: 'Custom interval in milliseconds',
        displayOptions: {
          show: {
            interval: ['custom'],
          },
        },
      },
      {
        displayName: 'Start Immediately',
        name: 'startImmediately',
        type: 'boolean',
        default: true,
        description: 'Whether to trigger immediately when workflow starts',
      },
    ],
  };

  protected async executeNode(context: IExecuteFunctions, itemIndex: number): Promise<INodeExecutionData> {
    const interval = this.getParameterValue(context, 'interval', itemIndex, 60000);
    const customInterval = this.getParameterValue(context, 'customInterval', itemIndex, 60000);
    const startImmediately = this.getParameterValue(context, 'startImmediately', itemIndex, true);

    const actualInterval = interval === 'custom' ? customInterval : interval;

    return this.createJsonData({
      timestamp: new Date().toISOString(),
      interval: actualInterval,
      startImmediately,
      triggerType: 'timer',
      message: `Timer triggered with ${actualInterval}ms interval`,
    });
  }
}