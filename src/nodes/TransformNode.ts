import { BaseNode } from './BaseNode';
import { INodeTypeDescription, IExecuteFunctions, INodeExecutionData, NODE_CATEGORIES, CONNECTION_TYPES } from '../types/n8nTypes';

export class TransformNode extends BaseNode {
  description: INodeTypeDescription = {
    displayName: 'Transform',
    name: 'transform',
    icon: 'transform',
    group: [NODE_CATEGORIES.TRANSFORM],
    version: 1,
    description: 'Transform and manipulate data',
    defaults: {
      name: 'Transform',
    },
    inputs: [CONNECTION_TYPES.MAIN],
    outputs: [CONNECTION_TYPES.MAIN],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          {
            name: 'Add Field',
            value: 'addField',
            description: 'Add a new field to the data',
          },
          {
            name: 'Remove Field',
            value: 'removeField',
            description: 'Remove a field from the data',
          },
          {
            name: 'Rename Field',
            value: 'renameField',
            description: 'Rename an existing field',
          },
          {
            name: 'Map Fields',
            value: 'mapFields',
            description: 'Map multiple fields at once',
          },
          {
            name: 'Filter Data',
            value: 'filterData',
            description: 'Filter data based on conditions',
          },
          {
            name: 'Sort Data',
            value: 'sortData',
            description: 'Sort data by field',
          },
        ],
        default: 'addField',
        description: 'The transformation operation to perform',
      },
      {
        displayName: 'Field Name',
        name: 'fieldName',
        type: 'string',
        default: '',
        description: 'Name of the field to operate on',
        displayOptions: {
          show: {
            operation: ['addField', 'removeField', 'renameField', 'sortData'],
          },
        },
      },
      {
        displayName: 'Field Value',
        name: 'fieldValue',
        type: 'string',
        default: '',
        description: 'Value for the new field',
        displayOptions: {
          show: {
            operation: ['addField'],
          },
        },
      },
      {
        displayName: 'New Field Name',
        name: 'newFieldName',
        type: 'string',
        default: '',
        description: 'New name for the field',
        displayOptions: {
          show: {
            operation: ['renameField'],
          },
        },
      },
      {
        displayName: 'Field Mapping',
        name: 'fieldMapping',
        type: 'json',
        default: '{}',
        description: 'JSON object mapping old field names to new field names',
        displayOptions: {
          show: {
            operation: ['mapFields'],
          },
        },
      },
      {
        displayName: 'Filter Condition',
        name: 'filterCondition',
        type: 'string',
        default: '',
        placeholder: 'field === "value"',
        description: 'JavaScript expression to filter data',
        displayOptions: {
          show: {
            operation: ['filterData'],
          },
        },
      },
      {
        displayName: 'Sort Order',
        name: 'sortOrder',
        type: 'options',
        options: [
          {
            name: 'Ascending',
            value: 'asc',
          },
          {
            name: 'Descending',
            value: 'desc',
          },
        ],
        default: 'asc',
        description: 'Sort order',
        displayOptions: {
          show: {
            operation: ['sortData'],
          },
        },
      },
    ],
  };

  protected async executeNode(context: IExecuteFunctions, itemIndex: number): Promise<INodeExecutionData[]> {
    const operation = this.getParameterValue(context, 'operation', itemIndex, 'addField');
    const inputData = context.getInputData();
    const results: INodeExecutionData[] = [];

    for (const item of inputData) {
      let transformedData = { ...item.json };

      switch (operation) {
        case 'addField':
          const fieldName = this.getParameterValue(context, 'fieldName', itemIndex, '');
          const fieldValue = this.getParameterValue(context, 'fieldValue', itemIndex, '');
          if (fieldName) {
            transformedData[fieldName] = fieldValue;
          }
          break;

        case 'removeField':
          const removeFieldName = this.getParameterValue(context, 'fieldName', itemIndex, '');
          if (removeFieldName && transformedData[removeFieldName] !== undefined) {
            delete transformedData[removeFieldName];
          }
          break;

        case 'renameField':
          const oldFieldName = this.getParameterValue(context, 'fieldName', itemIndex, '');
          const newFieldName = this.getParameterValue(context, 'newFieldName', itemIndex, '');
          if (oldFieldName && newFieldName && transformedData[oldFieldName] !== undefined) {
            transformedData[newFieldName] = transformedData[oldFieldName];
            delete transformedData[oldFieldName];
          }
          break;

        case 'mapFields':
          const fieldMappingString = this.getParameterValue(context, 'fieldMapping', itemIndex, '{}');
          try {
            const fieldMapping = JSON.parse(fieldMappingString);
            const newData: Record<string, any> = {};
            for (const [oldField, newField] of Object.entries(fieldMapping)) {
              if (transformedData[oldField] !== undefined) {
                newData[newField as string] = transformedData[oldField];
              }
            }
            transformedData = newData;
          } catch (error) {
            throw new Error('Invalid JSON format in field mapping');
          }
          break;

        case 'filterData':
          const filterCondition = this.getParameterValue(context, 'filterCondition', itemIndex, '');
          if (filterCondition) {
            try {
              // Simple evaluation - in production, use a safer expression evaluator
              const shouldInclude = this.evaluateCondition(filterCondition, transformedData);
              if (!shouldInclude) {
                continue; // Skip this item
              }
            } catch (error) {
              throw new Error(`Filter condition evaluation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
          }
          break;

        case 'sortData':
          // For sorting, we need to collect all items first
          // This is handled after the loop
          break;
      }

      results.push(this.createJsonData(transformedData));
    }

    // Handle sorting operation
    if (operation === 'sortData') {
      const sortFieldName = this.getParameterValue(context, 'fieldName', itemIndex, '');
      const sortOrder = this.getParameterValue(context, 'sortOrder', itemIndex, 'asc');
      
      if (sortFieldName) {
        results.sort((a, b) => {
          const aValue = a.json[sortFieldName];
          const bValue = b.json[sortFieldName];
          
          if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
          if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
          return 0;
        });
      }
    }

    return results;
  }

  private evaluateCondition(condition: string, data: Record<string, any>): boolean {
    // Simple condition evaluator - replace with a proper expression evaluator in production
    try {
      // Create a function that has access to the data fields
      const func = new Function(...Object.keys(data), `return ${condition}`);
      return func(...Object.values(data));
    } catch {
      return true; // Default to including the item if evaluation fails
    }
  }
}