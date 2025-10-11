import { IExecuteFunctions, INodeExecutionData } from '../types/n8nTypes';

export class ExecutionContext implements IExecuteFunctions {
  private nodeParameters: Record<string, any>;
  private inputData: INodeExecutionData[];
  private credentials: Record<string, Record<string, any>>;

  constructor(
    nodeParameters: Record<string, any> = {},
    inputData: INodeExecutionData[] = [],
    credentials: Record<string, Record<string, any>> = {}
  ) {
    this.nodeParameters = nodeParameters;
    this.inputData = inputData;
    this.credentials = credentials;
  }

  getNodeParameter(parameterName: string, itemIndex: number): any {
    return this.nodeParameters[parameterName];
  }

  getInputData(inputIndex: number = 0): INodeExecutionData[] {
    return this.inputData;
  }

  async getCredentials(type: string): Promise<Record<string, any>> {
    return this.credentials[type] || {};
  }

  helpers = {
    httpRequest: async (options: any): Promise<any> => {
      try {
        const response = await fetch(options.url, {
          method: options.method || 'GET',
          headers: options.headers || {},
          body: options.body ? JSON.stringify(options.body) : undefined,
        });

        const data = await response.json();
        
        return {
          statusCode: response.status,
          headers: Object.fromEntries(response.headers.entries()),
          body: data,
        };
      } catch (error) {
        throw new Error(`HTTP request failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    },

    returnJsonArray: (data: any): INodeExecutionData[] => {
      if (!Array.isArray(data)) {
        return [{ json: data }];
      }
      
      return data.map(item => ({
        json: typeof item === 'object' ? item : { value: item },
      }));
    },
  };

  // Helper methods for setting up execution context
  setNodeParameters(parameters: Record<string, any>): void {
    this.nodeParameters = parameters;
  }

  setInputData(data: INodeExecutionData[]): void {
    this.inputData = data;
  }

  setCredentials(type: string, credentials: Record<string, any>): void {
    this.credentials[type] = credentials;
  }
}