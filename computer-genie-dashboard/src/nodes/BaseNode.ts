import { INodeType, INodeTypeDescription, IExecuteFunctions, INodeExecutionData } from '../types/n8nTypes';

export abstract class BaseNode implements INodeType {
  abstract description: INodeTypeDescription;

  async execute(context: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = context.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const result = await this.executeNode(context, i);
        if (Array.isArray(result)) {
          returnData.push(...result);
        } else {
          returnData.push(result);
        }
      } catch (error) {
        throw new Error(`Node execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    return [returnData];
  }

  protected abstract executeNode(context: IExecuteFunctions, itemIndex: number): Promise<INodeExecutionData | INodeExecutionData[]>;

  protected createJsonData(data: Record<string, any>): INodeExecutionData {
    return {
      json: data,
    };
  }

  protected getParameterValue(context: IExecuteFunctions, parameterName: string, itemIndex: number, defaultValue?: any): any {
    try {
      return context.getNodeParameter(parameterName, itemIndex);
    } catch {
      return defaultValue;
    }
  }

  protected async makeHttpRequest(context: IExecuteFunctions, options: any): Promise<any> {
    return context.helpers.httpRequest(options);
  }
}