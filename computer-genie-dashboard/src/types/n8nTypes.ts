// n8n-inspired type definitions for our workflow builder

export interface INodeExecutionData {
  json: Record<string, any>;
  binary?: Record<string, any>;
  pairedItem?: {
    item: number;
    input?: number;
  };
}

export interface INodeType {
  description: INodeTypeDescription;
  execute?(context: IExecuteFunctions): Promise<INodeExecutionData[][]>;
}

export interface INodeTypeDescription {
  displayName: string;
  name: string;
  icon: string;
  group: string[];
  version: number;
  description: string;
  defaults: {
    name: string;
  };
  inputs: string[];
  outputs: string[];
  credentials?: INodeCredentials[];
  properties: INodeProperties[];
  requestDefaults?: {
    baseURL?: string;
    headers?: Record<string, string>;
  };
}

export interface INodeCredentials {
  name: string;
  required: boolean;
  displayName?: string;
  documentationUrl?: string;
}

export interface INodePropertyDisplayOptions {
  show?: Record<string, any[]>;
  hide?: Record<string, any[]>;
}

export interface INodeProperties {
  displayName: string;
  name: string;
  type: 'string' | 'number' | 'boolean' | 'options' | 'multiOptions' | 'collection' | 'fixedCollection' | 'json' | 'dateTime' | 'color' | 'hidden' | 'credentialsSelect';
  default?: any;
  description?: string;
  placeholder?: string;
  required?: boolean;
  displayOptions?: INodePropertyDisplayOptions;
  options?: INodePropertyOptions[];
  noDataExpression?: boolean;
  credentialTypes?: string[];
  routing?: {
    request?: {
      method: string;
      url: string;
      headers?: Record<string, string>;
      body?: any;
    };
    output?: {
      postReceive?: any[];
    };
  };
}

export interface INodePropertyOptions {
  name: string;
  value: string | number | boolean;
  description?: string;
  action?: string;
  routing?: {
    request?: {
      method: string;
      url: string;
      headers?: Record<string, string>;
      body?: any;
    };
    output?: {
      postReceive?: any[];
    };
  };
}

export interface IExecuteFunctions {
  getNodeParameter(parameterName: string, itemIndex: number): any;
  getInputData(inputIndex?: number): INodeExecutionData[];
  getCredentials(type: string): Promise<Record<string, any>>;
  helpers: {
    httpRequest(options: any): Promise<any>;
    returnJsonArray(data: any): INodeExecutionData[];
  };
}

export interface IWorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    nodeType: string;
    type: string;
    label: string;
    description?: string;
    properties?: Record<string, any>;
    credentials?: Record<string, any>;
  };
}

export interface IWorkflowEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface IWorkflowExecution {
  id: string;
  workflowId: string;
  status: 'running' | 'success' | 'error' | 'waiting';
  startTime: Date;
  endTime?: Date;
  data?: INodeExecutionData[][];
  error?: string;
}

// Node categories
export const NODE_CATEGORIES = {
  CORE: 'core',
  TRIGGER: 'trigger',
  ACTION: 'action',
  TRANSFORM: 'transform',
  INTEGRATION: 'integration',
  UTILITY: 'utility',
} as const;

// Connection types
export const CONNECTION_TYPES = {
  MAIN: 'main',
  AI: 'ai',
} as const;