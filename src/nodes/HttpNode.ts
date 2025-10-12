import { BaseNode } from './BaseNode';
import { INodeTypeDescription, INodeExecutionData, IExecuteFunctions, NODE_CATEGORIES } from '../types/n8nTypes';

export class HttpNode extends BaseNode {
  description: INodeTypeDescription = {
    displayName: 'HTTP Request',
    name: 'httpRequest',
    icon: 'fa:globe',
    group: [NODE_CATEGORIES.INTEGRATION],
    version: 1,
    description: 'Make HTTP requests to any URL',
    defaults: {
      name: 'HTTP Request',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'httpBasicAuth',
        required: false,
      },
      {
        name: 'httpHeaderAuth',
        required: false,
      },
    ],
    properties: [
      {
        displayName: 'Resource',
        name: 'resource',
        type: 'options',
        noDataExpression: true,
        options: [
          {
            name: 'HTTP Request',
            value: 'httpRequest',
          },
          {
            name: 'REST API',
            value: 'restApi',
          },
          {
            name: 'GraphQL',
            value: 'graphql',
          },
        ],
        default: 'httpRequest',
        description: 'The type of HTTP operation to perform',
      },
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        displayOptions: {
          show: {
            resource: ['httpRequest'],
          },
        },
        options: [
          {
            name: 'GET',
            value: 'get',
            description: 'Retrieve data from the server',
          },
          {
            name: 'POST',
            value: 'post',
            description: 'Send data to the server',
          },
          {
            name: 'PUT',
            value: 'put',
            description: 'Update data on the server',
          },
          {
            name: 'DELETE',
            value: 'delete',
            description: 'Delete data from the server',
          },
          {
            name: 'PATCH',
            value: 'patch',
            description: 'Partially update data on the server',
          },
        ],
        default: 'get',
        description: 'The HTTP method to use',
      },
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        displayOptions: {
          show: {
            resource: ['restApi'],
          },
        },
        options: [
          {
            name: 'Create Resource',
            value: 'create',
            description: 'Create a new resource',
          },
          {
            name: 'Get Resource',
            value: 'get',
            description: 'Retrieve a resource',
          },
          {
            name: 'Update Resource',
            value: 'update',
            description: 'Update an existing resource',
          },
          {
            name: 'Delete Resource',
            value: 'delete',
            description: 'Delete a resource',
          },
          {
            name: 'List Resources',
            value: 'list',
            description: 'List all resources',
          },
        ],
        default: 'get',
        description: 'The REST operation to perform',
      },
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        displayOptions: {
          show: {
            resource: ['graphql'],
          },
        },
        options: [
          {
            name: 'Query',
            value: 'query',
            description: 'Execute a GraphQL query',
          },
          {
            name: 'Mutation',
            value: 'mutation',
            description: 'Execute a GraphQL mutation',
          },
          {
            name: 'Subscription',
            value: 'subscription',
            description: 'Execute a GraphQL subscription',
          },
        ],
        default: 'query',
        description: 'The GraphQL operation to perform',
      },
      {
        displayName: 'Authentication',
        name: 'authentication',
        type: 'credentialsSelect',
        required: false,
        credentialTypes: ['httpBasicAuth', 'apiKey', 'oauth2'],
        default: '',
        description: 'Select authentication credentials for this request',
      },
      {
        displayName: 'URL',
        name: 'url',
        type: 'string',
        required: true,
        default: '',
        placeholder: 'https://api.example.com/endpoint',
        description: 'The URL to make the request to',
      },
      {
        displayName: 'Headers',
        name: 'headers',
        type: 'collection',
        placeholder: 'Add Header',
        default: {},
        options: [
          {
            name: 'Content-Type',
            value: 'content-type',
          },
          {
            name: 'Authorization',
            value: 'authorization',
          },
          {
            name: 'User-Agent',
            value: 'user-agent',
          },
          {
            name: 'Accept',
            value: 'accept',
          },
        ],
        description: 'Headers to send with the request',
      },
      {
        displayName: 'Body',
        name: 'body',
        type: 'json',
        displayOptions: {
          show: {
            operation: ['post', 'put', 'patch', 'create', 'update', 'mutation'],
          },
        },
        default: '{}',
        description: 'The request body as JSON',
      },
      {
        displayName: 'Query Parameters',
        name: 'queryParameters',
        type: 'collection',
        placeholder: 'Add Parameter',
        default: {},
        options: [
          {
            name: 'limit',
            value: 'limit',
          },
          {
            name: 'offset',
            value: 'offset',
          },
          {
            name: 'sort',
            value: 'sort',
          },
          {
            name: 'filter',
            value: 'filter',
          },
        ],
        description: 'Query parameters to append to the URL',
      },
      {
        displayName: 'GraphQL Query',
        name: 'graphqlQuery',
        type: 'string',
        displayOptions: {
          show: {
            resource: ['graphql'],
          },
        },
        default: '',
        placeholder: 'query { users { id name email } }',
        description: 'The GraphQL query, mutation, or subscription',
      },
      {
        displayName: 'Variables',
        name: 'variables',
        type: 'json',
        displayOptions: {
          show: {
            resource: ['graphql'],
          },
        },
        default: '{}',
        description: 'Variables for the GraphQL operation',
      },
      {
        displayName: 'Authentication',
        name: 'authentication',
        type: 'options',
        options: [
          {
            name: 'None',
            value: 'none',
          },
          {
            name: 'Basic Auth',
            value: 'basicAuth',
          },
          {
            name: 'Bearer Token',
            value: 'bearerToken',
          },
          {
            name: 'API Key',
            value: 'apiKey',
          },
        ],
        default: 'none',
        description: 'Authentication method to use',
      },
      {
        displayName: 'Token',
        name: 'token',
        type: 'string',
        displayOptions: {
          show: {
            authentication: ['bearerToken'],
          },
        },
        default: '',
        description: 'Bearer token for authentication',
      },
      {
        displayName: 'API Key',
        name: 'apiKey',
        type: 'string',
        displayOptions: {
          show: {
            authentication: ['apiKey'],
          },
        },
        default: '',
        description: 'API key for authentication',
      },
      {
        displayName: 'API Key Header Name',
        name: 'apiKeyHeader',
        type: 'string',
        displayOptions: {
          show: {
            authentication: ['apiKey'],
          },
        },
        default: 'X-API-Key',
        description: 'Header name for the API key',
      },
    ],
  };

  async executeNode(context: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = context.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const resource = context.getNodeParameter('resource', i) as string;
        const operation = context.getNodeParameter('operation', i) as string;
        const url = context.getNodeParameter('url', i) as string;
        const headers = context.getNodeParameter('headers', i, {}) as Record<string, string>;
        const authentication = context.getNodeParameter('authentication', i) as string;

        // Build request configuration based on resource and operation
        let requestConfig: any = {
          url,
          headers: { ...headers },
        };

        // Set HTTP method based on resource and operation
        if (resource === 'httpRequest') {
          requestConfig.method = operation.toUpperCase();
        } else if (resource === 'restApi') {
          switch (operation) {
            case 'create':
              requestConfig.method = 'POST';
              break;
            case 'get':
              requestConfig.method = 'GET';
              break;
            case 'update':
              requestConfig.method = 'PUT';
              break;
            case 'delete':
              requestConfig.method = 'DELETE';
              break;
            case 'list':
              requestConfig.method = 'GET';
              break;
            default:
              requestConfig.method = 'GET';
          }
        } else if (resource === 'graphql') {
          requestConfig.method = 'POST';
          requestConfig.headers['Content-Type'] = 'application/json';
          
          const graphqlQuery = context.getNodeParameter('graphqlQuery', i) as string;
          const variables = context.getNodeParameter('variables', i, {}) as Record<string, any>;
          
          requestConfig.data = {
            query: graphqlQuery,
            variables,
          };
        }

        // Add body for applicable operations
        if (['post', 'put', 'patch', 'create', 'update'].includes(operation) && resource !== 'graphql') {
          const body = context.getNodeParameter('body', i, {}) as any;
          requestConfig.data = body;
          if (!requestConfig.headers['Content-Type']) {
            requestConfig.headers['Content-Type'] = 'application/json';
          }
        }

        // Add query parameters
        const queryParameters = context.getNodeParameter('queryParameters', i, {}) as Record<string, string>;
        if (Object.keys(queryParameters).length > 0) {
          const urlObj = new URL(url);
          Object.entries(queryParameters).forEach(([key, value]) => {
            urlObj.searchParams.append(key, value);
          });
          requestConfig.url = urlObj.toString();
        }

        // Handle authentication
        switch (authentication) {
          case 'bearerToken':
            const token = context.getNodeParameter('token', i) as string;
            requestConfig.headers['Authorization'] = `Bearer ${token}`;
            break;
          case 'apiKey':
            const apiKey = context.getNodeParameter('apiKey', i) as string;
            const apiKeyHeader = context.getNodeParameter('apiKeyHeader', i) as string;
            requestConfig.headers[apiKeyHeader] = apiKey;
            break;
        }

        // Make the HTTP request
        const response = await context.helpers.httpRequest(requestConfig);

        returnData.push({
          json: {
            statusCode: response.status || 200,
            headers: response.headers || {},
            body: response.data,
            url: requestConfig.url,
            method: requestConfig.method,
            resource,
            operation,
          },
        });
      } catch (error) {
        returnData.push({
          json: {
            error: error instanceof Error ? error.message : 'Unknown error',
            statusCode: 500,
          },
        });
      }
    }

    return [returnData];
  }
}