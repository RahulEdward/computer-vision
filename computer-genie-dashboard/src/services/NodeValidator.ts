import { INodeData, INodeProperties } from '../types/n8nTypes';
import { ICredentialData } from '../types/credentials';

export interface IValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface IValidationResult {
  isValid: boolean;
  errors: IValidationError[];
  warnings: IValidationError[];
}

export class NodeValidator {
  /**
   * Validates a node's configuration
   */
  static validateNode(nodeData: INodeData): IValidationResult {
    const errors: IValidationError[] = [];
    const warnings: IValidationError[] = [];

    // Validate required fields
    this.validateRequiredFields(nodeData, errors);
    
    // Validate field types and formats
    this.validateFieldTypes(nodeData, errors, warnings);
    
    // Validate credentials
    this.validateCredentials(nodeData, errors, warnings);
    
    // Validate URLs and endpoints
    this.validateUrls(nodeData, errors, warnings);
    
    // Validate JSON fields
    this.validateJsonFields(nodeData, errors, warnings);

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validates required fields are present
   */
  private static validateRequiredFields(nodeData: INodeData, errors: IValidationError[]): void {
    const nodeType = nodeData.type;
    const parameters = nodeData.parameters || {};

    // Get node type definition (this would come from node registry in real implementation)
    const nodeProperties = this.getNodeProperties(nodeType);

    nodeProperties.forEach(property => {
      if (property.required && !parameters[property.name]) {
        errors.push({
          field: property.name,
          message: `${property.displayName} is required`,
          severity: 'error'
        });
      }
    });
  }

  /**
   * Validates field types and formats
   */
  private static validateFieldTypes(
    nodeData: INodeData, 
    errors: IValidationError[], 
    warnings: IValidationError[]
  ): void {
    const parameters = nodeData.parameters || {};
    const nodeProperties = this.getNodeProperties(nodeData.type);

    nodeProperties.forEach(property => {
      const value = parameters[property.name];
      if (value === undefined || value === null) return;

      switch (property.type) {
        case 'number':
          if (isNaN(Number(value))) {
            errors.push({
              field: property.name,
              message: `${property.displayName} must be a valid number`,
              severity: 'error'
            });
          }
          break;

        case 'boolean':
          if (typeof value !== 'boolean' && value !== 'true' && value !== 'false') {
            errors.push({
              field: property.name,
              message: `${property.displayName} must be a boolean value`,
              severity: 'error'
            });
          }
          break;

        case 'options':
          if (property.options && !property.options.some(opt => opt.value === value)) {
            errors.push({
              field: property.name,
              message: `${property.displayName} contains an invalid option`,
              severity: 'error'
            });
          }
          break;
      }
    });
  }

  /**
   * Validates credentials configuration
   */
  private static validateCredentials(
    nodeData: INodeData, 
    errors: IValidationError[], 
    warnings: IValidationError[]
  ): void {
    const parameters = nodeData.parameters || {};
    const authentication = parameters.authentication;

    if (authentication && typeof authentication === 'string') {
      // Check if credential exists (this would check against credential store)
      if (!this.credentialExists(authentication)) {
        errors.push({
          field: 'authentication',
          message: 'Selected credential does not exist or is invalid',
          severity: 'error'
        });
      }
    }

    // Validate credential requirements for specific node types
    if (nodeData.type === 'http' && !authentication) {
      warnings.push({
        field: 'authentication',
        message: 'Consider adding authentication for external API calls',
        severity: 'warning'
      });
    }
  }

  /**
   * Validates URL fields
   */
  private static validateUrls(
    nodeData: INodeData, 
    errors: IValidationError[], 
    warnings: IValidationError[]
  ): void {
    const parameters = nodeData.parameters || {};
    const url = parameters.url;

    if (url && typeof url === 'string') {
      try {
        new URL(url);
        
        // Check for common security issues
        if (url.startsWith('http://') && !url.includes('localhost')) {
          warnings.push({
            field: 'url',
            message: 'Consider using HTTPS for external URLs',
            severity: 'warning'
          });
        }
      } catch {
        errors.push({
          field: 'url',
          message: 'Invalid URL format',
          severity: 'error'
        });
      }
    }
  }

  /**
   * Validates JSON fields
   */
  private static validateJsonFields(
    nodeData: INodeData, 
    errors: IValidationError[], 
    warnings: IValidationError[]
  ): void {
    const parameters = nodeData.parameters || {};
    const nodeProperties = this.getNodeProperties(nodeData.type);

    nodeProperties.forEach(property => {
      if (property.type === 'json') {
        const value = parameters[property.name];
        if (value && typeof value === 'string') {
          try {
            JSON.parse(value);
          } catch {
            errors.push({
              field: property.name,
              message: `${property.displayName} contains invalid JSON`,
              severity: 'error'
            });
          }
        }
      }
    });
  }

  /**
   * Gets node properties for validation (mock implementation)
   */
  private static getNodeProperties(nodeType: string): INodeProperties[] {
    // This would normally come from a node registry
    // For now, return basic HTTP node properties
    if (nodeType === 'http') {
      return [
        {
          displayName: 'URL',
          name: 'url',
          type: 'string',
          required: true,
          default: '',
          description: 'The URL to make the request to'
        },
        {
          displayName: 'Method',
          name: 'method',
          type: 'options',
          required: true,
          default: 'GET',
          options: [
            { name: 'GET', value: 'GET' },
            { name: 'POST', value: 'POST' },
            { name: 'PUT', value: 'PUT' },
            { name: 'DELETE', value: 'DELETE' }
          ],
          description: 'The HTTP method to use'
        }
      ];
    }
    return [];
  }

  /**
   * Checks if a credential exists (mock implementation)
   */
  private static credentialExists(credentialId: string): boolean {
    // This would check against the actual credential store
    return credentialId.length > 0;
  }

  /**
   * Validates workflow connectivity
   */
  static validateWorkflowConnections(nodes: INodeData[]): IValidationResult {
    const errors: IValidationError[] = [];
    const warnings: IValidationError[] = [];

    // Check for orphaned nodes
    const connectedNodes = new Set<string>();
    
    nodes.forEach(node => {
      // Add logic to check node connections
      // This is a simplified version
      if (node.type !== 'trigger' && !this.hasIncomingConnection(node, nodes)) {
        warnings.push({
          field: 'connections',
          message: `Node "${node.name}" has no incoming connections`,
          severity: 'warning'
        });
      }
    });

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Checks if a node has incoming connections (simplified)
   */
  private static hasIncomingConnection(node: INodeData, allNodes: INodeData[]): boolean {
    // This would check the actual workflow connections
    // For now, return true as a placeholder
    return true;
  }
}