import { WorkflowNode, ExecutionContext } from './workflowEngine';

// Base class for node executors
export abstract class NodeExecutor {
  abstract execute(node: WorkflowNode, context: ExecutionContext): Promise<any>;
}

// HTTP Request Executor
export class HttpRequestExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { url, method = 'GET', headers = {}, body } = node.data.config || {};
    
    if (!url) {
      throw new Error('URL is required for HTTP request');
    }

    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return { status: response.status, data };
  }
}

// Data Transform Executor
export class DataTransformExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { transformCode, inputKey } = node.data.config || {};
    
    if (!transformCode) {
      throw new Error('Transform code is required');
    }

    const inputData = inputKey ? context.results[inputKey] : context.variables;
    
    // Create a safe execution environment
    const func = new Function('data', 'context', `
      return (function() {
        ${transformCode}
      })();
    `);

    return func(inputData, context);
  }
}

// Condition Executor
export class ConditionExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { condition, trueValue, falseValue } = node.data.config || {};
    
    if (!condition) {
      throw new Error('Condition is required');
    }

    // Create a safe execution environment for condition evaluation
    const func = new Function('context', `
      return (${condition});
    `);

    const result = func(context);
    return result ? trueValue : falseValue;
  }
}

// Timer Executor
export class TimerExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { delay = 1000 } = node.data.config || {};
    
    await new Promise(resolve => setTimeout(resolve, delay));
    return { delayed: delay };
  }
}

// File Operation Executor
export class FileOperationExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { operation, content, filename } = node.data.config || {};
    
    if (operation === 'create') {
      // In a real implementation, this would write to the file system
      // For now, we'll simulate file creation
      return { 
        operation: 'create',
        filename,
        size: content?.length || 0,
        created: true 
      };
    } else if (operation === 'read') {
      // Simulate file reading
      return { 
        operation: 'read',
        filename,
        content: `Simulated content of ${filename}`,
        size: 100 
      };
    }
    
    throw new Error(`Unsupported file operation: ${operation}`);
  }
}

// Code Execution Executor
export class CodeExecutionExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { code } = node.data;
    
    if (!code) {
      throw new Error('Code is required for execution');
    }

    // Create a safe execution environment
    const logs: string[] = [];
    const originalConsole = console.log;
    
    console.log = (...args) => {
      logs.push(args.join(' '));
    };

    try {
      const func = new Function('context', `
        ${code}
      `);
      
      const result = func(context);
      console.log = originalConsole;
      
      return { result, logs };
    } catch (error) {
      console.log = originalConsole;
      throw error;
    }
  }
}

// Email Executor
export class EmailExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { to, subject, body } = node.data.config || {};
    
    if (!to || !subject) {
      throw new Error('To and subject are required for email');
    }

    // Simulate email sending
    return {
      sent: true,
      to,
      subject,
      timestamp: Date.now(),
      messageId: `msg_${Math.random().toString(36).substr(2, 9)}`,
    };
  }
}

// Database Executor
export class DatabaseExecutor extends NodeExecutor {
  async execute(node: WorkflowNode, context: ExecutionContext): Promise<any> {
    const { operation, query, data } = node.data.config || {};
    
    // Simulate database operations
    if (operation === 'select') {
      return {
        operation: 'select',
        query,
        results: [
          { id: 1, name: 'Sample Record 1', created: Date.now() },
          { id: 2, name: 'Sample Record 2', created: Date.now() },
        ],
        count: 2,
      };
    } else if (operation === 'insert') {
      return {
        operation: 'insert',
        insertedId: Math.floor(Math.random() * 1000),
        data,
        success: true,
      };
    }
    
    throw new Error(`Unsupported database operation: ${operation}`);
  }
}''