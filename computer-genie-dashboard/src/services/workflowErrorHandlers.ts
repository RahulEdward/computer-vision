import { WorkflowNode, ExecutionContext } from './workflowEngine';

export abstract class ErrorHandler {
  abstract shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean>;
}

export class DefaultErrorHandler extends ErrorHandler {
  async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
    return retryCount < maxRetries;
  }
}

export class NetworkErrorHandler extends ErrorHandler {
  async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
    // Retry on network errors
    return retryCount < maxRetries;
  }
}

export class ValidationErrorHandler extends ErrorHandler {
  async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
    // Do not retry on validation errors
    return false;
  }
}

export class TimeoutErrorHandler extends ErrorHandler {
    async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
        // Retry on timeout errors
        return retryCount < maxRetries;
    }
}

export class ResourceErrorHandler extends ErrorHandler {
    async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
        // Retry on resource errors
        return retryCount < maxRetries;
    }
}

export class AuthenticationErrorHandler extends ErrorHandler {
    async shouldRetry(error: Error, retryCount: number, maxRetries: number): Promise<boolean> {
        // Do not retry on authentication errors
        return false;
    }
}

export abstract class RecoveryStrategy {
  abstract name: string;
  abstract recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean>;
}

export class DefaultRecoveryStrategy extends RecoveryStrategy {
    name = 'default';
    async recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean> {
        return false;
    }
}

export class RetryRecoveryStrategy extends RecoveryStrategy {
  name = 'retry';
  async recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean> {
    // Logic to retry the node execution
    return true;
  }
}

export class FallbackRecoveryStrategy extends RecoveryStrategy {
  name = 'fallback';
  async recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean> {
    // Logic to execute a fallback node
    return false;
  }
}

export class CircuitBreakerRecoveryStrategy extends RecoveryStrategy {
  name = 'circuit_breaker';
  private failures = new Map<string, number>();
  private threshold = 3;

  async recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean> {
    const failureCount = this.failures.get(node.id) || 0;
    if (failureCount >= this.threshold) {
      // Circuit is open
      return false;
    }

    this.failures.set(node.id, failureCount + 1);
    return true; // Attempt recovery
  }
}

export class GracefulDegradationStrategy extends RecoveryStrategy {
    name = 'graceful_degradation';
    async recover(node: WorkflowNode, error: Error, context: ExecutionContext): Promise<boolean> {
        // Logic for graceful degradation
        return false;
    }
}