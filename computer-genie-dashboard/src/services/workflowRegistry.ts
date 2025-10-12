import { workflowEngine } from './workflowEngine';
import {
  HttpRequestExecutor,
  DataTransformExecutor,
  ConditionExecutor,
  TimerExecutor,
  FileOperationExecutor,
  CodeExecutionExecutor,
  EmailExecutor,
  DatabaseExecutor,
} from './nodeExecutors';
import {
  DefaultErrorHandler,
  NetworkErrorHandler,
  ValidationErrorHandler,
  TimeoutErrorHandler,
  ResourceErrorHandler,
  AuthenticationErrorHandler,
} from './workflowErrorHandlers';
import {
  DefaultRecoveryStrategy,
  RetryRecoveryStrategy,
  FallbackRecoveryStrategy,
  CircuitBreakerRecoveryStrategy,
  GracefulDegradationStrategy,
} from './workflowErrorHandlers';

export function registerCoreServices() {
  // Register Node Executors
  workflowEngine.registerNodeExecutor('http', new HttpRequestExecutor());
  workflowEngine.registerNodeExecutor('transform', new DataTransformExecutor());
  workflowEngine.registerNodeExecutor('condition', new ConditionExecutor());
  workflowEngine.registerNodeExecutor('timer', new TimerExecutor());
  workflowEngine.registerNodeExecutor('file', new FileOperationExecutor());
  workflowEngine.registerNodeExecutor('code', new CodeExecutionExecutor());
  workflowEngine.registerNodeExecutor('email', new EmailExecutor());
  workflowEngine.registerNodeExecutor('database', new DatabaseExecutor());

  // Register Error Handlers
  workflowEngine.registerErrorHandler('default', new DefaultErrorHandler());
  workflowEngine.registerErrorHandler('network', new NetworkErrorHandler());
  workflowEngine.registerErrorHandler('validation', new ValidationErrorHandler());
  workflowEngine.registerErrorHandler('timeout', new TimeoutErrorHandler());
  workflowEngine.registerErrorHandler('resource', new ResourceErrorHandler());
  workflowEngine.registerErrorHandler('authentication', new AuthenticationErrorHandler());

  // Register Recovery Strategies
  workflowEngine.registerRecoveryStrategy('default', new DefaultRecoveryStrategy());
  workflowEngine.registerRecoveryStrategy('retry', new RetryRecoveryStrategy());
  workflowEngine.registerRecoveryStrategy('fallback', new FallbackRecoveryStrategy());
  workflowEngine.registerRecoveryStrategy('circuit_breaker', new CircuitBreakerRecoveryStrategy());
  workflowEngine.registerRecoveryStrategy('graceful_degradation', new GracefulDegradationStrategy());
}