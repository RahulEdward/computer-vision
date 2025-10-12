'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon, ClockIcon, BugAntIcon } from '@heroicons/react/24/outline';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  maxRetries?: number;
  retryDelay?: number;
  enableAutoRetry?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
  isRetrying: boolean;
  lastErrorTime: number;
}

export class ErrorBoundary extends Component<Props, State> {
  private retryTimer: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRetrying: false,
      lastErrorTime: 0
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      lastErrorTime: Date.now()
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Call the onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // Auto-retry for transient errors
    if (this.props.enableAutoRetry && this.shouldAutoRetry(error)) {
      this.scheduleAutoRetry();
    }
  }

  componentWillUnmount() {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
    }
  }

  shouldAutoRetry = (error: Error): boolean => {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;
    
    // Don't auto-retry if max retries reached
    if (retryCount >= maxRetries) return false;
    
    // Auto-retry for network errors, timeouts, and temporary failures
    const retryableErrors = [
      'NetworkError',
      'TimeoutError',
      'AbortError',
      'fetch',
      'network',
      'timeout',
      'connection'
    ];
    
    return retryableErrors.some(keyword => 
      error.message.toLowerCase().includes(keyword.toLowerCase()) ||
      error.name.toLowerCase().includes(keyword.toLowerCase())
    );
  };

  scheduleAutoRetry = () => {
    const { retryDelay = 2000 } = this.props;
    const { retryCount } = this.state;
    
    // Exponential backoff
    const delay = retryDelay * Math.pow(2, retryCount);
    
    this.setState({ isRetrying: true });
    
    this.retryTimer = setTimeout(() => {
      this.handleRetry();
    }, delay);
  };

  handleRetry = () => {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }

    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1,
      isRetrying: false
    }));
  };

  handleReset = () => {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }

    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRetrying: false,
      lastErrorTime: 0
    });
  };

  getErrorSeverity = (): 'critical' | 'error' | 'warning' => {
    const { error } = this.state;
    if (!error) return 'error';
    
    // Critical errors that require immediate attention
    const criticalKeywords = ['security', 'auth', 'permission', 'cors'];
    if (criticalKeywords.some(keyword => 
      error.message.toLowerCase().includes(keyword) ||
      error.name.toLowerCase().includes(keyword)
    )) {
      return 'critical';
    }
    
    // Network/temporary errors are warnings
    const warningKeywords = ['network', 'timeout', 'fetch', 'connection'];
    if (warningKeywords.some(keyword => 
      error.message.toLowerCase().includes(keyword) ||
      error.name.toLowerCase().includes(keyword)
    )) {
      return 'warning';
    }
    
    return 'error';
  };

  render() {
    const { hasError, error, isRetrying, retryCount } = this.state;
    const { maxRetries = 3 } = this.props;
    
    if (hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const severity = this.getErrorSeverity();
      const severityColors = {
        critical: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800', icon: 'text-red-500' },
        error: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-800', icon: 'text-orange-500' },
        warning: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-800', icon: 'text-yellow-500' }
      };
      
      const colors = severityColors[severity];

      // Auto-retry in progress
      if (isRetrying) {
        return (
          <div className={`flex flex-col items-center justify-center p-8 ${colors.bg} ${colors.border} border rounded-lg`}>
            <ClockIcon className={`w-12 h-12 ${colors.icon} mb-4 animate-spin`} />
            <h2 className={`text-lg font-semibold ${colors.text} mb-2`}>
              Retrying... (Attempt {retryCount + 1}/{maxRetries})
            </h2>
            <p className={`${colors.text.replace('800', '600')} text-center mb-4 max-w-md`}>
              Attempting to recover from the error automatically.
            </p>
          </div>
        );
      }

      // Default error UI
      return (
        <div className={`flex flex-col items-center justify-center p-8 ${colors.bg} ${colors.border} border rounded-lg`}>
          <ExclamationTriangleIcon className={`w-12 h-12 ${colors.icon} mb-4`} />
          <h2 className={`text-lg font-semibold ${colors.text} mb-2`}>
            {severity === 'critical' ? 'Critical Error' : 
             severity === 'error' ? 'Something went wrong' : 
             'Temporary Issue'}
          </h2>
          <p className={`${colors.text.replace('800', '600')} text-center mb-4 max-w-md`}>
            {error?.message || 'An unexpected error occurred in the workflow.'}
          </p>
          
          {retryCount > 0 && (
            <p className={`text-sm ${colors.text.replace('800', '500')} mb-4`}>
              Retry attempts: {retryCount}/{maxRetries}
            </p>
          )}
          
          {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
            <details className={`mb-4 p-4 ${colors.bg.replace('50', '100')} rounded border max-w-2xl overflow-auto`}>
              <summary className={`cursor-pointer font-medium ${colors.text} flex items-center gap-2`}>
                <BugAntIcon className="w-4 h-4" />
                Error Details (Development)
              </summary>
              <pre className={`mt-2 text-xs ${colors.text.replace('800', '700')} whitespace-pre-wrap`}>
                {error?.stack}
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}

          <div className="flex gap-3">
            <button
              onClick={this.handleRetry}
              disabled={retryCount >= maxRetries}
              className={`flex items-center gap-2 px-4 py-2 ${
                retryCount >= maxRetries 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : severity === 'critical' 
                    ? 'bg-red-600 hover:bg-red-700' 
                    : severity === 'error'
                      ? 'bg-orange-600 hover:bg-orange-700'
                      : 'bg-yellow-600 hover:bg-yellow-700'
              } text-white rounded-lg transition-colors`}
            >
              <ArrowPathIcon className="w-4 h-4" />
              {retryCount >= maxRetries ? 'Max Retries Reached' : 'Try Again'}
            </button>
            
            {retryCount > 0 && (
              <button
                onClick={this.handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Reset
              </button>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Hook for functional components to handle errors
export const useErrorHandler = () => {
  const [error, setError] = React.useState<Error | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
  }, []);

  const handleError = React.useCallback((error: Error) => {
    setError(error);
  }, []);

  React.useEffect(() => {
    if (error) {
      throw error;
    }
  }, [error]);

  return { handleError, resetError };
};

// Workflow-specific error boundary
interface WorkflowErrorBoundaryProps {
  children: ReactNode;
  workflowId?: string;
  workflowName?: string;
  onWorkflowError?: (workflowId: string, error: Error, recovery?: WorkflowRecoveryAction) => void;
  onWorkflowRecovery?: (workflowId: string, action: WorkflowRecoveryAction) => Promise<boolean>;
  enableAutoSave?: boolean;
  enableRollback?: boolean;
}

export interface WorkflowRecoveryAction {
  type: 'retry' | 'rollback' | 'safe_mode' | 'reset' | 'reload';
  description: string;
  automatic?: boolean;
}

export const WorkflowErrorBoundary: React.FC<WorkflowErrorBoundaryProps> = ({
  children,
  workflowId,
  workflowName,
  onWorkflowError,
  onWorkflowRecovery,
  enableAutoSave = true,
  enableRollback = true
}) => {
  const [isRecovering, setIsRecovering] = React.useState(false);
  const [recoveryAction, setRecoveryAction] = React.useState<WorkflowRecoveryAction | null>(null);

  const handleError = (error: Error, errorInfo: ErrorInfo) => {
    if (workflowId && onWorkflowError) {
      // Determine best recovery action based on error type
      const recovery = determineRecoveryAction(error, { enableRollback, enableAutoSave });
      onWorkflowError(workflowId, error, recovery);
    }
  };

  const determineRecoveryAction = (error: Error, options: { enableRollback: boolean; enableAutoSave: boolean }): WorkflowRecoveryAction => {
    const errorMessage = error.message.toLowerCase();
    
    // Network/API errors - retry
    if (errorMessage.includes('network') || errorMessage.includes('fetch') || errorMessage.includes('timeout')) {
      return { type: 'retry', description: 'Retry workflow execution', automatic: true };
    }
    
    // Validation errors - safe mode
    if (errorMessage.includes('validation') || errorMessage.includes('invalid')) {
      return { type: 'safe_mode', description: 'Run in safe mode with validation checks' };
    }
    
    // State corruption - rollback if available
    if ((errorMessage.includes('state') || errorMessage.includes('corrupt')) && options.enableRollback) {
      return { type: 'rollback', description: 'Rollback to last known good state' };
    }
    
    // Default to reset
    return { type: 'reset', description: 'Reset workflow to initial state' };
  };

  const executeRecovery = async (action: WorkflowRecoveryAction) => {
    if (!workflowId || !onWorkflowRecovery) return false;
    
    setIsRecovering(true);
    setRecoveryAction(action);
    
    try {
      const success = await onWorkflowRecovery(workflowId, action);
      if (success) {
        setRecoveryAction(null);
      }
      return success;
    } catch (recoveryError) {
      console.error('Recovery failed:', recoveryError);
      return false;
    } finally {
      setIsRecovering(false);
    }
  };

  const getRecoveryActions = (error: Error): WorkflowRecoveryAction[] => {
    const actions: WorkflowRecoveryAction[] = [
      { type: 'retry', description: 'Retry workflow execution' },
      { type: 'safe_mode', description: 'Run in safe mode with validation' },
    ];
    
    if (enableRollback) {
      actions.push({ type: 'rollback', description: 'Rollback to previous version' });
    }
    
    actions.push(
      { type: 'reset', description: 'Reset workflow to initial state' },
      { type: 'reload', description: 'Reload entire workflow editor' }
    );
    
    return actions;
  };

  return (
    <ErrorBoundary
      onError={handleError}
      enableAutoRetry={true}
      maxRetries={2}
      retryDelay={1000}
      fallback={
        <WorkflowErrorFallback
          workflowId={workflowId}
          workflowName={workflowName}
          isRecovering={isRecovering}
          recoveryAction={recoveryAction}
          onRecovery={executeRecovery}
          getRecoveryActions={getRecoveryActions}
        />
      }
    >
      {children}
    </ErrorBoundary>
  );
};

interface WorkflowErrorFallbackProps {
  workflowId?: string;
  workflowName?: string;
  isRecovering: boolean;
  recoveryAction: WorkflowRecoveryAction | null;
  onRecovery: (action: WorkflowRecoveryAction) => Promise<boolean>;
  getRecoveryActions: (error: Error) => WorkflowRecoveryAction[];
}

const WorkflowErrorFallback: React.FC<WorkflowErrorFallbackProps> = ({
  workflowId,
  workflowName,
  isRecovering,
  recoveryAction,
  onRecovery,
  getRecoveryActions
}) => {
  const [selectedAction, setSelectedAction] = React.useState<WorkflowRecoveryAction | null>(null);
  const [showAdvanced, setShowAdvanced] = React.useState(false);

  // Mock error for demonstration - in real implementation, this would come from error boundary
  const mockError = new Error('Workflow execution failed');
  const recoveryActions = getRecoveryActions(mockError);

  if (isRecovering && recoveryAction) {
    return (
      <div className="flex flex-col items-center justify-center p-6 bg-blue-50 border border-blue-200 rounded-lg">
        <ClockIcon className="w-10 h-10 text-blue-500 mb-3 animate-spin" />
        <h3 className="text-lg font-medium text-blue-800 mb-2">
          Recovering Workflow
        </h3>
        <p className="text-blue-700 text-center mb-4">
          {recoveryAction.description}...
        </p>
        <div className="w-full max-w-xs bg-blue-200 rounded-full h-2">
          <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-red-50 border border-red-200 rounded-lg max-w-2xl mx-auto">
      <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mb-4" />
      <h3 className="text-xl font-semibold text-red-800 mb-2">
        Workflow Execution Failed
      </h3>
      {workflowName && (
        <p className="text-red-600 font-medium mb-2">
          Workflow: {workflowName}
        </p>
      )}
      <p className="text-red-700 text-center mb-6 max-w-md">
        The workflow encountered an error during execution. Choose a recovery option below to continue.
      </p>

      <div className="w-full space-y-3 mb-4">
        <h4 className="font-medium text-red-800 mb-2">Quick Recovery Options:</h4>
        {recoveryActions.slice(0, 2).map((action, index) => (
          <button
            key={index}
            onClick={() => onRecovery(action)}
            className="w-full flex items-center justify-between p-3 bg-white border border-red-200 rounded-lg hover:bg-red-50 transition-colors"
          >
            <span className="text-red-800">{action.description}</span>
            <ArrowPathIcon className="w-4 h-4 text-red-500" />
          </button>
        ))}
      </div>

      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-red-600 hover:text-red-800 text-sm mb-4 underline"
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <div className="w-full space-y-2 mb-4">
          <h4 className="font-medium text-red-800 mb-2">Advanced Recovery:</h4>
          {recoveryActions.slice(2).map((action, index) => (
            <button
              key={index}
              onClick={() => onRecovery(action)}
              className="w-full flex items-center justify-between p-2 bg-white border border-red-200 rounded hover:bg-red-50 transition-colors text-sm"
            >
              <span className="text-red-700">{action.description}</span>
              <ArrowPathIcon className="w-3 h-3 text-red-500" />
            </button>
          ))}
        </div>
      )}

      <div className="flex gap-3 mt-4">
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Reload Page
        </button>
        <button
          onClick={() => window.history.back()}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          Go Back
        </button>
      </div>
    </div>
  );
};