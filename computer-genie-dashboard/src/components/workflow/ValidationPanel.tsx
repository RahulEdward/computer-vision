'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ExclamationTriangleIcon, 
  ExclamationCircleIcon,
  CheckCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { IValidationResult, IValidationError } from '../../services/NodeValidator';

interface ValidationPanelProps {
  validationResults: Record<string, IValidationResult>;
  isOpen: boolean;
  onClose: () => void;
  onNodeSelect?: (nodeId: string) => void;
}

export const ValidationPanel: React.FC<ValidationPanelProps> = ({
  validationResults,
  isOpen,
  onClose,
  onNodeSelect
}) => {
  const totalErrors = Object.values(validationResults).reduce(
    (sum, result) => sum + result.errors.length, 0
  );
  
  const totalWarnings = Object.values(validationResults).reduce(
    (sum, result) => sum + result.warnings.length, 0
  );

  const hasIssues = totalErrors > 0 || totalWarnings > 0;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed right-0 top-0 h-full w-96 bg-white shadow-xl border-l border-gray-200 z-50 flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center gap-2">
              {hasIssues ? (
                <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />
              ) : (
                <CheckCircleIcon className="w-5 h-5 text-green-500" />
              )}
              <h2 className="text-lg font-semibold">
                Workflow Validation
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Summary */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <ExclamationCircleIcon className="w-4 h-4 text-red-500" />
                <span className="text-sm font-medium text-red-700">
                  {totalErrors} Errors
                </span>
              </div>
              <div className="flex items-center gap-1">
                <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500" />
                <span className="text-sm font-medium text-yellow-700">
                  {totalWarnings} Warnings
                </span>
              </div>
            </div>
            
            {!hasIssues && (
              <div className="flex items-center gap-2 mt-2">
                <CheckCircleIcon className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-700">
                  All nodes are valid
                </span>
              </div>
            )}
          </div>

          {/* Validation Results */}
          <div className="flex-1 overflow-y-auto">
            {Object.entries(validationResults).map(([nodeId, result]) => (
              <NodeValidationItem
                key={nodeId}
                nodeId={nodeId}
                result={result}
                onNodeSelect={onNodeSelect}
              />
            ))}
          </div>

          {/* Actions */}
          {hasIssues && (
            <div className="p-4 border-t border-gray-200">
              <button
                onClick={() => {
                  // Auto-fix common issues
                  console.log('Auto-fixing validation issues...');
                }}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Auto-fix Common Issues
              </button>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

interface NodeValidationItemProps {
  nodeId: string;
  result: IValidationResult;
  onNodeSelect?: (nodeId: string) => void;
}

const NodeValidationItem: React.FC<NodeValidationItemProps> = ({
  nodeId,
  result,
  onNodeSelect
}) => {
  const hasIssues = result.errors.length > 0 || result.warnings.length > 0;

  if (!hasIssues) {
    return null;
  }

  return (
    <div className="p-4 border-b border-gray-100">
      <button
        onClick={() => onNodeSelect?.(nodeId)}
        className="w-full text-left hover:bg-gray-50 p-2 rounded-lg transition-colors"
      >
        <div className="flex items-center gap-2 mb-2">
          {result.errors.length > 0 ? (
            <ExclamationCircleIcon className="w-4 h-4 text-red-500" />
          ) : (
            <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500" />
          )}
          <span className="font-medium text-gray-900">
            {nodeId}
          </span>
        </div>
      </button>

      <div className="space-y-2 ml-6">
        {/* Errors */}
        {result.errors.map((error, index) => (
          <ValidationErrorItem
            key={`error-${index}`}
            error={error}
            type="error"
          />
        ))}

        {/* Warnings */}
        {result.warnings.map((warning, index) => (
          <ValidationErrorItem
            key={`warning-${index}`}
            error={warning}
            type="warning"
          />
        ))}
      </div>
    </div>
  );
};

interface ValidationErrorItemProps {
  error: IValidationError;
  type: 'error' | 'warning';
}

const ValidationErrorItem: React.FC<ValidationErrorItemProps> = ({
  error,
  type
}) => {
  return (
    <div className={`p-2 rounded-lg ${
      type === 'error' 
        ? 'bg-red-50 border border-red-200' 
        : 'bg-yellow-50 border border-yellow-200'
    }`}>
      <div className="flex items-start gap-2">
        {type === 'error' ? (
          <ExclamationCircleIcon className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
        ) : (
          <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium ${
            type === 'error' ? 'text-red-800' : 'text-yellow-800'
          }`}>
            {error.field}
          </p>
          <p className={`text-xs ${
            type === 'error' ? 'text-red-600' : 'text-yellow-600'
          }`}>
            {error.message}
          </p>
        </div>
      </div>
    </div>
  );
};

// Validation status indicator for nodes
interface ValidationStatusProps {
  result: IValidationResult;
  size?: 'sm' | 'md' | 'lg';
}

export const ValidationStatus: React.FC<ValidationStatusProps> = ({
  result,
  size = 'md'
}) => {
  const iconSize = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  }[size];

  if (result.errors.length > 0) {
    return (
      <div className="flex items-center gap-1">
        <ExclamationCircleIcon className={`${iconSize} text-red-500`} />
        <span className="text-xs text-red-600">{result.errors.length}</span>
      </div>
    );
  }

  if (result.warnings.length > 0) {
    return (
      <div className="flex items-center gap-1">
        <ExclamationTriangleIcon className={`${iconSize} text-yellow-500`} />
        <span className="text-xs text-yellow-600">{result.warnings.length}</span>
      </div>
    );
  }

  return (
    <CheckCircleIcon className={`${iconSize} text-green-500`} />
  );
};