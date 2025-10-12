import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PlusIcon, KeyIcon } from '@heroicons/react/24/outline';
import { INodeProperties, INodePropertyOptions } from '../../types/n8nTypes';
import { credentialManager } from '../../services/CredentialManager';
import CredentialEditor from './CredentialEditor';

interface NodePropertyEditorProps {
  nodeId: string;
  properties: INodeProperties[];
  values: Record<string, any>;
  onChange: (nodeId: string, values: Record<string, any>) => void;
  onClose: () => void;
}

const NodePropertyEditor: React.FC<NodePropertyEditorProps> = ({
  nodeId,
  properties,
  values,
  onChange,
  onClose
}) => {
  const [localValues, setLocalValues] = useState<Record<string, any>>(values);
  const [showCredentialEditor, setShowCredentialEditor] = useState(false);
  const [credentialType, setCredentialType] = useState<string>('');

  useEffect(() => {
    setLocalValues(values);
  }, [values]);

  const handleValueChange = (propertyName: string, value: any) => {
    const newValues = { ...localValues, [propertyName]: value };
    setLocalValues(newValues);
    onChange(nodeId, newValues);
  };

  const shouldShowProperty = (property: INodeProperties): boolean => {
    if (!property.displayOptions?.show) return true;
    
    for (const [fieldName, expectedValues] of Object.entries(property.displayOptions.show)) {
      const currentValue = localValues[fieldName];
      if (!expectedValues.includes(currentValue)) {
        return false;
      }
    }
    return true;
  };

  const renderProperty = (property: INodeProperties) => {
    if (!shouldShowProperty(property)) return null;
    
    const value = localValues[property.name] || property.default;

    switch (property.type) {
      case 'string':
        return (
          <input
            type="text"
            value={value || ''}
            onChange={(e) => handleValueChange(property.name, e.target.value)}
            placeholder={property.placeholder}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        );

      case 'number':
        return (
          <input
            type="number"
            value={value || ''}
            onChange={(e) => handleValueChange(property.name, parseFloat(e.target.value) || 0)}
            placeholder={property.placeholder}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        );

      case 'boolean':
        return (
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={value || false}
              onChange={(e) => handleValueChange(property.name, e.target.checked)}
              className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
            />
            <span className="text-sm text-gray-700">
              {property.description || 'Enable this option'}
            </span>
          </label>
        );

      case 'options':
        return (
          <select
            value={value || ''}
            onChange={(e) => handleValueChange(property.name, e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">Select an option...</option>
            {property.options?.map((option: INodePropertyOptions, index: number) => (
              <option key={`${option.name}-${index}`} value={String(option.value)}>
                {option.name}
              </option>
            ))}
          </select>
        );

      case 'multiOptions':
        return (
          <div className="space-y-2">
            {property.options?.map((option: INodePropertyOptions, idx: number) => (
              <label key={`${option.name}-${idx}`} className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={(value || []).includes(option.value)}
                  onChange={(e) => {
                    const currentValues = value || [];
                    const newValues = e.target.checked
                      ? [...currentValues, option.value]
                      : currentValues.filter((v: any) => v !== option.value);
                    handleValueChange(property.name, newValues);
                  }}
                  className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <span className="text-sm text-gray-700">{option.name}</span>
              </label>
            ))}
          </div>
        );

      case 'json':
        return (
          <textarea
            value={typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                handleValueChange(property.name, parsed);
              } catch {
                handleValueChange(property.name, e.target.value);
              }
            }}
            placeholder={property.placeholder || '{}'}
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent font-mono text-sm"
          />
        );

      case 'collection':
        return (
          <div className="space-y-2">
            <div className="text-sm text-gray-600">Collection properties:</div>
            {property.options?.map((option: INodePropertyOptions, i: number) => (
              <div key={`collection-${i}`} className="pl-4 border-l-2 border-gray-200">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {option.name}
                </label>
                {/* Recursively render nested properties if they exist */}
                <input
                  type="text"
                  value={(value && value[String(option.value)]) || ''}
                  onChange={(e) => {
                    const newValue = { ...value, [String(option.value)]: e.target.value };
                    handleValueChange(property.name, newValue);
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            ))}
          </div>
        );

      case 'credentialsSelect':
        const availableCredentials = credentialManager.getCredentials()
          .filter(cred => !property.credentialTypes || property.credentialTypes.includes(cred.type));
        
        return (
          <div className="space-y-2">
            <div className="flex space-x-2">
              <select
                value={value || ''}
                onChange={(e) => handleValueChange(property.name, e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="">Select credential</option>
                {availableCredentials.map(cred => (
                  <option key={cred.id} value={cred.id}>
                    {cred.name} ({cred.type})
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  setCredentialType(property.credentialTypes?.[0] || '');
                  setShowCredentialEditor(true);
                }}
                className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 flex items-center space-x-1"
                title="Create new credential"
              >
                <PlusIcon className="h-4 w-4" />
                <KeyIcon className="h-4 w-4" />
              </button>
            </div>
            {property.credentialTypes && property.credentialTypes.length > 0 && (
              <p className="text-xs text-gray-500">
                Supported types: {property.credentialTypes.join(', ')}
              </p>
            )}
          </div>
        );

      default:
        return (
          <input
            type="text"
            value={value || ''}
            onChange={(e) => handleValueChange(property.name, e.target.value)}
            placeholder={property.placeholder}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        );
    }
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        className="fixed right-0 top-0 h-full w-96 bg-white shadow-xl border-l border-gray-200 z-50 flex flex-col"
      >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Node Properties</h3>
        <button
          onClick={onClose}
          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Properties */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-6">
          {properties.map((property) => (
            <div key={property.name} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {property.displayName}
                {property.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              
              {property.description && (
                <p className="text-xs text-gray-500">{property.description}</p>
              )}
              
              {renderProperty(property)}
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Close
          </button>
          <button
            onClick={() => {
              // Apply changes and close
              onChange(nodeId, localValues);
              onClose();
            }}
            className="flex-1 px-4 py-2 text-sm font-medium text-white bg-purple-600 border border-transparent rounded-lg hover:bg-purple-700 transition-colors"
          >
            Apply
          </button>
        </div>
      </div>
      </motion.div>

      <CredentialEditor
        isOpen={showCredentialEditor}
        onClose={() => setShowCredentialEditor(false)}
        credentialType={credentialType}
        onSave={(credential) => {
          setShowCredentialEditor(false);
          // Optionally auto-select the newly created credential
          const credentialProperty = properties.find(p => p.type === 'credentialsSelect');
          if (credentialProperty) {
            handleValueChange(credentialProperty.name, credential.id);
          }
        }}
      />
    </>
  );
};

export default NodePropertyEditor;