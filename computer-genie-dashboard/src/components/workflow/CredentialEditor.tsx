import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  XMarkIcon,
  KeyIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  EyeIcon,
  EyeSlashIcon
} from '@heroicons/react/24/outline';
import { credentialManager } from '../../services/CredentialManager';
import { ICredentialType, ICredentialData, ICredentialProperty } from '../../types/credentials';

interface CredentialEditorProps {
  isOpen: boolean;
  onClose: () => void;
  credentialId?: string;
  credentialType?: string;
  onSave: (credential: ICredentialData) => void;
}

const CredentialEditor: React.FC<CredentialEditorProps> = ({
  isOpen,
  onClose,
  credentialId,
  credentialType,
  onSave
}) => {
  const [selectedType, setSelectedType] = useState<string>(credentialType || '');
  const [credentialName, setCredentialName] = useState<string>('');
  const [credentialData, setCredentialData] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [showPasswords, setShowPasswords] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const credentialTypes = credentialManager.getCredentialTypes();
  const selectedCredentialType = credentialTypes.find(type => type.name === selectedType);

  useEffect(() => {
    if (credentialId) {
      const credential = credentialManager.getCredential(credentialId);
      if (credential) {
        setSelectedType(credential.type);
        setCredentialName(credential.name);
        setCredentialData(credentialManager.decryptCredentialData(credential.data as any));
      }
    } else if (credentialType) {
      setSelectedType(credentialType);
    }
  }, [credentialId, credentialType]);

  const handleInputChange = (propertyName: string, value: any) => {
    setCredentialData(prev => ({
      ...prev,
      [propertyName]: value
    }));
    
    // Clear error when user starts typing
    if (errors[propertyName]) {
      setErrors(prev => ({
        ...prev,
        [propertyName]: ''
      }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!credentialName.trim()) {
      newErrors.name = 'Credential name is required';
    }
    
    if (!selectedType) {
      newErrors.type = 'Please select a credential type';
    }
    
    if (selectedCredentialType) {
      selectedCredentialType.properties.forEach(property => {
        if (property.required && !credentialData[property.name]) {
          newErrors[property.name] = `${property.displayName} is required`;
        }
      });
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = async () => {
    if (!validateForm()) return;
    
    setIsLoading(true);
    try {
      let savedCredential: ICredentialData;
      
      if (credentialId) {
        savedCredential = await credentialManager.updateCredential(credentialId, {
          name: credentialName,
          type: selectedType,
          data: credentialData,
          isActive: true
        });
      } else {
        savedCredential = await credentialManager.saveCredential({
          name: credentialName,
          type: selectedType,
          data: credentialData,
          isActive: true
        });
      }
      
      onSave(savedCredential);
      onClose();
    } catch (error) {
      console.error('Failed to save credential:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTest = async () => {
    if (!validateForm()) return;
    
    setIsLoading(true);
    setTestResult(null);
    
    try {
      const testCredential: ICredentialData = {
        id: 'test',
        name: credentialName,
        type: selectedType,
        data: credentialData,
        createdAt: new Date(),
        updatedAt: new Date(),
        isActive: true
      };
      
      const result = await credentialManager.testCredential(testCredential);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'Test failed'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const togglePasswordVisibility = (propertyName: string) => {
    setShowPasswords(prev => ({
      ...prev,
      [propertyName]: !prev[propertyName]
    }));
  };

  const renderProperty = (property: ICredentialProperty) => {
    const value = credentialData[property.name] || property.default || '';
    const isPassword = property.type === 'password' || property.typeOptions?.password;
    const showPassword = showPasswords[property.name];
    const hasError = errors[property.name];

    return (
      <div key={property.name} className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          {property.displayName}
          {property.required && <span className="text-red-500 ml-1">*</span>}
        </label>
        
        {property.description && (
          <p className="text-xs text-gray-500">{property.description}</p>
        )}
        
        <div className="relative">
          {property.type === 'options' ? (
            <select
              value={value}
              onChange={(e) => handleInputChange(property.name, e.target.value)}
              className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                hasError ? 'border-red-500' : 'border-gray-300'
              }`}
            >
              <option value="">Select an option</option>
              {property.options?.map(option => (
                <option key={option.value} value={option.value}>
                  {option.name}
                </option>
              ))}
            </select>
          ) : property.type === 'boolean' ? (
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => handleInputChange(property.name, e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm">{property.displayName}</span>
            </label>
          ) : (
            <>
              <input
                type={isPassword && !showPassword ? 'password' : property.type === 'number' ? 'number' : 'text'}
                value={value}
                onChange={(e) => handleInputChange(property.name, e.target.value)}
                placeholder={property.placeholder}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  hasError ? 'border-red-500' : 'border-gray-300'
                } ${isPassword ? 'pr-10' : ''}`}
              />
              {isPassword && (
                <button
                  type="button"
                  onClick={() => togglePasswordVisibility(property.name)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? (
                    <EyeSlashIcon className="h-4 w-4" />
                  ) : (
                    <EyeIcon className="h-4 w-4" />
                  )}
                </button>
              )}
            </>
          )}
        </div>
        
        {hasError && (
          <p className="text-sm text-red-500">{hasError}</p>
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex items-center space-x-3">
              <KeyIcon className="h-6 w-6 text-blue-600" />
              <h2 className="text-xl font-semibold">
                {credentialId ? 'Edit Credential' : 'Create Credential'}
              </h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
            <div className="space-y-6">
              {/* Credential Name */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Credential Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={credentialName}
                  onChange={(e) => setCredentialName(e.target.value)}
                  placeholder="Enter a name for this credential"
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    errors.name ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                {errors.name && (
                  <p className="text-sm text-red-500">{errors.name}</p>
                )}
              </div>

              {/* Credential Type */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Credential Type <span className="text-red-500">*</span>
                </label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  disabled={!!credentialId}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    errors.type ? 'border-red-500' : 'border-gray-300'
                  } ${credentialId ? 'bg-gray-100' : ''}`}
                >
                  <option value="">Select credential type</option>
                  {credentialTypes.map(type => (
                    <option key={type.name} value={type.name}>
                      {type.displayName}
                    </option>
                  ))}
                </select>
                {errors.type && (
                  <p className="text-sm text-red-500">{errors.type}</p>
                )}
              </div>

              {/* Credential Properties */}
              {selectedCredentialType && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    {selectedCredentialType.displayName} Configuration
                  </h3>
                  {selectedCredentialType.description && (
                    <p className="text-sm text-gray-600">
                      {selectedCredentialType.description}
                    </p>
                  )}
                  
                  <div className="space-y-4">
                    {selectedCredentialType.properties.map(renderProperty)}
                  </div>
                </div>
              )}

              {/* Test Result */}
              {testResult && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-4 rounded-md flex items-start space-x-3 ${
                    testResult.success 
                      ? 'bg-green-50 border border-green-200' 
                      : 'bg-red-50 border border-red-200'
                  }`}
                >
                  {testResult.success ? (
                    <CheckCircleIcon className="h-5 w-5 text-green-600 mt-0.5" />
                  ) : (
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-600 mt-0.5" />
                  )}
                  <div>
                    <p className={`text-sm font-medium ${
                      testResult.success ? 'text-green-800' : 'text-red-800'
                    }`}>
                      {testResult.success ? 'Test Successful' : 'Test Failed'}
                    </p>
                    <p className={`text-sm ${
                      testResult.success ? 'text-green-700' : 'text-red-700'
                    }`}>
                      {testResult.message}
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-6 border-t bg-gray-50">
            <button
              onClick={handleTest}
              disabled={isLoading || !selectedType}
              className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ShieldCheckIcon className="h-4 w-4" />
              <span>{isLoading ? 'Testing...' : 'Test Connection'}</span>
            </button>
            
            <div className="flex space-x-3">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Saving...' : credentialId ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default CredentialEditor;