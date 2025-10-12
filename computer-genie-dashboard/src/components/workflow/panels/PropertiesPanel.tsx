import React, { useState, useCallback, useMemo } from 'react';
import {
  Settings,
  Save,
  RotateCcw,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle,
  Info,
  HelpCircle,
  ChevronDown,
  ChevronRight,
  Plus,
  Minus,
  Edit3,
  Code,
  Database,
  Globe,
  Calendar,
  Clock,
  User,
  Tag,
  FileText,
  Link,
  Zap,
  Activity,
  BarChart3,
  Shield,
  Cpu,
  Memory,
  Timer,
  RefreshCw,
  Play,
  Pause,
  Square,
  X
} from 'lucide-react';

export interface PropertyField {
  key: string;
  label: string;
  type: 'text' | 'number' | 'boolean' | 'select' | 'textarea' | 'json' | 'code' | 'url' | 'email' | 'password' | 'date' | 'time' | 'color' | 'file' | 'array' | 'object';
  value: any;
  description?: string;
  placeholder?: string;
  required?: boolean;
  readonly?: boolean;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    custom?: (value: any) => string | null;
  };
  options?: Array<{ label: string; value: any; description?: string }>;
  group?: string;
  conditional?: {
    field: string;
    value: any;
    operator?: 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than';
  };
  advanced?: boolean;
}

export interface PropertyGroup {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  isExpanded: boolean;
  isAdvanced?: boolean;
}

export interface SelectedNodeData {
  id: string;
  type: string;
  label: string;
  description?: string;
  category?: string;
  status?: string;
  properties: PropertyField[];
  groups: PropertyGroup[];
  metadata?: {
    createdAt?: number;
    updatedAt?: number;
    createdBy?: string;
    version?: string;
    tags?: string[];
  };
  performance?: {
    executionTime?: number;
    memoryUsage?: number;
    cpuUsage?: number;
    lastExecuted?: number;
    executionCount?: number;
    successRate?: number;
  };
  validation?: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  };
  connections?: {
    inputs: number;
    outputs: number;
    maxInputs?: number;
    maxOutputs?: number;
  };
}

interface PropertiesPanelProps {
  selectedNode: SelectedNodeData | null;
  onPropertyChange: (nodeId: string, key: string, value: any) => void;
  onNodeUpdate: (nodeId: string, updates: Partial<SelectedNodeData>) => void;
  onNodeDelete: (nodeId: string) => void;
  onNodeDuplicate: (nodeId: string) => void;
  onNodeExecute?: (nodeId: string) => void;
  className?: string;
}

export const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  selectedNode,
  onPropertyChange,
  onNodeUpdate,
  onNodeDelete,
  onNodeDuplicate,
  onNodeExecute,
  className = ''
}) => {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['basic']));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [editingLabel, setEditingLabel] = useState(false);
  const [tempLabel, setTempLabel] = useState('');
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroups(prev => {
      const newSet = new Set(prev);
      if (newSet.has(groupId)) {
        newSet.delete(groupId);
      } else {
        newSet.add(groupId);
      }
      return newSet;
    });
  }, []);

  const handlePropertyChange = useCallback((key: string, value: any) => {
    if (!selectedNode) return;

    // Validate the value
    const property = selectedNode.properties.find(p => p.key === key);
    if (property?.validation) {
      const error = validateProperty(property, value);
      setValidationErrors(prev => ({
        ...prev,
        [key]: error || ''
      }));
    }

    onPropertyChange(selectedNode.id, key, value);
  }, [selectedNode, onPropertyChange]);

  const validateProperty = (property: PropertyField, value: any): string | null => {
    if (property.required && (value === undefined || value === null || value === '')) {
      return 'This field is required';
    }

    if (property.validation) {
      const { min, max, pattern, custom } = property.validation;

      if (typeof value === 'number') {
        if (min !== undefined && value < min) {
          return `Value must be at least ${min}`;
        }
        if (max !== undefined && value > max) {
          return `Value must be at most ${max}`;
        }
      }

      if (typeof value === 'string') {
        if (min !== undefined && value.length < min) {
          return `Must be at least ${min} characters`;
        }
        if (max !== undefined && value.length > max) {
          return `Must be at most ${max} characters`;
        }
        if (pattern && !new RegExp(pattern).test(value)) {
          return 'Invalid format';
        }
      }

      if (custom) {
        return custom(value);
      }
    }

    return null;
  };

  const handleLabelEdit = useCallback(() => {
    if (!selectedNode) return;
    setTempLabel(selectedNode.label);
    setEditingLabel(true);
  }, [selectedNode]);

  const handleLabelSave = useCallback(() => {
    if (!selectedNode) return;
    onNodeUpdate(selectedNode.id, { label: tempLabel });
    setEditingLabel(false);
  }, [selectedNode, tempLabel, onNodeUpdate]);

  const handleLabelCancel = useCallback(() => {
    setEditingLabel(false);
    setTempLabel('');
  }, []);

  const groupedProperties = useMemo(() => {
    if (!selectedNode) return {};

    const grouped: Record<string, PropertyField[]> = {};
    selectedNode.properties.forEach(property => {
      const group = property.group || 'basic';
      if (!grouped[group]) {
        grouped[group] = [];
      }

      // Check conditional visibility
      if (property.conditional) {
        const conditionField = selectedNode.properties.find(p => p.key === property.conditional!.field);
        if (conditionField) {
          const { value: conditionValue, operator = 'equals' } = property.conditional;
          const fieldValue = conditionField.value;

          let isVisible = false;
          switch (operator) {
            case 'equals':
              isVisible = fieldValue === conditionValue;
              break;
            case 'not_equals':
              isVisible = fieldValue !== conditionValue;
              break;
            case 'contains':
              isVisible = String(fieldValue).includes(String(conditionValue));
              break;
            case 'greater_than':
              isVisible = Number(fieldValue) > Number(conditionValue);
              break;
            case 'less_than':
              isVisible = Number(fieldValue) < Number(conditionValue);
              break;
          }

          if (!isVisible) return;
        }
      }

      // Check advanced visibility
      if (property.advanced && !showAdvanced) return;

      grouped[group].push(property);
    });

    return grouped;
  }, [selectedNode, showAdvanced]);

  const renderPropertyField = useCallback((property: PropertyField) => {
    const error = validationErrors[property.key];
    const hasError = !!error;

    const baseInputClasses = `w-full px-3 py-2 bg-gray-800 border rounded-lg text-white placeholder-gray-400 focus:outline-none transition-colors ${
      hasError 
        ? 'border-red-500 focus:border-red-400' 
        : 'border-gray-600 focus:border-blue-400'
    } ${property.readonly ? 'opacity-50 cursor-not-allowed' : ''}`;

    const renderInput = () => {
      switch (property.type) {
        case 'text':
        case 'email':
        case 'url':
        case 'password':
          return (
            <input
              type={property.type}
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              placeholder={property.placeholder}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );

        case 'number':
          return (
            <input
              type="number"
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, Number(e.target.value))}
              placeholder={property.placeholder}
              min={property.validation?.min}
              max={property.validation?.max}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );

        case 'boolean':
          return (
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={property.value || false}
                onChange={(e) => handlePropertyChange(property.key, e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                disabled={property.readonly}
              />
              <span className="text-sm text-gray-300">
                {property.value ? 'Enabled' : 'Disabled'}
              </span>
            </label>
          );

        case 'select':
          return (
            <select
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              className={baseInputClasses}
              disabled={property.readonly}
            >
              <option value="">Select an option...</option>
              {property.options?.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          );

        case 'textarea':
          return (
            <textarea
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              placeholder={property.placeholder}
              rows={4}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );

        case 'code':
        case 'json':
          return (
            <div className="relative">
              <textarea
                value={property.value || ''}
                onChange={(e) => handlePropertyChange(property.key, e.target.value)}
                placeholder={property.placeholder}
                rows={6}
                className={`${baseInputClasses} font-mono text-sm`}
                readOnly={property.readonly}
              />
              <div className="absolute top-2 right-2">
                <Code className="w-4 h-4 text-gray-400" />
              </div>
            </div>
          );

        case 'date':
          return (
            <input
              type="date"
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );

        case 'time':
          return (
            <input
              type="time"
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );

        case 'color':
          return (
            <div className="flex items-center space-x-2">
              <input
                type="color"
                value={property.value || '#000000'}
                onChange={(e) => handlePropertyChange(property.key, e.target.value)}
                className="w-12 h-10 border border-gray-600 rounded cursor-pointer"
                disabled={property.readonly}
              />
              <input
                type="text"
                value={property.value || ''}
                onChange={(e) => handlePropertyChange(property.key, e.target.value)}
                placeholder="#000000"
                className={`${baseInputClasses} flex-1`}
                readOnly={property.readonly}
              />
            </div>
          );

        case 'array':
          const arrayValue = Array.isArray(property.value) ? property.value : [];
          return (
            <div className="space-y-2">
              {arrayValue.map((item, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={item}
                    onChange={(e) => {
                      const newArray = [...arrayValue];
                      newArray[index] = e.target.value;
                      handlePropertyChange(property.key, newArray);
                    }}
                    className={baseInputClasses}
                    readOnly={property.readonly}
                  />
                  {!property.readonly && (
                    <button
                      onClick={() => {
                        const newArray = arrayValue.filter((_, i) => i !== index);
                        handlePropertyChange(property.key, newArray);
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <Minus className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
              {!property.readonly && (
                <button
                  onClick={() => {
                    handlePropertyChange(property.key, [...arrayValue, '']);
                  }}
                  className="flex items-center space-x-1 text-blue-400 hover:text-blue-300 text-sm"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add item</span>
                </button>
              )}
            </div>
          );

        default:
          return (
            <input
              type="text"
              value={property.value || ''}
              onChange={(e) => handlePropertyChange(property.key, e.target.value)}
              placeholder={property.placeholder}
              className={baseInputClasses}
              readOnly={property.readonly}
            />
          );
      }
    };

    return (
      <div key={property.key} className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-300 flex items-center space-x-1">
            <span>{property.label}</span>
            {property.required && <span className="text-red-400">*</span>}
            {property.readonly && <Lock className="w-3 h-3 text-gray-500" />}
          </label>
          {property.description && (
            <div className="group relative">
              <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
              <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block bg-gray-900 text-white text-xs rounded p-2 whitespace-nowrap z-10 border border-gray-700">
                {property.description}
              </div>
            </div>
          )}
        </div>
        
        {renderInput()}
        
        {hasError && (
          <div className="flex items-center space-x-1 text-red-400 text-xs">
            <AlertTriangle className="w-3 h-3" />
            <span>{error}</span>
          </div>
        )}
      </div>
    );
  }, [validationErrors, handlePropertyChange]);

  if (!selectedNode) {
    return (
      <div className={`bg-gray-900 border-l border-gray-700 flex flex-col items-center justify-center ${className}`}>
        <div className="text-center p-8">
          <Settings className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-400 mb-2">No Node Selected</h3>
          <p className="text-sm text-gray-500">
            Select a node to view and edit its properties
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900 border-l border-gray-700 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          {editingLabel ? (
            <div className="flex items-center space-x-2 flex-1">
              <input
                type="text"
                value={tempLabel}
                onChange={(e) => setTempLabel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleLabelSave();
                  if (e.key === 'Escape') handleLabelCancel();
                }}
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white text-lg font-semibold focus:border-blue-400 focus:outline-none"
                autoFocus
              />
              <button
                onClick={handleLabelSave}
                className="text-green-400 hover:text-green-300"
              >
                <CheckCircle className="w-4 h-4" />
              </button>
              <button
                onClick={handleLabelCancel}
                className="text-red-400 hover:text-red-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center space-x-2 flex-1">
              <h2 className="text-lg font-semibold text-white truncate">
                {selectedNode.label}
              </h2>
              <button
                onClick={handleLabelEdit}
                className="text-gray-400 hover:text-white"
              >
                <Edit3 className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-1 bg-blue-900/30 text-blue-400 rounded text-xs">
              {selectedNode.type}
            </span>
            {selectedNode.category && (
              <span className="px-2 py-1 bg-gray-800 text-gray-300 rounded text-xs">
                {selectedNode.category}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-1">
            {onNodeExecute && (
              <button
                onClick={() => onNodeExecute(selectedNode.id)}
                className="text-green-400 hover:text-green-300"
                title="Execute"
              >
                <Play className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => onNodeDuplicate(selectedNode.id)}
              className="text-blue-400 hover:text-blue-300"
              title="Duplicate"
            >
              <Copy className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNodeDelete(selectedNode.id)}
              className="text-red-400 hover:text-red-300"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Status and validation */}
        <div className="mt-3 space-y-2">
          {selectedNode.status && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Status:</span>
              <span className={`text-xs px-2 py-1 rounded ${
                selectedNode.status === 'completed' ? 'bg-green-900/30 text-green-400' :
                selectedNode.status === 'error' ? 'bg-red-900/30 text-red-400' :
                selectedNode.status === 'running' ? 'bg-blue-900/30 text-blue-400' :
                'bg-gray-800 text-gray-300'
              }`}>
                {selectedNode.status}
              </span>
            </div>
          )}

          {selectedNode.validation && !selectedNode.validation.isValid && (
            <div className="bg-red-900/20 border border-red-700 rounded p-2">
              <div className="flex items-center space-x-1 text-red-400 text-xs font-medium">
                <AlertTriangle className="w-3 h-3" />
                <span>Validation Issues</span>
              </div>
              {selectedNode.validation.errors.length > 0 && (
                <div className="mt-1 space-y-1">
                  {selectedNode.validation.errors.map((error, index) => (
                    <div key={index} className="text-xs text-red-300">
                      • {error}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Advanced toggle */}
        <div className="mt-3 flex items-center justify-between">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-1 text-xs text-gray-400 hover:text-gray-300"
          >
            <Settings className="w-3 h-3" />
            <span>Advanced</span>
            {showAdvanced ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Properties */}
      <div className="flex-1 overflow-y-auto">
        {selectedNode.groups.map(group => {
          const groupProperties = groupedProperties[group.id] || [];
          const isExpanded = expandedGroups.has(group.id);
          
          if (groupProperties.length === 0) return null;
          if (group.isAdvanced && !showAdvanced) return null;

          return (
            <div key={group.id} className="border-b border-gray-800">
              <button
                onClick={() => toggleGroup(group.id)}
                className="w-full p-3 flex items-center justify-between hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center space-x-2">
                  {group.icon && <div className="text-gray-400">{group.icon}</div>}
                  <div className="text-left">
                    <div className="text-sm font-medium text-white">
                      {group.label}
                    </div>
                    {group.description && (
                      <div className="text-xs text-gray-400">
                        {group.description}
                      </div>
                    )}
                  </div>
                </div>
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                )}
              </button>

              {isExpanded && (
                <div className="p-4 space-y-4">
                  {groupProperties.map(renderPropertyField)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer with metadata */}
      {selectedNode.metadata && (
        <div className="p-4 border-t border-gray-700 space-y-2">
          <div className="text-xs text-gray-400 font-medium">Metadata</div>
          <div className="space-y-1 text-xs">
            {selectedNode.metadata.createdBy && (
              <div className="flex justify-between">
                <span className="text-gray-400">Created by:</span>
                <span className="text-white">{selectedNode.metadata.createdBy}</span>
              </div>
            )}
            {selectedNode.metadata.version && (
              <div className="flex justify-between">
                <span className="text-gray-400">Version:</span>
                <span className="text-white">{selectedNode.metadata.version}</span>
              </div>
            )}
            {selectedNode.metadata.createdAt && (
              <div className="flex justify-between">
                <span className="text-gray-400">Created:</span>
                <span className="text-white">
                  {new Date(selectedNode.metadata.createdAt).toLocaleDateString()}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};