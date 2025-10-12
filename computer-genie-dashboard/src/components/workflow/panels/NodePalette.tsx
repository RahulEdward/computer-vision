import React, { useState, useCallback, useMemo } from 'react';
import { 
  Search,
  ChevronDown,
  ChevronRight,
  Zap,
  Play,
  Code,
  Filter,
  Database,
  Globe,
  Send,
  FileText,
  Calendar,
  User,
  Settings,
  Folder,
  Plus,
  Star,
  Clock,
  Activity,
  Link,
  Shuffle,
  BarChart3,
  Shield,
  Cpu,
  Cloud,
  Smartphone,
  Mail,
  MessageSquare,
  Image,
  Video,
  Music,
  Download,
  Upload,
  Trash2,
  Edit3,
  Copy,
  Save,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Info,
  HelpCircle
} from 'lucide-react';

export interface NodeTemplate {
  id: string;
  type: 'enterprise' | 'group' | 'child';
  category: string;
  subcategory?: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  tags: string[];
  isPopular?: boolean;
  isNew?: boolean;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  defaultData: Record<string, any>;
  requiredInputs?: string[];
  providedOutputs?: string[];
  documentation?: string;
}

export interface NodeCategory {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  isExpanded: boolean;
  subcategories?: string[];
}

const nodeCategories: NodeCategory[] = [
  {
    id: 'triggers',
    name: 'Triggers',
    description: 'Start your workflows',
    icon: <Zap className="w-4 h-4" />,
    color: 'text-yellow-400',
    isExpanded: true,
  },
  {
    id: 'actions',
    name: 'Actions',
    description: 'Perform operations',
    icon: <Play className="w-4 h-4" />,
    color: 'text-blue-400',
    isExpanded: true,
    subcategories: ['http', 'database', 'file', 'email', 'api'],
  },
  {
    id: 'transforms',
    name: 'Transforms',
    description: 'Process and modify data',
    icon: <Code className="w-4 h-4" />,
    color: 'text-purple-400',
    isExpanded: false,
  },
  {
    id: 'conditions',
    name: 'Conditions',
    description: 'Control workflow flow',
    icon: <Filter className="w-4 h-4" />,
    color: 'text-orange-400',
    isExpanded: false,
  },
  {
    id: 'integrations',
    name: 'Integrations',
    description: 'Connect to external services',
    icon: <Link className="w-4 h-4" />,
    color: 'text-green-400',
    isExpanded: false,
    subcategories: ['cloud', 'social', 'productivity', 'analytics'],
  },
  {
    id: 'utilities',
    name: 'Utilities',
    description: 'Helper and utility nodes',
    icon: <Settings className="w-4 h-4" />,
    color: 'text-gray-400',
    isExpanded: false,
  },
  {
    id: 'groups',
    name: 'Groups',
    description: 'Organize workflow sections',
    icon: <Folder className="w-4 h-4" />,
    color: 'text-indigo-400',
    isExpanded: false,
  },
];

const nodeTemplates: NodeTemplate[] = [
  // Triggers
  {
    id: 'manual-trigger',
    type: 'child',
    category: 'triggers',
    name: 'Manual Trigger',
    description: 'Start workflow manually',
    icon: <Play className="w-4 h-4" />,
    color: 'text-yellow-400',
    tags: ['trigger', 'manual', 'start'],
    isPopular: true,
    difficulty: 'beginner',
    defaultData: {
      label: 'Manual Trigger',
      type: 'trigger',
      level: 0,
      status: 'idle',
    },
    providedOutputs: ['trigger_data'],
  },
  {
    id: 'webhook-trigger',
    type: 'child',
    category: 'triggers',
    name: 'Webhook',
    description: 'Trigger via HTTP webhook',
    icon: <Globe className="w-4 h-4" />,
    color: 'text-yellow-400',
    tags: ['trigger', 'webhook', 'http'],
    isPopular: true,
    difficulty: 'intermediate',
    defaultData: {
      label: 'Webhook Trigger',
      type: 'trigger',
      level: 0,
      status: 'idle',
      config: { method: 'POST', path: '/webhook' },
    },
    providedOutputs: ['webhook_data', 'headers', 'query_params'],
  },
  {
    id: 'schedule-trigger',
    type: 'child',
    category: 'triggers',
    name: 'Schedule',
    description: 'Trigger on schedule',
    icon: <Calendar className="w-4 h-4" />,
    color: 'text-yellow-400',
    tags: ['trigger', 'schedule', 'cron'],
    difficulty: 'intermediate',
    defaultData: {
      label: 'Schedule Trigger',
      type: 'trigger',
      level: 0,
      status: 'idle',
      config: { cron: '0 9 * * *' },
    },
    providedOutputs: ['timestamp'],
  },

  // Actions
  {
    id: 'http-request',
    type: 'child',
    category: 'actions',
    subcategory: 'http',
    name: 'HTTP Request',
    description: 'Make HTTP API calls',
    icon: <Globe className="w-4 h-4" />,
    color: 'text-blue-400',
    tags: ['action', 'http', 'api', 'request'],
    isPopular: true,
    difficulty: 'beginner',
    defaultData: {
      label: 'HTTP Request',
      type: 'action',
      category: 'http',
      level: 1,
      status: 'idle',
      config: { method: 'GET', url: '' },
    },
    requiredInputs: ['url'],
    providedOutputs: ['response', 'status_code', 'headers'],
  },
  {
    id: 'database-query',
    type: 'child',
    category: 'actions',
    subcategory: 'database',
    name: 'Database Query',
    description: 'Execute database queries',
    icon: <Database className="w-4 h-4" />,
    color: 'text-blue-400',
    tags: ['action', 'database', 'sql', 'query'],
    difficulty: 'intermediate',
    defaultData: {
      label: 'Database Query',
      type: 'action',
      category: 'database',
      level: 1,
      status: 'idle',
      config: { connection: '', query: '' },
    },
    requiredInputs: ['query'],
    providedOutputs: ['results', 'row_count'],
  },
  {
    id: 'send-email',
    type: 'child',
    category: 'actions',
    subcategory: 'email',
    name: 'Send Email',
    description: 'Send email messages',
    icon: <Mail className="w-4 h-4" />,
    color: 'text-blue-400',
    tags: ['action', 'email', 'notification'],
    isPopular: true,
    difficulty: 'beginner',
    defaultData: {
      label: 'Send Email',
      type: 'action',
      category: 'email',
      level: 1,
      status: 'idle',
      config: { to: '', subject: '', body: '' },
    },
    requiredInputs: ['to', 'subject', 'body'],
    providedOutputs: ['message_id', 'status'],
  },

  // Transforms
  {
    id: 'data-transform',
    type: 'child',
    category: 'transforms',
    name: 'Data Transform',
    description: 'Transform and map data',
    icon: <Code className="w-4 h-4" />,
    color: 'text-purple-400',
    tags: ['transform', 'data', 'mapping'],
    isPopular: true,
    difficulty: 'intermediate',
    defaultData: {
      label: 'Data Transform',
      type: 'transform',
      level: 1,
      status: 'idle',
      config: { script: '' },
    },
    requiredInputs: ['input_data'],
    providedOutputs: ['transformed_data'],
  },
  {
    id: 'json-parser',
    type: 'child',
    category: 'transforms',
    name: 'JSON Parser',
    description: 'Parse and extract JSON data',
    icon: <Code className="w-4 h-4" />,
    color: 'text-purple-400',
    tags: ['transform', 'json', 'parser'],
    difficulty: 'beginner',
    defaultData: {
      label: 'JSON Parser',
      type: 'transform',
      level: 1,
      status: 'idle',
      config: { path: '' },
    },
    requiredInputs: ['json_data'],
    providedOutputs: ['parsed_data'],
  },

  // Conditions
  {
    id: 'if-condition',
    type: 'child',
    category: 'conditions',
    name: 'If Condition',
    description: 'Conditional branching',
    icon: <Filter className="w-4 h-4" />,
    color: 'text-orange-400',
    tags: ['condition', 'if', 'branch'],
    isPopular: true,
    difficulty: 'beginner',
    defaultData: {
      label: 'If Condition',
      type: 'condition',
      level: 1,
      status: 'idle',
      config: { condition: '' },
    },
    requiredInputs: ['input_data'],
    providedOutputs: ['true_branch', 'false_branch'],
  },

  // Groups
  {
    id: 'workflow-group',
    type: 'group',
    category: 'groups',
    name: 'Workflow Group',
    description: 'Group related nodes',
    icon: <Folder className="w-4 h-4" />,
    color: 'text-indigo-400',
    tags: ['group', 'organization'],
    difficulty: 'beginner',
    defaultData: {
      label: 'New Group',
      type: 'group',
      level: 0,
      childIds: [],
      isCollapsed: false,
      color: 'blue',
    },
  },
];

interface NodePaletteProps {
  onNodeDragStart?: (event: React.DragEvent, nodeTemplate: NodeTemplate) => void;
  onAddNode?: (type: string, position?: { x: number; y: number }) => void;
  className?: string;
}

export const NodePalette: React.FC<NodePaletteProps> = ({ 
  onNodeDragStart, 
  onAddNode,
  className = '' 
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['triggers', 'actions'])
  );
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [showPopularOnly, setShowPopularOnly] = useState(false);

  const filteredTemplates = useMemo(() => {
    return nodeTemplates.filter(template => {
      const matchesSearch = searchTerm === '' || 
        template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesCategory = selectedCategory === null || template.category === selectedCategory;
      const matchesPopular = !showPopularOnly || template.isPopular;
      
      return matchesSearch && matchesCategory && matchesPopular;
    });
  }, [searchTerm, selectedCategory, showPopularOnly]);

  const groupedTemplates = useMemo(() => {
    const grouped: Record<string, NodeTemplate[]> = {};
    filteredTemplates.forEach(template => {
      if (!grouped[template.category]) {
        grouped[template.category] = [];
      }
      grouped[template.category].push(template);
    });
    return grouped;
  }, [filteredTemplates]);

  const toggleCategory = useCallback((categoryId: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(categoryId)) {
        newSet.delete(categoryId);
      } else {
        newSet.add(categoryId);
      }
      return newSet;
    });
  }, []);

  const handleDragStart = useCallback((event: React.DragEvent, template: NodeTemplate) => {
    if (onNodeDragStart) {
      onNodeDragStart(event, template);
    }
  }, [onNodeDragStart]);

  const handleNodeClick = useCallback((template: NodeTemplate) => {
    if (onAddNode) {
      onAddNode(template.id);
    }
  }, [onAddNode]);

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-400';
      case 'intermediate': return 'text-yellow-400';
      case 'advanced': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getDifficultyIcon = (difficulty?: string) => {
    switch (difficulty) {
      case 'beginner': return <CheckCircle className="w-3 h-3" />;
      case 'intermediate': return <AlertTriangle className="w-3 h-3" />;
      case 'advanced': return <Shield className="w-3 h-3" />;
      default: return <Info className="w-3 h-3" />;
    }
  };

  return (
    <div className={`bg-gray-900 border-r border-gray-700 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white mb-3">Node Palette</h2>
        
        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-400 focus:outline-none"
          />
        </div>

        {/* Filters */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowPopularOnly(!showPopularOnly)}
            className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
              showPopularOnly 
                ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-700' 
                : 'bg-gray-800 text-gray-400 border border-gray-600 hover:border-gray-500'
            }`}
          >
            <Star className="w-3 h-3" />
            <span>Popular</span>
          </button>
          
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              selectedCategory === null
                ? 'bg-blue-900/30 text-blue-400 border border-blue-700'
                : 'bg-gray-800 text-gray-400 border border-gray-600 hover:border-gray-500'
            }`}
          >
            All
          </button>
        </div>
      </div>

      {/* Categories and Nodes */}
      <div className="flex-1 overflow-y-auto">
        {nodeCategories.map(category => {
          const categoryTemplates = groupedTemplates[category.id] || [];
          const isExpanded = expandedCategories.has(category.id);
          const hasTemplates = categoryTemplates.length > 0;

          if (!hasTemplates && searchTerm) return null;

          return (
            <div key={category.id} className="border-b border-gray-800">
              {/* Category Header */}
              <button
                onClick={() => {
                  toggleCategory(category.id);
                  setSelectedCategory(selectedCategory === category.id ? null : category.id);
                }}
                className={`w-full p-3 flex items-center justify-between hover:bg-gray-800 transition-colors ${
                  selectedCategory === category.id ? 'bg-gray-800' : ''
                }`}
              >
                <div className="flex items-center space-x-2">
                  <div className={category.color}>
                    {category.icon}
                  </div>
                  <div className="text-left">
                    <div className="text-sm font-medium text-white">
                      {category.name}
                    </div>
                    <div className="text-xs text-gray-400">
                      {category.description}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {hasTemplates && (
                    <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">
                      {categoryTemplates.length}
                    </span>
                  )}
                  {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  )}
                </div>
              </button>

              {/* Category Nodes */}
              {isExpanded && hasTemplates && (
                <div className="pb-2">
                  {categoryTemplates.map(template => (
                    <div
                      key={template.id}
                      draggable
                      onDragStart={(e) => handleDragStart(e, template)}
                      onClick={() => handleNodeClick(template)}
                      className="mx-3 mb-2 p-3 bg-gray-800 border border-gray-700 rounded-lg cursor-grab hover:border-gray-600 hover:bg-gray-750 transition-all duration-200 active:cursor-grabbing"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-2 flex-1">
                          <div className={template.color}>
                            {template.icon}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2">
                              <span className="text-sm font-medium text-white truncate">
                                {template.name}
                              </span>
                              {template.isNew && (
                                <span className="px-1 py-0.5 bg-green-900/30 text-green-400 text-xs rounded">
                                  NEW
                                </span>
                              )}
                              {template.isPopular && (
                                <Star className="w-3 h-3 text-yellow-400 fill-current" />
                              )}
                            </div>
                            <p className="text-xs text-gray-400 mt-1">
                              {template.description}
                            </p>
                            
                            {/* Tags */}
                            <div className="flex flex-wrap gap-1 mt-2">
                              {template.tags.slice(0, 3).map(tag => (
                                <span
                                  key={tag}
                                  className="px-1 py-0.5 bg-gray-700 text-gray-300 text-xs rounded"
                                >
                                  {tag}
                                </span>
                              ))}
                              {template.tags.length > 3 && (
                                <span className="text-xs text-gray-500">
                                  +{template.tags.length - 3}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        
                        {/* Difficulty indicator */}
                        {template.difficulty && (
                          <div className={`flex items-center space-x-1 ${getDifficultyColor(template.difficulty)}`}>
                            {getDifficultyIcon(template.difficulty)}
                            <span className="text-xs capitalize">
                              {template.difficulty}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Input/Output indicators */}
                      {(template.requiredInputs || template.providedOutputs) && (
                        <div className="mt-2 pt-2 border-t border-gray-700 flex justify-between text-xs">
                          {template.requiredInputs && (
                            <div className="text-red-400">
                              Inputs: {template.requiredInputs.length}
                            </div>
                          )}
                          {template.providedOutputs && (
                            <div className="text-green-400">
                              Outputs: {template.providedOutputs.length}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-700 text-xs text-gray-400">
        <div className="flex justify-between">
          <span>{filteredTemplates.length} nodes available</span>
          <span>Drag to canvas</span>
        </div>
      </div>
    </div>
  );
};