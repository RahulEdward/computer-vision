'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FolderIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
  ArrowDownTrayIcon,
  ArrowUpTrayIcon,
  TagIcon,
  CalendarIcon,
  CodeBracketIcon,
  PlayIcon,
  DocumentDuplicateIcon,
  ArchiveBoxIcon,
} from '@heroicons/react/24/outline';
import { fileSystemService, SavedWorkflow, SavedScript, FileSystemStats } from '../../services/fileSystem';

interface FileManagerProps {
  onWorkflowSelect?: (workflow: SavedWorkflow) => void;
  onScriptSelect?: (script: SavedScript) => void;
  onCreateNew?: (type: 'workflow' | 'script') => void;
}

const FileManager: React.FC<FileManagerProps> = ({
  onWorkflowSelect,
  onScriptSelect,
  onCreateNew,
}) => {
  const [activeTab, setActiveTab] = useState<'workflows' | 'scripts'>('workflows');
  const [workflows, setWorkflows] = useState<SavedWorkflow[]>([]);
  const [scripts, setScripts] = useState<SavedScript[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [stats, setStats] = useState<FileSystemStats | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  const [editingItem, setEditingItem] = useState<{ type: 'workflow' | 'script'; id: string; name: string } | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = () => {
    setWorkflows(fileSystemService.getWorkflows());
    setScripts(fileSystemService.getScripts());
    setStats(fileSystemService.getStats());
  };

  const filteredWorkflows = workflows.filter(workflow => {
    const matchesSearch = !searchQuery || 
      workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      workflow.description?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesTags = selectedTags.length === 0 || 
      selectedTags.some(tag => workflow.tags?.includes(tag));
    
    return matchesSearch && matchesTags;
  });

  const filteredScripts = scripts.filter(script => {
    const matchesSearch = !searchQuery || 
      script.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      script.description?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesTags = selectedTags.length === 0 || 
      selectedTags.some(tag => script.tags?.includes(tag));
    
    return matchesSearch && matchesTags;
  });

  const allTags = [...new Set([
    ...workflows.flatMap(w => w.tags || []),
    ...scripts.flatMap(s => s.tags || [])
  ])];

  const handleDelete = async (type: 'workflow' | 'script', id: string) => {
    if (!confirm('Are you sure you want to delete this item?')) return;

    try {
      if (type === 'workflow') {
        await fileSystemService.deleteWorkflow(id);
      } else {
        await fileSystemService.deleteScript(id);
      }
      loadData();
    } catch (error) {
      console.error('Error deleting item:', error);
    }
  };

  const handleRename = async (type: 'workflow' | 'script', id: string, newName: string) => {
    try {
      if (type === 'workflow') {
        await fileSystemService.updateWorkflow(id, { name: newName });
      } else {
        await fileSystemService.updateScript(id, { name: newName });
      }
      loadData();
      setEditingItem(null);
    } catch (error) {
      console.error('Error renaming item:', error);
    }
  };

  const handleExport = async (type: 'workflow' | 'script', id: string) => {
    try {
      if (type === 'workflow') {
        await fileSystemService.exportWorkflow(id);
      } else {
        await fileSystemService.exportScript(id);
      }
    } catch (error) {
      console.error('Error exporting item:', error);
    }
  };

  const handleImport = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    try {
      for (const file of Array.from(files)) {
        if (file.name.endsWith('.json')) {
          // Import workflow
          await fileSystemService.importWorkflow(file);
        } else {
          // Import script
          await fileSystemService.importScript(file);
        }
      }
      loadData();
      setShowImportModal(false);
    } catch (error) {
      console.error('Error importing files:', error);
      alert('Error importing files: ' + (error as Error).message);
    }
  };

  const handleBackup = async () => {
    try {
      await fileSystemService.createBackup();
    } catch (error) {
      console.error('Error creating backup:', error);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString() + ' ' + new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-slate-950 to-slate-900 text-slate-100">
      {/* Header */}
      <div className="p-6 border-b border-slate-700/50 bg-gradient-to-r from-slate-900/50 to-slate-800/30 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-blue-500/20 to-blue-600/10 rounded-lg border border-blue-500/20">
              <FolderIcon className="w-6 h-6 text-blue-400" />
            </div>
            <h2 className="text-2xl font-bold text-slate-100 tracking-tight">File Manager</h2>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowImportModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-medium rounded-lg transition-all duration-200 shadow-lg shadow-blue-900/30"
            >
              <ArrowUpTrayIcon className="w-4 h-4" />
              <span>Import</span>
            </button>
            <button
              onClick={handleBackup}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 text-white font-medium rounded-lg transition-all duration-200 shadow-lg shadow-emerald-900/30"
            >
              <ArchiveBoxIcon className="w-4 h-4" />
              <span>Backup</span>
            </button>
          </div>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/40 backdrop-blur-sm rounded-xl p-4 border border-slate-600/30 shadow-lg shadow-slate-900/20">
              <div className="text-sm font-medium text-slate-400 mb-1">Workflows</div>
              <div className="text-2xl font-bold text-slate-100">{stats.totalWorkflows}</div>
            </div>
            <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/40 backdrop-blur-sm rounded-xl p-4 border border-slate-600/30 shadow-lg shadow-slate-900/20">
              <div className="text-sm font-medium text-slate-400 mb-1">Scripts</div>
              <div className="text-2xl font-bold text-slate-100">{stats.totalScripts}</div>
            </div>
            <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/40 backdrop-blur-sm rounded-xl p-4 border border-slate-600/30 shadow-lg shadow-slate-900/20">
              <div className="text-sm font-medium text-slate-400 mb-1">Storage</div>
              <div className="text-2xl font-bold text-slate-100">{formatFileSize(stats.storageUsed)}</div>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="flex space-x-2 bg-gradient-to-r from-slate-800/50 to-slate-700/30 backdrop-blur-sm rounded-xl p-2 border border-slate-600/30">
          <button
            onClick={() => setActiveTab('workflows')}
            className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
              activeTab === 'workflows' 
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-900/30' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
            }`}
          >
            <FolderIcon className="w-4 h-4" />
            <span>Workflows</span>
          </button>
          <button
            onClick={() => setActiveTab('scripts')}
            className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
              activeTab === 'scripts' 
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-900/30' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
            }`}
          >
            <CodeBracketIcon className="w-4 h-4" />
            <span>Scripts</span>
          </button>
        </div>

        {/* Search and Filters */}
        <div className="mt-6 space-y-4">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search files..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-800/60 to-slate-700/40 border border-slate-600/50 rounded-lg text-slate-100 placeholder-slate-400 focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all duration-200 shadow-inner"
            />
          </div>

          {allTags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {allTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => {
                    setSelectedTags(prev => 
                      prev.includes(tag) 
                        ? prev.filter(t => t !== tag)
                        : [...prev, tag]
                    );
                  }}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
                    selectedTags.includes(tag)
                      ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-900/30'
                      : 'bg-gradient-to-r from-slate-700/60 to-slate-600/40 text-slate-300 hover:from-slate-600/60 hover:to-slate-500/40 hover:text-slate-100 border border-slate-600/30'
                  }`}
                >
                  <TagIcon className="w-3 h-3" />
                  <span>{tag}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-slate-100 tracking-tight">
            {activeTab === 'workflows' ? 'Workflows' : 'Scripts'}
          </h3>
          <button
            onClick={() => onCreateNew?.(activeTab)}
            className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-medium rounded-lg transition-all duration-200 shadow-lg shadow-blue-900/30"
          >
            <PlusIcon className="w-4 h-4" />
            <span>New {activeTab === 'workflows' ? 'Workflow' : 'Script'}</span>
          </button>
        </div>

        <div className="space-y-2">
          {activeTab === 'workflows' ? (
            filteredWorkflows.length > 0 ? (
              filteredWorkflows.map(workflow => (
                <motion.div
                  key={workflow.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      {editingItem?.id === workflow.id ? (
                        <input
                          type="text"
                          value={editingItem.name}
                          onChange={(e) => setEditingItem({ ...editingItem, name: e.target.value })}
                          onBlur={() => handleRename('workflow', workflow.id, editingItem.name)}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              handleRename('workflow', workflow.id, editingItem.name);
                            }
                          }}
                          className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                          autoFocus
                        />
                      ) : (
                        <div
                          className="cursor-pointer"
                          onClick={() => onWorkflowSelect?.(workflow)}
                        >
                          <h4 className="font-medium text-white hover:text-purple-400 transition-colors">
                            {workflow.name}
                          </h4>
                          {workflow.description && (
                            <p className="text-sm text-gray-400 mt-1">{workflow.description}</p>
                          )}
                          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                            <div className="flex items-center space-x-1">
                              <CalendarIcon className="w-3 h-3" />
                              <span>{formatDate(workflow.updatedAt)}</span>
                            </div>
                            <div>{workflow.nodes.length} nodes</div>
                            <div>{workflow.edges.length} connections</div>
                          </div>
                          {workflow.tags && workflow.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {workflow.tags.map(tag => (
                                <span
                                  key={tag}
                                  className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center space-x-2 ml-4">
                      <button
                        onClick={() => onWorkflowSelect?.(workflow)}
                        className="p-1 text-gray-400 hover:text-green-400 transition-colors"
                        title="Open workflow"
                      >
                        <PlayIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingItem({ type: 'workflow', id: workflow.id, name: workflow.name })}
                        className="p-1 text-gray-400 hover:text-blue-400 transition-colors"
                        title="Rename"
                      >
                        <PencilIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleExport('workflow', workflow.id)}
                        className="p-1 text-gray-400 hover:text-purple-400 transition-colors"
                        title="Export"
                      >
                        <ArrowDownTrayIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete('workflow', workflow.id)}
                        className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <FolderIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No workflows found</p>
                <p className="text-sm">Create a new workflow to get started</p>
              </div>
            )
          ) : (
            filteredScripts.length > 0 ? (
              filteredScripts.map(script => (
                <motion.div
                  key={script.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      {editingItem?.id === script.id ? (
                        <input
                          type="text"
                          value={editingItem.name}
                          onChange={(e) => setEditingItem({ ...editingItem, name: e.target.value })}
                          onBlur={() => handleRename('script', script.id, editingItem.name)}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              handleRename('script', script.id, editingItem.name);
                            }
                          }}
                          className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                          autoFocus
                        />
                      ) : (
                        <div
                          className="cursor-pointer"
                          onClick={() => onScriptSelect?.(script)}
                        >
                          <h4 className="font-medium text-white hover:text-purple-400 transition-colors">
                            {script.name}
                          </h4>
                          {script.description && (
                            <p className="text-sm text-gray-400 mt-1">{script.description}</p>
                          )}
                          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                            <div className="flex items-center space-x-1">
                              <CalendarIcon className="w-3 h-3" />
                              <span>{formatDate(script.updatedAt)}</span>
                            </div>
                            <div className="px-2 py-1 bg-gray-700 rounded text-xs">
                              {script.language}
                            </div>
                            <div>{script.code.length} chars</div>
                          </div>
                          {script.tags && script.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {script.tags.map(tag => (
                                <span
                                  key={tag}
                                  className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center space-x-2 ml-4">
                      <button
                        onClick={() => onScriptSelect?.(script)}
                        className="p-1 text-gray-400 hover:text-green-400 transition-colors"
                        title="Open script"
                      >
                        <PlayIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingItem({ type: 'script', id: script.id, name: script.name })}
                        className="p-1 text-gray-400 hover:text-blue-400 transition-colors"
                        title="Rename"
                      >
                        <PencilIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleExport('script', script.id)}
                        className="p-1 text-gray-400 hover:text-purple-400 transition-colors"
                        title="Export"
                      >
                        <ArrowDownTrayIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete('script', script.id)}
                        className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <DocumentTextIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No scripts found</p>
                <p className="text-sm">Create a new script to get started</p>
              </div>
            )
          )}
        </div>
      </div>

      {/* Import Modal */}
      <AnimatePresence>
        {showImportModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowImportModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold mb-4">Import Files</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Select files to import
                  </label>
                  <input
                    type="file"
                    multiple
                    accept=".json,.js,.ts,.py,.txt,.md,.yml,.yaml"
                    onChange={(e) => handleImport(e.target.files)}
                    className="w-full p-2 bg-gray-700 border border-gray-600 rounded focus:border-purple-500 focus:outline-none"
                  />
                </div>
                <div className="text-sm text-gray-400">
                  <p>Supported formats:</p>
                  <ul className="list-disc list-inside mt-1 space-y-1">
                    <li>JSON files for workflows</li>
                    <li>JS, TS, PY, TXT, MD, YML files for scripts</li>
                  </ul>
                </div>
              </div>
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => setShowImportModal(false)}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FileManager;