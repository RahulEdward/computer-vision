'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSemanticSearch } from '@/lib/semantic-search';
import { useDashboardStore } from '@/lib/store';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: 'workflow' | 'node' | 'template' | 'documentation';
  tags: string[];
  score: number;
}

export default function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [showResults, setShowResults] = useState(false);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const { search, addItems, getSuggestions, isReady } = useSemanticSearch();
  const { recordInteraction } = useDashboardStore();

  // Sample data for demonstration
  useEffect(() => {
    if (isReady) {
      addItems([
        {
          id: '1',
          title: 'HTTP Request Workflow',
          description: 'Make HTTP requests to external APIs and process responses',
          tags: ['http', 'api', 'request', 'webhook'],
          type: 'workflow',
          content: 'This workflow allows you to make HTTP requests to any REST API endpoint'
        },
        {
          id: '2',
          title: 'Email Notification Node',
          description: 'Send email notifications with custom templates',
          tags: ['email', 'notification', 'smtp', 'template'],
          type: 'node',
          content: 'Configure SMTP settings and send personalized emails'
        },
        {
          id: '3',
          title: 'Database Query Template',
          description: 'Template for querying SQL databases',
          tags: ['database', 'sql', 'query', 'mysql', 'postgresql'],
          type: 'template',
          content: 'Pre-built template for common database operations'
        },
        {
          id: '4',
          title: 'File Processing Workflow',
          description: 'Process and transform files automatically',
          tags: ['file', 'processing', 'transform', 'automation'],
          type: 'workflow',
          content: 'Automatically process uploaded files with various transformations'
        },
        {
          id: '5',
          title: 'Conditional Logic Node',
          description: 'Add conditional branching to your workflows',
          tags: ['condition', 'logic', 'branching', 'if-else'],
          type: 'node',
          content: 'Create complex conditional logic with multiple branches'
        },
        {
          id: '6',
          title: 'Getting Started Guide',
          description: 'Learn how to create your first automation workflow',
          tags: ['tutorial', 'guide', 'getting-started', 'beginner'],
          type: 'documentation',
          content: 'Step-by-step guide for beginners to create automation workflows'
        }
      ]);
    }
  }, [isReady, addItems]);

  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      setShowResults(false);
      return;
    }

    setIsSearching(true);
    recordInteraction();

    try {
      const searchResults = await search(searchQuery, {
        limit: 10,
        useSemanticSearch: true
      });

      setResults(searchResults.items.map((item, index) => ({
        ...item,
        score: searchResults.scores[index] || 0
      })));
      setShowResults(true);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    
    // Get suggestions
    if (value.length > 1) {
      const newSuggestions = getSuggestions(value, 5);
      setSuggestions(newSuggestions);
    } else {
      setSuggestions([]);
    }

    // Debounced search
    const timeoutId = setTimeout(() => {
      performSearch(value);
    }, 300);

    return () => clearTimeout(timeoutId);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < results.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && results[selectedIndex]) {
        handleResultClick(results[selectedIndex]);
      }
    } else if (e.key === 'Escape') {
      setShowResults(false);
      setSelectedIndex(-1);
    }
  };

  const handleResultClick = (result: SearchResult) => {
    recordInteraction();
    setQuery(result.title);
    setShowResults(false);
    setSelectedIndex(-1);
    
    // Handle different result types
    switch (result.type) {
      case 'workflow':
        console.log('Opening workflow:', result.id);
        break;
      case 'node':
        console.log('Adding node:', result.id);
        break;
      case 'template':
        console.log('Loading template:', result.id);
        break;
      case 'documentation':
        console.log('Opening documentation:', result.id);
        break;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'workflow': return '🔧';
      case 'node': return '⚙️';
      case 'template': return '📋';
      case 'documentation': return '📚';
      default: return '📄';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'workflow': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'node': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'template': return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      case 'documentation': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  return (
    <div className="relative">
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2 text-slate-800 dark:text-slate-200">
          🔍 Intelligent Search
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
          Search workflows, nodes, templates, and documentation with AI-powered semantic understanding
        </p>
      </div>

      {/* Search Input */}
      <div className="relative">
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => query && setShowResults(true)}
            placeholder="Search for workflows, nodes, templates..."
            className="w-full px-4 py-3 pl-10 pr-10 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
          
          {/* Search Icon */}
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
            {isSearching ? (
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            )}
          </div>

          {/* Clear Button */}
          {query && (
            <button
              onClick={() => {
                setQuery('');
                setResults([]);
                setShowResults(false);
                setSuggestions([]);
              }}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {/* Suggestions */}
        {suggestions.length > 0 && !showResults && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg shadow-lg z-50">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => {
                  setQuery(suggestion);
                  performSearch(suggestion);
                }}
                className="w-full px-4 py-2 text-left hover:bg-slate-50 dark:hover:bg-slate-600 first:rounded-t-lg last:rounded-b-lg"
              >
                <span className="text-slate-600 dark:text-slate-300">{suggestion}</span>
              </button>
            ))}
          </div>
        )}

        {/* Search Results */}
        <AnimatePresence>
          {showResults && results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg shadow-xl z-50 max-h-96 overflow-y-auto"
            >
              {results.map((result, index) => (
                <motion.button
                  key={result.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => handleResultClick(result)}
                  className={`w-full px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-600 border-b border-slate-100 dark:border-slate-600 last:border-b-0 first:rounded-t-lg last:rounded-b-lg transition-colors ${
                    selectedIndex === index ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                  }`}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-lg mt-0.5">{getTypeIcon(result.type)}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <h4 className="font-medium text-slate-900 dark:text-slate-100 truncate">
                          {result.title}
                        </h4>
                        <span className={`px-2 py-0.5 text-xs rounded-full ${getTypeColor(result.type)}`}>
                          {result.type}
                        </span>
                        {result.score > 0 && (
                          <span className="text-xs text-slate-500 dark:text-slate-400">
                            {Math.round(result.score * 100)}%
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 line-clamp-2">
                        {result.description}
                      </p>
                      {result.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {result.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 text-xs bg-slate-100 dark:bg-slate-600 text-slate-600 dark:text-slate-300 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                          {result.tags.length > 3 && (
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              +{result.tags.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </motion.button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* No Results */}
        {showResults && results.length === 0 && query && !isSearching && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg shadow-lg p-4 z-50"
          >
            <div className="text-center text-slate-500 dark:text-slate-400">
              <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.47-.881-6.08-2.33" />
              </svg>
              <p className="text-sm">No results found for "{query}"</p>
              <p className="text-xs mt-1">Try different keywords or check spelling</p>
            </div>
          </motion.div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="mt-4">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Quick Actions</h4>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              setQuery('create workflow');
              performSearch('create workflow');
            }}
            className="px-3 py-2 text-sm bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
          >
            🔧 Create Workflow
          </button>
          <button
            onClick={() => {
              setQuery('email template');
              performSearch('email template');
            }}
            className="px-3 py-2 text-sm bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
          >
            📧 Email Templates
          </button>
          <button
            onClick={() => {
              setQuery('database query');
              performSearch('database query');
            }}
            className="px-3 py-2 text-sm bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
          >
            🗄️ Database
          </button>
          <button
            onClick={() => {
              setQuery('getting started');
              performSearch('getting started');
            }}
            className="px-3 py-2 text-sm bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300 rounded-lg hover:bg-yellow-100 dark:hover:bg-yellow-900/30 transition-colors"
          >
            📚 Help
          </button>
        </div>
      </div>
    </div>
  );
}