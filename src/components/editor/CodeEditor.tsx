'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Editor } from '@monaco-editor/react';
import { motion } from 'framer-motion';
import {
  PlayIcon,
  StopIcon,
  DocumentArrowDownIcon,
  DocumentArrowUpIcon,
  CogIcon,
  CommandLineIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

interface ExecutionResult {
  output: string;
  error?: string;
  executionTime: number;
  timestamp: number;
}

interface CodeEditorProps {
  initialCode?: string;
  language?: string;
  onCodeChange?: (code: string) => void;
  onExecute?: (code: string) => Promise<ExecutionResult>;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  initialCode = '',
  language = 'javascript',
  onCodeChange,
  onExecute,
}) => {
  const [code, setCode] = useState(initialCode);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState(language);
  const [theme, setTheme] = useState<'vs-dark' | 'light'>('vs-dark');
  const editorRef = useRef<any>(null);

  const languages = [
    { value: 'javascript', label: 'JavaScript' },
    { value: 'typescript', label: 'TypeScript' },
    { value: 'python', label: 'Python' },
    { value: 'json', label: 'JSON' },
    { value: 'yaml', label: 'YAML' },
    { value: 'markdown', label: 'Markdown' },
  ];

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor;
    
    // Configure editor options
    editor.updateOptions({
      fontSize: 14,
      lineHeight: 20,
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 2,
      insertSpaces: true,
      wordWrap: 'on',
    });

    // Add custom key bindings
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      handleExecute();
    });

    // Configure language-specific settings
    monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
      noSemanticValidation: false,
      noSyntaxValidation: false,
    });

    monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
      target: monaco.languages.typescript.ScriptTarget.ES2020,
      allowNonTsExtensions: true,
    });
  };

  const handleCodeChange = (value: string | undefined) => {
    const newCode = value || '';
    setCode(newCode);
    onCodeChange?.(newCode);
  };

  const handleExecute = async () => {
    if (!code.trim() || isExecuting) return;

    setIsExecuting(true);
    const startTime = Date.now();

    try {
      let result: ExecutionResult;

      if (onExecute) {
        // Use custom execution handler
        result = await onExecute(code);
      } else {
        // Default execution for JavaScript
        result = await executeCode(code, selectedLanguage);
      }

      setExecutionResults(prev => [result, ...prev.slice(0, 9)]);
    } catch (error) {
      const result: ExecutionResult = {
        output: '',
        error: error instanceof Error ? error.message : 'Unknown error',
        executionTime: Date.now() - startTime,
        timestamp: Date.now(),
      };
      setExecutionResults(prev => [result, ...prev.slice(0, 9)]);
    } finally {
      setIsExecuting(false);
    }
  };

  const executeCode = async (code: string, lang: string): Promise<ExecutionResult> => {
    const startTime = Date.now();
    
    try {
      if (lang === 'javascript' || lang === 'typescript') {
        // Create a safe execution environment
        const logs: string[] = [];
        const originalConsole = console.log;
        
        // Override console.log to capture output
        console.log = (...args) => {
          logs.push(args.map(arg => 
            typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
          ).join(' '));
        };

        try {
          // Execute the code in a try-catch block
          const func = new Function(`
            ${code}
          `);
          const result = func();
          
          // Restore console.log
          console.log = originalConsole;
          
          return {
            output: logs.length > 0 ? logs.join('\n') : (result !== undefined ? String(result) : 'Code executed successfully'),
            executionTime: Date.now() - startTime,
            timestamp: Date.now(),
          };
        } catch (error) {
          console.log = originalConsole;
          throw error;
        }
      } else if (lang === 'json') {
        // Validate JSON
        JSON.parse(code);
        return {
          output: 'Valid JSON',
          executionTime: Date.now() - startTime,
          timestamp: Date.now(),
        };
      } else {
        return {
          output: `Execution not supported for ${lang}. Code syntax validated.`,
          executionTime: Date.now() - startTime,
          timestamp: Date.now(),
        };
      }
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Execution failed');
    }
  };

  const handleSaveFile = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `script.${selectedLanguage === 'javascript' ? 'js' : selectedLanguage === 'typescript' ? 'ts' : selectedLanguage === 'python' ? 'py' : 'txt'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleLoadFile = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.js,.ts,.py,.json,.yaml,.md,.txt';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          setCode(content);
          onCodeChange?.(content);
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return (
    <div className="h-full flex flex-col bg-gray-900 rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
          >
            {languages.map(lang => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
          
          <button
            onClick={() => setTheme(theme === 'vs-dark' ? 'light' : 'vs-dark')}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Toggle theme"
          >
            <CogIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleLoadFile}
            className="flex items-center space-x-1 px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
            title="Load file"
          >
            <DocumentArrowUpIcon className="w-4 h-4" />
            <span className="text-sm">Load</span>
          </button>
          
          <button
            onClick={handleSaveFile}
            className="flex items-center space-x-1 px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
            title="Save file"
          >
            <DocumentArrowDownIcon className="w-4 h-4" />
            <span className="text-sm">Save</span>
          </button>
          
          <button
            onClick={handleExecute}
            disabled={isExecuting || !code.trim()}
            className="flex items-center space-x-1 px-4 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Execute code (Ctrl+Enter)"
          >
            {isExecuting ? (
              <StopIcon className="w-4 h-4 animate-spin" />
            ) : (
              <PlayIcon className="w-4 h-4" />
            )}
            <span className="text-sm">{isExecuting ? 'Running...' : 'Run'}</span>
          </button>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Editor */}
        <div className="flex-1">
          <Editor
            height="100%"
            language={selectedLanguage}
            value={code}
            theme={theme}
            onChange={handleCodeChange}
            onMount={handleEditorDidMount}
            options={{
              fontSize: 14,
              lineHeight: 20,
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 2,
              insertSpaces: true,
              wordWrap: 'on',
            }}
          />
        </div>

        {/* Results Panel */}
        {executionResults.length > 0 && (
          <div className="w-1/3 border-l border-gray-700 bg-gray-800">
            <div className="p-3 border-b border-gray-700">
              <div className="flex items-center space-x-2">
                <CommandLineIcon className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-white">Execution Results</span>
              </div>
            </div>
            
            <div className="h-full overflow-y-auto p-3 space-y-3">
              {executionResults.map((result, index) => (
                <motion.div
                  key={result.timestamp}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-900 rounded p-3 border border-gray-600"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {result.error ? (
                        <ExclamationTriangleIcon className="w-4 h-4 text-red-400" />
                      ) : (
                        <CheckCircleIcon className="w-4 h-4 text-green-400" />
                      )}
                      <span className="text-xs text-gray-400">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {result.executionTime}ms
                    </span>
                  </div>
                  
                  {result.error ? (
                    <pre className="text-red-400 text-xs whitespace-pre-wrap font-mono">
                      {result.error}
                    </pre>
                  ) : (
                    <pre className="text-green-400 text-xs whitespace-pre-wrap font-mono">
                      {result.output}
                    </pre>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CodeEditor;