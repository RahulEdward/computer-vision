'use client';

import React, { useEffect, useRef, useState } from 'react';
import Editor from '@monaco-editor/react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/lib/store';
import { useCollaboration } from '@/lib/webrtc-collaboration';
import * as Y from 'yjs';

interface MonacoEditorProps {
  roomId?: string;
}

export default function MonacoEditor({ roomId = 'editor-room' }: MonacoEditorProps) {
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);
  const bindingRef = useRef<any | null>(null);
  const [language, setLanguage] = useState('javascript');
  const [theme, setTheme] = useState('vs-dark');
  const [code, setCode] = useState(`// Computer Genie Automation Script
// This is a collaborative code editor with real-time sync

class AutomationWorkflow {
  constructor(name) {
    this.name = name;
    this.steps = [];
    this.variables = new Map();
  }

  addStep(step) {
    this.steps.push({
      id: Date.now(),
      ...step,
      timestamp: new Date().toISOString()
    });
    return this;
  }

  setVariable(key, value) {
    this.variables.set(key, value);
    return this;
  }

  async execute() {
    console.log(\`Executing workflow: \${this.name}\`);
    
    for (const step of this.steps) {
      try {
        console.log(\`Executing step: \${step.name}\`);
        
        switch (step.type) {
          case 'http':
            await this.executeHttpRequest(step);
            break;
          case 'email':
            await this.sendEmail(step);
            break;
          case 'database':
            await this.databaseOperation(step);
            break;
          case 'condition':
            if (!this.evaluateCondition(step.condition)) {
              console.log('Condition not met, skipping subsequent steps');
              break;
            }
            break;
          default:
            console.warn(\`Unknown step type: \${step.type}\`);
        }
        
        step.status = 'completed';
      } catch (error) {
        step.status = 'failed';
        step.error = error.message;
        console.error(\`Step failed: \${step.name}\`, error);
        
        if (step.continueOnError !== true) {
          throw error;
        }
      }
    }
    
    return {
      status: 'completed',
      steps: this.steps,
      variables: Object.fromEntries(this.variables)
    };
  }

  async executeHttpRequest(step) {
    const response = await fetch(step.url, {
      method: step.method || 'GET',
      headers: step.headers || {},
      body: step.body ? JSON.stringify(step.body) : undefined
    });
    
    const data = await response.json();
    this.setVariable(\`\${step.name}_response\`, data);
    return data;
  }

  async sendEmail(step) {
    // Email sending logic
    console.log(\`Sending email to: \${step.to}\`);
    console.log(\`Subject: \${step.subject}\`);
    console.log(\`Body: \${step.body}\`);
    
    // Simulate email sending
    await new Promise(resolve => setTimeout(resolve, 1000));
    return { sent: true, messageId: Date.now() };
  }

  async databaseOperation(step) {
    // Database operation logic
    console.log(\`Database operation: \${step.operation}\`);
    
    switch (step.operation) {
      case 'select':
        return { rows: [], count: 0 };
      case 'insert':
        return { insertedId: Date.now() };
      case 'update':
        return { modifiedCount: 1 };
      case 'delete':
        return { deletedCount: 1 };
    }
  }

  evaluateCondition(condition) {
    // Simple condition evaluation
    // In production, use a proper expression evaluator
    try {
      return new Function('variables', \`return \${condition}\`)(
        Object.fromEntries(this.variables)
      );
    } catch (error) {
      console.error('Condition evaluation failed:', error);
      return false;
    }
  }
}

// Example usage
const workflow = new AutomationWorkflow('User Registration Flow')
  .addStep({
    name: 'Validate User Data',
    type: 'condition',
    condition: 'variables.get("email") && variables.get("password")'
  })
  .addStep({
    name: 'Create User Account',
    type: 'database',
    operation: 'insert',
    table: 'users',
    data: {
      email: '{{email}}',
      password: '{{hashedPassword}}',
      createdAt: new Date()
    }
  })
  .addStep({
    name: 'Send Welcome Email',
    type: 'email',
    to: '{{email}}',
    subject: 'Welcome to Computer Genie!',
    body: 'Thank you for joining us. Your automation journey begins now!'
  })
  .addStep({
    name: 'Log Registration',
    type: 'http',
    url: 'https://api.analytics.com/events',
    method: 'POST',
    body: {
      event: 'user_registered',
      userId: '{{userId}}',
      timestamp: new Date().toISOString()
    }
  });

// Execute the workflow
workflow
  .setVariable('email', 'user@example.com')
  .setVariable('password', 'securePassword123')
  .setVariable('hashedPassword', 'hashed_password_here')
  .execute()
  .then(result => {
    console.log('Workflow completed:', result);
  })
  .catch(error => {
    console.error('Workflow failed:', error);
  });
`);

  const collaboration = useCollaboration(roomId);
  const { theme: appTheme, recordInteraction } = useDashboardStore();

  useEffect(() => {
    setTheme(appTheme === 'dark' ? 'vs-dark' : 'vs-light');
  }, [appTheme]);

  // Cleanup collaboration binding
  useEffect(() => {
    return () => {
      if (bindingRef.current?.destroy) {
        bindingRef.current.destroy();
      }
    };
  }, []);

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    
    // Setup collaborative editing with simple text sync
    if (collaboration) {
      // Listen for remote changes
      const yText = collaboration.getSharedText('monaco');
      
      const updateFromRemote = () => {
        const remoteContent = yText.toString();
        const currentContent = editor.getValue();
        
        if (remoteContent !== currentContent) {
          const position = editor.getPosition();
          editor.setValue(remoteContent);
          if (position) {
            editor.setPosition(position);
          }
        }
      };

      yText.observe(updateFromRemote);
      
      // Send local changes
      const onContentChange = editor.onDidChangeModelContent(() => {
        const content = editor.getValue();
        collaboration.syncEditorContent(content, 'monaco');
      });

      // Store cleanup function
      bindingRef.current = {
        destroy: () => {
          yText.unobserve(updateFromRemote);
          onContentChange.dispose();
        }
      };
    }

    // Configure editor
    editor.updateOptions({
      fontSize: 14,
      lineHeight: 20,
      fontFamily: 'JetBrains Mono, Consolas, Monaco, monospace',
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      suggestOnTriggerCharacters: true,
      quickSuggestions: true,
      wordWrap: 'on',
      lineNumbers: 'on',
      renderWhitespace: 'selection',
      bracketPairColorization: { enabled: true }
    });

    // Add custom commands
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      recordInteraction();
      saveCode();
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyR, () => {
      recordInteraction();
      runCode();
    });

    // Add custom themes
    monaco.editor.defineTheme('computer-genie-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
        { token: 'keyword', foreground: '569CD6' },
        { token: 'string', foreground: 'CE9178' },
        { token: 'number', foreground: 'B5CEA8' },
        { token: 'function', foreground: 'DCDCAA' },
      ],
      colors: {
        'editor.background': '#0F172A',
        'editor.foreground': '#E2E8F0',
        'editorLineNumber.foreground': '#64748B',
        'editorCursor.foreground': '#3B82F6',
        'editor.selectionBackground': '#1E40AF33',
      }
    });

    monaco.editor.setTheme('computer-genie-dark');
  };

  const saveCode = () => {
    if (editorRef.current) {
      const currentCode = editorRef.current.getValue();
      localStorage.setItem('editor-code', currentCode);
      // Show success notification
      console.log('Code saved successfully');
    }
  };

  const runCode = () => {
    if (editorRef.current) {
      const currentCode = editorRef.current.getValue();
      try {
        // In a real implementation, you'd send this to a secure execution environment
        console.log('Executing code...');
        eval(currentCode);
      } catch (error) {
        console.error('Code execution error:', error);
      }
    }
  };

  const formatCode = () => {
    if (editorRef.current) {
      recordInteraction();
      editorRef.current.getAction('editor.action.formatDocument').run();
    }
  };

  const insertTemplate = (template: string) => {
    if (editorRef.current) {
      recordInteraction();
      const position = editorRef.current.getPosition();
      editorRef.current.executeEdits('insert-template', [{
        range: {
          startLineNumber: position.lineNumber,
          startColumn: position.column,
          endLineNumber: position.lineNumber,
          endColumn: position.column
        },
        text: template
      }]);
    }
  };

  const templates = {
    httpRequest: `
// HTTP Request Template
const response = await fetch('https://api.example.com/data', {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_TOKEN'
  }
});
const data = await response.json();
console.log(data);
`,
    emailSender: `
// Email Sender Template
const emailStep = {
  name: 'Send Notification',
  type: 'email',
  to: 'recipient@example.com',
  subject: 'Automation Notification',
  body: 'Your automation workflow has completed successfully!'
};
`,
    databaseQuery: `
// Database Query Template
const dbStep = {
  name: 'Query Database',
  type: 'database',
  operation: 'select',
  table: 'users',
  where: { active: true },
  limit: 100
};
`,
    condition: `
// Condition Template
const conditionStep = {
  name: 'Check Condition',
  type: 'condition',
  condition: 'variables.get("status") === "active"',
  continueOnError: false
};
`
  };

  return (
    <div className="h-full flex flex-col bg-slate-900 rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="bg-slate-800 border-b border-slate-700 p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <select
            value={language}
            onChange={(e) => {
              recordInteraction();
              setLanguage(e.target.value);
            }}
            className="bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600"
          >
            <option value="javascript">JavaScript</option>
            <option value="typescript">TypeScript</option>
            <option value="python">Python</option>
            <option value="json">JSON</option>
            <option value="yaml">YAML</option>
          </select>

          <div className="flex space-x-2">
            <button
              onClick={formatCode}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
            >
              Format
            </button>
            <button
              onClick={saveCode}
              className="px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
            >
              Save (Ctrl+S)
            </button>
            <button
              onClick={runCode}
              className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm"
            >
              Run (Ctrl+R)
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-slate-400 text-sm">Templates:</span>
          {Object.entries(templates).map(([name, template]) => (
            <button
              key={name}
              onClick={() => insertTemplate(template)}
              className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs hover:bg-slate-600 transition-colors"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1">
        <Editor
          height="100%"
          language={language}
          theme={theme}
          value={code}
          onChange={(value) => setCode(value || '')}
          onMount={handleEditorDidMount}
          options={{
            selectOnLineNumbers: true,
            roundedSelection: false,
            readOnly: false,
            cursorStyle: 'line',
            automaticLayout: true,
            glyphMargin: true,
            folding: true,
            lineNumbersMinChars: 3,
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            wrappingIndent: 'indent',
            renderLineHighlight: 'all',
            contextmenu: true,
            mouseWheelZoom: true,
            smoothScrolling: true,
            cursorBlinking: 'blink',
            cursorSmoothCaretAnimation: 'on',
            renderWhitespace: 'selection',
            renderControlCharacters: false,
            fontLigatures: true,
            bracketPairColorization: { enabled: true },
            guides: {
              bracketPairs: true,
              indentation: true
            },
            suggest: {
              showKeywords: true,
              showSnippets: true,
              showFunctions: true,
              showConstructors: true,
              showFields: true,
              showVariables: true,
              showClasses: true,
              showStructs: true,
              showInterfaces: true,
              showModules: true,
              showProperties: true,
              showEvents: true,
              showOperators: true,
              showUnits: true,
              showValues: true,
              showConstants: true,
              showEnums: true,
              showEnumMembers: true,
              showColors: true,
              showFiles: true,
              showReferences: true,
              showFolders: true,
              showTypeParameters: true,
              showUsers: true,
              showIssues: true
            }
          }}
        />
      </div>

      {/* Status Bar */}
      <div className="bg-slate-800 border-t border-slate-700 px-4 py-2 flex items-center justify-between text-sm text-slate-400">
        <div className="flex items-center space-x-4">
          <span>Language: {language}</span>
          <span>Theme: {theme}</span>
          <span>Lines: {code.split('\n').length}</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span>Real-time sync enabled</span>
        </div>
      </div>
    </div>
  );
}