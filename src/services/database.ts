// Database service for storing workflows, user data, and execution logs
// Using IndexedDB for browser-based storage with fallback to localStorage

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'admin' | 'user' | 'viewer';
  createdAt: number;
  lastLoginAt: number;
  preferences: {
    theme: 'dark' | 'light';
    notifications: boolean;
    autoSave: boolean;
  };
}

export interface WorkflowRecord {
  id: string;
  name: string;
  description: string;
  nodes: any[];
  edges: any[];
  userId: string;
  isPublic: boolean;
  tags: string[];
  version: number;
  createdAt: number;
  updatedAt: number;
  lastExecutedAt?: number;
  executionCount: number;
  status: 'draft' | 'published' | 'archived';
}

export interface ExecutionLog {
  id: string;
  workflowId: string;
  userId: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  startTime: number;
  endTime?: number;
  duration?: number;
  logs: Array<{
    timestamp: number;
    level: 'info' | 'warn' | 'error' | 'debug';
    message: string;
    nodeId?: string;
    data?: any;
  }>;
  result?: any;
  error?: string;
  metrics: {
    nodesExecuted: number;
    totalNodes: number;
    memoryUsed: number;
    cpuTime: number;
  };
}

export interface ScriptRecord {
  id: string;
  name: string;
  description: string;
  language: string;
  code: string;
  userId: string;
  isPublic: boolean;
  tags: string[];
  version: number;
  createdAt: number;
  updatedAt: number;
  lastExecutedAt?: number;
  executionCount: number;
}

export interface CollaborationSession {
  id: string;
  workflowId: string;
  hostUserId: string;
  participants: Array<{
    userId: string;
    joinedAt: number;
    role: 'host' | 'editor' | 'viewer';
    cursor?: { x: number; y: number };
  }>;
  isActive: boolean;
  createdAt: number;
  endedAt?: number;
}

export interface DatabaseStats {
  totalWorkflows: number;
  totalUsers: number;
  totalExecutions: number;
  totalScripts: number;
  activeSessions: number;
  storageUsed: number; // in bytes
  lastBackup?: number;
}

class DatabaseService {
  private db: IDBDatabase | null = null;
  private dbName = 'ComputerGenieDB';
  private dbVersion = 1;
  private isInitialized = false;

  constructor() {
    this.initializeDatabase();
  }

  private async initializeDatabase(): Promise<void> {
    if (typeof window === 'undefined' || !window.indexedDB) {
      console.warn('IndexedDB not available, falling back to localStorage');
      this.isInitialized = true;
      return;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => {
        console.error('Failed to open database:', request.error);
        this.isInitialized = true; // Fallback to localStorage
        resolve();
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.isInitialized = true;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create object stores
        if (!db.objectStoreNames.contains('users')) {
          const userStore = db.createObjectStore('users', { keyPath: 'id' });
          userStore.createIndex('email', 'email', { unique: true });
        }

        if (!db.objectStoreNames.contains('workflows')) {
          const workflowStore = db.createObjectStore('workflows', { keyPath: 'id' });
          workflowStore.createIndex('userId', 'userId');
          workflowStore.createIndex('isPublic', 'isPublic');
          workflowStore.createIndex('status', 'status');
          workflowStore.createIndex('tags', 'tags', { multiEntry: true });
        }

        if (!db.objectStoreNames.contains('executionLogs')) {
          const logStore = db.createObjectStore('executionLogs', { keyPath: 'id' });
          logStore.createIndex('workflowId', 'workflowId');
          logStore.createIndex('userId', 'userId');
          logStore.createIndex('status', 'status');
          logStore.createIndex('startTime', 'startTime');
        }

        if (!db.objectStoreNames.contains('scripts')) {
          const scriptStore = db.createObjectStore('scripts', { keyPath: 'id' });
          scriptStore.createIndex('userId', 'userId');
          scriptStore.createIndex('language', 'language');
          scriptStore.createIndex('tags', 'tags', { multiEntry: true });
        }

        if (!db.objectStoreNames.contains('collaborationSessions')) {
          const sessionStore = db.createObjectStore('collaborationSessions', { keyPath: 'id' });
          sessionStore.createIndex('workflowId', 'workflowId');
          sessionStore.createIndex('hostUserId', 'hostUserId');
          sessionStore.createIndex('isActive', 'isActive');
        }
      };
    });
  }

  private async waitForInitialization(): Promise<void> {
    while (!this.isInitialized) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  private async performTransaction<T>(
    storeName: string,
    mode: IDBTransactionMode,
    operation: (store: IDBObjectStore) => IDBRequest<T>
  ): Promise<T> {
    await this.waitForInitialization();

    if (!this.db) {
      // Fallback to localStorage
      return this.performLocalStorageOperation(storeName, operation);
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], mode);
      const store = transaction.objectStore(storeName);
      const request = operation(store);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private async performLocalStorageOperation<T>(
    storeName: string,
    operation: (store: any) => any
  ): Promise<T> {
    const key = `${this.dbName}_${storeName}`;
    const data = JSON.parse(localStorage.getItem(key) || '[]');
    
    // Mock IDBObjectStore interface for localStorage
    const mockStore = {
      add: (value: any) => {
        data.push(value);
        localStorage.setItem(key, JSON.stringify(data));
        return { result: value };
      },
      put: (value: any) => {
        const index = data.findIndex((item: any) => item.id === value.id);
        if (index >= 0) {
          data[index] = value;
        } else {
          data.push(value);
        }
        localStorage.setItem(key, JSON.stringify(data));
        return { result: value };
      },
      get: (id: string) => {
        const item = data.find((item: any) => item.id === id);
        return { result: item };
      },
      delete: (id: string) => {
        const index = data.findIndex((item: any) => item.id === id);
        if (index >= 0) {
          data.splice(index, 1);
          localStorage.setItem(key, JSON.stringify(data));
        }
        return { result: undefined };
      },
      getAll: () => {
        return { result: data };
      }
    };

    const request = operation(mockStore);
    return request.result;
  }

  // User management
  async createUser(user: Omit<User, 'id' | 'createdAt' | 'lastLoginAt'>): Promise<User> {
    const newUser: User = {
      ...user,
      id: this.generateId(),
      createdAt: Date.now(),
      lastLoginAt: Date.now(),
    };

    await this.performTransaction('users', 'readwrite', (store) => store.add(newUser));
    return newUser;
  }

  async getUserById(id: string): Promise<User | null> {
    const result = await this.performTransaction('users', 'readonly', (store) => store.get(id));
    return result || null;
  }

  async getUserByEmail(email: string): Promise<User | null> {
    if (!this.db) {
      const users = await this.getAllUsers();
      return users.find(user => user.email === email) || null;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['users'], 'readonly');
      const store = transaction.objectStore('users');
      const index = store.index('email');
      const request = index.get(email);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }

  async updateUser(id: string, updates: Partial<User>): Promise<User | null> {
    const user = await this.getUserById(id);
    if (!user) return null;

    const updatedUser = { ...user, ...updates };
    await this.performTransaction('users', 'readwrite', (store) => store.put(updatedUser));
    return updatedUser;
  }

  async getAllUsers(): Promise<User[]> {
    return this.performTransaction('users', 'readonly', (store) => store.getAll());
  }

  // Workflow management
  async saveWorkflow(workflow: Omit<WorkflowRecord, 'id' | 'createdAt' | 'updatedAt' | 'version' | 'executionCount'>): Promise<WorkflowRecord> {
    const newWorkflow: WorkflowRecord = {
      ...workflow,
      id: this.generateId(),
      version: 1,
      executionCount: 0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    await this.performTransaction('workflows', 'readwrite', (store) => store.add(newWorkflow));
    return newWorkflow;
  }

  async updateWorkflow(id: string, updates: Partial<WorkflowRecord>): Promise<WorkflowRecord | null> {
    const workflow = await this.getWorkflowById(id);
    if (!workflow) return null;

    const updatedWorkflow = {
      ...workflow,
      ...updates,
      version: workflow.version + 1,
      updatedAt: Date.now(),
    };

    await this.performTransaction('workflows', 'readwrite', (store) => store.put(updatedWorkflow));
    return updatedWorkflow;
  }

  async getWorkflowById(id: string): Promise<WorkflowRecord | null> {
    const result = await this.performTransaction('workflows', 'readonly', (store) => store.get(id));
    return result || null;
  }

  async getWorkflowsByUserId(userId: string): Promise<WorkflowRecord[]> {
    if (!this.db) {
      const workflows = await this.getAllWorkflows();
      return workflows.filter(workflow => workflow.userId === userId);
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['workflows'], 'readonly');
      const store = transaction.objectStore('workflows');
      const index = store.index('userId');
      const request = index.getAll(userId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllWorkflows(): Promise<WorkflowRecord[]> {
    return this.performTransaction('workflows', 'readonly', (store) => store.getAll());
  }

  async deleteWorkflow(id: string): Promise<boolean> {
    try {
      await this.performTransaction('workflows', 'readwrite', (store) => store.delete(id));
      return true;
    } catch (error) {
      console.error('Failed to delete workflow:', error);
      return false;
    }
  }

  // Execution log management
  async createExecutionLog(log: Omit<ExecutionLog, 'id'>): Promise<ExecutionLog> {
    const newLog: ExecutionLog = {
      ...log,
      id: this.generateId(),
    };

    await this.performTransaction('executionLogs', 'readwrite', (store) => store.add(newLog));
    return newLog;
  }

  async updateExecutionLog(id: string, updates: Partial<ExecutionLog>): Promise<ExecutionLog | null> {
    const log = await this.getExecutionLogById(id);
    if (!log) return null;

    const updatedLog = { ...log, ...updates };
    await this.performTransaction('executionLogs', 'readwrite', (store) => store.put(updatedLog));
    return updatedLog;
  }

  async getExecutionLogById(id: string): Promise<ExecutionLog | null> {
    const result = await this.performTransaction('executionLogs', 'readonly', (store) => store.get(id));
    return result || null;
  }

  async getExecutionLogsByWorkflowId(workflowId: string): Promise<ExecutionLog[]> {
    if (!this.db) {
      const logs = await this.getAllExecutionLogs();
      return logs.filter(log => log.workflowId === workflowId);
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['executionLogs'], 'readonly');
      const store = transaction.objectStore('executionLogs');
      const index = store.index('workflowId');
      const request = index.getAll(workflowId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllExecutionLogs(): Promise<ExecutionLog[]> {
    return this.performTransaction('executionLogs', 'readonly', (store) => store.getAll());
  }

  // Script management
  async saveScript(script: Omit<ScriptRecord, 'id' | 'createdAt' | 'updatedAt' | 'version' | 'executionCount'>): Promise<ScriptRecord> {
    const newScript: ScriptRecord = {
      ...script,
      id: this.generateId(),
      version: 1,
      executionCount: 0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    await this.performTransaction('scripts', 'readwrite', (store) => store.add(newScript));
    return newScript;
  }

  async updateScript(id: string, updates: Partial<ScriptRecord>): Promise<ScriptRecord | null> {
    const script = await this.getScriptById(id);
    if (!script) return null;

    const updatedScript = {
      ...script,
      ...updates,
      version: script.version + 1,
      updatedAt: Date.now(),
    };

    await this.performTransaction('scripts', 'readwrite', (store) => store.put(updatedScript));
    return updatedScript;
  }

  async getScriptById(id: string): Promise<ScriptRecord | null> {
    const result = await this.performTransaction('scripts', 'readonly', (store) => store.get(id));
    return result || null;
  }

  async getScriptsByUserId(userId: string): Promise<ScriptRecord[]> {
    if (!this.db) {
      const scripts = await this.getAllScripts();
      return scripts.filter(script => script.userId === userId);
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['scripts'], 'readonly');
      const store = transaction.objectStore('scripts');
      const index = store.index('userId');
      const request = index.getAll(userId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllScripts(): Promise<ScriptRecord[]> {
    return this.performTransaction('scripts', 'readonly', (store) => store.getAll());
  }

  async deleteScript(id: string): Promise<boolean> {
    try {
      await this.performTransaction('scripts', 'readwrite', (store) => store.delete(id));
      return true;
    } catch (error) {
      console.error('Failed to delete script:', error);
      return false;
    }
  }

  // Collaboration session management
  async createCollaborationSession(session: Omit<CollaborationSession, 'id' | 'createdAt'>): Promise<CollaborationSession> {
    const newSession: CollaborationSession = {
      ...session,
      id: this.generateId(),
      createdAt: Date.now(),
    };

    await this.performTransaction('collaborationSessions', 'readwrite', (store) => store.add(newSession));
    return newSession;
  }

  async updateCollaborationSession(id: string, updates: Partial<CollaborationSession>): Promise<CollaborationSession | null> {
    const session = await this.getCollaborationSessionById(id);
    if (!session) return null;

    const updatedSession = { ...session, ...updates };
    await this.performTransaction('collaborationSessions', 'readwrite', (store) => store.put(updatedSession));
    return updatedSession;
  }

  async getCollaborationSessionById(id: string): Promise<CollaborationSession | null> {
    const result = await this.performTransaction('collaborationSessions', 'readonly', (store) => store.get(id));
    return result || null;
  }

  async getActiveCollaborationSessions(): Promise<CollaborationSession[]> {
    if (!this.db) {
      const sessions = await this.getAllCollaborationSessions();
      return sessions.filter(session => session.isActive);
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['collaborationSessions'], 'readonly');
      const store = transaction.objectStore('collaborationSessions');
      const index = store.index('isActive');
      const request = index.getAll(true);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllCollaborationSessions(): Promise<CollaborationSession[]> {
    return this.performTransaction('collaborationSessions', 'readonly', (store) => store.getAll());
  }

  // Statistics and analytics
  async getDatabaseStats(): Promise<DatabaseStats> {
    const [workflows, users, executions, scripts, sessions] = await Promise.all([
      this.getAllWorkflows(),
      this.getAllUsers(),
      this.getAllExecutionLogs(),
      this.getAllScripts(),
      this.getActiveCollaborationSessions(),
    ]);

    const storageUsed = this.calculateStorageUsage();

    return {
      totalWorkflows: workflows.length,
      totalUsers: users.length,
      totalExecutions: executions.length,
      totalScripts: scripts.length,
      activeSessions: sessions.length,
      storageUsed,
      lastBackup: this.getLastBackupTime(),
    };
  }

  // Search functionality
  async searchWorkflows(query: string, userId?: string): Promise<WorkflowRecord[]> {
    const workflows = userId 
      ? await this.getWorkflowsByUserId(userId)
      : await this.getAllWorkflows();

    const searchTerm = query.toLowerCase();
    return workflows.filter(workflow => 
      workflow.name.toLowerCase().includes(searchTerm) ||
      workflow.description.toLowerCase().includes(searchTerm) ||
      workflow.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );
  }

  async searchScripts(query: string, userId?: string): Promise<ScriptRecord[]> {
    const scripts = userId 
      ? await this.getScriptsByUserId(userId)
      : await this.getAllScripts();

    const searchTerm = query.toLowerCase();
    return scripts.filter(script => 
      script.name.toLowerCase().includes(searchTerm) ||
      script.description.toLowerCase().includes(searchTerm) ||
      script.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );
  }

  // Backup and restore
  async exportData(): Promise<string> {
    const [workflows, users, executions, scripts, sessions] = await Promise.all([
      this.getAllWorkflows(),
      this.getAllUsers(),
      this.getAllExecutionLogs(),
      this.getAllScripts(),
      this.getAllCollaborationSessions(),
    ]);

    const exportData = {
      version: this.dbVersion,
      timestamp: Date.now(),
      data: {
        workflows,
        users,
        executions,
        scripts,
        sessions,
      },
    };

    return JSON.stringify(exportData, null, 2);
  }

  async importData(jsonData: string): Promise<boolean> {
    try {
      const importData = JSON.parse(jsonData);
      
      if (!importData.data) {
        throw new Error('Invalid import data format');
      }

      const { workflows, users, executions, scripts, sessions } = importData.data;

      // Clear existing data (optional - could be made configurable)
      await this.clearAllData();

      // Import data
      if (users) {
        for (const user of users) {
          await this.performTransaction('users', 'readwrite', (store) => store.add(user));
        }
      }

      if (workflows) {
        for (const workflow of workflows) {
          await this.performTransaction('workflows', 'readwrite', (store) => store.add(workflow));
        }
      }

      if (scripts) {
        for (const script of scripts) {
          await this.performTransaction('scripts', 'readwrite', (store) => store.add(script));
        }
      }

      if (executions) {
        for (const execution of executions) {
          await this.performTransaction('executionLogs', 'readwrite', (store) => store.add(execution));
        }
      }

      if (sessions) {
        for (const session of sessions) {
          await this.performTransaction('collaborationSessions', 'readwrite', (store) => store.add(session));
        }
      }

      this.setLastBackupTime(Date.now());
      return true;
    } catch (error) {
      console.error('Failed to import data:', error);
      return false;
    }
  }

  async clearAllData(): Promise<void> {
    const stores = ['users', 'workflows', 'scripts', 'executionLogs', 'collaborationSessions'];
    
    for (const storeName of stores) {
      if (this.db) {
        await new Promise<void>((resolve, reject) => {
          const transaction = this.db!.transaction([storeName], 'readwrite');
          const store = transaction.objectStore(storeName);
          const request = store.clear();

          request.onsuccess = () => resolve();
          request.onerror = () => reject(request.error);
        });
      } else {
        localStorage.removeItem(`${this.dbName}_${storeName}`);
      }
    }
  }

  // Utility methods
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private calculateStorageUsage(): number {
    if (typeof window === 'undefined') return 0;
    
    let totalSize = 0;
    for (let key in localStorage) {
      if (key.startsWith(this.dbName)) {
        totalSize += localStorage[key].length;
      }
    }
    return totalSize;
  }

  private getLastBackupTime(): number | undefined {
    const backup = localStorage.getItem(`${this.dbName}_lastBackup`);
    return backup ? parseInt(backup) : undefined;
  }

  private setLastBackupTime(timestamp: number): void {
    localStorage.setItem(`${this.dbName}_lastBackup`, timestamp.toString());
  }
}

// Export singleton instance
export const databaseService = new DatabaseService();