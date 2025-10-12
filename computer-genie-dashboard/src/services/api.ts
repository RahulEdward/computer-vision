// API service for all dashboard functionality
// Provides RESTful endpoints for workflows, users, executions, and collaboration

import { databaseService, User, WorkflowRecord, ExecutionLog, ScriptRecord, CollaborationSession } from './database';

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface SearchParams extends PaginationParams {
  query?: string;
  tags?: string[];
  status?: string;
  userId?: string;
}

class ApiService {
  private baseUrl: string;
  private currentUser: User | null = null;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  // Authentication methods
  async login(email: string, password: string): Promise<ApiResponse<User>> {
    try {
      // In a real implementation, this would validate against a backend
      // For now, we'll simulate authentication with the database
      const user = await databaseService.getUserByEmail(email);
      
      if (!user) {
        return {
          success: false,
          error: 'User not found'
        };
      }

      // Update last login time
      await databaseService.updateUser(user.id, { lastLoginAt: Date.now() });
      this.currentUser = user;

      return {
        success: true,
        data: user,
        message: 'Login successful'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Login failed'
      };
    }
  }

  async register(userData: Omit<User, 'id' | 'createdAt' | 'lastLoginAt'>): Promise<ApiResponse<User>> {
    try {
      // Check if user already exists
      const existingUser = await databaseService.getUserByEmail(userData.email);
      if (existingUser) {
        return {
          success: false,
          error: 'User already exists'
        };
      }

      const user = await databaseService.createUser(userData);
      this.currentUser = user;

      return {
        success: true,
        data: user,
        message: 'Registration successful'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Registration failed'
      };
    }
  }

  async logout(): Promise<ApiResponse> {
    this.currentUser = null;
    return {
      success: true,
      message: 'Logout successful'
    };
  }

  getCurrentUser(): User | null {
    return this.currentUser;
  }

  // User management endpoints
  async getUsers(params: PaginationParams = {}): Promise<ApiResponse<User[]>> {
    try {
      const users = await databaseService.getAllUsers();
      const sortedUsers = this.sortData(users, params.sortBy || 'createdAt', params.sortOrder || 'desc');
      const paginatedUsers = this.paginateData(sortedUsers, params.page || 1, params.limit || 10);

      return {
        success: true,
        data: paginatedUsers
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch users'
      };
    }
  }

  async getUserById(id: string): Promise<ApiResponse<User>> {
    try {
      const user = await databaseService.getUserById(id);
      if (!user) {
        return {
          success: false,
          error: 'User not found'
        };
      }

      return {
        success: true,
        data: user
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch user'
      };
    }
  }

  async updateUser(id: string, updates: Partial<User>): Promise<ApiResponse<User>> {
    try {
      const user = await databaseService.updateUser(id, updates);
      if (!user) {
        return {
          success: false,
          error: 'User not found'
        };
      }

      return {
        success: true,
        data: user,
        message: 'User updated successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update user'
      };
    }
  }

  // Workflow management endpoints
  async getWorkflows(params: SearchParams = {}): Promise<ApiResponse<WorkflowRecord[]>> {
    try {
      let workflows: WorkflowRecord[];

      if (params.query) {
        workflows = await databaseService.searchWorkflows(params.query, params.userId);
      } else if (params.userId) {
        workflows = await databaseService.getWorkflowsByUserId(params.userId);
      } else {
        workflows = await databaseService.getAllWorkflows();
      }

      // Filter by status if provided
      if (params.status) {
        workflows = workflows.filter(w => w.status === params.status);
      }

      // Filter by tags if provided
      if (params.tags && params.tags.length > 0) {
        workflows = workflows.filter(w => 
          params.tags!.some(tag => w.tags.includes(tag))
        );
      }

      const sortedWorkflows = this.sortData(workflows, params.sortBy || 'updatedAt', params.sortOrder || 'desc');
      const paginatedWorkflows = this.paginateData(sortedWorkflows, params.page || 1, params.limit || 10);

      return {
        success: true,
        data: paginatedWorkflows
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch workflows'
      };
    }
  }

  async getWorkflowById(id: string): Promise<ApiResponse<WorkflowRecord>> {
    try {
      const workflow = await databaseService.getWorkflowById(id);
      if (!workflow) {
        return {
          success: false,
          error: 'Workflow not found'
        };
      }

      return {
        success: true,
        data: workflow
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch workflow'
      };
    }
  }

  async createWorkflow(workflowData: Omit<WorkflowRecord, 'id' | 'createdAt' | 'updatedAt' | 'version' | 'executionCount'>): Promise<ApiResponse<WorkflowRecord>> {
    try {
      const workflow = await databaseService.saveWorkflow(workflowData);
      return {
        success: true,
        data: workflow,
        message: 'Workflow created successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create workflow'
      };
    }
  }

  async updateWorkflow(id: string, updates: Partial<WorkflowRecord>): Promise<ApiResponse<WorkflowRecord>> {
    try {
      const workflow = await databaseService.updateWorkflow(id, updates);
      if (!workflow) {
        return {
          success: false,
          error: 'Workflow not found'
        };
      }

      return {
        success: true,
        data: workflow,
        message: 'Workflow updated successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update workflow'
      };
    }
  }

  async deleteWorkflow(id: string): Promise<ApiResponse> {
    try {
      const success = await databaseService.deleteWorkflow(id);
      if (!success) {
        return {
          success: false,
          error: 'Failed to delete workflow'
        };
      }

      return {
        success: true,
        message: 'Workflow deleted successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete workflow'
      };
    }
  }

  // Execution log endpoints
  async getExecutionLogs(params: SearchParams = {}): Promise<ApiResponse<ExecutionLog[]>> {
    try {
      let logs: ExecutionLog[];

      if (params.userId) {
        logs = await databaseService.getAllExecutionLogs();
        logs = logs.filter(log => log.userId === params.userId);
      } else {
        logs = await databaseService.getAllExecutionLogs();
      }

      // Filter by status if provided
      if (params.status) {
        logs = logs.filter(log => log.status === params.status);
      }

      const sortedLogs = this.sortData(logs, params.sortBy || 'startTime', params.sortOrder || 'desc');
      const paginatedLogs = this.paginateData(sortedLogs, params.page || 1, params.limit || 20);

      return {
        success: true,
        data: paginatedLogs
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch execution logs'
      };
    }
  }

  async getExecutionLogsByWorkflow(workflowId: string, params: PaginationParams = {}): Promise<ApiResponse<ExecutionLog[]>> {
    try {
      const logs = await databaseService.getExecutionLogsByWorkflowId(workflowId);
      const sortedLogs = this.sortData(logs, params.sortBy || 'startTime', params.sortOrder || 'desc');
      const paginatedLogs = this.paginateData(sortedLogs, params.page || 1, params.limit || 20);

      return {
        success: true,
        data: paginatedLogs
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch execution logs'
      };
    }
  }

  async createExecutionLog(logData: Omit<ExecutionLog, 'id'>): Promise<ApiResponse<ExecutionLog>> {
    try {
      const log = await databaseService.createExecutionLog(logData);
      return {
        success: true,
        data: log,
        message: 'Execution log created successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create execution log'
      };
    }
  }

  async updateExecutionLog(id: string, updates: Partial<ExecutionLog>): Promise<ApiResponse<ExecutionLog>> {
    try {
      const log = await databaseService.updateExecutionLog(id, updates);
      if (!log) {
        return {
          success: false,
          error: 'Execution log not found'
        };
      }

      return {
        success: true,
        data: log,
        message: 'Execution log updated successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update execution log'
      };
    }
  }

  // Script management endpoints
  async getScripts(params: SearchParams = {}): Promise<ApiResponse<ScriptRecord[]>> {
    try {
      let scripts: ScriptRecord[];

      if (params.query) {
        scripts = await databaseService.searchScripts(params.query, params.userId);
      } else if (params.userId) {
        scripts = await databaseService.getScriptsByUserId(params.userId);
      } else {
        scripts = await databaseService.getAllScripts();
      }

      // Filter by tags if provided
      if (params.tags && params.tags.length > 0) {
        scripts = scripts.filter(s => 
          params.tags!.some(tag => s.tags.includes(tag))
        );
      }

      const sortedScripts = this.sortData(scripts, params.sortBy || 'updatedAt', params.sortOrder || 'desc');
      const paginatedScripts = this.paginateData(sortedScripts, params.page || 1, params.limit || 10);

      return {
        success: true,
        data: paginatedScripts
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch scripts'
      };
    }
  }

  async getScriptById(id: string): Promise<ApiResponse<ScriptRecord>> {
    try {
      const script = await databaseService.getScriptById(id);
      if (!script) {
        return {
          success: false,
          error: 'Script not found'
        };
      }

      return {
        success: true,
        data: script
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch script'
      };
    }
  }

  async createScript(scriptData: Omit<ScriptRecord, 'id' | 'createdAt' | 'updatedAt' | 'version' | 'executionCount'>): Promise<ApiResponse<ScriptRecord>> {
    try {
      const script = await databaseService.saveScript(scriptData);
      return {
        success: true,
        data: script,
        message: 'Script created successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create script'
      };
    }
  }

  async updateScript(id: string, updates: Partial<ScriptRecord>): Promise<ApiResponse<ScriptRecord>> {
    try {
      const script = await databaseService.updateScript(id, updates);
      if (!script) {
        return {
          success: false,
          error: 'Script not found'
        };
      }

      return {
        success: true,
        data: script,
        message: 'Script updated successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update script'
      };
    }
  }

  async deleteScript(id: string): Promise<ApiResponse> {
    try {
      const success = await databaseService.deleteScript(id);
      if (!success) {
        return {
          success: false,
          error: 'Failed to delete script'
        };
      }

      return {
        success: true,
        message: 'Script deleted successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete script'
      };
    }
  }

  // Collaboration endpoints
  async getCollaborationSessions(params: PaginationParams = {}): Promise<ApiResponse<CollaborationSession[]>> {
    try {
      const sessions = await databaseService.getActiveCollaborationSessions();
      const sortedSessions = this.sortData(sessions, params.sortBy || 'createdAt', params.sortOrder || 'desc');
      const paginatedSessions = this.paginateData(sortedSessions, params.page || 1, params.limit || 10);

      return {
        success: true,
        data: paginatedSessions
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch collaboration sessions'
      };
    }
  }

  async createCollaborationSession(sessionData: Omit<CollaborationSession, 'id' | 'createdAt'>): Promise<ApiResponse<CollaborationSession>> {
    try {
      const session = await databaseService.createCollaborationSession(sessionData);
      return {
        success: true,
        data: session,
        message: 'Collaboration session created successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create collaboration session'
      };
    }
  }

  async updateCollaborationSession(id: string, updates: Partial<CollaborationSession>): Promise<ApiResponse<CollaborationSession>> {
    try {
      const session = await databaseService.updateCollaborationSession(id, updates);
      if (!session) {
        return {
          success: false,
          error: 'Collaboration session not found'
        };
      }

      return {
        success: true,
        data: session,
        message: 'Collaboration session updated successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update collaboration session'
      };
    }
  }

  // Analytics and statistics endpoints
  async getDashboardStats(): Promise<ApiResponse<any>> {
    try {
      const dbStats = await databaseService.getDatabaseStats();
      
      // Calculate additional analytics
      const workflows = await databaseService.getAllWorkflows();
      const executions = await databaseService.getAllExecutionLogs();
      
      const recentExecutions = executions
        .filter(e => e.startTime > Date.now() - 24 * 60 * 60 * 1000) // Last 24 hours
        .length;

      const successRate = executions.length > 0 
        ? (executions.filter(e => e.status === 'completed').length / executions.length) * 100
        : 0;

      const popularTags = this.getPopularTags(workflows);

      const stats = {
        ...dbStats,
        recentExecutions,
        successRate: Math.round(successRate * 100) / 100,
        popularTags,
        avgExecutionTime: this.calculateAverageExecutionTime(executions),
      };

      return {
        success: true,
        data: stats
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch dashboard stats'
      };
    }
  }

  // Data management endpoints
  async exportData(): Promise<ApiResponse<string>> {
    try {
      const data = await databaseService.exportData();
      return {
        success: true,
        data,
        message: 'Data exported successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to export data'
      };
    }
  }

  async importData(jsonData: string): Promise<ApiResponse> {
    try {
      const success = await databaseService.importData(jsonData);
      if (!success) {
        return {
          success: false,
          error: 'Failed to import data'
        };
      }

      return {
        success: true,
        message: 'Data imported successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to import data'
      };
    }
  }

  async clearAllData(): Promise<ApiResponse> {
    try {
      await databaseService.clearAllData();
      return {
        success: true,
        message: 'All data cleared successfully'
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to clear data'
      };
    }
  }

  // Utility methods
  private sortData<T>(data: T[], sortBy: string, sortOrder: 'asc' | 'desc'): T[] {
    return [...data].sort((a, b) => {
      const aValue = (a as any)[sortBy];
      const bValue = (b as any)[sortBy];

      if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
  }

  private paginateData<T>(data: T[], page: number, limit: number): T[] {
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    return data.slice(startIndex, endIndex);
  }

  private getPopularTags(workflows: WorkflowRecord[]): Array<{ tag: string; count: number }> {
    const tagCounts: Record<string, number> = {};
    
    workflows.forEach(workflow => {
      workflow.tags.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });

    return Object.entries(tagCounts)
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }

  private calculateAverageExecutionTime(executions: ExecutionLog[]): number {
    const completedExecutions = executions.filter(e => e.duration);
    if (completedExecutions.length === 0) return 0;

    const totalTime = completedExecutions.reduce((sum, e) => sum + (e.duration || 0), 0);
    return Math.round(totalTime / completedExecutions.length);
  }

  // Real-time updates (WebSocket simulation)
  private listeners: Map<string, Function[]> = new Map();

  subscribe(event: string, callback: Function): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  // Simulate real-time updates
  notifyWorkflowUpdate(workflow: WorkflowRecord): void {
    this.emit('workflow:updated', workflow);
  }

  notifyExecutionUpdate(execution: ExecutionLog): void {
    this.emit('execution:updated', execution);
  }

  notifyCollaborationUpdate(session: CollaborationSession): void {
    this.emit('collaboration:updated', session);
  }
}

// Export singleton instance
export const apiService = new ApiService();