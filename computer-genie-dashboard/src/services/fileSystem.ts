export interface SavedWorkflow {
  id: string;
  name: string;
  description?: string;
  nodes: any[];
  edges: any[];
  createdAt: number;
  updatedAt: number;
  version: string;
  tags?: string[];
}

export interface SavedScript {
  id: string;
  name: string;
  language: string;
  code: string;
  description?: string;
  createdAt: number;
  updatedAt: number;
  tags?: string[];
}

export interface FileSystemStats {
  totalWorkflows: number;
  totalScripts: number;
  storageUsed: number;
  lastBackup?: number;
}

class FileSystemService {
  private readonly WORKFLOWS_KEY = 'computer_genie_workflows';
  private readonly SCRIPTS_KEY = 'computer_genie_scripts';
  private readonly SETTINGS_KEY = 'computer_genie_settings';

  // Workflow Management
  async saveWorkflow(workflow: Omit<SavedWorkflow, 'id' | 'createdAt' | 'updatedAt' | 'version'>): Promise<SavedWorkflow> {
    const workflows = this.getWorkflows();
    const now = Date.now();
    
    const savedWorkflow: SavedWorkflow = {
      ...workflow,
      id: `workflow_${now}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: now,
      updatedAt: now,
      version: '1.0.0',
    };

    workflows.push(savedWorkflow);
    this.setWorkflows(workflows);
    
    return savedWorkflow;
  }

  async updateWorkflow(id: string, updates: Partial<SavedWorkflow>): Promise<SavedWorkflow | null> {
    const workflows = this.getWorkflows();
    const index = workflows.findIndex(w => w.id === id);
    
    if (index === -1) return null;

    const updatedWorkflow = {
      ...workflows[index],
      ...updates,
      updatedAt: Date.now(),
    };

    workflows[index] = updatedWorkflow;
    this.setWorkflows(workflows);
    
    return updatedWorkflow;
  }

  async deleteWorkflow(id: string): Promise<boolean> {
    const workflows = this.getWorkflows();
    const filteredWorkflows = workflows.filter(w => w.id !== id);
    
    if (filteredWorkflows.length === workflows.length) return false;
    
    this.setWorkflows(filteredWorkflows);
    return true;
  }

  getWorkflows(): SavedWorkflow[] {
    try {
      const data = localStorage.getItem(this.WORKFLOWS_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error loading workflows:', error);
      return [];
    }
  }

  getWorkflow(id: string): SavedWorkflow | null {
    const workflows = this.getWorkflows();
    return workflows.find(w => w.id === id) || null;
  }

  private setWorkflows(workflows: SavedWorkflow[]): void {
    try {
      localStorage.setItem(this.WORKFLOWS_KEY, JSON.stringify(workflows));
    } catch (error) {
      console.error('Error saving workflows:', error);
      throw new Error('Failed to save workflow. Storage may be full.');
    }
  }

  // Script Management
  async saveScript(script: Omit<SavedScript, 'id' | 'createdAt' | 'updatedAt'>): Promise<SavedScript> {
    const scripts = this.getScripts();
    const now = Date.now();
    
    const savedScript: SavedScript = {
      ...script,
      id: `script_${now}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: now,
      updatedAt: now,
    };

    scripts.push(savedScript);
    this.setScripts(scripts);
    
    return savedScript;
  }

  async updateScript(id: string, updates: Partial<SavedScript>): Promise<SavedScript | null> {
    const scripts = this.getScripts();
    const index = scripts.findIndex(s => s.id === id);
    
    if (index === -1) return null;

    const updatedScript = {
      ...scripts[index],
      ...updates,
      updatedAt: Date.now(),
    };

    scripts[index] = updatedScript;
    this.setScripts(scripts);
    
    return updatedScript;
  }

  async deleteScript(id: string): Promise<boolean> {
    const scripts = this.getScripts();
    const filteredScripts = scripts.filter(s => s.id !== id);
    
    if (filteredScripts.length === scripts.length) return false;
    
    this.setScripts(filteredScripts);
    return true;
  }

  getScripts(): SavedScript[] {
    try {
      const data = localStorage.getItem(this.SCRIPTS_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error loading scripts:', error);
      return [];
    }
  }

  getScript(id: string): SavedScript | null {
    const scripts = this.getScripts();
    return scripts.find(s => s.id === id) || null;
  }

  private setScripts(scripts: SavedScript[]): void {
    try {
      localStorage.setItem(this.SCRIPTS_KEY, JSON.stringify(scripts));
    } catch (error) {
      console.error('Error saving scripts:', error);
      throw new Error('Failed to save script. Storage may be full.');
    }
  }

  // File Import/Export
  async exportWorkflow(id: string): Promise<void> {
    const workflow = this.getWorkflow(id);
    if (!workflow) throw new Error('Workflow not found');

    const dataStr = JSON.stringify(workflow, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `${workflow.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async importWorkflow(file: File): Promise<SavedWorkflow> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const content = e.target?.result as string;
          const workflowData = JSON.parse(content);
          
          // Validate workflow structure
          if (!workflowData.name || !workflowData.nodes || !workflowData.edges) {
            throw new Error('Invalid workflow file format');
          }

          // Create new workflow with imported data
          const importedWorkflow = await this.saveWorkflow({
            name: `${workflowData.name} (Imported)`,
            description: workflowData.description,
            nodes: workflowData.nodes,
            edges: workflowData.edges,
            tags: workflowData.tags,
          });

          resolve(importedWorkflow);
        } catch (error) {
          reject(new Error('Failed to import workflow: ' + (error as Error).message));
        }
      };

      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  async exportScript(id: string): Promise<void> {
    const script = this.getScript(id);
    if (!script) throw new Error('Script not found');

    const blob = new Blob([script.code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const extension = this.getFileExtension(script.language);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${script.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async importScript(file: File, language?: string): Promise<SavedScript> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const content = e.target?.result as string;
          const fileName = file.name.replace(/\.[^/.]+$/, '');
          const detectedLanguage = language || this.detectLanguageFromExtension(file.name);
          
          const importedScript = await this.saveScript({
            name: `${fileName} (Imported)`,
            language: detectedLanguage,
            code: content,
            description: `Imported from ${file.name}`,
          });

          resolve(importedScript);
        } catch (error) {
          reject(new Error('Failed to import script: ' + (error as Error).message));
        }
      };

      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  private getFileExtension(language: string): string {
    const extensions: Record<string, string> = {
      javascript: 'js',
      typescript: 'ts',
      python: 'py',
      json: 'json',
      yaml: 'yml',
      markdown: 'md',
    };
    return extensions[language] || 'txt';
  }

  private detectLanguageFromExtension(filename: string): string {
    const extension = filename.split('.').pop()?.toLowerCase();
    const languages: Record<string, string> = {
      js: 'javascript',
      ts: 'typescript',
      py: 'python',
      json: 'json',
      yml: 'yaml',
      yaml: 'yaml',
      md: 'markdown',
    };
    return languages[extension || ''] || 'javascript';
  }

  // Search and Filter
  searchWorkflows(query: string, tags?: string[]): SavedWorkflow[] {
    const workflows = this.getWorkflows();
    const lowerQuery = query.toLowerCase();
    
    return workflows.filter(workflow => {
      const matchesQuery = !query || 
        workflow.name.toLowerCase().includes(lowerQuery) ||
        workflow.description?.toLowerCase().includes(lowerQuery);
      
      const matchesTags = !tags?.length || 
        tags.some(tag => workflow.tags?.includes(tag));
      
      return matchesQuery && matchesTags;
    });
  }

  searchScripts(query: string, language?: string, tags?: string[]): SavedScript[] {
    const scripts = this.getScripts();
    const lowerQuery = query.toLowerCase();
    
    return scripts.filter(script => {
      const matchesQuery = !query || 
        script.name.toLowerCase().includes(lowerQuery) ||
        script.description?.toLowerCase().includes(lowerQuery) ||
        script.code.toLowerCase().includes(lowerQuery);
      
      const matchesLanguage = !language || script.language === language;
      
      const matchesTags = !tags?.length || 
        tags.some(tag => script.tags?.includes(tag));
      
      return matchesQuery && matchesLanguage && matchesTags;
    });
  }

  // Statistics and Management
  getStats(): FileSystemStats {
    const workflows = this.getWorkflows();
    const scripts = this.getScripts();
    
    // Calculate approximate storage usage
    const workflowsSize = JSON.stringify(workflows).length;
    const scriptsSize = JSON.stringify(scripts).length;
    
    return {
      totalWorkflows: workflows.length,
      totalScripts: scripts.length,
      storageUsed: workflowsSize + scriptsSize,
      lastBackup: this.getLastBackupTime(),
    };
  }

  async createBackup(): Promise<void> {
    const workflows = this.getWorkflows();
    const scripts = this.getScripts();
    
    const backup = {
      version: '1.0.0',
      timestamp: Date.now(),
      workflows,
      scripts,
    };

    const dataStr = JSON.stringify(backup, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `computer_genie_backup_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Update last backup time
    localStorage.setItem(`${this.SETTINGS_KEY}_last_backup`, Date.now().toString());
  }

  async restoreBackup(file: File): Promise<{ workflows: number; scripts: number }> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const backup = JSON.parse(content);
          
          if (!backup.workflows || !backup.scripts) {
            throw new Error('Invalid backup file format');
          }

          // Restore workflows
          this.setWorkflows(backup.workflows);
          
          // Restore scripts
          this.setScripts(backup.scripts);
          
          resolve({
            workflows: backup.workflows.length,
            scripts: backup.scripts.length,
          });
        } catch (error) {
          reject(new Error('Failed to restore backup: ' + (error as Error).message));
        }
      };

      reader.onerror = () => reject(new Error('Failed to read backup file'));
      reader.readAsText(file);
    });
  }

  private getLastBackupTime(): number | undefined {
    const lastBackup = localStorage.getItem(`${this.SETTINGS_KEY}_last_backup`);
    return lastBackup ? parseInt(lastBackup) : undefined;
  }

  // Cleanup and Maintenance
  async clearAll(): Promise<void> {
    localStorage.removeItem(this.WORKFLOWS_KEY);
    localStorage.removeItem(this.SCRIPTS_KEY);
  }

  async clearWorkflows(): Promise<void> {
    localStorage.removeItem(this.WORKFLOWS_KEY);
  }

  async clearScripts(): Promise<void> {
    localStorage.removeItem(this.SCRIPTS_KEY);
  }
}

// Export singleton instance
export const fileSystemService = new FileSystemService();