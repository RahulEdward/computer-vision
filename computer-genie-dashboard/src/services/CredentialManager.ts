import { 
  ICredentialManager, 
  ICredentialType, 
  ICredentialData, 
  ICredentialTestResult 
} from '../types/credentials';

class CredentialManager implements ICredentialManager {
  private credentials: Map<string, ICredentialData> = new Map();
  private credentialTypes: Map<string, ICredentialType> = new Map();
  private encryptionKey = 'n8n-workflow-builder-key'; // In production, use proper encryption

  constructor() {
    this.initializeCredentialTypes();
    this.loadCredentials();
  }

  private initializeCredentialTypes() {
    // HTTP Basic Auth
    this.credentialTypes.set('httpBasicAuth', {
      name: 'httpBasicAuth',
      displayName: 'HTTP Basic Auth',
      description: 'Basic authentication for HTTP requests',
      icon: 'key',
      properties: [
        {
          displayName: 'Username',
          name: 'username',
          type: 'string',
          required: true,
          placeholder: 'Enter username'
        },
        {
          displayName: 'Password',
          name: 'password',
          type: 'password',
          required: true,
          placeholder: 'Enter password',
          typeOptions: {
            password: true
          }
        }
      ]
    });

    // API Key
    this.credentialTypes.set('apiKey', {
      name: 'apiKey',
      displayName: 'API Key',
      description: 'API Key authentication',
      icon: 'key',
      properties: [
        {
          displayName: 'API Key',
          name: 'apiKey',
          type: 'password',
          required: true,
          placeholder: 'Enter API key',
          typeOptions: {
            password: true
          }
        },
        {
          displayName: 'Header Name',
          name: 'headerName',
          type: 'string',
          default: 'Authorization',
          placeholder: 'Authorization'
        },
        {
          displayName: 'Prefix',
          name: 'prefix',
          type: 'string',
          default: 'Bearer',
          placeholder: 'Bearer'
        }
      ]
    });

    // OAuth2
    this.credentialTypes.set('oauth2', {
      name: 'oauth2',
      displayName: 'OAuth2',
      description: 'OAuth2 authentication',
      icon: 'shield',
      properties: [
        {
          displayName: 'Client ID',
          name: 'clientId',
          type: 'string',
          required: true,
          placeholder: 'Enter client ID'
        },
        {
          displayName: 'Client Secret',
          name: 'clientSecret',
          type: 'password',
          required: true,
          placeholder: 'Enter client secret',
          typeOptions: {
            password: true
          }
        },
        {
          displayName: 'Authorization URL',
          name: 'authUrl',
          type: 'string',
          required: true,
          placeholder: 'https://example.com/oauth/authorize'
        },
        {
          displayName: 'Token URL',
          name: 'tokenUrl',
          type: 'string',
          required: true,
          placeholder: 'https://example.com/oauth/token'
        },
        {
          displayName: 'Scope',
          name: 'scope',
          type: 'string',
          placeholder: 'read write'
        }
      ]
    });

    // Database Connection
    this.credentialTypes.set('database', {
      name: 'database',
      displayName: 'Database Connection',
      description: 'Database connection credentials',
      icon: 'database',
      properties: [
        {
          displayName: 'Host',
          name: 'host',
          type: 'string',
          required: true,
          placeholder: 'localhost'
        },
        {
          displayName: 'Port',
          name: 'port',
          type: 'number',
          default: 5432,
          placeholder: '5432'
        },
        {
          displayName: 'Database',
          name: 'database',
          type: 'string',
          required: true,
          placeholder: 'mydb'
        },
        {
          displayName: 'Username',
          name: 'username',
          type: 'string',
          required: true,
          placeholder: 'username'
        },
        {
          displayName: 'Password',
          name: 'password',
          type: 'password',
          required: true,
          placeholder: 'password',
          typeOptions: {
            password: true
          }
        }
      ]
    });
  }

  private loadCredentials() {
    // Load from localStorage in browser environment
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('n8n-credentials');
      if (stored) {
        try {
          const credentialsArray = JSON.parse(stored);
          credentialsArray.forEach((cred: ICredentialData) => {
            this.credentials.set(cred.id, {
              ...cred,
              createdAt: new Date(cred.createdAt),
              updatedAt: new Date(cred.updatedAt)
            });
          });
        } catch (error) {
          console.error('Failed to load credentials:', error);
        }
      }
    }
  }

  private saveCredentials() {
    if (typeof window !== 'undefined') {
      const credentialsArray = Array.from(this.credentials.values());
      localStorage.setItem('n8n-credentials', JSON.stringify(credentialsArray));
    }
  }

  getCredentialTypes(): ICredentialType[] {
    return Array.from(this.credentialTypes.values());
  }

  getCredentials(): ICredentialData[] {
    return Array.from(this.credentials.values());
  }

  getCredential(id: string): ICredentialData | null {
    return this.credentials.get(id) || null;
  }

  async saveCredential(credential: Omit<ICredentialData, 'id' | 'createdAt' | 'updatedAt'>): Promise<ICredentialData> {
    const id = `cred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const now = new Date();
    
    const newCredential: ICredentialData = {
      ...credential,
      id,
      createdAt: now,
      updatedAt: now,
      data: this.encryptCredentialData(credential.data)
    };

    this.credentials.set(id, newCredential);
    this.saveCredentials();
    
    return newCredential;
  }

  async updateCredential(id: string, updates: Partial<ICredentialData>): Promise<ICredentialData> {
    const existing = this.credentials.get(id);
    if (!existing) {
      throw new Error(`Credential with id ${id} not found`);
    }

    const updated: ICredentialData = {
      ...existing,
      ...updates,
      id, // Ensure ID doesn't change
      updatedAt: new Date(),
      data: updates.data ? this.encryptCredentialData(updates.data) : existing.data
    };

    this.credentials.set(id, updated);
    this.saveCredentials();
    
    return updated;
  }

  async deleteCredential(id: string): Promise<boolean> {
    const deleted = this.credentials.delete(id);
    if (deleted) {
      this.saveCredentials();
    }
    return deleted;
  }

  async testCredential(credentialData: ICredentialData): Promise<ICredentialTestResult> {
    try {
      const decryptedData = this.decryptCredentialData(credentialData.data as any);
      
      // Basic validation based on credential type
      switch (credentialData.type) {
        case 'httpBasicAuth':
          if (!decryptedData.username || !decryptedData.password) {
            return {
              success: false,
              message: 'Username and password are required'
            };
          }
          break;
          
        case 'apiKey':
          if (!decryptedData.apiKey) {
            return {
              success: false,
              message: 'API key is required'
            };
          }
          break;
          
        case 'oauth2':
          if (!decryptedData.clientId || !decryptedData.clientSecret) {
            return {
              success: false,
              message: 'Client ID and Client Secret are required'
            };
          }
          break;
          
        case 'database':
          if (!decryptedData.host || !decryptedData.database || !decryptedData.username) {
            return {
              success: false,
              message: 'Host, database, and username are required'
            };
          }
          break;
      }

      // In a real implementation, you would test the actual connection
      return {
        success: true,
        message: 'Credential test successful'
      };
    } catch (error) {
      return {
        success: false,
        message: `Credential test failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  encryptCredentialData(data: Record<string, any>): string {
    // Simple base64 encoding for demo purposes
    // In production, use proper encryption like AES
    return btoa(JSON.stringify(data));
  }

  decryptCredentialData(encryptedData: string): Record<string, any> {
    try {
      return JSON.parse(atob(encryptedData));
    } catch (error) {
      throw new Error('Failed to decrypt credential data');
    }
  }

  getCredentialType(typeName: string): ICredentialType | null {
    return this.credentialTypes.get(typeName) || null;
  }
}

// Create singleton instance
export const credentialManager = new CredentialManager();