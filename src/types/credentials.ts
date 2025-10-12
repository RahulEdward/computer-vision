export interface ICredentialType {
  name: string;
  displayName: string;
  description?: string;
  icon?: string;
  properties: ICredentialProperty[];
  authenticate?: {
    type: 'generic' | 'oauth2' | 'oauth1' | 'apiKey' | 'basic';
    properties: Record<string, any>;
  };
}

export interface ICredentialProperty {
  displayName: string;
  name: string;
  type: 'string' | 'password' | 'hidden' | 'boolean' | 'number' | 'options';
  required?: boolean;
  default?: any;
  description?: string;
  placeholder?: string;
  options?: Array<{
    name: string;
    value: string;
  }>;
  typeOptions?: {
    password?: boolean;
    multipleValues?: boolean;
    multipleValueButtonText?: string;
  };
}

export interface ICredentialData {
  id: string;
  name: string;
  type: string;
  data: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
  isActive: boolean;
}

export interface ICredentialTestResult {
  success: boolean;
  message: string;
  details?: any;
}

export interface ICredentialManager {
  getCredentialTypes(): ICredentialType[];
  getCredentials(): ICredentialData[];
  getCredential(id: string): ICredentialData | null;
  saveCredential(credential: Omit<ICredentialData, 'id' | 'createdAt' | 'updatedAt'>): Promise<ICredentialData>;
  updateCredential(id: string, updates: Partial<ICredentialData>): Promise<ICredentialData>;
  deleteCredential(id: string): Promise<boolean>;
  testCredential(credentialData: ICredentialData): Promise<ICredentialTestResult>;
  encryptCredentialData(data: Record<string, any>): string;
  decryptCredentialData(encryptedData: string): Record<string, any>;
}