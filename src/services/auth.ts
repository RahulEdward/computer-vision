// Authentication service with user management and security features
// Provides login, registration, session management, and authorization

import { apiService } from './api';
import { User } from './database';

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  role?: 'admin' | 'user' | 'viewer';
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
  error: string | null;
}

export interface SessionInfo {
  token: string;
  expiresAt: number;
  refreshToken: string;
  user: User;
}

class AuthService {
  private currentUser: User | null = null;
  private sessionToken: string | null = null;
  private refreshToken: string | null = null;
  private sessionExpiresAt: number = 0;
  private listeners: Function[] = [];

  constructor() {
    this.initializeFromStorage();
    this.setupSessionRefresh();
  }

  // Initialize authentication state from localStorage
  private initializeFromStorage(): void {
    try {
      const sessionData = localStorage.getItem('auth_session');
      if (sessionData) {
        const session: SessionInfo = JSON.parse(sessionData);
        
        // Check if session is still valid
        if (session.expiresAt > Date.now()) {
          this.currentUser = session.user;
          this.sessionToken = session.token;
          this.refreshToken = session.refreshToken;
          this.sessionExpiresAt = session.expiresAt;
          this.notifyListeners();
        } else {
          this.clearSession();
        }
      }
    } catch (error) {
      console.error('Failed to initialize auth from storage:', error);
      this.clearSession();
    }
  }

  // Setup automatic session refresh
  private setupSessionRefresh(): void {
    setInterval(() => {
      if (this.shouldRefreshSession()) {
        this.refreshSession();
      }
    }, 60000); // Check every minute
  }

  // Authentication methods
  async login(credentials: LoginCredentials): Promise<{ success: boolean; error?: string }> {
    try {
      // Validate credentials
      const validation = this.validateLoginCredentials(credentials);
      if (!validation.isValid) {
        return { success: false, error: validation.error };
      }

      // Attempt login through API
      const response = await apiService.login(credentials.email, credentials.password);
      
      if (!response.success || !response.data) {
        return { success: false, error: response.error || 'Login failed' };
      }

      // Create session
      const sessionDuration = credentials.rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000; // 30 days or 1 day
      const session = this.createSession(response.data, sessionDuration);
      
      // Store session
      this.storeSession(session);
      
      // Update current state
      this.currentUser = response.data;
      this.sessionToken = session.token;
      this.refreshToken = session.refreshToken;
      this.sessionExpiresAt = session.expiresAt;

      // Notify listeners
      this.notifyListeners();

      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Login failed' 
      };
    }
  }

  async register(data: RegisterData): Promise<{ success: boolean; error?: string }> {
    try {
      // Validate registration data
      const validation = this.validateRegistrationData(data);
      if (!validation.isValid) {
        return { success: false, error: validation.error };
      }

      // Create user data
      const userData = {
        name: data.name,
        email: data.email,
        role: data.role || 'user' as const,
        avatar: this.generateAvatar(data.name),
        preferences: {
          theme: 'dark' as const,
          notifications: true,
          autoSave: true,
        },
      };

      // Register through API
      const response = await apiService.register(userData);
      
      if (!response.success || !response.data) {
        return { success: false, error: response.error || 'Registration failed' };
      }

      // Auto-login after registration
      const loginResult = await this.login({
        email: data.email,
        password: data.password,
        rememberMe: false,
      });

      return loginResult;
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Registration failed' 
      };
    }
  }

  async logout(): Promise<void> {
    try {
      // Notify API
      await apiService.logout();
      
      // Clear session
      this.clearSession();
      
      // Notify listeners
      this.notifyListeners();
    } catch (error) {
      console.error('Logout error:', error);
      // Clear session anyway
      this.clearSession();
      this.notifyListeners();
    }
  }

  // Session management
  private createSession(user: User, duration: number): SessionInfo {
    const now = Date.now();
    return {
      token: this.generateToken(),
      refreshToken: this.generateToken(),
      expiresAt: now + duration,
      user,
    };
  }

  private storeSession(session: SessionInfo): void {
    try {
      localStorage.setItem('auth_session', JSON.stringify(session));
    } catch (error) {
      console.error('Failed to store session:', error);
    }
  }

  private clearSession(): void {
    this.currentUser = null;
    this.sessionToken = null;
    this.refreshToken = null;
    this.sessionExpiresAt = 0;
    
    try {
      localStorage.removeItem('auth_session');
    } catch (error) {
      console.error('Failed to clear session storage:', error);
    }
  }

  private shouldRefreshSession(): boolean {
    if (!this.sessionToken || !this.refreshToken) return false;
    
    // Refresh if session expires in the next 5 minutes
    const fiveMinutes = 5 * 60 * 1000;
    return this.sessionExpiresAt - Date.now() < fiveMinutes;
  }

  private async refreshSession(): Promise<void> {
    if (!this.refreshToken || !this.currentUser) return;

    try {
      // In a real implementation, this would call a refresh endpoint
      // For now, we'll extend the current session
      const newSession = this.createSession(this.currentUser, 24 * 60 * 60 * 1000); // 1 day
      this.storeSession(newSession);
      
      this.sessionToken = newSession.token;
      this.refreshToken = newSession.refreshToken;
      this.sessionExpiresAt = newSession.expiresAt;
    } catch (error) {
      console.error('Failed to refresh session:', error);
      this.logout();
    }
  }

  // User management
  async updateProfile(updates: Partial<User>): Promise<{ success: boolean; error?: string }> {
    if (!this.currentUser) {
      return { success: false, error: 'Not authenticated' };
    }

    try {
      const response = await apiService.updateUser(this.currentUser.id, updates);
      
      if (!response.success || !response.data) {
        return { success: false, error: response.error || 'Update failed' };
      }

      // Update current user
      this.currentUser = response.data;
      
      // Update stored session
      if (this.sessionToken && this.refreshToken) {
        const session: SessionInfo = {
          token: this.sessionToken,
          refreshToken: this.refreshToken,
          expiresAt: this.sessionExpiresAt,
          user: this.currentUser,
        };
        this.storeSession(session);
      }

      this.notifyListeners();
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Update failed' 
      };
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<{ success: boolean; error?: string }> {
    if (!this.currentUser) {
      return { success: false, error: 'Not authenticated' };
    }

    try {
      // Validate new password
      const validation = this.validatePassword(newPassword);
      if (!validation.isValid) {
        return { success: false, error: validation.error };
      }

      // In a real implementation, this would verify current password and update
      // For now, we'll simulate success
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Password change failed' 
      };
    }
  }

  // Authorization
  hasPermission(permission: string): boolean {
    if (!this.currentUser) return false;

    const rolePermissions = {
      admin: ['read', 'write', 'delete', 'manage_users', 'manage_system'],
      user: ['read', 'write'],
      viewer: ['read'],
    };

    const userPermissions = rolePermissions[this.currentUser.role] || [];
    return userPermissions.includes(permission);
  }

  canAccessResource(resourceType: string, resourceUserId?: string): boolean {
    if (!this.currentUser) return false;

    // Admins can access everything
    if (this.currentUser.role === 'admin') return true;

    // Users can access their own resources
    if (resourceUserId && resourceUserId === this.currentUser.id) return true;

    // Check specific resource permissions
    switch (resourceType) {
      case 'public_workflow':
        return true;
      case 'private_workflow':
        return resourceUserId === this.currentUser.id;
      case 'user_profile':
        return resourceUserId === this.currentUser.id || this.currentUser.role === 'admin';
      default:
        return false;
    }
  }

  // Getters
  getCurrentUser(): User | null {
    return this.currentUser;
  }

  isAuthenticated(): boolean {
    return this.currentUser !== null && this.sessionToken !== null && this.sessionExpiresAt > Date.now();
  }

  getAuthState(): AuthState {
    return {
      isAuthenticated: this.isAuthenticated(),
      user: this.currentUser,
      loading: false,
      error: null,
    };
  }

  getSessionToken(): string | null {
    return this.isAuthenticated() ? this.sessionToken : null;
  }

  // Event listeners
  subscribe(listener: Function): () => void {
    this.listeners.push(listener);
    
    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  private notifyListeners(): void {
    const authState = this.getAuthState();
    this.listeners.forEach(listener => {
      try {
        listener(authState);
      } catch (error) {
        console.error('Auth listener error:', error);
      }
    });
  }

  // Validation methods
  private validateLoginCredentials(credentials: LoginCredentials): { isValid: boolean; error?: string } {
    if (!credentials.email) {
      return { isValid: false, error: 'Email is required' };
    }

    if (!this.isValidEmail(credentials.email)) {
      return { isValid: false, error: 'Invalid email format' };
    }

    if (!credentials.password) {
      return { isValid: false, error: 'Password is required' };
    }

    return { isValid: true };
  }

  private validateRegistrationData(data: RegisterData): { isValid: boolean; error?: string } {
    if (!data.name || data.name.trim().length < 2) {
      return { isValid: false, error: 'Name must be at least 2 characters' };
    }

    if (!data.email) {
      return { isValid: false, error: 'Email is required' };
    }

    if (!this.isValidEmail(data.email)) {
      return { isValid: false, error: 'Invalid email format' };
    }

    const passwordValidation = this.validatePassword(data.password);
    if (!passwordValidation.isValid) {
      return passwordValidation;
    }

    if (data.password !== data.confirmPassword) {
      return { isValid: false, error: 'Passwords do not match' };
    }

    return { isValid: true };
  }

  private validatePassword(password: string): { isValid: boolean; error?: string } {
    if (!password) {
      return { isValid: false, error: 'Password is required' };
    }

    if (password.length < 8) {
      return { isValid: false, error: 'Password must be at least 8 characters' };
    }

    if (!/(?=.*[a-z])/.test(password)) {
      return { isValid: false, error: 'Password must contain at least one lowercase letter' };
    }

    if (!/(?=.*[A-Z])/.test(password)) {
      return { isValid: false, error: 'Password must contain at least one uppercase letter' };
    }

    if (!/(?=.*\d)/.test(password)) {
      return { isValid: false, error: 'Password must contain at least one number' };
    }

    return { isValid: true };
  }

  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  // Utility methods
  private generateToken(): string {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
  }

  private generateAvatar(name: string): string {
    // Generate a simple avatar URL based on name
    const initials = name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
    
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=6366f1&color=ffffff&size=128`;
  }

  // Security utilities
  async hashPassword(password: string): Promise<string> {
    // In a real implementation, use proper password hashing (bcrypt, scrypt, etc.)
    // This is a simple example for demonstration
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  async verifyPassword(password: string, hash: string): Promise<boolean> {
    const passwordHash = await this.hashPassword(password);
    return passwordHash === hash;
  }

  // Rate limiting (simple implementation)
  private loginAttempts: Map<string, { count: number; lastAttempt: number }> = new Map();

  private checkRateLimit(email: string): boolean {
    const now = Date.now();
    const attempts = this.loginAttempts.get(email);
    
    if (!attempts) {
      this.loginAttempts.set(email, { count: 1, lastAttempt: now });
      return true;
    }

    // Reset attempts after 15 minutes
    if (now - attempts.lastAttempt > 15 * 60 * 1000) {
      this.loginAttempts.set(email, { count: 1, lastAttempt: now });
      return true;
    }

    // Allow up to 5 attempts
    if (attempts.count >= 5) {
      return false;
    }

    attempts.count++;
    attempts.lastAttempt = now;
    return true;
  }

  private resetRateLimit(email: string): void {
    this.loginAttempts.delete(email);
  }
}

// Export singleton instance
export const authService = new AuthService();