export interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: {
    upload: number;
    download: number;
  };
  timestamp: number;
}

export interface WorkflowExecution {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  startTime: number;
  endTime?: number;
  logs: string[];
}

export interface CollaborationEvent {
  type: 'user_joined' | 'user_left' | 'cursor_move' | 'selection_change';
  userId: string;
  userName: string;
  data?: any;
  timestamp: number;
}

export type WebSocketMessage = {
  type: 'system_metrics';
  data: SystemMetrics;
} | {
  type: 'workflow_execution';
  data: WorkflowExecution;
} | {
  type: 'collaboration_event';
  data: CollaborationEvent;
} | {
  type: 'notification';
  data: {
    id: string;
    title: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
    timestamp: number;
  };
};

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private isConnecting = false;

  constructor(private url: string = 'ws://localhost:3001') {}

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return Promise.resolve();
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.isConnecting = false;
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.warn('WebSocket connection failed - this is expected in development mode without a backend server');
          this.isConnecting = false;
          reject(new Error('WebSocket connection failed - backend server not available'));
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private handleMessage(message: WebSocketMessage) {
    const listeners = this.listeners.get(message.type);
    if (listeners) {
      listeners.forEach(callback => callback(message.data));
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`WebSocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} - backend server not available`);
      
      setTimeout(() => {
        this.connect().catch(() => {
          // Silently handle reconnection failures in development mode
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.log('WebSocket reconnection attempts exhausted - running in offline mode with mock data');
    }
  }

  subscribe<T>(eventType: string, callback: (data: T) => void) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(eventType);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  send(message: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// Auto-connect when service is imported
if (typeof window !== 'undefined') {
  websocketService.connect().catch(() => {
    console.log('WebSocket auto-connect failed - running in development mode with mock data');
  });
}