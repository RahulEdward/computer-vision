import SimplePeer from 'simple-peer';
import * as Y from 'yjs';
import { WebrtcProvider } from 'y-webrtc';

export class CollaborationEngine {
  private doc: Y.Doc;
  private provider: WebrtcProvider;
  private peers: Map<string, SimplePeer.Instance> = new Map();
  private cursors: Y.Map<any>;
  private awareness: any;

  constructor(roomId: string) {
    this.doc = new Y.Doc();
    this.provider = new WebrtcProvider(roomId, this.doc, {
      signaling: ['wss://signaling.yjs.dev'],
      maxConns: 20,
      filterBcConns: true,
      peerOpts: {}
    });

    this.cursors = this.doc.getMap('cursors');
    this.awareness = this.provider.awareness;
    
    this.setupEventListeners();
  }

  private setupEventListeners() {
    // Listen for awareness changes (user cursors, selections)
    this.awareness.on('change', (changes: any) => {
      changes.added.forEach((clientId: number) => {
        const user = this.awareness.getStates().get(clientId);
        this.onUserJoined?.(user);
      });

      changes.removed.forEach((clientId: number) => {
        this.onUserLeft?.(clientId);
      });

      changes.updated.forEach((clientId: number) => {
        const user = this.awareness.getStates().get(clientId);
        this.onUserUpdated?.(user);
      });
    });

    // Listen for document changes
    this.doc.on('update', (update: Uint8Array) => {
      this.onDocumentUpdate?.(update);
    });
  }

  // Callbacks - set these from your React components
  onUserJoined?: (user: any) => void;
  onUserLeft?: (clientId: number) => void;
  onUserUpdated?: (user: any) => void;
  onDocumentUpdate?: (update: Uint8Array) => void;

  // Update user cursor position
  updateCursor(x: number, y: number) {
    this.awareness.setLocalStateField('cursor', { x, y, timestamp: Date.now() });
  }

  // Update user selection
  updateSelection(selection: any) {
    this.awareness.setLocalStateField('selection', selection);
  }

  // Set user info
  setUser(user: { id: string; name: string; avatar?: string; color?: string }) {
    this.awareness.setLocalStateField('user', user);
  }

  // Get shared document for collaborative editing
  getSharedDoc() {
    return this.doc;
  }

  // Get shared text for editor content
  getSharedText(key: string = 'editor') {
    return this.doc.getText(key);
  }

  // Get shared map for workflow data
  getWorkflowMap() {
    return this.doc.getMap('workflow');
  }

  // Get shared array for nodes
  getNodesArray() {
    return this.doc.getArray('nodes');
  }

  // Get shared array for edges
  getEdgesArray() {
    return this.doc.getArray('edges');
  }

  // Simple text synchronization for editor
  syncEditorContent(content: string, key: string = 'editor') {
    const yText = this.getSharedText(key);
    const currentContent = yText.toString();
    
    if (currentContent !== content) {
      yText.delete(0, yText.length);
      yText.insert(0, content);
    }
  }

  // Get current editor content
  getEditorContent(key: string = 'editor'): string {
    return this.getSharedText(key).toString();
  }

  // Broadcast custom event to all peers
  broadcast(event: string, data: any) {
    this.awareness.setLocalStateField('broadcast', {
      event,
      data,
      timestamp: Date.now()
    });
  }

  // Clean up
  destroy() {
    this.provider.destroy();
    this.doc.destroy();
    this.peers.forEach(peer => peer.destroy());
  }
}

// React hook for collaboration
import { useEffect, useRef, useState } from 'react';
import { useDashboardStore } from './store';

export function useCollaboration(roomId: string) {
  const [engine, setEngine] = useState<CollaborationEngine | null>(null);
  const { addUser, removeUser, updateUser, setCurrentUser } = useDashboardStore();
  
  useEffect(() => {
    const collaborationEngine = new CollaborationEngine(roomId);
    
    collaborationEngine.onUserJoined = (user) => {
      addUser({
        id: user.user?.id || `user-${Date.now()}`,
        name: user.user?.name || 'Anonymous',
        avatar: user.user?.avatar,
        cursor: user.cursor
      });
    };
    
    collaborationEngine.onUserLeft = (clientId) => {
      removeUser(clientId.toString());
    };
    
    collaborationEngine.onUserUpdated = (user) => {
      if (user.user?.id) {
        updateUser(user.user.id, {
          cursor: user.cursor,
          selection: user.selection
        });
      }
    };
    
    setEngine(collaborationEngine);
    
    return () => {
      collaborationEngine.destroy();
    };
  }, [roomId, addUser, removeUser, updateUser]);
  
  return engine;
}