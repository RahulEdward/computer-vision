import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface User {
  id: string;
  name: string;
  avatar?: string;
  cursor?: { x: number; y: number };
  selection?: string;
}

interface DashboardState {
  // Real-time collaboration
  users: Map<string, User>;
  currentUser: User | null;
  
  // Workflow builder
  nodes: any[];
  edges: any[];
  selectedNodes: string[];
  
  // UI state
  sidebarOpen: boolean;
  theme: 'light' | 'dark' | 'auto';
  layout: 'grid' | 'list' | 'kanban';
  
  // Voice control
  voiceEnabled: boolean;
  listening: boolean;
  
  // AR/3D
  arEnabled: boolean;
  threeDView: boolean;
  
  // Performance
  lastInteraction: number;
  
  // Actions
  addUser: (user: User) => void;
  removeUser: (userId: string) => void;
  updateUser: (userId: string, updates: Partial<User>) => void;
  setCurrentUser: (user: User) => void;
  
  addNode: (node: any) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, updates: any) => void;
  
  toggleSidebar: () => void;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  setLayout: (layout: 'grid' | 'list' | 'kanban') => void;
  
  toggleVoice: () => void;
  setListening: (listening: boolean) => void;
  
  toggleAR: () => void;
  toggle3D: () => void;
  
  recordInteraction: () => void;
}

// Initialize theme from localStorage
const getInitialTheme = (): 'light' | 'dark' | 'auto' => {
  if (typeof window === 'undefined') return 'dark';
  const stored = localStorage.getItem('theme');
  if (stored === 'light' || stored === 'dark' || stored === 'auto') {
    return stored;
  }
  return 'dark';
};

export const useDashboardStore = create<DashboardState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    users: new Map(),
    currentUser: null,
    nodes: [],
    edges: [],
    selectedNodes: [],
    sidebarOpen: true,
    theme: getInitialTheme(),
    layout: 'grid',
    voiceEnabled: false,
    listening: false,
    arEnabled: false,
    threeDView: false,
    lastInteraction: Date.now(),
    
    // Actions
    addUser: (user) => set((state) => {
      const newUsers = new Map(state.users);
      newUsers.set(user.id, user);
      return { users: newUsers };
    }),
    
    removeUser: (userId) => set((state) => {
      const newUsers = new Map(state.users);
      newUsers.delete(userId);
      return { users: newUsers };
    }),
    
    updateUser: (userId, updates) => set((state) => {
      const newUsers = new Map(state.users);
      const user = newUsers.get(userId);
      if (user) {
        newUsers.set(userId, { ...user, ...updates });
      }
      return { users: newUsers };
    }),
    
    setCurrentUser: (user) => set({ currentUser: user }),
    
    addNode: (node) => set((state) => ({
      nodes: [...state.nodes, node]
    })),
    
    removeNode: (nodeId) => set((state) => ({
      nodes: state.nodes.filter(n => n.id !== nodeId)
    })),
    
    updateNode: (nodeId, updates) => set((state) => ({
      nodes: state.nodes.map(n => n.id === nodeId ? { ...n, ...updates } : n)
    })),
    
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setTheme: (theme) => {
      console.log('🎨 setTheme called with:', theme);
      console.log('📍 Current HTML classes:', document.documentElement.className);
      
      // Persist theme to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('theme', theme);
        console.log('💾 Theme saved to localStorage:', theme);
        
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
          document.body.classList.add('dark');
          console.log('✅ Added dark class to html and body');
        } else {
          document.documentElement.classList.remove('dark');
          document.body.classList.remove('dark');
          console.log('✅ Removed dark class from html and body');
        }
        
        console.log('📍 New HTML classes:', document.documentElement.className);
      }
      
      set({ theme });
      console.log('✅ Store updated with theme:', theme);
    },
    setLayout: (layout) => set({ layout }),
    
    toggleVoice: () => set((state) => ({ voiceEnabled: !state.voiceEnabled })),
    setListening: (listening) => set({ listening }),
    
    toggleAR: () => set((state) => ({ arEnabled: !state.arEnabled })),
    toggle3D: () => set((state) => ({ threeDView: !state.threeDView })),
    
    recordInteraction: () => set({ lastInteraction: Date.now() })
  }))
);