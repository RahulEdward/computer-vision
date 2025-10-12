'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '@/lib/store';
import { useCollaboration } from '@/lib/webrtc-collaboration';

interface CollaborationUser {
  id: string;
  name: string;
  avatar?: string;
  cursor?: { x: number; y: number };
  selection?: string;
  status: 'online' | 'away' | 'busy';
  lastSeen: Date;
}

export default function CollaborationPanel() {
  const { users, currentUser } = useDashboardStore();
  const [showUserDetails, setShowUserDetails] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{
    id: string;
    userId: string;
    message: string;
    timestamp: Date;
  }>>([]);
  const [newMessage, setNewMessage] = useState('');
  const [showChat, setShowChat] = useState(false);

  // Convert Map to Array for easier rendering
  const userList = Array.from(users.values());

  const sendMessage = () => {
    if (!newMessage.trim() || !currentUser) return;

    const message = {
      id: Date.now().toString(),
      userId: currentUser.id,
      message: newMessage.trim(),
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, message]);
    setNewMessage('');

    // In a real implementation, broadcast this message via WebRTC
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      case 'busy': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'online': return 'Online';
      case 'away': return 'Away';
      case 'busy': return 'Busy';
      default: return 'Offline';
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200">
          👥 Collaboration
        </h3>
        <div className="flex items-center space-x-2">
          <div className="flex -space-x-2">
            {userList.slice(0, 3).map((user) => (
              <div
                key={user.id}
                className="w-6 h-6 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 border-2 border-white dark:border-slate-800 flex items-center justify-center text-xs text-white font-medium"
                title={user.name}
              >
                {user.name.charAt(0).toUpperCase()}
              </div>
            ))}
            {userList.length > 3 && (
              <div className="w-6 h-6 rounded-full bg-slate-400 border-2 border-white dark:border-slate-800 flex items-center justify-center text-xs text-white font-medium">
                +{userList.length - 3}
              </div>
            )}
          </div>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {userList.length} online
          </span>
        </div>
      </div>

      {/* Current User */}
      {currentUser && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white font-medium">
                {currentUser.name.charAt(0).toUpperCase()}
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white dark:border-slate-800" />
            </div>
            <div>
              <div className="font-medium text-slate-900 dark:text-slate-100">
                {currentUser.name} (You)
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">
                Online
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Users */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Active Users ({userList.length})
        </h4>
        
        <div className="space-y-2 max-h-48 overflow-y-auto">
          <AnimatePresence>
            {userList.map((user) => (
              <motion.div
                key={user.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
                onClick={() => setShowUserDetails(showUserDetails === user.id ? null : user.id)}
              >
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center text-white text-sm font-medium">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-white dark:border-slate-800 ${getStatusColor('online')}`} />
                  </div>
                  <div>
                    <div className="font-medium text-sm text-slate-900 dark:text-slate-100">
                      {user.name}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {getStatusText('online')}
                    </div>
                  </div>
                </div>
                
                {user.cursor && (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      Active
                    </span>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {userList.length === 0 && (
          <div className="text-center py-4 text-slate-500 dark:text-slate-400">
            <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <p className="text-sm">No other users online</p>
            <p className="text-xs mt-1">Share your room ID to invite collaborators</p>
          </div>
        )}
      </div>

      {/* User Details Modal */}
      <AnimatePresence>
        {showUserDetails && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
            onClick={() => setShowUserDetails(null)}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="bg-white dark:bg-slate-800 rounded-lg p-6 max-w-sm w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              {(() => {
                const user = userList.find(u => u.id === showUserDetails);
                if (!user) return null;
                
                return (
                  <div className="space-y-4">
                    <div className="flex items-center space-x-4">
                      <div className="w-16 h-16 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center text-white text-xl font-medium">
                        {user.name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                          {user.name}
                        </h3>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {getStatusText('online')}
                        </p>
                      </div>
                    </div>
                    
                    {user.cursor && (
                      <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-3">
                        <h4 className="text-sm font-medium mb-2">Current Activity</h4>
                        <p className="text-xs text-slate-600 dark:text-slate-400">
                          Cursor at ({user.cursor.x}, {user.cursor.y})
                        </p>
                        {user.selection && (
                          <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                            Selected: {user.selection}
                          </p>
                        )}
                      </div>
                    )}
                    
                    <div className="flex space-x-2">
                      <button
                        onClick={() => {
                          // Start private chat
                          setShowUserDetails(null);
                          setShowChat(true);
                        }}
                        className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                      >
                        Message
                      </button>
                      <button
                        onClick={() => setShowUserDetails(null)}
                        className="px-3 py-2 bg-slate-200 dark:bg-slate-600 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-500 transition-colors text-sm"
                      >
                        Close
                      </button>
                    </div>
                  </div>
                );
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Toggle */}
      <button
        onClick={() => setShowChat(!showChat)}
        className="w-full px-3 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors text-sm flex items-center justify-center space-x-2"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
        <span>{showChat ? 'Hide Chat' : 'Show Chat'}</span>
        {chatMessages.length > 0 && (
          <span className="bg-blue-500 text-white text-xs rounded-full px-2 py-0.5">
            {chatMessages.length}
          </span>
        )}
      </button>

      {/* Chat Panel */}
      <AnimatePresence>
        {showChat && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-50 dark:bg-slate-800 rounded-lg overflow-hidden"
          >
            <div className="p-3 border-b border-slate-200 dark:border-slate-700">
              <h4 className="font-medium text-slate-900 dark:text-slate-100">Team Chat</h4>
            </div>
            
            <div className="h-48 overflow-y-auto p-3 space-y-2">
              {chatMessages.length === 0 ? (
                <div className="text-center text-slate-500 dark:text-slate-400 py-8">
                  <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <p className="text-sm">No messages yet</p>
                  <p className="text-xs mt-1">Start a conversation with your team</p>
                </div>
              ) : (
                chatMessages.map((message) => {
                  const user = userList.find(u => u.id === message.userId) || currentUser;
                  const isCurrentUser = message.userId === currentUser?.id;
                  
                  return (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${isCurrentUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-xs px-3 py-2 rounded-lg ${
                        isCurrentUser 
                          ? 'bg-blue-500 text-white' 
                          : 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100'
                      }`}>
                        {!isCurrentUser && (
                          <div className="text-xs font-medium mb-1 opacity-75">
                            {user?.name || 'Unknown User'}
                          </div>
                        )}
                        <div className="text-sm">{message.message}</div>
                        <div className={`text-xs mt-1 opacity-75 ${
                          isCurrentUser ? 'text-blue-100' : 'text-slate-500 dark:text-slate-400'
                        }`}>
                          {message.timestamp.toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </div>
                      </div>
                    </motion.div>
                  );
                })
              )}
            </div>
            
            <div className="p-3 border-t border-slate-200 dark:border-slate-700">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Type a message..."
                  className="flex-1 px-3 py-2 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={sendMessage}
                  disabled={!newMessage.trim()}
                  className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Room Info */}
      <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          Room Information
        </h4>
        <div className="space-y-1 text-xs text-slate-600 dark:text-slate-400">
          <div className="flex justify-between">
            <span>Room ID:</span>
            <span className="font-mono">room-{Date.now().toString().slice(-6)}</span>
          </div>
          <div className="flex justify-between">
            <span>Connection:</span>
            <span className="text-green-600 dark:text-green-400">Connected</span>
          </div>
          <div className="flex justify-between">
            <span>Sync Status:</span>
            <span className="text-green-600 dark:text-green-400">Real-time</span>
          </div>
        </div>
      </div>
    </div>
  );
}