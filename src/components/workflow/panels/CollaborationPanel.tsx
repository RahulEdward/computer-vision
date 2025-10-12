import React, { useState, useCallback, useMemo } from 'react';
import {
  Users,
  MessageSquare,
  Video,
  Phone,
  Share2,
  UserPlus,
  UserMinus,
  Crown,
  Eye,
  Edit3,
  Lock,
  Unlock,
  Bell,
  BellOff,
  Send,
  Smile,
  Paperclip,
  MoreHorizontal,
  Circle,
  CheckCircle,
  AlertCircle,
  Clock,
  Calendar,
  Settings,
  X,
  ChevronDown,
  ChevronUp,
  Search,
  Filter,
  Star,
  Pin,
  Reply,
  Forward,
  Copy,
  Trash2,
  Flag,
  VolumeX as Mute,
  Volume2,
  Camera,
  Mic,
  MicOff,
  VideoOff,
  ScreenShare,
  StopCircle
} from 'lucide-react';

export interface Collaborator {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: 'owner' | 'editor' | 'viewer' | 'commenter';
  status: 'online' | 'away' | 'busy' | 'offline';
  lastSeen: number;
  permissions: {
    canEdit: boolean;
    canComment: boolean;
    canShare: boolean;
    canManageUsers: boolean;
  };
  cursor?: {
    x: number;
    y: number;
    nodeId?: string;
  };
  selection?: {
    nodeIds: string[];
    edgeIds: string[];
  };
  color: string;
}

export interface ChatMessage {
  id: string;
  userId: string;
  content: string;
  timestamp: number;
  type: 'text' | 'system' | 'file' | 'node_reference' | 'workflow_change';
  replyTo?: string;
  reactions?: Array<{
    emoji: string;
    userIds: string[];
  }>;
  attachments?: Array<{
    id: string;
    name: string;
    type: string;
    size: number;
    url: string;
  }>;
  nodeReference?: {
    nodeId: string;
    nodeName: string;
  };
  isEdited?: boolean;
  isPinned?: boolean;
}

export interface VideoCall {
  id: string;
  participants: string[];
  isActive: boolean;
  startTime: number;
  settings: {
    audioEnabled: boolean;
    videoEnabled: boolean;
    screenShareEnabled: boolean;
    recordingEnabled: boolean;
  };
}

interface CollaborationPanelProps {
  collaborators: Collaborator[];
  messages: ChatMessage[];
  videoCall?: VideoCall;
  currentUserId: string;
  workflowId: string;
  onSendMessage: (content: string, type?: ChatMessage['type'], replyTo?: string) => void;
  onInviteUser: (email: string, role: Collaborator['role']) => void;
  onRemoveUser: (userId: string) => void;
  onChangeUserRole: (userId: string, role: Collaborator['role']) => void;
  onStartVideoCall: () => void;
  onJoinVideoCall: () => void;
  onLeaveVideoCall: () => void;
  onToggleAudio: () => void;
  onToggleVideo: () => void;
  onToggleScreenShare: () => void;
  onReactToMessage: (messageId: string, emoji: string) => void;
  onPinMessage: (messageId: string) => void;
  onDeleteMessage: (messageId: string) => void;
  onEditMessage: (messageId: string, content: string) => void;
  className?: string;
}

export const CollaborationPanel: React.FC<CollaborationPanelProps> = ({
  collaborators,
  messages,
  videoCall,
  currentUserId,
  workflowId,
  onSendMessage,
  onInviteUser,
  onRemoveUser,
  onChangeUserRole,
  onStartVideoCall,
  onJoinVideoCall,
  onLeaveVideoCall,
  onToggleAudio,
  onToggleVideo,
  onToggleScreenShare,
  onReactToMessage,
  onPinMessage,
  onDeleteMessage,
  onEditMessage,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'chat' | 'users' | 'activity'>('chat');
  const [messageInput, setMessageInput] = useState('');
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [editingMessage, setEditingMessage] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<Collaborator['role']>('viewer');
  const [searchQuery, setSearchQuery] = useState('');
  const [showEmojiPicker, setShowEmojiPicker] = useState<string | null>(null);

  const currentUser = collaborators.find(c => c.id === currentUserId);
  const onlineUsers = collaborators.filter(c => c.status === 'online');
  
  const filteredMessages = useMemo(() => {
    if (!searchQuery) return messages;
    return messages.filter(m => 
      m.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      collaborators.find(c => c.id === m.userId)?.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [messages, searchQuery, collaborators]);

  const pinnedMessages = messages.filter(m => m.isPinned);

  const handleSendMessage = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!messageInput.trim()) return;

    onSendMessage(messageInput, 'text', replyingTo || undefined);
    setMessageInput('');
    setReplyingTo(null);
  }, [messageInput, replyingTo, onSendMessage]);

  const handleInviteUser = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;

    onInviteUser(inviteEmail, inviteRole);
    setInviteEmail('');
    setShowInviteDialog(false);
  }, [inviteEmail, inviteRole, onInviteUser]);

  const handleEditMessage = useCallback((messageId: string, content: string) => {
    setEditingMessage(messageId);
    setEditContent(content);
  }, []);

  const handleSaveEdit = useCallback(() => {
    if (!editingMessage || !editContent.trim()) return;
    
    onEditMessage(editingMessage, editContent);
    setEditingMessage(null);
    setEditContent('');
  }, [editingMessage, editContent, onEditMessage]);

  const formatTime = (timestamp: number): string => {
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  const getStatusColor = (status: Collaborator['status']): string => {
    switch (status) {
      case 'online': return 'bg-green-400';
      case 'away': return 'bg-yellow-400';
      case 'busy': return 'bg-red-400';
      default: return 'bg-gray-400';
    }
  };

  const getRoleIcon = (role: Collaborator['role']): React.ReactNode => {
    switch (role) {
      case 'owner': return <Crown className="w-3 h-3 text-yellow-400" />;
      case 'editor': return <Edit3 className="w-3 h-3 text-blue-400" />;
      case 'viewer': return <Eye className="w-3 h-3 text-gray-400" />;
      case 'commenter': return <MessageSquare className="w-3 h-3 text-green-400" />;
    }
  };

  const renderMessage = (message: ChatMessage) => {
    const author = collaborators.find(c => c.id === message.userId);
    const isOwn = message.userId === currentUserId;
    const replyMessage = message.replyTo ? messages.find(m => m.id === message.replyTo) : null;
    const isEditing = editingMessage === message.id;

    return (
      <div key={message.id} className={`group relative ${isOwn ? 'ml-8' : 'mr-8'}`}>
        {message.isPinned && (
          <div className="flex items-center space-x-1 text-xs text-yellow-400 mb-1">
            <Pin className="w-3 h-3" />
            <span>Pinned message</span>
          </div>
        )}
        
        {replyMessage && (
          <div className="ml-4 mb-1 pl-2 border-l-2 border-gray-600 text-xs text-gray-400">
            <div className="flex items-center space-x-1">
              <Reply className="w-3 h-3" />
              <span>Replying to {collaborators.find(c => c.id === replyMessage.userId)?.name}</span>
            </div>
            <div className="truncate">{replyMessage.content}</div>
          </div>
        )}

        <div className={`flex items-start space-x-2 ${isOwn ? 'flex-row-reverse space-x-reverse' : ''}`}>
          <div className="relative">
            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white text-sm font-medium">
              {author?.avatar ? (
                <img src={author.avatar} alt={author.name} className="w-8 h-8 rounded-full" />
              ) : (
                author?.name.charAt(0).toUpperCase()
              )}
            </div>
            <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-gray-900 ${getStatusColor(author?.status || 'offline')}`} />
          </div>

          <div className={`flex-1 ${isOwn ? 'text-right' : ''}`}>
            <div className="flex items-center space-x-2 mb-1">
              <span className="text-sm font-medium text-white">{author?.name}</span>
              {getRoleIcon(author?.role || 'viewer')}
              <span className="text-xs text-gray-400">{formatTime(message.timestamp)}</span>
              {message.isEdited && (
                <span className="text-xs text-gray-500">(edited)</span>
              )}
            </div>

            <div className={`inline-block max-w-xs lg:max-w-md px-3 py-2 rounded-lg ${
              isOwn 
                ? 'bg-blue-600 text-white' 
                : message.type === 'system' 
                  ? 'bg-gray-700 text-gray-300'
                  : 'bg-gray-800 text-white'
            }`}>
              {isEditing ? (
                <div className="space-y-2">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white text-sm resize-none"
                    rows={2}
                  />
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleSaveEdit}
                      className="text-xs text-green-400 hover:text-green-300"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingMessage(null)}
                      className="text-xs text-gray-400 hover:text-gray-300"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-sm">{message.content}</p>
                  
                  {message.nodeReference && (
                    <div className="mt-2 p-2 bg-gray-700 rounded text-xs">
                      <div className="flex items-center space-x-1">
                        <Circle className="w-3 h-3 text-blue-400" />
                        <span>Referenced node: {message.nodeReference.nodeName}</span>
                      </div>
                    </div>
                  )}

                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {message.attachments.map(attachment => (
                        <div key={attachment.id} className="flex items-center space-x-2 p-2 bg-gray-700 rounded text-xs">
                          <Paperclip className="w-3 h-3" />
                          <span>{attachment.name}</span>
                          <span className="text-gray-400">({(attachment.size / 1024).toFixed(1)} KB)</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {message.reactions && message.reactions.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {message.reactions.map((reaction, index) => (
                        <button
                          key={index}
                          onClick={() => onReactToMessage(message.id, reaction.emoji)}
                          className="flex items-center space-x-1 px-2 py-1 bg-gray-700 rounded-full text-xs hover:bg-gray-600"
                        >
                          <span>{reaction.emoji}</span>
                          <span>{reaction.userIds.length}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Message Actions */}
            <div className={`mt-1 opacity-0 group-hover:opacity-100 transition-opacity ${isOwn ? 'text-right' : ''}`}>
              <div className="flex items-center space-x-1 text-xs">
                <button
                  onClick={() => setReplyingTo(message.id)}
                  className="text-gray-400 hover:text-white"
                  title="Reply"
                >
                  <Reply className="w-3 h-3" />
                </button>
                
                <button
                  onClick={() => setShowEmojiPicker(message.id)}
                  className="text-gray-400 hover:text-white"
                  title="React"
                >
                  <Smile className="w-3 h-3" />
                </button>

                {currentUser?.permissions.canEdit && (
                  <button
                    onClick={() => onPinMessage(message.id)}
                    className={`hover:text-white ${message.isPinned ? 'text-yellow-400' : 'text-gray-400'}`}
                    title="Pin message"
                  >
                    <Pin className="w-3 h-3" />
                  </button>
                )}

                {isOwn && (
                  <>
                    <button
                      onClick={() => handleEditMessage(message.id, message.content)}
                      className="text-gray-400 hover:text-white"
                      title="Edit"
                    >
                      <Edit3 className="w-3 h-3" />
                    </button>
                    
                    <button
                      onClick={() => onDeleteMessage(message.id)}
                      className="text-gray-400 hover:text-red-400"
                      title="Delete"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`bg-gray-900 border-l border-gray-700 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-white">Collaboration</h3>
          
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1 text-sm text-gray-400">
              <Circle className={`w-2 h-2 ${onlineUsers.length > 0 ? 'text-green-400' : 'text-gray-400'}`} />
              <span>{onlineUsers.length} online</span>
            </div>
            
            {!videoCall?.isActive ? (
              <button
                onClick={onStartVideoCall}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
                title="Start video call"
              >
                <Video className="w-4 h-4" />
              </button>
            ) : (
              <div className="flex items-center space-x-1">
                <button
                  onClick={onToggleAudio}
                  className={`p-1 rounded transition-colors ${
                    videoCall.settings.audioEnabled 
                      ? 'text-green-400 hover:text-green-300' 
                      : 'text-red-400 hover:text-red-300'
                  }`}
                  title="Toggle audio"
                >
                  {videoCall.settings.audioEnabled ? <Mic className="w-3 h-3" /> : <MicOff className="w-3 h-3" />}
                </button>
                
                <button
                  onClick={onToggleVideo}
                  className={`p-1 rounded transition-colors ${
                    videoCall.settings.videoEnabled 
                      ? 'text-green-400 hover:text-green-300' 
                      : 'text-red-400 hover:text-red-300'
                  }`}
                  title="Toggle video"
                >
                  {videoCall.settings.videoEnabled ? <Camera className="w-3 h-3" /> : <VideoOff className="w-3 h-3" />}
                </button>
                
                <button
                  onClick={onLeaveVideoCall}
                  className="p-1 text-red-400 hover:text-red-300 rounded transition-colors"
                  title="Leave call"
                >
                  <StopCircle className="w-3 h-3" />
                </button>
              </div>
            )}
            
            <button
              onClick={() => setShowInviteDialog(true)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Invite user"
            >
              <UserPlus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 bg-gray-800 rounded-lg p-1">
          {(['chat', 'users', 'activity'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-3 py-2 text-sm font-medium rounded transition-colors ${
                activeTab === tab
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'chat' && (
          <div className="h-full flex flex-col">
            {/* Search */}
            <div className="p-3 border-b border-gray-700">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search messages..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:border-blue-400"
                />
              </div>
            </div>

            {/* Pinned Messages */}
            {pinnedMessages.length > 0 && (
              <div className="p-3 border-b border-gray-700 bg-gray-800/50">
                <div className="text-xs font-medium text-yellow-400 mb-2 flex items-center space-x-1">
                  <Pin className="w-3 h-3" />
                  <span>Pinned Messages</span>
                </div>
                <div className="space-y-2 max-h-20 overflow-y-auto">
                  {pinnedMessages.slice(0, 2).map(message => (
                    <div key={message.id} className="text-xs text-gray-300 truncate">
                      <span className="font-medium">{collaborators.find(c => c.id === message.userId)?.name}:</span>
                      <span className="ml-1">{message.content}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-4">
              {filteredMessages.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <MessageSquare className="w-8 h-8 mx-auto mb-2" />
                  <p>No messages yet</p>
                  <p className="text-sm">Start a conversation with your team</p>
                </div>
              ) : (
                filteredMessages.map(renderMessage)
              )}
            </div>

            {/* Reply indicator */}
            {replyingTo && (
              <div className="px-3 py-2 bg-gray-800 border-t border-gray-700">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2 text-gray-400">
                    <Reply className="w-4 h-4" />
                    <span>Replying to {collaborators.find(c => c.id === messages.find(m => m.id === replyingTo)?.userId)?.name}</span>
                  </div>
                  <button
                    onClick={() => setReplyingTo(null)}
                    className="text-gray-400 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Message Input */}
            <div className="p-3 border-t border-gray-700">
              <form onSubmit={handleSendMessage} className="flex items-center space-x-2">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={messageInput}
                    onChange={(e) => setMessageInput(e.target.value)}
                    placeholder="Type a message..."
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-400"
                  />
                  <button
                    type="button"
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                  >
                    <Paperclip className="w-4 h-4" />
                  </button>
                </div>
                
                <button
                  type="submit"
                  disabled={!messageInput.trim()}
                  className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </div>
        )}

        {activeTab === 'users' && (
          <div className="p-3 space-y-3">
            {collaborators.map(user => (
              <div key={user.id} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center text-white font-medium">
                      {user.avatar ? (
                        <img src={user.avatar} alt={user.name} className="w-10 h-10 rounded-full" />
                      ) : (
                        user.name.charAt(0).toUpperCase()
                      )}
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-gray-900 ${getStatusColor(user.status)}`} />
                  </div>
                  
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-white">{user.name}</span>
                      {getRoleIcon(user.role)}
                    </div>
                    <div className="text-sm text-gray-400">{user.email}</div>
                    <div className="text-xs text-gray-500">
                      {user.status === 'online' ? 'Online' : `Last seen ${formatTime(user.lastSeen)}`}
                    </div>
                  </div>
                </div>

                {currentUser?.permissions.canManageUsers && user.id !== currentUserId && (
                  <div className="flex items-center space-x-1">
                    <select
                      value={user.role}
                      onChange={(e) => onChangeUserRole(user.id, e.target.value as Collaborator['role'])}
                      className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm focus:outline-none focus:border-blue-400"
                    >
                      <option value="viewer">Viewer</option>
                      <option value="commenter">Commenter</option>
                      <option value="editor">Editor</option>
                      <option value="owner">Owner</option>
                    </select>
                    
                    <button
                      onClick={() => onRemoveUser(user.id)}
                      className="p-1 text-red-400 hover:text-red-300"
                      title="Remove user"
                    >
                      <UserMinus className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'activity' && (
          <div className="p-3 space-y-3">
            <div className="text-center py-8 text-gray-500">
              <Clock className="w-8 h-8 mx-auto mb-2" />
              <p>Activity feed coming soon</p>
              <p className="text-sm">Track workflow changes and user actions</p>
            </div>
          </div>
        )}
      </div>

      {/* Invite Dialog */}
      {showInviteDialog && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-96 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">Invite User</h4>
              <button
                onClick={() => setShowInviteDialog(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleInviteUser} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:border-blue-400"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Role
                </label>
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value as Collaborator['role'])}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-blue-400"
                >
                  <option value="viewer">Viewer - Can view workflow</option>
                  <option value="commenter">Commenter - Can view and comment</option>
                  <option value="editor">Editor - Can edit workflow</option>
                  <option value="owner">Owner - Full access</option>
                </select>
              </div>

              <div className="flex items-center justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setShowInviteDialog(false)}
                  className="px-4 py-2 text-gray-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                >
                  Send Invite
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};