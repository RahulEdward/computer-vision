'use client';

import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  VideoCameraIcon,
  MicrophoneIcon,
  PhoneXMarkIcon,
  UserGroupIcon,
  ChatBubbleLeftRightIcon,
  ShareIcon,
  CursorArrowRaysIcon,
} from '@heroicons/react/24/outline';

interface User {
  id: string;
  name: string;
  avatar: string;
  isOnline: boolean;
  cursor?: { x: number; y: number };
}

interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  message: string;
  timestamp: Date;
}

const CollaborationPanel = () => {
  const [isVideoEnabled, setIsVideoEnabled] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [users, setUsers] = useState<User[]>([
    { id: '1', name: 'John Doe', avatar: '👨‍💻', isOnline: true, cursor: { x: 100, y: 150 } },
    { id: '2', name: 'Jane Smith', avatar: '👩‍💼', isOnline: true, cursor: { x: 200, y: 300 } },
    { id: '3', name: 'Mike Johnson', avatar: '👨‍🎨', isOnline: false },
  ]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      userId: '1',
      userName: 'John Doe',
      message: 'Hey everyone! Ready to work on the new workflow?',
      timestamp: new Date(Date.now() - 300000),
    },
    {
      id: '2',
      userId: '2',
      userName: 'Jane Smith',
      message: 'Yes! I have some ideas for the API integration part.',
      timestamp: new Date(Date.now() - 180000),
    },
  ]);
  const [newMessage, setNewMessage] = useState('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const startVideoCall = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: true 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsVideoEnabled(true);
      setIsAudioEnabled(true);
      setIsConnected(true);
    } catch (error) {
      console.error('Error accessing media devices:', error);
    }
  };

  const endCall = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
    setIsVideoEnabled(false);
    setIsAudioEnabled(false);
    setIsConnected(false);
  };

  const toggleVideo = () => {
    setIsVideoEnabled(!isVideoEnabled);
  };

  const toggleAudio = () => {
    setIsAudioEnabled(!isAudioEnabled);
  };

  const sendMessage = () => {
    if (newMessage.trim()) {
      const message: ChatMessage = {
        id: Date.now().toString(),
        userId: 'current-user',
        userName: 'You',
        message: newMessage,
        timestamp: new Date(),
      };
      setChatMessages([...chatMessages, message]);
      setNewMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="space-y-6">
      {/* Collaboration Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-xl p-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <UserGroupIcon className="h-8 w-8 text-purple-400" />
            <div>
              <h2 className="text-xl font-semibold text-white">Collaboration Space</h2>
              <p className="text-gray-300">Real-time collaboration with your team</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isConnected ? endCall : startVideoCall}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isConnected
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isConnected ? (
                <>
                  <PhoneXMarkIcon className="h-4 w-4" />
                  <span>End Call</span>
                </>
              ) : (
                <>
                  <VideoCameraIcon className="h-4 w-4" />
                  <span>Start Call</span>
                </>
              )}
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-all duration-200"
            >
              <ShareIcon className="h-4 w-4" />
              <span>Share Screen</span>
            </motion.button>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Call Area */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2 bg-black/20 backdrop-blur-xl border border-white/10 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Video Conference</h3>
            {isConnected && (
              <div className="flex items-center space-x-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={toggleVideo}
                  className={`p-2 rounded-lg transition-all duration-200 ${
                    isVideoEnabled
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-gray-600 hover:bg-gray-700 text-white'
                  }`}
                >
                  <VideoCameraIcon className="h-5 w-5" />
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={toggleAudio}
                  className={`p-2 rounded-lg transition-all duration-200 ${
                    isAudioEnabled
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-gray-600 hover:bg-gray-700 text-white'
                  }`}
                >
                  <MicrophoneIcon className="h-5 w-5" />
                </motion.button>
              </div>
            )}
          </div>
          
          <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ height: '400px' }}>
            {isConnected ? (
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <VideoCameraIcon className="h-16 w-16 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">Click "Start Call" to begin video conference</p>
                </div>
              </div>
            )}
            
            {/* Cursor indicators for other users */}
            {users.filter(user => user.isOnline && user.cursor).map(user => (
              <motion.div
                key={user.id}
                className="absolute pointer-events-none"
                style={{ left: user.cursor!.x, top: user.cursor!.y }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
              >
                <CursorArrowRaysIcon className="h-6 w-6 text-purple-400" />
                <div className="bg-purple-600 text-white text-xs px-2 py-1 rounded mt-1">
                  {user.name}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Chat and Users Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-6"
        >
          {/* Online Users */}
          <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Online Users</h3>
            <div className="space-y-3">
              {users.map(user => (
                <div key={user.id} className="flex items-center space-x-3">
                  <div className="relative">
                    <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white">
                      {user.avatar}
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-gray-900 ${
                      user.isOnline ? 'bg-green-400' : 'bg-gray-500'
                    }`} />
                  </div>
                  <div className="flex-1">
                    <p className="text-white text-sm font-medium">{user.name}</p>
                    <p className="text-gray-400 text-xs">
                      {user.isOnline ? 'Online' : 'Offline'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-4">
              <ChatBubbleLeftRightIcon className="h-5 w-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">Team Chat</h3>
            </div>
            
            <div className="space-y-3 mb-4 max-h-64 overflow-y-auto">
              {chatMessages.map(message => (
                <div key={message.id} className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-purple-400 text-sm font-medium">
                      {message.userName}
                    </span>
                    <span className="text-gray-500 text-xs">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm">{message.message}</p>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            
            <div className="flex space-x-2">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message..."
                className="flex-1 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={sendMessage}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-all duration-200"
              >
                Send
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Collaboration Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Collaboration Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 p-4 rounded-lg border border-blue-500/30">
            <h4 className="text-white font-medium mb-2">Real-time Cursors</h4>
            <p className="text-gray-300 text-sm">See where your teammates are working in real-time</p>
          </div>
          
          <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 p-4 rounded-lg border border-green-500/30">
            <h4 className="text-white font-medium mb-2">Live Editing</h4>
            <p className="text-gray-300 text-sm">Collaborate on workflows simultaneously</p>
          </div>
          
          <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-4 rounded-lg border border-purple-500/30">
            <h4 className="text-white font-medium mb-2">Voice & Video</h4>
            <p className="text-gray-300 text-sm">High-quality WebRTC communication</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default CollaborationPanel;