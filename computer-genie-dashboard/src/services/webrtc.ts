// WebRTC service for real-time peer-to-peer collaboration
// Provides video/audio calls, screen sharing, and real-time data synchronization

import { apiService } from './api';
import { authService } from './auth';

export interface PeerConnection {
  id: string;
  userId: string;
  userName: string;
  avatar?: string;
  connection: RTCPeerConnection;
  dataChannel?: RTCDataChannel;
  localStream?: MediaStream;
  remoteStream?: MediaStream;
  isHost: boolean;
  status: 'connecting' | 'connected' | 'disconnected' | 'failed';
  lastSeen: number;
}

export interface CollaborationMessage {
  type: 'cursor' | 'selection' | 'edit' | 'chat' | 'workflow_update' | 'node_update' | 'edge_update';
  userId: string;
  userName: string;
  timestamp: number;
  data: any;
}

export interface MediaSettings {
  video: boolean;
  audio: boolean;
  screenShare: boolean;
  quality: 'low' | 'medium' | 'high';
}

export interface CollaborationState {
  sessionId: string | null;
  isHost: boolean;
  peers: Map<string, PeerConnection>;
  localStream: MediaStream | null;
  screenStream: MediaStream | null;
  mediaSettings: MediaSettings;
  isConnected: boolean;
  messages: CollaborationMessage[];
}

class WebRTCService {
  private sessionId: string | null = null;
  private isHost: boolean = false;
  private peers: Map<string, PeerConnection> = new Map();
  private localStream: MediaStream | null = null;
  private screenStream: MediaStream | null = null;
  private signalingSocket: WebSocket | null = null;
  private listeners: Map<string, Function[]> = new Map();

  private mediaSettings: MediaSettings = {
    video: true,
    audio: true,
    screenShare: false,
    quality: 'medium',
  };

  private rtcConfiguration: RTCConfiguration = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
      // In production, add TURN servers for better connectivity
      // { urls: 'turn:your-turn-server.com', username: 'user', credential: 'pass' }
    ],
    iceCandidatePoolSize: 10,
  };

  constructor() {
    this.setupSignalingConnection();
  }

  // Session management
  async createSession(workflowId: string): Promise<{ success: boolean; sessionId?: string; error?: string }> {
    try {
      const user = authService.getCurrentUser();
      if (!user) {
        return { success: false, error: 'Not authenticated' };
      }

      // Create collaboration session in database
      const response = await apiService.createCollaborationSession({
        workflowId,
        hostUserId: user.id,
        participants: [{
          userId: user.id,
          joinedAt: Date.now(),
          role: 'host',
        }],
        isActive: true,
      });

      if (!response.success || !response.data) {
        return { success: false, error: response.error || 'Failed to create session' };
      }

      this.sessionId = response.data.id;
      this.isHost = true;

      // Initialize media if needed
      if (this.mediaSettings.video || this.mediaSettings.audio) {
        await this.initializeLocalMedia();
      }

      // Setup signaling for this session
      this.setupSessionSignaling();

      this.emit('session:created', { sessionId: this.sessionId });
      return { success: true, sessionId: this.sessionId };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create session'
      };
    }
  }

  async joinSession(sessionId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const user = authService.getCurrentUser();
      if (!user) {
        return { success: false, error: 'Not authenticated' };
      }

      // Get session info from database
      const sessionResponse = await apiService.getCollaborationSessions();
      const session = sessionResponse.data?.find(s => s.id === sessionId);
      
      if (!session || !session.isActive) {
        return { success: false, error: 'Session not found or inactive' };
      }

      this.sessionId = sessionId;
      this.isHost = false;

      // Update session with new participant
      await apiService.updateCollaborationSession(sessionId, {
        participants: [
          ...session.participants,
          {
            userId: user.id,
            joinedAt: Date.now(),
            role: 'editor',
          }
        ]
      });

      // Initialize media if needed
      if (this.mediaSettings.video || this.mediaSettings.audio) {
        await this.initializeLocalMedia();
      }

      // Setup signaling for this session
      this.setupSessionSignaling();

      // Connect to existing peers
      await this.connectToExistingPeers();

      this.emit('session:joined', { sessionId });
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to join session'
      };
    }
  }

  async leaveSession(): Promise<void> {
    try {
      if (this.sessionId) {
        // Notify other peers
        this.broadcastMessage({
          type: 'chat',
          userId: authService.getCurrentUser()?.id || '',
          userName: authService.getCurrentUser()?.name || '',
          timestamp: Date.now(),
          data: { message: 'Left the session', type: 'system' }
        });

        // Update session in database
        if (this.isHost) {
          await apiService.updateCollaborationSession(this.sessionId, {
            isActive: false,
            endedAt: Date.now(),
          });
        }
      }

      // Close all peer connections
      this.peers.forEach(peer => {
        peer.connection.close();
        if (peer.dataChannel) {
          peer.dataChannel.close();
        }
      });
      this.peers.clear();

      // Stop local media
      if (this.localStream) {
        this.localStream.getTracks().forEach(track => track.stop());
        this.localStream = null;
      }

      if (this.screenStream) {
        this.screenStream.getTracks().forEach(track => track.stop());
        this.screenStream = null;
      }

      // Close signaling connection
      if (this.signalingSocket) {
        this.signalingSocket.close();
        this.signalingSocket = null;
      }

      this.sessionId = null;
      this.isHost = false;

      this.emit('session:left', {});
    } catch (error) {
      console.error('Error leaving session:', error);
    }
  }

  // Media management
  async initializeLocalMedia(): Promise<MediaStream | null> {
    try {
      const constraints: MediaStreamConstraints = {
        video: this.mediaSettings.video ? this.getVideoConstraints() : false,
        audio: this.mediaSettings.audio ? this.getAudioConstraints() : false,
      };

      this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Add tracks to existing peer connections
      this.peers.forEach(peer => {
        if (this.localStream) {
          this.localStream.getTracks().forEach(track => {
            peer.connection.addTrack(track, this.localStream!);
          });
        }
      });

      this.emit('media:local_stream', { stream: this.localStream });
      return this.localStream;
    } catch (error) {
      console.error('Failed to initialize local media:', error);
      this.emit('media:error', { error: 'Failed to access camera/microphone' });
      return null;
    }
  }

  async startScreenShare(): Promise<MediaStream | null> {
    try {
      this.screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
      });

      // Replace video track in peer connections
      this.peers.forEach(peer => {
        const videoSender = peer.connection.getSenders().find(
          sender => sender.track?.kind === 'video'
        );
        
        if (videoSender && this.screenStream) {
          const videoTrack = this.screenStream.getVideoTracks()[0];
          videoSender.replaceTrack(videoTrack);
        }
      });

      // Handle screen share end
      this.screenStream.getVideoTracks()[0].addEventListener('ended', () => {
        this.stopScreenShare();
      });

      this.mediaSettings.screenShare = true;
      this.emit('media:screen_share_started', { stream: this.screenStream });
      return this.screenStream;
    } catch (error) {
      console.error('Failed to start screen share:', error);
      this.emit('media:error', { error: 'Failed to start screen sharing' });
      return null;
    }
  }

  async stopScreenShare(): Promise<void> {
    if (this.screenStream) {
      this.screenStream.getTracks().forEach(track => track.stop());
      this.screenStream = null;
    }

    // Restore camera video
    if (this.localStream && this.mediaSettings.video) {
      this.peers.forEach(peer => {
        const videoSender = peer.connection.getSenders().find(
          sender => sender.track?.kind === 'video'
        );
        
        if (videoSender && this.localStream) {
          const videoTrack = this.localStream.getVideoTracks()[0];
          if (videoTrack) {
            videoSender.replaceTrack(videoTrack);
          }
        }
      });
    }

    this.mediaSettings.screenShare = false;
    this.emit('media:screen_share_stopped', {});
  }

  async toggleVideo(): Promise<void> {
    if (this.localStream) {
      const videoTrack = this.localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        this.mediaSettings.video = videoTrack.enabled;
        this.emit('media:video_toggled', { enabled: videoTrack.enabled });
      }
    }
  }

  async toggleAudio(): Promise<void> {
    if (this.localStream) {
      const audioTrack = this.localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        this.mediaSettings.audio = audioTrack.enabled;
        this.emit('media:audio_toggled', { enabled: audioTrack.enabled });
      }
    }
  }

  // Peer connection management
  private async createPeerConnection(peerId: string, isInitiator: boolean): Promise<PeerConnection> {
    const connection = new RTCPeerConnection(this.rtcConfiguration);
    const user = authService.getCurrentUser();

    const peer: PeerConnection = {
      id: peerId,
      userId: peerId,
      userName: `User ${peerId.slice(0, 8)}`,
      connection,
      isHost: false,
      status: 'connecting',
      lastSeen: Date.now(),
    };

    // Add local stream tracks
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => {
        connection.addTrack(track, this.localStream!);
      });
    }

    // Handle remote stream
    connection.ontrack = (event) => {
      peer.remoteStream = event.streams[0];
      this.emit('peer:stream', { peerId, stream: event.streams[0] });
    };

    // Handle ICE candidates
    connection.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignalingMessage({
          type: 'ice-candidate',
          targetPeer: peerId,
          candidate: event.candidate,
        });
      }
    };

    // Handle connection state changes
    connection.onconnectionstatechange = () => {
      peer.status = this.mapConnectionState(connection.connectionState);
      peer.lastSeen = Date.now();
      this.emit('peer:status_changed', { peerId, status: peer.status });

      if (connection.connectionState === 'failed' || connection.connectionState === 'disconnected') {
        this.removePeer(peerId);
      }
    };

    // Create data channel for real-time collaboration
    if (isInitiator) {
      peer.dataChannel = connection.createDataChannel('collaboration', {
        ordered: true,
      });
      this.setupDataChannel(peer.dataChannel, peerId);
    } else {
      connection.ondatachannel = (event) => {
        peer.dataChannel = event.channel;
        this.setupDataChannel(peer.dataChannel, peerId);
      };
    }

    this.peers.set(peerId, peer);
    return peer;
  }

  private setupDataChannel(dataChannel: RTCDataChannel, peerId: string): void {
    dataChannel.onopen = () => {
      console.log(`Data channel opened with peer ${peerId}`);
      this.emit('peer:data_channel_open', { peerId });
    };

    dataChannel.onmessage = (event) => {
      try {
        const message: CollaborationMessage = JSON.parse(event.data);
        this.handleCollaborationMessage(message);
      } catch (error) {
        console.error('Failed to parse collaboration message:', error);
      }
    };

    dataChannel.onclose = () => {
      console.log(`Data channel closed with peer ${peerId}`);
      this.emit('peer:data_channel_close', { peerId });
    };

    dataChannel.onerror = (error) => {
      console.error(`Data channel error with peer ${peerId}:`, error);
      this.emit('peer:data_channel_error', { peerId, error });
    };
  }

  private async connectToExistingPeers(): Promise<void> {
    // In a real implementation, get list of existing peers from signaling server
    // For now, this is a placeholder
  }

  private removePeer(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (peer) {
      peer.connection.close();
      if (peer.dataChannel) {
        peer.dataChannel.close();
      }
      this.peers.delete(peerId);
      this.emit('peer:disconnected', { peerId });
    }
  }

  // Signaling
  private setupSignalingConnection(): void {
    // In a real implementation, connect to a signaling server (WebSocket)
    // For now, we'll simulate signaling
    console.log('Setting up signaling connection...');
  }

  private setupSessionSignaling(): void {
    if (!this.sessionId) return;

    // Setup WebSocket connection for signaling
    const wsUrl = `ws://localhost:8080/signaling/${this.sessionId}`;
    
    try {
      this.signalingSocket = new WebSocket(wsUrl);
      
      this.signalingSocket.onopen = () => {
        console.log('Signaling connection established');
        this.emit('signaling:connected', {});
      };

      this.signalingSocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleSignalingMessage(message);
        } catch (error) {
          console.error('Failed to parse signaling message:', error);
        }
      };

      this.signalingSocket.onclose = () => {
        console.log('Signaling connection closed');
        this.emit('signaling:disconnected', {});
      };

      this.signalingSocket.onerror = (error) => {
        console.error('Signaling error:', error);
        this.emit('signaling:error', { error });
      };
    } catch (error) {
      console.error('Failed to setup signaling:', error);
      // Fallback to simulated signaling for demo purposes
      this.setupSimulatedSignaling();
    }
  }

  private setupSimulatedSignaling(): void {
    // Simulate signaling for demo purposes
    console.log('Using simulated signaling for demo');
    this.emit('signaling:connected', {});
  }

  private sendSignalingMessage(message: any): void {
    if (this.signalingSocket && this.signalingSocket.readyState === WebSocket.OPEN) {
      this.signalingSocket.send(JSON.stringify(message));
    } else {
      // Fallback for demo
      console.log('Signaling message (simulated):', message);
    }
  }

  private async handleSignalingMessage(message: any): Promise<void> {
    switch (message.type) {
      case 'peer-joined':
        await this.handlePeerJoined(message.peerId);
        break;
      case 'peer-left':
        this.removePeer(message.peerId);
        break;
      case 'offer':
        await this.handleOffer(message.peerId, message.offer);
        break;
      case 'answer':
        await this.handleAnswer(message.peerId, message.answer);
        break;
      case 'ice-candidate':
        await this.handleIceCandidate(message.peerId, message.candidate);
        break;
    }
  }

  private async handlePeerJoined(peerId: string): Promise<void> {
    if (this.isHost) {
      // Create offer for new peer
      const peer = await this.createPeerConnection(peerId, true);
      const offer = await peer.connection.createOffer();
      await peer.connection.setLocalDescription(offer);
      
      this.sendSignalingMessage({
        type: 'offer',
        targetPeer: peerId,
        offer,
      });
    }
  }

  private async handleOffer(peerId: string, offer: RTCSessionDescriptionInit): Promise<void> {
    const peer = await this.createPeerConnection(peerId, false);
    await peer.connection.setRemoteDescription(offer);
    
    const answer = await peer.connection.createAnswer();
    await peer.connection.setLocalDescription(answer);
    
    this.sendSignalingMessage({
      type: 'answer',
      targetPeer: peerId,
      answer,
    });
  }

  private async handleAnswer(peerId: string, answer: RTCSessionDescriptionInit): Promise<void> {
    const peer = this.peers.get(peerId);
    if (peer) {
      await peer.connection.setRemoteDescription(answer);
    }
  }

  private async handleIceCandidate(peerId: string, candidate: RTCIceCandidateInit): Promise<void> {
    const peer = this.peers.get(peerId);
    if (peer) {
      await peer.connection.addIceCandidate(candidate);
    }
  }

  // Collaboration messaging
  sendCollaborationMessage(message: Omit<CollaborationMessage, 'userId' | 'userName' | 'timestamp'>): void {
    const user = authService.getCurrentUser();
    if (!user) return;

    const fullMessage: CollaborationMessage = {
      ...message,
      userId: user.id,
      userName: user.name,
      timestamp: Date.now(),
    };

    this.broadcastMessage(fullMessage);
    this.handleCollaborationMessage(fullMessage);
  }

  private broadcastMessage(message: CollaborationMessage): void {
    const messageStr = JSON.stringify(message);
    
    this.peers.forEach(peer => {
      if (peer.dataChannel && peer.dataChannel.readyState === 'open') {
        try {
          peer.dataChannel.send(messageStr);
        } catch (error) {
          console.error(`Failed to send message to peer ${peer.id}:`, error);
        }
      }
    });
  }

  private handleCollaborationMessage(message: CollaborationMessage): void {
    this.emit('collaboration:message', message);

    // Handle specific message types
    switch (message.type) {
      case 'cursor':
        this.emit('collaboration:cursor', message);
        break;
      case 'selection':
        this.emit('collaboration:selection', message);
        break;
      case 'edit':
        this.emit('collaboration:edit', message);
        break;
      case 'chat':
        this.emit('collaboration:chat', message);
        break;
      case 'workflow_update':
        this.emit('collaboration:workflow_update', message);
        break;
      case 'node_update':
        this.emit('collaboration:node_update', message);
        break;
      case 'edge_update':
        this.emit('collaboration:edge_update', message);
        break;
    }
  }

  // Utility methods
  private getVideoConstraints(): MediaTrackConstraints {
    const quality = this.mediaSettings.quality;
    
    switch (quality) {
      case 'low':
        return { width: 320, height: 240, frameRate: 15 };
      case 'medium':
        return { width: 640, height: 480, frameRate: 30 };
      case 'high':
        return { width: 1280, height: 720, frameRate: 30 };
      default:
        return { width: 640, height: 480, frameRate: 30 };
    }
  }

  private getAudioConstraints(): MediaTrackConstraints {
    return {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };
  }

  private mapConnectionState(state: RTCPeerConnectionState): PeerConnection['status'] {
    switch (state) {
      case 'connecting':
        return 'connecting';
      case 'connected':
        return 'connected';
      case 'disconnected':
        return 'disconnected';
      case 'failed':
        return 'failed';
      default:
        return 'connecting';
    }
  }

  // Getters
  getCollaborationState(): CollaborationState {
    return {
      sessionId: this.sessionId,
      isHost: this.isHost,
      peers: this.peers,
      localStream: this.localStream,
      screenStream: this.screenStream,
      mediaSettings: this.mediaSettings,
      isConnected: this.sessionId !== null,
      messages: [], // In a real implementation, maintain message history
    };
  }

  getPeers(): PeerConnection[] {
    return Array.from(this.peers.values());
  }

  getMediaSettings(): MediaSettings {
    return { ...this.mediaSettings };
  }

  updateMediaSettings(settings: Partial<MediaSettings>): void {
    this.mediaSettings = { ...this.mediaSettings, ...settings };
    this.emit('media:settings_updated', this.mediaSettings);
  }

  // Event system
  subscribe(event: string, callback: Function): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);

    return () => {
      const callbacks = this.listeners.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebRTC event callback for ${event}:`, error);
        }
      });
    }
  }
}

// Export singleton instance
export const webrtcService = new WebRTCService();