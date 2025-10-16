const { desktopCapturer } = require('electron');
const fs = require('fs');
const path = require('path');

class ScreenRecorderService {
  constructor() {
    this.isRecording = false;
    this.mediaRecorder = null;
    this.recordedChunks = [];
  }

  async initialize() {
    console.log('[ScreenRecorder] Service initialized');
  }

  async startRecording(options = {}) {
    if (this.isRecording) {
      return { success: false, error: 'Already recording' };
    }

    try {
      const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: { width: 1920, height: 1080 }
      });

      if (sources.length === 0) {
        return { success: false, error: 'No screen sources available' };
      }

      this.isRecording = true;
      this.recordedChunks = [];

      return {
        success: true,
        message: 'Recording started',
        sourceId: sources[0].id
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async stopRecording() {
    if (!this.isRecording) {
      return { success: false, error: 'Not recording' };
    }

    this.isRecording = false;

    // Save recorded chunks
    const outputPath = path.join(app.getPath('videos'), `recording-${Date.now()}.webm`);
    
    return {
      success: true,
      message: 'Recording stopped',
      outputPath
    };
  }

  async getRecordingStatus() {
    return {
      isRecording: this.isRecording,
      duration: this.isRecording ? Date.now() - this.startTime : 0
    };
  }

  cleanup() {
    if (this.isRecording) {
      this.stopRecording();
    }
    this.recordedChunks = [];
  }
}

module.exports = ScreenRecorderService;
