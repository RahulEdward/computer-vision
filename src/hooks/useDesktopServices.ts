'use client';

import { useState, useEffect } from 'react';

// Type definitions for desktop services
interface ServiceStatus {
  initialized: boolean;
  services: string[];
  count: number;
}

interface OCRResult {
  success: boolean;
  text?: string;
  confidence?: number;
  error?: string;
}

interface ColorResult {
  success: boolean;
  hex?: string;
  rgb?: { r: number; g: number; b: number };
  hsl?: { h: number; s: number; l: number };
  error?: string;
}

export function useDesktopServices() {
  const [isElectron, setIsElectron] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);

  useEffect(() => {
    // Check if running in Electron
    setIsElectron(typeof window !== 'undefined' && !!(window as any).electronAPI);
    
    if ((window as any).electronAPI) {
      // Get service status
      (window as any).electronAPI.getServiceStatus().then(setServiceStatus);
    }
  }, []);

  // OCR Service
  const recognizeText = async (image: string): Promise<OCRResult> => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.ocr.recognizeText(image);
  };

  // Color Picker Service
  const getPixelColor = async (x: number, y: number): Promise<ColorResult> => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.colorPicker.getPixelColor(x, y);
  };

  // Window Manager Service
  const listWindows = async () => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.windowManager.listWindows();
  };

  const focusWindow = async (id: number) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.windowManager.focusWindow(id);
  };

  // Text Expander Service
  const registerShortcut = async (trigger: string, expansion: string) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.textExpander.registerShortcut(trigger, expansion);
  };

  const getShortcuts = async () => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.textExpander.getShortcuts();
  };

  // Clipboard Service
  const getClipboardHistory = async () => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.clipboard.getHistory();
  };

  const transformClipboard = async (content: string, transformer: string) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.clipboard.transform(content, transformer);
  };

  // File Watcher Service
  const watchFile = async (path: string, options?: any) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.fileWatcher.watch(path, options);
  };

  const unwatchFile = async (path: string) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.fileWatcher.unwatch(path);
  };

  // Screen Recorder Service
  const startRecording = async (options?: any) => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.screenRecorder.startRecording(options);
  };

  const stopRecording = async () => {
    if (!isElectron) return { success: false, error: 'Not running in desktop app' };
    return await (window as any).electronAPI.screenRecorder.stopRecording();
  };

  return {
    isElectron,
    serviceStatus,
    
    // OCR
    recognizeText,
    
    // Color Picker
    getPixelColor,
    
    // Window Manager
    listWindows,
    focusWindow,
    
    // Text Expander
    registerShortcut,
    getShortcuts,
    
    // Clipboard
    getClipboardHistory,
    transformClipboard,
    
    // File Watcher
    watchFile,
    unwatchFile,
    
    // Screen Recorder
    startRecording,
    stopRecording
  };
}
