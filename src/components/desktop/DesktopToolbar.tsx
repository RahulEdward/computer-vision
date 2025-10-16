'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  EyeDropperIcon,
  DocumentTextIcon,
  WindowIcon,
  ClipboardIcon,
  FolderIcon,
  VideoCameraIcon,
  CogIcon
} from '@heroicons/react/24/outline';
import { useDesktopServices } from '@/hooks/useDesktopServices';

export default function DesktopToolbar() {
  const {
    isElectron,
    serviceStatus,
    getPixelColor,
    recognizeText,
    getClipboardHistory,
    startRecording,
    stopRecording
  } = useDesktopServices();

  const [isRecording, setIsRecording] = useState(false);
  const [selectedColor, setSelectedColor] = useState<string | null>(null);

  if (!isElectron) {
    return null;
  }

  const handleColorPicker = async () => {
    // In a real implementation, you'd show a crosshair cursor
    // and let user click to pick color
    const result = await getPixelColor(100, 100); // Example coordinates
    if (result.success && result.hex) {
      setSelectedColor(result.hex);
    }
  };

  const handleOCR = async () => {
    // In a real implementation, you'd capture screen region
    // and pass it to OCR service
    const result = await recognizeText('screenshot.png');
    if (result.success) {
      console.log('OCR Result:', result.text);
    }
  };

  const handleRecording = async () => {
    if (isRecording) {
      await stopRecording();
      setIsRecording(false);
    } else {
      await startRecording();
      setIsRecording(true);
    }
  };

  const tools = [
    {
      id: 'color-picker',
      name: 'Color Picker',
      icon: EyeDropperIcon,
      action: handleColorPicker,
      shortcut: '⌘⇧C'
    },
    {
      id: 'ocr',
      name: 'OCR Text',
      icon: DocumentTextIcon,
      action: handleOCR,
      shortcut: '⌘⇧O'
    },
    {
      id: 'windows',
      name: 'Windows',
      icon: WindowIcon,
      action: () => console.log('Window manager'),
      shortcut: '⌘⇧W'
    },
    {
      id: 'clipboard',
      name: 'Clipboard',
      icon: ClipboardIcon,
      action: () => getClipboardHistory(),
      shortcut: '⌘⇧V'
    },
    {
      id: 'files',
      name: 'File Watcher',
      icon: FolderIcon,
      action: () => console.log('File watcher'),
      shortcut: '⌘⇧F'
    },
    {
      id: 'recording',
      name: isRecording ? 'Stop Recording' : 'Start Recording',
      icon: VideoCameraIcon,
      action: handleRecording,
      shortcut: '⌘⇧R',
      active: isRecording
    }
  ];

  return (
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-xl p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <CogIcon className="w-5 h-5" />
          Desktop Tools
        </h3>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            serviceStatus?.initialized ? 'bg-green-400' : 'bg-red-400'
          }`} />
          <span className="text-sm text-gray-300">
            {serviceStatus?.initialized ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        {tools.map((tool) => (
          <motion.button
            key={tool.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={tool.action}
            className={`p-3 rounded-lg border transition-all group ${
              tool.active
                ? 'bg-red-500/20 border-red-500/50 text-red-300'
                : 'bg-white/5 border-white/10 text-gray-300 hover:bg-white/10 hover:border-white/20'
            }`}
            title={`${tool.name} (${tool.shortcut})`}
          >
            <tool.icon className="w-6 h-6 mx-auto mb-2" />
            <div className="text-xs font-medium">{tool.name}</div>
            <div className="text-xs opacity-60 mt-1">{tool.shortcut}</div>
          </motion.button>
        ))}
      </div>

      {selectedColor && (
        <div className="mt-4 p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-3">
            <div 
              className="w-8 h-8 rounded border border-white/20"
              style={{ backgroundColor: selectedColor }}
            />
            <div>
              <div className="text-white font-medium">{selectedColor}</div>
              <div className="text-gray-400 text-sm">Picked Color</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
