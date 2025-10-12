import { useDashboardStore } from './store';

export class VoiceController {
  private recognition: SpeechRecognition | null = null;
  private synthesis: SpeechSynthesis;
  private isListening = false;
  private commands: Map<string, () => void> = new Map();

  constructor() {
    this.synthesis = window.speechSynthesis;
    this.initializeSpeechRecognition();
    this.setupCommands();
  }

  private initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
      this.recognition = new (window as any).webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      this.recognition = new SpeechRecognition();
    }

    if (this.recognition) {
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';

      this.recognition.onstart = () => {
        this.isListening = true;
        useDashboardStore.getState().setListening(true);
      };

      this.recognition.onend = () => {
        this.isListening = false;
        useDashboardStore.getState().setListening(false);
      };

      this.recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');

        this.processCommand(transcript.toLowerCase().trim());
      };

      this.recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        this.speak('Sorry, I didn\'t catch that. Please try again.');
      };
    }
  }

  private setupCommands() {
    const store = useDashboardStore.getState();

    this.commands.set('create new workflow', () => {
      this.speak('Creating new workflow');
      // Trigger workflow creation
    });

    this.commands.set('save workflow', () => {
      this.speak('Saving workflow');
      // Trigger save
    });

    this.commands.set('run workflow', () => {
      this.speak('Running workflow');
      // Trigger execution
    });

    this.commands.set('toggle sidebar', () => {
      store.toggleSidebar();
      this.speak('Sidebar toggled');
    });

    this.commands.set('switch to dark mode', () => {
      store.setTheme('dark');
      this.speak('Switched to dark mode');
    });

    this.commands.set('switch to light mode', () => {
      store.setTheme('light');
      this.speak('Switched to light mode');
    });

    this.commands.set('show grid view', () => {
      store.setLayout('grid');
      this.speak('Switched to grid view');
    });

    this.commands.set('show list view', () => {
      store.setLayout('list');
      this.speak('Switched to list view');
    });

    this.commands.set('enable 3d view', () => {
      store.toggle3D();
      this.speak('3D view enabled');
    });

    this.commands.set('enable ar mode', () => {
      store.toggleAR();
      this.speak('AR mode enabled');
    });

    this.commands.set('help', () => {
      this.speak('Available commands: create new workflow, save workflow, run workflow, toggle sidebar, switch to dark mode, switch to light mode, show grid view, show list view, enable 3D view, enable AR mode');
    });
  }

  private processCommand(transcript: string) {
    // Find the best matching command
    let bestMatch = '';
    let bestScore = 0;

    for (const [command] of this.commands) {
      const score = this.calculateSimilarity(transcript, command);
      if (score > bestScore && score > 0.7) {
        bestMatch = command;
        bestScore = score;
      }
    }

    if (bestMatch) {
      const action = this.commands.get(bestMatch);
      if (action) {
        action();
      }
    } else {
      // Try partial matches for common actions
      if (transcript.includes('create') && transcript.includes('workflow')) {
        this.commands.get('create new workflow')?.();
      } else if (transcript.includes('save')) {
        this.commands.get('save workflow')?.();
      } else if (transcript.includes('run') || transcript.includes('execute')) {
        this.commands.get('run workflow')?.();
      } else {
        this.speak('Command not recognized. Say "help" for available commands.');
      }
    }
  }

  private calculateSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const editDistance = this.levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  speak(text: string) {
    if (this.synthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 0.8;
      this.synthesis.speak(utterance);
    }
  }

  startListening() {
    if (this.recognition && !this.isListening) {
      this.recognition.start();
    }
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
    }
  }

  addCommand(command: string, action: () => void) {
    this.commands.set(command.toLowerCase(), action);
  }

  removeCommand(command: string) {
    this.commands.delete(command.toLowerCase());
  }
}

// React hook for voice control
import { useEffect, useRef } from 'react';

export function useVoiceControl() {
  const voiceController = useRef<VoiceController | null>(null);
  const { voiceEnabled, listening } = useDashboardStore();

  useEffect(() => {
    if (voiceEnabled && !voiceController.current) {
      voiceController.current = new VoiceController();
    }
  }, [voiceEnabled]);

  const startListening = () => {
    voiceController.current?.startListening();
  };

  const stopListening = () => {
    voiceController.current?.stopListening();
  };

  const speak = (text: string) => {
    voiceController.current?.speak(text);
  };

  const addCommand = (command: string, action: () => void) => {
    voiceController.current?.addCommand(command, action);
  };

  return {
    startListening,
    stopListening,
    speak,
    addCommand,
    listening,
    isSupported: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window
  };
}