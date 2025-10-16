const { globalShortcut, clipboard } = require('electron');
const robot = require('robotjs');

class TextExpanderService {
  constructor() {
    this.shortcuts = new Map();
    this.isListening = false;
    this.typedBuffer = '';
    this.maxBufferLength = 50;
  }

  async initialize() {
    // Register built-in shortcuts
    this.registerShortcut('@@email', 'user@example.com');
    this.registerShortcut('@@date', () => new Date().toLocaleDateString());
    this.registerShortcut('@@time', () => new Date().toLocaleTimeString());
    this.registerShortcut('@@sig', 'Best regards,\nYour Name');
    
    console.log('[TextExpander] Service initialized');
  }

  registerShortcut(trigger, expansion) {
    this.shortcuts.set(trigger, expansion);
    return { success: true, trigger, expansion };
  }

  async startListening() {
    if (this.isListening) return { success: false, error: 'Already listening' };

    this.isListening = true;
    
    // Note: This is a simplified version
    // In production, use a proper keyboard hook library
    console.log('[TextExpander] Started listening');
    
    return { success: true };
  }

  stopListening() {
    this.isListening = false;
    this.typedBuffer = '';
    return { success: true };
  }

  handleKeyPress(key) {
    if (!this.isListening) return;

    // Add to buffer
    this.typedBuffer += key;
    
    // Keep buffer size manageable
    if (this.typedBuffer.length > this.maxBufferLength) {
      this.typedBuffer = this.typedBuffer.slice(-this.maxBufferLength);
    }

    // Check for matches
    for (const [trigger, expansion] of this.shortcuts) {
      if (this.typedBuffer.endsWith(trigger)) {
        this.expandText(trigger, expansion);
        break;
      }
    }
  }

  async expandText(trigger, expansion) {
    try {
      // Get expansion value (could be function)
      const text = typeof expansion === 'function' ? expansion() : expansion;

      // Delete trigger text
      for (let i = 0; i < trigger.length; i++) {
        robot.keyTap('backspace');
      }

      // Type expansion
      robot.typeString(text);

      // Clear buffer
      this.typedBuffer = '';

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  getShortcuts() {
    return Array.from(this.shortcuts.entries()).map(([trigger, expansion]) => ({
      trigger,
      expansion: typeof expansion === 'function' ? '<dynamic>' : expansion
    }));
  }

  cleanup() {
    this.stopListening();
    this.shortcuts.clear();
  }
}

module.exports = TextExpanderService;
