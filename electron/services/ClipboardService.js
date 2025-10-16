const { clipboard } = require('electron');

class ClipboardService {
  constructor() {
    this.history = [];
    this.maxHistory = 100;
    this.transformers = new Map();
    this.watchInterval = null;
    this.lastValue = '';
  }

  async initialize() {
    // Register built-in transformers
    this.registerTransformer('uppercase', (text) => text.toUpperCase());
    this.registerTransformer('lowercase', (text) => text.toLowerCase());
    this.registerTransformer('trim', (text) => text.trim());
    this.registerTransformer('extractEmails', (text) => {
      const emails = text.match(/[\w.-]+@[\w.-]+\.\w+/g);
      return emails ? emails.join('\n') : '';
    });
    this.registerTransformer('extractURLs', (text) => {
      const urls = text.match(/https?:\/\/[^\s]+/g);
      return urls ? urls.join('\n') : '';
    });
    this.registerTransformer('formatJSON', (text) => {
      try {
        return JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        return text;
      }
    });
    this.registerTransformer('removeDuplicates', (text) => {
      return [...new Set(text.split('\n'))].join('\n');
    });
    this.registerTransformer('sortLines', (text) => {
      return text.split('\n').sort().join('\n');
    });

    // Start watching clipboard
    this.startWatching();
    
    console.log('[Clipboard] Service initialized');
  }

  startWatching() {
    this.watchInterval = setInterval(() => {
      const current = clipboard.readText();
      if (current && current !== this.lastValue) {
        this.addToHistory(current);
        this.lastValue = current;
      }
    }, 500);
  }

  stopWatching() {
    if (this.watchInterval) {
      clearInterval(this.watchInterval);
      this.watchInterval = null;
    }
  }

  addToHistory(content) {
    const item = {
      content,
      timestamp: Date.now(),
      type: this.detectType(content)
    };

    this.history.unshift(item);

    // Keep only last N items
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(0, this.maxHistory);
    }
  }

  detectType(content) {
    if (/^https?:\/\//.test(content)) return 'url';
    if (/[\w.-]+@[\w.-]+\.\w+/.test(content)) return 'email';
    if (/^\d+$/.test(content)) return 'number';
    if (content.startsWith('{') || content.startsWith('[')) return 'json';
    return 'text';
  }

  getHistory() {
    return this.history;
  }

  registerTransformer(name, fn) {
    this.transformers.set(name, fn);
  }

  async transform(content, transformerName) {
    const transformer = this.transformers.get(transformerName);
    if (!transformer) {
      return { success: false, error: 'Transformer not found' };
    }

    try {
      const result = transformer(content);
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  getTransformers() {
    return Array.from(this.transformers.keys());
  }

  cleanup() {
    this.stopWatching();
    this.history = [];
    this.transformers.clear();
  }
}

module.exports = ClipboardService;
