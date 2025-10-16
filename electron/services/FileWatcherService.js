const chokidar = require('chokidar');
const path = require('path');

class FileWatcherService {
  constructor() {
    this.watchers = new Map();
    this.callbacks = new Map();
  }

  async initialize() {
    console.log('[FileWatcher] Service initialized');
  }

  async watch(watchPath, options = {}) {
    if (this.watchers.has(watchPath)) {
      return { success: false, error: 'Already watching this path' };
    }

    try {
      const watcher = chokidar.watch(watchPath, {
        persistent: true,
        ignoreInitial: true,
        awaitWriteFinish: {
          stabilityThreshold: 2000,
          pollInterval: 100
        },
        ...options
      });

      const callbacks = {
        add: [],
        change: [],
        unlink: []
      };

      watcher
        .on('add', (filePath) => {
          console.log(`[FileWatcher] File added: ${filePath}`);
          this.triggerCallbacks('add', filePath);
        })
        .on('change', (filePath) => {
          console.log(`[FileWatcher] File changed: ${filePath}`);
          this.triggerCallbacks('change', filePath);
        })
        .on('unlink', (filePath) => {
          console.log(`[FileWatcher] File deleted: ${filePath}`);
          this.triggerCallbacks('unlink', filePath);
        })
        .on('error', (error) => {
          console.error(`[FileWatcher] Error: ${error}`);
        });

      this.watchers.set(watchPath, watcher);
      this.callbacks.set(watchPath, callbacks);

      return {
        success: true,
        path: watchPath,
        message: 'Watching started'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async unwatch(watchPath) {
    const watcher = this.watchers.get(watchPath);
    if (!watcher) {
      return { success: false, error: 'Path not being watched' };
    }

    try {
      await watcher.close();
      this.watchers.delete(watchPath);
      this.callbacks.delete(watchPath);

      return {
        success: true,
        message: 'Watching stopped'
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  registerCallback(watchPath, event, callback) {
    const callbacks = this.callbacks.get(watchPath);
    if (callbacks && callbacks[event]) {
      callbacks[event].push(callback);
    }
  }

  triggerCallbacks(event, filePath) {
    this.callbacks.forEach((callbacks) => {
      if (callbacks[event]) {
        callbacks[event].forEach(cb => cb(filePath));
      }
    });
  }

  getWatchedPaths() {
    return Array.from(this.watchers.keys());
  }

  async cleanup() {
    for (const [path, watcher] of this.watchers) {
      await watcher.close();
    }
    this.watchers.clear();
    this.callbacks.clear();
  }
}

module.exports = FileWatcherService;
