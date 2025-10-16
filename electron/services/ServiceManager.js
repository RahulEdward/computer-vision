const OCRService = require('./OCRService');
const ColorPickerService = require('./ColorPickerService');
const WindowManagerService = require('./WindowManagerService');
const TextExpanderService = require('./TextExpanderService');
const ClipboardService = require('./ClipboardService');
const FileWatcherService = require('./FileWatcherService');
const ScreenRecorderService = require('./ScreenRecorderService');

class ServiceManager {
  constructor() {
    this.services = {
      ocr: new OCRService(),
      colorPicker: new ColorPickerService(),
      windowManager: new WindowManagerService(),
      textExpander: new TextExpanderService(),
      clipboard: new ClipboardService(),
      fileWatcher: new FileWatcherService(),
      screenRecorder: new ScreenRecorderService()
    };
    
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    console.log('[ServiceManager] Initializing all services...');
    
    try {
      // Initialize all services
      await Promise.all(
        Object.values(this.services).map(service => service.initialize())
      );
      
      this.initialized = true;
      console.log('[ServiceManager] All services initialized successfully');
      
      return { success: true, message: 'All services initialized' };
    } catch (error) {
      console.error('[ServiceManager] Initialization failed:', error);
      return { success: false, error: error.message };
    }
  }

  getService(name) {
    return this.services[name];
  }

  async executeService(serviceName, method, ...args) {
    const service = this.services[serviceName];
    if (!service) {
      return { success: false, error: `Service '${serviceName}' not found` };
    }

    if (typeof service[method] !== 'function') {
      return { success: false, error: `Method '${method}' not found in service '${serviceName}'` };
    }

    try {
      const result = await service[method](...args);
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async cleanup() {
    console.log('[ServiceManager] Cleaning up all services...');
    
    await Promise.all(
      Object.values(this.services).map(service => {
        if (service.cleanup) {
          return service.cleanup();
        }
      })
    );
    
    this.initialized = false;
  }

  getStatus() {
    return {
      initialized: this.initialized,
      services: Object.keys(this.services),
      count: Object.keys(this.services).length
    };
  }
}

module.exports = ServiceManager;
