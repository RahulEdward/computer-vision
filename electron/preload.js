const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // OCR
  ocr: {
    recognize: (image) => ipcRenderer.invoke('ocr:recognize', image),
    recognizeRegion: (image, region) => ipcRenderer.invoke('ocr:recognizeRegion', image, region)
  },

  // Color Picker
  colorPicker: {
    getPixelColor: (x, y) => ipcRenderer.invoke('colorPicker:getPixelColor', x, y),
    findColor: (color, tolerance) => ipcRenderer.invoke('colorPicker:findColor', color, tolerance)
  },

  // Window Manager
  window: {
    list: () => ipcRenderer.invoke('window:list'),
    focus: (windowId) => ipcRenderer.invoke('window:focus', windowId),
    resize: (windowId, width, height) => ipcRenderer.invoke('window:resize', windowId, width, height),
    move: (windowId, x, y) => ipcRenderer.invoke('window:move', windowId, x, y)
  },

  // Text Expander
  textExpander: {
    register: (trigger, expansion) => ipcRenderer.invoke('textExpander:register', trigger, expansion),
    start: () => ipcRenderer.invoke('textExpander:start'),
    stop: () => ipcRenderer.invoke('textExpander:stop')
  },

  // Clipboard
  clipboard: {
    getHistory: () => ipcRenderer.invoke('clipboard:getHistory'),
    transform: (content, transformer) => ipcRenderer.invoke('clipboard:transform', content, transformer)
  },

  // File Watcher
  fileWatcher: {
    watch: (path, options) => ipcRenderer.invoke('fileWatcher:watch', path, options),
    unwatch: (path) => ipcRenderer.invoke('fileWatcher:unwatch', path)
  },

  // Screen Recorder
  screenRecorder: {
    start: (options) => ipcRenderer.invoke('screenRecorder:start', options),
    stop: () => ipcRenderer.invoke('screenRecorder:stop')
  },

  // Update events
  onUpdateAvailable: (callback) => ipcRenderer.on('update-available', callback),
  onUpdateDownloaded: (callback) => ipcRenderer.on('update-downloaded', callback)
});
