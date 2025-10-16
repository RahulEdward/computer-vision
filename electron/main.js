const { app, BrowserWindow, ipcMain, globalShortcut, screen } = require('electron');
const { autoUpdater } = require('electron-updater');
const Store = require('electron-store');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

// Initialize store
const store = new Store();

// Services
const OCRService = require('./services/OCRService');
const ColorPickerService = require('./services/ColorPickerService');
const WindowManagerService = require('./services/WindowManagerService');
const TextExpanderService = require('./services/TextExpanderService');
const ClipboardService = require('./services/ClipboardService');
const FileWatcherService = require('./services/FileWatcherService');
const ScreenRecorderService = require('./services/ScreenRecorderService');

let mainWindow;
let services = {};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    frame: true,
    titleBarStyle: 'default',
    backgroundColor: '#0a0118',
    show: false
  });

  // Load app
  const url = isDev
    ? 'http://localhost:3000'
    : `file://${path.join(__dirname, '../.next/server/app/index.html')}`;

  mainWindow.loadURL(url);

  // Show when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Initialize services
  initializeServices();

  // Auto-updater
  if (!isDev) {
    autoUpdater.checkForUpdatesAndNotify();
  }
}

function initializeServices() {
  services.ocr = new OCRService();
  services.colorPicker = new ColorPickerService();
  services.windowManager = new WindowManagerService();
  services.textExpander = new TextExpanderService();
  services.clipboard = new ClipboardService();
  services.fileWatcher = new FileWatcherService();
  services.screenRecorder = new ScreenRecorderService();

  // Initialize all services
  Object.values(services).forEach(service => {
    if (service.initialize) {
      service.initialize();
    }
  });
}

// IPC Handlers

// OCR
ipcMain.handle('ocr:recognize', async (event, image) => {
  return await services.ocr.recognizeText(image);
});

ipcMain.handle('ocr:recognizeRegion', async (event, image, region) => {
  return await services.ocr.recognizeRegion(image, region);
});

// Color Picker
ipcMain.handle('colorPicker:getPixelColor', async (event, x, y) => {
  return await services.colorPicker.getPixelColor(x, y);
});

ipcMain.handle('colorPicker:findColor', async (event, color, tolerance) => {
  return await services.colorPicker.findColorOnScreen(color, tolerance);
});

// Window Manager
ipcMain.handle('window:list', async () => {
  return await services.windowManager.listWindows();
});

ipcMain.handle('window:focus', async (event, windowId) => {
  return await services.windowManager.focusWindow(windowId);
});

ipcMain.handle('window:resize', async (event, windowId, width, height) => {
  return await services.windowManager.resizeWindow(windowId, width, height);
});

ipcMain.handle('window:move', async (event, windowId, x, y) => {
  return await services.windowManager.moveWindow(windowId, x, y);
});

// Text Expander
ipcMain.handle('textExpander:register', async (event, trigger, expansion) => {
  return services.textExpander.registerShortcut(trigger, expansion);
});

ipcMain.handle('textExpander:start', async () => {
  return services.textExpander.startListening();
});

ipcMain.handle('textExpander:stop', async () => {
  return services.textExpander.stopListening();
});

// Clipboard
ipcMain.handle('clipboard:getHistory', async () => {
  return services.clipboard.getHistory();
});

ipcMain.handle('clipboard:transform', async (event, content, transformer) => {
  return await services.clipboard.transform(content, transformer);
});

// File Watcher
ipcMain.handle('fileWatcher:watch', async (event, path, options) => {
  return await services.fileWatcher.watch(path, options);
});

ipcMain.handle('fileWatcher:unwatch', async (event, path) => {
  return await services.fileWatcher.unwatch(path);
});

// Screen Recorder
ipcMain.handle('screenRecorder:start', async (event, options) => {
  return await services.screenRecorder.startRecording(options);
});

ipcMain.handle('screenRecorder:stop', async () => {
  return await services.screenRecorder.stopRecording();
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Auto-updater events
autoUpdater.on('update-available', () => {
  mainWindow.webContents.send('update-available');
});

autoUpdater.on('update-downloaded', () => {
  mainWindow.webContents.send('update-downloaded');
});

// Cleanup
app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  Object.values(services).forEach(service => {
    if (service.cleanup) {
      service.cleanup();
    }
  });
});
