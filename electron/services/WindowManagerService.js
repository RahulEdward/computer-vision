const { BrowserWindow, screen } = require('electron');

class WindowManagerService {
  constructor() {
    this.windows = new Map();
  }

  async initialize() {
    console.log('[WindowManager] Service initialized');
  }

  async listWindows() {
    const windows = BrowserWindow.getAllWindows();
    return windows.map(win => ({
      id: win.id,
      title: win.getTitle(),
      bounds: win.getBounds(),
      isVisible: win.isVisible(),
      isMinimized: win.isMinimized(),
      isMaximized: win.isMaximized(),
      isFocused: win.isFocused()
    }));
  }

  async focusWindow(windowId) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      win.focus();
      win.show();
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async resizeWindow(windowId, width, height) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      win.setSize(width, height);
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async moveWindow(windowId, x, y) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      win.setPosition(x, y);
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async minimizeWindow(windowId) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      win.minimize();
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async maximizeWindow(windowId) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      if (win.isMaximized()) {
        win.unmaximize();
      } else {
        win.maximize();
      }
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async closeWindow(windowId) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      win.close();
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  async getWindowInfo(windowId) {
    const win = BrowserWindow.fromId(windowId);
    if (win) {
      return {
        success: true,
        info: {
          id: win.id,
          title: win.getTitle(),
          bounds: win.getBounds(),
          isVisible: win.isVisible(),
          isMinimized: win.isMinimized(),
          isMaximized: win.isMaximized(),
          isFocused: win.isFocused()
        }
      };
    }
    return { success: false, error: 'Window not found' };
  }

  cleanup() {
    this.windows.clear();
  }
}

module.exports = WindowManagerService;
