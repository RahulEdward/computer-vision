# 🚀 How to Start the Desktop App

## ⚡ Quick Start (3 Options)

### Option 1: Double-Click Batch File (Easiest)
1. Navigate to `computer-genie-dashboard` folder
2. Double-click `start-app.bat`
3. Two terminal windows will open
4. Wait for the Electron window to appear

### Option 2: PowerShell Script
1. Right-click `start-app.ps1`
2. Select "Run with PowerShell"
3. Two terminal windows will open
4. Wait for the Electron window to appear

### Option 3: Manual Command
1. Open terminal in `computer-genie-dashboard`
2. Run: `npm run electron:dev`
3. Wait for the app to start

---

## 📋 What Happens

1. **Next.js Dev Server starts** (Terminal 1)
   - Compiles the React app
   - Runs on http://localhost:3000
   - Shows "Ready in X.XXs"

2. **Electron App launches** (Terminal 2)
   - Waits for Next.js to be ready
   - Opens desktop window
   - Initializes 7 desktop services
   - Shows Desktop Toolbar

---

## ✅ Success Indicators

You'll know it's working when you see:

### In Terminal:
```
✓ Ready in 2.5s
○ Local: http://localhost:3000
[ServiceManager] Initializing all services...
[OCRService] Initialized
[ColorPickerService] Initialized
[WindowManagerService] Initialized
[TextExpanderService] Initialized
[ClipboardService] Initialized
[FileWatcherService] Initialized
[ScreenRecorderService] Initialized
[ServiceManager] All services initialized successfully
```

### In Electron Window:
- Desktop window opens
- Dark purple gradient background
- Desktop Toolbar at top with 6 tools
- Green "Connected" status
- Workflow Builder below

---

## 🎯 Testing the App

Once running, try:

1. **Check Service Status**
   - Look for green dot in toolbar
   - Should say "Connected"

2. **Test Tools**
   - Click Color Picker button
   - Click OCR button
   - Click other toolbar buttons

3. **Try Shortcuts**
   - `Ctrl+Shift+C` - Color Picker
   - `Ctrl+Shift+O` - OCR
   - `Ctrl+Shift+W` - Windows
   - `Ctrl+Shift+V` - Clipboard
   - `Ctrl+Shift+F` - File Watcher
   - `Ctrl+Shift+R` - Recording

4. **Open DevTools**
   - Press `Ctrl+Shift+I`
   - Check Console for logs

---

## 🛑 Stopping the App

- Close the Electron window
- Press `Ctrl+C` in both terminal windows
- Or just close the terminal windows

---

## 🐛 Troubleshooting

### Port 3000 Already in Use
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Electron Doesn't Open
- Make sure Next.js finished starting
- Check http://localhost:3000 in browser
- Try running `npm run electron` manually

### Services Show "Disconnected"
```powershell
npx electron-rebuild
```

### Missing Dependencies
```powershell
npm install --legacy-peer-deps
```

---

## 📁 Files Created

- `start-app.bat` - Windows batch script
- `start-app.ps1` - PowerShell script
- `HOW_TO_START.md` - This file

---

## 🎉 You're Ready!

Just double-click `start-app.bat` and start automating! 🚀
