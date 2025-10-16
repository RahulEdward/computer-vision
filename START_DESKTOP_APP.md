# 🚀 Starting the Desktop App

## Quick Start (Recommended)

Open your terminal in the `computer-genie-dashboard` folder and run:

```powershell
npm run electron:dev
```

This will:
1. Start the Next.js dev server on http://localhost:3000
2. Wait for it to be ready
3. Launch the Electron desktop app automatically

---

## Alternative: Two Terminal Method

If the above doesn't work, use two separate terminals:

### Terminal 1: Start Next.js
```powershell
cd computer-genie-dashboard
npm run dev
```

Wait until you see: `✓ Ready in X.XXs`

### Terminal 2: Start Electron
```powershell
cd computer-genie-dashboard
npm run electron
```

---

## What You Should See

1. **Terminal Output:**
   ```
   ✓ Ready in 2.5s
   ○ Local: http://localhost:3000
   [ServiceManager] Initializing all services...
   [OCRService] Initialized
   [ColorPickerService] Initialized
   ...
   ```

2. **Electron Window:**
   - Desktop app window opens
   - Dashboard loads with workflow builder
   - Desktop toolbar appears at the top
   - Service status shows "Connected" (green dot)

---

## Troubleshooting

### Port 3000 Already in Use
```powershell
# Find and kill the process
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Electron Doesn't Start
```powershell
# Make sure Next.js is running first
# Check http://localhost:3000 in browser
# Then run: npm run electron
```

### Services Not Initializing
```powershell
# Rebuild native modules
npx electron-rebuild
```

### Missing Dependencies
```powershell
# Reinstall with legacy peer deps
npm install --legacy-peer-deps
```

---

## Testing the Desktop Features

Once the app is running:

1. **Check Service Status**
   - Look for the Desktop Toolbar at the top
   - Green dot = Connected ✅
   - Red dot = Disconnected ❌

2. **Test Each Tool**
   - Click each button in the toolbar
   - Check console for logs
   - Try global shortcuts (Ctrl+Shift+C, etc.)

3. **Open DevTools**
   - Press `Ctrl+Shift+I`
   - Check Console tab for service logs
   - Look for any errors

---

## Next Steps

After the app is running:
1. ✅ Verify all services are connected
2. 🎨 Test the color picker
3. 📝 Try OCR on an image
4. 🪟 List open windows
5. 📋 Check clipboard history
6. 🎥 Test screen recording

---

## Stopping the App

- Press `Ctrl+C` in the terminal(s)
- Or close the Electron window

---

**Ready to automate! 🚀**
