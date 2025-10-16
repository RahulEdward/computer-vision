# 🖥️ Computer Genie Desktop - Quick Start Guide

## What You Get

Computer Genie Desktop is a powerful automation platform that combines:
- 🎨 Beautiful web-based workflow builder
- 🖥️ Native desktop automation capabilities
- 🤖 AI-powered task automation
- 📊 Real-time monitoring and analytics

## Desktop Features

### 7 Powerful Services

1. **OCR Service** - Extract text from images and screenshots
2. **Color Picker** - Pick colors from anywhere on screen
3. **Window Manager** - Control application windows
4. **Text Expander** - Create keyboard shortcuts for text
5. **Clipboard Manager** - Enhanced clipboard with history
6. **File Watcher** - Monitor file system changes
7. **Screen Recorder** - Record screen with audio

## Quick Start

### 1. Install Dependencies

```powershell
# Run the installation script
.\install-desktop-deps.ps1

# Or manually install
npm install
```

### 2. Set Up Environment

Copy `.env.example` to `.env` and configure:

```env
# Database
DATABASE_URL="file:./dev.db"

# NextAuth
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key-here"

# Stripe (optional for SaaS features)
STRIPE_SECRET_KEY="sk_test_..."
STRIPE_WEBHOOK_SECRET="whsec_..."
```

### 3. Initialize Database

```powershell
npx prisma generate
npx prisma db push
```

### 4. Run the App

**Option A: Development Mode**
```powershell
# Terminal 1: Start Next.js
npm run dev

# Terminal 2: Start Electron
npm run electron:dev
```

**Option B: Combined (Recommended)**
```powershell
npm run electron:dev
```

This will:
1. Start the Next.js dev server
2. Wait for it to be ready
3. Launch Electron automatically

## Using Desktop Features

### In the Dashboard

Once the app launches, you'll see the **Desktop Toolbar** at the top:

```
┌─────────────────────────────────────────────────┐
│  🖥️ Desktop Tools              ● Connected      │
├─────────────────────────────────────────────────┤
│  [🎨]    [📝]    [🪟]    [📋]    [📁]    [🎥]   │
│ Color   OCR   Windows Clipboard Files  Record  │
│ ⌘⇧C    ⌘⇧O    ⌘⇧W     ⌘⇧V     ⌘⇧F    ⌘⇧R    │
└─────────────────────────────────────────────────┘
```

### Global Shortcuts

- `Ctrl+Shift+C` - Color Picker
- `Ctrl+Shift+O` - OCR Text Recognition
- `Ctrl+Shift+W` - Window Manager
- `Ctrl+Shift+V` - Clipboard History
- `Ctrl+Shift+F` - File Watcher
- `Ctrl+Shift+R` - Screen Recording

### Using Services in Code

```typescript
import { useDesktopServices } from '@/hooks/useDesktopServices';

function MyComponent() {
  const { 
    recognizeText, 
    getPixelColor,
    listWindows 
  } = useDesktopServices();

  const handleOCR = async () => {
    const result = await recognizeText('screenshot.png');
    console.log(result.text);
  };

  const handleColorPick = async () => {
    const color = await getPixelColor(100, 100);
    console.log(color.hex); // #FF5733
  };

  return (
    <button onClick={handleOCR}>Extract Text</button>
  );
}
```

## Building for Production

### Windows
```powershell
npm run electron:build:win
```

Output: `dist/Computer Genie Setup.exe`

### macOS
```bash
npm run electron:build:mac
```

Output: `dist/Computer Genie.dmg`

### Linux
```bash
npm run electron:build:linux
```

Output: `dist/Computer Genie.AppImage`

## Project Structure

```
computer-genie-dashboard/
├── electron/                    # Electron main process
│   ├── main.js                 # Main entry point
│   ├── preload.js              # Secure API bridge
│   └── services/               # Desktop services
│       ├── ServiceManager.js   # Service orchestrator
│       ├── OCRService.js
│       ├── ColorPickerService.js
│       ├── WindowManagerService.js
│       ├── TextExpanderService.js
│       ├── ClipboardService.js
│       ├── FileWatcherService.js
│       └── ScreenRecorderService.js
│
├── src/
│   ├── app/                    # Next.js pages
│   ├── components/
│   │   ├── desktop/           # Desktop UI components
│   │   │   └── DesktopToolbar.tsx
│   │   └── workflow/          # Workflow builder
│   ├── hooks/
│   │   └── useDesktopServices.ts  # React hook
│   └── services/              # Business logic
│
└── prisma/
    └── schema.prisma          # Database schema
```

## Troubleshooting

### Services Not Starting

**Problem**: Desktop toolbar shows "Disconnected"

**Solution**:
```powershell
# Rebuild native modules
npx electron-rebuild

# Reinstall dependencies
rm -rf node_modules
npm install
```

### RobotJS Build Errors

**Problem**: `robotjs` fails to build

**Solution**:
```powershell
# Install Windows Build Tools
npm install --global windows-build-tools

# Rebuild
npx electron-rebuild
```

### Permission Errors

**Problem**: Services can't access system features

**Solution**:
- Run as Administrator (Windows)
- Grant accessibility permissions (macOS)
- Check antivirus settings

### Electron Window Not Opening

**Problem**: Electron starts but no window appears

**Solution**:
```powershell
# Check if Next.js is running
curl http://localhost:3000

# Clear Electron cache
rm -rf %APPDATA%\computer-genie-desktop
```

## Development Tips

### Hot Reload

The app supports hot reload for both:
- **React Components**: Auto-reload on save
- **Electron Main Process**: Restart Electron manually

### Debugging

**React DevTools**: Built-in when running in dev mode

**Electron DevTools**: 
```javascript
// In main.js
mainWindow.webContents.openDevTools();
```

**Service Logs**:
```javascript
// Check console for service logs
[ServiceManager] Initializing all services...
[OCRService] Initialized
[ColorPickerService] Initialized
...
```

### Adding New Services

1. Create service file in `electron/services/`
2. Add to `ServiceManager.js`
3. Expose API in `preload.js`
4. Add React hook method in `useDesktopServices.ts`
5. Update toolbar UI if needed

## Performance Tips

### Optimize OCR
```javascript
// Use smaller images
await recognizeText(image, { 
  scale: 0.5,  // Reduce size
  lang: 'eng'  // Specify language
});
```

### Debounce File Watcher
```javascript
await watchFile(path, {
  debounce: 1000  // Wait 1s before triggering
});
```

### Limit Clipboard History
```javascript
// In ClipboardService.js
this.maxHistory = 20;  // Reduce from 50
```

## Security Best Practices

1. **Never expose sensitive APIs** in preload.js
2. **Validate all IPC inputs** in main process
3. **Use context isolation** (already enabled)
4. **Keep dependencies updated**
5. **Sign your builds** for production

## Resources

- 📚 [Full Documentation](./DESKTOP_FEATURES_COMPLETE.md)
- 🎯 [Workflow Guide](./COMPLETE_WORKFLOW_GUIDE.md)
- 💰 [SaaS Features](./SAAS_COMPLETE_SUMMARY.md)
- 🔧 [Backend Setup](./BACKEND_COMPLETE.md)

## Support

Having issues? Check:
1. Console logs in DevTools
2. Electron main process logs
3. Service initialization status
4. Network connectivity (for web features)

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure environment
3. ✅ Run the app
4. 🎯 Create your first workflow
5. 🚀 Build automation magic!

---

**Made with ❤️ by the Computer Genie Team**
