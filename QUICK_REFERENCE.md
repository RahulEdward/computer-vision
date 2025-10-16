# 🚀 Computer Genie Desktop - Quick Reference Card

## Installation (One Command)
```powershell
npm install && npx prisma generate && npx prisma db push
```

## Run App
```powershell
npm run electron:dev
```

---

## 🎯 Desktop Services

| Service | Shortcut | Function |
|---------|----------|----------|
| 🎨 Color Picker | `Ctrl+Shift+C` | Pick colors from screen |
| 📝 OCR | `Ctrl+Shift+O` | Extract text from images |
| 🪟 Windows | `Ctrl+Shift+W` | Manage application windows |
| 📋 Clipboard | `Ctrl+Shift+V` | Access clipboard history |
| 📁 File Watcher | `Ctrl+Shift+F` | Monitor file changes |
| 🎥 Recorder | `Ctrl+Shift+R` | Record screen |

---

## 💻 Code Usage

### React Hook
```typescript
import { useDesktopServices } from '@/hooks/useDesktopServices';

const { recognizeText, getPixelColor } = useDesktopServices();

// Use services
const result = await recognizeText('image.png');
const color = await getPixelColor(100, 100);
```

### Direct API
```typescript
// In Electron renderer
const result = await window.electronAPI.ocr.recognizeText(image);
const color = await window.electronAPI.colorPicker.getPixelColor(x, y);
```

---

## 📦 Project Structure

```
computer-genie-dashboard/
├── electron/
│   ├── main.js              # Electron entry
│   ├── preload.js           # IPC bridge
│   └── services/            # 7 services + manager
├── src/
│   ├── app/                 # Next.js pages
│   ├── components/desktop/  # Desktop UI
│   └── hooks/               # React hooks
└── Documentation/           # 5 guide files
```

---

## 🔧 Common Commands

```powershell
# Development
npm run dev                  # Web only
npm run electron:dev         # Desktop app

# Build
npm run build               # Build Next.js
npm run electron:build:win  # Build Windows app

# Database
npx prisma studio           # Open database GUI
npx prisma generate         # Generate client
npx prisma db push          # Apply schema

# Troubleshooting
npx electron-rebuild        # Rebuild native modules
rm -rf node_modules && npm install  # Fresh install
```

---

## 🐛 Quick Fixes

### Services Not Working
```powershell
npx electron-rebuild
```

### Build Errors
```powershell
npm install --global windows-build-tools
npx electron-rebuild
```

### Port Already in Use
```powershell
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README_DESKTOP.md` | Quick start guide |
| `DESKTOP_FEATURES_COMPLETE.md` | Full feature docs |
| `DESKTOP_SETUP_CHECKLIST.md` | Setup verification |
| `COMPLETE_PLATFORM_SUMMARY.md` | Platform overview |
| `DESKTOP_IMPLEMENTATION_SUMMARY.md` | Implementation details |

---

## 🎨 Service Examples

### OCR
```typescript
const result = await recognizeText('screenshot.png');
console.log(result.text); // Extracted text
```

### Color Picker
```typescript
const color = await getPixelColor(100, 100);
console.log(color.hex); // #FF5733
```

### Clipboard
```typescript
const history = await getClipboardHistory();
const transformed = await transformClipboard(text, 'uppercase');
```

### File Watcher
```typescript
await watchFile('./config.json', { debounce: 1000 });
```

---

## 🔒 Security Checklist

- ✅ Context isolation enabled
- ✅ Node integration disabled
- ✅ Secure IPC via contextBridge
- ✅ Input validation
- ✅ Environment variables for secrets

---

## 📊 Performance Tips

1. **OCR**: Use smaller images
2. **File Watcher**: Increase debounce time
3. **Clipboard**: Reduce history limit
4. **Recording**: Lower quality for smaller files

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Configure `.env`
3. ✅ Run `npm run electron:dev`
4. 🎨 Customize UI
5. 🔧 Add custom services
6. 🚀 Build workflows

---

## 💡 Pro Tips

- Use DevTools: `Ctrl+Shift+I` in Electron
- Check logs: Console shows service status
- Test incrementally: One service at a time
- Read docs: Each service has detailed docs

---

## 🆘 Getting Help

1. Check console logs
2. Review documentation
3. Verify setup checklist
4. Test in isolation

---

**Version**: 1.0.0  
**Status**: Production Ready ✅  
**Last Updated**: January 2025

---

## 🎉 You're Ready!

Everything is set up and ready to go. Just run:

```powershell
npm run electron:dev
```

And start automating! 🚀
