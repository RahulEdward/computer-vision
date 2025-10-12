# 🧞 Computer Genie Dashboard

<div align="center">

![Next.js](https://img.shields.io/badge/Next.js-15.5.4-black?style=for-the-badge&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=for-the-badge&logo=typescript)
![React](https://img.shields.io/badge/React-19.1-61DAFB?style=for-the-badge&logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.0-38B2AC?style=for-the-badge&logo=tailwind-css)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A next-generation automation dashboard with real-time collaboration, 3D visualization, AI-powered search, and advanced workflow management.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### 🎨 **Modern UI/UX**
- **Dark/Light Theme Toggle** - Seamless theme switching with persistence
- **Responsive Design** - Works flawlessly on desktop, tablet, and mobile
- **Smooth Animations** - Powered by Framer Motion for fluid interactions
- **Glassmorphism Effects** - Modern, elegant design language
- **Adaptive Layout** - UI reorganizes based on user behavior

### 🚀 **Core Capabilities**

#### 1. **Real-time Collaboration** (Figma-like)
- WebRTC-based peer-to-peer collaboration
- Live cursor tracking and user presence
- Collaborative editing with conflict resolution
- Real-time chat and team communication
- CRDT-based synchronization with Yjs

#### 2. **Interactive Workflow Builder**
- Node-based visual programming with React Flow
- Drag-and-drop interface with custom nodes
- Real-time execution with status indicators
- Visual debugging and error handling
- Support for triggers, actions, and conditions

#### 3. **Advanced Code Editor**
- Monaco Editor integration (VS Code engine)
- Multi-language support (JS, TS, Python, JSON, YAML)
- IntelliSense and auto-completion
- Real-time collaborative editing
- Custom themes and syntax highlighting

#### 4. **3D Workflow Visualization**
- Three.js powered 3D rendering
- Interactive node manipulation in 3D space
- Real-time data flow visualization
- Performance metrics in 3D
- Smooth camera controls and navigation

#### 5. **Voice Control Interface**
- Web Speech API integration
- Natural language commands
- Voice feedback and confirmations
- Custom command registration
- Multi-language support

#### 6. **Augmented Reality Preview**
- Mobile AR support using device camera
- Workflow visualization in AR space
- Interactive AR objects
- Real-time data overlay
- Cross-platform compatibility

#### 7. **Semantic Search with AI**
- Fuzzy search with Fuse.js
- TensorFlow.js integration (optional)
- Auto-complete suggestions
- Related content discovery
- Multi-type search (workflows, nodes, templates, docs)

#### 8. **Performance Monitoring**
- Real-time metrics (FPS, memory, latency)
- Long task detection
- Layout shift monitoring
- Network performance tracking
- Performance score calculation
- **Target: <100ms interaction latency**

### 📱 **Progressive Web App (PWA)**
- Offline functionality with service workers
- Install prompt for desktop and mobile
- Background sync for data synchronization
- Push notifications support
- App shortcuts for quick actions

---

## 🎯 Demo

### Screenshots

#### Dashboard Overview
![Dashboard](docs/screenshots/dashboard.png)

#### Workflow Builder
![Workflow Builder](docs/screenshots/workflow-builder.png)

#### 3D Visualization
![3D View](docs/screenshots/3d-view.png)

#### Dark Mode
![Dark Mode](docs/screenshots/dark-mode.png)

### Live Demo
🔗 [https://computer-genie-dashboard.vercel.app](https://computer-genie-dashboard.vercel.app)

---

## 🚀 Installation

### Prerequisites
- **Node.js** 18+ 
- **npm** or **yarn**
- Modern browser with WebRTC support

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/computer-genie-dashboard.git
cd computer-genie-dashboard

# Install dependencies
npm install --legacy-peer-deps

# Set up environment variables
cp .env.example .env.local

# Start development server
npm run dev

# Open browser
# Navigate to http://localhost:3000
```

### Environment Variables

Create a `.env.local` file:

```env
# WebRTC Signaling Server
NEXT_PUBLIC_WEBRTC_SIGNALING_URL=wss://signaling.yjs.dev

# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:3001

# Feature Flags
NEXT_PUBLIC_ENABLE_AR=true
NEXT_PUBLIC_ENABLE_VOICE=true
NEXT_PUBLIC_ENABLE_3D=true

# Analytics (Optional)
NEXT_PUBLIC_GA_ID=your-ga-id
```

---

## 📦 Build & Deploy

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Docker Deployment

```bash
# Build Docker image
docker build -t computer-genie-dashboard .

# Run container
docker run -p 3000:3000 computer-genie-dashboard
```

### Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel --prod
```

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15.5.4 (App Router)
- **Language**: TypeScript 5.0
- **Styling**: Tailwind CSS 4.0
- **State Management**: Zustand
- **Animations**: Framer Motion

### Real-time & Collaboration
- **WebRTC**: Simple Peer
- **CRDT**: Yjs + y-webrtc
- **WebSocket**: Socket.io Client

### Visualization
- **3D Graphics**: Three.js + React Three Fiber
- **Workflow**: React Flow
- **Code Editor**: Monaco Editor

### AI & Search
- **Search**: Fuse.js (fuzzy search)
- **ML**: TensorFlow.js (optional)
- **NLP**: Web Speech API

### Performance
- **Monitoring**: Custom performance monitor
- **Optimization**: Code splitting, lazy loading
- **Caching**: Service workers, IndexedDB

---

## 📖 Documentation

### Project Structure

```
computer-genie-dashboard/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── layout.tsx         # Root layout with theme
│   │   └── page.tsx           # Home page
│   ├── components/            # React components
│   │   ├── dashboard/         # Main dashboard
│   │   ├── workflow/          # Workflow builder
│   │   ├── editor/            # Code editor
│   │   ├── 3d/               # 3D visualization
│   │   ├── ar/               # AR preview
│   │   ├── search/           # Semantic search
│   │   ├── collaboration/    # Real-time collab
│   │   └── ThemeProvider.tsx # Theme management
│   ├── lib/                   # Utility libraries
│   │   ├── store.ts          # Zustand store
│   │   ├── webrtc-collaboration.ts
│   │   ├── voice-control.ts
│   │   ├── semantic-search.ts
│   │   └── performance-monitor.ts
│   ├── services/             # API services
│   └── types/                # TypeScript types
├── public/                    # Static assets
│   ├── manifest.json         # PWA manifest
│   ├── sw.js                 # Service worker
│   └── offline.html          # Offline page
├── docs/                      # Documentation
├── .env.example              # Environment template
├── next.config.ts            # Next.js config
├── tailwind.config.ts        # Tailwind config
└── package.json              # Dependencies
```

### Key Concepts

#### Theme System
The dashboard uses a class-based dark mode system:

```typescript
// Toggle theme
const { theme, setTheme } = useDashboardStore();
setTheme(theme === 'dark' ? 'light' : 'dark');

// Theme persists to localStorage automatically
```

#### Real-time Collaboration
```typescript
// Initialize collaboration
const collaboration = useCollaboration('room-id');

// Update cursor position
collaboration.updateCursor(x, y);

// Broadcast events
collaboration.broadcast('event-name', data);
```

#### Voice Commands
```typescript
// Start voice control
const { startListening, speak } = useVoiceControl();
startListening();

// Add custom command
addCommand('create workflow', () => {
  // Your action
});
```

---

## 🎨 Customization

### Theme Colors

Edit `tailwind.config.ts`:

```typescript
theme: {
  extend: {
    colors: {
      primary: '#3b82f6',
      secondary: '#8b5cf6',
      // Add your colors
    }
  }
}
```

### Custom Nodes

Create custom workflow nodes:

```typescript
// src/components/workflow/nodes/CustomNode.tsx
export const CustomNode = ({ data }) => {
  return (
    <div className="custom-node">
      {/* Your node UI */}
    </div>
  );
};
```

---

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

---

## 📊 Performance Metrics

### Target Metrics
- **First Contentful Paint (FCP)**: < 1.5s
- **Largest Contentful Paint (LCP)**: < 2.5s
- **First Input Delay (FID)**: < 100ms
- **Cumulative Layout Shift (CLS)**: < 0.1
- **Time to Interactive (TTI)**: < 3.5s

### Optimization Techniques
- Code splitting and lazy loading
- Image optimization with Next.js Image
- Bundle size optimization
- Service worker caching
- Performance monitoring

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow TypeScript best practices
- Write meaningful commit messages
- Add tests for new features
- Update documentation
- Ensure all tests pass

---

## 🐛 Troubleshooting

### Common Issues

#### Theme not changing?
- Verify `darkMode: 'class'` in `tailwind.config.ts`
- Check browser console for errors
- Clear cache and restart dev server

#### WebRTC not working?
- Ensure HTTPS connection
- Check firewall settings
- Verify signaling server availability

#### Build errors?
- Delete `.next` folder: `rm -rf .next`
- Clear node_modules: `rm -rf node_modules`
- Reinstall: `npm install --legacy-peer-deps`

#### Performance issues?
- Check bundle size: `npm run analyze`
- Monitor performance panel in dashboard
- Use React DevTools Profiler

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Next.js Team** - Amazing React framework
- **Vercel** - Deployment platform
- **Yjs Team** - Real-time collaboration
- **Three.js Community** - 3D visualization
- **React Flow** - Node-based UI
- **Monaco Editor** - Code editing
- **Tailwind CSS** - Styling framework

---

## 📞 Support

- **Documentation**: [docs.computer-genie.dev](https://docs.computer-genie.dev)
- **Issues**: [GitHub Issues](https://github.com/yourusername/computer-genie-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/computer-genie-dashboard/discussions)
- **Email**: support@computer-genie.dev

---

## 🗺️ Roadmap

### Q1 2025
- [ ] Multi-user workspace support
- [ ] Advanced workflow templates
- [ ] Mobile app (React Native)
- [ ] Plugin system

### Q2 2025
- [ ] AI-powered workflow suggestions
- [ ] Advanced analytics dashboard
- [ ] Integration marketplace
- [ ] Enterprise features

### Q3 2025
- [ ] Self-hosted option
- [ ] Advanced security features
- [ ] Workflow versioning
- [ ] Team management

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/computer-genie-dashboard&type=Date)](https://star-history.com/#yourusername/computer-genie-dashboard&Date)

---

<div align="center">

**Built with ❤️ by the Computer Genie Team**

[Website](https://computer-genie.dev) • [Twitter](https://twitter.com/computergenie) • [Discord](https://discord.gg/computergenie)

</div>
