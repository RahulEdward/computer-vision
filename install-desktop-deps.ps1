# Desktop Dependencies Installation Script for Windows
# Run this script to install all required dependencies for desktop features

Write-Host "🖥️  Installing Desktop Feature Dependencies..." -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js $nodeVersion found" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
    exit 1
}

# Check if npm is installed
Write-Host "Checking npm installation..." -ForegroundColor Yellow
$npmVersion = npm --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ npm $npmVersion found" -ForegroundColor Green
} else {
    Write-Host "❌ npm not found. Please install npm first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📦 Installing Electron and desktop dependencies..." -ForegroundColor Cyan

# Install main dependencies
npm install --save electron electron-is-dev

# Install desktop service dependencies
npm install --save tesseract.js robotjs chokidar screenshot-desktop

# Install dev dependencies for Electron
npm install --save-dev electron-builder electron-rebuild concurrently wait-on cross-env

Write-Host ""
Write-Host "🔧 Rebuilding native modules for Electron..." -ForegroundColor Cyan
npx electron-rebuild

Write-Host ""
Write-Host "✅ Desktop dependencies installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Yellow
Write-Host "1. Run 'npm run dev' to start the Next.js dev server" -ForegroundColor White
Write-Host "2. In another terminal, run 'npm run electron:dev' to start Electron" -ForegroundColor White
Write-Host "3. The desktop toolbar will appear at the top of the dashboard" -ForegroundColor White
Write-Host ""
Write-Host "📚 See DESKTOP_FEATURES_COMPLETE.md for full documentation" -ForegroundColor Cyan
