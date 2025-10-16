# Computer Genie SaaS - Quick Setup Script (PowerShell)

Write-Host "🧞‍♂️ Computer Genie SaaS - Quick Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
Write-Host "Checking Node.js..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Node.js $nodeVersion found" -ForegroundColor Green

# Check if npm is installed
Write-Host "Checking npm..." -ForegroundColor Yellow
$npmVersion = npm --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm not found" -ForegroundColor Red
    exit 1
}
Write-Host "✅ npm $npmVersion found" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
npm install --legacy-peer-deps
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Check if .env.local exists
if (!(Test-Path ".env.local")) {
    Write-Host "⚙️  Creating .env.local file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env.local"
    Write-Host "✅ .env.local created" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env.local with your database credentials!" -ForegroundColor Yellow
    Write-Host "   Minimum required:" -ForegroundColor Yellow
    Write-Host "   - DATABASE_URL" -ForegroundColor Yellow
    Write-Host "   - NEXTAUTH_SECRET (run: openssl rand -base64 32)" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✅ .env.local already exists" -ForegroundColor Green
}

# Ask if user wants to setup database
Write-Host "Do you want to setup the database now? (y/n)" -ForegroundColor Cyan
$setupDb = Read-Host
if ($setupDb -eq "y" -or $setupDb -eq "Y") {
    Write-Host ""
    Write-Host "🗄️  Setting up database..." -ForegroundColor Yellow
    
    # Generate Prisma Client
    Write-Host "Generating Prisma Client..." -ForegroundColor Yellow
    npx prisma generate
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to generate Prisma Client" -ForegroundColor Red
        Write-Host "   Make sure DATABASE_URL is set in .env.local" -ForegroundColor Yellow
    } else {
        Write-Host "✅ Prisma Client generated" -ForegroundColor Green
        
        # Run migrations
        Write-Host "Running database migrations..." -ForegroundColor Yellow
        npx prisma migrate dev --name init
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to run migrations" -ForegroundColor Red
            Write-Host "   Make sure your database is running and DATABASE_URL is correct" -ForegroundColor Yellow
        } else {
            Write-Host "✅ Database migrations completed" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env.local with your credentials" -ForegroundColor White
Write-Host "2. Run: npm run dev" -ForegroundColor White
Write-Host "3. Visit: http://localhost:3000/landing" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   - BACKEND_SETUP.md - Complete setup guide" -ForegroundColor White
Write-Host "   - QUICK_START_HINDI.md - Hindi quick start" -ForegroundColor White
Write-Host "   - SAAS_COMPLETE_SUMMARY.md - Feature overview" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green
