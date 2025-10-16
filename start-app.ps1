# Computer Genie Desktop - PowerShell Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Computer Genie Desktop - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "[1/2] Starting Next.js Dev Server..." -ForegroundColor Yellow
Write-Host ""

# Start Next.js in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath'; npm run dev"

Write-Host "Waiting for Next.js to be ready (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "[2/2] Starting Electron Desktop App..." -ForegroundColor Yellow
Write-Host ""

# Start Electron in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath'; npm run electron"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Desktop App is Starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Two windows have opened:" -ForegroundColor White
Write-Host "  1. Next.js Dev Server (keep running)" -ForegroundColor White
Write-Host "  2. Electron Desktop App (keep running)" -ForegroundColor White
Write-Host ""
Write-Host "The Electron window should appear shortly..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
