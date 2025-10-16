@echo off
echo ========================================
echo  Computer Genie Desktop - Starting...
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Starting Next.js Dev Server...
echo.
start "Next.js Dev Server" cmd /k "npm run dev"

echo Waiting for Next.js to be ready...
timeout /t 10 /nobreak >nul

echo.
echo [2/2] Starting Electron Desktop App...
echo.
start "Electron Desktop App" cmd /k "npm run electron"

echo.
echo ========================================
echo  Desktop App is Starting!
echo ========================================
echo.
echo Two windows have opened:
echo  1. Next.js Dev Server (keep running)
echo  2. Electron Desktop App (keep running)
echo.
echo The Electron window should appear shortly...
echo.
echo Press any key to close this window...
pause >nul
