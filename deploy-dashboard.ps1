# Computer Genie Dashboard Deployment Script (PowerShell)
# This script deploys only the dashboard component while keeping other components intact

param(
    [string]$Environment = "staging",
    [bool]$BuildOnly = $false
)

# Configuration
$DashboardDir = "computer-genie-dashboard"
$DockerImageName = "computer-genie/dashboard"

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if dashboard directory exists
function Test-DashboardDirectory {
    if (-not (Test-Path $DashboardDir)) {
        Write-Error "Dashboard directory '$DashboardDir' not found!"
        exit 1
    }
    Write-Success "Dashboard directory found"
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if Node.js is installed
    try {
        $nodeVersion = node --version
        $versionNumber = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
        if ($versionNumber -lt 18) {
            Write-Error "Node.js version 18 or higher is required. Current version: $nodeVersion"
            exit 1
        }
        Write-Success "Node.js version check passed: $nodeVersion"
    }
    catch {
        Write-Error "Node.js is not installed. Please install Node.js 18 or higher."
        exit 1
    }
    
    # Check if npm is installed
    try {
        npm --version | Out-Null
        Write-Success "npm is available"
    }
    catch {
        Write-Error "npm is not installed."
        exit 1
    }
    
    # Check if Docker is installed (for Docker deployment)
    try {
        docker --version | Out-Null
        Write-Success "Docker is available"
    }
    catch {
        Write-Warning "Docker is not installed. Docker deployment will not be available."
    }
    
    Write-Success "Prerequisites check completed"
}

# Install dependencies
function Install-Dependencies {
    Write-Info "Installing dashboard dependencies..."
    Push-Location $DashboardDir
    
    try {
        if (Test-Path "package-lock.json") {
            npm ci
        } else {
            npm install
        }
        Write-Success "Dependencies installed"
    }
    catch {
        Write-Error "Failed to install dependencies"
        Pop-Location
        exit 1
    }
    finally {
        Pop-Location
    }
}

# Run tests
function Invoke-Tests {
    Write-Info "Running dashboard tests..."
    Push-Location $DashboardDir
    
    try {
        # Run linting
        Write-Info "Running linter..."
        npm run lint
        Write-Success "Linting passed"
        
        # Run type checking
        Write-Info "Running type check..."
        npm run type-check
        Write-Success "Type checking passed"
        
        # Run tests
        Write-Info "Running tests..."
        npm run test
        Write-Success "Tests passed"
    }
    catch {
        Write-Error "Tests failed"
        Pop-Location
        exit 1
    }
    finally {
        Pop-Location
    }
}

# Build dashboard
function Build-Dashboard {
    Write-Info "Building dashboard for $Environment environment..."
    Push-Location $DashboardDir
    
    try {
        # Set environment variables
        $env:NODE_ENV = "production"
        
        # Build the application
        npm run build
        Write-Success "Dashboard build completed"
    }
    catch {
        Write-Error "Dashboard build failed"
        Pop-Location
        exit 1
    }
    finally {
        Pop-Location
    }
}

# Build Docker image
function Build-DockerImage {
    try {
        docker --version | Out-Null
        Write-Info "Building Docker image..."
        
        # Build the Docker image
        docker build -t "$DockerImageName`:latest" -t "$DockerImageName`:$Environment" $DashboardDir
        
        Write-Success "Docker image built: $DockerImageName`:$Environment"
    }
    catch {
        Write-Warning "Docker not available, skipping Docker image build"
    }
}

# Deploy to Vercel
function Deploy-ToVercel {
    Write-Info "Deploying to Vercel ($Environment)..."
    
    try {
        vercel --version | Out-Null
    }
    catch {
        Write-Warning "Vercel CLI not installed. Installing..."
        npm install -g vercel
    }
    
    Push-Location $DashboardDir
    
    try {
        if ($Environment -eq "production") {
            vercel --prod
        } else {
            vercel
        }
        Write-Success "Deployed to Vercel"
    }
    catch {
        Write-Error "Vercel deployment failed"
        Pop-Location
        exit 1
    }
    finally {
        Pop-Location
    }
}

# Deploy with Docker Compose
function Deploy-WithDocker {
    try {
        docker-compose --version | Out-Null
        Write-Info "Deploying with Docker Compose..."
        
        # Use the dashboard-specific docker-compose file
        if (Test-Path "docker-compose.dashboard.yml") {
            docker-compose -f docker-compose.dashboard.yml up -d
            Write-Success "Dashboard deployed with Docker Compose"
        } else {
            Write-Error "docker-compose.dashboard.yml not found"
            exit 1
        }
    }
    catch {
        Write-Warning "Docker Compose not available"
    }
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting dashboard deployment process..."
    Write-Info "Environment: $Environment"
    Write-Info "Build only: $BuildOnly"
    
    Test-DashboardDirectory
    Test-Prerequisites
    Install-Dependencies
    Invoke-Tests
    Build-Dashboard
    
    if ($BuildOnly) {
        Write-Success "Build completed. Skipping deployment."
        return
    }
    
    Build-DockerImage
    
    # Choose deployment method based on environment
    switch ($Environment) {
        "vercel" {
            Deploy-ToVercel
        }
        "docker" {
            Deploy-WithDocker
        }
        "staging" {
            Deploy-ToVercel
        }
        "production" {
            Deploy-ToVercel
        }
        default {
            Write-Error "Unknown deployment environment: $Environment"
            Write-Info "Available environments: staging, production, vercel, docker"
            exit 1
        }
    }
    
    Write-Success "Dashboard deployment completed successfully!"
}

# Show usage
function Show-Usage {
    Write-Host "Usage: .\deploy-dashboard.ps1 [-Environment <env>] [-BuildOnly <bool>]"
    Write-Host ""
    Write-Host "Environment options:"
    Write-Host "  staging     - Deploy to staging environment (default)"
    Write-Host "  production  - Deploy to production environment"
    Write-Host "  vercel      - Deploy to Vercel"
    Write-Host "  docker      - Deploy with Docker Compose"
    Write-Host ""
    Write-Host "BuildOnly options:"
    Write-Host "  `$false      - Build and deploy (default)"
    Write-Host "  `$true       - Build only, skip deployment"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy-dashboard.ps1                           # Deploy to staging"
    Write-Host "  .\deploy-dashboard.ps1 -Environment production   # Deploy to production"
    Write-Host "  .\deploy-dashboard.ps1 -Environment staging -BuildOnly `$true  # Build only for staging"
    Write-Host "  .\deploy-dashboard.ps1 -Environment docker       # Deploy with Docker"
}

# Handle script arguments
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Show-Usage
    exit 0
}

# Start deployment
try {
    Start-Deployment
}
catch {
    Write-Error "Deployment failed: $_"
    exit 1
}