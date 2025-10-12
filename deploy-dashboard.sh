#!/bin/bash

# Computer Genie Dashboard Deployment Script
# This script deploys only the dashboard component while keeping other components intact

set -e

# Configuration
DASHBOARD_DIR="computer-genie-dashboard"
DOCKER_IMAGE_NAME="computer-genie/dashboard"
DEPLOYMENT_ENV="${1:-staging}"
BUILD_ONLY="${2:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if dashboard directory exists
check_dashboard_directory() {
    if [ ! -d "$DASHBOARD_DIR" ]; then
        log_error "Dashboard directory '$DASHBOARD_DIR' not found!"
        exit 1
    fi
    log_success "Dashboard directory found"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18 or higher."
        exit 1
    fi
    
    # Check Node.js version
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        log_error "Node.js version 18 or higher is required. Current version: $(node --version)"
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed."
        exit 1
    fi
    
    # Check if Docker is installed (for Docker deployment)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. Docker deployment will not be available."
    fi
    
    log_success "Prerequisites check completed"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dashboard dependencies..."
    cd "$DASHBOARD_DIR"
    
    if [ -f "package-lock.json" ]; then
        npm ci
    else
        npm install
    fi
    
    cd ..
    log_success "Dependencies installed"
}

# Run tests
run_tests() {
    log_info "Running dashboard tests..."
    cd "$DASHBOARD_DIR"
    
    # Run linting
    if npm run lint; then
        log_success "Linting passed"
    else
        log_error "Linting failed"
        exit 1
    fi
    
    # Run type checking
    if npm run type-check; then
        log_success "Type checking passed"
    else
        log_error "Type checking failed"
        exit 1
    fi
    
    # Run tests
    if npm run test; then
        log_success "Tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
    
    cd ..
}

# Build dashboard
build_dashboard() {
    log_info "Building dashboard for $DEPLOYMENT_ENV environment..."
    cd "$DASHBOARD_DIR"
    
    # Set environment variables
    export NODE_ENV="production"
    
    # Build the application
    if npm run build; then
        log_success "Dashboard build completed"
    else
        log_error "Dashboard build failed"
        exit 1
    fi
    
    cd ..
}

# Build Docker image
build_docker_image() {
    if command -v docker &> /dev/null; then
        log_info "Building Docker image..."
        
        # Build the Docker image
        docker build -t "$DOCKER_IMAGE_NAME:latest" -t "$DOCKER_IMAGE_NAME:$DEPLOYMENT_ENV" "$DASHBOARD_DIR"
        
        log_success "Docker image built: $DOCKER_IMAGE_NAME:$DEPLOYMENT_ENV"
    else
        log_warning "Docker not available, skipping Docker image build"
    fi
}

# Deploy to Vercel
deploy_to_vercel() {
    log_info "Deploying to Vercel ($DEPLOYMENT_ENV)..."
    
    if ! command -v vercel &> /dev/null; then
        log_warning "Vercel CLI not installed. Installing..."
        npm install -g vercel
    fi
    
    cd "$DASHBOARD_DIR"
    
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        vercel --prod
    else
        vercel
    fi
    
    cd ..
    log_success "Deployed to Vercel"
}

# Deploy with Docker Compose
deploy_with_docker() {
    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        log_info "Deploying with Docker Compose..."
        
        # Use the dashboard-specific docker-compose file
        if [ -f "docker-compose.dashboard.yml" ]; then
            docker-compose -f docker-compose.dashboard.yml up -d
            log_success "Dashboard deployed with Docker Compose"
        else
            log_error "docker-compose.dashboard.yml not found"
            exit 1
        fi
    else
        log_warning "Docker Compose not available"
    fi
}

# Main deployment function
deploy() {
    log_info "Starting dashboard deployment process..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Build only: $BUILD_ONLY"
    
    check_dashboard_directory
    check_prerequisites
    install_dependencies
    run_tests
    build_dashboard
    
    if [ "$BUILD_ONLY" = "true" ]; then
        log_success "Build completed. Skipping deployment."
        return
    fi
    
    build_docker_image
    
    # Choose deployment method based on environment
    case $DEPLOYMENT_ENV in
        "vercel")
            deploy_to_vercel
            ;;
        "docker")
            deploy_with_docker
            ;;
        "staging"|"production")
            deploy_to_vercel
            ;;
        *)
            log_error "Unknown deployment environment: $DEPLOYMENT_ENV"
            log_info "Available environments: staging, production, vercel, docker"
            exit 1
            ;;
    esac
    
    log_success "Dashboard deployment completed successfully!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [ENVIRONMENT] [BUILD_ONLY]"
    echo ""
    echo "ENVIRONMENT options:"
    echo "  staging     - Deploy to staging environment (default)"
    echo "  production  - Deploy to production environment"
    echo "  vercel      - Deploy to Vercel"
    echo "  docker      - Deploy with Docker Compose"
    echo ""
    echo "BUILD_ONLY options:"
    echo "  false       - Build and deploy (default)"
    echo "  true        - Build only, skip deployment"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy to staging"
    echo "  $0 production         # Deploy to production"
    echo "  $0 staging true       # Build only for staging"
    echo "  $0 docker             # Deploy with Docker"
}

# Handle script arguments
case $1 in
    "-h"|"--help"|"help")
        show_usage
        exit 0
        ;;
    *)
        deploy
        ;;
esac