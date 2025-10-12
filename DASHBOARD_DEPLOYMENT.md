# Dashboard Deployment Guide

This guide explains how to deploy only the Computer Genie Dashboard while keeping all other components in the IDE git repository intact.

## Overview

The dashboard deployment process is designed to:
- Deploy only the dashboard functionality (`computer-genie-dashboard/` directory)
- Keep other components (Python packages, AI features, automation engine) unchanged
- Provide multiple deployment options (Vercel, Docker, local)
- Maintain separation between dashboard and other services

## Repository Structure

```
computer-vision/
├── computer-genie-dashboard/          # Dashboard application (Next.js)
├── computer_genie/                    # Python package (not deployed)
├── computer_genie_ai_features/        # AI features (not deployed)
├── intelligent_automation_engine/     # Automation engine (not deployed)
├── .github/workflows/deploy-dashboard.yml  # Dashboard-specific CI/CD
├── docker-compose.dashboard.yml       # Dashboard-only Docker setup
├── vercel.json                        # Vercel configuration for dashboard
├── deploy-dashboard.sh                # Bash deployment script
├── deploy-dashboard.ps1               # PowerShell deployment script
└── DASHBOARD_DEPLOYMENT.md            # This file
```

## Deployment Options

### 1. Automated Deployment (GitHub Actions)

The repository includes a GitHub Actions workflow that automatically deploys the dashboard when changes are made to the `computer-genie-dashboard/` directory.

#### Triggers
- Push to `main` branch with changes in `computer-genie-dashboard/`
- Manual workflow dispatch
- Pull requests affecting dashboard code

#### Setup
1. Configure secrets in your GitHub repository:
   ```
   VERCEL_TOKEN          # Vercel deployment token
   VERCEL_ORG_ID         # Vercel organization ID
   VERCEL_PROJECT_ID     # Vercel project ID
   DOCKER_USERNAME       # Docker Hub username
   DOCKER_PASSWORD       # Docker Hub password
   SLACK_WEBHOOK         # Slack notification webhook (optional)
   ```

2. The workflow will automatically:
   - Test the dashboard code
   - Build the application
   - Run security audits
   - Deploy to staging/production
   - Build and push Docker images

### 2. Manual Deployment with Scripts

#### Using Bash (Linux/macOS/WSL)

```bash
# Make script executable
chmod +x deploy-dashboard.sh

# Deploy to staging
./deploy-dashboard.sh staging

# Deploy to production
./deploy-dashboard.sh production

# Build only (no deployment)
./deploy-dashboard.sh staging true

# Deploy with Docker
./deploy-dashboard.sh docker
```

#### Using PowerShell (Windows)

```powershell
# Deploy to staging
.\deploy-dashboard.ps1 -Environment staging

# Deploy to production
.\deploy-dashboard.ps1 -Environment production

# Build only (no deployment)
.\deploy-dashboard.ps1 -Environment staging -BuildOnly $true

# Deploy with Docker
.\deploy-dashboard.ps1 -Environment docker
```

### 3. Vercel Deployment

#### Prerequisites
- Vercel account
- Vercel CLI installed: `npm install -g vercel`

#### Configuration
The `vercel.json` file is configured to deploy only the dashboard:

```json
{
  "version": 2,
  "name": "computer-genie-dashboard",
  "builds": [
    {
      "src": "computer-genie-dashboard/package.json",
      "use": "@vercel/next"
    }
  ]
}
```

#### Manual Vercel Deployment
```bash
cd computer-genie-dashboard
vercel --prod  # For production
vercel         # For preview
```

### 4. Docker Deployment

#### Using Docker Compose
```bash
# Start dashboard with Docker Compose
docker-compose -f docker-compose.dashboard.yml up -d

# Stop dashboard
docker-compose -f docker-compose.dashboard.yml down

# View logs
docker-compose -f docker-compose.dashboard.yml logs -f dashboard
```

#### Using Docker directly
```bash
# Build image
docker build -t computer-genie/dashboard ./computer-genie-dashboard

# Run container
docker run -d \
  --name dashboard \
  -p 3000:3000 \
  -e NODE_ENV=production \
  computer-genie/dashboard
```

#### Production Docker Setup with Nginx
```bash
# Start with production profile (includes Nginx)
docker-compose -f docker-compose.dashboard.yml --profile production up -d

# This will start:
# - Dashboard application (port 3000)
# - Nginx reverse proxy (ports 80/443)
```

## Environment Configuration

### Environment Variables

Create a `.env.local` file in the `computer-genie-dashboard/` directory:

```env
# Application
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0

# Database (if needed)
DATABASE_URL=postgresql://user:password@localhost:5432/dashboard

# Redis (if needed)
REDIS_URL=redis://localhost:6379

# Authentication
NEXTAUTH_SECRET=your-secret-key
NEXTAUTH_URL=https://your-dashboard-domain.com

# API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
```

### Production Environment Setup

1. **Domain Configuration**
   - Point your domain to the deployment server
   - Configure SSL certificates
   - Update CORS settings if needed

2. **Database Setup** (if using external database)
   ```bash
   # PostgreSQL example
   createdb computer_genie_dashboard
   ```

3. **Redis Setup** (if using session storage)
   ```bash
   # Start Redis
   redis-server
   ```

## Monitoring and Health Checks

### Health Check Endpoint
The dashboard includes a health check endpoint at `/api/health`:

```bash
curl http://localhost:3000/api/health
```

### Docker Health Checks
The Docker setup includes automatic health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Monitoring with Docker Compose
```bash
# Check service status
docker-compose -f docker-compose.dashboard.yml ps

# View resource usage
docker stats dashboard

# Check logs
docker-compose -f docker-compose.dashboard.yml logs dashboard
```

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Clear cache and rebuild
   cd computer-genie-dashboard
   rm -rf .next node_modules
   npm install
   npm run build
   ```

2. **Port Conflicts**
   ```bash
   # Check what's using port 3000
   lsof -i :3000
   
   # Kill process if needed
   kill -9 <PID>
   ```

3. **Docker Issues**
   ```bash
   # Clean up Docker resources
   docker system prune -a
   
   # Rebuild without cache
   docker build --no-cache -t computer-genie/dashboard ./computer-genie-dashboard
   ```

4. **Permission Issues**
   ```bash
   # Fix script permissions
   chmod +x deploy-dashboard.sh
   
   # Fix Docker permissions (Linux)
   sudo usermod -aG docker $USER
   ```

### Logs and Debugging

1. **Application Logs**
   ```bash
   # Docker logs
   docker logs dashboard
   
   # PM2 logs (if using PM2)
   pm2 logs dashboard
   ```

2. **Nginx Logs** (if using Nginx)
   ```bash
   # Access logs
   docker exec dashboard-nginx tail -f /var/log/nginx/access.log
   
   # Error logs
   docker exec dashboard-nginx tail -f /var/log/nginx/error.log
   ```

## Security Considerations

### Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Configure proper CORS settings
- [ ] Set secure environment variables
- [ ] Enable security headers (configured in Nginx)
- [ ] Regular security audits (`npm audit`)
- [ ] Keep dependencies updated
- [ ] Use secrets management for sensitive data

### Network Security
```bash
# Configure firewall (example for Ubuntu)
sudo ufw allow 80
sudo ufw allow 443
sudo ufw deny 3000  # Don't expose Node.js directly
```

## Scaling and Performance

### Horizontal Scaling
```yaml
# docker-compose.dashboard.yml - multiple instances
services:
  dashboard:
    deploy:
      replicas: 3
    # ... other configuration
```

### Load Balancing
Configure Nginx for load balancing multiple dashboard instances:

```nginx
upstream dashboard {
    server dashboard-1:3000;
    server dashboard-2:3000;
    server dashboard-3:3000;
}
```

### Performance Monitoring
- Use the built-in performance monitoring features
- Monitor with external tools (New Relic, DataDog, etc.)
- Set up alerts for high resource usage

## Backup and Recovery

### Database Backup (if applicable)
```bash
# PostgreSQL backup
pg_dump computer_genie_dashboard > dashboard_backup.sql

# Restore
psql computer_genie_dashboard < dashboard_backup.sql
```

### Configuration Backup
```bash
# Backup environment files
cp computer-genie-dashboard/.env.local backup/
cp docker-compose.dashboard.yml backup/
```

## Support and Maintenance

### Regular Maintenance Tasks
1. Update dependencies monthly
2. Review and rotate secrets quarterly
3. Monitor disk space and logs
4. Test backup and recovery procedures
5. Review security audit reports

### Getting Help
- Check the main project documentation
- Review GitHub Issues
- Contact the development team
- Check deployment logs for specific errors

## Migration from Other Setups

### From Full Stack Deployment
If you were previously deploying the entire repository:

1. Stop the full deployment
2. Extract dashboard-specific environment variables
3. Use the dashboard-only deployment methods above
4. Update DNS/proxy settings to point to new dashboard deployment

### From Development to Production
1. Update environment variables for production
2. Configure production database connections
3. Set up SSL certificates
4. Configure monitoring and alerting
5. Test all functionality in production environment

---

For more information about the dashboard features and development, see the main [README.md](computer-genie-dashboard/README.md) and [ENTERPRISE_FEATURES.md](computer-genie-dashboard/ENTERPRISE_FEATURES.md) files.