# Computer Genie Dashboard - GitHub Push Script
# Run this script to push your code to GitHub

Write-Host "🚀 Computer Genie Dashboard - GitHub Push Script" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check Git configuration
Write-Host ""
Write-Host "Checking Git configuration..." -ForegroundColor Yellow

$userName = git config --global user.name
$userEmail = git config --global user.email

if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Host ""
    Write-Host "⚠️  Git is not configured!" -ForegroundColor Yellow
    Write-Host ""
    
    # Ask for user details
    $userName = Read-Host "Enter your name (e.g., John Doe)"
    $userEmail = Read-Host "Enter your email (GitHub email)"
    
    # Configure Git
    git config --global user.name "$userName"
    git config --global user.email "$userEmail"
    
    Write-Host ""
    Write-Host "✓ Git configured successfully!" -ForegroundColor Green
} else {
    Write-Host "✓ Git already configured" -ForegroundColor Green
    Write-Host "  Name: $userName" -ForegroundColor Gray
    Write-Host "  Email: $userEmail" -ForegroundColor Gray
}

# Ask for GitHub username
Write-Host ""
Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan
Write-Host ""

$githubUsername = Read-Host "Enter your GitHub username"

if ([string]::IsNullOrWhiteSpace($githubUsername)) {
    Write-Host "✗ GitHub username is required!" -ForegroundColor Red
    exit 1
}

$repoUrl = "https://github.com/$githubUsername/computer-genie-dashboard.git"

Write-Host ""
Write-Host "Repository URL: $repoUrl" -ForegroundColor Gray
Write-Host ""

# Check if repository exists on GitHub
Write-Host "⚠️  IMPORTANT: Make sure you have created the repository on GitHub!" -ForegroundColor Yellow
Write-Host "   Go to: https://github.com/new" -ForegroundColor Yellow
Write-Host "   Repository name: computer-genie-dashboard" -ForegroundColor Yellow
Write-Host "   DO NOT initialize with README, .gitignore, or license" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Have you created the repository on GitHub? (yes/no)"

if ($confirm -ne "yes" -and $confirm -ne "y") {
    Write-Host ""
    Write-Host "Please create the repository first, then run this script again." -ForegroundColor Yellow
    Write-Host "Opening GitHub in browser..." -ForegroundColor Gray
    Start-Process "https://github.com/new"
    exit 0
}

# Check if remote already exists
$remoteExists = git remote get-url origin 2>$null

if ($remoteExists) {
    Write-Host ""
    Write-Host "⚠️  Remote 'origin' already exists: $remoteExists" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to overwrite it? (yes/no)"
    
    if ($overwrite -eq "yes" -or $overwrite -eq "y") {
        git remote remove origin
        Write-Host "✓ Removed existing remote" -ForegroundColor Green
    } else {
        Write-Host "Keeping existing remote" -ForegroundColor Gray
    }
}

# Add remote
Write-Host ""
Write-Host "Adding GitHub remote..." -ForegroundColor Yellow

try {
    git remote add origin $repoUrl
    Write-Host "✓ Remote added successfully" -ForegroundColor Green
} catch {
    Write-Host "Note: Remote might already exist" -ForegroundColor Gray
}

# Check for uncommitted changes
Write-Host ""
Write-Host "Checking for uncommitted changes..." -ForegroundColor Yellow

$status = git status --porcelain

if ($status) {
    Write-Host "Found uncommitted changes. Committing..." -ForegroundColor Yellow
    git add .
    git commit -m "docs: Add GitHub push script and final updates"
    Write-Host "✓ Changes committed" -ForegroundColor Green
} else {
    Write-Host "✓ No uncommitted changes" -ForegroundColor Green
}

# Rename branch to main
Write-Host ""
Write-Host "Setting branch to 'main'..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Branch set to 'main'" -ForegroundColor Green

# Push to GitHub
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  You will be prompted for credentials:" -ForegroundColor Yellow
Write-Host "   Username: Your GitHub username" -ForegroundColor Yellow
Write-Host "   Password: Your Personal Access Token (NOT your password!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Don't have a token? Create one here:" -ForegroundColor Yellow
Write-Host "https://github.com/settings/tokens/new" -ForegroundColor Yellow
Write-Host ""

$pushConfirm = Read-Host "Ready to push? (yes/no)"

if ($pushConfirm -ne "yes" -and $pushConfirm -ne "y") {
    Write-Host ""
    Write-Host "Push cancelled. You can push manually later with:" -ForegroundColor Yellow
    Write-Host "git push -u origin main" -ForegroundColor Gray
    exit 0
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host ""

try {
    git push -u origin main
    
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host "🎉 SUCCESS! Code pushed to GitHub!" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your repository is now live at:" -ForegroundColor Cyan
    Write-Host "https://github.com/$githubUsername/computer-genie-dashboard" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Visit your repository on GitHub" -ForegroundColor Gray
    Write-Host "2. Add a description and topics" -ForegroundColor Gray
    Write-Host "3. Enable GitHub Pages (optional)" -ForegroundColor Gray
    Write-Host "4. Set up branch protection rules" -ForegroundColor Gray
    Write-Host "5. Deploy to Vercel: vercel --prod" -ForegroundColor Gray
    Write-Host ""
    
    # Ask to open in browser
    $openBrowser = Read-Host "Open repository in browser? (yes/no)"
    if ($openBrowser -eq "yes" -or $openBrowser -eq "y") {
        Start-Process "https://github.com/$githubUsername/computer-genie-dashboard"
    }
    
} catch {
    Write-Host ""
    Write-Host "=================================================" -ForegroundColor Red
    Write-Host "✗ Push failed!" -ForegroundColor Red
    Write-Host "=================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "1. Repository doesn't exist on GitHub" -ForegroundColor Gray
    Write-Host "2. Wrong credentials (use Personal Access Token, not password)" -ForegroundColor Gray
    Write-Host "3. No internet connection" -ForegroundColor Gray
    Write-Host "4. Repository already has content" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Error details:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Need help? Check PUSH_TO_GITHUB.md for detailed instructions" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Script completed! 🚀" -ForegroundColor Green
Write-Host ""
