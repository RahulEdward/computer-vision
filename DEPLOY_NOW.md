# 🚀 Deploy Kaise Karein - Step by Step

## Option 1: Vercel (Sabse Aasan) ⭐ RECOMMENDED

### Step 1: Vercel Account
1. https://vercel.com pe jao
2. GitHub se sign up karo
3. Free plan select karo

### Step 2: Database Setup
1. https://vercel.com/storage/postgres pe jao
2. "Create Database" click karo
3. Database name do: `computer-genie-prod`
4. Region select karo (closest to users)
5. Create karo

### Step 3: Get Database URL
1. Database dashboard mein jao
2. "Connection String" copy karo
3. Yeh aapka `DATABASE_URL` hai

### Step 4: Deploy Project
```bash
# Terminal mein:
cd computer-genie-dashboard
npm install -g vercel
vercel login
vercel
```

### Step 5: Environment Variables
Vercel dashboard mein:
1. Project Settings → Environment Variables
2. Add karo:
   - `DATABASE_URL` = (step 3 se)
   - `NEXTAUTH_SECRET` = (generate karo: `openssl rand -base64 32`)
   - `NEXTAUTH_URL` = (Vercel URL milega)

### Step 6: Deploy
```bash
vercel --prod
```

**Done! 🎉**

---

## Option 2: Railway (Easy + Free Tier)

### Step 1: Railway Account
1. https://railway.app pe jao
2. GitHub se sign up karo

### Step 2: New Project
1. "New Project" click karo
2. "Deploy from GitHub repo" select karo
3. Repository select karo

### Step 3: Add PostgreSQL
1. "New" → "Database" → "PostgreSQL"
2. Automatically DATABASE_URL mil jayega

### Step 4: Environment Variables
Railway dashboard mein add karo:
- `NEXTAUTH_SECRET`
- `NEXTAUTH_URL`
- Other variables

### Step 5: Deploy
- Automatic deploy hoga!

---

## Option 3: DigitalOcean App Platform

### Step 1: Account
1. https://www.digitalocean.com pe jao
2. Sign up karo ($200 free credit)

### Step 2: Create App
1. "Create" → "Apps"
2. GitHub repo connect karo
3. Branch select karo (main)

### Step 3: Database
1. "Create" → "Databases" → "PostgreSQL"
2. Connection string copy karo

### Step 4: Configure
1. Environment variables add karo
2. Build command: `npm run build`
3. Run command: `npm start`

### Step 5: Deploy
- Click "Create Resources"

---

## Option 4: AWS (Advanced)

### Requirements:
- AWS Account
- EC2 instance
- RDS PostgreSQL
- S3 for static files
- CloudFront for CDN

### Quick Setup:
```bash
# EC2 pe:
git clone your-repo
cd computer-genie-dashboard
npm install
npm run build

# PM2 se run karo:
npm install -g pm2
pm2 start npm --name "computer-genie" -- start
pm2 save
pm2 startup
```

---

## Pre-Deployment Checklist ✅

### 1. Environment Variables
- [ ] DATABASE_URL set hai
- [ ] NEXTAUTH_SECRET strong hai
- [ ] NEXTAUTH_URL correct hai
- [ ] NODE_ENV=production

### 2. Database
- [ ] PostgreSQL setup hai
- [ ] Migrations run ho gaye
- [ ] Connection test kiya

### 3. Security
- [ ] Secrets secure hain
- [ ] CORS configured hai
- [ ] Rate limiting enabled hai

### 4. Performance
- [ ] Build successful hai
- [ ] No errors in console
- [ ] Images optimized hain

### 5. Testing
- [ ] Login/Signup working
- [ ] Dashboard loading
- [ ] Workflows accessible
- [ ] APIs responding

---

## Post-Deployment Steps

### 1. DNS Setup
```
A Record: @ → Your server IP
CNAME: www → your-domain.com
```

### 2. SSL Certificate
- Vercel: Automatic
- Railway: Automatic
- Others: Use Let's Encrypt

### 3. Monitoring
- Set up error tracking
- Enable analytics
- Configure alerts

### 4. Backup
- Database backups daily
- Code in Git
- Environment variables documented

---

## Quick Commands

### Build for Production
```bash
npm run build
```

### Start Production Server
```bash
npm start
```

### Database Migration
```bash
npx prisma migrate deploy
```

### Generate Prisma Client
```bash
npx prisma generate
```

### Check Build
```bash
npm run build && npm start
```

---

## Troubleshooting

### Build Fails?
```bash
# Clear cache
rm -rf .next
rm -rf node_modules
npm install
npm run build
```

### Database Connection Error?
- Check DATABASE_URL format
- Verify database is running
- Check firewall rules

### 500 Error?
- Check server logs
- Verify environment variables
- Check database connection

---

## Cost Estimates

### Vercel
- Free: Hobby projects
- Pro: $20/month
- Enterprise: Custom

### Railway
- Free: $5 credit/month
- Developer: $10/month
- Team: $20/month

### DigitalOcean
- Basic: $12/month (app + database)
- Professional: $24/month
- Business: $48/month

### AWS
- Variable: $20-100/month
- Depends on usage

---

## Support

### Issues?
1. Check logs
2. Verify environment variables
3. Test database connection
4. Check deployment status

### Need Help?
- Vercel: https://vercel.com/support
- Railway: https://railway.app/help
- DigitalOcean: https://www.digitalocean.com/support

---

## Ready to Deploy? 🚀

**Recommended Path:**
1. Use Vercel (easiest)
2. Set up PostgreSQL database
3. Configure environment variables
4. Deploy with one command
5. Done in 15 minutes!

**Let's go!** 🎉
