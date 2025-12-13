# Frontend Deployment Guide

This guide addresses common frontend deployment issues and provides solutions for production deployment.

## 🔧 Fixed Issues

### 1. **Production Dockerfile Added**

- Created `frontend/Dockerfile` for production deployment
- Multi-stage build with Node.js builder and Nginx server
- Optimized for production with proper caching

### 2. **Vite Configuration Enhanced**

- Updated `frontend/vite.config.ts` with:
  - Better chunk splitting for optimal loading
  - Proper build optimization
  - Development and preview server configuration
  - Environment variable handling

### 3. **Vercel Configuration Fixed**

- Updated `vercel.json` with proper monorepo support
- Added security headers
- Improved build command using `npm ci` instead of `npm install`
- Added redirects for API routes

### 4. **Package Scripts Enhanced**

- Added `build:check`, `lint:fix`, `type-check`, and `clean` scripts
- Improved build process with TypeScript checking

### 5. **Docker Optimization**

- Created `.dockerignore` file to reduce build context
- Proper nginx configuration with gzip compression and caching

## 🚀 Deployment Options

### Option 1: Vercel Deployment (Recommended)

1. **Connect your repository to Vercel**
2. **Vercel will automatically detect the configuration**
3. **Build settings are pre-configured in `vercel.json`**

```bash
# Vercel will run these commands automatically:
cd frontend && npm ci && npm run build
```

### Option 2: Docker Deployment

```bash
# Build the Docker image
docker build -t nexus-ai-frontend ./frontend

# Run the container
docker run -p 80:80 nexus-ai-frontend
```

### Option 3: Manual Build and Deploy

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run type checking
npm run type-check

# Build for production
npm run build

# The built files will be in the 'dist' directory
# Deploy the 'dist' directory to your web server
```

## 🛠️ Common Issues and Solutions

### Issue 1: Build Fails with TypeScript Errors

**Solution:** Run type checking before build

```bash
cd frontend
npm run type-check
npm run build
```

### Issue 2: Vercel Build Hangs or Fails

**Solution:** Ensure Node.js version compatibility

- Vercel uses Node.js 18.x by default
- Check `package.json` for compatibility

### Issue 3: Environment Variables Not Working

**Solution:** Use Vite environment variable prefixes

- Variables must start with `VITE_` to be accessible in the browser
- Update your environment variables:

```env
VITE_API_URL=your_api_url
VITE_APP_NAME=NEXUS AI
```

### Issue 4: Routing Issues (404 on Refresh)

**Solution:** Ensure SPA routing configuration

- Vercel configuration includes proper rewrites
- Nginx configuration handles client-side routing

## 📁 Project Structure

```text
NEXUS-AI.io/
├── frontend/
│   ├── src/                    # Source code
│   ├── public/                 # Static assets
│   ├── dist/                   # Build output
│   ├── package.json           # Dependencies and scripts
│   ├── vite.config.ts         # Vite configuration
│   ├── Dockerfile             # Production Docker build
│   ├── Dockerfile.dev         # Development Docker build
│   ├── nginx.conf             # Nginx configuration
│   └── .dockerignore          # Docker ignore rules
├── vercel.json                # Vercel deployment config
└── DEPLOYMENT.md             # This file
```

## 🔍 Pre-deployment Checklist

- [ ] Node.js version is 18+
- [ ] All dependencies are compatible
- [ ] TypeScript compilation passes
- [ ] Environment variables are properly prefixed with `VITE_`
- [ ] Build script runs without errors
- [ ] Docker build works (if using Docker)
- [ ] Vercel configuration is correct

## 🐛 Troubleshooting

### Build Fails

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run type-check

# Run linter
npm run lint
```

### Docker Issues

```bash
# Build with verbose output
docker build --no-cache -t nexus-ai-frontend ./frontend

# Check Docker logs
docker logs <container-id>
```

### Vercel Issues

1. Check build logs in Vercel dashboard
2. Ensure `vercel.json` is in the root directory
3. Verify Node.js version compatibility
4. Check environment variables in Vercel dashboard

## 📝 Environment Variables

Update your environment variables to use Vite prefixes:

```env
# Development (.env.local)
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=NEXUS AI (Dev)

# Production (.env.production)
VITE_API_URL=https://your-api-domain.com
VITE_APP_NAME=NEXUS AI
```

## 🎯 Performance Optimization

The configuration includes:

- **Code splitting** for optimal loading
- **Gzip compression** via Nginx
- **Asset caching** with long-term cache headers
- **Security headers** for production
- **Bundle optimization** with rollup

## 📞 Support

If you continue to experience issues:

1. Check the build logs for specific error messages
2. Verify all dependencies are compatible
3. Ensure proper environment variable configuration
4. Test locally before deploying
