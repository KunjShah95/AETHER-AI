# Docker Deployment Guide

This guide explains how to deploy the AetherAI project using separate Dockerfiles for frontend and backend.

## 🐳 Docker Architecture

The project is now split into separate Docker containers:

- **Frontend**: React/TypeScript application served by Nginx
- **Backend**: Python terminal application with AI services
- **Nginx**: Reverse proxy for routing traffic
- **Redis**: Optional cache service

## 📁 Docker Files Structure

```text
NEXUS-AI.io/
├── docker-compose.yml              # Main compose file
├── frontend/
│   ├── Dockerfile              # Frontend production build
│   ├── Dockerfile.dev         # Frontend development build
│   ├── .dockerignore        # Frontend ignore rules
│   └── nginx.conf           # Nginx configuration
└── terminal/
    ├── Dockerfile           # Backend production build
    ├── Dockerfile.dev      # Backend development build
    └── .dockerignore     # Backend ignore rules
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```env
# Frontend Environment
NODE_ENV=production

# Backend Environment
PYTHON_ENV=production

# API Keys (required for backend)
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_hf_token
MCP_API_KEY=your_mcp_api_key
```

### 2. Build and Run All Services

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 3. Individual Service Management

```bash
# Build only frontend
docker-compose build frontend

# Build only backend
docker-compose build backend

# Run only frontend
docker-compose up frontend

# Run only backend
docker-compose up backend

# View specific service logs
docker-compose logs -f frontend
docker-compose logs -f backend
```

## 🔧 Development vs Production

### Development Mode

For development, use the dev Dockerfiles:

```bash
# Development build (frontend)
cd frontend
docker build -f Dockerfile.dev -t aetherai-frontend:dev .

# Development build (backend)
cd terminal
docker build -f Dockerfile.dev -t aetherai-backend:dev .
```

### Production Mode

For production, use the production Dockerfiles:

```bash
# Production build (frontend)
cd frontend
docker build -t aetherai-frontend:prod .

# Production build (backend)
cd terminal
docker build -t aetherai-backend:prod .
```

## 🌐 Service Ports

- **Frontend**: `http://localhost:3000`
- **Backend**: `http://localhost:8000`
- **Nginx**: `http://localhost:80`
- **Redis**: `http://localhost:6379`

## 📦 Individual Docker Builds

### Frontend Only

```bash
# Build frontend
cd frontend
docker build -t aetherai-frontend .

# Run frontend container
docker run -p 3000:80 aetherai-frontend
```

### Backend Only

```bash
# Build backend
cd terminal
docker build -t aetherai-backend .

# Run backend container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  aetherai-backend
```

## 🔍 Docker Commands Reference

### Container Management

```bash
# List containers
docker ps

# View container logs
docker logs aetherai-frontend
docker logs aetherai-backend

# Execute commands in running container
docker exec -it aetherai-frontend /bin/sh
docker exec -it aetherai-backend /bin/bash

# Stop and remove containers
docker-compose down

# Remove images
docker rmi aetherai-frontend aetherai-backend
```

### Development Commands

```bash
# Rebuild specific service
docker-compose build --no-cache frontend
docker-compose build --no-cache backend

# View real-time logs
docker-compose logs -f --tail=100

# Scale services (if needed)
docker-compose up --scale backend=2
```

## 🏗️ Dockerfile Details

### Frontend Dockerfile

- **Base**: Node.js 18 Alpine
- **Purpose**: Build React app and serve with Nginx
- **Port**: 80 (HTTP)
- **Features**:
  - Multi-stage build
  - Optimized production serving
  - Security headers
  - Gzip compression

### Backend Dockerfile

- **Base**: Python 3.11 Slim
- **Purpose**: Run Python terminal application
- **Port**: 8000 (HTTP)
- **Features**:
  - Non-root user
  - System dependencies for audio/voice
  - Health checks
  - Volume mounts for data persistence

## 🔐 Security Features

### Frontend

- Non-root user execution
- Minimal attack surface
- Security headers
- Asset caching

### Backend

- Non-root user execution
- System dependencies properly isolated
- Environment variables for secrets
- Health check monitoring

## 📊 Monitoring and Health Checks

### Health Checks

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl http://localhost:8000/health
```

### Resource Usage

```bash
# View resource usage
docker stats

# View container info
docker inspect aetherai-frontend
docker inspect aetherai-backend
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in `docker-compose.yml`
2. **Permission errors**: Check volume mount permissions
3. **API key errors**: Verify `.env` file is properly loaded
4. **Build failures**: Check Docker daemon is running

### Debug Commands

```bash
# Debug frontend build
docker build --no-cache frontend

# Debug backend build
docker build --no-cache backend

# View detailed logs
docker-compose logs --details

# Shell access for debugging
docker exec -it aetherai-frontend /bin/sh
docker exec -it aetherai-backend /bin/bash
```

## 🔄 Deployment Strategies

### Local Development

```bash
docker-compose -f docker-compose.yml up --build
```

### Production Deployment

```bash
# With environment-specific configs
docker-compose -f docker-compose.prod.yml up --build -d
```

### CI/CD Integration

```bash
# Build and push to registry
docker build -t registry/aetherai-frontend:latest frontend
docker build -t registry/aetherai-backend:latest terminal
```

## 📝 Best Practices

1. **Use `.dockerignore` files** to reduce build context
2. **Leverage multi-stage builds** for optimization
3. **Run as non-root users** for security
4. **Use health checks** for monitoring
5. **Separate development and production** configurations
6. **Use environment variables** for configuration
7. **Implement proper logging** and monitoring

## 🎯 Next Steps

1. Set up SSL certificates for production
2. Configure load balancing if needed
3. Set up monitoring and alerting
4. Implement backup strategies for data volumes
5. Configure CI/CD pipelines for automated builds
