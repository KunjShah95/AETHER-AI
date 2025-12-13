# Docker Configuration for NEXUS-AI

This document explains the Docker setup for the NEXUS-AI project, including TypeScript configuration fixes and container orchestration.

## Table of Contents

1. [TypeScript Configuration Fixes](#typescript-configuration-fixes)
2. [Docker Architecture](#docker-architecture)
3. [Getting Started](#getting-started)
4. [Development Workflow](#development-workflow)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## TypeScript Configuration Fixes

### Issues Resolved

The TypeScript configuration in `frontend/tsconfig.json` has been fixed to address the following Microsoft Edge Tools warnings:

#### 1. Force Consistent Casing in File Names

```json
"forceConsistentCasingInFileNames": true
```

**Why this was needed:**.

- Prevents cross-platform issues when working with different operating systems
- Ensures file name casing is consistent across Windows, macOS, and Linux
- Reduces import/export errors caused by case sensitivity differences
- Improves code reliability in team environments

#### 2. Strict Type Checking

```json
"strict": true
```

**Why this was needed:**

- Enables comprehensive type checking to catch type errors early
- Reduces runtime errors and improves code quality
- Provides better IntelliSense and error detection
- Makes the codebase more maintainable and easier to refactor

### Configuration Details

The root `tsconfig.json` now includes:

```json
{
  "files": [],
  "compilerOptions": {
    "strict": true,
    "forceConsistentCasingInFileNames": true
  },
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

This configuration is inherited by the referenced TypeScript configuration files (`tsconfig.app.json` and `tsconfig.node.json`), ensuring consistent type checking across the entire frontend application.

## Docker Architecture

### Multi-Stage Build Process

The Docker setup uses a multi-stage build process to optimize image size and separation of concerns:

``` text
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Frontend Build    │    │   Terminal Build    │    │  Production Runtime │
│   (Node.js 18)      │    │  (Python 3.11)      │    │   (Python 3.11)     │
│                     │    │                     │    │                     │
│ • Install npm deps  │───▶│ • Install pip deps  │───▶│ • Copy built assets │
│ • Build React app   │    │ • Install system    │    │ • Copy Python app   │
│ • Generate static   │    │   dependencies      │    │ • Setup user        │
│   files             │    │                     │    │ • Configure runtime │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Container Services

The docker-compose.yml defines the following services:

#### Core Application

- **aetherai**: Main application container
  - Multi-stage built image
  - Serves both frontend (port 3000) and backend (port 8000)
  - Production-ready configuration

#### Development Services

- **frontend-dev**: Frontend development environment
  - Hot reloading with Vite
  - Development dependencies included
  - Interactive development mode

- **terminal-dev**: Terminal development environment
  - Python development with hot reloading
  - Full development toolchain
  - Interactive terminal access

#### Supporting Services

- **redis**: Caching and session management
  - Persistent data storage
  - Optimized configuration
  - Health checks enabled

- **chromadb**: Vector database for RAG
  - Knowledge base storage
  - Vector embeddings support
  - API endpoints for search

- **nginx**: Production reverse proxy
  - SSL termination
  - Static file serving
  - Load balancing
  - Rate limiting

## Getting Started

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Quick Start

#### Development Mode

```bash
# Start development environment
docker-compose --profile dev up

# Access services:
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# Terminal dev: http://localhost:8001
```

#### Production Mode

```bash
# Build and start production services
docker-compose --profile production up -d

# Access services:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys (at least one required)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
OLLAMA_HOST=localhost:11434

# Database Configuration
REDIS_URL=redis://redis:6379
CHROMADB_URL=http://chromadb:8000

# Security
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key

# Development
DEBUG=false
LOG_LEVEL=INFO
```

## Development Workflow

### Frontend Development

```bash
# Start frontend development container
docker-compose --profile dev up frontend-dev

# The container supports hot reloading
# Changes to frontend code are reflected immediately
```

### Terminal Development

```bash
# Start terminal development container
docker-compose --profile dev up terminal-dev

# Access interactive terminal
docker exec -it aetherai-terminal-dev bash
```

### Testing

```bash
# Run tests in container
docker-compose exec terminal-dev python -m pytest

# Run frontend tests
docker-compose exec frontend-dev npm test
```

## Production Deployment

### Building Images

```bash
# Build production image
docker build -t aetherai:latest .

# Build specific target
docker build --target production -t aetherai:prod .
```

### Deployment Options

#### Docker Compose (Single Host)

```bash
docker-compose up -d
```

#### Kubernetes (Multi-Host)

```bash
# Convert compose to Kubernetes manifests
kompose convert -f docker-compose.yml

# Apply to cluster
kubectl apply -f .
```

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml aetherai
```

## Configuration

### Customization

#### Environment-Specific Configs

- Development: Uses `Dockerfile.dev` files with hot reloading
- Production: Uses optimized multi-stage builds
- Staging: Can be added with custom docker-compose files

#### Resource Limits

Edit `docker-compose.yml` to adjust resource constraints:

```yaml
services:
  aetherai:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

#### Scaling

```bash
# Scale specific services
docker-compose up --scale aetherai=3 -d
```

### Volume Mounts

- **Configuration**: `./config:/app/config:ro`
- **Data**: `./data:/app/data`
- **Logs**: `./logs:/app/logs`
- **User Data**: `~/.aetherai:/home/aetherai/.aetherai`

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check what's using a port
lsof -i :3000

# Stop conflicting services
docker-compose down
```

#### Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x *.sh
```

#### Memory Issues

```bash
# Monitor container resources
docker stats

# Adjust memory limits in compose file
```

#### Build Failures

```bash
# Clear Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose logs aetherai
```

### Logs

```bash
# View logs for all services
docker-compose logs

# Follow logs for specific service
docker-compose logs -f aetherai
```

### Debugging

#### Enter Containers

```bash
# Frontend container
docker exec -it aetherai-frontend-dev sh

# Terminal container
docker exec -it aetherai-terminal-dev bash

# Redis container
docker exec -it aetherai-redis redis-cli
```

#### Network Debugging

```bash
# Test connectivity between containers
docker-compose exec aetherai ping chromadb
docker-compose exec aetherai curl http://chromadb:8000/api/v1/heartbeat
```

## Security Considerations

### Production Hardening

1. **Non-root User**: All containers run as non-root users
2. **Read-only Filesystems**: Configuration mounted as read-only
3. **Resource Limits**: CPU and memory constraints
4. **Network Isolation**: Custom bridge network
5. **Secrets Management**: Use Docker secrets or external secret management

### SSL/TLS Configuration

Uncomment SSL sections in nginx.conf and provide certificates:

```bash
# Generate self-signed certificate for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/nginx/ssl/key.pem \
  -out config/nginx/ssl/cert.pem
```

### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
```

## Performance Optimization

### Image Size Optimization

- Multi-stage builds reduce final image size
- Alpine Linux base images where possible
- Minimal runtime dependencies
- Efficient layer caching

### Runtime Optimization

- Connection pooling for databases
- Redis for session storage
- CDN integration for static assets
- Horizontal scaling support

### Monitoring

```bash
# Resource usage
docker stats

# Container metrics
docker-compose exec aetherai python -c "import psutil; print(psutil.cpu_percent())"
```

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild with latest code
docker-compose up -d --build

# Clean up old images
docker image prune -f
```

### Backup

```bash
# Backup volumes
docker run --rm -v aetherai_redis-data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz -C /data .

# Backup ChromaDB
docker run --rm -v aetherai_chromadb-data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb-backup.tar.gz -C /data .
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Complete system cleanup
docker system prune -a
