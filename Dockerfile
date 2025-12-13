# Multi-stage Dockerfile for NEXUS-AI project
# This Dockerfile builds both the frontend and terminal components

# ==========================================
# Frontend Build Stage
# ==========================================
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ .

# Build the frontend application
RUN npm run build

# ==========================================
# Terminal Build Stage
# ==========================================
FROM python:3.11-slim AS terminal-builder

# Set working directory for terminal
WORKDIR /app/terminal

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libsndfile1-dev \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY terminal/requirements.txt terminal/requirements.txt
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy terminal source code
COPY terminal/ .

# ==========================================
# Production Runtime Stage
# ==========================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NODE_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r aetherai && useradd -r -g aetherai aetherai

# Set working directory
WORKDIR /app

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy Python application from builder stage
COPY --from=terminal-builder /app/terminal ./terminal
COPY --from=terminal-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy project root files
COPY pyproject.toml .
COPY setup.cfg .

# Install the Python package
RUN pip install -e .

# Change ownership to app user
RUN chown -R aetherai:aetherai /app

# Switch to app user
USER aetherai

# Expose ports
EXPOSE 3000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Default command
CMD ["aetherai", "--help"]
