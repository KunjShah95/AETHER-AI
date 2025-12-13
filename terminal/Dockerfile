# Production Dockerfile for backend (AetherAI Terminal)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    libsndfile1-dev \
    libasound2-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aetherai && useradd -r -g aetherai aetherai

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/.nexus && \
    chown -R aetherai:aetherai /app

# Switch to non-root user
USER aetherai

# Expose port for web server (if using FastAPI)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command - run the main application
CMD ["python", "main.py", "--cli"]
