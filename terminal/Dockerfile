# Dockerfile for AETHER-AI Terminal Assistant
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY terminal/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (for pyproject.toml and terminal code)
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create .aether directory for application data
RUN mkdir -p /root/.aether

# Install the application
RUN pip install -e .

# Set working directory to terminal for tests
WORKDIR /app/terminal

# Default command - make container interactive by default
CMD ["bash"]
