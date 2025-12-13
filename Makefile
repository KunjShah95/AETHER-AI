.PHONY: help build up down logs shell clean test pypi-build pypi-test pypi-publish

# Default target
help:
	@echo "AetherAI Management"
	@echo "==================="
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build       - Build Docker image"
	@echo "  make up          - Start containers in detached mode"
	@echo "  make down        - Stop and remove containers"
	@echo "  make logs        - View container logs"
	@echo "  make shell       - Open shell in running container"
	@echo "  make clean       - Remove containers, volumes, and images"
	@echo "  make test        - Test Docker image"
	@echo "  make rebuild     - Clean and rebuild everything"
	@echo "  make push        - Push image to registry"
	@echo ""
	@echo "PyPI Publishing:"
	@echo "  make pypi-build   - Build wheel and sdist"
	@echo "  make pypi-check   - Verify package with twine"
	@echo "  make pypi-test    - Upload to TestPyPI"
	@echo "  make pypi-publish - Upload to PyPI (production)"
	@echo "  make pypi-clean   - Remove build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make dev-install  - Install in editable mode with dev deps"
	@echo "  make lint         - Run linters (ruff, black, mypy)"
	@echo "  make format       - Auto-format code with black"
	@echo "  make test-py      - Run pytest"
	@echo ""

# Build Docker image
build:
	@echo "Building AetherAI Docker image..."
	docker-compose build --no-cache

# Start containers
up:
	@echo "Starting AetherAI containers..."
	docker-compose up -d
	@echo "Containers started! Use 'make logs' to view output"

# Stop containers
down:
	@echo "Stopping AetherAI containers..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Open shell in container
shell:
	docker exec -it aetherai_terminal /bin/bash

# Run the application interactively
run:
	docker exec -it aetherai_terminal python terminal/main.py

# Clean up everything
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --rmi all
	@echo "Cleanup complete!"

# Test the image
test:
	@echo "Testing AetherAI Docker image..."
	docker-compose run --rm aetherai python -c "import sys; print(f'Python {sys.version}'); sys.exit(0)"
	@echo "Test passed!"

# Rebuild everything from scratch
rebuild: clean build up

# Push to registry (requires login)
push:
	@echo "Pushing image to GitHub Container Registry..."
	docker tag aetherai:latest ghcr.io/kunjshah95/aetherai:latest
	docker push ghcr.io/kunjshah95/aetherai:latest

# Pull from registry
pull:
	@echo "Pulling latest image from registry..."
	docker pull ghcr.io/kunjshah95/aetherai:latest

# ============================================================================
# PyPI Publishing Targets
# ============================================================================

# Clean build artifacts
pypi-clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info aetherai.egg-info terminal.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Build package (wheel + sdist)
pypi-build: pypi-clean
	@echo "Building AetherAI package..."
	python -m pip install --upgrade build
	python -m build
	@echo "Build complete! Check dist/ directory"

# Check package with twine
pypi-check: pypi-build
	@echo "Checking package..."
	python -m pip install --upgrade twine
	twine check dist/*
	@echo "Package check passed!"

# Upload to TestPyPI
pypi-test: pypi-check
	@echo "Uploading to TestPyPI..."
	twine upload --repository testpypi dist/*
	@echo "Upload complete! Test with: pip install --index-url https://test.pypi.org/simple/ aetherai"

# Upload to PyPI (production)
pypi-publish: pypi-check
	@echo "Uploading to PyPI..."
	twine upload dist/*
	@echo "Published! Install with: pip install aetherai"

# ============================================================================
# Development Targets
# ============================================================================

# Install in editable mode with dev dependencies
dev-install:
	@echo "Installing in development mode..."
	python -m pip install -e ".[dev]"
	@echo "Development install complete!"

# Run linters
lint:
	@echo "Running linters..."
	python -m ruff check aetherai terminal --fix
	python -m black --check aetherai terminal
	python -m mypy aetherai terminal --ignore-missing-imports
	@echo "Linting complete!"

# Auto-format code
format:
	@echo "Formatting code..."
	python -m black aetherai terminal
	python -m ruff check aetherai terminal --fix
	@echo "Formatting complete!"

# Run pytest
test-py:
	@echo "Running tests..."
	python -m pytest terminal/tests -v
	@echo "Tests complete!"

# Run all checks (lint + test)
check: lint test-py
	@echo "All checks passed!"

# Full release preparation
release-prep: format lint test-py pypi-check
	@echo "Release preparation complete!"
	@echo "Ready to publish with: make pypi-publish"

