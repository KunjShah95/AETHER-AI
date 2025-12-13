.PHONY: help install build pypi-build pypi-test pypi-publish dev-install lint format test-py check release-prep

# Default target
help:
	@echo "AetherAI Management"
	@echo "==================="
	@echo ""
	@echo "Project Setup:"
	@echo "  make install      - Install the project"
	@echo "  make build        - Build the frontend"
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
	@echo "  make check        - Run all checks (lint + test)"
	@echo "  make release-prep - Full release preparation"
	@echo ""

# ============================================================================
# Project Setup
# ============================================================================

# Install the project
install:
	@echo "Installing AetherAI..."
	python -m pip install -e .
	@echo "Installation complete!"

# Build frontend
build:
	@echo "Building frontend..."
	cd frontend && npm install && npm run build
	@echo "Frontend build complete!"

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
