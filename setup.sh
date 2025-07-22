#!/bin/bash

# Personal Finance Analytics Platform - Setup Script
# This script installs all dependencies needed for development and testing

echo "Setting up Personal Finance Analytics Platform..."

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install all dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install development tools
echo "Installing development tools..."
pip install black isort mypy bandit safety locust

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Warning: Docker is not installed. Please install Docker to run the full test suite."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Warning: Docker Compose is not installed. Please install Docker Compose to run the full test suite."
fi

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run 'make test' to run all tests"
echo "2. Run 'docker-compose up -d' to start all services"
echo "3. Run 'make install-dev' if you need to reinstall development tools"
echo ""
echo "Available commands:"
echo "  make test          - Run all tests including new features and CI/CD validation"
echo "  make install-dev   - Install development dependencies"
echo "  make clean         - Clean up Docker containers and cache"
echo "  docker-compose up  - Start all services" 