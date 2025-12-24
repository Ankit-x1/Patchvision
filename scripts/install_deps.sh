#!/bin/bash

# PatchVision Dependency Installer
# For Ubuntu/Debian systems

set -e

echo "Installing PatchVision dependencies..."

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    g++ \
    make \
    cmake

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy

echo "Installation complete!"
echo "To activate virtual environment: source venv/bin/activate"