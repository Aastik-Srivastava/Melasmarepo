#!/usr/bin/env bash
# exit on error
set -o errexit

# Print Python version
python --version

# Upgrade pip and install build tools
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

# Clean pip cache
pip cache purge

# Install Python dependencies with specific options
pip install --no-cache-dir -r requirements.txt --use-pep517

# Create necessary directories
mkdir -p media/uploads
mkdir -p media/reports
mkdir -p staticfiles

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate