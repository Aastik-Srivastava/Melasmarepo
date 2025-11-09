#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install system dependencies
apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libpq-dev

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p media/uploads
mkdir -p media/reports
mkdir -p staticfiles

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate