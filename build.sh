#!/usr/bin/env bash
set -o errexit

echo "Python version:"
python --version

echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Running migrations..."
python manage.py migrate

# Create necessary directories
mkdir -p media/uploads
mkdir -p media/reports
mkdir -p staticfiles

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate