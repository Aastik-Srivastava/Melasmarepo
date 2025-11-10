"""
WSGI config for PythonAnywhere deployment
"""

import os
import sys

# Add your project directory to the sys.path
path = '/home/YOUR_USERNAME/melascan'
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['DJANGO_SETTINGS_MODULE'] = 'melascan.settings_prod'

# Import Django WSGI handler
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()