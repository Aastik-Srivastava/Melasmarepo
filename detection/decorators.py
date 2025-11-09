"""
Custom decorators for Supabase authentication.
"""

from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from .supabase_auth import get_session_from_django, get_supabase_auth


def supabase_login_required(view_func):
    """
    Decorator to check if user is authenticated via Supabase.
    Redirects to login if not authenticated.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        session = get_session_from_django(request)
        if not session:
            messages.error(request, 'Please login to access this page.')
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def get_current_user(request):
    """
    Get current Supabase user from request session.
    
    Returns:
        User data dictionary or None
    """
    session = get_session_from_django(request)
    if not session:
        return None
    
    try:
        supabase_auth = get_supabase_auth()
        user = supabase_auth.get_user(session['access_token'])
        if user:
            return {
                'id': user.id,
                'email': user.email,
                'metadata': user.user_metadata or {},
            }
    except Exception as e:
        print(f"Error getting user: {e}")
    
    return None

