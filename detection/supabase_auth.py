"""
Supabase Authentication Integration
Handles user registration, login, logout, and session management via Supabase.
"""

import os
from supabase import create_client, Client
from gotrue.errors import AuthError
import logging
from django.conf import settings
from django.contrib.sessions.models import Session


class SupabaseAuth:
    """Wrapper for Supabase authentication."""
    
    def __init__(self):
        """Initialize Supabase client."""
        supabase_url = getattr(settings, 'SUPABASE_URL', os.environ.get('SUPABASE_URL'))
        supabase_key = getattr(settings, 'SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY'))

        # Debug print to verify values (remove or comment out in production)
        print('DEBUG: Supabase URL:', repr(supabase_url))
        print('DEBUG: Supabase KEY:', repr(supabase_key))

        # Strip quotes and whitespace if present
        if supabase_url:
            supabase_url = supabase_url.strip().strip("'\"")
        if supabase_key:
            supabase_key = supabase_key.strip().strip("'\"")

        print('DEBUG: Supabase URL (stripped):', repr(supabase_url))
        print('DEBUG: Supabase KEY (stripped):', repr(supabase_key))

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY "
                "in environment variables or Django settings."
            )

        self.client: Client = create_client(supabase_url, supabase_key)
    
    def register(self, email, password, metadata=None):
        """
        Register a new user.
        
        Args:
            email: User email
            password: User password
            metadata: Optional user metadata (name, gender, etc.)
        
        Returns:
            Tuple of (user_data, session) or (None, error_message)
        """
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": metadata or {}
                }
            })

            # Normalize different response shapes from supabase client
            # Some versions return an object with `.user` and `.session`,
            # others return a dict with 'data' and 'error' keys.
            logging.debug(f"Supabase sign_up raw response: {response!r}")

            # Case: object with attributes
            user = getattr(response, 'user', None)
            session = getattr(response, 'session', None)
            if user is not None or session is not None:
                return user, session

            # Case: dict-like response
            try:
                # some clients return dict with 'data' and 'error'
                data = response.get('data') if isinstance(response, dict) else None
                error = response.get('error') if isinstance(response, dict) else None
                if error:
                    return None, str(error)
                if data:
                    # data may contain user/session fields
                    u = data.get('user') or data.get('user_id') or data
                    s = data.get('session') if isinstance(data, dict) else None
                    return u, s
            except Exception:
                pass

            # Unknown response shape
            logging.error(f"Unrecognized sign_up response from Supabase: {response!r}")
            return None, f"Registration failed: unexpected response from auth provider"
        except AuthError as e:
            logging.warning(f"Supabase AuthError during register: {e}")
            return None, str(e)
        except Exception as e:
            logging.exception("Unexpected exception during Supabase registration")
            return None, f"Registration failed: {str(e)}"
    
    def login(self, email, password):
        """
        Login user.
        
        Args:
            email: User email
            password: User password
        
        Returns:
            Tuple of (user_data, session) or (None, error_message)
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return response.user, response.session
        except AuthError as e:
            return None, "Invalid email or password."
        except Exception as e:
            return None, f"Login failed: {str(e)}"
    
    def logout(self, session_token):
        """
        Logout user.
        
        Args:
            session_token: Session access token
        """
        try:
            self.client.auth.set_session(access_token=session_token, refresh_token="")
            self.client.auth.sign_out()
        except Exception as e:
            print(f"Logout error: {e}")
    
    def get_user(self, session_token):
        """
        Get current user from session token.
        
        Args:
            session_token: Session access token
        
        Returns:
            User data or None
        """
        try:
            self.client.auth.set_session(access_token=session_token, refresh_token="")
            user = self.client.auth.get_user(session_token)
            return user
        except Exception as e:
            print(f"Get user error: {e}")
            return None
    
    def update_user_metadata(self, session_token, metadata):
        """
        Update user metadata.
        
        Args:
            session_token: Session access token
            metadata: Dictionary of metadata to update
        """
        try:
            self.client.auth.set_session(access_token=session_token, refresh_token="")
            self.client.auth.update_user({"data": metadata})
        except Exception as e:
            print(f"Update metadata error: {e}")


# Global instance
_supabase_auth = None


def get_supabase_auth():
    """Get or create SupabaseAuth instance."""
    global _supabase_auth
    if _supabase_auth is None:
        _supabase_auth = SupabaseAuth()
    return _supabase_auth


def store_session_in_django(request, supabase_session):
    """
    Store Supabase session in Django session.
    
    Args:
        request: Django request object
        supabase_session: Supabase session object
    """
    if supabase_session:
        request.session['supabase_access_token'] = supabase_session.access_token
        request.session['supabase_refresh_token'] = supabase_session.refresh_token
        request.session['supabase_user_id'] = supabase_session.user.id if supabase_session.user else None
        request.session.save()


def get_session_from_django(request):
    """
    Get Supabase session from Django session.
    
    Args:
        request: Django request object
    
    Returns:
        Dictionary with session tokens or None
    """
    access_token = request.session.get('supabase_access_token')
    refresh_token = request.session.get('supabase_refresh_token')
    user_id = request.session.get('supabase_user_id')
    
    if access_token and user_id:
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user_id': user_id,
        }
    return None


def clear_session_from_django(request):
    """Clear Supabase session from Django session."""
    if 'supabase_access_token' in request.session:
        del request.session['supabase_access_token']
    if 'supabase_refresh_token' in request.session:
        del request.session['supabase_refresh_token']
    if 'supabase_user_id' in request.session:
        del request.session['supabase_user_id']
    request.session.save()

