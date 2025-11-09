# Supabase Authentication Setup Guide

## Overview

MelaScan now uses Supabase for user authentication instead of Django's built-in authentication. This provides better security, scalability, and user management.

## Setup Steps

### 1. Create a Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Click "New Project"
4. Fill in:
   - Project Name: `melascan` (or your choice)
   - Database Password: (choose a strong password)
   - Region: (choose closest to you)
5. Click "Create new project"

### 2. Get Your Credentials

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy the following:
   - **Project URL**: `https://rxcakfbqzptpaxiyotwe.supabase.co`
   - **anon/public key**: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ4Y2FrZmJxenB0cGF4aXlvdHdlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0MTYyNzMsImV4cCI6MjA3Nzk5MjI3M30.cFSUlb4rzSSxksKpIxY7uXXZoWNP8l3_jFTmwclXmO4'
### 3. Configure Environment Variables

**Option A: Using .env file (Recommended)**

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```bash
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   ```

3. Install python-dotenv (already in requirements.txt):
   ```bash
   pip install python-dotenv
   ```

4. Load environment variables in `settings.py` (add at top):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

**Option B: Export in shell**

```bash
export SUPABASE_URL="https://your-project-id.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key-here"
```

**Option C: Set directly in settings.py (Not recommended for production)**

Edit `melascan/settings.py`:
```python
SUPABASE_URL = 'https://your-project-id.supabase.co'
SUPABASE_ANON_KEY = 'your-anon-key-here'
```

### 4. Test the Setup

1. Start the Django server:
   ```bash
   python manage.py runserver
   ```

2. Try registering a new user at `http://127.0.0.1:8000/register/`

3. Check your Supabase dashboard → **Authentication** → **Users** to see the new user

## Troubleshooting

### "Supabase credentials not found"

- Make sure environment variables are set correctly
- Check that `.env` file is in the project root
- Verify you're using the `anon` key, not `service_role` key

### "Registration failed"

- Check Supabase project is active
- Verify credentials are correct
- Check Supabase dashboard for error logs

### "Invalid email or password"

- Ensure user exists in Supabase
- Check email confirmation settings in Supabase dashboard
- For development, you can disable email confirmation:
  1. Go to Supabase Dashboard → **Authentication** → **Settings**
  2. Disable "Enable email confirmations"

## Features

- ✅ User registration with email/password
- ✅ User login/logout
- ✅ Session management via Django sessions
- ✅ User metadata storage (name, gender, date of birth)
- ✅ Profile updates sync to Supabase

## Migration from Django Auth

Existing Django users will need to register again with Supabase. Their data (reports, profiles) will be preserved if they use the same email address.

