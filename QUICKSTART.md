# Quick Start Guide

## Initial Setup (First Time Only)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

3. **Create superuser**:
   ```bash
   python manage.py createsuperuser
   ```
   Follow the prompts to create an admin account.

4. **Initialize model data**:
   ```bash
   python manage.py init_models
   ```
   This creates sample ML model performance data.

5. **Run the server**:
   ```bash
   python manage.py runserver
   ```

6. **Access the application**:
   - Main app: http://127.0.0.1:8000/
   - Admin: http://127.0.0.1:8000/admin/

## Daily Usage

After initial setup, just run:
```bash
python manage.py runserver
```

## Testing the Application

1. **Register a new account** at http://127.0.0.1:8000/register/
2. **Login** with your credentials
3. **Upload an image** at "Start Detection"
4. **View results** on the dashboard
5. **Download PDF report** if available

## Admin Tasks

- Login at `/admin/` with superuser credentials
- Manage users, reports, and model performance metrics
- Add/edit ML model performance data

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **Database errors**: Run migrations again (`python manage.py migrate`)
- **Static files**: Run `python manage.py collectstatic` (for production)
- **Media files**: Ensure `media/` directory exists and is writable

