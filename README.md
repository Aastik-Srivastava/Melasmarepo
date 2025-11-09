# MelaScan - Intelligent Melasma Detection Web Application

A Django-based web application for melasma detection using multiple ML models (EfficientNet, ResNet, DenseNet, etc.). The platform automatically selects the best-performing model based on accuracy and provides comprehensive reports.

## Features

- **User Authentication**: Secure registration, login, and logout with session management
- **Dashboard**: Real-time statistics and model performance metrics
- **Melasma Detection**: Upload images for AI-powered analysis
- **PDF Reports**: Automatically generated detailed reports with results and model metrics
- **Profile Management**: Edit user profile information
- **Admin Interface**: Manage models, users, and reports via Django admin

## Technology Stack

- **Backend**: Django 5.0.1
- **Frontend**: HTML5, TailwindCSS, JavaScript
- **ML Framework**: TensorFlow 2.15.0 (placeholder for model integration)
- **Database**: SQLite (development) / PostgreSQL (production)
- **PDF Generation**: ReportLab
- **Image Processing**: Pillow

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd melascan
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create a superuser** (for admin access):
   ```bash
   python manage.py createsuperuser
   ```

6. **Populate initial model data** (optional):
   You can add model performance data via Django admin or use this command:
   ```bash
   python manage.py shell
   ```
   Then in the shell:
   ```python
   from detection.models import ModelPerformance
   ModelPerformance.objects.create(
       model_name='EfficientNet-B3',
       accuracy=0.962,
       precision=0.978,
       recall=0.954,
       f1_score=0.966,
       is_active=True
   )
   ModelPerformance.objects.create(
       model_name='ResNet-50',
       accuracy=0.945,
       precision=0.960,
       recall=0.935,
       f1_score=0.947,
       is_active=True
   )
   ModelPerformance.objects.create(
       model_name='DenseNet-121',
       accuracy=0.938,
       precision=0.952,
       recall=0.928,
       f1_score=0.940,
       is_active=True
   )
   ```

7. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

8. **Access the application**:
   - Main app: http://127.0.0.1:8000/
   - Admin panel: http://127.0.0.1:8000/admin/

## Project Structure

```
melascan/
├── melascan/           # Main Django project settings
│   ├── settings.py     # Project configuration
│   ├── urls.py         # Main URL routing
│   └── wsgi.py         # WSGI configuration
├── detection/          # Main application
│   ├── models.py       # Database models
│   ├── views.py        # View functions
│   ├── forms.py        # Form definitions
│   ├── urls.py         # App URL routing
│   ├── admin.py        # Admin configuration
│   ├── ml_integration.py    # ML model integration
│   └── report_generator.py  # PDF generation
├── templates/          # HTML templates
│   └── detection/
│       ├── login.html
│       ├── register.html
│       ├── dashboard.html
│       ├── profile.html
│       └── detect.html
├── media/              # User uploaded files
│   ├── uploads/        # Uploaded images
│   ├── reports/        # Generated PDF reports
│   └── models/         # ML model files (to be added)
├── static/             # Static files (CSS, JS)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

### For Users

1. **Register**: Create a new account with your details
2. **Login**: Sign in to access the dashboard
3. **Upload Image**: Go to "Start Detection" and upload a skin image
4. **View Results**: Check the dashboard for your latest report
5. **Download PDF**: Click the download button to get a detailed PDF report

### For Administrators

1. **Access Admin**: Login at `/admin/` with superuser credentials
2. **Manage Models**: Add/edit ML model performance metrics
3. **View Reports**: Monitor all user reports and statistics
4. **User Management**: Manage user accounts and profiles

## ML Model Integration

The application currently uses a placeholder prediction system. To integrate your trained models:

1. **Add Model Files**: Place your trained models (`.h5`, `.pth`, etc.) in `media/models/`

2. **Update `detection/ml_integration.py`**:
   - Implement `load_model()` function to load your model
   - Update `get_prediction()` to use actual model inference
   - Modify preprocessing as needed for your model input requirements

3. **Example for TensorFlow/Keras**:
   ```python
   import tensorflow as tf
   
   def load_model(model_path):
       return tf.keras.models.load_model(model_path)
   
   def get_prediction(image_file, model_performance):
       processed_image = preprocess_image(image_file)
       model = get_cached_model(model_performance.model_name)
       prediction = model.predict(processed_image)
       # Process prediction and return result
   ```

4. **Update Model Performance**: Add model metrics via admin or shell

## Database Schema

### Models

- **UserProfile**: Extended user information (name, gender, date of birth)
- **ModelPerformance**: ML model metrics (accuracy, precision, recall, F1-score)
- **MelasmaReport**: Detection reports with results and PDF files

## Production Deployment

### Database Configuration

Update `melascan/settings.py` to use PostgreSQL:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'melascan_db',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Security Settings

Update `settings.py` for production:

```python
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']
SECRET_KEY = 'your-secret-key-here'  # Use environment variable
```

### Static Files

Collect static files:
```bash
python manage.py collectstatic
```

## Future Enhancements

- Multi-disease detection (melasma, acne, pigmentation)
- Real-time webcam input
- Email report delivery
- AI-based skin improvement recommendations
- Model comparison interface
- Batch image processing

## License

© MelaScan WebApp — Developed by Team M — 2025

## Support

For issues or questions, please contact the development team.

