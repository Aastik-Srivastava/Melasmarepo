from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import HttpResponse, Http404
from django.db.models import Count, Q
from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.conf import settings

from .models import UserProfile, ModelPerformance, MelasmaReport
from .forms import ProfileForm, ImageUploadForm, RegistrationForm
from .ml_integration import get_prediction
from .report_generator import generate_pdf_report
from .hybrid_pipeline import init_hybrid_models
# Hybrid models are initialized lazily by `hybrid_pipeline_ext` when needed to avoid
# heavy model loading at import/startup time.

import os
import time


def get_unique_filename(original_name):
    """Generate a unique filename based on timestamp."""
    base, ext = os.path.splitext(original_name)
    timestamp = str(int(time.time()))
    return f"{base}_{timestamp}{ext}"


def register_view(request):
    """User registration using Django's built-in auth.

    This replaces Supabase signup and creates a local Django user and UserProfile.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            # Save the Django user
            django_user = form.save()
            # Create UserProfile
            name = form.cleaned_data.get('name')
            gender = form.cleaned_data.get('gender')
            dob = form.cleaned_data.get('date_of_birth')
            UserProfile.objects.get_or_create(
                user=django_user,
                defaults={
                    'name': name or django_user.username,
                    'gender': gender or 'M',
                    'date_of_birth': dob,
                }
            )
            # Authenticate and login
            user = authenticate(request, username=django_user.username, password=form.cleaned_data.get('password1'))
            if user:
                auth_login(request, user)
                messages.success(request, 'Registration successful! You are now logged in.')
                return redirect('dashboard')
            else:
                messages.warning(request, 'Registration succeeded but automatic login failed. Please login manually.')
                return redirect('login')
    else:
        form = RegistrationForm()

    return render(request, 'detection/register.html', {'form': form})


def login_view(request):
    """User login view using Django auth."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if email and password:
            # Try to authenticate by username first, then email
            user = authenticate(request, username=email, password=password)
            if not user:
                # If users register with email as username, this works; otherwise try lookup by email
                try:
                    django_user = User.objects.filter(email=email).first()
                    if django_user:
                        user = authenticate(request, username=django_user.username, password=password)
                except Exception:
                    user = None

            if user:
                auth_login(request, user)
                messages.success(request, 'Login successful!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid email or password.')
        else:
            messages.error(request, 'Please provide both email and password.')

    return render(request, 'detection/login.html')


@login_required
def logout_view(request):
    """User logout view using Django auth."""
    auth_logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')


@login_required
def dashboard_view(request):
    """Main dashboard view."""
    django_user = request.user
    
    # Get best model
    best_model = ModelPerformance.get_best_model()
    
    # Get statistics (using Django User model for compatibility)
    total_users = User.objects.count()
    benign_count = MelasmaReport.objects.filter(result='Benign').count()
    melasma_count = MelasmaReport.objects.filter(result='Melasma Detected').count()
    normal_count = MelasmaReport.objects.filter(result='Normal Skin').count()
    
    # Get user's last report
    last_report = None
    if django_user:
        last_report = MelasmaReport.objects.filter(user=django_user).first()
    
    # Get all model metrics
    model_metrics = ModelPerformance.objects.filter(is_active=True).order_by('-accuracy')
    
    context = {
        'best_model': best_model,
        'total_users': total_users,
        'benign_count': benign_count,
        'melasma_count': melasma_count,
        'normal_count': normal_count,
        'last_report': last_report,
        'model_metrics': model_metrics,
        'user_email': django_user.email,
        'user_name': getattr(django_user, 'first_name', '') or django_user.username,
    }
    
    return render(request, 'detection/dashboard.html', context)


@login_required
def profile_view(request):
    """User profile view."""
    django_user = request.user
    if not django_user:
        messages.error(request, 'User profile not found.')
        return redirect('dashboard')
    
    profile, created = UserProfile.objects.get_or_create(
        user=django_user,
        defaults={
            'name': django_user.get_full_name() or django_user.username,
            'gender': 'M',
        }
    )
    
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            
            # No external auth provider to update; profile saved locally
            
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = ProfileForm(instance=profile)
    
    return render(request, 'detection/profile.html', {'form': form, 'profile': profile})


@login_required
def detect_view(request):
    """Melasma detection view with segmentation and classification modes."""
    django_user = request.user
    if not django_user:
        messages.error(request, 'User profile not found.')
        return redirect('dashboard')
    
    # Get the mode from query param, default to segment
    mode = request.GET.get('mode', 'segment')
    if mode not in ['segment', 'classify']:
        mode = 'segment'
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']
            
            # Save the file to media/uploads/
            filename = get_unique_filename(uploaded_image.name)
            filepath = os.path.join('uploads', filename)
            
            # Save the file
            with open(os.path.join(settings.MEDIA_ROOT, filepath), 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)

            # Full filesystem path to the saved image
            full_image_path = os.path.join(settings.MEDIA_ROOT, filepath)

            # Process image based on mode
            if mode == 'segment':
                from .hybrid_pipeline_ext import process_segmentation
                prediction_result = process_segmentation(full_image_path)
            else:
                from .hybrid_pipeline_ext import process_classification
                prediction_result = process_classification(full_image_path)

            if not prediction_result or 'error' in prediction_result:
                messages.error(request, prediction_result.get('error', 'Error processing image. Please try again.'))
                return redirect('detect')

            # Create report
            report = MelasmaReport.objects.create(
                user=django_user,
                result=prediction_result['result'],
                model_used=prediction_result['model_used'],
                accuracy=0.0,  # These could be fetched from ModelPerformance if needed
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                uploaded_image=filepath,
            )
            
            # For classification mode, generate PDF report
            if mode == 'classify':
                pdf_file = generate_pdf_report(report, django_user, prediction_result=prediction_result)
                report.report_pdf = pdf_file
                report.save()
                messages.success(request, f'Classification complete! Result: {prediction_result["result"]} (Confidence: {prediction_result["confidence"]:.1f}%)')
            else:
                messages.success(request, f'Segmentation complete! Affected area: {prediction_result["affected_percentage"]:.1f}%')
            
            # Get URLs for display
            uploaded_image_url = os.path.join(settings.MEDIA_URL, filepath)
            overlay_url = None
            
            # Check if there's an overlay image
            overlay_path = prediction_result.get('overlay_path')
            if overlay_path:
                # overlay is saved under MEDIA_ROOT/reports/<name>
                overlay_url = settings.MEDIA_URL + 'reports/' + os.path.basename(overlay_path)

            context = {
                'form': form,
                'mode': mode,
                'uploaded_image_url': uploaded_image_url,
                'overlay_url': overlay_url,
                'prediction_result': prediction_result,
                # Provide the report id so the template can show a download button only after classification
                'report_id': report.id if mode == 'classify' else None,
            }
            return render(request, 'detection/detect.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'detection/detect.html', {'form': form, 'mode': mode})


@login_required
def download_report_pdf(request, report_id):
    """Download PDF report."""
    django_user = request.user
    if not django_user:
        messages.error(request, 'User profile not found.')
        return redirect('dashboard')
    
    report = get_object_or_404(MelasmaReport, id=report_id, user=django_user)
    
    if not report.report_pdf:
        messages.error(request, 'Report PDF not available.')
        return redirect('dashboard')
    
    try:
        response = HttpResponse(report.report_pdf.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="melascan_report_{report.id}.pdf"'
        return response
    except Exception as e:
        messages.error(request, 'Error downloading report.')
        return redirect('dashboard')
