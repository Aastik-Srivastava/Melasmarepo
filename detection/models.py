from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class UserProfile(models.Model):
    """Extended user profile with additional information."""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    name = models.CharField(max_length=200)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='M')
    date_of_birth = models.DateField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} ({self.user.username})"


class ModelPerformance(models.Model):
    """ML model performance metrics."""
    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-accuracy']
        verbose_name_plural = 'Model Performances'
    
    def __str__(self):
        return f"{self.model_name} (Acc: {self.accuracy:.2%})"
    
    @classmethod
    def get_best_model(cls):
        """Returns the model with highest accuracy."""
        return cls.objects.filter(is_active=True).order_by('-accuracy').first()


class MelasmaReport(models.Model):
    """Melasma detection report for each user."""
    RESULT_CHOICES = [
        ('Melasma Detected', 'Melasma Detected'),
        ('Normal Skin', 'Normal Skin'),
        ('Benign', 'Benign'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    date = models.DateTimeField(default=timezone.now)
    result = models.CharField(max_length=50, choices=RESULT_CHOICES)
    model_used = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    uploaded_image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    report_pdf = models.FileField(upload_to='reports/%Y/%m/%d/', null=True, blank=True)
    
    class Meta:
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.user.username} - {self.result} ({self.date.strftime('%Y-%m-%d')})"

