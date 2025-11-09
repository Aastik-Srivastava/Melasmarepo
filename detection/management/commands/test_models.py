"""
Management command to test ML model loading.
Usage: python manage.py test_models
"""

from django.core.management.base import BaseCommand
from detection.ml_integration import initialize_models, SEGMENTATION_MODEL, CLASSIFICATION_MODEL, VGG_MODEL
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Test ML model loading and initialization'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write(self.style.SUCCESS('Testing ML Model Loading'))
        self.stdout.write(self.style.SUCCESS('=' * 60))
        
        # Check if model files exist
        seg_path = os.path.join(settings.MEDIA_ROOT, 'models', 'hybrid_best.pt')
        cls_path = os.path.join(settings.MEDIA_ROOT, 'models', 'fcm_classifier.pkl')
        
        self.stdout.write(f'\n📁 Checking model files...')
        self.stdout.write(f'Segmentation model: {seg_path}')
        if os.path.exists(seg_path):
            size = os.path.getsize(seg_path) / (1024 * 1024)  # MB
            self.stdout.write(self.style.SUCCESS(f'  ✅ Found ({size:.1f} MB)'))
        else:
            self.stdout.write(self.style.ERROR(f'  ❌ Not found'))
        
        self.stdout.write(f'Classification model: {cls_path}')
        if os.path.exists(cls_path):
            size = os.path.getsize(cls_path) / 1024  # KB
            self.stdout.write(self.style.SUCCESS(f'  ✅ Found ({size:.1f} KB)'))
        else:
            self.stdout.write(self.style.ERROR(f'  ❌ Not found'))
        
        # Initialize models
        self.stdout.write(f'\n🔄 Initializing models...')
        try:
            initialize_models()
            
            # Check model status
            self.stdout.write(f'\n📊 Model Status:')
            if SEGMENTATION_MODEL is not None:
                self.stdout.write(self.style.SUCCESS('  ✅ Segmentation model: Loaded'))
            else:
                self.stdout.write(self.style.ERROR('  ❌ Segmentation model: Failed'))
            
            if CLASSIFICATION_MODEL is not None:
                self.stdout.write(self.style.SUCCESS('  ✅ Classification model: Loaded'))
            else:
                self.stdout.write(self.style.ERROR('  ❌ Classification model: Failed'))
            
            if VGG_MODEL is not None:
                self.stdout.write(self.style.SUCCESS('  ✅ VGG16 model: Loaded'))
            else:
                self.stdout.write(self.style.WARNING('  ⚠️  VGG16 model: Not available'))
            
            self.stdout.write(self.style.SUCCESS('\n' + '=' * 60))
            self.stdout.write(self.style.SUCCESS('Model initialization complete!'))
            self.stdout.write(self.style.SUCCESS('=' * 60))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Error during initialization: {e}'))
            import traceback
            self.stdout.write(traceback.format_exc())


